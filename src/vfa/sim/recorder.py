"""Episode recorder (T06): per-tick arrays, events, skill parameters, outcome.

``EpisodeRecorder.record(frame, phase, privileged)`` appends one row per control
tick (or every ``stride`` ticks) from a ``SensorFrame`` and, optionally, the
``PrivilegedState``. ``finish(result, ...)`` freezes it into an ``Episode``, which
``save`` writes as one ``.npz`` file (no pickle) plus one ``.jsonl`` summary line
appended to ``episodes.jsonl`` beside it. ``load_episode`` reads the ``.npz``
back exactly: every array bit for bit (dtype and shape included), and the
phases, events, parameters, context, outcome, and metadata as saved.

Columns (``COLUMNS``; refer to ``SensorFrame`` and ``PrivilegedState`` for units):
``t``, ``phase`` (an index into ``Episode.phases``), ``wrench_raw``, ``wrench``,
``acc``, ``finger_force``, ``contact``, ``width``, ``width_rate``, ``ee_pos``,
``ee_quat``, ``ee_vel``, ``load``, ``peak_load``, ``armed``, ``watching``,
``watch_id`` (which carry), ``slip``, and the privileged ``cube_pos``, ``cube_quat``, ``cube_vel``,
``cube_finger_force``, ``broken`` (``PRIVILEGED_COLUMNS``: NaN, or False, on rows
recorded without a ``PrivilegedState``). The privileged columns are for scoring,
logging, and plots; nothing a policy or a memory key reads may use them
(ARCHITECTURE §4).

Beside the arrays an episode keeps JSON parts: ``context`` (what was observable
at the attempt's start: the observed cube pose and size, the initial joints;
T11's ``key_pre``), ``params`` (θ at the end) with ``params_history`` (every θ
and the time it was set, e.g. before and after ``PROBE_DONE``), ``outcome``,
``meta`` (logging only), and the event ``info``. They are stored as JSON
returns them (tuples become lists, keys become strings), so an episode equals
its loaded copy. The summary line is strict JSON: non-finite numbers become
``null``. Each line is one ``write`` in append mode, so worker processes can
share one ``episodes.jsonl``. ``save`` writes the ``.npz`` to a temporary file
and renames it, so a killed run never leaves a truncated episode behind.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vfa.sim.sensors import Event, EventRecord, PrivilegedState, SensorFrame

# Episode results (ARCHITECTURE §4 failure modes, plus SUCCESS).
RESULTS: tuple[str, ...] = (
    "SUCCESS",
    "MISS",
    "SLIP",
    "DROP",
    "BREAK",
    "TIMEOUT",
    "SAFETY_STOP",
    "PLACE_BREAK",
    "TOPPLE",
    "PLACE_MISS",
)

_F8 = np.dtype(np.float64)
_B1 = np.dtype(np.bool_)
# name -> (row shape, dtype). Observable columns first, privileged last.
SENSOR_COLUMNS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "t": ((), _F8),
    "phase": ((), np.dtype(np.int32)),
    "wrench_raw": ((6,), _F8),
    "wrench": ((6,), _F8),
    "acc": ((3,), _F8),
    "finger_force": ((2,), _F8),
    "contact": ((2,), _B1),
    "width": ((), _F8),
    "width_rate": ((), _F8),
    "ee_pos": ((3,), _F8),
    "ee_quat": ((4,), _F8),
    "ee_vel": ((6,), _F8),
    "load": ((), _F8),
    "peak_load": ((), _F8),
    "armed": ((), _B1),
    "watching": ((), _B1),
    "watch_id": ((), np.dtype(np.int32)),
    "slip": ((), _B1),
}
PRIVILEGED_COLUMNS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "cube_pos": ((3,), _F8),
    "cube_quat": ((4,), _F8),
    "cube_vel": ((6,), _F8),
    "cube_finger_force": ((2,), _F8),
    "broken": ((), _B1),
}
COLUMNS = {**SENSOR_COLUMNS, **PRIVILEGED_COLUMNS}
_JSON_KEY = "episode_json"
SUMMARY_FILE = "episodes.jsonl"


def _jsonable(value: object) -> object:
    """``value`` as ``json.loads(json.dumps(value))`` returns it (NaN allowed)."""
    try:
        return json.loads(json.dumps(value, default=_numpy_default))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"not JSON-serializable: {value!r} ({exc})") from None


def _numpy_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _strict(value: object) -> object:
    """Replace non-finite floats with ``None`` (strict JSON for the summary line)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _strict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strict(v) for v in value]
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True)


@dataclass(slots=True, eq=False)
class Episode:
    """A finished recording. ``==`` compares every array exactly (NaN equal to NaN)."""

    arrays: dict[str, np.ndarray]
    phases: tuple[str, ...]
    events: tuple[EventRecord, ...]
    params: dict
    outcome: dict
    meta: dict
    context: dict = field(default_factory=dict)
    params_history: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.arrays["t"].shape[0])

    @property
    def result(self) -> str:
        return str(self.outcome["result"])

    @property
    def success(self) -> bool:
        return bool(self.outcome["success"])

    def phase_names(self) -> list[str]:
        """The phase of every row, by name."""
        return [self.phases[i] for i in self.arrays["phase"]]

    def phase_spans(self) -> list[tuple[str, float, float]]:
        """``(phase, t_start, t_end)`` for each run of rows in one phase.

        A span ends where the next one starts (the last ends at the last row).
        """
        t, idx = self.arrays["t"], self.arrays["phase"]
        if t.size == 0:
            return []
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        ends = np.r_[starts[1:], t.size - 1]
        pairs = zip(starts, ends, strict=True)
        return [(self.phases[idx[s]], float(t[s]), float(t[e])) for s, e in pairs]

    def events_named(self, name: Event | str) -> list[EventRecord]:
        return [e for e in self.events if e.name == Event(name)]

    def _doc(self) -> dict:
        return {
            "phases": list(self.phases),
            "events": _events_json(self.events),
            "params": self.params,
            "params_history": self.params_history,
            "context": self.context,
            "outcome": self.outcome,
            "meta": self.meta,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Episode):
            return NotImplemented
        if self.arrays.keys() != other.arrays.keys():
            return False
        for key, a in self.arrays.items():
            b = other.arrays[key]
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
                return False
        return _canonical(self._doc()) == _canonical(other._doc())

    __hash__ = None  # mutable arrays

    def summary(self, file: str | None = None) -> dict:
        """The ``.jsonl`` summary line (strict JSON)."""
        a = self.arrays

        def peak(values: np.ndarray) -> float | None:
            finite = values[np.isfinite(values)]
            return float(finite.max()) if finite.size else None

        line = {
            "file": file,
            "result": self.result,
            "success": self.success,
            "outcome": self.outcome,
            "params": self.params,
            "context": self.context,
            "events": [[e.name.value, e.t] for e in self.events],
            "ticks": self.n,
            "t_start": float(a["t"][0]) if self.n else None,
            "t_end": float(a["t"][-1]) if self.n else None,
            "phases": [[name, t0, t1] for name, t0, t1 in self.phase_spans()],
            "peak_finger_force": peak(a["finger_force"].sum(axis=1)) if self.n else None,
            "peak_load": peak(a["load"]) if self.n else None,
            "peak_cube_finger_force": peak(a["cube_finger_force"].sum(axis=1)) if self.n else None,
            "meta": self.meta,
        }
        return _strict(line)

    def save(self, path: str | Path, summary: str | Path | None = SUMMARY_FILE) -> Path:
        """Write ``path`` (``.npz`` added if missing) and append the summary line.

        ``summary`` is the ``.jsonl`` file, relative to the ``.npz`` file's
        directory unless absolute; ``None`` skips it. The line's ``file`` is the
        ``.npz`` path relative to the summary's directory. The ``.npz`` is
        written to a temporary file and renamed, so it appears whole or not at
        all. Returns the ``.npz`` path.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        target = None
        if summary is not None:
            target = Path(summary)
            if not target.is_absolute():
                target = path.parent / target
            target.parent.mkdir(parents=True, exist_ok=True)
        # A short random name beside the target (a long episode name must not overflow
        # it), created with mode 0o666 so the kernel applies the umask, as open() does.
        while True:
            tmp = path.parent / f".episode.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
                break
            except FileExistsError:
                continue
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(
                    fh, **self.arrays, **{_JSON_KEY: np.array(json.dumps(self._doc()))}
                )
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        if target is not None:
            rel = os.path.relpath(path, target.parent)
            line = json.dumps(self.summary(Path(rel).as_posix()), allow_nan=False) + "\n"
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line)
        return path


def _events_json(events: Iterable[EventRecord]) -> list:
    return [{"name": e.name.value, "t": e.t, "info": e.info} for e in events]


def load_episode(path: str | Path) -> Episode:
    """Read an episode written by ``Episode.save`` (no pickle); validate its layout."""
    path = Path(path)
    try:
        if not path.exists():
            raise ValueError(f"{path}: no such episode file")
        if not path.is_file():  # a directory or a FIFO would fail late, or block
            raise ValueError(f"{path}: not an episode file (not a regular file)")
        with open(path, "rb") as fh:
            if fh.read(4) != b"PK\x03\x04":  # numpy would guess "pickle" for any non-zip
                raise ValueError(f"{path}: not a readable episode file (not an .npz archive)")
        with np.load(path, allow_pickle=False) as npz:
            files = set(npz.files)
            missing = (set(COLUMNS) | {_JSON_KEY}) - files
            if missing:
                raise ValueError(f"{path}: not an episode file (missing {sorted(missing)})")
            arrays = {name: npz[name] for name in COLUMNS}
            doc = json.loads(str(npz[_JSON_KEY]))
            if not all(isinstance(a, np.ndarray) for a in arrays.values()):
                raise ValueError(f"{path}: not an episode file (a column is not an array)")
    except (
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        OSError,
        json.JSONDecodeError,
        NotImplementedError,  # zip flags numpy cannot read (a damaged header)
        RuntimeError,  # an "encrypted" zip member
        RecursionError,  # absurdly nested JSON
        MemoryError,  # a crafted array header with an absurd shape
    ) as exc:
        raise ValueError(f"{path}: not a readable episode file ({exc!r})") from None
    n = arrays["t"].shape[0] if arrays["t"].ndim == 1 else -1
    for name, (shape, dtype) in COLUMNS.items():
        a = arrays[name]
        if a.dtype != dtype or a.shape != (n, *shape):
            raise ValueError(
                f"{path}: column {name!r} is {a.dtype}{a.shape}, expected {dtype}{(n, *shape)}"
            )
    try:
        phases = doc["phases"]
        if not isinstance(phases, list) or not all(isinstance(p, str) for p in phases):
            raise TypeError("phases must be a list of str")
        events = []
        if not isinstance(doc["events"], list):
            raise TypeError("events must be a list")
        for e in doc["events"]:
            t, info = e["t"], e["info"]
            if isinstance(t, bool) or not isinstance(t, int | float) or not isinstance(info, dict):
                raise TypeError(f"event {e!r} needs a numeric t and an info object")
            events.append(EventRecord(Event(e["name"]), t, info))
        events = tuple(events)
        parts = [doc[k] for k in ("params", "outcome", "meta", "context")]
        history = doc["params_history"]
        if not all(isinstance(x, dict) for x in parts) or not isinstance(history, list):
            raise TypeError("params, outcome, meta, context must be objects")
        outcome = parts[1]
        if outcome.get("result") not in RESULTS or not isinstance(outcome.get("success"), bool):
            raise TypeError(f"outcome needs a result in RESULTS and a bool success: {outcome!r}")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{path}: damaged episode metadata ({exc!r})") from None
    if n and not 0 <= int(arrays["phase"].min()) <= int(arrays["phase"].max()) < len(phases):
        raise ValueError(f"{path}: phase indices outside the {len(phases)} phase names")
    return Episode(arrays, tuple(phases), events, *parts[:3], parts[3], history)


class EpisodeRecorder:
    """Collects one episode. ``record`` every control tick, then ``finish``.

    ``stride`` keeps every ``stride``-th call (1 = every tick; T11 fixtures can
    keep 100 Hz with 5); events are kept whatever the stride. ``context`` holds
    what was observable at the start (the observed cube pose and size, the
    initial joints), ``params`` the skill's θ (``set_params`` changes it and
    keeps the history), ``meta`` anything for logs (class, seed, hidden
    properties). All are JSON-ready mappings. A call that raises leaves the
    recorder as it was.
    """

    def __init__(
        self,
        meta: Mapping | None = None,
        params: Mapping | None = None,
        stride: int = 1,
        capacity: int = 1024,
        context: Mapping | None = None,
    ) -> None:
        for name, value in (("stride", stride), ("capacity", capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        self.stride = stride
        self.meta = _json_dict("meta", meta)
        self.context = _json_dict("context", context)
        self.params = _json_dict("params", params)
        self.params_history: list = [{"t": None, "params": self.params}] if params else []
        self.events: list[EventRecord] = []
        self._phases: dict[str, int] = {}
        self._buf = {name: np.empty((capacity, *shp), dt) for name, (shp, dt) in COLUMNS.items()}
        self._n = 0
        self._calls = 0
        self._last_t = -math.inf
        self.finished: Episode | None = None

    @property
    def n(self) -> int:
        return self._n

    def set_params(self, params: Mapping, t: float | None = None) -> None:
        """Replace θ (e.g. at ``PROBE_DONE``); ``t`` [s] dates it in ``params_history``."""
        self._check_open()
        new = _json_dict("params", params)
        if t is not None and not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t!r}")
        self.params = new
        self.params_history.append({"t": None if t is None else float(t), "params": new})

    def record(
        self, frame: SensorFrame, phase: str, privileged: PrivilegedState | None = None
    ) -> None:
        self._check_open()
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"phase must be a non-empty str, got {phase!r}")
        phase = str.__str__(phase)  # a str Enum member records as its value
        if not frame.t > self._last_t:
            raise ValueError(f"time must increase: {frame.t} after {self._last_t}")
        if self._calls % self.stride:
            self._last_t = frame.t
            self._calls += 1
            return
        i = self._n
        if i == self._buf["t"].shape[0]:
            self._grow()
        index = self._phases.get(phase, len(self._phases))
        b = self._buf
        b["t"][i] = frame.t
        b["phase"][i] = index
        b["wrench_raw"][i] = frame.wrench_raw
        b["wrench"][i] = frame.wrench
        b["acc"][i] = frame.acc
        b["finger_force"][i] = frame.finger_force
        b["contact"][i] = frame.contact
        b["width"][i] = frame.width
        b["width_rate"][i] = frame.width_rate
        b["ee_pos"][i] = frame.ee_pos
        b["ee_quat"][i] = frame.ee_quat
        b["ee_vel"][i] = frame.ee_vel
        b["load"][i] = frame.load
        b["peak_load"][i] = frame.peak_load
        b["armed"][i] = frame.armed
        b["watching"][i] = frame.watching
        b["watch_id"][i] = frame.watch_id
        b["slip"][i] = frame.slip
        if privileged is None:
            for name in ("cube_pos", "cube_quat", "cube_vel", "cube_finger_force"):
                b[name][i] = np.nan
            b["broken"][i] = False
        else:
            b["cube_pos"][i] = privileged.cube_pos
            b["cube_quat"][i] = privileged.cube_quat
            b["cube_vel"][i] = privileged.cube_vel
            b["cube_finger_force"][i] = privileged.cube_finger_force
            b["broken"][i] = privileged.broken
        # Every column took its value: commit the row.
        self._phases.setdefault(phase, index)
        self._last_t = frame.t
        self._calls += 1
        self._n = i + 1

    def snapshot(self) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
        """Copies of the observable columns recorded so far, and the phase names.

        For a decision in the middle of an attempt (T11's ``key_probe`` at
        ``PROBE_DONE``); the privileged columns are left out on purpose.
        """
        arrays = {name: self._buf[name][: self._n].copy() for name in SENSOR_COLUMNS}
        return arrays, tuple(self._phases)

    def add_events(self, events: Iterable[EventRecord]) -> None:
        self._check_open()
        clean = []
        for e in events:
            if not isinstance(e, EventRecord):
                raise TypeError(f"expected an EventRecord, got {type(e).__name__}")
            info = _json_dict("event info", e.info)
            clean.append(EventRecord(Event(e.name), float(e.t), info))
        self.events.extend(clean)

    def finish(self, result: str, success: bool | None = None, **info: object) -> Episode:
        """Freeze the episode. ``result`` is one of ``RESULTS``; ``success`` defaults to
        ``result == "SUCCESS"``. Extra keyword ``info`` (JSON-ready) joins the outcome."""
        self._check_open()
        if result not in RESULTS:
            raise ValueError(f"result must be one of {RESULTS}, got {result!r}")
        success = result == "SUCCESS" if success is None else success
        if not isinstance(success, bool):
            raise TypeError(f"success must be a bool, got {success!r}")
        outcome = _json_dict("outcome", {"result": result, "success": success, **info})
        arrays = {name: buf[: self._n].copy() for name, buf in self._buf.items()}
        self.finished = Episode(
            arrays,
            tuple(self._phases),
            tuple(self.events),
            self.params,
            outcome,
            dict(self.meta),
            dict(self.context),
            list(self.params_history),
        )
        return self.finished

    def _check_open(self) -> None:
        if self.finished is not None:
            raise RuntimeError("the episode is finished")

    def _grow(self) -> None:
        for name, buf in self._buf.items():
            bigger = np.empty((2 * buf.shape[0], *buf.shape[1:]), buf.dtype)
            bigger[: buf.shape[0]] = buf
            self._buf[name] = bigger


def _json_dict(name: str, value: Mapping | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return _jsonable(dict(value))  # type: ignore[return-value]
