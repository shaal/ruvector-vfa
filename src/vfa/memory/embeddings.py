"""Force, trajectory, and context embeddings: retrieval keys for memory (T11, ARCHITECTURE §5.1).

An attempt becomes vectors where "similar physics" means "nearby". Memory
stores what existed at each decision point, so each key uses only that:

| Key | Groups | Dims | When |
|---|---|---|---|
| ``key_pre`` | context | 12 | attempt start |
| ``key_probe`` | context, probe, probe_profile | 50 | after ``PROBE_DONE`` |
| ``key_retry`` | context, force_wrist, force_moments, failure | 213 | a retry |
| ``full_embedding`` | every group but probe_profile and failure | 354 | analysis |

**Views.** Every feature reads an ``EpisodeView``: the observable signals at
50 Hz (``VIEW_HZ``; the rate the fixture stores), the observed context, the probe
features, θ, and the outcome. A view comes from a finished ``Episode``
(``from_episode``), from a recorder in the middle of an attempt
(``from_recorder``), or from the fixture (``from_fixture``); recordings at any
rate that is a multiple of 50 Hz are decimated to it (the rate is read from the
timestamps), so live attempts and the fixture embed the same way. No view holds
a privileged column, the hidden properties, or an event's ``info``.

**Groups** (raw features; ``DIMS``):

- ``context`` (12): the observed cube x, y, z, yaw, size and the initial arm
  joints (7; the env starts every episode at the same joints, so they carry
  nothing yet and standardize to 0).
- ``probe`` (6): the probe's heft load and peak [N], log10 of the indentation
  [mm], whether and when [s] it slid, and whether it dropped. The closure-force
  slope is ``2·F/indentation`` (``ProbeFeatures``; T08 measures stiffness from
  the settled opening, not from a ramp), so its log carries the indentation's
  information.
- ``probe_profile`` (32): the PROBE phase's wrist Fz and ‖Fxy‖, each resampled to
  16 points. Its torque mostly encodes the cube's pose and its pad force the
  commanded probe force, so neither is in ``key_probe`` (both stay in the
  ``force_wrist`` and ``force_pad`` groups).
- ``force_wrist`` (192): Fz, ‖Fxy‖, and ‖τ‖ in PROBE, SQUEEZE, LIFT, and HOLD
  (3 × 4 × 16; a phase the attempt never reached is missing).
- ``force_pad`` (65): the summed pad force in those phases (4 × 16) and its peak.
  After the probe it follows the commanded grip (θ), not the object, so it is in
  ``full_embedding`` only: in ``key_retry`` it would make a failure's nearest
  successes the ones at the grip that just failed.
- ``force_moments`` (5): contact onset [s from the probe's start; mostly the
  cube's size], log closure slope, heft load, and the counts of ``SLIP`` and
  ``DROP`` events.
- ``trajectory`` (64): the ee path relative to the observed cube (16 × 3) and
  its speed (16), from the first motion to the end.
- ``failure`` (4): one-hot BREAK, under-grip (SLIP or DROP), MISS, other.
  SLIP and DROP merge because ``run_attempt`` labels an expired hold SLIP or
  DROP by the cube's true height.
- ``outcome`` (10): one-hot SUCCESS, BREAK, under-grip, MISS, other, and θ
  scaled to its bounds.

``force_features`` is ``force_wrist`` + ``force_pad`` + ``force_moments`` (262).

**Scaling.** Each feature is standardized with the mean and standard deviation
fitted on the calibration fixture (``configs/embedding_stats.npz``, written by
``python -m vfa.memory.embeddings fit``) and clipped to ±``Z_CLIP``, so one
outlying sample cannot fill a key; a missing feature becomes 0, the fitted mean.
Each group is divided by ``sqrt(dims)`` (equal expected weight per group) and
multiplied by its weight, then the key is L2-normalized, so a dot product is a
cosine. The weights: context 0.5 (the observed pose and size carry no class
information by construction); probe 2, because its six scalars separate the
classes whose grips conflict (brittle from heavy_rigid by the heft load,
light_rigid from fragile_soft by the indentation) far better than the 32-sample
profile, which would otherwise hold about half of ``key_probe``'s norm (chosen
on the quick tuning seeds by a rule fixed before one scored check on fresh
seeds, in T11's review); the others 1. In ``key_retry`` the failure one-hot is a
large share of the norm: neighbours mostly share the failure type. A failed
attempt's post-probe wrist profile resembles no success's, so ``key_retry`` is
for failure neighbours; success neighbours on a retry come from the failed
attempt's ``key_probe``. Similarities are not comparable across keys
(``key_pre`` ranks every cube of the same size and pose alike): a retrieval
policy needs a threshold per key.

RuVector mapping: the keys are the ``percepts`` of the Rust
``memory_system::Episode``: ``EpisodicMemory::recall_similar`` flattens them and
ranks by a raw dot product, which on these unit vectors is the cosine
similarity (ARCHITECTURE §5.4, §10).
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vfa.skills.params import FIELDS, HIGH, LOW

REPO_ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = REPO_ROOT / "configs" / "embedding_stats.npz"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "calib_episodes.npz"
VIEW_HZ = 50.0  # view rows per second, as the fixture stores them
SAMPLES = 16  # resampled points per signal and phase
FORCE_PHASES: tuple[str, ...] = ("PROBE", "SQUEEZE", "LIFT", "HOLD")
MOTION_PHASES = frozenset({"APPROACH", "DESCEND", "PROBE", "SQUEEZE", "LIFT", "HOLD"})
# vfa.sim.env.START_ARM_QPOS (tests check they agree); not imported, to keep Gymnasium
# out of the memory package's imports.
START_ARM_QPOS: tuple[float, ...] = (0.0, -0.03493, 0.0, -1.922671, 0.0, 1.887741, -0.785398)
CONTACT_N = 0.2  # summed pad force above this is a contact (a pad's flag turns on at 0.2 N)
FAILURES: tuple[str, ...] = ("BREAK", "UNDERGRIP", "MISS", "OTHER")
OUTCOMES: tuple[str, ...] = ("SUCCESS", *FAILURES)
UNDERGRIP = frozenset({"SLIP", "DROP"})
DIMS: dict[str, int] = {
    "context": 12,
    "probe": 6,
    "probe_profile": 2 * SAMPLES,
    "force_wrist": 3 * len(FORCE_PHASES) * SAMPLES,
    "force_pad": len(FORCE_PHASES) * SAMPLES + 1,
    "force_moments": 5,
    "trajectory": 4 * SAMPLES,
    "failure": len(FAILURES),
    "outcome": len(OUTCOMES) + len(FIELDS),
}
KEYS: dict[str, tuple[str, ...]] = {
    "key_pre": ("context",),
    "key_probe": ("context", "probe", "probe_profile"),
    "key_retry": ("context", "force_wrist", "force_moments", "failure"),
    "full_embedding": (
        "context",
        "probe",
        "force_wrist",
        "force_pad",
        "force_moments",
        "trajectory",
        "outcome",
    ),
}
WEIGHTS: dict[str, float] = {
    "context": 0.5,
    "probe": 2.0,
    "probe_profile": 1.0,
    "force_wrist": 1.0,
    "force_pad": 1.0,
    "force_moments": 1.0,
    "trajectory": 1.0,
    "failure": 1.0,
    "outcome": 1.0,
}
GROUPS: tuple[str, ...] = tuple(DIMS)
MIN_STD = 1e-6
Z_CLIP = 5.0


# ------------------------------------------------------------------------ view


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """The observable record of one attempt at 50 Hz (refer to the module doc)."""

    t: np.ndarray  # (n,) [s]
    phase: np.ndarray  # (n,) phase names
    wrench: np.ndarray  # (n, 6) tared, low-passed wrist wrench (world axes)
    finger: np.ndarray  # (n,) summed pad force [N]
    ee: np.ndarray  # (n, 3) ee position [m]
    context: np.ndarray  # (12,) cube x, y, z, yaw, size + 7 initial joints
    probe: Mapping[str, Any] | None  # ProbeFeatures as a dict, or None (no PROBE_DONE)
    theta: np.ndarray  # (5,) θ that ran at the end (FIELDS order)
    result: str | None  # None while the attempt runs
    slip_events: int = 0
    drop_events: int = 0

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        phases: Sequence[str],
        context: Mapping[str, Any],
        *,
        probe: Any = None,
        theta: Mapping[str, float] | np.ndarray | None = None,
        result: str | None = None,
        slip_events: int = 0,
        drop_events: int = 0,
    ) -> EpisodeView:
        """From recorder columns (``t``, ``phase``, ``wrench``, ``finger_force``,
        ``ee_pos``) at any rate that is a multiple of 50 Hz, decimated to 50 Hz."""
        t = np.asarray(arrays["t"], dtype=float)
        keep = slice(None, None, _decimation(t))
        names = np.asarray(phases, dtype=object)[np.asarray(arrays["phase"])[keep]]
        return cls(
            t=t[keep],
            phase=names.astype(str),
            wrench=np.asarray(arrays["wrench"], dtype=float)[keep],
            finger=np.sum(np.asarray(arrays["finger_force"], dtype=float)[keep], axis=1),
            ee=np.asarray(arrays["ee_pos"], dtype=float)[keep],
            context=context_features(context),
            probe=_probe_dict(probe),
            theta=_theta_vector(theta),
            result=result,
            slip_events=int(slip_events),
            drop_events=int(drop_events),
        )

    @classmethod
    def from_episode(cls, episode: Any) -> EpisodeView:
        """From a finished ``vfa.sim.recorder.Episode`` (its observable columns only)."""
        return cls.from_arrays(
            episode.arrays,
            episode.phases,
            episode.context,
            probe=episode.meta.get("probe"),
            theta=episode.params or None,
            result=episode.result,
            slip_events=len(episode.events_named("SLIP")),
            drop_events=len(episode.events_named("DROP")),
        )

    @classmethod
    def from_recorder(cls, recorder: Any, probe: Any = None) -> EpisodeView:
        """From an ``EpisodeRecorder`` in the middle of an attempt (e.g. at ``PROBE_DONE``;
        ``probe`` may be the escalation's ``ProbeFeatures``)."""
        arrays, phases = recorder.snapshot()
        events = recorder.events
        return cls.from_arrays(
            arrays,
            phases,
            recorder.context,
            probe=probe if probe is not None else recorder.meta.get("probe"),
            theta=recorder.params or None,
            slip_events=sum(e.name == "SLIP" for e in events),
            drop_events=sum(e.name == "DROP" for e in events),
        )

    @classmethod
    def from_fixture(cls, fixture: Mapping[str, np.ndarray], index: int) -> EpisodeView:
        """Episode ``index`` of the calibration fixture (``scripts/make_fixtures.py``)."""
        rows = fixture["rows"][fixture["offsets"][index] : fixture["offsets"][index + 1]]
        rows = rows.astype(float)
        col = {name: i for i, name in enumerate(fixture["row_columns"])}
        names = np.asarray(fixture["phase_names"])[rows[:, col["phase"]].astype(int)]
        wrench = rows[:, [col[c] for c in ("fx", "fy", "fz", "tx", "ty", "tz")]]
        values = fixture["probe"][index]
        probe = None
        if not np.all(np.isnan(values)):
            probe = {
                str(k): (None if np.isnan(v) else float(v))
                for k, v in zip(fixture["probe_columns"], values, strict=True)
            }
        ctx = dict(zip(fixture["context_columns"], fixture["context"][index], strict=True))
        return cls(
            t=rows[:, col["t"]],
            phase=names.astype(str),
            wrench=wrench,
            finger=rows[:, col["finger_left"]] + rows[:, col["finger_right"]],
            ee=rows[:, [col["ee_x"], col["ee_y"], col["ee_z"]]],
            context=context_features({"cube": ctx}),
            probe=probe,
            theta=np.asarray(fixture["theta_final"][index], dtype=float),
            result=str(fixture["result"][index]),
            slip_events=int(fixture["slip_events"][index]),
            drop_events=int(fixture["drop_events"][index]),
        )


def _decimation(t: np.ndarray) -> int:
    """Rows to skip so that ``t`` (a recording's timestamps) lands at 50 Hz."""
    if t.size < 2:
        return 1
    step = float(np.median(np.diff(t)))
    if not step > 0:
        raise ValueError("recording timestamps must increase")
    k = round(1.0 / (VIEW_HZ * step))
    if k < 1 or abs(k * step * VIEW_HZ - 1.0) > 1e-3:
        raise ValueError(f"a recording at {1.0 / step:.4g} Hz cannot be decimated to 50 Hz")
    return k


def _probe_dict(probe: Any) -> dict[str, Any] | None:
    if probe is None:
        return None
    if hasattr(probe, "to_dict"):  # ProbeFeatures
        probe = probe.to_dict()
    return {
        str(k): (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in dict(probe).items()
    }


def _theta_vector(theta: Mapping[str, float] | np.ndarray | None) -> np.ndarray:
    if theta is None:
        return np.full(len(FIELDS), np.nan)
    if isinstance(theta, Mapping):
        return np.array([float(theta[f]) for f in FIELDS])
    return np.asarray(theta, dtype=float).reshape(len(FIELDS))


# -------------------------------------------------------------------- features


def context_features(source: Mapping[str, Any]) -> np.ndarray:
    """Raw context (12): an observation (``cube_pos``, ``cube_yaw``, ``cube_size``,
    ``joint_pos``) or a recorder context (``cube``: x, y, z, yaw, size; ``arm_qpos``)."""
    if "cube_pos" in source:
        pos = np.asarray(source["cube_pos"], dtype=float).reshape(3)
        yaw = float(np.asarray(source["cube_yaw"]).reshape(-1)[0])
        size = float(np.asarray(source["cube_size"]).reshape(-1)[0])
        joints = np.asarray(source.get("joint_pos", START_ARM_QPOS), dtype=float)
    else:
        cube = source["cube"]
        pos = np.array([cube["x"], cube["y"], cube["z"]], dtype=float)
        yaw, size = float(cube["yaw"]), float(cube["size"])
        joints = np.asarray(source.get("arm_qpos", START_ARM_QPOS), dtype=float)
    return np.concatenate([pos, [yaw, size], joints.reshape(7)])


def _resample(values: np.ndarray, n: int = SAMPLES) -> np.ndarray:
    """Linear resampling of each row of ``values`` to ``n`` points over its span."""
    k = values.shape[-1]
    if k == 1:
        return np.repeat(values, n, axis=-1)
    grid = np.linspace(0.0, k - 1.0, n)
    return np.stack([np.interp(grid, np.arange(k), row) for row in values])


def _phase_rows(view: EpisodeView, phase: str) -> np.ndarray:
    rows = np.flatnonzero(view.phase == phase)
    return rows[np.all(np.isfinite(view.wrench[rows]), axis=1)]


def _wrist(view: EpisodeView, phase: str, torque: bool = True) -> np.ndarray:
    """Fz, ‖Fxy‖ (and ‖τ‖) of ``phase`` resampled to 16 points each (NaN if absent)."""
    n = 3 if torque else 2
    rows = _phase_rows(view, phase)
    if rows.size == 0:
        return np.full(n * SAMPLES, np.nan)
    w = view.wrench[rows]
    signals = [w[:, 2], np.hypot(w[:, 0], w[:, 1])]
    if torque:
        signals.append(np.linalg.norm(w[:, 3:], axis=1))
    return _resample(np.stack(signals)).reshape(-1)


def _pad(view: EpisodeView, phase: str) -> np.ndarray:
    rows = _phase_rows(view, phase)
    if rows.size == 0:
        return np.full(SAMPLES, np.nan)
    return _resample(view.finger[rows][None, :])[0]


def _contact_onset(view: EpisodeView) -> float:
    """Seconds from the probe's start to the first summed pad force above ``CONTACT_N``."""
    rows = np.flatnonzero(view.phase == "PROBE")
    if rows.size == 0:
        return math.nan
    touch = rows[view.finger[rows] > CONTACT_N]
    return float(view.t[touch[0]] - view.t[rows[0]]) if touch.size else math.nan


def _probe_value(view: EpisodeView, key: str) -> float:
    if view.probe is None:
        return math.nan
    v = view.probe.get(key)
    return math.nan if v is None else float(v)


def probe_scalars(view: EpisodeView) -> np.ndarray:
    """Raw probe group (6); NaN where the probe did not finish."""
    indentation = _probe_value(view, "indentation")
    log_ind = math.log10(indentation * 1e3) if indentation > 0 else math.nan
    return np.array(
        [
            _probe_value(view, "heft_load"),
            _probe_value(view, "heft_peak"),
            log_ind,
            _probe_value(view, "heft_slip"),
            _probe_value(view, "heft_slip_time"),
            _probe_value(view, "heft_drop"),
        ]
    )


def probe_profile(view: EpisodeView) -> np.ndarray:
    """Raw probe_profile group (32): the PROBE phase's Fz and ‖Fxy‖."""
    return _wrist(view, "PROBE", torque=False)


def force_groups(view: EpisodeView) -> dict[str, np.ndarray]:
    """Raw ``force_wrist``, ``force_pad``, and ``force_moments`` groups."""
    finger = view.finger[np.isin(view.phase, FORCE_PHASES)]
    slope = _probe_value(view, "closure_slope")
    peak = float(np.max(finger)) if finger.size else math.nan
    return {
        "force_wrist": np.concatenate([_wrist(view, p) for p in FORCE_PHASES]),
        "force_pad": np.concatenate([*(_pad(view, p) for p in FORCE_PHASES), [peak]]),
        "force_moments": np.array(
            [
                _contact_onset(view),
                math.log10(slope) if slope > 0 else math.nan,
                _probe_value(view, "heft_load"),
                float(view.slip_events),
                float(view.drop_events),
            ]
        ),
    }


def _as_view(source: Any) -> EpisodeView:
    return source if isinstance(source, EpisodeView) else EpisodeView.from_episode(source)


def force_features(episode: Any) -> np.ndarray:
    """Raw force features (262) of an ``Episode`` or ``EpisodeView``:
    ``force_wrist`` + ``force_pad`` + ``force_moments``."""
    groups = force_groups(_as_view(episode))
    return np.concatenate([groups["force_wrist"], groups["force_pad"], groups["force_moments"]])


def trajectory_features(episode: Any) -> np.ndarray:
    """Raw trajectory group (64) of an ``Episode`` or ``EpisodeView``: the ee path
    relative to the observed cube, and its speed."""
    view = _as_view(episode)
    rows = np.flatnonzero(np.isin(view.phase, list(MOTION_PHASES)))
    if rows.size < 2:
        return np.full(DIMS["trajectory"], math.nan)
    rel = view.ee[rows] - view.context[:3]
    speed = np.linalg.norm(np.diff(view.ee[rows], axis=0), axis=1) / np.diff(view.t[rows])
    return np.concatenate([_resample(rel.T).reshape(-1), _resample(speed[None, :])[0]])


def _bucket(result: str | None) -> str | None:
    if result is None:
        return None
    if result in ("SUCCESS", "BREAK", "MISS"):
        return result
    return "UNDERGRIP" if result in UNDERGRIP else "OTHER"


def failure_features(view: EpisodeView) -> np.ndarray:
    """Raw failure group (4): one-hot of the failure (all 0 for a success or no result)."""
    out = np.zeros(DIMS["failure"])
    bucket = _bucket(view.result)
    if bucket in FAILURES:
        out[FAILURES.index(bucket)] = 1.0
    return out


def outcome_features(view: EpisodeView) -> np.ndarray:
    """Raw outcome group (10): one-hot result and θ scaled to its bounds."""
    onehot = np.zeros(len(OUTCOMES))
    bucket = _bucket(view.result)
    if bucket is not None:
        onehot[OUTCOMES.index(bucket)] = 1.0
    return np.concatenate([onehot, (view.theta - LOW) / (HIGH - LOW)])


def raw_groups(view: EpisodeView) -> dict[str, np.ndarray]:
    """Every raw group of ``view`` (NaN marks a missing feature)."""
    return {
        "context": view.context,
        "probe": probe_scalars(view),
        "probe_profile": probe_profile(view),
        **force_groups(view),
        "trajectory": trajectory_features(view),
        "failure": failure_features(view),
        "outcome": outcome_features(view),
    }


# ---------------------------------------------------------------------- scaling


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    """Per-feature mean and standard deviation of every group."""

    mean: Mapping[str, np.ndarray]
    std: Mapping[str, np.ndarray]

    @classmethod
    def fit(cls, views: Sequence[EpisodeView]) -> EmbeddingStats:
        raws = [raw_groups(v) for v in views]
        mean, std = {}, {}
        for g in GROUPS:
            x = np.stack([r[g] for r in raws])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # an all-missing feature
                m = np.nanmean(x, axis=0)
                s = np.nanstd(x, axis=0)
            mean[g] = np.nan_to_num(m, nan=0.0)
            std[g] = np.where(np.isfinite(s) & (s > MIN_STD), s, 1.0)
        return cls(mean, std)

    def save(self, path: str | Path = STATS_PATH) -> Path:
        path = Path(path)
        arrays = {f"{g}_mean": self.mean[g] for g in GROUPS}
        arrays |= {f"{g}_std": self.std[g] for g in GROUPS}
        np.savez(path, groups=np.array(GROUPS), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path = STATS_PATH) -> EmbeddingStats:
        with np.load(path, allow_pickle=False) as f:
            if tuple(f["groups"]) != GROUPS:
                raise ValueError(f"{path}: groups {tuple(f['groups'])} != {GROUPS}")
            mean = {g: f[f"{g}_mean"] for g in GROUPS}
            std = {g: f[f"{g}_std"] for g in GROUPS}
        for g in GROUPS:
            if mean[g].shape != (DIMS[g],) or std[g].shape != (DIMS[g],):
                raise ValueError(f"{path}: group {g} has the wrong dimension")
        return cls(mean, std)


class Embedder:
    """Builds the unit-norm keys from raw groups (refer to the module doc)."""

    def __init__(
        self, stats: EmbeddingStats | None = None, weights: Mapping[str, float] | None = None
    ) -> None:
        self.stats = stats if stats is not None else EmbeddingStats.load()
        self.weights = dict(WEIGHTS if weights is None else weights)

    def scaled(self, group: str, raw: np.ndarray) -> np.ndarray:
        """``raw`` standardized and clipped, missing features at 0, scaled by
        weight / sqrt(dims)."""
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (DIMS[group],):
            raise ValueError(f"{group}: expected {DIMS[group]} features, got {raw.shape}")
        z = (raw - self.stats.mean[group]) / self.stats.std[group]
        z = np.clip(np.where(np.isfinite(z), z, 0.0), -Z_CLIP, Z_CLIP)
        return z * (self.weights[group] / math.sqrt(DIMS[group]))

    def key(self, name: str, groups: Mapping[str, np.ndarray]) -> np.ndarray:
        """The unit-norm key ``name`` (``KEYS``) from raw groups."""
        parts = [self.scaled(g, groups[g]) for g in KEYS[name]]
        return _unit(np.concatenate(parts))

    def key_pre(self, context: Mapping[str, Any] | np.ndarray) -> np.ndarray:
        raw = context if isinstance(context, np.ndarray) else context_features(context)
        return self.key("key_pre", {"context": raw})

    def key_probe(self, view: EpisodeView) -> np.ndarray:
        groups = {
            "context": view.context,
            "probe": probe_scalars(view),
            "probe_profile": probe_profile(view),
        }
        return self.key("key_probe", groups)

    def key_retry(self, previous: EpisodeView) -> np.ndarray:
        """The key for a retry: its context (the same cube) and the failed attempt."""
        groups = {
            "context": previous.context,
            **force_groups(previous),
            "failure": failure_features(previous),
        }
        return self.key("key_retry", groups)

    def full_embedding(self, view: EpisodeView) -> np.ndarray:
        return self.key("full_embedding", raw_groups(view))


def key_dims(name: str) -> int:
    return sum(DIMS[g] for g in KEYS[name])


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        out = np.zeros_like(vector)
        out[0] = 1.0  # a degenerate key (every feature at its mean): a fixed unit vector
        return out
    return vector / norm


# -------------------------------------------------------------------- fixture


def load_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def fixture_views(fixture: Mapping[str, np.ndarray]) -> list[EpisodeView]:
    return [EpisodeView.from_fixture(fixture, i) for i in range(len(fixture["offsets"]) - 1)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fit configs/embedding_stats.npz on the fixture")
    parser.add_argument("command", choices=["fit"])
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--out", type=Path, default=STATS_PATH)
    args = parser.parse_args(argv)
    views = fixture_views(load_fixture(args.fixture))
    path = EmbeddingStats.fit(views).save(args.out)
    print(f"wrote {path} from {len(views)} fixture episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
