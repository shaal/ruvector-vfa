"""Vector-store seam for memory: the ``MemoryBackend`` protocol and ``NumpyBackend`` (T12).

ARCHITECTURE §5.4. A backend holds ``(id, vector, payload)`` records and answers
top-k queries. ``MemoryBackend`` is the protocol cognition code talks to;
``NumpyBackend`` is the default (brute force, fine for thousands of records);
T22's ``RuVectorBackend`` runs the same calls through the Rust crate and must
return the same top-k.

``NumpyBackend(metric="cosine")`` stores every vector L2-normalized and scores
by the dot product, so the score is the cosine similarity: the Rust crate's
``recall_similar`` / ``find_similar`` rank by a raw dot product, and on unit
vectors the two agree (the parity rule of ARCHITECTURE §10). The dot product
sums left to right, as the crate's ``dot_product`` does, so equal vectors get
equal scores (a BLAS product does not guarantee that). With
``metric="euclidean"`` the score is the negative Euclidean distance (higher is
nearer), and vectors are stored as given.

``search(vector, k, filter)`` returns ``Hit``s by descending score; ties keep
insertion order. ``filter`` is ``None``, a mapping (every key must equal the
payload's value; a missing key never matches), or a callable on the payload
(it must not edit the payload). A hit's payload is a copy, so editing it never
changes the store. ``update(id, patch)`` merges a patch into a
payload. ``check(vector)`` validates a vector exactly as ``add`` would, so a
caller can validate several before it adds any; ``add`` must not fail for a
vector that passed ``check`` and a JSON payload (``EpisodicMemory.store``
relies on it to keep its key indexes in step). ``save(path)`` writes a new
or empty directory (it raises otherwise) with ``vectors.npy``,
``records.jsonl``, and ``meta.json``; ``NumpyBackend.load(path)`` reads it
back, and the same queries then return the same hits. Payloads must be plain JSON (numpy scalars
and arrays are converted; NaN is rejected).

RuVector mapping: the store behind ``memory_system::EpisodicMemory`` and
``SemanticMemory`` (a ``Vec<Episode>`` and a ``HashMap<String, Vec<f64>>``
scanned by dot product); ``ruvector-core``'s index is the scalable drop-in
(ARCHITECTURE §5.4, T22).
"""

from __future__ import annotations

import copy
import json
import math
import operator
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

METRICS: tuple[str, ...] = ("cosine", "euclidean")
UNIT_TOL = 1e-9
UNIT_EPS = 4 * np.finfo(float).eps  # a norm this close to 1 is left as is
NORM_RANGE = (1e-150, 1e150)  # inside it, the sum of squares is a normal float
Filter = Mapping[str, Any] | Callable[[Mapping[str, Any]], bool] | None


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: the record id, its score (higher is nearer), its payload."""

    id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryBackend(Protocol):
    """What memory needs from a vector store (refer to the module doc)."""

    def add(self, id: str, vector: np.ndarray, payload: Mapping[str, Any]) -> None: ...

    def check(self, vector: np.ndarray) -> np.ndarray: ...

    def search(self, vector: np.ndarray, k: int, filter: Filter = None) -> list[Hit]: ...

    def update(self, id: str, payload_patch: Mapping[str, Any]) -> None: ...

    def vector(self, id: str) -> np.ndarray: ...

    def payload(self, id: str) -> dict[str, Any]: ...

    @property
    def ids(self) -> list[str]: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> MemoryBackend: ...

    def __contains__(self, id: object) -> bool: ...

    def __len__(self) -> int: ...


class NumpyBackend:
    """Brute-force cosine or Euclidean search over records in numpy arrays."""

    def __init__(self, metric: str = "cosine", dim: int | None = None) -> None:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        self.metric = metric
        self.dim = dim
        self._buf = np.empty((0, dim or 0))  # capacity-doubling storage; rows [:len] used
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._payloads: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id: object) -> bool:
        return id in self._index

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def _vectors(self) -> np.ndarray:
        return self._buf[: len(self._ids)]

    def vector(self, id: str) -> np.ndarray:
        """The stored vector of ``id`` (unit norm for cosine)."""
        return self._vectors[self._index[id]].copy()

    def payload(self, id: str) -> dict[str, Any]:
        return copy.deepcopy(self._payloads[self._index[id]])

    def add(self, id: str, vector: np.ndarray, payload: Mapping[str, Any] | None = None) -> None:
        if not isinstance(id, str) or not id:
            raise ValueError(f"id must be a non-empty str, got {id!r}")
        if id in self._index:
            raise KeyError(f"id {id!r} is already stored")
        v = self.check(vector)
        payload = json_ready(payload or {})
        if self.dim is None:
            self.dim = v.size
            self._buf = np.empty((0, v.size))
        n = len(self._ids)
        if n == self._buf.shape[0]:
            grown = np.empty((max(16, 2 * n), self.dim))
            grown[:n] = self._buf[:n]
            self._buf = grown
        self._buf[n] = v
        self._index[id] = n
        self._ids.append(id)
        self._payloads.append(payload)

    def search(self, vector: np.ndarray, k: int, filter: Filter = None) -> list[Hit]:
        k = check_k(k)
        if not self._ids or k == 0:
            return []
        scores = self.scores(vector)
        if filter is None:
            order = np.argsort(-scores, kind="stable")[:k]
        else:
            filter = plain_filter(filter)
            kept = np.flatnonzero([matches(p, filter) for p in self._payloads])
            order = kept[np.argsort(-scores[kept], kind="stable")[:k]]
        return [
            Hit(self._ids[i], float(scores[i]), copy.deepcopy(self._payloads[i])) for i in order
        ]

    def scores(self, vector: np.ndarray) -> np.ndarray:
        """The score of every stored record for ``vector``, in insertion order."""
        q = self.check(vector)
        if not len(self._ids):
            return np.empty(0)
        if self.metric == "cosine":
            return np.cumsum(self._vectors * q, axis=1)[:, -1]  # left to right
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scores = -np.linalg.norm(self._vectors - q, axis=1)
        scale = np.maximum(np.max(np.abs(self._vectors), axis=1), np.max(np.abs(q)))
        odd = (scale > 0.0) & ((scale < NORM_RANGE[0]) | (scale > NORM_RANGE[1]))
        if odd.any():  # the squares over- or underflow: rescale those rows
            s = scale[odd, None]
            scores[odd] = -scale[odd] * np.linalg.norm(self._vectors[odd] / s - q / s, axis=1)
        return scores

    def update(self, id: str, payload_patch: Mapping[str, Any]) -> None:
        i = self._index[id]
        self._payloads[i] = {**self._payloads[i], **json_ready(payload_patch)}

    def save(self, path: str | Path) -> None:
        path = new_save_dir(path)
        meta = {"metric": self.metric, "dim": self.dim, "count": len(self)}
        atomic_write(path / "vectors.npy", lambda f: np.save(f, self._vectors))
        lines = "".join(
            json.dumps({"id": i, "payload": p}, allow_nan=False) + "\n"
            for i, p in zip(self._ids, self._payloads, strict=True)
        )
        atomic_write(path / "records.jsonl", lambda f: f.write(lines.encode()))
        atomic_write(path / "meta.json", lambda f: f.write(json.dumps(meta).encode()))

    @classmethod
    def load(cls, path: str | Path) -> NumpyBackend:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        backend = cls(meta["metric"], meta["dim"])
        vectors = np.load(path / "vectors.npy", allow_pickle=False)
        records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
        if len(records) != meta["count"] or vectors.shape[0] != meta["count"]:
            raise ValueError(
                f"{path}: {len(records)} records and {vectors.shape[0]} vectors,"
                f" meta.json says {meta['count']}"
            )
        if meta["count"]:
            backend._buf = vectors.astype(float, copy=True).reshape(meta["count"], -1)
        if len({r["id"] for r in records}) != len(records):
            raise ValueError(f"{path}: duplicate ids")
        for i, r in enumerate(records):
            backend._index[r["id"]] = i
            backend._ids.append(r["id"])
            backend._payloads.append(r["payload"])
        return backend

    def check(self, vector: np.ndarray) -> np.ndarray:
        """``vector`` as the store would hold it (unit norm for cosine); raises
        ``ValueError`` if ``add`` would reject it."""
        v = np.asarray(vector, dtype=float).reshape(-1)
        if self.dim is not None and v.size != self.dim:
            raise ValueError(f"vector must have {self.dim} entries, got {v.size}")
        if not np.all(np.isfinite(v)):
            raise ValueError("vector must be finite")
        if v.size == 0:
            raise ValueError("vector must not be empty")
        if self.metric == "cosine":
            scale = float(np.max(np.abs(v)))
            if scale == 0.0:
                raise ValueError("a cosine store cannot hold a zero vector")
            norm = float(np.linalg.norm(v))
            if not NORM_RANGE[0] < norm < NORM_RANGE[1]:  # the squares over- or underflow
                v = v / scale
                norm = float(np.linalg.norm(v))
            if abs(norm - 1.0) > UNIT_EPS:  # a stored vector re-added stays bit-identical
                v = v / norm
        return v


def check_k(k: object) -> int:
    """``k`` as an int (numpy ints accepted); raises ``ValueError`` unless it is >= 0."""
    try:
        n = operator.index(k)  # type: ignore[arg-type]
    except TypeError:
        n = -1
    if isinstance(k, bool) or n < 0:
        raise ValueError(f"k must be an int >= 0, got {k!r}")
    return n


def plain_filter(filter: Filter) -> Filter:
    """A mapping filter converted as payloads are (tuples to lists, numpy to
    Python), so it compares like with like; other filters unchanged."""
    return json_ready(filter) if isinstance(filter, Mapping) else filter


def matches(payload: Mapping[str, Any], filter: Filter) -> bool:
    """True if ``payload`` passes ``filter`` (refer to the module doc)."""
    if filter is None:
        return True
    if callable(filter):
        return bool(filter(payload))
    return all(k in payload and payload[k] == v for k, v in filter.items())


def json_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A plain-JSON copy of ``payload`` (numpy values converted, NaN rejected)."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    return json.loads(json.dumps(dict(payload), default=_plain, allow_nan=False))


def _plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("payload values must be finite")
        return item
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def new_save_dir(path: str | Path) -> Path:
    """Create directory ``path`` for a save; raises if it exists and is not empty
    (a save never overwrites another one in place)."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"{path}: save into a new or empty directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path: Path, write: Callable[[Any], object]) -> None:
    """Write ``path`` through a temporary file renamed into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)
