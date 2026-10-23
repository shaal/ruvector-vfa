"""Three-tier memory: working, episodic, semantic, with utility reinforcement (T12).

ARCHITECTURE §5.2–§5.4, mirroring ``ruvector_robotics::cognitive::memory_system``.

- ``WorkingMemory(max_size)``: a bounded buffer of ``MemoryItem``s for the current
  attempt (percepts, probe features, retrieved hits). When full, adding an item
  evicts the least important one; ``get`` counts accesses. The Rust
  ``WorkingMemory``, except that ``max_size`` must be at least 1.
- ``EpisodicMemory``: every attempt, as an ``EpisodeRecord`` (its keys, θ as
  chosen and as it ended, outcome, failure mode, opaque situation id, and
  ``utility``). Each key type has its own index (``KEY_TYPES``; a record joins
  the indexes of the keys it has: ``key_retry`` only for a failure,
  ``key_probe`` only if the probe finished). ``store`` validates every key
  before it adds any and keeps a plain-JSON copy of the record.
  ``recall_similar(key_type, query, k, success=None, filter=None)`` ranks by
  ``ranking_score``: ``similarity × (0.5 + utility)``, divided instead of
  multiplied when the similarity is negative, so a higher utility always ranks
  a record higher. ``reinforce(id, success)`` moves a used record's utility by
  ``UTILITY_STEP`` toward 1 or 0 (clamped to [0, 1]; a new record starts at
  0.5, where the score is the similarity). Recalled records are copies.
- ``SemanticMemory``: named concept vectors (T13's skill prototypes) with a
  payload; ``find_similar(query, k)`` ranks by dot product.
- ``MemorySystem``: the three tiers. ``save(path)`` writes episodic and
  semantic memory (working memory is per attempt and starts empty) to a new
  version directory ``path/v<n>/``, then points ``path/memory.json`` at it
  (one atomic file replace) and deletes the older versions and any stale
  temporary pointer file. A crash at any step leaves ``memory.json`` pointing
  at a complete save. ``save`` refuses a ``path`` that holds anything else
  (Finder's ``.DS_Store`` aside), and one process at a time may save to a
  path. The tier saves write only into a new or empty directory. A reload
  answers the same queries with the same hits. Every index comes from one
  ``backend_factory``; ``load`` refuses a directory saved by another backend
  class or metric.

Every vector store is a ``MemoryBackend`` (``NumpyBackend`` by default), with
cosine similarity on vectors stored at unit norm, so a dot product equals the
cosine (the Rust parity rule, ARCHITECTURE §10).

RuVector mapping: ``WorkingMemory`` and ``MemoryItem`` are the Rust types of the
same names; ``EpisodicMemory`` is ``memory_system::EpisodicMemory`` with one
instance per key type (the Rust ``Episode`` keeps one key as its single percept;
θ, ids, and the signed outcome ride in ``actions`` / ``reward`` and a side
table, T22); ``SemanticMemory`` is ``memory_system::SemanticMemory``. The
utility reinforcement is ours: the crate ranks by similarity alone.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from vfa.memory.backend import (
    Filter,
    MemoryBackend,
    NumpyBackend,
    atomic_write,
    check_k,
    json_ready,
    matches,
    new_save_dir,
    plain_filter,
)

KEY_TYPES: tuple[str, ...] = ("key_pre", "key_probe", "key_retry", "full_embedding")
INITIAL_UTILITY = 0.5
UTILITY_STEP = 0.1
SUCCESS_RESULT = "SUCCESS"
# What MemorySystem.save may find in its directory: the pointer, a temporary
# pointer file left by a crash (backend.atomic_write), the versions, and Finder's
# .DS_Store.
VERSION_DIR = re.compile(r"v\d{6,}(\.tmp)?")
POINTER_TMP = re.compile(r"\.memory\.json\.\d+\.tmp")
SAVE_ENTRY = re.compile(rf"memory\.json|\.DS_Store|{POINTER_TMP.pattern}|{VERSION_DIR.pattern}")


# ------------------------------------------------------------------- working


@dataclass(slots=True)
class MemoryItem:
    """One item of working memory (the Rust ``MemoryItem``)."""

    key: str
    data: list[float]
    importance: float
    timestamp: int
    access_count: int = 0


class WorkingMemory:
    """A bounded buffer: when full, ``add`` evicts the least important item first."""

    def __init__(self, max_size: int) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError(f"max_size must be an int >= 1, got {max_size!r}")
        self.max_size = max_size
        self.items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        if len(self.items) >= self.max_size:
            least = min(range(len(self.items)), key=lambda i: self.items[i].importance)
            del self.items[least]
        self.items.append(item)

    def get(self, key: str) -> MemoryItem | None:
        for item in self.items:
            if item.key == key:
                item.access_count += 1
                return item
        return None

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


# ------------------------------------------------------------------- episodic


@dataclass(slots=True)
class EpisodeRecord:
    """One attempt in episodic memory (vectors live in the key indexes)."""

    id: str
    situation: int  # the situation's opaque key: equal for every attempt at one cube
    attempt: int
    theta: dict[str, float]  # θ as chosen for the attempt (AttemptResult.theta)
    result: str  # SUCCESS or the failure mode
    utility: float = INITIAL_UTILITY
    timestamp: int = 0
    info: dict[str, Any] = field(default_factory=dict)  # e.g. probe features
    theta_final: dict[str, float] | None = None  # after escalations and reflexes; None: = theta

    @property
    def success(self) -> bool:
        return self.result == SUCCESS_RESULT

    @property
    def failure_mode(self) -> str | None:
        return None if self.success else self.result

    def payload(self) -> dict[str, Any]:
        """The index payload: every field but ``info`` (which may be large, e.g. a
        trajectory), plus ``success`` and ``failure_mode``."""
        data = asdict(self)
        del data["info"]
        return {**data, "success": self.success, "failure_mode": self.failure_mode}


@dataclass(frozen=True, slots=True)
class Recall:
    """A recalled record: its ``similarity`` to the query (the backend score: the
    cosine, or the negative Euclidean distance) and the ranking ``score``."""

    record: EpisodeRecord
    similarity: float
    score: float


BackendFactory = Callable[[], MemoryBackend]


def load_index(path: Path, backend_factory: BackendFactory) -> MemoryBackend:
    """Load an index saved by the backend ``backend_factory`` makes; raises if the
    saved metric differs from the factory's."""
    template = backend_factory()
    index = type(template).load(path)
    want, got = getattr(template, "metric", None), getattr(index, "metric", None)
    if want != got:
        raise ValueError(f"{path}: saved with metric {got!r}, the factory makes {want!r}")
    return index


def ranking_score(similarity: float, utility: float) -> float:
    """``similarity × (0.5 + utility)``; divided for a negative similarity, so the
    score always rises with both similarity and utility."""
    gain = 0.5 + utility
    return similarity * gain if similarity >= 0.0 else similarity / gain


class EpisodicMemory:
    """Every attempt, one index per key type (refer to the module doc)."""

    def __init__(self, backend_factory: BackendFactory = NumpyBackend) -> None:
        self.indexes: dict[str, MemoryBackend] = {k: backend_factory() for k in KEY_TYPES}
        self.records: dict[str, EpisodeRecord] = {}

    def __len__(self) -> int:
        return len(self.records)

    def store(self, record: EpisodeRecord, keys: Mapping[str, np.ndarray]) -> None:
        """Add a copy of ``record`` under each key it has (``keys``: key type → vector).

        Every key is validated before the first one is added, so a bad key
        leaves the memory unchanged.
        """
        record = _plain_record(record)
        if record.id in self.records:
            raise KeyError(f"record {record.id!r} is already stored")
        unknown = set(keys) - set(KEY_TYPES)
        if unknown:
            raise ValueError(f"unknown key types {sorted(unknown)}")
        for name, vector in keys.items():
            if record.id in self.indexes[name]:
                raise KeyError(f"index {name} already holds {record.id!r}")
            self.indexes[name].check(vector)
        payload = record.payload()
        for name, vector in keys.items():
            self.indexes[name].add(record.id, vector, payload)
        self.records[record.id] = record

    def recall_similar(
        self,
        key_type: str,
        query: np.ndarray,
        k: int,
        success: bool | None = None,
        filter: Filter = None,
    ) -> list[Recall]:
        """The ``k`` records ranked by ``ranking_score`` under ``key_type`` (refer to
        the module doc); ties keep the similarity order.

        ``success`` keeps only successes (True) or failures (False); ``filter``
        is a payload mapping or callable, as in ``MemoryBackend.search``.
        """
        index = self.indexes[key_type]
        k = check_k(k)
        if success is not None and not isinstance(success, bool | np.bool_):
            raise TypeError(f"success must be a bool or None, got {success!r}")
        filter = plain_filter(filter)
        if k == 0 or not len(index):
            index.check(query)
            return []

        def keep(payload: Mapping[str, Any]) -> bool:
            if success is not None and bool(payload["success"]) != bool(success):
                return False
            return matches(payload, filter)

        # Search the nearest m by similarity; widen until no record beyond them can
        # outrank the k-th (none scores above ranking_score(sim, highest utility)).
        top_utility = max(r.utility for r in self.records.values())
        m = min(len(index), max(4 * k, 32))
        while True:
            hits = index.search(query, m, None if success is None and filter is None else keep)
            ranked = sorted(
                (
                    (ranking_score(h.score, self.records[h.id].utility), i)
                    for i, h in enumerate(hits)
                ),
                key=lambda pair: -pair[0],
            )
            if len(hits) < m or m == len(index):
                break
            bound = ranking_score(hits[-1].score, top_utility)
            if len(ranked) >= k and ranked[k - 1][0] >= bound:
                break
            m = min(len(index), 2 * m)
        return [
            Recall(copy.deepcopy(self.records[hits[i].id]), hits[i].score, score)
            for score, i in ranked[:k]
        ]

    def reinforce(self, id: str, success: bool, step: float = UTILITY_STEP) -> float:
        """Move ``id``'s utility by ``step`` toward 1 (success) or 0; returns the new value."""
        if not isinstance(success, bool | np.bool_):
            raise TypeError(f"success must be a bool, got {success!r}")
        if not (math.isfinite(step) and step >= 0.0):
            raise ValueError(f"step must be finite and >= 0, got {step!r}")
        record = self.records[id]
        delta = step if success else -step
        record.utility = round(min(1.0, max(0.0, record.utility + delta)), 12)
        for index in self.indexes.values():
            if id in index:
                index.update(id, {"utility": record.utility})
        return record.utility

    def save(self, path: Path) -> None:
        """Save into a new or empty directory (``MemorySystem.save`` handles
        replacing an earlier save)."""
        path = new_save_dir(path)
        for name, index in self.indexes.items():
            index.save(path / name)
        lines = "".join(
            json.dumps(asdict(r), allow_nan=False) + "\n" for r in self.records.values()
        )
        atomic_write(path / "records.jsonl", lambda f: f.write(lines.encode()))

    @classmethod
    def load(cls, path: Path, backend_factory: BackendFactory = NumpyBackend) -> EpisodicMemory:
        memory = cls(backend_factory)
        memory.indexes = {name: load_index(path / name, backend_factory) for name in KEY_TYPES}
        for line in (path / "records.jsonl").read_text().splitlines():
            record = _plain_record(EpisodeRecord(**json.loads(line)))
            if record.id in memory.records:
                raise ValueError(f"{path}: record {record.id!r} appears twice")
            memory.records[record.id] = record
        for name, index in memory.indexes.items():
            stray = set(index.ids) - set(memory.records)
            if stray:
                raise ValueError(f"{path / name}: ids without a record, e.g. {min(stray)!r}")
        return memory


# ------------------------------------------------------------------- semantic


class SemanticMemory:
    """Named concept vectors (skill prototypes); ``find_similar`` by dot product."""

    def __init__(self, backend_factory: BackendFactory = NumpyBackend) -> None:
        self._factory = backend_factory
        self.index: MemoryBackend = backend_factory()

    def __len__(self) -> int:
        return len(self.index)

    def store(
        self, name: str, embedding: np.ndarray, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Store (or replace) concept ``name``; a failed replacement keeps the old one."""
        payload = json_ready(payload or {})
        self.index.check(embedding)
        if name in self.index:  # rebuild without it, then swap the new index in
            fresh = self._factory()
            for other in self.index.ids:
                if other != name:
                    fresh.add(other, self.index.vector(other), self.index.payload(other))
            fresh.add(name, embedding, payload)
            self.index = fresh
        else:
            self.index.add(name, embedding, payload)

    def retrieve(self, name: str) -> tuple[np.ndarray, dict[str, Any]] | None:
        if name not in self.index:
            return None
        return self.index.vector(name), self.index.payload(name)

    def find_similar(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        return [(h.id, h.score) for h in self.index.search(query, k)]

    def clear(self) -> None:
        """Drop every concept (e.g. before a fresh consolidation)."""
        self.index = self._factory()

    def rebuild(self, concepts: Sequence[tuple[str, np.ndarray, Mapping[str, Any]]]) -> None:
        """Replace every concept with ``concepts`` (name, embedding, payload) at once;
        on any error the old concepts stay."""
        fresh = self._factory()
        for name, embedding, payload in concepts:
            fresh.add(name, embedding, payload)
        self.index = fresh

    def save(self, path: Path) -> None:
        self.index.save(path)

    @classmethod
    def load(cls, path: Path, backend_factory: BackendFactory = NumpyBackend) -> SemanticMemory:
        memory = cls(backend_factory)
        memory.index = load_index(path, backend_factory)
        return memory


# ------------------------------------------------------------------- facade


class MemorySystem:
    """Working, episodic, and semantic memory; saved to and loaded from one directory
    (refer to the module doc)."""

    def __init__(
        self, working_size: int = 32, backend_factory: BackendFactory = NumpyBackend
    ) -> None:
        self.working = WorkingMemory(working_size)
        self.episodic = EpisodicMemory(backend_factory)
        self.semantic = SemanticMemory(backend_factory)
        self._backend = type(self.semantic.index).__name__

    def save(self, path: str | Path) -> Path:
        """Save into directory ``path`` (refer to the module doc); returns ``path``."""
        path = Path(path)
        if path.exists() and not path.is_dir():
            raise ValueError(f"{path} is not a directory")
        path.mkdir(parents=True, exist_ok=True)
        foreign = sorted(
            e.name
            for e in path.iterdir()
            if not SAVE_ENTRY.fullmatch(e.name)
            or (VERSION_DIR.fullmatch(e.name) and not e.is_dir())
        )
        if foreign:
            raise ValueError(
                f"{path} holds other files {foreign[:3]}: give memory its own directory"
            )
        current = _current_version(path)
        version = f"v{(int(current[1:]) + 1 if current else 0):06d}"
        for leftover in (path / version, path / f"{version}.tmp"):  # from a crashed save
            if leftover.exists():
                shutil.rmtree(leftover)
        tmp = path / f"{version}.tmp"
        self.episodic.save(tmp / "episodic")
        self.semantic.save(tmp / "semantic")
        os.replace(tmp, path / version)
        meta = {
            "version": version,
            "working_size": self.working.max_size,
            "episodes": len(self.episodic),
            "backend": self._backend,
        }
        atomic_write(path / "memory.json", lambda f: f.write(json.dumps(meta).encode()))
        for entry in path.iterdir():  # older versions, now unreferenced, and stale pointers
            if entry.is_dir() and entry.name != version and VERSION_DIR.fullmatch(entry.name):
                shutil.rmtree(entry)
            elif entry.is_file() and POINTER_TMP.fullmatch(entry.name):
                entry.unlink()
        return path

    @classmethod
    def load(cls, path: str | Path, backend_factory: BackendFactory = NumpyBackend) -> MemorySystem:
        path = Path(path)
        meta = json.loads((path / "memory.json").read_text())
        system = cls(meta["working_size"], backend_factory)
        if meta["backend"] != system._backend:
            raise ValueError(
                f"{path}: saved by {meta['backend']}, the factory makes {system._backend}"
            )
        system.episodic = EpisodicMemory.load(path / meta["version"] / "episodic", backend_factory)
        system.semantic = SemanticMemory.load(path / meta["version"] / "semantic", backend_factory)
        if len(system.episodic) != meta["episodes"]:
            raise ValueError(
                f"{path}: {len(system.episodic)} episodes, expected {meta['episodes']}"
            )
        return system


def _current_version(path: Path) -> str | None:
    """The version ``memory.json`` points to, or None before the first save."""
    meta = path / "memory.json"
    return json.loads(meta.read_text())["version"] if meta.is_file() else None


def _plain_record(record: EpisodeRecord) -> EpisodeRecord:
    """A plain-JSON copy of ``record`` (numpy values converted); validates utility."""
    utility = float(record.utility)
    if not 0.0 <= utility <= 1.0:
        raise ValueError(f"utility must be in [0, 1], got {record.utility!r}")
    return replace(
        record,
        id=str(record.id),
        situation=int(record.situation),
        attempt=int(record.attempt),
        theta=json_ready(record.theta),
        result=str(record.result),
        utility=utility,
        timestamp=int(record.timestamp),
        info=json_ready(record.info),
        theta_final=None if record.theta_final is None else json_ready(record.theta_final),
    )


def unit_norm(vectors: Sequence[np.ndarray]) -> bool:
    """True if every vector has unit length (the parity rule)."""
    return all(math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=1e-9) for v in vectors)
