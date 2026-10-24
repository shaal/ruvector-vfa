"""``RuVectorBackend``: memory's vector store run through the real Rust crate (T22).

ARCHITECTURE §5.4, §10; ``docs/RUVECTOR_MAPPING.md`` §3. The ranking runs in
``ruvector-robotics``' ``EpisodicMemory`` (``recall_similar``), reached through
the PyO3 binding in ``rust/ruvector_vfa_py`` (build it with ``uv sync --extra
ruvector``, then ``uv run maturin develop --release -m
rust/ruvector_vfa_py/Cargo.toml``; a sync removes it again, and a debug build
searches many times slower).

**Mapping.** Each ``add(id, vector, payload)`` stores one Rust ``Episode``: the
unit-norm key is its single percept; ``actions`` holds the record id and θ as
JSON (the payload's ``theta`` at store time, ``{}`` without one); ``reward`` is
the signed outcome (+1 for a success, −1 for a failure, 0 when the payload has
no ``success``); ``timestamp`` is the insertion index. The crate's episode has
no id, payload, or score, so a Python side table (``NumpyBackend``'s) keeps the
ids, the unit vectors, and the payloads: ``vector``, ``payload``, ``update``,
``save``, and ``load`` use it, and a later ``update`` (utility reinforcement)
changes the side table, not the stored episode.

**Search.** ``search(vector, k, filter)`` asks the crate for its ranking
(``recall_similar``, through the binding's ``recall_similar_scored``, which
returns each episode's score by the crate's own formula), maps each episode
back to its id through ``actions``, and applies the filter in Python, so a
filtered search (every search System 2 makes) brings the whole ranking back
through PyO3 (the crate has no filter). The crate's dot
product sums left to right and its sort is stable, as ``NumpyBackend``'s are,
so both return the same ids, in the same order, with the same scores
(``tests/test_ruvector_backend.py``). Cosine only: the crate ranks by the raw
dot product, which is the cosine on the unit vectors this backend stores.

RuVector mapping: the store is ``memory_system::EpisodicMemory`` with
``Episode { percepts, actions, reward, timestamp }``; ``SemanticMemory``,
``SkillLibrary``, and ``CognitiveCore`` are exposed by the binding too, while
the cognition code keeps its Python logic (refer to docs/RUVECTOR_MAPPING.md).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from vfa.memory.backend import Filter, Hit, NumpyBackend, check_k, matches, plain_filter

BUILD_HINT = (
    "the ruvector backend needs the Rust extension: uv sync --extra ruvector && "
    "uv run maturin develop --release -m rust/ruvector_vfa_py/Cargo.toml"
)


def rust_module() -> Any:
    """The compiled binding (``ruvector_vfa_py``); ``ImportError`` with the build steps."""
    try:
        import ruvector_vfa_py
    except ImportError as exc:  # not built in this environment
        raise ImportError(BUILD_HINT) from exc
    return ruvector_vfa_py


def available() -> bool:
    try:
        rust_module()
    except ImportError:
        return False
    return True


class RuVectorBackend(NumpyBackend):
    """``MemoryBackend`` whose search runs in the Rust crate (refer to the module doc)."""

    def __init__(self, metric: str = "cosine", dim: int | None = None) -> None:
        if metric != "cosine":
            raise ValueError("the Rust crate ranks by the raw dot product: metric must be cosine")
        super().__init__(metric, dim)
        self._rust = rust_module().EpisodicMemory()

    def add(self, id: str, vector: np.ndarray, payload: Any = None) -> None:
        super().add(id, vector, payload)  # validates, normalizes, keeps the side table
        self._store(len(self._ids) - 1)

    def _store(self, i: int) -> None:
        payload = self._payloads[i]
        success = payload.get("success")
        reward = 0.0 if not isinstance(success, bool) else (1.0 if success else -1.0)
        theta = json.dumps(payload.get("theta", {}), sort_keys=True)
        self._rust.store([self._vectors[i].tolist()], [self._ids[i], theta], reward, i)

    def search(self, vector: np.ndarray, k: int, filter: Filter = None) -> list[Hit]:
        k = check_k(k)
        if not self._ids or k == 0:
            return []
        q = self.check(vector)
        ranked = self._rust.recall_similar_scored(
            q.tolist(), min(k, len(self._ids)) if filter is None else len(self._ids)
        )
        filter = plain_filter(filter)
        hits: list[Hit] = []
        for actions, score in ranked:
            i = self._index[actions[0]]
            if filter is None or matches(self._payloads[i], filter):
                hits.append(Hit(self._ids[i], float(score), copy.deepcopy(self._payloads[i])))
                if len(hits) == k:
                    break
        return hits

    @classmethod
    def load(cls, path: str | Path) -> RuVectorBackend:
        backend = super().load(path)  # the side table, then the episodes in insertion order
        for i in range(len(backend._ids)):
            backend._store(i)
        return backend
