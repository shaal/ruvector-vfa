"""Skill library: learn skills from demonstrations and consolidate successes (T13).

ARCHITECTURE §5.2–§5.3, mirroring ``ruvector_robotics::cognitive::skill_learning``.

- ``Demonstration``: an end-effector trajectory (``(n, 3)`` positions), its
  timestamps, a metadata string, and (ours) the θ of the attempt.
- ``SkillLibrary.learn_from_demonstration(name, demos)`` averages the
  trajectories point by point over the shortest one (summed in demo order, then
  divided by the count, as the crate does), averages θ the same way, and sets
  ``confidence = 1 − 1/(n + 1)``. It replaces any skill of that name (with
  ``execution_count`` 0). No demos give a skill with an empty trajectory and
  confidence 0. ``execute_skill(name)`` returns a copy of the trajectory and
  counts the execution (None for an unknown name); ``improve_skill(name,
  feedback)`` adds ``feedback`` to the confidence, clamped to [0, 1] (it first
  rejects non-finite feedback, then does nothing for an unknown name).
- ``SkillLibrary.consolidate(memory)`` turns the successful episodes of a
  ``MemorySystem`` into skills. It clusters their ``key_probe`` vectors (records
  without a finished probe have no ``key_probe`` and are skipped), scaled to unit
  norm whatever the index metric, by cosine similarity: a leader pass in
  insertion order (a vector joins the most similar cluster if the similarity to
  its unit-norm centroid is at least ``CLUSTER_THRESHOLD``, else it starts a new
  cluster), then spherical k-means passes until nothing moves or
  ``CLUSTER_ITERATIONS`` passes ran. Each pass keeps the clusters with at least
  ``MIN_SUPPORT`` members (all of them if none has that many) and reassigns every
  vector to the most similar kept centroid. So when some cluster reaches
  ``MIN_SUPPORT`` and the passes converge, every skill rests on at least
  ``MIN_SUPPORT`` demonstrations (confidence ≥ 0.75). ``MIN_SUPPORT`` is absolute,
  so a small memory gives fewer, coarser skills. It never reads class labels: a
  record holds none. Each cluster becomes ``skill_00``, ``skill_01``, … (in order
  of the cluster's first record), learned from one ``Demonstration`` per member:
  the trajectory in ``record.info[TRAJECTORY_KEY]`` (empty when absent, which
  empties the skill's trajectory, as the crate's shortest-demo rule does) and θ as
  chosen (``record.theta``, the value System 2 sets, not ``theta_final``: the
  reflexes add their raise on top of it at run time). The
  unit-norm centroid goes into ``SemanticMemory`` under the skill name, with the
  mean θ and the member ids as payload; the library holds the confidence.
  Consolidation starts over each time: it replaces every concept in
  ``SemanticMemory`` and the skills that earlier consolidations made, so a
  positional name can move to another cluster, and ``improve_skill`` feedback
  lasts until the next consolidation. It is all or nothing: on an error (θ fields
  that differ, a bad trajectory, a zero key, a cluster whose vectors sum to zero) the library
  and ``SemanticMemory`` stay as they were. A skill taught by name must not use
  the ``skill_`` prefix, not even a name a consolidation made. A library belongs
  to one ``MemorySystem``: consolidating it on another rewrites its skills and
  leaves the first memory's concepts stale.

For System 2 (T15): ``SemanticMemory.find_similar`` gives skill names and
scores; ``retrieve(name)`` gives the mean θ and member ids; the confidence comes
from ``SkillLibrary.get(name)``. The centroids
are ``key_probe`` vectors, so the fallback can query only from ``PROBE_DONE`` on.
``MemorySystem.save`` does not save the library: after a load, consolidate again
(deterministic in record order; ``improve_skill`` feedback is lost).

RuVector mapping: ``Demonstration``, ``Skill``, and ``SkillLibrary`` are the
types of the same names in ``ruvector_robotics::cognitive::skill_learning``,
with the same averaging, confidence, execution count, and clamping. Deviations:
θ rides along on ``Demonstration`` and ``Skill`` (a side table in the Rust
backend, T22); ``improve_skill`` rejects non-finite feedback (Rust stores NaN
for NaN and clamps ±inf); a skill name must be a non-empty string and
timestamps must be integers (Rust takes any ``&str`` and ``i64``s by type);
consolidation is ours (the crate has none).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vfa.memory.backend import NORM_RANGE

if TYPE_CHECKING:
    from vfa.memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)

CLUSTER_KEY = "key_probe"
CLUSTER_THRESHOLD = 0.8  # cosine to a cluster's centroid to join it (leader pass)
CLUSTER_ITERATIONS = 10  # spherical k-means passes after the leader pass
MIN_SUPPORT = 3  # a k-means pass dissolves smaller clusters into the nearest kept one
TRAJECTORY_KEY = "ee_trajectory"  # EpisodeRecord.info entry: (n, 3) ee positions [m]
SKILL_PREFIX = "skill_"
ZERO_SUM = 1e-9  # per member: a smaller sum of unit rows is rounding residue, not a direction


def _timestamps(values: Any) -> np.ndarray:
    """A 1-D int64 array; every entry must be a whole number in the crate's i64 range."""
    if np.ndim(values) != 1:
        raise ValueError(f"timestamps must be 1-D, got {np.ndim(values)} dimensions")
    stamps = []
    for value in values:
        whole = isinstance(value, int | np.integer) or (
            isinstance(value, float | np.floating) and float(value).is_integer()
        )
        if isinstance(value, bool | np.bool_) or not whole or not -(2**63) <= int(value) < 2**63:
            raise ValueError(f"timestamps must be integers (the crate's i64), got {value!r}")
        stamps.append(int(value))
    return np.array(stamps, dtype=np.int64)


def _trajectory(values: Any) -> np.ndarray:
    array = np.asarray(values if values is not None else [], dtype=float)
    if array.size == 0:
        return np.zeros((0, 3))
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"a trajectory must have shape (n, 3), got {array.shape}")
    return array


@dataclass(slots=True, eq=False)
class Demonstration:
    """One recorded attempt to learn from (Rust ``Demonstration`` plus θ)."""

    trajectory: np.ndarray  # (n, 3) ee positions [m]
    timestamps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    metadata: str = ""
    theta: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trajectory = _trajectory(self.trajectory)
        self.timestamps = _timestamps(self.timestamps)


@dataclass(slots=True, eq=False)
class Skill:
    """A learned skill (Rust ``Skill`` plus the mean θ)."""

    name: str
    trajectory: np.ndarray  # (m, 3), m = the shortest demonstration
    confidence: float
    execution_count: int = 0
    theta: dict[str, float] = field(default_factory=dict)


def _mean_theta(demos: Sequence[Demonstration]) -> dict[str, float]:
    names = set(demos[0].theta)
    if any(set(d.theta) != names for d in demos):
        raise ValueError("every demonstration needs the same θ fields")
    total = dict.fromkeys(demos[0].theta, 0.0)
    for demo in demos:
        for name in total:
            total[name] += float(demo.theta[name])
    return {name: value / len(demos) for name, value in total.items()}


class SkillLibrary:
    """Skills by name (refer to the module doc)."""

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}
        self._consolidated: dict[str, Skill] = {}  # name -> the object consolidate stored

    def __len__(self) -> int:
        return len(self.skills)

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def learn_from_demonstration(self, name: str, demos: Sequence[Demonstration]) -> Skill:
        """Average ``demos`` into skill ``name`` (replacing any skill of that name)."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"a skill name must be a non-empty str, got {name!r}")
        if not demos:
            skill = Skill(name, np.zeros((0, 3)), 0.0)
        else:
            length = min(len(d.trajectory) for d in demos)
            total = np.zeros((length, 3))
            for demo in demos:  # the crate's order: sum over demos, then divide
                total += demo.trajectory[:length]
            confidence = 1.0 - 1.0 / (len(demos) + 1.0)
            skill = Skill(name, total / len(demos), confidence, theta=_mean_theta(demos))
        self.skills[name] = skill
        return _copy(skill)

    def execute_skill(self, name: str) -> np.ndarray | None:
        """The skill's trajectory (a copy), counting the execution; None if unknown."""
        skill = self.skills.get(name)
        if skill is None:
            return None
        skill.execution_count += 1
        return skill.trajectory.copy()

    def improve_skill(self, name: str, feedback: float) -> None:
        """Add ``feedback`` to the confidence, clamped to [0, 1]; no-op if unknown."""
        if not math.isfinite(feedback):
            raise ValueError(f"feedback must be finite, got {feedback!r}")
        skill = self.skills.get(name)
        if skill is not None:
            skill.confidence = min(max(skill.confidence + float(feedback), 0.0), 1.0)

    def consolidate(
        self,
        memory: MemorySystem,
        threshold: float = CLUSTER_THRESHOLD,
        iterations: int = CLUSTER_ITERATIONS,
        min_support: int = MIN_SUPPORT,
    ) -> list[Skill]:
        """Rebuild the consolidated skills from ``memory``'s successes (refer to the
        module doc); returns them in name order."""
        taught = [
            name
            for name, skill in self.skills.items()
            if name.startswith(SKILL_PREFIX) and self._consolidated.get(name) is not skill
        ]
        if taught:
            raise ValueError(f"skills {taught[:3]} use the consolidation prefix {SKILL_PREFIX!r}")
        index = memory.episodic.indexes[CLUSTER_KEY]
        records = [r for r in memory.episodic.records.values() if r.success and r.id in index]
        vectors = np.array([_unit(index.vector(r.id)) for r in records])
        flat = [r.id for r, v in zip(records, vectors, strict=True) if not np.any(v)]
        if flat:  # a Euclidean index can hold a zero key
            raise ValueError(f"records {flat[:3]} have a zero {CLUSTER_KEY} (no direction)")
        clusters = cluster(vectors, threshold, iterations, min_support)  # [] if no records
        staged = SkillLibrary()  # learn everything before anything changes
        concepts = []
        for number, members in enumerate(clusters):
            name = f"{SKILL_PREFIX}{number:02d}"
            demos = [
                Demonstration(
                    _trajectory(records[i].info.get(TRAJECTORY_KEY)),
                    metadata=records[i].id,
                    theta=records[i].theta,
                )
                for i in members
            ]
            skill = staged.learn_from_demonstration(name, demos)
            total = vectors[members].sum(axis=0)
            if not np.linalg.norm(total) > ZERO_SUM * len(members):  # rounding residue only
                raise ValueError(f"{name}: its key vectors sum to zero (no centroid)")
            payload = {"theta": skill.theta, "members": [records[i].id for i in members]}
            concepts.append((name, _unit(total), payload))
        memory.semantic.rebuild(concepts)
        for name in self._consolidated:
            self.skills.pop(name, None)
        self.skills.update(staged.skills)
        self._consolidated = dict(staged.skills)
        for (name, _, payload), skill in zip(concepts, staged.skills.values(), strict=True):
            logger.info(
                "[MEM] consolidated %s from %d successes, confidence %.2f",
                name,
                len(payload["members"]),
                skill.confidence,
            )
        return [_copy(skill) for skill in staged.skills.values()]


def _copy(skill: Skill) -> Skill:
    return Skill(
        skill.name,
        skill.trajectory.copy(),
        skill.confidence,
        skill.execution_count,
        dict(skill.theta),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` at unit norm (a zero vector stays zero). Outside ``NORM_RANGE`` it is
    first divided by its largest entry, so the norm neither overflows nor underflows."""
    norm = float(np.linalg.norm(vector))
    if not NORM_RANGE[0] < norm < NORM_RANGE[1] and np.any(vector):
        vector = vector / float(np.max(np.abs(vector)))
        norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def cluster(
    vectors: np.ndarray, threshold: float, iterations: int, min_support: int = 1
) -> list[list[int]]:
    """Leader clustering, then spherical k-means with a minimum cluster size
    (refer to the module doc).

    ``vectors``: ``(n, d)`` unit-norm rows. Returns member indexes per cluster,
    clusters ordered by their first member, members in row order.
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be a cosine in [-1, 1], got {threshold!r}")
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations!r}")
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support!r}")
    sums: list[np.ndarray] = []
    labels = np.zeros(len(vectors), dtype=int)
    for i, vector in enumerate(vectors):
        if sums:
            sims = np.array([float(_unit(s) @ vector) for s in sums])
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                sums[best] = sums[best] + vector
                labels[i] = best
                continue
        sums.append(vector.copy())
        labels[i] = len(sums) - 1
    for _ in range(iterations if len(vectors) else 0):
        sizes = np.bincount(labels)
        kept = np.flatnonzero(sizes >= min_support)
        if not len(kept):
            kept = np.arange(len(sizes))
        centroids = np.array([_unit(vectors[labels == c].sum(axis=0)) for c in kept])
        new = kept[np.argmax(vectors @ centroids.T, axis=1)]
        if np.array_equal(new, labels):
            break
        _, labels = np.unique(new, return_inverse=True)  # drop empty clusters
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(i)
    return sorted(groups.values(), key=lambda members: members[0])
