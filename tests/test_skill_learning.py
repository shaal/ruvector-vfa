"""T13: the skill library (``vfa.memory.skill_learning``).

ROADMAP acceptance: the averaging and confidence math match the Rust semantics
(``ruvector_robotics::cognitive::skill_learning``, its unit tests ported here);
on the successes in ``tests/fixtures/calib_episodes.npz`` the clusters reach
purity ≥ 0.7 against the hidden class labels (labels used only to score).
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import numpy as np
import pytest

from vfa.memory import embeddings as M
from vfa.memory import skill_learning as SL
from vfa.memory.backend import NumpyBackend
from vfa.memory.memory_system import EpisodeRecord, MemorySystem
from vfa.memory.skill_learning import (
    CLUSTER_THRESHOLD,
    TRAJECTORY_KEY,
    Demonstration,
    SkillLibrary,
    cluster,
)

# Fixed before the first purity run: a cluster count above this would make
# purity meaningless (singletons score 1.0). Two clusters per hidden class.
MAX_CLUSTERS = 12


def demo(points, theta=None) -> Demonstration:
    points = np.asarray(points, dtype=float)
    return Demonstration(points, np.arange(len(points)), "", dict(theta or {}))


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# ------------------------------------------------- Rust semantics (ported tests)


def test_learn_single_demo():
    lib = SkillLibrary()
    skill = lib.learn_from_demonstration("wave", [demo([[1.0, 2.0, 3.0]])])
    assert skill.trajectory.tolist() == [[1.0, 2.0, 3.0]]
    assert skill.confidence > 0.0
    assert len(lib) == 1


def test_learn_multiple_demos_averages_over_the_shortest():
    lib = SkillLibrary()
    skill = lib.learn_from_demonstration(
        "reach", [demo([[0, 0, 0], [2, 2, 2]]), demo([[2, 2, 2], [4, 4, 4]])]
    )
    assert skill.trajectory.tolist() == [[1, 1, 1], [3, 3, 3]]
    mixed = lib.learn_from_demonstration(
        "mixed", [demo([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), demo([[3, 3, 3], [4, 4, 4]])]
    )
    assert mixed.trajectory.tolist() == [[2, 2, 2], [3, 3, 3]]


def test_average_sums_in_demo_order_then_divides():
    """The crate adds the demos one by one, then divides: 1e16 + 1 − 1e16 is 0 in
    that order, while an exact mean would give 1/3."""
    values = [1e16, 1.0, -1e16]
    demos = [demo([[v, 0, 0]], {"grip_force": v}) for v in values]
    skill = SkillLibrary().learn_from_demonstration("order", demos)
    assert skill.trajectory[0, 0] == 0.0
    assert skill.theta["grip_force"] == 0.0
    assert math.fsum(values) / 3 != 0.0


def test_confidence_is_one_minus_one_over_n_plus_one():
    lib = SkillLibrary()
    previous = 0.0
    for n in range(1, 8):
        skill = lib.learn_from_demonstration(f"s{n}", [demo([[n, 0, 0]])] * n)
        assert skill.confidence == 1.0 - 1.0 / (n + 1.0)
        assert skill.confidence > previous
        previous = skill.confidence


def test_theta_is_averaged_and_fields_must_match():
    lib = SkillLibrary()
    skill = lib.learn_from_demonstration(
        "grip",
        [
            demo([[0, 0, 0]], {"grip_force": 4.0, "lift_accel": 0.3}),
            demo([[0, 0, 0]], {"grip_force": 6.0, "lift_accel": 0.5}),
        ],
    )
    assert skill.theta == {"grip_force": 5.0, "lift_accel": 0.4}
    with pytest.raises(ValueError, match="same θ fields"):
        lib.learn_from_demonstration(
            "bad", [demo([[0, 0, 0]], {"grip_force": 1.0}), demo([[0, 0, 0]], {})]
        )


def test_empty_demos_give_a_zero_confidence_skill_that_replaces():
    lib = SkillLibrary()
    lib.learn_from_demonstration("grab", [demo([[1, 1, 1]])])
    lib.execute_skill("grab")
    skill = lib.learn_from_demonstration("grab", [])
    assert skill.confidence == 0.0
    assert skill.trajectory.shape == (0, 3)
    assert lib.get("grab").execution_count == 0
    assert len(lib) == 1


def test_relearning_replaces_the_skill_and_resets_the_count():
    lib = SkillLibrary()
    lib.learn_from_demonstration("grab", [demo([[0, 0, 0]])])
    lib.execute_skill("grab")
    lib.improve_skill("grab", 0.3)
    skill = lib.learn_from_demonstration("grab", [demo([[2, 2, 2]])] * 3)
    assert lib.get("grab").execution_count == 0
    assert lib.get("grab").confidence == skill.confidence == 0.75
    assert lib.get("grab").trajectory.tolist() == [[2, 2, 2]]


def test_execute_increments_count_and_returns_a_copy():
    lib = SkillLibrary()
    lib.learn_from_demonstration("grab", [demo([[0.0, 0.0, 0.0]])])
    trajectory = lib.execute_skill("grab")
    assert lib.get("grab").execution_count == 1
    trajectory[0, 0] = 99.0
    assert lib.execute_skill("grab").tolist() == [[0.0, 0.0, 0.0]]
    assert lib.get("grab").execution_count == 2
    assert lib.execute_skill("nonexistent") is None


def test_returned_skill_is_a_copy():
    lib = SkillLibrary()
    skill = lib.learn_from_demonstration("grab", [demo([[1, 1, 1]], {"grip_force": 2.0})])
    skill.trajectory[0, 0] = 99.0
    skill.theta["grip_force"] = 99.0
    skill.confidence = 0.0
    stored = lib.get("grab")
    assert stored.trajectory.tolist() == [[1, 1, 1]]
    assert stored.theta == {"grip_force": 2.0}
    assert stored.confidence == 0.5


def test_improve_skill_clamps_and_ignores_unknown_names():
    lib = SkillLibrary()
    lib.learn_from_demonstration("push", [demo([[1, 0, 0]])])
    initial = lib.get("push").confidence
    lib.improve_skill("push", 0.1)
    assert lib.get("push").confidence > initial
    lib.improve_skill("push", 10.0)
    assert lib.get("push").confidence == 1.0
    lib.improve_skill("push", -20.0)
    assert lib.get("push").confidence == 0.0
    lib.improve_skill("missing", 0.5)
    assert lib.get("missing") is None
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            lib.improve_skill("push", bad)
    assert lib.get("push").confidence == 0.0


def test_skill_names_must_be_non_empty_strings():
    lib = SkillLibrary()
    for bad in (7, "", None):
        with pytest.raises(ValueError, match="non-empty str"):
            lib.learn_from_demonstration(bad, [])
    assert len(lib) == 0


def test_trajectory_must_be_n_by_3_and_timestamps_integers():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        Demonstration(np.zeros((4, 2)))
    assert Demonstration([]).trajectory.shape == (0, 3)
    with pytest.raises(ValueError, match="integers"):
        Demonstration(np.zeros((3, 3)), [0.0, 0.002, 0.004])
    stamps = Demonstration(np.zeros((2, 3)), [0.0, 2.0]).timestamps
    assert stamps.tolist() == [0, 2] and stamps.dtype == np.int64
    for bad in ([1e9 + 0.5], [True, False], [[1, 2], [3, 4]], 5, [2**63], [math.nan]):
        with pytest.raises(ValueError, match="timestamps"):
            Demonstration(np.zeros((1, 3)), bad)


# ---------------------------------------------------------------- consolidation


def record(i: int, result: str = "SUCCESS", grip: float = 5.0, trajectory=None) -> EpisodeRecord:
    """θ as chosen is ``grip``; θ at the end (after a reflex) is 1 N higher."""
    info = {} if trajectory is None else {TRAJECTORY_KEY: trajectory}
    return EpisodeRecord(
        f"ep_{i:04d}",
        i,
        1,
        {"grip_force": grip},
        result,
        info=info,
        theta_final={"grip_force": grip + 1.0},
    )


def two_groups(memory: MemorySystem, rng: np.random.Generator) -> None:
    """Six successes near e0 (grip 2 N), six near e1 (grip 10 N), and distractors."""
    for i in range(12):
        axis, grip = (0, 2.0) if i % 2 == 0 else (1, 10.0)
        v = np.zeros(8)
        v[axis] = 1.0
        v[2:] = rng.normal(scale=0.05, size=6)
        memory.episodic.store(
            record(i, grip=grip + 0.1 * (i // 2), trajectory=[[axis, 0, 0], [axis, 1, 0]]),
            {"key_probe": unit(v)},
        )
    failure = np.zeros(8)
    failure[3] = 1.0
    memory.episodic.store(record(20, "BREAK", 8.0), {"key_probe": failure})
    memory.episodic.store(record(21), {"key_pre": unit(np.ones(8))})  # no probe


def test_consolidate_learns_one_skill_per_cluster(caplog):
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(3))
    lib = SkillLibrary()
    with caplog.at_level(logging.INFO, logger="vfa.memory.skill_learning"):
        skills = lib.consolidate(memory)
    assert [s.name for s in skills] == ["skill_00", "skill_01"]
    even, odd = skills
    assert even.theta["grip_force"] == pytest.approx(2.0 + 0.1 * 2.5)
    assert odd.theta["grip_force"] == pytest.approx(10.0 + 0.1 * 2.5)
    assert even.confidence == 1.0 - 1.0 / 7.0
    assert even.trajectory.tolist() == [[0, 0, 0], [0, 1, 0]]
    assert len(memory.semantic) == 2
    vector, payload = memory.semantic.retrieve("skill_01")
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert payload["members"] == [f"ep_{i:04d}" for i in range(1, 12, 2)]
    assert payload["theta"] == odd.theta  # θ as chosen, not theta_final
    assert "confidence" not in payload  # the library holds it (improve_skill changes it)
    assert lib.get("skill_01").confidence == odd.confidence
    assert memory.semantic.find_similar(unit([1, 0, 0, 0, 0, 0, 0, 0]), 1)[0][0] == "skill_00"
    assert "[MEM] consolidated skill_00 from 6 successes, confidence 0.86" in caplog.text


def test_reconsolidation_replaces_only_its_own_skills():
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(4))
    lib = SkillLibrary()
    lib.learn_from_demonstration("taught", [demo([[0, 0, 0]])])
    assert len(lib.consolidate(memory)) == 2
    lib.improve_skill("skill_01", -0.5)
    merged = lib.consolidate(memory, threshold=-1.0, iterations=0)  # one cluster
    assert [s.name for s in merged] == ["skill_00"]
    assert sorted(lib.skills) == ["skill_00", "taught"]
    assert [name for name, _ in memory.semantic.find_similar(unit(np.ones(8)), 5)] == ["skill_00"]
    assert lib.get("skill_00").confidence == 1.0 - 1.0 / 13.0


def test_consolidate_is_all_or_nothing():
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(5))
    lib = SkillLibrary()
    before = [s.theta for s in lib.consolidate(memory)]
    lib.improve_skill("skill_00", -0.5)
    odd = EpisodeRecord("ep_0100", 100, 1, {"grip_force": 3.0, "lift_accel": 0.3}, "SUCCESS")
    memory.episodic.store(odd, {"key_probe": unit([1, 0, 0, 0, 0, 0, 0, 0])})
    with pytest.raises(ValueError, match="same θ fields"):
        lib.consolidate(memory)
    assert [lib.get(n).theta for n in ("skill_00", "skill_01")] == before
    assert lib.get("skill_00").confidence == 1.0 - 1.0 / 7.0 - 0.5
    assert sorted(lib.skills) == ["skill_00", "skill_01"]
    assert len(memory.semantic) == 2
    # Two opposite successes in one cluster: a zero centroid, refused before any change.
    flat = MemorySystem()
    flat.episodic.store(record(1), {"key_probe": unit([1, 0])})
    flat.episodic.store(record(2), {"key_probe": unit([-1, 0])})
    flat.semantic.store("kept", unit([0, 1]))
    with pytest.raises(ValueError, match="sum to zero"):
        lib.consolidate(flat, threshold=-1.0, iterations=0)
    assert [n for n, _ in flat.semantic.find_similar(unit([0, 1]), 5)] == ["kept"]
    assert sorted(lib.skills) == ["skill_00", "skill_01"]
    # The failures left the library consistent: a good consolidation still works.
    assert [s.name for s in lib.consolidate(flat, threshold=0.5)] == ["skill_00", "skill_01"]
    assert sorted(lib.skills) == ["skill_00", "skill_01"]


def test_zero_centroid_is_refused_on_a_euclidean_index():
    flat = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
    flat.episodic.store(record(1), {"key_probe": np.array([1.0, 0.0])})
    flat.episodic.store(record(2), {"key_probe": np.array([-1.0, 0.0])})
    flat.semantic.store("kept", np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="sum to zero"):
        SkillLibrary().consolidate(flat, threshold=-1.0, iterations=0)
    assert flat.semantic.retrieve("kept") is not None and len(flat.semantic) == 1
    # Opposite vectors that do not cancel exactly leave a ~1e-16 residue: still no centroid.
    for factory in (NumpyBackend, lambda: NumpyBackend("euclidean")):
        noisy = MemorySystem(backend_factory=factory)
        noisy.episodic.store(record(1), {"key_probe": np.array([0.1, 0.2, 0.3])})
        noisy.episodic.store(record(2), {"key_probe": np.array([-0.3, -0.6, -0.9])})
        with pytest.raises(ValueError, match="sum to zero"):
            SkillLibrary().consolidate(noisy, threshold=-1.0, iterations=0)


def test_extreme_magnitudes_on_a_euclidean_index():
    """Raw vectors far outside the float norm range still cluster by direction; a
    zero key has no direction and is refused before any change."""
    memory = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
    for i in range(3):
        memory.episodic.store(record(i, grip=2.0), {"key_probe": np.array([1.0, 0.0])})
        memory.episodic.store(record(10 + i, grip=10.0), {"key_probe": np.array([0.0, 1.0])})
    memory.episodic.store(record(20, grip=10.0), {"key_probe": np.array([0.0, 1e200])})
    memory.episodic.store(record(21, grip=10.0), {"key_probe": np.array([0.0, 1e-200])})
    lib = SkillLibrary()
    assert [s.theta for s in lib.consolidate(memory)] == [{"grip_force": 2.0}, {"grip_force": 10.0}]
    huge = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
    for i in range(3):
        huge.episodic.store(record(i), {"key_probe": np.array([1e200, 0.0])})
    assert len(SkillLibrary().consolidate(huge)) == 1
    members = [memory.semantic.retrieve(n)[1]["members"] for n in ("skill_00", "skill_01")]
    memory.episodic.store(record(30, grip=10.0), {"key_probe": np.zeros(2)})
    with pytest.raises(ValueError, match="zero key_probe"):
        lib.consolidate(memory)
    assert [memory.semantic.retrieve(n)[1]["members"] for n in ("skill_00", "skill_01")] == members
    for scale in (1e-200, 1e308):  # only far-out rows: they still form a cluster
        far = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
        for i in range(3):
            far.episodic.store(record(i), {"key_probe": np.array([scale, scale])})
        (skill,) = SkillLibrary().consolidate(far)
        assert far.semantic.retrieve(skill.name)[0] == pytest.approx([2**-0.5, 2**-0.5])


def test_unit_scaling_leaves_normal_rows_bit_identical():
    rows = np.random.default_rng(10).normal(size=(100, 50))
    for row in rows:
        assert np.array_equal(SL._unit(row), row / np.linalg.norm(row))


def test_nearly_opposite_vectors_still_have_a_centroid():
    memory = MemorySystem()
    memory.episodic.store(record(1), {"key_probe": unit([1, 0])})
    memory.episodic.store(record(2), {"key_probe": unit([-1, 1e-8])})
    (skill,) = SkillLibrary().consolidate(memory, threshold=-1.0, iterations=0)
    assert memory.semantic.retrieve(skill.name)[0] == pytest.approx([0.0, 1.0])


def test_a_failing_semantic_rebuild_changes_nothing():
    """A mean θ of inf passes learning but not the JSON payload check inside
    ``SemanticMemory.rebuild``: the library and the old concepts stay."""
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(9))
    lib = SkillLibrary()
    lib.consolidate(memory)
    before = {name: dict(skill.theta) for name, skill in lib.skills.items()}
    for i in (30, 31):
        memory.episodic.store(record(i, grip=1e308), {"key_probe": unit([0, 0, 0, 0, 0, 0, 0, 1])})
    with pytest.raises(ValueError, match="Out of range"):
        lib.consolidate(memory, min_support=1)
    assert {name: skill.theta for name, skill in lib.skills.items()} == before
    assert sorted(n for n, _ in memory.semantic.find_similar(unit(np.ones(8)), 9)) == sorted(before)


def test_consolidate_clusters_by_cosine_on_any_metric():
    """A Euclidean index stores raw vectors; consolidation still compares directions."""
    rng = np.random.default_rng(6)
    cosine, euclidean = (
        MemorySystem(),
        MemorySystem(backend_factory=lambda: NumpyBackend("euclidean")),
    )
    for i in range(8):
        v = np.array([1.0, 0.0] if i % 2 == 0 else [0.6, 0.8]) * (1 + 9 * rng.random())
        for memory in (cosine, euclidean):
            memory.episodic.store(record(i, grip=1.0 + 3 * (i % 2)), {"key_probe": v})
    thetas = [[s.theta for s in SkillLibrary().consolidate(m)] for m in (cosine, euclidean)]
    assert thetas[0] == thetas[1] == [{"grip_force": 1.0}, {"grip_force": 4.0}]
    centroid, _ = euclidean.semantic.retrieve("skill_01")
    assert centroid == pytest.approx([0.6, 0.8])  # unit norm, not a raw sum


def test_taught_skills_may_not_use_the_consolidation_prefix():
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(7))
    lib = SkillLibrary()
    lib.learn_from_demonstration("skill_00", [demo([[0, 0, 0]])])
    with pytest.raises(ValueError, match="consolidation prefix"):
        lib.consolidate(memory)
    assert lib.get("skill_00").trajectory.tolist() == [[0, 0, 0]]
    assert len(memory.semantic) == 0
    # Teaching over a name that a consolidation made is refused the same way.
    del lib.skills["skill_00"]
    lib.consolidate(memory)
    lib.execute_skill("skill_00")  # using a consolidated skill is fine
    lib.improve_skill("skill_00", 0.01)
    assert len(lib.consolidate(memory)) == 2
    taught = lib.learn_from_demonstration("skill_00", [demo([[9, 9, 9]], {"grip_force": 42.0})])
    with pytest.raises(ValueError, match="consolidation prefix"):
        lib.consolidate(memory)
    assert lib.get("skill_00").theta == taught.theta


def test_consolidate_returns_copies():
    memory = MemorySystem()
    two_groups(memory, np.random.default_rng(8))
    lib = SkillLibrary()
    skills = lib.consolidate(memory)
    skills[0].theta["grip_force"] = 99.0
    skills[0].trajectory[0, 0] = 99.0
    skills[0].confidence = 0.0
    stored = lib.get("skill_00")
    assert stored.theta["grip_force"] != 99.0 and stored.trajectory[0, 0] != 99.0
    assert stored.confidence == 1.0 - 1.0 / 7.0


def test_consolidate_on_empty_memory_clears_semantic():
    memory = MemorySystem()
    lib = SkillLibrary()
    memory.semantic.store("stale", unit(np.ones(4)))
    assert lib.consolidate(memory) == []
    assert len(memory.semantic) == 0
    with pytest.raises(ValueError, match="cosine"):
        lib.consolidate(memory, threshold=5.0)


def test_missing_trajectory_empties_the_skill_trajectory():
    memory = MemorySystem()
    memory.episodic.store(record(1, trajectory=[[0, 0, 0], [1, 1, 1]]), {"key_probe": unit([1, 0])})
    memory.episodic.store(record(2), {"key_probe": unit([1, 0.01])})
    (skill,) = SkillLibrary().consolidate(memory)
    assert skill.trajectory.shape == (0, 3)
    assert skill.confidence == 1.0 - 1.0 / 3.0


def test_cluster_leader_then_kmeans():
    rows = np.array([unit([1, 0]), unit([0, 1]), unit([1, 0.2]), unit([0.2, 1]), unit([1, 1])])
    assert cluster(rows, 0.99, 0) == [[0], [1], [2], [3], [4]]
    # Leader pass at 0.7: row 2 joins row 0 (cos 0.98), row 3 joins row 1; row 4 ties
    # at cos 0.773 with both centroids, argmax keeps the first, and k-means leaves it.
    clusters = cluster(rows, 0.7, 10)
    assert clusters == cluster(rows, 0.7, 0) == [[0, 2, 4], [1, 3]]
    assert cluster(rows, 0.7, 10) == clusters  # deterministic
    assert cluster(np.zeros((0, 2)), 0.5, 3) == []
    for bad in (1.5, -2.0, math.nan):
        with pytest.raises(ValueError, match="cosine"):
            cluster(rows, bad, 1)
    with pytest.raises(ValueError, match="iterations"):
        cluster(rows, 0.5, -1)
    with pytest.raises(ValueError, match="min_support"):
        cluster(rows, 0.5, 1, 0)


def test_cluster_dissolves_clusters_below_min_support():
    near = [unit([1, 0.01 * i]) for i in range(4)]
    rows = np.array([*near[:2], unit([0.3, 1]), *near[2:]])
    assert cluster(rows, 0.99, 10) == [[0, 1, 3, 4], [2]]
    assert cluster(rows, 0.99, 10, min_support=3) == [[0, 1, 2, 3, 4]]
    assert cluster(rows, 0.99, 0, min_support=3) == [[0, 1, 3, 4], [2]]  # no passes
    # No cluster reaches the minimum: all are kept.
    assert cluster(rows[[0, 2]], 0.99, 10, min_support=3) == [[0], [1]]


# ----------------------------------------------------------------- the fixture


def purity(clusters: list[list[str]], labels: dict[str, str]) -> float:
    total = sum(len(c) for c in clusters)
    return sum(Counter(labels[i] for i in c).most_common(1)[0][1] for c in clusters) / total


def test_fixture_successes_cluster_by_hidden_class():
    """Consolidation on the calibration successes: the clusters follow the hidden
    classes (purity ≥ 0.7, at most MAX_CLUSTERS). The labels score the result
    only; the memory never holds them."""
    fx = M.load_fixture()
    views = M.fixture_views(fx)
    emb = M.Embedder()
    columns = [str(c) for c in fx["theta_columns"]]
    memory = MemorySystem()
    labels = {}
    for i, view in enumerate(views):
        rec = EpisodeRecord(
            f"fx_{i:03d}",
            int(fx["group"][i]),
            int(fx["attempt"][i]),
            dict(zip(columns, map(float, fx["theta"][i]), strict=True)),
            str(fx["result"][i]),
            info={TRAJECTORY_KEY: view.ee.tolist()},
            theta_final=dict(zip(columns, map(float, fx["theta_final"][i]), strict=True)),
        )
        keys = {"key_pre": emb.key_pre(view.context)}
        if view.probe is not None:
            keys["key_probe"] = emb.key_probe(view)
        memory.episodic.store(rec, keys)
        labels[rec.id] = str(fx["hidden_class"][i])
    lib = SkillLibrary()
    skills = lib.consolidate(memory)
    clusters = [memory.semantic.retrieve(s.name)[1]["members"] for s in skills]
    successes = [r.id for r in memory.episodic.records.values() if r.success]
    clustered = sum(len(c) for c in clusters)
    score = purity(clusters, labels)
    baseline = Counter(labels[i] for i in successes).most_common(1)[0][1] / len(successes)
    print(
        f"\nthreshold {CLUSTER_THRESHOLD}: {len(clusters)} clusters over {clustered} of "
        f"{len(successes)} successes, purity {score:.3f} (one cluster: {baseline:.3f})"
    )
    for skill, members in zip(skills, clusters, strict=True):
        counts = Counter(labels[i] for i in members)
        print(
            f"  {skill.name}: n={len(members):2d} conf={skill.confidence:.2f} "
            f"grip={skill.theta['grip_force']:5.2f} N  {dict(counts.most_common())}"
        )
    assert clustered == sum(1 for i in successes if i in memory.episodic.indexes["key_probe"])
    # Chance level for these cluster sizes: shuffle the labels over the members.
    rng = np.random.default_rng(13)
    pool = np.array([labels[i] for c in clusters for i in c])
    sizes = np.cumsum([len(c) for c in clusters])[:-1]
    null = [
        purity([list(g) for g in np.split(rng.permutation(pool), sizes)], {x: x for x in pool})
        for _ in range(1000)
    ]
    print(
        f"  chance purity for these sizes: mean {np.mean(null):.3f}, "
        f"p99 {np.quantile(null, 0.99):.3f}"
    )
    assert len(clusters) <= MAX_CLUSTERS
    assert score >= 0.7
    assert score > np.quantile(null, 0.99)
    assert all(set(memory.semantic.retrieve(s.name)[1]) == {"theta", "members"} for s in skills)
