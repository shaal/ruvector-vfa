"""T12: the memory system (``vfa.memory.backend``, ``vfa.memory.memory_system``).

ROADMAP acceptance: top-k ordering equals brute force; filters work; utility
updates change the ranking as specified; save → load → identical search results;
the Rust-parity rule holds (stored vectors are unit norm, so dot == cosine).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vfa.memory import embeddings as M
from vfa.memory import memory_system as MS
from vfa.memory.backend import MemoryBackend, NumpyBackend
from vfa.memory.memory_system import (
    KEY_TYPES,
    EpisodeRecord,
    EpisodicMemory,
    MemoryItem,
    MemorySystem,
    SemanticMemory,
    WorkingMemory,
    ranking_score,
    unit_norm,
)

RNG = np.random.default_rng(12)


def random_vectors(n: int, dim: int = 16) -> np.ndarray:
    return RNG.normal(size=(n, dim))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --------------------------------------------------------------------- backend


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_top_k_equals_brute_force(metric):
    data = random_vectors(200)
    backend = NumpyBackend(metric)
    for i, v in enumerate(data):
        backend.add(f"r{i}", v, {"i": i, "even": i % 2 == 0})
    assert isinstance(backend, MemoryBackend) and len(backend) == 200
    for q in random_vectors(10):
        if metric == "cosine":
            truth = unit(data) @ unit(q)
        else:
            truth = -np.linalg.norm(data - q, axis=1)
        order = np.argsort(-truth, kind="stable")[:7]
        hits = backend.search(q, 7)
        assert [h.id for h in hits] == [f"r{i}" for i in order]
        assert np.allclose([h.score for h in hits], truth[order], atol=1e-12)
    assert backend.search(data[0], 0) == []
    assert len(backend.search(data[0], 500)) == 200


def test_ties_keep_insertion_order_and_filters_work():
    backend = NumpyBackend()
    v = np.array([1.0, 0.0, 0.0])
    for i in range(5):
        backend.add(f"t{i}", v * (i + 1), {"tag": "a" if i < 3 else "b", "i": i})
    assert [h.id for h in backend.search(v, 5)] == ["t0", "t1", "t2", "t3", "t4"]
    assert [h.id for h in backend.search(v, 5, {"tag": "b"})] == ["t3", "t4"]
    assert [h.id for h in backend.search(v, 5, lambda p: p["i"] % 2 == 1)] == ["t1", "t3"]
    assert backend.search(v, 5, {"tag": "c"}) == []
    backend.update("t4", {"tag": "a", "new": 1})
    assert backend.payload("t4") == {"tag": "a", "i": 4, "new": 1}
    # Hits and payloads are copies: editing them never changes the store.
    backend.add("n", v, {"theta": {"grip_force": 5.0}})
    backend.search(v, 6)[-1].payload["theta"]["grip_force"] = 99.0
    backend.payload("n")["theta"]["grip_force"] = 99.0
    assert backend.payload("n") == {"theta": {"grip_force": 5.0}}
    assert len(backend.search(v, np.int64(2))) == 2  # numpy ints are fine for k


def test_rust_parity_stored_vectors_are_unit_so_dot_is_cosine():
    backend = NumpyBackend()
    data = random_vectors(50) * RNG.uniform(0.1, 10.0, size=(50, 1))  # any length in
    for i, v in enumerate(data):
        backend.add(f"p{i}", v, {})
    stored = [backend.vector(f"p{i}") for i in range(50)]
    assert unit_norm(stored)
    q = unit(random_vectors(1)[0])
    for h in backend.search(q, 50):
        i = int(h.id[1:])
        cosine = float(data[i] @ q / np.linalg.norm(data[i]))
        assert h.score == pytest.approx(cosine, abs=1e-12)
        assert h.score == pytest.approx(float(backend.vector(h.id) @ q), abs=1e-15)


def test_backend_rejects_bad_input():
    backend = NumpyBackend()
    backend.add("a", np.ones(3), {})
    with pytest.raises(KeyError):
        backend.add("a", np.ones(3), {})
    with pytest.raises(ValueError):
        backend.add("b", np.ones(4), {})
    with pytest.raises(ValueError):
        backend.add("c", np.zeros(3), {})
    with pytest.raises(ValueError):
        backend.add("d", np.array([1.0, np.nan, 0.0]), {})
    with pytest.raises(ValueError):
        backend.add("e", np.ones(3), {"x": float("nan")})
    with pytest.raises(ValueError):
        NumpyBackend("manhattan")
    with pytest.raises(ValueError):
        backend.search(np.ones(3), -1)
    # A huge or tiny finite vector still stores at unit norm.
    for i, v in enumerate(([1e200, 1e200, 0.0], [1e-320, 1e-320, 0.0], [1e-160, 3e-161, 0.0])):
        backend.add(f"s{i}", np.array(v), {})
        assert unit_norm([backend.vector(f"s{i}")])
    # A mapping filter never matches a key the payload lacks.
    assert backend.search(np.ones(3), 9, {"missing": None}) == []
    # Euclidean distances between huge or tiny vectors still order correctly.
    for near, far, q in ((1e199, 1e200, -1e200), (1e-170, 3e-170, 0.0)):
        euclid = NumpyBackend("euclidean")
        euclid.add("far", np.array([far]), {})
        euclid.add("near", np.array([near]), {})
        hits = euclid.search(np.array([q]), 2)
        assert [h.id for h in hits] == ["near", "far"]
        assert [h.score for h in hits] == pytest.approx([-abs(near - q), -abs(far - q)])


def test_equal_vectors_score_equal_at_full_key_size():
    """The dot product sums left to right (as the Rust crate does), so duplicates
    of a 354-entry key tie exactly and keep insertion order."""
    rng = np.random.default_rng(4)
    backend = NumpyBackend()
    for i, v in enumerate(rng.normal(size=(30, 354))):
        backend.add(f"x{i}", v, {})
    twin = rng.normal(size=354)
    for i in range(6):
        backend.add(f"d{i}", twin, {"tags": (1, 2)})
    for q in rng.normal(size=(50, 354)):
        hits = [h for h in backend.search(q, 36) if h.id.startswith("d")]
        assert [h.id for h in hits] == [f"d{i}" for i in range(6)]
        assert len({h.score for h in hits}) == 1
        sequential = 0.0
        for x, y in zip(backend.vector("x0").tolist(), backend.check(q).tolist(), strict=True):
            sequential += x * y
        assert backend.scores(q)[0] == sequential
    assert len(backend.search(twin, 9, {"tags": (1, 2)})) == 6  # tuple filters match lists


def test_backend_save_load_gives_identical_results(tmp_path):
    backend = NumpyBackend()
    data = random_vectors(40)
    for i, v in enumerate(data):
        backend.add(f"s{i}", v, {"i": i, "theta": {"grip_force": float(i)}})
    backend.save(tmp_path / "b")
    loaded = NumpyBackend.load(tmp_path / "b")
    assert loaded.ids == backend.ids  # insertion order, hence tie order, survives
    for q in random_vectors(5):
        assert backend.search(q, 10) == loaded.search(q, 10)
    loaded.add("new", data[0], {})  # a loaded store keeps growing
    assert len(loaded) == 41
    empty = NumpyBackend()
    empty.save(tmp_path / "e")
    assert len(NumpyBackend.load(tmp_path / "e")) == 0


# ---------------------------------------------------------------------- tiers


def test_working_memory_evicts_the_least_important():
    wm = WorkingMemory(2)  # the Rust crate's test, ported
    wm.add(MemoryItem("a", [], 0.1, 1))
    wm.add(MemoryItem("b", [], 0.9, 2))
    wm.add(MemoryItem("c", [], 0.5, 3))
    assert len(wm) == 2 and wm.get("a") is None
    item = wm.get("b")
    assert item is not None and item.access_count == 1
    wm.clear()
    assert len(wm) == 0
    with pytest.raises(ValueError):
        WorkingMemory(0)


def record(i: int, result: str = "SUCCESS", utility: float = 0.5) -> EpisodeRecord:
    return EpisodeRecord(
        f"ep_{i:04d}",
        situation=1000 + i,
        attempt=1,
        theta={"grip_force": 4.0 + i},
        result=result,
        utility=utility,
    )


def test_utility_changes_the_ranking_as_specified():
    memory = EpisodicMemory()
    q = np.array([1.0, 0.0])
    near = np.array([0.9, np.sqrt(1 - 0.81)])  # cosine 0.9
    far = np.array([0.8, 0.6])  # cosine 0.8
    memory.store(record(1), {"key_probe": near})
    memory.store(record(2), {"key_probe": far})
    first = memory.recall_similar("key_probe", q, 2)
    assert [r.record.id for r in first] == ["ep_0001", "ep_0002"]
    assert [r.score for r in first] == pytest.approx([0.9 * 1.0, 0.8 * 1.0])
    # Two successes with ep_0002 raise its utility to 0.7: 0.8 × 1.2 = 0.96 > 0.9.
    assert memory.reinforce("ep_0002", True) == pytest.approx(0.6)
    assert memory.reinforce("ep_0002", True) == pytest.approx(0.7)
    second = memory.recall_similar("key_probe", q, 2)
    assert [r.record.id for r in second] == ["ep_0002", "ep_0001"]
    assert second[0].score == pytest.approx(0.8 * 1.2) and second[0].similarity == pytest.approx(
        0.8
    )
    # A failure lowers it; utility stays in [0, 1].
    for _ in range(10):
        memory.reinforce("ep_0002", False)
    assert memory.records["ep_0002"].utility == 0.0
    for _ in range(20):
        memory.reinforce("ep_0001", True)
    assert memory.records["ep_0001"].utility == 1.0
    assert memory.indexes["key_probe"].payload("ep_0001")["utility"] == 1.0
    with pytest.raises(TypeError):
        memory.reinforce("ep_0001", "False")  # a truthy string is not a success


def test_negative_similarity_still_ranks_higher_utility_first():
    for similarity in (-0.8, -0.1, 0.0, 0.3):
        scores = [ranking_score(similarity, u) for u in (0.0, 0.5, 1.0)]
        assert scores == sorted(scores) and scores[1] == pytest.approx(similarity)
    memory = EpisodicMemory()
    q = np.array([1.0, 0.0])
    memory.store(record(1), {"key_probe": np.array([-0.6, 0.8])})  # cosine -0.6
    memory.store(record(2), {"key_probe": np.array([-0.6, -0.8])})  # cosine -0.6
    memory.reinforce("ep_0002", True)
    memory.reinforce("ep_0001", False)
    assert [r.record.id for r in memory.recall_similar("key_probe", q, 2)] == [
        "ep_0002",
        "ep_0001",
    ]
    # Euclidean scores (negative distances) follow the same rule.
    far = EpisodicMemory(lambda: NumpyBackend("euclidean"))
    far.store(record(1), {"key_probe": np.array([2.0, 0.0])})
    far.store(record(2), {"key_probe": np.array([3.0, 0.0])})
    far.reinforce("ep_0002", True)
    hits = far.recall_similar("key_probe", np.zeros(2), 2)
    assert [r.record.id for r in hits] == ["ep_0001", "ep_0002"]
    assert [r.score for r in hits] == pytest.approx([-2.0, -3.0 / 1.1])


@pytest.mark.parametrize("success", [None, True, False])
def test_recall_with_utilities_equals_brute_force(success):
    """Random utilities and similarities of both signs: recall_similar returns the
    brute-force ranking, including when the top k sit far down the similarity order."""
    rng = np.random.default_rng(7)
    memory = EpisodicMemory()
    data = rng.normal(size=(300, 8))
    utilities = rng.uniform(0.0, 1.0, size=300)
    utilities[:150] = np.where(rng.uniform(size=150) < 0.5, 0.0, 1.0)
    for i, v in enumerate(data):
        result = "SUCCESS" if i % 3 else "SLIP"
        memory.store(record(i, result, float(utilities[i])), {"key_probe": v})
    for q in rng.normal(size=(20, 8)):
        sims = unit(data) @ unit(q)
        truth = [
            (f"ep_{i:04d}", ranking_score(float(sims[i]), float(utilities[i])))
            for i in np.argsort(-sims, kind="stable")
            if success is None or (i % 3 != 0) == success
        ]
        truth.sort(key=lambda pair: -pair[1])
        for k in (1, 5, 40, 300):
            got = memory.recall_similar("key_probe", q, k, success=success)
            assert [r.record.id for r in got] == [i for i, _ in truth[:k]]
            assert [r.score for r in got] == pytest.approx([s for _, s in truth[:k]], abs=1e-12)


def test_recall_reaches_a_high_utility_record_far_down_the_similarity_order():
    """100 close records at utility 0 rank below a farther one at utility 1, which
    sits beyond the first search window, so recall must widen the window."""
    memory = EpisodicMemory()
    q = np.array([1.0, 0.0])
    for i in range(100):
        memory.store(record(i, utility=0.0), {"key_probe": np.array([0.99, 0.1 + 0.001 * i])})
    memory.store(record(500, utility=1.0), {"key_probe": np.array([0.5, np.sqrt(0.75)])})
    top = memory.recall_similar("key_probe", q, 3)
    assert [r.record.id for r in top] == ["ep_0500", "ep_0000", "ep_0001"]
    assert top[0].score == pytest.approx(0.5 * 1.5) and top[0].similarity == pytest.approx(0.5)
    # The window's bound must assume utility 1: 40 records at cosine 0.9 and
    # utility 0.5 (score 0.9) fill it; one at 0.85 and utility 1 scores 1.275.
    tight = EpisodicMemory()
    for i in range(40):
        tight.store(record(i), {"key_probe": np.array([0.9, np.sqrt(1 - 0.81)])})
    tight.store(record(99, utility=1.0), {"key_probe": np.array([0.85, np.sqrt(1 - 0.85**2)])})
    assert tight.recall_similar("key_probe", q, 1)[0].record.id == "ep_0099"
    with pytest.raises(TypeError):
        tight.recall_similar("key_probe", q, 1, success="False")
    # The bound uses the highest stored utility (0.8 here), not a smaller guess:
    # 40 records at 0.9 and utility 0.72 score 1.098; one at 0.85 and utility 0.8
    # scores 1.105, beyond the window.
    below = EpisodicMemory()
    for i in range(40):
        below.store(record(i, utility=0.72), {"key_probe": np.array([0.9, np.sqrt(1 - 0.81)])})
    below.store(record(99, utility=0.8), {"key_probe": np.array([0.85, np.sqrt(1 - 0.85**2)])})
    assert below.recall_similar("key_probe", q, 1)[0].record.id == "ep_0099"


def test_store_validates_every_key_first_and_keeps_a_plain_copy():
    memory = EpisodicMemory()
    memory.store(record(1), {"key_pre": np.ones(3), "key_probe": np.ones(4)})
    bad = {"key_pre": np.ones(3), "key_probe": np.zeros(4)}  # the zero vector fails
    with pytest.raises(ValueError):
        memory.store(record(2), bad)
    assert len(memory) == 1 and all(len(memory.indexes[k]) == 1 for k in bad)
    theta = {"grip_force": np.float64(6.0)}
    rec = EpisodeRecord(
        "ep_np",
        np.int64(7),
        np.int32(2),
        theta,
        "SLIP",
        info={"force": np.float32(1.5)},
        theta_final={"grip_force": 7.5},
    )
    memory.store(rec, {"key_pre": np.ones(3)})
    theta["grip_force"] = 99.0  # the caller's dict, not the stored copy
    stored = memory.records["ep_np"]
    assert stored.theta == {"grip_force": 6.0} and type(stored.situation) is int
    assert stored.theta_final == {"grip_force": 7.5} and stored.failure_mode == "SLIP"
    assert stored.info == {"force": 1.5} and "info" not in memory.indexes["key_pre"].payload(
        "ep_np"
    )
    recalled = memory.recall_similar("key_pre", np.ones(3), 1, success=False)[0].record
    recalled.theta["grip_force"] = 0.0
    assert memory.records["ep_np"].theta == {"grip_force": 6.0}
    with pytest.raises(ValueError):
        memory.store(record(3, utility=1.5), {})


def test_episodic_filters_and_key_indexes():
    memory = EpisodicMemory()
    rng = np.random.default_rng(3)
    for i in range(12):
        result = "SUCCESS" if i % 3 else "BREAK"
        keys = {"key_pre": rng.normal(size=4), "key_probe": rng.normal(size=6)}
        if result != "SUCCESS":
            keys["key_retry"] = rng.normal(size=8)  # only failures have one
        memory.store(record(i, result), keys)
    q = rng.normal(size=6)
    assert all(r.record.success for r in memory.recall_similar("key_probe", q, 12, success=True))
    assert len(memory.recall_similar("key_probe", q, 12, success=False)) == 4
    assert len(memory.recall_similar("key_retry", rng.normal(size=8), 12)) == 4
    picked = memory.recall_similar("key_probe", q, 12, filter={"situation": 1005})
    assert [r.record.id for r in picked] == ["ep_0005"]
    breaks = memory.recall_similar("key_probe", q, 12, filter={"failure_mode": "BREAK"})
    assert sorted(r.record.id for r in breaks) == ["ep_0000", "ep_0003", "ep_0006", "ep_0009"]
    memory.reinforce("ep_0003", True)
    for name in ("key_pre", "key_probe", "key_retry"):  # every index holding it
        assert memory.indexes[name].payload("ep_0003")["utility"] == pytest.approx(0.6)
    with pytest.raises(KeyError):
        memory.store(record(1), {"key_pre": rng.normal(size=4)})
    with pytest.raises(ValueError):
        memory.store(record(99), {"key_other": rng.normal(size=4)})
    assert set(memory.indexes) == set(KEY_TYPES)


def test_semantic_memory_finds_and_replaces_concepts():
    semantic = SemanticMemory()
    semantic.store("skill_00", np.array([1.0, 0.0]), {"grip_force": 3.0})
    semantic.store("skill_01", np.array([0.0, 1.0]), {"grip_force": 15.0})
    assert semantic.find_similar(np.array([0.9, 0.1]), 1)[0][0] == "skill_00"
    semantic.store("skill_02", np.array([1.0, 3.0]))  # re-normalizing it would move a bit
    before = semantic.retrieve("skill_02")[0]
    semantic.store("skill_00", np.array([0.0, -1.0]), {"grip_force": 2.0})  # replaced
    assert np.array_equal(semantic.retrieve("skill_02")[0], before)  # others bit-identical
    assert len(semantic) == 3
    vector, payload = semantic.retrieve("skill_00")
    assert np.allclose(vector, [0.0, -1.0]) and payload == {"grip_force": 2.0}
    assert semantic.retrieve("missing") is None
    # A failed replacement keeps the old concept.
    with pytest.raises(ValueError):
        semantic.store("skill_00", np.array([1.0, 1.0]), {"grip_force": float("nan")})
    with pytest.raises(ValueError):
        semantic.store("skill_00", np.zeros(2), {})
    assert len(semantic) == 3 and semantic.retrieve("skill_00")[1] == {"grip_force": 2.0}
    semantic.clear()
    assert len(semantic) == 0 and semantic.find_similar(np.array([1.0, 0.0]), 3) == []


def test_memory_system_save_load_gives_identical_results(tmp_path):
    system = MemorySystem(working_size=4)
    rng = np.random.default_rng(5)
    dims = {"key_pre": 5, "key_probe": 7, "key_retry": 9, "full_embedding": 11}
    for i in range(30):
        keys = {name: rng.normal(size=d) for name, d in dims.items()}
        if i % 2:
            del keys["key_retry"]  # only failures have one
        system.episodic.store(record(i, "SUCCESS" if i % 2 else "DROP"), keys)
    system.episodic.reinforce("ep_0003", True)
    system.semantic.store("skill_00", rng.normal(size=7), {"n": 3})
    path = system.save(tmp_path / "memory")
    loaded = MemorySystem.load(path)
    for name, d in dims.items():
        for q in rng.normal(size=(5, d)):
            for success in (None, True, False):
                a = system.episodic.recall_similar(name, q, 6, success=success)
                b = loaded.episodic.recall_similar(name, q, 6, success=success)
                assert [(r.record.id, r.similarity, r.score) for r in a] == [
                    (r.record.id, r.similarity, r.score) for r in b
                ]
                assert len(b) == (0 if name == "key_retry" and success else 6)
    for q in rng.normal(size=(5, 7)):
        assert system.semantic.find_similar(q, 1) == loaded.semantic.find_similar(q, 1)
    assert loaded.episodic.records["ep_0003"].utility == pytest.approx(0.6)
    assert loaded.working.max_size == 4
    assert loaded.semantic.retrieve("skill_00")[1] == {"n": 3}
    assert loaded.episodic.records == system.episodic.records
    # A Euclidean system keeps its metric through save, load, and a replaced concept.
    euclid = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
    euclid.semantic.store("a", np.array([3.0, 0.0]))
    euclid.semantic.store("b", np.array([0.0, 1.0]))
    again = MemorySystem.load(euclid.save(tmp_path / "euclid"), lambda: NumpyBackend("euclidean"))
    again.semantic.store("b", np.array([0.0, 2.0]))
    assert again.semantic.find_similar(np.zeros(2), 2) == [("b", -2.0), ("a", -3.0)]
    with pytest.raises(ValueError, match="metric"):
        MemorySystem.load(tmp_path / "euclid")  # the default factory makes cosine

    class OtherBackend(NumpyBackend):
        pass

    with pytest.raises(ValueError, match="saved by NumpyBackend"):
        MemorySystem.load(path, OtherBackend)
    meta = json.loads((path / "memory.json").read_text())
    (path / "memory.json").write_text(json.dumps({**meta, "episodes": 29}))
    with pytest.raises(ValueError, match="expected 29"):
        MemorySystem.load(path)


@pytest.mark.parametrize("crash_at", ["semantic", "pointer"])
def test_an_interrupted_save_keeps_the_previous_one(tmp_path, monkeypatch, crash_at):
    """A crash while the new version is written, or after it is complete but before
    memory.json points at it, leaves the previous save loadable; the next save
    clears the leftovers."""
    system = MemorySystem()
    system.episodic.store(record(1), {"key_probe": np.ones(3)})
    path = system.save(tmp_path / "memory")
    system.episodic.store(record(2), {"key_probe": np.ones(3)})
    system.episodic.reinforce("ep_0001", True)

    real_write = MS.atomic_write

    def crash(*args):
        raise OSError("killed")

    def crash_on_pointer(target, write):
        if target.name == "memory.json":
            crash()
        real_write(target, write)

    if crash_at == "semantic":
        monkeypatch.setattr(SemanticMemory, "save", crash)
    else:
        monkeypatch.setattr(MS, "atomic_write", crash_on_pointer)
    with pytest.raises(OSError):
        system.save(path)
    old = MemorySystem.load(path)
    assert set(old.episodic.records) == {"ep_0001"}
    assert old.episodic.records["ep_0001"].utility == 0.5
    monkeypatch.undo()
    assert len(MemorySystem.load(system.save(path)).episodic) == 2
    assert sorted(e.name for e in path.iterdir()) == ["memory.json", "v000001"]


def test_save_tolerates_finder_files_and_long_version_numbers(tmp_path):
    system = MemorySystem()
    system.episodic.store(record(1), {"key_probe": np.ones(3)})
    path = system.save(tmp_path / "memory")
    (path / ".DS_Store").write_bytes(b"\0")
    (path / ".memory.json.99999.tmp").write_text("{")  # a crashed pointer write
    meta = json.loads((path / "memory.json").read_text())
    (path / meta["version"]).rename(path / "v999999")
    (path / "memory.json").write_text(json.dumps({**meta, "version": "v999999"}))
    for _ in range(2):
        system.save(path)
    assert sorted(e.name for e in path.iterdir()) == [".DS_Store", "memory.json", "v1000001"]
    assert len(MemorySystem.load(path).episodic) == 1
    # The tier and backend saves never overwrite a directory in place.
    for save in (system.episodic.save, system.semantic.save):
        with pytest.raises(ValueError, match="new or empty"):
            save(path)
    (path / "v1000002").write_text("not a version")  # a file named like the next one
    with pytest.raises(ValueError, match="other files"):
        system.save(path)
    assert len(MemorySystem.load(path).episodic) == 1


def test_save_never_deletes_files_it_did_not_write(tmp_path):
    system = MemorySystem()
    system.episodic.store(record(1), {"key_probe": np.ones(3)})
    run = system.save(tmp_path / "run1")
    (run / "seed0.json").write_text("{}")  # a runner writes into the same directory
    (tmp_path / "file").write_text("x")
    for target in (run, tmp_path / "file", tmp_path):
        with pytest.raises(ValueError):
            system.save(target)
    assert (run / "seed0.json").read_text() == "{}"
    assert len(MemorySystem.load(run).episodic) == 1
    (tmp_path / "link").symlink_to(system.save(tmp_path / "memory"))
    system.episodic.store(record(2), {"key_probe": np.ones(3)})
    assert len(MemorySystem.load(system.save(tmp_path / "link")).episodic) == 2
    assert len(MemorySystem.load(tmp_path / "memory").episodic) == 2


def test_real_probe_keys_recall_the_same_class():
    """On the calibration fixture: the nearest other cube by key_probe mostly comes
    from the same hidden class (a sanity check, labels for scoring only)."""
    fixture = M.load_fixture()
    views = M.fixture_views(fixture)
    emb = M.Embedder()
    memory = EpisodicMemory()
    first = [i for i in range(len(views)) if fixture["attempt"][i] == 1]
    for i in first:
        memory.store(
            EpisodeRecord(f"fx_{i}", int(fixture["group"][i]), 1, {}, str(fixture["result"][i])),
            {"key_probe": emb.key_probe(views[i]), "key_pre": emb.key_pre(views[i].context)},
        )
    same = 0
    for i in first:
        hits = memory.recall_similar(
            "key_probe", emb.key_probe(views[i]), 1, filter=lambda p, i=i: p["id"] != f"fx_{i}"
        )
        j = int(hits[0].record.id[3:])
        same += fixture["hidden_class"][j] == fixture["hidden_class"][i]
    rate = same / len(first)
    print(f"nearest other cube by key_probe has the same class: {rate:.3f}")
    assert rate > 0.6
