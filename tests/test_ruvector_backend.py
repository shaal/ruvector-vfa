"""T22: the real RuVector backend (PyO3 binding over ``ruvector-robotics``).

ROADMAP acceptance: ``RuVectorBackend`` and ``NumpyBackend`` return the same
top-k ids in the same order on 1 000 random queries (here also the same
scores); ``vfa demo --quick --backend ruvector`` matches the numpy run (a
benchmark chunk with both backends here, the full demo in docs/RESULTS.md §12).
Every test skips when the Rust extension is not built (``uv sync --extra
ruvector && uv run maturin develop -m rust/ruvector_vfa_py/Cargo.toml``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

R = pytest.importorskip("ruvector_vfa_py", reason="the Rust extension is not built")

from vfa.experiments import benchmark as B  # noqa: E402
from vfa.memory import embeddings as EM  # noqa: E402
from vfa.memory.backend import NumpyBackend  # noqa: E402
from vfa.memory.memory_system import EpisodeRecord, MemorySystem  # noqa: E402
from vfa.memory.ruvector_backend import RuVectorBackend  # noqa: E402

DIM = EM.key_dims("key_probe")


def _pair(n: int, rng: np.random.Generator, dupes: int = 0) -> tuple[NumpyBackend, RuVectorBackend]:
    a, b = NumpyBackend(), RuVectorBackend()
    vectors = rng.standard_normal((n, DIM))
    vectors[n - dupes :] = vectors[:dupes]  # exact ties: the order must be insertion order
    for i, v in enumerate(vectors):
        payload = {
            "success": bool(i % 3),
            "theta": {"grip_force": float(i % 25 + 1)},
            "situation": i % 7,
        }
        a.add(f"r{i:04d}", v, payload)
        b.add(f"r{i:04d}", v, payload)
    return a, b


def test_parity_on_1000_random_queries() -> None:
    rng = np.random.default_rng(22)
    a, b = _pair(1000, rng, dupes=50)
    for _ in range(1000):
        q = rng.standard_normal(DIM)
        ha, hb = a.search(q, 5), b.search(q, 5)
        assert [(h.id, h.score) for h in ha] == [(h.id, h.score) for h in hb]
    for v in (a.vector("r0000"), -a.vector("r0001")):  # a stored vector: ties at the top
        assert [h.id for h in a.search(v, 20)] == [h.id for h in b.search(v, 20)]


def test_filters_payloads_and_updates_match() -> None:
    rng = np.random.default_rng(5)
    a, b = _pair(300, rng)
    q = rng.standard_normal(DIM)
    for flt in ({"situation": 3}, {"success": False}, lambda p: p["theta"]["grip_force"] > 20):
        ha, hb = a.search(q, 7, flt), b.search(q, 7, flt)
        assert [(h.id, h.score, h.payload) for h in ha] == [(h.id, h.score, h.payload) for h in hb]
    a.update("r0007", {"utility": 0.9})
    b.update("r0007", {"utility": 0.9})
    assert b.payload("r0007") == a.payload("r0007") and "r0007" in b and len(b) == 300
    np.testing.assert_array_equal(a.vector("r0011"), b.vector("r0011"))
    assert b.search(q, 0) == [] and RuVectorBackend().search(q, 3) == []
    with pytest.raises(ValueError):
        RuVectorBackend("euclidean")


def test_save_and_load_keep_the_ranking(tmp_path) -> None:
    rng = np.random.default_rng(9)
    _, b = _pair(200, rng)
    b.save(tmp_path / "idx")
    again = RuVectorBackend.load(tmp_path / "idx")
    q = rng.standard_normal(DIM)
    assert [(h.id, h.score) for h in again.search(q, 10)] == [
        (h.id, h.score) for h in b.search(q, 10)
    ]
    assert len(again._rust) == 200  # the episodes went back into the crate


def test_episodes_carry_the_id_theta_and_signed_outcome() -> None:
    b = RuVectorBackend()
    b.add("win", np.ones(4), {"success": True, "theta": {"grip_force": 8.0}})
    b.add("loss", -np.ones(4), {"success": False, "theta": {"grip_force": 2.0}})
    b.add("plain", np.array([1.0, 0, 0, 0]), {})
    episodes = b._rust.recall_similar([1.0, 1.0, 1.0, 1.0], 3)
    assert episodes[0] == (["win", json.dumps({"grip_force": 8.0})], 1.0, 0)
    assert {e[0][0]: e[1] for e in episodes} == {"win": 1.0, "loss": -1.0, "plain": 0.0}


def test_memory_system_runs_on_the_crate(tmp_path) -> None:
    rng = np.random.default_rng(1)
    systems = [MemorySystem(), MemorySystem(backend_factory=RuVectorBackend)]
    keys = rng.standard_normal((40, DIM))
    for system in systems:
        for i, key in enumerate(keys):
            record = EpisodeRecord(
                f"ep{i:03d}", i % 8, 1, {"grip_force": 8.0}, "SUCCESS" if i % 2 else "BREAK"
            )
            system.episodic.store(record, {"key_probe": key})
        system.episodic.reinforce("ep005", True)
    q = rng.standard_normal(DIM)
    hits = [s.episodic.recall_similar("key_probe", q, 6) for s in systems]
    assert [(h.record.id, h.similarity, h.score) for h in hits[0]] == [
        (h.record.id, h.similarity, h.score) for h in hits[1]
    ]
    loaded = MemorySystem.load(systems[1].save(tmp_path / "mem"), RuVectorBackend)
    assert [h.record.id for h in loaded.episodic.recall_similar("key_probe", q, 6)] == [
        h.record.id for h in hits[1]
    ]


def test_the_binding_exposes_the_crate_types() -> None:
    semantic = R.SemanticMemory()
    semantic.store("a", [1.0, 0.0])
    semantic.store("b", [0.0, 1.0])
    assert semantic.find_similar([1.0, 0.25], 2) == [("a", 1.0), ("b", 0.25)]
    assert semantic.retrieve("b") == [0.0, 1.0] and len(semantic) == 2
    library = R.SkillLibrary()
    demos = [
        ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [0, 1], ""),
        ([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], [0, 1], ""),
    ]
    name, trajectory, confidence, count = library.learn_from_demonstration("grip", demos)
    assert trajectory == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]] and confidence == pytest.approx(2 / 3)
    library.improve_skill("grip", 0.9)
    assert library.get("grip")[2] == 1.0  # clamped, as the Python mirror does (T13)
    core = R.CognitiveCore()
    assert (core.state, core.mode) == ("Idle", "Reactive")
    assert core.perceive("probe", [1.0], 0.9, 0) == "Perceiving"
    assert core.think() is not None and core.state == "Thinking"
    core.learn(True, 1.0, "ok")
    assert core.state == "Idle" and core.cumulative_reward == pytest.approx(0.01)
    assert R.RUVECTOR_REV == "3e8429739fc0a3878b553a87bb91e7f66d8c0cbd"


@pytest.mark.slow
def test_a_memory_chunk_matches_the_numpy_backend(tmp_path) -> None:
    """memory_full on 6 situations: the same records with either backend."""
    config = B.load_benchmark_config()
    records = []
    for backend in ("numpy", "ruvector"):
        job = B.plan_condition(
            config, "memory_full", "quick", [500], tmp_path / backend, backend=backend
        )[0]
        job = B.Job(**{**{f: getattr(job, f) for f in B.Job.__dataclass_fields__}, "end": 6})
        B.run_chunk(job)
        rows = json.loads(open(job.path).read())["records"]
        records.append([{k: v for k, v in r.items() if k != "wall_s"} for r in rows])
    assert records[0] == records[1]
