"""T18: the component ablations and the frozen memory of ``RetrievalPolicy`` (no physics)."""

from __future__ import annotations

import numpy as np
import pytest

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    FAILURE,
    PROBE,
    START,
    ObservedAttempt,
    Percept,
)
from vfa.cognition.system2 import RetrievalPolicy
from vfa.memory import embeddings as M
from vfa.memory.memory_system import EpisodeRecord
from vfa.skills.params import SkillParams

VIEW = M.fixture_views(M.load_fixture())[0]  # a probed calibration attempt


def store(policy: RetrievalPolicy, rid: str, situation: int, key, result: str, grip: float) -> None:
    theta = {**DEFAULT_THETA.to_dict(), "grip_force": grip}
    record = EpisodeRecord(rid, situation, 1, theta, result, theta_final=theta)
    policy.memory.episodic.store(record, {"key_probe": key})


def observed(result: str, grip: float, view=VIEW) -> ObservedAttempt:
    final = SkillParams(grip_force=grip)
    return ObservedAttempt(result, result == "SUCCESS", final, final, None, (), None, view)


def test_without_reinforcement_every_utility_stays_at_its_start():
    policy = RetrievalPolicy(reinforce=False)
    store(policy, "used", 1, policy.key("key_probe", VIEW), "SUCCESS", 5.0)
    policy.begin_situation(2)
    policy.begin_attempt(1)
    policy.learn_attempt(observed("BREAK", 6.0), ["used"], SkillParams(grip_force=6.0))
    policy.learn_attempt(observed("SUCCESS", 4.0), ["used"], SkillParams(grip_force=4.0))
    assert {r.utility for r in policy.memory.episodic.records.values()} == {0.5}
    assert len(policy.memory.episodic) == 3  # still stored


def test_without_the_semantic_tier_no_skill_and_no_consolidation(monkeypatch):
    from vfa.memory.skill_learning import Demonstration

    policy = RetrievalPolicy(semantic=False)
    theta = {**DEFAULT_THETA.to_dict(), "grip_force": 3.0}
    policy.library.learn_from_demonstration("skill_00", [Demonstration([], theta=theta)] * 3)
    policy.memory.semantic.store("skill_00", np.array([1.0, 0.0]), {"theta": theta})
    store(policy, "far", 1, np.array([0.0, 1.0]), "SUCCESS", 20.0)
    policy.begin_situation(2)
    policy.begin_attempt(1)
    assert policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).theta == DEFAULT_THETA
    monkeypatch.setattr(policy.library, "consolidate", lambda memory: pytest.fail("consolidated"))
    for _ in range(20):
        policy.end_situation()


def test_without_failure_bounds_a_retry_repeats_the_default_grip():
    for bounds, expect in ((True, "bisect"), (False, "default")):
        policy = RetrievalPolicy(failure_bounds=bounds)
        store(policy, "own", 7, np.array([1.0, 0.0]), "BREAK", 8.0)
        store(policy, "nb", 8, np.array([1.0, 0.0]), "UNDERGRIP", 2.0)  # a strong failure
        policy.begin_situation(7)
        policy.begin_attempt(2)
        decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
        assert decision.reasoning.startswith(expect)
        if not bounds:
            assert decision.theta == DEFAULT_THETA and decision.retrieved == ()
        else:
            assert decision.theta.grip_force < 8.0
    assert policy.decide(Percept("escalation", {}), DEFAULT_THETA).theta.grip_force > 8.0


def test_without_the_probe_trigger_the_retry_decision_is_the_proposal():
    policy = RetrievalPolicy(probe_trigger=False)
    store(policy, "nb", 9, policy.key("key_probe", VIEW), "SUCCESS", 3.0)
    policy.begin_situation(4)
    policy.begin_attempt(1)
    obs = {
        "cube_pos": np.array([0.5, 0.0, 0.02]),
        "cube_yaw": np.array([0.0]),
        "cube_size": np.array([0.05]),
    }
    assert policy.decide(Percept(START, {"obs": obs}), DEFAULT_THETA).theta == DEFAULT_THETA
    held = SkillParams(grip_force=11.0)
    probe = policy.decide(Percept(PROBE, {"view": VIEW}), held)
    assert probe.theta == held and probe.retrieved == ()  # no decision at the probe
    policy.begin_attempt(2)
    retry = policy.decide(Percept(FAILURE, {"previous": observed("BREAK", 8.0)}), DEFAULT_THETA)
    assert retry.trigger == FAILURE and retry.retrieved == ("nb",)
    assert retry.theta.grip_force == pytest.approx(3.0, rel=0.3)
    with_probe = RetrievalPolicy()
    with_probe.begin_situation(4)
    with_probe.begin_attempt(2)
    plain = with_probe.decide(Percept(FAILURE, {"previous": observed("BREAK", 8.0)}), DEFAULT_THETA)
    assert plain.theta == DEFAULT_THETA  # the T15 retry start keeps the default θ


def test_without_the_probe_trigger_a_miss_still_bisects_on_own_failures():
    policy = RetrievalPolicy(probe_trigger=False)
    store(policy, "own", 5, np.array([1.0, 0.0]), "BREAK", 8.0)
    policy.begin_situation(5)
    policy.begin_attempt(2)
    miss = observed("MISS", 8.0, view=None)
    decision = policy.decide(Percept(FAILURE, {"previous": miss}), DEFAULT_THETA)
    assert decision.theta.grip_force < 8.0 and decision.retrieved == ("own",)


def test_a_frozen_memory_takes_no_writes_but_bounds_the_cube_it_holds():
    policy = RetrievalPolicy()
    key = policy.key("key_probe", VIEW)
    store(policy, "used", 1, key, "SUCCESS", 5.0)
    policy.freeze()
    before = {rid: r.utility for rid, r in policy.memory.episodic.records.items()}
    policy.begin_situation(2)
    policy.begin_attempt(1)
    rid = policy.learn_attempt(observed("BREAK", 6.0), ["used"], SkillParams(grip_force=6.0))
    assert {r: x.utility for r, x in policy.memory.episodic.records.items()} == before
    assert rid not in policy.memory.episodic.records and len(policy.scratch) == 1
    policy.begin_attempt(2)
    decision = policy.propose(key, DEFAULT_THETA)
    assert decision.theta.grip_force <= 6.0 * 0.9 + 1e-9  # the scratch break bounds the retry
    policy.end_situation()
    policy.begin_situation(3)
    assert policy.scratch == [] and policy._own_failures() == []


def test_the_benchmark_maps_each_ablation_to_one_flag(tmp_path):
    from vfa.experiments import benchmark as B

    assert {c: tuple(v) for c, v in B.MEMORY_POLICIES.items()} == {
        "memory_full": (),
        "memory_ctx_only": ("keys",),
        "memory_no_reinforce": ("reinforce",),
        "memory_no_semantic": ("semantic",),
        "memory_no_probe": ("probe_trigger",),
        "memory_no_failure_bounds": ("failure_bounds",),
    }
    for flags in B.MEMORY_POLICIES.values():
        RetrievalPolicy(**flags)
    config = B.load_benchmark_config()
    jobs = B.plan_condition(config, B.FROZEN, "heldout", out=tmp_path)
    assert [(j.seed, j.start, j.end) for j in jobs] == [(2000, 0, 60)]
    assert jobs[0].key["warmup"] == {"seed": 1000, "situations": 60}
    with pytest.raises(ValueError, match="warm-up"):
        B.plan_condition(config, B.FROZEN, "full", out=tmp_path)
