"""T15: retry until success with memory (``vfa.cognition.retry``), with physics.

``solve_situation`` resets the arm and the same cube after each failure, stores
every attempt, reinforces the records its decisions used, and consolidates the
skills every N situations. The quick-profile learning run is the Verify
command's benchmark step (``--condition memory_full --profile quick``).
"""

from __future__ import annotations

import re

import gymnasium as gym
import pytest

from vfa.cognition import system2 as S2
from vfa.cognition.cognitive_core import CognitiveCore
from vfa.cognition.retry import solve_situation
from vfa.cognition.system2 import RetrievalPolicy
from vfa.experiments import benchmark as B
from vfa.memory.skill_learning import TRAJECTORY_KEY
from vfa.sim import objects as OB
from vfa.skills.pick import GraspLiftSkill, run_attempt

CONFIG = B.load_benchmark_config()
# Quick seed 500, situations 0-3: light_rigid (8 N holds), brittle (8 N breaks),
# heavy_rigid, fragile_soft (8 N breaks).
SITUATIONS = B.make_situations(500, 4, OB.load_object_config())


class Resets(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.options = []

    def reset(self, **kwargs):
        self.options.append(kwargs.get("options"))
        return self.env.reset(**kwargs)


def run(policy: RetrievalPolicy, situations=SITUATIONS, env=None):
    env = env if env is not None else B.make_env(CONFIG)
    core = CognitiveCore(policy)
    results = {}
    for s in situations:
        results[s.index] = solve_situation(
            env,
            core,
            GraspLiftSkill(),
            CONFIG.max_attempts,
            key=s.key,
            seed_of=lambda a, s=s: B.env_seed(s.seed, s.index, a),
            options={"object": s.spec},
        )
    return core, results


@pytest.fixture(scope="module")
def solved():
    env = Resets(B.make_env(CONFIG))
    policy = RetrievalPolicy()
    core, results = run(policy, env=env)
    return env, policy, core, results


def test_every_attempt_is_stored_with_observable_data(solved):
    _, policy, _, results = solved
    attempts = [r for rs in results.values() for r in rs]
    records = list(policy.memory.episodic.records.values())
    assert len(records) == len(attempts) > len(SITUATIONS)  # some cube needed a retry
    assert all(rs[-1].success for rs in results.values())
    for record, result in zip(records, attempts, strict=True):
        assert record.result == (
            "UNDERGRIP" if result.result in ("SLIP", "DROP") else result.result
        )
        assert record.theta_final == result.theta_final.to_dict()
        assert len(record.info[TRAJECTORY_KEY]) > 0
    for name in ("key_pre", "full_embedding", "key_probe"):
        assert len(policy.memory.episodic.indexes[name]) == len(records)
    failures = sum(not r.success for r in records)
    assert len(policy.memory.episodic.indexes["key_retry"]) == failures > 0


def test_retries_reset_the_same_cube(solved):
    env, _, _, results = solved
    specs = [o["object"] for o in env.options]
    # Two resets per attempt (System 2 looks first), all on the situation's cube.
    expected = [s.spec for s in SITUATIONS for _ in results[s.index] for _ in range(2)]
    assert specs == expected


def test_used_records_are_reinforced(solved):
    _, policy, core, _ = solved
    used = {i for d in core.decisions for i in d.retrieved}
    assert used
    utilities = {i: policy.memory.episodic.records[i].utility for i in used}
    assert any(u != 0.5 for u in utilities.values())
    unused = set(policy.memory.episodic.records) - used
    assert all(policy.memory.episodic.records[i].utility == 0.5 for i in unused)


def test_memory_holds_no_hidden_information(solved):
    _, policy, _, _ = solved
    memory = policy.memory
    policy.library.consolidate(memory)  # the skills' payloads too
    assert len(memory.semantic) > 0
    parts = [list(memory.episodic.records.values())]
    for index in memory.episodic.indexes.values():
        parts.append([index.payload(i) for i in index.ids])
    parts.append([memory.semantic.index.payload(i) for i in memory.semantic.index.ids])
    text = repr(parts)
    for s in SITUATIONS:
        assert s.class_name not in text
        hidden = s.spec.hidden()
        values = [hidden["mass"], hidden["friction"][0], hidden["break_force"]]
        # The sampled properties (fixed class constants such as 0.005 would match by chance).
        for value in [float(v) for v in values if v is not None and float(v) != round(v, 3)]:
            assert not re.search(rf"(?<![\d.]){re.escape(repr(value))}(?!\d)", text), value


def test_the_first_attempt_is_paired_with_naive(solved):
    """Empty memory: memory_full's first attempt is the naive attempt, bit for bit (this
    cube has no slip escalation, the one place the policy could differ)."""
    _, _, _, results = solved
    s = SITUATIONS[0]
    naive = run_attempt(
        B.make_env(CONFIG),
        GraspLiftSkill(),
        B.NAIVE,
        seed=B.env_seed(s.seed, s.index, 1),
        options={"object": s.spec},
    )
    first = results[s.index][0]
    assert (first.result, first.sim_s, first.theta_final) == (
        naive.result,
        naive.sim_s,
        naive.theta_final,
    )


def test_skills_are_consolidated_every_n_situations(monkeypatch):
    monkeypatch.setattr(S2, "CONSOLIDATE_EVERY", 2)
    policy = RetrievalPolicy()
    run(policy, SITUATIONS[:2])
    assert policy.situations == 2 and len(policy.memory.semantic) >= 1
    assert len(policy.library) == len(policy.memory.semantic)


def test_context_only_keys():
    policy = RetrievalPolicy(keys="context")
    run(policy, SITUATIONS[:1])
    dims = {k: idx.dim for k, idx in policy.memory.episodic.indexes.items() if len(idx)}
    assert set(dims.values()) == {12}


def test_max_attempts_is_checked():
    with pytest.raises(ValueError, match="max_attempts"):
        solve_situation(
            None, CognitiveCore(RetrievalPolicy()), GraspLiftSkill(), 0, key=1, seed_of=int
        )


def test_each_attempt_reinforces_only_its_own_decisions():
    """The ids an attempt reinforces come from that attempt's slice of the core's
    decisions, and the θ it stores is its probe decision's."""
    from vfa.cognition.cognitive_core import ESCALATION, PROBE, START, Decision
    from vfa.skills.params import SkillParams
    from vfa.skills.pick import AttemptResult

    class Core:
        def __init__(self, policy):
            self.policy, self.decisions, self.n = policy, [], 0

        def run_attempt(self, env, skill, *, seed, options, previous):
            self.n += 1
            grip = SkillParams(grip_force=float(self.n))
            self.decisions += [
                Decision(B.NAIVE, START),
                Decision(grip, PROBE, retrieved=(f"r{self.n}",)),
                Decision(SkillParams(grip_force=25.0), ESCALATION, retrieved=()),
            ]
            ok = self.n == 3
            return AttemptResult(
                ok, "SUCCESS" if ok else "BREAK", B.NAIVE, grip, None, [], None, 0.0, 0.0
            )

    class Spy:
        def __init__(self):
            self.calls = []

        def begin_situation(self, key):
            pass

        def begin_attempt(self, attempt):
            pass

        def learn_attempt(self, observed, used, chosen):
            self.calls.append((used, chosen.grip_force))

        def end_situation(self):
            pass

    spy = Spy()
    results = solve_situation(None, Core(spy), None, 5, key=1, seed_of=int)
    assert [r.success for r in results] == [False, False, True]
    assert spy.calls == [(["r1"], 1.0), (["r2"], 2.0), (["r3"], 3.0)]


def test_the_chunk_key_covers_the_embedding_statistics(tmp_path, monkeypatch):
    stats = tmp_path / "stats.npz"
    stats.write_bytes(b"a")
    monkeypatch.setattr(B, "STATS_PATH", stats)
    before = B.code_fingerprint()
    stats.write_bytes(b"b")
    assert B.code_fingerprint() != before


def test_aggregate_leaves_out_memory_conditions_never_run(tmp_path):
    result = B.aggregate(CONFIG, "quick", out=tmp_path, partial=True)
    assert set(result["missing"]) == set(B.CONDITIONS)
