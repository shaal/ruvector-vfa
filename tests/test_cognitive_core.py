"""T14: the dual-process cognitive core (``vfa.cognition.cognitive_core``).

ROADMAP acceptance: a scripted attempt produces the exact mode sequence
``DELIBERATIVE → REACTIVE → DELIBERATIVE(probe) → REACTIVE → LEARNING``; a slip
escalation produces a System 2 entry; a safety fault produces ``EMERGENCY`` and
ends the attempt; System 2 runs only at its three triggers, never inside a
control tick.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from enum import Enum

import gymnasium as gym
import numpy as np
import pytest

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    ESCALATION,
    FAILURE,
    PROBE,
    START,
    UNDERGRIP_RESULT,
    CognitiveCore,
    CognitiveMode,
    CognitiveState,
    Decision,
    DefaultPolicy,
    Handoff,
    ObservedAttempt,
    Outcome,
    Percept,
)
from vfa.control.safety import SafetyLimits
from vfa.memory.embeddings import Embedder, EpisodeView, failure_features
from vfa.sim.env import PandaVFAEnv
from vfa.skills.params import SkillParams
from vfa.skills.pick import (
    PROBE_DONE,
    SLIP_UNRECOVERED,
    GraspLiftSkill,
    ProbeFeatures,
    run_attempt,
)

LIGHT = {"object_class": "light_rigid"}


def make_env(**kwargs) -> PandaVFAEnv:
    return PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False, **kwargs)


class StepFlag(gym.Wrapper):
    """Marks the time spent inside ``env.step`` (one control tick)."""

    def __init__(self, env):
        super().__init__(env)
        self.in_step = False
        self.steps = 0

    def step(self, action):
        self.in_step = True
        try:
            return self.env.step(action)
        finally:
            self.in_step = False
            self.steps += 1


class Recorder(DefaultPolicy):
    """The stub policy, recording each call: trigger, inside a step?, step count."""

    def __init__(self, env: StepFlag | None = None, grip_at: dict[str, float] | None = None):
        super().__init__()
        self.env = env
        self.calls: list[tuple[str, bool, int]] = []
        self.percepts: list[Percept] = []
        self.grip_at = grip_at or {}

    def decide(self, percept: Percept, theta: SkillParams) -> Decision:
        if self.env is not None:
            self.calls.append((percept.source, self.env.in_step, self.env.steps))
        else:
            self.calls.append((percept.source, False, -1))
        self.percepts.append(percept)
        decision = super().decide(percept, theta)
        if percept.source in self.grip_at:
            new = replace(decision.theta, grip_force=self.grip_at[percept.source])
            return Decision(new, percept.source, "scripted", retrieved=("ep_0001",))
        return decision


def sequence(core: CognitiveCore) -> list[tuple[str, str | None]]:
    return [(h.kind, h.trigger) for h in core.handoffs]


# ------------------------------------------------------------- acceptance


def test_scripted_attempt_mode_sequence(tmp_path):
    log = tmp_path / "run" / "handoffs.jsonl"
    core = CognitiveCore(log_path=log)
    result = core.run_attempt(make_env(), GraspLiftSkill(), seed=0, options=LIGHT)
    assert result.success
    assert sequence(core) == [
        ("DELIBERATIVE", START),
        ("REACTIVE", None),
        ("DELIBERATIVE", PROBE),
        ("REACTIVE", None),
        ("LEARNING", None),
    ]
    assert core.state is CognitiveState.IDLE and core.mode is CognitiveMode.REACTIVE
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["kind"] for line in lines] == [h.kind for h in core.handoffs]
    start, _, probe, _, learned = lines
    assert start["theta"] == DEFAULT_THETA.to_dict() and start["latency_ms"] >= 0.0
    assert probe["t"] > 0.0 and probe["retrieved"] == []
    assert learned["detail"] == {"result": "SUCCESS", "reward": 1.0}
    assert all(line["attempt"] == 1 for line in lines)


class ScriptedSkill(GraspLiftSkill):
    """A few ticks, ``PROBE_DONE``, a few more, ``SLIP_UNRECOVERED``, then a DROP."""

    def _run(self):
        hold = self._target()
        self._enter("PROBE")
        yield from self._hold(hold, 0.01)
        self.probe = ProbeFeatures(1.5, 0.001, 3000.0, 2.0, 2.2, False, None, False)
        self._escalate(PROBE_DONE, self.probe, ran=("grasp_depth", "approach_speed"))
        self._enter("LIFT")
        yield from self._hold(hold, 0.01)
        self._escalate(SLIP_UNRECOVERED, info={"grip": 9.0}, ran=("grasp_depth",))
        yield from self._hold(hold, 0.01)
        self.stop_reason = "DROP"
        return False


def test_slip_escalation_enters_system2():
    env = StepFlag(make_env())
    policy = Recorder(env, grip_at={ESCALATION: 12.0})
    core = CognitiveCore(policy)
    result = core.run_attempt(env, ScriptedSkill(), seed=0, options=LIGHT)
    assert [e.kind for e in result.escalations] == [PROBE_DONE, SLIP_UNRECOVERED]
    assert sequence(core) == [
        ("DELIBERATIVE", START),
        ("REACTIVE", None),
        ("DELIBERATIVE", PROBE),
        ("REACTIVE", None),
        ("DELIBERATIVE", ESCALATION),
        ("REACTIVE", None),
        ("LEARNING", None),
    ]
    assert core.handoffs[4].retrieved == ["ep_0001"]
    assert result.theta_final.grip_force == 12.0  # System 2's θ reached System 1
    assert result.result == "DROP" and core.handoffs[-1].detail["reward"] == -1.0
    assert policy.percepts[2].data["escalation"].info == {"grip": 9.0}


def test_safety_fault_forces_emergency_and_ends_the_attempt(caplog):
    env = make_env(safety=SafetyLimits(max_force=0.3))
    core = CognitiveCore()
    with caplog.at_level(logging.WARNING, logger="vfa.cognition.cognitive_core"):
        result = core.run_attempt(env, GraspLiftSkill(), seed=0)
    assert result.result == "SAFETY_STOP" and not result.success
    kinds = [h.kind for h in core.handoffs]
    assert kinds[:2] == ["DELIBERATIVE", "REACTIVE"]
    assert kinds[-2:] == ["EMERGENCY", "LEARNING"]
    assert kinds.count("EMERGENCY") == 1
    assert core.handoffs[-2].detail["reason"] == "wrist_force"
    assert core.mode is CognitiveMode.EMERGENCY
    assert "[SAFETY] EMERGENCY attempt 1" in caplog.text
    # The next attempt starts in System 2 again, here as a retry after the failure.
    policy = Recorder()
    core.policy = policy
    core.run_attempt(make_env(), GraspLiftSkill(), seed=0, options=LIGHT, previous=result)
    assert core.handoffs[len(kinds)].kind == "DELIBERATIVE"
    assert core.handoffs[len(kinds)].trigger == FAILURE and core.handoffs[-1].attempt == 2
    previous = policy.percepts[0].data["previous"]
    assert isinstance(previous, ObservedAttempt) and previous.result == "SAFETY_STOP"
    assert previous.safety["reason"] == "wrist_force"
    assert core.mode is CognitiveMode.REACTIVE


def test_system2_runs_only_at_triggers_never_inside_a_tick():
    env = StepFlag(make_env())
    policy = Recorder(env)
    core = CognitiveCore(policy)
    first = core.run_attempt(env, GraspLiftSkill(), seed=0, options=LIGHT)
    assert first.success
    second = core.run_attempt(env, GraspLiftSkill(), seed=0, options=LIGHT, previous=first)
    assert second.success
    triggers = [t for t, _, _ in policy.calls]
    assert triggers == [START, PROBE, START, PROBE]  # a success does not make a retry
    assert not any(inside for _, inside, _ in policy.calls)
    # Before the attempt: no tick ran yet; at the probe: ticks ran, the skill waits.
    assert policy.calls[0][2] == 0 and policy.calls[1][2] > 0
    assert len(policy.calls) == sum(1 + len(r.escalations) for r in (first, second))
    # The START percept is the observation only: no privileged info.
    assert "privileged" not in policy.percepts[0].data["obs"]
    assert set(policy.percepts[0].data) == {"obs"}


class Resets(gym.Wrapper):
    """Keeps every reset observation."""

    def __init__(self, env):
        super().__init__(env)
        self.observations = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.observations.append({k: np.array(v) for k, v in obs.items()})
        return obs, info


def test_system2_sees_the_cube_the_attempt_runs_on_and_its_theta_runs():
    env = Resets(make_env())
    policy = Recorder(grip_at={START: 5.0, PROBE: 6.0})
    core = CognitiveCore(policy)
    result = core.run_attempt(env, GraspLiftSkill(), seed=4, options=LIGHT)
    seen, ran = env.observations
    assert all(np.array_equal(seen[k], ran[k]) for k in seen)
    assert all(np.array_equal(policy.percepts[0].data["obs"][k], ran[k]) for k in ran)
    assert result.theta.grip_force == 5.0 and result.theta_final.grip_force == 6.0
    probe = policy.percepts[1].data
    assert isinstance(probe["view"], EpisodeView) and probe["probe"] is result.probe
    # T15 can build key_probe at PROBE_DONE: the same key as from the finished episode.
    emb = Embedder()
    assert np.allclose(
        emb.key_probe(probe["view"]), emb.key_probe(EpisodeView.from_episode(result.episode))
    )


def test_a_failed_attempt_reaches_system2_without_hidden_information():
    env = make_env()
    policy = Recorder()
    core = CognitiveCore(policy)
    broken = core.run_attempt(
        env, GraspLiftSkill(), seed=0, options={"object_class": "fragile_soft"}
    )
    assert broken.result == "BREAK"
    core.run_attempt(
        env, GraspLiftSkill(), seed=0, options={"object_class": "fragile_soft"}, previous=broken
    )
    previous = policy.percepts[2].data["previous"]
    assert policy.calls[2][0] == FAILURE and isinstance(previous, ObservedAttempt)
    assert not hasattr(previous, "episode") and isinstance(previous.view, EpisodeView)
    assert previous.result == "BREAK" and previous.theta_final == broken.theta_final
    strings, numbers = leaves(policy.percepts)
    assert not any(word in text for text in strings for word in ("fragile_soft", "privileged"))
    hidden = broken.episode.meta["hidden"]
    secrets = [hidden["mass"], hidden["break_force"], hidden["friction"][0]]
    assert all(float(v) != round(float(v), 3) for v in secrets)  # no chance matches
    for value in secrets:
        assert not any(np.any(np.asarray(n) == value) for n in numbers), value


def leaves(obj, seen=None) -> tuple[list[str], list]:
    """Every string and number (arrays whole) reachable from ``obj``."""
    seen = set() if seen is None else seen
    strings, numbers = [], []
    if obj is None or id(obj) in seen:
        return strings, numbers
    seen.add(id(obj))
    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in "USO":
        strings += [str(x) for x in obj.ravel()]
    elif isinstance(obj, bool | int | float | np.number | np.ndarray):
        numbers.append(obj)
    elif isinstance(obj, Enum):
        strings.append(str(obj.value))
    elif hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            s, n = leaves(getattr(obj, name), seen)
            strings += [name, *s]
            numbers += n
    elif isinstance(obj, dict):
        for key, value in obj.items():
            s, n = leaves(value, seen)
            strings += [str(key), *s]
            numbers += n
    elif isinstance(obj, list | tuple | set | frozenset):
        for item in obj:
            s, n = leaves(item, seen)
            strings += s
            numbers += n
    else:  # an unknown object could hide anything: the walk must see it all
        raise TypeError(f"the leak walk does not know {type(obj).__name__}")
    return strings, numbers


class ExpiredHold(GraspLiftSkill):
    """Holds a moment, then ends as a hold that expired: run_attempt labels it SLIP or
    DROP from the cube's true height."""

    def _run(self):
        yield from self._hold(self._target(), 0.02)
        self.stop_reason = "HOLD_EXPIRED"
        return False


def test_slip_and_drop_reach_system2_as_one_label():
    env = make_env()
    policy = Recorder()
    core = CognitiveCore(policy)
    expired = core.run_attempt(env, ExpiredHold(), seed=0, options=LIGHT)
    assert expired.result in ("SLIP", "DROP")
    core.run_attempt(env, ExpiredHold(), seed=0, options=LIGHT, previous=expired)
    previous = policy.percepts[1].data["previous"]
    assert previous.result == previous.view.result == UNDERGRIP_RESULT
    as_drop = failure_features(replace(previous.view, result="DROP"))
    assert np.array_equal(failure_features(previous.view), as_drop)  # key_retry unchanged
    assert not any(t in ("SLIP", "DROP") for t in leaves(policy.percepts[1])[0])
    for label in ("SLIP", "DROP"):  # this cube only ever drops: check SLIP directly
        seen = ObservedAttempt.of(replace(expired, result=label))
        assert seen.result == seen.view.result == UNDERGRIP_RESULT
    for label in ("BREAK", "MISS", "TIMEOUT", "SAFETY_STOP"):
        assert ObservedAttempt.of(replace(expired, result=label, episode=None)).result == label


def test_a_bad_env_is_refused_before_anything_is_logged(tmp_path):
    log = tmp_path / "handoffs.jsonl"
    core = CognitiveCore(log_path=log)
    bad = PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=True)
    with pytest.raises(ValueError, match="auto_carry"):
        core.run_attempt(bad, GraspLiftSkill(), seed=0)
    assert core.attempt == 0 and core.handoffs == [] and not log.exists()


def test_fields_that_already_ran_keep_their_values():
    """A policy that changes only fields that already ran at SLIP_UNRECOVERED changes
    nothing: the logged θ is the θ that ran, and the recording equals plain System 1."""

    class ChangesRanFields(Recorder):
        def decide(self, percept, theta):
            decision = super().decide(percept, theta)
            if percept.source == ESCALATION:
                theta = replace(decision.theta, lift_accel=3.0, grasp_depth=0.4)
                return Decision(theta, ESCALATION, "ran fields")
            return decision

    options = {"object_class": "heavy_slippery"}
    policy = ChangesRanFields()
    policy.theta = SLIPPING
    core = CognitiveCore(policy)
    cored = core.run_attempt(make_env(), GraspLiftSkill(), seed=725293, options=options)
    plain = run_attempt(make_env(), GraspLiftSkill(), SLIPPING, seed=725293, options=options)
    escalation = [h for h in core.handoffs if h.trigger == ESCALATION]
    assert len(escalation) == 1 and escalation[0].theta == cored.theta_final.to_dict()
    assert (cored.result, cored.sim_s) == (plain.result, plain.sim_s)
    a, b = plain.episode, cored.episode
    assert all(np.array_equal(a.arrays[k], b.arrays[k], equal_nan=True) for k in a.arrays)


# θ from tests/test_skill.py::test_unrecovered_slip_escalates_once: the slide is
# chaotic, so it keeps full precision.
SLIPPING = SkillParams(
    grip_force=8.873970736532769,
    close_speed=0.29465118035080506,
    lift_accel=0.8300015898098342,
    grasp_depth=0.4996141190118628,
    approach_speed=0.38648231964594904,
)


def test_a_real_slip_escalation_reaches_system2_and_its_rescue_runs():
    rescue = 25.0
    policy = Recorder(grip_at={ESCALATION: rescue})
    policy.theta = SLIPPING
    core = CognitiveCore(policy)
    result = core.run_attempt(
        make_env(), GraspLiftSkill(), seed=725293, options={"object_class": "heavy_slippery"}
    )
    assert [e.kind for e in result.escalations] == [PROBE_DONE, SLIP_UNRECOVERED]
    escalation = [h for h in core.handoffs if h.trigger == ESCALATION]
    assert len(escalation) == 1 and escalation[0].theta["grip_force"] == rescue
    # The fields that already ran keep their values: the logged θ is the θ that ran.
    assert escalation[0].theta == result.theta_final.to_dict()
    assert result.theta_final.lift_accel == SLIPPING.lift_accel
    assert result.success


def test_the_stub_records_the_same_attempt_as_plain_system1():
    """The stub keeps the reflex's grip at SLIP_UNRECOVERED and returns None, so the
    recording equals an attempt without the core (no extra re-arm of the detector)."""
    options = {"object_class": "heavy_slippery"}
    plain = run_attempt(make_env(), GraspLiftSkill(), DEFAULT_THETA, seed=1, options=options)
    core = CognitiveCore()
    cored = core.run_attempt(make_env(), GraspLiftSkill(), seed=1, options=options)
    slip = [e for e in cored.escalations if e.kind == SLIP_UNRECOVERED]
    assert len(slip) == 1 and slip[0].theta.grip_force > DEFAULT_THETA.grip_force
    assert cored.theta_final.grip_force == slip[0].theta.grip_force
    assert (cored.result, cored.sim_s) == (plain.result, plain.sim_s)
    a, b = plain.episode, cored.episode
    assert a.arrays.keys() == b.arrays.keys()
    assert all(np.array_equal(a.arrays[k], b.arrays[k], equal_nan=True) for k in a.arrays)
    assert [(e.name, e.t) for e in a.events] == [(e.name, e.t) for e in b.events]


def test_handoff_lines_are_strict_json(tmp_path):
    log = tmp_path / "handoffs.jsonl"
    core = CognitiveCore(log_path=log)
    core._log(
        Handoff(
            "EMERGENCY",
            1,
            0.1,
            detail={
                "value": float("nan"),
                "limit": np.float64(np.inf),
                "extra": (1, np.int64(2)),
                "scalar": np.array(1.5),
            },
        )
    )

    def reject(name):
        raise ValueError(name)

    line = json.loads(log.read_text(), parse_constant=reject)
    assert line["detail"] == {"value": None, "limit": None, "extra": [1, 2], "scalar": 1.5}


def test_a_seed_is_required():
    with pytest.raises(ValueError, match="seed"):
        CognitiveCore().run_attempt(make_env(), GraspLiftSkill(), seed=None)


# ------------------------------------------------- the crate's cycle (ported)


def percept(source: str = START, confidence: float = 0.9) -> Percept:
    return Percept(source, {"x": 1.0}, confidence)


def test_initial_state():
    core = CognitiveCore()
    assert core.state is CognitiveState.IDLE and core.mode is CognitiveMode.REACTIVE
    assert len(core.percepts) == 0 and core.cumulative_reward == 0.0


def test_perceive_above_and_below_threshold():
    core = CognitiveCore()
    assert core.perceive(percept(confidence=0.9)) is CognitiveState.PERCEIVING
    assert len(core.percepts) == 1
    core.perceive(percept(confidence=0.1))
    assert len(core.percepts) == 1


def test_think_on_an_empty_buffer_is_none():
    core = CognitiveCore()
    assert core.think() is None and core.state is CognitiveState.THINKING


def test_think_then_act_hands_theta_to_system1():
    core = CognitiveCore(DefaultPolicy(SkillParams(grip_force=5.0)))
    core.perceive(percept(START))
    decision = core.think()
    assert core.mode is CognitiveMode.DELIBERATIVE and decision.theta.grip_force == 5.0
    assert core.act(decision).grip_force == 5.0
    assert core.state is CognitiveState.ACTING and core.mode is CognitiveMode.REACTIVE
    core.perceive(percept(PROBE))
    assert core.think().theta.grip_force == 5.0  # the stub keeps the θ in force
    core.perceive(percept("tick"))
    with pytest.raises(ValueError, match="trigger"):
        core.think()


def test_learn_adjusts_threshold_and_reward():
    core = CognitiveCore()
    core.perceive(percept())
    core.learn(Outcome(True, 1.0, "SUCCESS"))
    assert core.attention_threshold == pytest.approx(0.49)
    assert core.cumulative_reward == pytest.approx(0.01)
    assert len(core.percepts) == 0 and core.state is CognitiveState.IDLE
    core.learn(Outcome(False, -1.0, "BREAK"))
    core.learn(Outcome(False, -1.0, "BREAK"))
    assert core.attention_threshold == pytest.approx(0.51)
    assert core.cumulative_reward == pytest.approx(-0.01)
    for _ in range(100):
        core.learn(Outcome(False, -1.0, "BREAK"))
    assert core.attention_threshold == pytest.approx(0.9)


def test_percept_buffer_keeps_the_newest():
    core = CognitiveCore(max_percepts=3)
    for i in range(5):
        core.perceive(Percept(START, {"i": i}, 1.0))
    assert [p.data["i"] for p in core.percepts] == [2, 3, 4]


def test_bad_settings_and_policies():
    with pytest.raises(ValueError, match="max_percepts"):
        CognitiveCore(max_percepts=0)
    with pytest.raises(ValueError, match="attention_threshold"):
        CognitiveCore(attention_threshold=1.5)

    class Broken:
        def decide(self, percept, theta):
            return theta

    core = CognitiveCore(Broken())
    core.perceive(percept())
    with pytest.raises(TypeError, match="Decision"):
        core.think()


def test_outcome_from_a_result():
    class R:
        success, result = False, "BREAK"

    assert Outcome.from_result(R()) == Outcome(False, -1.0, "BREAK")


def test_default_policy():
    policy = DefaultPolicy()
    held = SkillParams(grip_force=13.0)
    assert policy.decide(percept(START), held).theta == DEFAULT_THETA
    assert policy.decide(percept(FAILURE), held).theta == DEFAULT_THETA
    assert policy.decide(percept(PROBE), held).theta == held
    assert policy.decide(percept(ESCALATION), held).theta == held
    assert np.isclose(DEFAULT_THETA.grip_force, 8.0)
