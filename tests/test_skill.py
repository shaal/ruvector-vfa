"""T08: the grasp-and-lift skill with a touch-and-heft probe (``vfa.skills``).

ROADMAP acceptance: with θ hand-picked from ``docs/FEASIBILITY.md`` the skill
lifts each of the five classes in at least 8 of 10 seeds; a deliberately wrong θ
fails with the right failure mode (``fragile_soft`` + high force → ``BREAK``,
``heavy_slippery`` + low force → ``SLIP``/``DROP``); the ``PROBE_DONE`` callback
fires exactly once per attempt and can change θ for the remaining phases.

The 5×10 sweep and the other multi-attempt checks are ``slow``
(``uv run pytest -m slow -q tests/test_skill.py``); the fast suite runs one attempt.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from vfa.sim import objects as OB
from vfa.sim.env import PandaVFAEnv
from vfa.sim.sensors import Event
from vfa.skills import pick
from vfa.skills.params import BOUNDS, FIELDS, HIGH, LOW, SkillParams
from vfa.skills.pick import (
    PROBE_DONE,
    SLIP_UNRECOVERED,
    Escalation,
    GraspLiftSkill,
    run_attempt,
)
from vfa.skills.reflexes import SlipReflex

CONFIG = OB.load_object_config()
CLASSES = ("light_rigid", "heavy_rigid", "slippery", "fragile_soft", "heavy_slippery")
# θ hand-picked from docs/FEASIBILITY.md §3: a grip [N per finger] inside each
# class's T05 window with margin over its hardest cube's need (0.97, 7.4, 12.1,
# 1.73, 16.6 N at the T05 lift's 0.58 m/s² peak) and, for fragile_soft, a summed
# grip below the class's lowest break force (6 N).
HAND_PICKED = {
    "light_rigid": 4.0,
    "heavy_rigid": 16.0,
    "slippery": 16.0,
    "fragile_soft": 2.0,
    "heavy_slippery": 20.0,
}
SEEDS = range(10)


def theta_for(grip: float, **changes: float) -> SkillParams:
    base = {
        "grip_force": grip,
        "close_speed": 0.1,
        "lift_accel": 0.6,
        "grasp_depth": 0.5,
        "approach_speed": 0.4,
    }
    return SkillParams(**{**base, **changes})


def make_env() -> PandaVFAEnv:
    return PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False)


class NoPrivileged(gym.Wrapper):
    """Strips ``info["privileged"]``: the skill must act on observable signals only."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, {k: v for k, v in info.items() if k != "privileged"}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            reward,
            terminated,
            truncated,
            {k: v for k, v in info.items() if k != "privileged"},
        )


def events_of(result: pick.AttemptResult, name: Event) -> list[float]:
    return [e.t for e in result.episode.events if e.name is name]


def phase_at(result: pick.AttemptResult, t: float) -> str:
    for name, t0, t1 in result.episode.phase_spans():
        if t0 <= t <= t1:
            return name
    raise AssertionError(f"no phase at t={t}")


# ------------------------------------------------------------------ params


def test_params_bounds_and_defaults():
    theta = SkillParams()
    assert np.all(LOW <= theta.to_vector()) and np.all(theta.to_vector() <= HIGH)
    assert tuple(BOUNDS) == FIELDS
    for name in FIELDS:
        low, high = BOUNDS[name]
        SkillParams(**{name: low})
        SkillParams(**{name: high})
        with pytest.raises(ValueError):
            SkillParams(**{name: low - 1e-6})
        with pytest.raises(ValueError):
            SkillParams(**{name: high * 1.01})
        with pytest.raises(ValueError):
            SkillParams(**{name: float("nan")})
        with pytest.raises(TypeError):
            SkillParams(**{name: True})
    assert isinstance(SkillParams(grip_force=np.float32(4.0)).grip_force, float)
    assert SkillParams(grip_force=np.int64(4)).grip_force == 4.0
    with pytest.raises(TypeError):
        SkillParams(grip_force=np.True_)


def test_params_vector_and_json_round_trip():
    theta = SkillParams(grip_force=12.5, close_speed=0.2, lift_accel=1.5, grasp_depth=0.4)
    v = theta.to_vector()
    assert v.dtype == np.float64 and v.shape == (len(FIELDS),)
    assert SkillParams.from_vector(v) == theta
    assert SkillParams.from_json(theta.to_json()) == theta
    assert SkillParams.from_dict(theta.to_dict()) == theta
    assert list(theta.to_dict()) == list(FIELDS)
    clipped = SkillParams.from_vector(HIGH + 1.0, clip=True)
    assert np.array_equal(clipped.to_vector(), HIGH)
    with pytest.raises(ValueError):
        SkillParams.from_vector(HIGH + 1.0)
    with pytest.raises(ValueError):
        SkillParams.from_vector(v[:4])
    with pytest.raises(ValueError):
        SkillParams.from_vector(np.full(5, np.nan), clip=True)
    with pytest.raises(ValueError):
        SkillParams.from_dict({**theta.to_dict(), "extra": 1.0})
    with pytest.raises(ValueError):
        SkillParams.from_dict({k: v for k, v in theta.to_dict().items() if k != "grip_force"})
    with pytest.raises(ValueError):
        SkillParams.from_json('{"grip_force": NaN}')


def test_slip_reflex_steps_are_bounded():
    reflex = SlipReflex()
    assert reflex.react(2.0) == pytest.approx(2.5)
    assert reflex.react(2.5) == pytest.approx(3.125)
    assert reflex.react(3.125) is None  # spent: the skill escalates
    big = SlipReflex()
    assert big.react(20.0) == pytest.approx(22.0)  # at most +2 N
    assert big.react(24.0) == pytest.approx(BOUNDS["grip_force"][1])  # capped
    with pytest.raises(ValueError):
        SlipReflex(max_triggers=-1)
    with pytest.raises(ValueError):
        SlipReflex(step_fraction=0.0)


def test_skill_rejects_an_env_it_cannot_drive():
    skill = GraspLiftSkill()
    with pytest.raises(ValueError, match="step_hz"):
        skill.begin(PandaVFAEnv("lift", "osc", auto_carry=False), SkillParams())
    with pytest.raises(ValueError, match="auto_carry"):
        skill.begin(PandaVFAEnv("lift", "osc", step_hz=500), SkillParams())
    with pytest.raises(ValueError, match="lift"):
        skill.begin(PandaVFAEnv("reach", "osc", step_hz=500, auto_carry=False), SkillParams())
    with pytest.raises(TypeError):
        skill.begin(make_env(), {"grip_force": 4.0})


# --------------------------------------------------------------- attempts


def test_probe_done_fires_once_and_its_theta_takes_effect():
    """A θ that would break the cube (20 N) is swapped for 2 N at PROBE_DONE: success.

    The swap also asks for a faster lift, which runs, and for another grasp depth
    and approach speed, which already ran and so keep their values. The env strips
    ``info["privileged"]``, so the skill acts on observable signals.
    """
    env = NoPrivileged(make_env())
    calls: list[Escalation] = []
    swapped = theta_for(2.0, close_speed=0.3, lift_accel=1.5, grasp_depth=0.4, approach_speed=0.35)
    ran = theta_for(2.0, close_speed=0.3, lift_accel=1.5)

    def on_escalation(esc: Escalation) -> SkillParams | None:
        calls.append(esc)
        return swapped if esc.kind == PROBE_DONE else None

    start = theta_for(20.0)
    result = run_attempt(
        env,
        GraspLiftSkill(),
        start,
        on_escalation,
        seed=0,
        options={"object_class": "fragile_soft"},
    )
    assert result.result == "SUCCESS" and result.success and result.failure_mode is None
    assert [e.kind for e in calls] == [PROBE_DONE]
    probe = calls[0]
    assert probe.phase == "PROBE" and probe.theta == start and probe.probe is result.probe
    assert result.theta == start and result.theta_final == ran
    (lift,) = [t1 - t0 for name, t0, t1 in result.episode.phase_spans() if name == "LIFT"]
    assert lift == pytest.approx(pick.lift_duration(1.5), abs=0.01)
    assert env.unwrapped.close_speed == 0.3  # the squeeze ran at θ's closing speed
    # The probe read the cube: a light cube rose with the 1 cm heft, a soft one sank.
    features = result.probe
    assert features.force == pick.PROBE_FORCE
    assert not features.heft_slip and 0.3 < features.heft_load < 2.0
    assert features.indentation > 3e-4 and features.closure_slope > 0
    ep = result.episode
    assert [h["params"] for h in ep.params_history] == [start.to_dict(), ran.to_dict()]
    assert ep.params == ran.to_dict()
    assert ep.meta["probe"]["indentation"] == pytest.approx(features.indentation)
    assert ep.phases == ("SETTLE", *pick.PHASES)  # SETTLE: the env reset
    assert not events_of(result, Event.BREAK)
    assert 0.0 < result.sim_s < 10.0
    with pytest.raises(RuntimeError):
        env.unwrapped.end_episode("SUCCESS")  # the episode is over


# -------------------------------------------------------------------- slow


@pytest.mark.slow
def test_hand_picked_theta_lifts_every_class_in_8_of_10_seeds():
    env = make_env()
    skill = GraspLiftSkill()
    table = {}
    for cls in CLASSES:
        theta = theta_for(HAND_PICKED[cls])
        results = []
        for seed in SEEDS:
            probes: list[Escalation] = []
            r = run_attempt(
                env,
                skill,
                theta,
                lambda e, probes=probes: probes.append(e) if e.kind == PROBE_DONE else None,
                seed=seed,
                options={"object_class": cls},
            )
            assert len(probes) == 1, (cls, seed, r.result)  # exactly once per attempt
            assert sum(e.kind == PROBE_DONE for e in r.escalations) == 1
            results.append(r)
        table[cls] = [r.result for r in results]
        wins = sum(r.success for r in results)
        print(f"{cls:15s} {wins}/10  {table[cls]}  sim {max(r.sim_s for r in results):.2f} s max")
        assert wins >= 8, (cls, table[cls])


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
def test_fragile_cube_with_high_force_breaks_in_the_squeeze(seed):
    r = run_attempt(
        make_env(),
        GraspLiftSkill(),
        theta_for(12.0),
        seed=seed,
        options={"object_class": "fragile_soft"},
    )
    assert r.result == "BREAK", r.result
    (t_break,) = events_of(r, Event.BREAK)
    assert phase_at(r, t_break) == "SQUEEZE"  # the probe itself is safe
    assert [e.kind for e in r.escalations] == [PROBE_DONE]


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
def test_heavy_slippery_cube_with_low_force_slips_or_drops(seed):
    escalations: list[Escalation] = []
    r = run_attempt(
        make_env(),
        GraspLiftSkill(),
        theta_for(2.0),
        escalations.append,
        seed=seed,
        options={"object_class": "heavy_slippery"},
    )
    assert r.result in ("SLIP", "DROP"), r.result
    assert not events_of(r, Event.BREAK)
    assert [e.kind for e in escalations][0] == PROBE_DONE
    assert sum(e.kind == PROBE_DONE for e in escalations) == 1
    print(seed, r.result, [e.kind for e in escalations], r.sim_s)


@pytest.mark.slow
def test_probe_done_theta_can_also_break_the_cube():
    """The reverse swap: 2 N, then 20 N from PROBE_DONE on, breaks a fragile cube."""
    r = run_attempt(
        make_env(),
        GraspLiftSkill(),
        theta_for(2.0),
        lambda e: theta_for(20.0) if e.kind == PROBE_DONE else None,
        seed=1,
        options={"object_class": "fragile_soft"},
    )
    assert r.result == "BREAK"
    assert phase_at(r, events_of(r, Event.BREAK)[0]) == "SQUEEZE"


@pytest.mark.slow
@pytest.mark.parametrize("y", [-0.15, 0.15])
def test_slowest_theta_corner_fits_the_time_limit(y):
    """Every speed and acceleration at its lower bound, the largest grip, the deepest
    grasp, on the farthest, smallest, heaviest cube: the whole hold fits the 10 s."""
    slow = SkillParams(
        grip_force=BOUNDS["grip_force"][1],
        close_speed=BOUNDS["close_speed"][0],
        lift_accel=BOUNDS["lift_accel"][0],
        grasp_depth=BOUNDS["grasp_depth"][1],
        approach_speed=BOUNDS["approach_speed"][0],
    )
    spec = OB.sample_object(CONFIG.classes["heavy_rigid"], np.random.default_rng(0))
    spec = dataclasses.replace(spec, size=0.04, mass=1.0, x=0.45, y=y, yaw=0.785)
    env = make_env()
    r = run_attempt(env, GraspLiftSkill(), slow, seed=0, options={"object": spec})
    assert r.result == "SUCCESS", r.result
    t0 = float(r.episode.arrays["t"][-1]) - r.sim_s  # the reset's end
    (hold,) = [s0 - t0 for name, s0, _ in r.episode.phase_spans() if name == "HOLD"]
    assert hold + pick.HOLD_S < env.time_limit - 0.05, hold
    print(f"y={y}: success at {r.sim_s:.3f} s, HOLD from {hold:.3f} s")


def worst_fragile(seed: int, **changes: float) -> OB.ObjectSpec:
    """The probe's worst case: a small (4 cm), heavy (0.2 kg), low-friction (μ 0.6),
    softest (time constant 0.08) fragile cube that breaks at 6 N."""
    spec = OB.sample_object(CONFIG.classes["fragile_soft"], np.random.default_rng(10_000 + seed))
    base = {
        "size": 0.04,
        "mass": 0.2,
        "friction": (0.6, *spec.friction[1:]),
        "solref": (0.08, spec.solref[1]),
        "break_force": 6.0,
    }
    return dataclasses.replace(spec, **{**base, **changes})


@pytest.mark.slow
def test_probe_spares_the_weakest_fragile_cube_at_the_shallowest_grasp():
    """It slides in the probe grip; at the shallowest grasp the fingers must not catch
    its top edge in the set-down (at 0.35 three of 40 such cubes peaked at 6–7 N)."""
    theta = theta_for(2.0, grasp_depth=BOUNDS["grasp_depth"][0])
    env = make_env()
    for seed in range(10):
        r = run_attempt(
            env, GraspLiftSkill(), theta, seed=seed, options={"object": worst_fragile(seed)}
        )
        ep = r.episode
        probe = ep.arrays["phase"] == ep.phases.index("PROBE")
        peak = float(np.max(np.sum(ep.arrays["cube_finger_force"][probe], axis=1)))
        assert peak < 4.5, (seed, peak, r.result)
    # A natural cube whose probe broke at grasp depth 0.30 (now out of bounds).
    r = run_attempt(
        env, GraspLiftSkill(), theta, seed=1568, options={"object_class": "fragile_soft"}
    )
    assert r.result == "SUCCESS", r.result


@pytest.mark.slow
def test_a_missed_cube_ends_the_attempt_before_probe_done():
    """The cube moves after the camera snapshot: the probe closes on nothing."""

    class MoveCube(gym.Wrapper):
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            u = self.env.unwrapped
            adr = u.model.jnt_qposadr[u.model.body_jntadr[u.ids.cube_body]]
            u.data.qpos[adr + 1] += 0.12
            mujoco.mj_forward(u.model, u.data)
            return obs, info

    escalations: list[Escalation] = []
    r = run_attempt(
        MoveCube(make_env()),
        GraspLiftSkill(),
        theta_for(4.0),
        escalations.append,
        seed=0,
        options={"object_class": "light_rigid"},
    )
    assert r.result == "MISS" and r.probe is None and not escalations
    assert r.episode.result == "MISS"


@pytest.mark.slow
def test_unrecovered_slip_escalates_once():
    """A grip under the need slips in the lift: two reflex raises (recorded as θ),
    then one escalation, whose θ (25 N) sets the grip and the closing speed and
    saves the cube.

    θ is a uniform random draw, kept to full precision: the slide is chaotic, so
    rounded values can take another path.
    """
    escalations: list[Escalation] = []
    start = SkillParams(
        grip_force=8.873970736532769,
        close_speed=0.29465118035080506,
        lift_accel=0.8300015898098342,
        grasp_depth=0.4996141190118628,
        approach_speed=0.38648231964594904,
    )
    rescue = theta_for(25.0, close_speed=0.5, lift_accel=3.0)

    def on_escalation(esc: Escalation) -> SkillParams | None:
        escalations.append(esc)
        return rescue if esc.kind == SLIP_UNRECOVERED else None

    r = run_attempt(
        make_env(),
        GraspLiftSkill(),
        start,
        on_escalation,
        seed=725293,
        options={"object_class": "heavy_slippery"},
    )
    assert [e.kind for e in escalations] == [PROBE_DONE, SLIP_UNRECOVERED]
    unrecovered = escalations[1]
    assert unrecovered.phase in ("LIFT", "HOLD")
    reflex = SlipReflex()
    first = reflex.react(start.grip_force)
    second = reflex.react(first)
    assert unrecovered.info["grip"] == pytest.approx(second)
    assert unrecovered.theta.grip_force == pytest.approx(second)
    grips = [h["params"]["grip_force"] for h in r.episode.params_history]
    assert grips == pytest.approx([start.grip_force, first, second, 25.0])
    # The grasp, the approach, and the lift had already run: they keep their values.
    assert r.theta_final == dataclasses.replace(start, grip_force=25.0, close_speed=0.5)
    assert r.result == "SUCCESS", r.result


def test_label_of_an_attempt_the_skill_ended():
    """Ground truth labels a skill-ended attempt: SLIP with the cube 2 cm up, else DROP."""
    env = make_env()
    env.reset(seed=0, options={"object_class": "light_rigid"})
    assert pick._label(env, "MISS") == "MISS"
    assert pick._label(env, "DROP") == "DROP"
    assert pick._label(env, "HOLD_EXPIRED") == "DROP"  # the cube rests on the table
    adr = env.model.jnt_qposadr[env.model.body_jntadr[env.ids.cube_body]]
    env.data.qpos[adr + 2] += pick.SLIP_RISE_M + 0.001
    mujoco.mj_forward(env.model, env.data)
    assert pick._label(env, "HOLD_EXPIRED") == "SLIP"
