"""T09: the safety supervisor and emergency stop (``vfa.control.safety``).

ROADMAP acceptance: fault injection (driving the ee into the table, commanding a
target outside the workspace, a large external force) triggers a stop within 50
ms of sim time with the right reason; normal T08 attempts never trigger it
(``tests/test_skill.py`` asserts it over the T08 seeds, which the Verify command
runs with this file). Most unit tests feed each check a synthetic frame.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from vfa.control.osc import WORKSPACE_HIGH, WORKSPACE_LOW, control_step
from vfa.control.safety import REASONS, SafetyLimits, SafetySupervisor
from vfa.sim import env as E
from vfa.sim.sensors import Event, SensorFrame
from vfa.skills.params import SkillParams
from vfa.skills.pick import GraspLiftSkill, run_attempt

LATENCY_S = 0.05  # ROADMAP T09
HOLD_CHECK_S = 0.5


# ------------------------------------------------------------------ helpers


def frame(**changes) -> SensorFrame:
    """A calm frame: the ee at rest inside the workspace, no wrist load."""
    base = dict(
        t=1.0,
        wrench_raw=np.zeros(6),
        wrench=np.zeros(6),
        acc=np.array([0.0, 0.0, 9.81]),
        finger_force=np.zeros(2),
        contact=np.zeros(2, dtype=bool),
        width=0.08,
        width_rate=0.0,
        ee_pos=np.array([0.5, 0.0, 0.3]),
        ee_quat=np.array([0.0, 1.0, 0.0, 0.0]),
        ee_vel=np.zeros(6),
        load=0.0,
        peak_load=0.0,
        armed=False,
        watching=False,
        watch_id=0,
        slip=False,
        slip_time=None,
    )
    return SensorFrame(**{**base, **changes})


@pytest.fixture(scope="module")
def env_osc() -> E.PandaVFAEnv:
    return E.PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False)


def supervisor(env: E.PandaVFAEnv, **limits) -> SafetySupervisor:
    env.reset(seed=0)
    return SafetySupervisor(env.model, env.data, env.ids, SafetyLimits(**limits), env.dt)


def stop_event(info: dict) -> dict | None:
    for e in info["events"]:
        if e["name"] == Event.SAFETY_STOP.value:
            return e
    return None


def finger_table_contact(env: E.PandaVFAEnv) -> bool:
    ids, data = env.ids, env.data
    fingers = ids.left_finger_geoms | ids.right_finger_geoms
    for i in range(data.ncon):
        g1, g2 = int(data.contact.geom1[i]), int(data.contact.geom2[i])
        if (g1 in fingers and g2 == ids.table_geom) or (g2 in fingers and g1 == ids.table_geom):
            return True
    return False


def assert_holds(env: E.PandaVFAEnv) -> None:
    """The emergency hold: the OSC target is the stop pose, and it stays near it."""
    held = np.array(env.osc.target.pos)
    assert np.allclose(env.osc.reference().vel, 0.0)
    path = []
    for _ in range(round(HOLD_CHECK_S / env.dt)):
        control_step(env.model, env.data, (env.osc, env.gripper), 1)
        path.append(env.data.site_xpos[env.ids.ee_site].copy())
    drift = np.linalg.norm(np.array(path) - held, axis=1)
    assert drift.max() < 0.02, drift.max()  # T03: ≈ 2 cm per m/s of overshoot
    assert drift[-1] < 0.005, drift[-1]


class TorqueDriver:
    """Torque control at 500 Hz with the env's own OSC (as T07's torque tests do)."""

    def __init__(self, env: E.PandaVFAEnv, obs: dict) -> None:
        self.env = env
        self.target = np.array(obs["ee_pos"], dtype=float)
        self.info: dict = {}
        self.done = False

    def tick(self, vel: np.ndarray) -> None:
        env = self.env
        mujoco.mj_forward(env.model, env.data)
        env.osc.set_target(self.target, yaw=env.grasp_yaw, vel=(*vel, 0.0, 0.0, 0.0))
        env.osc.advance_reference()
        tau = env.osc.compute()
        grip = env.gripper.apply()
        a = np.clip(np.r_[tau / env._arm_limits, grip / env._grip_limit], -1.0, 1.0)
        _, _, terminated, truncated, self.info = env.step(a)
        self.done = terminated or truncated

    def move(self, goal, speed: float, until=None) -> None:
        goal = np.asarray(goal, dtype=float)
        while not self.done:
            gap = goal - self.target
            dist = float(np.linalg.norm(gap))
            if dist < 1e-9:
                break
            step = min(dist, speed * self.env.dt)
            self.target = self.target + gap / dist * step
            self.tick(gap / dist * speed if dist > step else np.zeros(3))
            if until is not None and until():
                return

    def wait(self, seconds: float) -> None:
        for _ in range(round(seconds / self.env.dt)):
            if self.done:
                return
            self.tick(np.zeros(3))


# --------------------------------------------------------------- unit checks


@pytest.mark.parametrize(
    "changes, target, reason",
    [
        ({"wrench": np.array([0.0, 31.0, 0.0, 0.0, 0.0, 0.0])}, None, "wrist_force"),
        ({"wrench": np.array([0.0, 0.0, 0.0, 0.0, 4.1, 0.0])}, None, "wrist_torque"),
        ({"ee_pos": np.array([0.5, 0.0, 0.49])}, None, "workspace"),
        ({}, np.array([0.5, 0.31, 0.3]), "workspace"),
        ({"ee_vel": np.array([0.0, 1.1, 0.0, 0.0, 0.0, 0.0])}, None, "ee_speed"),
        ({"ee_vel": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.1])}, None, "ee_speed"),
    ],
)
def test_each_limit_trips_on_a_synthetic_frame(env_osc, changes, target, reason):
    sup = supervisor(env_osc)
    assert sup.check(frame(), np.array([0.5, 0.0, 0.3])) is None
    v = sup.check(frame(t=1.002, **changes), target)
    assert v is not None and v.reason == reason and v.t == pytest.approx(1.002)
    assert v.reason in REASONS
    assert sup.check(frame(t=1.004, **changes), target) is None  # latched: one stop


def test_box_margin_spares_a_lagging_ee_but_not_its_target(env_osc):
    sup = supervisor(env_osc)
    edge = np.array(WORKSPACE_HIGH)
    assert sup.check(frame(ee_pos=edge + 0.029), edge) is None  # ee within the 3 cm margin
    v = sup.check(frame(t=1.002), edge + np.array([0.001, 0.0, 0.0]))
    assert v.reason == "workspace" and v.detail == "target"
    assert v.value == pytest.approx(0.001)
    sup = supervisor(env_osc)
    assert sup.check(frame(ee_pos=np.array(WORKSPACE_LOW) - 0.031)).detail == "ee"
    sup = supervisor(env_osc)
    assert sup.check(frame(ee_pos=np.array([np.nan, 0.0, 0.3]))).reason == "workspace"


def test_joint_margin_and_sustained_saturation(env_osc):
    sup = supervisor(env_osc)
    adr = env_osc.ids.arm_qpos[3]
    low, _ = env_osc.model.joint("joint4").range
    env_osc.data.qpos[adr] = low + 0.019
    v = sup.check(frame())
    assert v.reason == "joint_limit" and v.detail == "joint4"
    assert v.value == pytest.approx(0.019)

    sup = supervisor(env_osc)  # the reset put the joints back
    ticks = round(sup.limits.saturation_s / env_osc.dt)
    for k in range(ticks - 1):
        assert sup.check(frame(t=1.0 + k * env_osc.dt), saturated=True) is None
    assert sup.check(frame(t=2.0), saturated=False) is None  # a gap restarts the count
    for k in range(ticks - 1):
        assert sup.check(frame(t=3.0 + k * env_osc.dt), saturated=True) is None
    v = sup.check(frame(t=4.0), saturated=True)
    assert v.reason == "torque_saturation" and v.value == pytest.approx(sup.limits.saturation_s)


def test_nan_wrench_before_the_tare_is_skipped(env_osc):
    sup = supervisor(env_osc)
    assert sup.check(frame(wrench=np.full(6, np.nan))) is None


def test_limits_reject_bad_values():
    with pytest.raises(ValueError):
        SafetyLimits(max_force=0.0)
    with pytest.raises(ValueError):
        SafetyLimits(joint_margin=-0.1)
    with pytest.raises(ValueError):
        SafetyLimits(workspace_low=(0.5, 0.0, 0.0), workspace_high=(0.4, 1.0, 1.0))
    with pytest.raises(TypeError):
        E.PandaVFAEnv("lift", safety={"max_force": 10.0})


# ------------------------------------------------------- fault injection (env)


@pytest.mark.parametrize("speed", [0.1, 0.3, 0.5])
def test_driving_the_ee_into_the_table_stops_on_wrist_force(speed):
    env = E.PandaVFAEnv("lift", "torque", step_hz=500)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube = obs["cube_pos"]
    spot = np.array([0.36, -0.2 if cube[1] > 0 else 0.2])  # bare table, inside the box
    drive = TorqueDriver(env, obs)
    drive.move((*spot, 0.06), 0.3)
    drive.wait(0.3)
    assert not drive.done and not finger_table_contact(env)
    touched: list[float] = []

    def contact() -> bool:
        if not touched and finger_table_contact(env):
            touched.append(float(env.data.time))
        return False

    drive.move((*spot, -0.05), speed, until=lambda: contact() or drive.done)
    assert drive.done and env.result == "SAFETY_STOP"
    stop = stop_event(drive.info)
    assert stop["info"]["reason"] == "wrist_force", stop
    assert touched, "the fingertips never reached the table"
    latency = stop["t"] - touched[0]
    assert 0.0 <= latency <= LATENCY_S, latency
    assert env.emergency.reason == "wrist_force"
    arm, limits = env.ids.arm_actuators, env._arm_limits  # one gravity compensation write
    gravity = np.clip(env.data.qfrc_bias[env.ids.arm_dof], -limits, limits)
    assert np.array_equal(env.data.ctrl[arm], gravity)
    assert env.episode.result == "SAFETY_STOP"
    assert [e.name for e in env.episode.events_named(Event.SAFETY_STOP)] == [Event.SAFETY_STOP]
    assert_holds(env)


def test_a_target_outside_the_workspace_stops_at_once(monkeypatch):
    """A broken command limit (the env's clip box widened) lets the target leave the box."""
    env = E.PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False)
    env.reset(seed=0)
    monkeypatch.setattr(E, "TARGET_HIGH", np.array([0.9, 0.3, 0.45]))
    for _ in range(round(1.0 / env.dt)):
        before = env.ee_target.copy()  # the commanded target so far
        _, _, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0, -1.0]))
        if terminated:
            break
    assert terminated and env.result == "SAFETY_STOP"
    stop = stop_event(info)
    assert stop["info"]["reason"] == "workspace" and stop["info"]["detail"] == "target"
    # The first step whose target left the box stopped (one 2 ms tick at 500 Hz).
    assert before[0] <= WORKSPACE_HIGH[0] < stop["info"]["target"][0]
    assert stop["t"] == pytest.approx(info["sim_time"])
    assert np.all(np.array(env.osc.target.pos) <= WORKSPACE_HIGH)  # the hold is inside
    assert_holds(env)


@pytest.mark.parametrize("axis", range(6))
def test_a_large_external_force_on_the_hand_stops_on_wrist_force(axis):
    """100 N on the hand along ±x, ±y, ±z: the wrist F/T sees it (forces on the arm
    links above the wrist are a known limit, refer to ``vfa.control.safety``)."""
    push = np.zeros(3)
    push[axis % 3] = 100.0 if axis < 3 else -100.0
    env = E.PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False)
    env.reset(seed=0)
    hold = np.array([0.0, 0.0, 0.0, -1.0])
    for _ in range(50):
        env.step(hold)
    pushed_at = float(env.data.time)
    env.data.xfrc_applied[env.ids.hand_body, :3] = push
    for _ in range(round(1.0 / env.dt)):
        obs, _, terminated, _, info = env.step(hold)
        if terminated:
            break
    assert terminated and env.result == "SAFETY_STOP"
    assert env.observation_space.contains(obs)  # the stop step's observation is valid
    stop = stop_event(info)
    assert stop["info"]["reason"] == "wrist_force", stop
    assert 0.0 <= stop["t"] - pushed_at <= LATENCY_S, stop["t"] - pushed_at
    env.data.xfrc_applied[:] = 0.0  # the push ends: the hold must keep the pose
    assert_holds(env)


def test_run_attempt_reports_the_stop_and_its_reason():
    """A skill attempt that trips the supervisor (an absurdly low force limit) ends
    with SAFETY_STOP, and the result carries the reason."""
    env = E.PandaVFAEnv(
        "lift", "osc", step_hz=500, auto_carry=False, safety=SafetyLimits(max_force=0.3)
    )
    r = run_attempt(env, GraspLiftSkill(), SkillParams(), seed=0)
    assert r.result == "SAFETY_STOP" and not r.success
    assert r.safety["reason"] == "wrist_force" and r.safety["limit"] == 0.3
    assert r.episode.result == "SAFETY_STOP"


def test_a_stop_reported_between_steps_engages_the_hold_and_the_penalty():
    """A SAFETY_STOP from outside the env (T14) ends the episode like an own stop."""
    env = E.PandaVFAEnv("reach", "osc", step_hz=500)
    env.reset(seed=0)
    for _ in range(100):
        env.step(np.array([0.5, 0.0, 0.0, -1.0]))  # moving: the hold must take over
    # Malformed values must not lose the stop; extra keys travel with it.
    env.events.report(
        Event.SAFETY_STOP, float(env.data.time), reason="operator", value=None, source="T14"
    )
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0, -1.0]))
    assert terminated and not truncated and env.result == "SAFETY_STOP"
    assert env.emergency.reason == "operator" and np.isnan(env.emergency.value)
    assert env.emergency.info()["source"] == "T14"
    assert env.observation_space.contains(obs)  # ee_target: the last command, in its box
    # No tick ran: the stop's cost only (the penalty plus the distance for every step left).
    dist = float(np.linalg.norm(obs["ee_pos"] - obs["goal"]))
    left = (env.time_limit - env.elapsed) * env.step_hz
    assert reward == pytest.approx(-E.SAFETY_PENALTY - dist * left)
    assert_holds(env)


def test_an_own_stop_costs_the_safety_penalty(monkeypatch):
    env = E.PandaVFAEnv("reach", "osc", step_hz=500)
    obs, _ = env.reset(seed=0)
    monkeypatch.setattr(E, "TARGET_HIGH", np.array([0.9, 0.3, 0.45]))
    for _ in range(round(1.0 / env.dt)):
        obs, reward, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0, -1.0]))
        if terminated:
            break
    assert env.result == "SAFETY_STOP"
    dist = float(np.linalg.norm(obs["ee_pos"] - obs["goal"]))
    left = (env.time_limit - env.elapsed) * env.step_hz
    assert reward == pytest.approx(-dist - E.SAFETY_PENALTY - dist * left)
    # Holding still until the time limit would have scored -dist per step: the stop
    # always scores less (ARCHITECTURE §8).
    assert reward < -dist * (1.0 + left)


def test_the_target_box_starts_at_x_033():
    """The env's target and reach boxes start at x = 0.33 (refer to ``TARGET_LOW``)."""
    assert E.TARGET_LOW[0] == 0.33 and E.REACH_LOW[0] == E.TARGET_LOW[0]
    assert np.all(E.TARGET_LOW >= WORKSPACE_LOW) and np.all(E.TARGET_HIGH <= WORKSPACE_HIGH)


def test_a_report_with_info_json_cannot_hold_fails_at_the_reporter():
    """Nothing half delivered: the bad report raises, and the episode runs on."""
    env = E.PandaVFAEnv("reach", "osc", step_hz=500)
    env.reset(seed=0)
    with pytest.raises(TypeError, match="JSON-ready"):
        env.events.report(Event.SAFETY_STOP, float(env.data.time), source=object())
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert not terminated and env.emergency is None
    env.events.report(Event.SAFETY_STOP, float(env.data.time), reason="operator")
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert terminated and env.result == "SAFETY_STOP" and stop_event(info) is not None


def test_odd_reports_never_lose_a_stop():
    """An overflowing value, an info changed after the report, and a stop reported
    just before the skill ends the episode itself."""
    env = E.PandaVFAEnv("reach", "osc", step_hz=500)
    env.reset(seed=0)
    env.events.report(Event.SAFETY_STOP, float(env.data.time), reason="operator", value=10**400)
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert terminated and env.result == "SAFETY_STOP" and np.isnan(env.emergency.value)

    env.reset(seed=0)
    detail = {"notes": []}
    env.events.report(Event.SAFETY_STOP, float(env.data.time), reason="operator", extra=detail)
    detail["notes"].append(object())  # changed after the report: the event kept a copy
    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    assert terminated and env.result == "SAFETY_STOP"

    env.reset(seed=0)
    env.step(np.array([0.0, 0.0, 0.0, -1.0]))
    env.events.report(Event.SAFETY_STOP, float(env.data.time), reason="operator")
    env.end_episode("MISS")
    assert env.result == "SAFETY_STOP" and env.emergency.reason == "operator"
    assert env.episode.result == "SAFETY_STOP"
    assert env.episode.events_named(Event.SAFETY_STOP)


def test_run_attempt_reports_a_stop_requested_as_the_skill_ends():
    """System 2 (T14) requests a stop in the tick the skill ends the attempt itself."""

    class StopAtTheEnd(GraspLiftSkill):
        def act(self, obs, info):
            action = super().act(obs, info)
            if action is None:
                env = self._env
                env.events.report(Event.SAFETY_STOP, float(env.data.time), reason="operator")
            return action

    env = E.PandaVFAEnv("lift", "osc", step_hz=500, auto_carry=False)
    theta = SkillParams(grip_force=2.0)  # heavy_slippery at 2 N: the skill ends on DROP
    r = run_attempt(env, StopAtTheEnd(), theta, seed=0, options={"object_class": "heavy_slippery"})
    assert r.result == r.episode.result == env.result == "SAFETY_STOP"
    assert r.safety["reason"] == "operator" and not r.success


def test_safety_off_never_stops():
    env = E.PandaVFAEnv("lift", "torque", step_hz=50, safety=None)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        _, _, terminated, _, _ = env.step(rng.uniform(-1.0, 1.0, 8))
        assert not terminated or env.result != "SAFETY_STOP"
    assert env.supervisor is None and env.emergency is None
