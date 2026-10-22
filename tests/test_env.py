"""T07: the Gymnasium environment (``vfa.sim.env``).

ROADMAP acceptance: ``check_env`` passes for all three ids; the same seed gives an
identical observation sequence for 50 random-action steps; a scripted OSC policy
solves ``PandaReach-v0`` in at least 9 of 10 seeds. The other tests pin the task
outcomes of ARCHITECTURE §3, the carry logic, the recorder, and the input checks.
"""

from __future__ import annotations

import math
import warnings

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vfa.sim import env as E
from vfa.sim import objects as OB
from vfa.sim.recorder import load_episode
from vfa.sim.sensors import Event

CONFIG = OB.load_object_config()
CONTROLS = ("osc", "torque")
# Hand-picked grips [N per finger] inside each class's T05 window (FEASIBILITY §3).
GRIP = {
    "light_rigid": 4.0,
    "heavy_rigid": 16.0,
    "slippery": 16.0,
    "fragile_soft": 1.5,
    "heavy_slippery": 20.0,
}
SETTLE_TAU = 0.1  # s: the scripted target's time constant near a waypoint
SCRIPT_ACCEL = 1.0  # m/s²: the scripted target's acceleration limit
# Gymnasium's advice for the unbounded physical signals (refer to _obs_spaces).
_INF_WARNINGS = ("minimum value is -infinity", "maximum value is infinity")


# ------------------------------------------------------------- scripted policy


class Script:
    """A scripted OSC policy: waypoints for the ee target, a grip per segment."""

    def __init__(self, env: gym.Env, obs: dict) -> None:
        self.env, self.obs = env, obs
        self.u: E.PandaVFAEnv = env.unwrapped
        self.done = False
        self.infos: list[dict] = []
        self.rewards: list[float] = []

    def _step(self, a: np.ndarray) -> None:
        self.obs, reward, terminated, truncated, info = self.env.step(a.astype(np.float32))
        self.rewards.append(reward)
        self.infos.append(info)
        self.done = terminated or truncated

    def move(self, goal, speed: float, grip: float, max_s: float = 5.0) -> Script:
        """Move the target to ``goal`` at up to ``speed`` [m/s] while gripping with ``grip``.

        The target speeds up at ``SCRIPT_ACCEL`` and slows down with the time constant
        ``SETTLE_TAU`` near the goal: an abrupt start or stop with a cube in the grip
        dips the held-load estimate enough to fire a SLIP (measured: 20-33 % dips at
        0.15-0.2 m/s steps).
        """
        goal = np.asarray(goal, dtype=float)
        dv = SCRIPT_ACCEL / self.u.step_hz
        vel = np.zeros(3)
        t0 = self.u.elapsed
        while not self.done and self.u.elapsed - t0 < max_s:
            gap = goal - self.obs["ee_target"]
            if np.linalg.norm(gap) < 1e-4:
                break
            want = gap / SETTLE_TAU
            norm = np.linalg.norm(want)
            if norm > speed:
                want *= speed / norm
            vel += np.clip(want - vel, -dv, dv)
            self._step(np.array([*(vel / E.MAX_EE_SPEED), grip]))
        return self

    def wait(self, seconds: float, grip: float) -> Script:
        t0 = self.u.elapsed
        while not self.done and self.u.elapsed - t0 < seconds - 1e-9:
            self._step(np.array([0.0, 0.0, 0.0, grip]))
        return self

    def grasp(self, grip_n: float) -> Script:
        """Approach the observed cube, descend, and squeeze with ``grip_n`` per finger."""
        cube = self.obs["cube_pos"].copy()
        self.move(cube + (0.0, 0.0, 0.12), 0.4, -1.0).wait(0.5, -1.0)
        self.move(cube, 0.1, -1.0).wait(0.3, -1.0)
        ramp = grip_n / (250.0 * self.u.close_speed)  # Gripper.squeeze ramps 250·v N/s
        return self.wait(0.7 + ramp, grip_n / E.GRIP_FORCE_MAX)


class TorqueScript:
    """Drives ``control="torque"`` at 500 Hz with the env's own OSC and gripper laws."""

    def __init__(self, env: E.PandaVFAEnv, obs: dict) -> None:
        self.env, self.obs = env, obs
        self.target = obs["ee_pos"].copy()
        self.vel = np.zeros(6)
        self.infos: list[dict] = []
        self.observations: list[dict] = [obs]
        self.done = False

    def tick(self) -> None:
        env = self.env
        mujoco.mj_forward(env.model, env.data)
        env.osc.set_target(self.target, yaw=env.grasp_yaw, vel=self.vel)
        env.osc.advance_reference()
        tau = env.osc.compute()
        grip = env.gripper.apply()
        a = np.clip(np.r_[tau / env._arm_limits, grip / env._grip_limit], -1.0, 1.0)
        self.obs, _, terminated, truncated, info = env.step(a.astype(np.float32))
        self.observations.append(self.obs)
        self.infos.append(info)
        self.done = terminated or truncated

    def grip(self, force_n: float | None) -> TorqueScript:
        if force_n is None:
            self.env.gripper.open()
        else:
            self.env.gripper.squeeze(force_n, speed=0.1)
        return self

    def move(self, goal, speed: float) -> TorqueScript:
        goal = np.asarray(goal, dtype=float)
        while not self.done:
            gap = goal - self.target
            dist = float(np.linalg.norm(gap))
            if dist < 1e-6:
                break
            step = min(dist, speed / 500.0)
            self.target = self.target + gap / dist * step
            self.vel = np.r_[gap / dist * speed if dist > step else np.zeros(3), np.zeros(3)]
            self.tick()
        self.vel = np.zeros(6)
        return self

    def wait(self, seconds: float) -> TorqueScript:
        for _ in range(round(seconds * 500)):
            if self.done:
                break
            self.tick()
        return self


def lift(env: gym.Env, seed: int, cls: str, grip_n: float, open_after: float | None = None):
    obs, _ = env.reset(seed=seed, options={"object_class": cls})
    s = Script(env, obs).grasp(grip_n)
    g = grip_n / E.GRIP_FORCE_MAX
    s.move(obs["cube_pos"] + (0.0, 0.0, 0.15), 0.15, g)
    if open_after is not None:
        s.wait(open_after, g).wait(2.0, -1.0)
    return s.wait(3.0, g)


def pick_place(env: gym.Env, seed: int, cls: str, grip_n: float):
    obs, _ = env.reset(seed=seed, options={"object_class": cls})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    g = grip_n / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(grip_n)
    s.move(cube + (0.0, 0.0, 0.15), 0.15, g)
    s.move((goal[0], goal[1], cube[2] + 0.15), 0.2, g)
    s.move((goal[0], goal[1], cube[2] + 0.002), 0.05, g).wait(0.3, g)
    s.wait(0.5, -1.0).move((goal[0], goal[1], 0.2), 0.1, -1.0)
    return s.wait(3.0, -1.0)


def reach(env: gym.Env, seed: int) -> Script:
    obs, _ = env.reset(seed=seed)
    s = Script(env, obs)
    while not s.done:
        a = np.zeros(4)
        a[:3] = np.clip((s.obs["goal"] - s.obs["ee_target"]) / s.u.max_step, -1.0, 1.0)
        a[3] = -1.0
        s._step(a)
    return s


def names(script: Script) -> list[str]:
    """The observable events of the run (``info["events"]``)."""
    return [e["name"] for info in script.infos for e in info["events"]]


def all_names(script: Script) -> list[str]:
    """Every event of the run, ground truth included, in time order."""
    events = [e for info in script.infos for e in (*info["events"], *info["privileged"]["events"])]
    return [e["name"] for e in sorted(events, key=lambda e: e["t"])]


# ----------------------------------------------------------------- checker


def test_ids_are_registered() -> None:
    for env_id, task in E.ENV_IDS.items():
        env = gym.make(env_id)
        assert env.unwrapped.task == task
        assert env.unwrapped.control == "osc"
        env.close()
    env = gym.make("vfa.sim.env:vfa/PandaLift-v0", control="torque")
    assert env.action_space.shape == (8,)
    env.close()


@pytest.mark.parametrize("control", CONTROLS)
@pytest.mark.parametrize("env_id", sorted(E.ENV_IDS))
def test_check_env_passes(env_id: str, control: str) -> None:
    env = gym.make(env_id, control=control)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env.unwrapped, skip_render_check=True)
    other = [str(w.message) for w in caught if not any(s in str(w.message) for s in _INF_WARNINGS)]
    assert other == []  # no checker complaint beyond the unbounded physical signals
    env.close()


@pytest.mark.render
def test_check_env_with_render_and_frame() -> None:
    env = gym.make("vfa/PandaPickPlace-v0", render_mode="rgb_array")
    check_env(env.unwrapped)  # also builds the env through its spec in every render mode
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (*E.RENDER_SIZE, 3) and frame.dtype == np.uint8
    assert frame.std() > 10  # a picture, not a blank buffer
    env.close()
    env.close()  # closing twice is allowed


# ------------------------------------------------------------- determinism


def _rollout(env_id: str, control: str, seed: int, steps: int = 50):
    env = gym.make(env_id, control=control)
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    out = [(obs, 0.0, info)]
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        out.append((obs, reward, info))
        assert not (terminated or truncated)
    env.close()
    return out


@pytest.mark.parametrize(
    "env_id, control",
    [(i, "osc") for i in sorted(E.ENV_IDS)] + [("vfa/PandaLift-v0", "torque")],
)
def test_same_seed_same_observation_sequence(env_id: str, control: str) -> None:
    a, b = _rollout(env_id, control, 11), _rollout(env_id, control, 11)
    for (obs_a, r_a, info_a), (obs_b, r_b, info_b) in zip(a, b, strict=True):
        assert obs_a.keys() == obs_b.keys()
        for key in obs_a:
            assert obs_a[key].dtype == obs_b[key].dtype
            assert np.array_equal(obs_a[key], obs_b[key]), key  # bit for bit
        assert r_a == r_b
        assert info_a == info_b
    c = _rollout(env_id, control, 12, steps=1)
    assert not np.array_equal(a[1][0]["joint_pos"], c[1][0]["joint_pos"])


def test_reset_seed_reproduces_after_other_episodes() -> None:
    """The same seed gives the same episode whatever ran in the env before."""
    action = np.array([0.2, 0.1, -0.5, 0.3], dtype=np.float32)

    def run(env: E.PandaVFAEnv) -> list[dict]:
        obs, _ = env.reset(seed=5)
        return [obs] + [env.step(action)[0] for _ in range(60)]

    fresh = run(E.PandaVFAEnv("lift"))
    env = gym.make("vfa/PandaLift-v0")
    lift(env, 1, "fragile_soft", 20.0)  # touches and breaks a cube: warmstart came back
    env.unwrapped.reset(seed=2)
    for _ in range(20):
        env.unwrapped.step(env.action_space.sample())
    again = run(env.unwrapped)
    for o_fresh, o_again in zip(fresh, again, strict=True):
        for key in o_fresh:
            assert np.array_equal(o_fresh[key], o_again[key]), key


# --------------------------------------------------------------- reach


def test_scripted_osc_policy_solves_reach() -> None:
    env = gym.make("vfa/PandaReach-v0")
    results = []
    for seed in range(10):
        s = reach(env, seed)
        info = s.infos[-1]
        results.append(info["result"])
        if info["result"] == "SUCCESS":
            dist = np.linalg.norm(s.obs["ee_pos"] - s.obs["goal"])
            assert dist <= E.REACH_TOLERANCE
            assert info["t"] < E.TIME_LIMIT_S["reach"]
            assert s.rewards[-1] == pytest.approx(-dist)
    print(f"reach: {results.count('SUCCESS')}/10 seeds solved, results {results}")
    assert results.count("SUCCESS") >= 9
    env.close()


def test_reach_truncates_at_its_time_limit() -> None:
    env = gym.make("vfa/PandaReach-v0")
    obs, _ = env.reset(seed=0)
    steps, truncated, terminated = 0, False, False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        steps += 1
    assert truncated and not terminated
    assert info["result"] == "TIMEOUT" and not info["is_success"]
    assert steps == round(E.TIME_LIMIT_S["reach"] * E.STEP_HZ)
    assert env.unwrapped.episode.result == "TIMEOUT"
    with pytest.raises(RuntimeError, match="reset"):
        env.unwrapped.step(np.zeros(4))
    env.close()


# --------------------------------------------------------- hidden properties


def test_hidden_properties_stay_out_of_the_observation() -> None:
    """Same seed, five classes: the observation before contact does not tell them apart."""
    env = E.PandaVFAEnv("lift")
    action = np.array([0.3, -0.2, -0.4, -1.0], dtype=np.float32)
    runs = {}
    for cls in CONFIG.names:
        obs, info = env.reset(seed=3, options={"object_class": cls})
        assert info["privileged"]["object"]["class_name"] == cls
        seq = [obs]
        for _ in range(40):
            obs, *_ = env.step(action)
            seq.append(obs)
        runs[cls] = seq
    assert set(seq[0]) == set(env.observation_space.spaces)
    ref = runs[CONFIG.names[0]]
    for cls in CONFIG.names[1:]:
        for o_ref, o in zip(ref, runs[cls], strict=True):
            for key in o:  # bit for bit (the env runs without solver warmstart until contact)
                assert np.array_equal(o[key], o_ref[key]), (cls, key)
    assert env.model.opt.disableflags & E._WARMSTART  # no contact yet
    assert not set(OB.HIDDEN_FIELDS) & set(seq[0])


def test_hidden_properties_stay_out_of_the_torque_observation() -> None:
    """Torque mode, the arm moving near the cube: identical across classes until a touch."""
    runs = {}
    for cls in CONFIG.names:
        env = E.PandaVFAEnv("lift", control="torque", step_hz=500)
        obs, _ = env.reset(seed=4, options={"object_class": cls})
        t = TorqueScript(env, obs).grip(None)
        cube = obs["cube_pos"].copy()
        t.move(cube + (0.0, 0.0, 0.08), 0.4).move(cube + (0.1, -0.1, 0.05), 0.3).wait(0.1)
        assert env._untouched
        runs[cls] = t.observations
    ref = runs[CONFIG.names[0]]
    for cls in CONFIG.names[1:]:
        for o_ref, o in zip(ref, runs[cls], strict=True):
            for key in o_ref:
                assert np.array_equal(o[key], o_ref[key]), (cls, key)


def test_reset_with_an_object_spec_repeats_the_situation() -> None:
    """T15 retries one situation: the same spec gives the same hidden properties."""
    spec = OB.sample_object(CONFIG.classes["slippery"], np.random.default_rng(4))
    env = E.PandaVFAEnv("lift")
    obs_a, info_a = env.reset(seed=9, options={"object": spec})
    obs_b, info_b = env.reset(seed=9, options={"object": spec})
    assert info_a["privileged"]["object"] == info_b["privileged"]["object"] == spec.hidden()
    assert env.object_spec == spec
    for key in obs_a:
        assert np.array_equal(obs_a[key], obs_b[key])
    assert obs_a["cube_pos"][:2].tolist() == [spec.x, spec.y]
    assert obs_a["cube_size"][0] == spec.size


# ------------------------------------------------------------- lift outcomes


def test_scripted_lift_succeeds_and_is_recorded(tmp_path) -> None:
    env = gym.make("vfa/PandaLift-v0")
    s = lift(env, 0, "light_rigid", GRIP["light_rigid"])
    info = s.infos[-1]
    assert info["result"] == "SUCCESS" and info["is_success"]
    assert s.rewards[-1] > E.SUCCESS_BONUS  # the bonus on top of the shaped terms
    assert names(s) == ["CONTACT_ONSET", "GRASP_CLOSED"]  # LIFTED is ground truth
    assert all_names(s) == ["CONTACT_ONSET", "GRASP_CLOSED", "LIFTED"]
    phases = [i["phase"] for i in s.infos]
    assert phases[0] == "APPROACH" and "GRASP" in phases and phases[-1] == "CARRY"
    # The shaped reward climbs: approach < grasped < lifted.
    assert s.rewards[0] < s.rewards[phases.index("GRASP") + 5] < max(s.rewards[:-1])
    ep = env.unwrapped.episode
    assert ep.result == "SUCCESS"
    assert [e.name.value for e in ep.events] == all_names(s)
    assert ep.phases[:2] == ("SETTLE", "APPROACH")
    assert ep.context["cube"]["size"] == pytest.approx(s.obs["cube_size"][0])
    assert ep.meta["hidden"]["class_name"] == "light_rigid"
    assert ep.n == round((E.SETTLE_S + info["t"]) * 500)
    assert load_episode(ep.save(tmp_path / "ep")) == ep
    env.close()


def test_lift_breaks_a_fragile_cube_with_a_hard_squeeze() -> None:
    env = gym.make("vfa/PandaLift-v0")
    s = lift(env, 0, "fragile_soft", 20.0)
    assert s.done and s.infos[-1]["result"] == "BREAK"
    assert "BREAK" in all_names(s) and "BREAK" not in names(s)
    assert s.infos[-1]["privileged"]["broken"]
    env.close()


def test_lift_ends_in_drop_when_the_gripper_opens_in_the_air() -> None:
    env = gym.make("vfa/PandaLift-v0")
    s = lift(env, 0, "light_rigid", GRIP["light_rigid"], open_after=0.3)
    assert s.infos[-1]["result"] == "DROP"
    seen = all_names(s)
    assert seen.index("LIFTED") < seen.index("DROP") and seen[-1] == "DROP"
    assert env.unwrapped.sensors.slip.watching is False  # the carry ended at the drop
    env.close()


def test_safety_stop_ends_the_episode() -> None:
    """T09's supervisor reports SAFETY_STOP through the event detector."""
    env = gym.make("vfa/PandaLift-v0")
    env.reset(seed=0)
    env.step(np.zeros(4, dtype=np.float32))
    u = env.unwrapped
    t = float(u.data.time)
    u.events.report(Event.SAFETY_STOP, t, reason="test")
    *_, terminated, truncated, info = env.step(np.array([0, 0, 0, 1.0], dtype=np.float32))
    assert terminated and not truncated and info["result"] == "SAFETY_STOP"
    assert float(u.data.time) == t  # the new action never ran
    assert u.gripper.command.mode == "position"  # ... nor its squeeze
    assert [(e["name"], e["info"]) for e in info["events"]] == [("SAFETY_STOP", {"reason": "test"})]
    assert u.episode.result == "SAFETY_STOP"
    assert [e.name for e in u.episode.events] == [Event.SAFETY_STOP]
    env.close()


# -------------------------------------------------------- pick-and-place


def test_scripted_pick_and_place_succeeds() -> None:
    env = gym.make("vfa/PandaPickPlace-v0")
    s = pick_place(env, 0, "light_rigid", GRIP["light_rigid"])
    assert s.infos[-1]["result"] == "SUCCESS"
    assert all_names(s) == ["CONTACT_ONSET", "GRASP_CLOSED", "LIFTED", "PLACED"]
    # info["events"] holds the observable events only; ground truth is privileged.
    assert names(s) == ["CONTACT_ONSET", "GRASP_CLOSED"]
    hidden = [e["name"] for info in s.infos for e in info["privileged"]["events"]]
    assert hidden == ["LIFTED", "PLACED"]
    assert {e.value for e in E.PRIVILEGED_EVENTS} == {"BREAK", "LIFTED", "PLACED"}
    phases = list(dict.fromkeys(i["phase"] for i in s.infos))
    # The set-down is PLACE (no SLIP or DROP from the table taking the load), then RELEASE.
    assert phases == ["APPROACH", "GRASP", "CARRY", "PLACE", "RELEASE"]
    # Lowering the cube onto the target keeps the shaped reward of the carry.
    by_phase = {
        p: [r for r, i in zip(s.rewards, s.infos, strict=True) if i["phase"] == p] for p in phases
    }
    assert min(by_phase["PLACE"] + by_phase["RELEASE"][:-1]) >= 0.95 * max(by_phase["CARRY"])
    assert s.rewards[-1] > E.SUCCESS_BONUS
    env.close()


def test_pick_and_place_open_high_up_is_a_drop() -> None:
    env = gym.make("vfa/PandaPickPlace-v0")
    s = lift(env, 0, "light_rigid", GRIP["light_rigid"], open_after=0.3)
    assert s.infos[-1]["result"] == "DROP"
    assert "RELEASE" not in [i["phase"] for i in s.infos]
    env.close()


# ----------------------------------------------------------- interfaces


def test_control_rate_steps_for_skills() -> None:
    """T08 steps at 500 Hz: one control tick per step, 1 mm of target per step at most."""
    env = E.PandaVFAEnv("lift", step_hz=500)
    obs, _ = env.reset(seed=0)
    t0 = float(env.data.time)
    obs2, *_ = env.step(np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert float(env.data.time) - t0 == pytest.approx(0.002)
    assert obs2["ee_target"][0] - obs["ee_target"][0] == pytest.approx(0.001)
    assert env.metadata["render_fps"] == 500
    with pytest.raises(ValueError, match="step_hz"):
        E.PandaVFAEnv("lift", step_hz=30)


def test_torque_mode_writes_raw_actuator_commands() -> None:
    env = E.PandaVFAEnv("lift", control="torque")
    env.reset(seed=0)
    a = np.array([0.5, -0.25, 0.1, 0.0, 1.0, -1.0, 0.2, 0.3], dtype=np.float32)
    env.step(a)
    ctrl = env.data.ctrl
    limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    assert np.allclose(ctrl[env.ids.arm_actuators], a[:7].astype(float) * limits)
    assert ctrl[env.ids.gripper_actuator] == pytest.approx(0.3 * 100.0)  # positive opens


def test_skill_can_drive_carries_and_phases() -> None:
    """With ``auto_carry=False`` the env leaves the slip watch to the caller (T08)."""
    env = E.PandaVFAEnv("lift", auto_carry=False)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    env.set_phase("PROBE")
    s = Script(env, obs).grasp(GRIP["light_rigid"])
    assert {i["phase"] for i in s.infos} == {"PROBE"}
    assert env.sensors.watch_id == 0
    env.set_phase(None)
    assert env.phase in ("GRASP", "APPROACH")
    with pytest.raises(ValueError):
        env.set_phase("")


def test_input_checks() -> None:
    with pytest.raises(ValueError, match="task"):
        E.PandaVFAEnv("push")
    with pytest.raises(ValueError, match="control"):
        E.PandaVFAEnv("lift", control="velocity")
    with pytest.raises(ValueError, match="object_class"):
        E.PandaVFAEnv("lift", object_class="glass")
    with pytest.raises(ValueError, match="render_mode"):
        E.PandaVFAEnv("lift", render_mode="human")
    with pytest.raises(ValueError, match="close_speed"):
        E.PandaVFAEnv("lift", close_speed=2.0)
    with pytest.raises(ValueError, match="record_stride"):
        E.PandaVFAEnv("lift", record_stride=0)
    env = E.PandaVFAEnv("lift")
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))
    with pytest.raises(ValueError, match="unknown reset options"):
        env.reset(seed=0, options={"spec": None})
    with pytest.raises(TypeError, match="ObjectSpec"):
        env.reset(seed=0, options={"object": "light_rigid"})
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([math.nan, 0.0, 0.0, 0.0]))
    before = env.ee_target.copy()
    obs, *_ = env.step(np.array([5.0, 0.0, 0.0, -1.0]))  # clipped to 1
    assert obs["ee_target"] - before == pytest.approx([env.max_step, 0.0, 0.0])
    assert env.observation_space.contains(obs)


def test_no_recording_when_disabled() -> None:
    env = E.PandaVFAEnv("reach", record=False)
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.zeros(4, dtype=np.float32))
    assert env.recorder is None and env.episode is None


def test_start_pose_and_target_box() -> None:
    """Episodes start inside the OSC workspace; the target never leaves it."""
    env = E.PandaVFAEnv("lift")
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs["ee_pos"], E.START_POS, atol=1e-4)
    assert np.allclose(obs["ee_target"], E.START_POS, atol=1e-4)
    assert np.abs(obs["ee_vel"]).max() < 1e-4  # at rest after the settle
    for _ in range(30):  # straight up into the box ceiling, and diagonally out
        obs, *_ = env.step(np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32))
    assert np.all(obs["ee_target"] <= E.TARGET_HIGH)
    assert obs["ee_target"][2] == pytest.approx(E.TARGET_HIGH[2])


def test_diagonal_action_keeps_the_speed_limit() -> None:
    env = E.PandaVFAEnv("reach")
    env.reset(seed=0)
    before = env.ee_target.copy()
    env.step(np.array([-1.0, 1.0, -1.0, -1.0], dtype=np.float32))
    assert np.linalg.norm(env.ee_target - before) == pytest.approx(env.max_step)


def test_open_low_then_squeeze_again_does_not_hide_a_drop() -> None:
    """A one-step open near the table, a re-squeeze, then a drop from 15 cm is a DROP."""
    for task in ("pick_place", "lift"):
        env = E.PandaVFAEnv(task, close_speed=0.5)
        obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
        cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
        g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
        s = Script(env, obs).grasp(GRIP["light_rigid"])
        s.move(cube + (0.0, 0.0, 0.01), 0.05, g).wait(0.2, g)
        s._step(np.array([0.0, 0.0, 0.0, -1.0]))  # open for one step ...
        s.wait(0.3, g)  # ... and squeeze again
        assert env.phase == "CARRY"
        s.move(cube + (0.0, 0.0, 0.15), 0.15, g)
        if task == "pick_place":
            s.move((goal[0], goal[1], cube[2] + 0.15), 0.2, g)
        s.wait(0.3, g).wait(1.0, -1.0)
        assert s.infos[-1]["result"] == "DROP", task


def test_place_success_needs_a_low_release() -> None:
    """Ground truth backs the release rule: a cube that left the fingers high fell."""
    env = E.PandaVFAEnv("pick_place", auto_carry=False)  # no carry watch: no DROP event
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["light_rigid"])
    s.move(cube + (0.0, 0.0, 0.08), 0.15, g).move((goal[0], goal[1], cube[2] + 0.08), 0.2, g)
    s.wait(0.3, g).wait(2.0, -1.0)
    assert all_names(s)[-1] == "PLACED" and "DROP" not in all_names(s)
    assert s.infos[-1]["result"] == "DROP"


@pytest.mark.parametrize(
    "cls, heft",
    [
        ("light_rigid", 0.022),
        ("slippery", 0.015),
        pytest.param("light_rigid", 0.015, marks=pytest.mark.slow),
        pytest.param("heavy_rigid", 0.022, marks=pytest.mark.slow),
    ],
)
def test_heft_set_down_is_not_a_drop(cls: str, heft: float) -> None:
    """A heft (T06: 1-2 cm) set back on the table raises no SLIP or DROP; a lift follows."""
    env = E.PandaVFAEnv("lift")
    obs, _ = env.reset(seed=0, options={"object_class": cls})
    cube = obs["cube_pos"].copy()
    g = GRIP[cls] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP[cls])
    s.move(cube + (0.0, 0.0, heft), 0.05, g).wait(0.3, g)
    assert env.phase == "CARRY"
    s.move(cube, 0.05, g).wait(0.3, g)
    assert env.phase == "PLACE" and not s.done
    assert not {"SLIP", "DROP"} & set(names(s))
    s.move(cube + (0.0, 0.0, 0.15), 0.15, g).wait(3.0, g)
    assert s.infos[-1]["result"] == "SUCCESS"
    if cls == "light_rigid" and heft > 0.02:  # the heft lifts the cube past the 2 cm line
        assert all_names(s).count("LIFTED") == 2  # (a heavy cube's sag keeps it below)


def test_slip_at_lift_off_is_not_a_drop_after_lift_off() -> None:
    """A grip too weak to lift: SLIP and DROP fire, the cube never left the table: TIMEOUT."""
    env = E.PandaVFAEnv("lift")
    s = lift(env, 0, "heavy_slippery", 4.0)
    while not s.done:
        s._step(np.array([0.0, 0.0, 0.0, 4.0 / E.GRIP_FORCE_MAX]))
    assert "DROP" in names(s) and "LIFTED" not in all_names(s)
    assert s.infos[-1]["result"] == "TIMEOUT"


def test_dip_near_the_table_does_not_hide_a_later_drop() -> None:
    """Lift past 2 cm, dip back below it (a set-down), raise again, open high: DROP."""
    env = E.PandaVFAEnv("lift")
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube = obs["cube_pos"].copy()
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["light_rigid"])
    s.move(cube + (0.0, 0.0, 0.03), 0.05, g).wait(0.2, g)
    s.move(cube + (0.0, 0.0, 0.015), 0.05, g).wait(0.2, g)
    assert env.phase == "PLACE" and env.events.lifted  # still in the air
    s.move(cube + (0.0, 0.0, 0.15), 0.15, g).wait(0.3, g)
    assert env.phase == "CARRY"
    s.wait(1.0, -1.0)
    assert s.infos[-1]["result"] == "DROP"


def test_break_in_the_air_after_a_heft_is_not_a_place_break() -> None:
    env = E.PandaVFAEnv("pick_place")
    obs, _ = env.reset(seed=0, options={"object_class": "fragile_soft"})
    cube = obs["cube_pos"].copy()
    g = GRIP["fragile_soft"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["fragile_soft"])
    s.move(cube + (0.0, 0.0, 0.015), 0.05, g).wait(0.2, g).move(cube, 0.05, g).wait(0.3, g)
    assert env.phase == "PLACE"
    s.move(cube + (0.0, 0.0, 0.10), 0.15, g).wait(0.2, g)
    s.wait(2.0, 20.0 / E.GRIP_FORCE_MAX)  # squeeze far past the break force in the air
    assert s.infos[-1]["result"] == "BREAK"


def test_warmstart_returns_at_the_first_touch() -> None:
    env = gym.make("vfa/PandaLift-v0")
    s = lift(env, 0, "light_rigid", GRIP["light_rigid"])
    u = env.unwrapped
    assert s.infos[-1]["result"] == "SUCCESS"
    assert not u.model.opt.disableflags & E._WARMSTART
    u.reset(seed=1)
    assert u.model.opt.disableflags & E._WARMSTART
    env.close()


def test_torque_mode_slow_set_down_is_a_place() -> None:
    """Torque mode reads the measured motion and load: a 4 mm/s set-down is a place."""
    env = E.PandaVFAEnv("pick_place", control="torque", step_hz=500)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.3).wait(0.3).move(cube, 0.1).wait(0.3)
    t.grip(GRIP["light_rigid"]).wait(0.9)
    t.move(cube + (0.0, 0.0, 0.06), 0.1).move((goal[0], goal[1], cube[2] + 0.06), 0.15)
    lowering = len(t.infos)  # (this driver stops abruptly: a jolt may fire SLIP before)
    t.move((goal[0], goal[1], cube[2] + 0.008), 0.05).wait(0.2)
    t.move((goal[0], goal[1], cube[2] - 0.002), 0.004).wait(0.2)  # 4 mm/s onto the table
    assert env.phase == "PLACE"
    t.grip(None).wait(0.5).move((goal[0], goal[1], 0.2), 0.1).wait(1.0)
    events = [e["name"] for i in t.infos[lowering:] for e in i["events"]]
    assert not {"SLIP", "DROP"} & set(events) and t.infos[-1]["result"] == "SUCCESS"


def test_torque_mode_payload_sag_is_no_set_down() -> None:
    """A heavy cube lifted 1.5 cm sags the hand 3+ mm when the lift stops: still a carry."""
    env = E.PandaVFAEnv("lift", control="torque", step_hz=500)
    obs, _ = env.reset(seed=1, options={"object_class": "heavy_rigid"})
    cube = obs["cube_pos"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.4).wait(0.1).move(cube, 0.15).wait(0.2)
    t.grip(GRIP["heavy_rigid"]).wait(1.0).move(cube + (0.0, 0.0, 0.015), 0.2).wait(1.0)
    assert env.phase == "CARRY"
    assert env._carry_peak - (float(t.obs["ee_pos"][2]) - env._grasp_z) > E.CARRY_RISE  # sagged


def test_torque_mode_slow_raise_from_a_set_down_is_a_new_carry() -> None:
    """Torque mode: a set-down, a 4 mm/s raise (a new carry), an open high: DROP."""
    env = E.PandaVFAEnv("lift", control="torque", step_hz=500)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube = obs["cube_pos"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.4).wait(0.1).move(cube, 0.15).wait(0.2)
    t.grip(GRIP["light_rigid"]).wait(0.6).move(cube + (0.0, 0.0, 0.015), 0.05).wait(0.05)
    assert env.phase == "CARRY"
    t.move(cube + (0.0, 0.0, -0.002), 0.05).wait(0.2)  # back on the table
    assert env.phase == "PLACE"
    t.move(cube + (0.0, 0.0, 0.01), 0.004)  # 4 mm/s: slower than the set-down rebound
    assert env.phase == "CARRY"
    t.move(cube + (0.0, 0.0, 0.032), 0.05)
    t.grip(None).wait(1.0)
    assert t.infos[-1]["result"] == "DROP"


def test_pushing_the_cube_onto_the_target_is_not_a_place() -> None:
    """A heft, a set-down, then a push along the table to the target: PLACED, no success."""
    env = E.PandaVFAEnv("pick_place")
    obs, _ = env.reset(seed=2, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    size = float(obs["cube_size"][0])
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["light_rigid"])
    s.move(cube + (0.0, 0.0, 0.025), 0.05, g).wait(0.2, g)  # the heft clears 2 cm: LIFTED
    s.move(cube, 0.05, g).wait(0.2, g).wait(0.4, -1.0)  # set down, open
    s.move(cube + (0.0, 0.0, 0.10), 0.2, -1.0)
    way = (goal[:2] - cube[:2]) / np.linalg.norm(goal[:2] - cube[:2])
    behind = cube[:2] - way * (size / 2 + 0.03)
    z = cube[2]
    s.move((*behind, 0.10), 0.2, 0.3).move((*behind, z), 0.1, 0.3).wait(0.3, 0.3)
    end = goal[:2] - way * (size / 2 + 0.012)
    s.move((*end, z), 0.05, 0.3, max_s=8.0).wait(0.2, 0.3)
    s.move((*end, 0.15), 0.1, 0.3).wait(1.0, 0.3)
    assert "PLACED" in all_names(s)
    assert s.infos[-1]["result"] != "SUCCESS"


@pytest.mark.parametrize(
    "cls, lift_h",
    [("light_rigid", 0.10), pytest.param("heavy_rigid", 0.05, marks=pytest.mark.slow)],
)
def test_straight_line_transport_places(cls: str, lift_h: float) -> None:
    """A slanted transport straight down to the target is a place through the air."""
    env = E.PandaVFAEnv("pick_place")
    obs, _ = env.reset(seed=0, options={"object_class": cls})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    g = GRIP[cls] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP[cls])
    s.move(cube + (0.0, 0.0, lift_h), 0.15, g).wait(0.2, g)
    s.move(goal + (0.0, 0.0, 0.002), 0.2, g).wait(0.3, g)  # one straight line to the goal
    s.wait(0.5, -1.0).move(goal + (0.0, 0.0, 0.15), 0.1, -1.0).wait(1.0, -1.0)
    assert s.infos[-1]["result"] == "SUCCESS"


def test_low_carry_places() -> None:
    """A lift-off, then a transport with the cube 1.5 cm up: still a place through the air.

    (``PLACED`` needs a ``LIFTED`` first, T06: a cube that never cleared 2 cm was not lifted.)
    """
    env = E.PandaVFAEnv("pick_place")
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["light_rigid"])
    s.move(cube + (0.0, 0.0, 0.03), 0.05, g).move(cube + (0.0, 0.0, 0.015), 0.05, g)
    s.move((goal[0], goal[1], cube[2] + 0.015), 0.2, g)
    s.move(goal + (0.0, 0.0, 0.002), 0.05, g).wait(0.3, g)
    s.wait(0.5, -1.0).move(goal + (0.0, 0.0, 0.15), 0.1, -1.0).wait(1.0, -1.0)
    assert s.infos[-1]["result"] == "SUCCESS"


@pytest.mark.parametrize("speed", [0.05, pytest.param(0.3, marks=pytest.mark.slow)])
def test_torque_mode_set_down_rebound_is_no_new_carry(speed: float) -> None:
    """The ee rebounds at 5-6 mm/s when a torque-mode set-down stops: still PLACE."""
    env = E.PandaVFAEnv("lift", control="torque", step_hz=500)
    obs, _ = env.reset(seed=0, options={"object_class": "heavy_rigid"})
    cube = obs["cube_pos"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.4).wait(0.1).move(cube, 0.15).wait(0.2)
    t.grip(GRIP["heavy_rigid"]).wait(1.0).move(cube + (0.0, 0.0, 0.05), 0.1).wait(0.2)
    carries = env.sensors.watch_id
    t.move(cube + (0.0, 0.0, -0.005), speed).wait(1.0)
    assert env.phase == "PLACE" and env.sensors.watch_id == carries
    assert not {"SLIP", "DROP"} & {e["name"] for i in t.infos for e in i["events"]}


def test_drop_then_nudge_onto_the_target_is_a_drop() -> None:
    """Dropped from 8 cm short of the target, then nudged in: the touchdown decides, DROP."""
    env = E.PandaVFAEnv("pick_place", auto_carry=False, close_speed=0.5)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    size = float(obs["cube_size"][0])
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs)
    s.move(cube + (0.0, 0.0, 0.05), 0.5, -1.0).move(cube, 0.3, -1.0).wait(0.3, g)
    way = (goal[:2] - cube[:2]) / np.linalg.norm(goal[:2] - cube[:2])
    drop = goal[:2] - way * 0.03
    s.move(cube + (0.0, 0.0, 0.08), 0.2, g).move((*drop, cube[2] + 0.08), 0.3, g)
    s.wait(0.2, g).wait(0.6, -1.0)  # open 8 cm up
    c = np.array(s.infos[-1]["privileged"]["cube_pos"])
    to_goal = (goal[:2] - c[:2]) / np.linalg.norm(goal[:2] - c[:2])
    behind = c[:2] - to_goal * (size / 2 + 0.015)
    s.move((*behind, 0.12), 0.5, 0.3).move((*behind, cube[2]), 0.3, 0.3).wait(0.1, 0.3)
    end = goal[:2] - to_goal * (size / 2 + 0.006)  # nudge it onto the target, fingers closed
    s.move((*end, cube[2]), 0.05, 0.3, max_s=3.0).wait(0.1, 0.3).move((*end, 0.12), 0.3, 0.3)
    s.wait(2.0, 0.3)
    assert "PLACED" in all_names(s)
    assert s.infos[-1]["result"] == "DROP"


@pytest.mark.slow
def test_push_with_hops_onto_the_target_is_not_a_place() -> None:
    """A heft in place, then a fast push: the cube hops, but no carried touchdown: no place."""
    env = E.PandaVFAEnv("pick_place", close_speed=0.5)
    obs, _ = env.reset(seed=0, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    size = float(obs["cube_size"][0])
    g = GRIP["light_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs)
    s.move(cube + (0.0, 0.0, 0.05), 0.5, -1.0).move(cube, 0.3, -1.0).wait(0.05, -1.0)
    s.wait(0.25, g).move(cube + (0.0, 0.0, 0.026), 0.2, g).move(cube, 0.2, g)  # heft: LIFTED
    s.wait(0.05, g).wait(0.2, -1.0).move(cube + (0.0, 0.0, 0.07), 0.5, -1.0)
    way = (goal[:2] - cube[:2]) / np.linalg.norm(goal[:2] - cube[:2])
    behind = cube[:2] - way * (size / 2 + 0.03)
    z = cube[2] + 0.012
    s.move((*behind, 0.07), 0.5, 0.3).move((*behind, z), 0.3, 0.3).wait(0.1, 0.3)
    end = goal[:2] - way * (size / 2 + 0.012 - 0.02)
    s.move((*end, z), 0.5, 0.3, max_s=6.0).wait(0.1, 0.3).move((*end, 0.08), 0.5, 0.3)
    while not s.done:
        s.wait(0.5, 0.3)
    assert s.infos[-1]["result"] != "SUCCESS"


# ----------------------------------------------------------------- slow


@pytest.mark.slow
@pytest.mark.parametrize("cls", list(GRIP))
def test_scripted_lift_and_place_every_class(cls: str) -> None:
    """The hand-picked grip lifts and places every class through the env (3 seeds each)."""
    env = gym.make("vfa/PandaLift-v0")
    lifted = [lift(env, seed, cls, GRIP[cls]).infos[-1]["result"] for seed in range(3)]
    env.close()
    env = gym.make("vfa/PandaPickPlace-v0")
    placed = [pick_place(env, seed, cls, GRIP[cls]).infos[-1]["result"] for seed in range(3)]
    env.close()
    print(f"{cls}: lift {lifted}, pick-and-place {placed}")
    assert lifted == ["SUCCESS"] * 3
    assert placed == ["SUCCESS"] * 3
