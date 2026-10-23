"""T19: pick-and-place with a force-aware place (``vfa.skills.place``).

ROADMAP acceptance: hand-picked θ places each class in at least 7 of 10 seeds
(``test_hand_picked_theta_places_every_class``, slow); ``memory_full`` beats
``random_retry`` on attempts-to-success on the pick-and-place quick profile
(a benchmark run, docs/RESULTS.md §10). The other tests pin θ's place fields,
the impact break rule, the place labels, the pick-and-place keys, the
benchmark plumbing, and the two env limits T07 left (regression tests).
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from test_env import GRIP, Script, TorqueScript
from vfa.cognition import system2 as S2
from vfa.control.osc import control_step
from vfa.experiments import baselines as BL
from vfa.experiments import benchmark as B
from vfa.memory import embeddings as EM
from vfa.sim import env as E
from vfa.sim import objects as OB
from vfa.sim.sensors import REALISTIC_NOISE, SensorConfig
from vfa.skills import params as PR
from vfa.skills import place as P
from vfa.skills.params import SkillParams
from vfa.skills.pick import GraspLiftSkill, run_attempt

# Hand-picked θ per class: grips from each class's best calibration grip (RESULTS §3)
# with a margin for the transport, the default place fields (place on the table).
HAND_PICKED = {
    "light_rigid": SkillParams(grip_force=2.0, lift_accel=0.3),
    "heavy_rigid": SkillParams(grip_force=10.0, lift_accel=0.3),
    "slippery": SkillParams(grip_force=12.0, lift_accel=0.3),
    "fragile_soft": SkillParams(grip_force=1.5, lift_accel=0.3),
    "heavy_slippery": SkillParams(grip_force=15.0, lift_accel=0.3),
    "brittle": SkillParams(grip_force=3.0, lift_accel=0.3),
}
CONFIG = OB.load_object_config()


def _env(**kwargs) -> E.PandaVFAEnv:
    return P.make_env(SensorConfig(noise=REALISTIC_NOISE), **kwargs)


# ------------------------------------------------------------------- θ


def test_theta_place_fields_leave_the_lift_vector_alone() -> None:
    assert PR.FIELDS == ("grip_force", "close_speed", "lift_accel", "grasp_depth", "approach_speed")
    assert PR.LOW.shape == PR.HIGH.shape == (5,)
    assert PR.ALL_FIELDS == PR.FIELDS + ("place_speed", "release_height")
    theta = SkillParams()
    assert (theta.place_speed, theta.release_height) == (0.05, 0.0)
    assert theta.to_vector().shape == (5,)
    lift_only = {k: v for k, v in theta.to_dict().items() if k in PR.FIELDS}
    assert SkillParams.from_dict(lift_only) == theta  # a pre-T19 θ gets the defaults
    assert SkillParams.from_dict(theta.to_dict()) == theta
    with pytest.raises(ValueError):
        SkillParams.from_dict({**lift_only, "place_height": 0.0})
    with pytest.raises(ValueError):
        SkillParams(place_speed=0.5)
    with pytest.raises(ValueError):
        SkillParams(release_height=0.02)  # above 1.5 cm a light cube's release is a DROP
    assert "place" not in theta.describe() and "place=0.05m/s" in theta.describe(place=True)


def test_random_retry_draws_the_place_fields_from_their_own_stream() -> None:
    lift, place = BL.make_baseline("random_retry"), BL.make_baseline("random_retry", place=True)
    for key, attempt in [(1, 2), (987654321, 3), (5, 5)]:
        a, b = lift.theta(key, attempt), place.theta(key, attempt)
        assert a.to_vector().tolist() == b.to_vector().tolist()  # the lift's draws
        assert (a.place_speed, a.release_height) == (0.05, 0.0)
        assert PR.PLACE_LOW[0] <= b.place_speed <= PR.PLACE_HIGH[0]
        assert PR.PLACE_LOW[1] <= b.release_height <= PR.PLACE_HIGH[1]
    assert place.theta(1, 1) == BL.NAIVE


# ------------------------------------------------------------- the skill


def test_the_skill_needs_its_env() -> None:
    skill = P.PickPlaceSkill()
    with pytest.raises(ValueError):
        skill.validate(E.PandaVFAEnv("lift", step_hz=500, auto_carry=False))
    with pytest.raises(ValueError):  # the env's own release phases, not the skill's
        skill.validate(E.PandaVFAEnv("pick_place", step_hz=500, auto_carry=False))
    with pytest.raises(ValueError):
        GraspLiftSkill().validate(_env())
    skill.validate(_env())


def test_a_default_place_runs_every_phase_and_succeeds() -> None:
    env = _env()
    result = run_attempt(
        env,
        P.PickPlaceSkill(),
        HAND_PICKED["light_rigid"],
        seed=3,
        options={"object_class": "light_rigid"},
    )
    assert result.result == "SUCCESS"
    ep = result.episode
    seen = list(dict.fromkeys(np.asarray(ep.phases)[ep.arrays["phase"]]))
    assert seen[-4:] == ["TRANSPORT", "PLACE_DESCEND", "CONTACT", "RELEASE"]
    assert result.sim_s < E.TIME_LIMIT_S["pick_place"]
    assert ep.meta["task"] == "pick_place"


def test_the_slowest_corner_finishes_in_time() -> None:
    """Every speed and acceleration at its lower bound, the largest grip, the deepest grasp,
    the slowest place, on the cube farthest from the target: done before 15 s."""
    slow = SkillParams(
        grip_force=25.0,
        close_speed=0.05,
        lift_accel=0.3,
        grasp_depth=0.4,
        approach_speed=0.35,
        place_speed=PR.BOUNDS["place_speed"][0],
        release_height=0.0,
    )
    spec = OB.ObjectSpec(
        "heavy_rigid", 0.04, 1.0, (0.9, 0.005, 0.0001), (0.02, 1.0), None, 0.45, -0.15, 0.0
    )
    result = run_attempt(_env(), P.PickPlaceSkill(), slow, seed=0, options={"object": spec})
    assert result.result == "SUCCESS"
    assert result.sim_s < E.TIME_LIMIT_S["pick_place"] - 1.0


@pytest.mark.parametrize(
    "where, expected",
    [("off_target", "PLACE_MISS"), ("tilted", "TOPPLE"), ("in_the_air", "DROP")],
)
def test_the_place_labels(where: str, expected: str) -> None:
    env = _env()
    env.reset(seed=0, options={"object_class": "light_rigid"})
    skill = P.PickPlaceSkill()
    skill.stop_reason = "PLACE_UNJUDGED"
    size = env.object_spec.size
    yaw = 0.0
    z = size / 2 - 1e-4  # sunk 0.1 mm into the table, as a resting cube
    if where == "tilted":
        yaw, z = None, size / 2 * np.sqrt(2) - 1e-4  # resting on an edge
    elif where == "in_the_air":
        z = 0.2
    q = env.ids.cube_qpos
    qpos = env.data.qpos[q : q + 7]
    qpos[:3] = (0.55, 0.0, z)
    if yaw is None:
        half = np.pi / 8  # 45° about x
        qpos[3:] = (np.cos(half), np.sin(half), 0.0, 0.0)
    else:
        qpos[3:] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    assert skill.label(env) == expected


# --------------------------------------------------------- impact rule


def test_a_hard_landing_breaks_a_brittle_cube_only_with_the_impact_rule() -> None:
    def drop(impact: bool) -> OB.BreakMonitor:
        env = E.PandaVFAEnv("lift")
        spec = OB.sample_object(CONFIG.classes["brittle"], np.random.default_rng(0))
        spec = replace(spec, break_force=12.0)
        env.reset(seed=0, options={"object": spec})
        monitor = OB.BreakMonitor(env.model, env.data, spec, env.ids, impact=impact)
        env.data.qpos[env.ids.cube_qpos + 2] += 0.03  # 3 cm up, then let go
        mujoco.mj_forward(env.model, env.data)
        for _ in range(300):  # the arm holds its start pose
            control_step(env.model, env.data, (env.osc, env.gripper), 1)
            monitor.update()
        return monitor

    on, off = drop(True), drop(False)
    assert on.broken and on.peak_table > 12.0 and on.break_load == pytest.approx(on.peak_table)
    assert not off.broken and off.peak_force == 0.0 and off.peak_table == 0.0
    env = E.PandaVFAEnv("pick_place")
    env.reset(seed=0)
    assert env.monitor.impact
    env = E.PandaVFAEnv("lift")
    env.reset(seed=0)
    assert not env.monitor.impact  # the frozen lift benchmark never reads the table force


# -------------------------------------------------- memory on pick-and-place


def test_place_keys_extend_only_the_retry_and_full_keys() -> None:
    env = _env()
    result = run_attempt(
        env,
        P.PickPlaceSkill(),
        HAND_PICKED["light_rigid"],
        seed=3,
        options={"object_class": "light_rigid"},
    )
    view = EM.EpisodeView.from_episode(result.episode)
    lift, place = EM.Embedder(), EM.PlaceEmbedder()
    np.testing.assert_array_equal(lift.key_probe(view), place.key_probe(view))
    np.testing.assert_array_equal(lift.key_pre(view.context), place.key_pre(view.context))
    raw = EM.place_features(view)
    assert raw.shape == (EM.PLACE_DIMS,) and np.all(np.isfinite(raw))  # every place phase ran
    for name in ("key_retry", "full_embedding"):
        a = getattr(lift, name)(view)
        b = getattr(place, name)(view)
        assert b.shape == (a.size + EM.PLACE_DIMS,)
        assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-12
    assert EM.failure_features(replace(view, result="PLACE_BREAK"))[0] == 1.0  # the BREAK bucket


def test_a_place_break_bounds_the_grip_like_a_break() -> None:
    own = [S2.Evidence("a", 1.0, "PLACE_BREAK", 8.0, {}, 8.0)]
    window = S2.grip_window(own, own=1)
    assert window.high == pytest.approx(8.0 * (1 - S2.MARGIN)) and window.used == ("a",)
    neighbour = [S2.Evidence("b", 0.9, "PLACE_BREAK", 8.0, {}, 8.0)]
    assert S2.grip_window(neighbour).high == pytest.approx(8.0 * (1 - S2.MARGIN))
    assert S2.grip_window([S2.Evidence("c", 1.0, "TOPPLE", 8.0, {}, 8.0)], own=1).used == ()


# ----------------------------------------------------------- benchmark


def test_the_pick_place_benchmark_matches_its_hash() -> None:
    expected = B.PICK_PLACE_HASH.read_text().strip()
    assert B.benchmark_hash(B.PICK_PLACE_FILES) == expected
    config = B.load_pick_place_config()
    assert config.task == "pick_place" and set(config.profiles) == {"quick", "calib", "heldout"}
    lift = B.load_benchmark_config()
    lift_seeds = {s for p in lift.profiles.values() for s in p.seeds}
    assert not lift_seeds & {s for p in config.profiles.values() for s in p.seeds}


def test_pick_place_jobs_stay_apart_from_the_lift() -> None:
    config = B.load_pick_place_config()
    jobs = B.plan_condition(config, "memory_full", "quick")
    assert all(j.task == "pick_place" and j.key["task"] == "pick_place" for j in jobs)
    assert all(str(B.PICK_PLACE_OUT) in j.path for j in jobs)
    lift = B.plan_condition(B.load_benchmark_config(), "naive", "quick")
    assert all(j.task == "lift" and "task" not in j.key for j in lift)
    for condition in ("best_fixed", "bandit", "memory_ctx_only"):
        with pytest.raises(ValueError):
            B.plan_condition(config, condition, "quick")
    env, skill, _ = B._worker_env("realistic", "pick_place")
    assert env.task == "pick_place" and isinstance(skill, P.PickPlaceSkill)
    env, skill, _ = B._worker_env("realistic", "lift")
    assert env.task == "lift" and type(skill) is GraspLiftSkill


# -------------------------------------------- the two env limits from T07


def test_a_cube_dragged_while_pinched_touches_down_where_it_is_let_go() -> None:
    """A heavy cube grazes the table 8 cm short and is dragged, still pinched, onto the
    target: the touchdown follows it while both pads press (it used to latch at the
    graze: TIMEOUT)."""
    env = E.PandaVFAEnv("pick_place")
    obs, _ = env.reset(seed=0, options={"object_class": "heavy_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    g = GRIP["heavy_rigid"] / E.GRIP_FORCE_MAX
    s = Script(env, obs).grasp(GRIP["heavy_rigid"])
    s.move(cube + (0.0, 0.0, 0.05), 0.15, g).wait(0.2, g)
    way = (goal[:2] - cube[:2]) / np.linalg.norm(goal[:2] - cube[:2])
    graze = goal[:2] - way * 0.08
    s.move((*graze, cube[2] + 0.05), 0.2, g)
    s.move((*graze, cube[2] + 0.002), 0.05, g).wait(0.2, g)
    assert np.hypot(*(env._touchdown_xy - goal[:2])) > E.PLACE_TOUCHDOWN_RADIUS
    s.move((goal[0], goal[1], cube[2] + 0.002), 0.05, g, max_s=8.0).wait(0.3, g)
    s.wait(0.5, -1.0).move(goal + (0.0, 0.0, 0.15), 0.1, -1.0).wait(1.0, -1.0)
    assert s.infos[-1]["result"] == "SUCCESS"


def test_torque_mode_lowering_in_the_air_after_a_load_spike_is_no_set_down() -> None:
    """A fast lift start slipped a heavy_slippery cube in the grip and spiked the held load
    (1.4x); a 5 mm lowering in the air then passed the old set-down gate (PLACE with the
    cube 5 mm up). The reference load now ignores spikes: still CARRY."""
    env = E.PandaVFAEnv("pick_place", control="torque", step_hz=500)
    obs, _ = env.reset(seed=1, options={"object_class": "heavy_slippery"})
    cube = obs["cube_pos"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.4).wait(0.1).move(cube, 0.15).wait(0.2)
    t.grip(GRIP["heavy_slippery"]).wait(1.0)
    t.move(cube + (0.0, 0.0, 0.018), 0.3).wait(0.3)
    t.move(cube + (0.0, 0.0, 0.013), 0.02).wait(0.3)
    assert not env._cube_contacts() & 1  # in the air
    assert env.phase == "CARRY"


def test_the_set_down_reference_needs_a_held_level() -> None:
    """A spike on the first carry tick cannot become the reference: it needs a level the
    load held for SETDOWN_WINDOW_S, and a younger carry has none (no set-down)."""
    env = E.PandaVFAEnv("pick_place", control="torque", step_hz=500)
    env.reset(seed=0)
    env._start_carry(0.0)
    full = round(E.SETDOWN_WINDOW_S / env.dt)
    loads = [9.75] + [7.08] * 200
    for k, load in enumerate(loads, 1):
        env._track_carry_load(load)
        if k < full:
            assert env._carry_load == 0.0
    assert env._carry_load == 7.08


def _noisy_torque_carry(seed: int, final_z: float) -> tuple[E.PandaVFAEnv, TorqueScript]:
    """light_rigid under realistic noise in torque mode: grasp, carry 6 cm up to the
    target, lower at 2 cm/s to ``final_z`` above the cube's start height, hold."""
    env = E.PandaVFAEnv(
        "pick_place",
        control="torque",
        step_hz=500,
        sensor_config=SensorConfig(noise=REALISTIC_NOISE),
    )
    obs, _ = env.reset(seed=seed, options={"object_class": "light_rigid"})
    cube, goal = obs["cube_pos"].copy(), obs["goal"].copy()
    t = TorqueScript(env, obs).grip(None)
    t.move(cube + (0.0, 0.0, 0.12), 0.3).wait(0.3).move(cube, 0.1).wait(0.3)
    t.grip(GRIP["light_rigid"]).wait(0.9)
    t.move(cube + (0.0, 0.0, 0.06), 0.1).move((goal[0], goal[1], cube[2] + 0.06), 0.15)
    t.move((goal[0], goal[1], cube[2] + final_z), 0.02).wait(0.3)
    return env, t


@pytest.mark.parametrize("seed", [0, 3])
def test_torque_mode_noise_dip_in_the_air_is_no_set_down(seed: int) -> None:
    """Under realistic noise a two-tick dip 5 % below a 0.5 N reference set these cubes
    down 1.6 cm in the air; the 4σ noise margin keeps the carry."""
    env, _ = _noisy_torque_carry(seed, 0.016)
    assert not env._cube_contacts() & 1
    assert env.phase == "CARRY"


def test_torque_mode_noisy_set_down_still_places() -> None:
    env, t = _noisy_torque_carry(0, 0.008)
    goal = t.obs["goal"]
    t.move((goal[0], goal[1], t.obs["ee_pos"][2] - 0.01), 0.004).wait(0.2)  # onto the table
    assert env.phase == "PLACE"
    t.grip(None).wait(0.5).move((goal[0], goal[1], 0.2), 0.1).wait(1.0)
    assert t.infos[-1]["result"] == "SUCCESS"


def test_pick_place_out_option_takes_every_form(tmp_path) -> None:
    forms = (
        ["--out", str(tmp_path / "a")],
        [f"--out={tmp_path / 'b'}"],
        ["--ou", str(tmp_path / "c")],
    )
    for form in forms:
        assert B.main(["aggregate", "--task", "pick_place", "--partial", *form]) == 0
    for name in "abc":
        assert (tmp_path / name / "quick" / "metrics.json").exists()


# ------------------------------------------------------------ acceptance


@pytest.mark.slow
@pytest.mark.parametrize("cls", list(HAND_PICKED))
def test_hand_picked_theta_places_every_class(cls: str) -> None:
    """ROADMAP: hand-picked θ places each class in at least 7 of 10 seeds."""
    env = _env()
    skill = P.PickPlaceSkill()
    results = [
        run_attempt(env, skill, HAND_PICKED[cls], seed=seed, options={"object_class": cls}).result
        for seed in range(10)
    ]
    print(cls, results)
    assert sum(r == "SUCCESS" for r in results) >= 7, results
