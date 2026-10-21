"""T03: operational-space impedance controller.

Covers the acceptance list in ROADMAP T03: reach 10 random workspace targets
(< 1 cm, < 5° within 2 s of sim time), compliance (a lower stiffness deflects
more under the same ``xfrc_applied`` push), and throughput (a 10 s lift-length
episode in <= 1 s wall on one core, number printed). Also pins the torque law
against an independent reference, the null-space posture term, torque
clipping, the speed limit, and input validation.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import mujoco
import numpy as np
import pytest

from vfa.control import osc as O
from vfa.control import throughput
from vfa.control.gripper import Gripper
from vfa.sim import scene as S

REACH_SECONDS = 2.0
POS_TOL_M = 0.01
ROT_TOL_RAD = math.radians(5.0)
CUBE_ASIDE = (0.2, 0.6)  # cube out of the reach workspace


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return S.build_scene()


@pytest.fixture(scope="module")
def ids(model: mujoco.MjModel) -> S.SceneIds:
    return S.scene_ids(model)


def _fresh(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, *CUBE_ASIDE)
    return data


def _run(model, data, controllers, seconds, hz=O.CONTROL_HZ) -> None:
    substeps = O.substeps_for(model, hz)
    for _ in range(round(seconds * hz)):
        O.control_step(model, data, controllers, substeps)
    mujoco.mj_forward(model, data)


def _ee_errors(data, ids, pos, rot) -> tuple[float, float]:
    """Pose error measured from the site, independent of the controller's own error."""
    pos_err = float(np.linalg.norm(data.site_xpos[ids.ee_site] - np.asarray(pos)))
    rot_err = O.rotation_angle(data.site_xmat[ids.ee_site].reshape(3, 3), rot)
    return pos_err, rot_err


def _reach(model, ids, target, yaw=0.0, hz=O.CONTROL_HZ, gains=None) -> tuple[float, float]:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, gains, ids=ids, control_hz=hz)
    osc.set_target(target, yaw=yaw)
    _run(model, data, [osc], REACH_SECONDS, hz)
    return _ee_errors(data, ids, target, O.top_down_rot(yaw))


# ------------------------------------------------------------------ acceptance


def test_reach_10_random_workspace_targets(model, ids) -> None:
    rng = np.random.default_rng(3)
    targets = rng.uniform(O.WORKSPACE_LOW, O.WORKSPACE_HIGH, size=(10, 3))
    for target in targets:
        pos_err, rot_err = _reach(model, ids, target)
        assert pos_err < POS_TOL_M, (target, pos_err)
        assert rot_err < ROT_TOL_RAD, (target, math.degrees(rot_err))


@pytest.mark.parametrize("hz", [O.CONTROL_HZ, 250, 100])
@pytest.mark.parametrize("corner", range(8))
def test_reach_every_workspace_corner(model, ids, corner: int, hz: int) -> None:
    """The box corners are the hardest targets (longest moves, most extended poses).

    At 100 Hz each torque is held for 5 physics steps (10 ms): the far-high
    corners went unstable there with damping from diag(Λ); the apparent-mass
    damping keeps them stable.
    """
    pick = [(corner >> axis) & 1 for axis in range(3)]
    target = np.where(pick, O.WORKSPACE_HIGH, O.WORKSPACE_LOW)
    pos_err, rot_err = _reach(model, ids, target, hz=hz)
    assert pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD, (target, pos_err, rot_err)


@pytest.mark.parametrize("yaw", [-O._YAW_LIMIT, O._YAW_LIMIT])
def test_reach_every_workspace_corner_at_the_yaw_band_edges(model, ids, yaw: float) -> None:
    """|yaw| <= π/2 + GRASP_YAW_MARGIN is reachable in the whole box, clear of joint 7's limit."""
    q7 = model.joint(S.ARM_JOINTS[6])
    for target in itertools.product(*zip(O.WORKSPACE_LOW, O.WORKSPACE_HIGH, strict=True)):
        data = _fresh(model)
        osc = O.OperationalSpaceController(model, data, ids=ids)
        osc.set_target(target, yaw=yaw)
        headroom = math.inf
        for _ in range(round(REACH_SECONDS * O.CONTROL_HZ)):
            O.control_step(model, data, [osc], 1)
            q = data.qpos[ids.arm_qpos[6]]
            headroom = min(headroom, q - q7.range[0], q7.range[1] - q)
            assert not osc.at_joint_limit, (yaw, target)
        mujoco.mj_forward(model, data)
        pos_err, rot_err = _ee_errors(data, ids, target, O.top_down_rot(yaw))
        assert pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD, (yaw, target, pos_err, rot_err)
        assert headroom > O.JOINT_LIMIT_MARGIN, (yaw, target, headroom)


@pytest.mark.parametrize("yaw", [-1.2, -0.5, 0.5, 1.2])
def test_reach_with_yaw(model, ids, yaw: float) -> None:
    pos_err, rot_err = _reach(model, ids, (0.45, 0.1, 0.2), yaw=yaw)
    assert pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD, (yaw, pos_err, rot_err)


def test_set_target_rejects_yaws_outside_the_band_or_turns_past_pi(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    for yaw in (O._YAW_LIMIT + 1e-6, -2.0, 3.0):
        with pytest.raises(ValueError, match="within"):
            osc.set_target((0.5, 0.0, 0.3), yaw=yaw)
    osc.set_target((0.5, 0.0, 0.3), rot=O.top_down_rot(2.0))  # rot= skips the yaw checks


def test_turn_past_pi_after_a_twist_is_rejected_and_wrap_near_avoids_it(model, ids) -> None:
    """Regression (T03 review): an external twist left the hand at yaw −1.61; the next
    target +1.56 was 3.17 rad away, the reference turned the short way through yaw −π,
    and joint 7 stalled at its limit (34 mm / 26° off after 4 s, saturated False)."""
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    target = (0.5, 0.0, 0.2)
    osc.set_target(target, yaw=-math.pi / 2)
    _run(model, data, [osc], 2.0)
    data.xfrc_applied[ids.hand_body, 3:] = (0.0, 0.0, -2.4)  # twist about −z
    _run(model, data, [osc], 1.0)
    osc.hold_current()
    data.xfrc_applied[:] = 0.0
    assert osc.ee_yaw() < -math.pi / 2 - 0.03  # outside [−π/2, π/2)
    naive = O.wrap_grasp_yaw(math.pi / 2 - 0.01)
    with pytest.raises(ValueError, match="±π"):
        osc.set_target(target, yaw=naive)
    yaw = O.wrap_grasp_yaw(math.pi / 2 - 0.01, near=osc.ee_yaw())
    assert yaw == pytest.approx(naive - math.pi)  # the same grasp, a 0.03 rad turn
    osc.set_target(target, yaw=yaw)
    _run(model, data, [osc], 1.0)
    pos_err, rot_err = _ee_errors(data, ids, target, O.top_down_rot(yaw))
    assert pos_err < 1e-3 and rot_err < math.radians(0.5)


def test_wrap_grasp_yaw_near_picks_the_closest_equivalent_in_the_band() -> None:
    half = math.pi / 2
    for yaw, near, expected in (
        (half - 0.01, -half - 0.04, -half - 0.01),  # the far edge's twin is closer
        (half - 0.01, 0.0, half - 0.01),
        (0.0, -half - 0.04, 0.0),  # ±π are outside the band
        (half + 0.02, half + 0.04, half + 0.02),  # stays above +π/2 inside the margin
        (7.0, 0.0, 7.0 - 2 * math.pi),
    ):
        got = O.wrap_grasp_yaw(yaw, near=near)
        assert got == pytest.approx(expected), (yaw, near, got)
        assert abs(got) <= O._YAW_LIMIT
        turns = (yaw - got) / math.pi
        assert turns == pytest.approx(round(turns), abs=1e-9)
    with pytest.raises(ValueError):
        O.wrap_grasp_yaw(0.0, near=math.nan)


def test_at_joint_limit_flags_a_stall_the_torque_limits_miss(model, ids) -> None:
    """Yaw 2.3 (via rot=) at (0.5, 0, 0.3) runs joint 7 into its range end: the hand
    stalls 6.6° short with no motor saturated (measured in T03)."""
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.5, 0.0, 0.3), rot=O.top_down_rot(2.3))
    flagged = 0
    for _ in range(round(3.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], 1)
        flagged += osc.at_joint_limit
    mujoco.mj_forward(model, data)
    assert flagged > 0.3 * O.CONTROL_HZ, flagged
    assert osc.at_joint_limit and not osc.saturated
    assert osc.errors()[1] > math.radians(5.0), "the hand must stall short of yaw 2.3"


def test_ee_yaw_reads_the_top_down_yaw(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    for yaw in (-1.2, 0.0, 0.7):
        osc.set_target((0.45, 0.1, 0.25), yaw=yaw)
        _run(model, data, [osc], 2.0)
        assert osc.ee_yaw() == pytest.approx(yaw, abs=math.radians(0.5))


def test_long_move_from_a_non_home_pose_does_not_overshoot(model, ids) -> None:
    """Regression (T03 review): per-axis apparent-mass damping underdamped the coupled
    modes and overshot this move by 12.7 mm (500 Hz) / 13.4 mm (100 Hz)."""
    start, goal = np.array([0.579, -0.019, 0.15]), np.array([0.389, -0.033, 0.237])
    for hz in (O.CONTROL_HZ, 100):
        data = _fresh(model)
        osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=hz)
        osc.set_target(start)
        _run(model, data, [osc], 3.0, hz)
        direction = (goal - data.site_xpos[ids.ee_site]) / np.linalg.norm(
            goal - data.site_xpos[ids.ee_site]
        )
        osc.set_target(goal)
        overshoot = 0.0
        substeps = O.substeps_for(model, hz)
        for _ in range(round(2.5 * hz)):
            O.control_step(model, data, [osc], substeps)
            overshoot = max(overshoot, float((data.site_xpos[ids.ee_site] - goal) @ direction))
        mujoco.mj_forward(model, data)
        assert overshoot < 5e-4, (hz, overshoot)
        assert np.linalg.norm(data.site_xpos[ids.ee_site] - goal) < 1e-3


def test_wrap_grasp_yaw_uses_the_half_turn_symmetry() -> None:
    for yaw, expected in ((0.0, 0.0), (1.0, 1.0), (2.5, 2.5 - math.pi), (-2.0, math.pi - 2.0)):
        assert O.wrap_grasp_yaw(yaw) == pytest.approx(expected)
    for yaw in np.linspace(-10.0, 10.0, 41):
        wrapped = O.wrap_grasp_yaw(float(yaw))
        assert -math.pi / 2 <= wrapped < math.pi / 2
        # Same grasp: the two yaws differ by a whole number of half turns.
        turns = (float(yaw) - wrapped) / math.pi
        assert turns == pytest.approx(round(turns), abs=1e-9)
    with pytest.raises(ValueError):
        O.wrap_grasp_yaw(math.nan)


def _push_deflection(model, ids, stiffness, push, seconds=2.0) -> tuple[np.ndarray, float]:
    """Steady ee displacement [m] and rotation [rad] under a pure force at the ee point."""
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, O.OSCGains(pos_stiffness=stiffness), ids=ids)
    osc.set_target((0.5, 0.0, 0.3))
    _run(model, data, [osc], 3.0)
    before = data.site_xpos[ids.ee_site].copy()
    rot_before = data.site_xmat[ids.ee_site].reshape(3, 3).copy()
    push = np.asarray(push, dtype=float)
    substeps = O.substeps_for(model, O.CONTROL_HZ)
    for _ in range(round(seconds * O.CONTROL_HZ)):
        # xfrc_applied acts at the hand CoM; add the moment that moves it to the ee point.
        arm = data.site_xpos[ids.ee_site] - data.xipos[ids.hand_body]
        data.xfrc_applied[ids.hand_body] = np.concatenate((push, np.cross(arm, push)))
        O.control_step(model, data, [osc], substeps)
    mujoco.mj_forward(model, data)
    rot = O.rotation_angle(data.site_xmat[ids.ee_site].reshape(3, 3), rot_before)
    return data.site_xpos[ids.ee_site] - before, rot


def test_lower_stiffness_deflects_more_under_the_same_push(model, ids) -> None:
    push = (0.0, 10.0, 0.0)  # N, along world +y
    stiff, _ = _push_deflection(model, ids, 1500.0, push)
    soft, _ = _push_deflection(model, ids, 300.0, push)
    assert soft[1] > 3.0 * stiff[1] > 0.0, (stiff, soft)
    # Impedance semantics: the static deflection is F / K.
    assert stiff[1] == pytest.approx(10.0 / 1500.0, rel=0.03)
    assert soft[1] == pytest.approx(10.0 / 300.0, rel=0.03)


@pytest.mark.parametrize(("stiffness", "push"), [(1500.0, 70.0), (300.0, 30.0)])
def test_large_pushes_still_deflect_by_f_over_k(model, ids, stiffness, push) -> None:
    """The speed limit acts on the reference, not on the spring, so no force cap.

    Regression: an earlier spring-saturating speed limit capped the restoring
    force at v_max·D (≈ 67 N at 1500 N/m, ≈ 30 N at 300 N/m) and let the ee drift.
    """
    deflection, _ = _push_deflection(model, ids, stiffness, (0.0, push, 0.0), seconds=3.0)
    assert deflection[1] == pytest.approx(push / stiffness, rel=0.02)


def test_per_axis_stiffness_and_pure_force_gives_no_rotation(model, ids) -> None:
    push = (6.0, 6.0, -6.0)
    k = np.array([2000.0, 500.0, 1000.0])
    deflection, rot = _push_deflection(model, ids, tuple(k), push)
    np.testing.assert_allclose(deflection, np.array(push) / k, rtol=0.03, atol=2e-4)
    assert rot < math.radians(0.5)


def test_lift_episode_carries_the_cube() -> None:
    """The timed episode is a real grasp-and-lift (contacts and all), not an idle loop.

    It is not a tuned grasp: the cube creeps in the 1 N grip (refer to throughput.py).
    """
    run = throughput.run_lift_episode(O.CONTROL_HZ)
    assert run.sim_s == pytest.approx(10.0)
    assert run.lift_m > 0.08
    assert run.min_finger_force_n == pytest.approx(throughput.SQUEEZE_N, rel=0.05)


@pytest.mark.perf
def test_throughput_10s_lift_episode_under_1s_wall(capsys) -> None:
    runs = [throughput.run_lift_episode(O.CONTROL_HZ) for _ in range(3)]
    best = min(runs, key=lambda r: r.wall_s)
    with capsys.disabled():
        print(
            f"\n[T03 throughput] {best.sim_s:.1f} s sim at {best.control_hz} Hz: "
            f"wall {best.wall_s:.3f} s, cpu {best.cpu_s:.3f} s, "
            f"{best.realtime_factor:.1f}x real time (runs: "
            + ", ".join(f"{r.wall_s:.3f}" for r in runs)
            + " s)"
        )
    assert best.sim_s == pytest.approx(10.0)
    assert best.wall_s <= 1.0
    # One core: CPU time cannot exceed wall time by more than a sliver.
    assert best.cpu_s <= 1.1 * best.wall_s
    assert best.lift_m > 0.08  # the same grasp-and-lift as test_lift_episode_carries_the_cube


# --------------------------------------------------------------- torque law


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """Symmetric square root of a symmetric positive semi-definite matrix."""
    w, v = np.linalg.eigh(a)
    return (v * np.sqrt(np.maximum(w, 0.0))) @ v.T


def _reference_tau(model, data, ids, osc: O.OperationalSpaceController) -> np.ndarray:
    """The ARCHITECTURE §3 law written out directly, toward ``osc.reference()``."""
    ref = osc.reference()
    arm = ids.arm_dof
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ids.ee_site)
    jac = np.vstack((jacp[:, arm], jacr[:, arm]))
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    mass = full[np.ix_(arm, arm)]
    mass_inv = np.linalg.inv(mass)
    lam_inv = jac @ mass_inv @ jac.T
    lam = np.linalg.inv(lam_inv)
    g = osc.gains
    k = np.array([*g.pos_stiffness, *g.rot_stiffness])
    # Modal damping in the symmetric-root form D = 2ζ·Λ½·(Λ^-½·K·Λ^-½)½·Λ½
    # (a different factorization from the controller's K½·V·(2ζ/Ω)·Vᵀ·K½).
    lam_half = _sqrtm_psd(lam)
    lam_half_inv = np.linalg.inv(lam_half)
    d = 2.0 * g.damping_ratio * lam_half @ _sqrtm_psd(lam_half_inv @ np.diag(k) @ lam_half_inv)
    d = d @ lam_half
    for block, fixed in ((slice(0, 3), g.pos_damping), (slice(3, 6), g.rot_damping)):
        if fixed is not None:  # explicit damping: a diagonal block, no coupling
            d[block, :] = 0.0
            d[:, block] = 0.0
            d[block, block] = np.diag(fixed)
    # Orientation error from rotation matrices: R_err = R_t·Rᵀ → axis·angle.
    r_err = ref.rot @ data.site_xmat[ids.ee_site].reshape(3, 3).T
    angle = math.acos(min(1.0, max(-1.0, 0.5 * (np.trace(r_err) - 1.0))))
    axis = np.array(
        [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
    )
    rot_err = axis * (angle / (2.0 * math.sin(angle))) if angle > 1e-9 else 0.5 * axis
    err = np.concatenate((ref.pos - data.site_xpos[ids.ee_site], rot_err))
    qvel = data.qvel[arm]
    wrench = k * err + d @ (ref.vel - jac @ qvel)
    posture = g.null_stiffness * (osc.posture - data.qpos[ids.arm_qpos]) - g.null_damping * qvel
    null_t = np.eye(7) - jac.T @ lam @ jac @ mass_inv
    tau = jac.T @ wrench + data.qfrc_bias[arm] + null_t @ posture
    limit = np.array(S.ARM_TORQUE_LIMITS)
    return np.clip(tau, -limit, limit)


@pytest.mark.parametrize(
    "gains",
    [
        O.OSCGains(),
        O.OSCGains(pos_stiffness=(800.0, 1200.0, 400.0), rot_stiffness=30.0, damping_ratio=0.7),
        O.OSCGains(pos_damping=40.0, rot_damping=(2.0, 3.0, 4.0), null_stiffness=0.0),
    ],
)
@pytest.mark.parametrize("unlimited", [True, False], ids=["ref=target", "ref=rate-limited"])
def test_compute_matches_the_reference_law(model, ids, gains, unlimited: bool) -> None:
    rng = np.random.default_rng(11)
    data = _fresh(model)
    if unlimited:  # the reference jumps onto the target: the law acts on the target itself
        gains = dataclasses.replace(gains, max_lin_speed=math.inf, max_ang_speed=math.inf)
    osc = O.OperationalSpaceController(model, data, gains, ids=ids)
    for _ in range(5):
        data.qpos[ids.arm_qpos] = np.array(S.HOME_ARM_QPOS) + rng.normal(0.0, 0.15, 7)
        data.qvel[ids.arm_dof] = rng.normal(0.0, 0.3, 7)
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ids.ee_site]
        rot = data.site_xmat[ids.ee_site].reshape(3, 3)
        tilt = O.top_down_rot(float(rng.normal(0.0, 0.05))) @ O.TOP_DOWN_ROT.T
        osc.set_target(ee + rng.normal(0.0, 0.01, 3), rot=tilt @ rot, vel=rng.normal(0.0, 0.02, 6))
        osc.advance_reference()
        if unlimited:
            ref = osc.reference()
            np.testing.assert_array_equal(ref.pos, osc.target.pos)
            np.testing.assert_array_equal(ref.vel, osc.target.vel)
        # The controller's 1e-9 Tikhonov term on Λ⁻¹ changes τ by < 1e-7 (relative).
        reference = _reference_tau(model, data, ids, osc)
        np.testing.assert_allclose(osc.compute(), reference, rtol=1e-6, atol=1e-6)


def test_fresh_controller_holds_its_pose_and_returns_after_a_push(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    ee0 = data.site_xpos[ids.ee_site].copy()
    _run(model, data, [osc], 2.0)
    assert np.linalg.norm(data.site_xpos[ids.ee_site] - ee0) < 1e-4
    data.xfrc_applied[ids.hand_body, :3] = (0.0, 20.0, -20.0)
    _run(model, data, [osc], 0.5)
    pushed = float(np.linalg.norm(data.site_xpos[ids.ee_site] - ee0))
    data.xfrc_applied[:] = 0.0
    _run(model, data, [osc], 2.0)
    assert pushed > 0.01, "the push must move the ee"
    assert np.linalg.norm(data.site_xpos[ids.ee_site] - ee0) < 5e-4, "the target must stay put"


def test_hold_current_resets_target_and_reference(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.4, 0.2, 0.2), yaw=0.5)
    _run(model, data, [osc], 0.3)  # halfway there
    osc.hold_current()
    ee = data.site_xpos[ids.ee_site].copy()
    for snapshot in (osc.target, osc.reference()):
        np.testing.assert_array_equal(snapshot.pos, ee)
        np.testing.assert_array_equal(snapshot.vel, np.zeros(6))
        assert O.rotation_angle(snapshot.rot, data.site_xmat[ids.ee_site].reshape(3, 3)) < 1e-6
    data.qpos[ids.arm_qpos[0]] += 0.05  # move the arm: the stored target must not follow
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(osc.target.pos, ee)


def test_null_space_posture_moves_the_elbow_not_the_ee(model, ids) -> None:
    data = _fresh(model)
    target = np.array([0.5, 0.0, 0.3])
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target(target)
    _run(model, data, [osc], 2.0)
    q_before = data.qpos[ids.arm_qpos].copy()
    elbow_out = np.array(S.HOME_ARM_QPOS) + np.array([0.6, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0])
    osc = O.OperationalSpaceController(model, data, ids=ids, posture=elbow_out)
    osc.set_target(target)
    worst = 0.0
    substeps = O.substeps_for(model, O.CONTROL_HZ)
    for _ in range(round(3.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], substeps)
        worst = max(worst, float(np.linalg.norm(data.site_xpos[ids.ee_site] - target)))
    mujoco.mj_forward(model, data)
    moved = np.abs(data.qpos[ids.arm_qpos] - q_before)
    assert moved.max() > 0.1, "the posture term must move the redundant DOF"
    assert worst < 0.002, f"the posture term leaked {worst * 1000:.2f} mm into the ee"
    pos_err, rot_err = _ee_errors(data, ids, target, O.top_down_rot())
    assert pos_err < 5e-4 and rot_err < math.radians(0.2)


def test_singular_start_pose_keeps_torques_finite_and_bounded(model, ids) -> None:
    """Arm straight up (q = 0): joints 1, 3, 5, 7 align and Λ⁻¹ is singular.

    From near-singular poses the controller can stall (documented in
    ARCHITECTURE §3); it must never produce NaN or out-of-limit torques.
    Without the Tikhonov term, inverting Λ⁻¹ fails here.
    """
    data = _fresh(model)
    data.qpos[ids.arm_qpos] = 0.0
    mujoco.mj_forward(model, data)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ids.ee_site)
    jac = np.vstack((jacp[:, ids.arm_dof], jacr[:, ids.arm_dof]))
    assert np.linalg.matrix_rank(jac, tol=1e-9) < 6, "q = 0 must be a kinematic singularity"
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.5, 0.0, 0.3))
    limits = np.array(S.ARM_TORQUE_LIMITS)
    for _ in range(round(1.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], 1)
        tau = data.ctrl[ids.arm_actuators]
        assert np.all(np.isfinite(tau)) and np.all(np.abs(tau) <= limits + 1e-12)
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_non_finite_torques_raise_instead_of_reaching_the_motors(model, ids) -> None:
    data = _fresh(model)
    gains = O.OSCGains(max_lin_speed=math.inf)
    osc = O.OperationalSpaceController(model, data, gains, ids=ids)
    osc.set_target((1e308, 0.0, 0.3))  # K·e overflows to inf
    ctrl, before = data.ctrl.copy(), osc.reference()
    with pytest.raises(FloatingPointError):
        osc.apply()
    np.testing.assert_array_equal(data.ctrl, ctrl)
    after = osc.reference()
    np.testing.assert_array_equal(after.pos, before.pos)  # a failed tick rolls back
    np.testing.assert_array_equal(after.vel, before.vel)
    osc = O.OperationalSpaceController(model, _fresh(model), ids=ids)
    osc.data.qvel[ids.arm_dof[2]] = math.nan
    with pytest.raises(FloatingPointError):
        osc.compute()
    # A non-finite qpos breaks the dynamics (eigh raises LinAlgError): still a
    # FloatingPointError, and apply() still rolls the reference back.
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.4, 0.1, 0.3))
    data.qpos[ids.arm_qpos[1]] = math.nan
    mujoco.mj_forward(model, data)
    ctrl, before = data.ctrl.copy(), osc.reference()
    with pytest.raises(FloatingPointError):
        osc.apply()
    np.testing.assert_array_equal(data.ctrl, ctrl)
    np.testing.assert_array_equal(osc.reference().pos, before.pos)


def test_torques_are_clipped_to_the_motor_limits(model, ids) -> None:
    data = _fresh(model)
    gains = O.OSCGains(pos_stiffness=1e5, rot_stiffness=1e4, max_lin_speed=50.0, max_ang_speed=50.0)
    osc = O.OperationalSpaceController(model, data, gains, ids=ids)
    osc.set_target((0.3, 0.3, 0.05), yaw=1.0)
    mujoco.mj_forward(model, data)
    tau = osc.apply()
    assert osc.saturated
    limits = np.array(S.ARM_TORQUE_LIMITS)
    assert np.all(np.abs(tau) <= limits + 1e-12)
    assert np.any(np.isclose(np.abs(tau), limits))
    np.testing.assert_array_equal(data.ctrl[ids.arm_actuators], tau)


@pytest.mark.parametrize("max_speed", [0.25, 0.5])
def test_speed_limit_bounds_the_reference_and_the_ee(model, ids, max_speed: float) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, O.OSCGains(max_lin_speed=max_speed), ids=ids)
    target = np.array([0.3, 0.3, 0.05])
    osc.set_target(target)
    jacp = np.zeros((3, model.nv))
    fastest = ref_fastest = 0.0
    previous = osc.reference().pos
    distance = float(np.linalg.norm(target - previous))
    for _ in range(round((distance / max_speed + 0.5) * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], 1)
        snapshot = osc.reference()
        ref = snapshot.pos
        if not np.array_equal(ref, target):  # chasing: no feed-forward velocity
            np.testing.assert_array_equal(snapshot.vel, np.zeros(6))
        ref_fastest = max(ref_fastest, float(np.linalg.norm(ref - previous)) * O.CONTROL_HZ)
        previous = ref
        mujoco.mj_jacSite(model, data, jacp, None, ids.ee_site)
        fastest = max(fastest, float(np.linalg.norm(jacp @ data.qvel)))
    assert ref_fastest == pytest.approx(max_speed, rel=1e-9)  # moves at the limit, never over
    np.testing.assert_array_equal(osc.reference().pos, target)  # and lands exactly
    assert 0.8 * max_speed < fastest <= 1.2 * max_speed  # the ee tracks it within 20 %


def test_angular_speed_limit_bounds_the_reference_turn(model, ids) -> None:
    data = _fresh(model)
    gains = O.OSCGains(max_ang_speed=0.5)
    osc = O.OperationalSpaceController(model, data, gains, ids=ids)
    osc.set_target(data.site_xpos[ids.ee_site], yaw=1.2)
    jacr = np.zeros((3, model.nv))
    fastest = ref_fastest = 0.0
    previous = osc.reference().rot
    for _ in range(round(3.5 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc], 1)
        snapshot = osc.reference()
        ref = snapshot.rot
        if O.rotation_angle(ref, osc.target.rot) > 0.0:  # chasing: no feed-forward velocity
            np.testing.assert_array_equal(snapshot.vel[3:], np.zeros(3))
        ref_fastest = max(ref_fastest, O.rotation_angle(ref, previous) * O.CONTROL_HZ)
        previous = ref
        mujoco.mj_jacSite(model, data, None, jacr, ids.ee_site)
        fastest = max(fastest, float(np.linalg.norm(jacr @ data.qvel)))
    assert ref_fastest == pytest.approx(0.5, rel=1e-6)
    assert 0.8 * 0.5 < fastest <= 1.2 * 0.5
    assert osc.errors()[1] < math.radians(0.5)


def test_reference_follows_a_slow_target_exactly_with_its_velocity(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    start = data.site_xpos[ids.ee_site].copy()
    step = 0.2 / O.CONTROL_HZ  # 0.2 m/s, under the 0.5 m/s limit
    for k in range(1, 50):
        pos = start + (k * step, 0.0, 0.0)
        osc.set_target(pos, vel=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
        O.control_step(model, data, [osc], 1)
        ref = osc.reference()
        np.testing.assert_array_equal(ref.pos, pos)
        np.testing.assert_allclose(ref.vel, (0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
    # A feed-forward velocity above the limit is clipped to it.
    osc.set_target(osc.target.pos, vel=(3.0, 4.0, 0.0, 0.0, 0.0, 6.0))
    osc.advance_reference()
    np.testing.assert_allclose(osc.reference().vel, (0.3, 0.4, 0.0, 0.0, 0.0, 1.5))


def test_feedforward_velocity_reduces_tracking_lag(model, ids) -> None:
    """A target moving on a circle: passing its velocity cuts the lag."""
    center, radius, omega = np.array([0.5, 0.0, 0.3]), 0.08, 2.0 * math.pi * 0.5

    def lag(use_ff: bool) -> float:
        data = _fresh(model)
        osc = O.OperationalSpaceController(model, data, ids=ids)
        osc.set_target(center + (radius, 0.0, 0.0))
        _run(model, data, [osc], 1.5)
        errs = []
        for k in range(round(2.0 * O.CONTROL_HZ)):
            t = k / O.CONTROL_HZ
            pos = center + radius * np.array([math.cos(omega * t), math.sin(omega * t), 0.0])
            vel = radius * omega * np.array([-math.sin(omega * t), math.cos(omega * t), 0.0])
            osc.set_target(pos, vel=(*vel, 0.0, 0.0, 0.0) if use_ff else None)
            O.control_step(model, data, [osc], 1)
            errs.append(float(np.linalg.norm(osc.target.pos - data.site_xpos[ids.ee_site])))
        return float(np.mean(errs[len(errs) // 2 :]))

    with_ff, without_ff = lag(True), lag(False)
    assert with_ff < 0.5 * without_ff, (with_ff, without_ff)


def test_pose_error_is_world_axis_angle_with_shortest_arc(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    current = data.site_xmat[ids.ee_site].reshape(3, 3)
    for angle in (0.3, -1.0, 3.0):
        rz = O.top_down_rot(angle) @ O.TOP_DOWN_ROT.T  # world-z rotation by angle
        osc.set_target(data.site_xpos[ids.ee_site], rot=rz @ current)
        pos_err, rot_err = osc.pose_error()
        np.testing.assert_allclose(pos_err, 0.0, atol=1e-12)
        np.testing.assert_allclose(rot_err, (0.0, 0.0, angle), atol=1e-9)
    # A 3.5 rad turn is reached the short way round: −(2π − 3.5).
    rz = O.top_down_rot(3.5) @ O.TOP_DOWN_ROT.T
    osc.set_target(data.site_xpos[ids.ee_site], rot=rz @ current)
    np.testing.assert_allclose(osc.pose_error()[1], (0.0, 0.0, 3.5 - 2 * math.pi), atol=1e-9)
    assert osc.errors()[1] == pytest.approx(2 * math.pi - 3.5)


def test_top_down_rot_matches_home_and_points_down(model, ids) -> None:
    data = _fresh(model)
    home = data.site_xmat[ids.ee_site].reshape(3, 3)
    assert O.rotation_angle(home, O.top_down_rot()) < math.radians(0.01)
    for yaw in (0.0, 0.7, -2.0):
        rot = O.top_down_rot(yaw)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rot[:, 2], (0.0, 0.0, -1.0), atol=1e-12)  # hand z down
        assert O.rotation_angle(rot, O.top_down_rot()) == pytest.approx(abs(yaw), abs=1e-9)
    # yaw 0: the fingers slide along world x (finger joint axis = hand y).
    np.testing.assert_allclose(O.top_down_rot()[:, 1], (1.0, 0.0, 0.0), atol=1e-12)


def test_set_gains_takes_effect(model, ids) -> None:
    data = _fresh(model)
    unlimited = {"max_lin_speed": math.inf, "max_ang_speed": math.inf}
    osc = O.OperationalSpaceController(model, data, O.OSCGains(**unlimited), ids=ids)
    osc.set_target(data.site_xpos[ids.ee_site] + (0.0, 0.004, 0.0))
    osc.advance_reference()
    mujoco.mj_forward(model, data)
    stiff = osc.compute()
    osc.set_gains(O.OSCGains(pos_stiffness=150.0, **unlimited))
    soft = osc.compute()
    bias = data.qfrc_bias[ids.arm_dof]
    assert np.linalg.norm(soft - bias) < 0.2 * np.linalg.norm(stiff - bias)
    with pytest.raises(TypeError):
        osc.set_gains({"pos_stiffness": 1.0})  # type: ignore[arg-type]


# ------------------------------------------------------------- control loop


def test_substeps_for_the_physics_rate(model) -> None:
    assert O.substeps_for(model, 500) == 1
    assert O.substeps_for(model, 250) == 2
    assert O.substeps_for(model, 100) == 5
    assert O.substeps_for(model, 125.0) == 4
    for bad in (300, 499, 1000, 0, -100, math.nan, math.inf, True, "500", 5e-324, 1e-10, 0.4):
        with pytest.raises(ValueError):
            O.substeps_for(model, bad)
    # Below the floor (ARCHITECTURE §2): rates that divide 500 Hz but lose the reach.
    for slow in (50, 25, 10, 0.5, O.MIN_CONTROL_HZ - 1e-9):
        with pytest.raises(ValueError, match="floor"):
            O.substeps_for(model, slow)
    fine = S.build_scene()
    fine.opt.timestep = 1e-6  # 100 Hz would hold each torque for 10000 steps
    with pytest.raises(ValueError, match="substeps"):
        O.substeps_for(fine, 100)


def test_control_step_holds_commands_across_substeps(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=100)
    calls = []

    class Spy:
        def apply(self) -> None:
            calls.append(float(data.time))

    t0 = data.time
    O.control_step(model, data, [osc, Spy()], substeps=5)
    assert data.time == pytest.approx(t0 + 5 * model.opt.timestep)
    assert calls == [t0], "controllers run once per tick, before the physics steps"
    # A controller built for one rate cannot be ticked at another.
    t_before = data.time
    with pytest.raises(ValueError, match="100 Hz"):
        O.control_step(model, data, [osc], substeps=np.int64(2))  # a 250 Hz tick
    assert data.time == t_before
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=250)
    O.control_step(model, data, [osc], substeps=np.int64(2))  # numpy ints are fine
    t1 = data.time
    for bad in (0, -1, 2.0, True, np.True_, O.MAX_SUBSTEPS + 1, 2**64):
        with pytest.raises(ValueError):
            O.control_step(model, data, [osc], substeps=bad)
        assert data.time == t1, "a rejected tick must not step the physics"


def test_control_step_accepts_a_one_shot_iterable_and_rejects_duplicates(model, ids) -> None:
    """Regression: the rate check used up a generator, so no controller ran and the arm sagged."""
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    ee0 = data.site_xpos[ids.ee_site].copy()
    for _ in range(round(0.5 * O.CONTROL_HZ)):
        O.control_step(model, data, (c for c in (osc, gripper)), 1)
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.site_xpos[ids.ee_site] - ee0) < 1e-4
    assert np.any(data.ctrl[ids.arm_actuators] != 0.0)
    t0, ref = data.time, osc.reference()
    with pytest.raises(ValueError, match="twice"):
        O.control_step(model, data, [osc, osc], 1)  # would advance the reference twice
    assert data.time == t0
    np.testing.assert_array_equal(osc.reference().pos, ref.pos)


def test_rate_and_gains_are_read_only(model, ids) -> None:
    """The reference step is sized for control_hz; gains change only through set_gains."""
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=250)
    with pytest.raises(AttributeError):
        osc.control_hz = 100  # type: ignore[misc]
    with pytest.raises(AttributeError):
        osc.gains = O.OSCGains(pos_stiffness=100.0)  # type: ignore[misc]
    assert osc.control_hz == 250.0 and osc.gains == O.OSCGains()


def test_controller_and_gripper_share_a_control_tick(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    gripper.close(0.03)
    _run(model, data, [osc, gripper], 1.0)
    assert gripper.width() == pytest.approx(0.03, abs=1e-3)
    assert osc.errors()[0] < 1e-3


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pos_stiffness": 0.0},
        {"pos_stiffness": -1.0},
        {"pos_stiffness": (1.0, 2.0)},
        {"pos_stiffness": "1500"},
        {"rot_stiffness": math.nan},
        {"rot_stiffness": (1.0, 2.0, math.inf)},
        {"damping_ratio": -0.1},
        {"pos_damping": -1.0},
        {"rot_damping": (1.0, -1.0, 1.0)},
        {"null_stiffness": -1.0},
        {"null_damping": True},
        {"max_lin_speed": 0.0},
        {"max_ang_speed": -1.0},
        {"max_ang_speed": -math.inf},
        {"max_lin_speed": math.nan},
        {"max_lin_speed": True},
    ],
)
def test_gains_reject_bad_values(kwargs) -> None:
    with pytest.raises(ValueError):
        O.OSCGains(**kwargs)


def test_gains_are_frozen_and_normalized() -> None:
    g = O.OSCGains(pos_stiffness=np.float32(800.0), rot_damping=np.array([1, 2, 3]))
    assert g.pos_stiffness == (800.0, 800.0, 800.0)
    assert g.rot_damping == (1.0, 2.0, 3.0)
    assert all(type(v) is float for v in g.pos_stiffness + g.rot_damping)
    with pytest.raises(AttributeError):
        g.pos_stiffness = (1.0, 1.0, 1.0)  # type: ignore[misc]


def test_set_target_rejects_bad_input(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    before = osc.target
    bad_calls = [
        {"pos": (0.5, 0.0)},
        {"pos": (0.5, 0.0, math.nan)},
        {"pos": "abc"},
        {"pos": (0.5, 0.0, 0.3), "yaw": math.inf},
        {"pos": (0.5, 0.0, 0.3), "vel": (0.0,) * 5},
        {"pos": (0.5, 0.0, 0.3), "rot": np.eye(3) * 2.0},
        {"pos": (0.5, 0.0, 0.3), "rot": np.diag([1.0, 1.0, -1.0])},  # reflection
        {"pos": (0.5, 0.0, 0.3), "rot": np.eye(2)},
        {"pos": (0.5, 0.0, 0.3), "rot": np.eye(3), "yaw": 0.3},
        {"pos": (0.5, 0.0, 0.3), "rot": np.eye(3) + 0.5j},
        {"pos": (0.5, 0.0, 0.3), "rot": np.eye(3, dtype=bool)},
        {"pos": (0.5, 0.0, 0.3), "rot": [[1, 0, 0], [0, 1, 0], [0, 0, "1"]]},
    ]
    for kwargs in bad_calls:
        with pytest.raises(ValueError):
            osc.set_target(**kwargs)
        assert osc.target is before, f"a rejected target must not change state: {kwargs}"


def test_target_is_read_only(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    osc.set_target((0.5, 0.0, 0.3), yaw=0.4)
    with pytest.raises(AttributeError):
        osc.target = osc.target  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        osc.target.rot = np.eye(3)  # type: ignore[misc]
    for array in (osc.target.pos, osc.target.rot, osc.target.vel):
        with pytest.raises(ValueError):
            array[0] = 1.0
    # The caller's arrays stay writable: the controller froze its own copies.
    pos, rot = np.array([0.5, 0.0, 0.3]), O.top_down_rot(0.2)
    osc.set_target(pos, rot=rot)
    pos[0], rot[0, 0] = 9.0, 9.0
    assert osc.target.pos[0] == 0.5 and osc.target.rot[0, 0] != 9.0


def test_rotation_angle_rejects_non_matrices() -> None:
    assert O.rotation_angle(np.eye(3), O.top_down_rot()) == pytest.approx(math.pi)
    for bad in (np.full((3, 3), math.nan), np.eye(2), np.eye(3) + 1j, "abc"):
        with pytest.raises(ValueError):
            O.rotation_angle(bad, np.eye(3))


def test_controller_needs_forwarded_data_and_a_valid_posture(model, ids) -> None:
    with pytest.raises(ValueError, match="mj_forward"):
        O.OperationalSpaceController(model, mujoco.MjData(model), ids=ids)
    data = _fresh(model)
    for hz in (300, 0, math.nan, 50):
        with pytest.raises(ValueError):
            O.OperationalSpaceController(model, data, ids=ids, control_hz=hz)
    for posture in ((0.0,) * 6, (0.0,) * 6 + (math.nan,)):
        with pytest.raises(ValueError):
            O.OperationalSpaceController(model, data, ids=ids, posture=posture)


def test_control_step_gives_controllers_the_current_kinematics(model, ids) -> None:
    """mj_step1 runs before apply(): kinematics and qfrc_bias match the current qpos/qvel."""
    data = _fresh(model)
    data.qvel[ids.arm_dof] = 0.3
    probe = mujoco.MjData(model)
    seen = []

    class Spy:
        def apply(self) -> None:
            probe.qpos[:], probe.qvel[:] = data.qpos, data.qvel
            mujoco.mj_forward(model, probe)
            seen.append(
                np.allclose(data.site_xpos[ids.ee_site], probe.site_xpos[ids.ee_site], atol=1e-12)
                and np.allclose(data.qfrc_bias, probe.qfrc_bias, atol=1e-9)
            )

    for _ in range(3):
        O.control_step(model, data, [Spy()], substeps=1)
    assert seen == [True, True, True]


def test_reaching_the_box_floor_neither_overshoots_nor_touches_the_table(model, ids) -> None:
    """Regression: with the chase velocity fed forward, long moves overshot ≈ 1 cm and the
    fingertips (≈ 9.8 mm below the ee) hit the table at z = 0.02."""
    fingers = ids.left_finger_geoms | ids.right_finger_geoms
    for target in ((0.65, 0.3, 0.02), (0.3, -0.3, 0.02), (0.5, 0.0, 0.02)):
        target = np.array(target)
        data = _fresh(model)
        osc = O.OperationalSpaceController(model, data, ids=ids)
        osc.set_target(target)
        start = data.site_xpos[ids.ee_site].copy()
        direction = (target - start) / np.linalg.norm(target - start)
        overshoot = 0.0
        for _ in range(round(REACH_SECONDS * O.CONTROL_HZ)):
            O.control_step(model, data, [osc], 1)
            overshoot = max(overshoot, float((data.site_xpos[ids.ee_site] - target) @ direction))
            for i in range(data.ncon):
                pair = {int(data.contact[i].geom1), int(data.contact[i].geom2)}
                assert not (ids.table_geom in pair and pair & fingers), "fingers hit the table"
        assert overshoot < 0.005, (target, overshoot)


def test_hold_current_rejects_non_finite_kinematics(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    data.site_xpos[ids.ee_site, 0] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        osc.hold_current()


def test_reference_snapshots_are_read_only(model, ids) -> None:
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    ref = osc.reference()
    for array in (ref.pos, ref.rot, ref.vel):
        with pytest.raises(ValueError):
            array[0] = 1.0


@pytest.mark.parametrize("inf", [math.inf, np.inf, np.float64(np.inf), np.array(np.inf)])
def test_speed_limits_accept_any_real_infinity(inf) -> None:
    assert O.OSCGains(max_lin_speed=inf, max_ang_speed=inf).max_lin_speed == math.inf


def test_wrap_grasp_yaw_stays_half_open_at_float_edges() -> None:
    for k in range(-6, 7):
        for nudge in (-1e-15, 0.0, 1e-15):
            wrapped = O.wrap_grasp_yaw(-math.pi / 2 + k * math.pi + nudge)
            assert -math.pi / 2 <= wrapped < math.pi / 2, (k, nudge, wrapped)
    assert O.wrap_grasp_yaw(-1.5707963267948968) < math.pi / 2


# --------------------------------------------------------------------- CLI


def test_throughput_cli_prints_the_table(capsys) -> None:
    assert throughput.main(["--hz", "500", "--seconds", "1", "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "control_hz" in out and "x_realtime" in out
    row = next(line for line in out.splitlines() if line.strip().startswith("500"))
    assert float(row.split()[2]) > 0.0  # wall_s


@pytest.mark.parametrize(
    "argv",
    [
        ["--hz", "300"],
        ["--hz", "50"],
        ["--hz", *["500"] * 9],
        ["--seconds", "0"],
        ["--seconds", "nan"],
        ["--seconds", "60.5"],
        ["--repeats", "0"],
        ["--repeats", "11"],
    ],
)
def test_throughput_cli_rejects_bad_args(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        throughput.main(argv)
    assert exc.value.code == 2


def test_throughput_cli_accepts_its_upper_bounds(monkeypatch, capsys) -> None:
    calls = []

    def fake(hz, seconds, model):
        calls.append((hz, seconds))
        return throughput.ThroughputResult(float(hz), 1, seconds, 1.0, 1.0, 0.0, 0.0)

    monkeypatch.setattr(throughput, "run_lift_episode", fake)
    rates = ["500", "250", "125", "100"] * 2  # MAX_RATES = 8
    argv = ["--hz", *rates, "--repeats", "10", "--seconds", "60"]
    assert throughput.main(argv) == 0
    assert len(calls) == 8 * 10 and all(seconds == 60.0 for _, seconds in calls)
    assert "x_realtime" in capsys.readouterr().out


def test_run_lift_episode_reports_the_simulated_time_and_rejects_bad_input(model) -> None:
    short = throughput.run_lift_episode(100, 0.004, model)  # rounds up to one 10 ms tick
    assert short.sim_s == pytest.approx(0.01) and short.control_hz == 100.0
    for bad in (True, "1", 0.0, 61.0, math.inf):
        with pytest.raises(ValueError):
            throughput.run_lift_episode(500, bad, model)
    with pytest.raises(ValueError):
        throughput.run_lift_episode(300, 1.0, model)


# -------------------------------------------------------------------- slow


@pytest.mark.slow
@pytest.mark.parametrize("seed", [2024, 7])
def test_reach_sweep_chained_targets_with_random_yaw(model, ids, seed: int) -> None:
    """30 chained targets (each starts where the last ended) with yaw in ±π/2.

    A turn of θ takes θ / max_ang_speed (up to 2.1 s for ≈ π), so each move gets
    that on top of the 2 s reach. Seed 7 failed a flat 2 s budget (T03 review).
    """
    rng = np.random.default_rng(seed)
    data = _fresh(model)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    worst_pos = worst_rot = 0.0
    yaw = 0.0
    for _ in range(30):
        target = rng.uniform(O.WORKSPACE_LOW, O.WORKSPACE_HIGH)
        new_yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
        turn, yaw = abs(new_yaw - yaw), new_yaw
        osc.set_target(target, yaw=yaw)
        _run(model, data, [osc], REACH_SECONDS + turn / osc.gains.max_ang_speed)
        pos_err, rot_err = _ee_errors(data, ids, target, O.top_down_rot(yaw))
        worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
    assert worst_pos < POS_TOL_M and worst_rot < ROT_TOL_RAD, (worst_pos, worst_rot)
