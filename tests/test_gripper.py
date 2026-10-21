"""T03/T05: gripper interface (``open``, ``close(width)``, ``squeeze(force_n, speed)``, ``width``).

Pins the force mapping documented in ``vfa/control/gripper.py``: a squeeze
request lands on the cube as that normal force per finger, the menagerie servo
(``SceneConfig(gripper="servo")``) caps it at 25·width N, and the same class
drives the force motor that T05 made the scene default. T05 adds the closing
speed (``squeeze(f, speed=v)``) and the end stop.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from vfa.control import gripper as G
from vfa.control import osc as O
from vfa.control.gripper import Gripper, GripperCommand
from vfa.sim import scene as S

SERVO = S.SceneConfig(gripper="servo")


def _servo_model() -> mujoco.MjModel:
    """The scene with the pristine menagerie gripper servo (the T02/T03 default)."""
    return S.build_scene(SERVO)


def _model(kind: str) -> mujoco.MjModel:
    return S.build_scene(S.SceneConfig(gripper=kind))


def _motor_model(ctrl_limit: float = 100.0, gear: float = 1.0) -> mujoco.MjModel:
    """A force motor on the tendon with a custom ctrl range or gear (force range ±100 N)."""
    spec = S.build_spec(SERVO)
    act = spec.actuator(S.GRIPPER_ACTUATOR)
    act.set_to_motor()
    act.ctrlrange = [-ctrl_limit, ctrl_limit]
    act.forcerange = [-100.0, 100.0]
    act.gear[0] = gear
    return spec.compile()


def _floating_cube(model: mujoco.MjModel):
    """Gravity off, the cube floats between the open fingers, faces on the pads."""
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    model.opt.gravity[:] = 0.0
    q = ids.cube_qpos
    data.qpos[q : q + 3] = data.site_xpos[ids.ee_site]
    data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    return ids, data, osc, Gripper(model, data, ids)


def _settle(model, data, osc, gripper, seconds=0.8) -> None:
    for _ in range(round(seconds * O.CONTROL_HZ)):
        O.control_step(model, data, [osc, gripper], 1)
    mujoco.mj_forward(model, data)


KINDS = pytest.mark.parametrize("kind", ["servo", "motor"])


@KINDS
@pytest.mark.parametrize("force", [0.25, 0.5, 1.0, 1.2])
def test_squeeze_puts_the_requested_force_on_the_cube(kind: str, force: float) -> None:
    model = _model(kind)
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(force)
    _settle(model, data, osc, gripper)
    left, right = S.finger_cube_forces(model, data, ids)
    assert left == pytest.approx(force, rel=1e-3)
    assert right == pytest.approx(force, rel=1e-3)
    assert gripper.actuator_squeeze() == pytest.approx(force, rel=1e-3)
    assert not gripper.saturated
    assert gripper.command == GripperCommand("force", force, G.MAX_CLOSE_SPEED)  # the default
    assert gripper.width() == pytest.approx(0.05, abs=0.002)  # stopped by the 5 cm cube


def test_squeeze_saturates_at_the_servo_cap() -> None:
    """Menagerie servo: the ctrl floor caps the squeeze at 25·width N per finger."""
    model = _servo_model()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(5.0)
    _settle(model, data, osc, gripper)
    cap = gripper.max_squeeze_force()
    assert cap == pytest.approx(25.0 * gripper.width(), rel=1e-9)
    assert cap == pytest.approx(1.25, abs=0.01)
    assert gripper.saturated
    assert data.ctrl[ids.gripper_actuator] == 0.0
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(cap, rel=1e-3)


@KINDS
def test_close_on_the_cube_squeezes_by_the_servo_stiffness(kind: str) -> None:
    """The motor emulates the servo's 100 N/m, so both squeeze by 25·(w − w*)."""
    model = _model(kind)
    ids, data, osc, gripper = _floating_cube(model)
    gripper.close(0.04)  # 1 cm narrower than the cube
    _settle(model, data, osc, gripper)
    expected = 0.25 * gripper.servo_kp * (gripper.width() - 0.04)  # 25·(w − w*)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(expected, rel=1e-3)
    assert expected == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("motor", [False, True], ids=["servo", "motor"])
def test_close_and_open_track_the_width_without_an_object(motor: bool) -> None:
    model = S.build_scene() if motor else _servo_model()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.2, 0.6)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    assert gripper.max_width == pytest.approx(0.08)
    assert gripper.width() == pytest.approx(0.08)
    for width in (0.06, 0.02, 0.0):
        gripper.close(width)
        _settle(model, data, osc, gripper, 1.0)
        assert gripper.width() == pytest.approx(width, abs=1e-3), (motor, width)
    gripper.open()
    _settle(model, data, osc, gripper, 1.0)
    assert gripper.width() == pytest.approx(0.08, abs=1e-3)
    assert gripper.command == GripperCommand("position", 0.08)


def test_servo_position_command_is_the_native_menagerie_ctrl() -> None:
    model = _servo_model()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    for width in (0.0, 0.03, 0.08):
        gripper.close(width)
        # rel 1e-8: panda.xml rounds the gain 0.04·100/255 to 0.01568627451.
        assert data.ctrl[S.scene_ids(model).gripper_actuator] == pytest.approx(
            255.0 * width / 0.08, rel=1e-8, abs=1e-12
        )


@pytest.mark.parametrize("force", [2.0, 10.0, 20.0, 45.0])
def test_force_motor_reaches_forces_the_servo_cannot(force: float) -> None:
    model = S.build_scene()  # T05: the scene default
    ids, data, osc, gripper = _floating_cube(model)
    assert gripper.max_squeeze_force() == pytest.approx(50.0)  # 100 N tendon / 2 fingers
    gripper.squeeze(force)
    _settle(model, data, osc, gripper)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(force, rel=1e-3)
    assert not gripper.saturated


def test_force_motor_squeeze_saturates_at_the_force_range() -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(80.0)
    _settle(model, data, osc, gripper, 0.8)  # 0.06 s closing, then 125 N/s up to the cap
    assert gripper.saturated
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(50.0, rel=1e-3)


def test_max_squeeze_force_scales_with_width() -> None:
    model = _servo_model()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    for width in (0.0, 0.02, 0.05, 0.08):
        assert gripper.max_squeeze_force(width) == pytest.approx(25.0 * width, abs=1e-9)


def test_motor_max_squeeze_force_follows_the_end_stop() -> None:
    """Full 50 N per finger from 3 cm up; linear down to 0 at the 5 mm stop."""
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    lo = G.END_STOP_WIDTH
    full = G.END_STOP_WIDTH + G.END_STOP_BAND
    assert full == pytest.approx(0.03)
    for width, expected in ((0.0, 0.0), (lo, 0.0), (0.0175, 25.0), (full, 50.0), (0.08, 50.0)):
        assert gripper.max_squeeze_force(width) == pytest.approx(expected, abs=1e-9), width


def test_gripper_rejects_bad_commands_without_changing_state() -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    gripper.squeeze(0.5)
    before, ctrl = gripper.command, data.ctrl.copy()
    for call, arg in (
        (gripper.squeeze, -0.1),
        (gripper.squeeze, math.nan),
        (gripper.squeeze, math.inf),
        (gripper.squeeze, True),
        (gripper.squeeze, "1"),
        (gripper.close, -0.001),
        (gripper.close, 0.0801),
        (gripper.close, math.nan),
        (gripper.close, None),
    ):
        with pytest.raises(ValueError):
            call(arg)
        assert gripper.command == before
        np.testing.assert_array_equal(data.ctrl, ctrl)
    bad_speeds = (0.0, 5e-324, 0.5 * G.MIN_CLOSE_SPEED, -0.1, G.MAX_CLOSE_SPEED + 0.01)
    for speed in (*bad_speeds, math.nan, math.inf, True, "0.1"):
        with pytest.raises(ValueError):
            gripper.squeeze(1.0, speed=speed)  # type: ignore[arg-type]
        assert gripper.command == before
        np.testing.assert_array_equal(data.ctrl, ctrl)
    with pytest.raises(ValueError):
        gripper.max_squeeze_force(math.nan)


def test_gripper_rejects_an_unsupported_actuator() -> None:
    spec = S.build_spec()
    act = spec.actuator(S.GRIPPER_ACTUATOR)
    act.dyntype = mujoco.mjtDyn.mjDYN_FILTER
    act.dynprm[0] = 0.01
    model = spec.compile()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    with pytest.raises(ValueError, match="activation"):
        Gripper(model, data)


@pytest.mark.parametrize("motor", [False, True], ids=["servo", "motor"])
def test_apply_solves_the_actuator_line_while_the_fingers_move(motor: bool) -> None:
    """ctrl = (F* − b0 − b1·L − b2·L̇)/g with F* carrying the servo damping −kd·L̇."""
    model = S.build_scene() if motor else _servo_model()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    data.qpos[ids.finger_qpos] = 0.03
    data.qvel[[model.jnt_dofadr[model.joint(n).id] for n in S.FINGER_JOINTS]] = -0.05
    mujoco.mj_forward(model, data)
    gripper = Gripper(model, data, ids)
    length, speed = 0.03, -0.05  # tendon L = mean finger slide, L̇ likewise
    assert data.ten_length[0] == pytest.approx(length) and data.ten_velocity[0] == pytest.approx(
        speed
    )
    g = float(model.actuator_gainprm[ids.gripper_actuator, 0])
    b = model.actuator_biasprm[ids.gripper_actuator, :3] if not motor else np.zeros(3)
    kp, kd = 100.0, 10.0
    for command, desired in (
        (lambda: gripper.close(0.02), kp * (0.01 - length) - kd * speed),
        (lambda: gripper.squeeze(0.6, speed=None), -0.6 / 0.5 - kd * speed),  # T03 law
    ):
        command()
        expected = (desired - b[0] - b[1] * length - b[2] * speed) / g
        assert data.ctrl[ids.gripper_actuator] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        # And MuJoCo turns that ctrl into exactly the desired tendon force.
        mujoco.mj_forward(model, data)
        assert data.actuator_force[ids.gripper_actuator] == pytest.approx(desired, rel=1e-9)


def test_force_range_clips_before_the_ctrl_range() -> None:
    """A motor whose ctrl range (±200) is wider than its force range (±100)."""
    model = _motor_model(ctrl_limit=200.0)
    ids, data, osc, gripper = _floating_cube(model)
    assert gripper.max_squeeze_force() == pytest.approx(50.0)
    gripper.squeeze(80.0, speed=None)  # T03 law: the full request at once
    assert gripper.saturated
    assert data.ctrl[ids.gripper_actuator] == pytest.approx(-100.0, abs=0.5)
    _settle(model, data, osc, gripper, 0.4)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(50.0, rel=1e-3)
    gripper.squeeze(1e4, speed=None)  # the largest accepted request: flagged, finite
    assert gripper.saturated and math.isfinite(data.ctrl[ids.gripper_actuator])


@KINDS
def test_squeeze_from_open_overshoots_then_settles_on_the_request(kind: str) -> None:
    """Pins the T03 closing transient: no closing-speed limit in squeeze(f, speed=None)."""
    model = _model(kind)
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(1.0, speed=None)
    peak = 0.0
    for _ in range(round(0.8 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc, gripper], 1)
        mujoco.mj_forward(model, data)
        peak = max(peak, *S.finger_cube_forces(model, data, ids))
    assert 1.5 < peak < 2.5, peak
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(1.0, rel=1e-3)


def test_gripper_rejects_a_geared_actuator_and_out_of_range_requests() -> None:
    for gear in (2.0, -1.0):
        model = _motor_model(gear=gear)
        data = mujoco.MjData(model)
        S.reset_to_home(model, data)
        with pytest.raises(ValueError, match="gear"):
            Gripper(model, data)
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    for width in (-0.001, 0.081, 10.0):
        with pytest.raises(ValueError):
            gripper.max_squeeze_force(width)
    with pytest.raises(ValueError):
        gripper.squeeze(1e4 + 1.0)


def test_apply_raises_on_a_non_finite_finger_state() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data, ids)
    data.qvel[model.jnt_dofadr[model.joint(S.FINGER_JOINTS[0]).id]] = math.nan
    mujoco.mj_forward(model, data)
    ctrl, command = float(data.ctrl[ids.gripper_actuator]), gripper.command
    with pytest.raises(FloatingPointError):
        gripper.apply()
    assert data.ctrl[ids.gripper_actuator] == ctrl
    # A command that cannot be applied does not become the active command.
    for call, arg in ((gripper.squeeze, 1.0), (gripper.close, 0.02)):
        with pytest.raises(FloatingPointError):
            call(arg)
        assert gripper.command is command
        assert data.ctrl[ids.gripper_actuator] == ctrl


def test_command_is_read_only() -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    with pytest.raises(AttributeError):
        gripper.command = GripperCommand("force", -50.0)  # type: ignore[misc]
    assert gripper.command == GripperCommand("position", 0.08)


def test_gripper_reads_a_custom_servo_and_an_unlimited_motor() -> None:
    spec = S.build_spec(SERVO)
    act = spec.actuator(S.GRIPPER_ACTUATOR)
    act.gainprm[0] = 0.04 * 200.0 / 255.0  # a 200 N/m, 20 N·s/m servo, same ctrl mapping
    act.biasprm[:3] = [0.0, -200.0, -20.0]
    model = spec.compile()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    assert (gripper.servo_kp, gripper.servo_kd) == (200.0, 20.0)
    gripper.close(0.03)
    assert data.ctrl[S.scene_ids(model).gripper_actuator] == pytest.approx(255.0 * 0.03 / 0.08)
    assert gripper.max_squeeze_force(0.05) == pytest.approx(50.0 * 0.05)  # kp·L/2 = 50·w

    spec = S.build_spec()
    act = spec.actuator(S.GRIPPER_ACTUATOR)
    act.set_to_motor()
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    model = spec.compile()
    ids, data, osc, gripper = _floating_cube(model)
    # No force limit: the cap is the largest accepted request (the end stop scale).
    assert gripper.max_squeeze_force() == pytest.approx(G.MAX_SQUEEZE_N)
    gripper.squeeze(30.0)
    _settle(model, data, osc, gripper)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(30.0, rel=1e-3)
    assert not gripper.saturated


def test_max_squeeze_force_clamps_a_slightly_out_of_range_current_width() -> None:
    model = _servo_model()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data, ids)
    data.qpos[ids.finger_qpos] = -1e-6  # soft joint limits let the fingers stray past 0
    assert gripper.max_squeeze_force() == 0.0
    data.qpos[ids.finger_qpos] = 0.04 + 1e-6
    assert gripper.max_squeeze_force() == pytest.approx(25.0 * 0.08)


def test_gripper_follows_the_model_tendon_coupling() -> None:
    """Regression (T03 review): the width and cap math hard-coded the coupling c = 0.5.
    With c = 1, close(0.04) settled at 0.020 m and max_squeeze_force(0.05) said 2.5 N."""
    model = _servo_model()
    ids = S.scene_ids(model)
    tendon = int(model.actuator_trnid[ids.gripper_actuator, 0])
    adr = int(model.tendon_adr[tendon])
    model.wrap_prm[adr : adr + 2] = 1.0  # L = q1 + q2 = w
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.2, 0.6)
    osc = O.OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    gripper.close(0.03)
    _settle(model, data, osc, gripper, 1.0)
    assert gripper.width() == pytest.approx(0.03, abs=1e-3)
    assert gripper.max_squeeze_force(0.05) == pytest.approx(5.0)  # c·kp·(c·w) = 100·w
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(2.0)  # above the c = 0.5 cap (1.25 N), below this one (5 N)
    _settle(model, data, osc, gripper)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(2.0, rel=1e-3)
    assert not gripper.saturated


# ------------------------------------------------- T05: closing speed and end stop


def _free_gripper(model: mujoco.MjModel, control_hz: float = O.CONTROL_HZ):
    """Home pose, cube far away: the fingers close on nothing."""
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.2, 0.6)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=control_hz)
    return ids, data, osc, Gripper(model, data, ids)


def _run_ticks(model, data, osc, gripper, seconds: float, control_hz: float = O.CONTROL_HZ):
    substeps = O.substeps_for(model, control_hz)
    widths = []
    for _ in range(round(seconds * control_hz)):
        O.control_step(model, data, [osc, gripper], substeps)
        widths.append(gripper.width())
    mujoco.mj_forward(model, data)
    return np.array(widths)


@pytest.mark.parametrize("force", [0.5, 20.0])
@pytest.mark.parametrize("speed", [0.02, 0.1, 0.5])
def test_squeeze_with_speed_closes_at_that_speed(speed: float, force: float) -> None:
    """The bumpless start accelerates the fingers smoothly up to `speed`, never past it.

    T05 round 3: a clamped target fed no speed forward, so the damping held a weak
    squeeze below the speed (0.5 N at 0.5 m/s closed at 0.37 m/s, 0.25 N at 0.13).
    """
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(force, speed=speed)
    widths = [gripper.width()]
    while widths[-1] > G.END_STOP_WIDTH + G.END_STOP_BAND + 0.001:  # above the band
        O.control_step(model, data, [osc, gripper], 1)
        widths.append(gripper.width())
        assert len(widths) < 5 * O.CONTROL_HZ
    rate = -np.diff(widths) * O.CONTROL_HZ
    assert rate.max() < 1.02 * speed, rate.max()
    assert rate[-1] > 0.97 * speed, rate[-1]
    assert not gripper.saturated


@pytest.mark.parametrize("speed", [0.05, 0.4])
@pytest.mark.parametrize("force", [3.0, 20.0])
def test_squeeze_with_speed_settles_on_the_request(speed: float, force: float) -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(force, speed=speed)
    assert gripper.command == GripperCommand("force", force, speed)
    _settle(model, data, osc, gripper, 0.015 / speed + force / (250.0 * speed) + 0.6)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(force, rel=1e-3)
    assert gripper.actuator_squeeze() == pytest.approx(force, rel=1e-3)
    assert not gripper.saturated


def test_in_contact_the_squeeze_ramps_at_250_speed_newtons_per_second() -> None:
    """TRACK_KP·c² = 250: the target runs ahead of the blocked fingers at `speed`."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.close(0.054)  # 2 mm clearance per side
    _settle(model, data, osc, gripper, 0.5)
    speed, force = 0.02, 8.0
    gripper.squeeze(force, speed=speed)
    trace = []
    for _ in range(round(2.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc, gripper], 1)
        mujoco.mj_forward(model, data)
        trace.append((float(data.time), gripper.actuator_squeeze()))
    t, f = np.array(trace).T
    mid = (f > 2.0) & (f < 6.0)  # well inside the ramp
    slope = np.polyfit(t[mid], f[mid], 1)[0]
    assert slope == pytest.approx(G.TRACK_KP * 0.25 * speed, rel=0.05), slope
    assert f[-1] == pytest.approx(force, rel=1e-3)


def test_a_stronger_squeeze_ramps_up_from_the_current_force() -> None:
    """Probe → squeeze: the new target starts at the applied force, so the grip never dips."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(2.0, speed=0.05)
    _settle(model, data, osc, gripper, 0.8)
    assert sum(S.finger_cube_forces(model, data, ids)) == pytest.approx(4.0, rel=1e-3)
    gripper.squeeze(6.0, speed=0.05)
    low = math.inf
    for _ in range(round(1.0 * O.CONTROL_HZ)):
        O.control_step(model, data, [osc, gripper], 1)
        mujoco.mj_forward(model, data)
        low = min(low, sum(S.finger_cube_forces(model, data, ids)))
    assert low > 0.99 * 4.0, low
    assert sum(S.finger_cube_forces(model, data, ids)) == pytest.approx(12.0, rel=1e-3)
    gripper.squeeze(1.0, speed=0.05)  # a weaker request applies at once
    _settle(model, data, osc, gripper, 0.5)
    assert sum(S.finger_cube_forces(model, data, ids)) == pytest.approx(2.0, rel=1e-3)


def test_a_lost_object_leaves_a_bounded_target_and_the_end_stop_catches_the_fingers() -> None:
    """Anti-windup: a long hold does not wind the target up past the full squeeze.

    After the object is gone the fingers close under at most that squeeze (the
    target leads by f/(c·TRACK_KP) = 40 mm of tendon at 20 N), and the end stop
    brakes them to rest above the stop.
    """
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    force = 20.0
    gripper.squeeze(force, speed=0.05)
    _settle(model, data, osc, gripper, 3.0)
    lead = float(data.ten_length[0]) - gripper._track_length
    # At rest the bound's force is the squeeze: 2f/TRACK_KP minus the feed-forward lag.
    expected = (2.0 * force - G.TRACK_KD * 0.5 * 0.05) / G.TRACK_KP + G._TRACK_SLACK
    assert lead == pytest.approx(expected, rel=1e-3)
    q = ids.cube_qpos
    data.qpos[q : q + 3] = (0.2, 0.6, 0.3)  # the cube vanishes from the grasp
    data.qvel[ids.cube_dof : ids.cube_dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    widths = _run_ticks(model, data, osc, gripper, 1.0)
    assert widths.min() > -0.001, widths.min()
    assert 0.0 < widths[-1] < G.END_STOP_WIDTH + 0.002, widths[-1]


@pytest.mark.parametrize("control_hz", [500, 100])
@pytest.mark.parametrize("speed", [None, 0.5])
@pytest.mark.parametrize("force", [5.0, 10.0, 20.0, 50.0])
def test_a_squeeze_on_nothing_stops_at_the_end_stop(control_hz, speed, force) -> None:
    """T03 handoff: a force motor squeezing >= 5 N on nothing jammed the fingers (-11..-49 mm).

    With a speed the fingers never pass the stop. Without one (T03 law, no speed
    cap) they reach several m/s and can overrun it for a moment (the finger-finger
    contact exclusion keeps them from jamming). Below the stop nothing pushes
    them back, so they rest at or a little inside it, never below 0.
    """
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model, control_hz)
    gripper.squeeze(force, speed=speed)
    widths = _run_ticks(model, data, osc, gripper, 1.5, control_hz)
    if speed is not None:
        assert widths.min() > 0.0, widths.min()
    assert 0.0 <= widths[-1] <= G.END_STOP_WIDTH + 0.003, widths[-1]
    assert gripper.saturated  # the stop caps the request
    gripper.open()
    _run_ticks(model, data, osc, gripper, 1.0, control_hz)
    assert gripper.width() == pytest.approx(0.08, abs=1e-3)


def test_speed_limited_squeeze_is_stable_at_the_100_hz_floor() -> None:
    model = S.build_scene()
    ids, data, _, _ = _floating_cube(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=100)
    gripper = Gripper(model, data, ids)
    gripper.squeeze(10.0, speed=0.5)
    _run_ticks(model, data, osc, gripper, 1.0, 100)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(10.0, rel=1e-3)


@pytest.mark.parametrize("timeconst", [0.03, 0.05])
def test_closing_speed_sets_the_peak_force_on_a_soft_cube(timeconst: float) -> None:
    """The force profile matters (ROADMAP T05): same request, faster closing, higher peak."""
    peaks = {}
    for speed in (0.02, 0.5):
        model = S.build_scene()
        S.set_cube_solref(model, (timeconst, 1.0))
        ids, data, osc, gripper = _floating_cube(model)
        gripper.squeeze(2.0, speed=speed)
        peak = 0.0
        for _ in range(round((0.03 / speed + 0.5) * O.CONTROL_HZ)):
            O.control_step(model, data, [osc, gripper], 1)
            peak = max(peak, sum(S.finger_cube_forces(model, data, ids)))
        peaks[speed] = peak / 4.0
    assert peaks[0.02] < 1.02, peaks
    assert peaks[0.5] > 1.15, peaks


def _contact_trace(model, data, osc, gripper, ids, seconds: float):
    """Per-tick (time, actuator squeeze per finger, summed finger-cube force, width)."""
    rows = []
    for _ in range(round(seconds * O.CONTROL_HZ)):
        O.control_step(model, data, [osc, gripper], 1)
        mujoco.mj_forward(model, data)
        rows.append(
            (
                float(data.time),
                gripper.actuator_squeeze(),
                sum(S.finger_cube_forces(model, data, ids)),
                gripper.width(),
            )
        )
    return np.array(rows)


@pytest.mark.parametrize("speed", [0.02, 0.5])
def test_resending_the_same_squeeze_every_tick_changes_nothing(speed: float) -> None:
    """T07 re-sends the grip action every step: the ramp must not restart."""
    traces = []
    for resend in (False, True):
        model = S.build_scene()
        ids, data, osc, gripper = _floating_cube(model)
        gripper.squeeze(4.0, speed=speed)
        rows = []
        for _ in range(round((0.03 / speed + 0.5) * O.CONTROL_HZ)):
            if resend:
                gripper.squeeze(4.0, speed=speed)
            O.control_step(model, data, [osc, gripper], 1)
            rows.append(gripper.actuator_squeeze())
        traces.append(np.array(rows))
    np.testing.assert_allclose(traces[1], traces[0], rtol=0, atol=1e-9)


def test_a_stronger_squeeze_continues_from_the_current_force_without_a_step() -> None:
    """Bumpless start: in contact the new squeeze ramps at 250·v N/s from the old one."""
    for speed in (0.05, 0.5):
        model = S.build_scene()
        ids, data, osc, gripper = _floating_cube(model)
        gripper.squeeze(1.0, speed=0.05)
        _settle(model, data, osc, gripper, 1.2)
        gripper.squeeze(5.0, speed=speed)
        rows = _contact_trace(model, data, osc, gripper, ids, 0.6)
        ramp = G.TRACK_KP * 0.25 * speed  # N/s per finger
        assert rows[0, 1] - 1.0 < 1.5 * ramp / O.CONTROL_HZ + 0.06, (speed, rows[0, 1])
        assert rows[:, 1].max() <= 5.0 + 1e-6  # never above the request
        rising = (rows[:, 1] > 1.5) & (rows[:, 1] < 4.5)
        if rising.sum() > 3:
            slope = np.polyfit(rows[rising, 0], rows[rising, 1], 1)[0]
            assert slope == pytest.approx(ramp, rel=0.1), (speed, slope)
        assert rows[-1, 1] == pytest.approx(5.0, rel=1e-3)


@pytest.mark.parametrize("delay_ticks", [0, 5, 20])
def test_switching_speed_and_force_at_contact_never_lets_go(delay_ticks: int) -> None:
    """Round-2 review: a fast probe then a slow, stronger squeeze, sent at contact,
    pulled the fingers open (finger force 0 N). The bumpless start keeps the force."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(2.0, speed=0.5)
    for _ in range(2 * O.CONTROL_HZ):
        O.control_step(model, data, [osc, gripper], 1)
        if sum(S.finger_cube_forces(model, data, ids)) > 0:
            break
    for _ in range(delay_ticks):
        O.control_step(model, data, [osc, gripper], 1)
    gripper.squeeze(10.0, speed=0.05)
    rows = _contact_trace(model, data, osc, gripper, ids, 0.8)
    assert rows[:, 1].min() > 0.5, rows[:, 1].min()  # the motor never pulls open
    assert rows[:, 2].min() > 0.9 * 4.0, rows[:, 2].min()  # never below the probe's 2·2 N
    assert rows[-1, 2] == pytest.approx(20.0, rel=1e-3)


def test_changing_only_the_speed_keeps_the_force_continuous() -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(30.0, speed=0.5)
    for _ in range(80):
        O.control_step(model, data, [osc, gripper], 1)
    mujoco.mj_forward(model, data)
    before = gripper.actuator_squeeze()
    assert 3.0 < before < 27.0, before  # mid-ramp
    gripper.squeeze(30.0, speed=0.001)
    O.control_step(model, data, [osc, gripper], 1)
    mujoco.mj_forward(model, data)
    assert gripper.actuator_squeeze() == pytest.approx(before, abs=0.05)


def test_a_gap_between_apply_calls_does_not_release_a_jump() -> None:
    """A long gap (here data.time jumps 1 s, as if apply() had not run) restarts the
    target at the fingers instead of leaping them ahead."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    speed = 0.05
    gripper.squeeze(20.0, speed=speed)
    _run_ticks(model, data, osc, gripper, 0.2)
    data.time += 1.0
    widths = _run_ticks(model, data, osc, gripper, 0.2)
    rate = -np.diff(widths) * O.CONTROL_HZ
    # Uncapped, the stale target would leap 1 s ahead: 1.25 m/s, 25x.
    assert rate.max() < 1.05 * speed, rate.max()


def test_a_weaker_squeeze_in_contact_drops_at_once_without_opening() -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(20.0, speed=0.2)
    _settle(model, data, osc, gripper, 1.0)
    width = gripper.width()
    gripper.squeeze(2.0, speed=0.2)
    rows = _contact_trace(model, data, osc, gripper, ids, 0.5)
    assert rows[:, 1].min() > 1.9  # the motor never pulls the fingers open
    assert rows[:, 1].max() < 2.1 + 1e-6
    assert rows[-1, 2] == pytest.approx(4.0, rel=1e-3)
    assert rows[:, 3].max() < width + 0.002  # no opening pulse


def test_a_reset_restarts_the_speed_target_at_the_fingers() -> None:
    """reset_to_home moves time back; a stale target would close the fingers fast."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    speed = 0.1
    gripper.squeeze(20.0, speed=speed)
    _settle(model, data, osc, gripper, 1.5)
    S.reset_to_home(model, data)  # fingers open again, time 0
    S.place_cube(model, data, 0.2, 0.6)
    osc.hold_current()
    gripper.squeeze(20.0, speed=speed)  # the same command again keeps its target...
    widths = _run_ticks(model, data, osc, gripper, 0.3)
    rate = -np.diff(widths) * O.CONTROL_HZ
    assert rate.max() < 1.05 * speed, rate.max()  # ...but the reset restarted it
    assert rate[-1] > 0.95 * speed, rate[-1]  # at the fingers: closing at once


def test_without_a_speed_the_t03_law_has_no_speed_cap_above_the_band() -> None:
    """The end-stop brake acts only inside the band (below 3 cm)."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(6.0, speed=None)
    widths = _run_ticks(model, data, osc, gripper, 0.2)
    above = widths > G.END_STOP_WIDTH + G.END_STOP_BAND
    rate = -np.diff(widths)[above[1:]] * O.CONTROL_HZ
    # Measured 1.77 m/s; a brake at every width held it near 0.93 m/s.
    assert rate.max() > 1.5, rate.max()


@pytest.mark.parametrize("control_hz", [500, 100])
@pytest.mark.parametrize("vary", ["force", "speed"])
def test_a_squeeze_that_changes_every_tick_still_closes_and_ramps(control_hz, vary) -> None:
    """Round-3 review: a changed command re-placed the target at the fingers, so a
    command that changed every tick (T07's continuous grip action) never closed."""
    model = S.build_scene()
    ids, data, _, _ = _floating_cube(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=control_hz)
    gripper = Gripper(model, data, ids)
    substeps = O.substeps_for(model, control_hz)
    widths = []
    for k in range(round(1.0 * control_hz)):
        jitter = 1e-6 * (k % 2)
        if vary == "force":
            gripper.squeeze(10.0 + jitter, speed=0.1)
        else:
            gripper.squeeze(10.0, speed=0.1 + jitter)
        O.control_step(model, data, [osc, gripper], substeps)
        widths.append(gripper.width())
    rate = -np.diff(widths[: round(0.2 * control_hz)]) * control_hz
    assert rate.max() < 1.05 * 0.1, rate.max()
    assert rate[-1] > 0.95 * 0.1, rate[-1]  # closing at the speed
    mujoco.mj_forward(model, data)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(10.0, rel=1e-3)  # 0.15 s closing + 0.4 s ramp


@pytest.mark.parametrize("speed", [0.05, 0.2])
def test_at_the_100_hz_floor_the_speed_and_the_ramp_hold(speed: float) -> None:
    """The target advances 10 ms per apply() at 100 Hz (_MAX_TRACK_DT), not less."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model, 100)
    gripper.squeeze(20.0, speed=speed)
    widths = _run_ticks(model, data, osc, gripper, 0.2, 100)  # 0.15 s to reach the speed
    rate = -np.diff(widths) * 100
    assert rate[-1] == pytest.approx(speed, rel=0.02), rate
    model = S.build_scene()
    ids, data, _, _ = _floating_cube(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=100)
    gripper = Gripper(model, data, ids)
    gripper.close(0.054)
    _run_ticks(model, data, osc, gripper, 0.5, 100)
    gripper.squeeze(20.0, speed=speed)
    trace = []
    for _ in range(round((0.002 / speed + 20.0 / (250.0 * speed) + 0.3) * 100)):
        O.control_step(model, data, [osc, gripper], O.substeps_for(model, 100))
        mujoco.mj_forward(model, data)
        trace.append((float(data.time), gripper.actuator_squeeze()))
    t, f = np.array(trace).T
    mid = (f > 6.0) & (f < 14.0)
    slope = np.polyfit(t[mid], f[mid], 1)[0]
    assert slope == pytest.approx(G.TRACK_KP * 0.25 * speed, rel=0.05), slope
    assert f[-1] == pytest.approx(20.0, rel=1e-3)


@pytest.mark.parametrize("control_hz", [500, 100])
@pytest.mark.parametrize(("size", "timeconst", "force"), [(0.05, 0.04, 30.0), (0.042, 0.08, 50.0)])
def test_the_default_squeeze_does_not_drive_the_fingers_through_a_soft_cube(
    control_hz, size, timeconst, force
) -> None:
    """Round-3 review: the T03 law (now speed=None) hit soft cubes at ~5 m/s and the
    fingers passed through them (-50 mm, 145 N per finger; open() could not free
    them). The default squeeze closes at MAX_CLOSE_SPEED and stops on the cube."""
    model = S.build_scene()
    S.set_cube_size(model, size)
    S.set_cube_solref(model, (timeconst, 1.0))
    ids, data, _, _ = _floating_cube(model)
    osc = O.OperationalSpaceController(model, data, ids=ids, control_hz=control_hz)
    gripper = Gripper(model, data, ids)
    gripper.squeeze(force)
    widths = _run_ticks(model, data, osc, gripper, 1.5, control_hz)
    assert widths.min() > 0.5 * size, widths.min()  # measured 0.60-0.87 x size
    expected = min(force, gripper.max_squeeze_force())
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(expected, rel=1e-2)
    gripper.open()
    _run_ticks(model, data, osc, gripper, 2.0, control_hz)
    assert gripper.width() == pytest.approx(0.08, abs=1e-3)


def test_a_held_squeeze_survives_being_pushed_open() -> None:
    """T05 review round 4: the lag term had no cap, so a held stiff cube that grew
    4 mm in one tick (fingers pushed open at 0.5 m/s) dropped the squeeze to 0.01 N
    for > 0.5 s. Capped at c·v, the grip recovers like the pre-round-3 code (18 ms)."""
    model = S.build_scene()
    S.set_cube_solref(model, (0.004, 1.0))
    ids, data, osc, gripper = _floating_cube(model)
    gripper.close(0.051)
    _settle(model, data, osc, gripper, 0.4)
    gripper.squeeze(2.0, speed=0.02)
    _settle(model, data, osc, gripper, 0.001 / 0.02 + 2.0 / (250 * 0.02) + 0.4)
    S.set_cube_size(model, 0.054)  # the cube grows 4 mm in one tick
    rows = _contact_trace(model, data, osc, gripper, ids, 0.5)
    assert rows[:, 1].min() > 1.0, rows[:, 1].min()
    low = rows[:, 2] < 0.9 * 4.0
    assert low.sum() / O.CONTROL_HZ < 0.05, low.sum()  # measured 18 ms
    assert rows[-1, 2] == pytest.approx(4.0, rel=1e-3)


@pytest.mark.parametrize("timeconst", [0.04, 0.06])
@pytest.mark.parametrize(("force", "speed"), [(2.0, 0.02), (1.0, 0.001)])
def test_a_slower_squeeze_at_contact_never_pulls_off_a_soft_cube(timeconst, force, speed) -> None:
    """T05 review round 4: switching a fast probe to a slower squeeze at contact braked
    the fingers, and the motor pulled them off fragile-range cubes (contact force 0
    for 3-14 ticks). A squeeze never pulls the fingers off what their pads touch."""
    model = S.build_scene()
    S.set_cube_solref(model, (timeconst, 1.0))
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(2.0, speed=0.5)
    for _ in range(2 * O.CONTROL_HZ):
        O.control_step(model, data, [osc, gripper], 1)
        if sum(S.finger_cube_forces(model, data, ids)) > 0:
            break
    gripper.squeeze(force, speed=speed)
    rows = _contact_trace(model, data, osc, gripper, ids, 0.8)
    assert rows[:, 1].min() > -1e-9, rows[:, 1].min()  # the motor never pulls open
    assert rows[:, 2].min() > 0.0, rows[:, 2].min()  # contact never lost


def test_a_changed_squeeze_after_a_gap_restarts_without_reopening() -> None:
    """T05 review round 4: after a gap in apply() calls a changed command kept the
    stale target, and the motor pulled the fingers back open (0.06 m/s)."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(20.0, speed=0.05)
    _run_ticks(model, data, osc, gripper, 0.2)
    for _ in range(50):  # the arm keeps its controller; the gripper misses 50 ticks
        O.control_step(model, data, [osc], 1)
    gripper.squeeze(20.0, speed=0.1)
    widths = np.array([gripper.width()] + list(_run_ticks(model, data, osc, gripper, 0.2)))
    assert np.diff(widths).max() < 1e-5, np.diff(widths).max()  # never reopens
    rate = -np.diff(widths) * O.CONTROL_HZ
    assert rate.max() < 1.05 * 0.1, rate.max()


@pytest.mark.parametrize("resend", [False, True])
def test_the_same_squeeze_after_a_gap_restarts_without_reopening(resend: bool) -> None:
    """T05 review round 5: the gap restart covered only a changed command; resuming
    apply() (or re-sending the same command) after 50 missed ticks reopened the
    fingers for 19 ticks (0.06 m/s) toward the stale target."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(20.0, speed=0.05)
    _run_ticks(model, data, osc, gripper, 0.2)
    for _ in range(50):
        O.control_step(model, data, [osc], 1)
    if resend:
        gripper.squeeze(20.0, speed=0.05)
    widths = np.array([gripper.width()] + list(_run_ticks(model, data, osc, gripper, 0.2)))
    assert np.diff(widths).max() < 1e-5, np.diff(widths).max()  # never reopens
    rate = -np.diff(widths) * O.CONTROL_HZ
    assert rate.max() < 1.05 * 0.05, rate.max()
    assert rate[-1] > 0.95 * 0.05, rate[-1]


@pytest.mark.parametrize(
    ("control_hz", "first", "missed"), [(100, 21, 2), (100, 19, 2), (500, 100, 10)]
)
def test_a_gap_inside_the_end_stop_band_does_not_fly_the_fingers_open(control_hz, first, missed):
    """T05 review round 6: ctrl froze during a gap while it held the end-stop brake's
    opening force, and the restarted target kept pushing the fingers open (+48 mm)."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model, control_hz)
    substeps = O.substeps_for(model, control_hz)
    gripper.squeeze(20.0, speed=0.5)
    widths = []
    for k in range(round(0.6 * control_hz)):
        controllers = [osc] if first <= k < first + missed else [osc, gripper]
        O.control_step(model, data, controllers, substeps)
        widths.append(gripper.width())
    widths = np.array(widths)
    after = widths[first + missed :]
    assert after.max() < widths[first + missed - 1] + 0.001, after.max()  # measured +0.6 mm
    assert after[-1] < G.END_STOP_WIDTH + 0.003, after[-1]


def test_a_stronger_squeeze_after_a_gap_in_a_hold_has_no_step() -> None:
    """T05 review round 7: restarting at the fingers unconditionally after a gap made
    a stronger squeeze step from a 2 N hold to 7.5·v = 3.75 N on its first tick."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(2.0, speed=0.5)
    _settle(model, data, osc, gripper, 0.6)
    for _ in range(50):  # the gripper misses 50 ticks while it holds
        O.control_step(model, data, [osc], 1)
    gripper.squeeze(4.0, speed=0.5)
    rows = _contact_trace(model, data, osc, gripper, ids, 0.3)
    ramp = G.TRACK_KP * 0.25 * 0.5 / O.CONTROL_HZ  # one tick of the 250·v N/s ramp
    assert rows[0, 1] < 2.0 + 1.5 * ramp + 0.05, rows[0, 1]
    assert rows[-1, 1] == pytest.approx(4.0, rel=1e-3)


def test_a_gap_while_the_fingers_are_pushed_open_does_not_keep_pushing_them_open() -> None:
    """T05 review round 8: a gap that froze a closing force while a push opened the
    fingers placed the restarted target 48 mm open of them, and the motor then pushed
    them open off contact for 78 ticks. The restart is capped where a hold sits."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    gripper.squeeze(5.0, speed=0.5)
    _settle(model, data, osc, gripper, 1.0)
    opening_off_contact = 0
    for k in range(round(0.4 * O.CONTROL_HZ)):
        push = k < round(0.04 * O.CONTROL_HZ)
        data.xfrc_applied[finger, :3] = 30.0 * data.xmat[finger].reshape(3, 3)[:, 1] if push else 0
        gap = k < round(0.02 * O.CONTROL_HZ)
        O.control_step(model, data, [osc] if gap else [osc, gripper], 1)
        if not gap and data.ctrl[ids.gripper_actuator] > 0.05 and not gripper._touching():
            opening_off_contact += 1
    assert opening_off_contact == 0, opening_off_contact


def test_apply_below_the_100_hz_floor_raises() -> None:
    """T05 review round 8: a gripper-only control_step at 50 Hz restarted the target on
    every call, so the fingers stalled (or, capped, went unstable) without an error."""
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(20.0, speed=0.05)
    O.control_step(model, data, [gripper], 10)  # 50 Hz, at the squeeze: no gap yet
    O.control_step(model, data, [gripper], 10)  # the first gap restarts the target
    with pytest.raises(RuntimeError, match="100 Hz"):
        O.control_step(model, data, [gripper], 10)  # the second gap in a row raises
    model = S.build_scene()
    ids, data, osc, gripper = _free_gripper(model)
    gripper.squeeze(20.0, speed=0.05)
    widths = []
    for _ in range(50):  # 100 Hz, gripper alone: fine
        O.control_step(model, data, [gripper], 5)
        widths.append(gripper.width())
    assert -np.diff(widths)[-1] * 100 == pytest.approx(0.05, rel=0.02)
