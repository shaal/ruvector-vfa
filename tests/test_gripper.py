"""T03: gripper interface (``open``, ``close(width)``, ``squeeze(force_n)``, ``width``).

Pins the force mapping documented in ``vfa/control/gripper.py``: a squeeze
request lands on the cube as that normal force per finger, the menagerie servo
caps it at 25·width N, and the same class works unchanged when T05 swaps the
servo for a force ``motor``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from vfa.control import osc as O
from vfa.control.gripper import Gripper, GripperCommand
from vfa.sim import scene as S


def _motor_model(ctrl_limit: float = 100.0, gear: float = 1.0) -> mujoco.MjModel:
    """The scene with actuator8 turned into a force motor (ctrl = tendon force, ±100 N)."""
    spec = S.build_spec()
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


@pytest.mark.parametrize("force", [0.25, 0.5, 1.0, 1.2])
def test_squeeze_puts_the_requested_force_on_the_cube(force: float) -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(force)
    _settle(model, data, osc, gripper)
    left, right = S.finger_cube_forces(model, data, ids)
    assert left == pytest.approx(force, rel=1e-3)
    assert right == pytest.approx(force, rel=1e-3)
    assert gripper.actuator_squeeze() == pytest.approx(force, rel=1e-3)
    assert not gripper.saturated
    assert gripper.command == GripperCommand("force", force)
    assert gripper.width() == pytest.approx(0.05, abs=0.002)  # stopped by the 5 cm cube


def test_squeeze_saturates_at_the_servo_cap() -> None:
    """Menagerie servo: the ctrl floor caps the squeeze at 25·width N per finger."""
    model = S.build_scene()
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


def test_close_on_the_cube_squeezes_by_the_servo_stiffness() -> None:
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.close(0.04)  # 1 cm narrower than the cube
    _settle(model, data, osc, gripper)
    expected = 0.25 * gripper.servo_kp * (gripper.width() - 0.04)  # 25·(w − w*)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(expected, rel=1e-3)
    assert expected == pytest.approx(0.25, abs=0.01)


@pytest.mark.parametrize("motor", [False, True], ids=["servo", "motor"])
def test_close_and_open_track_the_width_without_an_object(motor: bool) -> None:
    model = _motor_model() if motor else S.build_scene()
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
    model = S.build_scene()
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
    model = _motor_model()
    ids, data, osc, gripper = _floating_cube(model)
    assert gripper.max_squeeze_force() == pytest.approx(50.0)  # 100 N tendon / 2 fingers
    gripper.squeeze(force)
    _settle(model, data, osc, gripper)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(force, rel=1e-3)
    assert not gripper.saturated


def test_force_motor_squeeze_saturates_at_the_force_range() -> None:
    model = _motor_model()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(80.0)
    _settle(model, data, osc, gripper, 0.4)
    assert gripper.saturated
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(50.0, rel=1e-3)


def test_max_squeeze_force_scales_with_width() -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data)
    for width in (0.0, 0.02, 0.05, 0.08):
        assert gripper.max_squeeze_force(width) == pytest.approx(25.0 * width, abs=1e-9)


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
    model = _motor_model() if motor else S.build_scene()
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
        (lambda: gripper.squeeze(0.6), -0.6 / 0.5 - kd * speed),
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
    gripper.squeeze(80.0)
    assert gripper.saturated
    assert data.ctrl[ids.gripper_actuator] == pytest.approx(-100.0, abs=0.5)
    _settle(model, data, osc, gripper, 0.4)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(50.0, rel=1e-3)
    gripper.squeeze(1e4)  # the largest accepted request: still flagged, still finite
    assert gripper.saturated and math.isfinite(data.ctrl[ids.gripper_actuator])


def test_squeeze_from_open_overshoots_then_settles_on_the_request() -> None:
    """Pins the documented closing transient: no closing-speed limit in squeeze()."""
    model = S.build_scene()
    ids, data, osc, gripper = _floating_cube(model)
    gripper.squeeze(1.0)
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
    ctrl = float(data.ctrl[ids.gripper_actuator])
    with pytest.raises(FloatingPointError):
        gripper.apply()
    assert data.ctrl[ids.gripper_actuator] == ctrl


def test_gripper_reads_a_custom_servo_and_an_unlimited_motor() -> None:
    spec = S.build_spec()
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
    assert gripper.max_squeeze_force() == math.inf
    gripper.squeeze(30.0)
    _settle(model, data, osc, gripper)
    for f in S.finger_cube_forces(model, data, ids):
        assert f == pytest.approx(30.0, rel=1e-3)
    assert not gripper.saturated


def test_max_squeeze_force_clamps_a_slightly_out_of_range_current_width() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    gripper = Gripper(model, data, ids)
    data.qpos[ids.finger_qpos] = -1e-6  # soft joint limits let the fingers stray past 0
    assert gripper.max_squeeze_force() == 0.0
    data.qpos[ids.finger_qpos] = 0.04 + 1e-6
    assert gripper.max_squeeze_force() == pytest.approx(25.0 * 0.08)
