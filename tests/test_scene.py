"""T02: Panda + table + cube scene built with MjSpec from the pristine Menagerie Panda.

Covers the acceptance list in ROADMAP T02: the model compiles; 7 arm torque
motors + the menagerie gripper actuator exist; the wrist F/T reads the
hand+fingers weight at rest; a gravity-compensated hold drifts < 1 cm in 2 s.
Also pins the vendored files, the per-episode hooks, and the touch sensors.
"""

from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np
import pytest

from vfa import cli
from vfa.sim import scene as S
from vfa.sim import scene_check

# Tree hash of assets/menagerie/franka_emika_panda (refer to assets/menagerie/NOTICE.md).
VENDORED_TREE_SHA256 = "1418e49508d92ec0a2ecf1c9f7a62d458b1d2ba940a19f980b1231e68b55f274"
HAND_PLUS_FINGERS_KG = 0.73 + 2 * 0.015  # menagerie panda.xml inertials


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return S.build_scene()


@pytest.fixture(scope="module")
def ids(model: mujoco.MjModel) -> S.SceneIds:
    return S.scene_ids(model)


@pytest.fixture(scope="module")
def held(model: mujoco.MjModel) -> scene_check.HoldResult:
    return scene_check.hold_home(model, mujoco.MjData(model), seconds=2.0)


# ------------------------------------------------------------------ vendored model


def test_vendored_panda_is_pristine() -> None:
    root = S.PANDA_DIR
    files = sorted(p for p in root.rglob("*") if p.is_file())
    tree = hashlib.sha256()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tree.update(path.relative_to(root).as_posix().encode() + b"\0" + digest.encode() + b"\n")
    assert {"panda.xml", "hand.xml", "LICENSE"} <= {p.name for p in files}
    assert tree.hexdigest() == VENDORED_TREE_SHA256, "vendored Menagerie files were modified"
    notice = (root.parent / "NOTICE.md").read_text()
    assert "feadf76d42f8a2162426f7d226a3b539556b3bf5" in notice
    assert "Apache" in notice


# ------------------------------------------------------------------- compiled model


def test_model_compiles_with_expected_structure(model: mujoco.MjModel) -> None:
    # 7 arm hinges + 2 finger slides + cube free joint (7 qpos / 6 dof).
    assert (model.nq, model.nv, model.nu) == (16, 15, 8)
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.nkey == 1 and model.key(0).name == S.HOME_KEY
    for name in (S.CUBE_BODY, S.HAND_BODY):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    for name in (S.EE_SITE, S.WRIST_SITE, S.PLACE_SITE, S.LEFT_TOUCH_SITE, S.RIGHT_TOUCH_SITE):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, S.OVERVIEW_CAMERA) >= 0


def test_arm_actuators_are_torque_motors(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    assert len(ids.arm_actuators) == 7
    for k, (act, joint, limit) in enumerate(
        zip(ids.arm_actuators, S.ARM_JOINTS, S.ARM_TORQUE_LIMITS, strict=True)
    ):
        assert act == k, "arm motors must be ctrl[0:7] in joint order"
        assert model.actuator(act).name == S.ARM_ACTUATORS[k]
        assert model.actuator_trntype[act] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_trnid[act, 0] == model.joint(joint).id
        assert model.actuator_dyntype[act] == mujoco.mjtDyn.mjDYN_NONE
        assert model.actuator_gaintype[act] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_gainprm[act, 0] == 1.0
        assert model.actuator_biastype[act] == mujoco.mjtBias.mjBIAS_NONE
        assert not np.any(model.actuator_biasprm[act])
        assert model.actuator_gear[act, 0] == 1.0
        assert model.actuator_ctrllimited[act] and model.actuator_forcelimited[act]
        np.testing.assert_array_equal(model.actuator_ctrlrange[act], [-limit, limit])
        np.testing.assert_array_equal(model.actuator_forcerange[act], [-limit, limit])
    np.testing.assert_array_equal(
        model.actuator_ctrlrange[ids.arm_actuators, 1], [87, 87, 87, 87, 12, 12, 12]
    )


def test_motor_ctrl_is_joint_torque(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    data.ctrl[ids.arm_actuators] = [5.0, -4.0, 3.0, -2.0, 1.0, -0.5, 0.25]
    data.ctrl[ids.arm_actuators[0]] = 200.0  # beyond the 87 N·m limit: must clip
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.qfrc_actuator[ids.arm_dof], [87.0, -4.0, 3.0, -2.0, 1.0, -0.5, 0.25]
    )


def test_gripper_keeps_menagerie_tendon_actuator(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    pristine = mujoco.MjModel.from_xml_path(str(S.PANDA_XML))
    p = mujoco.mj_name2id(pristine, mujoco.mjtObj.mjOBJ_ACTUATOR, S.GRIPPER_ACTUATOR)
    g = ids.gripper_actuator
    assert g == 7
    assert model.actuator_trntype[g] == mujoco.mjtTrn.mjTRN_TENDON
    assert model.tendon(model.actuator_trnid[g, 0]).name == "split"
    for name in (
        "actuator_trntype",
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gear",
    ):
        np.testing.assert_array_equal(getattr(model, name)[g], getattr(pristine, name)[p], name)


def test_sites_sit_where_documented(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    hand = model.body(S.HAND_BODY).id
    assert model.site_bodyid[ids.ee_site] == hand
    assert model.site_bodyid[ids.wrist_site] == hand
    # Wrist site = the link7 → hand flange (hand body origin).
    np.testing.assert_allclose(data.site_xpos[ids.wrist_site], data.xpos[hand], atol=1e-12)
    assert model.body_parentid[hand] == model.body("link7").id
    # ee = fingertip center: midway between the two main pads, at their height.
    left_pad = _main_pad(model, ids.left_finger_geoms)
    right_pad = _main_pad(model, ids.right_finger_geoms)
    mid = 0.5 * (data.geom_xpos[left_pad] + data.geom_xpos[right_pad])
    np.testing.assert_allclose(data.site_xpos[ids.ee_site], mid, atol=1e-3)
    # Place target lies on the table top.
    assert data.site_xpos[ids.place_site][2] == pytest.approx(S.TABLE_TOP_Z, abs=1e-3)


def _main_pad(model: mujoco.MjModel, geoms: frozenset[int]) -> int:
    """The finger's main fingertip pad box (menagerie ``fingertip_pad_collision_1``)."""
    (pad,) = [
        g
        for g in geoms
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        and np.allclose(model.geom_size[g], [0.0085, 0.004, 0.0085])
    ]
    return pad


def test_sensors_attach_to_the_right_sites(model: mujoco.MjModel) -> None:
    expected = {
        S.WRIST_FORCE_SENSOR: (mujoco.mjtSensor.mjSENS_FORCE, S.WRIST_SITE, 3),
        S.WRIST_TORQUE_SENSOR: (mujoco.mjtSensor.mjSENS_TORQUE, S.WRIST_SITE, 3),
        S.LEFT_TOUCH_SENSOR: (mujoco.mjtSensor.mjSENS_TOUCH, S.LEFT_TOUCH_SITE, 1),
        S.RIGHT_TOUCH_SENSOR: (mujoco.mjtSensor.mjSENS_TOUCH, S.RIGHT_TOUCH_SITE, 1),
    }
    assert model.nsensor == len(expected)
    for name, (stype, site, dim) in expected.items():
        sensor = model.sensor(name)
        assert sensor.type[0] == stype
        assert sensor.objtype[0] == mujoco.mjtObj.mjOBJ_SITE
        assert model.site(sensor.objid[0]).name == site
        assert sensor.dim[0] == dim


def test_home_keyframe_resets_arm_gripper_and_cube(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    np.testing.assert_allclose(data.qpos[ids.arm_qpos], S.HOME_ARM_QPOS)
    np.testing.assert_allclose(data.qpos[ids.finger_qpos], S.HOME_FINGER_QPOS)
    cfg = S.SceneConfig()
    np.testing.assert_allclose(
        data.qpos[ids.cube_qpos : ids.cube_qpos + 7],
        [*cfg.cube_xy, S.TABLE_TOP_Z + cfg.cube_size / 2, 1, 0, 0, 0],
    )
    np.testing.assert_allclose(data.ctrl, [0.0] * 7 + [S.GRIPPER_OPEN_CTRL])


def test_cube_geom_has_priority_and_condim(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    g = ids.cube_geom
    assert model.geom_priority[g] == 1  # ARCHITECTURE D8
    assert model.geom_condim[g] == 4
    np.testing.assert_allclose(model.geom_size[g], [0.025] * 3)
    assert model.body_mass[ids.cube_body] == pytest.approx(0.1)
    assert model.geom_priority[g] > model.geom_priority[list(ids.left_finger_geoms)].max()
    assert model.geom_priority[g] > model.geom_priority[ids.table_geom]


# ------------------------------------------------------- F/T and gravity compensation


def test_wrist_ft_at_rest_reads_hand_weight(held: scene_check.HoldResult) -> None:
    expected = HAND_PLUS_FINGERS_KG * 9.81
    assert held.expected_force_n == pytest.approx(expected, rel=1e-6)
    assert held.force_norm == pytest.approx(expected, rel=0.15)
    # At home the hand z axis points down, so the support force reads as -z.
    assert held.wrist_force[2] < -0.95 * held.force_norm
    assert held.finger_touch == (0.0, 0.0)


def test_wrist_torque_at_rest_is_the_weight_moment(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    """At rest the wrist torque is r × F, with r from the site to the hand-subtree COM."""
    data = mujoco.MjData(model)
    scene_check.hold_home(model, data, seconds=0.5)
    force, torque = S.read_wrist_ft(model, data, ids)
    rot = data.site_xmat[ids.wrist_site].reshape(3, 3)
    r = rot.T @ (data.subtree_com[ids.hand_body] - data.site_xpos[ids.wrist_site])
    np.testing.assert_allclose(torque, np.cross(r, force), atol=1e-6)
    assert abs(torque[1]) > 0.05  # the hand COM is off-axis, so the moment is not ~0


def test_helpers_accept_precomputed_ids(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    data = mujoco.MjData(model)
    scene_check.hold_home(model, data, seconds=0.1)
    np.testing.assert_array_equal(
        S.gravity_compensation(model, data, ids), S.gravity_compensation(model, data)
    )
    for a, b in zip(S.read_wrist_ft(model, data, ids), S.read_wrist_ft(model, data), strict=True):
        np.testing.assert_array_equal(a, b)
    assert S.read_finger_touch(model, data, ids) == S.read_finger_touch(model, data)
    assert S.finger_cube_forces(model, data, ids) == S.finger_cube_forces(model, data)


def test_config_timestep_reaches_the_model() -> None:
    assert S.build_scene(S.SceneConfig(timestep=0.001)).opt.timestep == 0.001


def test_reset_to_home_requires_the_home_key() -> None:
    pristine = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(KeyError):
        S.reset_to_home(pristine, mujoco.MjData(pristine))


def test_gravity_compensated_hold_drifts_less_than_1cm(held: scene_check.HoldResult) -> None:
    assert held.seconds == pytest.approx(2.0)
    assert held.ee_drift_m < 0.01
    assert held.max_joint_drift_rad < math.radians(0.5)


def test_without_gravity_compensation_the_arm_falls(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    """Negative control: the hold test passes because of the compensation torques."""
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    ee0 = data.site_xpos[ids.ee_site].copy()
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)  # ctrl stays 0
    assert np.linalg.norm(data.site_xpos[ids.ee_site] - ee0) > 0.01


def test_gravity_compensation_helper_matches_bias(model: mujoco.MjModel, ids: S.SceneIds) -> None:
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    tau = S.gravity_compensation(model, data)
    np.testing.assert_array_equal(tau, data.qfrc_bias[ids.arm_dof])
    assert np.all(np.abs(tau) <= model.actuator_ctrlrange[ids.arm_actuators, 1])


def test_hold_rejects_bad_durations(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    for bad in (0.0, -1.0, math.nan, math.inf, scene_check.MAX_HOLD_SECONDS + 1):
        with pytest.raises(ValueError):
            scene_check.hold_home(model, data, bad)


# ------------------------------------------------------------- per-episode hooks

_HOOK_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "bvh_aabb",
    "geom_friction",
    "geom_solref",
    "qpos0",
    "body_pos",
    "key_qpos",
)


@pytest.mark.parametrize("size_first", [True, False])
def test_hooks_match_a_fresh_compile(size_first: bool) -> None:
    edited = S.build_scene()
    if size_first:
        S.set_cube_size(edited, 0.065)
        S.set_cube_mass(edited, 0.7)
    else:
        S.set_cube_mass(edited, 0.7)
        S.set_cube_size(edited, 0.065)
    S.set_cube_friction(edited, (0.3, 0.01, 0.0002))
    S.set_cube_solref(edited, (0.01, 0.8))
    fresh = S.build_scene(
        S.SceneConfig(
            cube_size=0.065,
            cube_mass=0.7,
            cube_friction=(0.3, 0.01, 0.0002),
            cube_solref=(0.01, 0.8),
        )
    )
    compared = 0
    for name in dir(fresh):
        value = getattr(fresh, name, None)
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            np.testing.assert_allclose(
                getattr(edited, name), value, rtol=1e-9, atol=1e-12, err_msg=name
            )
            compared += 1
    assert compared > 200, "expected to compare every numeric MjModel array"
    assert set(_HOOK_FIELDS) <= {n for n in dir(fresh)}
    for name in ("meaninertia", "meanmass", "meansize", "extent"):
        assert getattr(edited.stat, name) == pytest.approx(getattr(fresh.stat, name), rel=1e-9)


@pytest.mark.parametrize("value", [np.float32(0.5), np.int64(1), np.array(0.5), 0.5, 1])
def test_hooks_accept_numpy_and_python_scalars(value) -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    S.set_cube_mass(model, value)
    S.set_cube_friction(model, value)
    assert model.body_mass[ids.cube_body] == pytest.approx(float(value))
    assert model.geom_friction[ids.cube_geom, 0] == pytest.approx(float(value))
    S.set_cube_size(model, np.float32(0.06))
    assert model.geom_size[ids.cube_geom, 0] == pytest.approx(0.03)
    S.set_cube_friction(model, np.array([0.4, 0.01, 0.001]))
    np.testing.assert_allclose(model.geom_friction[ids.cube_geom], [0.4, 0.01, 0.001])


def test_place_cube_sets_pose_and_stops_the_cube_immediately() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    data.qvel[ids.cube_dof : ids.cube_dof + 6] = [1.0, -2.0, 3.0, 0.5, 0.5, 0.5]
    S.place_cube(model, data, 0.42, 0.11, yaw=0.3)
    q = ids.cube_qpos
    np.testing.assert_allclose(data.qpos[q : q + 3], [0.42, 0.11, 0.025])
    np.testing.assert_allclose(data.qpos[q + 3 : q + 7], [math.cos(0.15), 0, 0, math.sin(0.15)])
    assert not np.any(data.qvel[ids.cube_dof : ids.cube_dof + 6])
    # mj_forward ran: the body pose already reflects the new qpos.
    np.testing.assert_allclose(data.xpos[ids.cube_body], [0.42, 0.11, 0.025])


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (0.0, 0.0),  # inside the robot base
        (0.1, 0.1),  # footprint reaches into the base keep-out
        (0.84, 0.0),  # centre on the table, footprint hangs over the +x edge
        (0.82, 0.0),  # only a 45°-yawed corner would overhang: half·√2 reach matters
        (0.5, 0.69),  # footprint hangs over the +y edge
        (5.0, 5.0),  # off the table entirely
    ],
)
def test_place_cube_rejects_poses_off_the_table_or_in_the_base(x: float, y: float) -> None:
    model = S.build_scene()
    data = mujoco.MjData(model)
    with pytest.raises(ValueError):
        S.place_cube(model, data, x, y)


def test_scalar_friction_keeps_torsional_and_rolling() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    before = model.geom_friction[ids.cube_geom].copy()
    S.set_cube_friction(model, 0.2)
    np.testing.assert_allclose(model.geom_friction[ids.cube_geom], [0.2, *before[1:]])


def test_hooks_govern_cube_contacts_and_dynamics() -> None:
    """Friction, solref, and mass written through the hooks reach the contact solver."""
    model = S.build_scene()
    ids = S.scene_ids(model)
    S.set_cube_friction(model, (0.25, 0.002, 0.0001))
    S.set_cube_solref(model, (0.005, 0.9))
    S.set_cube_mass(model, 0.6)
    S.set_cube_size(model, 0.04)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.45, -0.1, yaw=0.3)
    for _ in range(250):  # 0.5 s: settle on the table under gravity compensation
        mujoco.mj_step1(model, data)
        data.ctrl[ids.arm_actuators] = data.qfrc_bias[ids.arm_dof]
        mujoco.mj_step2(model, data)
    mujoco.mj_forward(model, data)

    table_contacts = [
        i
        for i in range(data.ncon)
        if {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        == {ids.cube_geom, ids.table_geom}
    ]
    assert table_contacts, "the cube must rest on the table"
    normal = 0.0
    wrench = np.zeros(6)
    for i in table_contacts:
        con = data.contact[i]
        assert con.dim == 4
        np.testing.assert_allclose(con.friction[:3], [0.25, 0.25, 0.002])
        np.testing.assert_allclose(con.solref, [0.005, 0.9])
        mujoco.mj_contactForce(model, data, i, wrench)
        normal += wrench[0]
    assert normal == pytest.approx(0.6 * 9.81, rel=0.02)
    # Resized cube rests flat at its new half-height, where place_cube put it.
    np.testing.assert_allclose(
        data.qpos[ids.cube_qpos : ids.cube_qpos + 3], [0.45, -0.1, 0.02], atol=1e-3
    )


def test_resized_cube_resets_flat_on_the_table() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    S.set_cube_size(model, 0.08)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    assert data.qpos[ids.cube_qpos + 2] == pytest.approx(0.04)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.qpos[ids.cube_qpos + 2] == pytest.approx(0.04, abs=5e-4)


@pytest.mark.parametrize(
    ("hook", "bad"),
    [
        (S.set_cube_mass, 0.0),
        (S.set_cube_mass, -0.1),
        (S.set_cube_mass, math.nan),
        (S.set_cube_mass, math.inf),
        (S.set_cube_size, 0.0),
        (S.set_cube_size, -0.05),
        (S.set_cube_size, math.nan),
        (S.set_cube_size, 1e-5),  # below CUBE_SIZE_RANGE
        (S.set_cube_size, 2.0),  # above CUBE_SIZE_RANGE
        (S.set_cube_size, True),
        (S.set_cube_mass, True),
        (S.set_cube_mass, "0.5"),
        (S.set_cube_friction, "123"),
        (S.set_cube_friction, True),
        (S.set_cube_friction, b"abc"),
        (S.set_cube_mass, 10**400),
        (S.set_cube_solref, 0.02),
        (S.set_cube_solref, (0.003, 1.0)),  # timeconst < 2 × timestep: MuJoCo would clamp
        (S.set_cube_friction, -0.5),
        (S.set_cube_friction, 0.0),
        (S.set_cube_friction, (1.0, 0.005)),
        (S.set_cube_friction, (1.0, math.nan, 0.0)),
        (S.set_cube_solref, (0.02,)),
        (S.set_cube_solref, (0.02, -1.0)),
        (S.set_cube_solref, (math.inf, 1.0)),
    ],
)
def test_hooks_reject_bad_values_without_side_effects(hook, bad) -> None:
    model = S.build_scene()
    snapshot = {name: getattr(model, name).copy() for name in _HOOK_FIELDS}
    with pytest.raises(ValueError):
        hook(model, bad)
    for name, before in snapshot.items():
        np.testing.assert_array_equal(getattr(model, name), before, err_msg=name)


def test_place_cube_rejects_non_finite_pose(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    with pytest.raises(ValueError):
        S.place_cube(model, data, math.nan, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cube_size": 0.0},
        {"cube_mass": -1.0},
        {"timestep": 0.0},
        {"cube_friction": (1.0, -0.1, 0.0)},
        {"cube_solref": (0.02, 0.0)},
        {"cube_xy": (5.0, 0.0)},
        {"place_xy": (0.0, 3.0)},
        {"cube_xy": (math.nan, 0.0)},
        {"cube_xy": (0.0, 0.0)},  # inside the robot base
        {"cube_xy": (0.85, 0.0)},  # on the edge: the footprint hangs off
        {"place_xy": (0.05, 0.0)},  # inside the base keep-out
        {"cube_size": 1.0},
        {"cube_solref": (0.001, 1.0)},
        {"table_half_xy": (0.55, -0.7)},
        {"table_height": 0.04},  # not taller than the table top thickness
        {"offscreen_size": (0, 960)},
        {"cube_mass": True},
        {"table_center_xy": (0.5, 0.1), "table_half_xy": (0.35, 0.3)},  # base off the table
        {"offscreen_size": (640.5, 480)},
        {"offscreen_size": (10**12, 960)},
        {"cube_rgba": (1.0, 0.0, 0.0)},
        {"cube_rgba": (1.0, 0.0, 0.0, math.nan)},
        {"cube_rgba": "abcd"},
    ],
)
def test_scene_config_rejects_bad_values(kwargs) -> None:
    with pytest.raises(ValueError):
        S.SceneConfig(**kwargs)


def test_scene_config_is_frozen_and_normalized() -> None:
    import dataclasses

    cfg = S.SceneConfig(cube_size=np.float32(0.06), offscreen_size=(np.int64(640), 480))
    assert isinstance(cfg.cube_size, float) and cfg.offscreen_size == (640, 480)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cube_size = 5.0  # type: ignore[misc]
    listy = S.SceneConfig(cube_rgba=[0.15, 0.45, 0.85, 1.0])  # tomllib gives lists
    assert listy == S.SceneConfig() and hash(listy) == hash(S.SceneConfig())
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, cube_size=5.0)  # replace re-validates


def test_resize_rejects_a_footprint_that_would_leave_the_table() -> None:
    edge_x = 0.3 + 0.55 - 0.025 * math.sqrt(2.0)
    model = S.build_scene(S.SceneConfig(cube_xy=(edge_x, 0.0)))
    snapshot = {name: getattr(model, name).copy() for name in _HOOK_FIELDS}
    with pytest.raises(ValueError):
        S.set_cube_size(model, 0.07)
    for name, before in snapshot.items():
        np.testing.assert_array_equal(getattr(model, name), before, err_msg=name)


def test_solref_accepts_direct_mode_and_respects_refsafe() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    S.set_cube_solref(model, (-2000.0, 0.0))  # direct (-stiffness, -damping), zero damping
    np.testing.assert_array_equal(model.geom_solref[ids.cube_geom], [-2000.0, 0.0])
    with pytest.raises(ValueError):
        S.set_cube_solref(model, (0.002, 1.0))  # refsafe on: MuJoCo would clamp to 0.004
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_REFSAFE
    S.set_cube_solref(model, (0.002, 1.0))  # refsafe off: MuJoCo uses it as given
    np.testing.assert_array_equal(model.geom_solref[ids.cube_geom], [0.002, 1.0])
    discrete = S.build_scene()
    discrete.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    S.set_cube_solref(discrete, (0.003, 1.0))  # no clamp under the discrete integrator


def test_scene_config_accepts_a_cube_footprint_touching_the_table_edge() -> None:
    half = 0.025
    edge_x = 0.3 + 0.55 - half * math.sqrt(2.0)  # table centre + half extent - reach
    cfg = S.SceneConfig(cube_xy=(edge_x, 0.0), cube_mass=np.float32(0.2))
    assert isinstance(cfg.cube_mass, float)
    model = S.build_scene(cfg)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    assert data.qpos[S.scene_ids(model).cube_qpos] == pytest.approx(edge_x)


# ------------------------------------------------------------------ touch sensors


def test_touch_sensors_match_the_contact_helper_during_a_squeeze() -> None:
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    # Gravity off: the cube floats between the open fingers, faces aligned with
    # the pads, so only the squeeze acts on it.
    model.opt.gravity[:] = 0.0
    q = ids.cube_qpos
    data.qpos[q : q + 3] = data.site_xpos[ids.ee_site]
    data.qpos[q + 3 : q + 7] = (1.0, 0.0, 0.0, 0.0)
    data.ctrl[ids.gripper_actuator] = 0.0  # close
    for _ in range(150):
        mujoco.mj_step1(model, data)
        data.ctrl[ids.arm_actuators] = data.qfrc_bias[ids.arm_dof]
        mujoco.mj_step2(model, data)
    mujoco.mj_forward(model, data)
    touch = S.read_finger_touch(model, data)
    helper = S.finger_cube_forces(model, data)
    assert min(touch) > 0.5, f"both fingers must squeeze the cube, got {touch}"
    np.testing.assert_allclose(touch, helper, rtol=1e-9, atol=1e-9)
    assert S.finger_cube_forces(model, data, ids) == helper
    # Every finger-cube contact point lies inside its pad's touch-site box.
    sites = {S.LEFT_TOUCH_SITE: ids.left_finger_geoms, S.RIGHT_TOUCH_SITE: ids.right_finger_geoms}
    checked = 0
    for i in range(data.ncon):
        con = data.contact[i]
        if ids.cube_geom not in (con.geom1, con.geom2):
            continue
        finger_geom = con.geom1 if con.geom2 == ids.cube_geom else con.geom2
        site = next(model.site(n).id for n, geoms in sites.items() if finger_geom in geoms)
        local = data.site_xmat[site].reshape(3, 3).T @ (con.pos - data.site_xpos[site])
        assert np.all(np.abs(local) <= model.site_size[site]), local
        checked += 1
    assert checked >= 2
    # The fingers stop at the 5 cm cube: opening ≈ cube size.
    assert data.qpos[ids.finger_qpos].sum() == pytest.approx(0.05, abs=0.002)


def test_touch_counts_any_object_but_the_helper_counts_only_the_cube() -> None:
    """Touch counts pad contacts with any object; finger_cube_forces counts only the cube."""
    spec = S.build_spec()
    other = spec.worldbody.add_body(name="other", pos=[0.5, 0.4, 0.3])
    other.add_freejoint()
    other.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025] * 3, mass=0.1)
    model = spec.compile()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    model.opt.gravity[:] = 0.0
    q = model.jnt_qposadr[model.body("other").jntadr[0]]
    data.qpos[q : q + 3] = data.site_xpos[ids.ee_site]
    data.ctrl[ids.gripper_actuator] = 0.0
    for _ in range(150):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert min(S.read_finger_touch(model, data, ids)) > 0.5
    assert S.finger_cube_forces(model, data, ids) == (0.0, 0.0)


def test_touch_reads_a_fingertip_press_on_the_table_but_the_helper_does_not() -> None:
    """Lower the open gripper onto the table with a simple Cartesian impedance law."""
    model = S.build_scene()
    ids = S.scene_ids(model)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    S.place_cube(model, data, 0.3, 0.4)  # out of the way
    ee0 = data.site_xpos[ids.ee_site].copy()
    rot0 = data.site_xmat[ids.ee_site].reshape(3, 3).copy()
    target = np.array([ee0[0], ee0[1], -0.01])  # 1 cm below the table top
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    quat = np.zeros(4)
    arm = ids.arm_dof
    for k in range(1200):
        mujoco.mj_step1(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, ids.ee_site)
        rot = data.site_xmat[ids.ee_site].reshape(3, 3)
        mujoco.mju_mat2Quat(quat, (rot0 @ rot.T).flatten())
        ori_err = 2.0 * quat[1:] * np.sign(quat[0])
        goal = ee0 + min(1.0, k / 800) * (target - ee0)
        force = 600.0 * (goal - data.site_xpos[ids.ee_site]) - 60.0 * (jacp @ data.qvel)
        moment = 30.0 * ori_err - 3.0 * (jacr @ data.qvel)
        tau = jacp[:, arm].T @ force + jacr[:, arm].T @ moment + data.qfrc_bias[arm]
        data.ctrl[ids.arm_actuators] = tau - data.qvel[arm]
        mujoco.mj_step2(model, data)
    mujoco.mj_forward(model, data)
    touch = S.read_finger_touch(model, data, ids)
    assert min(touch) > 1.0, touch
    assert S.finger_cube_forces(model, data, ids) == (0.0, 0.0)


def test_touch_under_reads_a_fingertip_press_on_the_cube_top() -> None:
    """Fingertip-end contacts sit on the finger mesh, mostly outside the pad sites."""
    model = S.build_scene()
    ids = S.scene_ids(model)
    S.set_cube_size(model, 0.12)
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    model.opt.gravity[:] = 0.0
    ee = data.site_xpos[ids.ee_site].copy()
    q = ids.cube_qpos
    data.qpos[q : q + 3] = [ee[0], ee[1], ee[2] - 0.012 - 0.06]  # top just below the tips
    data.xfrc_applied[ids.cube_body] = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0]  # push up into them
    for _ in range(60):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    touch = S.read_finger_touch(model, data, ids)
    helper = S.finger_cube_forces(model, data, ids)
    assert min(touch) > 0.5
    assert all(h > 2.0 * t for h, t in zip(helper, touch, strict=True)), (touch, helper)


# ---------------------------------------------------------------- render and CLI


@pytest.mark.render
def test_render_png_shows_scene(tmp_path, model: mujoco.MjModel) -> None:
    from matplotlib import image as mpimg

    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    path = scene_check.render_png(model, data, tmp_path / "sub" / "scene.png", 320, 240)
    pixels = mpimg.imread(path)
    assert pixels.shape[:2] == (240, 320)
    assert pixels[..., :3].std() > 0.05, "image is blank"
    rgb = pixels[..., :3]
    cube_blue = (rgb[..., 2] > 0.6) & (rgb[..., 0] < 0.35)
    assert cube_blue.sum() > 20, "the blue cube is not visible"


@pytest.mark.render
def test_scene_check_cli_end_to_end(tmp_path, capsys) -> None:
    png = tmp_path / "scene.png"
    assert cli.main(["scene-check", "--png", str(png), "--seconds", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "|F| =" in out and "ee drift" in out
    assert png.is_file() and png.stat().st_size > 1000


def test_scene_check_cli_no_render(capsys) -> None:
    assert scene_check.main(["--no-render", "--seconds", "0.1"]) == 0
    assert "png:" not in capsys.readouterr().out


@pytest.mark.parametrize("seconds", ["nan", "inf", "-1", "0", "61"])
def test_cli_rejects_bad_seconds(seconds: str) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["scene-check", "--no-render", "--seconds", seconds])
    assert exc.value.code == 2


def test_cli_rejects_a_bad_png_path_before_simulating(tmp_path, capsys, monkeypatch) -> None:
    def must_not_build(*_args, **_kwargs):
        raise AssertionError("the scene was built before the PNG path was checked")

    monkeypatch.setattr(scene_check, "build_scene", must_not_build)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    for png in (tmp_path, tmp_path / "scene.jpg", blocker / "sub" / "scene.png"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["scene-check", "--png", str(png)])
        assert exc.value.code == 2
        assert "--png" in capsys.readouterr().err


def test_cli_lists_scene_check_as_implemented(capsys) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "scene-check" in ln)
    assert "not implemented" not in line
