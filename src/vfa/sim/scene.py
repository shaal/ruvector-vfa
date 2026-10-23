"""Panda + table + cube scene, built in code from the pristine Menagerie Panda.

The vendored ``panda.xml`` is never edited (ARCHITECTURE D1). ``build_spec``
loads it into a ``mujoco.MjSpec`` and edits the spec:

- the seven arm position servos become torque ``motor`` actuators
  (limits 87/87/87/87/12/12/12 N·m, ``ARM_TORQUE_LIMITS``);
- the gripper actuator (``actuator8``, on the menagerie ``split`` tendon) becomes a
  force ``motor``: ctrl = tendon force [N], ±100 N, positive opens; each finger
  gets half of it (T05: the menagerie position servo cannot squeeze harder than
  25·width N per finger, 1.25 N on a 5 cm cube). ``SceneConfig(gripper="servo")``
  keeps the pristine servo (ctrl 0..255 maps to 0..0.08 m opening);
- a table (top surface at world ``z = 0``; the robot base sits on it at the
  origin, so world frame = robot base frame), a floor, lights, and an
  ``overview`` camera;
- a free-joint ``cube`` (default 5 cm, 0.1 kg). Its geom has ``priority=1`` and
  ``condim=4`` (ARCHITECTURE D8), so the cube's friction and ``solref`` govern
  every contact it makes;
- sites: ``ee`` (fingertip center, the Franka TCP 0.1034 m below the flange),
  ``wrist_ft`` (the link7 → hand flange), ``place_target`` (on the table), and
  one touch site per fingertip pad;
- sensors: ``wrist_force``, ``wrist_torque``, and (T06) the ``wrist_acc``
  accelerometer on ``wrist_ft``; ``left_touch`` and ``right_touch``; (T06) the
  ``ee`` site's world pose and velocity (``ee_pos``, ``ee_quat``, ``ee_linvel``,
  ``ee_angvel``) and the gripper tendon's length and rate (``gripper_pos``,
  ``gripper_vel``; the opening is twice the length). The sensors fill one
  contiguous block of ``sensordata`` in this order (``SENSOR_NAMES``), so
  ``vfa.sim.sensors`` reads them with one copy.

Touch vs. cube force: a ``touch`` sensor counts a contact on its finger when
MuJoCo's ray test (contact point along the contact normal) hits the pad site,
whatever the other geom is (cube, table, another object). Face-on pad contacts,
as in a grasp, are counted in full. Contacts at the fingertip end on the finger
mesh (pressing down on a cube top or the table) mostly miss the pad site, so
touch under-reads them. ``finger_cube_forces`` counts every finger↔cube contact
and nothing else. Use it for cube-specific logic such as T04's break rule.

Wrist F/T convention (MuJoCo ``force``/``torque`` sensors): the reading is the
interaction wrench between ``link7`` and the hand subtree (hand, fingers, and
anything they hold), expressed in the ``wrist_ft`` site frame (= hand frame,
whose +z points from the flange toward the fingertips). At rest its magnitude
equals the hand subtree's weight; T05/T06 tare it.

Per-episode hooks (``set_cube_mass``, ``set_cube_friction``, ``set_cube_size``,
``set_cube_solref``, ``place_cube``) edit a compiled ``MjModel`` in place and
keep every derived constant consistent with a fresh compile, so an episode
loop never has to recompile. T04's ``apply_object`` writes through them.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# Repo root: src/vfa/sim/scene.py → parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
PANDA_DIR = REPO_ROOT / "assets" / "menagerie" / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"

ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS: tuple[str, ...] = tuple(f"motor{i}" for i in range(1, 8))
ARM_TORQUE_LIMITS: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
GRIPPER_ACTUATOR = "actuator8"
# Gripper actuator kinds (``SceneConfig.gripper``). "motor" (default, T05): a force
# motor on the tendon, ctrl = tendon force [N] within ±GRIPPER_FORCE_LIMIT, positive
# opens. "servo": the pristine menagerie position servo (ctrl 0..255 = 0..0.08 m).
GRIPPER_KINDS: tuple[str, ...] = ("motor", "servo")
GRIPPER_FORCE_LIMIT = 100.0  # N on the tendon: up to 50 N squeeze per finger
GRIPPER_OPEN_CTRL = 255.0  # servo ctrl for a fully open gripper (the home keyframe)
FINGER_JOINTS: tuple[str, str] = ("finger_joint1", "finger_joint2")

# Menagerie "home" keyframe arm configuration (the ``home`` key's first 7 qpos).
HOME_ARM_QPOS: tuple[float, ...] = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
HOME_FINGER_QPOS: tuple[float, float] = (0.04, 0.04)

HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
CUBE_JOINT = "cube_joint"
TABLE_GEOM = "table"
EE_SITE = "ee"
WRIST_SITE = "wrist_ft"
PLACE_SITE = "place_target"
LEFT_TOUCH_SITE = "left_pad"
RIGHT_TOUCH_SITE = "right_pad"
WRIST_FORCE_SENSOR = "wrist_force"
WRIST_TORQUE_SENSOR = "wrist_torque"
LEFT_TOUCH_SENSOR = "left_touch"
RIGHT_TOUCH_SENSOR = "right_touch"
WRIST_ACC_SENSOR = "wrist_acc"
EE_POS_SENSOR = "ee_pos"
EE_QUAT_SENSOR = "ee_quat"
EE_LINVEL_SENSOR = "ee_linvel"
EE_ANGVEL_SENSOR = "ee_angvel"
GRIPPER_POS_SENSOR = "gripper_pos"
GRIPPER_VEL_SENSOR = "gripper_vel"
GRIPPER_TENDON = "split"
# Every scene sensor, in model order: they fill sensordata[0:SENSOR_DIM] contiguously.
SENSOR_NAMES: tuple[str, ...] = (
    WRIST_FORCE_SENSOR,
    WRIST_TORQUE_SENSOR,
    LEFT_TOUCH_SENSOR,
    RIGHT_TOUCH_SENSOR,
    WRIST_ACC_SENSOR,
    EE_POS_SENSOR,
    EE_QUAT_SENSOR,
    EE_LINVEL_SENSOR,
    EE_ANGVEL_SENSOR,
    GRIPPER_POS_SENSOR,
    GRIPPER_VEL_SENSOR,
)
OVERVIEW_CAMERA = "overview"
HOME_KEY = "home"

# Ratio of frictional to normal constraint impedance (MuJoCo ``opt.impratio``).
IMPRATIO = 10.0

# Franka TCP: fingertip center, 0.1034 m along the hand z axis from the flange.
EE_OFFSET = 0.1034
# Table top surface height in the world (= robot base) frame.
TABLE_TOP_Z = 0.0
# Touch-site box in the finger frame. It covers all five menagerie fingertip pad
# boxes (``fingertip_pad_collision_1..5``: x in ±0.0085, y in [0, 0.0095],
# z in [0.036, 0.0535]) with ≥ 1.5 mm to spare. Face-on contacts with a cube
# land on the small pads 2–5, whose inner face is at y = 0; MuJoCo puts the
# contact point mid-penetration, slightly at y < 0, so the box reaches y = -2 mm.
_TOUCH_POS = (0.0, 0.0045, 0.0445)
_TOUCH_HALF = (0.01, 0.0065, 0.0105)
# Cube edge length limits [m]. Below 5 mm box contacts get unreliable; above
# 30 cm the cube no longer fits beside the robot on the table.
CUBE_SIZE_RANGE: tuple[float, float] = (0.005, 0.3)
# No cube footprint may come closer to the base axis than this [m]. The link0
# collision mesh reaches 0.155 m from the axis.
BASE_KEEPOUT_RADIUS = 0.16


@dataclass(slots=True, frozen=True)
class SceneConfig:
    """Build-time scene parameters (immutable, validated once). Units: m, kg, s.

    ``cube_size`` is the full edge length (5 cm default), not MuJoCo's half-size.
    ``cube_friction`` is MuJoCo's (sliding, torsional, rolling) triple.
    ``cube_solref`` is MuJoCo's (timeconst, dampratio) pair; smaller timeconst =
    stiffer contact.
    """

    timestep: float = 0.002
    cube_size: float = 0.05
    cube_mass: float = 0.1
    cube_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    cube_solref: tuple[float, float] = (0.02, 1.0)
    cube_xy: tuple[float, float] = (0.5, 0.0)
    cube_rgba: tuple[float, float, float, float] = (0.15, 0.45, 0.85, 1.0)
    place_xy: tuple[float, float] = (0.5, 0.25)
    # Table top: half extents (x, y) and center (x, y). The top surface is at z=0.
    table_half_xy: tuple[float, float] = (0.55, 0.7)
    table_center_xy: tuple[float, float] = (0.3, 0.0)
    table_thickness: float = 0.05
    table_height: float = 0.75  # floor is at z = -table_height
    offscreen_size: tuple[int, int] = (1280, 960)  # (width, height) framebuffer
    gripper: str = "motor"  # one of GRIPPER_KINDS

    def __post_init__(self) -> None:
        def put(name: str, value: object) -> None:
            object.__setattr__(self, name, value)  # frozen: normalize once, here

        if not isinstance(self.gripper, str) or self.gripper not in GRIPPER_KINDS:
            raise ValueError(f"gripper must be one of {GRIPPER_KINDS}, got {self.gripper!r}")

        put("timestep", _positive("timestep", self.timestep))
        put("cube_size", _cube_size(self.cube_size))
        put("cube_mass", _positive("cube_mass", self.cube_mass))
        put("table_thickness", _positive("table_thickness", self.table_thickness))
        put("table_height", _positive("table_height", self.table_height))
        if self.table_height <= self.table_thickness:
            raise ValueError("table_height must exceed table_thickness")
        put("cube_friction", _friction_triple(self.cube_friction))
        put("cube_solref", _solref_pair(self.cube_solref, self.timestep))
        put("cube_xy", _pair("cube_xy", self.cube_xy))
        put("place_xy", _pair("place_xy", self.place_xy))
        put("table_center_xy", _pair("table_center_xy", self.table_center_xy))
        hx, hy = _pair("table_half_xy", self.table_half_xy)
        put("table_half_xy", (_positive("table_half_xy", hx), _positive("table_half_xy", hy)))
        put("cube_rgba", _rgba(self.cube_rgba))
        size = _positive_ints("offscreen_size", self.offscreen_size, 2)
        if max(size) > _MAX_OFFSCREEN:
            raise ValueError(f"offscreen_size must be <= {_MAX_OFFSCREEN} px, got {size!r}")
        put("offscreen_size", size)
        cx, cy = self.table_center_xy
        if abs(cx) + BASE_KEEPOUT_RADIUS > hx or abs(cy) + BASE_KEEPOUT_RADIUS > hy:
            raise ValueError("the table must hold the robot base (keep-out circle at the origin)")
        _check_cube_xy(
            "cube_xy", self.cube_xy, 0.5 * self.cube_size, self.table_center_xy, self.table_half_xy
        )
        _check_cube_xy("place_xy", self.place_xy, 0.0, self.table_center_xy, self.table_half_xy)


@dataclass(slots=True, frozen=True)
class SceneIds:
    """Integer ids and addresses resolved once per compiled model."""

    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    arm_actuators: np.ndarray
    gripper_actuator: int
    finger_qpos: np.ndarray
    hand_body: int
    cube_body: int
    cube_geom: int
    cube_qpos: int  # start of the cube's 7-wide free-joint qpos block
    cube_dof: int  # start of the cube's 6-wide free-joint dof block
    table_geom: int
    ee_site: int
    wrist_site: int
    place_site: int
    wrist_force_adr: int
    wrist_torque_adr: int
    left_touch_adr: int
    right_touch_adr: int
    wrist_acc_adr: int
    ee_pos_adr: int
    ee_quat_adr: int
    ee_linvel_adr: int
    ee_angvel_adr: int
    gripper_pos_adr: int
    gripper_vel_adr: int
    left_finger_geoms: frozenset[int]
    right_finger_geoms: frozenset[int]
    # Per geom (ngeom + 1 entries; the last one stands for "no geom"): 1 = a left
    # finger geom, 2 = a right finger geom, 0 = anything else.
    finger_side: np.ndarray


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    """Load the pristine Panda and edit it into the VFA scene spec (not compiled)."""
    cfg = cfg if cfg is not None else SceneConfig()
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.modelname = "panda_vfa"
    spec.option.timestep = cfg.timestep
    # T05: elliptic friction cones with impratio 10 (MuJoCo's advice for grasping).
    # With the default pyramidal cones and impratio 1 a held cube slid 2.9-9.4 mm
    # in the grip over a 2 s lift and hold at 1.1x the grip it needs (1.5-5.5 mm at
    # 2x); with these, 0.12-0.47 mm, and below the needed grip it still drops
    # (ARCHITECTURE §3, D11).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO
    spec.visual.global_.offwidth = cfg.offscreen_size[0]
    spec.visual.global_.offheight = cfg.offscreen_size[1]

    _arm_servos_to_torque_motors(spec)
    if cfg.gripper == "motor":
        _gripper_servo_to_force_motor(spec)
    # T05: the two fingers never collide with each other. A squeeze on nothing
    # otherwise drove their collision meshes into each other, where they jammed
    # (T03: -11 mm at 5 N, -49 mm at 20 N, stuck after open()). The finger joint
    # limits (0..0.04 m) and the Gripper end stop still stop them.
    spec.add_exclude(bodyname1=LEFT_FINGER_BODY, bodyname2=RIGHT_FINGER_BODY)
    _add_environment(spec, cfg)
    _add_cube(spec, cfg)
    _add_robot_sites_and_sensors(spec)
    _set_home_key(spec, cfg)
    return spec


def build_scene(cfg: SceneConfig | None = None) -> mujoco.MjModel:
    """Build and compile the VFA scene. Reset with ``reset_to_home``."""
    return build_spec(cfg).compile()


def scene_ids(model: mujoco.MjModel) -> SceneIds:
    """Resolve the names this package relies on into ids and addresses."""

    def oid(objtype: mujoco.mjtObj, name: str) -> int:
        i = mujoco.mj_name2id(model, objtype, name)
        if i < 0:
            raise KeyError(f"{objtype.name} {name!r} not found in model")
        return int(i)

    def sensor_adr(name: str) -> int:
        return int(model.sensor_adr[oid(mujoco.mjtObj.mjOBJ_SENSOR, name)])

    def body_geoms(body: int) -> frozenset[int]:
        start, n = int(model.body_geomadr[body]), int(model.body_geomnum[body])
        return frozenset(range(start, start + n))

    jnt = mujoco.mjtObj.mjOBJ_JOINT
    arm_joints = [oid(jnt, n) for n in ARM_JOINTS]
    cube_joint = oid(jnt, CUBE_JOINT)
    left = body_geoms(oid(mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER_BODY))
    right = body_geoms(oid(mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER_BODY))
    side = np.zeros(model.ngeom + 1, dtype=np.int8)
    side[sorted(left)] = 1
    side[sorted(right)] = 2
    return SceneIds(
        arm_qpos=np.array([model.jnt_qposadr[j] for j in arm_joints], dtype=np.int64),
        arm_dof=np.array([model.jnt_dofadr[j] for j in arm_joints], dtype=np.int64),
        arm_actuators=np.array(
            [oid(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS], dtype=np.int64
        ),
        gripper_actuator=oid(mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR),
        finger_qpos=np.array(
            [model.jnt_qposadr[oid(jnt, n)] for n in FINGER_JOINTS], dtype=np.int64
        ),
        hand_body=oid(mujoco.mjtObj.mjOBJ_BODY, HAND_BODY),
        cube_body=oid(mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY),
        cube_geom=oid(mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM),
        cube_qpos=int(model.jnt_qposadr[cube_joint]),
        cube_dof=int(model.jnt_dofadr[cube_joint]),
        table_geom=oid(mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM),
        ee_site=oid(mujoco.mjtObj.mjOBJ_SITE, EE_SITE),
        wrist_site=oid(mujoco.mjtObj.mjOBJ_SITE, WRIST_SITE),
        place_site=oid(mujoco.mjtObj.mjOBJ_SITE, PLACE_SITE),
        wrist_force_adr=sensor_adr(WRIST_FORCE_SENSOR),
        wrist_torque_adr=sensor_adr(WRIST_TORQUE_SENSOR),
        left_touch_adr=sensor_adr(LEFT_TOUCH_SENSOR),
        right_touch_adr=sensor_adr(RIGHT_TOUCH_SENSOR),
        wrist_acc_adr=sensor_adr(WRIST_ACC_SENSOR),
        ee_pos_adr=sensor_adr(EE_POS_SENSOR),
        ee_quat_adr=sensor_adr(EE_QUAT_SENSOR),
        ee_linvel_adr=sensor_adr(EE_LINVEL_SENSOR),
        ee_angvel_adr=sensor_adr(EE_ANGVEL_SENSOR),
        gripper_pos_adr=sensor_adr(GRIPPER_POS_SENSOR),
        gripper_vel_adr=sensor_adr(GRIPPER_VEL_SENSOR),
        left_finger_geoms=left,
        right_finger_geoms=right,
        finger_side=side,
    )


# --------------------------------------------------------------------------- reset


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the ``home`` keyframe (arm home, gripper open, cube on the table)."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
    if key < 0:
        raise KeyError(f"model has no {HOME_KEY!r} keyframe; build it with build_scene()")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


# The read helpers below take an optional ``ids``. Resolving names costs more
# than one physics step, so control loops pass ``ids = scene_ids(model)`` once.


def gravity_compensation(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> np.ndarray:
    """Arm torques that cancel gravity + Coriolis at the current state (``qfrc_bias``).

    Valid after ``mj_forward`` or ``mj_step1`` on the current state.
    """
    ids = ids if ids is not None else scene_ids(model)
    return data.qfrc_bias[ids.arm_dof].copy()


# ------------------------------------------------------------------------- sensors


def read_wrist_ft(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (untared) wrist force [N] and torque [N·m] in the ``wrist_ft`` site frame."""
    ids = ids if ids is not None else scene_ids(model)
    force = data.sensordata[ids.wrist_force_adr : ids.wrist_force_adr + 3].copy()
    torque = data.sensordata[ids.wrist_torque_adr : ids.wrist_torque_adr + 3].copy()
    return force, torque


def read_wrist_acc(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> np.ndarray:
    """Wrist accelerometer [m/s²] in the ``wrist_ft`` site frame: proper acceleration.

    MuJoCo's ``accelerometer`` includes gravity, as a real one does: at rest it
    reads ``+g`` along the world's up axis (with the hand pointing down, that is
    the site's −z). T05's held-mass slip detector divides the vertical wrist
    force by its vertical component.
    """
    ids = ids if ids is not None else scene_ids(model)
    return data.sensordata[ids.wrist_acc_adr : ids.wrist_acc_adr + 3].copy()


def read_finger_touch(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> tuple[float, float]:
    """Left/right fingertip-pad touch sensor readings [N] (summed normal force).

    Counts finger contacts whose ray test hits the pad site, with any object
    (cube, table, ...). Exact for face-on pad grasps; under-reads fingertip-end
    presses on the finger mesh. For the complete cube-only force use
    ``finger_cube_forces``.
    """
    ids = ids if ids is not None else scene_ids(model)
    return float(data.sensordata[ids.left_touch_adr]), float(data.sensordata[ids.right_touch_adr])


def finger_cube_forces(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> tuple[float, float]:
    """Summed contact normal force [N] between each finger and the cube.

    Contact helper that does not depend on the touch sites: it sums the normal
    force of every finger-geom / cube-geom contact, pads and finger mesh alike.
    T04's break rule uses the sum of both. With elliptic friction cones (the
    scene's, since T05) a contact's normal force is the first entry of its
    constraint force block, which is what ``mj_contactForce`` returns; the
    contacts are filtered as arrays and read in one gather (T06: 55 → ≈ 13 µs
    with 36 contacts in a grasp). Other cones go through ``mj_contactForce``.
    A contact without constraint rows (``efc_address < 0``) carries no force.
    """
    ids = ids if ids is not None else scene_ids(model)
    n = data.ncon
    if n == 0:
        return 0.0, 0.0
    geom = data.contact.geom[:n]
    cube = ids.cube_geom
    other = np.where(geom[:, 0] == cube, geom[:, 1], np.where(geom[:, 1] == cube, geom[:, 0], -1))
    side = ids.finger_side[other]  # other = -1 reads the trailing "no geom" entry (0)
    hits = np.flatnonzero(side)
    if hits.size == 0:
        return 0.0, 0.0
    adr = data.contact.efc_address[:n][hits]
    if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        normal = np.where(adr >= 0, data.efc_force[np.maximum(adr, 0)], 0.0)
    else:
        normal = np.zeros(hits.size)
        wrench = np.zeros(6)
        for k, i in enumerate(hits):
            mujoco.mj_contactForce(model, data, int(i), wrench)
            normal[k] = wrench[0]
    # Plain left-to-right sums in contact order: the same rounding as the T02-T05
    # loop (Python's sum() compensates since 3.12 and would differ by an ulp).
    left = right = 0.0
    for force, is_left in zip(normal.tolist(), (side[hits] == 1).tolist(), strict=True):
        if is_left:
            left += force
        else:
            right += force
    return left, right


def table_cube_force(
    model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
) -> float:
    """Summed contact normal force [N] between the table and the cube (T19's impact
    rule; read as ``finger_cube_forces`` reads the fingers')."""
    ids = ids if ids is not None else scene_ids(model)
    n = data.ncon
    if n == 0:
        return 0.0
    geom = data.contact.geom[:n]
    cube, table = ids.cube_geom, ids.table_geom
    hits = np.flatnonzero(
        ((geom[:, 0] == cube) & (geom[:, 1] == table))
        | ((geom[:, 1] == cube) & (geom[:, 0] == table))
    )
    if hits.size == 0:
        return 0.0
    adr = data.contact.efc_address[:n][hits]
    if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        normal = np.where(adr >= 0, data.efc_force[np.maximum(adr, 0)], 0.0)
    else:
        normal = np.zeros(hits.size)
        wrench = np.zeros(6)
        for k, i in enumerate(hits):
            mujoco.mj_contactForce(model, data, int(i), wrench)
            normal[k] = wrench[0]
    total = 0.0
    for force in normal.tolist():
        total += force
    return total


# ------------------------------------------------------------- per-episode hooks


def set_cube_mass(model: mujoco.MjModel, mass: float) -> None:
    """Set the cube mass [kg]; updates its inertia and the derived constants."""
    mass = _positive("mass", mass)
    ids = scene_ids(model)
    model.body_mass[ids.cube_body] = mass
    _refresh_cube_inertia(model, ids)
    _refresh_constants(model)


def set_cube_friction(model: mujoco.MjModel, friction: float | Sequence[float]) -> None:
    """Set the cube friction: a sliding coefficient, or the full (slide, torsion, roll).

    A scalar keeps the current torsional and rolling coefficients. Because the
    cube geom has ``priority=1`` these values govern every cube contact.
    """
    ids = scene_ids(model)
    if _is_scalar(friction):
        slide = _real("friction", friction)
        triple = _friction_triple((slide, *model.geom_friction[ids.cube_geom, 1:]))
    else:
        triple = _friction_triple(friction)
    model.geom_friction[ids.cube_geom] = triple


def set_cube_solref(model: mujoco.MjModel, solref: Sequence[float]) -> None:
    """Set the cube contact softness as MuJoCo ``solref`` (timeconst, dampratio).

    A timeconst below 2 × timestep is rejected, because MuJoCo would silently
    clamp it. MuJoCo skips that clamp when ``refsafe`` is disabled or the
    integrator is ``mjINT_DISCRETE``, and so does this check.
    """
    ids = scene_ids(model)
    model.geom_solref[ids.cube_geom] = _solref_pair(
        solref, float(model.opt.timestep), refsafe=_refsafe(model)
    )


def set_cube_size(model: mujoco.MjModel, size: float) -> None:
    """Set the cube edge length [m], keeping its mass and its resting contact with the table.

    Updates the geom size and bounding volumes, the inertia, and the cube's rest
    height in ``qpos0``, ``qpos_spring``, ``body_pos``, and every keyframe, so a
    reset puts the cube flat on the table. Call ``reset_to_home`` or
    ``place_cube`` afterwards. The size must lie in ``CUBE_SIZE_RANGE``, and the
    resized footprint at the rest pose (``qpos0`` and every keyframe) must stay
    on the table and outside the base keep-out.
    """
    size = _cube_size(size)
    ids = scene_ids(model)
    old_half = float(model.geom_size[ids.cube_geom, 0])
    half = 0.5 * size
    q = ids.cube_qpos
    table_xy, table_half = _table_rect(model, ids)
    for xy in (model.qpos0[q : q + 2], *model.key_qpos[:, q : q + 2]):
        _check_cube_xy("cube rest (x, y)", (float(xy[0]), float(xy[1])), half, table_xy, table_half)
    scale = half / old_half
    model.geom_size[ids.cube_geom] = half
    model.geom_rbound[ids.cube_geom] *= scale
    model.geom_aabb[ids.cube_geom] *= scale
    bvh_start = int(model.body_bvhadr[ids.cube_body])
    bvh_end = bvh_start + int(model.body_bvhnum[ids.cube_body])
    model.bvh_aabb[bvh_start:bvh_end] *= scale
    dz = half - old_half
    z = ids.cube_qpos + 2
    model.qpos0[z] += dz
    model.qpos_spring[z] += dz
    model.body_pos[ids.cube_body, 2] += dz
    if model.nkey:
        model.key_qpos[:, z] += dz
    _refresh_cube_inertia(model, ids)
    _refresh_constants(model)


def place_cube(
    model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, yaw: float = 0.0
) -> None:
    """Put the cube at rest on the table at (x, y) with a yaw angle [rad].

    Zeroes the cube velocity and runs ``mj_forward``. Rejects a pose whose
    footprint leaves the table top or enters the robot-base keep-out.
    """
    x, y, yaw = _real("x", x), _real("y", y), _real("yaw", yaw)
    ids = scene_ids(model)
    half = float(model.geom_size[ids.cube_geom, 0])
    _check_cube_xy("cube (x, y)", (x, y), half, *_table_rect(model, ids))
    q = ids.cube_qpos
    data.qpos[q : q + 3] = (x, y, TABLE_TOP_Z + half)
    data.qpos[q + 3 : q + 7] = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
    data.qvel[ids.cube_dof : ids.cube_dof + 6] = 0.0
    mujoco.mj_forward(model, data)


# ----------------------------------------------------------------------- builders


def _arm_servos_to_torque_motors(spec: mujoco.MjSpec) -> None:
    """Turn menagerie actuator1..7 (position servos) into torque motors, in place.

    Editing in place keeps the actuator order: ctrl[0:7] = arm torques [N·m],
    ctrl[7] = gripper (``_gripper_servo_to_force_motor`` or the menagerie servo).
    """
    limited = mujoco.mjtLimited.mjLIMITED_TRUE
    for i, (joint, limit) in enumerate(zip(ARM_JOINTS, ARM_TORQUE_LIMITS, strict=True)):
        act = spec.actuator(f"actuator{i + 1}")
        if act is None or act.target != joint:
            raise RuntimeError(f"menagerie actuator{i + 1} does not drive {joint}")
        act.set_to_motor()
        act.name = ARM_ACTUATORS[i]
        act.ctrllimited = limited
        act.forcelimited = limited
        act.ctrlrange = [-limit, limit]
        act.forcerange = [-limit, limit]


def _gripper_servo_to_force_motor(spec: mujoco.MjSpec) -> None:
    """Turn the menagerie gripper servo (actuator8) into a force motor on the same tendon.

    ctrl = tendon force [N] (gain 1, no bias), limited to ±GRIPPER_FORCE_LIMIT;
    positive opens. The tendon couples the fingers with coef 0.5, so each finger
    gets half the tendon force. ``vfa.control.gripper.Gripper`` writes it.
    """
    act = spec.actuator(GRIPPER_ACTUATOR)
    if act is None or act.target != GRIPPER_TENDON:
        raise RuntimeError(f"menagerie {GRIPPER_ACTUATOR} does not drive the split tendon")
    limited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.set_to_motor()
    act.ctrllimited = limited
    act.forcelimited = limited
    act.ctrlrange = [-GRIPPER_FORCE_LIMIT, GRIPPER_FORCE_LIMIT]
    act.forcerange = [-GRIPPER_FORCE_LIMIT, GRIPPER_FORCE_LIMIT]


def _add_environment(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.7, 0.85],
        rgb2=[0.1, 0.12, 0.15],
        width=512,
        height=512,
    )
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.32, 0.34, 0.36],
        rgb2=[0.26, 0.28, 0.3],
        width=256,
        height=256,
    )
    floor_mat = spec.add_material(name="floor_mat", texrepeat=[6, 6], reflectance=0.05)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    spec.add_material(name="table_mat", rgba=[0.62, 0.5, 0.38, 1.0], specular=0.1)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    world = spec.worldbody
    world.add_light(name="key", pos=[1.0, -1.0, 2.0], dir=[-0.4, 0.4, -1.0], castshadow=True)
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, TABLE_TOP_Z - cfg.table_height],
        material="floor_mat",
    )
    (cx, cy), (hx, hy) = cfg.table_center_xy, cfg.table_half_xy
    half_t = 0.5 * cfg.table_thickness
    world.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hx, hy, half_t],
        pos=[cx, cy, TABLE_TOP_Z - half_t],
        material="table_mat",
    )
    # Visual-only legs.
    leg_half_h = 0.5 * (cfg.table_height - cfg.table_thickness)
    leg_z = TABLE_TOP_Z - cfg.table_thickness - leg_half_h
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.03, 0.03, leg_half_h],
                pos=[cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), leg_z],
                material="table_mat",
                contype=0,
                conaffinity=0,
            )
    world.add_site(
        name=PLACE_SITE,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.035, 0.0005, 0.0],
        pos=[cfg.place_xy[0], cfg.place_xy[1], TABLE_TOP_Z + 0.0005],
        rgba=[0.1, 0.8, 0.2, 0.5],
    )
    eye = np.array([1.45, -1.05, 0.75])
    target = np.array([0.3, 0.0, 0.2])
    world.add_camera(name=OVERVIEW_CAMERA, pos=eye.tolist(), quat=_look_at_quat(eye, target))


def _add_cube(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    half = 0.5 * cfg.cube_size
    body = spec.worldbody.add_body(
        name=CUBE_BODY, pos=[cfg.cube_xy[0], cfg.cube_xy[1], TABLE_TOP_Z + half]
    )
    body.add_freejoint(name=CUBE_JOINT)
    body.add_geom(
        name=CUBE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, half],
        mass=cfg.cube_mass,
        friction=list(cfg.cube_friction),
        solref=list(cfg.cube_solref),
        priority=1,  # ARCHITECTURE D8: the cube's contact params win
        condim=4,  # torsional friction, so twisting slip is modeled
        rgba=list(cfg.cube_rgba),
    )


def _add_robot_sites_and_sensors(spec: mujoco.MjSpec) -> None:
    hidden = 4  # site group not drawn by the default renderer options
    hand = spec.body(HAND_BODY)
    hand.add_site(name=WRIST_SITE, pos=[0.0, 0.0, 0.0], size=[0.01, 0.0, 0.0], group=hidden)
    hand.add_site(name=EE_SITE, pos=[0.0, 0.0, EE_OFFSET], size=[0.006, 0.0, 0.0], group=hidden)
    for body_name, site_name in (
        (LEFT_FINGER_BODY, LEFT_TOUCH_SITE),
        (RIGHT_FINGER_BODY, RIGHT_TOUCH_SITE),
    ):
        spec.body(body_name).add_site(
            name=site_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(_TOUCH_POS),
            size=list(_TOUCH_HALF),
            group=hidden,
        )

    site, tendon, sens = mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_TENDON, mujoco.mjtSensor
    # Frame sensors without a reference object report in the world frame.
    for name, stype, objtype, obj in (
        (WRIST_FORCE_SENSOR, sens.mjSENS_FORCE, site, WRIST_SITE),
        (WRIST_TORQUE_SENSOR, sens.mjSENS_TORQUE, site, WRIST_SITE),
        (LEFT_TOUCH_SENSOR, sens.mjSENS_TOUCH, site, LEFT_TOUCH_SITE),
        (RIGHT_TOUCH_SENSOR, sens.mjSENS_TOUCH, site, RIGHT_TOUCH_SITE),
        (WRIST_ACC_SENSOR, sens.mjSENS_ACCELEROMETER, site, WRIST_SITE),
        (EE_POS_SENSOR, sens.mjSENS_FRAMEPOS, site, EE_SITE),
        (EE_QUAT_SENSOR, sens.mjSENS_FRAMEQUAT, site, EE_SITE),
        (EE_LINVEL_SENSOR, sens.mjSENS_FRAMELINVEL, site, EE_SITE),
        (EE_ANGVEL_SENSOR, sens.mjSENS_FRAMEANGVEL, site, EE_SITE),
        (GRIPPER_POS_SENSOR, sens.mjSENS_TENDONPOS, tendon, GRIPPER_TENDON),
        (GRIPPER_VEL_SENSOR, sens.mjSENS_TENDONVEL, tendon, GRIPPER_TENDON),
    ):
        spec.add_sensor(name=name, type=stype, objtype=objtype, objname=obj)


def _set_home_key(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    key = spec.key(HOME_KEY)
    if key is None:
        raise RuntimeError("menagerie panda.xml has no 'home' keyframe")
    cube_qpos = [cfg.cube_xy[0], cfg.cube_xy[1], TABLE_TOP_Z + 0.5 * cfg.cube_size, 1, 0, 0, 0]
    key.qpos = [*HOME_ARM_QPOS, *HOME_FINGER_QPOS, *cube_qpos]
    # Torque motors start at zero; callers add gravity compensation every tick. The
    # gripper motor starts with no force (fingers open, at rest); the servo, open.
    gripper = GRIPPER_OPEN_CTRL if cfg.gripper == "servo" else 0.0
    key.ctrl = [0.0] * len(ARM_ACTUATORS) + [gripper]


# ------------------------------------------------------------------------ helpers


def _refsafe(model: mujoco.MjModel) -> bool:
    """True if MuJoCo clamps a solref time constant to 2 × timestep in this model."""
    return not (
        model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE
        or model.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
    )


def _refresh_cube_inertia(model: mujoco.MjModel, ids: SceneIds) -> None:
    """Solid-cube inertia about its center: I = m·(2h)²/6 = (2/3)·m·h² per axis."""
    half = float(model.geom_size[ids.cube_geom, 0])
    model.body_inertia[ids.cube_body] = (2.0 / 3.0) * model.body_mass[ids.cube_body] * half**2


def _refresh_constants(model: mujoco.MjModel) -> None:
    """Recompute mass-dependent constants (subtree mass, invweight0, ...) at qpos0.

    ``mj_setConst`` overwrites the MjData it is given, so it gets a scratch one.
    """
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> list[float]:
    """Quaternion of a camera at ``eye`` looking at ``target`` with world +z up."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # MuJoCo cameras look along their local -z, with +x right and +y up.
    mat = np.column_stack([right, up, -forward])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat.tolist()


def _table_rect(
    model: mujoco.MjModel, ids: SceneIds
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Table-top center (x, y) and half extents (x, y), read from the table geom."""
    g = ids.table_geom
    center = (float(model.geom_pos[g, 0]), float(model.geom_pos[g, 1]))
    half = (float(model.geom_size[g, 0]), float(model.geom_size[g, 1]))
    return center, half


_TEXT = str | bytes | bytearray
_MAX_OFFSCREEN = 16384  # px; far above any useful render, below GL limits that overflow


def _rgba(values: object) -> tuple[float, float, float, float]:
    rgba = _reals("cube_rgba", values, 4)
    if not all(0.0 <= v <= 1.0 for v in rgba):
        raise ValueError(f"cube_rgba entries must be in [0, 1], got {values!r}")
    return rgba  # type: ignore[return-value]


def _is_scalar(value: object) -> bool:
    return not isinstance(value, _TEXT) and np.ndim(value) == 0


def _real(name: str, value: object) -> float:
    """A finite real number (Python or numpy, 0-d arrays too) as ``float``. No bools."""
    if isinstance(value, bool | np.bool_) or not _is_scalar(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    item = value.item() if isinstance(value, np.ndarray) else value
    if not isinstance(item, numbers.Real) or isinstance(item, bool):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(item)
    except OverflowError:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return result


def _positive(name: str, value: object) -> float:
    result = _real(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _cube_size(size: object) -> float:
    result = _real("cube size", size)
    lo, hi = CUBE_SIZE_RANGE
    if not lo <= result <= hi:
        raise ValueError(f"cube size must be in [{lo}, {hi}] m, got {size!r}")
    return result


def _positive_ints(name: str, values: object, n: int) -> tuple[int, ...]:
    if _is_scalar(values) or isinstance(values, _TEXT):
        raise ValueError(f"{name} must be {n} integers > 0, got {values!r}")
    items = list(values)  # type: ignore[call-overload]
    if len(items) != n or not all(
        isinstance(v, numbers.Integral) and not isinstance(v, bool | np.bool_) and v > 0
        for v in items
    ):
        raise ValueError(f"{name} must be {n} integers > 0, got {values!r}")
    return tuple(int(v) for v in items)


def _reals(name: str, values: object, n: int) -> tuple[float, ...]:
    if _is_scalar(values) or isinstance(values, _TEXT):
        raise ValueError(f"{name} must be {n} finite numbers, got {values!r}")
    items = list(values)  # type: ignore[call-overload]
    if len(items) != n:
        raise ValueError(f"{name} must be {n} finite numbers, got {values!r}")
    return tuple(_real(name, v) for v in items)


def _pair(name: str, values: object) -> tuple[float, float]:
    return _reals(name, values, 2)  # type: ignore[return-value]


def _friction_triple(friction: object) -> tuple[float, float, float]:
    values = _reals("friction", friction, 3)
    if values[0] <= 0 or values[1] < 0 or values[2] < 0:
        raise ValueError(f"friction needs slide > 0, torsion >= 0, roll >= 0; got {friction!r}")
    return values  # type: ignore[return-value]


def _solref_pair(solref: object, timestep: float, refsafe: bool = True) -> tuple[float, float]:
    values = _reals("solref", solref, 2)
    # MuJoCo: (timeconst > 0, dampratio > 0), or direct (-stiffness < 0, -damping <= 0).
    standard = values[0] > 0 and values[1] > 0
    direct = values[0] < 0 and values[1] <= 0
    if not (standard or direct):
        raise ValueError(
            f"solref must be (timeconst > 0, dampratio > 0) or (-stiffness < 0, -damping <= 0), "
            f"got {solref!r}"
        )
    if refsafe and standard and values[0] < 2.0 * timestep:
        raise ValueError(
            f"solref timeconst {values[0]} < 2 × timestep ({2.0 * timestep}); "
            "MuJoCo would clamp it silently"
        )
    return values  # type: ignore[return-value]


def _check_cube_xy(
    name: str,
    xy: tuple[float, float],
    half: float,
    table_xy: tuple[float, float],
    table_half: tuple[float, float],
) -> None:
    """Reject a cube footprint (any yaw) that leaves the table or enters the base keep-out."""
    reach = half * math.sqrt(2.0)  # footprint circumradius in xy
    tol = 1e-9  # a footprint exactly on the edge counts as on the table
    if abs(xy[0] - table_xy[0]) + reach > table_half[0] + tol or (
        abs(xy[1] - table_xy[1]) + reach > table_half[1] + tol
    ):
        raise ValueError(f"{name}={xy!r} puts the footprint off the table top")
    if math.hypot(xy[0], xy[1]) - reach < BASE_KEEPOUT_RADIUS:
        raise ValueError(
            f"{name}={xy!r} is inside the robot-base keep-out ({BASE_KEEPOUT_RADIUS} m)"
        )
