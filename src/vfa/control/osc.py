"""Operational-space impedance controller: System 1's low-level torque law.

The controller makes the ``ee`` site (fingertip center) behave like a 6-D
spring-damper attached to a reference pose (ARCHITECTURE §3, D2)::

    F     = K·e + D·(v_ref − ẋ)                       task wrench at the ee site
    τ     = Jᵀ·F + qfrc_bias + Nᵀ·τ_posture            arm joint torques
    τ_cmd = clip(τ, −τ_max, τ_max)                     motor limits (87/87/87/87/12/12/12 N·m)

- ``e`` is the 6-D error from the ee to the reference pose: position [m] and
  orientation as an axis-angle vector [rad] (shortest arc), world frame.
- Reference pose: the caller sets a *target*; the controller keeps a reference
  pose that moves toward it at most ``max_lin_speed`` / ``max_ang_speed``
  (a rate limiter, advanced once per control tick by ``apply``). While it
  chases a far target, ``v_ref = 0``: the damper then resists the motion, the
  ee trails the reference, and it arrives without overshoot. (Feeding the
  chase velocity as ``v_ref`` overshot every long move by ≈ v/(e·ω) ≈ 1 cm and
  put the fingertips on the table at the box floor; measured in T03.) Once the
  reference sits on the target it follows the target exactly, with the
  target's feed-forward velocity as ``v_ref``. The speed limit never caps the
  spring: at rest an external wrench ``W`` at the ee deflects it by ``K⁻¹·W``
  until a motor reaches its torque limit (e.g. a 70 N push along +x
  saturates joint 6).
- ``K`` is the stiffness in physical units (N/m, N·m/rad). Lower ``K`` = softer.
- ``D`` is the damping, a 6×6 matrix. By default it gives every mode of the
  ee's mass-spring system ``Λ·ẍ + D·ẋ + K·x = 0`` the damping ratio ``ζ``
  (``damping_ratio``, default 1 = critical) in every arm configuration, where
  ``Λ = (J·M⁻¹·Jᵀ)⁻¹`` is the task-space inertia. One 6×6 ``eigh`` per tick,
  ``K½·Λ⁻¹·K½ = V·Ω²·Vᵀ``, gives the mode frequencies ``Ω`` and
  ``D = K½·V·(2ζ·Ω⁻¹)·Vᵀ·K½`` (modal "double diagonalization", Albu-Schäffer
  et al. 2003). At ζ = 1 the ee arrives without overshoot. Heavy modes settle
  slowly: at the far-high box corner the slowest mode has ω ≈ 2.9 rad/s, so a
  reach from ``home`` is still ≈ 3 mm / 2.4° off after 2 s. The fastest mode at
  the default gains has ω ≈ 33 rad/s, so ``2ζω·dt`` ≤ 0.67 with torques held
  for 10 ms (100 Hz): stable. History (measured in T03): per-axis damping from
  the apparent mass ``1/(Λ⁻¹)_ii`` underdamped the coupled modes (effective ζ
  0.41–0.96 per axis) and overshot long moves by up to 13 mm; per-axis damping from the
  diagonal of ``Λ`` went unstable at 100–125 Hz at the far-high corners.
  Explicit ``pos_damping`` / ``rot_damping`` replace that block with a diagonal
  and drop its coupling to the other block.
- ``qfrc_bias`` cancels gravity and Coriolis/centrifugal forces. The
  menagerie joint damping (1 N·m·s/rad per joint) is not compensated; it only
  adds damping and does not change any static deflection.
- ``τ_posture = K_null·(q_home − q) − D_null·q̇`` pulls the redundant 7th DOF
  (the elbow) toward ``home``. ``Nᵀ = I − Jᵀ·Λ·J·M⁻¹`` is the dynamically
  consistent null-space projector, so the posture term adds no force at the ee.
- ``J`` comes from ``mj_jacSite`` and ``M`` is the arm block of ``mj_fullM``.

The default target orientation is top-down: the hand z axis points at the table
and the fingers close along world x. An optional yaw turns the hand about world
z. Joint 7's range (±2.8973 rad) bounds the yaw, and not symmetrically: at
yaw +π/2 joint 7 has only ≈ 0.05 rad left at some box corners, at −π/2 it has
≈ 1.6 rad. ``set_target`` accepts ``|yaw| <= π/2 + GRASP_YAW_MARGIN`` (the band
reached at every box corner in T03 with ≥ 0.013 rad of joint-7 headroom). Yaw
2.0 rad (via ``rot=``) runs joint 7 into its limit at the four y = −0.3
corners (up to 16 mm / 13° off after 2 s). ``set_target`` also rejects a yaw more than
π away from the reference's yaw: the reference turns the shortest way, which
would then pass yaw ±π and run joint 7 into its limit (a review found this
after an external twist left the hand at yaw −1.61 and the next target was
+1.56). The parallel gripper is symmetric under a half turn, so
``wrap_grasp_yaw`` maps a grasp yaw into [−π/2, π/2), or, given ``near``, to
the equivalent yaw in the band that is closest to it (the smallest turn).
``at_joint_limit`` reports an arm joint within ``JOINT_LIMIT_MARGIN`` of its
range, where the torque law can stall without saturating a motor.

Rates (ARCHITECTURE §2): physics runs at ``1 / timestep`` (500 Hz). The torque
law runs at ``control_hz`` (default ``CONTROL_HZ``) and its torques are held
across physics substeps, at ``MIN_CONTROL_HZ`` (100 Hz) or faster. Slower
ticks lose the reach accuracy (measured in T03 at the 8 box corners after
2 s: 3.9 mm and 2.9° at 50 Hz, 82 mm and 50° at 25 Hz), so ``substeps_for``
rejects them. ``control_step`` runs one control tick and rejects a controller
whose ``control_hz`` does not match the tick rate.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from vfa.sim.scene import ARM_JOINTS, HOME_ARM_QPOS, SceneIds, _real, _reals, scene_ids

# The OSC rate chosen in T03 (ARCHITECTURE §2): the physics rate, so one control
# tick per physics step. It keeps a 10 s lift-length episode at >= 10x real time
# on one core, with headroom for the per-tick work that later tasks add.
# Measured with ``python -m vfa.control.throughput``.
CONTROL_HZ = 500

# Slowest allowed control tick (ARCHITECTURE §2: 100–500 Hz). The reach tests
# pass at 100 Hz (box corners <= 0.58 mm); at 50 Hz they end 3.9 mm off, at
# 25 Hz 82 mm (measured in T03).
MIN_CONTROL_HZ = 100.0

# A control tick may hold its torques for at most this many physics steps.
MAX_SUBSTEPS = 1000

# ee rotation (columns = ee x, y, z axes in the world frame) for the top-down
# grasp at yaw 0: the hand z axis points down (−z) and the fingers slide along
# world x. This is the menagerie ``home`` orientation without its 1e-4 rounding.
TOP_DOWN_ROT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
TOP_DOWN_ROT.setflags(write=False)

# |yaw| <= π/2 + GRASP_YAW_MARGIN is reachable at every box corner (T03: joint 7
# keeps >= 0.013 rad of headroom at +π/2 + 0.05; at +π/2 + 0.1 it hits the limit).
GRASP_YAW_MARGIN = 0.05

# ``at_joint_limit`` is set when an arm joint is this close [rad] to its range end.
JOINT_LIMIT_MARGIN = 0.01

# Box (world = robot base frame, metres) where top-down targets are reachable
# from ``home``: the reach tests sample it. The table top is z = 0; z = 0.02
# puts the fingertip center 2 cm above it (a 4 cm cube's center).
WORKSPACE_LOW = np.array([0.30, -0.30, 0.02])
WORKSPACE_HIGH = np.array([0.65, 0.30, 0.45])
WORKSPACE_LOW.setflags(write=False)
WORKSPACE_HIGH.setflags(write=False)

# Tikhonov term added to Λ⁻¹ = J·M⁻¹·Jᵀ before Λ and D come from it. Measured
# in T03: Λ⁻¹'s smallest eigenvalue is >= 0.01 at reach endpoints over the
# workspace box with |yaw| <= π/2, so 1e-9 changes Λ by < 1e-7 (relative). At
# an exact singularity (e.g. the arm straight up, q = 0) it keeps Λ finite.
_LAMBDA_REG = 1e-9 * np.eye(6)

_YAW_LIMIT = 0.5 * math.pi + GRASP_YAW_MARGIN
_DOWN_COS = math.cos(math.radians(45.0))  # "hand points down": hand z within 45° of −z


@dataclass(slots=True, frozen=True)
class OSCGains:
    """Impedance gains. Scalars apply to all three axes; triples are (x, y, z) in the world.

    Units: ``pos_stiffness`` N/m, ``rot_stiffness`` N·m/rad, ``pos_damping``
    N·s/m, ``rot_damping`` N·m·s/rad, ``null_stiffness`` N·m/rad, ``null_damping``
    N·m·s/rad, ``max_lin_speed`` m/s, ``max_ang_speed`` rad/s. ``pos_damping`` /
    ``rot_damping`` = ``None`` means "modal damping with ``damping_ratio``"
    (module docstring). The speed limits bound how fast the
    reference pose moves toward the target; ``math.inf`` disables them.
    """

    pos_stiffness: float | Sequence[float] = 1500.0
    rot_stiffness: float | Sequence[float] = 60.0
    damping_ratio: float = 1.0
    pos_damping: float | Sequence[float] | None = None
    rot_damping: float | Sequence[float] | None = None
    null_stiffness: float = 10.0
    null_damping: float = 2.0
    max_lin_speed: float = 0.5
    max_ang_speed: float = 1.5

    def __post_init__(self) -> None:
        def put(name: str, value: object) -> None:
            object.__setattr__(self, name, value)  # frozen: normalize once, here

        put("pos_stiffness", _axis3("pos_stiffness", self.pos_stiffness, positive=True))
        put("rot_stiffness", _axis3("rot_stiffness", self.rot_stiffness, positive=True))
        put("damping_ratio", _nonneg("damping_ratio", self.damping_ratio))
        for name in ("pos_damping", "rot_damping"):
            value = getattr(self, name)
            if value is not None:
                put(name, _axis3(name, value, positive=False))
        put("null_stiffness", _nonneg("null_stiffness", self.null_stiffness))
        put("null_damping", _nonneg("null_damping", self.null_damping))
        put("max_lin_speed", _speed("max_lin_speed", self.max_lin_speed))
        put("max_ang_speed", _speed("max_ang_speed", self.max_ang_speed))


@dataclass(slots=True, frozen=True)
class OSCTarget:
    """Target (or reference) pose of the ee site in the world frame, with a velocity.

    Frozen. The snapshots the controller hands out (``target``,
    ``reference()``) also have read-only arrays; change the target with
    ``OperationalSpaceController.set_target``.
    """

    pos: np.ndarray  # (3,) m
    rot: np.ndarray  # (3, 3) rotation matrix; columns = ee axes in the world
    vel: np.ndarray  # (6,) [linear m/s, angular rad/s], world frame


class OperationalSpaceController:
    """Task-space impedance controller for the seven Panda arm motors.

    Bound to one ``(model, data)`` pair and one control rate. ``apply`` (called
    once per control tick, after ``mj_step1``; ``control_step`` does that)
    advances the reference pose by one tick and writes the torques. ``compute``
    only evaluates the torque law at the current state and reference.

    A new controller holds the ee pose of ``data`` (which needs kinematics:
    ``mj_forward`` or ``reset_to_home``). After teleporting the arm, call
    ``hold_current`` so the reference does not jump.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gains: OSCGains | None = None,
        ids: SceneIds | None = None,
        posture: Sequence[float] = HOME_ARM_QPOS,
        control_hz: float = CONTROL_HZ,
    ) -> None:
        self.model = model
        self.data = data
        self.ids = ids if ids is not None else scene_ids(model)
        self.posture = np.array(_reals("posture", posture, 7))
        substeps_for(model, control_hz)  # validates the rate
        self._control_hz = float(control_hz)
        self._dt = 1.0 / self._control_hz
        arm = self.ids.arm_dof
        if not np.array_equal(arm, np.arange(arm[0], arm[0] + 7)):
            raise ValueError("the arm dofs must be contiguous")
        self._cols = slice(int(arm[0]), int(arm[0]) + 7)  # arm dofs in qvel / M / J
        self._qpos = self.ids.arm_qpos
        self._tau_max = np.abs(model.actuator_ctrlrange[self.ids.arm_actuators, 1]).copy()
        ranges = np.array([model.joint(name).range for name in ARM_JOINTS])
        self._q_low = ranges[:, 0] + JOINT_LIMIT_MARGIN
        self._q_high = ranges[:, 1] - JOINT_LIMIT_MARGIN
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._jac = np.zeros((6, 7))
        self._mass = np.zeros((model.nv, model.nv))
        self._err = np.zeros(6)
        self._quat = np.zeros(4)
        self._quat_err = np.zeros(4)
        self._rotvec = np.zeros(3)
        self._quat_target = np.zeros(4)
        self._mat = np.zeros(9)
        self._ref_pos = np.zeros(3)
        self._ref_quat = np.zeros(4)
        self._ref_vel = np.zeros(6)
        self.saturated = False  # True if the last torque command hit a motor limit
        self.at_joint_limit = False  # True if an arm joint was at its range end (last compute)
        self.set_gains(gains if gains is not None else OSCGains())
        self.hold_current()

    @property
    def control_hz(self) -> float:
        """Tick rate [Hz] the reference step is sized for (fixed at construction)."""
        return self._control_hz

    @property
    def gains(self) -> OSCGains:
        """The active gains (immutable). Change them with ``set_gains``."""
        return self._gains

    # ------------------------------------------------------------------ targets

    @property
    def target(self) -> OSCTarget:
        """The current target (immutable). Set it with ``set_target`` or ``hold_current``."""
        return self._target

    def set_target(
        self,
        pos: Sequence[float],
        yaw: float = 0.0,
        rot: np.ndarray | None = None,
        vel: Sequence[float] | None = None,
    ) -> None:
        """Set the ee target: position [m], top-down orientation turned by ``yaw`` [rad].

        ``yaw`` must be within ±(π/2 + ``GRASP_YAW_MARGIN``) and within π of the
        reference's yaw (module docstring); use ``wrap_grasp_yaw(yaw,
        near=self.ee_yaw())`` to pick the equivalent grasp yaw. ``rot`` (a 3×3
        rotation matrix) overrides the top-down orientation, skips both checks,
        and then ``yaw`` must stay 0. ``vel`` is an optional 6-D feed-forward
        velocity [m/s, rad/s] for trajectory tracking (default zero). The
        reference pose moves toward the new target from where it is now.
        """
        position = np.array(_reals("pos", pos, 3))
        if rot is None:
            yaw = _real("yaw", yaw)
            if abs(yaw) > _YAW_LIMIT:
                raise ValueError(
                    f"yaw must be within ±{_YAW_LIMIT:.4f} rad (π/2 + GRASP_YAW_MARGIN), "
                    f"got {yaw!r}; refer to wrap_grasp_yaw"
                )
            self._check_turn(yaw)
            rotation = top_down_rot(yaw)
        else:
            if _real("yaw", yaw) != 0.0:
                raise ValueError("pass either rot or yaw, not both")
            rotation = _rotation("rot", rot)
        velocity = np.zeros(6) if vel is None else np.array(_reals("vel", vel, 6))
        self._set_target(position, rotation, velocity)

    def _check_turn(self, yaw: float) -> None:
        """Reject a yaw target the reference would reach by turning through yaw ±π."""
        ref = self._mat
        mujoco.mju_quat2Mat(ref, self._ref_quat)
        if ref[8] > -_DOWN_COS:  # the reference hand does not point down: no yaw to compare
            return
        ref_yaw = math.atan2(ref[4], ref[1])  # the ee y axis (column 1) in the world xy plane
        if abs(yaw - ref_yaw) > math.pi:
            raise ValueError(
                f"yaw {yaw:.4f} rad is {abs(yaw - ref_yaw):.4f} rad from the reference yaw "
                f"{ref_yaw:.4f}: the shortest turn passes yaw ±π, past joint 7's range. Use "
                "wrap_grasp_yaw(yaw, near=controller.ee_yaw()) or turn in two steps"
            )

    def ee_yaw(self) -> float:
        """Yaw [rad] of the ee about world z, as in ``top_down_rot(yaw)``.

        Meaningful while the hand points down. Reads ``data``'s kinematics
        (after ``control_step``, run ``mj_forward`` first).
        """
        rot = self.data.site_xmat[self.ids.ee_site]
        return math.atan2(float(rot[4]), float(rot[1]))

    def hold_current(self) -> None:
        """Make the ee pose in ``data`` the target and the reference, at rest.

        After ``control_step``, run ``mj_forward`` first (refer to ``control_step``).
        """
        site = self.ids.ee_site
        rotation = self.data.site_xmat[site].reshape(3, 3).copy()
        if not rotation.any():
            raise ValueError("data has no kinematics yet; run mj_forward (or reset_to_home) first")
        position = self.data.site_xpos[site].copy()
        if not (np.all(np.isfinite(rotation)) and np.all(np.isfinite(position))):
            raise ValueError("data has non-finite ee kinematics")
        self._set_target(position, rotation, np.zeros(6))
        self._ref_pos[:] = self._target.pos
        self._ref_quat[:] = self._quat_target
        self._ref_vel[:] = 0.0

    def reference(self) -> OSCTarget:
        """Snapshot of the reference pose and velocity the spring-damper pulls toward."""
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self._ref_quat)
        snapshot = (self._ref_pos.copy(), rot.reshape(3, 3), self._ref_vel.copy())
        for array in snapshot:
            array.setflags(write=False)
        return OSCTarget(*snapshot)

    def set_gains(self, gains: OSCGains) -> None:
        """Swap the gains (e.g. a softer stiffness while in contact)."""
        if not isinstance(gains, OSCGains):
            raise TypeError(f"gains must be OSCGains, got {type(gains).__name__}")
        self._gains = gains
        self._stiffness = np.array((*gains.pos_stiffness, *gains.rot_stiffness))
        self._sqrt_stiffness = np.sqrt(self._stiffness)
        self._two_zeta = 2.0 * gains.damping_ratio
        fixed = np.full(6, np.nan)
        if gains.pos_damping is not None:
            fixed[:3] = gains.pos_damping
        if gains.rot_damping is not None:
            fixed[3:] = gains.rot_damping
        self._fixed = np.flatnonzero(~np.isnan(fixed))  # axes with explicit damping
        self._fixed_damping = np.nan_to_num(fixed)[self._fixed]

    def _set_target(self, pos: np.ndarray, rot: np.ndarray, vel: np.ndarray) -> None:
        for array in (pos, rot, vel):
            array.setflags(write=False)
        self._target = OSCTarget(pos=pos, rot=rot, vel=vel)
        mujoco.mju_mat2Quat(self._quat_target, rot.reshape(9))

    # ------------------------------------------------------------------- errors

    def pose_error(self) -> tuple[np.ndarray, np.ndarray]:
        """(position error [m], axis-angle orientation error [rad]) to the target, world frame."""
        err = self._error_to(self._target.pos, self._quat_target)
        return err[:3].copy(), err[3:].copy()

    def errors(self) -> tuple[float, float]:
        """(position error norm [m], orientation error angle [rad]) to the target.

        Reads ``data``'s kinematics: after ``control_step``, run ``mj_forward`` first.
        """
        err = self._error_to(self._target.pos, self._quat_target)
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def _error_to(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        site = self.ids.ee_site
        err = self._err
        err[:3] = pos - self.data.site_xpos[site]
        mujoco.mju_mat2Quat(self._quat, self.data.site_xmat[site])
        err[3:] = _rotvec_between(quat, self._quat, self._quat_err, self._rotvec)
        return err

    # ------------------------------------------------------------------ control

    def advance_reference(self) -> None:
        """Move the reference pose one control tick toward the target (rate-limited).

        On the target (within one tick's step), the reference takes the target
        pose and its feed-forward velocity, clipped to the limits. Otherwise it
        steps toward the target at the limit with ``v_ref = 0`` (refer to the
        module docstring: no overshoot on arrival).
        """
        g, t = self.gains, self._target
        gap = t.pos - self._ref_pos
        dist = math.hypot(*gap)
        if dist <= g.max_lin_speed * self._dt:
            self._ref_pos[:] = t.pos
            self._ref_vel[:3] = _clip_norm(t.vel[:3], g.max_lin_speed)
        else:
            self._ref_pos += gap * (g.max_lin_speed * self._dt / dist)
            self._ref_vel[:3] = 0.0
        turn = _rotvec_between(self._quat_target, self._ref_quat, self._quat_err, self._rotvec)
        angle = math.hypot(*turn)
        if angle <= g.max_ang_speed * self._dt:
            self._ref_quat[:] = self._quat_target
            self._ref_vel[3:] = _clip_norm(t.vel[3:], g.max_ang_speed)
        else:
            mujoco.mju_axisAngle2Quat(self._quat_err, turn / angle, g.max_ang_speed * self._dt)
            mujoco.mju_mulQuat(self._ref_quat, self._quat_err, self._ref_quat.copy())
            mujoco.mju_normalize4(self._ref_quat)
            self._ref_vel[3:] = 0.0

    def compute(self) -> np.ndarray:
        """Arm torque command [N·m] (7,) at the current state and reference, clipped.

        Raises ``FloatingPointError`` instead of returning a non-finite torque
        (e.g. from absurd gains or a non-finite state).
        """
        model, data, g, cols = self.model, self.data, self.gains, self._cols
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.ids.ee_site)
        jac = self._jac
        jac[:3] = self._jacp[:, cols]
        jac[3:] = self._jacr[:, cols]
        mujoco.mj_fullM(model, data, self._mass)
        try:
            jm = jac @ np.linalg.inv(self._mass[cols, cols])  # J·M⁻¹
            lam_inv = jm @ jac.T + _LAMBDA_REG
            lam, damping = self._inertia_and_damping(lam_inv)
        except np.linalg.LinAlgError as exc:  # e.g. a non-finite qpos
            raise FloatingPointError(f"non-finite arm dynamics ({exc}); check the state") from exc
        qvel = data.qvel[cols]
        error = self._error_to(self._ref_pos, self._ref_quat)
        wrench = self._stiffness * error + damping @ (self._ref_vel - jac @ qvel)

        # Nᵀ·τ_p = τ_p − Jᵀ·Λ·J·M⁻¹·τ_p (dynamically consistent null space).
        posture = g.null_stiffness * (self.posture - data.qpos[self._qpos])
        posture -= g.null_damping * qvel
        tau = jac.T @ (wrench - lam @ (jm @ posture)) + posture + data.qfrc_bias[cols]
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError(f"non-finite arm torque {tau}; check gains, target, state")
        self.saturated = bool(np.any(np.abs(tau) > self._tau_max))
        q = data.qpos[self._qpos]
        self.at_joint_limit = bool(np.any((q < self._q_low) | (q > self._q_high)))
        return np.minimum(np.maximum(tau, -self._tau_max), self._tau_max)

    def _inertia_and_damping(self, lam_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Task-space inertia Λ and the 6×6 damping D (ratio ζ in every mode of
        Λ·ẍ + D·ẋ + K·x = 0; explicit blocks override), from one ``eigh``."""
        # K½·Λ⁻¹·K½ = V·Ω²·Vᵀ, so Λ = K½·V·Ω⁻²·Vᵀ·K½. The mode shapes Φ = K⁻½·V·Ω
        # satisfy ΦᵀΛΦ = I and ΦᵀKΦ = Ω², so D = Φ⁻ᵀ·(2ζΩ)·Φ⁻¹ = K½·V·(2ζ/Ω)·Vᵀ·K½.
        root = self._sqrt_stiffness
        omega_sq, modes = np.linalg.eigh(lam_inv * root * root[:, None])
        # Λ⁻¹ carries the 1e-9 Tikhonov term, so Ω² > 0 up to rounding; floor it.
        omega_sq = np.maximum(omega_sq, 1e-12 * omega_sq[-1])
        shapes = modes * root[:, None]
        lam = (shapes / omega_sq) @ shapes.T  # kept finite at a singularity by the Tikhonov term
        fixed = self._fixed
        if fixed.size == 6:
            return lam, np.diag(self._fixed_damping)
        damping = (shapes * (self._two_zeta / np.sqrt(omega_sq))) @ shapes.T
        if fixed.size:
            damping[fixed, :] = 0.0
            damping[:, fixed] = 0.0
            damping[fixed, fixed] = self._fixed_damping
        return lam, damping

    def apply(self) -> np.ndarray:
        """One control tick: advance the reference, compute, write the arm motors. Returns τ.

        If ``compute`` raises, the reference is restored and ``ctrl`` is untouched.
        """
        saved = (self._ref_pos.copy(), self._ref_quat.copy(), self._ref_vel.copy())
        self.advance_reference()
        try:
            tau = self.compute()
        except FloatingPointError:
            self._ref_pos[:], self._ref_quat[:], self._ref_vel[:] = saved
            raise
        self.data.ctrl[self.ids.arm_actuators] = tau
        return tau


# ----------------------------------------------------------------- control loop


class Actuated(Protocol):
    """Anything ``control_step`` can drive: ``apply()`` writes its ``data.ctrl`` entries."""

    def apply(self) -> object: ...


def substeps_for(model: mujoco.MjModel, control_hz: float) -> int:
    """Physics steps per control tick (1..``MAX_SUBSTEPS``).

    ``control_hz`` must be at least ``MIN_CONTROL_HZ``, and ``1/timestep`` must
    be a whole multiple of it.
    """
    hz = _pos("control_hz", control_hz)
    if hz < MIN_CONTROL_HZ:
        raise ValueError(
            f"control_hz={control_hz!r} is below the {MIN_CONTROL_HZ:g} Hz floor "
            "(slower ticks lose the reach accuracy; refer to the osc module docstring)"
        )
    dt = float(model.opt.timestep)
    if hz * dt * MAX_SUBSTEPS < 1.0:
        raise ValueError(f"control_hz={control_hz!r} needs more than {MAX_SUBSTEPS} substeps")
    ratio = 1.0 / (dt * hz)
    n = round(ratio)
    if n < 1 or abs(ratio - n) > 1e-9 * ratio:
        raise ValueError(
            f"control_hz={control_hz!r} must divide the physics rate "
            f"{1.0 / dt:g} Hz into a whole number of substeps"
        )
    return n


def control_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controllers: Sequence[Actuated],
    substeps: int = 1,
) -> None:
    """One control tick: refresh the state, apply each controller, step ``substeps`` times.

    ``mj_step1`` computes kinematics, the mass matrix, and ``qfrc_bias`` at the
    current state, so the torques match the state they act on. The commands are
    then held for all ``substeps`` physics steps. A controller with a
    ``control_hz`` attribute must match the tick rate ``1/(substeps·timestep)``,
    and each controller may appear once. If an ``apply()`` raises, the tick
    stops before any physics step; controllers applied earlier in the tick have
    already advanced (an ``OperationalSpaceController`` rolls back only its own
    failed tick). Afterwards ``data``'s kinematics (``site_xpos``, ...) still
    describe the state before the last physics step (≈ 1 mm behind at
    0.5 m/s). Run ``mj_forward`` before reading poses (``errors``,
    ``pose_error``, ``ee_yaw``, ``hold_current``, the scene readers).
    """
    controllers = tuple(controllers)  # a one-shot iterator would be used up below
    if len({id(controller) for controller in controllers}) != len(controllers):
        raise ValueError("a controller appears twice in one tick; it would advance twice")
    if (
        not isinstance(substeps, numbers.Integral)
        or isinstance(substeps, bool | np.bool_)
        or not 1 <= substeps <= MAX_SUBSTEPS
    ):
        raise ValueError(f"substeps must be an int in [1, {MAX_SUBSTEPS}], got {substeps!r}")
    tick_hz = 1.0 / (int(substeps) * float(model.opt.timestep))
    for controller in controllers:
        rate = getattr(controller, "control_hz", None)
        if rate is not None and not math.isclose(rate, tick_hz, rel_tol=1e-9):
            raise ValueError(
                f"{type(controller).__name__} runs at {rate:g} Hz but this tick is {tick_hz:g} Hz"
            )
    mujoco.mj_step1(model, data)
    for controller in controllers:
        controller.apply()
    mujoco.mj_step2(model, data)
    if substeps > 1:
        mujoco.mj_step(model, data, int(substeps) - 1)


# ---------------------------------------------------------------------- helpers


def top_down_rot(yaw: float = 0.0) -> np.ndarray:
    """Top-down ee rotation turned by ``yaw`` [rad] about world z."""
    yaw = _real("yaw", yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ TOP_DOWN_ROT


def wrap_grasp_yaw(yaw: float, near: float | None = None) -> float:
    """The equivalent grasp yaw to command: a parallel grasp is the same after a half turn.

    Without ``near``: the equivalent yaw in [−π/2, π/2). With ``near`` (e.g.
    ``controller.ee_yaw()``): of the equivalent yaws within ±(π/2 +
    ``GRASP_YAW_MARGIN``), the one closest to ``near``, so the hand turns the least.
    """
    yaw = _real("yaw", yaw)
    wrapped = (yaw + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    # Float rounding can land exactly on the open end (+π/2); fold it back.
    wrapped = wrapped - math.pi if wrapped >= 0.5 * math.pi else wrapped
    if near is None:
        return wrapped
    near = _real("near", near)
    candidates = [c for c in (wrapped - math.pi, wrapped + math.pi) if abs(c) <= _YAW_LIMIT]
    return min([wrapped, *candidates], key=lambda c: abs(c - near))


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle [rad] of the rotation between two rotation matrices."""
    a, b = _matrix3("a", a), _matrix3("b", b)
    cos = 0.5 * (float(np.trace(a @ b.T)) - 1.0)
    return math.acos(min(1.0, max(-1.0, cos)))


def _rotvec_between(
    quat_to: np.ndarray, quat_from: np.ndarray, scratch: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """World-frame axis-angle vector of the shortest turn from ``quat_from`` to ``quat_to``."""
    mujoco.mju_negQuat(scratch, quat_from)
    mujoco.mju_mulQuat(scratch, quat_to, scratch.copy())  # q_to · q_from⁻¹
    mujoco.mju_quat2Vel(out, scratch, 1.0)  # shortest arc, |θ| <= π
    return out


def _clip_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    norm = math.hypot(*vec)
    return vec * (limit / norm) if norm > limit else vec


def _matrix3(name: str, value: object) -> np.ndarray:
    mat = np.asarray(value)
    if mat.dtype.kind not in "iuf" or mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must be a finite real 3x3 matrix")
    return mat.astype(float)


def _rotation(name: str, value: object) -> np.ndarray:
    try:
        mat = _matrix3(name, value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite real 3x3 rotation matrix") from None
    if not np.allclose(mat @ mat.T, np.eye(3), atol=1e-6) or np.linalg.det(mat) < 0:
        raise ValueError(f"{name} must be a proper rotation matrix (orthonormal, det +1)")
    return mat


def _axis3(name: str, value: object, positive: bool) -> tuple[float, float, float]:
    if np.ndim(value) == 0 and not isinstance(value, str | bytes):
        values = (_real(name, value),) * 3
    else:
        values = _reals(name, value, 3)
    check = _pos if positive else _nonneg
    return tuple(check(name, v) for v in values)  # type: ignore[return-value]


def _speed(name: str, value: object) -> float:
    """A speed limit: finite > 0, or ``math.inf`` (any real +inf scalar) for no limit."""
    if not isinstance(value, bool | np.bool_) and np.ndim(value) == 0:
        item = value.item() if isinstance(value, np.ndarray | np.generic) else value
        if isinstance(item, numbers.Real) and not isinstance(item, bool) and item == math.inf:
            return math.inf
    return _pos(name, value)


def _pos(name: str, value: object) -> float:
    result = _real(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _nonneg(name: str, value: object) -> float:
    result = _real(name, value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result
