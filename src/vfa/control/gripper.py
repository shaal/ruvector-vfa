"""Parallel-jaw gripper interface: ``open``, ``close(width)``, ``squeeze(force_n)``, ``width``.

The gripper is the menagerie tendon actuator ``actuator8`` (T02). It drives the
``split`` tendon, ``L = c·q1 + c·q2`` with the coupling ``c = 0.5``, where
``q1``, ``q2`` are the two finger slides (0..0.04 m each). The opening between
the fingers is the width ``w = q1 + q2 = L/c = 2·L`` (0..0.08 m). The class
reads ``c`` from the model. MuJoCo computes the tendon force as::

    F = g·ctrl + b0 + b1·L + b2·L̇          (gain g, affine bias b; clipped to forcerange)

The menagerie servo has ``g = 0.0157`` (= 0.04·100/255), ``b = (0, −100, −10)``,
ctrl 0..255, force ±100 N: a position servo with stiffness 100 N/m and damping
10 N·s/m on ``L``. Each finger joint receives ``coef·F = 0.5·F``. The class
requires ``gear = 1`` (MuJoCo's bias then acts on ``L`` itself).

Both commands pick a desired tendon force ``F*`` and solve the line above for
``ctrl`` at the current ``L``, ``L̇`` (re-solved every control tick by ``apply``):

- **Position** (``open``, ``close(width)``): ``F* = kp·(c·w* − L) − kd·L̇`` with
  the servo's own ``kp = −b1``, ``kd = −b2``. For the menagerie servo this gives
  the native command ``ctrl = 255·L*/0.04``. Closed on an object of width
  ``w_obj``, the squeeze per finger is ``0.5·kp·(w_obj − w*)/2 = 25·(w_obj − w*)`` N.
- **Force** (``squeeze(f)``): ``F* = −f / c − kd·L̇`` = ``−2f − 10·L̇``,
  so at rest each finger presses with ``f`` N, the squeeze (grip) force in a
  face-on two-finger grasp. ``f`` is per finger: the normal force each pad
  applies, which is the ``F_min ≈ m(g + a)/(2μ)`` of ARCHITECTURE §4.
  The servo damping stays while the fingers close, but nothing limits the
  closing speed: squeezing from wide open overshoots on impact, then settles
  on the request. Measured on a floating cube at 500 Hz: servo 1.96–2.07×
  (3 cm at 0.5 N, 5 and 7 cm at 1 N); force motor 1.6–2.0× (5 cm at 6–20 N,
  3 cm at 6–10 N), but at 15–20 N on a 3 cm cube the fingers punch through the
  soft contact (peak ≈ 1330–1380 N). Closing to 4 mm clearance by position
  first still leaves 1.37–1.71× (2–20 N). T05 owns the closing profile
  (ARCHITECTURE §3).

Saturation (menagerie servo): the ctrl floor (0) caps the squeeze at
``kp·L/2 = 25·w`` N per finger, i.e. **1.25 N on a 5 cm cube**, far below the
§4 force windows. ``squeeze`` then clips ``ctrl`` and sets ``saturated``.
``max_squeeze_force`` reports the cap at the current width. T05 decides whether
``actuator8`` becomes a force ``motor``; the force mapping then works unchanged
(``g = 1``, ``b = 0``: ``ctrl = F*``, with the menagerie ``kp``/``kd`` emulated
for the position commands; tested). T05 handoff: with such a motor, a squeeze
of >= 5 N that closes on *nothing* drives the light fingers through their 0
joint limit and into each other, and they jam (measured in T03: −11 mm at 5 N,
−49 mm at 20 N, still there after ``open()``). The servo cannot do this (its
force falls to 0 as the fingers close). A motor swap needs an end stop or a
closing-speed limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from vfa.sim.scene import FINGER_JOINTS, SceneIds, _real, scene_ids

# Position-servo gains used when the actuator has no position bias of its own
# (a force motor): the menagerie servo's 100 N/m and 10 N·s/m on the tendon.
DEFAULT_SERVO_KP = 100.0
DEFAULT_SERVO_KD = 10.0
# Upper bound on a squeeze request [N per finger], far above any real gripper,
# so the force arithmetic stays finite even on an actuator without limits.
MAX_SQUEEZE_N = 1e4


@dataclass(slots=True, frozen=True)
class GripperCommand:
    """The active command: a target width [m] or a per-finger squeeze force [N]."""

    mode: Literal["position", "force"]
    value: float


class Gripper:
    """Width and squeeze-force commands for the Panda gripper, bound to one ``(model, data)``.

    ``open``, ``close``, and ``squeeze`` store the command and write ``ctrl`` at
    once. Call ``apply`` every control tick (``control_step`` does) so the
    command tracks the current finger state.
    The actuator and tendon parameters are read once, at construction: build a
    new ``Gripper`` after editing them in the model.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, ids: SceneIds | None = None
    ) -> None:
        self.model = model
        self.data = data
        self.ids = ids if ids is not None else scene_ids(model)
        act = self.ids.gripper_actuator
        # int(): numpy ints do not compare reliably with pybind11 enums inside ``in``.
        bias_type = int(model.actuator_biastype[act])
        if int(model.actuator_trntype[act]) != mujoco.mjtTrn.mjTRN_TENDON:
            raise ValueError("gripper actuator must drive a tendon")
        if int(model.actuator_dyntype[act]) != mujoco.mjtDyn.mjDYN_NONE:
            raise ValueError("gripper actuator must have no activation dynamics")
        if int(model.actuator_gaintype[act]) != mujoco.mjtGain.mjGAIN_FIXED:
            raise ValueError("gripper actuator must have a fixed gain")
        if bias_type not in (int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE)):
            raise ValueError("gripper actuator bias must be none or affine")
        self._act = act
        self.actuators = (int(act),)  # ctrl ids it writes (control_step checks overlaps)
        self._tendon = int(model.actuator_trnid[act, 0])
        self._gain = float(model.actuator_gainprm[act, 0])
        affine = bias_type == int(mujoco.mjtBias.mjBIAS_AFFINE)
        self._bias = model.actuator_biasprm[act, :3].copy() if affine else np.zeros(3)
        if self._gain <= 0:
            raise ValueError("gripper actuator gain must be > 0")
        # A native position servo has b1 < 0 (stiffness) and b2 <= 0 (damping).
        has_servo = self._bias[1] < 0
        self.servo_kp = float(-self._bias[1]) if has_servo else DEFAULT_SERVO_KP
        self.servo_kd = float(-self._bias[2]) if has_servo else DEFAULT_SERVO_KD
        self._ctrl_lo, self._ctrl_hi = (float(v) for v in model.actuator_ctrlrange[act])
        if not model.actuator_ctrllimited[act]:
            self._ctrl_lo, self._ctrl_hi = -math.inf, math.inf
        self._force_lo, self._force_hi = (float(v) for v in model.actuator_forcerange[act])
        if not model.actuator_forcelimited[act]:
            self._force_lo, self._force_hi = -math.inf, math.inf
        self._coef = self._finger_coef(model)
        if float(model.actuator_gear[act, 0]) != 1.0:
            raise ValueError("gripper actuator gear must be 1")
        jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in FINGER_JOINTS]
        self.max_width = float(sum(model.jnt_range[j, 1] for j in jnt))
        self.saturated = False  # True if the last apply() clipped ctrl or force
        self._command = GripperCommand("position", self.max_width)

    @property
    def command(self) -> GripperCommand:
        """The active command (immutable). Change it with ``open``, ``close``, or ``squeeze``."""
        return self._command

    # ------------------------------------------------------------------ commands

    def open(self) -> None:
        """Open fully (position command at ``max_width``)."""
        self.close(self.max_width)

    def close(self, width: float = 0.0) -> None:
        """Position command: drive the opening toward ``width`` [m] (0 = fully closed)."""
        width = _real("width", width)
        if not 0.0 <= width <= self.max_width:
            raise ValueError(f"width must be in [0, {self.max_width}] m, got {width!r}")
        self._set_command(GripperCommand("position", width))

    def squeeze(self, force_n: float) -> None:
        """Force command: press each finger on the object with ``force_n`` [N] (at rest).

        Before contact the fingers close under that force, damped by the servo
        damping. If the actuator cannot reach the force at the current width,
        the command saturates (``saturated``); refer to ``max_squeeze_force``.
        """
        force_n = _real("force_n", force_n)
        if not 0.0 <= force_n <= MAX_SQUEEZE_N:
            raise ValueError(f"force_n must be in [0, {MAX_SQUEEZE_N:g}] N, got {force_n!r}")
        self._set_command(GripperCommand("force", force_n))

    def _set_command(self, command: GripperCommand) -> None:
        """Make ``command`` active and write ``ctrl``; keep the old command if that raises."""
        previous, self._command = self._command, command
        try:
            self.apply()
        except FloatingPointError:
            self._command = previous
            raise

    def apply(self) -> float:
        """Write ``ctrl`` for the active command at the current finger state. Returns it.

        Raises ``FloatingPointError`` instead of writing a non-finite ``ctrl``
        (a non-finite finger state).
        """
        length = float(self.data.ten_length[self._tendon])
        speed = float(self.data.ten_velocity[self._tendon])
        command = self._command
        if command.mode == "position":
            target_length = self._coef * command.value  # L = c·w
            force = self.servo_kp * (target_length - length) - self.servo_kd * speed
        else:
            force = -command.value / self._coef - self.servo_kd * speed
        reachable = min(max(force, self._force_lo), self._force_hi)
        b0, b1, b2 = self._bias
        ctrl = (reachable - b0 - b1 * length - b2 * speed) / self._gain
        clipped = min(max(ctrl, self._ctrl_lo), self._ctrl_hi)
        if not math.isfinite(clipped):
            raise FloatingPointError(f"non-finite gripper ctrl {clipped} (L={length}, L̇={speed})")
        self.saturated = _differs(reachable, force) or _differs(clipped, ctrl)
        self.data.ctrl[self._act] = clipped
        return float(clipped)

    # ------------------------------------------------------------------ readback

    def width(self) -> float:
        """Current opening between the fingers [m] (``q1 + q2``)."""
        return float(self.data.qpos[self.ids.finger_qpos].sum())

    def max_squeeze_force(self, width: float | None = None) -> float:
        """Largest per-finger squeeze [N] the actuator can hold at rest at ``width`` [m].

        ``width`` defaults to the current opening (clamped to [0, ``max_width``],
        since the soft joint limits let it stray slightly outside).
        """
        if width is None:
            w = min(max(self.width(), 0.0), self.max_width)
        else:
            w = _real("width", width)
            if not 0.0 <= w <= self.max_width:
                raise ValueError(f"width must be in [0, {self.max_width}] m, got {width!r}")
        length = self._coef * w
        b0, b1, _ = self._bias
        # Most negative tendon force: the ctrl bound that minimizes g·ctrl, then forcerange.
        force = self._gain * self._ctrl_lo + b0 + b1 * length
        force = max(force, self._force_lo)
        return float(max(0.0, -force * self._coef))

    def actuator_squeeze(self) -> float:
        """Per-finger squeeze [N] the actuator applies now: ``−coef·F`` (after mj_forward).

        Positive = closing. It equals the pad's normal force on the object only
        in a static face-on grasp; refer to ``finger_cube_forces`` for contact force.
        """
        return float(-self.data.actuator_force[self._act] * self._coef)

    def _finger_coef(self, model: mujoco.MjModel) -> float:
        adr, num = int(model.tendon_adr[self._tendon]), int(model.tendon_num[self._tendon])
        coefs = model.wrap_prm[adr : adr + num]
        if num != 2 or not np.allclose(coefs, coefs[0]) or coefs[0] <= 0:
            raise ValueError("gripper tendon must couple the two fingers with one positive coef")
        return float(coefs[0])


def _differs(a: float, b: float) -> bool:
    """True if a clip moved ``b`` to ``a`` by more than float rounding."""
    if not math.isfinite(b):
        return a != b
    return bool(abs(a - b) > 1e-9 * (1.0 + abs(b)))
