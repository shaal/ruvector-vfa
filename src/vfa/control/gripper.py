"""Parallel-jaw gripper: ``open``, ``close(width)``, ``squeeze(force_n, speed)``, ``width``.

The gripper actuator is ``actuator8`` on the menagerie ``split`` tendon,
``L = c·q1 + c·q2`` with the coupling ``c = 0.5``, where ``q1``, ``q2`` are the
two finger slides (0..0.04 m each). The opening between the fingers is the
width ``w = q1 + q2 = L/c = 2·L`` (0..0.08 m). The class reads ``c`` from the
model. MuJoCo computes the tendon force as::

    F = g·ctrl + b0 + b1·L + b2·L̇          (gain g, affine bias b; clipped to forcerange)

Since T05 the scene builds ``actuator8`` as a force motor (``g = 1``, ``b = 0``,
±100 N: ctrl = tendon force). ``SceneConfig(gripper="servo")`` keeps the
menagerie position servo (``g = 0.0157`` = 0.04·100/255, ``b = (0, −100, −10)``,
ctrl 0..255): stiffness 100 N/m and damping 10 N·s/m on ``L``. Each finger
joint receives ``coef·F = 0.5·F``. The class requires ``gear = 1`` (MuJoCo's
bias then acts on ``L`` itself).

Every command picks a desired tendon force ``F*`` and solves the line above for
``ctrl`` at the current ``L``, ``L̇`` (re-solved every control tick by ``apply``):

- **Position** (``open``, ``close(width)``): ``F* = kp·(c·w* − L) − kd·L̇`` with
  the servo's own ``kp = −b1``, ``kd = −b2`` (the motor emulates 100 N/m and
  10 N·s/m). For the menagerie servo this gives the native command
  ``ctrl = 255·L*/0.04``. Closed on an object of width ``w_obj``, the squeeze
  per finger is ``0.5·kp·(w_obj − w*)/2 = 25·(w_obj − w*)`` N.
- **Force, no speed limit** (``squeeze(f)``, the T03 law): ``F* = −f/c − kd·L̇``
  = ``−2f − 10·L̇``, so at rest each finger presses with ``f`` N, the squeeze
  (grip) force in a face-on two-finger grasp. ``f`` is per finger: the normal
  force each pad applies, which is the ``F_min ≈ m(g + a)/(2μ)`` of
  ARCHITECTURE §4. The servo damping slows the fingers, but nothing caps the
  closing speed: squeezing from wide open overshoots on impact, then settles
  on the request. Measured on a floating cube at 500 Hz in T03: servo
  1.96–2.07× (3 cm at 0.5 N, 5 and 7 cm at 1 N); force motor 1.6–2.0× (5 cm
  at 6–20 N, 3 cm at 6–10 N).
- **Force with a closing speed** (``squeeze(f, speed=v)``, T05): the libfranka
  ``grasp(width, speed, force)`` idea. A width target starts where its force
  equals the force the actuator applies now (a bumpless start: a new squeeze
  continues from the current force, without a step or a dip) and moves toward
  0 at ``v`` [m/s, the rate of change of the opening;
  ``MIN_CLOSE_SPEED``–``MAX_CLOSE_SPEED``].
  A PD law on the tendon pulls the fingers along it,
  ``F_track = TRACK_KP·(c·w_t − L) + TRACK_KD·(c·ẇ_t − L̇)``, and
  ``F* = max(−f/c, F_track)`` (never more than ``f``): the fingers close at
  ``v`` until an object blocks them. When the moving fingers hit it, the
  squeeze steps by the servo's damping on the target velocity,
  ``c²·TRACK_KD·v`` = ``7.5·v`` N per finger (capped at ``f``), and the target runs
  ahead, so it rises at ``TRACK_KP·c²·v`` = ``250·v`` N/s per finger until it
  reaches ``f`` and holds it. The force branch has no damping: a
  current-controlled motor has no velocity feedback, and the T03 servo damping
  (10 N·s/m) would hide most of the closing-speed effect (median overshoot
  1.267 → 1.100 at 2 N in T05). The target never leads the fingers by more
  than the full squeeze needs (anti-windup), and a clamped target feeds no
  velocity forward: after the object is lost the fingers close under at most
  that squeeze, and the end stop brakes them. Re-sending the same command
  every tick changes nothing (a changed one starts bumplessly). The target
  advances at most 10 ms per ``apply`` (call it every control tick), and
  after ``reset_to_home`` (time moves back) it restarts at the fingers. A
  command outlives a reset: send ``open()`` after one.
  In contact the speed sets the force profile. T05 spike, ``fragile_soft``
  cubes on the table (30 per cell, 500 Hz): the median peak summed force over
  ``2f`` is 1.000 at 0.02 m/s and 1.267 / 1.199 / 1.147 / 1.103 at 0.5 m/s
  for ``f`` = 2 / 2.5 / 3 / 3.5 N; on stiff ``light_rigid`` cubes 3.13 at 1 N
  and 1.41 at 3 N (the fingers' impact barely depends on the request).
  A slow speed with a large force is a slow grasp: 30 N at 0.02 m/s ramps
  for 6 s.

**End stop** (force modes): below ``END_STOP_WIDTH + END_STOP_BAND`` the
closing force is capped, falling linearly to 0 at ``END_STOP_WIDTH``, and
inside that band a brake slows fingers closing faster than ``END_STOP_SPEED``
scaled the same way.
Without it a force motor that squeezes ≥ 5 N on *nothing* drives the fingers
through their 0 limit and into each other, where they jam (measured in T03:
−11 mm at 5 N, −49 mm at 20 N, still there after ``open()``; the scene also
excludes finger-finger contacts since T05). The full force range needs an
object at least ``END_STOP_WIDTH + END_STOP_BAND`` (3 cm) wide. The stop
only removes the closing force, so fingers that arrive moving come to rest
a little inside it (at 0.5 m/s: 3.4 mm at 500 Hz, 1.6 mm at 100 Hz), never
below 0. Without a speed they arrive at several m/s and overrun past 0 for a
moment before they come back; refer to ARCHITECTURE §3.

Saturation: the force range, the ctrl range, and the end stop can cap a
request; ``squeeze`` then sets ``saturated`` (the speed branch's intended
ramp does not count). ``max_squeeze_force`` reports the cap at a width. The
menagerie servo's ctrl floor (0) caps its squeeze at ``kp·L/2 = 25·w`` N per
finger, i.e. **1.25 N on a 5 cm cube**, far below the ARCHITECTURE §4 force
windows; that is why T05 made the scene a force motor.
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
# Closing-speed servo on the tendon (squeeze with ``speed``). Critically damped for
# the fingers' tendon inertia, 2 × (0.015 kg finger + 0.1 kg joint armature) =
# 0.23 kg, and stable at the 100 Hz control floor: TRACK_KD·T/m = 1.3 at T = 10 ms.
TRACK_KP = 1000.0  # N/m
TRACK_KD = 30.0  # N·s/m
# End stop [m]: in a force mode the closing force is capped below
# END_STOP_WIDTH + END_STOP_BAND, falling linearly to 0 at END_STOP_WIDTH. Inside
# that band a brake (TRACK_KD) slows fingers that close faster than END_STOP_SPEED
# [m/s] scaled the same way, so they reach the stop at rest.
END_STOP_WIDTH = 0.005
END_STOP_BAND = 0.025
END_STOP_SPEED = 0.5
# Closing speed bounds [m/s, rate of change of the opening]. The top is the end
# stop's design speed, so a speed-limited squeeze never drives the fingers past 0.
MIN_CLOSE_SPEED = 0.001
MAX_CLOSE_SPEED = END_STOP_SPEED
# Anti-windup slack [m of tendon length]: the clamped target leads the fingers
# by the full squeeze plus this (0.1 N of tendon force).
_TRACK_SLACK = 0.0001
# The speed target advances at most this long per apply() [s] (the 100 Hz control
# floor), so a gap between apply() calls pauses it instead of releasing a jump.
_MAX_TRACK_DT = 0.01


@dataclass(slots=True, frozen=True)
class GripperCommand:
    """The active command: a target width [m] or a per-finger squeeze force [N].

    ``speed`` [m/s] caps the closing speed of a squeeze (``None``: no cap).
    """

    mode: Literal["position", "force"]
    value: float
    speed: float | None = None


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
        self._track_length = 0.0  # speed-limited squeeze: tendon-length target [m]
        self._track_time = 0.0  # data.time of its last advance [s]

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

    def squeeze(self, force_n: float, speed: float | None = None) -> None:
        """Force command: press each finger on the object with ``force_n`` [N] (at rest).

        ``speed=None``: the fingers close under that force, damped by the servo
        damping (no speed cap). ``speed`` [m/s]: the fingers close at most that
        fast, and in contact the squeeze rises at ``TRACK_KP·c²·speed`` N/s per
        finger up to ``force_n`` (module docstring). If the actuator cannot
        reach the force at the current width, the command saturates
        (``saturated``); refer to ``max_squeeze_force``.
        """
        force_n = _real("force_n", force_n)
        if not 0.0 <= force_n <= MAX_SQUEEZE_N:
            raise ValueError(f"force_n must be in [0, {MAX_SQUEEZE_N:g}] N, got {force_n!r}")
        if speed is not None:
            speed = _real("speed", speed)
            if not MIN_CLOSE_SPEED <= speed <= MAX_CLOSE_SPEED:
                raise ValueError(
                    f"speed must be in [{MIN_CLOSE_SPEED:g}, {MAX_CLOSE_SPEED:g}] m/s, "
                    f"got {speed!r}"
                )
        self._set_command(GripperCommand("force", force_n, speed))

    def _set_command(self, command: GripperCommand) -> None:
        """Make ``command`` active and write ``ctrl``; keep the old command if that raises."""
        previous, self._command = self._command, command
        saved = (self._track_length, self._track_time)
        if command.speed is not None and command != previous:
            self._start_track(command.speed)  # re-sending the same command keeps the target
        try:
            self.apply()
        except FloatingPointError:
            self._command = previous
            self._track_length, self._track_time = saved
            raise

    def apply(self) -> float:
        """Write ``ctrl`` for the active command at the current finger state. Returns it.

        Raises ``FloatingPointError`` instead of writing a non-finite ``ctrl``
        (a non-finite finger state).
        """
        length = float(self.data.ten_length[self._tendon])
        speed = float(self.data.ten_velocity[self._tendon])
        command = self._command
        stopped = False
        if command.mode == "position":
            target_length = self._coef * command.value  # L = c·w
            force = self.servo_kp * (target_length - length) - self.servo_kd * speed
        else:
            squeeze = -command.value / self._coef
            if command.speed is None:
                force = squeeze - self.servo_kd * speed
            else:
                force = max(squeeze, self._track_force(length, speed, squeeze, command.speed))
            ceiling = self._end_stop(length)
            stopped = force < ceiling and _differs(ceiling, force)
            force = max(force, ceiling) + self._end_stop_brake(length, speed)
        reachable = min(max(force, self._force_lo), self._force_hi)
        b0, b1, b2 = self._bias
        ctrl = (reachable - b0 - b1 * length - b2 * speed) / self._gain
        clipped = min(max(ctrl, self._ctrl_lo), self._ctrl_hi)
        if not math.isfinite(clipped):
            raise FloatingPointError(f"non-finite gripper ctrl {clipped} (L={length}, L̇={speed})")
        self.saturated = stopped or _differs(reachable, force) or _differs(clipped, ctrl)
        self.data.ctrl[self._act] = clipped
        return float(clipped)

    def _start_track(self, close_speed: float) -> None:
        """Place the speed target so its force equals the force the actuator applies now.

        A bumpless start: ``TRACK_KP·(target − L) + TRACK_KD·(−c·v − L̇) = F_now``.
        So a new squeeze (after a position command, a gentler squeeze, or another
        speed) continues from the current force without a step or a dip.
        """
        now = float(self.data.actuator_force[self._act])
        length = float(self.data.ten_length[self._tendon])
        speed = float(self.data.ten_velocity[self._tendon])
        feed_forward = TRACK_KD * (-self._coef * close_speed - speed)
        self._track_length = length + (now - feed_forward) / TRACK_KP
        self._track_time = float(self.data.time)

    def _track_force(
        self, length: float, speed: float, squeeze: float, close_speed: float
    ) -> float:
        """Closing-speed servo: advance the target to ``data.time`` and return its tendon force."""
        if self.data.time < self._track_time:  # a reset moved time back: restart
            self._start_track(close_speed)
        now = float(self.data.time)
        elapsed = min(now - self._track_time, _MAX_TRACK_DT)
        self._track_time = now
        free = self._track_length - self._coef * close_speed * elapsed
        # Anti-windup: lead the fingers by no more than the full squeeze needs. A
        # clamped target feeds no velocity forward (its jumps are not motion).
        bound = length + squeeze / TRACK_KP - _TRACK_SLACK
        if free >= bound:
            target, target_speed = free, -self._coef * close_speed
        else:
            target, target_speed = bound, 0.0
        self._track_length = target
        return TRACK_KP * (target - length) + TRACK_KD * (target_speed - speed)

    def _end_stop(self, length: float) -> float:
        """Most negative (closing) tendon force allowed at ``length``: 0 at the stop."""
        floor = self._force_lo if math.isfinite(self._force_lo) else -MAX_SQUEEZE_N / self._coef
        return floor * self._end_stop_reach(length)

    def _end_stop_brake(self, length: float, speed: float) -> float:
        """Opening tendon force that slows fingers closing faster than the band allows."""
        reach = self._end_stop_reach(length)
        if reach >= 1.0:  # above the band: no brake
            return 0.0
        allowed = -self._coef * END_STOP_SPEED * reach  # L̇ floor
        return TRACK_KD * (allowed - speed) if speed < allowed else 0.0

    def _end_stop_reach(self, length: float) -> float:
        """0 at the stop, 1 at and above END_STOP_WIDTH + END_STOP_BAND."""
        width = length / self._coef
        return min(max((width - END_STOP_WIDTH) / END_STOP_BAND, 0.0), 1.0)

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
        # Most negative tendon force: the ctrl bound that minimizes g·ctrl, then
        # forcerange, then the end stop.
        force = self._gain * self._ctrl_lo + b0 + b1 * length
        force = max(force, self._force_lo, self._end_stop(length))
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
