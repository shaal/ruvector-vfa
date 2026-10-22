"""Safety supervisor and emergency stop (T09, ARCHITECTURE §8).

``SafetySupervisor.check`` runs every control tick, after the sensors, and
returns the first violated limit (``SafetyViolation``) or ``None``, in this
order:

1. ``workspace`` (detail ``target``): the commanded ee target (OSC control) lies
   outside the OSC workspace box. The env clips its own targets to a tighter
   box (``TARGET_LOW``–``TARGET_HIGH``, near face x = 0.33), so only a fault
   commands outside it.
2. ``wrist_force`` / ``wrist_torque``: the norm of the tared, low-passed wrist
   force or torque passes its limit (a collision, a push, a jammed grasp).
3. ``workspace`` (detail ``ee``): the measured ee lies more than
   ``SafetyLimits.box_margin`` past a face of the box (a tracking ee lags its
   target).
4. ``ee_speed``: the ee's linear or angular speed passes its cap.
5. ``joint_limit``: an arm joint comes within ``joint_margin`` of its range end.
6. ``torque_saturation``: the arm torque command stays at a motor limit for
   ``saturation_s`` (the OSC clips its law there and stops tracking it).

The supervisor latches on the first violation; later checks return ``None``.
The env (``PandaVFAEnv``) then engages the emergency mode: the OSC holds the
current pose (``hold_current``: impedance around it plus gravity compensation),
in torque control the arm motors get the gravity compensation torques, the
gripper keeps its last command (a held cube stays held), and the env reports
``SAFETY_STOP`` with the violation as its info, which ends the episode.

**Limits** (``SafetyLimits`` defaults): the force and speed limits sit about
twice above the peaks of normal T08 attempts, the torque limit about four
times. Over 450 attempts (50 hand-picked θ, 400 uniform-random θ, 200 of them
with ``REALISTIC_NOISE``; 341 successes, 66 breaks, 40 drops, 3 slips) the
largest tared wrist force was 15.1 N (a 0.95 kg cube lifted at 2.8 m/s²),
torque 0.92 N·m, ee speed 0.51 m/s and 1.52 rad/s; no joint came within
0.50 rad of its range end; the ee stayed within x 0.446–0.602, y ±0.151, z 0.0156–0.400 m; and
no arm torque ever saturated. Saturation must last 0.2 s: full-speed target
reversals (bang-bang exploration of the env's osc actions) saturate joint 2 for
up to 134 ms in free space near the far, low edge of the box; there such a
reversal can still carry the ee more than 3 cm past x = 0.65 (a ``workspace``
stop). Such exploration also ends in other correct stops: a full-speed descent
onto the target floor (z 0.02) near the far corners overshoots about 1 cm and
usually strikes the table with 30–41 N (up to 49 N), and at the far high
corners joint 4 comes within about 0.03 rad of its range end.

**Latency** (``tests/test_safety.py``). A check stops the episode in the tick
its limit is crossed. A table hit at 0.1–0.5 m/s crosses 30 N within 0–12 ms of
the first fingertip contact (0 ms at 0.5 m/s); a slower drive presses longer
before it does (about 18 ms at 0.06 m/s, 216 ms at 0.05 m/s). A 50–200 N push on
the hand stops in 2–10 ms; a target leaving the box, at once.

**Known limits.** (1) Only the wrist F/T senses force, so it sees forces on the
hand, the fingers, and what they hold, not on the arm links above the wrist: a
50–100 N push on link 4–7 deflects the ee by up to 7 cm without a stop, and a
200 N push stops late or not at all (only through saturation, speed, or the
box; 200 N on link 4 moved the ee 14.6 cm). An external-torque observer would
close the gap. (2) The wrist cannot tell a held cube's inertia from a
collision: setting a 0.6–1.0 kg cube down at 0.1 m/s or faster, or reversing
it at full speed, can pass 30 N. T19 must place more slowly or raise
``max_force`` in its release phases. (3) The checks add about 20 µs per
500 Hz tick (T08 attempts: 221 vs 201 µs, and 197 vs 179 µs in a second run).

``SafetyViolation.value`` is the measured value, except for ``workspace`` (the
distance outside the box; a measured ee stops past ``limit`` = the margin) and
``joint_limit`` (the radians left to the range end; it stops below ``limit``).

RuVector mapping: the supervisor is the trigger of
``cognitive_core::CognitiveMode::Emergency``, the crate's override mode for
safety-critical situations (its decisions carry priority 255; T14 switches the
core into it); ``SAFETY_STOP`` ends the attempt like any other failure
``Outcome``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from vfa.control.osc import WORKSPACE_HIGH, WORKSPACE_LOW
from vfa.sim.scene import ARM_JOINTS, SceneIds
from vfa.sim.sensors import SensorFrame

logger = logging.getLogger(__name__)

REASONS: tuple[str, ...] = (
    "workspace",
    "wrist_force",
    "wrist_torque",
    "ee_speed",
    "joint_limit",
    "torque_saturation",
)
_EPS_BOX = 1e-9  # a target on the box edge is inside


@dataclass(slots=True, frozen=True)
class SafetyLimits:
    """Thresholds of ``SafetySupervisor`` (refer to the module doc for the margins)."""

    max_force: float = 30.0  # N: norm of the tared, low-passed wrist force
    max_torque: float = 4.0  # N·m: norm of the tared, low-passed wrist torque
    box_margin: float = 0.03  # m: how far the measured ee may stray past the workspace
    max_lin_speed: float = 1.0  # m/s
    max_ang_speed: float = 3.0  # rad/s
    joint_margin: float = 0.02  # rad from an arm joint's range end
    saturation_s: float = 0.2  # s of continuous arm torque saturation
    workspace_low: tuple[float, float, float] = tuple(float(v) for v in WORKSPACE_LOW)
    workspace_high: tuple[float, float, float] = tuple(float(v) for v in WORKSPACE_HIGH)

    def __post_init__(self) -> None:
        for name in (
            "max_force",
            "max_torque",
            "max_lin_speed",
            "max_ang_speed",
            "saturation_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not (
                isinstance(value, int | float) and math.isfinite(value) and value > 0
            ):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        for name in ("box_margin", "joint_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not (
                isinstance(value, int | float) and math.isfinite(value) and value >= 0
            ):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        low = np.asarray(self.workspace_low, dtype=float)
        high = np.asarray(self.workspace_high, dtype=float)
        if low.shape != (3,) or high.shape != (3,) or not np.all(low < high):
            raise ValueError("workspace_low and workspace_high must be 3 values with low < high")
        object.__setattr__(self, "workspace_low", tuple(float(v) for v in low))
        object.__setattr__(self, "workspace_high", tuple(float(v) for v in high))


@dataclass(slots=True, frozen=True)
class SafetyViolation:
    """The first violated limit: ``reason`` (one of ``REASONS``), when, by how much."""

    reason: str
    t: float  # the frame's time (data.time) [s]
    value: float  # the measured value (for ``workspace``: the distance outside the box)
    limit: float
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def info(self) -> dict[str, Any]:
        """JSON-ready event info for ``SAFETY_STOP``."""
        return {
            "reason": self.reason,
            "value": self.value,
            "limit": self.limit,
            "detail": self.detail,
            **self.extra,
        }


class SafetySupervisor:
    """Checks the limits every control tick (refer to the module doc). One per episode."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ids: SceneIds,
        limits: SafetyLimits | None = None,
        dt: float | None = None,
    ) -> None:
        self.model, self.data, self.ids = model, data, ids
        self.limits = limits if limits is not None else SafetyLimits()
        self.dt = float(model.opt.timestep) if dt is None else float(dt)
        lim = self.limits
        ranges = np.array([model.joint(name).range for name in ARM_JOINTS])
        self._q_low = ranges[:, 0] + lim.joint_margin
        self._q_high = ranges[:, 1] - lim.joint_margin
        self._box_low = np.asarray(lim.workspace_low)
        self._box_high = np.asarray(lim.workspace_high)
        self._sat_ticks = max(1, round(lim.saturation_s / self.dt))
        self._sat_run = 0
        self.violation: SafetyViolation | None = None

    def check(
        self,
        frame: SensorFrame,
        target: np.ndarray | None = None,
        saturated: bool = False,
    ) -> SafetyViolation | None:
        """The violation this tick (latched: ``None`` after the first one).

        ``frame`` is this tick's sensor frame, ``target`` the commanded ee position
        (OSC control; ``None`` in torque control), ``saturated`` whether the arm
        torque command hit a motor limit this tick.
        """
        if self.violation is not None:
            return None
        self._sat_run = self._sat_run + 1 if saturated else 0
        found = self._first(frame, target)
        if found is not None:
            self.violation = found
            logger.warning(
                "[SAFETY] EMERGENCY t=%.3f s %s: %.4g (limit %.4g) %s",
                found.t,
                found.reason,
                found.value,
                found.limit,
                found.detail,
            )
        return found

    def _first(self, frame: SensorFrame, target: np.ndarray | None) -> SafetyViolation | None:
        lim, t = self.limits, float(frame.t)
        low, high = self._box_low, self._box_high
        if target is not None:
            out = _outside(np.asarray(target, dtype=float), low - _EPS_BOX, high + _EPS_BOX)
            if out > 0.0:
                return SafetyViolation(
                    "workspace", t, out, 0.0, "target", {"target": _list(target)}
                )
        force = frame.wrench[:3]
        if np.all(np.isfinite(force)):
            norm = float(np.linalg.norm(force))
            if norm > lim.max_force:
                return SafetyViolation("wrist_force", t, norm, lim.max_force)
        torque = frame.wrench[3:]
        if np.all(np.isfinite(torque)):
            norm = float(np.linalg.norm(torque))
            if norm > lim.max_torque:
                return SafetyViolation("wrist_torque", t, norm, lim.max_torque)
        if _outside(frame.ee_pos, low - lim.box_margin, high + lim.box_margin) > 0.0:
            out = _outside(frame.ee_pos, low, high)  # how far past the box (> the margin)
            return SafetyViolation(
                "workspace", t, out, lim.box_margin, "ee", {"ee": _list(frame.ee_pos)}
            )
        lin = float(np.linalg.norm(frame.ee_vel[:3]))
        if lin > lim.max_lin_speed:
            return SafetyViolation("ee_speed", t, lin, lim.max_lin_speed, "linear")
        ang = float(np.linalg.norm(frame.ee_vel[3:]))
        if ang > lim.max_ang_speed:
            return SafetyViolation("ee_speed", t, ang, lim.max_ang_speed, "angular")
        q = self.data.qpos[self.ids.arm_qpos]
        beyond = np.maximum(self._q_low - q, q - self._q_high)
        if np.any(beyond > 0.0):
            j = int(np.argmax(beyond))
            margin = lim.joint_margin - float(beyond[j])  # rad left to the range end
            return SafetyViolation("joint_limit", t, margin, lim.joint_margin, ARM_JOINTS[j])
        if self._sat_run >= self._sat_ticks:
            held = self._sat_run * self.dt
            return SafetyViolation("torque_saturation", t, held, lim.saturation_s)
        return None


def _outside(point: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """How far [m] ``point`` lies outside the box (0 inside; any non-finite point is out)."""
    if not np.all(np.isfinite(point)):
        return math.inf
    gap = np.maximum(low - point, point - high)
    return float(np.linalg.norm(np.maximum(gap, 0.0)))


def _list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1)]
