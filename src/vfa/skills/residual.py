"""Residual System 1 policy: a learned, bounded correction during the grasp (T21).

``ResidualPolicy`` is a linear map from observable features to two bounded
outputs, applied on top of ``GraspLiftSkill``'s commands in SQUEEZE, LIFT, and
HOLD. It is off by default: a skill made without one runs exactly as before.

**Features** (``FEATURES``, 7): the phase one-hot (SQUEEZE, LIFT, HOLD); the
tared, low-passed wrist force Fz and ‖Fxy‖ and the held load, each in units of
10 N; and the slip signal, 1 − load/peak of the slip detector while it watches a
carry (0 otherwise, clipped to [0, 1]; it rises as the cube slides in the grip).

**Outputs.** ``z = W · x`` (W is 2 × 7).

- Δgrip-force, as a per-tick step: System 1's grip command (θ's grip, or a slip
  reflex's raise) is reached through a residual command that rises by at most
  ``STEP_MIN + (STEP_MAX − STEP_MIN) · sigmoid(z_grip)`` N per finger per control
  tick (5–50 N/s; a lower command applies at once). At ``W = 0`` the step allows
  27.5 N/s, about the gripper's own squeeze ramp at the default 0.1 m/s closing
  speed (25 N/s), so an untrained residual changes little. With a residual,
  SQUEEZE waits until the motor reaches θ's grip (the floor guarantees progress),
  within the time the lift and the hold still need before the env's time limit.
  A step limit, not an additive offset: on memory's grasps from the calibration
  seeds (``scripts/probe_residual.py``) the jerk² comes about equally from
  SQUEEZE (the squeeze ramp) and LIFT (where the grip rises only through
  slip-reflex raises and escalations, besides sensor noise). A constant 2 N less
  in SQUEEZE cut the jerk 7 %, while limiting every rise to 10 N/s cut it 24 % for
  0.7 points of success (docs/RESULTS.md §11).
- Δz-velocity: ``VZ_MAX · tanh(z_z)`` m/s added to the ee target's vertical
  motion in LIFT and HOLD only (in SQUEEZE the hand rests on the cube at the
  table), integrated into an offset clamped to ``Z_OFFSET`` so the hold stays
  above the env's 10 cm success line.

The weights are trained by ``vfa.experiments.train_residual`` (numpy ES) and the
benchmark's ``memory_full+residual`` condition reads the committed
``configs/residual.npz``.

RuVector mapping: the residual is part of the REACTIVE executor
(``cognitive_core::CognitiveMode::Reactive``): it reshapes the ``ActionCommand``
that System 1 sends every control tick. The crate has no learned residual; it
is ours.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESIDUAL_PATH = REPO_ROOT / "configs" / "residual.npz"
PHASES: tuple[str, ...] = ("SQUEEZE", "LIFT", "HOLD")
FEATURES: tuple[str, ...] = (
    "squeeze",
    "lift",
    "hold",
    "fz",
    "fxy",
    "load",
    "slip",
)
OUTPUTS: tuple[str, ...] = ("grip_step", "z_velocity")
FORCE_SCALE = 10.0  # N per feature unit
STEP_MIN = 0.01  # N per finger per control tick (5 N/s at 500 Hz): the command always rises
STEP_MAX = 0.1  # N per finger per control tick (50 N/s at 500 Hz)
VZ_MAX = 0.02  # m/s
Z_OFFSET = (-0.005, 0.02)  # m: the clamp of the integrated vertical offset


@dataclass(slots=True, frozen=True, eq=False)
class ResidualPolicy:
    """The linear residual (refer to the module doc); ``weights`` is (2, 7)."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros((len(OUTPUTS), len(FEATURES))))

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (len(OUTPUTS), len(FEATURES)) or not np.all(np.isfinite(w)):
            raise ValueError(f"weights must be finite with shape {(len(OUTPUTS), len(FEATURES))}")
        w = w.copy()
        w.setflags(write=False)
        object.__setattr__(self, "weights", w)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ResidualPolicy) and np.array_equal(self.weights, other.weights)

    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def from_vector(cls, vector: Any) -> ResidualPolicy:
        return cls(np.asarray(vector, dtype=np.float64).reshape(len(OUTPUTS), len(FEATURES)))

    def to_vector(self) -> np.ndarray:
        return self.weights.reshape(-1).copy()

    def to_dict(self) -> dict[str, Any]:
        """JSON form (a benchmark chunk key carries it)."""
        return {
            "features": list(FEATURES),
            "outputs": list(OUTPUTS),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResidualPolicy:
        if list(data.get("features", ())) != list(FEATURES) or list(
            data.get("outputs", ())
        ) != list(OUTPUTS):
            raise ValueError("residual weights for other features or outputs")
        return cls(np.asarray(data["weights"], dtype=np.float64))

    def save(self, path: str | Path = RESIDUAL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, features=np.array(FEATURES), outputs=np.array(OUTPUTS))
        return path

    @classmethod
    def load(cls, path: str | Path = RESIDUAL_PATH) -> ResidualPolicy:
        with np.load(path, allow_pickle=False) as f:
            if tuple(f["features"]) != FEATURES or tuple(f["outputs"]) != OUTPUTS:
                raise ValueError(f"{path}: residual weights for other features or outputs")
            return cls(f["weights"])

    # ------------------------------------------------------------ acting

    def outputs(self, x: np.ndarray) -> tuple[float, float]:
        """(grip step [N per finger per tick], Δz velocity [m/s]) for features ``x``."""
        z = self.weights @ x
        gate = 1.0 / (1.0 + math.exp(-float(np.clip(z[0], -50.0, 50.0))))
        step = STEP_MIN + (STEP_MAX - STEP_MIN) * gate
        return step, VZ_MAX * math.tanh(float(z[1]))


def features(
    phase: str, wrench: np.ndarray, load: float, slip_load: float, slip_peak: float, watching: bool
) -> np.ndarray:
    """The feature vector ``FEATURES`` (refer to the module doc). NaN readings count 0."""
    x = np.zeros(len(FEATURES))
    if phase in PHASES:
        x[PHASES.index(phase)] = 1.0
    fx, fy, fz = (float(v) for v in np.asarray(wrench, dtype=float)[:3])  # world axes
    x[3] = fz / FORCE_SCALE if math.isfinite(fz) else 0.0
    fxy = math.hypot(fx, fy)
    x[4] = fxy / FORCE_SCALE if math.isfinite(fxy) else 0.0
    x[5] = load / FORCE_SCALE if math.isfinite(load) else 0.0
    if watching and math.isfinite(slip_load) and math.isfinite(slip_peak) and slip_peak > 0.0:
        x[6] = min(max(1.0 - slip_load / slip_peak, 0.0), 1.0)
    return x


@dataclass(slots=True)
class ResidualState:
    """One attempt's residual command and vertical offset."""

    command: float | None = None  # the residual grip command [N per finger]
    z_offset: float = 0.0  # [m]

    def grip(self, target: float, step: float, start: float) -> float:
        """The next grip command toward ``target`` (``start``: the actuator's squeeze,
        where the residual command begins)."""
        if self.command is None:
            self.command = start
        if target <= self.command:
            self.command = target  # a lower command applies at once
        else:
            self.command = min(target, self.command + step)
        return self.command

    def move(self, vz: float, dt: float) -> float:
        lo, hi = Z_OFFSET
        self.z_offset = min(max(self.z_offset + vz * dt, lo), hi)
        return self.z_offset
