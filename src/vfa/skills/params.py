"""Skill parameters θ of the grasp-and-lift skill (T08, ARCHITECTURE §4).

``SkillParams`` holds the five numbers System 2 chooses and System 1 executes:

- ``grip_force`` [N per finger]: the squeeze after the probe (``Gripper.squeeze``).
- ``close_speed`` [m/s]: the gripper's speed limit for that squeeze. The fingers
  already touch the cube after the probe, so it sets the force ramp
  (250·``close_speed`` N/s per finger), not an impact (refer to ``vfa.skills.pick``).
- ``lift_accel`` [m/s²]: the peak acceleration of the minimum-jerk lift.
- ``grasp_depth`` [fraction of the cube's height]: how far below the cube's top
  face the fingertip center (the ``ee`` site) closes; 0.5 is the cube's center.
  The lower bound keeps the probe safe: at the probe grip a heavy cube stays on
  the table while the fingers slide 1 cm up, and from a shallow grasp they
  caught the top edge of a 4 cm cube and pressed it in the set-down. Worst-case
  fragile cubes (4 cm, 0.2 kg, μ 0.6, break at 6 N, softest contact) broke in
  38 of 40 probes at 0.30 and 3 of 40 at 0.35; at 0.40, 0 of 80 (also with
  sampled softness), with a probe peak of at most 3.45 N summed.
- ``approach_speed`` [m/s]: the peak target speed of the approach and the descent.

``BOUNDS`` fit the lift task's 10 s limit with the 1.5 s hold included: at the
slowest corner tested (all speeds and accelerations at their lower bounds, the
largest grip, the deepest grasp) on the farthest, smallest, heaviest cube
(x 0.45, y ±0.15, 4 cm, 1 kg) the hold ends at 9.77 s (``tests/test_skill.py``).
The grip's upper bound, 25 N, is 1.5× the hardest cube's need in the T05 map
(16.6 N, ``docs/FEASIBILITY.md`` §3). The closing speed, 0.05–0.5 m/s, is the
top of the gripper's 0.001–0.5 m/s range (slower ramps take too long). The
defaults are plausible values inside the bounds, not tuned ones (T10 defines
the baselines).

``to_vector`` / ``from_vector`` use ``FIELDS`` order. ``to_dict`` / ``to_json``
and ``from_dict`` / ``from_json`` give the JSON form (the recorder's ``params``).
Every constructor checks that each value is a finite number inside its bounds.

RuVector mapping: the Rust ``skill_learning::Skill`` stores only a trajectory, a
confidence, and an execution count; θ has no Rust type. ARCHITECTURE §10 keeps it
in a Python side table beside each ``Skill`` and each ``Episode`` (JSON in
``actions``), so ``SkillParams`` is the payload of that side table.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

FIELDS: tuple[str, ...] = (
    "grip_force",
    "close_speed",
    "lift_accel",
    "grasp_depth",
    "approach_speed",
)
BOUNDS: dict[str, tuple[float, float]] = {
    "grip_force": (0.5, 25.0),  # N per finger
    "close_speed": (0.05, 0.5),  # m/s
    "lift_accel": (0.3, 3.0),  # m/s²
    "grasp_depth": (0.4, 0.5),  # fraction of the cube height below its top face
    "approach_speed": (0.35, 0.5),  # m/s
}
UNITS: dict[str, str] = {
    "grip_force": "N",
    "close_speed": "m/s",
    "lift_accel": "m/s²",
    "grasp_depth": "",
    "approach_speed": "m/s",
}
LOW = np.array([BOUNDS[f][0] for f in FIELDS])
HIGH = np.array([BOUNDS[f][1] for f in FIELDS])


@dataclass(slots=True, frozen=True)
class SkillParams:
    """θ for ``GraspLiftSkill`` (refer to the module doc for units and bounds)."""

    grip_force: float = 8.0
    close_speed: float = 0.1
    lift_accel: float = 0.6
    grasp_depth: float = 0.5
    approach_speed: float = 0.4

    def __post_init__(self) -> None:
        for name in FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            value = float(value)
            low, high = BOUNDS[name]
            if not (math.isfinite(value) and low <= value <= high):
                raise ValueError(f"{name} must be in [{low}, {high}], got {value!r}")
            object.__setattr__(self, name, value)

    # ------------------------------------------------------------- vector form

    def to_vector(self) -> np.ndarray:
        """θ as a float64 vector in ``FIELDS`` order."""
        return np.array([getattr(self, f) for f in FIELDS], dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: Any, clip: bool = False) -> SkillParams:
        """θ from a vector in ``FIELDS`` order; ``clip`` clamps it into the bounds first."""
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape != (len(FIELDS),):
            raise ValueError(f"vector must have {len(FIELDS)} entries, got shape {v.shape}")
        if clip:
            if not np.all(np.isfinite(v)):
                raise ValueError(f"vector must be finite, got {v!r}")
            v = np.clip(v, LOW, HIGH)
        return cls(**{f: float(x) for f, x in zip(FIELDS, v, strict=True)})

    # --------------------------------------------------------------- JSON form

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SkillParams:
        """θ from a mapping with exactly the ``FIELDS`` keys (no defaults, no extras)."""
        if not isinstance(data, Mapping):
            raise TypeError(f"expected a mapping, got {type(data).__name__}")
        keys = set(data)
        if keys != set(FIELDS):
            missing, extra = sorted(set(FIELDS) - keys), sorted(keys - set(FIELDS))
            raise ValueError(f"SkillParams keys: missing {missing}, unknown {extra}")
        return cls(**{f: data[f] for f in FIELDS})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> SkillParams:
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """A short log form, e.g. ``grip=8.0N close=0.10m/s ...``."""
        short = {"grip_force": "grip", "close_speed": "close", "lift_accel": "lift"}
        parts = []
        for f in fields(self):
            name = short.get(f.name, f.name.split("_")[0])
            parts.append(f"{name}={getattr(self, f.name):.3g}{UNITS[f.name]}")
        return " ".join(parts)
