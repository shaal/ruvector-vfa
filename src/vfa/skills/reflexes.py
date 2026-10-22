"""System 1 reflexes (T08): the slip reflex of the grasp-and-lift skill.

``SlipReflex.react(grip)`` answers one ``SLIP`` event during a carry (``LIFT``,
``HOLD``): it returns a stronger grip, ``grip`` raised by ``step_fraction``
(at most ``max_step_n``, never above ``max_force``), until it has fired
``max_triggers`` times; after that it returns ``None`` and the skill escalates
``SLIP_UNRECOVERED`` to System 2.

The skill re-arms the slip detector after a raise (``SlipDetector.rearm``), so
the reflex fires again on a further drop. **Known limit:** when the fingers
re-catch a sliding cube, the load can overshoot the re-armed peak, and a
one-tick force cut of the gripper's speed servo about 0.1 s later then reads
as a second slip, even when the cube has stopped. On a constructed worst case
(``fragile_soft``, 4 cm, 0.2 kg, μ 0.6, break at 6 N, grip 2 N, lift 3 m/s²,
150 poses and softnesses × 2 closing speeds) that second raise broke 151 of 300
cubes; on natural ``fragile_soft`` draws at the same θ, 0 of 400 broke. A
variant that kept re-arming until a raised grip had settled removed those
breaks, but it also hid real slides: on 290 random-θ attempts with a reflex
raise it lifted 43 cubes and escalated ``SLIP_UNRECOVERED`` 8 times, against 61
and 252 with the single re-arm, which the reflex keeps.

The step is small on purpose (25 %, at most 2 N, twice). A false ``SLIP`` on a
``fragile_soft`` cube held at 2 N per finger then reaches 2.5 N, and a second
one 3.1 N (a summed 6.25 N, the bottom of the class's 6–12 N break range); a
grip far below the need (``heavy_slippery`` at 2 N needs 6.5–16.6 N) is not
rescued (2 → 2.5 → 3.1 N): fixing a wrong θ is System 2's job.

RuVector mapping: a stimulus → response rule of the REACTIVE mode
(``cognitive_core::CognitiveMode::Reactive``), which acts without the
deliberative ``think()`` step; ``SLIP_UNRECOVERED`` is the switch to
``CognitiveMode::Deliberative``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vfa.skills.params import BOUNDS

REFLEX_STEP_FRACTION = 0.25
REFLEX_MAX_STEP_N = 2.0
REFLEX_MAX_TRIGGERS = 2


@dataclass(slots=True)
class SlipReflex:
    """Bounded grip raises on ``SLIP`` (refer to the module doc). One per attempt."""

    step_fraction: float = REFLEX_STEP_FRACTION
    max_step_n: float = REFLEX_MAX_STEP_N
    max_triggers: int = REFLEX_MAX_TRIGGERS
    max_force: float = BOUNDS["grip_force"][1]
    triggers: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for name in ("step_fraction", "max_step_n", "max_force"):
            value = getattr(self, name)
            if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        if isinstance(self.max_triggers, bool) or not isinstance(self.max_triggers, int):
            raise ValueError(f"max_triggers must be an int >= 0, got {self.max_triggers!r}")
        if self.max_triggers < 0:
            raise ValueError(f"max_triggers must be an int >= 0, got {self.max_triggers!r}")

    def react(self, grip: float) -> float | None:
        """The raised grip [N per finger] for one ``SLIP``, or ``None`` once spent."""
        if self.triggers >= self.max_triggers:
            return None
        self.triggers += 1
        step = min(self.step_fraction * grip, self.max_step_n)
        return min(grip + step, self.max_force)
