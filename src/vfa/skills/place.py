"""Pick-and-place skill with a force-aware place: System 1 (T19, ARCHITECTURE §4, §6).

``PickPlaceSkill`` runs ``GraspLiftSkill``'s phases up to the lift, then carries
the cube to the place target and sets it down (``PandaVFAEnv`` with
``task="pick_place"``, ``control="osc"``, ``step_hz=500``, ``auto_carry=False``,
and an ``EventConfig`` whose release phases are ``RELEASE_PHASES``;
``make_env`` builds it)::

    APPROACH → DESCEND → PROBE → SQUEEZE → LIFT
      → TRANSPORT → PLACE_DESCEND → CONTACT (F/T unload) → RELEASE → RETREAT

**Phases.** TRANSPORT carries the cube from the top of the lift to
``PRE_PLACE_M`` above its release point over the target (the env's ``goal``)
on one straight minimum-jerk path, no faster than ``TRANSPORT_SPEED`` and with
a peak acceleration of at most ``θ.lift_accel``, under the lift's slip watch,
reflex, and ``DROP`` check. PLACE_DESCEND first stops the slip watch (the table taking
the load is not a drop), then moves the ee target down at ``θ.place_speed``
(reached over ``RAMP_S``) and watches the wrist's held load: once it stays
below ``UNLOAD_FRACTION`` of the load carried into the descent for
``UNLOAD_S``, the table carries the cube, the target stops where it is, and
CONTACT holds it for ``CONTACT_S``. The release point is the grasp height plus
``θ.release_height``: with a height of 0 the target may go ``MAX_PRESS_M``
further down to find the table; with a height above 0 the descent stops there,
and an unload on the way (a cube that slid down in the grip) still ends it in
CONTACT. RELEASE opens the fingers and waits ``OPEN_S``; RETREAT raises the open
hand ``RETREAT_M`` and waits up to ``JUDGE_S`` for the env's verdict.

**Outcomes.** The env judges the place from ground truth: ``SUCCESS`` (the cube
came down within 4 cm of the target, rests flat within 2 cm of it, unbroken,
the pads clear), ``PLACE_BREAK`` (a break in or after a release phase),
``TOPPLE``, or ``DROP`` (the cube left the fingers more than 2 cm up). An
attempt the skill ends after RETREAT without a verdict is labelled from ground
truth for scoring only: ``TOPPLE`` if the cube rests on no face, else
``PLACE_MISS`` (it rests outside the target tolerance, the roadmap's place
"MISS": a separate label from the probe's ``MISS``, which means no grasp), or
``DROP`` if it is not on the table. The pick part ends as in the lift
(``MISS``, ``BREAK``, ``DROP``; a ``SLIP`` escalation and the reflex work
through the transport).

**What the skill reads.** As the lift skill: the observation (the place target
is ``obs["goal"]``), the observable events, and the robot's own state. The
expected table height comes from the grasp pose: the cube rested on the table
there.

RuVector mapping: as ``GraspLiftSkill``, the REACTIVE executor of
``cognitive_core::CognitiveMode::Reactive``; the place adds ``ActionType::Move``
and ``ActionType::Release`` commands. Its θ rides in the side table of
ARCHITECTURE §10 like the lift's.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vfa.sim import env as E
from vfa.sim.sensors import EventConfig, SensorConfig, read_privileged
from vfa.skills import pick
from vfa.skills.params import PLACE_FIELDS
from vfa.skills.pick import Actions, GraspLiftSkill, lift_duration, move_duration

logger = logging.getLogger(__name__)

PHASES: tuple[str, ...] = (
    *pick.PHASES[:-1],
    "TRANSPORT",
    "PLACE_DESCEND",
    "CONTACT",
    "RELEASE",
    "RETREAT",
)
# The env's release phases: no DROP in them, and a break in or after one is a
# PLACE_BREAK (until a new carry starts).
RELEASE_PHASES = frozenset({"PLACE_DESCEND", "CONTACT", "RELEASE", "RETREAT"})
TRANSPORT_SPEED = 0.3  # peak ee speed of the transport [m/s]
PRE_PLACE_M = 0.01  # the descent at θ.place_speed starts this far above the release point
RAMP_S = 0.03  # the descent reaches θ.place_speed in this time
UNLOAD_FRACTION = 0.5  # the table carries the cube: held load below this share ...
UNLOAD_S = 0.01  # ... for this long
MIN_LOAD_N = 0.15  # below this carried load there is no unload to see (the slip watch's arm)
MAX_PRESS_M = 0.005  # with release_height 0 the target goes at most this below the grasp height
CONTACT_S = 0.1  # CONTACT: hold while the table takes the load
OPEN_S = 0.3  # RELEASE: the fingers open
RETREAT_M = 0.08  # RETREAT: raise the open hand
JUDGE_S = 0.5  # wait for the env's verdict after the retreat
LOAD_WINDOW_S = 0.1  # the carried load: the mean over the last 0.1 s before the descent


def make_env(sensor_config: SensorConfig | None = None, **kwargs: Any) -> E.PandaVFAEnv:
    """The env the skill drives: pick-and-place, osc at 500 Hz, the skill's carries,
    the skill's release phases (``kwargs``: more ``PandaVFAEnv`` options)."""
    return E.PandaVFAEnv(
        "pick_place",
        "osc",
        step_hz=500,
        auto_carry=False,
        sensor_config=sensor_config,
        event_config=EventConfig(release_phases=RELEASE_PHASES),
        **kwargs,
    )


class PickPlaceSkill(GraspLiftSkill):
    """The pick-and-place phase machine (refer to the module doc)."""

    task = "pick_place"
    # A slip escalation (LIFT or TRANSPORT) changes only the grip and the closing speed:
    # the transport already heads for the release point.
    slip_ran = (*GraspLiftSkill.slip_ran, *PLACE_FIELDS)

    def validate(self, env: Any) -> None:
        super().validate(env)
        if not RELEASE_PHASES <= env.event_config.release_phases:
            raise ValueError("make the env with EventConfig(release_phases=RELEASE_PHASES)")

    def _after_lift(self, grasp: np.ndarray, top: np.ndarray) -> Actions:
        env, theta = self._env, self.theta
        sensors = env.sensors
        goal = np.array(self._obs["goal"], dtype=float)
        release_z = grasp[2] + theta.release_height
        pre = np.array([goal[0], goal[1], release_z + PRE_PLACE_M])

        self._enter("TRANSPORT")
        distance = float(np.linalg.norm(pre - top))
        duration = max(
            move_duration(distance, TRANSPORT_SPEED), lift_duration(theta.lift_accel, distance)
        )
        loads: list[float] = []

        def carried() -> bool:
            loads.append(float(self._obs["load"][0]))
            return self._carry()

        if (yield from self._move(top, pre, duration, carried)):
            return False
        tail = loads[-max(1, round(LOAD_WINDOW_S / self._dt)) :]
        carried_load = float(np.mean(tail))

        # PLACE_DESCEND: a guarded move down at θ.place_speed until the table takes the load.
        sensors.unwatch_slip()  # the table taking the load is not a slip or a drop
        self._enter("PLACE_DESCEND")
        theta = self.theta  # an escalation in the transport may have changed it
        floor = release_z - (MAX_PRESS_M if theta.release_height <= 0.0 else 0.0)
        speed = theta.place_speed
        target = pre.copy()
        below = 0  # ticks with the held load under the unload line
        unload_ticks = max(2, round(UNLOAD_S / self._dt))
        watch = carried_load >= MIN_LOAD_N
        contact = False
        k = 0
        while target[2] > floor:
            k += 1
            v = speed * min(1.0, k * self._dt / RAMP_S)
            target = np.array([pre[0], pre[1], max(target[2] - v * self._dt, floor)])
            yield self._action(target)
            if watch and float(self._obs["load"][0]) < UNLOAD_FRACTION * carried_load:
                below += 1
                if below >= unload_ticks:
                    contact = True
                    break
            else:
                below = 0
        if contact:
            logger.info(
                "[S1] unload at %.1f mm above the grasp height", 1e3 * (target[2] - grasp[2])
            )
            self._enter("CONTACT")
            yield from self._hold(target, CONTACT_S)
        self._enter("RELEASE")
        self._grip = 0.0
        yield from self._hold(target, OPEN_S)
        self._enter("RETREAT")
        up = target + (0.0, 0.0, RETREAT_M)
        yield from self._move(target, up, move_duration(RETREAT_M, TRANSPORT_SPEED))
        yield from self._hold(up, JUDGE_S)
        self.stop_reason = "PLACE_UNJUDGED"
        return False

    def label(self, env: E.PandaVFAEnv) -> str:
        """Ground-truth label of an attempt the skill ended (refer to the module doc)."""
        if self.stop_reason != "PLACE_UNJUDGED":
            return super().label(env)
        priv = read_privileged(env.model, env.data, env.ids, env.monitor)
        if not env._cube_contacts() & 1:  # not on the table
            return "DROP"
        if not E._flat(priv.cube_quat, env.event_config.place_tilt_deg):
            return "TOPPLE"
        return "PLACE_MISS"
