"""Grasp-and-lift skill with a touch-and-heft probe: System 1 (T08, ARCHITECTURE §4, §6).

``GraspLiftSkill`` is a phase machine that drives ``PandaVFAEnv`` (lift task,
``control="osc"``, ``step_hz=500``: one control tick per step, and
``auto_carry=False``: the skill starts and ends its own carries)::

    APPROACH → DESCEND → PROBE (gentle close + 1 cm heft + set-down) → SQUEEZE → LIFT → HOLD

``run_attempt(env, skill, theta, on_escalation)`` resets the env, runs one
attempt, and returns an ``AttemptResult`` (success, failure mode, the recorded
``Episode``).

**Phases.** Every ee motion is a minimum-jerk path whose per-tick step is the
env action (the env feeds the step's velocity forward, as T05 requires).
APPROACH moves above the cube by ``APPROACH_HEIGHT`` at a peak speed of
``θ.approach_speed``, and DESCEND lowers to the grasp height (``θ.grasp_depth``
of the cube's height below its top face). PROBE closes at ``PROBE_SPEED`` to
``PROBE_FORCE`` per finger, which is safe for every class (a summed 3 N, half
the lowest ``fragile_soft`` break force; ``params`` explains the grasp depth
this needs). It then raises the hand ``HEFT_M``
with the slip detector watching, holds, stops the watch, and lowers back to the
grasp height whatever the load, because at the probe grip most heavy or
slippery cubes stay on the table while the fingers slide up. A heft of 1 cm
never reaches the 2 cm ``LIFTED`` line, so a ``DROP`` in the probe cannot end
the episode. Then the skill escalates ``PROBE_DONE`` with ``ProbeFeatures``
and waits for the callback, which may return a new θ for the rest of the
attempt. SQUEEZE ramps the grip to ``θ.grip_force`` at ``θ.close_speed``.
LIFT raises the hand ``LIFT_M`` with a peak acceleration of ``θ.lift_accel``
under a new slip watch, and HOLD keeps it there until the env reports success
or ``HOLD_S`` runs out.

**Closing speed.** The probe always closes slowly, so the fingers already
touch the cube when ``θ.close_speed`` applies. The speed then sets only the
force ramp (250·``close_speed`` N/s per finger). Measured on 12 ``fragile_soft``
cubes at 2, 2.5, and 3 N per finger: a ramp at 0.05 and at 0.5 m/s give the
same peak (median 1.000× the summed command) and no break. The T05 impact
effect comes only from a close from open.

**Reflex and escalations.** In LIFT and HOLD a ``SLIP`` runs ``SlipReflex``
(a bounded grip raise, then a re-armed detector; refer to
``vfa.skills.reflexes`` for a known limit). Once the reflex is spent, a
further ``SLIP`` escalates ``SLIP_UNRECOVERED`` (once per attempt). A ``DROP``
ends the attempt. ``PROBE_DONE`` fires exactly once in every attempt that
finishes its probe; a probe close that ends without both pads closed on the
cube is a ``MISS``, which ends the attempt before ``PROBE_DONE``. A θ returned
at an escalation applies to what is still to run: at ``PROBE_DONE`` everything
but ``grasp_depth`` and ``approach_speed``, at ``SLIP_UNRECOVERED`` only the grip
and the closing speed. The fields that already ran keep their values (a warning
says so), so θ always records what ran. A θ returned at ``SLIP_UNRECOVERED``
replaces the grip the reflex raised; return ``None`` (or ``esc.theta``) to keep
it. A new grip ramps at 250·``close_speed`` N/s per finger, so a slow closing
speed makes a late rescue slow too.

**What the skill reads.** Only the observation, the observable events in
``info["events"]``, and the robot's own state (the gripper's motor force). Its
decisions never read ``info["privileged"]``, the object spec, or the break
monitor.
An attempt that the env ends keeps the env's result (``SUCCESS``, ``BREAK``,
``DROP``, ``TIMEOUT``). One that the skill ends keeps the skill's reason for
``MISS`` and ``DROP`` (a ``DROP`` event before the env saw a lift-off); after a
HOLD that never succeeded, ``run_attempt`` reads ground truth for the label
only: ``SLIP`` if the cube's center is 2 cm or more above its rest height,
``DROP`` if not (T05's rule).

**Recorder.** θ goes to ``params`` at the start and at every change
(``params_history``, with the env's sim time): an escalation's θ and each reflex
raise (``grip_force``), so the last entry is the θ that ran at the end. The
probe features go to ``meta["probe"]``.

RuVector mapping: ``GraspLiftSkill`` is the REACTIVE executor of
``cognitive_core::CognitiveMode::Reactive``: it turns a θ into motor commands
(``ActionCommand`` with ``ActionType::Move`` and ``ActionType::Grasp``) every
control tick. An ``Escalation`` is the hand-off that switches the core to
``CognitiveMode::Deliberative`` (System 2); its probe features become a
``Percept``. ``AttemptResult`` carries the ``Outcome { success, reward,
description }`` of one attempt.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from vfa.sim import env as E
from vfa.sim.recorder import Episode
from vfa.sim.sensors import read_privileged
from vfa.skills.params import SkillParams
from vfa.skills.reflexes import SlipReflex

logger = logging.getLogger(__name__)

PHASES: tuple[str, ...] = ("APPROACH", "DESCEND", "PROBE", "SQUEEZE", "LIFT", "HOLD")
PROBE_DONE = "PROBE_DONE"
SLIP_UNRECOVERED = "SLIP_UNRECOVERED"

APPROACH_HEIGHT = 0.06  # ee above the grasp pose before the descent [m]
APPROACH_SETTLE_S = 0.1
DESCEND_SETTLE_S = 0.2
PROBE_FORCE = 1.5  # N per finger: a summed 3 N, half the lowest fragile break force
PROBE_SPEED = 0.05  # m/s
PROBE_SETTLE_S = 0.1  # hold after the probe squeeze is reached
PROBE_MARGIN_S = 0.5  # close timeout beyond the nominal close + ramp time
HEFT_M = 0.01  # T06 handoff: 1-2 cm; a 1 cm heft never fires LIFTED (2 cm)
HEFT_RISE_S = max(0.3, math.sqrt(10.0 * HEFT_M))  # no harsher than the T05 lift
HEFT_HOLD_S = 0.2
HEFT_LOAD_S = 0.1  # the heft load is the mean over the hold's last 0.1 s
SET_DOWN_S = 0.4  # minimum-jerk return to the grasp height
SET_DOWN_SETTLE_S = 0.2
SQUEEZE_SETTLE_S = 0.15  # T05: hold after the motor reaches the force
SQUEEZE_MARGIN_S = 0.5
RAMP_N_PER_S_PER_MPS = 250.0  # the gripper ramps the squeeze at 250·speed N/s per finger
GRIP_TOLERANCE = 0.01  # a squeeze is reached within 1 % of the request
LIFT_M = 0.12  # the env's success line is 10 cm; 2 cm cover payload sag and creep
HOLD_S = 1.5  # the env needs 1 s above 10 cm; the lift path already starts it
SLIP_RISE_M = 0.02  # T05: a failed attempt with the cube this high is a SLIP
MIN_JERK_PEAK_ACC = 10.0 / math.sqrt(3.0)  # peak acceleration of a unit min-jerk path
MIN_JERK_PEAK_SPEED = 1.875  # peak speed of a unit min-jerk path
MIN_MOVE_S = 0.1


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def lift_duration(accel: float, height: float = LIFT_M) -> float:
    """Duration [s] of a minimum-jerk lift of ``height`` with peak acceleration ``accel``."""
    return math.sqrt(MIN_JERK_PEAK_ACC * height / accel)


def move_duration(distance: float, speed: float) -> float:
    """Duration [s] of a minimum-jerk move of ``distance`` with peak speed ``speed``."""
    return max(MIN_MOVE_S, MIN_JERK_PEAK_SPEED * distance / speed)


# ----------------------------------------------------------------- data types


@dataclass(slots=True, frozen=True)
class ProbeFeatures:
    """The touch-and-heft probe's force signature (observable signals only)."""

    force: float  # probe squeeze per finger [N]
    indentation: float  # cube size minus the opening, settled at the probe force [m]
    closure_slope: float | None  # summed probe force / indentation [N/m]; None if <= 0
    heft_load: float  # mean held load over the last HEFT_LOAD_S of the heft hold [N]
    heft_peak: float  # largest held load during the heft [N]
    heft_slip: bool  # the slip detector fired during the heft
    heft_slip_time: float | None  # since the heft started [s]
    heft_drop: bool  # DROP fired during the heft

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class Escalation:
    """A System 1 → System 2 hand-off (``PROBE_DONE`` or ``SLIP_UNRECOVERED``)."""

    kind: str
    t: float  # since the env reset ended [s]
    phase: str
    theta: SkillParams  # θ in force when it fired
    probe: ProbeFeatures | None = None
    info: Mapping[str, Any] = field(default_factory=dict)


EscalationHandler = Callable[[Escalation], SkillParams | None]


@dataclass(slots=True)
class AttemptResult:
    """One attempt: its outcome, θ before and after updates, and the recorded episode."""

    success: bool
    result: str  # SUCCESS or the failure mode (recorder.RESULTS)
    theta: SkillParams  # θ at the start
    theta_final: SkillParams  # θ after every update
    probe: ProbeFeatures | None
    escalations: list[Escalation]
    episode: Episode | None  # None if the env does not record
    sim_s: float  # since the env reset ended
    wall_s: float

    @property
    def failure_mode(self) -> str | None:
        return None if self.success else self.result


# ---------------------------------------------------------------- the skill

Actions = Generator[np.ndarray, None, bool]


class GraspLiftSkill:
    """The grasp-and-lift phase machine (refer to the module doc).

    ``begin(env, theta, on_escalation)`` starts an attempt on a freshly reset
    env; then each ``act(obs, info)`` returns the action for the next
    ``env.step``, or ``None`` once the skill has ended the attempt on its own
    (``stop_reason``: ``MISS``, ``DROP``, or ``HOLD_EXPIRED``).
    """

    def __init__(self) -> None:
        self.phase = "IDLE"
        self.theta: SkillParams | None = None
        self.probe: ProbeFeatures | None = None
        self.escalations: list[Escalation] = []
        self.stop_reason: str | None = None
        self.done = True
        self._program: Actions | None = None

    # ------------------------------------------------------------ interface

    def begin(
        self,
        env: E.PandaVFAEnv,
        theta: SkillParams,
        on_escalation: EscalationHandler | None = None,
    ) -> None:
        if not isinstance(theta, SkillParams):
            raise TypeError(f"theta must be a SkillParams, got {type(theta).__name__}")
        validate_env(env)
        if self._program is not None:
            self._program.close()
        self._env = env
        self._dt = 1.0 / env.step_hz
        self._on_escalation = on_escalation
        self.theta = theta
        self.probe = None
        self.escalations = []
        self.stop_reason = None
        self.done = False
        self.phase = "IDLE"
        self._grip = 0.0  # 0: open
        self._reflex = SlipReflex()
        self._unrecovered = False
        self._obs: dict[str, np.ndarray] | None = None
        self._info: dict[str, Any] = {}
        self._new: set[str] = set()
        self._phase_seen: set[str] = set()  # events since the phase began
        self._program = self._run()

    def act(self, obs: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> np.ndarray | None:
        """The next action, after observing ``obs`` and ``info`` (``None``: the skill ended)."""
        if self.done or self._program is None:
            return None
        self._obs = obs
        self._info = info
        self._new = {e["name"] for e in info.get("events", ())}
        self._phase_seen |= self._new
        try:
            return next(self._program)
        except StopIteration:
            self.done = True
            self._program = None
            return None

    def stop(self) -> None:
        """End the attempt from outside (the env terminated it): drop the program."""
        if self._program is not None:
            self._program.close()
            self._program = None
        self.done = True

    # ------------------------------------------------------------- program

    def _run(self) -> Actions:
        env, theta = self._env, self.theta
        obs = self._obs
        if env.recorder is not None:
            env.recorder.set_params(theta.to_dict(), t=self._sim_time())
        cube = np.array(obs["cube_pos"], dtype=float)
        size = float(obs["cube_size"][0])
        z = max(size * (1.0 - theta.grasp_depth), float(E.TARGET_LOW[2]))
        grasp = np.array([cube[0], cube[1], z])
        above = grasp + (0.0, 0.0, APPROACH_HEIGHT)

        self._enter("APPROACH")
        start = self._target()
        distance = float(np.linalg.norm(above - start))
        yield from self._move(start, above, move_duration(distance, theta.approach_speed))
        yield from self._hold(above, APPROACH_SETTLE_S)

        self._enter("DESCEND")
        yield from self._move(above, grasp, move_duration(APPROACH_HEIGHT, theta.approach_speed))
        yield from self._hold(grasp, DESCEND_SETTLE_S)

        # PROBE: a slow, gentle close; then a 1 cm heft and a set-down.
        self._enter("PROBE")
        env.close_speed = PROBE_SPEED
        self._grip = PROBE_FORCE
        width0 = float(self._obs["gripper_width"][0])
        nominal = max(width0 - size, 0.0) / PROBE_SPEED
        nominal += PROBE_FORCE / (RAMP_N_PER_S_PER_MPS * PROBE_SPEED)
        settle = round(PROBE_SETTLE_S / self._dt)
        reached: int | None = None
        for k in range(round((nominal + PROBE_MARGIN_S) / self._dt)):
            yield self._action(grasp)
            if (
                reached is None
                and "GRASP_CLOSED" in self._phase_seen
                and env.gripper.actuator_squeeze() >= (1.0 - GRIP_TOLERANCE) * PROBE_FORCE
            ):
                reached = k
            if reached is not None and k - reached >= settle:
                break
        # A grasp needs both pads closed on the cube in this phase: a pad contact
        # alone can come from the table.
        if "GRASP_CLOSED" not in self._phase_seen or not np.all(self._obs["finger_contact"]):
            self.stop_reason = "MISS"
            logger.info("[S1] MISS: the probe close found no grasp")
            return False
        # The fingers' sink into the cube at the probe force: the opening against
        # the camera's cube size (soft cubes sink about ten times deeper).
        indentation = size - float(self._obs["gripper_width"][0])

        sensors = env.sensors
        heft_top = grasp + (0.0, 0.0, HEFT_M)
        t_heft = self._time()
        loads: list[float] = []
        slip_time: list[float] = []
        drop: list[bool] = []

        def watch_heft() -> bool:
            loads.append(float(self._obs["load"][0]))
            if "SLIP" in self._new and not slip_time:
                slip_time.append(self._time() - t_heft)
            if "DROP" in self._new:
                drop.append(True)
            return False

        sensors.watch_slip()
        yield from self._move(grasp, heft_top, HEFT_RISE_S, watch_heft)
        yield from self._hold(heft_top, HEFT_HOLD_S, watch_heft)
        sensors.unwatch_slip()  # the table taking the load is not a slip or a drop
        yield from self._move(heft_top, grasp, SET_DOWN_S)
        yield from self._hold(grasp, SET_DOWN_SETTLE_S)

        tail = loads[-max(1, round(HEFT_LOAD_S / self._dt)) :]
        self.probe = ProbeFeatures(
            force=PROBE_FORCE,
            indentation=indentation,
            closure_slope=2.0 * PROBE_FORCE / indentation if indentation > 0.0 else None,
            heft_load=float(np.mean(tail)),
            heft_peak=float(np.max(loads)),
            heft_slip=bool(slip_time),
            heft_slip_time=slip_time[0] if slip_time else None,
            heft_drop=bool(drop),
        )
        if env.recorder is not None:
            env.recorder.meta["probe"] = _json_safe(self.probe.to_dict())
        self._escalate(PROBE_DONE, probe=self.probe, ran=("grasp_depth", "approach_speed"))
        theta = self.theta

        # SQUEEZE: ramp from the probe grip to θ's.
        self._enter("SQUEEZE")
        env.close_speed = theta.close_speed
        self._grip = theta.grip_force
        hold = self._target()
        ramp = abs(theta.grip_force - env.gripper.actuator_squeeze())
        ramp /= RAMP_N_PER_S_PER_MPS * theta.close_speed
        settle = round(SQUEEZE_SETTLE_S / self._dt)
        reached = None
        for k in range(round((ramp + SQUEEZE_MARGIN_S) / self._dt)):
            yield self._action(hold)
            if reached is None and _near(env.gripper.actuator_squeeze(), self._grip):
                reached = k
            if reached is not None and k - reached >= settle:
                break

        # LIFT and HOLD, under a new slip watch.
        self._enter("LIFT")
        sensors.watch_slip()
        base = self._target()
        top = base + (0.0, 0.0, LIFT_M)
        if (yield from self._move(base, top, lift_duration(theta.lift_accel), self._carry)):
            return False
        self._enter("HOLD")
        if (yield from self._hold(top, HOLD_S, self._carry)):
            return False
        self.stop_reason = "HOLD_EXPIRED"
        return False

    def _carry(self) -> bool:
        """Per-tick check in LIFT and HOLD: the slip reflex; True ends the attempt (DROP)."""
        if "DROP" in self._new:
            self.stop_reason = "DROP"
            logger.info("[S1] DROP in %s", self.phase)
            return True
        sensors = self._env.sensors
        if "SLIP" not in self._new:
            return False
        old = self._grip
        raised = self._reflex.react(old)
        if raised is not None:
            logger.info(
                "[S1] slip reflex %d/%d: grip %.2f → %.2f N",
                self._reflex.triggers,
                self._reflex.max_triggers,
                old,
                raised,
            )
            self._grip = raised
            self._set_theta(replace(self.theta, grip_force=raised))  # what runs from now
            sensors.slip.rearm()
        elif not self._unrecovered:
            self._unrecovered = True
            new = self._escalate(
                SLIP_UNRECOVERED,
                info={"grip": old},
                ran=("grasp_depth", "approach_speed", "lift_accel"),
            )
            if new is not None:
                self._env.close_speed = new.close_speed
                self._grip = new.grip_force
                sensors.slip.rearm()
        return False

    # ------------------------------------------------------------- helpers

    def _escalate(
        self,
        kind: str,
        probe: ProbeFeatures | None = None,
        info: Mapping[str, Any] | None = None,
        ran: tuple[str, ...] = (),
    ) -> SkillParams | None:
        """Hand off to System 2; apply the θ it returns. The fields in ``ran`` have
        already acted (their phases are over), so they keep the values that ran."""
        esc = Escalation(kind, self._time(), self.phase, self.theta, probe, dict(info or {}))
        self.escalations.append(esc)
        if probe is not None:
            slope = "n/a" if probe.closure_slope is None else f"{probe.closure_slope:.0f}N/m"
            logger.info(
                "[S1] escalate %s  heft=%.2fN peak=%.2fN slope=%s slip=%d",
                kind,
                probe.heft_load,
                probe.heft_peak,
                slope,
                int(probe.heft_slip),
            )
        else:
            logger.info("[S1] escalate %s %s", kind, dict(esc.info))
        if self._on_escalation is None:
            return None
        new = self._on_escalation(esc)
        if new is None:
            return None
        if not isinstance(new, SkillParams):
            raise TypeError(f"on_escalation must return SkillParams or None, got {new!r}")
        fixed = {f: getattr(self.theta, f) for f in ran}
        if any(getattr(new, f) != v for f, v in fixed.items()):
            logger.warning("[S1] %s: %s already ran; keeping %s", kind, list(ran), fixed)
            new = replace(new, **fixed)
        self._set_theta(new)
        return new

    def _set_theta(self, theta: SkillParams) -> None:
        """θ in force from now on, recorded with the sim time (``params_history``)."""
        if theta == self.theta:
            return
        self.theta = theta
        if self._env.recorder is not None:
            self._env.recorder.set_params(theta.to_dict(), t=self._sim_time())
        logger.info("[S1] θ ← %s", theta.describe())

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self._phase_seen = set()
        self._env.set_phase(phase)
        logger.info("[S1] %s", phase)

    def _time(self) -> float:
        return float(self._info.get("t", 0.0))

    def _sim_time(self) -> float:
        return float(self._env.data.time)

    def _target(self) -> np.ndarray:
        return np.array(self._obs["ee_target"], dtype=float)

    def _action(self, pos: np.ndarray) -> np.ndarray:
        """The env action that moves the ee target to ``pos`` this tick, with the grip."""
        env = self._env
        pos = np.clip(pos, E.TARGET_LOW, E.TARGET_HIGH)
        a = np.empty(4)
        delta = (pos - self._obs["ee_target"]) / env.max_step
        norm = float(np.linalg.norm(delta))
        a[:3] = delta / norm if norm > 1.0 else delta
        a[3] = self._grip / E.GRIP_FORCE_MAX if self._grip > 0.0 else -1.0
        return a

    def _move(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        duration: float,
        check: Callable[[], bool] | None = None,
    ) -> Actions:
        """A minimum-jerk move; returns True if ``check`` ended it."""
        n = max(1, round(duration / self._dt))
        for k in range(1, n + 1):
            yield self._action(start + (goal - start) * min_jerk(k / n))
            if check is not None and check():
                return True
        return False

    def _hold(
        self, pos: np.ndarray, duration: float, check: Callable[[], bool] | None = None
    ) -> Actions:
        for _ in range(round(duration / self._dt)):
            yield self._action(pos)
            if check is not None and check():
                return True
        return False


# ----------------------------------------------------------- run an attempt


def validate_env(env: Any) -> None:
    """Raise unless ``env`` is a ``PandaVFAEnv`` the skill can drive."""
    if not isinstance(env, E.PandaVFAEnv):
        raise TypeError(f"expected a PandaVFAEnv (env.unwrapped), got {type(env).__name__}")
    if env.task != "lift" or env.control != "osc":
        raise ValueError("the skill drives the lift task with control='osc'")
    if env.ticks_per_step != 1:
        raise ValueError("the skill runs at control rate: make the env with step_hz=500")
    if env.auto_carry:
        raise ValueError("the skill starts its own carries: make the env with auto_carry=False")


def run_attempt(
    env: Any,
    skill: GraspLiftSkill,
    theta: SkillParams,
    on_escalation: EscalationHandler | None = None,
    *,
    seed: int | None = None,
    options: Mapping[str, Any] | None = None,
) -> AttemptResult:
    """Reset ``env`` (``seed``, ``options``) and run one attempt of ``skill`` with ``theta``.

    ``env`` may be a Gymnasium wrapper of a ``PandaVFAEnv``; ``on_escalation``
    receives every ``Escalation`` and may return a new θ (refer to the module doc).
    The attempt always starts with a reset: to choose θ from the observation
    first (System 2's attempt-start trigger), reset with the same ``seed`` and
    ``options`` before, which gives the same cube.
    """
    unwrapped = env.unwrapped
    validate_env(unwrapped)
    wall0 = time.perf_counter()
    obs, info = env.reset(seed=seed, options=options)
    skill.begin(unwrapped, theta, on_escalation)
    ended = False
    while True:
        action = skill.act(obs, info)
        if action is None:
            break
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            ended = True
            break
    skill.stop()
    if ended:
        result = unwrapped.result
    else:
        result = _label(unwrapped, skill.stop_reason)
        unwrapped.end_episode(result)
    logger.info("[S1] attempt %s (%s)", result, skill.stop_reason or "env")
    return AttemptResult(
        success=result == "SUCCESS",
        result=result,
        theta=theta,
        theta_final=skill.theta,
        probe=skill.probe,
        escalations=list(skill.escalations),
        episode=unwrapped.episode,
        sim_s=unwrapped.elapsed,
        wall_s=time.perf_counter() - wall0,
    )


def _label(env: E.PandaVFAEnv, stop_reason: str | None) -> str:
    """Ground-truth label for an attempt the skill ended on its own (scoring only)."""
    if stop_reason in ("MISS", "DROP"):
        return stop_reason
    priv = read_privileged(env.model, env.data, env.ids, env.monitor)
    return "SLIP" if float(priv.cube_pos[2]) - priv.rest_z >= SLIP_RISE_M else "DROP"


def _near(value: float, target: float) -> bool:
    return abs(value - target) <= GRIP_TOLERANCE * max(abs(target), 1e-9)


def _json_safe(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in data.items()
    }
