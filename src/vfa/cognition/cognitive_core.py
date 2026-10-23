"""Dual-process cognitive core: System 1 / System 2 with explicit handoffs (T14).

ARCHITECTURE §6, mirroring ``ruvector_robotics::cognitive::cognitive_core``.

``CognitiveCore.run_attempt(env, skill, seed=, options=, previous=)`` runs one
grasp-and-lift attempt. System 1 (``GraspLiftSkill`` with its slip reflex and the
OSC, every control tick) is ``REACTIVE``. System 2 (a ``System2Policy``) is
``DELIBERATIVE`` and runs only at a trigger:

- ``START``: before the attempt, on the observed cube. The core resets the env
  to read the observation; ``pick.run_attempt`` then resets it again with the
  same seed and options, which gives the same cube (so a seed is required).
- ``FAILURE``: replaces ``START`` when ``previous`` (the last attempt at this
  cube) failed; the percept carries an ``ObservedAttempt`` of it.
- ``PROBE``: at the skill's ``PROBE_DONE`` escalation.
- ``ESCALATION``: at ``SLIP_UNRECOVERED`` (any other escalation kind too).

At an escalation the fields that already ran (``Escalation.ran``) keep their
values in the decision, so the logged θ is the θ System 1 runs, and an unchanged
θ goes back to the skill as ``None`` (a returned θ would re-arm the slip
detector, so the stub would change the recording).

Escalations reach System 2 through the skill's callback, which the skill calls
between two ``env.step`` calls and waits for, so System 2 never runs inside a
control tick and no physics time passes while it thinks. A safety stop (the
env's ``SafetySupervisor``: the env holds the arm and ends the attempt with
``SAFETY_STOP``, or with ``BREAK`` if a break in the same tick outranks it)
switches the core to ``EMERGENCY`` until the next attempt starts. After every
attempt the core learns from the ``Outcome`` (``LEARNING``), then is ``IDLE``.

Each step is a ``Handoff`` (``kind``: ``DELIBERATIVE``, ``REACTIVE``,
``EMERGENCY``, or ``LEARNING``; the attempt number, sim time, trigger, θ,
retrieved record ids, System 2 latency) in ``core.handoffs`` and, with
``log_path`` (``runs/<id>/handoffs.jsonl``), one JSON line. A first attempt that
finishes its probe without a slip escalation logs ``DELIBERATIVE(start) →
REACTIVE → DELIBERATIVE(probe) → REACTIVE → LEARNING``.
Console lines carry ``[S2]``, ``[S1]``, and ``[SAFETY]`` (``[MEM]`` comes with
memory, T15).

Percepts hold observable data only (any hidden class information would leak):
the observation (never ``info["privileged"]``) at ``START``; the escalation, its
probe features, and an ``EpisodeView`` of the attempt so far (T11's observable
columns, built from the recorder; None without one) at ``PROBE`` and
``ESCALATION``; and at ``FAILURE`` an ``ObservedAttempt``: the failed attempt's
result label (``SLIP`` and ``DROP`` merged into ``UNDERGRIP``, as in T11's keys:
``run_attempt`` tells an expired hold apart by the cube's true height), θ as
chosen and as it ended, probe features, escalations, safety info, and an
``EpisodeView``. The ``AttemptResult`` itself never reaches a
policy: its episode holds the class, the hidden properties, and ground-truth
columns. JSON lines map NaN and ±inf to null.

``DefaultPolicy`` is the stub System 2: the default θ (T10's ``naive``) at
``START`` and ``FAILURE``, the θ in force at ``PROBE`` and ``ESCALATION`` (it
never lowers a grip the reflex raised). T15 plugs in retrieval.

RuVector mapping: ``CognitiveCore``, ``CognitiveMode`` (``Reactive``,
``Deliberative``, ``Emergency``), ``CognitiveState`` (``Idle``, ``Perceiving``,
``Thinking``, ``Acting``, ``Learning``), ``Percept``, ``Decision``, and
``Outcome`` are the types of the same names in
``ruvector_robotics::cognitive::cognitive_core``. ``perceive`` drops a percept
under the attention threshold and keeps at most ``max_percepts``; ``learn``
adds ``reward × learning_rate`` to the cumulative reward, moves the attention
threshold by 0.01 (down on success, up on failure, within [0.1, 0.9]), clears
the percepts, and ends ``IDLE``, all as in the crate. Deviations: the crate's
``think`` picks the most confident percept, ours asks the System 2 policy about
the newest one; the crate's mode is fixed in its config, ours follows the
triggers; a ``Percept`` carries a mapping (the crate: ``Vec<f64>``), a
``Decision`` carries θ, its trigger, and the retrieved ids (the crate: an
``ActionCommand``), ``Percept.timestamp`` is sim seconds (the crate: ``i64``),
``Outcome.reward`` is +1 for a success and −1 for a failure, ``Decision.utility``
comes from the policy (the stub's is 0), the settings are constructor arguments
(no ``CognitiveConfig``), ``max_percepts`` must be at least 1, and ``think``
rejects a percept that is not a trigger and takes ``keep``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vfa.memory.embeddings import UNDERGRIP, EpisodeView
from vfa.skills import pick
from vfa.skills.params import SkillParams
from vfa.skills.pick import (
    PROBE_DONE,
    AttemptResult,
    Escalation,
    GraspLiftSkill,
    ProbeFeatures,
)

logger = logging.getLogger(__name__)

START = "start"
FAILURE = "failure"
PROBE = "probe"
ESCALATION = "escalation"
TRIGGERS = (START, FAILURE, PROBE, ESCALATION)
UNDERGRIP_RESULT = "UNDERGRIP"  # SLIP or DROP: run_attempt tells them apart by the true height
DEFAULT_THETA = SkillParams()  # T10's naive θ: the θ in force before System 2 decides


class CognitiveMode(Enum):
    REACTIVE = "REACTIVE"  # System 1
    DELIBERATIVE = "DELIBERATIVE"  # System 2
    EMERGENCY = "EMERGENCY"  # the safety supervisor


class CognitiveState(Enum):
    IDLE = "IDLE"
    PERCEIVING = "PERCEIVING"
    THINKING = "THINKING"
    ACTING = "ACTING"
    LEARNING = "LEARNING"


@dataclass(slots=True)
class Percept:
    """What System 2 sees at a trigger (observable data only)."""

    source: str  # the trigger
    data: Mapping[str, Any]
    confidence: float = 1.0
    timestamp: float = 0.0  # sim time since the env reset ended [s]


@dataclass(slots=True)
class Decision:
    """System 2's choice: θ for the rest of the attempt."""

    theta: SkillParams
    trigger: str
    reasoning: str = ""
    utility: float = 0.0
    retrieved: tuple[str, ...] = ()  # memory record ids the choice used


@dataclass(slots=True)
class Outcome:
    success: bool
    reward: float
    description: str

    @classmethod
    def from_result(cls, result: AttemptResult) -> Outcome:
        return cls(result.success, 1.0 if result.success else -1.0, result.result)


@dataclass(slots=True, frozen=True)
class ObservedAttempt:
    """What System 2 may know of a finished attempt (refer to the module doc)."""

    result: str  # SUCCESS or the failure mode, SLIP and DROP merged into UNDERGRIP
    success: bool
    theta: SkillParams  # as chosen
    theta_final: SkillParams  # as it ended (escalations, reflex raises)
    probe: ProbeFeatures | None
    escalations: tuple[Escalation, ...]
    safety: dict[str, Any] | None
    view: EpisodeView | None  # observable columns; None if the env did not record

    @classmethod
    def of(cls, result: AttemptResult) -> ObservedAttempt:
        label = UNDERGRIP_RESULT if result.result in UNDERGRIP else result.result
        view = None
        if result.episode is not None:
            view = replace(EpisodeView.from_episode(result.episode), result=label)
        return cls(
            label,
            result.success,
            result.theta,
            result.theta_final,
            result.probe,
            tuple(result.escalations),
            None if result.safety is None else dict(result.safety),
            view,
        )


class System2Policy(Protocol):
    def decide(self, percept: Percept, theta: SkillParams) -> Decision:
        """θ for the trigger in ``percept``; ``theta`` is the θ in force (the default
        θ at ``START`` and ``FAILURE``)."""
        ...


class DefaultPolicy:
    """The stub System 2 (refer to the module doc)."""

    def __init__(self, theta: SkillParams | None = None) -> None:
        self.theta = DEFAULT_THETA if theta is None else theta

    def decide(self, percept: Percept, theta: SkillParams) -> Decision:
        if percept.source in (START, FAILURE):
            return Decision(self.theta, percept.source, "default θ")
        return Decision(theta, percept.source, "keep θ")


@dataclass(slots=True)
class Handoff:
    """One step of the dual-process loop (a line of ``handoffs.jsonl``)."""

    kind: str  # DELIBERATIVE | REACTIVE | EMERGENCY | LEARNING
    attempt: int
    t: float  # sim time since the env reset ended [s]
    trigger: str | None = None
    theta: dict[str, float] | None = None
    retrieved: list[str] = field(default_factory=list)
    latency_ms: float | None = None  # System 2's wall time
    detail: dict[str, Any] = field(default_factory=dict)


class CognitiveCore:
    """Perceive → think → act → learn around System 1 (refer to the module doc)."""

    def __init__(
        self,
        policy: System2Policy | None = None,
        *,
        log_path: str | Path | None = None,
        attention_threshold: float = 0.5,
        learning_rate: float = 0.01,
        max_percepts: int = 100,
    ) -> None:
        if max_percepts < 1:
            raise ValueError(f"max_percepts must be >= 1, got {max_percepts!r}")
        if not 0.0 <= attention_threshold <= 1.0:  # a trigger's percept (confidence 1) passes
            raise ValueError(f"attention_threshold must be in [0, 1], got {attention_threshold!r}")
        self.policy: System2Policy = DefaultPolicy() if policy is None else policy
        self.log_path = None if log_path is None else Path(log_path)
        self.attention_threshold = float(attention_threshold)
        self.learning_rate = float(learning_rate)
        self.percepts: deque[Percept] = deque(maxlen=max_percepts)
        self.decisions: list[Decision] = []
        self.handoffs: list[Handoff] = []
        self.cumulative_reward = 0.0
        self.attempt = 0
        self.theta = DEFAULT_THETA
        self._mode = CognitiveMode.REACTIVE
        self._state = CognitiveState.IDLE
        self._t = 0.0
        self._env: Any = None

    @property
    def mode(self) -> CognitiveMode:
        return self._mode

    @property
    def state(self) -> CognitiveState:
        return self._state

    # ------------------------------------------------------ the crate's cycle

    def perceive(self, percept: Percept) -> CognitiveState:
        """Buffer ``percept`` unless its confidence is under the attention threshold."""
        self._state = CognitiveState.PERCEIVING
        if percept.confidence >= self.attention_threshold:
            self.percepts.append(percept)
        return self._state

    def think(self, keep: Mapping[str, float] | None = None) -> Decision | None:
        """System 2 on the newest percept (None if there is none). ``keep``: θ fields
        that already ran, kept at their values in the decision."""
        self._state = CognitiveState.THINKING
        if not self.percepts:
            return None
        percept = self.percepts[-1]
        if percept.source not in TRIGGERS:
            raise ValueError(f"percept source must be a trigger {TRIGGERS}, got {percept.source!r}")
        if self._mode is not CognitiveMode.EMERGENCY:
            self._mode = CognitiveMode.DELIBERATIVE
        start = time.perf_counter()
        decision = self.policy.decide(percept, self.theta)
        latency = (time.perf_counter() - start) * 1e3
        if not isinstance(decision, Decision) or not isinstance(decision.theta, SkillParams):
            raise TypeError(f"the policy must return a Decision with SkillParams, got {decision!r}")
        if keep:
            decision = replace(decision, theta=replace(decision.theta, **keep))
        self.decisions.append(decision)
        self._log(
            Handoff(
                CognitiveMode.DELIBERATIVE.value,
                self.attempt,
                self._t,
                percept.source,
                decision.theta.to_dict(),
                list(decision.retrieved),
                latency,
                {"reasoning": decision.reasoning},
            )
        )
        logger.info(
            "[S2] attempt %d trigger=%s %s (%s, %.2f ms)",
            self.attempt,
            percept.source,
            decision.theta.describe(),
            decision.reasoning,
            latency,
        )
        return decision

    def act(self, decision: Decision) -> SkillParams:
        """Hand θ to System 1."""
        self._state = CognitiveState.ACTING
        if self._mode is not CognitiveMode.EMERGENCY:
            self._mode = CognitiveMode.REACTIVE
        self.theta = decision.theta
        self._log(
            Handoff(CognitiveMode.REACTIVE.value, self.attempt, self._t, None, self.theta.to_dict())
        )
        logger.info("[S1] REACTIVE after %s", decision.trigger)
        return decision.theta

    def learn(self, outcome: Outcome) -> None:
        """Feedback from one attempt (the crate's rule), then IDLE."""
        self._state = CognitiveState.LEARNING
        self.cumulative_reward += outcome.reward * self.learning_rate
        if outcome.success:
            self.attention_threshold = max(self.attention_threshold - 0.01, 0.1)
        else:
            self.attention_threshold = min(self.attention_threshold + 0.01, 0.9)
        self._log(
            Handoff(
                CognitiveState.LEARNING.value,
                self.attempt,
                self._t,
                detail={"result": outcome.description, "reward": outcome.reward},
            )
        )
        logger.info("[S2] learn attempt %d: %s", self.attempt, outcome.description)
        self.percepts.clear()
        self._state = CognitiveState.IDLE

    # ------------------------------------------------------------ an attempt

    def run_attempt(
        self,
        env: Any,
        skill: GraspLiftSkill,
        *,
        seed: int,
        options: Mapping[str, Any] | None = None,
        previous: AttemptResult | None = None,
    ) -> AttemptResult:
        """One attempt with System 2 at its triggers (refer to the module doc)."""
        if seed is None:
            raise ValueError("a seed is required: System 2 observes the cube before the attempt")
        pick.validate_env(env.unwrapped)  # before anything is logged
        trigger = FAILURE if previous is not None and not previous.success else START
        self.attempt += 1
        self._mode = CognitiveMode.DELIBERATIVE
        self._t = 0.0
        self.theta = DEFAULT_THETA
        self._env = env.unwrapped
        obs, _ = env.reset(seed=seed, options=options)
        data: dict[str, Any] = {"obs": {k: np.array(v) for k, v in obs.items()}}
        if trigger == FAILURE:
            data["previous"] = ObservedAttempt.of(previous)
        self.perceive(Percept(trigger, data))
        theta = self.act(self.think())
        result = pick.run_attempt(
            env, skill, theta, self._on_escalation, seed=seed, options=options
        )
        self._t = result.sim_s
        if result.safety is not None:
            self._mode = CognitiveMode.EMERGENCY
            self._log(
                Handoff(
                    CognitiveMode.EMERGENCY.value, self.attempt, self._t, detail=dict(result.safety)
                )
            )
            logger.warning("[SAFETY] EMERGENCY attempt %d: %s", self.attempt, result.safety)
        self.learn(Outcome.from_result(result))
        return result

    def _on_escalation(self, esc: Escalation) -> SkillParams | None:
        """System 2 at a System 1 escalation; the skill waits for the θ (None: keep)."""
        self._t = esc.t
        self.theta = esc.theta  # the θ in force, with any reflex raise
        trigger = PROBE if esc.kind == PROBE_DONE else ESCALATION
        recorder = getattr(self._env, "recorder", None)
        view = None if recorder is None else EpisodeView.from_recorder(recorder, esc.probe)
        data = {"escalation": esc, "probe": esc.probe, "view": view}
        self.perceive(Percept(trigger, data, timestamp=esc.t))
        theta = self.act(self.think(keep={f: getattr(esc.theta, f) for f in esc.ran}))
        return None if theta == esc.theta else theta

    def _log(self, handoff: Handoff) -> None:
        self.handoffs.append(handoff)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(_plain(asdict(handoff)), allow_nan=False)
            with self.log_path.open("a") as f:
                f.write(line + "\n")


def _plain(value: Any) -> Any:
    """JSON-ready: numpy values to Python, NaN and ±inf to None, tuples to lists."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
