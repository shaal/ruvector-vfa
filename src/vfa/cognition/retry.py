"""Retry until success, storing every attempt (T15, ARCHITECTURE §7).

``solve_situation(env, core, skill, max_attempts, key=, seed_of=, options=)``
runs attempts at one cube until one succeeds or ``max_attempts`` ran. Every
attempt resets the arm and the cube with ``options`` (the same hidden
properties) and the reset seed ``seed_of(attempt)``; after a failure the next
attempt is a retry (``CognitiveCore``'s ``FAILURE`` trigger). After each attempt
the policy (a ``RetrievalPolicy``) stores it, with the θ of its probe decision,
and reinforces the records its decisions in that attempt used (the ids in that
attempt's slice of ``core.decisions``); after the situation it counts toward the
next skill consolidation. ``on_attempt(result, attempt)`` lets a runner score the attempt
(it may read privileged data; the policy never sees what it reads).

RuVector mapping: one situation is a run of the crate's perceive → think → act
→ learn cycle per attempt (``cognitive_core::CognitiveCore``), with every
attempt stored as an ``Episode`` in ``memory_system::EpisodicMemory``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vfa.cognition.cognitive_core import PROBE, CognitiveCore, ObservedAttempt
from vfa.skills.pick import AttemptResult, GraspLiftSkill


def solve_situation(
    env: Any,
    core: CognitiveCore,
    skill: GraspLiftSkill,
    max_attempts: int,
    *,
    key: int,
    seed_of: Callable[[int], int],
    options: Mapping[str, Any] | None = None,
    on_attempt: Callable[[AttemptResult, int], None] | None = None,
) -> list[AttemptResult]:
    """The attempts at one cube, in order (refer to the module doc)."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
    policy = core.policy
    policy.begin_situation(key)
    results: list[AttemptResult] = []
    previous = None
    for attempt in range(1, max_attempts + 1):
        policy.begin_attempt(attempt)
        first = len(core.decisions)
        result = core.run_attempt(
            env, skill, seed=seed_of(attempt), options=options, previous=previous
        )
        mine = core.decisions[first:]
        used = [rid for d in mine for rid in d.retrieved]
        chosen = next((d.theta for d in mine if d.trigger == PROBE), None)
        policy.learn_attempt(ObservedAttempt.of(result), used, chosen)
        if on_attempt is not None:
            on_attempt(result, attempt)
        results.append(result)
        if result.success:
            break
        previous = result
    policy.end_situation()
    return results
