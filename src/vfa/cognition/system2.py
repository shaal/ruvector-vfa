"""System 2: retrieval-augmented choice of θ (T15, ARCHITECTURE §7).

``RetrievalPolicy`` is the ``System2Policy`` of the memory conditions. It reads
only percepts (observable data, T14) and its own memory.

**Triggers.** At ``START`` it queries ``key_pre`` and at ``FAILURE`` (a retry)
``key_retry``. Neither key separates the classes (T11), and the fields that run
before the probe (``grasp_depth``, ``approach_speed``) never vary in memory, so
both return the default θ; their hits are logged. The decision that matters is
at ``PROBE``: it queries ``key_probe`` (built from the percept's ``EpisodeView``
of the attempt so far) for the ``K`` best-ranked success and failure neighbours
from other situations among those at similarity ≥ ``SIM_STRONG`` (filtered
before the cut, so a weak neighbour with a high utility takes no place), and
reads every earlier attempt at this cube.

**Grip window** (``grip_window``). It starts at the ``SkillParams`` bounds. An
under-grip (SLIP or DROP, merged by T14) at grip g raises the lower bound to
g·(1 + ``MARGIN``); a ``BREAK`` at g lowers the upper bound to g·(1 − ``MARGIN``).
g is the grip that ran at the end (``theta_final``, after any reflex raise),
never a ground-truth force. The cube's own failures come first, as facts: a
grip that slipped (an under-grip's final grip, and in an attempt that slipped or
broke, a chosen grip that a reflex or an escalation had to raise) or broke (a
``BREAK``'s final grip), kept latest
first unless they contradict the ones kept (noise). The highest slip and the
lowest break bound the window, clamped to the range (a slip at the 25 N reflex
cap closes it on 25 N); where the two margins would overlap, the failed grips
themselves bound it and the grip is their midpoint. Failure neighbours with
similarity ≥ ``SIM_STRONG`` follow in ranking order (similarity × (0.5 +
utility)); a neighbour's bound that would empty the window is skipped.

**Center.** The success neighbours with similarity ≥ ``SIM_STRONG`` give a
similarity-weighted geometric mean of their final grip, and a weighted mean of
the other post-probe fields (``close_speed``, ``lift_accel``). With none, the
nearest ``SemanticMemory`` skill at ``SIM_STRONG`` or more gives its mean θ
(T13). The grip is the center if it lies in the window, else the window's
geometric midpoint (a bisection on the failures); with no center it is the
midpoint once a failure has narrowed the window, and the default θ's grip
otherwise: with an empty memory the probe decision keeps the default θ, like
``naive`` (only a slip escalation, below, differs).

**Noise.** Only a grip taken from a center gets noise: log-normal with
σ = ``SIGMA0`` · (1 − best similarity · confidence), where the confidence is
1 − 1/(n + 1) for n strong success neighbours (the crate's skill formula) or the
skill's own; the draw is clipped to the window and seeded by (situation key,
attempt).

**Slip escalation.** At ``SLIP_UNRECOVERED`` the grip moves to the geometric
midpoint between the grip in force and the window's upper bound, if that is at
least ``MARGIN`` higher; otherwise θ stays.

**Learning** (``learn_attempt``, called by ``retry.solve_situation``). Every
attempt is stored as an ``EpisodeRecord``: the merged result label, θ as System
2 chose it at the probe (``chosen``; the attempt's start θ if no probe decision
ran) and as it ended, the opaque situation key, and the ee trajectory (for T13's
skills, which average the chosen θ), under ``key_pre``, ``full_embedding``,
``key_probe`` (once the probe finished), and ``key_retry`` (failures). The
records a decision used (its ``Decision.retrieved``: the success neighbours
whose center set the grip, and the failures that set a bound) are reinforced
with the outcome. ``Decision.evidence`` lists what the probe decision weighed
(this cube's own failures, the strong success and failure neighbours, with
their similarity, result, and grip); the handoff log keeps it (T17's plots).
``end_situation`` consolidates the skills every ``CONSOLIDATE_EVERY`` situations.
With ``keys="context"`` (``memory_ctx_only``, the force ablation) every key is
the context key ``key_pre``.

**Pick-and-place** (T19). A ``PLACE_BREAK`` counts as a ``BREAK`` everywhere
above: the cube broke at the set-down because its grip was close to its break
force (the touchdown raised the finger force past it). The place fields keep
the policy's ``theta`` values (the defaults); the pick-and-place keys add the
place-phase forces (``PlaceEmbedder``, passed as ``embedder``).

**Component ablations** (T18; constructor flags, each on by default):
``reinforce=False`` never reinforces, so every utility stays 0.5 and ranking is
by similarity alone; ``semantic=False`` never consolidates and never asks the
skill library, so the center comes from episodic hits only; ``failure_bounds=
False`` sets no bound from any failure (this cube's own or a neighbour's), so
the window stays at the ``SkillParams`` bounds and a retry without a success
neighbour repeats the default grip; ``probe_trigger=False`` keeps θ at
``PROBE`` and makes the whole proposal at ``FAILURE`` instead, from the failed
attempt's ``key_probe`` (its own failures only, after a probe-less ``MISS``);
``START`` keeps the default θ as always, so a first attempt runs ``naive``'s θ
until a slip escalation, which raises the grip toward 25 N (no probe window
caps it, unlike in ``memory_full``).

**Frozen memory** (``freeze()``, T18's held-out run): nothing is stored,
reinforced, or consolidated any more. The attempts at the current cube go to
a per-situation scratch instead (the role of ``WorkingMemory``), cleared at the
next ``begin_situation``, so a retry is still bounded by this cube's failures.

The thresholds were set before the first benchmark run on the ``quick``
profile, which T10 keeps for tuning. On the calibration fixture's 243 probed
attempts, the best same-class ``key_probe`` neighbour from another cube is at
≥ 0.8 in 99 % (median 0.920), and the best other-class one in 53 % (median
0.806): 0.8 keeps nearly every same-class hit, and the ranking puts those first.
That fixture holds about 20 cubes per class; an online memory holds only the
earlier cubes, so an early cube often finds no neighbour at 0.8 and keeps the
default grip (docs/RESULTS.md §6).

RuVector mapping: ``RetrievalPolicy`` replaces the crate's
``CognitiveCore::think`` (which picks the most confident percept) with
retrieval over ``memory_system::EpisodicMemory`` (``recall_similar``) and
``memory_system::SemanticMemory`` (``find_similar``), and learns with
``skill_learning::SkillLibrary``. The window, the bounds, and the utility
reinforcement are ours.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    ESCALATION,
    FAILURE,
    PROBE,
    START,
    UNDERGRIP_RESULT,
    Decision,
    ObservedAttempt,
    Percept,
)
from vfa.memory.embeddings import Embedder, EpisodeView
from vfa.memory.memory_system import EpisodeRecord, MemorySystem
from vfa.memory.skill_learning import TRAJECTORY_KEY, SkillLibrary
from vfa.skills.params import BOUNDS, SkillParams

logger = logging.getLogger(__name__)

K = 5  # success and failure neighbours per query
SIM_STRONG = 0.8  # a neighbour at least this similar is evidence about this cube
MARGIN = 0.1  # a failure at grip g bounds the window at g·(1 ± MARGIN)
SIGMA0 = 0.15  # log-grip noise at zero similarity × confidence
CONSOLIDATE_EVERY = 10  # situations between skill consolidations
POST_PROBE = ("grip_force", "close_speed", "lift_accel")
GRIP_LOW, GRIP_HIGH = (float(v) for v in BOUNDS["grip_force"])
KEY_MODES = ("full", "context")
# A pick-and-place PLACE_BREAK bounds the grip like a BREAK (T19: the cubes that broke
# at the set-down were squeezed 0.03-0.22 N below their break force; the touchdown raised
# the finger force past it).
BREAKS = ("BREAK", "PLACE_BREAK")
NO_RECORDER = "System 2 builds its keys from the recording: make the env with record=True"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One remembered attempt as the window sees it."""

    id: str
    similarity: float  # 1.0 for this cube's own attempts
    result: str
    grip: float  # the grip that ran at the end
    theta: Mapping[str, float]  # θ that ran at the end
    chosen: float | None = None  # the grip System 2 chose (below ``grip``: it slipped)


@dataclass(frozen=True, slots=True)
class Window:
    low: float
    high: float
    used: tuple[str, ...]  # ids of the failures that set a bound
    tight: bool = False  # the edges are grips this cube failed at: only the midpoint

    @property
    def midpoint(self) -> float:
        return math.sqrt(self.low * self.high)

    @property
    def narrowed(self) -> bool:
        return bool(self.used)


def _under(f: Evidence) -> bool:
    return f.result == UNDERGRIP_RESULT or f.result in ("SLIP", "DROP")


def _broke(f: Evidence) -> bool:
    return f.result in BREAKS


def grip_window(
    failures: Sequence[Evidence],
    low: float = GRIP_LOW,
    high: float = GRIP_HIGH,
    own: int = 0,
) -> Window:
    """The grip window (refer to the module doc). The first ``own`` failures are this
    cube's own, latest first; the rest are neighbours, strongest first."""
    slip, broke = _own_facts(failures[:own])
    used = [f for _, f in (slip, broke) if f is not None]
    lo, hi, tight = low, high, False
    if slip[1] is not None:
        lo = min(slip[0] * (1.0 + MARGIN), high)  # clamped to the range
    if broke[1] is not None:
        hi = max(broke[0] * (1.0 - MARGIN), low)
    if slip[1] is not None and broke[1] is not None and lo >= hi:
        lo, hi, tight = slip[0], broke[0], True  # the margins overlap: the failed grips
    low, high = lo, hi
    for f in failures[own:]:
        if not _under(f) and not _broke(f):
            continue
        bound = f.grip * (1.0 + MARGIN if _under(f) else 1.0 - MARGIN)
        if low < bound < high:
            low, high = (bound, high) if _under(f) else (low, bound)
            used.append(f.id)
    return Window(low, high, tuple(dict.fromkeys(used)), tight)


def _own_facts(
    mine: Sequence[Evidence],
) -> tuple[tuple[float, str | None], tuple[float, str | None]]:
    """The highest grip this cube slipped at and the lowest it broke at, as (grip, id).

    Each failure is a fact: an under-grip slipped at its final grip, a BREAK broke at
    it, and in either, a final grip above the chosen one means the chosen grip slipped
    first (a reflex or an escalation raised it). Facts are kept latest first, each unless it
    contradicts the ones kept (noise)."""
    facts = []
    for f in mine:
        if _under(f):
            facts.append(("slip", f.grip, f.id))
        elif _broke(f):
            facts.append(("break", f.grip, f.id))
        if f.chosen is not None and f.chosen < f.grip and (_under(f) or _broke(f)):
            facts.append(("slip", f.chosen, f.id))
    slip: tuple[float, str | None] = (GRIP_LOW, None)
    broke: tuple[float, str | None] = (GRIP_HIGH, None)
    for kind, grip, rid in facts:
        if kind == "slip" and (broke[1] is None or grip < broke[0]) and grip >= slip[0]:
            slip = (grip, rid) if slip[1] is None or grip > slip[0] else slip
        elif kind == "break" and (slip[1] is None or grip > slip[0]) and grip <= broke[0]:
            broke = (grip, rid) if broke[1] is None or grip < broke[0] else broke
    return slip, broke


def center(successes: Sequence[Evidence]) -> dict[str, float] | None:
    """Similarity-weighted center of the post-probe fields (geometric for the grip)."""
    weights = np.array([max(s.similarity, 0.0) for s in successes])
    if not len(successes) or weights.sum() <= 0.0:
        return None
    w = weights / weights.sum()
    out = {"grip_force": float(np.exp(np.sum(w * np.log([s.grip for s in successes]))))}
    for name in POST_PROBE[1:]:
        out[name] = float(np.sum(w * np.array([s.theta[name] for s in successes])))
    return out


def choose_grip(
    window: Window, mid: float | None, default: float, noise: float, rng: np.random.Generator
) -> tuple[float, str]:
    """The grip for ``window`` and a center grip ``mid`` (refer to the module doc)."""
    if mid is not None and window.low <= mid <= window.high and not window.tight:
        grip = mid * math.exp(noise * rng.standard_normal()) if noise > 0.0 else mid
        return float(np.clip(grip, window.low, window.high)), "center"
    if mid is not None or window.narrowed:
        return window.midpoint, "bisect"
    return float(np.clip(default, window.low, window.high)), "default"


class RetrievalPolicy:
    """System 2 over a ``MemorySystem`` (refer to the module doc)."""

    def __init__(
        self,
        memory: MemorySystem | None = None,
        library: SkillLibrary | None = None,
        *,
        keys: str = "full",
        embedder: Embedder | None = None,
        theta: SkillParams = DEFAULT_THETA,
        reinforce: bool = True,
        semantic: bool = True,
        probe_trigger: bool = True,
        failure_bounds: bool = True,
    ) -> None:
        if keys not in KEY_MODES:
            raise ValueError(f"keys must be one of {KEY_MODES}, got {keys!r}")
        self.memory = MemorySystem() if memory is None else memory
        for name, index in self.memory.episodic.indexes.items():
            if getattr(index, "metric", "cosine") != "cosine":
                raise ValueError(f"index {name} must be cosine (unit keys, ARCHITECTURE §10)")
        self.library = SkillLibrary() if library is None else library
        self.keys = keys
        self.embedder = Embedder() if embedder is None else embedder
        self.theta = theta
        self.reinforce = reinforce
        self.semantic = semantic
        self.probe_trigger = probe_trigger
        self.failure_bounds = failure_bounds
        self.frozen = False
        self.scratch: list[EpisodeRecord] = []  # this cube's attempts while frozen
        self.situation: int | None = None
        self.attempt = 0
        self.situations = 0
        self._window: Window | None = None
        self._stored = 0

    # ---------------------------------------------------------- the keys

    def key(self, name: str, view: EpisodeView) -> np.ndarray:
        if self.keys == "context" or name == "key_pre":
            return self.embedder.key_pre(view.context)
        if name == "key_probe":
            return self.embedder.key_probe(view)
        if name == "key_retry":
            return self.embedder.key_retry(view)
        return self.embedder.full_embedding(view)

    # -------------------------------------------------------- System 2

    def decide(self, percept: Percept, theta: SkillParams) -> Decision:
        if percept.source == START:
            obs = percept.data["obs"]
            query = self.embedder.key_pre(obs)
            hits = self._recall("key_pre", query, None)
            self._window = None
            return Decision(self.theta, START, f"default θ; key_pre {_hits(hits)}")
        if percept.source == FAILURE:
            previous: ObservedAttempt = percept.data["previous"]
            if not self.probe_trigger:  # the whole proposal, one attempt late
                view = previous.view
                query = None
                if view is not None and view.probe is not None:
                    query = self.key("key_probe", view)
                return self.propose(query, theta, FAILURE)
            hits = []
            if previous.view is not None:
                hits = self._recall("key_retry", self.key("key_retry", previous.view), False)
            self._window = None
            return Decision(self.theta, FAILURE, f"default θ; key_retry {_hits(hits)}")
        if percept.source == PROBE:
            if not self.probe_trigger:
                return Decision(theta, PROBE, "keep θ: no decision at the probe")
            return self._probe(percept, theta)
        if percept.source == ESCALATION:
            return self._escalation(theta)
        raise ValueError(f"unknown trigger {percept.source!r}")

    def _probe(self, percept: Percept, theta: SkillParams) -> Decision:
        view: EpisodeView | None = percept.data.get("view")
        if view is None:
            raise ValueError(NO_RECORDER)
        return self.propose(self.key("key_probe", view), theta)

    def propose(
        self, query: np.ndarray | None, theta: SkillParams, trigger: str = PROBE
    ) -> Decision:
        """The probe decision for the ``key_probe`` vector ``query`` (None: no view);
        ``trigger`` names the decision point (``FAILURE`` without the probe trigger)."""
        own = self._own_failures()
        successes, failures, semantic = [], [], None
        if query is not None and len(self.memory.episodic):
            successes = self._recall("key_probe", query, True, strong=True)
            failures = self._recall("key_probe", query, False, strong=True)
            if not successes:
                semantic = self._skill(query)
        if self.failure_bounds:
            window = grip_window(own + failures, own=len(own))
        else:
            window = grip_window([])
        self._window = window
        mid = center(successes)
        confidence, best = 0.0, 0.0
        if mid is not None:
            best = max(e.similarity for e in successes)
            confidence = 1.0 - 1.0 / (len(successes) + 1.0)
        elif semantic is not None:
            mid, best, confidence = semantic
        noise = SIGMA0 * (1.0 - best * confidence) if mid is not None else 0.0
        rng = np.random.default_rng((self.situation or 0, self.attempt, 0x5332))
        grip, how = choose_grip(
            window, None if mid is None else mid["grip_force"], self.theta.grip_force, noise, rng
        )
        fields = {"grip_force": grip}
        if mid is not None and how == "center":
            fields.update({k: mid[k] for k in POST_PROBE[1:]})
        chosen = replace(theta, **fields)
        centered = [e.id for e in successes] if how == "center" and successes else []
        used = tuple(dict.fromkeys(centered + list(window.used)))
        for e in successes[:3]:
            logger.info(
                "[MEM] hit %s sim=%.2f outcome=success θ.grip=%.1fN", e.id, e.similarity, e.grip
            )
        reason = (
            f"{how}: grip {grip:.2f} N in [{window.low:.2f}, {window.high:.2f}] "
            f"({len(successes)} successes, {len(window.used)} bounds"
            f"{', skill' if semantic is not None and how == 'center' else ''})"
        )
        evidence = tuple(
            {
                "id": e.id,
                "role": role,
                "similarity": round(e.similarity, 4),
                "result": e.result,
                "grip": round(e.grip, 3),
            }
            for role, group in (("own", own), ("success", successes), ("failure", failures))
            for e in group
        )
        return Decision(
            chosen, trigger, reason, utility=best * confidence, retrieved=used, evidence=evidence
        )

    def _escalation(self, theta: SkillParams) -> Decision:
        high = GRIP_HIGH if self._window is None else self._window.high
        raised = math.sqrt(theta.grip_force * high)
        if raised >= theta.grip_force * (1.0 + MARGIN):
            return Decision(
                replace(theta, grip_force=raised), ESCALATION, "raise toward the upper bound"
            )
        return Decision(theta, ESCALATION, "keep θ: no room above the grip")

    # ------------------------------------------------------- the memory

    def _recall(
        self, key_type: str, query: np.ndarray, success: bool | None, strong: bool = False
    ) -> list[Evidence]:
        """The ``K`` best-ranked neighbours from other situations; with ``strong``, among
        those at similarity ≥ ``SIM_STRONG`` only (filtered before the cut)."""
        index = self.memory.episodic.indexes[key_type]
        if not len(index):
            return []
        here = self.situation
        if strong:  # the strong candidates by similarity alone, then the K best ranked
            near = index.search(query, len(index), filter=lambda p: p["situation"] != here)
            ids = {h.id for h in near if h.score >= SIM_STRONG}
            keep = lambda p: p["id"] in ids  # noqa: E731
        else:
            keep = lambda p: p["situation"] != here  # noqa: E731
        if strong and not ids:
            return []
        hits = self.memory.episodic.recall_similar(key_type, query, K, success=success, filter=keep)
        return [_evidence(h.record, h.similarity) for h in hits]

    def _own_failures(self) -> list[Evidence]:
        records = list(self.memory.episodic.records.values()) + self.scratch
        own = [r for r in records if r.situation == self.situation and not r.success]
        return [_evidence(r, 1.0) for r in sorted(own, key=lambda r: -r.attempt)]

    def _skill(self, query: np.ndarray) -> tuple[dict[str, float], float, float] | None:
        if not self.semantic or not len(self.memory.semantic):
            return None
        name, sim = self.memory.semantic.find_similar(query, 1)[0]
        skill = self.library.get(name)
        if sim < SIM_STRONG or skill is None or not skill.theta:
            return None
        return {k: float(skill.theta[k]) for k in POST_PROBE}, float(sim), skill.confidence

    # --------------------------------------------------------- learning

    def freeze(self) -> None:
        """No more writes to memory (refer to the module doc)."""
        self.frozen = True
        self.scratch = []

    def begin_situation(self, key: int) -> None:
        self.situation = int(key)
        self.attempt = 0
        self.scratch = []

    def begin_attempt(self, attempt: int) -> None:
        self.attempt = int(attempt)

    def learn_attempt(
        self, observed: ObservedAttempt, used: Sequence[str], chosen: SkillParams | None = None
    ) -> str:
        """Store the attempt (``chosen``: θ of its probe decision), reinforce the records
        its decisions used; returns its id. Frozen: only the scratch keeps it."""
        if self.frozen:
            rid = f"scratch_{len(self.scratch) + 1:02d}"
            theta = observed.theta if chosen is None else chosen
            self.scratch.append(
                EpisodeRecord(
                    rid,
                    int(self.situation or 0),
                    self.attempt,
                    theta.to_dict(),
                    observed.result,
                    theta_final=observed.theta_final.to_dict(),
                )
            )
            return rid
        self._stored += 1
        while f"ep_{self._stored:05d}" in self.memory.episodic.records:  # a memory given
            self._stored += 1
        rid = f"ep_{self._stored:05d}"
        view = observed.view
        if view is None:
            raise ValueError(NO_RECORDER)
        info: dict[str, Any] = {}
        keys: dict[str, np.ndarray] = {}
        info[TRAJECTORY_KEY] = view.ee.tolist()
        keys["key_pre"] = self.key("key_pre", view)
        keys["full_embedding"] = self.key("full_embedding", view)
        if view.probe is not None:
            keys["key_probe"] = self.key("key_probe", view)
        if not observed.success:
            keys["key_retry"] = self.key("key_retry", view)
        record = EpisodeRecord(
            rid,
            int(self.situation or 0),
            self.attempt,
            (observed.theta if chosen is None else chosen).to_dict(),
            observed.result,
            timestamp=self._stored,
            info=info,
            theta_final=observed.theta_final.to_dict(),
        )
        self.memory.episodic.store(record, keys)
        for used_id in dict.fromkeys(used) if self.reinforce else ():
            if used_id in self.memory.episodic.records:
                self.memory.episodic.reinforce(used_id, observed.success)
        logger.info(
            "[MEM] store %s %s grip=%.1fN; reinforce %d %s",
            rid,
            observed.result,
            observed.theta_final.grip_force,
            len(set(used)) if self.reinforce else 0,
            "+" if observed.success else "−",
        )
        return rid

    def end_situation(self) -> None:
        self.situations += 1
        if self.frozen or not self.semantic:
            return
        if self.situations % CONSOLIDATE_EVERY == 0:
            self.library.consolidate(self.memory)


def _evidence(record: EpisodeRecord, similarity: float) -> Evidence:
    final = record.theta_final if record.theta_final is not None else record.theta
    grip, chosen = float(final["grip_force"]), float(record.theta["grip_force"])
    return Evidence(record.id, float(similarity), record.result, grip, final, chosen)


def _hits(hits: Sequence[Evidence]) -> str:
    if not hits:
        return "hits=0"
    return f"hits={len(hits)} (best sim {max(h.similarity for h in hits):.2f})"
