"""T15: System 2's retrieval policy (``vfa.cognition.system2``), no physics.

ROADMAP acceptance: on a 1-D toy problem with a hidden window, ``RetrievalPolicy``
reaches success in fewer mean attempts than random search over 200 trials.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    ESCALATION,
    PROBE,
    START,
    UNDERGRIP_RESULT,
    Percept,
)
from vfa.cognition.system2 import (
    GRIP_HIGH,
    GRIP_LOW,
    MARGIN,
    Evidence,
    K,
    RetrievalPolicy,
    Window,
    center,
    choose_grip,
    grip_window,
)
from vfa.memory.backend import NumpyBackend
from vfa.memory.memory_system import EpisodeRecord, MemorySystem

# ---------------------------------------------------------------- the toy

# Hidden grip windows [N per finger]: the toy's "classes". A probe key points at an
# angle set by the class (plus noise), so cubes of one class feel alike.
WINDOWS = ((1.0, 2.5), (3.0, 5.0), (6.0, 9.0), (10.0, 14.0), (16.0, 22.0))
CAP = 10  # attempts per trial; an unsolved trial counts CAP + 1


def toy_key(cls: int, rng: np.random.Generator) -> np.ndarray:
    angle = cls * 0.9 + rng.normal(scale=0.05)
    v = np.array([math.cos(angle), math.sin(angle), 0.1 * rng.normal(), 0.1 * rng.normal()])
    return v / np.linalg.norm(v)


def toy_outcome(grip: float, window: tuple[float, float]) -> str:
    if grip < window[0]:
        return UNDERGRIP_RESULT
    return "BREAK" if grip > window[1] else "SUCCESS"


def retrieval_trials(trials: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    policy = RetrievalPolicy()
    attempts = []
    for trial in range(trials):
        cls = int(rng.integers(len(WINDOWS)))
        key = toy_key(cls, rng)
        policy.begin_situation(trial)
        for attempt in range(1, CAP + 1):
            policy.begin_attempt(attempt)
            decision = policy.propose(key, DEFAULT_THETA)
            grip = decision.theta.grip_force
            result = toy_outcome(grip, WINDOWS[cls])
            theta = decision.theta.to_dict()
            record = EpisodeRecord(f"t{trial}_{attempt}", trial, attempt, theta, result)
            policy.memory.episodic.store(record, {"key_probe": key})
            for used in decision.retrieved:
                policy.memory.episodic.reinforce(used, result == "SUCCESS")
            if result == "SUCCESS":
                attempts.append(attempt)
                break
        else:
            attempts.append(CAP + 1)
    return attempts


def random_trials(trials: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    grips = np.random.default_rng((seed, 1))  # its own draws: the class sequence stays equal
    attempts = []
    for _ in range(trials):
        cls = int(rng.integers(len(WINDOWS)))
        toy_key(cls, rng)  # the same class sequence as the retrieval trials
        for attempt in range(1, CAP + 1):
            if toy_outcome(float(grips.uniform(GRIP_LOW, GRIP_HIGH)), WINDOWS[cls]) == "SUCCESS":
                attempts.append(attempt)
                break
        else:
            attempts.append(CAP + 1)
    return attempts


def test_retrieval_beats_random_search_on_the_toy():
    retrieval = retrieval_trials(200, seed=7)
    random = random_trials(200, seed=7)
    late = np.mean(retrieval[100:])
    print(
        f"\nmean attempts over 200 trials: retrieval {np.mean(retrieval):.2f} "
        f"(last 100: {late:.2f}), random search {np.mean(random):.2f}"
    )
    assert np.mean(retrieval) < np.mean(random)
    assert late < np.mean(retrieval[:100])  # it learns across trials


def test_bounds_alone_beat_random_search():
    """With a fresh memory every trial (no transfer), the failure bounds alone bisect
    the window: still fewer attempts than random search."""
    rng = np.random.default_rng(8)
    bisect, random = [], random_trials(200, seed=8)
    for trial in range(200):
        cls = int(rng.integers(len(WINDOWS)))
        key = toy_key(cls, rng)
        policy = RetrievalPolicy()
        policy.begin_situation(trial)
        for attempt in range(1, CAP + 1):
            policy.begin_attempt(attempt)
            grip = policy.propose(key, DEFAULT_THETA).theta.grip_force
            result = toy_outcome(grip, WINDOWS[cls])
            record = EpisodeRecord(f"t{attempt}", trial, attempt, {"grip_force": grip}, result)
            policy.memory.episodic.store(record, {"key_probe": key})
            if result == "SUCCESS":
                bisect.append(attempt)
                break
        else:
            bisect.append(CAP + 1)
    print(f"\nfresh memory: bisection {np.mean(bisect):.2f}, random {np.mean(random):.2f}")
    assert np.mean(bisect) < np.mean(random)


# ------------------------------------------------------------ the window


def ev(result: str, grip: float, sim: float = 1.0, id: str = "e") -> Evidence:
    return Evidence(
        id, sim, result, grip, {"grip_force": grip, "close_speed": 0.1, "lift_accel": 0.6}
    )


def test_failures_bound_the_grip_window():
    w = grip_window([ev("BREAK", 8.0, id="b"), ev(UNDERGRIP_RESULT, 2.0, id="u")])
    assert w.high == pytest.approx(8.0 * (1 - MARGIN)) and w.low == pytest.approx(
        2.0 * (1 + MARGIN)
    )
    assert w.used == ("b", "u") and w.midpoint == pytest.approx(math.sqrt(w.low * w.high))
    # SLIP and DROP count as under-grip too; MISS, TIMEOUT, SAFETY_STOP give no bound.
    assert grip_window([ev("SLIP", 5.0)]).low == pytest.approx(5.5)
    assert grip_window([ev("DROP", 5.0)]).low == pytest.approx(5.5)
    for other in ("MISS", "TIMEOUT", "SAFETY_STOP"):
        assert not grip_window([ev(other, 5.0)]).narrowed


def test_a_bound_that_would_empty_the_window_is_skipped():
    """The strongest evidence comes first; a later contradiction is ignored."""
    w = grip_window([ev("BREAK", 5.0, id="first"), ev(UNDERGRIP_RESULT, 6.0, id="second")])
    assert w.used == ("first",) and w.low == GRIP_LOW and w.high == pytest.approx(4.5)


def test_center_is_a_similarity_weighted_geometric_mean():
    c = center([ev("SUCCESS", 2.0, sim=0.9), ev("SUCCESS", 8.0, sim=0.9)])
    assert c["grip_force"] == pytest.approx(4.0)
    c = center([ev("SUCCESS", 2.0, sim=1.0), ev("SUCCESS", 8.0, sim=0.0)])
    assert c["grip_force"] == pytest.approx(2.0)
    assert center([]) is None


def test_choose_grip():
    rng = np.random.default_rng(0)
    full = Window(GRIP_LOW, GRIP_HIGH, ())
    assert choose_grip(full, None, 8.0, 0.0, rng) == (8.0, "default")
    narrowed = Window(GRIP_LOW, 7.2, ("b",))
    assert choose_grip(narrowed, None, 8.0, 0.0, rng) == (
        pytest.approx(math.sqrt(GRIP_LOW * 7.2)),
        "bisect",
    )
    assert choose_grip(narrowed, 9.0, 8.0, 0.0, rng)[1] == "bisect"  # center outside
    grip, how = choose_grip(narrowed, 3.0, 8.0, 0.5, rng)
    assert how == "center" and GRIP_LOW <= grip <= 7.2


# ------------------------------------------------------------ the policy


def store(policy: RetrievalPolicy, rid: str, situation: int, key, result: str, grip: float) -> None:
    theta = {**DEFAULT_THETA.to_dict(), "grip_force": grip}
    record = EpisodeRecord(rid, situation, 1, theta, result, theta_final=theta)
    policy.memory.episodic.store(record, {"key_probe": key})


def test_empty_memory_decides_like_naive():
    """No memory, no noise: the probe decision keeps the default θ, so the first cube
    of a memory run is paired exactly with the naive baseline."""
    policy = RetrievalPolicy()
    policy.begin_situation(1)
    policy.begin_attempt(1)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert decision.theta == DEFAULT_THETA and decision.retrieved == ()


def test_similar_successes_set_the_grip_and_are_used():
    policy = RetrievalPolicy()
    store(policy, "near", 10, np.array([1.0, 0.0]), "SUCCESS", 3.0)
    store(policy, "far", 11, np.array([0.0, 1.0]), "SUCCESS", 20.0)
    store(policy, "same", 12, np.array([1.0, 0.0]), "SUCCESS", 25.0)  # this cube: ignored
    policy.begin_situation(12)
    policy.begin_attempt(2)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert decision.retrieved == ("near",)
    assert decision.theta.grip_force == pytest.approx(3.0, rel=0.5)  # noise is bounded


def test_own_failures_bound_the_retry():
    policy = RetrievalPolicy()
    store(policy, "s1", 7, np.array([1.0, 0.0]), "BREAK", 8.0)
    store(policy, "nb", 8, np.array([0.95, math.sqrt(1 - 0.95**2)]), "SUCCESS", 3.0)
    policy.begin_situation(7)
    policy.begin_attempt(2)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert decision.theta.grip_force == pytest.approx(3.0, rel=0.3)
    assert decision.retrieved == ("nb", "s1")
    # The decision's evidence: what it weighed, with roles (logged for the plots).
    assert [(e["id"], e["role"], e["result"]) for e in decision.evidence] == [
        ("s1", "own", "BREAK"),
        ("nb", "success", "SUCCESS"),
    ]
    assert decision.evidence[1]["similarity"] == pytest.approx(0.95)


def test_dissimilar_failures_are_not_evidence():
    policy = RetrievalPolicy()
    store(policy, "other", 3, np.array([0.0, 1.0]), "BREAK", 8.0)
    policy.begin_situation(4)
    policy.begin_attempt(1)
    assert policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).theta == DEFAULT_THETA


def test_start_and_escalation():
    policy = RetrievalPolicy()
    policy.begin_situation(1)
    obs = {
        "cube_pos": np.array([0.5, 0.0, 0.02]),
        "cube_yaw": np.array([0.0]),
        "cube_size": np.array([0.05]),
    }
    start = policy.decide(Percept(START, {"obs": obs}), DEFAULT_THETA)
    assert start.theta == DEFAULT_THETA and start.retrieved == ()
    up = policy.decide(Percept(ESCALATION, {}), DEFAULT_THETA)
    assert up.theta.grip_force == pytest.approx(math.sqrt(8.0 * GRIP_HIGH))
    top = DEFAULT_THETA.__class__(**{**DEFAULT_THETA.to_dict(), "grip_force": GRIP_HIGH})
    assert policy.decide(Percept(ESCALATION, {}), top).theta == top
    with pytest.raises(ValueError, match="record=True"):
        policy.decide(Percept(PROBE, {"view": None}), DEFAULT_THETA)


def test_settings_are_checked():
    with pytest.raises(ValueError, match="keys"):
        RetrievalPolicy(keys="force")
    euclid = MemorySystem(backend_factory=lambda: NumpyBackend("euclidean"))
    with pytest.raises(ValueError, match="cosine"):
        RetrievalPolicy(euclid)


def test_weak_neighbours_take_no_place_among_the_strong():
    """Five weak successes with a high utility outrank by score a strong success with
    a low one; the strong one is still used."""
    policy = RetrievalPolicy()
    for i in range(5):
        store(policy, f"weak{i}", 20 + i, np.array([0.6, 0.8]), "SUCCESS", 20.0)
        policy.memory.episodic.records[f"weak{i}"].utility = 1.0
    store(policy, "strong", 30, np.array([1.0, 0.0]), "SUCCESS", 3.0)
    policy.memory.episodic.records["strong"].utility = 0.0
    policy.begin_situation(40)
    policy.begin_attempt(1)
    assert policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).retrieved == ("strong",)


def test_own_failures_latest_first_and_never_a_repeated_grip():
    policy = RetrievalPolicy()
    key = np.array([1.0, 0.0])
    policy.begin_situation(5)
    theta = {**DEFAULT_THETA.to_dict()}
    # An older BREAK at 5 N contradicts the latest under-grip at 6 N: the latest wins.
    for attempt, (result, grip) in enumerate([("BREAK", 5.0), (UNDERGRIP_RESULT, 6.0)], 1):
        t = {**theta, "grip_force": grip}
        policy.memory.episodic.store(
            EpisodeRecord(f"a{attempt}", 5, attempt, t, result, theta_final=t), {"key_probe": key}
        )
    policy.begin_attempt(3)
    assert policy.propose(key, DEFAULT_THETA).theta.grip_force == pytest.approx(
        math.sqrt(6.6 * GRIP_HIGH)
    )
    # Under-grips near the top: the bound clamps at the grip itself, so the grip moves.
    near = RetrievalPolicy()
    near.begin_situation(6)
    grips = []
    for attempt in range(1, 4):
        near.begin_attempt(attempt)
        grip = near.propose(key, DEFAULT_THETA).theta.grip_force if attempt > 1 else 23.63
        grips.append(grip)
        t = {**theta, "grip_force": grip}
        record = EpisodeRecord(f"n{attempt}", 6, attempt, t, UNDERGRIP_RESULT, theta_final=t)
        near.memory.episodic.store(record, {"key_probe": key})
    assert grips[1] > grips[0] and grips[2] == GRIP_HIGH  # the window closed on the top


def test_noise_is_seeded_by_situation_and_attempt():
    def grip(situation: int, attempt: int) -> float:
        policy = RetrievalPolicy()
        store(policy, "s", 99, np.array([1.0, 0.0]), "SUCCESS", 5.0)
        policy.begin_situation(situation)
        policy.begin_attempt(attempt)
        return policy.propose(np.array([0.9, math.sqrt(1 - 0.81)]), DEFAULT_THETA).theta.grip_force

    assert grip(1, 1) == grip(1, 1)
    assert len({grip(1, 1), grip(1, 2), grip(2, 1)}) == 3


def test_a_skill_is_the_fallback_when_episodic_hits_are_weak():
    from vfa.memory.skill_learning import Demonstration

    policy = RetrievalPolicy()
    store(policy, "far", 1, np.array([0.0, 1.0]), "SUCCESS", 20.0)  # not similar
    theta = {**DEFAULT_THETA.to_dict(), "grip_force": 3.0}
    policy.library.learn_from_demonstration("skill_00", [Demonstration([], theta=theta)] * 3)
    policy.memory.semantic.store("skill_00", np.array([1.0, 0.0]), {"theta": theta})
    policy.begin_situation(2)
    policy.begin_attempt(1)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert (
        decision.theta.grip_force == pytest.approx(3.0, rel=0.3) and "skill" in decision.reasoning
    )


def test_learning_reinforces_with_the_outcome_and_stores_the_chosen_theta():
    from vfa.cognition.cognitive_core import ObservedAttempt
    from vfa.memory import embeddings as M
    from vfa.skills.params import SkillParams

    view = M.fixture_views(M.load_fixture())[0]
    policy = RetrievalPolicy()
    store(policy, "used", 1, policy.key("key_probe", view), "SUCCESS", 5.0)
    policy.begin_situation(2)
    policy.begin_attempt(1)
    chosen = SkillParams(grip_force=4.0)
    failed = ObservedAttempt(
        "BREAK", False, DEFAULT_THETA, SkillParams(grip_force=6.0), None, (), None, view
    )
    rid = policy.learn_attempt(failed, ["used"], chosen)
    assert policy.memory.episodic.records["used"].utility == pytest.approx(0.4)
    assert policy.memory.episodic.records[rid].theta["grip_force"] == 4.0
    won = ObservedAttempt("SUCCESS", True, DEFAULT_THETA, chosen, None, (), None, view)
    policy.learn_attempt(won, ["used"], chosen)
    assert policy.memory.episodic.records["used"].utility == pytest.approx(0.5)


def own(policy: RetrievalPolicy, situation: int, rows) -> None:
    """Store this cube's attempts: (attempt, result, grip as chosen, grip at the end)."""
    for attempt, result, chosen, final in rows:
        t = {**DEFAULT_THETA.to_dict(), "grip_force": chosen}
        f = {**DEFAULT_THETA.to_dict(), "grip_force": final}
        record = EpisodeRecord(f"o{attempt}", situation, attempt, t, result, theta_final=f)
        policy.memory.episodic.store(record, {"key_probe": np.array([1.0, 0.0])})


def test_an_under_grip_at_the_top_tries_the_top():
    """The reflex caps the grip at 25 N; a slip there still moves the next grip."""
    policy = RetrievalPolicy()
    own(policy, 3, [(1, UNDERGRIP_RESULT, 8.0, 17.32), (2, UNDERGRIP_RESULT, 21.82, GRIP_HIGH)])
    policy.begin_situation(3)
    policy.begin_attempt(3)
    assert policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).theta.grip_force == GRIP_HIGH


def test_bounds_use_the_grip_that_ran_at_the_end():
    policy = RetrievalPolicy()
    own(policy, 4, [(1, UNDERGRIP_RESULT, 5.0, 12.0)])  # the reflex raised 5 N to 12 N
    policy.begin_situation(4)
    policy.begin_attempt(2)
    grip = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).theta.grip_force
    assert grip == pytest.approx(math.sqrt(12.0 * (1 + MARGIN) * GRIP_HIGH))


def test_own_failures_outrank_neighbours_and_bounds_are_strict():
    policy = RetrievalPolicy()
    own(policy, 5, [(1, UNDERGRIP_RESULT, 10.0, 10.0)])
    store(policy, "nb", 6, np.array([1.0, 0.0]), "BREAK", 9.0)  # a similar cube broke at 9 N
    policy.begin_situation(5)
    policy.begin_attempt(2)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert decision.theta.grip_force == pytest.approx(math.sqrt(11.0 * GRIP_HIGH))
    assert decision.retrieved == ("o1",)  # the contradicting neighbour set no bound
    # A margin bound exactly at the window's edge is not inside it: the grip itself bounds.
    edge = grip_window([ev("BREAK", GRIP_LOW / (1 - MARGIN))], own=1)
    assert edge.high == edge.low == GRIP_LOW  # the window closes on the bottom


@pytest.mark.parametrize(
    ("rows", "window"),
    [
        # Latest first. Consistent own failures whose margins overlap: the grips themselves.
        ([("BREAK", 11.0), (UNDERGRIP_RESULT, 10.0)], (10.0, 11.0)),
        ([(UNDERGRIP_RESULT, 9.5), ("BREAK", 10.0)], (9.5, 10.0)),
        # Margins that fit: both apply.
        ([("BREAK", 20.0), (UNDERGRIP_RESULT, 5.0)], (5.5, 18.0)),
        # Contradicting own failures (noise): the latest alone.
        ([("BREAK", 9.0), (UNDERGRIP_RESULT, 10.0)], (GRIP_LOW, 8.1)),
        ([(UNDERGRIP_RESULT, 10.0), ("BREAK", 9.0)], (11.0, GRIP_HIGH)),
        # The lowest break and the highest slip count.
        ([("BREAK", 9.0), ("BREAK", 20.0)], (GRIP_LOW, 8.1)),
        ([(UNDERGRIP_RESULT, 3.0), (UNDERGRIP_RESULT, 5.0)], (5.5, GRIP_HIGH)),
        # A slip and a break at the same grip contradict: the latest alone.
        ([("BREAK", 10.0), (UNDERGRIP_RESULT, 10.0)], (GRIP_LOW, 9.0)),
        ([(UNDERGRIP_RESULT, 10.0), ("BREAK", 10.0)], (11.0, GRIP_HIGH)),
        # Facts at the range limits count.
        ([(UNDERGRIP_RESULT, GRIP_LOW)], (GRIP_LOW * (1 + MARGIN), GRIP_HIGH)),
        ([("BREAK", GRIP_HIGH)], (GRIP_LOW, GRIP_HIGH * (1 - MARGIN))),
        # Only under-grips and breaks are facts.
        ([("MISS", 3.0), ("TIMEOUT", 9.0), ("SAFETY_STOP", 9.0)], (GRIP_LOW, GRIP_HIGH)),
        # At the range limits the window closes on the limit.
        ([(UNDERGRIP_RESULT, GRIP_HIGH), (UNDERGRIP_RESULT, GRIP_HIGH)], (GRIP_HIGH, GRIP_HIGH)),
        ([("BREAK", GRIP_LOW)], (GRIP_LOW, GRIP_LOW)),
    ],
)
def test_own_failures_bound_together(rows, window):
    failures = [ev(result, grip, id=f"a{i}") for i, (result, grip) in enumerate(rows)]
    w = grip_window(failures, own=len(failures))
    assert (w.low, w.high) == pytest.approx(window)


def test_the_k_best_ranked_strong_successes_set_the_center():
    policy = RetrievalPolicy()
    for i in range(K + 2):
        store(policy, f"s{i}", 50 + i, np.array([1.0, 0.0]), "SUCCESS", 5.0)
        policy.memory.episodic.records[f"s{i}"].utility = i / (K + 2)
    policy.begin_situation(99)
    policy.begin_attempt(1)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert decision.retrieved == tuple(f"s{i}" for i in range(K + 1, 1, -1))


def test_a_memory_given_keeps_its_records():
    from vfa.cognition.cognitive_core import ObservedAttempt
    from vfa.memory import embeddings as M

    view = M.fixture_views(M.load_fixture())[0]
    first = RetrievalPolicy()
    first.begin_situation(1)
    first.begin_attempt(1)
    observed = ObservedAttempt("SUCCESS", True, DEFAULT_THETA, DEFAULT_THETA, None, (), None, view)
    first.learn_attempt(observed, [])
    second = RetrievalPolicy(first.memory, first.library)
    second.begin_situation(2)
    second.begin_attempt(1)
    second.learn_attempt(observed, [])
    assert len(second.memory.episodic) == 2


def own_ev(result: str, final: float, chosen: float, id: str) -> Evidence:
    return Evidence(id, 1.0, result, final, {"grip_force": final}, chosen)


def test_a_raised_grip_means_the_chosen_grip_slipped():
    """A reflex or an escalation raised the grip before the cube broke: the chosen grip
    slipped, so it bounds from below too."""
    w = grip_window([own_ev("BREAK", 6.5, 1.9, "late"), own_ev("BREAK", 8.0, 8.0, "early")], own=2)
    assert (w.low, w.high) == pytest.approx((1.9 * (1 + MARGIN), 6.5 * (1 - MARGIN)))


def test_older_facts_that_agree_are_kept():
    """Latest first: a break at 12 N and a slip at 10 N agree; an older break at 9 N
    contradicts the slip and is dropped, not the consistent pair."""
    rows = [
        own_ev("BREAK", 12.0, 12.0, "a3"),
        own_ev(UNDERGRIP_RESULT, 10.0, 5.0, "a2"),
        own_ev("BREAK", 9.0, 9.0, "a1"),
    ]
    w = grip_window(rows, own=3)
    assert (w.low, w.high, w.tight) == (10.0, 12.0, True) and w.used == ("a2", "a3")


def test_a_tight_window_never_repeats_a_failed_grip():
    """When the margins overlap, the edges are grips this cube failed at: the grip is
    the midpoint, whatever a similar success suggests."""
    rng = np.random.default_rng(0)
    w = grip_window(
        [own_ev("BREAK", 11.5, 11.5, "b"), own_ev(UNDERGRIP_RESULT, 10.0, 10.0, "s")], own=2
    )
    for _ in range(50):
        grip, how = choose_grip(w, 10.2, 8.0, 0.5, rng)
        assert (grip, how) == (pytest.approx(math.sqrt(10.0 * 11.5)), "bisect")


def test_a_bisect_credits_no_success():
    policy = RetrievalPolicy()
    own(policy, 8, [(1, "BREAK", 8.0, 8.0)])
    store(policy, "far_center", 9, np.array([1.0, 0.0]), "SUCCESS", 9.0)  # outside [0.5, 7.2]
    policy.begin_situation(8)
    policy.begin_attempt(2)
    decision = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA)
    assert "bisect" in decision.reasoning and decision.retrieved == ("o1",)


def test_a_neighbour_bound_on_the_edge_is_not_applied():
    w = grip_window([ev("BREAK", GRIP_LOW / (1 - MARGIN), sim=0.9, id="nb")])
    assert (w.low, w.high, w.used) == (GRIP_LOW, GRIP_HIGH, ())


def test_the_probe_decision_sees_a_raised_grip_as_a_slip():
    """End to end through the memory: the chosen grip (record.theta) under a final grip
    the reflex raised bounds the next grip from below."""
    policy = RetrievalPolicy()
    own(policy, 11, [(1, "BREAK", 1.9, 6.5)])
    policy.begin_situation(11)
    policy.begin_attempt(2)
    grip = policy.propose(np.array([1.0, 0.0]), DEFAULT_THETA).theta.grip_force
    assert grip == pytest.approx(math.sqrt(1.9 * (1 + MARGIN) * 6.5 * (1 - MARGIN)))


def test_a_chosen_slip_survives_a_contradicted_final_slip():
    """Latest first: a break at 8 N contradicts an older slip at 9 N (its final grip),
    but that attempt's chosen 3 N, which the reflex raised, still slipped."""
    rows = [own_ev("BREAK", 8.0, 8.0, "late"), own_ev(UNDERGRIP_RESULT, 9.0, 3.0, "early")]
    w = grip_window(rows, own=2)
    assert (w.low, w.high) == pytest.approx((3.0 * (1 + MARGIN), 8.0 * (1 - MARGIN)))
