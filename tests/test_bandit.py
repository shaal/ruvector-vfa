"""The Thompson-sampling bandit baseline (T17a): posterior, sampler, buckets, policy."""

from dataclasses import replace

import numpy as np
import pytest

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    ESCALATION,
    FAILURE,
    PROBE,
    START,
    ObservedAttempt,
    Percept,
)
from vfa.experiments import bandit as B
from vfa.experiments.baselines import NAIVE
from vfa.memory.embeddings import load_fixture
from vfa.skills.params import BOUNDS, SkillParams
from vfa.skills.pick import ProbeFeatures


def _probe(heft: float, slope: float | None) -> ProbeFeatures:
    return ProbeFeatures(1.0, 1e-4, slope, heft, heft, False, None, False)


def _observed(success: bool) -> ObservedAttempt:
    result = "SUCCESS" if success else "BREAK"
    return ObservedAttempt(result, success, NAIVE, NAIVE, None, (), None, None)


def test_posterior_update():
    p = B.BetaParams.uniform()
    assert (p.alpha, p.beta, p.mean()) == (1.0, 1.0, 0.5)
    p.update(1.0)
    p.update(0.0)
    p.update(0.0)
    assert (p.alpha, p.beta) == (2.0, 3.0)
    assert p.variance() == pytest.approx(2 * 3 / (25 * 6))

    bandit = B.ThompsonGripBandit([NAIVE] * 3, heft_edges=[0.0])
    bandit.record_outcome(1, 2, 1.0)
    bandit.record_outcome(1, 2, 0.0)
    bandit.record_outcome(0, 0, 0.0)
    assert (bandit.posterior(1, 2).alpha, bandit.posterior(1, 2).beta) == (2.0, 2.0)
    assert (bandit.posterior(0, 0).alpha, bandit.posterior(0, 0).beta) == (1.0, 2.0)
    untouched = [(b, a) for b in range(2) for a in range(3) if (b, a) not in {(1, 2), (0, 0)}]
    assert all(bandit.alpha[b, a] == bandit.beta[b, a] == 1.0 for b, a in untouched)
    with pytest.raises(ValueError):
        bandit.record_outcome(0, 0, 1.5)


def test_the_same_seed_gives_the_same_choices():
    def run(seed: int) -> list[int]:
        bandit = B.ThompsonGripBandit([NAIVE] * 6)
        rng = np.random.default_rng(seed)
        rewards = np.random.default_rng(99)
        arms = []
        for _ in range(200):
            arm = bandit.select_arm(0, rng)
            bandit.record_outcome(0, arm, float(rewards.random() < 0.1 * arm))
            arms.append(arm)
        return arms

    assert run(7) == run(7)
    assert run(7) != run(8)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 5.0), (10.0, 3.0), (0.5, 0.5)])
def test_draws_match_the_beta_mean_and_variance(alpha, beta):
    p = B.BetaParams(alpha, beta)
    draws = p.sample(np.random.default_rng(0), 20_000)
    assert np.mean(draws) == pytest.approx(p.mean(), rel=0.02)
    assert np.var(draws) == pytest.approx(p.variance(), rel=0.02)


def test_the_selection_draws_match_each_arms_beta():
    bandit = B.ThompsonGripBandit([NAIVE] * 3)
    bandit.alpha[0] = [1.0, 2.0, 10.0]
    bandit.beta[0] = [1.0, 5.0, 3.0]
    rng = np.random.default_rng(0)
    draws = np.array([bandit.draws(0, rng) for _ in range(20_000)])
    for arm in range(3):
        p = bandit.posterior(0, arm)
        assert np.mean(draws[:, arm]) == pytest.approx(p.mean(), rel=0.02)
        assert np.var(draws[:, arm]) == pytest.approx(p.variance(), rel=0.02)


def test_the_bandit_finds_the_best_of_five_bernoulli_arms():
    probs = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    bandit = B.ThompsonGripBandit([NAIVE] * probs.size)
    rng, world = np.random.default_rng(0), np.random.default_rng(1)
    arms = []
    for _ in range(1000):
        arm = bandit.select_arm(0, rng)
        bandit.record_outcome(0, arm, float(world.random() < probs[arm]))
        arms.append(arm)
    assert np.mean(np.array(arms[-200:]) == 4) >= 0.9


def test_a_tie_goes_to_a_random_tied_arm():
    class TiedDraws:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(0)

        def beta(self, a, b):
            return np.array([0.1, 0.9, 0.9, 0.2])

        def choice(self, options):
            return self.rng.choice(options)

    bandit = B.ThompsonGripBandit([NAIVE] * 4)
    rng = TiedDraws()
    assert {bandit.select_arm(0, rng) for _ in range(50)} == {1, 2}


def test_the_config_grid_and_its_arms():
    config = B.load_config()
    arms = config.arms()
    assert len(arms) == 24 and config.bins == (3, 3)
    assert [a.grip_force for a in arms[:3]] == [0.5] * 3
    assert [a.close_speed for a in arms[:3]] == [0.05, 0.275, 0.5]
    lo, hi = BOUNDS["grip_force"]
    assert min(config.grip_force) >= lo and max(config.grip_force) <= hi
    for arm in arms:
        assert replace(arm, grip_force=NAIVE.grip_force, close_speed=NAIVE.close_speed) == NAIVE


def test_buckets_are_the_fixture_terciles():
    fixture = load_fixture()
    columns = [str(c) for c in fixture["probe_columns"]]
    heft = fixture["probe"][:, columns.index("heft_load")]
    slope = fixture["probe"][:, columns.index("closure_slope")]
    contextual = B.ThompsonGripBandit.from_config(contextual=True)
    np.testing.assert_allclose(contextual.heft_edges, np.quantile(heft, [1 / 3, 2 / 3]))
    np.testing.assert_allclose(contextual.slope_edges, np.quantile(slope, [1 / 3, 2 / 3]))
    assert contextual.n_buckets == 9 and len(contextual.arms) == 24
    h0, h1 = contextual.heft_edges
    s0, s1 = contextual.slope_edges
    assert contextual.bucket(_probe(h0 - 1.0, s0 - 1.0)) == 0
    assert contextual.bucket(_probe(h0, s0)) == 4  # on an edge: the upper bin
    assert contextual.bucket(_probe(h1 + 1.0, s1 + 1.0)) == 8
    assert contextual.bucket(_probe(h0 - 1.0, None)) == 2  # no indentation: the stiffest bin
    assert contextual.bucket({"heft_load": h1, "closure_slope": s0 - 1.0}) == 6
    flat = B.ThompsonGripBandit.from_config(contextual=False)
    assert flat.n_buckets == 1 and flat.bucket(_probe(h1 + 1.0, s1 + 1.0)) == 0


def test_the_policy_plays_an_arm_at_the_probe_and_credits_it():
    bandit = B.ThompsonGripBandit(B.load_config().arms(), heft_edges=[0.0])
    policy = B.BanditPolicy(bandit, np.random.default_rng(3))
    policy.begin_situation(12345)
    policy.begin_attempt(1)
    start = policy.decide(Percept(START, {}), DEFAULT_THETA)
    assert start.theta == DEFAULT_THETA and policy.pending is None
    decision = policy.decide(Percept(PROBE, {"probe": _probe(0.5, 2e4)}), DEFAULT_THETA)
    bucket, arm = policy.pending
    assert bucket == 1 and decision.trigger == PROBE
    played = bandit.arms[arm]
    assert decision.theta == replace(
        DEFAULT_THETA, grip_force=played.grip_force, close_speed=played.close_speed
    )
    raised = replace(decision.theta, grip_force=min(decision.theta.grip_force + 2.0, 25.0))
    assert policy.decide(Percept(ESCALATION, {}), raised).theta == raised  # keep the reflex raise
    policy.learn_attempt(_observed(False), [], decision.theta)
    assert (bandit.alpha[bucket, arm], bandit.beta[bucket, arm]) == (1.0, 2.0)
    assert policy.pending is None

    policy.begin_attempt(2)
    assert policy.decide(Percept(FAILURE, {}), DEFAULT_THETA).theta == DEFAULT_THETA
    policy.learn_attempt(_observed(False), [], None)  # a MISS before the probe
    assert bandit.alpha.sum() == bandit.alpha.size and bandit.beta.sum() == bandit.alpha.size + 1

    policy.begin_attempt(3)
    policy.decide(Percept(PROBE, {"probe": _probe(-1.0, 2e4)}), DEFAULT_THETA)
    bucket, arm = policy.pending
    policy.learn_attempt(_observed(True), [], None)
    assert bucket == 0 and bandit.alpha[bucket, arm] == 2.0


def test_a_new_attempt_drops_a_pending_arm():
    bandit = B.ThompsonGripBandit([NAIVE, replace(NAIVE, grip_force=2.0)])
    policy = B.BanditPolicy(bandit, np.random.default_rng(0))
    policy.decide(Percept(PROBE, {"probe": _probe(0.5, 2e4)}), DEFAULT_THETA)
    policy.begin_attempt(2)
    policy.learn_attempt(_observed(True), [], None)
    assert bandit.alpha.sum() == 2.0 and bandit.beta.sum() == 2.0


def test_make_policy_builds_a_fresh_bandit():
    rng = np.random.default_rng(0)
    first = B.make_policy("bandit", rng)
    first.bandit.record_outcome(0, 0, 1.0)
    assert B.make_policy("bandit", rng).bandit.alpha.sum() == first.bandit.alpha.size
    assert B.make_policy("bandit_flat", rng).bandit.n_buckets == 1
    settings = B.describe(B.load_config(), contextual=True)
    built = B.make_policy("bandit", rng, settings).bandit
    assert built.arms == first.bandit.arms
    np.testing.assert_array_equal(built.heft_edges, first.bandit.heft_edges)
    np.testing.assert_array_equal(built.slope_edges, first.bandit.slope_edges)
    small = {**settings, "grip_force": [2.0], "close_speed": [0.1]}
    assert B.make_policy("bandit", rng, small).bandit.arms == [replace(NAIVE, grip_force=2.0)]
    with pytest.raises(ValueError):
        B.make_policy("memory_full", rng)


def test_config_rejects_bad_files(tmp_path):
    path = tmp_path / "bandit.toml"
    path.write_text(
        "[arms]\ngrip_force = [30.0]\nclose_speed = [0.1]\n[buckets]\n"
        "heft_load = 3\nclosure_slope = 3\n"
    )
    with pytest.raises(ValueError):
        B.load_config(path)  # 30 N is outside the grip bounds
    path.write_text("[arms]\ngrip_force = [8.0]\nclose_speed = [0.1]\n[buckets]\nheft_load = 3\n")
    with pytest.raises(ValueError):
        B.load_config(path)
    assert isinstance(B.load_config().arms()[0], SkillParams)
