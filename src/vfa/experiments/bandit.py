"""Thompson-sampling bandit: a self-learning baseline without similarity search (T17a).

ARCHITECTURE §9. The bandit learns from success and failure like memory, but it
never compares forces: if memory beats it, the gain comes from retrieval, not
from learning alone.

**Arms and buckets** (``configs/bandit.toml``). An arm is a ``grip_force`` ×
``close_speed`` pair of a fixed grid (8 × 3 = 24 arms; the other θ fields keep
``naive``'s values). A context bucket is a bin of the probe's heft load × a bin
of its closure-force slope (3 × 3 = 9 buckets); a feature's bin edges are its
quantiles over every episode of the calibration fixture
(``tests/fixtures/calib_episodes.npz``), so no evaluation data sets them. The
fixture also holds retries (of ``naive``, and of ``random_retry`` at other grasp
depths), so the bins are not equal shares of the probes the bandit sees (``naive``'s first attempts
fill the slope bins 0.20 / 0.44 / 0.36). A value on an edge goes to the upper
bin; a missing slope (no indentation, so the stiffest contact) goes to the top
bin. ``bandit_flat`` has one bucket: an online learner of the single best θ.
A chunk's key holds the resolved arms and edges, and the run builds the bandit
from them (``from_settings``).

**Selection and update.** Each (bucket, arm) has a ``Beta(α, β)`` posterior,
prior Beta(1, 1). ``select_arm`` draws one sample per arm with
``rng.beta(α, β)`` and takes the largest (a tie goes to a random one of the
tied arms, drawn from the same ``rng``); ``record_outcome`` adds the reward to α
and its complement to β (a success: α += 1; a failure: β += 1).

**The policy.** ``BanditPolicy`` is a System 2 policy for ``CognitiveCore`` and
``solve_situation``, so it sees what System 2 sees and decides at the same
point: ``naive`` θ at ``START`` and ``FAILURE``; at ``PROBE`` it buckets the
probe features and plays the selected arm; at ``ESCALATION`` it keeps the θ in
force (with any reflex raise). An attempt that reached its probe updates that
arm before the next decision; an attempt that ended before it (a ``MISS``)
updates nothing. Learning carries across situations, so one seed runs
sequentially. The policy reads only the probe features and the outcome.

RuVector mapping: ``ThompsonGripBandit`` ports ``MetaThompsonEngine``
(``select_arm``, ``record_outcome``) and ``BetaParams`` (``uniform``, ``mean``,
``variance``, ``sample``, ``update``) of
``ruvector-domain-expansion/src/transfer.rs`` for one domain with uniform
priors; the crate's ``ContextBucket`` (difficulty tier, category) becomes the
probe bin pair, its ``ArmId`` strings an arm index. Deviations: the crate
breaks a tie to the first arm, ours to a random tied arm; and the crate's
``BetaParams::sample`` is not a Beta draw. It maps one uniform u in [0.001,
0.999) through a one-term normal approximation when α and β are both above 1
(mean − std·√(−2 ln u) for u < 0.5, mean + std·√(−2 ln(1 − u)) otherwise), and
through a power-law formula otherwise, which the Beta(1, 1) prior itself hits.
On a Python transcription of that code (200 000 draws) Beta(1, 1) gives mean
0.530 and variance 0.128 (exact: 0.500, 0.083), Beta(2, 5) variance 0.075
(exact 0.026), and Beta(1, 4) mean 0.388 (exact 0.200). This port uses numpy's
exact ``Generator.beta``. The crate's transfer priors, cost EMAs, and dual-path
checks are not ported.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from vfa.cognition.cognitive_core import (
    DEFAULT_THETA,
    FAILURE,
    PROBE,
    START,
    Decision,
    ObservedAttempt,
    Percept,
)
from vfa.experiments.baselines import NAIVE, REPO_ROOT
from vfa.memory.embeddings import FIXTURE_PATH, load_fixture
from vfa.skills.params import SkillParams

CONFIG_PATH = REPO_ROOT / "configs" / "bandit.toml"
BANDIT_CONDITIONS: tuple[str, ...] = ("bandit", "bandit_flat")
FEATURES: tuple[str, ...] = ("heft_load", "closure_slope")  # the bucket features, in order


@dataclass(slots=True)
class BetaParams:
    """A Beta(α, β) posterior over an arm's success probability (the crate's type)."""

    alpha: float = 1.0
    beta: float = 1.0

    @classmethod
    def uniform(cls) -> BetaParams:
        return cls(1.0, 1.0)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1.0))

    def sample(self, rng: np.random.Generator, size: int | None = None) -> Any:
        """An exact Beta draw (not the crate's approximation; refer to the module doc)."""
        return rng.beta(self.alpha, self.beta, size)

    def update(self, reward: float) -> None:
        self.alpha += reward
        self.beta += 1.0 - reward


@dataclass(frozen=True, slots=True)
class BanditConfig:
    """``configs/bandit.toml``: the arm grid and the bins per bucket feature."""

    grip_force: tuple[float, ...]
    close_speed: tuple[float, ...]
    bins: tuple[int, int]  # heft_load, closure_slope

    def arms(self) -> list[SkillParams]:
        """Every grip × closing-speed pair, grip-major; the other fields from ``NAIVE``."""
        return [
            replace(NAIVE, grip_force=g, close_speed=c)
            for g in self.grip_force
            for c in self.close_speed
        ]


def load_config(path: str | Path = CONFIG_PATH) -> BanditConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if set(data) != {"arms", "buckets"}:
        raise ValueError(f"{path}: expected exactly the [arms] and [buckets] tables")
    if set(data["arms"]) != {"grip_force", "close_speed"} or set(data["buckets"]) != set(FEATURES):
        raise ValueError(f"{path}: [arms] needs grip_force, close_speed; [buckets] {FEATURES}")
    bins = tuple(int(data["buckets"][f]) for f in FEATURES)
    if min(bins) < 1:
        raise ValueError(f"{path}: every feature needs at least 1 bin, got {bins}")
    config = BanditConfig(
        tuple(float(g) for g in data["arms"]["grip_force"]),
        tuple(float(c) for c in data["arms"]["close_speed"]),
        (bins[0], bins[1]),
    )
    if not config.grip_force or not config.close_speed:
        raise ValueError(f"{path}: the arm grid is empty")
    config.arms()  # SkillParams checks every arm against its bounds
    return config


def bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """The quantiles k/bins (k = 1 .. bins − 1) of the finite ``values``."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return np.quantile(finite, np.arange(1, bins) / bins)


def fixture_edges(
    bins: tuple[int, int], path: str | Path = FIXTURE_PATH
) -> tuple[np.ndarray, np.ndarray]:
    """The bucket features' bin edges over every episode of the calibration fixture."""
    fixture = load_fixture(path)
    columns = [str(c) for c in fixture["probe_columns"]]
    probe = fixture["probe"]
    heft, slope = (
        bin_edges(probe[:, columns.index(f)], n) for f, n in zip(FEATURES, bins, strict=True)
    )
    return heft, slope


class ThompsonGripBandit:
    """Thompson sampling over θ arms per probe bucket (refer to the module doc)."""

    def __init__(
        self,
        arms: Sequence[SkillParams],
        heft_edges: Sequence[float] = (),
        slope_edges: Sequence[float] = (),
    ) -> None:
        if not arms:
            raise ValueError("a bandit needs at least one arm")
        self.arms = list(arms)
        self.heft_edges = np.asarray(heft_edges, dtype=float)
        self.slope_edges = np.asarray(slope_edges, dtype=float)
        for edges in (self.heft_edges, self.slope_edges):
            if not (np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0)):
                raise ValueError(f"bin edges must be finite and increasing, got {edges}")
        n_buckets = (self.heft_edges.size + 1) * (self.slope_edges.size + 1)
        self.alpha = np.ones((n_buckets, len(self.arms)))
        self.beta = np.ones((n_buckets, len(self.arms)))

    @classmethod
    def from_settings(cls, settings: Mapping[str, Sequence[float]]) -> ThompsonGripBandit:
        """A fresh bandit from ``describe``'s resolved arms and edges."""
        arms = [
            replace(NAIVE, grip_force=float(g), close_speed=float(c))
            for g in settings["grip_force"]
            for c in settings["close_speed"]
        ]
        return cls(arms, settings["heft_edges"], settings["slope_edges"])

    @classmethod
    def from_config(
        cls, contextual: bool, config: BanditConfig | None = None
    ) -> ThompsonGripBandit:
        """``bandit`` (``contextual``: the probe buckets) or ``bandit_flat`` (one bucket)."""
        config = load_config() if config is None else config
        if not contextual:
            return cls(config.arms())
        return cls(config.arms(), *fixture_edges(config.bins))

    @property
    def n_buckets(self) -> int:
        return self.alpha.shape[0]

    def bucket(self, probe: Any) -> int:
        """The bucket of a probe (``ProbeFeatures`` or its dict form)."""
        get = probe.get if isinstance(probe, Mapping) else lambda name: getattr(probe, name)
        heft = float(get("heft_load"))
        slope = get("closure_slope")
        slope = np.inf if slope is None else float(slope)
        if not np.isfinite(heft) or np.isnan(slope):
            raise ValueError(f"probe features must be numbers, got heft {heft}, slope {slope}")
        h = int(np.searchsorted(self.heft_edges, heft, side="right"))
        s = int(np.searchsorted(self.slope_edges, slope, side="right"))
        return h * (self.slope_edges.size + 1) + s

    def posterior(self, bucket: int, arm: int) -> BetaParams:
        return BetaParams(float(self.alpha[bucket, arm]), float(self.beta[bucket, arm]))

    def draws(self, bucket: int, rng: np.random.Generator) -> np.ndarray:
        """One exact Beta draw per arm of ``bucket``."""
        return rng.beta(self.alpha[bucket], self.beta[bucket])

    def select_arm(self, bucket: int, rng: np.random.Generator) -> int:
        """The arm with the largest draw (ties: a random tied arm)."""
        draws = self.draws(bucket, rng)
        best = np.flatnonzero(draws == draws.max())
        return int(best[0]) if best.size == 1 else int(rng.choice(best))

    def record_outcome(self, bucket: int, arm: int, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1], got {reward!r}")
        self.alpha[bucket, arm] += reward
        self.beta[bucket, arm] += 1.0 - reward


class BanditPolicy:
    """``ThompsonGripBandit`` as a System 2 policy (refer to the module doc)."""

    def __init__(self, bandit: ThompsonGripBandit, rng: np.random.Generator) -> None:
        self.bandit = bandit
        self.rng = rng
        self.pending: tuple[int, int] | None = None  # (bucket, arm) of this attempt's probe

    def decide(self, percept: Percept, theta: SkillParams) -> Decision:
        if percept.source in (START, FAILURE):
            return Decision(DEFAULT_THETA, percept.source, "naive θ until the probe")
        if percept.source != PROBE:  # ESCALATION: keep the θ in force and the reflex raise
            return Decision(theta, percept.source, "keep θ")
        bucket = self.bandit.bucket(percept.data["probe"])
        arm = self.bandit.select_arm(bucket, self.rng)
        self.pending = (bucket, arm)
        played = self.bandit.arms[arm]
        post = self.bandit.posterior(bucket, arm)
        new = replace(theta, grip_force=played.grip_force, close_speed=played.close_speed)
        return Decision(
            new,
            PROBE,
            f"bandit bucket {bucket} arm {arm}, Beta({post.alpha:g}, {post.beta:g})",
            utility=post.mean(),
            evidence=({"bucket": bucket, "arm": arm, "alpha": post.alpha, "beta": post.beta},),
        )

    def begin_situation(self, key: int) -> None:
        self.pending = None

    def begin_attempt(self, attempt: int) -> None:
        self.pending = None

    def learn_attempt(
        self, observed: ObservedAttempt, used: Sequence[str], chosen: SkillParams | None = None
    ) -> None:
        """Credit the arm of this attempt's probe decision, if it had one."""
        if self.pending is not None:
            bucket, arm = self.pending
            self.bandit.record_outcome(bucket, arm, 1.0 if observed.success else 0.0)
        self.pending = None

    def end_situation(self) -> None:
        pass


def make_policy(
    condition: str,
    rng: np.random.Generator,
    settings: Mapping[str, Sequence[float]] | None = None,
) -> BanditPolicy:
    """A fresh policy for one run of ``bandit`` or ``bandit_flat`` (``settings``: a
    chunk key's resolved arms and edges; default: ``configs/bandit.toml``)."""
    if condition not in BANDIT_CONDITIONS:
        raise ValueError(f"condition must be one of {BANDIT_CONDITIONS}, got {condition!r}")
    if settings is None:
        settings = describe(load_config(), condition == "bandit")
    return BanditPolicy(ThompsonGripBandit.from_settings(settings), rng)


def describe(config: BanditConfig, contextual: bool) -> dict[str, Any]:
    """The resolved settings of a condition (its chunk key carries them)."""
    heft, slope = fixture_edges(config.bins) if contextual else (np.empty(0), np.empty(0))
    return {
        "grip_force": list(config.grip_force),
        "close_speed": list(config.close_speed),
        "heft_edges": heft.tolist(),
        "slope_edges": slope.tolist(),
    }
