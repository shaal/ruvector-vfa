"""Non-adaptive baselines for the frozen benchmark (T10, ARCHITECTURE §9).

A condition is a policy that picks θ for each attempt of a situation:

- ``naive`` (``NAIVE``): one engineer's default, repeated on every retry: 8 N
  per finger (16 N summed, about what a 0.75 kg cube at μ 0.5 needs at T05's
  lift) as a middle ground between dropping heavy objects and crushing light
  ones; a 0.1 m/s squeeze ramp; T05's lift (0.6 m/s² peak); a grasp at the
  cube's center; a 0.4 m/s approach. These are ``SkillParams()``'s defaults,
  chosen without benchmark data.
- ``best_fixed``: the single θ with the highest success@1 on the calibration
  seeds (``configs/best_fixed.toml``, found by ``benchmark calibrate`` over the
  ``[grid]`` of ``configs/benchmark.toml``), repeated on every retry: the
  non-adaptive baseline with the highest success@1.
- ``random_retry``: ``naive`` first, then θ drawn uniformly within the
  ``SkillParams`` bounds on each retry (from a generator seeded by the
  situation's opaque key and the attempt, so a re-run repeats it).

All get the same ``max_attempts``. With sensor noise each attempt sees its own
noise, so a repeated θ can still end differently.

RuVector mapping: the baselines stand in for a ``CognitiveCore`` with no
``MemorySystem``: after ``perceive()``, ``think()`` returns the same decision
whatever was perceived (θ has no Rust type; it rides in the side table of
ARCHITECTURE §10), so the comparison with memory isolates what retrieval adds.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vfa.skills.params import HIGH, LOW, SkillParams

REPO_ROOT = Path(__file__).resolve().parents[3]
BEST_FIXED_PATH = REPO_ROOT / "configs" / "best_fixed.toml"
NAIVE = SkillParams()
CONDITIONS: tuple[str, ...] = ("naive", "best_fixed", "random_retry")


@dataclass(frozen=True, slots=True)
class Baseline:
    """A non-adaptive policy: ``theta(key, attempt)`` (attempt from 1).

    ``key`` is the situation's opaque key (``Situation.key``): policies never see
    the situation index, which would let them infer the stratified class.
    """

    name: str
    fixed: SkillParams | None = None  # naive, best_fixed: the θ of every attempt

    def theta(self, key: int, attempt: int) -> SkillParams:
        if attempt < 1:
            raise ValueError(f"attempt counts from 1, got {attempt}")
        if self.fixed is not None:
            return self.fixed
        if attempt == 1:  # random_retry: naive first
            return NAIVE
        rng = np.random.default_rng((key, attempt, 0x52414E44))  # "RAND"
        return SkillParams.from_vector(rng.uniform(LOW, HIGH))


def load_best_fixed(path: str | Path = BEST_FIXED_PATH) -> SkillParams:
    """θ of the ``best_fixed`` baseline (``configs/best_fixed.toml``, ``[theta]``)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if set(data) - {"theta", "calibration"} or "theta" not in data:
        raise ValueError(f"{path}: expected a [theta] table (and optional [calibration])")
    return SkillParams.from_dict(data["theta"])


def make_baseline(name: str, best_fixed: SkillParams | None = None) -> Baseline:
    """The baseline ``name`` (``best_fixed`` loads ``configs/best_fixed.toml`` if not given)."""
    if name == "naive":
        return Baseline("naive", NAIVE)
    if name == "best_fixed":
        return Baseline("best_fixed", best_fixed if best_fixed is not None else load_best_fixed())
    if name == "random_retry":
        return Baseline("random_retry")
    raise ValueError(f"condition must be one of {CONDITIONS}, got {name!r}")


def grid_thetas(grid: dict[str, Sequence[float]]) -> list[SkillParams]:
    """Every θ of the calibration grid (other fields from ``NAIVE``), grip-major order."""
    grips = [float(g) for g in grid["grip_force"]]
    accels = [float(a) for a in grid["lift_accel"]]
    base = NAIVE.to_dict()
    return [
        SkillParams.from_dict({**base, "grip_force": g, "lift_accel": a})
        for g in grips
        for a in accels
    ]
