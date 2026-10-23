"""T18: the run plan, the paired bootstrap, the verdict rule, and the targets (no physics)."""

from __future__ import annotations

import numpy as np

from vfa.experiments import benchmark as B
from vfa.experiments import run_all as R


def rec(situation: int, attempt: int, result: str) -> dict:
    return {
        "situation": situation,
        "attempt": attempt,
        "result": result,
        "success": result == "SUCCESS",
    }


def test_the_plan_runs_every_condition_longest_first(tmp_path):
    jobs = R.plan(B.load_benchmark_config(), tmp_path)
    assert jobs[0].condition == B.FROZEN and jobs[0].seed == 2000
    kinds = [
        0 if j.condition == B.FROZEN else 1 if j.condition in B.LEARNING_CONDITIONS else 2
        for j in jobs
    ]
    assert kinds == sorted(kinds)
    full = {j.condition for j in jobs if j.seed in (1000, 1001, 1002)}
    assert full == set(R.FULL) and len(R.FULL) == 11
    assert {j.condition for j in jobs if j.seed == 2000} == set(R.HELDOUT)


def test_situations_count_first_tries_and_breaks():
    records = [rec(0, 1, "SUCCESS"), rec(25, 1, "BREAK"), rec(25, 2, "SUCCESS"), rec(30, 1, "DROP")]
    s = R.Situations.of({7: records}, max_attempts=5)
    assert s.units == ((7, 0), (7, 25), (7, 30))
    assert s.first.tolist() == [1.0, 0.0, 0.0] and s.tries.tolist() == [1.0, 2.0, 6.0]
    assert s.breaks.tolist() == [0.0, 1.0, 0.0] and s.late.tolist() == [False, True, True]


def test_bootstrap_is_paired_and_the_verdict_rule():
    rng = np.random.default_rng(1)
    rows = [rec(i, 1, "SUCCESS" if rng.random() < 0.7 else "BREAK") for i in range(60)]
    a = R.Situations.of({0: rows}, 5)
    same = R.bootstrap(a, a, draws=500)
    assert all(lo == hi == 0.0 for lo, hi in same.values())
    worse = R.Situations.of({0: [rec(i, 1, "BREAK") for i in range(60)]}, 5)
    gap = R.bootstrap(worse, a, draws=500)
    assert gap["success_at_1_late"][1] < 0 and gap["attempts_to_success"][0] > 0
    assert R.verdict(gap) == "it helps"
    assert R.verdict(R.bootstrap(a, worse, draws=500)) == "it hurts"
    assert R.verdict(same) == "no measurable effect"
    mixed = {"success_at_1_late": [0.05, 0.2], "attempts_to_success": [0.1, 0.4]}
    assert R.verdict(mixed) == "mixed"
    on_zero = {"success_at_1_late": [0.0, 0.025], "attempts_to_success": [-0.0167, 0.0]}
    assert R.verdict(on_zero) == "no measurable effect"  # a bound at 0 does not exclude 0


def test_targets_use_the_like_for_like_bar():
    def stats(late, overall, tries, excess, jerk):
        return {
            "success_at_1_late": {"mean": late},
            "success_at_1": {"mean": overall},
            "attempts_to_success": {"mean": tries},
            "peak_force_excess": {"mean": excess},
            "force_jerk": {"mean": jerk},
        }

    full = {
        "best_fixed": stats(0.658, 0.650, 2.7, 5.0, 38.0),
        "random_retry": stats(0.658, 0.639, 2.2, 7.8, 48.0),
        "memory_full": stats(0.805, 0.8, 1.2, 11.0, 36.0),
    }
    t = R.targets(full)
    assert not t["success_at_1_late"]["met"]  # between the loose 0.800 and the bar 0.808
    assert t["attempts_to_success"]["met"] and t["gentler_force"]["met"]
    full["memory_full"] = stats(0.81, 0.8, 1.2, 11.0, 36.0)
    assert R.targets(full)["success_at_1_late"]["met"]  # above both bars
    full["memory_full"] = stats(0.805, 0.8, 1.2, 11.0, 36.0)
    full["best_fixed"] = stats(0.640, 0.660, 2.7, 5.0, 38.0)  # now the loose bar is higher
    t = R.targets(full)
    assert t["success_at_1_late"]["bar"] < 0.805 < t["success_at_1_late"]["loose_bar"]
    assert not t["success_at_1_late"]["met"]
