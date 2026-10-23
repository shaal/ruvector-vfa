"""T10: the benchmark runner, the baselines, and the calibration fixture (fast checks).

The baseline numbers themselves come from the Verify commands
(``python -m vfa.experiments.benchmark``), recorded in ``docs/RESULTS.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from vfa.experiments import benchmark as B
from vfa.experiments.baselines import NAIVE, grid_thetas, load_best_fixed, make_baseline
from vfa.sim import objects as OB
from vfa.skills.params import HIGH, LOW

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_fixtures as MF  # noqa: E402

CONFIG = B.load_benchmark_config()
OBJECTS = OB.load_object_config()


def test_config_profiles_and_seeds():
    assert CONFIG.task == "lift" and CONFIG.max_attempts == 5
    assert CONFIG.noise == "realistic"
    assert not set(CONFIG.calib_seeds) & set(CONFIG.eval_seeds)
    assert len(CONFIG.eval_seeds) == 3
    assert CONFIG.profiles["quick"] == B.Profile((500, 501), 60)  # tune seeds, not eval seeds
    assert CONFIG.profiles["full"] == B.Profile(CONFIG.eval_seeds, 60)
    assert set(CONFIG.profiles["calib"].seeds) == set(CONFIG.calib_seeds)
    others = set(CONFIG.calib_seeds) | set(CONFIG.eval_seeds) | {500, 501}
    assert not set(CONFIG.profiles["heldout"].seeds) & others


def test_config_rejects_unknown_keys(tmp_path):
    text = B.CONFIG_PATH.read_text().replace("max_attempts = 5", "max_attempts = 5\nretries = 2")
    path = tmp_path / "benchmark.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="unknown"):
        B.load_benchmark_config(path)


def test_situations_are_stratified_shuffled_and_repeatable():
    n = len(OBJECTS.names)
    a = B.make_situations(1000, 60, OBJECTS)
    assert a == B.make_situations(1000, 60, OBJECTS)
    assert B.make_situations(1000, 60, OBJECTS, start=13) == a[13:]
    for block in range(60 // n):  # one cube per class in every block
        assert sorted(s.class_name for s in a[block * n : (block + 1) * n]) == sorted(OBJECTS.names)
    # The order differs block to block: the index alone does not give the class.
    orders = {tuple(s.class_name for s in a[b * n : (b + 1) * n]) for b in range(60 // n)}
    assert len(orders) > 5
    other = B.make_situations(1001, 60, OBJECTS)
    assert [s.spec.mass for s in other] != [s.spec.mass for s in a]
    keys = [s.key for s in a] + [s.key for s in other]
    assert len(set(keys)) == len(keys)  # opaque and distinct


def test_env_seeds_are_repeatable_and_distinct():
    seeds = {B.env_seed(s, i, a) for s in (0, 1000) for i in range(30) for a in range(1, 6)}
    assert len(seeds) == 2 * 30 * 5
    assert B.env_seed(1000, 3, 2) == B.env_seed(1000, 3, 2)


def test_baselines():
    key = B.make_situations(1000, 5, OBJECTS)[4].key
    naive = make_baseline("naive")
    assert naive.theta(key, 1) == naive.theta(key + 1, 5) == NAIVE
    best = load_best_fixed()
    assert make_baseline("best_fixed").theta(key, 3) == best
    assert best.grip_force in CONFIG.grid["grip_force"]
    rnd = make_baseline("random_retry")
    assert rnd.theta(key, 1) == NAIVE
    draws = [rnd.theta(key, a).to_vector() for a in range(2, 6)]
    assert all(np.all(LOW <= d) and np.all(d <= HIGH) for d in draws)
    assert len({tuple(d) for d in draws}) == 4
    assert rnd.theta(key, 3) == rnd.theta(key, 3)
    assert len(grid_thetas(CONFIG.grid)) == 15 * 2
    with pytest.raises(ValueError):
        make_baseline("memory_full")


def test_bandit_conditions_run_one_chunk_per_seed(tmp_path):
    for condition, edges in (("bandit", 2), ("bandit_flat", 0)):
        jobs = B.plan_condition(CONFIG, condition, "quick", out=tmp_path)
        assert [(j.seed, j.start, j.end) for j in jobs] == [(500, 0, 60), (501, 0, 60)]
        settings = jobs[0].key["bandit"]
        assert len(settings["grip_force"]) == 8 and len(settings["close_speed"]) == 3
        assert len(settings["heft_edges"]) == len(settings["slope_edges"]) == edges
        assert jobs[0].theta is None
    assert "bandit" not in B.plan_condition(CONFIG, "memory_full", "quick", out=tmp_path)[0].key


def test_summarize_counts_situations_and_attempts():
    def rec(situation, attempt, result, cls="light_rigid"):
        return {
            "situation": situation,
            "attempt": attempt,
            "class": cls,
            "result": result,
            "success": result == "SUCCESS",
            "peak_force_excess": 2.0 if result == "SUCCESS" else None,
            "force_jerk": 10.0,
        }

    records = [
        rec(0, 1, "SUCCESS"),
        rec(1, 1, "BREAK", "fragile_soft"),
        rec(1, 2, "SUCCESS", "fragile_soft"),
        *(rec(2, a, "DROP", "heavy_slippery") for a in range(1, 6)),
        rec(25, 1, "SUCCESS"),  # in the late window (situations 21-60)
    ]
    m = B.summarize(records, max_attempts=5)
    assert m["situations"] == 4 and m["attempts"] == 9
    assert m["success_at_1"] == pytest.approx(2 / 4)
    assert m["success_at_k"] == pytest.approx(3 / 4)
    assert m["success_at_1_late"] == pytest.approx(1.0)
    assert m["attempts_to_success"] == pytest.approx((1 + 2 + 6 + 1) / 4)
    assert m["break_rate"] == pytest.approx(1 / 9) and m["drop_rate"] == pytest.approx(5 / 9)
    assert m["break_situation_rate"] == pytest.approx(1 / 4)
    assert m["force_jerk"] == pytest.approx(10.0)  # successes only
    assert m["peak_force_excess"] == pytest.approx(2.0)
    assert m["per_class"]["fragile_soft"] == {
        "situations": 1,
        "success_at_1": 0.0,
        "success_at_k": 1.0,
    }
    both = B.combine({0: m, 1: {**m, "success_at_1": 0.0}})
    assert both["success_at_1"]["mean"] == pytest.approx(1 / 4)
    assert both["success_at_1"]["std"] == pytest.approx(np.std([0.5, 0.0], ddof=1))


def test_calibration_fixture_layout():
    path = MF.FIXTURE
    assert path.stat().st_size <= MF.MAX_BYTES
    f = MF.load_fixture(path)
    n = len(f["offsets"]) - 1
    assert f["offsets"][0] == 0 and f["offsets"][-1] == len(f["rows"])
    assert np.all(np.diff(f["offsets"]) > 0)
    assert f["rows"].dtype == np.float32 and f["rows"].shape[1] == len(f["row_columns"])
    for key in ("condition", "seed", "group", "attempt", "prev", "result", "success", "theta"):
        assert len(f[key]) == n, key
    assert set(f["hidden_class"]) == set(OBJECTS.names)  # every class
    assert f["success"].any() and (~f["success"]).any()
    assert set(f["seed"]) == set(CONFIG.calib_seeds)  # never an evaluation seed
    assert set(f["condition"]) == {"naive", "random_retry"}
    assert "situation" not in f  # the index orders the classes: only hidden_situation
    # Every retry points at the attempt before it, at the same cube; a second attempt
    # follows naive's first (random_retry starts with naive's θ).
    for e, p in enumerate(f["prev"]):
        if f["attempt"][e] == 1:
            assert p == -1
        else:
            assert f["group"][p] == f["group"][e] and f["attempt"][p] == f["attempt"][e] - 1
            assert f["attempt"][e] != 2 or f["condition"][p] == "naive"
    # The episode order is shuffled: it does not follow the situation index.
    order = np.arange(n)
    assert abs(np.corrcoef(order, f["hidden_situation"])[0, 1]) < 0.3
    # No duplicate episodes: random_retry's attempt 1 is naive's (stored once).
    assert not ((f["condition"] == "random_retry") & (f["attempt"] == 1)).any()
    # Probe features exist wherever the probe finished (every fixture attempt did).
    assert not np.isnan(f["probe"][:, list(f["probe_columns"]).index("heft_peak")]).any()


def test_freeze_help_writes_nothing(tmp_path, monkeypatch, capsys):
    """``freeze --help`` prints the help and exits before it writes a hash."""
    monkeypatch.setattr(B, "HASH_PATH", tmp_path / "benchmark.sha256")
    monkeypatch.setattr(B, "PICK_PLACE_HASH", tmp_path / "benchmark_pick_place.sha256")
    for argv in (["freeze", "--help"], ["freeze", "--task", "pick_place", "--help"]):
        with pytest.raises(SystemExit):
            B.main(argv)
    assert "re-freeze" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
    assert B.main(["freeze", "--task", "pick_place"]) == 0
    assert B.PICK_PLACE_HASH.read_text().strip() == B.benchmark_hash(B.PICK_PLACE_FILES)
