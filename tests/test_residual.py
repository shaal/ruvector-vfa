"""T21: the residual System 1 policy and its ES trainer.

ROADMAP acceptance: the training curve (validation reward, smoothed) and the
quick comparison are benchmark runs (docs/RESULTS.md §11). These tests pin the
residual's features, bounds, and command shaping, that a skill without one runs
as before, the ES pieces, the trainer's resume, and the benchmark condition.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from vfa.experiments import benchmark as B
from vfa.experiments import train_residual as T
from vfa.sim import env as E
from vfa.skills import residual as R
from vfa.skills.params import SkillParams
from vfa.skills.pick import GraspLiftSkill, run_attempt
from vfa.skills.place import PickPlaceSkill


def test_features_are_observable_and_bounded() -> None:
    x = R.features("LIFT", np.array([3.0, 4.0, -12.0, 0.1, 0.2, 0.3]), 9.0, 4.0, 8.0, True)
    assert x.tolist() == [0.0, 1.0, 0.0, -1.2, 0.5, 0.9, 0.5]
    assert R.features("LIFT", np.zeros(6), 1.0, 9.0, 8.0, True)[6] == 0.0  # clipped at 0
    assert R.features("LIFT", np.zeros(6), 1.0, 4.0, 8.0, False)[6] == 0.0  # no watch: no slip
    nan = R.features("PROBE", np.full(6, np.nan), math.nan, math.nan, math.nan, True)
    assert nan.tolist() == [0.0] * len(R.FEATURES)


def test_outputs_stay_in_their_bounds() -> None:
    for scale in (-1e6, -3.0, 0.0, 3.0, 1e6):
        policy = R.ResidualPolicy(np.full((2, len(R.FEATURES)), scale))
        step, vz = policy.outputs(np.ones(len(R.FEATURES)))
        assert 0.0 <= step <= R.STEP_MAX and -R.VZ_MAX <= vz <= R.VZ_MAX
    step, vz = R.ResidualPolicy().outputs(np.zeros(len(R.FEATURES)))
    assert step == pytest.approx((R.STEP_MIN + R.STEP_MAX) / 2) and vz == 0.0  # untrained
    floor, _ = R.ResidualPolicy(np.full((2, len(R.FEATURES)), -1e6)).outputs(
        np.ones(len(R.FEATURES))
    )
    assert floor == pytest.approx(R.STEP_MIN)  # the command always rises
    with pytest.raises(ValueError):
        R.ResidualPolicy(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        R.ResidualPolicy(np.full((2, len(R.FEATURES)), np.nan))


def test_the_command_rises_by_the_step_and_falls_at_once() -> None:
    state = R.ResidualState()
    assert state.grip(8.0, 0.5, start=1.5) == 2.0
    assert state.grip(8.0, 0.5, start=99.0) == 2.5  # the start only seeds the command
    assert state.grip(2.2, 0.5, start=0.0) == 2.2  # a lower command applies at once
    assert state.grip(2.4, 1.0, start=0.0) == 2.4  # never past the target
    for _ in range(10_000):
        state.move(-R.VZ_MAX, 0.002)
    assert state.z_offset == R.Z_OFFSET[0]


def test_weights_round_trip(tmp_path) -> None:
    policy = R.ResidualPolicy.from_vector(np.arange(2 * len(R.FEATURES)) / 10.0)
    assert R.ResidualPolicy.load(policy.save(tmp_path / "w.npz")).to_vector().tolist() == (
        policy.to_vector().tolist()
    )
    assert R.ResidualPolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy
    with pytest.raises(ValueError):
        R.ResidualPolicy.from_dict({**policy.to_dict(), "features": ["x"]})
    assert R.ResidualPolicy.load().to_vector().shape == (2 * len(R.FEATURES),)  # the installed one


def test_without_a_residual_the_skill_never_takes_its_path(monkeypatch) -> None:
    assert GraspLiftSkill().residual is None

    def boom(self, pos):
        raise AssertionError("the residual path ran")

    monkeypatch.setattr(GraspLiftSkill, "_residual_action", boom)
    env = B.make_env(B.load_benchmark_config())
    result = run_attempt(
        env, GraspLiftSkill(), SkillParams(), seed=3, options={"object_class": "light_rigid"}
    )
    assert result.result == "SUCCESS"


def test_a_slow_step_flattens_the_squeeze_ramp() -> None:
    env = B.make_env(B.load_benchmark_config())
    slow = np.zeros((2, len(R.FEATURES)))
    slow[0, 0] = -3.0  # SQUEEZE: about 7 N/s per finger
    jerks = []
    for skill in (GraspLiftSkill(), GraspLiftSkill(R.ResidualPolicy(slow))):
        result = run_attempt(
            env, skill, SkillParams(), seed=3, options={"object_class": "light_rigid"}
        )
        assert result.result == "SUCCESS" and result.theta_final.grip_force == 8.0  # θ untouched
        ep = result.episode
        phases = np.asarray(ep.phases)[ep.arrays["phase"]]
        rows = phases == "SQUEEZE"
        force = np.sum(ep.arrays["finger_force"][rows], axis=1)
        rate = np.diff(force) / np.diff(ep.arrays["t"][rows])
        jerks.append(float(np.sqrt(np.mean(rate * rate))))  # RMS force rate in SQUEEZE
    assert jerks[1] < 0.7 * jerks[0]


def test_squeeze_waits_for_the_grip_under_the_slowest_step() -> None:
    """At the step floor (5 N/s) the squeeze outlasts the gripper's own ramp time; LIFT
    must still start at θ's grip."""
    floor = np.zeros((2, len(R.FEATURES)))
    floor[0, :3] = -50.0  # every phase at the floor

    class Watch(GraspLiftSkill):
        def _enter(self, phase: str) -> None:
            if phase == "LIFT":
                self.at_lift = self._env.gripper.actuator_squeeze()
            super()._enter(phase)

    env = B.make_env(B.load_benchmark_config())
    skill = Watch(R.ResidualPolicy(floor))
    theta = SkillParams(grip_force=16.0)
    result = run_attempt(env, skill, theta, seed=3, options={"object_class": "heavy_rigid"})
    assert skill.at_lift == pytest.approx(16.0, rel=0.02)
    assert result.result == "SUCCESS" and result.sim_s < E.TIME_LIMIT_S["lift"]


@pytest.mark.slow
def test_the_squeeze_budget_keeps_the_slowest_corner_in_time() -> None:
    """At the slowest θ corner and the step floor the squeeze cannot reach 25 N in time:
    SQUEEZE stops where the lift and the hold still fit, and the attempt ends inside the
    time limit."""
    floor = np.zeros((2, len(R.FEATURES)))
    floor[0, :3] = -50.0
    slow = SkillParams(
        grip_force=25.0, close_speed=0.05, lift_accel=0.3, grasp_depth=0.4, approach_speed=0.35
    )
    env = B.make_env(B.load_benchmark_config())
    result = run_attempt(
        env,
        GraspLiftSkill(R.ResidualPolicy(floor)),
        slow,
        seed=0,
        options={"object_class": "heavy_rigid"},
    )
    lift = [h["t"] for h in result.episode.params_history]  # θ is set once: no raise here
    assert result.sim_s < E.TIME_LIMIT_S["lift"] and result.result != "TIMEOUT", (
        result.result,
        lift,
    )


def test_the_vertical_output_moves_the_hold() -> None:
    up = np.zeros((2, len(R.FEATURES)))
    up[1, 1:3] = 50.0  # LIFT and HOLD: +VZ_MAX
    env = B.make_env(B.load_benchmark_config())
    tops = []
    for skill in (GraspLiftSkill(), GraspLiftSkill(R.ResidualPolicy(up))):
        result = run_attempt(
            env, skill, SkillParams(), seed=3, options={"object_class": "light_rigid"}
        )
        ep = result.episode
        hold = np.asarray(ep.phases)[ep.arrays["phase"]] == "HOLD"
        tops.append(float(np.max(ep.arrays["ee_pos"][hold, 2])))
    assert tops[1] - tops[0] == pytest.approx(R.Z_OFFSET[1], abs=0.004)


def test_only_the_residual_condition_gets_a_residual(tmp_path) -> None:
    config = B.load_benchmark_config()
    for condition in ("memory_full+residual", "memory_full"):
        job = B.plan_condition(config, condition, "quick", seeds=[500], out=tmp_path)[0]
        job = B.Job(**{**{f: getattr(job, f) for f in B.Job.__dataclass_fields__}, "end": 1})
        B.run_chunk(job)
        assert B._WORKER["skill"].residual is None  # the cached worker skill stays plain


def test_pick_place_takes_no_residual() -> None:
    from vfa.skills import place

    with pytest.raises(ValueError):
        PickPlaceSkill(R.ResidualPolicy()).validate(place.make_env())


# -------------------------------------------------------------------- ES


def test_es_climbs_a_quadratic() -> None:
    """The trainer's ES pieces (antithetic pairs, centered ranks, Adam) on a toy reward."""
    target = np.array([1.0, -2.0, 0.5])
    center, adam = np.zeros(3), T.Adam(3, 0.1)
    rng = np.random.default_rng(0)
    for _ in range(300):
        eps = rng.standard_normal((T.PAIRS, 3))
        score = lambda v: -float(np.sum((v - target) ** 2))  # noqa: E731
        plus = np.array([score(center + T.SIGMA * e) for e in eps])
        minus = np.array([score(center - T.SIGMA * e) for e in eps])
        center = center + adam.step(T.es_gradient(eps, plus, minus, T.SIGMA))
    np.testing.assert_allclose(center, target, atol=0.15)
    assert T.centered_ranks(np.array([3.0, 1.0, 2.0])).tolist() == [0.5, -0.5, 0.0]
    assert T.smoothed([1.0, 3.0, 5.0], 2) == [1.0, 2.0, 4.0]


def test_training_resumes_by_generation(tmp_path, monkeypatch) -> None:
    """A toy replay and a toy evaluator: a deadline stops between generations, the same
    call continues from the checkpoint, and the end writes the weights and the curve."""
    pool = [{"seed": 0, "situation": i, "attempt": 1, "theta": {}} for i in range(6)]
    monkeypatch.setattr(T, "replay_set", lambda *a: {0: pool, 1: pool[:3]})

    def fake_evaluate(_pool, vectors, episodes):
        rows = []
        for v in vectors:
            r = -float(np.sum((np.asarray(v) - 0.3) ** 2))
            rows.append([[r, 1.0, 20.0, 5.0]] * len(episodes))
        return np.array(rows)

    monkeypatch.setattr(T, "evaluate", fake_evaluate)
    assert not T.train(tmp_path, 3, 4, 1, deadline=0.0)  # generation 0 only
    assert not T.train(tmp_path, 3, 4, 1, deadline=0.0)  # resumes: still generation 0
    curve = json.loads((tmp_path / "curve.json").read_text())
    assert [g["generation"] for g in curve["generations"]] == [0]
    assert T.train(tmp_path, 3, 4, 1, deadline=1e9)
    curve = json.loads((tmp_path / "curve.json").read_text())
    assert [g["generation"] for g in curve["generations"]] == [0, 1, 2, 3]
    assert curve["generations"][-1]["reward"] > curve["generations"][0]["reward"]
    assert (tmp_path / "residual.npz").exists() and (tmp_path / "curve.png").exists()
    curve["generations"].append({**curve["generations"][-1], "generation": 4})  # a crash
    (tmp_path / "curve.json").write_text(json.dumps(curve))  # after the curve, before the state
    assert T.train(tmp_path, 4, 4, 1, deadline=1e9)  # generation 4 runs again, once
    gens = [
        g["generation"] for g in json.loads((tmp_path / "curve.json").read_text())["generations"]
    ]
    assert gens == [0, 1, 2, 3, 4]
    with pytest.raises(SystemExit):
        T.train(tmp_path, 4, 2, 1, deadline=1e9)  # other settings: refused
    monkeypatch.setattr(T, "replay_set", lambda *a: {0: pool[1:], 1: pool[:3]})
    with pytest.raises(SystemExit):
        T.train(tmp_path, 4, 4, 1, deadline=1e9)  # another replay: refused
    np.savez(tmp_path / "state.npz", center=np.zeros(14), generation=1)  # from before the key
    with pytest.raises(SystemExit):
        T.train(tmp_path, 4, 4, 1, deadline=1e9)


# ------------------------------------------------------------- benchmark


def test_the_residual_condition_carries_its_weights(monkeypatch, tmp_path) -> None:
    config = B.load_benchmark_config()
    jobs = B.plan_condition(config, "memory_full+residual", "quick")
    assert all(j.key["residual"] == R.ResidualPolicy.load().to_dict() for j in jobs)
    assert "residual" not in B.plan_condition(config, "memory_full", "quick")[0].key
    monkeypatch.setattr(R, "RESIDUAL_PATH", tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError):
        B.plan_condition(config, "memory_full+residual", "quick")
    with pytest.raises(ValueError):
        B.plan_condition(B.load_pick_place_config(), "memory_full+residual", "quick")
