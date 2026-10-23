"""T11: force / trajectory / context embeddings (``vfa.memory.embeddings``).

ROADMAP acceptance: deterministic; every vector has unit norm; fixed dimensions,
documented and asserted; on the calibration fixture, same-class ``key_probe``
pairs are more similar than cross-class pairs (mean cosine gap ≥ 0.1), and the
context-only key shows a smaller gap (both printed).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vfa.experiments import benchmark as B
from vfa.experiments.baselines import NAIVE
from vfa.memory import embeddings as M
from vfa.sim import env as E
from vfa.sim import objects as OB
from vfa.sim.recorder import PRIVILEGED_COLUMNS
from vfa.skills.pick import PROBE_DONE, GraspLiftSkill, run_attempt

FIXTURE = M.load_fixture()
VIEWS = M.fixture_views(FIXTURE)
EMB = M.Embedder()
FIRST = [i for i in range(len(VIEWS)) if FIXTURE["attempt"][i] == 1]  # one per cube


def all_keys(view: M.EpisodeView) -> dict[str, np.ndarray]:
    return {
        "key_pre": EMB.key_pre(view.context),
        "key_probe": EMB.key_probe(view),
        "key_retry": EMB.key_retry(view),
        "full_embedding": EMB.full_embedding(view),
    }


def class_gap(name: str, rows: list[int]) -> float:
    """Mean cosine of same-class pairs minus that of cross-class pairs."""
    keys = np.stack([all_keys(VIEWS[i])[name] for i in rows])
    sims = keys @ keys.T
    upper = np.triu_indices(len(rows), 1)
    labels = FIXTURE["hidden_class"][rows]  # scoring only
    same = labels[upper[0]] == labels[upper[1]]
    return float(sims[upper][same].mean() - sims[upper][~same].mean())


def test_dimensions_are_fixed_and_documented():
    assert M.DIMS == {
        "context": 12,
        "probe": 6,
        "probe_profile": 32,
        "force_wrist": 192,
        "force_pad": 65,
        "force_moments": 5,
        "trajectory": 64,
        "failure": 4,
        "outcome": 10,
    }
    assert {k: M.key_dims(k) for k in M.KEYS} == {
        "key_pre": 12,
        "key_probe": 50,
        "key_retry": 213,
        "full_embedding": 354,
    }
    assert M.force_features(VIEWS[0]).shape == (262,)
    assert M.trajectory_features(VIEWS[0]).shape == (64,)
    # key_retry leaves out the pad force: after the probe it follows the commanded grip.
    assert "force_pad" not in M.KEYS["key_retry"] and "force_pad" in M.KEYS["full_embedding"]
    assert M.START_ARM_QPOS == E.START_ARM_QPOS
    for view in VIEWS[:20]:
        for group, raw in M.raw_groups(view).items():
            assert raw.shape == (M.DIMS[group],), group
        for name, key in all_keys(view).items():
            assert key.shape == (M.key_dims(name),), name


def test_every_key_is_unit_norm_and_deterministic():
    for view in VIEWS:
        a, b = all_keys(view), all_keys(view)
        for name in M.KEYS:
            assert np.all(np.isfinite(a[name])), name
            assert np.linalg.norm(a[name]) == pytest.approx(1.0, abs=1e-12), name
            assert np.array_equal(a[name], b[name]), name


def test_committed_stats_are_the_fit_on_the_fixture():
    # Within round-off, not bit for bit: summation order differs across CPUs (CI runs on
    # Linux x86, the stats were fitted on macOS arm64); a changed feature or fixture moves
    # a mean or std far more than 1e-9.
    fresh = M.EmbeddingStats.fit(VIEWS)
    stored = M.EmbeddingStats.load()
    for g in M.GROUPS:
        np.testing.assert_allclose(fresh.mean[g], stored.mean[g], rtol=1e-9, atol=1e-12, err_msg=g)
        np.testing.assert_allclose(fresh.std[g], stored.std[g], rtol=1e-9, atol=1e-12, err_msg=g)


def test_same_class_probe_keys_are_closer_than_cross_class():
    probe = class_gap("key_probe", FIRST)
    context = class_gap("key_pre", FIRST)
    print(f"cosine gap, same class minus cross class: key_probe {probe:.3f}, key_pre {context:.3f}")
    assert probe >= 0.1
    assert context < probe


def test_failure_buckets_and_missing_probe():
    view = VIEWS[FIRST[0]]
    for result, bucket in (
        ("BREAK", "BREAK"),
        ("SLIP", "UNDERGRIP"),
        ("DROP", "UNDERGRIP"),
        ("UNDERGRIP", "UNDERGRIP"),  # T14 merges the two labels before System 2
        ("MISS", "MISS"),
        ("TIMEOUT", "OTHER"),
        ("SAFETY_STOP", "OTHER"),
    ):
        onehot = M.failure_features(dataclasses.replace(view, result=result))
        assert onehot[M.FAILURES.index(bucket)] == 1.0 and onehot.sum() == 1.0
    assert M.failure_features(dataclasses.replace(view, result="SUCCESS")).sum() == 0.0
    blind = dataclasses.replace(view, probe=None, result="MISS")
    assert np.isnan(M.probe_scalars(blind)).all()
    key = EMB.key_probe(blind)
    assert np.all(np.isfinite(key)) and np.linalg.norm(key) == pytest.approx(1.0)


def test_live_attempts_embed_like_the_fixture_and_mid_attempt_like_at_the_end():
    """A calibration attempt run live gives the fixture's keys; the key_probe built
    from the recorder at PROBE_DONE equals the one from the finished episode."""
    i = next(
        i for i in FIRST if FIXTURE["condition"][i] == "naive" and FIXTURE["result"][i] == "BREAK"
    )
    seed, index = int(FIXTURE["seed"][i]), int(FIXTURE["hidden_situation"][i])
    situation = B.make_situations(seed, index + 1, OB.load_object_config(), start=index)[0]
    assert situation.key == FIXTURE["group"][i]
    env = B.make_env(B.load_benchmark_config())
    at_probe: list[np.ndarray] = []

    def on_escalation(esc):
        if esc.kind == PROBE_DONE:
            at_probe.append(EMB.key_probe(M.EpisodeView.from_recorder(env.recorder)))
            # The escalation's ProbeFeatures work too (T14/T15 hold those).
            mid = M.EpisodeView.from_recorder(env.recorder, probe=esc.probe)
            at_probe.append(EMB.key_probe(mid))

    r = run_attempt(
        env,
        GraspLiftSkill(),
        NAIVE,
        on_escalation,
        seed=B.env_seed(seed, index, 1),
        options={"object": situation.spec},
    )
    assert r.result == FIXTURE["result"][i]
    live = M.EpisodeView.from_episode(r.episode)
    for name, key in all_keys(live).items():
        cosine = float(key @ all_keys(VIEWS[i])[name])
        assert cosine > 0.9999, (name, cosine)  # the fixture stores float32 rows
    assert len(at_probe) == 2
    for key in at_probe:
        assert np.allclose(key, EMB.key_probe(live), atol=1e-12)
    # A second live run of the same attempt gives the same keys exactly.
    again = run_attempt(
        env,
        GraspLiftSkill(),
        NAIVE,
        seed=B.env_seed(seed, index, 1),
        options={"object": situation.spec},
    )
    for name, key in all_keys(M.EpisodeView.from_episode(again.episode)).items():
        assert np.array_equal(key, all_keys(live)[name]), name


def test_keys_never_read_privileged_data():
    """Scrambling the privileged columns, the hidden metadata, every event's info
    (BREAK's force, LIFTED's rise), and the outcome's ground-truth fields leaves
    every key unchanged."""
    env = B.make_env(B.load_benchmark_config())
    situation = next(
        s
        for s in B.make_situations(0, 12, OB.load_object_config())
        if s.class_name == "fragile_soft"
    )
    r = run_attempt(env, GraspLiftSkill(), NAIVE, seed=1, options={"object": situation.spec})
    ep = r.episode
    assert ep.events_named("BREAK")  # 8 N breaks this cube: an event with privileged info
    privileged = {
        name: (~ep.arrays[name] if ep.arrays[name].dtype == bool else ep.arrays[name] * 3.0 + 1.0)
        for name in PRIVILEGED_COLUMNS
    }
    scrambled = dataclasses.replace(
        ep,
        arrays={**ep.arrays, **privileged},
        meta={"probe": ep.meta["probe"]},  # no class, hidden properties, size, or seed
        events=tuple(dataclasses.replace(e, info={"force": -1.0}) for e in ep.events),
        outcome={**ep.outcome, "rise": -9.0, "peak_cube_force": -9.0},
    )
    a, b = M.EpisodeView.from_episode(ep), M.EpisodeView.from_episode(scrambled)
    for name in M.KEYS:
        assert np.array_equal(all_keys(a)[name], all_keys(b)[name]), name


def test_a_coarser_recording_decimates_to_the_same_view():
    """An env that records every 5th tick (100 Hz) gives the same 50 Hz view."""
    situation = B.make_situations(0, 1, OB.load_object_config())[0]
    views = []
    for stride in (1, 5):
        env = E.PandaVFAEnv(
            "lift",
            "osc",
            step_hz=500,
            auto_carry=False,
            record_stride=stride,
            sensor_config=B.make_env(B.load_benchmark_config()).sensor_config,
        )
        r = run_attempt(env, GraspLiftSkill(), NAIVE, seed=1, options={"object": situation.spec})
        views.append(M.EpisodeView.from_episode(r.episode))
    assert np.array_equal(views[0].t, views[1].t)
    for name in M.KEYS:
        assert np.array_equal(all_keys(views[0])[name], all_keys(views[1])[name]), name
    bad = {"t": np.arange(10) * 0.003, "phase": np.zeros(10, int)}
    with pytest.raises(ValueError, match="50 Hz"):
        M.EpisodeView.from_arrays(
            {
                **bad,
                "wrench": np.zeros((10, 6)),
                "finger_force": np.zeros((10, 2)),
                "ee_pos": np.zeros((10, 3)),
            },
            ("A",),
            {"cube": {"x": 0, "y": 0, "z": 0, "yaw": 0, "size": 0.05}},
        )
