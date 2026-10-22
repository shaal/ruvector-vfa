"""T05: the grip-force feasibility spike (``scripts/spike_grip.py``).

Pins the script's pieces (the F/T slip detector, the paired test, the sweep
plan, resumable chunks) and replays a few real attempts that carry the spike's
claims: too little grip drops the cube, enough grip holds it without creep, a
fast close overshoots the squeeze on a soft cube, the servo cannot squeeze past
25·width N, and the F/T detector agrees with the privileged ground truth.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from vfa.sim import objects as OB
from vfa.sim import scene as S

_SPEC = importlib.util.spec_from_file_location(
    "spike_grip", S.REPO_ROOT / "scripts" / "spike_grip.py"
)
SG = importlib.util.module_from_spec(_SPEC)
sys.modules["spike_grip"] = SG
_SPEC.loader.exec_module(SG)

G = 9.81


@pytest.fixture(scope="module")
def config() -> OB.ObjectConfig:
    return OB.load_object_config()


@pytest.fixture(scope="module")
def arm():
    return SG.Arm()


def _f_min(spec: OB.ObjectSpec) -> float:
    return spec.mass * G / (2.0 * spec.friction[0])


# ------------------------------------------------------------------ slip detector

HAND_KG = 0.76


def _feed(detector, cube_kg, hand_acc, cube_acc=None):
    """Drive the detector: a hand (HAND_KG) holding a cube, both accelerating.

    The wrist force is ``HAND_KG·(g + a_hand) + cube_kg·(g + a_cube)`` and the
    wrist accelerometer reads ``g + a_hand``. A slipping cube has ``a_cube < a_hand``.
    """
    hand_acc = np.asarray(hand_acc, float)
    cube_acc = hand_acc if cube_acc is None else np.asarray(cube_acc, float)
    cube_kg = np.broadcast_to(np.asarray(cube_kg, float), hand_acc.shape)
    detector.tare(HAND_KG * G, G)
    fired = []
    for k in range(hand_acc.size):
        f_z = HAND_KG * (G + hand_acc[k]) + cube_kg[k] * (G + cube_acc[k])
        fired.append(detector.update(k * SG.DT, f_z, G + hand_acc[k]))
    return np.array(fired)


def test_detector_reads_the_held_mass_through_any_lift_profile() -> None:
    """F_z / a_z minus the hand: the lightest cube (50 g) under a 0.58 m/s² lift never fires,
    though the hand's own inertial swing (0.44 N) is close to the cube's weight (0.49 N)."""
    t = np.arange(0, 2.0, SG.DT)
    acc = 0.58 * np.sin(2 * np.pi * t)
    det = SG.FTSlipDetector(SG.DT, G)
    assert not _feed(det, 0.05, acc).any()
    assert det.hand_mass == pytest.approx(HAND_KG)
    assert det.mass == pytest.approx(0.05, rel=1e-3)
    assert det.peak == pytest.approx(0.05 * G, rel=1e-3)


def test_detector_fires_on_a_drop() -> None:
    det = SG.FTSlipDetector(SG.DT, G)
    mass = np.r_[np.linspace(0, 0.3, 100), np.full(200, 0.3), np.zeros(200)]
    fired = _feed(det, mass, np.zeros(mass.size))
    assert not fired[:300].any()
    first = int(np.argmax(fired))
    window = round(SG.DEBOUNCE_S / SG.DT)
    assert 300 + window - 1 <= first < 300 + window + 4  # filter + 10 ms debounce
    assert det.fired_at == pytest.approx(first * SG.DT)
    assert fired[first:].all()  # latched
    assert det.min_ratio is not None and det.min_ratio < SG.DROP_FRACTION


def test_detector_fires_on_a_partial_slip_without_a_load_drop() -> None:
    """A cube sliding at 1.5 m/s² relative to the hand loses 15 % of its apparent mass:
    the load-drop rule (50 %) of the first draft missed exactly these."""
    t = np.arange(0, 1.0, SG.DT)
    hand = np.full(t.size, 0.3)
    cube = np.where(t < 0.5, 0.3, 0.3 - 1.5)
    det = SG.FTSlipDetector(SG.DT, G)
    fired = _feed(det, 0.4, hand, cube)
    assert not fired[t < 0.5].any() and fired[-1]
    assert 0.5 + SG.DEBOUNCE_S - SG.DT <= det.fired_at < 0.53


def test_detector_needs_an_armed_peak_and_a_valid_acceleration() -> None:
    det = SG.FTSlipDetector(SG.DT, G)
    light = 0.9 * SG.ARM_LOAD_N / G
    assert not _feed(det, np.r_[np.full(200, light), np.zeros(200)], np.zeros(400)).any()
    assert det.min_ratio is None  # never armed
    det = SG.FTSlipDetector(SG.DT, G)
    det.tare(HAND_KG * G, G)
    for f_z, a_z in ((1.0, 0.0), (1.0, math.nan), (1.0, math.inf), (math.nan, G), (1.0, 1e9)):
        assert det.update(0.0, f_z, a_z) is False  # skipped: no estimate, no crash
    assert det.mass == 0.0
    for f_z, a_z in ((7.0, 0.0), (7.0, math.inf), (math.nan, G), (7.0, math.nan)):
        with pytest.raises(ValueError, match="at rest"):
            det.tare(f_z, a_z)


def test_detector_ignores_a_one_tick_dip_and_survives_bad_samples() -> None:
    """Round-2 review: 4 of 13 false alarms were 2-4 ms dips; a NaN sample blinded it."""
    hand = np.zeros(600)
    mass = np.full(600, 0.3)
    mass[300] = 0.15  # a one-tick (2 ms) glitch to half the mass
    det = SG.FTSlipDetector(SG.DT, G)
    assert not _feed(det, mass, hand).any()  # the debounce ignores it...
    assert det.min_ratio_by_window[1] < SG.DROP_FRACTION  # ...a 1-tick rule would fire
    assert det.min_ratio >= SG.DROP_FRACTION
    det = SG.FTSlipDetector(SG.DT, G)
    det.tare(HAND_KG * G, G)
    for k in range(300):
        det.update(k * SG.DT, (HAND_KG + 0.3) * G, G)
    det.update(0.6, math.nan, G)  # a bad sample is skipped...
    det.update(0.602, (HAND_KG + 0.3) * G, math.inf)
    for k in range(302, 400):  # ...and a real drop afterwards still fires
        det.update(k * SG.DT, HAND_KG * G, G)
    assert det.fired_at is not None and math.isfinite(det.mass)


def test_min_ratio_scores_every_threshold_exactly() -> None:
    """The exact sweep: the detector fires at a threshold iff min_ratio is below it."""
    t = np.arange(0, 1.0, SG.DT)
    for slide in (0.5, 1.0, 1.5, 3.0):
        det = SG.FTSlipDetector(SG.DT, G)
        fired = _feed(det, 0.4, np.full(t.size, 0.3), np.where(t < 0.5, 0.3, 0.3 - slide))
        assert bool(fired[-1]) == (det.min_ratio < SG.DROP_FRACTION), (slide, det.min_ratio)


def test_low_pass_has_the_documented_cutoff() -> None:
    lp = SG.LowPass(20.0, SG.DT)
    assert lp.alpha == pytest.approx(1 - math.exp(-2 * math.pi * 20 * SG.DT))
    assert lp(1.0) == 1.0  # the first sample initializes
    assert lp(0.0) == pytest.approx(1 - lp.alpha)


# -------------------------------------------------------------- stats and plan


def test_mcnemar_one_sided() -> None:
    assert SG.mcnemar_one_sided(0, 0) == 1.0
    assert SG.mcnemar_one_sided(10, 0) == pytest.approx(2.0**-10)
    assert SG.mcnemar_one_sided(21, 0) == pytest.approx(2.0**-21)
    assert SG.mcnemar_one_sided(0, 5) == 1.0
    assert SG.mcnemar_one_sided(3, 3) == pytest.approx(42 / 64)  # P(X >= 3), X ~ B(6, 1/2)


@pytest.mark.parametrize(("profile", "attempts"), [("QUICK", 557), ("FULL", 2585)])
def test_plan_sizes_and_unique_chunk_files(config, profile, attempts) -> None:
    chunks = SG.plan(getattr(SG, profile), config)
    assert sum(len(c.jobs) for c in chunks) == attempts
    assert len({c.name for c in chunks}) == len(chunks)
    assert all(c.jobs for c in chunks)
    # Every map cell has at least 5 samples (ROADMAP T05).
    cells: dict = {}
    for c in chunks:
        if c.study == "map":
            for cls, _, grip, speed, _ in c.jobs:
                cells[(cls, grip, speed)] = cells.get((cls, grip, speed), 0) + 1
    assert min(cells.values()) >= 5
    assert {k[0] for k in cells} == set(SG.spike_classes(config))


def test_the_full_held_out_cubes_were_never_used_by_another_study(config) -> None:
    """T05 review round 3: the quick profile's noise study ran the first held-out cube
    (sample 9000), and the debounce was chosen on the quick profile."""

    def samples(profile: dict, held_out: bool) -> set:
        return {
            (cls, k)
            for c in SG.plan(profile, config)
            if (c.study == "noise") == held_out
            for cls, k, *_ in c.jobs
        }

    full_held_out = samples(SG.FULL, True)
    assert len(full_held_out) == 5 * SG.NOISE_SAMPLES
    assert {k for _, k in full_held_out} == set(range(9100, 9100 + SG.NOISE_SAMPLES))
    for profile in (SG.QUICK, SG.FULL):
        assert not full_held_out & samples(profile, False)
    assert not full_held_out & samples(SG.QUICK, True)
    assert not full_held_out & {(c, k) for c in SG.spike_classes(config) for k in range(9000, 9003)}


def test_wilson_interval() -> None:
    lo, hi = SG.wilson(33, 36)
    assert (round(lo, 3), round(hi, 3)) == (0.782, 0.971)
    assert SG.wilson(0, 10)[0] == 0.0 and SG.wilson(0, 5)[0] == 0.0  # clamped, not -2.8e-17
    assert SG.wilson(10, 10)[1] == 1.0
    assert all(math.isnan(x) for x in SG.wilson(0, 0))


def test_samples_are_paired_across_cells(config) -> None:
    a = SG.sample_spec(config, "fragile_soft", 3)
    assert a == SG.sample_spec(config, "fragile_soft", 3)
    assert a != SG.sample_spec(config, "fragile_soft", 4)
    assert a != SG.sample_spec(config, "light_rigid", 3)


def test_harness_grips_follow_the_need_and_stay_below_the_break_force(config) -> None:
    for k in range(20):
        spec = SG.sample_spec(config, "fragile_soft", k)
        need = _f_min(spec)
        grip, (when, low) = SG.harness_grip(spec, "release", 0.5)
        assert 2 * grip <= 0.8 * spec.break_force + 1e-12 and grip > need
        assert when == SG.RELEASE_AT and low == pytest.approx(0.5 * need)
        grip, release = SG.harness_grip(spec, "margin", 1.02)
        assert release is None
        assert grip == pytest.approx(1.02 * need * (G + SG.LIFT_PEAK_ACC) / G)
    assert SG.LIFT_PEAK_ACC == pytest.approx(0.577, abs=1e-3)
    with pytest.raises(ValueError, match="variant"):
        SG.harness_grip(spec, "squash", 1.0)


def test_sweep_is_chunked_resumable_and_aggregates(tmp_path, config) -> None:
    profile = {
        "forces": (2.0,),
        "speeds": (0.5,),
        "samples": 1,
        "speed_test_forces": (),
        "speed_test_samples": 0,
        "cal_forces": (),
        "cal_samples": 0,
        "release_samples": 0,
        "margin_samples": 0,
        "noise_samples": 0,
        "noise_first": SG.NOISE_FIRST_QUICK,
    }
    chunks = {c.name: c for c in SG.plan(profile, config)}

    def chunk_of(path):
        return chunks[path.stem]

    assert not SG.run_sweep(tmp_path, profile, workers=1, deadline_s=0.0)  # nothing ran
    assert SG.run_sweep(tmp_path, profile, workers=1, deadline_s=60.0)
    files = sorted((tmp_path / "chunks").glob("*.json"))
    assert [f.stem for f in files] == sorted(f"map__{c}__v0.5" for c in SG.spike_classes(config))
    stamps = {f: f.stat().st_mtime_ns for f in files}
    assert SG.run_sweep(tmp_path, profile, workers=1, deadline_s=60.0)  # all done: skips
    assert {f: f.stat().st_mtime_ns for f in files} == stamps
    rows = SG.load_results(tmp_path, profile)
    assert len(rows) == 5 and {r["study"] for r in rows} == {"map"}
    assert not list((tmp_path / "chunks").glob("*.tmp"))
    text, metrics = SG.summarize(rows, "test")
    assert "No single grip force succeeds" in text
    assert metrics["attempts"] == 5
    json.dumps(metrics, default=float)
    assert SG.aggregate(tmp_path, "test", profile) == 0
    strict = json.loads((tmp_path / "metrics.json").read_text())  # no NaN in the file
    assert strict["slip_detector"]["by_study"]["map"]["n"] == 5

    # A chunk from other code (fingerprint) or another plan (jobs) is stale: the
    # aggregation refuses it and the next sweep re-runs only that chunk.
    stale = files[0]
    record = json.loads(stale.read_text())
    record["fingerprint"] = "other code"
    stale.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=stale.stem):
        SG.load_results(tmp_path, profile)
    assert SG.run_sweep(tmp_path, profile, workers=1, deadline_s=60.0)
    assert stale.stat().st_mtime_ns != stamps[stale]
    assert all(f.stat().st_mtime_ns == stamps[f] for f in files[1:])
    bigger = {**profile, "forces": (2.0, 3.0)}
    with pytest.raises(ValueError, match="5 chunks missing or stale"):
        SG.load_results(tmp_path, bigger)

    # Hostile chunk files are stale, never crashes: JSON that is not an object, a
    # record with a missing row, a leftover .tmp from a killed write.
    for bad in ("null", "[]", "{}"):
        stale.write_text(bad)
        assert not SG.chunk_is_current(stale, chunk_of(stale), "x")
    record["fingerprint"] = SG.code_fingerprint()
    good_rows = record["results"]
    for rows in ([5], [{"cls": "x"}], []):  # a corrupt row, a partial row, a missing row
        record["results"] = rows
        stale.write_text(json.dumps(record))
        assert not SG.chunk_is_current(stale, chunk_of(stale), record["fingerprint"])
    record["results"] = good_rows
    stale.write_text(json.dumps(record))
    assert SG.chunk_is_current(stale, chunk_of(stale), record["fingerprint"])
    record["results"] = []
    stale.write_text(json.dumps(record))
    leftover = stale.with_suffix(".tmp")
    leftover.write_text("half")
    with pytest.raises(ValueError, match=stale.stem):
        SG.load_results(tmp_path, profile)
    assert SG.run_sweep(tmp_path, profile, workers=1, deadline_s=60.0)
    assert not leftover.exists()
    assert len(SG.load_results(tmp_path, profile)) == 5


def test_a_run_directory_belongs_to_one_profile(tmp_path) -> None:
    SG.claim_profile(tmp_path, "full")
    SG.claim_profile(tmp_path, "full")  # the same profile resumes
    with pytest.raises(ValueError, match="'full' run"):
        SG.claim_profile(tmp_path, "quick")
    assert SG.main(["--quick", "--aggregate", "--out", str(tmp_path)]) == 2
    (tmp_path / "profile.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="unreadable"):
        SG.claim_profile(tmp_path, "full")


@pytest.mark.parametrize("args", [["--workers", "0"], ["--deadline", "nan"], ["--deadline", "-1"]])
def test_the_cli_rejects_bad_arguments(tmp_path, args) -> None:
    assert SG.main(["--quick", "--out", str(tmp_path / "run"), *args]) == 2


# --------------------------------------------------------------- real attempts


def test_grip_below_the_need_drops_and_above_it_holds_without_creep(config, arm) -> None:
    """Elliptic cones + impratio 10: the F_min = m·g/(2μ) threshold is sharp."""
    spec = SG.sample_spec(config, "heavy_slippery", 0)
    f_min = _f_min(spec)
    low = SG.run_attempt(arm, spec, 0.9 * f_min, 0.2)
    assert low.outcome == "DROP" and low.gt_slip and low.ft_slip
    assert low.rel_drop > 0.05
    high = SG.run_attempt(arm, spec, 1.3 * f_min, 0.2)
    assert high.outcome == "SUCCESS" and not high.gt_slip and not high.ft_slip
    assert high.rel_drop < 0.001  # held, no creep
    assert high.rise > SG.SUCCESS_RISE_M
    assert high.ft_peak_load == pytest.approx(spec.mass * G, rel=0.1)  # the F/T sees the cube


def test_fast_closing_overshoots_the_squeeze_on_a_soft_cube(config, arm) -> None:
    spec = replace(SG.sample_spec(config, "fragile_soft", 0), break_force=None)
    slow = SG.run_attempt(arm, spec, 3.0, 0.02, keep_trace=False)
    fast = SG.run_attempt(arm, spec, 3.0, 0.5, keep_trace=False)
    assert slow.peak_force / 6.0 == pytest.approx(1.0, abs=0.01)
    assert fast.peak_force / 6.0 > 1.08
    assert slow.success and fast.success


def test_a_fragile_cube_breaks_past_its_threshold(config, arm) -> None:
    spec = SG.sample_spec(config, "fragile_soft", 0)
    result = SG.run_attempt(arm, spec, 0.6 * spec.break_force, 0.05, keep_trace=False)
    assert result.outcome == "BREAK" and not result.success
    assert spec.break_force < result.peak_force < 1.3 * spec.break_force
    assert result.break_time is not None and result.break_time > 0  # during the squeeze


def test_the_servo_cannot_squeeze_past_its_cap_but_the_motor_can(config) -> None:
    spec = replace(SG.sample_spec(config, "heavy_rigid", 0), break_force=None)
    servo = SG.run_attempt(SG.Arm(SG.S.SceneConfig(gripper="servo")), spec, 8.0, 0.1)
    motor = SG.run_attempt(SG.Arm(), spec, 8.0, 0.1)
    assert servo.settled_force / 2 == pytest.approx(25.0 * spec.size, rel=0.01)
    assert motor.settled_force / 2 == pytest.approx(8.0, rel=1e-3)
    assert servo.outcome == "DROP" and motor.outcome == "SUCCESS"


def test_a_mid_air_release_is_caught_by_the_ft_detector_first(config, arm) -> None:
    spec = SG.sample_spec(config, "heavy_rigid", 1)
    grip, release = SG.harness_grip(spec, "release", 0.5)
    result = SG.run_attempt(arm, spec, grip, SG.HARNESS_SPEED, release=release)
    assert result.gt_slip and result.ft_slip and result.outcome == "DROP"
    assert result.gt_slip_time > release[0]
    assert result.ft_slip_time <= result.gt_slip_time + 0.02


def test_a_partial_slip_at_lift_off_is_detected(config, arm) -> None:
    """At exactly the grip the lift needs, the cube slides part way at the peak
    acceleration (12 mm here) and is still lifted; 6 % more grip holds it."""
    spec = SG.sample_spec(config, "heavy_slippery", 3001)
    grip, _ = SG.harness_grip(spec, "margin", 1.0)
    result = SG.run_attempt(arm, spec, grip, SG.HARNESS_SPEED)
    assert result.gt_slip and SG.SLIP_GT_M < result.rel_drop < 0.05, result.rel_drop
    assert result.success  # still lifted
    assert result.ft_slip
    held = SG.run_attempt(arm, spec, SG.harness_grip(spec, "margin", 1.06)[0], SG.HARNESS_SPEED)
    assert not held.gt_slip and not held.ft_slip and held.rel_drop < 0.001


def test_elliptic_cones_remove_the_creep_of_a_held_cube(config) -> None:
    """D11: the pyramidal default lets a cube held at 1.1x its need slide millimeters."""
    spec = SG.sample_spec(config, "heavy_rigid", 0)
    grip = 1.1 * _f_min(spec) * (G + SG.LIFT_PEAK_ACC) / G
    elliptic = SG.run_attempt(SG.Arm(), spec, grip, 0.2, keep_trace=False)
    pyramidal_arm = SG.Arm()
    pyramidal_arm.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    pyramidal_arm.model.opt.impratio = 1.0
    pyramidal = SG.run_attempt(pyramidal_arm, spec, grip, 0.2, keep_trace=False)
    assert elliptic.rel_drop < 0.001 and elliptic.success
    assert pyramidal.rel_drop > 0.002, pyramidal.rel_drop


def test_the_descent_needs_the_path_velocity_fed_forward(config, arm) -> None:
    """Without vel= the coupled damping pushes the ee sideways during a vertical move."""
    spec = SG.sample_spec(config, "light_rigid", 0)
    drift = {}
    for feed_forward in (True, False):
        osc, gripper = arm.reset(spec)
        seen = OB.observe_cube(arm.model, arm.data, ids=arm.ids)
        cube = np.array([seen.x, seen.y, seen.z])
        above = cube + (0.0, 0.0, SG.APPROACH_HEIGHT)
        osc.set_target(above, yaw=seen.yaw)
        for _ in range(round(SG.APPROACH_S / SG.DT)):
            SG.control_step(arm.model, arm.data, (osc, gripper), 1)
        worst = 0.0
        for k in range(round((SG.DESCEND_S + 0.3) / SG.DT)):
            if feed_forward:
                SG.path_target(osc, above, cube, k * SG.DT, SG.DESCEND_S, seen.yaw)
            else:
                s = SG.min_jerk(k * SG.DT / SG.DESCEND_S)
                osc.set_target(above + s * (cube - above), yaw=seen.yaw)
            SG.control_step(arm.model, arm.data, (osc, gripper), 1)
            ee = arm.data.site_xpos[arm.ids.ee_site]
            worst = max(worst, float(np.hypot(*(ee[:2] - cube[:2]))))
        drift[feed_forward] = worst
    assert drift[True] < 0.003, drift
    assert drift[False] > 0.005, drift
