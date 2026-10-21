"""T06: sensors, the slip detector, and the event detector (``vfa.sim.sensors``).

Synthetic frames pin each event rule; real physics (the scripted grasp in
``scripts/record_episode.py``) pins the ROADMAP acceptance: ``CONTACT_ONSET``
and ``GRASP_CLOSED`` on a grasp, ``DROP`` when the gripper opens mid-lift.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys

import mujoco
import numpy as np
import pytest

from vfa.control.gripper import Gripper
from vfa.control.osc import OperationalSpaceController, control_step
from vfa.sim import objects as OB
from vfa.sim import scene as S
from vfa.sim.sensors import (
    REALISTIC_NOISE,
    T05_SLIP,
    Event,
    EventConfig,
    EventDetector,
    LowPass,
    PrivilegedState,
    SensorConfig,
    SensorFrame,
    SensorNoise,
    SensorSuite,
    SlipConfig,
    SlipDetector,
    read_privileged,
)

_SPEC = importlib.util.spec_from_file_location(
    "record_episode", S.REPO_ROOT / "scripts" / "record_episode.py"
)
RE = importlib.util.module_from_spec(_SPEC)
sys.modules["record_episode"] = RE
_SPEC.loader.exec_module(RE)

G = 9.81
DT = 0.002
HAND_KG = 0.76


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return S.build_scene()


@pytest.fixture(scope="module")
def ids(model) -> S.SceneIds:
    return S.scene_ids(model)


# ------------------------------------------------------------------ config


def test_noise_and_config_validation() -> None:
    assert not SensorNoise().active and REALISTIC_NOISE.active
    for bad in (-0.1, math.nan, math.inf, True, "0.1"):
        with pytest.raises(ValueError):
            SensorNoise(force=bad)
    with pytest.raises(ValueError, match="hysteresis"):
        SensorConfig(contact_on_n=0.1, contact_off_n=0.2)
    for bad in (0.0, 1.0, 1.2):
        with pytest.raises(ValueError):
            SlipConfig(drop_fraction=bad)
    with pytest.raises(ValueError):
        SlipConfig(noise_k=-1.0)
    with pytest.raises(TypeError):
        SensorConfig(noise=(0.05, 0.0, 0.0, 0.0))
    with pytest.raises(TypeError):
        EventConfig(release_phases="PLACE")
    # A generator is read once, not used up by the check (review: it became empty).
    assert EventConfig(release_phases=(p for p in ["OPEN"])).release_phases == {"OPEN"}
    with pytest.raises(ValueError):
        EventConfig(drop_fraction=1.0)


def test_slip_defaults_are_t05_plus_the_sized_noise_margin(model, ids) -> None:
    """FEASIBILITY §8: T05's detector plus noise_k = 4 (sized on the tune cubes)."""
    clean = SlipConfig()
    assert (clean.cutoff_hz, clean.drop_fraction, clean.debounce_s, clean.arm_load_n) == (
        T05_SLIP.cutoff_hz,
        T05_SLIP.drop_fraction,
        T05_SLIP.debounce_s,
        T05_SLIP.arm_load_n,
    )
    assert clean.noise_k == 4.0 and T05_SLIP.noise_k == 0.0 and clean.peak_window_s == 0.5
    assert SensorConfig().slip == SensorConfig(noise=REALISTIC_NOISE).slip == clean
    # The suite hands the detector the wrist force and accelerometer noise only.
    noisy = SensorConfig(noise=REALISTIC_NOISE)
    suite = SensorSuite(model, mujoco.MjData(model), ids, noisy, np.random.default_rng(0))
    assert suite.slip.noise == (REALISTIC_NOISE.force, REALISTIC_NOISE.accel)
    touch_only = SensorConfig(noise=SensorNoise(touch=0.05, torque=0.01))
    suite = SensorSuite(model, mujoco.MjData(model), ids, touch_only, np.random.default_rng(0))
    assert suite.slip.noise == (0.0, 0.0) and suite.slip.sigma() == 0.0


def test_noise_free_detector_is_bit_identical_to_t05() -> None:
    """With noise = (0, 0) the margin vanishes: SlipConfig() and T05_SLIP agree exactly."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = 600
        mass = np.r_[np.full(n // 2, rng.uniform(0.05, 0.5)), rng.uniform(0, 0.5, n - n // 2)]
        acc = rng.normal(0.0, 0.5, n)
        a, b = SlipDetector(DT, G, SlipConfig()), SlipDetector(DT, G, T05_SLIP)
        fa, fb = _feed(a, mass, acc), _feed(b, mass, acc)
        assert (fa == fb).all() and a.min_ratio == b.min_ratio and a.fired_at == b.fired_at


def test_low_pass_on_floats_and_arrays() -> None:
    lp = LowPass(20.0, DT)
    assert lp.alpha == pytest.approx(1 - math.exp(-2 * math.pi * 20 * DT))
    assert lp(1.0) == 1.0 and lp(0.0) == pytest.approx(1 - lp.alpha)
    va = LowPass(20.0, DT)
    x = np.array([1.0, 2.0])
    first = va(x)
    x[:] = 0.0  # the filter keeps its own copy
    np.testing.assert_array_equal(first, [1.0, 2.0])
    np.testing.assert_allclose(va(np.zeros(2)), [1 - lp.alpha, 2 * (1 - lp.alpha)])
    va.reset()
    np.testing.assert_array_equal(va(np.ones(2)), [1.0, 1.0])


# ----------------------------------------------------------- slip detector


def _feed(det: SlipDetector, cube_kg, hand_acc=None, cube_acc=None, rng=None, sigma=(0.0, 0.0)):
    """Drive a detector: a hand (HAND_KG) holding a cube; optional sensor noise."""
    cube_kg = np.asarray(cube_kg, float)
    hand_acc = np.zeros(cube_kg.size) if hand_acc is None else np.asarray(hand_acc, float)
    cube_acc = hand_acc if cube_acc is None else np.asarray(cube_acc, float)
    det.tare(HAND_KG * G, G)
    fired = []
    for k in range(cube_kg.size):
        f_z = HAND_KG * (G + hand_acc[k]) + cube_kg[k] * (G + cube_acc[k])
        a_z = G + hand_acc[k]
        if rng is not None:
            f_z += sigma[0] * rng.standard_normal()
            a_z += sigma[1] * rng.standard_normal()
        fired.append(det.update(k * DT, f_z, a_z))
    return np.array(fired)


def test_watching_off_estimates_the_load_but_never_arms_or_fires() -> None:
    """The squeeze can load the wrist for a moment (0.23 N on heavy_slippery); an
    armed detector read that as a slip and a drop before the lift."""
    det = SlipDetector(DT, G, watching=False)
    mass = np.r_[np.full(100, 0.03), np.zeros(200)]  # a 0.29 N transient, then gone
    assert not _feed(det, mass).any()
    assert det.peak_mass == 0.0 and not det.armed and det.min_ratio is None
    assert det.load == pytest.approx(0.0, abs=1e-6)
    det.watch()  # the lift starts: a held 0.3 kg cube now arms it
    for k in range(300):
        det.update(1.0 + k * DT, (HAND_KG + 0.3) * G, G)
    assert det.armed and det.peak == pytest.approx(0.3 * G, rel=1e-3)
    for k in range(100):
        det.update(2.0 + k * DT, HAND_KG * G, G)
    assert det.fired_at is not None


@pytest.mark.parametrize("noise", [(0.0, 0.0), (REALISTIC_NOISE.force, REALISTIC_NOISE.accel)])
def test_rearm_clears_the_latch_and_needs_a_further_drop(noise) -> None:
    """Review: with the old slow-peak design a rearm re-fired every 10 ms on a steady
    load. The peak now restarts at the fast estimate, so a steady load stays quiet."""
    det = SlipDetector(DT, G, SlipConfig(), noise=noise)
    mass = np.r_[np.full(300, 0.4), np.full(300, 0.3)]  # a 25 % drop, then steady
    fired = _feed(det, mass)
    assert fired[-1] and det.fired_at is not None
    det.rearm()
    assert det.fired_at is None and det.peak == pytest.approx(0.3 * G, rel=1e-6)
    for k in range(300):  # the same level: no new slip
        assert not det.update(2.0 + k * DT, (HAND_KG + 0.3) * G, G)
    for k in range(300):  # a further 20 % drop fires again
        det.update(3.0 + k * DT, (HAND_KG + 0.24) * G, G)
    assert det.fired_at is not None and det.fired_at >= 3.0


def _lift(
    config: SlipConfig,
    cube_kg: np.ndarray,
    seed: int,
    sigma: tuple[float, float],
    dt: float = DT,
    noise_spec: tuple[float, float] | None = None,
) -> SlipDetector:
    """As ``SensorSuite`` runs it: 0.3 s empty-handed (squeeze, not watching), then
    ``watch()`` and the load ``cube_kg`` (lift and hold), with white noise ``sigma``
    on both inputs. The detector knows ``noise_spec`` (default: ``sigma``)."""
    spec = sigma if noise_spec is None else noise_spec
    det = SlipDetector(dt, G, config, watching=False, noise=spec)
    rng = np.random.default_rng(seed)
    det.tare(HAND_KG * G, G)
    for k in range(round(0.3 / dt)):
        f_z = HAND_KG * G + sigma[0] * rng.standard_normal()
        det.update(k * dt, f_z, G + sigma[1] * rng.standard_normal())
    det.watch()
    for k, m in enumerate(cube_kg):
        f_z = (HAND_KG + m) * G + sigma[0] * rng.standard_normal()
        det.update(0.3 + k * dt, f_z, G + sigma[1] * rng.standard_normal())
    return det


REALISTIC_IN = (REALISTIC_NOISE.force, REALISTIC_NOISE.accel)


@pytest.mark.parametrize("dt", [0.002, 0.004])
def test_noise_margin_stops_false_alarms_and_still_catches_a_slip(dt) -> None:
    """Cubes held 2 s under REALISTIC_NOISE, 20 seeds each, at 500 and 250 Hz. T05's
    detector false-alarms on 50-100 g cubes; with the 4σ margin (σ from the noise spec
    at this tick rate) it stays quiet on 100 g and 150 g and nearly so on 50 g, and still
    catches a 1/3 loss within 50 ms (FEASIBILITY §8)."""
    n = round(2.0 / dt)

    def alarms(config: SlipConfig, kg: float) -> int:
        held = np.full(n, kg)
        return sum(_lift(config, held, s, REALISTIC_IN, dt).fired_at is not None for s in range(20))

    assert alarms(T05_SLIP, 0.1) >= 10 and alarms(T05_SLIP, 0.05) >= 15
    assert alarms(SlipConfig(), 0.1) == 0 and alarms(SlipConfig(), 0.15) == 0
    assert alarms(SlipConfig(), 0.05) <= 1
    for kg in (0.05, 0.3):
        for seed in range(5):
            mass = np.r_[np.full(n, kg), np.full(round(0.3 / dt), kg * 2 / 3)]
            det = _lift(SlipConfig(), mass, seed, REALISTIC_IN, dt)
            assert det.fired_at is not None and 0.3 + n * dt <= det.fired_at < 0.35 + n * dt


def test_the_margin_uses_the_spec_not_the_signal() -> None:
    """Without a noise spec the margin is 0 even if the inputs are noisy (T05 behaviour);
    a spec on noise-free inputs only widens the band (no false alarm, a later fire)."""
    held = np.full(1000, 0.05)
    blind = sum(
        _lift(SlipConfig(), held, s, REALISTIC_IN, noise_spec=(0.0, 0.0)).fired_at is not None
        for s in range(10)
    )
    assert blind >= 8
    det = _lift(SlipConfig(), np.r_[np.full(500, 0.3), np.full(200, 0.25)], 0, (0.0, 0.0),
                noise_spec=REALISTIC_IN)  # fmt: skip
    assert det.fired_at is not None  # a 17 % loss of 0.3 kg clears 10 % + 4σ (≈ 3 %)
    assert det.sigma() == pytest.approx(
        math.hypot(0.05, (HAND_KG + det.mass) * 0.03) * LowPass(20.0, DT).noise_gain()
    )


def test_rearm_while_unwatched_does_not_arm() -> None:
    """Review: rearm() before watch() armed the detector (and so DROP)."""
    det = SlipDetector(DT, G, watching=False)
    _feed(det, np.full(100, 0.3))
    det.rearm()
    assert not det.armed and det.fired_at is None
    det.watch()
    assert det.armed  # watching, and the 0.3 kg peak passed 0.15 N


def test_a_long_carry_keeps_false_alarms_to_the_lightest_cubes() -> None:
    """Round-2 review: with an all-time peak the false alarms of a held 50 g cube grow
    with the carry (the maximum of the noise keeps rising). The 0.5 s peak window halves
    them; what is left is the noise floor of the lightest cubes (FEASIBILITY §8)."""
    held = np.full(round(30.0 / DT), 0.05)

    def alarms(config: SlipConfig, cube: np.ndarray) -> int:
        return sum(_lift(config, cube, s, REALISTIC_IN).fired_at is not None for s in range(40))

    windowed = alarms(SlipConfig(), held)
    unbounded = alarms(SlipConfig(peak_window_s=1e6), held)  # the earlier design's peak
    assert windowed <= 12 and unbounded >= 2 * windowed, (windowed, unbounded)
    assert alarms(SlipConfig(), np.full(round(30.0 / DT), 0.075)) == 0


def test_debounce_is_at_least_two_ticks() -> None:
    """Below 167 Hz T05's 6 ms rounds to one tick; one noisy sample is not a drop."""
    for dt, ticks in ((0.002, 3), (0.004, 2), (0.008, 2), (0.01, 2)):
        assert SlipDetector(dt)._window == ticks
    # A peak window that cannot outlast the debounce would never fire (round-3 review).
    with pytest.raises(ValueError, match="peak_window_s"):
        SlipDetector(0.01, config=SlipConfig(peak_window_s=0.01), noise=REALISTIC_IN)


def test_unwatch_stops_the_detector_before_a_set_down() -> None:
    det = SlipDetector(DT, G)
    _feed(det, np.full(300, 0.3))
    det.unwatch()
    for k in range(300):  # the table takes the load
        det.update(1.0 + k * DT, HAND_KG * G, G)
    assert det.fired_at is None and not det.armed and det.load < 0.05


def test_detector_validation() -> None:
    with pytest.raises(ValueError):
        SlipDetector(0.0)
    with pytest.raises(ValueError):
        SlipDetector(DT, track_windows=(0,))
    det = SlipDetector(DT)
    for f_z, a_z in ((7.0, 0.0), (math.nan, G)):
        with pytest.raises(ValueError, match="at rest"):
            det.tare(f_z, a_z)


# ------------------------------------------------------------ event rules


def _frame(t, contact=(False, False), *, force=(0.0, 0.0), width=0.05, rate=0.0, load=math.nan,
           peak=math.nan, armed=False, watching=False, watch_id=0,
           slip_time=None) -> SensorFrame:  # fmt: skip
    return SensorFrame(
        t=t,
        wrench_raw=np.zeros(6),
        wrench=np.zeros(6),
        acc=np.array([0.0, 0.0, G]),
        finger_force=np.array(force, float),
        contact=np.array(contact, bool),
        width=width,
        width_rate=rate,
        ee_pos=np.zeros(3),
        ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ee_vel=np.zeros(6),
        load=load,
        peak_load=peak,
        armed=armed,
        watching=watching,
        watch_id=watch_id,
        slip=slip_time is not None,
        slip_time=slip_time,
    )


def _names(events) -> list[str]:
    return [e.name.value for e in events]


def test_contact_onset_once_per_grasp_and_grasp_closed_after_settling() -> None:
    det = EventDetector()
    out = []
    t = 0.0
    for _ in range(10):  # one pad touches while the fingers still close
        out += det.update(_frame(t, (True, False), rate=-0.1))
        t += DT
    for _ in range(3):  # a 6 ms bounce off: not a new grasp
        out += det.update(_frame(t, (False, False), rate=-0.1))
        t += DT
    for _ in range(4):  # both pads, but still closing
        out += det.update(_frame(t, (True, True), rate=-0.01))
        t += DT
    assert _names(out) == ["CONTACT_ONSET"] and out[0].t == 0.0
    start = t
    for _ in range(10):  # both pads, stopped (|rate| < 2 mm/s)
        out += det.update(_frame(t, (True, True), force=(2.0, 2.0), rate=0.001))
        t += DT
    assert _names(out) == ["CONTACT_ONSET", "GRASP_CLOSED"]
    assert out[1].t == pytest.approx(start + 0.01)  # held for grasp_settle_s
    assert out[1].info["force"] == 4.0
    for _ in range(60):  # released for 0.12 s >= rearm_s: the grasp ends
        out += det.update(_frame(t, (False, False)))
        t += DT
    assert not det.in_grasp
    out += det.update(_frame(t, (False, True)))
    assert _names(out)[-1] == "CONTACT_ONSET"


def _held(t, slip_time, watch_id=1):
    """A frame of a carry (the watch began after the grasp's onset, watch_id 0)."""
    return _frame(t, (True, True), load=1.0, peak=1.2, armed=True, watching=True,
                  watch_id=watch_id, slip_time=slip_time)  # fmt: skip


def test_slip_event_carries_the_detector_time_and_repeats_after_a_rearm() -> None:
    det = EventDetector()
    out = det.update(_frame(0.9, (True, True)))  # the grasp's onset, before the watch
    out += det.update(_held(1.0, 0.996))
    out += det.update(_held(1.002, 0.996))
    slips = [e for e in out if e.name is Event.SLIP]
    assert len(slips) == 1 and slips[0].t == 0.996
    out += det.update(_held(1.5, 1.498))
    assert _names(out)[-1] == "SLIP" and out[-1].t == 1.498


def test_a_watch_on_at_contact_onset_is_logged(caplog) -> None:
    det = EventDetector()
    with caplog.at_level(logging.WARNING, logger="vfa.sim.sensors"):
        det.update(_frame(0.0, (True, False), watching=True, watch_id=1))
    assert "while the slip watch is on" in caplog.text


def test_a_stale_watch_from_an_earlier_grasp_raises_no_slip_or_drop() -> None:
    """Round-3 review: a re-grasp with the last carry's watch still on reported a false
    DROP (and SLIP) in its squeeze. A carry counts only if its watch_slip() came after
    the grasp's CONTACT_ONSET."""
    det = EventDetector()
    out = det.update(_frame(0.0, (True, True)))  # grasp 1
    for k in range(1, 6):  # carry 1 (watch_id 1)
        out += det.update(_frame(k * DT, (True, True), load=1.0, watching=True, watch_id=1))
    t = 6 * DT
    for _ in range(60):  # the cube falls, contact lost: the grasp ends, the watch stays on
        out += det.update(_frame(t, load=0.0, watching=True, watch_id=1))
        t += DT
    assert _names(out).count("DROP") == 1
    out += det.update(_frame(t, (True, True), load=0.3, watching=True, watch_id=1))  # grasp 2
    for _ in range(20):  # its squeeze drags the cube: a load blip, then nothing
        t += DT
        out += det.update(_frame(t, (True, True), load=0.0, watching=True, watch_id=1, slip_time=t))
    assert _names(out).count("DROP") == 1 and "SLIP" not in _names(out)


def _carry(det, t, contact=(True, True), load=1.0, phase=None, watching=True):
    """A frame of carry 1; the caller starts with _onset (the grasp, watch_id 0)."""
    return det.update(_frame(t, contact, load=load, watching=watching, watch_id=1), phase)


def _onset(det, t=-DT):
    return det.update(_frame(t, (True, True)))


def test_drop_needs_a_carried_load_collapse_for_the_debounce() -> None:
    det = EventDetector()
    t, out = 0.0, []
    _onset(det)
    for _ in range(5):  # a load while the slip watch is off (the squeeze): never a drop
        out += _carry(det, t, load=1.0, watching=False)
        t += DT
    for _ in range(15):
        out += _carry(det, t, load=0.0, watching=False)
        t += DT
    assert "DROP" not in _names(out) and det._carry_peak == 0.0
    for _ in range(5):  # carried, but under the 0.15 N arming weight
        out += _carry(det, t, load=0.1)
        t += DT
    for _ in range(15):
        out += _carry(det, t, load=0.0)
        t += DT
    assert "DROP" not in _names(out)
    out += _carry(det, t, load=1.0)  # now carried: peak 1 N
    t += DT
    for _ in range(5):  # below 30 % for 10 ms only
        out += _carry(det, t, load=0.1)
        t += DT
    out += _carry(det, t, load=0.5)  # recovers
    t += DT
    low_from = t
    for _ in range(12):
        out += _carry(det, t, (False, False), load=0.0)
        t += DT
    drops = [e for e in out if e.name is Event.DROP]
    assert len(drops) == 1 and drops[0].t == pytest.approx(low_from + 0.02)
    assert drops[0].info == {"load": 0.0, "peak": 1.0}


def test_drop_debounce_restarts_when_the_rule_pauses() -> None:
    """Review: 4 low ticks, a pause (not watching), then 1 low tick dropped at once."""
    det = EventDetector()
    _onset(det)
    _carry(det, 0.0, load=1.0)
    t = DT
    for _ in range(4):
        _carry(det, t, load=0.0)
        t += DT
    for _ in range(50):
        _carry(det, t, load=0.0, watching=False)
        t += DT
    out = _carry(det, t, load=0.0)
    assert out == []
    for _ in range(10):
        t += DT
        out += _carry(det, t, load=0.0)
    assert _names(out) == ["DROP"]


def test_drop_ignores_the_slip_detectors_rearmed_peak() -> None:
    """Review: DROP read the slip detector's peak, which a reflex rearm lowers; a cube
    that then fell was not reported. DROP keeps its own carried peak."""
    det = EventDetector()
    out = _onset(det)
    out += det.update(_frame(0.0, (True, True), load=3.0, peak=3.0, armed=True, watching=True,
                             watch_id=1))  # fmt: skip
    for k in range(1, 40):  # the reflex rearmed: the slip peak follows the falling load
        load = max(0.0, 3.0 - 0.2 * k)
        frame = _frame(k * DT, (True, True), load=load, peak=load, armed=False, watching=True,
                       watch_id=1)  # fmt: skip
        out += det.update(frame)
    assert "DROP" in _names(out)


def test_no_drop_in_or_after_a_release_phase() -> None:
    det = EventDetector()
    t, out = 0.0, _onset(det)
    out += _carry(det, t, load=1.0, phase="HOLD")
    for phase in ("RELEASE", "RETREAT"):
        for _ in range(20):
            t += DT
            out += _carry(det, t, load=0.0, phase=phase)
    assert "DROP" not in _names(out)
    custom = EventDetector(EventConfig(release_phases=frozenset({"OPEN"})))
    _onset(custom)
    out = _carry(custom, 0.0, load=1.0, phase="HOLD")
    for k in range(1, 20):
        out += _carry(custom, k * DT, load=0.0, phase="PLACE")
    assert "DROP" in _names(out)  # "PLACE" is not a release phase for this detector


def test_each_carry_restarts_the_drop_state_and_release_holds_off_slip() -> None:
    """Round-2 review: a heft that ended in a (false) DROP left the latch set, so the real
    drop of the lift went unreported; a placement fired SLIP. Each watch_slip() is a new
    carry (watch_id), and SLIP, like DROP, waits out a release phase."""
    det = EventDetector()
    t, out = 0.0, _onset(det)

    def tick(load, watching, watch_id, phase=None, slip_time=None, contact=(True, True)):
        nonlocal t, out
        frame = _frame(t, contact, load=load, watching=watching, slip_time=slip_time)
        frame.watch_id = watch_id
        out += det.update(frame, phase)
        t += DT

    for _ in range(10):  # carry 1 (a heft): 1 N, then the load collapses
        tick(1.0, True, 1)
    for _ in range(15):
        tick(0.0, True, 1)
    assert _names(out).count("DROP") == 1
    for _ in range(15):  # carry 2 (the lift): its own peak, its own drop
        tick(2.0, True, 2)
    for _ in range(15):
        tick(0.0, True, 2)
    assert _names(out).count("DROP") == 2
    for _ in range(5):  # carry 3, set down in a release phase: no SLIP, no DROP
        tick(2.0, True, 3)
    tick(1.5, True, 3, phase="PLACE", slip_time=t)
    for _ in range(15):
        tick(0.0, True, 3, phase="PLACE")
    assert _names(out).count("DROP") == 2 and "SLIP" not in _names(out)
    tick(2.0, True, 4, slip_time=t)  # a new carry: SLIP counts again
    assert _names(out)[-1] == "SLIP"


def test_frames_must_come_in_time_order() -> None:
    det = EventDetector()
    assert _names(det.update(_frame(1.0, (True, False)))) == ["CONTACT_ONSET"]
    assert det.update(_frame(1.0, (True, False))) == []  # the same tick again: ignored
    with pytest.raises(RuntimeError, match="reset"):
        det.update(_frame(0.5))
    # A bad phase is refused before anything changes (round-4 review: the onset was lost).
    with pytest.raises(TypeError, match="phase must be a str"):
        det.update(_frame(2.0, (True, True)), 7)
    assert _names(det.update(_frame(2.0, (True, True)))) == []  # still the first grasp
    det.reset()
    assert _names(det.update(_frame(0.5, (True, False)))) == ["CONTACT_ONSET"]


def _priv(pos=(0.5, 0.0, 0.025), *, broken=False, break_time=None, vel=0.0, quat=(1, 0, 0, 0),
          place=(0.5, 0.25)) -> PrivilegedState:  # fmt: skip
    return PrivilegedState(
        cube_pos=np.array(pos, float),
        cube_quat=np.array(quat, float),
        cube_vel=np.array([vel, 0, 0, 0, 0, 0], float),
        cube_finger_force=np.array([1.0, 2.0]),
        broken=broken,
        break_time=break_time,
        rest_z=0.025,
        place_xy=np.array(place, float),
    )


TILTED = (math.cos(0.15), math.sin(0.15), 0.0, 0.0)  # 17° about x


def test_ground_truth_events_break_lifted_placed() -> None:
    det = EventDetector()
    out = det.update(_frame(1.0, (True, True)), "LIFT", _priv((0.5, 0.0, 0.044)))
    assert "LIFTED" not in _names(out)  # 1.9 cm < 2 cm
    out += det.update(_frame(1.002), "LIFT", _priv((0.5, 0.0, 0.10)))
    assert "LIFTED" not in _names(out)  # no pad contact: a flick, not a lift
    out += det.update(_frame(1.004, (True, False)), "LIFT", _priv((0.5, 0.0, 0.046)))
    assert _names(out)[-1] == "LIFTED" and out[-1].info["rise"] == pytest.approx(0.021)
    # Resting at the target but still touched, moving, tilted, or off target: not placed.
    for k, (contact, priv) in enumerate(
        (
            ((True, False), _priv((0.5, 0.25, 0.025))),
            ((False, False), _priv((0.5, 0.25, 0.025), vel=0.05)),
            ((False, False), _priv((0.5, 0.25, 0.025), quat=TILTED)),
            ((False, False), _priv((0.5, 0.275, 0.025))),
            ((False, False), _priv((0.5, 0.25, 0.035))),
        )
    ):
        assert "PLACED" not in _names(det.update(_frame(2.0 + k * DT, contact), "PLACE", priv))
    out = det.update(_frame(2.1), "RETREAT", _priv((0.51, 0.24, 0.0249)))
    assert _names(out) == ["PLACED"]
    # A quarter turn about x is another flat face: still placed.
    det2 = EventDetector()
    det2.update(_frame(0.0, (True, True)), "LIFT", _priv((0.5, 0.0, 0.1)))
    quarter = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    assert _names(det2.update(_frame(1.0), "X", _priv((0.5, 0.25, 0.025), quat=quarter))) == [
        "PLACED"
    ]
    # BREAK carries the monitor's time; a broken cube is never lifted or placed.
    det3 = EventDetector()
    out = det3.update(_frame(3.0, (True, True)), "SQUEEZE", _priv(broken=True, break_time=2.998))
    out += det3.update(_frame(3.002, (True, True)), "LIFT", _priv((0.5, 0, 0.2), broken=True))
    assert _names(out) == ["CONTACT_ONSET", "BREAK"] and out[1].t == 2.998
    assert out[1].info["force"] == 3.0


def test_safety_stop_is_reported_and_logged_on_its_channel(caplog) -> None:
    det = EventDetector()
    with caplog.at_level(logging.INFO, logger="vfa.sim.sensors"):
        rec = det.report("SAFETY_STOP", 1.25, reason="wrist force 120 N > 100 N")
        det.update(_frame(1.3, (True, False)))
    assert rec.name is Event.SAFETY_STOP and det.events[0] is rec
    assert "[SAFETY] event SAFETY_STOP t=1.250" in caplog.text
    assert "[S1] event CONTACT_ONSET" in caplog.text
    with pytest.raises(ValueError):
        det.report("NOT_AN_EVENT", 1.0)
    rec = det.report(Event.SAFETY_STOP, 2.0, name="joint 4", t=0.5)  # info may use these keys
    assert rec.t == 2.0 and rec.info == {"name": "joint 4", "t": 0.5}
    det.reset()
    assert det.events == [] and not det.in_grasp


# ------------------------------------------------------------ sensor suite


def _settled(model, ids):
    """Hold home under OSC for 0.5 s (fingers open): a hand at rest."""
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    osc = OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    for _ in range(250):
        control_step(model, data, (osc, gripper), 1)
    return data, osc, gripper


def test_suite_at_rest_reads_tared_zero_gravity_up_and_the_true_kinematics(model, ids) -> None:
    data, osc, gripper = _settled(model, ids)
    suite = SensorSuite(model, data, ids)
    frame = suite.read()
    assert np.isnan(frame.wrench).all() and math.isnan(frame.load) and not frame.armed
    np.testing.assert_allclose(frame.acc, [0, 0, G], atol=0.05)
    np.testing.assert_allclose(frame.wrench_raw[:3], [0, 0, 0.760 * G], atol=0.05)
    np.testing.assert_array_equal(frame.ee_pos, data.site_xpos[ids.ee_site])  # same instant
    assert frame.width == pytest.approx(gripper.width(), abs=1e-6)
    assert suite.read() is frame  # same tick, same frame
    with pytest.raises(RuntimeError, match="end_tare"):
        suite.end_tare()
    with pytest.raises(RuntimeError, match="tare"):
        suite.watch_slip()
    suite.begin_tare()
    for _ in range(50):
        control_step(model, data, (osc, gripper), 1)
        suite.read()
    suite.end_tare()
    assert suite.slip.tare_n == pytest.approx(0.760 * G, rel=0.01)
    for _ in range(50):
        control_step(model, data, (osc, gripper), 1)
        frame = suite.read()
    np.testing.assert_allclose(frame.wrench, 0.0, atol=0.02)
    assert abs(frame.load) < 0.01 and not frame.contact.any()
    S.reset_to_home(model, data)  # time goes back
    with pytest.raises(RuntimeError, match="reset"):
        suite.read()
    suite.reset()
    assert not suite.tared and math.isnan(suite.read().load)


def test_suite_checks_the_tick_length_and_tares_atomically(model, ids) -> None:
    data, osc, gripper = _settled(model, ids)
    suite = SensorSuite(model, data, ids)
    suite.read()
    for _ in range(2):  # two ticks without a read: the filters would run at the wrong rate
        control_step(model, data, (osc, gripper), 1)
    with pytest.raises(RuntimeError, match="every control tick"):
        suite.read()
    suite.reset()
    suite.begin_tare()
    for _ in range(20):
        control_step(model, data, (osc, gripper), 1)
        suite.read()
    suite.end_tare()
    good_tare, good_slip = suite.tare_wrench.copy(), suite.slip
    suite.watch_slip()
    with pytest.raises(RuntimeError, match="unwatch_slip"):  # no tare while carrying
        suite.begin_tare()
    suite.unwatch_slip()
    suite.begin_tare()
    with pytest.raises(RuntimeError, match="tare"):  # no carry while taring (round 4)
        suite.watch_slip()
    suite.reset()
    suite.begin_tare()
    for _ in range(20):
        control_step(model, data, (osc, gripper), 1)
        suite.read()
    suite.end_tare()
    good_tare, good_slip = suite.tare_wrench.copy(), suite.slip
    # A re-tare with one dropped sample.
    suite.begin_tare()
    for k in range(5):
        control_step(model, data, (osc, gripper), 1)
        if k == 2:
            data.sensordata[ids.wrist_torque_adr] = np.nan  # a dropped torque sample
        suite.read()
    # The dropped sample is left out of the window (review: it used to block the tare).
    suite.end_tare()
    assert suite.slip is not good_slip and np.isfinite(suite.tare_wrench).all()
    np.testing.assert_allclose(suite.tare_wrench, good_tare, atol=0.01)
    # A window of bad readings (a moving hand: a_z out of range) is refused and changes
    # nothing (review: end_tare used to change state first).
    good_tare, good_slip = suite.tare_wrench.copy(), suite.slip
    suite.begin_tare()
    control_step(model, data, (osc, gripper), 1)
    data.sensordata[ids.wrist_acc_adr : ids.wrist_acc_adr + 3] = 0.0  # free fall
    suite.read()
    with pytest.raises(ValueError, match="at rest"):
        suite.end_tare()
    assert suite.slip is good_slip and suite.taring
    np.testing.assert_array_equal(suite.tare_wrench, good_tare)


def test_a_dropped_sample_does_not_poison_the_filters(model, ids) -> None:
    """Review: one NaN pad sample left finger_force NaN and contact off for good."""
    data, osc, gripper = _settled(model, ids)
    suite = SensorSuite(model, data, ids)
    suite.begin_tare()
    for _ in range(10):
        control_step(model, data, (osc, gripper), 1)
        suite.read()
    suite.end_tare()
    control_step(model, data, (osc, gripper), 1)
    before = suite.read()
    control_step(model, data, (osc, gripper), 1)
    data.sensordata[[ids.left_touch_adr, ids.wrist_force_adr + 2]] = np.nan
    frame = suite.read()
    np.testing.assert_array_equal(frame.finger_force, before.finger_force)  # held
    assert np.isfinite(frame.wrench).all() and math.isfinite(frame.load)
    control_step(model, data, (osc, gripper), 1)
    after = suite.read()
    assert np.isfinite(after.finger_force).all() and np.isfinite(after.wrench).all()
    # A frame's arrays are its own: editing one does not reach the filter state.
    after.wrench += 100.0
    after.finger_force += 100.0
    control_step(model, data, (osc, gripper), 1)
    nxt = suite.read()
    assert np.abs(nxt.wrench).max() < 1.0 and np.abs(nxt.finger_force).max() < 1.0


def test_wrist_accelerometer_equals_the_site_acceleration(model, ids) -> None:
    """The deployed detector reads the scene's accelerometer; the T05/T06 sizing read
    mj_objectAcceleration on the wrist site. They agree to rounding through a grasp."""
    data, osc, gripper = _settled(model, ids)
    osc.set_target(data.site_xpos[ids.ee_site] + (0.05, -0.03, -0.15), yaw=0.4)
    gripper.squeeze(5.0)
    acc = np.zeros(6)
    worst = 0.0
    for _ in range(300):
        control_step(model, data, (osc, gripper), 1)
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, ids.wrist_site, acc, 0)
        rot = data.site_xmat[ids.wrist_site].reshape(3, 3)
        worst = max(worst, float(np.abs(rot @ S.read_wrist_acc(model, data, ids) - acc[3:]).max()))
    assert worst < 1e-9


def test_noise_is_seeded_has_the_configured_sigma_and_needs_a_generator(model, ids) -> None:
    with pytest.raises(ValueError, match="Generator"):
        SensorSuite(model, mujoco.MjData(model), ids, SensorConfig(noise=REALISTIC_NOISE))
    with pytest.raises(ValueError):
        SensorSuite(model, mujoco.MjData(model), ids, substeps=0)
    runs = []
    for seed in (5, 5, 6):
        data, osc, gripper = _settled(model, ids)
        cfg = SensorConfig(noise=REALISTIC_NOISE)
        suite = SensorSuite(model, data, ids, cfg, rng=np.random.default_rng(seed))
        rows = []
        for _ in range(1500):
            control_step(model, data, (osc, gripper), 1)
            f = suite.read()
            rows.append(np.r_[f.wrench_raw, f.acc, f.finger_force])
        runs.append(np.array(rows))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])
    std = runs[0].std(axis=0)
    # World axes carry the per-axis σ (an orthonormal rotation of white noise).
    np.testing.assert_allclose(std[0:3], REALISTIC_NOISE.force, rtol=0.1)
    np.testing.assert_allclose(std[3:6], REALISTIC_NOISE.torque, rtol=0.1)
    np.testing.assert_allclose(std[6:9], REALISTIC_NOISE.accel, rtol=0.1)
    # Pad forces are low-passed at 50 Hz: σ·sqrt(α / (2 − α)).
    alpha = LowPass(50.0, DT).alpha
    np.testing.assert_allclose(
        std[9:11], REALISTIC_NOISE.touch * math.sqrt(alpha / (2 - alpha)), rtol=0.15
    )
    assert not any(SensorSuite(model, data, ids).read().contact)


def test_privileged_state_reuses_the_monitor_forces(model, ids) -> None:
    config = OB.load_object_config()
    spec = OB.sample_object(config.classes["fragile_soft"], np.random.default_rng(0))
    data = mujoco.MjData(model)
    S.reset_to_home(model, data)
    OB.apply_object(model, data, spec)
    monitor = OB.BreakMonitor(model, data, spec, ids)
    control_step(model, data, (), 1)
    monitor.update()
    priv = read_privileged(model, data, ids, monitor)
    np.testing.assert_array_equal(priv.cube_finger_force, monitor.last_forces)
    assert priv.rest_z == pytest.approx(0.5 * spec.size)
    np.testing.assert_array_equal(priv.cube_pos, data.xpos[ids.cube_body])
    np.testing.assert_allclose(priv.place_xy, model.site_pos[ids.place_site, :2])
    alone = read_privileged(model, data, ids)
    assert not alone.broken and alone.break_time is None
    np.testing.assert_array_equal(alone.cube_finger_force, priv.cube_finger_force)


# ------------------------------------------------- scripted grasps (physics)


@pytest.fixture(scope="module")
def grasp(model):
    return RE.run_scripted_grasp("light_rigid", model=model)


@pytest.fixture(scope="module")
def dropped(model):
    return RE.run_scripted_grasp("light_rigid", open_at=0.6, model=model)


def test_scripted_grasp_reports_contact_onset_then_grasp_closed(grasp) -> None:
    """ROADMAP T06 acceptance (first half)."""
    assert grasp.result == "SUCCESS"
    assert _names(grasp.events) == ["CONTACT_ONSET", "GRASP_CLOSED", "LIFTED"]
    onset, closed, lifted = grasp.events
    spans = dict((name, (t0, t1)) for name, t0, t1 in grasp.phase_spans())
    assert spans["SQUEEZE"][0] < onset.t < closed.t < spans["SQUEEZE"][1]
    assert spans["LIFT"][0] < lifted.t < spans["LIFT"][1]
    # At GRASP_CLOSED both pads touch and the opening equals the cube's size (soft sink).
    k = int(np.searchsorted(grasp.arrays["t"], closed.t))
    assert grasp.arrays["contact"][k].all()
    assert closed.info["width"] == pytest.approx(grasp.meta["size"], abs=0.003)
    # While carried, the held load is the cube's weight (F/T over accelerometer, tared).
    hold = np.array(grasp.phase_names()) == "HOLD"
    weight = grasp.meta["hidden"]["mass"] * G
    assert np.median(grasp.arrays["load"][hold]) == pytest.approx(weight, rel=0.02)


def test_opening_the_gripper_mid_lift_reports_slip_then_drop(dropped) -> None:
    """ROADMAP T06 acceptance (second half)."""
    assert dropped.result == "DROP"
    names = _names(dropped.events)
    assert names == ["CONTACT_ONSET", "GRASP_CLOSED", "LIFTED", "SLIP", "DROP"]
    t_lift = next(t0 for name, t0, _ in dropped.phase_spans() if name == "LIFT")
    drop = dropped.events[-1]
    # The gripper opened 0.6 s into the lift; the drop follows within 50 ms.
    assert 0.6 < drop.t - t_lift < 0.65
    assert drop.info["load"] < 0.3 * drop.info["peak"]


def test_a_heft_and_set_down_is_neither_a_slip_nor_a_drop(model) -> None:
    """Round-2 review: with a one-way watch, setting a hefted cube back down fired SLIP and
    DROP, and the DROP latch then hid the real drop of the lift. With unwatch_slip() and a
    new carry per watch_slip(), the probe is quiet and the opened gripper still drops."""
    ep = RE.run_scripted_grasp("heavy_rigid", model=model, heft=0.015, open_at=0.6)
    names = _names(ep.events)
    t_lift = next(t0 for name, t0, _ in ep.phase_spans() if name == "LIFT")
    assert not [e for e in ep.events if e.name in (Event.SLIP, Event.DROP) and e.t < t_lift]
    assert names[-1] == "DROP" and ep.result == "DROP"
    probe = np.array(ep.phase_names()) == "PROBE"
    assert ep.arrays["watching"][probe].any() and not ep.arrays["watching"][probe].all()


def test_a_heft_that_loses_the_cube_ends_the_attempt(model) -> None:
    """Round-4 review: the harness lowered the closed fingers onto a cube the heft had
    lost and crushed it. A heft that holds under 0.15 N now skips the set-down and lift."""
    ep = RE.run_scripted_grasp("heavy_slippery", model=model, grip=1.0, heft=0.02)
    assert ep.result in ("DROP", "SLIP") and "LIFT" not in ep.phases
    assert "BREAK" not in _names(ep.events)


def test_a_2cm_heft_is_a_lift_off_and_the_set_down_unloads_the_hand(model) -> None:
    """Round-3 review: LIFTED fired once per episode, inside a 2 cm heft, and the harness
    left the cube hanging in the grip after the set-down. LIFTED now re-arms when the cube
    rests again, and the harness presses on until the table takes the load."""
    ep = RE.run_scripted_grasp("light_rigid", model=model, heft=0.02)
    phases = np.array(ep.phase_names())
    lifted = [phases[int(np.searchsorted(ep.arrays["t"], e.t))] for e in ep.events_named("LIFTED")]
    assert lifted == ["PROBE", "LIFT"] and ep.result == "SUCCESS"
    probe = np.flatnonzero(phases == "PROBE")
    first_lift = int(np.argmax(phases == "LIFT"))
    assert ep.arrays["load"][first_lift] < 0.25 * np.nanmax(ep.arrays["load"][probe])


@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [
        # The closing fingers drag this cube: before T06's watch_slip, SLIP + DROP fired here.
        ("heavy_slippery", {}, ["CONTACT_ONSET", "GRASP_CLOSED", "LIFTED"]),
        ("fragile_soft", {"grip": 8.0, "speed": 0.5}, ["CONTACT_ONSET", "BREAK"]),
        ("slippery", {"grip": 3.0}, ["CONTACT_ONSET", "GRASP_CLOSED", "SLIP", "DROP"]),
        # Realistic sensor noise on a clean lift of a 50 g fragile cube (sample 2), the
        # lightest in the classes: no false slip (review: the old design false-slipped).
        ("fragile_soft", {"sample": 2, "noise": REALISTIC_NOISE, "seed": 3}, None),
    ],
)
def test_scripted_events_per_case(model, cls, kwargs, expected) -> None:
    ep = RE.run_scripted_grasp(cls, model=model, **kwargs)
    if expected is None:
        assert ep.result == "SUCCESS" and "SLIP" not in _names(ep.events)
        assert "DROP" not in _names(ep.events)
    else:
        assert _names(ep.events) == expected


# ------------------------------------------------- the sizing study's replay


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, S.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def test_sizing_replay_equals_the_slip_detector_exactly() -> None:
    """FEASIBILITY §8 rests on the vectorized replay in scripts/size_slip_detector.py;
    it must give the same lowest debounced statistic as SlipDetector, bit for bit,
    with and without noise and the noise margin, in both semantics."""
    SZ = _load_script("size_slip_detector")
    rng = np.random.default_rng(0)
    for sigma in ((0.0, 0.0), (0.05, 0.03)):
        rows = []
        for i in range(12):
            n_sq, n_lift = int(rng.integers(50, 200)), int(rng.integers(300, 700))
            kg = rng.uniform(0.05, 0.6)
            lost = np.r_[np.zeros(n_lift // 2), np.full(n_lift - n_lift // 2, rng.uniform(0, 0.5))]
            hand_acc = 0.5 * np.sin(np.linspace(0, 3, n_lift))
            sq_f = HAND_KG * G + 0.02 * rng.standard_normal(n_sq)
            sq_f = sq_f + sigma[0] * rng.standard_normal(n_sq)
            sq_a = G + sigma[1] * rng.standard_normal(n_sq)
            f = HAND_KG * (G + hand_acc) + kg * (1 - lost) * (G + hand_acc)
            f = f + sigma[0] * rng.standard_normal(n_lift)
            a = G + hand_acc + sigma[1] * rng.standard_normal(n_lift)
            rows.append((i, sq_f, sq_a, f, a, HAND_KG))
        for semantics in ("suite", "t05"):
            debs = SZ.replay(rows, sigma, (0.0, 2.0, 4.0), (1, 3, 5), semantics)
            for k_noise in (0.0, 2.0, 4.0):
                config = SlipConfig(noise_k=k_noise)
                for k, row in enumerate(rows):
                    det = SZ.scalar_replay(row, config, sigma, semantics)
                    for w in (1, 3, 5):
                        d = debs[(k_noise, w)][k]
                        vec = float(np.nanmin(d)) if np.isfinite(d).any() else None
                        assert vec == det.min_ratio_by_window[w], (sigma, semantics, k_noise, k, w)
