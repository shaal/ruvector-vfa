"""T05 physics feasibility spike: does the grip force (and its profile) decide success?

ROADMAP T05 / ARCHITECTURE D4. A scripted top-down grasp, swept over grip force ×
closing speed × the five hidden classes (``configs/objects.toml``)::

    approach above the cube → descend → tare the wrist F/T → squeeze(f, speed)
    → lift 10 cm → hold 1 s

and three measurements around it:

1. **Squeeze calibration**: the real finger force versus the command, for the
   menagerie servo (``SceneConfig(gripper="servo")``) and the force motor that
   T05 made the scene default.
2. **Success map**: success / break / slip rate per (class, grip force, speed).
3. **Slip detector**: the held mass from the wrist F/T and accelerometer
   (``FTSlipDetector``), scored against privileged ground truth (the cube
   height relative to the fingers), on the map plus harness studies that make
   partial slips (``release``, ``margin``) and a held-out set with sensor noise
   (``noise``).

Usage::

    uv run python scripts/spike_grip.py --quick                 # < 5 min, runs/spike/quick
    uv run python scripts/spike_grip.py --workers 8             # full sweep, runs/spike/full
    uv run python scripts/spike_grip.py --aggregate --out runs/spike/full

The sweep is chunked (one JSON file per chunk under ``<out>/chunks``) and
resumable: a re-run skips chunks whose file is current (same jobs, same code
fingerprint), and a wall-clock deadline (``--deadline``, default 8 min) stops
it cleanly between chunks (CLAUDE.md, Runtime limits). ``--aggregate`` builds
the tables (``summary.md``, ``metrics.json``) and the figures from the chunk
files. A run directory belongs to one profile (``profile.json``).

Every attempt samples its cube with ``np.random.default_rng((class index, sample))``,
so every (force, speed) cell of a class grips the same cubes (a paired design).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from vfa.control.gripper import Gripper
from vfa.control.osc import CONTROL_HZ, OperationalSpaceController, control_step
from vfa.sim import objects as OB
from vfa.sim import scene as S

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "runs" / "spike"

# ------------------------------------------------------------------ the attempt

DT = 1.0 / CONTROL_HZ  # one control tick = one physics step (500 Hz)
APPROACH_HEIGHT = 0.12  # ee above the cube center before the descent [m]
APPROACH_S = 1.4  # move from home to above the cube, then settle [s]
DESCEND_S = 1.0  # minimum-jerk descent to the grasp height [s]
DESCEND_SETTLE_S = 0.3  # settle after the descent; its last TARE_S is the tare window [s]
TARE_S = 0.1  # the last part of the descent settle: F/T tare window [s]
SQUEEZE_SETTLE_S = 0.15  # hold after the gripper reports the force [s]
CAL_SETTLE_S = 0.5  # calibration: let soft contacts settle before reading the force [s]
STALL_SPEED = 0.002  # the fingers count as stopped below this opening rate [m/s]
SQUEEZE_MARGIN_S = 1.0  # squeeze timeout beyond the nominal close + ramp time [s]
LIFT_M = 0.10  # ee lift [m] (ROADMAP T05)
LIFT_S = 1.0  # minimum-jerk lift duration [s]
LIFT_PEAK_ACC = 10.0 / math.sqrt(3.0) * LIFT_M / LIFT_S**2  # 0.577 m/s²
HOLD_S = 1.0  # hold at the top [s] (ROADMAP T05)
# Success: at the end of the hold the cube center is at least this far above its
# rest height and the cube was not broken. The ee rises 10 cm; the 2 cm margin
# covers the OSC payload sag (6.5 mm/kg at 1500 N/m) and soft-contact creep.
SUCCESS_RISE_M = 0.08
# Ground-truth slip: the cube center falls this far relative to the fingertip
# center (ee site) after the lift starts. A held cube moves < 1 mm (elliptic
# cones); summary.md reports the whole distribution, micro-slips included.
SLIP_GT_M = 0.005
# Squeeze timing: the gripper's speed servo ramps the force at 250·speed N/s per finger.
RAMP_N_PER_S_PER_MPS = 250.0

# F/T slip detector (held-mass drop). The vertical wrist force over the wrist's
# vertical proper acceleration (an accelerometer in the F/T sensor: +g at rest) is
# the mass hanging below the wrist, F_z / a_z. Minus the hand's own mass (the
# tare) it is the held cube's mass while the cube moves with the hand, whatever
# the lift profile. In a slip the cube accelerates less than the hand, so the
# estimate falls. It fires below DROP_FRACTION of its running peak, once the peak
# passed ARM_LOAD_N / g (0.15 N ≈ 15 g: every class's lightest cube is 50 g).
FT_FILTER_HZ = 20.0  # first-order low-pass on the mass estimate
DROP_FRACTION = 0.9
# The drop must hold this long to fire (3 ticks at 500 Hz). Chosen from
# WINDOW_SWEEP on earlier runs, which included the quick profile's noise cubes
# (sample NOISE_FIRST_QUICK) and the first held-out cubes (samples 9000-9002):
# 1-2 ticks fire more on noise, 5-10 miss stick-slip partial slips (their dips
# are short). The full profile's held-out cubes (NOISE_FIRST_FULL) were first
# run after this choice and DROP_FRACTION were fixed (T05 review round 3).
DEBOUNCE_S = 0.006
# Debounce windows [ticks] whose lowest ratio every attempt also records, so the
# summary scores each (window, threshold) pair exactly.
WINDOW_SWEEP: tuple[int, ...] = (1, 2, 3, 5, 10)
ARM_LOAD_N = 0.15
# Valid proper vertical acceleration [m/s²]; outside it (near free fall, or an
# absurd reading) the mass estimate is skipped.
MIN_PROPER_ACC = 1.0
MAX_PROPER_ACC = 100.0
TRACE_HZ = 50  # downsampled traces stored per attempt (lift + hold)


@dataclass(slots=True)
class AttemptResult:
    """One scripted grasp. Times are seconds from the start of the lift."""

    cls: str
    sample: int
    grip: float  # commanded squeeze per finger [N]
    speed: float  # closing speed [m/s, rate of change of the opening]
    hidden: dict  # privileged object properties (logging and scoring only)
    size: float
    outcome: str  # SUCCESS, BREAK, SLIP (in hand, fell > SLIP_GT_M), DROP (lost)
    success: bool
    broken: bool
    break_time: float | None  # from the squeeze command [s]
    peak_force: float  # largest summed finger force [N]
    settled_force: float  # summed finger force when the lift starts [N]
    rise: float  # cube center rise at the end of the hold [m]
    rel_drop: float  # largest fall of the cube relative to the ee [m]
    gt_slip: bool
    gt_slip_time: float | None
    ft_slip: bool
    ft_slip_time: float | None
    ft_peak_load: float  # largest held-mass estimate, as a weight [N]
    ft_min_ratio: float | None  # lowest debounced held-mass / peak (fires below DROP_FRACTION)
    ft_min_ratio_by_window: dict  # the same for each window in WINDOW_SWEEP [ticks]
    tare_n: float  # the hand's weight at tare [N]
    squeeze_s: float  # time from the squeeze command to the lift [s]
    sim_s: float
    wall_s: float
    trace: dict = field(default_factory=dict)  # t, load, rel (TRACE_HZ, lift + hold)


class LowPass:
    """First-order low-pass filter (exponential smoothing) at a fixed sample rate."""

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt)
        self.value: float | None = None

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value


class FTSlipDetector:
    """Causal slip detector on the wrist F/T and accelerometer (no privileged signals).

    ``tare(f_z, a_z)`` stores the hand's own mass ``f_z / a_z`` (the caller
    averages both before contact). Each ``update(t, f_z, a_z)`` takes the
    world-frame vertical wrist force [N] and the wrist's vertical proper
    acceleration [m/s², +g at rest]. The held mass ``f_z / a_z − m_hand`` is
    low-passed at ``FT_FILTER_HZ``; ``load`` and ``peak`` report it as a weight
    [N]. The detector fires (and latches) once the estimate stays below
    ``DROP_FRACTION`` of its running peak for ``DEBOUNCE_S``, after the peak
    passed ``ARM_LOAD_N``. A non-finite or out-of-range sample is skipped.
    ``min_ratio`` is the lowest debounced ratio seen (``None`` before arming):
    the detector would fire at any threshold above it, so one run scores every
    threshold exactly.
    """

    def __init__(self, dt: float, g: float = 9.81) -> None:
        self.dt = dt
        self.g = g
        self.hand_mass = 0.0
        self.filter = LowPass(FT_FILTER_HZ, dt)
        self.mass = 0.0
        self.peak_mass = 0.0
        self.fired_at: float | None = None
        self.min_ratio: float | None = None
        window = max(1, round(DEBOUNCE_S / dt))
        self._recent: deque[float] = deque(maxlen=max(window, *WINDOW_SWEEP))
        self._window = window
        self.min_ratio_by_window: dict[int, float | None] = dict.fromkeys(WINDOW_SWEEP)

    @property
    def tare_n(self) -> float:
        return self.hand_mass * self.g

    @property
    def load(self) -> float:
        return self.mass * self.g

    @property
    def peak(self) -> float:
        return self.peak_mass * self.g

    def tare(self, f_z: float, a_z: float) -> None:
        if not (math.isfinite(f_z) and MIN_PROPER_ACC <= a_z <= MAX_PROPER_ACC):
            raise ValueError(
                f"tare needs finite readings with the hand at rest, got {f_z!r}, {a_z!r}"
            )
        self.hand_mass = f_z / a_z

    def update(self, t: float, f_z: float, a_z: float) -> bool:
        if math.isfinite(f_z) and MIN_PROPER_ACC <= a_z <= MAX_PROPER_ACC:
            self.mass = self.filter(f_z / a_z - self.hand_mass)
            self.peak_mass = max(self.peak_mass, self.mass)
            if self.peak >= ARM_LOAD_N:
                self._recent.append(self.mass / self.peak_mass)
                for w, low in self.min_ratio_by_window.items():
                    if len(self._recent) >= w:
                        ratio = max(list(self._recent)[-w:])
                        if low is None or ratio < low:
                            self.min_ratio_by_window[w] = ratio
                if len(self._recent) >= self._window:
                    # below the line for the whole window
                    debounced = max(list(self._recent)[-self._window :])
                    if self.min_ratio is None or debounced < self.min_ratio:
                        self.min_ratio = debounced
                    if self.fired_at is None and debounced < DROP_FRACTION:
                        self.fired_at = t
        return self.fired_at is not None


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s)


def min_jerk_rate(s: float) -> float:
    """d(min_jerk)/ds."""
    s = min(max(s, 0.0), 1.0)
    return 30.0 * s * s * (1.0 - s) ** 2


def path_target(
    osc: OperationalSpaceController,
    start: np.ndarray,
    goal: np.ndarray,
    t: float,
    T: float,
    yaw: float,
) -> None:
    """Set the target on a minimum-jerk path from ``start`` to ``goal`` over ``T`` s at ``t``.

    The path velocity is fed forward (``vel=``). Without it the damper resists
    the motion and its coupled modes push the ee sideways: a 12 cm descent in
    1.0 s drifted 7.2 mm off the line (median of 35 cubes, max 7.9 mm). With
    ``vel=``: 1.8 mm (max 2.0 mm). An earlier T05 run with a rate-limited
    descent target drifted 13.6 mm, and a fingertip landed on a 5.9 cm cube.
    """
    s = t / T
    vel = (goal - start) * (min_jerk_rate(s) / T)
    osc.set_target(start + (goal - start) * min_jerk(s), yaw=yaw, vel=(*vel, 0.0, 0.0, 0.0))


class Arm:
    """One model + data + controllers, reused across attempts (one per worker process)."""

    def __init__(self, cfg: S.SceneConfig | None = None) -> None:
        self.model = S.build_scene(cfg)
        self.data = mujoco.MjData(self.model)
        self.ids = S.scene_ids(self.model)
        self._acc = np.zeros(6)
        self._finger_dofs = [
            int(self.model.jnt_dofadr[self.model.joint(n).id]) for n in S.FINGER_JOINTS
        ]

    def reset(self, spec: OB.ObjectSpec) -> tuple[OperationalSpaceController, Gripper]:
        S.reset_to_home(self.model, self.data)
        OB.apply_object(self.model, self.data, spec)
        osc = OperationalSpaceController(self.model, self.data, ids=self.ids)
        return osc, Gripper(self.model, self.data, self.ids)

    def ft_world_z(self) -> float:
        """Vertical (world z) force that link7 applies to the hand subtree [N], raw."""
        force, _ = S.read_wrist_ft(self.model, self.data, self.ids)
        rot = self.data.site_xmat[self.ids.wrist_site].reshape(3, 3)
        return float(rot[2] @ force)

    def width_rate(self) -> float:
        """Rate of change of the finger opening [m/s] (the gripper encoders)."""
        return float(self.data.qvel[self._finger_dofs].sum())

    def wrist_acc_z(self) -> float:
        """Vertical proper acceleration of the wrist [m/s²], +g at rest.

        What an accelerometer in the F/T sensor reads. MuJoCo computes the body
        accelerations (``cacc``) for the force sensor in the same step, so the
        reading is consistent with ``ft_world_z``.
        """
        mujoco.mj_objectAcceleration(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.ids.wrist_site, self._acc, 0
        )
        return float(self._acc[5])


def run_attempt(
    arm: Arm,
    spec: OB.ObjectSpec,
    grip: float,
    speed: float,
    sample: int = 0,
    keep_trace: bool = True,
    settle_s: float = SQUEEZE_SETTLE_S,
    release: tuple[float, float] | None = None,
    ft_noise: tuple[float, float] = (0.0, 0.0),
    noise_rng: np.random.Generator | None = None,
) -> AttemptResult:
    """Scripted grasp-and-lift of ``spec`` with squeeze ``grip`` [N/finger] at ``speed`` [m/s].

    The lift starts ``settle_s`` after the gripper reports the force (its motor
    force reached 99 % of the request with the fingers stopped), or at a timeout.
    ``release = (t, f)`` lowers the squeeze to ``f`` N at ``t`` s after the lift
    starts (the release study: a slip in mid-air, after a clean lift).
    ``ft_noise = (σ_F, σ_a)`` adds white Gaussian noise [N, m/s²] from
    ``noise_rng`` to the detector's two inputs every tick (the noise study).
    """
    sigma_f, sigma_a = ft_noise
    if (sigma_f > 0 or sigma_a > 0) and not isinstance(noise_rng, np.random.Generator):
        raise ValueError("ft_noise > 0 needs a numpy Generator noise_rng")

    def sensed() -> tuple[float, float]:
        f_z, a_z = arm.ft_world_z(), arm.wrist_acc_z()
        if sigma_f > 0 or sigma_a > 0:
            f_z += float(noise_rng.normal(0.0, sigma_f))
            a_z += float(noise_rng.normal(0.0, sigma_a))
        return f_z, a_z

    wall0 = time.perf_counter()
    model, data, ids = arm.model, arm.data, arm.ids
    osc, gripper = arm.reset(spec)
    monitor = OB.BreakMonitor(model, data, spec, ids)
    seen = OB.observe_cube(model, data, ids=ids)
    rest_z = seen.z
    cube = np.array([seen.x, seen.y, seen.z])
    yaw = seen.yaw  # fingers close along the cube's face normal
    ctrl = (osc, gripper)

    def tick() -> None:
        control_step(model, data, ctrl, 1)
        monitor.update()

    def run(seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            tick()

    # Approach and descend (fingers open). Tare at the end of the descent settle.
    osc.set_target(cube + (0.0, 0.0, APPROACH_HEIGHT), yaw=yaw)
    run(APPROACH_S)
    above = np.array(osc.target.pos)  # the reference sits on it after the settle
    for k in range(round(DESCEND_S / DT)):
        path_target(osc, above, cube, k * DT, DESCEND_S, yaw)
        tick()
    osc.set_target(cube, yaw=yaw)
    run(DESCEND_SETTLE_S - TARE_S)
    tare_f, tare_a = [], []
    for _ in range(round(TARE_S / DT)):
        tick()
        f_z, a_z = sensed()
        tare_f.append(f_z)
        tare_a.append(a_z)
    detector = FTSlipDetector(DT, float(np.linalg.norm(model.opt.gravity)))
    detector.tare(float(np.mean(tare_f)), float(np.mean(tare_a)))

    # Squeeze: wait until the gripper reports the force (its own feedback) or time out.
    t_squeeze = float(data.time)
    gripper.squeeze(grip, speed=speed)
    nominal = (gripper.width() - spec.size) / speed + grip / (RAMP_N_PER_S_PER_MPS * speed)
    deadline = t_squeeze + nominal + SQUEEZE_MARGIN_S
    reached_at: float | None = None
    while data.time < deadline and not monitor.broken:
        tick()
        if (
            reached_at is None
            and gripper.actuator_squeeze() >= 0.99 * grip
            and abs(arm.width_rate()) < STALL_SPEED
        ):
            reached_at = float(data.time)
        if reached_at is not None and data.time >= reached_at + settle_s:
            break
    squeeze_s = float(data.time) - t_squeeze
    settled = sum(S.finger_cube_forces(model, data, ids))

    # Lift on a minimum-jerk path, then hold. GT slip and the F/T detector run each tick.
    start = np.array(osc.target.pos)  # the grasp pose (the ee sags under a heavy cube)
    t0 = float(data.time)
    rel0 = float(data.xpos[ids.cube_body, 2] - data.site_xpos[ids.ee_site, 2])
    rel_drop = 0.0
    gt_time: float | None = None
    stride = round(CONTROL_HZ / TRACE_HZ)
    trace: dict[str, list[float]] = {"t": [], "load": [], "rel": []}
    k = 0
    released = release is None
    while not monitor.broken and data.time < t0 + LIFT_S + HOLD_S:
        t = float(data.time) - t0
        if not released and t >= release[0]:
            gripper.squeeze(release[1], speed=speed)  # a weaker squeeze applies at once
            released = True
        path_target(osc, start, start + (0.0, 0.0, LIFT_M), t, LIFT_S, yaw)
        tick()
        t = float(data.time) - t0
        detector.update(t, *sensed())
        rel = float(data.xpos[ids.cube_body, 2] - data.site_xpos[ids.ee_site, 2]) - rel0
        rel_drop = max(rel_drop, -rel)
        if gt_time is None and -rel > SLIP_GT_M:
            gt_time = t
        if keep_trace and k % stride == 0:
            trace["t"].append(round(t, 4))
            trace["load"].append(round(detector.load, 4))
            trace["rel"].append(round(rel, 5))
        k += 1

    rise = float(data.xpos[ids.cube_body, 2] - rest_z)
    broken = monitor.broken
    success = not broken and rise >= SUCCESS_RISE_M
    if broken:
        outcome = "BREAK"
    elif success:
        outcome = "SUCCESS"
    elif rise >= 0.02:
        outcome = "SLIP"
    else:
        outcome = "DROP"
    hidden = spec.hidden()
    hidden["friction"] = list(hidden["friction"])
    hidden["solref"] = list(hidden["solref"])
    return AttemptResult(
        cls=spec.class_name,
        sample=sample,
        grip=grip,
        speed=speed,
        hidden=hidden,
        size=spec.size,
        outcome=outcome,
        success=success,
        broken=broken,
        break_time=None if monitor.break_time is None else monitor.break_time - t_squeeze,
        peak_force=monitor.peak_force,
        settled_force=float(settled),
        rise=rise,
        rel_drop=rel_drop,
        gt_slip=gt_time is not None,
        gt_slip_time=gt_time,
        ft_slip=detector.fired_at is not None,
        ft_slip_time=detector.fired_at,
        ft_peak_load=detector.peak,
        ft_min_ratio=detector.min_ratio,
        ft_min_ratio_by_window={str(w): r for w, r in detector.min_ratio_by_window.items()},
        tare_n=detector.tare_n,
        squeeze_s=squeeze_s,
        sim_s=float(data.time),
        wall_s=time.perf_counter() - wall0,
        trace=trace if keep_trace else {},
    )


def sample_spec(config: OB.ObjectConfig, cls: str, sample: int) -> OB.ObjectSpec:
    """The ``sample``-th cube of class ``cls``: same seed → same cube in every cell."""
    index = config.names.index(cls)
    return OB.sample_object(config.classes[cls], np.random.default_rng((index, sample)))


# ------------------------------------------------------------------- the sweep

# Full sweep. Grip force is per finger [N]; the actuator holds up to 50 N.
FORCES: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0)
SPEEDS: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.5)
SAMPLES = 5
# Closing-speed experiment on fragile_soft: many cubes at the forces near its break band.
SPEED_TEST_FORCES: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
SPEED_TEST_SPEEDS: tuple[float, ...] = (0.02, 0.5)
SPEED_TEST_SAMPLES = 30
SPEED_TEST_FIRST = 1000  # sample indices distinct from the map's
# Squeeze calibration: commands per finger [N], on two cubes of every class (unbreakable).
CAL_FORCES: tuple[float, ...] = (0.25, 0.5, 1.0, 1.25, 2.0, 4.0, 8.0, 16.0, 32.0, 45.0)
CAL_SAMPLES = 2
CAL_SPEED = 0.1
# Harness studies for the slip detector. They size the grip from the privileged
# need (F_min = m·g/(2μ) per finger), so they make the slips the force map's
# coarse grid misses; no policy sees F_min.
# Release: lift with a safe grip, then at RELEASE_AT s after the lift starts
# (mid-hold) lower the squeeze to a fraction of F_min: drops, and partial slips
# near 1.
RELEASE_FRACTIONS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.98, 0.995)
RELEASE_AT = LIFT_S + 0.3
RELEASE_SAMPLES = 4
RELEASE_FIRST = 2000  # sample indices distinct from the other studies
# Margin: squeeze with a factor of the grip the lift needs at its peak
# acceleration, m·(g + LIFT_PEAK_ACC)/(2μ): partial slips at lift-off.
MARGIN_FACTORS: tuple[float, ...] = (0.96, 0.98, 1.0, 1.02, 1.04, 1.06)
MARGIN_SAMPLES = 6
MARGIN_FIRST = 3000
HARNESS_SPEED = 0.1
# Held-out and noise study: new cubes in the near-threshold harness variants, with
# white Gaussian noise on the detector's two inputs every tick: none (the held-out
# check), then two levels. T06 owns the real sensor model. The full profile's cubes
# start at NOISE_FIRST_FULL, which no run used before the threshold and the
# debounce were fixed; the quick profile (a smoke test) keeps NOISE_FIRST_QUICK.
NOISE_LEVELS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.02, 0.01), (0.05, 0.02))
NOISE_VARIANTS: tuple[tuple[str, float], ...] = (
    ("margin", 0.98),
    ("margin", 1.0),
    ("margin", 1.04),
    ("margin", 1.3),
    ("release", 0.9),
)
NOISE_SAMPLES = 3
NOISE_FIRST_QUICK = 9000
NOISE_FIRST_FULL = 9100
# Window-edge coverage: this many fresh cubes per class (sample indices from
# COVERAGE_FIRST) score each window edge against the hidden need, without physics.
COVERAGE_SAMPLES = 4000
COVERAGE_FIRST = 20000

QUICK = {
    "forces": (1.0, 2.0, 3.0, 6.0, 12.0, 20.0),
    "speeds": (0.02, 0.5),
    "samples": 5,
    "speed_test_forces": (2.5, 3.0, 3.5),
    "speed_test_samples": 12,
    "cal_forces": (0.5, 1.25, 4.0, 16.0, 45.0),
    "cal_samples": 1,
    "release_samples": 1,
    "margin_samples": 1,
    "noise_samples": 1,
    "noise_first": NOISE_FIRST_QUICK,
}
FULL = {
    "forces": FORCES,
    "speeds": SPEEDS,
    "samples": SAMPLES,
    "speed_test_forces": SPEED_TEST_FORCES,
    "speed_test_samples": SPEED_TEST_SAMPLES,
    "cal_forces": CAL_FORCES,
    "cal_samples": CAL_SAMPLES,
    "release_samples": RELEASE_SAMPLES,
    "margin_samples": MARGIN_SAMPLES,
    "noise_samples": NOISE_SAMPLES,
    "noise_first": NOISE_FIRST_FULL,
}


@dataclass(slots=True, frozen=True)
class Chunk:
    """One unit of work: a list of attempts with one gripper kind. Its file is ``name``.json.

    A job is ``(class, sample, grip, speed, harness)``: ``harness`` is ``None``,
    or ``("release" | "margin", factor[, noise level index])``, and then the
    grip comes from the privileged need (``harness_grip``) and ``grip`` is unused.
    """

    name: str
    kind: str  # "motor" or "servo"
    study: str  # "map", "speed", "release", "margin", "noise", or "calibration"
    jobs: tuple[tuple[str, int, float, float, tuple | None], ...]
    unbreakable: bool = False  # calibration: measure the force, never break


def harness_grip(
    spec: OB.ObjectSpec, variant: str, factor: float
) -> tuple[float, tuple[float, float] | None]:
    """(grip, release) for a harness job, from the privileged need (harness only).

    ``release``: a safe grip of 2·F_min, then ``factor``·F_min at ``RELEASE_AT``.
    ``margin``: ``factor`` × the grip the lift needs at its peak acceleration.
    Fragile cubes stay below 0.4× their break force per finger (0.8× summed).
    """
    f_min = spec.mass * 9.81 / (2.0 * spec.friction[0])
    if variant == "release":
        grip, release = 2.0 * f_min, (RELEASE_AT, factor * f_min)
    elif variant == "margin":
        grip, release = factor * f_min * (9.81 + LIFT_PEAK_ACC) / 9.81, None
    else:
        raise ValueError(f"unknown harness variant {variant!r}")
    if spec.break_force is not None:
        grip = min(grip, 0.4 * spec.break_force)
    return grip, release


def plan(profile: dict, config: OB.ObjectConfig) -> list[Chunk]:
    """Every chunk of a profile, in a fixed order (largest first for load balance)."""
    chunks: list[Chunk] = []
    for cls in config.names:
        for speed in profile["speeds"]:
            jobs = tuple(
                (cls, k, f, speed, None)
                for f in profile["forces"]
                for k in range(profile["samples"])
            )
            chunks.append(Chunk(f"map__{cls}__v{speed:g}", "motor", "map", jobs))
    for speed in SPEED_TEST_SPEEDS:
        for f in profile["speed_test_forces"]:
            samples = range(SPEED_TEST_FIRST, SPEED_TEST_FIRST + profile["speed_test_samples"])
            jobs = tuple(("fragile_soft", k, f, speed, None) for k in samples)
            chunks.append(Chunk(f"speed__f{f:g}__v{speed:g}", "motor", "speed", jobs))
    for variant, factors, first, n in (
        ("release", RELEASE_FRACTIONS, RELEASE_FIRST, profile["release_samples"]),
        ("margin", MARGIN_FACTORS, MARGIN_FIRST, profile["margin_samples"]),
    ):
        for cls in config.names:
            jobs = tuple(
                (cls, k, 0.0, HARNESS_SPEED, (variant, factor))
                for factor in factors
                for k in range(first, first + n)
            )
            chunks.append(Chunk(f"{variant}__{cls}", "motor", variant, jobs))
    for level in range(len(NOISE_LEVELS)):
        for cls in config.names:
            jobs = tuple(
                (cls, k, 0.0, HARNESS_SPEED, (variant, factor, level))
                for variant, factor in NOISE_VARIANTS
                for k in range(
                    profile["noise_first"], profile["noise_first"] + profile["noise_samples"]
                )
            )
            chunks.append(Chunk(f"noise{level}__{cls}", "motor", "noise", jobs))
    for kind in ("servo", "motor"):
        jobs = tuple(
            (cls, k, f, CAL_SPEED, None)
            for f in profile["cal_forces"]
            for cls in config.names
            for k in range(profile["cal_samples"])
        )
        chunks.append(Chunk(f"cal__{kind}", kind, "calibration", jobs, unbreakable=True))
    return [c for c in chunks if c.jobs]


def code_fingerprint() -> str:
    """Hash of everything that decides an attempt's result: this script, the object
    config, the ``vfa`` sources, the vendored robot model, and the MuJoCo and numpy
    versions. A chunk file written by other code is stale: the sweep re-runs it
    and the aggregation refuses it."""
    digest = hashlib.sha256(f"{mujoco.__version__} {np.__version__}".encode())
    paths = [Path(__file__).resolve(), OB.OBJECTS_TOML.resolve()]
    paths += sorted((REPO_ROOT / "src" / "vfa").rglob("*.py"))
    paths += sorted(p for p in S.PANDA_DIR.rglob("*") if p.is_file())
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _jobs_json(chunk: Chunk) -> list:
    return json.loads(json.dumps([list(job) for job in chunk.jobs]))


def chunk_is_current(path: Path, chunk: Chunk, fingerprint: str) -> bool:
    """True if ``path`` holds this chunk's jobs, run by code with this fingerprint."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return False
    jobs = _jobs_json(chunk)
    return (
        data.get("fingerprint") == fingerprint
        and data.get("jobs") == jobs
        and len(data["results"]) == len(jobs)
        and all(isinstance(r, dict) and _ROW_KEYS <= r.keys() for r in data["results"])
    )


# Every result row carries these (``asdict(AttemptResult)`` plus the chunk's tags).
_ROW_KEYS = frozenset(AttemptResult.__dataclass_fields__) | {
    "kind",
    "study",
    "variant",
    "factor",
    "noise_level",
    "noise",
}


def run_chunk(chunk: Chunk, path: str, fingerprint: str) -> tuple[str, int, float]:
    """Run one chunk in a worker process and write its JSON file. Returns (name, n, wall)."""
    wall0 = time.perf_counter()
    config = OB.load_object_config()
    arm = Arm(S.SceneConfig(gripper=chunk.kind))
    results = []
    for cls, sample, grip, speed, harness in chunk.jobs:
        spec = sample_spec(config, cls, sample)
        if chunk.unbreakable:
            spec = replace(spec, break_force=None)
        release = None
        noise, noise_rng, level = (0.0, 0.0), None, None
        if harness is not None:
            grip, release = harness_grip(spec, harness[0], harness[1])
            if len(harness) > 2:
                level = harness[2]
                noise = NOISE_LEVELS[level]
                noise_rng = np.random.default_rng((config.names.index(cls), sample, level, 17))
        calibration = chunk.study == "calibration"
        result = run_attempt(
            arm,
            spec,
            grip,
            speed,
            sample,
            keep_trace=not calibration,
            settle_s=CAL_SETTLE_S if calibration else SQUEEZE_SETTLE_S,
            release=release,
            ft_noise=noise,
            noise_rng=noise_rng,
        )
        row = asdict(result)
        row["kind"] = chunk.kind
        row["study"] = chunk.study
        row["variant"], row["factor"] = (harness[0], harness[1]) if harness else (None, None)
        row["noise_level"], row["noise"] = level, list(noise)
        results.append(row)
    tmp = Path(path).with_suffix(".tmp")
    record = {
        "chunk": chunk.name,
        "fingerprint": fingerprint,
        "jobs": _jobs_json(chunk),
        "results": results,
    }
    tmp.write_text(json.dumps(record))
    tmp.replace(path)  # atomic: a killed run never leaves a half-written chunk
    return chunk.name, len(results), time.perf_counter() - wall0


def claim_profile(out: Path, name: str) -> None:
    """Mark ``out`` as a run of profile ``name``; refuse a directory of another profile.

    The profiles share chunk names, so a quick run in a full run's directory
    would overwrite its chunks.
    """
    marker = out / "profile.json"
    if marker.exists():
        try:
            owner = json.loads(marker.read_text()).get("profile")
        except (OSError, ValueError, AttributeError):
            raise ValueError(f"{marker} is unreadable; use another --out") from None
        if owner != name:
            raise ValueError(f"{out} holds a {owner!r} run; use another --out for {name!r}")
    else:
        marker.write_text(json.dumps({"profile": name}))


def run_sweep(out: Path, profile: dict, workers: int, deadline_s: float) -> bool:
    """Run every chunk that is missing or stale. Returns True when all are current."""
    config = OB.load_object_config()
    fingerprint = code_fingerprint()
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for leftover in chunk_dir.glob("*.tmp"):  # from a run killed mid-write
        leftover.unlink()
    todo = [
        c
        for c in plan(profile, config)
        if not chunk_is_current(chunk_dir / f"{c.name}.json", c, fingerprint)
    ]
    todo.sort(key=lambda c: -len(c.jobs))
    print(f"{len(todo)} chunks to run in {out} ({workers} workers)", flush=True)
    if not todo:
        return True
    start = time.perf_counter()
    done = 0
    if workers == 1:  # inline: no process pool (tests, debugging)
        for chunk in todo:
            if time.perf_counter() - start >= deadline_s:
                break
            name, n, wall = run_chunk(chunk, str(chunk_dir / f"{chunk.name}.json"), fingerprint)
            done += 1
            print(f"  [{done}/{len(todo)}] {name}: {n} attempts, {wall:.1f} s", flush=True)
        left = len(todo) - done
        if left:
            print(f"deadline reached: {left} chunks left; re-run the same command to continue")
        return left == 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < workers and time.perf_counter() - start < deadline_s:
                chunk = queue.pop(0)
                path = str(chunk_dir / f"{chunk.name}.json")
                pending[pool.submit(run_chunk, chunk, path, fingerprint)] = chunk.name
            if not pending:
                break
            finished = next(as_completed(pending))
            pending.pop(finished)
            name, n, wall = finished.result()
            done += 1
            print(f"  [{done}/{len(todo)}] {name}: {n} attempts, {wall:.1f} s", flush=True)
    left = len(todo) - done
    if left:
        print(f"deadline reached: {left} chunks left; re-run the same command to continue")
    return left == 0


def load_results(out: Path, profile: dict) -> list[dict]:
    """The rows of every chunk in the profile's plan. Raises ``ValueError`` naming
    the chunks that are missing or stale (other jobs, or other code)."""
    fingerprint = code_fingerprint()
    rows: list[dict] = []
    bad: list[str] = []
    for chunk in plan(profile, OB.load_object_config()):
        path = out / "chunks" / f"{chunk.name}.json"
        if not chunk_is_current(path, chunk, fingerprint):
            bad.append(chunk.name)
            continue
        rows.extend(json.loads(path.read_text())["results"])
    if bad:
        raise ValueError(f"{len(bad)} chunks missing or stale under {out}: {', '.join(bad)}")
    return rows


# ------------------------------------------------------------------ aggregation

WINDOW_RATE = 0.8  # a force is "in the window" for a class at >= 80 % success (T08: 8/10)


def _frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (95 % by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{lo:.2f}–{hi:.2f}" if n else "–"


def need_per_finger(spec: OB.ObjectSpec) -> float:
    """Hidden per-finger grip the lift needs at its peak acceleration, m(g + a)/(2μ)."""
    return spec.mass * (9.81 + LIFT_PEAK_ACC) / (2.0 * spec.friction[0])


def mcnemar_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar p-value: P(X >= b) for X ~ Binomial(b + c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(b, n + 1)) / 2.0**n


def summarize(rows: list[dict], profile_name: str) -> tuple[str, dict]:
    """Markdown tables and a metrics dict from the attempt rows."""
    config = OB.load_object_config()
    names = config.names
    lines: list[str] = []
    metrics: dict = {"profile": profile_name}
    add = lines.append

    sweep = [r for r in rows if r["study"] != "calibration"]
    wall = sum(r["wall_s"] for r in rows)
    sim = sum(r["sim_s"] for r in rows)
    metrics["attempts"] = len(rows)
    metrics["wall_s_per_attempt"] = wall / len(rows)
    metrics["realtime_factor"] = sim / wall
    add(
        f"Profile `{profile_name}`: {len(rows)} attempts, {sim:.0f} s simulated, "
        f"{wall:.0f} s of worker wall time ({wall / len(rows):.3f} s per attempt, "
        f"{sim / wall:.1f}x real time per worker)."
    )
    add("")

    # 1. Squeeze calibration.
    cal = [r for r in rows if r["study"] == "calibration"]
    if cal:
        add("### Squeeze calibration (settled per-finger force vs command)")
        add("")
        add("| command [N] | servo: mean ± std [min, max] | motor: mean ± std [min, max] |")
        add("|---|---|---|")
        metrics["calibration"] = {}
        for f in sorted({r["grip"] for r in cal}):
            cells = []
            for kind in ("servo", "motor"):
                got = np.array(
                    [r["settled_force"] / 2 for r in cal if r["kind"] == kind and r["grip"] == f]
                )
                metrics["calibration"].setdefault(kind, {})[f"{f:g}"] = {
                    "mean": float(got.mean()),
                    "std": float(got.std()),
                    "min": float(got.min()),
                    "max": float(got.max()),
                    "n": int(got.size),
                }
                cells.append(
                    f"{got.mean():.3f} ± {got.std():.3f} [{got.min():.3f}, {got.max():.3f}]"
                )
            add(f"| {f:g} | {cells[0]} | {cells[1]} |")
        add("")

    # 2. Success map.
    maps = [r for r in rows if r["study"] == "map"]
    forces = sorted({r["grip"] for r in maps})
    speeds = sorted({r["speed"] for r in maps})
    cell: dict[tuple[str, float, float], list[dict]] = {}
    for r in maps:
        cell.setdefault((r["cls"], r["grip"], r["speed"]), []).append(r)
    metrics["map"] = {}
    for v in speeds:
        add(f"### Success rate, closing speed {v:g} m/s (successes / attempts; B = breaks)")
        add("")
        add("| grip [N/finger] | " + " | ".join(names) + " | all classes |")
        add("|---|" + "---|" * (len(names) + 1))
        for f in forces:
            row = []
            total_k = total_n = 0
            for cls in names:
                rs = cell[(cls, f, v)]
                k = sum(r["success"] for r in rs)
                b = sum(r["broken"] for r in rs)
                total_k, total_n = total_k + k, total_n + len(rs)
                row.append(_frac(k, len(rs)) + (f" B{b}" if b else ""))
                metrics["map"][f"{cls}|{f:g}|{v:g}"] = {
                    "success": k,
                    "break": b,
                    "n": len(rs),
                    "outcomes": sorted(r["outcome"] for r in rs),
                }
            add(f"| {f:g} | " + " | ".join(row) + f" | {total_k / total_n:.0%} |")
        add("")

    add(f"### Success windows (grip forces with >= {WINDOW_RATE:.0%} success)")
    add("")
    add("| class | " + " | ".join(f"{v:g} m/s" for v in speeds) + " |")
    add("|---|" + "---|" * len(speeds))
    windows: dict = {}
    for cls in names:
        row = []
        for v in speeds:
            ok = [
                f
                for f in forces
                if np.mean([r["success"] for r in cell[(cls, f, v)]]) >= WINDOW_RATE
            ]
            windows[f"{cls}|{v:g}"] = ok
            row.append(f"{min(ok):g}–{max(ok):g} N ({len(ok)} forces)" if ok else "none")
        add(f"| {cls} | " + " | ".join(row) + " |")
    add("")
    metrics["windows"] = windows

    # The map has 5 cubes per cell, so its window edges are coarse. Score each edge
    # against the hidden need of many fresh cubes (no physics): the share of the
    # class that the lower edge holds and, for a breakable class, the share whose
    # break force the summed upper edge (2F) stays below, closing slowly (no
    # overshoot) and fast (times the speed study's median 0.5 m/s overshoot there).
    if speeds:
        v0 = speeds[0]
        fast_v = max(SPEED_TEST_SPEEDS)
        add(
            f"Window edges at {v0:g} m/s against {COVERAGE_SAMPLES} fresh cubes per class (no "
            f"physics; need = m(g + {LIFT_PEAK_ACC:.3f})/(2μ) per finger):"
        )
        add("")
        add(
            "| class | window [N] | cubes whose need <= lower edge | need: median, p95, max "
            f"| break force > 2 × upper edge: slow / fast ({fast_v:g} m/s) close |"
        )
        add("|---|---|---|---|---|")
        metrics["window_coverage"] = {}
        for cls in names:
            ok = windows[f"{cls}|{v0:g}"]
            specs = [sample_spec(config, cls, COVERAGE_FIRST + k) for k in range(COVERAGE_SAMPLES)]
            need = np.array([need_per_finger(s) for s in specs])
            breaks = np.array([s.break_force for s in specs if s.break_force is not None])
            lo = min(ok) if ok else None
            hi = max(ok) if ok else None
            held = float(np.mean(need <= lo)) if ok else None
            fast = [
                r["peak_force"] / (2 * hi)
                for r in rows
                if ok
                and r["study"] == "speed"
                and r["cls"] == cls
                and r["grip"] == hi
                and r["speed"] == fast_v
            ]
            over = float(np.median(fast)) if fast else None
            safe = float(np.mean(breaks > 2.0 * hi)) if ok and breaks.size else None
            safe_fast = (
                float(np.mean(breaks > 2.0 * hi * over))
                if over is not None and breaks.size
                else None
            )
            metrics["window_coverage"][cls] = {
                "window": [lo, hi],
                "held_by_lower_edge": held,
                "need_median": float(np.median(need)),
                "need_p95": float(np.percentile(need, 95)),
                "need_max": float(need.max()),
                "below_break_at_upper_edge_slow": safe,
                "fast_overshoot_at_upper_edge": over,
                "below_break_at_upper_edge_fast": safe_fast,
            }
            brk = "–"
            if safe is not None:
                brk = f"{safe:.1%} / " + (
                    f"{safe_fast:.1%} (× {over:.3f})" if safe_fast is not None else "–"
                )
            add(
                f"| {cls} | {f'{lo:g}–{hi:g}' if ok else 'none'} "
                f"| {'–' if held is None else f'{held:.1%}'} "
                f"| {np.median(need):.2f}, {np.percentile(need, 95):.2f}, {need.max():.2f} "
                f"| {brk} |"
            )
        add("")

    cells_fv = [(f, v) for f in forces for v in speeds]

    def rate(c: str, fv: tuple[float, float]) -> float:
        return float(np.mean([r["success"] for r in cell[(c, *fv)]]))

    worst = max(min(rate(c, fv) for c in names) for fv in cells_fv)
    served = {fv: [c for c in names if rate(c, fv) >= WINDOW_RATE] for fv in cells_fv}
    most = max(len(v) for v in served.values())
    widest = [fv for fv in cells_fv if len(served[fv]) == most]
    missed = sorted({c for fv in widest for c in names if c not in served[fv]})
    pooled = max(cells_fv, key=lambda fv: np.mean([rate(c, fv) for c in names]))
    pooled_rate = float(np.mean([rate(c, pooled) for c in names]))
    metrics["single_force"] = {
        "best_worst_class_rate": worst,
        "most_classes_served": most,
        "cells_serving_most": [list(fv) for fv in widest],
        "classes_missed_there": missed,
    }
    metrics["best_pooled"] = {"grip": pooled[0], "speed": pooled[1], "rate": pooled_rate}
    verdict = (
        f"**No single grip force succeeds on all {len(names)} classes.**"
        if most < len(names)
        else f"**One (force, speed) cell succeeds on all {len(names)} classes.**"
    )
    add(
        f"{verdict} Over the {len(cells_fv)} "
        f"(force, speed) cells the best worst-class success rate is {worst:.0%}. The most "
        f"classes one cell serves at >= {WINDOW_RATE:.0%} is {most} ("
        + ", ".join(f"{f:g} N at {v:g} m/s" for f, v in widest[:4])
        + (", ..." if len(widest) > 4 else "")
        + f"), and those cells miss {', '.join(missed) or 'no class'}. The best single "
        "(force, speed) over "
        f"all classes pooled is {pooled[0]:g} N at {pooled[1]:g} m/s with {pooled_rate:.0%} "
        "success (a preview of T10's `best_fixed`)."
    )
    add("")

    # 3. Closing speed on fragile_soft (paired: the same cubes at both speeds).
    fast_slow = [r for r in rows if r["study"] == "speed"]
    if fast_slow:
        slow, fast = min(SPEED_TEST_SPEEDS), max(SPEED_TEST_SPEEDS)
        add(f"### Closing speed on fragile_soft: breaks at {slow:g} vs {fast:g} m/s (same cubes)")
        add("")
        add(
            "| grip [N/finger] | slow breaks | fast breaks | fast-only | slow-only "
            "| McNemar p (one-sided) | peak/command slow | peak/command fast |"
        )
        add("|---|---|---|---|---|---|---|---|")
        by = {(r["grip"], r["speed"], r["sample"]): r for r in fast_slow}
        metrics["speed_effect"] = {}
        tot_b = tot_c = tot_slow = tot_fast = tot_n = 0
        for f in sorted({r["grip"] for r in fast_slow}):
            samples = sorted({r["sample"] for r in fast_slow if r["grip"] == f})
            s_b = [by[(f, slow, k)]["broken"] for k in samples]
            f_b = [by[(f, fast, k)]["broken"] for k in samples]
            b = sum(fb and not sb for sb, fb in zip(s_b, f_b, strict=True))
            c = sum(sb and not fb for sb, fb in zip(s_b, f_b, strict=True))
            ratio_s = np.median([by[(f, slow, k)]["peak_force"] / (2 * f) for k in samples])
            ratio_f = np.median([by[(f, fast, k)]["peak_force"] / (2 * f) for k in samples])
            p = mcnemar_one_sided(b, c)
            n = len(samples)
            tot_b, tot_c = tot_b + b, tot_c + c
            tot_slow, tot_fast, tot_n = tot_slow + sum(s_b), tot_fast + sum(f_b), tot_n + n
            metrics["speed_effect"][f"{f:g}"] = {
                "n": n,
                "slow_breaks": sum(s_b),
                "fast_breaks": sum(f_b),
                "fast_only": b,
                "slow_only": c,
                "p": p,
                "median_peak_ratio_slow": float(ratio_s),
                "median_peak_ratio_fast": float(ratio_f),
            }
            add(
                f"| {f:g} | {_frac(sum(s_b), n)} | {_frac(sum(f_b), n)} | {b} | {c} | {p:.2g} "
                f"| {ratio_s:.3f} | {ratio_f:.3f} |"
            )
        # The same cube appears at every force, so the pooled pairs are not
        # independent: test per cube (a sign test on its fast-only vs slow-only count).
        cubes = sorted({r["sample"] for r in fast_slow})
        forces_sp = sorted({r["grip"] for r in fast_slow})
        lean = [
            sum(
                by[(f, fast, k)]["broken"] - by[(f, slow, k)]["broken"]
                for f in forces_sp
                if (f, fast, k) in by and (f, slow, k) in by
            )
            for k in cubes
        ]
        n_fast, n_slow = sum(x > 0 for x in lean), sum(x < 0 for x in lean)
        p_cube = mcnemar_one_sided(n_fast, n_slow)
        metrics["speed_effect"]["pooled"] = {
            "n": tot_n,
            "slow_breaks": tot_slow,
            "fast_breaks": tot_fast,
            "fast_only": tot_b,
            "slow_only": tot_c,
            "cubes": len(cubes),
            "cubes_fast_worse": n_fast,
            "cubes_slow_worse": n_slow,
            "p_cube_sign_test": p_cube,
        }
        add(
            f"| all | {_frac(tot_slow, tot_n)} | {_frac(tot_fast, tot_n)} | {tot_b} | {tot_c} "
            "| – | | |"
        )
        add("")
        add(
            f"Per cube ({len(cubes)} cubes, each at every force): {n_fast} broke at more forces "
            f"when closing fast, {n_slow} when closing slow; sign test p = {p_cube:.2g} "
            "(one-sided)."
        )
        add("")

    # 4. Slip detector vs ground truth (attempts that reached the lift). The headline
    # is over the noise-free attempts; the noise study's noisy levels are separate.
    reached = [r for r in sweep if not r["broken"]]
    # Only the full profile's noise cubes (NOISE_FIRST_FULL) are held out: the quick
    # profile's (NOISE_FIRST_QUICK) were run when the debounce was chosen.
    full_noise = {r["sample"] for r in sweep if r["study"] == "noise"} <= set(
        range(NOISE_FIRST_FULL, NOISE_FIRST_FULL + NOISE_SAMPLES)
    )
    noise_label = "held-out" if full_noise else "noise-study"
    noise_note = "new cubes" if full_noise else "cubes seen when the debounce was chosen"
    lifted = [r for r in reached if not r.get("noise_level")]

    def online(r: dict) -> bool:
        return bool(r["ft_slip"])

    def gt_line(r: dict) -> bool:
        return bool(r["gt_slip"])

    def score(
        rs: list[dict],
        fires: Callable[[dict], bool] = online,
        gt: Callable[[dict], bool] = gt_line,
    ) -> dict:
        tp = sum(gt(r) and fires(r) for r in rs)
        fp = sum(fires(r) and not gt(r) for r in rs)
        fn = sum(gt(r) and not fires(r) for r in rs)
        return {
            "n": len(rs),
            "gt": sum(gt(r) for r in rs),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": len(rs) - tp - fp - fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
        }

    head = score(lifted)
    lags = np.array(
        [r["ft_slip_time"] - r["gt_slip_time"] for r in lifted if r["gt_slip"] and r["ft_slip"]]
    )
    metrics["slip_detector"] = {
        **head,
        "lag_median_s": float(np.median(lags)) if lags.size else None,
        "lag_p10_s": float(np.percentile(lags, 10)) if lags.size else None,
        "lag_p90_s": float(np.percentile(lags, 90)) if lags.size else None,
    }
    add(
        f"### F/T slip detector vs ground truth ({len(lifted)} noise-free attempts that "
        "reached the lift)"
    )
    add("")
    add("| | GT slip | GT no slip |")
    add("|---|---|---|")
    add(f"| F/T fired | {head['tp']} (TP) | {head['fp']} (FP) |")
    add(f"| F/T silent | {head['fn']} (FN) | {head['tn']} (TN) |")
    add("")
    metrics["slip_detector"]["precision_ci95"] = wilson(head["tp"], head["tp"] + head["fp"])
    metrics["slip_detector"]["recall_ci95"] = wilson(head["tp"], head["tp"] + head["fn"])
    p_ci = _ci(head["tp"], head["tp"] + head["fp"])
    r_ci = _ci(head["tp"], head["tp"] + head["fn"])
    add(
        f"Precision **{head['precision']:.3f}** (95 % CI {p_ci}), "
        f"recall **{head['recall']:.3f}** (95 % CI {r_ci}). "
        + (
            f"Detection time minus the GT time: median {np.median(lags) * 1000:.0f} ms "
            f"(p10 {np.percentile(lags, 10) * 1000:.0f} ms, p90 "
            f"{np.percentile(lags, 90) * 1000:.0f} ms)."
            if lags.size
            else ""
        )
    )
    add("")
    add(
        "By study (map and speed: lift-off failures; release: mid-air slips; margin: grips "
        f"at 0.96-1.06x the need, partial slips; {noise_label}: {noise_note}, noise-free; 95 % "
        "Wilson intervals):"
    )
    add("")
    add(
        "| study | attempts | GT slips | TP | FP | FN | precision (95 % CI) "
        "| recall (95 % CI) | median lag |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    metrics["slip_detector"]["by_study"] = {}
    groups = [(st, [r for r in lifted if r["study"] == st]) for st in ("map", "speed", "release")]
    groups += [("margin", [r for r in lifted if r["study"] == "margin"])]
    groups += [(noise_label, [r for r in lifted if r["study"] == "noise"])]
    for study, rs in groups:
        if not rs:
            continue
        sc = score(rs)
        s_lag = [r["ft_slip_time"] - r["gt_slip_time"] for r in rs if r["gt_slip"] and r["ft_slip"]]
        lag = f"{np.median(s_lag) * 1000:.0f} ms" if s_lag else "–"
        prec = f"{sc['precision']:.3f} ({_ci(sc['tp'], sc['tp'] + sc['fp'])})"
        rec = f"{sc['recall']:.3f} ({_ci(sc['tp'], sc['tp'] + sc['fn'])})"
        if sc["tp"] + sc["fp"] == 0:
            prec = "–"
        if sc["tp"] + sc["fn"] == 0:
            rec = "–"
        add(
            f"| {study} | {sc['n']} | {sc['gt']} | {sc['tp']} | {sc['fp']} | {sc['fn']} "
            f"| {prec} | {rec} | {lag} |"
        )
        metrics["slip_detector"]["by_study"][study] = {
            **sc,
            "lag_median_s": float(np.median(s_lag)) if s_lag else None,
        }
    add("")
    partial = [r for r in lifted if r["gt_slip"] and r["rel_drop"] < 0.05]
    micro = [r for r in lifted if not r["gt_slip"] and r["rel_drop"] >= 0.001]
    metrics["slip_detector"]["partial"] = {
        "n": len(partial),
        "detected": sum(r["ft_slip"] for r in partial),
    }
    metrics["slip_detector"]["micro"] = {
        "n": len(micro),
        "fired": sum(r["ft_slip"] for r in micro),
    }
    fp_falls = [r["rel_drop"] for r in lifted if r["ft_slip"] and not r["gt_slip"]]
    add(
        f"Partial slips (GT slip, fall < 50 mm): {sum(r['ft_slip'] for r in partial)}/"
        f"{len(partial)} detected. Micro-slips below the GT line (fall 1-"
        f"{SLIP_GT_M * 1000:g} mm, counted as no slip): the detector fired on "
        f"{sum(r['ft_slip'] for r in micro)}/{len(micro)}."
        + (
            f" The false alarms' falls: {min(fp_falls) * 1000:.2f}-{max(fp_falls) * 1000:.2f} mm."
            if fp_falls
            else ""
        )
    )
    add("")
    # Most ground-truth slips are cubes that never left the table, and most
    # non-slips moved < 1 mm. The calls that are not obvious fell 1-50 mm.
    hard = [r for r in lifted if 0.001 <= r["rel_drop"] < 0.05]
    hs = score(hard)
    metrics["slip_detector"]["hard_subset"] = hs
    add(
        f"Hard subset (the {hs['n']} noise-free attempts whose cube fell 1–50 mm relative to "
        f"the fingers): TP {hs['tp']}, FP {hs['fp']}, FN {hs['fn']}; precision "
        f"{hs['precision']:.3f} ({_ci(hs['tp'], hs['tp'] + hs['fp'])}), recall "
        f"{hs['recall']:.3f} ({_ci(hs['tp'], hs['tp'] + hs['fn'])})."
    )
    add("")
    add("Ground-truth line (largest fall relative to the fingers that counts as a slip):")
    add("")
    add("| GT line [mm] | GT slips | TP | FP | FN | precision | recall |")
    add("|---|---|---|---|---|---|---|")
    metrics["slip_detector"]["gt_lines"] = {}
    for line_mm in (1.0, 2.0, SLIP_GT_M * 1000, 10.0):
        sc = score(lifted, gt=lambda r, m=line_mm / 1000: r["rel_drop"] > m)
        metrics["slip_detector"]["gt_lines"][f"{line_mm:g}"] = sc
        add(
            f"| {line_mm:g}{' (used)' if line_mm == SLIP_GT_M * 1000 else ''} | {sc['gt']} "
            f"| {sc['tp']} | {sc['fp']} | {sc['fn']} | {sc['precision']:.3f} "
            f"| {sc['recall']:.3f} |"
        )
    add("")
    noisy = [r for r in reached if r["study"] == "noise"]
    if noisy:
        add(
            f"Sensor noise (the {noise_label} cubes again, white Gaussian noise on F_z and a_z "
            "every tick):"
        )
        add("")
        add(
            "| σ F_z [N] | σ a_z [m/s²] | attempts | GT slips | TP | FP | FN | precision | recall |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        metrics["slip_detector"]["noise"] = {}
        for level, (sf, sa) in enumerate(NOISE_LEVELS):
            rs = [r for r in noisy if r["noise_level"] == level]
            if not rs:
                continue
            sc = score(rs)
            add(
                f"| {sf:g} | {sa:g} | {sc['n']} | {sc['gt']} | {sc['tp']} | {sc['fp']} "
                f"| {sc['fn']} | {sc['precision']:.3f} | {sc['recall']:.3f} |"
            )
            metrics["slip_detector"]["noise"][f"{sf:g},{sa:g}"] = sc
        add("")
    add(
        "Threshold sweep (exact: each attempt records its lowest debounced held-mass / peak "
        f"ratio; the detector uses {DROP_FRACTION:g}), noise-free attempts:"
    )
    add("")
    add("| drop fraction | TP | FP | FN | precision | recall |")
    add("|---|---|---|---|---|---|")
    metrics["slip_detector"]["sweep"] = {}
    for frac in (0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97):
        sc = score(
            lifted,
            lambda r, f=frac: r["ft_min_ratio"] is not None and r["ft_min_ratio"] < f,
        )
        add(
            f"| {frac:g} | {sc['tp']} | {sc['fp']} | {sc['fn']} | {sc['precision']:.3f} "
            f"| {sc['recall']:.3f} |"
        )
        metrics["slip_detector"]["sweep"][f"{frac:g}"] = sc
    add("")
    window = round(DEBOUNCE_S * CONTROL_HZ)
    add(
        f"Debounce window at the {DROP_FRACTION:g} threshold (the detector uses {window} ticks "
        f"= {DEBOUNCE_S * 1000:g} ms), precision / recall:"
    )
    add("")
    levels = [("noise-free", lifted)] + [
        (f"σ {sf:g} N, {sa:g} m/s²", [r for r in noisy if r["noise_level"] == lv])
        for lv, (sf, sa) in enumerate(NOISE_LEVELS)
        if lv > 0
    ]
    add("| window [ticks] | " + " | ".join(name for name, _ in levels) + " |")
    add("|---|" + "---|" * len(levels))
    metrics["slip_detector"]["windows"] = {}
    for w in WINDOW_SWEEP:
        cells = []
        for name, rs in levels:
            sc = score(
                rs,
                lambda r, w=w: (
                    (low := r["ft_min_ratio_by_window"][str(w)]) is not None and low < DROP_FRACTION
                ),
            )
            cells.append(f"{sc['precision']:.3f} / {sc['recall']:.3f}")
            metrics["slip_detector"]["windows"][f"{w}|{name}"] = sc
        add(f"| {w} | " + " | ".join(cells) + " |")
    at = score(
        lifted, lambda r: r["ft_min_ratio"] is not None and r["ft_min_ratio"] < DROP_FRACTION
    )
    if (at["tp"], at["fp"]) != (head["tp"], head["fp"]):
        raise AssertionError("threshold sweep disagrees with the online detector")
    add("")
    bins = (0.0, 0.001, SLIP_GT_M, 0.02, 0.05, math.inf)
    labels = (
        "< 1 mm",
        f"1–{SLIP_GT_M * 1000:g} mm",
        f"{SLIP_GT_M * 1000:g}–20 mm",
        "20–50 mm",
        ">= 50 mm",
    )
    add(
        "Largest fall of the cube relative to the fingertip center (ground-truth signal), "
        "by outcome:"
    )
    add("")
    add("| outcome | " + " | ".join(labels) + " |")
    add("|---|" + "---|" * len(labels))
    hist = {}
    for outcome in ("SUCCESS", "SLIP", "DROP"):
        drops = np.array([r["rel_drop"] for r in lifted if r["outcome"] == outcome])
        counts = np.histogram(drops, bins=bins)[0] if drops.size else np.zeros(len(labels), int)
        hist[outcome] = counts.tolist()
        add(f"| {outcome} | " + " | ".join(str(c) for c in counts) + " |")
    metrics["rel_drop_hist"] = {"bins_m": list(bins[:-1]), "counts": hist}
    add("")
    return "\n".join(lines), metrics


def make_figures(rows: list[dict], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = OB.load_object_config()
    names = config.names
    paths: list[Path] = []

    # Heat maps: success rate per class over (grip force, closing speed).
    maps = [r for r in rows if r["study"] == "map"]
    forces = sorted({r["grip"] for r in maps})
    speeds = sorted({r["speed"] for r in maps})
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 5.2), sharey=True)
    for ax, cls in zip(np.atleast_1d(axes), names, strict=True):
        grid = np.zeros((len(forces), len(speeds)))
        for i, f in enumerate(forces):
            for j, v in enumerate(speeds):
                rs = [r for r in maps if (r["cls"], r["grip"], r["speed"]) == (cls, f, v)]
                grid[i, j] = np.mean([r["success"] for r in rs])
                brk = sum(r["broken"] for r in rs)
                text = f"{sum(r['success'] for r in rs)}/{len(rs)}" + (f"\nB{brk}" if brk else "")
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.25 < grid[i, j] < 0.85 else "white",
                )
        ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i, f in enumerate(forces):  # hatch the cells where cubes broke
            for j, v in enumerate(speeds):
                if any(
                    r["broken"] for r in maps if (r["cls"], r["grip"], r["speed"]) == (cls, f, v)
                ):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="///",
                            edgecolor="tab:red",
                            lw=0,
                        )
                    )
        ax.set_title(cls, fontsize=10)
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds], fontsize=8)
        ax.set_xlabel("closing speed [m/s]")
        ax.set_yticks(range(len(forces)), [f"{f:g}" for f in forces], fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("grip force [N per finger]")
    fig.suptitle("Grasp-and-lift success (successes/attempts; B = broken)", fontsize=11)
    fig.tight_layout()
    paths.append(out / "heatmap.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    # Squeeze calibration.
    cal = [r for r in rows if r["study"] == "calibration"]
    if cal:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        for kind, color in (("servo", "tab:red"), ("motor", "tab:blue")):
            cmds = sorted({r["grip"] for r in cal if r["kind"] == kind})
            got = [
                [r["settled_force"] / 2 for r in cal if r["kind"] == kind and r["grip"] == f]
                for f in cmds
            ]
            ax.errorbar(
                cmds,
                [np.mean(g) for g in got],
                yerr=[[np.mean(g) - np.min(g) for g in got], [np.max(g) - np.mean(g) for g in got]],
                fmt="o-",
                color=color,
                capsize=3,
                label=f"{kind} (min-max over cubes)",
            )
        lim = [0.2, 60]
        ax.plot(lim, lim, "k--", lw=0.8, label="measured = command")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("squeeze command [N per finger]")
        ax.set_ylabel("settled finger force [N per finger]")
        ax.set_title("Squeeze calibration: menagerie servo vs force motor")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", lw=0.3)
        fig.tight_layout()
        paths.append(out / "calibration.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)

    # Closing speed on fragile_soft.
    sp = [r for r in rows if r["study"] == "speed"]
    if sp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for v, color in zip(
            sorted({r["speed"] for r in sp}), ("tab:green", "tab:orange"), strict=False
        ):
            fs = sorted({r["grip"] for r in sp})
            rate = [
                np.mean([r["broken"] for r in sp if r["speed"] == v and r["grip"] == f]) for f in fs
            ]
            ax1.plot(fs, rate, "o-", color=color, label=f"{v:g} m/s")
            ratio = [r["peak_force"] / (2 * r["grip"]) for r in sp if r["speed"] == v]
            ax2.hist(ratio, bins=30, alpha=0.6, color=color, label=f"{v:g} m/s")
        ax1.set_xlabel("grip force [N per finger]")
        ax1.set_ylabel("break rate")
        ax1.set_title("fragile_soft: break rate by closing speed")
        ax1.legend()
        ax2.set_xlabel("peak summed finger force / commanded sum")
        ax2.set_ylabel("attempts")
        ax2.set_title("force overshoot (until the break, if any)")
        ax2.legend()
        fig.tight_layout()
        paths.append(out / "speed_effect.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)

    # Slip detector: example traces.
    lifted = [r for r in rows if not r["broken"] and r["trace"]]
    examples = []
    picks = (
        ("held", lambda r: r["study"] == "map" and r["outcome"] == "SUCCESS"),
        ("lift-off drop", lambda r: r["study"] == "map" and r["outcome"] == "DROP"),
        ("mid-air release", lambda r: r["study"] == "release" and r["gt_slip"]),
        (
            "partial slip",
            lambda r: r["study"] == "margin" and r["gt_slip"] and r["rel_drop"] < 0.05,
        ),
    )
    for label, keep in picks:
        pick = [r for r in lifted if keep(r)]
        if pick:
            examples.append((label, max(pick, key=lambda r: r["hidden"]["mass"])))
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(4.4 * len(examples), 3.8))
        for ax, (label, r) in zip(np.atleast_1d(axes), examples, strict=True):
            t = np.array(r["trace"]["t"])
            ax.plot(t, r["trace"]["load"], color="tab:blue", label="F/T held mass × g [N]")
            ax.set_xlabel("time from lift start [s]")
            ax.set_ylabel("held mass × g [N]", color="tab:blue")
            twin = ax.twinx()
            twin.plot(
                t,
                1000 * np.array(r["trace"]["rel"]),
                color="tab:red",
                label="cube z rel. fingers [mm]",
            )
            twin.set_ylabel("cube z relative to fingers [mm]", color="tab:red")
            for when, style, lab in (
                (r["gt_slip_time"], "r:", "GT slip"),
                (r["ft_slip_time"], "b--", "F/T detection"),
            ):
                if when is not None:
                    ax.axvline(when, color=style[0], ls=style[1:], label=lab)
            ax.set_title(
                f"{label}: {r['cls']}, {r['grip']:.3g} N, m={r['hidden']['mass']:.2f} kg",
                fontsize=9,
            )
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        paths.append(out / "slip_traces.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)
    return paths


def _json_safe(value: object) -> object:
    """``value`` with every NaN (a rate over no attempts) written as ``None`` (strict JSON)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float | np.floating) and math.isnan(value):
        return None
    return value


def aggregate(out: Path, profile_name: str, profile: dict) -> int:
    try:
        rows = load_results(out, profile)
    except ValueError as exc:
        print(f"cannot aggregate: {exc}; run the sweep first", file=sys.stderr)
        return 1
    text, metrics = summarize(rows, profile_name)
    (out / "summary.md").write_text(text + "\n")
    (out / "metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=1, allow_nan=False, default=float)
    )
    figures = make_figures(rows, out)
    print(text)
    print("figures: " + ", ".join(str(p) for p in figures))
    return 0


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/spike_grip.py", description="T05 grip-force feasibility spike."
    )
    parser.add_argument("--quick", action="store_true", help="small sweep (< 5 min)")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 2)),
        help="worker processes (default min(8, cpus - 2))",
    )
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument(
        "--deadline",
        type=float,
        default=8.0,
        help="stop submitting chunks after N minutes (running chunks finish, < 1 min)",
    )
    parser.add_argument(
        "--aggregate", action="store_true", help="only aggregate existing chunk files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2
    if not (math.isfinite(args.deadline) and args.deadline >= 0):
        print("--deadline must be a finite number of minutes >= 0", file=sys.stderr)
        return 2
    name = "quick" if args.quick else "full"
    profile = QUICK if args.quick else FULL
    out = args.out if args.out is not None else DEFAULT_OUT / name
    try:
        out.mkdir(parents=True, exist_ok=True)
        claim_profile(out, name)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.aggregate:
        if not run_sweep(out, profile, args.workers, 60 * args.deadline):
            return 3
    return aggregate(out, name, profile)


if __name__ == "__main__":
    raise SystemExit(main())
