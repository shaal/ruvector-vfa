"""Sensors, the F/T slip detector, and the event detector (T06).

Everything System 1 perceives, read once per control tick after ``control_step``,
and everything the memory later embeds (ARCHITECTURE §3, §5.1).

**Sensor suite** (``SensorSuite.read`` → ``SensorFrame``). All readings come from
the scene's sensor block (``vfa.sim.scene.SENSOR_NAMES``), copied once per tick.
After ``control_step`` they describe the state at the start of the last physics
step, one step (2 ms) before ``data.time``; the frame carries ``data.time``, the
same stamp as ``BreakMonitor.break_time``. No extra ``mj_forward`` is needed.

- **Wrist wrench**, world axes (the F/T reading rotated by the hand's forward
  kinematics), torque about the ``wrist_ft`` site. ``wrench_raw`` is untared;
  ``wrench`` is tared (minus the mean over a tare window) and low-passed. The
  tare is exact while the hand keeps the orientation it was taken in (any yaw of
  the top-down hand: the hand's weight stays on the yaw axis). It removes the
  hand's weight, not its inertia: a lift at 0.58 m/s² adds the hand's 0.44 N.
- **Held load**: the vertical wrist force over the accelerometer's vertical
  proper acceleration, minus the hand's mass (``SlipDetector``, T05). It removes
  the hand's inertia, so it is the weight of what the fingers carry.
- **Finger pads**: each ``touch`` sensor's normal force, low-passed, and a
  contact flag with hysteresis (on above ``contact_on_n``, off below
  ``contact_off_n``). Touch counts any object (the table too) and under-reads
  fingertip-end contacts (ARCHITECTURE §3): it is the observable signal; the
  privileged cube-only force is ``PrivilegedState.cube_finger_force``.
- **Gripper width** and its rate (the tendon length over the tendon coupling),
  and the **ee pose and velocity** in the world frame (encoders: noise-free).
- **Noise** (``SensorNoise``): white Gaussian noise on each wrist force, torque,
  and accelerometer axis and each pad force, drawn from a seeded
  ``np.random.Generator`` every tick. ``SensorNoise()`` is noise-free;
  ``REALISTIC_NOISE`` models a mid-range wrist F/T plus a MEMS IMU (refer to
  its comment).

**Slip detector** (``SlipDetector``, promoted from the T05 spike): fires once the
low-passed held mass stays below ``drop_fraction`` of its running peak for
``debounce_s``, after the peak passed ``arm_load_n``. It arms and fires only
between ``SensorSuite.watch_slip()`` (a heft or lift starts) and
``unwatch_slip()`` (the hand sets the object down). T06 added, with noise
only, a margin and a window: the drop must also exceed ``noise_k`` (4)
standard deviations of the filtered estimate, from the sensor's noise spec at
the tick rate, and the peak covers the last 0.5 s. Without noise the detector
is exactly T05's (holdout precision 0.918, recall 0.957); under
``REALISTIC_NOISE`` it scores 0.982 and 0.954, where T05's scores 0.708 and
0.886 (docs/FEASIBILITY.md §8, ``scripts/size_slip_detector.py``).
``T05_SLIP`` keeps the spike's values (no margin).

**Events** (``EventDetector``): ``CONTACT_ONSET``, ``GRASP_CLOSED``, ``SLIP``,
``DROP``, ``BREAK``, ``LIFTED``, ``PLACED``, ``SAFETY_STOP``, each an
``EventRecord`` with a timestamp. The first four come from the sensor frame
alone (System 1 can react to them); ``BREAK``, ``LIFTED``, and ``PLACED`` read
the privileged object state (scoring and logging); ``SAFETY_STOP`` is reported
by the safety supervisor (T09) through ``EventDetector.report``. The rules are
in ``EventDetector``'s docstring and ARCHITECTURE §4.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np

from vfa.sim.objects import REST_SINK_BAND, BreakMonitor
from vfa.sim.scene import (
    GRIPPER_TENDON,
    TABLE_TOP_Z,
    SceneIds,
    _real,
    finger_cube_forces,
    scene_ids,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------- config


def _nonneg(name: str, value: object) -> float:
    result = _real(name, value)
    if result < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _pos(name: str, value: object) -> float:
    result = _real(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _put(obj: object, name: str, value: object) -> None:
    object.__setattr__(obj, name, value)  # frozen dataclasses: normalize once


@dataclass(slots=True, frozen=True)
class SensorNoise:
    """White Gaussian sensor noise: σ per axis, drawn every control tick. 0 = noise-free."""

    force: float = 0.0  # each wrist force axis [N]
    torque: float = 0.0  # each wrist torque axis [N·m]
    accel: float = 0.0  # each wrist accelerometer axis [m/s²]
    touch: float = 0.0  # each fingertip pad force [N]

    def __post_init__(self) -> None:
        for name in ("force", "torque", "accel", "touch"):
            _put(self, name, _nonneg(f"noise.{name}", getattr(self, name)))

    @property
    def active(self) -> bool:
        return self.force > 0 or self.torque > 0 or self.accel > 0 or self.touch > 0


# A mid-range wrist F/T sensor and a MEMS IMU, sampled at the 500 Hz control rate.
# Force 0.05 N: between the ATI Mini45 (SI-145-5) Fz resolution, 1/16 N after
# dropping four counts of noise, and the Robotiq FT 300-S signal noise, 0.1 N
# (1 s standard deviation, product sheet). Torque 0.003 N·m: the FT 300-S
# figure. Accelerometer 0.03 m/s²: a Bosch BMI088's 175 µg/√Hz over a 250 Hz
# bandwidth. Pads 0.05 N: assumed of the same class as the wrist force.
REALISTIC_NOISE = SensorNoise(force=0.05, torque=0.003, accel=0.03, touch=0.05)


@dataclass(slots=True, frozen=True)
class SlipConfig:
    """``SlipDetector`` parameters: T05's detector plus T06's noise margin.

    With noise-free sensors the detector is exactly T05's (FEASIBILITY §5). With
    noise, the drop must also exceed ``noise_k`` standard deviations of the
    filtered held-load estimate, computed from the sensor's noise spec at the
    current tick rate, and the peak is the largest estimate of the last
    ``peak_window_s`` (FEASIBILITY §8 sized ``noise_k`` at 500 Hz).
    """

    cutoff_hz: float = 20.0  # first-order low-pass on the held-mass estimate
    drop_fraction: float = 0.9  # fire below this fraction of the running peak ...
    noise_k: float = 4.0  # ... minus noise_k σ of the filtered estimate (with noise)
    debounce_s: float = 0.006  # the drop must hold this long (at least 2 ticks)
    arm_load_n: float = 0.15  # the running peak must pass this weight first [N]
    # With noise: the peak covers only the last peak_window_s. The maximum of a
    # noisy estimate keeps growing the longer it runs (a 50 g cube held 20 s
    # false-alarmed on 13 of 40 seeds with an all-time peak); a slip's dip
    # develops within ≈ 0.3 s of its peak, so 0.5 s keeps it.
    peak_window_s: float = 0.5

    def __post_init__(self) -> None:
        _put(self, "cutoff_hz", _pos("slip.cutoff_hz", self.cutoff_hz))
        _put(self, "noise_k", _nonneg("slip.noise_k", self.noise_k))
        _put(self, "peak_window_s", _pos("slip.peak_window_s", self.peak_window_s))
        _put(self, "debounce_s", _nonneg("slip.debounce_s", self.debounce_s))
        _put(self, "arm_load_n", _pos("slip.arm_load_n", self.arm_load_n))
        frac = _real("slip.drop_fraction", self.drop_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"slip.drop_fraction must be in (0, 1), got {self.drop_fraction!r}")
        _put(self, "drop_fraction", frac)


# The T05 spike's detector (docs/FEASIBILITY.md §5). Without noise, SlipConfig()
# behaves exactly like it.
T05_SLIP = SlipConfig(
    cutoff_hz=20.0, drop_fraction=0.9, noise_k=0.0, debounce_s=0.006, arm_load_n=0.15
)


@dataclass(slots=True, frozen=True)
class SensorConfig:
    """``SensorSuite`` parameters. The slip detector gets the wrist force and
    accelerometer noise of ``noise``, so its margin follows the sensor."""

    noise: SensorNoise = SensorNoise()
    wrench_cutoff_hz: float = 20.0  # first-order low-pass on the tared wrench
    touch_cutoff_hz: float = 50.0  # first-order low-pass on the pad forces
    contact_on_n: float = 0.2  # a pad's filtered force above this turns its contact on [N]
    contact_off_n: float = 0.1  # ... and below this turns it off [N]
    slip: SlipConfig = SlipConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.noise, SensorNoise):
            raise TypeError(f"noise must be a SensorNoise, got {type(self.noise).__name__}")
        if not isinstance(self.slip, SlipConfig):
            raise TypeError(f"slip must be a SlipConfig, got {type(self.slip).__name__}")
        _put(self, "wrench_cutoff_hz", _pos("wrench_cutoff_hz", self.wrench_cutoff_hz))
        _put(self, "touch_cutoff_hz", _pos("touch_cutoff_hz", self.touch_cutoff_hz))
        _put(self, "contact_on_n", _pos("contact_on_n", self.contact_on_n))
        _put(self, "contact_off_n", _pos("contact_off_n", self.contact_off_n))
        if not self.contact_off_n < self.contact_on_n:
            raise ValueError("contact_off_n must be below contact_on_n (hysteresis)")


# ------------------------------------------------------------- filter, slip


class LowPass:
    """First-order low-pass filter (exponential smoothing) at a fixed sample rate.

    Works on floats and numpy arrays. The first sample initializes the state. A
    non-finite input (a dropped sample) leaves its lane unchanged; a lane whose
    state is not finite restarts at the next finite sample.
    """

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * _pos("cutoff_hz", cutoff_hz) * _pos("dt", dt))
        self.value: float | np.ndarray | None = None

    def __call__(self, x):
        v = self.value
        if v is None:
            self.value = x.copy() if isinstance(x, np.ndarray) else x
        elif isinstance(x, np.ndarray):
            new = v + self.alpha * (x - v)
            if not np.isfinite(new).all():
                ok = np.isfinite(x)
                new = np.where(ok, np.where(np.isfinite(v), new, x), v)
            self.value = new
        elif math.isfinite(x):
            self.value = v + self.alpha * (x - v) if math.isfinite(v) else x
        return self.value

    def noise_gain(self) -> float:
        """Standard deviation of the output over that of white-noise input."""
        return math.sqrt(self.alpha / (2.0 - self.alpha))

    def reset(self) -> None:
        self.value = None


# Valid vertical proper acceleration [m/s²] for the mass estimate; outside it
# (near free fall, or an absurd reading) the sample is skipped.
MIN_PROPER_ACC = 1.0
MAX_PROPER_ACC = 100.0


class SlipDetector:
    """Causal slip detector on the wrist F/T and accelerometer (no privileged signals).

    ``tare(f_z, a_z)`` stores the hand's own mass ``f_z / a_z`` (the caller
    averages both with the hand at rest before contact). Each
    ``update(t, f_z, a_z)`` takes the world-vertical wrist force [N] and the
    wrist's vertical proper acceleration [m/s², +g at rest]. The held mass
    ``f_z / a_z − m_hand`` is low-passed at ``config.cutoff_hz``; ``load`` and
    ``peak`` report it as a weight [N]. In a slip the cube accelerates less
    than the hand, so the estimate falls. The detector fires (and latches; refer
    to ``rearm``) once the estimate stays below ``drop_fraction`` of its running
    peak, minus a noise margin, for ``debounce_s`` (at least two ticks), after
    the peak passed ``arm_load_n``. A non-finite or out-of-range sample is
    skipped.

    **Noise** (T06). ``noise = (σ_F, σ_a)`` is the white noise on the two inputs
    [N, m/s²] (the sensor's spec). Each tick the estimate's own noise is
    ``σ = sqrt(σ_F² + ((m_hand + m)·σ_a)²) · sqrt(α / (2 − α))`` [N] (``α`` the
    filter's gain at this tick rate), and the detector compares
    ``(load + noise_k·σ) / peak`` with ``drop_fraction``; the peak is the largest
    estimate of the last ``peak_window_s`` (of valid samples). Noise lifts the maximum of the
    estimate by a few σ, more the longer it runs, and a bare 10 % band on a
    50 g cube is about 2.5σ under ``REALISTIC_NOISE``. With ``noise = (0, 0)``
    there is no margin and no window: the detector is T05's.

    ``min_ratio`` is the lowest debounced value of that statistic (``None``
    before arming): the detector fires at any threshold above it, so one run
    scores every threshold exactly; ``track_windows`` (debounce windows in
    ticks) records the same for other windows.

    ``watching=False`` estimates the load but neither arms nor fires until
    ``watch()``; ``unwatch()`` stops it again (before the hand sets the object
    down). The closing fingers can drag the cube on the table and load the
    wrist by more than ``arm_load_n`` for a moment (0.23 N on a
    ``heavy_slippery`` cube at 0.1 m/s), which an armed detector reads as a
    slip. The T05 spike fed the detector from the lift on only.
    """

    def __init__(
        self,
        dt: float,
        g: float = 9.81,
        config: SlipConfig | None = None,
        track_windows: tuple[int, ...] = (),
        watching: bool = True,
        noise: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.config = config if config is not None else SlipConfig()
        self.watching = bool(watching)
        self.dt = _pos("dt", dt)
        self.g = _pos("g", g)
        self.noise = (_nonneg("noise σ_F", noise[0]), _nonneg("noise σ_a", noise[1]))
        self.hand_mass = 0.0
        self.filter = LowPass(self.config.cutoff_hz, self.dt)
        # Margin per unit of σ, in kg of estimate: noise_k · gain / g.
        self._margin = self.config.noise_k * self.filter.noise_gain() / self.g
        self._noisy = self._margin > 0 and (self.noise[0] > 0 or self.noise[1] > 0)
        # With noise the peak is a sliding maximum: (tick, mass) pairs, decreasing.
        self._span = max(1, round(self.config.peak_window_s / self.dt))
        self._maxq: deque[tuple[int, float]] = deque()
        self._tick = 0
        self.mass = 0.0
        self.peak_mass = 0.0
        self.fired_at: float | None = None
        self.min_ratio: float | None = None
        if any(not isinstance(w, int) or w < 1 for w in track_windows):
            raise ValueError(f"track_windows must be ints >= 1, got {track_windows!r}")
        # T05's 6 ms is 3 ticks at 500 Hz; below 167 Hz it would round to a single
        # sample, and one noisy sample is not a drop.
        self._window = max(2, round(self.config.debounce_s / self.dt))
        if self._noisy and self._span <= self._window:
            raise ValueError(
                f"peak_window_s ({self.config.peak_window_s} s) must span more ticks than the "
                f"debounce ({self._window} ticks of {self.dt} s), or the detector cannot fire"
            )
        self._recent: deque[float] = deque(maxlen=max(self._window, *track_windows, 1))
        self.min_ratio_by_window: dict[int, float | None] = dict.fromkeys(track_windows)

    @property
    def tare_n(self) -> float:
        return self.hand_mass * self.g

    @property
    def load(self) -> float:
        return self.mass * self.g

    @property
    def peak(self) -> float:
        return self.peak_mass * self.g

    @property
    def armed(self) -> bool:
        return self.watching and self.peak >= self.config.arm_load_n

    def sigma(self) -> float:
        """The filtered estimate's noise σ at the current mass [N] (0 without noise)."""
        sf, sa = self.noise
        x = (self.hand_mass + self.mass) * sa
        return math.sqrt(sf * sf + x * x) * self.filter.noise_gain()

    def tare(self, f_z: float, a_z: float) -> None:
        if not (math.isfinite(f_z) and MIN_PROPER_ACC <= a_z <= MAX_PROPER_ACC):
            raise ValueError(
                f"tare needs finite readings with the hand at rest (a_z in "
                f"[{MIN_PROPER_ACC}, {MAX_PROPER_ACC}] m/s²), got {f_z!r}, {a_z!r}"
            )
        self.hand_mass = f_z / a_z

    def update(self, t: float, f_z: float, a_z: float) -> bool:
        if math.isfinite(f_z) and MIN_PROPER_ACC <= a_z <= MAX_PROPER_ACC:
            self.mass = self.filter(f_z / a_z - self.hand_mass)
            if not self.watching:
                return False
            if self._noisy:
                self._tick += 1
                q = self._maxq
                while q and q[-1][1] <= self.mass:
                    q.pop()
                q.append((self._tick, self.mass))
                while q[0][0] <= self._tick - self._span:
                    q.popleft()
                self.peak_mass = q[0][1]
            elif self.mass > self.peak_mass:
                self.peak_mass = self.mass
            if self.peak >= self.config.arm_load_n:
                recent = self._recent
                if self._noisy:
                    sf, sa = self.noise
                    x = (self.hand_mass + self.mass) * sa
                    margin = self._margin * math.sqrt(sf * sf + x * x)
                    recent.append((self.mass + margin) / self.peak_mass)
                else:
                    recent.append(self.mass / self.peak_mass)
                for w, low in self.min_ratio_by_window.items():
                    if len(recent) >= w:
                        ratio = max(list(recent)[-w:])
                        if low is None or ratio < low:
                            self.min_ratio_by_window[w] = ratio
                if len(recent) >= self._window:
                    # below the line for the whole window
                    if len(recent) == self._window:
                        debounced = max(recent)
                    else:
                        debounced = max(list(recent)[-self._window :])
                    if self.min_ratio is None or debounced < self.min_ratio:
                        self.min_ratio = debounced
                    if self.fired_at is None and debounced < self.config.drop_fraction:
                        self.fired_at = t
        return self.fired_at is not None

    def rearm(self) -> None:
        """Clear the latch and restart the peak at the current estimate (T08's slip reflex).

        A slide that continues at the same rate holds the estimate steady and does
        not fire again; a further drop of the same size does. ``min_ratio`` keeps
        the lowest value of the whole attempt.
        """
        self.fired_at = None
        self.peak_mass = self.mass
        self._maxq.clear()
        self._maxq.append((self._tick, self.mass))
        self._recent.clear()

    def watch(self) -> None:
        """Start (or restart) arming and firing from the current estimate."""
        self.watching = True
        self.rearm()

    def unwatch(self) -> None:
        """Stop arming and firing (the hand is about to set the object down). The
        load estimate keeps running; a fired detector stays latched."""
        self.watching = False


# ----------------------------------------------------------------- the suite


@dataclass(slots=True)
class SensorFrame:
    """Every observable signal of one control tick (world frame; refer to the module doc)."""

    t: float  # data.time; the readings describe the state one physics step earlier [s]
    wrench_raw: np.ndarray  # (6,) untared wrist force [N] and torque [N·m], with noise
    wrench: np.ndarray  # (6,) tared and low-passed; NaN before the tare
    acc: np.ndarray  # (3,) wrist proper acceleration [m/s²] (+g up at rest), with noise
    finger_force: np.ndarray  # (2,) left, right pad normal force [N], low-passed
    contact: np.ndarray  # (2,) bool: left, right pad contact (hysteresis)
    width: float  # gripper opening [m]
    width_rate: float  # its rate of change [m/s]; < 0 closing
    ee_pos: np.ndarray  # (3,) [m]
    ee_quat: np.ndarray  # (4,) w, x, y, z
    ee_vel: np.ndarray  # (6,) linear [m/s], angular [rad/s]
    load: float  # held load (weight of what the fingers carry) [N]; NaN before the tare
    peak_load: float  # its peak [N] (with noise: of the last peak_window_s); NaN untared
    armed: bool  # the slip detector watches and its peak passed the arming weight
    watching: bool  # the slip detector watches (from watch_slip(), the lift or heft)
    watch_id: int  # counts watch_slip() calls: a new value starts a new carry (0: none yet)
    slip: bool  # the slip detector has fired (latched until rearm)
    slip_time: float | None  # when it fired [s]


class SensorSuite:
    """The tared, filtered sensor readings of one episode, bound to ``(model, data)``.

    Call ``read()`` once per control tick, after ``control_step`` (a second call
    at the same ``data.time`` returns the same frame). Tare the wrist with the
    hand at rest before contact: ``begin_tare()``, a few ticks of ``read()``
    (T05 used 0.1 s), then ``end_tare()``. Before the tare the tared wrench and
    the held load are NaN. Call ``watch_slip()`` when the hand starts to carry
    the object (a heft, a lift): only then can the slip detector (and ``DROP``)
    arm and fire; call ``unwatch_slip()`` before the hand sets it down.
    ``reset()`` starts a new episode (it keeps the noise generator's state);
    ``read`` raises if time went back without one. ``substeps`` must match
    ``control_step``'s, so the filters know the tick length: ``read`` raises if
    consecutive reads are not one tick apart (call it every tick). Noise needs
    a seeded ``rng``: same seed, same readings. A non-finite reading (a dropped
    sample) leaves the filters where they were.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ids: SceneIds | None = None,
        config: SensorConfig | None = None,
        rng: np.random.Generator | None = None,
        substeps: int = 1,
    ) -> None:
        self.model = model
        self.data = data
        self.ids = ids if ids is not None else scene_ids(model)
        self.config = config if config is not None else SensorConfig()
        if not isinstance(self.config, SensorConfig):
            raise TypeError(f"config must be a SensorConfig, got {type(self.config).__name__}")
        noise = self.config.noise
        if noise.active and not isinstance(rng, np.random.Generator):
            raise ValueError("sensor noise > 0 needs a numpy Generator rng")
        self.rng = rng
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"substeps must be an int >= 1, got {substeps!r}")
        self.dt = substeps * float(model.opt.timestep)
        # Weights convert at the model's gravity; a gravity-free model (some tests)
        # has no held load to speak of, and its samples fail the a_z range check.
        self.g = float(np.linalg.norm(model.opt.gravity)) or 9.81
        ids = self.ids
        if ids.ee_angvel_adr != ids.ee_linvel_adr + 3:
            raise ValueError("ee_linvel and ee_angvel must be adjacent in sensordata")
        self._f = slice(ids.wrist_force_adr, ids.wrist_force_adr + 3)
        self._tau = slice(ids.wrist_torque_adr, ids.wrist_torque_adr + 3)
        self._acc = slice(ids.wrist_acc_adr, ids.wrist_acc_adr + 3)
        self._touch = (ids.left_touch_adr, ids.right_touch_adr)
        self._noise_idx = np.r_[
            np.arange(self._f.start, self._f.stop),
            np.arange(self._tau.start, self._tau.stop),
            np.arange(self._acc.start, self._acc.stop),
            np.array(self._touch),
        ]
        self._sigma = np.repeat([noise.force, noise.torque, noise.accel, noise.touch], [3, 3, 3, 2])
        tendon = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, GRIPPER_TENDON)
        adr, num = int(model.tendon_adr[tendon]), int(model.tendon_num[tendon])
        coefs = model.wrap_prm[adr : adr + num]
        if num != 2 or not np.allclose(coefs, coefs[0]) or coefs[0] <= 0:
            raise ValueError("gripper tendon must couple the two fingers with one positive coef")
        self._width_per_length = 1.0 / float(coefs[0])
        self.reset()

    def reset(self) -> None:
        """Forget the tare, the filters, the slip detector, and the contact state."""
        cfg = self.config
        self.tared = False
        self.tare_time: float | None = None
        self.tare_wrench = np.zeros(6)
        self._tare_samples: list[np.ndarray] | None = None
        self._wrench_lp = LowPass(cfg.wrench_cutoff_hz, self.dt)
        self._touch_lp = LowPass(cfg.touch_cutoff_hz, self.dt)
        self.slip = self._new_slip()
        self.watch_id = 0
        self._contact = np.zeros(2, dtype=bool)
        self._last: SensorFrame | None = None

    # ------------------------------------------------------------------ tare

    def begin_tare(self) -> None:
        """Start averaging the raw wrist wrench and accelerometer (hand empty, at rest)."""
        if self.slip.watching:
            raise RuntimeError("the hand carries (watch_slip): unwatch_slip() before a re-tare")
        self._tare_samples = []

    def end_tare(self) -> None:
        """Set the tare from the samples since ``begin_tare`` and restart the slip detector.

        Non-finite samples (dropped readings) are left out of the window.
        Validates first: on an error (no finite samples, a vertical acceleration
        outside the at-rest range) the previous tare stays in place and the tare
        window stays open.
        """
        if not self._tare_samples:
            raise RuntimeError("end_tare needs at least one finite read() after begin_tare()")
        mean = np.mean(self._tare_samples, axis=0)
        if not np.isfinite(mean).all():
            raise ValueError(f"tare window has non-finite readings (mean {mean.tolist()})")
        slip = self._new_slip()
        slip.tare(float(mean[2]), float(mean[8]))
        self.slip = slip
        self._tare_samples = None
        self.tare_wrench = mean[:6]
        self._wrench_lp.reset()
        self.tared = True
        self.tare_time = float(self.data.time)

    def _new_slip(self) -> SlipDetector:
        noise = self.config.noise
        return SlipDetector(
            self.dt, self.g, self.config.slip, watching=False, noise=(noise.force, noise.accel)
        )

    def watch_slip(self) -> None:
        """Start (or restart) the slip detector: call when the hand starts to carry
        the object (a heft or a lift). Until then the held load is reported, but the
        detector neither arms nor fires, and ``DROP`` cannot fire either. Each call
        starts a new carry (``SensorFrame.watch_id``), which resets the event
        detector's ``DROP`` state."""
        if not self.tared or self.taring:
            raise RuntimeError("tare the wrist (begin_tare/end_tare) before watch_slip()")
        self.slip.watch()
        self.watch_id += 1

    def unwatch_slip(self) -> None:
        """Stop the slip detector before the hand sets the object down (a heft that
        ends, a place): the table taking the load is not a slip or a drop."""
        self.slip.unwatch()

    @property
    def taring(self) -> bool:
        return self._tare_samples is not None

    # ------------------------------------------------------------------ read

    def read(self) -> SensorFrame:
        """The frame of this tick (call after ``control_step``)."""
        data = self.data
        t = float(data.time)
        last = self._last
        if last is not None:
            if t == last.t:
                return last
            if t < last.t:
                raise RuntimeError(f"time went back ({last.t} -> {t}); call reset() first")
            if abs(t - last.t - self.dt) > 1e-6 * self.dt:
                raise RuntimeError(
                    f"read() {t - last.t:.6g} s after the last one, but a tick is {self.dt:g} s: "
                    "call it every control tick, with substeps matching control_step "
                    "(reset() to start over)"
                )
        sd = data.sensordata.copy()
        if self.config.noise.active:
            sd[self._noise_idx] += self._sigma * self.rng.standard_normal(self._sigma.size)
        rot = data.site_xmat[self.ids.wrist_site].reshape(3, 3)
        wrench_raw = np.concatenate((rot @ sd[self._f], rot @ sd[self._tau]))
        acc = rot @ sd[self._acc]
        if self._tare_samples is not None:
            sample = np.concatenate((wrench_raw, acc))
            if np.isfinite(sample).all():
                self._tare_samples.append(sample)

        cfg = self.config
        touch = self._touch_lp(sd[list(self._touch)]).copy()
        contact = self._contact
        contact = np.where(contact, touch > cfg.contact_off_n, touch > cfg.contact_on_n)
        self._contact = contact

        if self.tared:
            wrench = self._wrench_lp(wrench_raw - self.tare_wrench).copy()
            slip = self.slip
            slip.update(t, float(wrench_raw[2]), float(acc[2]))
            load, peak, armed = slip.load, slip.peak, slip.armed
            watching, fired = slip.watching, slip.fired_at
        else:
            wrench = np.full(6, np.nan)
            load = peak = math.nan
            armed = watching = False
            fired = None

        ids = self.ids
        frame = SensorFrame(
            t=t,
            wrench_raw=wrench_raw,
            wrench=wrench,
            acc=acc,
            finger_force=touch,
            contact=contact.copy(),
            width=float(sd[ids.gripper_pos_adr]) * self._width_per_length,
            width_rate=float(sd[ids.gripper_vel_adr]) * self._width_per_length,
            ee_pos=sd[ids.ee_pos_adr : ids.ee_pos_adr + 3],
            ee_quat=sd[ids.ee_quat_adr : ids.ee_quat_adr + 4],
            ee_vel=sd[ids.ee_linvel_adr : ids.ee_linvel_adr + 6],
            load=load,
            peak_load=peak,
            armed=armed,
            watching=watching,
            watch_id=self.watch_id,
            slip=fired is not None,
            slip_time=fired,
        )
        self._last = frame
        return frame


# ------------------------------------------------------------ privileged state


@dataclass(slots=True)
class PrivilegedState:
    """Ground truth of one tick, for scoring and logging only (never an observation)."""

    cube_pos: np.ndarray  # (3,) world [m], same instant as the sensor frame
    cube_quat: np.ndarray  # (4,) w, x, y, z
    cube_vel: np.ndarray  # (6,) linear [m/s], angular [rad/s], world
    cube_finger_force: np.ndarray  # (2,) left, right finger-to-cube normal force [N]
    broken: bool
    break_time: float | None
    rest_z: float  # cube center height when it rests on the table [m]
    place_xy: np.ndarray  # (2,) place target [m]


def read_privileged(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: SceneIds | None = None,
    monitor: BreakMonitor | None = None,
) -> PrivilegedState:
    """The cube's true state after ``control_step`` (and after ``monitor.update()``).

    Reads the kinematics of the same instant as the sensors (the start of the
    last physics step). With a ``monitor`` the finger forces are its last
    reading (no second walk over the contacts) and ``broken`` is its flag;
    without one they are computed here and the cube counts as unbroken.
    """
    ids = ids if ids is not None else scene_ids(model)
    body = ids.cube_body
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
    if monitor is not None:
        forces = monitor.last_forces
        broken, break_time = monitor.broken, monitor.break_time
    else:
        forces = finger_cube_forces(model, data, ids)
        broken, break_time = False, None
    return PrivilegedState(
        cube_pos=data.xpos[body].copy(),
        cube_quat=data.xquat[body].copy(),
        cube_vel=np.concatenate((vel[3:], vel[:3])),
        cube_finger_force=np.array(forces, dtype=float),
        broken=bool(broken),
        break_time=break_time,
        rest_z=TABLE_TOP_Z + float(model.geom_size[ids.cube_geom, 0]),
        place_xy=model.site_pos[ids.place_site, :2].copy(),
    )


# --------------------------------------------------------------------- events


class Event(StrEnum):
    """Event names, shared by the recorder, the env, and memory (ARCHITECTURE §4)."""

    CONTACT_ONSET = "CONTACT_ONSET"
    GRASP_CLOSED = "GRASP_CLOSED"
    SLIP = "SLIP"
    DROP = "DROP"
    BREAK = "BREAK"
    LIFTED = "LIFTED"
    PLACED = "PLACED"
    SAFETY_STOP = "SAFETY_STOP"


@dataclass(slots=True, frozen=True)
class EventRecord:
    """One event: its name, when it happened [s], and JSON-ready details."""

    name: Event
    t: float
    info: dict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EventConfig:
    """``EventDetector`` rules (refer to its docstring)."""

    grasp_stall_speed: float = 0.002  # |width rate| below this counts as stopped [m/s]
    grasp_settle_s: float = 0.01  # both pads on and stopped this long → GRASP_CLOSED
    rearm_s: float = 0.1  # both pads off this long ends a grasp
    drop_fraction: float = 0.3  # held load below this fraction of its carried peak ...
    drop_debounce_s: float = 0.02  # ... for this long → DROP
    drop_arm_n: float = 0.15  # ... once that peak passed this weight [N]
    lift_height: float = 0.02  # cube center this far above its rest height → LIFTED [m]
    place_tolerance: float = 0.02  # cube center within this xy distance of the target [m]
    place_tilt_deg: float = 10.0  # a face within this angle of flat
    place_speed: float = 0.01  # cube speed below this counts as resting [m/s]
    release_phases: frozenset[str] = frozenset({"PLACE", "RELEASE"})

    def __post_init__(self) -> None:
        for name in (
            "grasp_stall_speed",
            "rearm_s",
            "lift_height",
            "place_tolerance",
            "place_speed",
            "drop_arm_n",
        ):
            _put(self, name, _pos(name, getattr(self, name)))
        for name in ("grasp_settle_s", "drop_debounce_s"):
            _put(self, name, _nonneg(name, getattr(self, name)))
        frac = _real("drop_fraction", self.drop_fraction)
        if not 0.0 < frac < 1.0:
            raise ValueError(f"drop_fraction must be in (0, 1), got {self.drop_fraction!r}")
        _put(self, "drop_fraction", frac)
        tilt = _real("place_tilt_deg", self.place_tilt_deg)
        if not 0.0 < tilt < 45.0:
            raise ValueError(f"place_tilt_deg must be in (0, 45), got {self.place_tilt_deg!r}")
        _put(self, "place_tilt_deg", tilt)
        phases = self.release_phases
        if isinstance(phases, str):
            raise TypeError(f"release_phases must be a set of str, got {phases!r}")
        phases = tuple(phases)  # a generator would be used up by the check
        if not all(isinstance(p, str) for p in phases):
            raise TypeError(f"release_phases must be a set of str, got {phases!r}")
        _put(self, "release_phases", frozenset(str.__str__(p) for p in phases))


# Tolerance for "held for X s" checks on float tick times.
_EPS_T = 1e-9


class EventDetector:
    """Turns sensor frames (and, for three events, ground truth) into timestamped events.

    Call ``update(frame, phase, privileged)`` once per tick; it returns the new
    events (also kept in ``events``). Rules (``EventConfig`` defaults):

    - ``CONTACT_ONSET``: the first pad contact of a grasp (either finger). A
      grasp ends once both pads have been off for ``rearm_s`` (0.1 s), which
      re-arms ``CONTACT_ONSET`` and ``GRASP_CLOSED``. Brief losses inside a
      grasp (an impact bounce, stick-slip flicker) do not repeat it.
    - ``GRASP_CLOSED``: both pads in contact and the opening's rate below
      ``grasp_stall_speed`` (2 mm/s) for ``grasp_settle_s`` (10 ms): the
      fingers closed on the object and stopped. Once per grasp.
    - ``SLIP``: the slip detector fired (stamped with its firing time), during a
      carry. Again after each ``SlipDetector.rearm`` that fires again. Not in a
      ``release_phases`` phase, nor after one until the next carry.
    - ``DROP``: the held load stays below ``drop_fraction`` (0.3) of its carried
      peak for ``drop_debounce_s`` (20 ms), once that peak passed ``drop_arm_n``
      (0.15 N): the hand no longer carries the object (it fell out, or slid
      down to the table). The carried peak is the detector's own: the largest
      load of the carry (since its ``watch_slip``), so a slip reflex's
      ``rearm`` does not lower it. Once per carry: each ``watch_slip`` (``SensorFrame.watch_id``)
      restarts it. Not in a ``release_phases`` phase, nor after one until the
      next carry, so an intended release is not a drop.
    - ``BREAK`` (privileged): the break monitor's flag turned on (stamped with
      its ``break_time``).
    - ``LIFTED`` (privileged): with a pad in contact and the cube unbroken, its
      center rose ``lift_height`` (2 cm, T05's SLIP/DROP line) above its rest
      height: a lift-off. Again after the cube rested on the table (within
      ``REST_SINK_BAND`` of its rest height), so a heft that raises the cube's
      center 2 cm and the real lift each report one.
    - ``PLACED`` (privileged): after ``LIFTED``, the cube rests unbroken on the
      table (center within ``REST_SINK_BAND`` of the rest height, speed below
      ``place_speed``, a face within ``place_tilt_deg`` of flat), within
      ``place_tolerance`` of the place target, and no pad touches it. Once.
    - ``SAFETY_STOP``: reported by the safety supervisor (T09) via ``report``.

    A **carry** runs from a ``watch_slip()`` that comes after the grasp's
    ``CONTACT_ONSET`` until ``unwatch_slip()``; ``SLIP`` and ``DROP`` fire only
    during one. A watch left on from an earlier grasp (a failed carry, a
    re-grasp) does not count, so call ``unwatch_slip()`` when a carry ends,
    whatever its outcome (after ``DROP``, or once the grasp has ended:
    ``in_grasp`` False). Two cases silence a carry, both logged as a warning:
    a watch that starts before the grasp's first contact, and a loss of both
    pads for ``rearm_s`` mid-carry (a new grasp starts; call ``watch_slip()``
    again). A cube dropped onto the place target reports ``DROP`` and then
    ``PLACED``: scoring must not take ``PLACED`` alone as success.

    ``GRASP_CLOSED`` is geometric: the fingers stopped on the object. A slow
    close reaches it at the first touch, while the squeeze force still ramps
    (``Gripper.squeeze``: 250·v N/s per finger), so it does not mark the end of
    the force ramp. Frames must come in time order (``reset`` for a new
    episode); a repeated frame time is ignored.
    """

    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config if config is not None else EventConfig()
        if not isinstance(self.config, EventConfig):
            raise TypeError(f"config must be an EventConfig, got {type(self.config).__name__}")
        self._place_cos = math.cos(math.radians(self.config.place_tilt_deg))
        self.events: list[EventRecord] = []
        self.reset()

    def reset(self) -> None:
        """Start a new episode (forget every event and state)."""
        self.events.clear()
        self.in_grasp = False
        self.closed = False
        self._off_since: float | None = None
        self._stall_since: float | None = None
        self._slip_time: float | None = None
        self._low_since: float | None = None
        self._dropped = False
        self._released = False
        self._carry_peak = 0.0
        self._watch_id = 0
        self._onset_watch_id = 0  # watch_id at this grasp's CONTACT_ONSET
        self._last_t = -math.inf
        self.broken = False
        self.lifted = False  # lifted now (re-armed when the cube rests again)
        self.ever_lifted = False
        self.placed = False

    def report(self, name: Event | str, t: float, /, **info: object) -> EventRecord:
        """Add an event raised outside the detector (``SAFETY_STOP`` from T09)."""
        record = EventRecord(Event(name), _real("t", t), dict(info))
        self._emit(record)
        return record

    def _emit(self, record: EventRecord) -> None:
        self.events.append(record)
        channel = "[SAFETY]" if record.name is Event.SAFETY_STOP else "[S1]"
        details = " ".join(f"{k}={v}" for k, v in record.info.items())
        logger.info("%s event %s t=%.3f s %s", channel, record.name.value, record.t, details)

    def update(
        self,
        frame: SensorFrame,
        phase: str | None = None,
        privileged: PrivilegedState | None = None,
    ) -> list[EventRecord]:
        cfg = self.config
        if phase is not None and not isinstance(phase, str):
            raise TypeError(f"phase must be a str, got {type(phase).__name__}")
        start = len(self.events)
        t = frame.t
        if not t > self._last_t:
            if t == self._last_t:
                return []
            raise RuntimeError(f"frame time went back ({self._last_t} -> {t}); call reset()")
        self._last_t = t
        any_contact = bool(frame.contact[0] or frame.contact[1])
        both = bool(frame.contact[0] and frame.contact[1])

        # Grasp lifecycle: CONTACT_ONSET, re-arm after rearm_s without contact.
        if any_contact:
            self._off_since = None
            if not self.in_grasp:
                self.in_grasp = True
                self.closed = False
                self._stall_since = None
                # A carry counts only if it starts after this onset (watch_slip()
                # after the grasp): a watch left on from the last grasp is stale.
                self._onset_watch_id = frame.watch_id
                if frame.watching:
                    logger.warning(
                        "[S1] CONTACT_ONSET at t=%.3f s while the slip watch is on: SLIP and "
                        "DROP stay silent until the next watch_slip() (unwatch_slip() when a "
                        "carry ends)",
                        t,
                    )
                left, right = (float(f) for f in frame.finger_force)
                self._emit(EventRecord(Event.CONTACT_ONSET, t, {"left": left, "right": right}))
        elif self.in_grasp:
            if self._off_since is None:
                self._off_since = t
            if t - self._off_since >= cfg.rearm_s - _EPS_T:
                self.in_grasp = False

        if self.in_grasp and not self.closed:
            if both and abs(frame.width_rate) < cfg.grasp_stall_speed:
                if self._stall_since is None:
                    self._stall_since = t
                if t - self._stall_since >= cfg.grasp_settle_s - _EPS_T:
                    self.closed = True
                    info = {"width": frame.width, "force": float(np.sum(frame.finger_force))}
                    self._emit(EventRecord(Event.GRASP_CLOSED, t, info))
            else:
                self._stall_since = None

        if frame.watch_id != self._watch_id:  # a new carry: forget the last one's state
            self._watch_id = frame.watch_id
            self._carry_peak = 0.0
            self._dropped = self._released = False
            self._low_since = None
        if phase is not None and str.__str__(phase) in cfg.release_phases:
            self._released = True

        carrying = frame.watching and frame.watch_id > self._onset_watch_id
        if frame.slip_time is not None and frame.slip_time != self._slip_time:
            self._slip_time = frame.slip_time
            if carrying and not self._released:
                info = {"load": frame.load, "peak": frame.peak_load}
                self._emit(EventRecord(Event.SLIP, frame.slip_time, info))

        if carrying and frame.load > self._carry_peak:
            self._carry_peak = frame.load
        peak = self._carry_peak
        if (
            carrying
            and peak >= cfg.drop_arm_n
            and not (self._dropped or self._released)
            and frame.load < cfg.drop_fraction * peak
        ):
            if self._low_since is None:
                self._low_since = t
            if t - self._low_since >= cfg.drop_debounce_s - _EPS_T:
                self._dropped = True
                self._emit(EventRecord(Event.DROP, t, {"load": frame.load, "peak": peak}))
        else:
            self._low_since = None

        if privileged is not None:
            self._ground_truth(frame, privileged, any_contact)
        return self.events[start:]

    def _ground_truth(self, frame: SensorFrame, priv: PrivilegedState, any_contact: bool) -> None:
        cfg = self.config
        t = frame.t
        if priv.broken and not self.broken:
            self.broken = True
            stamp = t if priv.break_time is None else float(priv.break_time)
            info = {"force": float(np.sum(priv.cube_finger_force))}
            self._emit(EventRecord(Event.BREAK, stamp, info))
        if self.broken:
            return
        rise = float(priv.cube_pos[2]) - priv.rest_z
        if not self.lifted and any_contact and rise >= cfg.lift_height:
            self.lifted = self.ever_lifted = True
            self._emit(EventRecord(Event.LIFTED, t, {"rise": rise}))
        elif self.lifted and abs(rise) < REST_SINK_BAND and not self.placed:
            self.lifted = False  # back on the table: the next lift-off is a new LIFTED
        if self.ever_lifted and not self.placed and not any_contact:
            dist = float(np.hypot(*(priv.cube_pos[:2] - priv.place_xy)))
            resting = abs(rise) < REST_SINK_BAND and np.linalg.norm(priv.cube_vel[:3]) < (
                cfg.place_speed
            )
            if resting and dist <= cfg.place_tolerance and _flat(priv.cube_quat, self._place_cos):
                self.placed = True
                self._emit(EventRecord(Event.PLACED, t, {"distance": dist}))


def _flat(quat: np.ndarray, min_cos: float) -> bool:
    """True if one of the cube's face normals is within acos(min_cos) of world z."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return bool(np.max(np.abs(mat[6:9])) >= min_cos)  # row 2: the body axes' z components
