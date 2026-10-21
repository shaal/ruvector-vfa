"""Record one scripted grasp-and-lift with the T06 sensing stack, then plot its force profile.

ROADMAP T06. The attempt follows the T05 spike (``scripts/spike_grip.py``)::

    APPROACH above the cube → DESCEND (tare the wrist at rest) → SQUEEZE(f, speed)
    → LIFT 10 cm → HOLD 1 s

and runs, every control tick after ``control_step``: the break rule
(``BreakMonitor``), the sensors (``SensorSuite``), the event detector, and the
``EpisodeRecorder``. ``--open-at`` opens the gripper that many seconds into the
lift, so the cube drops. ``--heft`` first raises the cube that far and sets it
down again (``PROBE``, T08's touch-and-heft), with the slip watch on only while
it is carried. It is a T06 test harness: T08's ``GraspLiftSkill`` replaces the
script.

Usage::

    uv run python scripts/record_episode.py --png runs/t06/force_profile.png
    uv run python scripts/record_episode.py --class fragile_soft --grip 8 --png runs/t06/break.png
    uv run python scripts/record_episode.py --open-at 0.6 --noise realistic --png runs/t06/drop.png

Writes ``<out>/<name>.npz``, appends ``<out>/episodes.jsonl``, and prints the events.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from vfa.control.gripper import MAX_CLOSE_SPEED, MIN_CLOSE_SPEED, Gripper
from vfa.control.osc import CONTROL_HZ, OperationalSpaceController, control_step
from vfa.sim import objects as OB
from vfa.sim import scene as S
from vfa.sim.recorder import Episode, EpisodeRecorder
from vfa.sim.sensors import (
    REALISTIC_NOISE,
    Event,
    EventDetector,
    SensorConfig,
    SensorNoise,
    SensorSuite,
    read_privileged,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / CONTROL_HZ
APPROACH_HEIGHT = 0.12  # ee above the cube center before the descent [m]
APPROACH_S = 1.4
DESCEND_S = 1.0
DESCEND_SETTLE_S = 0.3  # its last TARE_S is the tare window
TARE_S = 0.1
SQUEEZE_SETTLE_S = 0.15  # hold after the grasp closes and the motor reaches the force
SQUEEZE_MARGIN_S = 1.0  # squeeze timeout beyond the nominal close + ramp time (T05)
RAMP_N_PER_S_PER_MPS = 250.0  # the gripper ramps the squeeze at 250·speed N/s per finger
LIFT_M = 0.10
LIFT_S = 1.0
HOLD_S = 1.0
# Heft (T08's touch-and-heft probe: 1-2 cm, ARCHITECTURE §4). Below 1 cm the arm's
# payload sag (≈ 6.5 mm/kg) can leave a heavy cube barely off the table; it settles
# back during the hold and the slip detector reads a false SLIP (5 mm on heavy_rigid:
# 8 of 10 cubes). The rise takes at least sqrt(10·h) s, so its peak acceleration
# (5.77·h/T² on a minimum-jerk path) is no harsher than the lift's 0.58 m/s²; it holds
# HEFT_HOLD_S at the top.
MIN_HEFT_M = 0.01
MAX_HEFT_M = 0.02
HEFT_MIN_S = 0.3
HEFT_HOLD_S = 0.2
# Set-down: lower at HEFT_LOWER_SPEED until the table has carried the load (held load
# below HEFT_SET_DOWN of the heft's) for HEFT_HOLD_S, at most HEFT_PRESS_M below the
# grasp pose (the cube can slide up in the grip during the heft). Then the lift starts.
HEFT_LOWER_SPEED = 0.02  # m/s
HEFT_PRESS_M = 0.005
HEFT_SET_DOWN = 0.2
CARRY_MIN_N = 0.15  # a heft that holds less than this lost the cube: no set-down, no lift
SUCCESS_RISE_M = 0.08  # T05: the cube center ends >= 8 cm above its rest height
SLIP_RISE_M = 0.02  # T05: a failed attempt above this is a SLIP, else a DROP
# Hand-picked grips [N per finger] inside each class's T05 success window
# (docs/FEASIBILITY.md §3), all closed at 0.1 m/s.
DEFAULT_GRIP = {
    "light_rigid": 4.0,
    "heavy_rigid": 16.0,
    "slippery": 16.0,
    "fragile_soft": 1.5,
    "heavy_slippery": 20.0,
}
DEFAULT_SPEED = 0.1
NOISE = {"none": SensorNoise(), "realistic": REALISTIC_NOISE}
MAX_GRIP_N = S.GRIPPER_FORCE_LIMIT / 2.0  # the force motor's limit, per finger
MAX_SAMPLE = 10**9


def _min_jerk(s: float) -> tuple[float, float]:
    """Minimum-jerk blend and its rate d/ds at ``s`` in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10.0 - 15.0 * s + 6.0 * s * s), 30.0 * s * s * (1.0 - s) ** 2


def _path(osc, start: np.ndarray, goal: np.ndarray, t: float, duration: float, yaw: float) -> None:
    """Target on a minimum-jerk path, with its velocity fed forward (ARCHITECTURE §3)."""
    blend, rate = _min_jerk(t / duration)
    vel = (goal - start) * (rate / duration)
    osc.set_target(start + (goal - start) * blend, yaw=yaw, vel=(*vel, 0.0, 0.0, 0.0))


def run_scripted_grasp(
    cls: str = "light_rigid",
    sample: int = 0,
    grip: float | None = None,
    speed: float = DEFAULT_SPEED,
    open_at: float | None = None,
    noise: SensorNoise | None = None,
    seed: int = 0,
    model: mujoco.MjModel | None = None,
    config: OB.ObjectConfig | None = None,
    heft: float | None = None,
) -> Episode:
    """One scripted attempt on the ``sample``-th cube of class ``cls``, fully recorded.

    The cube comes from ``np.random.default_rng((class index, sample))`` (the T05
    convention). ``open_at`` [s from the lift start] opens the gripper mid-lift.
    ``noise`` (default none) draws from ``np.random.default_rng(seed)``.
    ``heft`` [m, 1-2 cm] raises the cube that far and sets it down before the lift.
    """
    if heft is not None and not (MIN_HEFT_M <= heft <= MAX_HEFT_M):
        raise ValueError(f"heft must be in [{MIN_HEFT_M}, {MAX_HEFT_M}] m, got {heft!r}")
    if not MIN_CLOSE_SPEED <= speed <= MAX_CLOSE_SPEED:
        raise ValueError(f"speed must be in [{MIN_CLOSE_SPEED}, {MAX_CLOSE_SPEED}] m/s")
    if open_at is not None and not (math.isfinite(open_at) and open_at >= 0):
        raise ValueError(f"open_at must be >= 0, got {open_at!r}")
    config = config if config is not None else OB.load_object_config()
    spec = OB.sample_object(
        config.classes[cls], np.random.default_rng((config.names.index(cls), sample))
    )
    grip = DEFAULT_GRIP[cls] if grip is None else grip
    if not (math.isfinite(grip) and 0 <= grip <= MAX_GRIP_N):
        raise ValueError(f"grip must be in [0, {MAX_GRIP_N:g}] N per finger, got {grip!r}")
    model = model if model is not None else S.build_scene()
    data = mujoco.MjData(model)
    ids = S.scene_ids(model)
    S.reset_to_home(model, data)
    OB.apply_object(model, data, spec)
    osc = OperationalSpaceController(model, data, ids=ids)
    gripper = Gripper(model, data, ids)
    monitor = OB.BreakMonitor(model, data, spec, ids)
    noise = noise if noise is not None else SensorNoise()
    sensors = SensorSuite(
        model, data, ids, SensorConfig(noise=noise), rng=np.random.default_rng(seed)
    )
    events = EventDetector()
    hidden = spec.hidden()
    meta = {
        "class": cls,
        "sample": sample,
        "seed": seed,
        "control_hz": CONTROL_HZ,
        "noise": {
            "force": noise.force,
            "torque": noise.torque,
            "accel": noise.accel,
            "touch": noise.touch,
        },
        "size": spec.size,
        "hidden": hidden,
        "open_at": open_at,
    }
    seen = OB.observe_cube(model, data, ids=ids)
    context = {
        "cube": {"x": seen.x, "y": seen.y, "z": seen.z, "yaw": seen.yaw, "size": seen.size},
        "arm_qpos": data.qpos[ids.arm_qpos].tolist(),
    }
    recorder = EpisodeRecorder(
        meta=meta, params={"grip_force": grip, "close_speed": speed}, context=context
    )
    cube = np.array([seen.x, seen.y, seen.z])
    yaw = seen.yaw

    def tick(phase: str):
        control_step(model, data, (osc, gripper), 1)
        monitor.update()
        frame = sensors.read()
        priv = read_privileged(model, data, ids, monitor)
        recorder.add_events(events.update(frame, phase, priv))
        recorder.record(frame, phase, priv)
        return frame

    wall0 = time.perf_counter()
    osc.set_target(cube + (0.0, 0.0, APPROACH_HEIGHT), yaw=yaw)
    for _ in range(round(APPROACH_S / DT)):
        tick("APPROACH")
    above = np.array(osc.target.pos)
    for k in range(round(DESCEND_S / DT)):
        _path(osc, above, cube, k * DT, DESCEND_S, yaw)
        tick("DESCEND")
    osc.set_target(cube, yaw=yaw)
    for _ in range(round((DESCEND_SETTLE_S - TARE_S) / DT)):
        tick("DESCEND")
    sensors.begin_tare()
    for _ in range(round(TARE_S / DT)):
        tick("DESCEND")
    sensors.end_tare()

    t_squeeze = float(data.time)
    nominal = (gripper.width() - spec.size) / speed + grip / (RAMP_N_PER_S_PER_MPS * speed)
    gripper.squeeze(grip, speed=speed)
    ready_at: float | None = None
    while data.time < t_squeeze + nominal + SQUEEZE_MARGIN_S and not monitor.broken:
        tick("SQUEEZE")
        if ready_at is None and events.closed and gripper.actuator_squeeze() >= 0.99 * grip:
            ready_at = float(data.time)
        if ready_at is not None and data.time >= ready_at + SQUEEZE_SETTLE_S:
            break

    lost = False  # the heft lost the cube: no set-down, no lift
    if heft is not None and not monitor.broken:
        base = np.array(osc.target.pos)
        top = base + (0.0, 0.0, heft)
        rise_s = max(HEFT_MIN_S, math.sqrt(10.0 * heft))
        sensors.watch_slip()  # carried from here ...
        for k in range(round(rise_s / DT)):
            _path(osc, base, top, k * DT, rise_s, yaw)
            tick("PROBE")
        osc.set_target(top, yaw=yaw)
        for _ in range(round(HEFT_HOLD_S / DT)):
            held = tick("PROBE").load
        sensors.unwatch_slip()  # ... until it goes back down: the table is not a slip
        lost = not held >= CARRY_MIN_N
        down, floor = top.copy(), base[2] - HEFT_PRESS_M
        quiet, quiet_ticks = 0, round(HEFT_HOLD_S / DT)
        budget = round(((heft + HEFT_PRESS_M) / HEFT_LOWER_SPEED + 2.0 * HEFT_HOLD_S) / DT)
        for _ in range(0 if lost else budget):
            if quiet == 0 and down[2] > floor:
                down[2] = max(floor, down[2] - HEFT_LOWER_SPEED * DT)
                osc.set_target(down, yaw=yaw)
            frame = tick("PROBE")
            quiet = quiet + 1 if frame.load < HEFT_SET_DOWN * held else 0
            if quiet >= quiet_ticks:
                break

    start = np.array(osc.target.pos)
    t0 = float(data.time)
    if not lost:
        sensors.watch_slip()
    opened = open_at is None
    while not (lost or monitor.broken) and data.time < t0 + LIFT_S + HOLD_S:
        t = float(data.time) - t0
        if not opened and t >= open_at:
            gripper.open()
            opened = True
        _path(osc, start, start + (0.0, 0.0, LIFT_M), t, LIFT_S, yaw)
        tick("LIFT" if t < LIFT_S else "HOLD")

    rise = float(data.xpos[ids.cube_body, 2]) - seen.z
    if monitor.broken:
        result = "BREAK"
    elif not any(e.name is Event.CONTACT_ONSET for e in recorder.events):
        result = "MISS"  # the fingers never touched the cube
    elif rise >= SUCCESS_RISE_M:
        result = "SUCCESS"
    elif rise >= SLIP_RISE_M:
        result = "SLIP"
    else:
        result = "DROP"
    return recorder.finish(
        result,
        rise=rise,
        peak_cube_force=monitor.peak_force,
        wall_s=time.perf_counter() - wall0,
        sim_s=float(data.time),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--class", dest="cls", default="light_rigid", help="object class")
    parser.add_argument("--sample", type=int, default=0, help="cube sample index")
    parser.add_argument("--grip", type=float, default=None, help="squeeze per finger [N]")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="closing speed [m/s]")
    parser.add_argument("--open-at", type=float, default=None, help="open s into the lift")
    parser.add_argument("--heft", type=float, default=None, help="probe: raise and set down [m]")
    parser.add_argument("--noise", choices=sorted(NOISE), default="none", help="sensor noise")
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "t06", help="output dir")
    parser.add_argument("--name", default=None, help="episode file name (no suffix)")
    parser.add_argument("--png", type=Path, default=None, help="force-profile PNG path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from vfa.viz.force_plot import plot_force_profile

    parser = build_parser()
    args = parser.parse_args(argv)
    config = OB.load_object_config()
    if args.cls not in config.names:
        parser.error(f"--class must be one of {config.names}")
    if args.open_at is not None and not (math.isfinite(args.open_at) and args.open_at >= 0):
        parser.error("--open-at must be >= 0")
    if args.heft is not None and not (MIN_HEFT_M <= args.heft <= MAX_HEFT_M):
        parser.error(f"--heft must be in [{MIN_HEFT_M}, {MAX_HEFT_M}] m")
    name = args.name
    if name is not None and (
        Path(name).name != name or name in ("", ".", "..") or len(name.encode()) > 200
    ):
        parser.error("--name must be a plain file name (no directories, at most 200 bytes)")
    if not (0 <= args.sample < MAX_SAMPLE and args.seed >= 0):
        parser.error(f"--sample must be in [0, {MAX_SAMPLE}) and --seed >= 0")
    if args.grip is not None and not (math.isfinite(args.grip) and 0 <= args.grip <= MAX_GRIP_N):
        parser.error(f"--grip must be in [0, {MAX_GRIP_N:g}] N")
    if not MIN_CLOSE_SPEED <= args.speed <= MAX_CLOSE_SPEED:
        parser.error(f"--speed must be in [{MIN_CLOSE_SPEED}, {MAX_CLOSE_SPEED}] m/s")
    episode = run_scripted_grasp(
        args.cls,
        args.sample,
        args.grip,
        args.speed,
        args.open_at,
        NOISE[args.noise],
        args.seed,
        config=config,
        heft=args.heft,
    )
    name = args.name or f"{args.cls}_s{args.sample}_{episode.result.lower()}"
    path = episode.save(args.out / name)
    print(f"{episode.result}: rise {100 * episode.outcome['rise']:.1f} cm, {episode.n} ticks")
    for e in episode.events:
        print(f"  {e.t:7.3f} s  {e.name.value:<14} {e.info}")
    print(f"saved {path} (+ {path.parent / 'episodes.jsonl'})")
    if args.png is not None:
        print(f"plot {plot_force_profile(episode, args.png)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
