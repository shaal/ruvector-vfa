"""Throughput probe: wall time of a 10 s lift-length episode under OSC + gripper control.

Usage::

    uv run python -m vfa.control.throughput                 # 500, 250, 100 Hz
    uv run python -m vfa.control.throughput --hz 500 --repeats 5

The episode is a scripted grasp-and-lift of the default cube (0.1 kg, 5 cm):
move above the cube, descend, squeeze 1 N per finger, lift 10 cm, hold. The
targets move on minimum-jerk paths, so the controller tracks a moving target as
a skill will, and the contacts cost what a real grasp costs. It is a timing
workload, not a tuned grasp. With T03's pyramidal friction cones the cube crept
down in the grip (≈ 1.2 mm/s, from 9.7 cm at 6.5 s to 9.3 cm at 10 s); T05's
elliptic cones (impratio 10) hold it at 9.87 cm at 10 s, at the same wall
time. The loop runs at ``control_hz`` with torques held
across physics substeps. Only the control loop is timed (not the scene build).
``cpu_s`` close to ``wall_s`` shows the loop ran on one core.

ROADMAP T03 target: <= 1 s wall per 10 s of sim time (>= 10x real time).
The CLI bounds (``MAX_RATES`` × ``MAX_REPEATS`` × 60 s) keep one command
under about 4 minutes on this Mac (CLAUDE.md: <= 9 min per command).
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from vfa.control.gripper import Gripper
from vfa.control.osc import CONTROL_HZ, OperationalSpaceController, control_step, substeps_for
from vfa.sim.scene import _real, build_scene, finger_cube_forces, reset_to_home, scene_ids

EPISODE_SECONDS = 10.0
SQUEEZE_N = 1.0  # per finger; below the menagerie servo's 1.25 N cap on a 5 cm cube
LIFT_M = 0.10
CANDIDATE_HZ: tuple[int, ...] = (500, 250, 100)
MAX_SECONDS = 60.0
MAX_RATES = 8
MAX_REPEATS = 10

# (start time [s], duration [s], target ee offset from the cube center [m]).
# The gripper squeezes at SQUEEZE_TIME and holds the squeeze to the end.
_ABOVE = np.array([0.0, 0.0, 0.12])
_GRASP = np.array([0.0, 0.0, 0.0])
_LIFTED = np.array([0.0, 0.0, LIFT_M])
_SEGMENTS: tuple[tuple[float, float, np.ndarray], ...] = (
    (0.0, 1.5, _ABOVE),
    (2.0, 1.2, _GRASP),
    (4.5, 1.5, _LIFTED),
)
SQUEEZE_TIME = 3.5


@dataclass(slots=True)
class ThroughputResult:
    """One timed episode."""

    control_hz: float
    substeps: int
    sim_s: float
    wall_s: float
    cpu_s: float
    lift_m: float  # cube center rise at the end of the episode
    min_finger_force_n: float  # min over fingers of the cube contact force at the end

    @property
    def realtime_factor(self) -> float:
        return self.sim_s / self.wall_s


def run_lift_episode(
    control_hz: float = CONTROL_HZ,
    seconds: float = EPISODE_SECONDS,
    model: mujoco.MjModel | None = None,
) -> ThroughputResult:
    """Run the scripted lift for ``seconds`` of sim time at ``control_hz`` and time the loop.

    The sim time is rounded to whole control ticks (at least one); ``sim_s``
    reports what ran.
    """
    seconds = _real("seconds", seconds)
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be in (0, {MAX_SECONDS:g}], got {seconds!r}")
    model = model if model is not None else build_scene()
    substeps = substeps_for(model, control_hz)
    data = mujoco.MjData(model)
    ids = scene_ids(model)
    reset_to_home(model, data)
    dt = substeps * float(model.opt.timestep)
    ticks = max(1, round(seconds / dt))
    cube0 = data.xpos[ids.cube_body].copy()
    osc = OperationalSpaceController(model, data, ids=ids, control_hz=control_hz)
    gripper = Gripper(model, data, ids)
    start = data.site_xpos[ids.ee_site].copy()
    waypoints = [start] + [cube0 + offset for _, _, offset in _SEGMENTS]
    squeezing = False

    wall0, cpu0 = time.perf_counter(), time.process_time()
    for k in range(ticks):
        t = k * dt
        osc.set_target(_scripted_target(t, waypoints))
        if not squeezing and t >= SQUEEZE_TIME:
            gripper.squeeze(SQUEEZE_N)
            squeezing = True
        control_step(model, data, (osc, gripper), substeps)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0

    mujoco.mj_forward(model, data)
    forces = finger_cube_forces(model, data, ids)
    return ThroughputResult(
        control_hz=float(control_hz),
        substeps=substeps,
        sim_s=ticks * dt,
        wall_s=wall,
        cpu_s=cpu,
        lift_m=float(data.xpos[ids.cube_body, 2] - cube0[2]),
        min_finger_force_n=float(min(forces)),
    )


def _scripted_target(t: float, waypoints: list[np.ndarray]) -> np.ndarray:
    """Target at time ``t``: minimum-jerk blend between consecutive waypoints."""
    target = waypoints[0]
    for (t0, duration, _), goal in zip(_SEGMENTS, waypoints[1:], strict=True):
        if t <= t0:
            break
        s = min(1.0, (t - t0) / duration)
        blend = s**3 * (10.0 - 15.0 * s + 6.0 * s * s)
        target = target + blend * (goal - target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vfa.control.throughput",
        description="Time a 10 s lift-length episode under OSC + gripper control.",
    )
    parser.add_argument(
        "--hz", type=int, nargs="+", default=list(CANDIDATE_HZ), help=f"<= {MAX_RATES} rates"
    )
    parser.add_argument("--seconds", type=float, default=EPISODE_SECONDS, help="sim time per run")
    parser.add_argument(
        "--repeats", type=int, default=3, help=f"runs per rate, best kept (1..{MAX_REPEATS})"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= MAX_REPEATS:
        parser.error(f"--repeats must be in [1, {MAX_REPEATS}], got {args.repeats}")
    if len(args.hz) > MAX_RATES:
        parser.error(f"--hz takes at most {MAX_RATES} rates, got {len(args.hz)}")
    if not (math.isfinite(args.seconds) and 0 < args.seconds <= MAX_SECONDS):
        parser.error(f"--seconds must be in (0, {MAX_SECONDS:g}], got {args.seconds}")
    model = build_scene()
    try:
        for hz in args.hz:
            substeps_for(model, hz)
    except ValueError as exc:
        parser.error(str(exc))
    dt = model.opt.timestep
    print(f"physics dt {dt} s, best of {args.repeats} runs per rate")
    print("control_hz  substeps  sim_s  wall_s  cpu_s  x_realtime  lift_cm  finger_N")
    for hz in args.hz:
        runs = [run_lift_episode(hz, args.seconds, model) for _ in range(args.repeats)]
        r = min(runs, key=lambda run: run.wall_s)
        print(
            f"{r.control_hz:>10g}  {r.substeps:>8}  {r.sim_s:5.2f}  {r.wall_s:6.3f}  "
            f"{r.cpu_s:5.3f}  {r.realtime_factor:10.1f}  {100 * r.lift_m:7.2f}  "
            f"{r.min_finger_force_n:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
