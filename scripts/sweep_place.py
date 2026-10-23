"""Place physics sweep (T19): class × place_speed × release_height at hand-picked grips.

Runs ``PickPlaceSkill`` with a fixed θ (no System 2; the realistic sensor noise)
on ``--cubes`` cubes per class, drawn from ``numpy.random.default_rng((77, k))``,
and records per attempt the result and, in the release phases, the peak
table-to-cube and finger-to-cube normal forces and the peak tared wrist force.
It prints one row per class and release height: per place speed, the
successes, the place breaks, and the largest table force [N]. docs/RESULTS.md
§10 quotes it.

    uv run python scripts/sweep_place.py --workers 8      # 360 attempts, about 1 min
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

# The hand-picked lift fields per class (grip [N per finger], lift acceleration), as
# tests/test_pick_place.py's HAND_PICKED.
GRIPS = {
    "light_rigid": (2.0, 0.3),
    "heavy_rigid": (10.0, 0.3),
    "slippery": (12.0, 0.3),
    "fragile_soft": (1.5, 0.3),
    "heavy_slippery": (15.0, 0.3),
    "brittle": (3.0, 0.3),
}


def one(job: tuple[str, int, float, float]) -> dict:
    """One attempt: (class, cube k, place_speed, release_height) → its record."""
    logging.disable(logging.CRITICAL)
    from vfa.sim import objects as OB
    from vfa.sim.scene import table_cube_force
    from vfa.sim.sensors import REALISTIC_NOISE, SensorConfig
    from vfa.skills import place as P
    from vfa.skills.params import SkillParams
    from vfa.skills.pick import run_attempt

    cls, k, speed, height = job
    env = P.make_env(SensorConfig(noise=REALISTIC_NOISE))
    spec = OB.sample_object(OB.load_object_config().classes[cls], np.random.default_rng((77, k)))
    grip, accel = GRIPS[cls]
    theta = replace(
        SkillParams(), grip_force=grip, lift_accel=accel, place_speed=speed, release_height=height
    )
    peak = {"table": 0.0, "finger": 0.0, "wrist": 0.0}
    inner = env.step

    def step(action):
        out = inner(action)
        if env.phase in P.RELEASE_PHASES:
            peak["table"] = max(peak["table"], table_cube_force(env.model, env.data, env.ids))
            peak["finger"] = max(peak["finger"], float(sum(env.monitor.last_forces)))
            peak["wrist"] = max(peak["wrist"], float(np.linalg.norm(out[0]["wrench"][:3])))
        return out

    env.step = step
    result = run_attempt(env, P.PickPlaceSkill(), theta, seed=k, options={"object": spec})
    return {
        "class": cls,
        "cube": k,
        "place_speed": speed,
        "release_height": height,
        "result": result.result,
        "sim_s": result.sim_s,
        "break_force": spec.break_force,
        **{f"peak_{name}": v for name, v in peak.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cubes", type=int, default=5)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.015])
    parser.add_argument("--out", type=Path, default=Path("runs/sweep_place"))
    args = parser.parse_args()
    jobs = list(itertools.product(GRIPS, range(args.cubes), args.speeds, args.heights))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs, chunksize=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    print("| class | release height | " + " | ".join(f"{v} m/s" for v in args.speeds) + " |")
    print("|---|---|" + "---|" * len(args.speeds))
    for cls, h in itertools.product(GRIPS, args.heights):
        cells = []
        for v in args.speeds:
            sel = [
                r
                for r in rows
                if r["class"] == cls and r["place_speed"] == v and r["release_height"] == h
            ]
            ok = sum(r["result"] == "SUCCESS" for r in sel)
            broke = sum(r["result"] == "PLACE_BREAK" for r in sel)
            table = max(r["peak_table"] for r in sel)
            cells.append(f"{ok}/{len(sel)}, {broke} pb, {table:.1f} N")
        print(f"| {cls} | {h * 100:g} cm | " + " | ".join(cells) + " |")
    worst = max(rows, key=lambda r: r["peak_wrist"])
    print(
        f"peak wrist force {worst['peak_wrist']:.1f} N ({worst['class']}); "
        f"longest attempt {max(r['sim_s'] for r in rows):.2f} s; {len(rows)} attempts"
    )


if __name__ == "__main__":
    main()
