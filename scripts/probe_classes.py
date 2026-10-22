"""Check the hidden classes under the benchmark's sensors (T10, docs/RESULTS.md §3).

Two measurements on the T08 skill with the benchmark env (``configs/benchmark.toml``:
realistic sensor noise, the safety supervisor on):

1. ``brittle`` (the class T10 added): a mid grip (5 N per finger) on 20 cubes
   must lift them; a high grip (12 N) on 10 cubes must break them.
2. Probe separability: the probe's heft peak, indentation, and whether the cube
   slid, on 20 cubes per class.

Usage (≈ 30 s with 10 workers)::

    uv run python scripts/probe_classes.py --workers 10
"""

from __future__ import annotations

import argparse
import collections
import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from vfa.experiments.benchmark import load_benchmark_config, make_env
from vfa.sim import objects as OB
from vfa.skills.params import SkillParams
from vfa.skills.pick import GraspLiftSkill, run_attempt

CUBES_MID, CUBES_HIGH, CUBES_PROBE = 20, 10, 20
PROBE_SEED0 = 100  # probe cubes use seeds 100.. (the brittle checks use 0..)


def job(args: tuple[str, str, float, int]) -> tuple[str, str, float, str, tuple | None]:
    logging.disable(logging.CRITICAL)
    kind, cls, grip, seed = args
    env = make_env(load_benchmark_config())
    theta = SkillParams(grip_force=grip, lift_accel=0.6)
    r = run_attempt(env, GraspLiftSkill(), theta, seed=seed, options={"object_class": cls})
    p = r.probe
    features = None if p is None else (p.heft_peak, 1000.0 * p.indentation, p.heft_slip)
    return kind, cls, grip, r.result, features


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)
    names = OB.load_object_config().names
    jobs = [("mid", "brittle", 5.0, s) for s in range(CUBES_MID)]
    jobs += [("high", "brittle", 12.0, s) for s in range(CUBES_HIGH)]
    jobs += [("probe", c, 5.0, PROBE_SEED0 + s) for c in names for s in range(CUBES_PROBE)]
    out: dict[tuple, list] = collections.defaultdict(list)
    with ProcessPoolExecutor(args.workers) as pool:
        for kind, cls, _grip, result, features in pool.map(job, jobs, chunksize=2):
            out[(kind, cls)].append((result, features))
    for kind, label in (("mid", "brittle at 5 N"), ("high", "brittle at 12 N")):
        print(f"{label}: {dict(collections.Counter(r for r, _ in out[(kind, 'brittle')]))}")
    print("class            heft peak [N]   indentation [mm]   slides")
    for cls in names:
        rows = [f for _, f in out[("probe", cls)] if f is not None]
        peak = [f[0] for f in rows]
        ind = [f[1] for f in rows]
        print(
            f"{cls:15s}  {min(peak):.2f}-{max(peak):.2f}       {min(ind):.3f}-{max(ind):.3f}"
            f"        {sum(f[2] for f in rows)}/{len(rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
