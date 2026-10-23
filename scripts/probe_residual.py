"""Hand-set grip shaping on memory's grasps: where the force jerk comes from (T21).

Replays every attempt of ``memory_full`` on the calibration profile (the
residual trainer's replay set: ``runs/residual/replay``, which
``vfa.experiments.train_residual`` builds) with the θ that ran, under each
variant below, and prints the success rate, the mean force jerk and peak-force
excess over successes, and the mean jerk² from SQUEEZE and from LIFT. No
System 2 runs; the grip command is changed in the skill's action. docs/RESULTS.md
§11 quotes it.

- ``none``: the skill as is.
- ``squeeze_minus2``: 2 N per finger less in SQUEEZE (a constant offset).
- ``rate10``: in SQUEEZE the command rises from the probe's 1.5 N at 10 N/s.
- ``close_0.05``: θ's closing speed at its lower bound (the squeeze ramps at half rate).
- ``limit_all_0.02`` / ``limit_all_0.01``: in SQUEEZE, LIFT, and HOLD every rise of
  the command is limited to 0.02 (0.01) N per tick above the motor's squeeze
  (10 and 5 N/s at 500 Hz); a lower command applies at once.

    uv run python scripts/probe_residual.py --workers 8      # about 2 min
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

VARIANTS: dict[str, dict] = {
    "none": {},
    "squeeze_minus2": {"bias": -2.0},
    "rate10": {"rate": 10.0},
    "close_0.05": {"close": 0.05},
    "limit_all_0.02": {"limit": 0.02},
    "limit_all_0.01": {"limit": 0.01},
}
REPLAY = "runs/residual/replay/calib/memory_full/*.json"


def one(job: tuple[str, dict, int]) -> tuple:
    logging.disable(logging.CRITICAL)
    from vfa.experiments import benchmark as B
    from vfa.sim import objects as OB
    from vfa.skills.params import SkillParams
    from vfa.skills.pick import GraspLiftSkill, run_attempt

    name, rec, seed = job
    v = VARIANTS[name]

    class Probe(GraspLiftSkill):
        def _enter(self, phase: str) -> None:
            super()._enter(phase)
            self._ticks = 0

        def _action(self, pos):
            a = super()._action(pos)
            self._ticks = getattr(self, "_ticks", 0) + 1
            if self._grip <= 0.0 or self.phase not in ("SQUEEZE", "LIFT", "HOLD"):
                return a
            grip = self._grip
            if "bias" in v and self.phase == "SQUEEZE":
                grip = max(grip + v["bias"], 0.5)
            if "rate" in v and self.phase == "SQUEEZE":
                grip = min(grip, 1.5 + v["rate"] * self._ticks * self._dt)
            if "limit" in v:
                grip = max(min(grip, self._env.gripper.actuator_squeeze() + v["limit"]), 0.5)
            a[3] = grip / 30.0
            return a

    env = B.make_env(B.load_benchmark_config())
    s = B.make_situations(seed, rec["situation"] + 1, OB.load_object_config(), rec["situation"])[0]
    theta = SkillParams.from_dict(rec["theta_final"])
    if "close" in v:
        theta = replace(theta, close_speed=v["close"])
    r = run_attempt(
        env,
        Probe(),
        theta,
        seed=B.env_seed(seed, s.index, rec["attempt"]),
        options={"object": s.spec},
    )
    out = B.attempt_record(r, s, rec["attempt"])
    split = None
    if out["success"]:
        ep = r.episode
        ph = np.asarray(ep.phases)[ep.arrays["phase"]]
        m = np.isin(ph, ("SQUEEZE", "LIFT"))
        force = np.sum(ep.arrays["finger_force"][m], axis=1)
        rate = np.diff(force) / np.diff(ep.arrays["t"][m])
        phase = ph[m][1:]
        n = len(rate)
        split = tuple(float(np.sum(rate[phase == p] ** 2) / n) for p in ("SQUEEZE", "LIFT"))
    return name, out["success"], out["force_jerk"], out["peak_force_excess"], split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = []
    for path in sorted(glob.glob(REPLAY)):
        seed = int(path.split("seed")[1].split("_")[0])
        for rec in json.load(open(path))["records"]:
            jobs += [(name, rec, seed) for name in VARIANTS]
    if not jobs:
        raise SystemExit(f"no replay under {REPLAY}: run vfa.experiments.train_residual first")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    for name in VARIANTS:
        rs = [r for r in rows if r[0] == name]
        ok = [r for r in rs if r[1]]
        jerk = np.mean([r[2] for r in ok if r[2] is not None])
        excess = np.mean([r[3] for r in ok if r[3] is not None])
        sq = np.mean([r[4][0] for r in ok])
        lift = np.mean([r[4][1] for r in ok])
        print(
            f"{name:16s} success {np.mean([r[1] for r in rs]):.3f} jerk {jerk:5.1f} N/s "
            f"excess {excess:5.2f} N  jerk² from SQUEEZE {sq:4.0f}, LIFT {lift:4.0f}  "
            f"({len(rs)} attempts)"
        )


if __name__ == "__main__":
    main()
