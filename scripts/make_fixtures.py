"""Build the calibration episode fixture for T11 and T13 (``tests/fixtures/calib_episodes.npz``).

ROADMAP T10. Runs the benchmark's ``calib`` profile (``configs/benchmark.toml``:
the calibration seeds, never the evaluation seeds) with the ``naive`` and
``random_retry`` conditions, exactly as ``vfa.experiments.benchmark`` does (same
situations, same per-attempt env seeds), and keeps attempts in compact form:
``naive`` attempts 1–2 and ``random_retry`` attempts 2–3 (its attempt 1 is
``naive``'s attempt 1, the same θ, cube, and noise: stored once).

Regenerate (1–2 min with 8 workers)::

    uv run python scripts/make_fixtures.py --workers 8

**Layout** (one ``numpy.savez_compressed`` file, no pickle; ``load_fixture`` reads it):

- ``rows`` float32 (N, len(``row_columns``)): every ``STRIDE``-th control tick
  (50 Hz) of every attempt, observable signals only: ``t`` [s], ``phase`` (an
  index into ``phase_names``), the tared wrist wrench (6), the held ``load``,
  the two pad forces, the gripper ``width``, and ``ee_pos`` (3). The ``SETTLE``
  rows come before the tare, so their wrench and load are NaN.
- ``offsets`` int64 (E + 1): episode ``e`` owns ``rows[offsets[e]:offsets[e + 1]]``.
- Per episode (length E, in a fixed shuffled order): ``condition``, ``seed``,
  ``group`` (the situation's opaque key: equal for every attempt at one cube,
  and it reveals nothing about the class), ``attempt`` (from 1), ``prev`` (the
  episode index of the attempt before it at the same cube, −1 for a first
  attempt: ``key_retry`` pairs),
  ``result`` (``SLIP`` comes from ground truth: an expired hold labeled by the
  cube's height), ``success``, ``theta`` and ``theta_final`` (E, 5;
  ``theta_columns`` order), ``probe`` (E, len(``probe_columns``); NaN when the
  attempt ended before ``PROBE_DONE``), ``context`` (E, 5: the observed cube x,
  y, z, yaw, size), and the observable ``slip_events`` and ``drop_events``
  counts.
- ``hidden_*`` (length E): the hidden class, mass, friction, break force (NaN =
  unbreakable), and the situation index (the index orders the stratified
  classes). **For test scoring only**: no key, feature, or policy may read them
  (ARCHITECTURE §4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vfa.experiments import benchmark as B
from vfa.experiments.baselines import make_baseline
from vfa.skills.params import FIELDS
from vfa.skills.pick import ProbeFeatures

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "calib_episodes.npz"
CONDITIONS = ("naive", "random_retry")
STRIDE = 10  # 500 Hz ticks per stored row: 50 Hz
# Attempts kept per condition (key_retry pairs). Later ones mostly repeat a failure
# (naive repeats its θ); with all five the file is over the 5 MB cap (CLAUDE.md).
KEEP_ATTEMPTS: dict[str, tuple[int, ...]] = {"naive": (1, 2), "random_retry": (2, 3)}
MAX_BYTES = 5 * 1024 * 1024
ROW_COLUMNS: tuple[str, ...] = (
    "t",
    "phase",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "load",
    "finger_left",
    "finger_right",
    "width",
    "ee_x",
    "ee_y",
    "ee_z",
)
CONTEXT_COLUMNS: tuple[str, ...] = ("x", "y", "z", "yaw", "size")
PROBE_COLUMNS: tuple[str, ...] = tuple(ProbeFeatures.__dataclass_fields__)
PHASE_NAMES: tuple[str, ...] = ("SETTLE", "APPROACH", "DESCEND", "PROBE", "SQUEEZE", "LIFT", "HOLD")


def compact(result, record: dict, situation: B.Situation, condition: str) -> dict:
    """One attempt in fixture form (observable rows + per-episode fields + hidden labels)."""
    ep = result.episode
    a = ep.arrays
    keep = slice(None, None, STRIDE)
    phase_index = np.array([PHASE_NAMES.index(p) for p in ep.phases])[a["phase"]]
    rows = np.column_stack(
        [
            a["t"][keep],
            phase_index[keep],
            a["wrench"][keep],
            a["load"][keep],
            a["finger_force"][keep],
            a["width"][keep],
            a["ee_pos"][keep],
        ]
    ).astype(np.float32)
    probe = np.full(len(PROBE_COLUMNS), np.nan)
    if result.probe is not None:
        values = result.probe.to_dict()
        probe = np.array([np.nan if values[k] is None else float(values[k]) for k in PROBE_COLUMNS])
    cube = ep.context["cube"]
    spec = situation.spec
    return {
        "rows": rows,
        "condition": condition,
        "seed": situation.seed,
        "group": situation.key,
        "attempt": record["attempt"],
        "slip_events": len(ep.events_named("SLIP")),
        "drop_events": len(ep.events_named("DROP")),
        "result": result.result,
        "success": result.success,
        "theta": result.theta.to_vector(),
        "theta_final": result.theta_final.to_vector(),
        "probe": probe,
        "context": np.array([cube[k] for k in CONTEXT_COLUMNS], dtype=float),
        "hidden_class": spec.class_name,
        "hidden_mass": spec.mass,
        "hidden_friction": spec.friction[0],
        "hidden_break_force": np.nan if spec.break_force is None else spec.break_force,
        "hidden_situation": situation.index,
    }


def run_part(job: tuple[str, int, int, int]) -> list[dict]:
    """One condition × seed × situation range, in a worker process."""
    logging.disable(logging.CRITICAL)
    condition, seed, start, end = job
    config = B.load_benchmark_config()
    env, skill, objects = B._worker_env(config.noise)
    policy = make_baseline(condition)
    out: list[dict] = []
    for situation in B.make_situations(seed, end, objects, start=start):
        B.run_situation(
            env,
            skill,
            policy,
            situation,
            config.max_attempts,
            lambda r, rec, s=situation: (
                out.append(compact(r, rec, s, condition))
                if rec["attempt"] in KEEP_ATTEMPTS[condition]
                else None
            ),
        )
    return out


def build(workers: int) -> dict[str, np.ndarray]:
    config = B.load_benchmark_config()
    prof = config.profiles["calib"]
    jobs = [
        (condition, seed, start, min(start + B.CHUNK, prof.situations))
        for condition in CONDITIONS
        for seed in prof.seeds
        for start in range(0, prof.situations, B.CHUNK)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run_part, jobs))
    episodes = [e for part in parts for e in part]
    # A fixed shuffle: run order would mirror the situation index (and so the class).
    order = np.random.default_rng(0xF1C7).permutation(len(episodes))
    episodes = [episodes[i] for i in order]
    lengths = [len(e["rows"]) for e in episodes]
    data = {
        "rows": np.concatenate([e["rows"] for e in episodes]),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "row_columns": np.array(ROW_COLUMNS),
        "phase_names": np.array(PHASE_NAMES),
        "theta_columns": np.array(FIELDS),
        "probe_columns": np.array(PROBE_COLUMNS),
        "context_columns": np.array(CONTEXT_COLUMNS),
        "benchmark_hash": np.array(B.benchmark_hash()),
    }
    for key in ("condition", "result", "hidden_class"):
        data[key] = np.array([e[key] for e in episodes])
    for key in ("seed", "group", "attempt", "slip_events", "drop_events", "hidden_situation"):
        data[key] = np.array([e[key] for e in episodes], dtype=np.int64)
    # The attempt before each retry at the same cube: naive's attempt 1 precedes both
    # naive's and random_retry's attempt 2 (random_retry starts with naive's θ).
    where = {
        (e["condition"], e["seed"], e["group"], e["attempt"]): i for i, e in enumerate(episodes)
    }
    prev = []
    for e in episodes:
        before = "naive" if e["attempt"] == 2 else e["condition"]
        prev.append(where.get((before, e["seed"], e["group"], e["attempt"] - 1), -1))
    data["prev"] = np.array(prev, dtype=np.int64)
    data["success"] = np.array([e["success"] for e in episodes], dtype=bool)
    for key in ("theta", "theta_final", "probe", "context"):
        data[key] = np.stack([e[key] for e in episodes])
    for key in ("hidden_mass", "hidden_friction", "hidden_break_force"):
        data[key] = np.array([e[key] for e in episodes], dtype=float)
    return data


def load_fixture(path: str | Path = FIXTURE) -> dict[str, np.ndarray]:
    """The fixture's arrays (refer to the module doc)."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=B.default_workers())
    parser.add_argument("--out", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)
    data = build(args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    size = args.out.stat().st_size
    n = len(data["offsets"]) - 1
    classes, counts = np.unique(data["hidden_class"], return_counts=True)
    print(f"wrote {args.out}: {n} episodes, {len(data['rows'])} rows, {size / 1e6:.2f} MB")
    print("  per class:", dict(zip(classes.tolist(), counts.tolist(), strict=True)))
    print("  results:", dict(zip(*np.unique(data["result"], return_counts=True), strict=True)))
    if size > MAX_BYTES:
        print(f"fixture is over {MAX_BYTES / 1e6:.1f} MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
