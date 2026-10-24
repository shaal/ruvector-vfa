"""Backend speed: ``NumpyBackend`` against ``RuVectorBackend`` (T22).

Usage::

    uv run python -m vfa.experiments.backend_speed --size 1000      # also 10000, 100000
    uv run python -m vfa.experiments.backend_speed --demo           # vfa demo --quick per backend

Both backends return the same hits (``tests/test_ruvector_backend.py``), so speed
is the only possible gain. ``--size N`` stores N random unit vectors of the real
``key_probe`` dimension (with a record-like payload) in a fresh store of each
backend, one at a time, and reports the insert time per record; then it times
``search(q, 5)`` for ``--queries`` random queries (1 000) and reports the median
and the 95th percentile. Everything runs on one core in this process. A backend
whose estimated run would pass ``--budget`` seconds (480, so one command stays
under 9 minutes) is skipped, and the output says so. ``--demo`` runs ``vfa demo
--quick`` with each backend, twice, alternating, and reports each run's wall
time. Results go to ``runs/backend_speed/`` as JSON.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np

from vfa.experiments.benchmark import REPO_ROOT, _write_json
from vfa.memory.backend import NumpyBackend
from vfa.memory.embeddings import key_dims
from vfa.memory.ruvector_backend import RuVectorBackend, rust_module

OUT = REPO_ROOT / "runs" / "backend_speed"
BACKENDS = {"numpy": NumpyBackend, "ruvector": RuVectorBackend}
PROBE_QUERIES = 20  # queries timed first to estimate a backend's whole run


def _payload(i: int) -> dict:
    return {
        "situation": i % 997,
        "attempt": 1 + i % 3,
        "theta": {"grip_force": 1.0 + i % 24, "close_speed": 0.1, "lift_accel": 0.6},
        "result": "SUCCESS" if i % 3 else "BREAK",
        "utility": 0.5,
        "success": bool(i % 3),
    }


def time_backend(name: str, vectors: np.ndarray, queries: np.ndarray, budget: float) -> dict:
    """Insert and search timings of one backend (refer to the module doc)."""
    store = BACKENDS[name]()
    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        store.add(f"r{i:06d}", v, _payload(i))
    insert = time.perf_counter() - t0
    probe = []
    for q in queries[:PROBE_QUERIES]:
        t = time.perf_counter()
        store.search(q, 5)
        probe.append(time.perf_counter() - t)
    estimate = insert + float(np.mean(probe)) * len(queries)
    if estimate > budget:
        return {"backend": name, "skipped": f"estimated {estimate:.0f} s > budget {budget:.0f} s"}
    latencies = []
    for q in queries:
        t = time.perf_counter()
        store.search(q, 5)
        latencies.append(time.perf_counter() - t)
    lat = np.array(latencies)
    return {
        "backend": name,
        "records": len(vectors),
        "insert_us_per_record": 1e6 * insert / len(vectors),
        "search_ms_median": 1e3 * float(np.median(lat)),
        "search_ms_p95": 1e3 * float(np.percentile(lat, 95)),
        "queries": len(queries),
    }


def run_size(size: int, n_queries: int, budget: float, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng((seed, size))
    dim = key_dims("key_probe")
    vectors = rng.standard_normal((size, dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.standard_normal((n_queries, dim))
    rows = []
    t0 = time.perf_counter()
    for name in BACKENDS:
        left = budget - (time.perf_counter() - t0)
        rows.append(time_backend(name, vectors, queries, left / (len(BACKENDS) - len(rows))))
        print(_line(rows[-1]), flush=True)
    return rows


def run_demo(rounds: int = 2) -> list[dict]:
    rows = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for r in range(rounds):
        for name in BACKENDS:
            out = OUT / f"demo-{stamp}-{name}-{r}"
            cmd = [sys.executable, "-m", "vfa.cli", "demo", "--quick", "--backend", name]
            t = time.perf_counter()
            done = subprocess.run([*cmd, "--out", str(out)], capture_output=True, text=True)
            wall = time.perf_counter() - t
            if done.returncode != 0:
                raise SystemExit(
                    f"{' '.join(cmd)} failed:\n{done.stdout[-2000:]}{done.stderr[-2000:]}"
                )
            summary = done.stdout[done.stdout.index("Summary") :].splitlines()[:5]
            rows.append({"backend": name, "round": r, "wall_s": wall, "summary": summary})
            print(f"demo --quick --backend {name}: {wall:.1f} s", flush=True)
    return rows


def _line(row: dict) -> str:
    if "skipped" in row:
        return f"{row['backend']:9s} skipped: {row['skipped']}"
    return (
        f"{row['backend']:9s} {row['records']:7d} records: "
        f"insert {row['insert_us_per_record']:8.1f} µs/record; "
        f"search top-5 median {row['search_ms_median']:8.3f} ms, "
        f"p95 {row['search_ms_p95']:8.3f} ms ({row['queries']} queries)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--size", type=int, help="records to store (1000, 10000, 100000)")
    group.add_argument("--demo", action="store_true", help="time vfa demo --quick per backend")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--budget", type=float, default=480.0, help="seconds for one command")
    args = parser.parse_args(argv)
    rust_module()  # fail early, with the build steps
    if args.demo:
        rows = run_demo()
        _write_json(OUT / "demo.json", {"runs": rows})
        return 0
    if args.size < 1 or args.queries < PROBE_QUERIES:
        raise SystemExit(f"--size must be >= 1 and --queries >= {PROBE_QUERIES}")
    rows = run_size(args.size, args.queries, args.budget)
    _write_json(OUT / f"size_{args.size}.json", {"size": args.size, "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
