"""The demo: baselines vs memory on the same cubes, retries until success (T16).

Usage::

    uv run vfa demo --quick            # 30 cubes, about 30 s
    uv run vfa demo --seed 501         # 60 cubes of another benchmark seed

The demo prints a banner naming each RuVector component and the active
backend, then runs three conditions on the same cubes of one benchmark seed
(the frozen benchmark's sensor noise and paired reset seeds, ARCHITECTURE §9):

1. **Baselines** (``naive``, ``random_retry``) in background processes.
2. **Memory** (``memory_full``) in the foreground, starting from an empty
   memory: the console shows the System 1 / System 2 handoffs (``[S1]``,
   ``[S2]``), the retrieval hits and stores (``[MEM]``), and the skill
   consolidations (``[MEM] consolidated skill_NN …``; every
   ``CONSOLIDATE_EVERY`` cubes and at the end). The class stays hidden until the
   tables: lines that show ground truth (the break monitor's, which name the
   class, and the privileged events', such as a BREAK's hidden break force) are
   not shown; the outcome still shows in ``[S1] attempt BREAK``. The displayed
   SLIP vs DROP label is the scorer's (an expired hold is labelled from the
   cube's true height); System 2 only ever sees the merged ``UNDERGRIP``.
3. **Tables**: per cube (class revealed, attempts and failure modes per
   condition) and a summary (success@1 overall and over the second half,
   success@k, attempts-to-success, break rate).

It writes ``runs/demo-<timestamp>/``: ``metrics.json``, every attempt's record
per condition (``records/``), the memory phase's episodes (``episodes/``), its
handoff log (``handoffs.jsonl``), and the memory (``memory/``,
``MemorySystem.save``). ``--video`` arrives with T17.

RuVector mapping: the banner lists the mapping of ARCHITECTURE §10. The memory
phase runs ``CognitiveCore`` (``cognitive_core::CognitiveCore``) with
``RetrievalPolicy`` over ``EpisodicMemory`` and ``SemanticMemory``
(``memory_system``) and ``SkillLibrary`` (``skill_learning``), on
``NumpyBackend``, a Python mirror of the crate's vector search.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from vfa.cognition import system2 as S2
from vfa.experiments import benchmark as B
from vfa.sim.env import PRIVILEGED_EVENTS

QUICK_SITUATIONS = 30
FULL_SITUATIONS = 60
DEFAULT_SEED = 500  # the first quick-profile seed (T10's tuning seeds)
BASELINES = ("naive", "random_retry")
CONDITIONS = (*BASELINES, "memory_full")
SPLIT = 15  # cubes per background baseline job
RUNS = B.REPO_ROOT / "runs"
SHORT = {"SUCCESS": "ok", "BREAK": "brk", "SLIP": "slp", "DROP": "drp", "MISS": "miss"}

BANNER = """\
ruvector-vfa demo: a force-aware Franka Panda that remembers how cubes felt
RuVector components (Python mirrors of ruvector_robotics::cognitive):
  EpisodicMemory  <-> memory_system::EpisodicMemory  every attempt, recalled by force similarity
  SemanticMemory  <-> memory_system::SemanticMemory  skill prototypes (cluster centroids)
  SkillLibrary    <-> skill_learning::SkillLibrary   skills learned from successful attempts
  CognitiveCore   <-> cognitive_core::CognitiveCore  System 1 / System 2 / EMERGENCY modes
backend: {backend} (a Python mirror; the Rust crate runs only with T22's --backend ruvector)"""


def build_parser(prog: str = "vfa demo") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_SITUATIONS} cubes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="benchmark seed")
    parser.add_argument("--video", action="store_true", help="a video of one attempt (T17)")
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument("--workers", type=int, default=B.default_workers())
    return parser


def run_baseline(condition: str, seed: int, start: int, end: int) -> list[dict[str, Any]]:
    """One background job: a baseline on cubes [start, end) of ``seed``."""
    logging.disable(logging.CRITICAL)
    config = B.load_benchmark_config()
    env, skill, objects = B._worker_env(config.noise)
    policy = B.make_baseline(condition)
    records: list[dict[str, Any]] = []
    for situation in B.make_situations(seed, end, objects, start=start):
        records += B.run_situation(env, skill, policy, situation, config.max_attempts)
    return records


def run_memory(
    seed: int, n: int, run: Path, config: B.BenchmarkConfig, policy: S2.RetrievalPolicy
) -> list[dict[str, Any]]:
    """The memory phase in this process: logs visible, episodes and memory saved."""
    from vfa.cognition.cognitive_core import CognitiveCore
    from vfa.cognition.retry import solve_situation
    from vfa.sim import objects as OB
    from vfa.skills.pick import GraspLiftSkill

    env = B.make_env(config)
    skill = GraspLiftSkill()
    core = CognitiveCore(policy, log_path=run / "handoffs.jsonl")
    episodes = run / "episodes"
    records: list[dict[str, Any]] = []
    for situation in B.make_situations(seed, n, OB.load_object_config()):

        def score(result, attempt, situation=situation) -> None:
            records.append(B.attempt_record(result, situation, attempt))
            if result.episode is not None:
                result.episode.save(episodes / f"cube{situation.index:03d}_a{attempt}")

        results = solve_situation(
            env,
            core,
            skill,
            config.max_attempts,
            key=situation.key,
            seed_of=lambda a, s=situation: B.env_seed(s.seed, s.index, a),
            options={"object": situation.spec},
            on_attempt=score,
        )
        last = results[-1]
        outcome = f"solved at attempt {len(results)}" if last.success else "not solved"
        path = " → ".join(SHORT.get(r.result, r.result.lower()) for r in results)
        print(
            f"[demo] cube {situation.index:02d}: {outcome} ({path}; final grip "
            f"{last.theta_final.grip_force:.1f} N)",
            flush=True,
        )
    if policy.situations % S2.CONSOLIDATE_EVERY:  # a last one, unless the cadence just ran
        policy.library.consolidate(policy.memory)
    policy.memory.save(run / "memory")
    return records


def halves(records: Sequence[dict[str, Any]], n: int) -> tuple[float, float]:
    first = B.first_attempts(records)
    half = n // 2
    return (
        sum(first[i] for i in range(half)) / half,
        sum(first[i] for i in range(half, n)) / (n - half),
    )


def cube_table(by_condition: dict[str, list[dict[str, Any]]], n: int) -> str:
    lines = [f"{'cube':>4}  {'class':<15}" + "".join(f"{c:<28}" for c in CONDITIONS)]
    for i in range(n):
        rows = {
            c: sorted((r for r in recs if r["situation"] == i), key=lambda r: r["attempt"])
            for c, recs in by_condition.items()
        }
        cls = next(iter(rows.values()))[0]["class"]
        cells = []
        for c in CONDITIONS:
            results = [SHORT.get(r["result"], r["result"].lower()) for r in rows[c]]
            cells.append(f"{len(results)}: {' '.join(results)}".ljust(28))
        lines.append(f"{i:>4}  {cls:<15}" + "".join(cells))
    return "\n".join(lines)


def summary_table(metrics: dict[str, dict[str, Any]], n: int) -> str:
    half = n // 2
    head = (
        f"{'condition':<14}{'success@1':>10}{f'2nd half ({half}-{n - 1})':>20}"
        f"{'success@5':>11}{'attempts':>10}{'break rate':>12}"
    )
    lines = [head]
    for c in CONDITIONS:
        m = metrics[c]
        lines.append(
            f"{c:<14}{m['success_at_1']:>10.3f}{m['second_half']:>20.3f}"
            f"{m['success_at_k']:>11.3f}{m['attempts_to_success']:>10.2f}{m['break_rate']:>12.3f}"
        )
    return "\n".join(lines)


_HIDDEN_EVENTS = frozenset(e.value for e in PRIVILEGED_EVENTS)


def _hide_ground_truth(record: logging.LogRecord) -> bool:
    """Drop the lines that show ground truth: the break monitor's (it names the class)
    and the privileged events' (a BREAK's force is the hidden break force, whose
    range tells the class)."""
    if record.name == "vfa.sim.objects":
        return False
    args = record.args if isinstance(record.args, tuple) else ()
    return not (record.name == "vfa.sim.sensors" and len(args) > 1 and args[1] in _HIDDEN_EVENTS)


def main(argv: Sequence[str] | None = None, prog: str = "vfa demo") -> int:
    args = build_parser(prog).parse_args(argv)
    if args.video:
        print("--video arrives with T17 (vfa.viz.video)", file=sys.stderr)
        return 2
    config = B.load_benchmark_config()
    n = QUICK_SITUATIONS if args.quick else FULL_SITUATIONS
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = args.out if args.out is not None else RUNS / f"demo-{stamp}"
    try:
        run.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{run} exists: give the demo a new directory (--out)", file=sys.stderr)
        return 2
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(_hide_ground_truth)
    vfa_log = logging.getLogger("vfa")
    saved = vfa_log.level, vfa_log.propagate
    vfa_log.addHandler(handler)
    vfa_log.setLevel(logging.INFO)
    vfa_log.propagate = False
    try:
        return _run(args, config, run, n)
    finally:
        vfa_log.removeHandler(handler)
        vfa_log.setLevel(saved[0])
        vfa_log.propagate = saved[1]


def _run(args: argparse.Namespace, config: B.BenchmarkConfig, run: Path, n: int) -> int:
    policy = S2.RetrievalPolicy()
    print(BANNER.format(backend=type(policy.memory.semantic.index).__name__), flush=True)
    print(
        f"\n{n} cubes of benchmark seed {args.seed} (6 hidden classes; the arm sees none), "
        f"up to {config.max_attempts} attempts each. Baselines run in the background.\n",
        flush=True,
    )
    t0 = time.perf_counter()
    jobs = [(c, args.seed, s, min(s + SPLIT, n)) for c in BASELINES for s in range(0, n, SPLIT)]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        futures = [(c, pool.submit(run_baseline, c, *rest)) for c, *rest in jobs]
        print("memory_full: empty memory, System 2 retrieves at every probe\n", flush=True)
        memory_records = run_memory(args.seed, n, run, config, policy)
        by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in BASELINES}
        for c, future in futures:
            by_condition[c] += future.result()
    by_condition["memory_full"] = memory_records
    metrics: dict[str, dict[str, Any]] = {}
    (run / "records").mkdir()
    for c, records in by_condition.items():
        summary = B.summarize(records, config.max_attempts)
        first, second = halves(records, n)
        metrics[c] = {**summary, "first_half": first, "second_half": second}
        B._write_json(run / "records" / f"{c}.json", records)
    wall = time.perf_counter() - t0
    B._write_json(
        run / "metrics.json",
        {
            "seed": args.seed,
            "situations": n,
            "max_attempts": config.max_attempts,
            "wall_s": wall,
            "skills": len(policy.library),
            "memory_records": len(policy.memory.episodic),
            "conditions": metrics,
        },
    )
    print(
        "\nPer cube (class revealed now; attempts: results, "
        f"brk = break, slp = slip, drp = drop):\n{cube_table(by_condition, n)}"
    )
    print(f"\nSummary ({n} cubes, seed {args.seed}):\n{summary_table(metrics, n)}")
    mem, naive = metrics["memory_full"], metrics["naive"]
    verdict = "beats" if mem["second_half"] > naive["second_half"] else "does not beat"
    print(
        f"\nmemory_full {verdict} naive on success@1 over cubes {n // 2}-{n - 1}: "
        f"{mem['second_half']:.3f} vs {naive['second_half']:.3f}. "
        f"Memory holds {len(policy.memory.episodic)} attempts and {len(policy.library)} skills."
    )
    print(f"Wrote {run} in {wall:.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
