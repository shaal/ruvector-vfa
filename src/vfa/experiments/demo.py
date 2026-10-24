"""The demo: baselines vs memory on the same cubes, retries until success (T16).

Usage::

    uv run vfa demo --quick            # 30 cubes, about 30 s
    uv run vfa demo --seed 501         # 60 cubes of another benchmark seed
    uv run vfa demo --quick --task pick_place   # pick-and-place (T19), about 1 min

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

It writes ``runs/demo-<timestamp>/``: ``metrics.json`` (with the item count per
memory tier after each cube and the skill list), every attempt's record per
condition (``records/``; the memory phase's carry their memory id), the memory
phase's episodes (``episodes/``), its handoff log (``handoffs.jsonl``), the
memory (``memory/``, ``MemorySystem.save``), the plots (``plots/``), and
``report.html`` (``vfa.viz.report``). ``--video`` also writes ``attempts.gif``:
every attempt of the first cube that needed a retry, with a force overlay.

``--task pick_place`` (T19) runs the same three conditions on the
pick-and-place benchmark (``configs/benchmark_pick_place.toml``, default seed
600): ``PickPlaceSkill`` carries each cube to the place target and sets it
down, ``random_retry`` also draws the place fields, and memory uses the
pick-and-place keys (``PlaceEmbedder``). The tables add the place failures
(``pbrk`` = PLACE_BREAK, ``topl`` = TOPPLE, ``pmis`` = PLACE_MISS).

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
DEFAULT_SEEDS = {"lift": DEFAULT_SEED, "pick_place": 600}  # pick_place: its quick seed
BASELINES = ("naive", "random_retry")
CONDITIONS = (*BASELINES, "memory_full")
SPLIT = 15  # cubes per background baseline job
RUNS = B.REPO_ROOT / "runs"
SHORT = {
    "SUCCESS": "ok",
    "BREAK": "brk",
    "SLIP": "slp",
    "DROP": "drp",
    "MISS": "miss",
    "PLACE_BREAK": "pbrk",
    "TOPPLE": "topl",
    "PLACE_MISS": "pmis",
}

BANNER = """\
ruvector-vfa demo: a force-aware Franka Panda that remembers how cubes felt
RuVector components (Python mirrors of ruvector_robotics::cognitive):
  EpisodicMemory  <-> memory_system::EpisodicMemory  every attempt, recalled by force similarity
  SemanticMemory  <-> memory_system::SemanticMemory  skill prototypes (cluster centroids)
  SkillLibrary    <-> skill_learning::SkillLibrary   skills learned from successful attempts
  CognitiveCore   <-> cognitive_core::CognitiveCore  System 1 / System 2 / EMERGENCY modes
backend: {backend}"""
BACKEND_NOTES = {
    "NumpyBackend": "a Python mirror of the crate's search; --backend ruvector runs the Rust crate",
    "RuVectorBackend": "ruvector-robotics' EpisodicMemory through the PyO3 binding (T22)",
}


def build_parser(prog: str = "vfa demo") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_SITUATIONS} cubes")
    parser.add_argument(
        "--seed", type=int, default=None, help="benchmark seed (default 500; pick_place 600)"
    )
    parser.add_argument("--task", choices=B.TASKS, default="lift", help="lift or pick_place")
    parser.add_argument(
        "--backend",
        choices=B.BACKENDS,
        default="numpy",
        help="memory's vector store: numpy, or ruvector (the Rust crate, T22)",
    )
    parser.add_argument("--video", action="store_true", help="a GIF of the first retried cube")
    parser.add_argument("--out", type=Path, default=None, help="run directory")
    parser.add_argument("--workers", type=int, default=B.default_workers())
    return parser


def load_config(task: str) -> B.BenchmarkConfig:
    return B.load_pick_place_config() if task == "pick_place" else B.load_benchmark_config()


def run_baseline(
    condition: str, seed: int, start: int, end: int, task: str = "lift"
) -> list[dict[str, Any]]:
    """One background job: a baseline on cubes [start, end) of ``seed``."""
    logging.disable(logging.CRITICAL)
    config = load_config(task)
    env, skill, objects = B._worker_env(config.noise, task)
    policy = B.make_baseline(condition, place=task == "pick_place")
    records: list[dict[str, Any]] = []
    for situation in B.make_situations(seed, end, objects, start=start):
        records += B.run_situation(env, skill, policy, situation, config.max_attempts)
    return records


def run_memory(
    seed: int,
    n: int,
    run: Path,
    config: B.BenchmarkConfig,
    policy: S2.RetrievalPolicy,
    video: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """The memory phase in this process: logs visible, episodes and memory saved.
    Returns the records and the item count per memory tier after each cube."""
    from vfa.cognition.cognitive_core import CognitiveCore
    from vfa.cognition.retry import solve_situation
    from vfa.sim import objects as OB

    if video:
        from vfa.viz.video import FrameGrab

        env: Any = FrameGrab(B.make_env(config, render_mode="rgb_array", render_size=(240, 320)))
    else:
        env = B.make_env(config)
    skill = B.make_skill(config.task)
    core = CognitiveCore(policy, log_path=run / "handoffs.jsonl")
    episodes = run / "episodes"
    records: list[dict[str, Any]] = []
    tiers: list[dict[str, int]] = []
    for situation in B.make_situations(seed, n, OB.load_object_config()):

        def score(result, attempt, situation=situation) -> None:
            record = B.attempt_record(result, situation, attempt)
            record["memory_id"] = next(reversed(policy.memory.episodic.records))  # just stored
            records.append(record)
            if result.episode is not None:
                result.episode.save(episodes / f"cube{situation.index:03d}_a{attempt}")
            if video:
                env.label = f"cube {situation.index}, attempt {attempt + 1}"

        if video:
            env.active = not (run / "attempts.gif").exists()
            env.frames.clear()
            env.label = f"cube {situation.index}, attempt 1"

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
        tiers.append(
            {
                "cube": situation.index,
                "episodic": len(policy.memory.episodic),
                "semantic": len(policy.memory.semantic),
                "working": len(policy.memory.working),
            }
        )
        if video and env.active and len(results) > 1:  # the first cube that needed a retry
            from vfa.viz.video import write_gif

            write_gif(env.frames, run / "attempts.gif", env.fps)
            env.active = False
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
        tiers[-1]["semantic"] = len(policy.memory.semantic)
    policy.memory.save(run / "memory")
    return records, tiers


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


def summary_table(metrics: dict[str, dict[str, Any]], n: int, place: bool = False) -> str:
    half = n // 2
    head = (
        f"{'condition':<14}{'success@1':>10}{f'2nd half ({half}-{n - 1})':>20}"
        f"{'success@5':>11}{'attempts':>10}{'break rate':>12}"
    ) + (f"{'place break':>13}" if place else "")
    lines = [head]
    for c in CONDITIONS:
        m = metrics[c]
        lines.append(
            f"{c:<14}{m['success_at_1']:>10.3f}{m['second_half']:>20.3f}"
            f"{m['success_at_k']:>11.3f}{m['attempts_to_success']:>10.2f}{m['break_rate']:>12.3f}"
            + (f"{m['place_break_rate']:>13.3f}" if place else "")
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
    config = load_config(args.task)
    if args.seed is None:
        args.seed = DEFAULT_SEEDS[args.task]
    n = QUICK_SITUATIONS if args.quick else FULL_SITUATIONS
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = args.out if args.out is not None else RUNS / f"demo-{stamp}"
    if args.backend == "ruvector":
        from vfa.memory.ruvector_backend import rust_module

        try:
            rust_module()  # before the run directory exists
        except ImportError as exc:
            print(exc, file=sys.stderr)
            return 2
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
    pick_place = config.task == "pick_place"
    memory = None
    if args.backend == "ruvector":
        from vfa.memory.memory_system import MemorySystem
        from vfa.memory.ruvector_backend import RuVectorBackend

        memory = MemorySystem(backend_factory=RuVectorBackend)
    if pick_place:
        from vfa.memory.embeddings import PlaceEmbedder

        policy = S2.RetrievalPolicy(memory, embedder=PlaceEmbedder())
    else:
        policy = S2.RetrievalPolicy(memory)
    name = type(policy.memory.semantic.index).__name__
    print(BANNER.format(backend=f"{name} ({BACKEND_NOTES.get(name, '')})"), flush=True)
    task = "pick-and-place: carry each cube to the target and set it down" if pick_place else "lift"
    print(
        f"\n{n} cubes of {config.task} benchmark seed {args.seed} (6 hidden classes; the arm "
        f"sees none), up to {config.max_attempts} attempts each ({task}). "
        "Baselines run in the background.\n",
        flush=True,
    )
    t0 = time.perf_counter()
    jobs = [
        (c, args.seed, s, min(s + SPLIT, n), config.task)
        for c in BASELINES
        for s in range(0, n, SPLIT)
    ]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as pool:
        futures = [(c, pool.submit(run_baseline, c, *rest)) for c, *rest in jobs]
        print("memory_full: empty memory, System 2 retrieves at every probe\n", flush=True)
        memory_records, tiers = run_memory(args.seed, n, run, config, policy, args.video)
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
            "task": config.task,
            "seed": args.seed,
            "situations": n,
            "max_attempts": config.max_attempts,
            "wall_s": wall,
            "backend": type(policy.memory.semantic.index).__name__,
            "memory_records": len(policy.memory.episodic),
            "tiers": tiers,
            "skills": [
                {
                    "name": name,
                    "members": len(policy.memory.semantic.retrieve(name)[1]["members"]),
                    "confidence": policy.library.get(name).confidence,
                    "grip_force": policy.library.get(name).theta["grip_force"],
                }
                for name in policy.memory.semantic.index.ids
            ],
            "conditions": metrics,
        },
    )
    legend = "brk = break, slp = slip, drp = drop"
    if pick_place:
        legend += ", pbrk = place break, topl = topple, pmis = place miss"
    print(
        f"\nPer cube (class revealed now; attempts: results, {legend}):\n"
        f"{cube_table(by_condition, n)}"
    )
    print(f"\nSummary ({n} cubes, seed {args.seed}):\n{summary_table(metrics, n, pick_place)}")
    mem, naive = metrics["memory_full"], metrics["naive"]
    verdict = "beats" if mem["second_half"] > naive["second_half"] else "does not beat"
    print(
        f"\nmemory_full {verdict} naive on success@1 over cubes {n // 2}-{n - 1}: "
        f"{mem['second_half']:.3f} vs {naive['second_half']:.3f}. "
        f"Memory holds {len(policy.memory.episodic)} attempts and {len(policy.library)} skills."
    )
    from vfa.viz.report import write_report

    report = write_report(run)
    print(f"Wrote {run} in {wall:.0f} s; report: {report}")
    if args.video:
        gif = run / "attempts.gif"
        print(f"video: {gif}" if gif.exists() else "video: no cube needed a retry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
