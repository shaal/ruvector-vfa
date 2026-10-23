"""Full experiments and ablations on the frozen benchmark (T18, ARCHITECTURE §9).

Usage::

    uv run python -m vfa.experiments.run_all --workers 8   # re-run until every chunk is done
    uv run python -m vfa.experiments.run_all aggregate     # tables, targets, learning curves
    uv run python -m vfa.experiments.run_all figures       # force-profile examples

**Runs.** Every condition of ``FULL`` on the ``full`` profile (the 3 evaluation
seeds × 60 situations), and the held-out run: ``HELDOUT`` on the ``heldout``
profile (seed 2000), where ``memory_frozen`` learns on the warm-up seed
(``benchmark.WARMUP``, unscored), freezes, and is scored with no further
writes. The chunks are ``vfa.experiments.benchmark``'s (same keys, files, and
resume rule), run in one process pool, the longest first; a deadline (default
300 s) stops submitting chunks, so an invocation ends within about 8 minutes.
Re-run the same command until it reports every chunk done.

**Aggregate.** Per condition, the benchmark metrics' mean ± sample std over
seeds. Each comparison with ``memory_full`` (the force ablation
``memory_ctx_only``, the four component ablations, and the bandits) gives the
difference (condition − ``memory_full``) per seed, as mean ± std, and a paired
bootstrap 95 % interval over situations pooled over the seeds (``BOOT`` draws,
seed 0; success@1 resamples situations 21–60 only): success@1 over situations
21–60, attempts-to-success, and break rate (pooled per attempt).

The verdicts and targets were fixed before the first evaluation run:

- A comparison's **verdict** reads the first two intervals. The condition is
  *worse* on a metric if the interval excludes 0 on the side that is worse for
  it (lower success@1, more attempts), *better* if it excludes 0 on the other
  side. For a component ablation the sentence is about the component: removing
  it makes the result worse on at least one metric and better on none → *it
  helps*; better on at least one and worse on none → *it hurts*; both →
  *mixed*; neither → *no measurable effect*.
- **Targets** (means over the 3 seeds): (1) ``memory_full`` success@1 over
  situations 21–60 ≥ ``best_fixed``'s over the same window + 0.15 (the
  like-for-like bar of docs/RESULTS.md §2; the looser reading, ``best_fixed``'s
  overall success@1 + 0.15, is reported too, and a result between the two
  counts as a miss); (2) ``memory_full`` attempts-to-success < ``random_retry``'s;
  (3) ``memory_full`` peak-force excess or force jerk < ``random_retry``'s.

``aggregate`` writes ``<out>/t18/results.json``, prints Markdown tables, and
draws the learning curves (success@1 and attempts-to-success per block of 6
situations, mean ± std over seeds) to ``docs/figures/t18_learning_curves.png``.

**Figures.** ``figures`` re-runs ``random_retry`` and ``memory_full`` on the
first evaluation seed with every episode saved (``<out>/t18_figures``), checks
that their records equal the scored chunks' (but for wall times), and draws,
per class, the first cube that both conditions lift: the summed pad force of
each one's successful attempt (``docs/figures/t18_force_profiles.png``).

RuVector mapping: none; this module runs the conditions (ARCHITECTURE §10 maps
the memory and cognition types they use).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vfa.experiments import benchmark as B
from vfa.experiments.baselines import REPO_ROOT

FULL: tuple[str, ...] = (
    "naive",
    "best_fixed",
    "random_retry",
    "memory_full",
    "memory_ctx_only",
    "memory_no_reinforce",
    "memory_no_semantic",
    "memory_no_probe",
    "memory_no_failure_bounds",
    "bandit",
    "bandit_flat",
)
HELDOUT: tuple[str, ...] = ("naive", "best_fixed", "random_retry", "memory_full", B.FROZEN)
ABLATIONS: tuple[str, ...] = (
    "memory_no_reinforce",
    "memory_no_semantic",
    "memory_no_probe",
    "memory_no_failure_bounds",
)
COMPARED: tuple[str, ...] = ("memory_ctx_only", *ABLATIONS, "bandit", "bandit_flat")
BOOT = 10_000
DEADLINE_S = 300.0
FIGURES = REPO_ROOT / "docs" / "figures"
CURVES = ("naive", "best_fixed", "random_retry", "memory_full", "memory_ctx_only", "bandit")
COLORS = {
    "naive": "#9ca3af",
    "best_fixed": "#6b7280",
    "random_retry": "#f59e0b",
    "memory_full": "#2563eb",
    "memory_ctx_only": "#93c5fd",
    "bandit": "#16a34a",
}
logger = logging.getLogger(__name__)


def plan(config: B.BenchmarkConfig, out: Path = B.DEFAULT_OUT) -> list[B.Job]:
    """Every chunk of the T18 runs, the longest first (held-out memory, learners, baselines)."""
    jobs = [j for c in FULL for j in B.plan_condition(config, c, "full", out=out)]
    jobs += [j for c in HELDOUT for j in B.plan_condition(config, c, "heldout", out=out)]

    def cost(job: B.Job) -> int:
        if job.condition == B.FROZEN:
            return 0
        return 1 if job.condition in B.LEARNING_CONDITIONS else 2

    return sorted(jobs, key=cost)


# -------------------------------------------------------------------- aggregate


@dataclass(frozen=True, slots=True)
class Situations:
    """One condition's situations, pooled over seeds, in (seed, index) order."""

    units: tuple[tuple[int, int], ...]
    first: np.ndarray  # attempt 1 succeeded
    tries: np.ndarray  # attempts-to-success (unsolved: max_attempts + 1)
    attempts: np.ndarray
    breaks: np.ndarray
    late: np.ndarray  # situation index >= LATE_FROM

    @classmethod
    def of(cls, by_seed: Mapping[int, Sequence[dict]], max_attempts: int) -> Situations:
        rows: dict[tuple[int, int], list[dict]] = {}
        for seed, records in by_seed.items():
            for r in records:
                rows.setdefault((seed, r["situation"]), []).append(r)
        units = tuple(sorted(rows))
        first, tries, attempts, breaks = [], [], [], []
        for unit in units:
            rs = sorted(rows[unit], key=lambda r: r["attempt"])
            win = next((r["attempt"] for r in rs if r["success"]), None)
            first.append(win == 1)
            tries.append(win if win is not None else max_attempts + 1)
            attempts.append(len(rs))
            breaks.append(sum(r["result"] == "BREAK" for r in rs))
        late = np.array([i >= B.LATE_FROM for _, i in units])
        return cls(
            units,
            np.array(first, dtype=float),
            np.array(tries, dtype=float),
            np.array(attempts, dtype=float),
            np.array(breaks, dtype=float),
            late,
        )


def bootstrap(
    a: Situations, b: Situations, draws: int = BOOT, seed: int = 0
) -> dict[str, list[float]]:
    """Paired bootstrap 95 % intervals of a − b over situations (refer to the module doc)."""
    if a.units != b.units:
        raise ValueError("the conditions must cover the same situations")
    rng = np.random.default_rng(seed)
    late = np.flatnonzero(a.late)
    pick = rng.integers(0, len(a.units), (draws, len(a.units)))
    pick_late = late[rng.integers(0, len(late), (draws, len(late)))]

    def rate(s: Situations, idx: np.ndarray) -> np.ndarray:
        return s.breaks[idx].sum(axis=1) / s.attempts[idx].sum(axis=1)

    diffs = {
        "success_at_1_late": a.first[pick_late].mean(axis=1) - b.first[pick_late].mean(axis=1),
        "attempts_to_success": a.tries[pick].mean(axis=1) - b.tries[pick].mean(axis=1),
        "break_rate": rate(a, pick) - rate(b, pick),
    }
    return {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in diffs.items()}


def verdict(interval: Mapping[str, Sequence[float]]) -> str:
    """The pre-registered sentence for a component ablation (refer to the module doc)."""
    worse = better = False
    for metric, sign in (("success_at_1_late", 1.0), ("attempts_to_success", -1.0)):
        lo, hi = (sign * x for x in interval[metric])
        lo, hi = min(lo, hi), max(lo, hi)
        worse |= hi < 0.0
        better |= lo > 0.0
    if worse and better:
        return "mixed"
    if worse:
        return "it helps"
    if better:
        return "it hurts"
    return "no measurable effect"


def blocks(by_seed: Mapping[int, Sequence[dict]], n: int, max_attempts: int) -> dict[str, Any]:
    """success@1 and attempts-to-success per block of ``BLOCK`` situations, per seed."""
    s1, att = [], []
    for records in by_seed.values():
        sit = Situations.of({0: records}, max_attempts)
        index = np.array([i for _, i in sit.units])
        s1.append(
            [sit.first[(index >= b) & (index < b + B.BLOCK)].mean() for b in range(0, n, B.BLOCK)]
        )
        att.append(
            [sit.tries[(index >= b) & (index < b + B.BLOCK)].mean() for b in range(0, n, B.BLOCK)]
        )
    s1_a, att_a = np.array(s1), np.array(att)
    ddof = 1 if len(s1) > 1 else 0
    return {
        "success_at_1": {"mean": s1_a.mean(0).tolist(), "std": s1_a.std(0, ddof=ddof).tolist()},
        "attempts_to_success": {
            "mean": att_a.mean(0).tolist(),
            "std": att_a.std(0, ddof=ddof).tolist(),
        },
    }


def _diff(per_seed: Mapping[int, dict], base: Mapping[int, dict], metric: str) -> dict:
    values = [per_seed[s][metric] - base[s][metric] for s in sorted(base)]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def aggregate(
    config: B.BenchmarkConfig, out: Path = B.DEFAULT_OUT, figures: Path | None = FIGURES
) -> dict[str, Any]:
    """The T18 tables, comparisons, and targets → ``<out>/t18/results.json``."""
    k = config.max_attempts
    result: dict[str, Any] = {
        "benchmark_hash": B.benchmark_hash(),
        "code": B.code_fingerprint(),
        "max_attempts": k,
        "warmup": dict(B.WARMUP),
        "profiles": {},
    }
    loaded: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for profile, conditions in (("full", FULL), ("heldout", HELDOUT)):
        n = config.profiles[profile].situations
        table: dict[str, Any] = {}
        for c in conditions:
            by_seed = B.load_records(B.plan_condition(config, c, profile, out=out))
            loaded[(profile, c)] = by_seed
            per_seed = {s: B.summarize(r, k) for s, r in sorted(by_seed.items())}
            table[c] = {
                "per_seed": per_seed,
                **B.combine(per_seed),
                "blocks": blocks(by_seed, n, k),
            }
        result["profiles"][profile] = table
    full = result["profiles"]["full"]
    base = full["memory_full"]["per_seed"]
    base_sit = Situations.of(loaded[("full", "memory_full")], k)
    comparisons = {}
    for c in COMPARED:
        sit = Situations.of(loaded[("full", c)], k)
        interval = bootstrap(sit, base_sit)
        comparisons[c] = {
            "diff": {
                m: _diff(full[c]["per_seed"], base, m)
                for m in ("success_at_1_late", "attempts_to_success", "break_rate")
            },
            "interval": interval,
            "verdict": verdict(interval),
        }
    result["comparisons"] = comparisons
    result["targets"] = targets(full)
    _write(out / "t18" / "results.json", result)
    if figures is not None:
        n = config.profiles["full"].situations
        learning_figure(full, n, k, figures / "t18_learning_curves.png")
    return result


def targets(full: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-registered targets (refer to the module doc)."""

    def mean(c: str, m: str) -> float:
        return full[c][m]["mean"]

    bar = mean("best_fixed", "success_at_1_late") + 0.15
    loose = mean("best_fixed", "success_at_1") + 0.15
    late = mean("memory_full", "success_at_1_late")
    excess = (mean("memory_full", "peak_force_excess"), mean("random_retry", "peak_force_excess"))
    jerk = (mean("memory_full", "force_jerk"), mean("random_retry", "force_jerk"))
    tries = (
        mean("memory_full", "attempts_to_success"),
        mean("random_retry", "attempts_to_success"),
    )
    return {
        "success_at_1_late": {"value": late, "bar": bar, "loose_bar": loose, "met": late >= bar},
        "attempts_to_success": {"value": tries[0], "bar": tries[1], "met": tries[0] < tries[1]},
        "gentler_force": {
            "peak_force_excess": list(excess),
            "force_jerk": list(jerk),
            "met": excess[0] < excess[1] or jerk[0] < jerk[1],
        },
    }


def _write(path: Path, payload: Any) -> None:
    B._write_json(path, payload)


# ---------------------------------------------------------------------- figures


def learning_figure(full: Mapping[str, Any], n: int, max_attempts: int, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, len(range(0, n, B.BLOCK)) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in CURVES:
        for ax, metric in ((a1, "success_at_1"), (a2, "attempts_to_success")):
            if c == "random_retry" and metric == "success_at_1":
                continue  # its first attempts are naive's
            mean = np.array(full[c]["blocks"][metric]["mean"])
            std = np.array(full[c]["blocks"][metric]["std"])
            ax.plot(x, mean, color=COLORS[c], lw=2.2 if c == "memory_full" else 1.4, label=c)
            ax.fill_between(x, mean - std, mean + std, color=COLORS[c], alpha=0.12, lw=0)
    a1.set_ylabel("success@1 (first attempt)")
    a2.set_ylabel(f"attempts-to-success (unsolved = {max_attempts + 1})")
    for ax in (a1, a2):
        ax.set_xlabel(f"block of {B.BLOCK} situations (one cube per class)")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    a1.set_ylim(0.0, 1.05)
    a1.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Learning curves on the 3 evaluation seeds (mean ± std over seeds; "
        "random_retry's first attempts are naive's)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def figures(
    config: B.BenchmarkConfig,
    out: Path = B.DEFAULT_OUT,
    workers: int = 2,
    deadline: float = DEADLINE_S,
    figures_dir: Path = FIGURES,
) -> int:
    """Force-profile examples (refer to the module doc). Returns an exit code."""
    seed = config.profiles["full"].seeds[0]
    fig_out = out / "t18_figures"
    save = (fig_out / "episodes").resolve()
    conditions = ("random_retry", "memory_full")
    jobs = {
        c: B.plan_condition(config, c, "full", [seed], out=fig_out, save_episodes=str(save))
        for c in conditions
    }
    if not B.run_jobs([j for js in jobs.values() for j in js], workers, deadline):
        return 3
    records = {}
    for c in conditions:
        scored = B.load_records(B.plan_condition(config, c, "full", [seed], out=out))[seed]
        rerun = B.load_records(jobs[c])[seed]
        if _strip(scored) != _strip(rerun):
            print(f"{c}: the re-run differs from the scored chunk", file=sys.stderr)
            return 4
        records[c] = rerun
    force_figure(records, save, seed, figures_dir / "t18_force_profiles.png")
    print(f"wrote {figures_dir / 't18_force_profiles.png'}")
    return 0


def _strip(records: Sequence[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "wall_s"} for r in records]


def force_figure(
    records: Mapping[str, Sequence[dict]], episodes: Path, seed: int, path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vfa.sim.recorder import load_episode
    from vfa.viz.plots import _squeeze_time

    wins = {c: {r["situation"]: r for r in rs if r["success"]} for c, rs in records.items()}
    classes = sorted({r["class"] for rs in records.values() for r in rs})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for ax, cls in zip(axes.flat, classes, strict=False):
        both = sorted(
            i
            for i in wins["memory_full"]
            if i in wins["random_retry"] and wins["memory_full"][i]["class"] == cls
        )
        if not both:
            ax.set_title(f"{cls}: no cube lifted by both", fontsize=9)
            continue
        i = both[0]
        for c, color in (
            ("random_retry", COLORS["random_retry"]),
            ("memory_full", COLORS["memory_full"]),
        ):
            r = wins[c][i]
            ep = load_episode(episodes / c / f"seed{seed}_s{i:03d}_a{r['attempt']}.npz")
            t, _ = _squeeze_time(ep)
            ax.plot(
                t,
                ep.arrays["finger_force"].sum(axis=1),
                color=color,
                lw=1.6,
                label=(
                    f"{c}: attempt {r['attempt']}, {r['theta_final']['grip_force']:.1f} N/finger, "
                    f"excess {r['peak_force_excess']:.1f} N"
                ),
            )
        ax.set_title(f"{cls}, situation {i}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(-1.0, 3.0)
    for ax in axes[1]:
        ax.set_xlabel("time from the squeeze [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("summed pad force [N]")
    fig.suptitle(
        f"Force profiles of the successful attempt, seed {seed}: the first cube of each class "
        "that both conditions lift (excess: peak summed force − m(g + a)/μ)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------- CLI


def _pm(stat: Mapping[str, float | None], digits: int = 3) -> str:
    mean, std = stat["mean"], stat["std"]
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f}" + ("" if std is None else f" ± {std:.{digits}f}")


def print_tables(result: Mapping[str, Any]) -> None:
    full = result["profiles"]["full"]
    print(
        "\n| Condition | success@1 | success@1, 21–60 | success@5 | attempts-to-success "
        "| break rate | peak-force excess [N] | force jerk [N/s] |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for c, m in full.items():
        print(
            f"| `{c}` | {_pm(m['success_at_1'])} | {_pm(m['success_at_1_late'])} "
            f"| {_pm(m['success_at_k'])} | {_pm(m['attempts_to_success'], 2)} "
            f"| {_pm(m['break_rate'])} | {_pm(m['peak_force_excess'], 2)} "
            f"| {_pm(m['force_jerk'], 1)} |"
        )
    print(
        "\n| vs `memory_full` | Δ success@1, 21–60 | Δ attempts-to-success | Δ break rate "
        "| verdict |"
    )
    print("|---|---|---|---|---|")
    for c, comp in result["comparisons"].items():
        cells = []
        for m, d in (("success_at_1_late", 3), ("attempts_to_success", 2), ("break_rate", 3)):
            lo, hi = comp["interval"][m]
            cells.append(f"{_pm(comp['diff'][m], d)} [{lo:+.{d}f}, {hi:+.{d}f}]")
        print(f"| `{c}` | {' | '.join(cells)} | {comp['verdict']} |")
    held = result["profiles"]["heldout"]
    print(
        "\n| Held-out (seed 2000) | success@1 | success@1, 21–60 | success@5 "
        "| attempts-to-success | break rate |"
    )
    print("|---|---|---|---|---|---|")
    for c, m in held.items():
        print(
            f"| `{c}` | {m['success_at_1']['mean']:.3f} | {m['success_at_1_late']['mean']:.3f} "
            f"| {m['success_at_k']['mean']:.3f} | {m['attempts_to_success']['mean']:.2f} "
            f"| {m['break_rate']['mean']:.3f} |"
        )
    t = result["targets"]
    s = t["success_at_1_late"]
    print(
        f"\ntarget 1 {'MET' if s['met'] else 'MISSED'}: memory_full success@1 21-60 "
        f"{s['value']:.3f} vs bar {s['bar']:.3f} (looser bar {s['loose_bar']:.3f})"
    )
    a = t["attempts_to_success"]
    print(
        f"target 2 {'MET' if a['met'] else 'MISSED'}: attempts-to-success {a['value']:.3f} "
        f"vs random_retry {a['bar']:.3f}"
    )
    g = t["gentler_force"]
    print(
        f"target 3 {'MET' if g['met'] else 'MISSED'}: peak-force excess "
        f"{g['peak_force_excess'][0]:.2f} vs {g['peak_force_excess'][1]:.2f} N, "
        f"force jerk {g['force_jerk'][0]:.1f} vs {g['force_jerk'][1]:.1f} N/s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = B.load_benchmark_config()
    command = argv[0] if argv and not argv[0].startswith("-") else "run"
    if argv and argv[0] == command:
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=B.DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=B.default_workers())
    parser.add_argument("--deadline", type=float, default=DEADLINE_S, help="seconds")
    parser.add_argument("--figures", type=Path, default=FIGURES, help="figure directory")
    args = parser.parse_args(argv)
    if command == "run":
        return 0 if B.run_jobs(plan(config, args.out), args.workers, args.deadline) else 3
    if command == "aggregate":
        try:
            result = aggregate(config, args.out, args.figures)
        except FileNotFoundError as exc:
            print(f"{exc}; run `python -m vfa.experiments.run_all` until done", file=sys.stderr)
            return 3
        print_tables(result)
        print(f"\nwrote {args.out / 't18' / 'results.json'}")
        return 0
    if command == "figures":
        return figures(config, args.out, args.workers, args.deadline, args.figures)
    raise SystemExit(f"unknown command {command!r} (run, aggregate, figures)")


if __name__ == "__main__":
    raise SystemExit(main())
