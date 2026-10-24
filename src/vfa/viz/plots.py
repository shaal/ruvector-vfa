"""Plots of a demo run: force profiles, retrieval hits, learning curves, embedding map,
and the RuVector components' views (T17, ARCHITECTURE §9).

Every function reads a ``runs/demo-<timestamp>/`` directory (``vfa demo``) and
writes one PNG:

- ``force_overlay``: a failed attempt, the successful neighbours that System 2
  retrieved for the retry, and the retry, on the summed pad force and the held
  load, aligned at the squeeze, with the failed attempt's phases shaded.
- ``retrieval_hits``: the query of that retry's probe decision against its top-k
  neighbours as memory held them at the time (similarity, outcome, grip; the
  ones the decision used are marked).
- ``learning_curves``: rolling success@1 and attempts-to-success over the cube
  index, per condition.
- ``embedding_map``: a 2-D PCA of the stored ``key_probe`` vectors, colored by
  the hidden class (diagnostic only: the class comes from the scoring records).
- ``mode_timeline``: per attempt, System 1 (``REACTIVE``) as bars over sim time,
  System 2 decisions (``DELIBERATIVE``, by trigger), ``EMERGENCY``, and
  ``LEARNING``, from ``handoffs.jsonl``.
- ``memory_tiers``: the item count per memory tier after each cube.

RuVector mapping: ``mode_timeline`` draws ``cognitive_core::CognitiveMode``
changes; ``memory_tiers`` counts ``memory_system::{WorkingMemory,
EpisodicMemory, SemanticMemory}``; ``retrieval_hits`` and ``embedding_map`` show
``EpisodicMemory::recall_similar`` over the stored keys.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vfa.memory.memory_system import MemorySystem  # noqa: E402
from vfa.sim.recorder import load_episode  # noqa: E402

CONDITION_COLORS = {"naive": "#9ca3af", "random_retry": "#f59e0b", "memory_full": "#2563eb"}
CLASS_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")
TRIGGER_COLORS = {
    "start": "#6b7280",
    "failure": "#dc2626",
    "probe": "#2563eb",
    "escalation": "#f59e0b",
}
SHADED = ("PROBE", "SQUEEZE", "LIFT", "HOLD")
WINDOW = 6  # rolling window: one block of cubes (one per class)


@dataclass(slots=True)
class Run:
    """A demo run directory, loaded."""

    path: Path
    metrics: dict[str, Any]
    records: dict[str, list[dict[str, Any]]]
    handoffs: list[dict[str, Any]]

    @classmethod
    def load(cls, path: str | Path) -> Run:
        path = Path(path)
        records = {
            p.stem: json.loads(p.read_text()) for p in sorted((path / "records").glob("*.json"))
        }
        lines = (path / "handoffs.jsonl").read_text().splitlines()
        return cls(
            path,
            json.loads((path / "metrics.json").read_text()),
            records,
            [json.loads(line) for line in lines if line.strip()],
        )

    @property
    def memory_records(self) -> list[dict[str, Any]]:
        return self.records["memory_full"]

    def by_memory_id(self) -> dict[str, dict[str, Any]]:
        return {r["memory_id"]: r for r in self.memory_records}

    def episode(self, record: Mapping[str, Any]) -> Any:
        name = f"cube{record['situation']:03d}_a{record['attempt']}.npz"
        return load_episode(self.path / "episodes" / name)

    def probe_decision(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """The probe handoff of a memory-phase record's attempt ({} if none ran). The
        core counts attempts in memory order, so attempt n is the n-th record."""
        n = 1 + next(i for i, r in enumerate(self.memory_records) if r is record)
        return next(
            (
                h
                for h in self.handoffs
                if h["attempt"] == n and h["kind"] == "DELIBERATIVE" and h["trigger"] == "probe"
            ),
            {},
        )

    def evidence(self, record: Mapping[str, Any], role: str | None = None) -> list[dict]:
        """What that probe decision weighed (``Decision.evidence``), most similar first."""
        rows = self.probe_decision(record).get("detail", {}).get("evidence", [])
        rows = [e for e in rows if role is None or e["role"] == role]
        return sorted(rows, key=lambda e: -e["similarity"])


def _save(fig: Any, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def pick_retry(run: Run) -> tuple[dict, dict, list[dict]] | None:
    """The cube solved on a retry whose probe decision weighed the most strong
    successes (then the most similar one): (failed record, retry record, those
    successes as logged)."""
    order = run.memory_records
    best, score = None, (0, -2.0)
    for record in order:
        if not record["success"] or record["attempt"] < 2:
            continue
        hits = run.evidence(record, "success")
        mine = (len(hits), max((h["similarity"] for h in hits), default=-2.0))
        if hits and mine > score:
            failed = next(
                r
                for r in order
                if r["situation"] == record["situation"] and r["attempt"] == record["attempt"] - 1
            )
            best, score = (failed, record, hits), mine
    return best


def _merged(result: str) -> str:
    """The label System 2 sees (T14): SLIP and DROP are one under-grip."""
    return "UNDERGRIP" if result in ("SLIP", "DROP") else result


def _squeeze_time(ep: Any) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(ep.phases)[ep.arrays["phase"]]
    t = ep.arrays["t"]
    start = t[np.argmax(names == "SQUEEZE")] if np.any(names == "SQUEEZE") else t[0]
    return t - start, names


def force_overlay(run: Run, out: Path) -> Path:
    chosen = pick_retry(run)
    fig, (ax_f, ax_l) = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    if chosen is None:
        ax_f.text(0.5, 0.5, "no retry weighed a strong success in this run", ha="center")
        return _save(fig, out)
    failed, retry, hits = chosen
    decision = run.probe_decision(retry)
    used = set(decision.get("retrieved", []))
    by_id = run.by_memory_id()
    ep = run.episode(failed)
    t, names = _squeeze_time(ep)
    for phase, color in zip(SHADED, ("#f3f4f6", "#e5e7eb", "#f3f4f6", "#e5e7eb"), strict=True):
        rows = names == phase
        if np.any(rows):
            for ax in (ax_f, ax_l):
                ax.axvspan(t[rows][0], t[rows][-1], color=color, zorder=0)
            ax_f.text(
                t[rows].mean(),
                0.97,
                phase,
                transform=ax_f.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#4b5563",
            )
    series = [
        (
            by_id[h["id"]],
            "#9ca3af",
            1.0,
            f"similar success {h['id']} (sim {h['similarity']:.2f}, {h['grip']:.1f} N)"
            f"{', set the grip' if h['id'] in used else ''}",
        )
        for h in hits[:3]
        if h["id"] in by_id
    ]
    label = _merged(failed["result"])
    bound = ", bounds the retry" if failed["memory_id"] in used else ""
    series += [
        (
            failed,
            "#dc2626",
            2.0,
            f"failed: {label} at {failed['theta_final']['grip_force']:.1f} N{bound}",
        )
    ]
    series += [
        (
            retry,
            "#16a34a",
            2.0,
            f"next attempt: SUCCESS at {retry['theta_final']['grip_force']:.1f} N",
        )
    ]
    for record, color, width, text in series:
        e = ep if record is failed else run.episode(record)
        tt, _ = _squeeze_time(e)
        neighbour = color == "#9ca3af"  # dashed on top, so a neighbour at the retry's grip shows
        style = {"ls": "--", "zorder": 4} if neighbour else {"alpha": 0.85, "zorder": 3}
        ax_f.plot(
            tt, e.arrays["finger_force"].sum(axis=1), color=color, lw=width, label=text, **style
        )
        ax_l.plot(tt, e.arrays["load"], color=color, lw=width, **style)
    ax_f.set_ylabel("summed pad force [N]")
    ax_l.set_ylabel("held load [N]")
    ax_l.set_xlabel("time from the squeeze [s] (phases shaded: the failed attempt's)")
    ax_f.set_xlim(-2.0, 3.0)
    ax_f.legend(fontsize=8, loc="center left")
    reason = decision.get("detail", {}).get("reasoning", "")
    ax_f.set_title(
        f"Force profiles, cube {failed['situation']}: a failed attempt, the strong successes "
        f"System 2 weighed at the retry's probe, and the retry\nretry decision: {reason}",
        fontsize=9,
    )
    return _save(fig, out)


def retrieval_hits(run: Run, out: Path) -> Path:
    chosen = pick_retry(run)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if chosen is None:
        ax.text(0.5, 0.5, "no retry weighed a strong success in this run", ha="center")
        return _save(fig, out)
    _, retry, _ = chosen
    decision = run.probe_decision(retry)
    used = set(decision.get("retrieved", []))
    rows = [e for role in ("own", "success", "failure") for e in run.evidence(retry, role)]
    sims = [e["similarity"] for e in rows]
    colors = ["#16a34a" if e["result"] == "SUCCESS" else "#dc2626" for e in rows]
    hatches = ["//" if e["role"] == "own" else "" for e in rows]
    bars = ax.bar(range(len(rows)), sims, color=colors, hatch=hatches, edgecolor="white")
    for bar, e in zip(bars, rows, strict=True):
        tag = "\n(used)" if e["id"] in used else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{e['result']}\n{e['grip']:.1f} N{tag}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0.8, color="#6b7280", ls="--", lw=1)
    ax.text(
        1.005,
        0.8,
        "strong\n(0.8)",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color="#6b7280",
    )
    ax.set_xticks(range(len(rows)), [f"{e['id']}\n{e['role']}" for e in rows], fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("cosine similarity (key_probe)")
    ax.set_title(
        f"Retrieval hits: what System 2 weighed at the probe of {retry['memory_id']} "
        f"(cube {retry['situation']}, attempt {retry['attempt']})\nhatched = this cube's own "
        "failures (similarity 1); green = success, red = failure; labels: outcome and grip",
        fontsize=9,
    )
    return _save(fig, out)


def _per_cube(
    records: Sequence[Mapping[str, Any]], max_attempts: int
) -> tuple[np.ndarray, np.ndarray]:
    cubes = sorted({r["situation"] for r in records})
    first, tries = [], []
    for c in cubes:
        rows = sorted((r for r in records if r["situation"] == c), key=lambda r: r["attempt"])
        win = next((r["attempt"] for r in rows if r["success"]), None)
        first.append(float(win == 1))
        tries.append(float(win if win is not None else max_attempts + 1))
    return np.array(first), np.array(tries)


def _rolling(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    return np.array([values[max(0, i - window + 1) : i + 1].mean() for i in range(len(values))])


def learning_curves(run: Run, out: Path) -> Path:
    max_attempts = int(run.metrics["max_attempts"])
    fig, (ax_s, ax_a) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    styles = {"naive": (4.5, "-"), "random_retry": (2.0, "--"), "memory_full": (2.5, "-")}
    for name, records in run.records.items():
        first, tries = _per_cube(records, max_attempts)
        x = np.arange(len(first))
        color = CONDITION_COLORS.get(name, "#111827")
        width, style = styles.get(name, (2.0, "-"))
        label = name if name != "random_retry" else "random_retry (attempt 1 = naive's)"
        ax_s.plot(x, _rolling(first), color=color, lw=width, ls=style, label=label)
        ax_a.plot(x, _rolling(tries), color=color, lw=width, ls=style, label=label)
    ax_s.set_ylabel(f"success@1 (rolling, {WINDOW} cubes)")
    ax_a.set_ylabel(f"attempts to success (rolling, {WINDOW})")
    ax_a.set_xlabel(
        f"cube index (memory_full learns in this order); an unsolved cube counts {max_attempts + 1}"
    )
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.legend(fontsize=8)
    ax_s.set_title(
        "Learning curves: rolling success@1 and attempts-to-success per condition", fontsize=10
    )
    return _save(fig, out)


def embedding_map(run: Run, out: Path) -> Path:
    path = run.path / "memory"
    factory = None
    if json.loads((path / "memory.json").read_text()).get("backend") == "RuVectorBackend":
        from vfa.memory.ruvector_backend import RuVectorBackend  # a --backend ruvector demo

        factory = RuVectorBackend
    memory = MemorySystem.load(path) if factory is None else MemorySystem.load(path, factory)
    index = memory.episodic.indexes["key_probe"]
    by_id = run.by_memory_id()
    ids = [i for i in index.ids if i in by_id]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if len(ids) < 3:
        ax.text(0.5, 0.5, "too few probed attempts", ha="center")
        return _save(fig, out)
    keys = np.array([index.vector(i) for i in ids])
    centered = keys - keys.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    classes = sorted({by_id[i]["class"] for i in ids})
    for color, cls in zip(CLASS_COLORS, classes, strict=False):
        sel = np.array([by_id[i]["class"] == cls for i in ids])
        ok = sel & np.array([by_id[i]["success"] for i in ids])
        ax.scatter(xy[ok, 0], xy[ok, 1], s=30, color=color, label=cls)
        ax.scatter(xy[sel & ~ok, 0], xy[sel & ~ok, 1], s=30, facecolors="none", edgecolors=color)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8)
    ax.set_title(
        "Embedding map: key_probe of the stored attempts (2-D PCA)\n"
        "color = hidden class (diagnostic only); filled = success, hollow = failure",
        fontsize=10,
    )
    return _save(fig, out)


def mode_timeline(run: Run, out: Path) -> Path:
    by_attempt: dict[int, list[dict[str, Any]]] = {}
    for h in run.handoffs:
        by_attempt.setdefault(h["attempt"], []).append(h)
    attempts = sorted(by_attempt)
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.16 * len(attempts) + 1.2)))
    for row, n in enumerate(attempts):
        steps = by_attempt[n]
        end = max(h["t"] for h in steps)
        for i, h in enumerate(steps):
            if h["kind"] == "REACTIVE":
                nxt = next((g["t"] for g in steps[i + 1 :] if g["kind"] != "REACTIVE"), end)
                ax.broken_barh(
                    [(h["t"], max(nxt - h["t"], 1e-3))], (row - 0.35, 0.7), color="#93c5fd"
                )
            elif h["kind"] == "DELIBERATIVE":
                ax.plot(h["t"], row, "D", ms=4, color=TRIGGER_COLORS.get(h["trigger"], "#111827"))
            elif h["kind"] == "EMERGENCY":
                ax.plot(h["t"], row, "X", ms=7, color="#b91c1c")
            elif h["kind"] == "LEARNING":
                ax.plot(h["t"], row, "|", ms=8, color="#111827")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#93c5fd", label="REACTIVE (System 1)")]
    handles += [
        plt.Line2D([], [], marker="D", ls="", color=c, label=f"DELIBERATIVE: {t}")
        for t, c in TRIGGER_COLORS.items()
    ]
    handles += [
        plt.Line2D([], [], marker="X", ls="", color="#b91c1c", label="EMERGENCY"),
        plt.Line2D([], [], marker="|", ls="", color="#111827", label="LEARNING"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    ax.set_yticks(range(len(attempts)), [str(a) for a in attempts], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("sim time in the attempt [s]")
    ax.set_ylabel("attempt (memory order)")
    ax.set_title("Mode timeline per attempt (CognitiveCore, from handoffs.jsonl)", fontsize=10)
    return _save(fig, out)


def memory_tiers(run: Run, out: Path) -> Path:
    tiers = run.metrics["tiers"]
    x = [t["cube"] for t in tiers]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for name, color in (("episodic", "#2563eb"), ("semantic", "#16a34a"), ("working", "#9ca3af")):
        ax.step(x, [t[name] for t in tiers], where="post", color=color, lw=2, label=f"{name} items")
    ax.set_xlabel("after cube")
    ax.set_ylabel("items")
    ax.legend(fontsize=8)
    ax.set_title(
        "Memory tiers over the run: episodic attempts, semantic skills (consolidated every 10 "
        "cubes), working memory",
        fontsize=10,
    )
    return _save(fig, out)
