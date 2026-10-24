"""A self-contained HTML report of a demo run (T17).

``write_report(run_dir)`` draws the plots of ``vfa.viz.plots`` into
``<run>/plots/`` and writes ``<run>/report.html`` with every image inlined as
base64: the summary table, the force profiles, the retrieval hits, the learning
curves, the embedding map, and a "RuVector components" section (the active
backend, the mode timeline per attempt, the item count per memory tier, and the
skill list with confidence).

RuVector mapping: the "RuVector components" section shows
``cognitive_core::CognitiveCore`` (its modes per attempt),
``memory_system::{WorkingMemory, EpisodicMemory, SemanticMemory}`` (their item
counts), and ``skill_learning::SkillLibrary`` (its skills), running on the
named backend.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from vfa.viz import plots as P

PLOTS = (
    ("force_overlay", "Force profiles", P.force_overlay),
    ("retrieval_hits", "Retrieval hits", P.retrieval_hits),
    ("learning_curves", "Learning curves", P.learning_curves),
    ("embedding_map", "Embedding map (diagnostic)", P.embedding_map),
)
COMPONENT_PLOTS = (
    ("mode_timeline", "Mode timeline per attempt (CognitiveCore)", P.mode_timeline),
    ("memory_tiers", "Item count per memory tier", P.memory_tiers),
)

STYLE = """
body { font: 15px/1.45 -apple-system, system-ui, sans-serif; max-width: 980px;
  margin: 24px auto; padding: 0 16px; color: #111827; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 28px; } h3 { font-size: 15px; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border-bottom: 1px solid #e5e7eb; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
img { max-width: 100%; border: 1px solid #e5e7eb; }
.note { color: #4b5563; font-size: 13px; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
"""


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _summary(run: P.Run) -> str:
    rows = []
    n = run.metrics["situations"]
    for name, m in run.metrics["conditions"].items():
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{m['success_at_1']:.3f}</td>"
            f"<td>{m['second_half']:.3f}</td><td>{m['success_at_k']:.3f}</td>"
            f"<td>{m['attempts_to_success']:.2f}</td><td>{m['break_rate']:.3f}</td></tr>"
        )
    head = (
        "<tr><th>condition</th><th>success@1</th>"
        f"<th>success@1, cubes {n // 2}–{n - 1}</th><th>success@{run.metrics['max_attempts']}</th>"
        "<th>attempts-to-success</th><th>break rate</th></tr>"
    )
    return f"<table>{head}{''.join(rows)}</table>"


def _skills(run: P.Run) -> str:
    skills = run.metrics.get("skills", [])
    if not skills:
        return "<p>No skill was consolidated.</p>"
    rows = "".join(
        f"<tr><td><code>{html.escape(s['name'])}</code></td><td>{s['members']}</td>"
        f"<td>{s['confidence']:.2f}</td><td>{s['grip_force']:.2f}</td></tr>"
        for s in skills
    )
    head = "<tr><th>skill</th><th>successes</th><th>confidence</th><th>mean grip [N]</th></tr>"
    return f"<table>{head}{rows}</table>"


def write_report(run_dir: str | Path) -> Path:
    """Draw the plots and write ``report.html``; returns its path."""
    run = P.Run.load(run_dir)
    plots_dir = run.path / "plots"
    images = {name: fn(run, plots_dir / f"{name}.png") for name, _, fn in PLOTS + COMPONENT_PLOTS}
    m = run.metrics
    backend = html.escape(str(m.get("backend", "unknown")))
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>ruvector-vfa demo, seed {m['seed']}</title><style>{STYLE}</style></head><body>",
        "<h1>ruvector-vfa demo report</h1>",
        f"<p>{m['situations']} cubes of benchmark seed {m['seed']}, up to {m['max_attempts']} "
        f"attempts each; {m['wall_s']:.0f} s. The arm never sees a cube's class.</p>",
        "<h2>Summary</h2>",
        _summary(run),
    ]
    for name, title, _ in PLOTS:
        parts += [f"<h2>{html.escape(title)}</h2>", _img(images[name], title)]
    parts += [
        "<h2>RuVector components</h2>",
        f"<p>Backend: <code>{backend}</code> ({html.escape(_backend_note(backend))}).</p>",
    ]
    for name, title, _ in COMPONENT_PLOTS:
        parts += [f"<h3>{html.escape(title)}</h3>", _img(images[name], title)]
    parts += [
        "<h3>Skill library (SemanticMemory prototypes)</h3>",
        _skills(run),
        "<p class='note'>Working memory holds one attempt's items and is not used by the "
        "T15 policy, so its count stays 0.</p>",
        "</body></html>",
    ]
    out = run.path / "report.html"
    out.write_text("\n".join(parts))
    return out


def _backend_note(backend: str) -> str:
    from vfa.experiments.demo import BACKEND_NOTES  # one wording for the console and the page

    return BACKEND_NOTES.get(backend, "")
