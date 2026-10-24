"""T16: the demo (``vfa demo``, ``vfa.experiments.demo``).

ROADMAP acceptance: ``vfa demo --quick`` finishes in < 3 min and exits 0; its
summary shows memory beating ``naive`` on success@1 over the second half; the
output starts with the RuVector banner and contains a ``[MEM]`` consolidation
line. The full run is ``slow`` (about 30 s); the fast tests cover the parts.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from vfa import cli
from vfa.experiments import demo
from vfa.sim import objects as OB
from vfa.viz import plots as P


def records(results_by_cube: dict[int, list[str]], cls: str = "brittle") -> list[dict]:
    return [
        {"situation": i, "attempt": a, "class": cls, "result": r, "success": r == "SUCCESS"}
        for i, results in results_by_cube.items()
        for a, r in enumerate(results, 1)
    ]


def test_the_banner_names_every_component_and_the_backend():
    text = demo.BANNER.format(backend="NumpyBackend")
    for rust in (
        "memory_system::EpisodicMemory",
        "memory_system::SemanticMemory",
        "skill_learning::SkillLibrary",
        "cognitive_core::CognitiveCore",
    ):
        assert rust in text
    assert "backend: NumpyBackend" in text
    assert "--backend ruvector" in demo.BACKEND_NOTES["NumpyBackend"]  # how to switch (T22)
    assert "PyO3" in demo.BACKEND_NOTES["RuVectorBackend"]


def test_halves_and_tables():
    recs = records(
        {0: ["BREAK", "SUCCESS"], 1: ["SUCCESS"], 2: ["DROP", "SUCCESS"], 3: ["SUCCESS"]}
    )
    assert demo.halves(recs, 4) == (0.5, 0.5)
    table = demo.cube_table({c: recs for c in demo.CONDITIONS}, 4)
    assert "2: brk ok" in table and "2: drp ok" in table and table.count("brittle") == 4
    long = records({0: ["BREAK"] * 5})
    assert "5: brk brk brk brk brk" in demo.cube_table({c: long for c in demo.CONDITIONS}, 1)
    metrics = {
        c: {
            "success_at_1": 0.5,
            "second_half": 0.5,
            "success_at_k": 1.0,
            "attempts_to_success": 1.5,
            "break_rate": 0.25,
        }
        for c in demo.CONDITIONS
    }
    assert "memory_full" in demo.summary_table(metrics, 4)


def test_the_cli_lists_the_demo_as_implemented():
    assert "(not implemented yet)" not in "\n".join(
        cli._describe(n, h, t) for n, h, t in cli._SUBCOMMANDS if n == "demo"
    )


def test_the_cli_runs_the_benchmark(capsys):
    """``vfa benchmark ...`` is ``python -m vfa.experiments.benchmark ...``."""
    assert all(name in cli._HANDLERS for name, _, _ in cli._SUBCOMMANDS)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["benchmark", "--help"])
    assert exit_info.value.code == 0 and "--condition" in capsys.readouterr().out


@pytest.mark.render
def test_the_video_grabs_overlay_frames(tmp_path):
    from PIL import Image

    from vfa.sim.env import PandaVFAEnv
    from vfa.viz.video import FrameGrab, write_gif

    env = PandaVFAEnv(
        "lift", "osc", step_hz=500, auto_carry=False, render_mode="rgb_array", render_size=(60, 80)
    )
    grab = FrameGrab(env, fps=50)  # a frame every 10 steps
    grab.reset(seed=0)
    for _ in range(10):
        grab.step(np.zeros(4))
    assert grab.frames == []  # not active
    grab.active, grab.label = True, "cube 0, attempt 1"
    for _ in range(20):
        grab.step(np.zeros(4))
    assert len(grab.frames) == 2 and grab.frames[0].size == (80, 60)
    gif = write_gif(grab.frames, tmp_path / "a.gif", grab.fps)
    assert Image.open(gif).n_frames == 2


def test_an_existing_run_directory_is_refused(tmp_path, capsys):
    assert demo.main(["--quick", "--out", str(tmp_path)]) == 2
    assert "exists" in capsys.readouterr().err


def test_the_break_monitor_lines_stay_hidden():
    def line(name: str, msg: str, args: tuple = ()) -> logging.LogRecord:
        return logging.LogRecord(name, logging.INFO, "", 0, msg, args, None)

    event = "%s event %s t=%.3f s %s"
    assert not demo._hide_ground_truth(line("vfa.sim.objects", "BREAK brittle"))
    assert not demo._hide_ground_truth(line("vfa.sim.sensors", event, ("[S1]", "BREAK", 1.0, "")))
    assert not demo._hide_ground_truth(line("vfa.sim.sensors", event, ("[S1]", "LIFTED", 1.0, "")))
    assert demo._hide_ground_truth(line("vfa.sim.sensors", event, ("[S1]", "SLIP", 1.0, "")))
    assert demo._hide_ground_truth(line("vfa.cognition.system2", "[MEM]"))


@pytest.mark.slow
def test_the_quick_demo(tmp_path, capsys):
    out = tmp_path / "run"
    vfa_log = logging.getLogger("vfa")
    before = (list(vfa_log.handlers), vfa_log.level, vfa_log.propagate)
    assert demo.main(["--quick", "--out", str(out)]) == 0
    assert (list(vfa_log.handlers), vfa_log.level, vfa_log.propagate) == before
    text = capsys.readouterr().out
    assert text.startswith("ruvector-vfa demo")
    assert "[MEM] consolidated skill_" in text and "[MEM] hit " in text
    live = text[: text.index("Per cube")]
    assert not any(cls in live for cls in OB.load_object_config().names)  # hidden until the end
    assert "event BREAK" not in live and "event LIFTED" not in live and "attempt BREAK" in live
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["situations"] == demo.QUICK_SITUATIONS and metrics["wall_s"] < 180
    mem, naive = metrics["conditions"]["memory_full"], metrics["conditions"]["naive"]
    assert mem["second_half"] > naive["second_half"]
    assert (out / "memory" / "memory.json").exists() and (out / "handoffs.jsonl").exists()
    assert any((out / "episodes").iterdir())
    # T17: every plot and the report, with the RuVector components section.
    plots = {p.stem for p in (out / "plots").glob("*.png")}
    assert plots == {
        "force_overlay",
        "retrieval_hits",
        "learning_curves",
        "embedding_map",
        "mode_timeline",
        "memory_tiers",
    }
    report = (out / "report.html").read_text()
    assert report.count("data:image/png;base64,") == 6 and "<h2>RuVector components</h2>" in report
    assert "Backend: <code>NumpyBackend</code>" in report and "skill_00" in report
    assert len(metrics["tiers"]) == demo.QUICK_SITUATIONS and metrics["skills"]
    run = P.Run.load(out)
    ids = [r["memory_id"] for r in run.memory_records]
    assert len(set(ids)) == len(ids) == len(run.memory_records)
    failed, retry, hits = P.pick_retry(run)  # the force and retrieval plots have data
    assert not failed["success"] and retry["success"] and hits
    assert all(h["role"] == "success" and h["similarity"] >= 0.8 for h in hits)
