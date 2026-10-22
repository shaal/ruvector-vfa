"""T06: the episode recorder (``vfa.sim.recorder``) and the force-profile plot.

ROADMAP acceptance: the recorder round-trips through save/load exactly, and a
force-profile PNG renders for one recorded episode.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from enum import Enum

import numpy as np
import pytest

from vfa.sim import scene as S
from vfa.sim.recorder import (
    COLUMNS,
    PRIVILEGED_COLUMNS,
    RESULTS,
    SUMMARY_FILE,
    Episode,
    EpisodeRecorder,
    load_episode,
)
from vfa.sim.sensors import Event, EventRecord, PrivilegedState, SensorFrame
from vfa.viz import force_plot

_SPEC = importlib.util.spec_from_file_location(
    "record_episode", S.REPO_ROOT / "scripts" / "record_episode.py"
)
RE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("record_episode", RE)
_SPEC.loader.exec_module(RE)


@pytest.fixture(scope="module")
def episode() -> Episode:
    """A real recording: the scripted light_rigid grasp, opened mid-lift (it drops)."""
    return RE.run_scripted_grasp("light_rigid", open_at=0.6)


def _frame(t: float, k: int = 0) -> SensorFrame:
    rng = np.random.default_rng(k)
    return SensorFrame(
        t=t,
        wrench_raw=rng.normal(size=6),
        wrench=np.full(6, np.nan) if k < 2 else rng.normal(size=6),
        acc=rng.normal(size=3),
        finger_force=rng.normal(size=2),
        contact=np.array([k % 2 == 0, k % 3 == 0]),
        width=0.05 + 1e-3 * k,
        width_rate=-0.01 * k,
        ee_pos=rng.normal(size=3),
        ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ee_vel=rng.normal(size=6),
        load=math.nan if k < 2 else 0.1 * k,
        peak_load=math.nan if k < 2 else 0.2 * k,
        armed=k > 3,
        watching=k > 2,
        watch_id=1 if k > 2 else 0,
        slip=k > 5,
        slip_time=0.01 if k > 5 else None,
    )


def _priv(k: int) -> PrivilegedState:
    return PrivilegedState(
        cube_pos=np.array([0.5, 0.0, 0.025 + 1e-3 * k]),
        cube_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        cube_vel=np.zeros(6),
        cube_finger_force=np.array([float(k), 2.0 * k]),
        broken=k > 6,
        break_time=None,
        rest_z=0.025,
        place_xy=np.array([0.5, 0.25]),
    )


def test_real_episode_round_trips_exactly(episode: Episode, tmp_path) -> None:
    """ROADMAP T06 acceptance: save → load returns the same episode, bit for bit."""
    path = episode.save(tmp_path / "ep")
    assert path.name == "ep.npz"
    loaded = load_episode(path)
    assert loaded == episode
    assert loaded.arrays.keys() == COLUMNS.keys()
    for name, (shape, dtype) in COLUMNS.items():
        a, b = episode.arrays[name], loaded.arrays[name]
        assert a.dtype == b.dtype == dtype and a.shape == b.shape == (episode.n, *shape)
        assert a.tobytes() == b.tobytes()  # NaNs included
    assert loaded.events == episode.events and loaded.phases == episode.phases
    assert [e.name for e in loaded.events][-2:] == [Event.SLIP, Event.DROP]
    assert loaded.params == {"grip_force": 4.0, "close_speed": 0.1}
    assert loaded.params_history == [{"t": None, "params": loaded.params}]
    # The observable start-of-attempt context (T11's key_pre), not the hidden truth.
    assert set(loaded.context) == {"cube", "arm_qpos"} and len(loaded.context["arm_qpos"]) == 7
    assert loaded.context["cube"]["size"] == pytest.approx(loaded.meta["size"])
    assert loaded.outcome["result"] == "DROP" and loaded.outcome["success"] is False
    assert loaded.meta["hidden"]["class_name"] == "light_rigid"
    # One changed sample anywhere breaks equality.
    loaded.arrays["finger_force"][100, 1] += 1e-12
    assert loaded != episode


def test_summary_line_is_strict_json_and_appended(episode: Episode, tmp_path) -> None:
    episode.save(tmp_path / "a")
    episode.save(tmp_path / "b")
    lines = (tmp_path / SUMMARY_FILE).read_text().splitlines()
    assert len(lines) == 2

    def no_nan(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant {token}")

    first = json.loads(lines[0], parse_constant=no_nan)
    assert first["file"] == "a.npz" and json.loads(lines[1])["file"] == "b.npz"
    assert first["result"] == "DROP" and first["success"] is False
    assert [name for name, _ in first["events"]] == [e.name.value for e in episode.events]
    assert first["ticks"] == episode.n
    assert [p[0] for p in first["phases"]] == ["APPROACH", "DESCEND", "SQUEEZE", "LIFT", "HOLD"]
    assert first["peak_cube_finger_force"] == pytest.approx(8.0, rel=0.05)  # 2 × 4 N
    # A custom summary file, and none.
    episode.save(tmp_path / "c", summary=tmp_path / "other.jsonl")
    episode.save(tmp_path / "d", summary=None)
    assert len((tmp_path / "other.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / SUMMARY_FILE).read_text().splitlines()) == 2


def test_synthetic_episode_round_trips_with_nans_strides_and_growth(tmp_path) -> None:
    rec = EpisodeRecorder(meta={"seed": 1, "tags": ("a", "b")}, stride=3, capacity=2)
    events = [EventRecord(Event.CONTACT_ONSET, 0.004, {"left": np.float64(0.5), "n": np.int64(2)})]
    for k in range(10):
        rec.record(_frame(0.002 * (k + 1), k), "A" if k < 5 else "B", _priv(k) if k % 2 else None)
        if k == 1:
            rec.add_events(events)
    rec.set_params({"grip_force": 2.5, "vec": np.array([1.0, 2.0])})
    ep = rec.finish("SLIP", note="x")
    assert ep.n == 4  # rows 0, 3, 6, 9 of 10
    np.testing.assert_allclose(ep.arrays["t"], [0.002, 0.008, 0.014, 0.020])
    assert ep.phases == ("A", "B") and ep.phase_names() == ["A", "A", "B", "B"]
    assert ep.phase_spans() == [("A", 0.002, 0.014), ("B", 0.014, 0.020)]
    # Rows without a PrivilegedState: NaN (and not broken).
    assert np.isnan(ep.arrays["cube_pos"][0]).all() and not ep.arrays["broken"][0]
    np.testing.assert_array_equal(ep.arrays["cube_finger_force"][1], [3.0, 6.0])
    assert ep.meta == {"seed": 1, "tags": ["a", "b"]}  # stored as JSON returns it
    assert ep.params == {"grip_force": 2.5, "vec": [1.0, 2.0]}
    assert ep.events[0].info == {"left": 0.5, "n": 2}
    assert ep.outcome == {"result": "SLIP", "success": False, "note": "x"}
    assert load_episode(ep.save(tmp_path / "syn.npz")) == ep
    assert set(PRIVILEGED_COLUMNS) <= set(ep.arrays)


def test_snapshot_copies_the_observable_rows_so_far() -> None:
    """T11: a mid-attempt view (``snapshot``) holds no privileged column."""
    rec = EpisodeRecorder(capacity=2)
    for k in range(5):
        rec.record(_frame(0.002 * (k + 1), k), "A" if k < 3 else "B", _priv(k))
    arrays, phases = rec.snapshot()
    assert phases == ("A", "B")
    assert not set(PRIVILEGED_COLUMNS) & set(arrays)
    assert all(len(v) == 5 for v in arrays.values())
    arrays["t"][0] = -1.0  # a copy: the recorder is unchanged
    assert rec.finish("SUCCESS").arrays["t"][0] == pytest.approx(0.002)


def test_recorder_rejects_bad_use(tmp_path) -> None:
    rec = EpisodeRecorder()
    with pytest.raises(ValueError, match="phase"):
        rec.record(_frame(0.002), "")
    rec.record(_frame(0.002), "A")
    with pytest.raises(ValueError, match="increase"):
        rec.record(_frame(0.002), "A")  # the same tick twice
    with pytest.raises(TypeError):
        rec.add_events(["SLIP"])
    with pytest.raises(TypeError):
        rec.set_params({"f": object()})
    with pytest.raises(TypeError):
        EpisodeRecorder(meta=[1, 2])
    with pytest.raises(ValueError):
        EpisodeRecorder(stride=0)
    with pytest.raises(ValueError, match="result"):
        rec.finish("FELL_OVER")
    with pytest.raises(TypeError):
        rec.finish("SLIP", success=1)
    ep = rec.finish("SUCCESS")
    assert ep.success and ep.result == "SUCCESS" and "SUCCESS" in RESULTS
    with pytest.raises(RuntimeError):
        rec.record(_frame(0.004), "A")
    with pytest.raises(RuntimeError):
        rec.finish("SUCCESS")
    np.savez(tmp_path / "other.npz", t=np.zeros(3))
    with pytest.raises(ValueError, match="not an episode"):
        load_episode(tmp_path / "other.npz")
    pickled = {name: np.zeros(1) for name in COLUMNS}
    pickled["episode_json"] = np.array([{"x": 1}], dtype=object)
    np.savez(tmp_path / "pickled.npz", **pickled)
    with pytest.raises(ValueError):  # object arrays need pickle: refused
        load_episode(tmp_path / "pickled.npz")
    (tmp_path / "garbage.npz").write_bytes(b"not a zip file at all")
    with pytest.raises(ValueError, match="not a readable episode"):
        load_episode(tmp_path / "garbage.npz")


def test_force_profile_png_renders(episode: Episode, tmp_path) -> None:
    """ROADMAP T06 acceptance: a force-profile PNG renders for one recorded episode."""
    out = force_plot.plot_force_profile(episode, tmp_path / "plots" / "profile.png")
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000
    # The CLI re-plots a saved episode.
    saved = episode.save(tmp_path / "ep")
    assert force_plot.main([str(saved), "--png", str(tmp_path / "cli.png")]) == 0
    assert (tmp_path / "cli.png").stat().st_size > 50_000
    with pytest.raises(ValueError, match="no recorded ticks"):
        force_plot.plot_force_profile(EpisodeRecorder().finish("MISS"), tmp_path / "empty.png")


def test_record_episode_cli_writes_episode_summary_and_png(tmp_path, capsys) -> None:
    png = tmp_path / "p.png"
    argv = ["--class", "fragile_soft", "--grip", "8", "--speed", "0.5", "--out", str(tmp_path)]
    assert RE.main([*argv, "--png", str(png)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("BREAK") and "CONTACT_ONSET" in out and "BREAK" in out
    assert (tmp_path / "fragile_soft_s0_break.npz").exists() and png.exists()
    assert json.loads((tmp_path / SUMMARY_FILE).read_text())["result"] == "BREAK"


class Phase(str, Enum):  # noqa: UP042 (the review's case: a str mixin, not StrEnum)
    APPROACH = "APPROACH"
    LIFT = "LIFT"


def test_phase_names_round_trip_exactly(tmp_path) -> None:
    """Review: (str, Enum) members were garbled ('P1.APPRO'); a trailing NUL was lost."""
    rec = EpisodeRecorder()
    for k, phase in enumerate((Phase.APPROACH, Phase.LIFT, "A\x00", "ünïcode")):
        rec.record(_frame(0.002 * (k + 1), k), phase)
    ep = rec.finish("SUCCESS")
    assert ep.phases == ("APPROACH", "LIFT", "A\x00", "ünïcode")
    assert all(type(p) is str for p in ep.phases)
    assert load_episode(ep.save(tmp_path / "p")) == ep


def test_context_and_params_history_round_trip(tmp_path) -> None:
    rec = EpisodeRecorder(params={"grip_force": 1.0}, context={"cube": (0.5, 0.0, 0.025)})
    rec.record(_frame(0.002), "PROBE")
    rec.set_params({"grip_force": 6.0}, t=0.002)  # System 2's update at PROBE_DONE
    rec.record(_frame(0.004), "SQUEEZE")
    ep = rec.finish("SUCCESS")
    assert ep.params == {"grip_force": 6.0}
    assert ep.params_history == [
        {"t": None, "params": {"grip_force": 1.0}},
        {"t": 0.002, "params": {"grip_force": 6.0}},
    ]
    assert ep.context == {"cube": [0.5, 0.0, 0.025]}
    loaded = load_episode(ep.save(tmp_path / "c"))
    assert loaded == ep and loaded.params_history == ep.params_history
    with pytest.raises(ValueError):
        EpisodeRecorder().set_params({"a": 1}, t=math.nan)


def test_a_failed_call_leaves_the_recorder_unchanged() -> None:
    """Review: a record() that failed on a bad array still registered its phase and
    advanced the clock; add_events kept the records before a bad one."""
    rec = EpisodeRecorder()
    bad = _frame(0.002)
    bad.wrench = np.zeros(5)
    with pytest.raises(ValueError):
        rec.record(bad, "BAD")
    rec.record(_frame(0.002), "GOOD")  # the same time is fine: the bad call left no trace
    ok = EventRecord(Event.SLIP, 0.002, {})
    with pytest.raises(TypeError):
        rec.add_events([ok, "SLIP"])
    assert rec.events == []
    rec.add_events([EventRecord("DROP", 0.004, {})])  # a plain-str name becomes an Event
    ep = rec.finish("DROP")
    assert ep.phases == ("GOOD",) and ep.n == 1 and ep.events[0].name is Event.DROP


def test_save_is_atomic_and_places_the_summary(tmp_path, monkeypatch) -> None:
    rec = EpisodeRecorder()
    rec.record(_frame(0.002), "A")
    ep = rec.finish("SUCCESS")
    # The summary may live elsewhere; its "file" is relative to it.
    path = ep.save(tmp_path / "runs" / "seed0" / "ep", summary=tmp_path / "index" / "all.jsonl")
    line = json.loads((tmp_path / "index" / "all.jsonl").read_text())
    assert line["file"] == "../runs/seed0/ep.npz" and path.exists()
    # Saved like open() would (round-3 review: mkstemp made it owner-only).
    umask = os.umask(0)
    os.umask(umask)
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask
    # A long name still fits once the temporary prefix is added (round-3 review).
    long_path = ep.save(tmp_path / "runs" / ("x" * 240), summary=None)
    assert load_episode(long_path) == ep

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError, match="disk full"):
        ep.save(tmp_path / "broken")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index", "runs"]  # no file, no tmp


def test_load_rejects_a_damaged_layout(tmp_path) -> None:
    rec = EpisodeRecorder()
    for k in range(3):
        rec.record(_frame(0.002 * (k + 1), k), "A")
    path = rec.finish("SUCCESS").save(tmp_path / "ep", summary=None)
    with np.load(path) as npz:
        good = {k: npz[k] for k in npz.files}
    for name, value in (
        ("wrench", np.zeros((3, 5))),
        ("t", np.zeros(4)),
        ("phase", np.array([0, 0, 7], dtype=np.int32)),
        ("contact", np.zeros((3, 2), dtype=np.float64)),
    ):
        np.savez(tmp_path / "bad.npz", **{**good, name: value})
        with pytest.raises(ValueError):
            load_episode(tmp_path / "bad.npz")
    # Damaged metadata: a ValueError, never a KeyError or TypeError (round-2 review).
    doc = json.loads(str(good["episode_json"]))
    for broken in (
        {k: v for k, v in doc.items() if k != "context"},
        {**doc, "phases": "ABC"},
        {**doc, "events": [{"name": "SLIP", "t": 1.0}]},
        {**doc, "events": [{"name": "SLIP", "t": "x", "info": {}}]},
        {**doc, "outcome": {"success": True}},
        {**doc, "events": {}},
        [1, 2, 3],
    ):
        np.savez(tmp_path / "bad.npz", **{**good, "episode_json": np.array(json.dumps(broken))})
        with pytest.raises(ValueError, match="damaged episode metadata"):
            load_episode(tmp_path / "bad.npz")
    # A member that is not an .npy array (numpy hands back raw bytes), and a directory.
    import zipfile

    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp_path / "raw.npz", "w") as dst:
        for info in src.infolist():
            dst.writestr(info.filename, b"hello" if info.filename == "t.npy" else src.read(info))
    with pytest.raises(ValueError, match="not an array"):
        load_episode(tmp_path / "raw.npz")
    (tmp_path / "dir.npz").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        load_episode(tmp_path / "dir.npz")


def test_force_plot_edge_cases(tmp_path) -> None:
    rec = EpisodeRecorder(params={"grip_force": "adaptive", "close_speed": 0.1})
    rec.record(_frame(0.002), "A")
    rec.record(_frame(0.004, 3), "B")
    rec.add_events([EventRecord(Event.SAFETY_STOP, 5000.0, {})])  # outside the range
    rec.add_events([EventRecord(Event.SLIP, 0.003, {}) for _ in range(20)])  # one instant
    ep = rec.finish("SAFETY_STOP")
    out = force_plot.plot_force_profile(ep, tmp_path / "noext")
    assert out.name == "noext.png" and out.exists()


def test_record_episode_cli_rejects_bad_arguments(tmp_path) -> None:
    for argv in (
        ["--class", "granite"],
        ["--grip", "-1"],
        ["--grip", "nan"],
        ["--grip", "1e6"],
        ["--heft", "0.2"],
        ["--heft", "0.03"],
        ["--heft", "0.005"],  # below the 1-2 cm probe: payload sag fakes a slip
        ["--sample", "1000000000"],
        ["--grip", "60"],
        ["--name", "sub/../../escape"],
        ["--name", "x" * 300],
        ["--speed", "0"],
        ["--speed", "2"],
        ["--open-at", "-0.5"],
        ["--sample", "-1"],
        ["--seed", "-3"],
    ):
        with pytest.raises(SystemExit) as exc:
            RE.main([*argv, "--out", str(tmp_path)])
        assert exc.value.code == 2, argv


def test_concurrent_saves_to_one_name_leave_a_whole_file(tmp_path) -> None:
    """Round-2 review: threads shared one temporary name and left an unreadable file."""
    import threading

    rec = EpisodeRecorder()
    for k in range(50):
        rec.record(_frame(0.002 * (k + 1), k % 7), "A")
    ep = rec.finish("SUCCESS")
    errors: list[BaseException] = []

    def save() -> None:
        try:
            ep.save(tmp_path / "same", summary=None)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=save) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and load_episode(tmp_path / "same.npz") == ep
    assert [p.name for p in tmp_path.iterdir()] == ["same.npz"]  # no temporary file left


def test_plot_keeps_a_dotted_name(tmp_path) -> None:
    rec = EpisodeRecorder()
    rec.record(_frame(0.002), "A")
    out = force_plot.plot_force_profile(rec.finish("SUCCESS"), tmp_path / "drop.0.6")
    assert out.name == "drop.0.6.png" and out.exists()


def test_sizing_chunks_that_do_not_read_are_rerun(tmp_path) -> None:
    """Round-2 review: a truncated chunk broke --physics and --aggregate for good."""
    spec = importlib.util.spec_from_file_location(
        "size_slip_detector", S.REPO_ROOT / "scripts" / "size_slip_detector.py"
    )
    SZ = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("size_slip_detector", SZ)
    spec.loader.exec_module(SZ)
    bad = tmp_path / "chunk.npz"
    bad.write_bytes(b"PK\x03\x04 truncated")
    assert SZ.chunk_current(bad, "any") is False
    assert SZ.chunk_current(tmp_path / "missing.npz", "any") is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heft": 0.005},  # below the 1-2 cm probe (payload sag fakes a slip)
        {"heft": 0.03},
        {"heft": math.nan},
        {"speed": 0.0},
        {"speed": 0.6},
        {"open_at": -1.0},
        {"grip": 60.0},
        {"grip": math.inf},
    ],
)
def test_run_scripted_grasp_validates_its_arguments(kwargs) -> None:
    """Round-6 review: only the CLI path was tested; the API checks before any physics."""
    with pytest.raises(ValueError):
        RE.run_scripted_grasp("light_rigid", **kwargs)


def test_load_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="no such episode file"):
        load_episode(tmp_path / "missing.npz")
