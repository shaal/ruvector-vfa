"""Force-profile plot of one recorded episode (T06).

``plot_force_profile(episode, path)`` draws four stacked panels on one time
axis: the fingertip pad forces, the tared wrist force with the held-load
estimate, the gripper opening, and the heights of the fingertips and the cube.
Gray bands mark the phases (labelled on top); dashed lines mark the events
(labelled above the plot). The privileged signals (the cube's contact force and
height) are dashed, so the plot shows what System 1 sensed and what really
happened side by side. Each measure has its own panel and y axis (no dual axes).

Uses matplotlib's Agg canvas directly (no ``pyplot`` state), so it runs headless
and in worker processes.

Usage::

    uv run python -m vfa.viz.force_plot runs/t06/light_rigid_s0_success.npz --png runs/t06/p.png
"""

from __future__ import annotations

import argparse
import numbers
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from vfa.sim.recorder import Episode, load_episode

# Validated categorical slots 1-3 (dataviz reference palette, light mode; slot 3
# is below 3:1 on the surface, so every series also has a legend label).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
IMAGE_SUFFIXES = (".png", ".pdf", ".svg", ".jpg", ".jpeg")
BAND = "#efeeea"


def plot_force_profile(episode: Episode, path: str | Path, dpi: int = 110) -> Path:
    """Write the force profile of ``episode`` as a PNG at ``path``; return the path.

    A path without an image suffix gets ``.png`` appended. Events outside the
    recorded time range are not drawn.
    """
    if episode.n == 0:
        raise ValueError("the episode has no recorded ticks")
    a = episode.arrays
    t = a["t"]
    fig = Figure(figsize=(10.0, 9.0), facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 1, sharex=True, height_ratios=(1.2, 1.2, 0.8, 0.9))
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        _phase_bands(ax, episode)

    ax = axes[0]
    ax.plot(t, a["finger_force"][:, 0], color=SERIES[0], lw=1.6, label="left pad")
    ax.plot(t, a["finger_force"][:, 1], color=SERIES[1], lw=1.6, label="right pad")
    cube = a["cube_finger_force"].sum(axis=1)
    if np.isfinite(cube).any():
        ax.plot(t, cube, color=SERIES[2], lw=1.4, ls="--", label="cube contact, both (truth)")
    _style(ax, "Finger force [N]")

    ax = axes[1]
    ax.plot(t, a["wrench"][:, 2], color=SERIES[0], lw=1.6, label="tared wrist $F_z$ (up)")
    fxy = np.hypot(a["wrench"][:, 0], a["wrench"][:, 1])
    ax.plot(t, fxy, color=SERIES[1], lw=1.6, label=r"tared wrist $|F_{xy}|$")
    ax.plot(t, a["load"], color=SERIES[2], lw=1.6, label="held load (F/T + accelerometer)")
    _style(ax, "Wrist force [N]")

    ax = axes[2]
    ax.plot(t, 1000.0 * a["width"], color=SERIES[0], lw=1.6)
    _style(ax, "Gripper opening [mm]", legend=False)

    ax = axes[3]
    ax.plot(t, 100.0 * a["ee_pos"][:, 2], color=SERIES[0], lw=1.6, label="fingertip center (ee)")
    if np.isfinite(a["cube_pos"][:, 2]).any():
        cube_z = 100.0 * a["cube_pos"][:, 2]
        ax.plot(t, cube_z, color=SERIES[1], lw=1.4, ls="--", label="cube center (truth)")
    _style(ax, "Height [cm]")
    ax.set_xlabel("time [s]", color=INK_2, fontsize=9)
    t0, t1 = float(t[0]), float(t[-1])
    pad = 0.0 if t1 > t0 else 0.001  # a one-row episode still gets a time axis
    ax.set_xlim(t0 - pad, t1 + pad)

    _event_lines(axes, episode)
    fig.suptitle(_title(episode), color=INK, fontsize=11, x=0.01, ha="left")
    path = Path(path)
    if path.suffix.lower() not in IMAGE_SUFFIXES:  # "run.001" keeps its name, plus .png
        path = path.with_name(path.name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


def _style(ax, ylabel: str, legend: bool = True) -> None:
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if legend:
        ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_2)


def _phase_bands(ax, episode: Episode) -> None:
    for k, (_, t0, t1) in enumerate(episode.phase_spans()):
        if k % 2 == 0:
            ax.axvspan(t0, t1, color=BAND, lw=0, zorder=0)


def _event_lines(axes, episode: Episode) -> None:
    top = axes[0]
    spans = episode.phase_spans()
    for name, t0, t1 in spans:
        top.text(
            0.5 * (t0 + t1),
            1.0,
            name,
            transform=top.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
        )
    t_lo, t_hi = top.get_xlim()
    width = max(t_hi - t_lo, 1e-9)
    last_x = -np.inf
    level = 0
    for e in episode.events:
        if not t_lo <= e.t <= t_hi:
            continue
        for ax in axes:
            ax.axvline(e.t, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=3)
        # Stagger labels that would collide (closer than 4 % of the axis), at most 6 deep.
        level = (level + 1) % 6 if (e.t - last_x) / width < 0.04 else 0
        last_x = e.t
        top.annotate(
            e.name.value,
            xy=(e.t, 0.97),
            xycoords=top.get_xaxis_transform(),
            xytext=(2 + 9 * level, 0),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK,
        )


def _title(episode: Episode) -> str:
    meta, params = episode.meta, episode.params
    parts = [str(meta.get("class", "episode"))]
    for key, label, unit in (("grip_force", "grip", "N/finger"), ("close_speed", "close", "m/s")):
        value = params.get(key)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            parts.append(f"{label} {value:g} {unit}")
        elif value is not None:
            parts.append(f"{label} {value}")
    parts.append(f"→ {episode.result}")
    return "  ·  ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the force profile of a saved episode.")
    parser.add_argument("episode", type=Path, help="episode .npz written by Episode.save")
    parser.add_argument("--png", type=Path, required=True, help="output PNG path")
    args = parser.parse_args(argv)
    print(plot_force_profile(load_episode(args.episode), args.png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
