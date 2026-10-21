"""Scene check: build the scene, hold ``home`` under gravity compensation, render a PNG.

Usage::

    uv run python -m vfa.sim.scene_check --png runs/scene.png
    uv run vfa scene-check --png runs/scene.png

The hold applies ``τ = qfrc_bias`` (gravity + Coriolis) to the seven torque
motors every physics step, split with ``mj_step1``/``mj_step2`` so the torque
matches the state it is applied to. No position feedback: any drift is the
error of pure gravity compensation.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from vfa.sim.scene import (
    OVERVIEW_CAMERA,
    build_scene,
    read_finger_touch,
    read_wrist_ft,
    reset_to_home,
    scene_ids,
)


@dataclass(slots=True)
class HoldResult:
    """Outcome of a gravity-compensated hold."""

    seconds: float
    ee_drift_m: float  # ‖ee(t_end) − ee(t_0)‖
    max_joint_drift_rad: float  # max_i |q_i(t_end) − q_i(t_0)| over the arm joints
    wrist_force: np.ndarray  # raw wrist force [N], site frame, at t_end
    wrist_torque: np.ndarray  # raw wrist torque [N·m], site frame, at t_end
    expected_force_n: float  # hand-subtree weight: subtree mass · |g|
    finger_touch: tuple[float, float]  # left/right touch sensors [N] at t_end

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.wrist_force))


# Upper bound on the hold [s of sim time]. 60 s runs in about 0.4 s wall time,
# far inside the 9-minute command budget (CLAUDE.md, Runtime limits).
MAX_HOLD_SECONDS = 60.0


def hold_home(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 2.0) -> HoldResult:
    """Reset to ``home`` and hold it with gravity compensation for ``seconds`` of sim time."""
    if not (math.isfinite(seconds) and 0 < seconds <= MAX_HOLD_SECONDS):
        raise ValueError(f"seconds must be in (0, {MAX_HOLD_SECONDS}], got {seconds!r}")
    ids = scene_ids(model)
    reset_to_home(model, data)
    ee0 = data.site_xpos[ids.ee_site].copy()
    q0 = data.qpos[ids.arm_qpos].copy()
    steps = max(1, round(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step1(model, data)
        data.ctrl[ids.arm_actuators] = data.qfrc_bias[ids.arm_dof]
        mujoco.mj_step2(model, data)
    # mj_step2 leaves sensors and site poses at the pre-integration state; refresh them.
    mujoco.mj_step1(model, data)
    data.ctrl[ids.arm_actuators] = data.qfrc_bias[ids.arm_dof]
    mujoco.mj_forward(model, data)

    force, torque = read_wrist_ft(model, data, ids)
    g = float(np.linalg.norm(model.opt.gravity))
    return HoldResult(
        seconds=steps * model.opt.timestep,
        ee_drift_m=float(np.linalg.norm(data.site_xpos[ids.ee_site] - ee0)),
        max_joint_drift_rad=float(np.max(np.abs(data.qpos[ids.arm_qpos] - q0))),
        wrist_force=force,
        wrist_torque=torque,
        expected_force_n=float(model.body_subtreemass[ids.hand_body]) * g,
        finger_touch=read_finger_touch(model, data, ids),
    )


def render_png(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    width: int = 960,
    height: int = 720,
    camera: str = OVERVIEW_CAMERA,
) -> Path:
    """Render the current state offscreen from ``camera`` and save it as a PNG."""
    from matplotlib import image as mpimg

    path = prepare_png_path(path)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    mpimg.imsave(path, pixels)
    return path


def prepare_png_path(path: Path) -> Path:
    """Check that ``path`` can take a PNG and create its parent directory.

    Raises ``ValueError`` for a non-``.png`` suffix or an existing directory, and
    ``OSError`` if the parent directory cannot be created.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"PNG path must end in .png, got {str(path)!r}")
    if path.is_dir():
        raise ValueError(f"PNG path is a directory: {str(path)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_parser(prog: str = "python -m vfa.sim.scene_check") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Build the Panda scene, hold home under gravity compensation, render a PNG.",
    )
    parser.add_argument("--png", type=Path, default=Path("runs/scene.png"), help="output PNG")
    parser.add_argument("--seconds", type=float, default=2.0, help="hold duration (sim time)")
    parser.add_argument("--no-render", action="store_true", help="skip the PNG")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser() if prog is None else build_parser(prog)
    args = parser.parse_args(argv)
    if not (math.isfinite(args.seconds) and 0 < args.seconds <= MAX_HOLD_SECONDS):
        parser.error(f"--seconds must be in (0, {MAX_HOLD_SECONDS}], got {args.seconds}")
    if not args.no_render:
        # Fail before the simulation, not after it.
        try:
            prepare_png_path(args.png)
        except (OSError, ValueError) as exc:
            parser.error(f"--png: {exc}")
    model = build_scene()
    data = mujoco.MjData(model)
    result = hold_home(model, data, args.seconds)

    f, t = result.wrist_force, result.wrist_torque
    ratio = result.force_norm / result.expected_force_n
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} dt={model.opt.timestep} s")
    print(f"hold: {result.seconds:.3f} s of gravity compensation at 'home'")
    print(f"  ee drift          {result.ee_drift_m * 1000:.4f} mm")
    print(f"  max joint drift   {np.degrees(result.max_joint_drift_rad):.5f} deg")
    print(f"wrist F [N]   (site frame)  [{f[0]:+.4f} {f[1]:+.4f} {f[2]:+.4f}]")
    print(f"wrist T [N·m] (site frame)  [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]")
    print(
        f"  |F| = {result.force_norm:.4f} N   expected hand+fingers weight "
        f"{result.expected_force_n:.4f} N   ratio {ratio:.4f}"
    )
    left, right = result.finger_touch
    print(f"finger touch [N]  left {left:.4f}  right {right:.4f}")
    if not args.no_render:
        path = render_png(model, data, args.png)
        print(f"png: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
