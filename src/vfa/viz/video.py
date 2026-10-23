"""An offscreen video of attempts with a force overlay (T17, ``vfa demo --video``).

``FrameGrab`` wraps a ``PandaVFAEnv`` made with ``render_mode="rgb_array"``.
While ``active`` it renders the overview camera every ``every`` steps and writes
the attempt, the sim time, the phase, the summed pad force, and the held load
(observable signals) onto the frame. ``write_gif`` saves the frames with
Pillow; with the ``video`` extra (``imageio[ffmpeg]``) ``write_mp4`` saves an
MP4. Rendering never touches the physics state (``PandaVFAEnv.render``).

RuVector mapping: none; the video shows System 1 (``cognitive_core``'s
``Reactive`` mode) acting on the θ that System 2 chose.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

FPS = 10  # 100 ms per GIF frame, exactly


class FrameGrab(gym.Wrapper):
    """Grab overlay frames while ``active`` (refer to the module doc)."""

    def __init__(self, env: gym.Env, fps: int = FPS, max_frames: int = 1200) -> None:
        super().__init__(env)
        hz = int(env.unwrapped.metadata.get("render_fps", 500))
        self.every = max(1, round(hz / fps))
        self.fps = fps
        self.max_frames = max_frames
        self.active = False
        self.label = ""
        self.frames: list[Image.Image] = []
        self._steps = 0

    def reset(self, **kwargs: Any) -> Any:
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        out = self.env.step(action)
        self._steps += 1
        if self.active and self._steps % self.every == 0 and len(self.frames) < self.max_frames:
            obs, _, _, _, info = out
            self.frames.append(self._frame(obs, info))
        return out

    def _frame(self, obs: Any, info: Any) -> Image.Image:
        image = Image.fromarray(np.asarray(self.env.unwrapped.render()))
        draw = ImageDraw.Draw(image)
        pad = float(np.sum(obs["finger_force"]))
        load = float(np.asarray(obs["load"]).reshape(-1)[0])
        lines = [
            self.label,
            f"t {float(info.get('t', 0.0)):4.1f} s  {info.get('phase', '')}",
            f"pad force {pad:5.1f} N   held load {load:5.2f} N",
        ]
        draw.rectangle((0, 0, image.width, 14 * len(lines) + 6), fill=(255, 255, 255))
        for i, line in enumerate(lines):
            draw.text((6, 3 + 14 * i), line, fill=(17, 24, 39))
        bar = int(min(pad / 50.0, 1.0) * (image.width - 12))  # 50 N = full width
        draw.rectangle((6, image.height - 12, 6 + bar, image.height - 4), fill=(37, 99, 235))
        return image


def write_gif(frames: Sequence[Image.Image], path: str | Path, fps: int = FPS) -> Path:
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    frames[0].save(
        path, save_all=True, append_images=list(frames[1:]), duration=int(1000 / fps), loop=0
    )
    return path


def write_mp4(frames: Sequence[Image.Image], path: str | Path, fps: int = FPS) -> Path:
    """Needs the ``video`` extra (``uv sync --extra video``)."""
    import imageio.v2 as imageio

    path = Path(path)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return path
