from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Frame = np.ndarray
Renderer = Callable[["RenderContext"], Frame]


@dataclass
class RenderContext:
    """Inputs needed by mode renderers to produce a 1-bit frame."""

    frame: np.ndarray
    pose_results: Any
    face_mesh_results: Any
    estimated_distance: float | None
    mode_time: float
    panel_width: int
    panel_height: int
    input_hub: Any = None
    # 0..1 progress of the caricature exit hold (viewer backing away); None
    # outside an auto-caricature exit. Drives the shrink-back animation.
    caricature_exit_progress: float | None = None


class ModeRegistry:
    """Maps a mode key to a renderer callable."""

    def __init__(self) -> None:
        self._renderers: dict[str, Renderer] = {}

    def register(self, mode: str, renderer: Renderer) -> None:
        self._renderers[mode] = renderer

    def render(self, mode: str, context: RenderContext) -> Frame:
        renderer = self._renderers.get(mode)
        if renderer is None:
            return np.zeros((context.panel_height, context.panel_width), dtype=np.uint8)
        return renderer(context)
