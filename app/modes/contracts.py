from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

Frame = np.ndarray
Renderer = Callable[["RenderContext"], Frame]


@dataclass
class RenderContext:
    """Inputs needed by mode renderers to produce a 1-bit frame."""

    frame: np.ndarray
    pose_results: Any
    face_mesh_results: Any
    estimated_distance: Optional[float]
    mode_time: float
    panel_width: int
    panel_height: int


class ModeRegistry:
    """Maps a mode key to a renderer callable."""

    def __init__(self) -> None:
        self._renderers: Dict[str, Renderer] = {}

    def register(self, mode: str, renderer: Renderer) -> None:
        self._renderers[mode] = renderer

    def render(self, mode: str, context: RenderContext) -> Frame:
        renderer = self._renderers.get(mode)
        if renderer is None:
            return np.zeros((context.panel_height, context.panel_width), dtype=np.uint8)
        return renderer(context)
