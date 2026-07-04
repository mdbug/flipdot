from __future__ import annotations

from typing import Any

import numpy as np

import app.services.human_pose as human_pose
import app.services.transition as transition
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame, ModeRegistry, RenderContext


class CrossFadingModeRegistry(ModeRegistry):
    """Registry that random-pixel blends from the previous mode on every mode change."""

    def __init__(self, blend_seconds: float) -> None:
        super().__init__()
        self._blend_seconds = blend_seconds
        self._last_mode: str | None = None
        self._blend_source: Frame | None = None
        self._last_output: Frame | None = None

    def render(self, mode: str, context: RenderContext) -> Frame:
        """Render ``mode``, blending from the previously displayed frame after a mode change."""
        dots = super().render(mode, context)
        if mode != self._last_mode:
            self._last_mode = mode
            # Blend from what was actually displayed, so a mode change
            # mid-blend continues seamlessly from the visible frame.
            self._blend_source = (
                self._last_output
                if self._last_output is not None and self._last_output.shape == dots.shape
                else None
            )
        if self._blend_source is not None:
            if 0.0 <= context.mode_time < self._blend_seconds:
                dots = transition.blend(
                    self._blend_source, dots, context.mode_time / self._blend_seconds
                )
            else:
                self._blend_source = None
        # Copy so later in-place mutations (e.g. the DEBUG overlay) never
        # leak into the next transition's blend source.
        self._last_output = dots.copy()
        return dots


def build_mode_registry(
    *,
    clock: Any,
    menu: Any,
    paint: Any,
    caricature: Any,
    percussion: Any,
    autodrum: Any,
    beatmirror: Any,
    tetris_game: Any,
    pong_game: Any,
    tank_game: Any,
    worldcup: Any,
    board: Any,
    font_preview: Any,
    script_mode: Any,
    life: Any,
    sandfall: Any,
    img_sleep: np.ndarray,
    mode_blend_seconds: float,
) -> ModeRegistry:
    """Build default renderers for all current modes.

    This keeps legacy mode implementations intact while moving dispatch into
    a data-driven registry. The returned registry cross-blends every mode
    change over ``mode_blend_seconds``.
    """

    registry = CrossFadingModeRegistry(mode_blend_seconds)

    def render_sleep(_: RenderContext) -> np.ndarray:
        return img_sleep.copy()

    def render_pose(context: RenderContext) -> np.ndarray:
        return human_pose.display_human_pose(
            context.pose_results,
            context.panel_width,
            context.panel_height,
            context.estimated_distance,
            context.face_mesh_results,
        )

    registry.register(ModeManager.MODE_SLEEP, render_sleep)
    registry.register(ModeManager.MODE_POSE, render_pose)
    registry.register(
        ModeManager.MODE_MENU, lambda c: menu.get_frame(c.pose_results, input_hub=c.input_hub)
    )
    registry.register(ModeManager.MODE_CLOCK, lambda c: clock.get_frame())
    registry.register(
        ModeManager.MODE_PAINT, lambda c: paint.get_frame(c.pose_results, input_hub=c.input_hub)
    )
    registry.register(ModeManager.MODE_CARICATURE, lambda c: caricature.get_frame(c))
    registry.register(ModeManager.MODE_PERCUSSION, lambda c: percussion.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_AUTODRUM, lambda c: autodrum.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_BEATMIRROR, lambda c: beatmirror.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_TETRIS, lambda c: tetris_game.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_PONG, lambda c: pong_game.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_TANK, lambda c: tank_game.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_WORLDCUP, lambda c: worldcup.get_frame(c.pose_results))
    registry.register(
        ModeManager.MODE_BOARD, lambda c: board.get_frame(c.pose_results, input_hub=c.input_hub)
    )
    registry.register(
        ModeManager.MODE_FONT_PREVIEW,
        lambda c: font_preview.get_frame(c.pose_results, input_hub=c.input_hub),
    )
    registry.register(ModeManager.MODE_SCRIPT, lambda c: script_mode.get_frame())
    registry.register(ModeManager.MODE_LIFE, lambda c: life.get_frame(c.pose_results))
    registry.register(ModeManager.MODE_SANDFALL, lambda c: sandfall.get_frame(c))

    return registry
