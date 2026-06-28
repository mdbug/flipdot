from __future__ import annotations

from typing import Any

import numpy as np

import app.services.human_pose as human_pose
import app.services.transition as transition
from app.core.mode_manager import ModeManager
from app.modes.contracts import ModeRegistry, RenderContext


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
    img_sleep: np.ndarray,
    clock_resolve_time: float,
    clock_disolve_time: float,
) -> ModeRegistry:
    """Build default renderers for all current modes.

    This keeps legacy mode implementations intact while moving dispatch into
    a data-driven registry.
    """

    registry = ModeRegistry()

    def render_sleep(_: RenderContext) -> np.ndarray:
        return img_sleep.copy()

    def render_pose(context: RenderContext) -> np.ndarray:
        dots = human_pose.display_human_pose(
            context.pose_results,
            context.panel_width,
            context.panel_height,
            context.estimated_distance,
            context.face_mesh_results,
        )
        if context.mode_time < clock_disolve_time:
            clock_dots = clock.get_frame()
            dots = transition.blend(clock_dots, dots, context.mode_time / clock_disolve_time)
        return dots

    def render_clock(context: RenderContext) -> np.ndarray:
        dots = clock.get_frame()
        if context.mode_time < clock_resolve_time:
            dots = transition.resolve(dots, context.mode_time / clock_resolve_time)
        return dots

    registry.register(ModeManager.MODE_SLEEP, render_sleep)
    registry.register(ModeManager.MODE_POSE, render_pose)
    registry.register(
        ModeManager.MODE_MENU, lambda c: menu.get_frame(c.pose_results, input_hub=c.input_hub)
    )
    registry.register(ModeManager.MODE_CLOCK, render_clock)
    registry.register(
        ModeManager.MODE_PAINT, lambda c: paint.get_frame(c.pose_results, input_hub=c.input_hub)
    )
    registry.register(ModeManager.MODE_CARICATURE, lambda c: caricature.get_frame(c.frame))
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

    return registry
