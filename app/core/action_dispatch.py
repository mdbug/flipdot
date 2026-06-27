from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.core.input_source import InputAction
from app.core.mode_manager import ModeManager


def _to_control_source(source: str) -> str | None:
    """Map an input-source label to its control source, or ``None`` if it has none."""
    if source == "controller":
        return ModeManager.CONTROL_CONTROLLER
    if source == "pose":
        return ModeManager.CONTROL_GESTURE
    return None


def dispatch_actions(
    *,
    actions: Iterable[InputAction],
    mode_manager: ModeManager,
    paint: Any,
    autodrum: Any,
    board: Any,
    font_preview: Any,
    allowed_sources: Iterable[str] | None = None,
) -> None:
    """Apply queued input actions to mode objects through one shared path."""
    allowed = set(allowed_sources) if allowed_sources is not None else None
    for action in actions:
        if allowed is not None and action.source not in allowed:
            continue

        if action.action == "toggle_menu":
            entered_via = _to_control_source(action.source)
            try:
                mode_manager.toggle_menu(entered_via=entered_via)
            except TypeError:
                mode_manager.toggle_menu()
        elif action.action == "paint_clear" and mode_manager.mode == ModeManager.MODE_PAINT:
            paint.clear()
        elif (
            action.action == "autodrum_next_song" and mode_manager.mode == ModeManager.MODE_AUTODRUM
        ):
            autodrum.next_song()
        elif action.action == "board_clear" and mode_manager.mode == ModeManager.MODE_BOARD:
            board.clear()
        elif action.action == "board_undo" and mode_manager.mode == ModeManager.MODE_BOARD:
            board.undo()
        elif (
            action.action == "font_preview_prev"
            and mode_manager.mode == ModeManager.MODE_FONT_PREVIEW
        ):
            font_preview.previous_variant()
        elif (
            action.action == "font_preview_next"
            and mode_manager.mode == ModeManager.MODE_FONT_PREVIEW
        ):
            font_preview.next_variant()
