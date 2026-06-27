from __future__ import annotations

from types import SimpleNamespace

from app.core.action_dispatch import dispatch_actions
from app.core.mode_manager import ModeManager


class DummyModeManager:
    def __init__(self, mode):
        self.mode = mode
        self.toggle_count = 0
        self.toggle_sources = []

    def toggle_menu(self, entered_via=None):
        self.toggle_count += 1
        self.toggle_sources.append(entered_via)


class DummyPaint:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class DummyAutoDrum:
    def __init__(self):
        self.next_song_count = 0

    def next_song(self):
        self.next_song_count += 1


class DummyBoard:
    def __init__(self):
        self.clear_count = 0
        self.undo_count = 0

    def clear(self):
        self.clear_count += 1

    def undo(self):
        self.undo_count += 1


class DummyFontPreview:
    def __init__(self):
        self.prev_count = 0
        self.next_count = 0

    def previous_variant(self):
        self.prev_count += 1

    def next_variant(self):
        self.next_count += 1


def _action(name: str, source: str = "controller"):
    return SimpleNamespace(action=name, source=source)


def test_toggle_menu_applies_in_any_mode():
    manager = DummyModeManager(ModeManager.MODE_CLOCK)
    dispatch_actions(
        actions=[_action("toggle_menu")],
        mode_manager=manager,
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
    )
    assert manager.toggle_count == 1
    assert manager.toggle_sources == [ModeManager.CONTROL_CONTROLLER]


def test_mode_gated_actions_only_apply_in_matching_mode():
    manager = DummyModeManager(ModeManager.MODE_CLOCK)
    paint = DummyPaint()
    autodrum = DummyAutoDrum()
    board = DummyBoard()
    font_preview = DummyFontPreview()

    dispatch_actions(
        actions=[
            _action("paint_clear"),
            _action("autodrum_next_song"),
            _action("board_clear"),
            _action("board_undo"),
            _action("font_preview_prev"),
            _action("font_preview_next"),
        ],
        mode_manager=manager,
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )

    assert paint.clear_count == 0
    assert autodrum.next_song_count == 0
    assert board.clear_count == 0
    assert board.undo_count == 0
    assert font_preview.prev_count == 0
    assert font_preview.next_count == 0


def test_controller_style_actions_use_same_dispatch_path():
    manager = DummyModeManager(ModeManager.MODE_TETRIS)
    paint = DummyPaint()
    autodrum = DummyAutoDrum()
    board = DummyBoard()
    font_preview = DummyFontPreview()

    dispatch_actions(
        actions=[
            _action("toggle_menu", source="controller"),
            _action("paint_clear", source="controller"),
        ],
        mode_manager=manager,
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )

    assert manager.toggle_count == 1
    assert paint.clear_count == 0


def test_matching_mode_actions_are_executed():
    paint = DummyPaint()
    autodrum = DummyAutoDrum()
    board = DummyBoard()
    font_preview = DummyFontPreview()

    dispatch_actions(
        actions=[_action("paint_clear")],
        mode_manager=DummyModeManager(ModeManager.MODE_PAINT),
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )
    dispatch_actions(
        actions=[_action("autodrum_next_song")],
        mode_manager=DummyModeManager(ModeManager.MODE_AUTODRUM),
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )
    dispatch_actions(
        actions=[_action("board_clear"), _action("board_undo")],
        mode_manager=DummyModeManager(ModeManager.MODE_BOARD),
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )
    dispatch_actions(
        actions=[_action("font_preview_prev"), _action("font_preview_next")],
        mode_manager=DummyModeManager(ModeManager.MODE_FONT_PREVIEW),
        paint=paint,
        autodrum=autodrum,
        board=board,
        font_preview=font_preview,
    )

    assert paint.clear_count == 1
    assert autodrum.next_song_count == 1
    assert board.clear_count == 1
    assert board.undo_count == 1
    assert font_preview.prev_count == 1
    assert font_preview.next_count == 1


def test_actions_can_be_filtered_by_source():
    manager = DummyModeManager(ModeManager.MODE_CLOCK)
    dispatch_actions(
        actions=[
            _action("toggle_menu", source="controller"),
            _action("toggle_menu", source="pose"),
        ],
        mode_manager=manager,
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
        allowed_sources={"pose"},
    )

    assert manager.toggle_count == 1
    assert manager.toggle_sources == [ModeManager.CONTROL_GESTURE]
