from __future__ import annotations

from app.core.mode_manager import ModeManager
from app.services.controller_mapping import ControllerInputBridge


class DummyInputHub:
    def __init__(self):
        self.actions = []
        self.pointers = []
        self.button_states = {}
        self.cleared_sources = []

    def submit_action(self, *, source, action):
        self.actions.append((source, action))

    def submit_pointer(self, *, source, x, y):
        self.pointers.append((source, x, y))

    def set_button_down(self, *, source, is_down):
        self.button_states[source] = bool(is_down)

    def clear_pointer(self, source):
        self.cleared_sources.append(source)


class DummyModeManager:
    def __init__(self):
        self.toggle_count = 0
        self.effective_source = ModeManager.CONTROL_CONTROLLER

    def toggle_menu(self):
        self.toggle_count += 1

    def get_effective_control_source(self):
        return self.effective_source


class DummyMenu:
    def __init__(self):
        self.prev_count = 0
        self.next_count = 0
        self.page_prev_count = 0
        self.page_next_count = 0
        self.activate_count = 0
        self.controller_mark_count = 0

    def select_prev_item(self):
        self.prev_count += 1

    def select_next_item(self):
        self.next_count += 1

    def set_page_prev(self):
        self.page_prev_count += 1

    def set_page_next(self):
        self.page_next_count += 1

    def activate_selected(self):
        self.activate_count += 1

    def mark_controller_navigation_active(self):
        self.controller_mark_count += 1


class DummyTetris:
    def __init__(self):
        self.moves = []
        self.ccw = 0
        self.cw = 0
        self.hard_drop = 0

    def queue_controller_move(self, dx):
        self.moves.append(dx)

    def queue_controller_rotate_ccw(self):
        self.ccw += 1

    def queue_controller_rotate_cw(self):
        self.cw += 1

    def queue_controller_hard_drop(self):
        self.hard_drop += 1


class DummyPong:
    def __init__(self):
        self.targets = []
        self.restart_called = False

    def set_controller_target(self, norm_y, side="right"):
        self.targets.append((side, norm_y))
        self.last_target = norm_y
        self.last_side = side

    def restart_if_game_over(self):
        self.restart_called = True


class DummyPercussion:
    def adjust_bpm(self, delta):
        self.delta = delta

    def cycle_pattern(self, delta):
        self.pattern_delta = delta

    def trigger_accent(self):
        self.accent = True


class DummyFontPreview:
    def previous_variant(self):
        self.prev = True

    def next_variant(self):
        self.next = True

    def adjust_spacing(self, delta):
        self.spacing_delta = delta


class DummyPaint:
    def clear(self):
        self.cleared = True


class DummyAutoDrum:
    def next_song(self):
        self.next_song_called = True


class DummyBoard:
    def apply_stroke(self, points):
        self.points = points

    def undo(self):
        self.undo_called = True


def _run_bridge(
    bridge,
    monkeypatch,
    now,
    *,
    mode,
    snapshot,
    primary_snapshot=None,
    secondary_snapshot=None,
):
    monkeypatch.setattr("app.services.controller_mapping.time.time", lambda: now)
    input_hub = DummyInputHub()
    mode_manager = DummyModeManager()
    menu = DummyMenu()
    tetris = DummyTetris()
    pong = DummyPong()
    bridge.process(
        snapshot=snapshot,
        primary_snapshot=primary_snapshot,
        secondary_snapshot=secondary_snapshot,
        mode=mode,
        input_hub=input_hub,
        mode_manager=mode_manager,
        menu=menu,
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
        tetris_game=tetris,
        pong_game=pong,
        percussion=DummyPercussion(),
    )
    return input_hub, mode_manager, menu, tetris, pong


def test_pressed_events_fire_even_when_level_state_missing(monkeypatch):
    # A quick tap captured by the input thread arrives as an edge while the
    # snapshot level state already shows the button released. It must still
    # trigger the discrete action exactly once.
    monkeypatch.setattr("app.services.controller_mapping.time.time", lambda: 5.0)
    bridge = ControllerInputBridge()
    tetris = DummyTetris()
    bridge.process(
        snapshot={"enabled": True, "connected": True, "pressed_buttons": []},
        mode=ModeManager.MODE_TETRIS,
        input_hub=DummyInputHub(),
        mode_manager=DummyModeManager(),
        menu=DummyMenu(),
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
        tetris_game=tetris,
        pong_game=DummyPong(),
        percussion=DummyPercussion(),
        pressed_events={"A"},
    )
    assert tetris.ccw == 1


def test_empty_pressed_events_suppress_level_edge(monkeypatch):
    # When the input thread reports no down-edges, a held button that is still
    # in the level state must not re-trigger a discrete action.
    monkeypatch.setattr("app.services.controller_mapping.time.time", lambda: 6.0)
    bridge = ControllerInputBridge()
    tetris = DummyTetris()
    bridge.process(
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["A"]},
        mode=ModeManager.MODE_TETRIS,
        input_hub=DummyInputHub(),
        mode_manager=DummyModeManager(),
        menu=DummyMenu(),
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
        tetris_game=tetris,
        pong_game=DummyPong(),
        percussion=DummyPercussion(),
        pressed_events=set(),
    )
    assert tetris.ccw == 0


def test_ab_hold_toggles_menu_once(monkeypatch):
    bridge = ControllerInputBridge()
    snapshot = {"enabled": True, "connected": True, "pressed_buttons": ["A", "B"]}

    input_hub, _, _, tetris, _ = _run_bridge(
        bridge,
        monkeypatch,
        10.0,
        mode=ModeManager.MODE_TETRIS,
        snapshot=snapshot,
    )
    assert input_hub.actions == []
    assert tetris.ccw == 0
    assert tetris.cw == 0

    input_hub, _, _, tetris, _ = _run_bridge(
        bridge,
        monkeypatch,
        12.1,
        mode=ModeManager.MODE_TETRIS,
        snapshot=snapshot,
    )
    assert input_hub.actions == [("controller", "toggle_menu")]
    assert tetris.ccw == 0
    assert tetris.cw == 0

    input_hub, _, _, _, _ = _run_bridge(
        bridge,
        monkeypatch,
        12.5,
        mode=ModeManager.MODE_TETRIS,
        snapshot=snapshot,
    )
    assert input_hub.actions == []


def test_tetris_mapping_uses_ccw_cw_and_hard_drop(monkeypatch):
    bridge = ControllerInputBridge()

    _, _, _, tetris, _ = _run_bridge(
        bridge,
        monkeypatch,
        5.0,
        mode=ModeManager.MODE_TETRIS,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["A"]},
    )
    assert tetris.ccw == 1
    assert tetris.cw == 0
    assert tetris.hard_drop == 0

    _, _, _, tetris, _ = _run_bridge(
        bridge,
        monkeypatch,
        5.2,
        mode=ModeManager.MODE_TETRIS,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["B"]},
    )
    assert tetris.ccw == 0
    assert tetris.cw == 1
    assert tetris.hard_drop == 0

    _, _, _, tetris, _ = _run_bridge(
        bridge,
        monkeypatch,
        5.4,
        mode=ModeManager.MODE_TETRIS,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Down"]},
    )
    assert tetris.hard_drop == 1
    assert tetris.ccw == 0
    assert tetris.cw == 0


def test_paint_mode_sets_controller_button_and_pointer(monkeypatch):
    bridge = ControllerInputBridge()
    input_hub, _, _, _, _ = _run_bridge(
        bridge,
        monkeypatch,
        20.0,
        mode=ModeManager.MODE_PAINT,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["A", "D-Right"]},
    )

    assert input_hub.button_states.get("controller") is True
    assert input_hub.pointers
    source, x, y = input_hub.pointers[-1]
    assert source == "controller"
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0


def test_menu_buttons_are_not_preempted_by_ab_hold(monkeypatch):
    bridge = ControllerInputBridge()

    # Simulate B already being held from a prior frame.
    _run_bridge(
        bridge,
        monkeypatch,
        30.0,
        mode=ModeManager.MODE_MENU,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["B"]},
    )

    # Press A while B is still held; selection should trigger immediately
    # in menu mode (no 2s A+B chord gating while already in menu).
    _, mode_manager, menu, _, _ = _run_bridge(
        bridge,
        monkeypatch,
        30.1,
        mode=ModeManager.MODE_MENU,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["A", "B"]},
    )

    assert menu.activate_count == 1
    assert menu.controller_mark_count == 1
    assert mode_manager.toggle_count == 0


def test_controller_input_is_ignored_when_gesture_controls_mode(monkeypatch):
    bridge = ControllerInputBridge()
    monkeypatch.setattr("app.services.controller_mapping.time.time", lambda: 15.0)

    input_hub = DummyInputHub()
    mode_manager = DummyModeManager()
    mode_manager.effective_source = ModeManager.CONTROL_GESTURE
    menu = DummyMenu()

    bridge.process(
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Down", "A"]},
        mode=ModeManager.MODE_MENU,
        input_hub=input_hub,
        mode_manager=mode_manager,
        menu=menu,
        paint=DummyPaint(),
        autodrum=DummyAutoDrum(),
        board=DummyBoard(),
        font_preview=DummyFontPreview(),
        tetris_game=DummyTetris(),
        pong_game=DummyPong(),
        percussion=DummyPercussion(),
    )

    assert menu.activate_count == 0
    assert menu.next_count == 0
    assert input_hub.button_states.get("controller") is False


def test_pong_secondary_only_controls_right_paddle(monkeypatch):
    bridge = ControllerInputBridge()

    _, _, _, _, pong = _run_bridge(
        bridge,
        monkeypatch,
        40.0,
        mode=ModeManager.MODE_PONG,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Down"]},
        primary_snapshot={"enabled": True, "connected": False, "pressed_buttons": []},
        secondary_snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Down"]},
    )

    assert pong.targets
    assert pong.targets[-1][0] == "right"


def test_pong_two_controllers_are_two_player(monkeypatch):
    bridge = ControllerInputBridge()

    _, _, _, _, pong = _run_bridge(
        bridge,
        monkeypatch,
        41.0,
        mode=ModeManager.MODE_PONG,
        snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Up", "D-Down"]},
        primary_snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Up"]},
        secondary_snapshot={"enabled": True, "connected": True, "pressed_buttons": ["D-Down"]},
    )

    sides = {side for side, _ in pong.targets}
    assert "right" in sides
    assert "left" in sides
