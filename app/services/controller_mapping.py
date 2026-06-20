from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.core.mode_manager import ModeManager


DPAD_KEYS = ("D-Up", "D-Down", "D-Left", "D-Right")


@dataclass
class ControllerBridgeState:
    prev_pressed: set[str] = field(default_factory=set)
    repeat_deadlines: dict[str, float | None] = field(
        default_factory=lambda: {key: None for key in DPAD_KEYS}
    )
    cursor_x: float = 0.5
    cursor_y: float = 0.5
    ab_hold_start: float | None = None
    ab_fired: bool = False
    pong_target_y: float = 0.5


class ControllerInputBridge:
    AB_MENU_HOLD_SEC = 2.0
    DPAD_INITIAL_REPEAT_SEC = 0.20
    DPAD_REPEAT_SEC = 0.09
    CURSOR_STEP_NORM = 0.05
    BPM_STEP = 8.0

    def __init__(self) -> None:
        self._state = ControllerBridgeState()

    def process(
        self,
        *,
        snapshot: dict,
        mode: str,
        input_hub,
        mode_manager,
        menu,
        paint,
        autodrum,
        board,
        font_preview,
        tetris_game,
        pong_game,
        percussion,
        pressed_events: set[str] | None = None,
    ) -> None:
        now = time.time()
        connected = bool(snapshot.get("enabled")) and bool(snapshot.get("connected"))
        pressed = set(snapshot.get("pressed_buttons", [])) if connected else set()
        get_effective_source = getattr(mode_manager, "get_effective_control_source", None)
        effective_source = (
            get_effective_source()
            if callable(get_effective_source)
            else ModeManager.CONTROL_CONTROLLER
        )
        controller_allowed = effective_source == ModeManager.CONTROL_CONTROLLER

        if not connected or not controller_allowed:
            input_hub.clear_pointer("controller")
            input_hub.set_button_down(source="controller", is_down=False)
            self._state.ab_hold_start = None
            self._state.ab_fired = False
            for key in DPAD_KEYS:
                self._state.repeat_deadlines[key] = None
            self._state.prev_pressed = pressed
            return

        # Prefer the input thread's latched down-edges when available so quick
        # taps are never lost to render-loop sampling. Fall back to diffing the
        # level state when no edge set is supplied (e.g. in unit tests).
        if pressed_events is None:
            just_pressed = pressed - self._state.prev_pressed
        else:
            just_pressed = set(pressed_events)
        released = self._state.prev_pressed - pressed

        for key in released:
            if key in self._state.repeat_deadlines:
                self._state.repeat_deadlines[key] = None

        # A+B hold chord toggles the menu globally, but should not preempt
        # direct menu navigation/selection while already in menu mode.
        chord_active = mode != ModeManager.MODE_MENU and "A" in pressed and "B" in pressed
        if chord_active:
            if self._state.ab_hold_start is None:
                self._state.ab_hold_start = now
            elif (not self._state.ab_fired) and (now - self._state.ab_hold_start >= self.AB_MENU_HOLD_SEC):
                input_hub.submit_action(source="controller", action="toggle_menu")
                self._state.ab_fired = True
        else:
            self._state.ab_hold_start = None
            self._state.ab_fired = False

        dpad_trigger = self._compute_dpad_triggers(pressed=pressed, just_pressed=just_pressed, now=now)

        # Expose controller pointer/button state for pointer-driven modes.
        if mode in (ModeManager.MODE_PAINT, ModeManager.MODE_BOARD):
            self._update_cursor_with_dpad(dpad_trigger)
            input_hub.submit_pointer(source="controller", x=self._state.cursor_x, y=self._state.cursor_y)
        else:
            input_hub.clear_pointer("controller")

        input_hub.set_button_down(
            source="controller",
            is_down=(mode == ModeManager.MODE_PAINT and "A" in pressed and not chord_active),
        )

        if chord_active:
            self._state.prev_pressed = pressed
            return

        if mode == ModeManager.MODE_MENU:
            self._handle_menu_mode(dpad_trigger=dpad_trigger, just_pressed=just_pressed, menu=menu, mode_manager=mode_manager)
        elif mode == ModeManager.MODE_PAINT:
            if "B" in just_pressed:
                paint.clear()
        elif mode == ModeManager.MODE_BOARD:
            if "A" in just_pressed:
                board.apply_stroke([
                    {
                        "x": self._state.cursor_x,
                        "y": self._state.cursor_y,
                    }
                ])
            if "B" in just_pressed:
                board.undo()
        elif mode == ModeManager.MODE_FONT_PREVIEW:
            if dpad_trigger["D-Left"]:
                font_preview.previous_variant()
            if dpad_trigger["D-Right"]:
                font_preview.next_variant()
            if dpad_trigger["D-Up"]:
                font_preview.adjust_spacing(1)
            if dpad_trigger["D-Down"]:
                font_preview.adjust_spacing(-1)
            if "A" in just_pressed:
                font_preview.next_variant()
            if "B" in just_pressed:
                font_preview.previous_variant()
        elif mode == ModeManager.MODE_TETRIS:
            if dpad_trigger["D-Left"]:
                tetris_game.queue_controller_move(-1)
            if dpad_trigger["D-Right"]:
                tetris_game.queue_controller_move(1)
            if dpad_trigger["D-Down"]:
                tetris_game.queue_controller_hard_drop()
            if "A" in just_pressed:
                tetris_game.queue_controller_rotate_ccw()
            if "B" in just_pressed:
                tetris_game.queue_controller_rotate_cw()
        elif mode == ModeManager.MODE_PONG:
            if dpad_trigger["D-Up"]:
                self._state.pong_target_y = max(0.0, self._state.pong_target_y - self.CURSOR_STEP_NORM)
            if dpad_trigger["D-Down"]:
                self._state.pong_target_y = min(1.0, self._state.pong_target_y + self.CURSOR_STEP_NORM)
            if dpad_trigger["D-Up"] or dpad_trigger["D-Down"]:
                pong_game.set_controller_target(self._state.pong_target_y)
            if "A" in just_pressed:
                pong_game.restart_if_game_over()
        elif mode == ModeManager.MODE_AUTODRUM:
            if "A" in just_pressed or dpad_trigger["D-Right"] or dpad_trigger["D-Left"]:
                autodrum.next_song()
        elif mode == ModeManager.MODE_PERCUSSION:
            if dpad_trigger["D-Left"]:
                percussion.adjust_bpm(-self.BPM_STEP)
            if dpad_trigger["D-Right"]:
                percussion.adjust_bpm(self.BPM_STEP)
            if dpad_trigger["D-Up"]:
                percussion.cycle_pattern(1)
            if dpad_trigger["D-Down"]:
                percussion.cycle_pattern(-1)
            if "A" in just_pressed:
                percussion.trigger_accent()

        self._state.prev_pressed = pressed

    def _compute_dpad_triggers(self, *, pressed: set[str], just_pressed: set[str], now: float) -> dict[str, bool]:
        triggered: dict[str, bool] = {key: False for key in DPAD_KEYS}
        for key in DPAD_KEYS:
            if key in just_pressed:
                triggered[key] = True
                self._state.repeat_deadlines[key] = now + self.DPAD_INITIAL_REPEAT_SEC
                continue

            if key not in pressed:
                continue

            deadline = self._state.repeat_deadlines.get(key)
            if deadline is not None and now >= deadline:
                triggered[key] = True
                self._state.repeat_deadlines[key] = now + self.DPAD_REPEAT_SEC

        return triggered

    def _update_cursor_with_dpad(self, dpad_trigger: dict[str, bool]) -> None:
        if dpad_trigger["D-Left"]:
            self._state.cursor_x = max(0.0, self._state.cursor_x - self.CURSOR_STEP_NORM)
        if dpad_trigger["D-Right"]:
            self._state.cursor_x = min(1.0, self._state.cursor_x + self.CURSOR_STEP_NORM)
        if dpad_trigger["D-Up"]:
            self._state.cursor_y = max(0.0, self._state.cursor_y - self.CURSOR_STEP_NORM)
        if dpad_trigger["D-Down"]:
            self._state.cursor_y = min(1.0, self._state.cursor_y + self.CURSOR_STEP_NORM)

    @staticmethod
    def _handle_menu_mode(*, dpad_trigger: dict[str, bool], just_pressed: set[str], menu, mode_manager) -> None:
        has_controller_navigation = (
            dpad_trigger["D-Up"]
            or dpad_trigger["D-Down"]
            or dpad_trigger["D-Left"]
            or dpad_trigger["D-Right"]
            or ("A" in just_pressed)
            or ("B" in just_pressed)
        )
        if has_controller_navigation:
            menu.mark_controller_navigation_active()

        if dpad_trigger["D-Up"]:
            menu.select_prev_item()
        if dpad_trigger["D-Down"]:
            menu.select_next_item()
        if dpad_trigger["D-Left"]:
            menu.set_page_prev()
        if dpad_trigger["D-Right"]:
            menu.set_page_next()
        if "A" in just_pressed:
            menu.activate_selected()
        if "B" in just_pressed:
            try:
                mode_manager.toggle_menu(entered_via=ModeManager.CONTROL_CONTROLLER)
            except TypeError:
                mode_manager.toggle_menu()
