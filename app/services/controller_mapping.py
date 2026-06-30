from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.mode_manager import ModeManager

DPAD_KEYS = ("D-Up", "D-Down", "D-Left", "D-Right")


@dataclass
class ControllerBridgeState:
    """Mutable state the bridge carries between frames (edges, repeats, cursors)."""

    prev_pressed: set[str] = field(default_factory=set)
    repeat_deadlines: dict[str, float | None] = field(
        default_factory=lambda: {key: None for key in DPAD_KEYS}
    )
    cursor_x: float = 0.5
    cursor_y: float = 0.5
    ab_hold_start: float | None = None
    ab_fired: bool = False
    pong_target_y_right: float = 0.5
    pong_target_y_left: float = 0.5
    tank_prev_right: set[str] = field(default_factory=set)
    tank_prev_left: set[str] = field(default_factory=set)


class ControllerInputBridge:
    """Translate gamepad button snapshots into mode-specific actions.

    Holds per-frame state (button edges, D-pad auto-repeat, paint/board
    cursor, Pong paddle targets) and, given the active mode, drives the
    matching mode object plus the shared input hub each frame.
    """

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
        primary_snapshot: dict | None = None,
        secondary_snapshot: dict | None = None,
        mode: str,
        input_hub: Any,
        mode_manager: Any,
        menu: Any,
        paint: Any,
        autodrum: Any,
        board: Any,
        font_preview: Any,
        tetris_game: Any,
        pong_game: Any,
        percussion: Any,
        tank_game: Any = None,
        pressed_events: set[str] | None = None,
    ) -> None:
        """Apply one controller snapshot to the active mode for this frame."""
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
            elif (not self._state.ab_fired) and (
                now - self._state.ab_hold_start >= self.AB_MENU_HOLD_SEC
            ):
                input_hub.submit_action(source="controller", action="toggle_menu")
                self._state.ab_fired = True
        else:
            self._state.ab_hold_start = None
            self._state.ab_fired = False

        dpad_trigger = self._compute_dpad_triggers(
            pressed=pressed, just_pressed=just_pressed, now=now
        )

        # Expose controller pointer/button state for pointer-driven modes.
        if mode in (ModeManager.MODE_PAINT, ModeManager.MODE_BOARD):
            self._update_cursor_with_dpad(dpad_trigger)
            input_hub.submit_pointer(
                source="controller", x=self._state.cursor_x, y=self._state.cursor_y
            )
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
            self._handle_menu_mode(
                dpad_trigger=dpad_trigger,
                just_pressed=just_pressed,
                menu=menu,
                mode_manager=mode_manager,
            )
        elif mode == ModeManager.MODE_PAINT:
            if "B" in just_pressed:
                paint.clear()
        elif mode == ModeManager.MODE_BOARD:
            if "A" in just_pressed:
                board.apply_stroke(
                    [
                        {
                            "x": self._state.cursor_x,
                            "y": self._state.cursor_y,
                        }
                    ]
                )
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
            primary_connected = bool(
                primary_snapshot
                and primary_snapshot.get("enabled")
                and primary_snapshot.get("connected")
            )
            secondary_connected = bool(
                secondary_snapshot
                and secondary_snapshot.get("enabled")
                and secondary_snapshot.get("connected")
            )

            primary_pressed = (
                set(primary_snapshot.get("pressed_buttons", []))
                if primary_connected and primary_snapshot is not None
                else set()
            )
            secondary_pressed = (
                set(secondary_snapshot.get("pressed_buttons", []))
                if secondary_connected and secondary_snapshot is not None
                else set()
            )

            if primary_connected and secondary_connected:
                right_pressed = primary_pressed
                left_pressed = secondary_pressed
            elif primary_connected:
                right_pressed = primary_pressed
                left_pressed = set()
            elif secondary_connected:
                right_pressed = secondary_pressed
                left_pressed = set()
            else:
                # Fall back to merged snapshot behavior for compatibility.
                right_pressed = pressed
                left_pressed = set()

            if "D-Up" in right_pressed:
                self._state.pong_target_y_right = max(
                    0.0, self._state.pong_target_y_right - self.CURSOR_STEP_NORM
                )
            if "D-Down" in right_pressed:
                self._state.pong_target_y_right = min(
                    1.0, self._state.pong_target_y_right + self.CURSOR_STEP_NORM
                )
            if ("D-Up" in right_pressed) or ("D-Down" in right_pressed):
                pong_game.set_controller_target(self._state.pong_target_y_right, side="right")

            if "D-Up" in left_pressed:
                self._state.pong_target_y_left = max(
                    0.0, self._state.pong_target_y_left - self.CURSOR_STEP_NORM
                )
            if "D-Down" in left_pressed:
                self._state.pong_target_y_left = min(
                    1.0, self._state.pong_target_y_left + self.CURSOR_STEP_NORM
                )
            if ("D-Up" in left_pressed) or ("D-Down" in left_pressed):
                pong_game.set_controller_target(self._state.pong_target_y_left, side="left")

            if "A" in just_pressed:
                pong_game.restart_if_game_over()
        elif mode == ModeManager.MODE_TANK:
            primary_connected = bool(
                primary_snapshot
                and primary_snapshot.get("enabled")
                and primary_snapshot.get("connected")
            )
            secondary_connected = bool(
                secondary_snapshot
                and secondary_snapshot.get("enabled")
                and secondary_snapshot.get("connected")
            )
            primary_pressed = (
                set(primary_snapshot.get("pressed_buttons", []))
                if primary_connected and primary_snapshot is not None
                else set()
            )
            secondary_pressed = (
                set(secondary_snapshot.get("pressed_buttons", []))
                if secondary_connected and secondary_snapshot is not None
                else set()
            )

            # Primary drives the right tank; secondary (when present) the left.
            # With a single controller, fall back to the merged snapshot so a
            # lone pad still steers the right tank against the AI.
            sides: list[tuple[str, set[str], str]] = []
            if primary_connected:
                sides.append(("right", primary_pressed, "tank_prev_right"))
            if secondary_connected:
                sides.append(("left", secondary_pressed, "tank_prev_left"))
            if not sides and connected:
                sides.append(("right", pressed, "tank_prev_right"))

            driven = {"left": False, "right": False}
            for side, held, prev_attr in sides:
                move_x = (1 if "D-Right" in held else 0) - (1 if "D-Left" in held else 0)
                move_y = (1 if "D-Down" in held else 0) - (1 if "D-Up" in held else 0)
                tank_game.set_controller_input(side, move_x=move_x, move_y=move_y)
                prev = getattr(self._state, prev_attr)
                if "A" in held and "A" not in prev:
                    tank_game.fire(side, now)
                setattr(self._state, prev_attr, set(held))
                driven[side] = True

            # Keep idle sides' edge state fresh so a later first press still fires.
            if not driven["right"]:
                self._state.tank_prev_right = set()
            if not driven["left"]:
                self._state.tank_prev_left = set()

            if "A" in just_pressed:
                tank_game.restart_if_game_over()
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

    def _compute_dpad_triggers(
        self, *, pressed: set[str], just_pressed: set[str], now: float
    ) -> dict[str, bool]:
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
    def _handle_menu_mode(
        *, dpad_trigger: dict[str, bool], just_pressed: set[str], menu, mode_manager
    ) -> None:
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
