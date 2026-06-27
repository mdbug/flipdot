import time
from collections.abc import Callable
from typing import Any

import numpy as np

import app.services.human_pose as human_pose
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame
from app.services.text import write

try:
    from mediapipe.python.solutions.pose import PoseLandmark
except ModuleNotFoundError:
    import mediapipe as mp

    PoseLandmark = mp.solutions.pose.PoseLandmark


class MenuItem:
    """A dwell-activated menu row: hover the pointer over it for ``CLICK_TIME`` to trigger."""

    CLICK_TIME = 2
    ROW_HEIGHT = 8
    TEXT_TOP_OFFSET = 1  # row 0 is a spacer so every item has the same rhythm

    def __init__(
        self,
        label: str,
        row: int,
        width: int,
        on_click: Callable[..., Any] | None = None,
    ) -> None:
        self.label = label
        self.row = row
        self.hovered = False
        self.width = width
        self.hover_start_time: float | None = None
        self.on_click = on_click
        self.page = 0

    @property
    def y(self) -> int:
        """Top pixel row of this item's text."""
        return self.row * MenuItem.ROW_HEIGHT + MenuItem.TEXT_TOP_OFFSET

    def is_hovered(self, y: int) -> bool:
        """Return whether panel row ``y`` falls within this item's interactive band."""
        # Hover includes spacer+item+spacer (7px); divider line remains non-interactive.
        hover_top = max(0, self.y - 1)
        hover_bottom = self.y + 6
        return hover_top <= y < hover_bottom

    def hover(self, hovering: bool, source: str | None = None) -> None:
        """Update hover state and fire ``on_click`` once the dwell reaches ``CLICK_TIME``."""
        if hovering and not self.hovered:
            self.hover_start_time = time.time()
        elif not hovering:
            self.hover_start_time = None
        else:
            hover_duration = self.get_hover_duration()
            if hover_duration >= MenuItem.CLICK_TIME and self.on_click:
                self.click(source)
                self.hover_start_time = None

        self.hovered = hovering

    def click(self, source: str | None = None) -> None:
        """Invoke ``on_click``, passing ``source`` when the callback accepts it."""
        if self.on_click:
            try:
                self.on_click(source)
            except TypeError:
                self.on_click()

    def get_hover_duration(self) -> float:
        """Return seconds the pointer has dwelled on this item (0 if not hovering)."""
        if self.hover_start_time:
            return time.time() - self.hover_start_time
        return 0

    def draw(self, frame: Frame) -> None:
        """Render this item onto ``frame``; implemented by concrete item types."""
        raise NotImplementedError

    def draw_hover(self, frame: Frame) -> None:
        """XOR a dwell-progress bar across the item proportional to hover time."""
        if self.hovered:
            duration = self.get_hover_duration()
            slice = min(int(self.width * duration / MenuItem.CLICK_TIME), 28)
            hover_top = max(0, self.y - 1)
            hover_bottom = min(frame.shape[0], self.y + 6)
            frame[hover_top:hover_bottom, 0:slice] = frame[hover_top:hover_bottom, 0:slice] ^ 1


class Button(MenuItem):
    """A menu item that runs its callback when activated."""

    def __init__(
        self,
        label: str,
        row: int,
        width: int,
        on_click: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(label, row, width, on_click)

    def draw(self, frame: Frame) -> None:
        """Render the button label and its divider, plus any hover progress."""
        write(frame, self.label, x=1, y=self.y, size=5, style="regular")
        sep = min(self.y + 6, frame.shape[0] - 1)
        frame[sep, 0 : self.width] = 1
        self.draw_hover(frame)


class Checkbox(MenuItem):
    """A menu item that toggles a boolean and shows a check box."""

    def __init__(
        self,
        label: str,
        row: int,
        width: int,
        checked: bool = False,
        on_click: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(label, row, width, on_click=on_click)
        self.checked = checked

    def draw(self, frame: Frame) -> None:
        """Render the label, divider, check box (blinking while hovered), and hover progress."""
        write(frame, self.label, x=7, y=self.y, size=5, style="regular")
        sep = min(self.y + 6, frame.shape[0] - 1)
        frame[sep, 0 : self.width] = 1
        self.draw_hover(frame)
        frame[self.y : self.y + 6, 0:6] = 0
        frame[self.y : self.y + 5, 0:5] = 1
        frame[self.y + 1 : self.y + 4, 1:4] = 0
        if self.checked:
            frame[self.y + 2, 2] = 1

        # blink while hovered
        if self.hovered and self.hover_start_time:
            frame[self.y + 2, 2] = int(time.time() * 2) % 2

    def hover(self, hovering: bool, source: str | None = None) -> None:
        super().hover(hovering, source=source)

    def click(self) -> None:  # type: ignore[override]
        """Toggle the checked state and notify ``on_click``."""
        self.checked = not self.checked
        if self.on_click:
            self.on_click()


class Menu:
    """Paged, dwell- and swipe-driven mode picker; pointer comes from pose or controller."""

    SWIPE_MIN_DX = 8
    SWIPE_MAX_DY = 3
    SWIPE_MAX_DT = 0.30
    SWIPE_MIN_SPEED_PX_S = 22.0
    SWIPE_MIN_HORIZONTAL_RATIO = 2.5
    SWIPE_MIN_FRAME_DX = 2
    FAST_SWIPE_MIN_DX = 7
    FAST_SWIPE_MIN_SPEED_PX_S = 32.0
    FAST_SWIPE_MAX_DY = 6
    SWIPE_COOLDOWN = 0.35
    SWIPE_OPPOSITE_GUARD_DT = 0.70
    SWIPE_RELEASE_MISSING_FRAMES = 2
    SWIPE_RELEASE_STILL_FRAMES = 3
    SWIPE_RELEASE_STILL_DX = 1
    SWIPE_RELEASE_STILL_DY = 1
    INDICATOR_HOVER_HYSTERESIS_PX = 2
    CONTROLLER_SUPPRESS_SEC = 0.7

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.page = 0
        self._swipe_origin: tuple[int, int, float] | None = None
        self._finger_sample: tuple[int, int, float] | None = None
        self._swipe_locked = False
        self._last_swipe_time = 0.0
        self._last_swipe_direction = 0
        self._missing_finger_frames = 0
        self._still_frames = 0
        self._indicator_hover_page: int | None = None
        self._indicator_hover_start: float | None = None
        self._selected_row = 0
        self._controller_navigation_until = 0.0
        self.pages: list[list[MenuItem]] = [
            [
                Button("CLOCK", 0, width, on_click=self._make_mode_setter(ModeManager.MODE_CLOCK)),
                Button(
                    "WRLDCUP", 1, width, on_click=self._make_mode_setter(ModeManager.MODE_WORLDCUP)
                ),
                Checkbox(
                    "POSE",
                    2,
                    width,
                    checked=mode_manager.pose_enabled,
                    on_click=mode_manager.toggle_pose_enabled,
                ),
            ],
            [
                Button(
                    "DRUM", 0, width, on_click=self._make_mode_setter(ModeManager.MODE_PERCUSSION)
                ),
                Button(
                    "BEATS", 1, width, on_click=self._make_mode_setter(ModeManager.MODE_AUTODRUM)
                ),
                Button(
                    "MIRROR", 2, width, on_click=self._make_mode_setter(ModeManager.MODE_BEATMIRROR)
                ),
            ],
            [
                Button(
                    "TETRIS", 0, width, on_click=self._make_mode_setter(ModeManager.MODE_TETRIS)
                ),
                Button("PONG", 1, width, on_click=self._make_mode_setter(ModeManager.MODE_PONG)),
                Button("PAINT", 2, width, on_click=self._make_mode_setter(ModeManager.MODE_PAINT)),
            ],
            [
                Button("BOARD", 0, width, on_click=self._make_mode_setter(ModeManager.MODE_BOARD)),
                Button("PAINT", 1, width, on_click=self._make_mode_setter(ModeManager.MODE_PAINT)),
                Button("CLOCK", 2, width, on_click=self._make_mode_setter(ModeManager.MODE_CLOCK)),
            ],
            [
                Button(
                    "FONTS",
                    0,
                    width,
                    on_click=self._make_mode_setter(ModeManager.MODE_FONT_PREVIEW),
                ),
                Button("CLOCK", 1, width, on_click=self._make_mode_setter(ModeManager.MODE_CLOCK)),
                Button("MENU", 2, width, on_click=self._make_mode_setter(ModeManager.MODE_MENU)),
            ],
        ]

    def _source_to_entered_via(self, source: str | None) -> str | None:
        if source == "controller":
            return ModeManager.CONTROL_CONTROLLER
        if source == "pose":
            return ModeManager.CONTROL_GESTURE
        return None

    def _make_mode_setter(self, mode: str) -> Callable[..., None]:
        """Return a click handler that switches ``mode_manager`` to ``mode``."""

        def _set_mode(source: str | None = None) -> None:
            entered_via = self._source_to_entered_via(source)
            try:
                self.mode_manager.set_mode(mode, entered_via=entered_via)
            except TypeError:
                self.mode_manager.set_mode(mode)

        return _set_mode

    @property
    def items(self) -> list[MenuItem]:
        """Items on the currently visible page."""
        return self.pages[self.page]

    def _clamp_selected_row(self) -> None:
        if not self.items:
            self._selected_row = 0
            return
        self._selected_row = max(0, min(self._selected_row, len(self.items) - 1))

    def select_next_item(self) -> None:
        """Move the controller selection down one row (wrapping)."""
        if not self.items:
            return
        self._selected_row = (self._selected_row + 1) % len(self.items)

    def select_prev_item(self) -> None:
        """Move the controller selection up one row (wrapping)."""
        if not self.items:
            return
        self._selected_row = (self._selected_row - 1) % len(self.items)

    def activate_selected(self, source: str = "controller") -> None:
        """Click the currently selected item."""
        if not self.items:
            return
        self._clamp_selected_row()
        self.items[self._selected_row].click(source)

    def set_page_next(self) -> None:
        self.next_page()

    def set_page_prev(self) -> None:
        self.prev_page()

    def mark_controller_navigation_active(self, now: float | None = None) -> None:
        """Suppress pose-pointer/swipe handling briefly after controller navigation."""
        if now is None:
            now = time.time()
        self._controller_navigation_until = max(
            self._controller_navigation_until,
            float(now) + self.CONTROLLER_SUPPRESS_SEC,
        )

    def next_page(self) -> None:
        """Advance to the next page (wrapping)."""
        self.page = (self.page + 1) % len(self.pages)
        self._clamp_selected_row()

    def prev_page(self) -> None:
        """Go to the previous page (wrapping)."""
        self.page = (self.page - 1) % len(self.pages)
        self._clamp_selected_row()

    def _reset_swipe_state(self) -> None:
        self._swipe_origin = None
        self._finger_sample = None
        self._swipe_locked = False
        self._missing_finger_frames = 0
        self._still_frames = 0

    def _get_page_indicator_layout(self) -> tuple[int, int]:
        page_count = len(self.pages)
        item_count = len(self.pages[0]) if self.pages else 0
        # After final separator: [spacer][item][spacer][separator] ...
        indicator_y = min(
            self.height,
            max(0, ((item_count - 1) * MenuItem.ROW_HEIGHT) + MenuItem.TEXT_TOP_OFFSET + 7),
        )
        return page_count, indicator_y

    def _get_indicator_page(self, panel_x: int | None, panel_y: int | None) -> int | None:
        if panel_x is None or panel_y is None:
            return None

        page_count, indicator_y = self._get_page_indicator_layout()
        if panel_y < indicator_y or panel_y >= self.height:
            return None

        for idx in range(page_count):
            x0 = (idx * self.width) // page_count
            x1 = ((idx + 1) * self.width) // page_count
            if x0 <= panel_x < x1:
                return idx

        return None

    def _is_in_indicator_page(
        self,
        panel_x: int | None,
        panel_y: int | None,
        page_idx: int | None,
        tolerance: int = 0,
    ) -> bool:
        if panel_x is None or panel_y is None:
            return False

        page_count, indicator_y = self._get_page_indicator_layout()
        if page_idx is None or page_idx < 0 or page_idx >= page_count:
            return False

        if panel_y < indicator_y or panel_y >= self.height:
            return False

        x0 = (page_idx * self.width) // page_count
        x1 = ((page_idx + 1) * self.width) // page_count
        return (x0 - tolerance) <= panel_x < (x1 + tolerance)

    def _update_indicator_hover(
        self,
        panel_x: int | None,
        panel_y: int | None,
        now: float,
        hovered_page: int | None = None,
    ) -> None:
        if hovered_page is None:
            hovered_page = self._get_indicator_page(panel_x, panel_y)

        # Keep dwell on the same page through minor boundary jitter, similar
        # to menu-item hover stability within an area.
        if self._indicator_hover_page is not None and self._is_in_indicator_page(
            panel_x,
            panel_y,
            self._indicator_hover_page,
            tolerance=Menu.INDICATOR_HOVER_HYSTERESIS_PX,
        ):
            hovered_page = self._indicator_hover_page

        if hovered_page is None:
            self._indicator_hover_page = None
            self._indicator_hover_start = None
            return

        if hovered_page != self._indicator_hover_page:
            self._indicator_hover_page = hovered_page
            self._indicator_hover_start = now
            return

        if self._indicator_hover_start is None:
            self._indicator_hover_start = now
            return

        hover_duration = now - self._indicator_hover_start
        if hover_duration < MenuItem.CLICK_TIME:
            return

        if hovered_page != self.page:
            self.page = hovered_page

            # Avoid stale dwell timers from triggering right after page jump.
            for item in self.items:
                item.hover(False)

        # Re-arm indicator dwell so sustained hovering does not repeatedly retrigger.
        self._indicator_hover_start = now

    def _update_swipe(self, panel_x: int | None, panel_y: int | None, now: float) -> None:
        """Track finger motion and flip pages on a qualifying horizontal swipe."""
        if panel_x is None or panel_y is None:
            self._missing_finger_frames += 1
            self._still_frames = 0

            # Unlock only after a brief tracking loss so return motion is not
            # interpreted as an opposite swipe from the same gesture.
            if (
                self._swipe_locked
                and self._missing_finger_frames >= Menu.SWIPE_RELEASE_MISSING_FRAMES
            ):
                self._swipe_locked = False
                self._swipe_origin = None
                self._finger_sample = None
            elif not self._swipe_locked:
                self._swipe_origin = None
                self._finger_sample = None

            return

        self._missing_finger_frames = 0

        if self._finger_sample is None:
            self._swipe_origin = (panel_x, panel_y, now)
            self._finger_sample = (panel_x, panel_y, now)
            return

        previous_x, previous_y, previous_t = self._finger_sample
        frame_dx = panel_x - previous_x
        frame_dy = panel_y - previous_y
        self._finger_sample = (panel_x, panel_y, now)

        if self._swipe_locked:
            if (
                abs(frame_dx) <= Menu.SWIPE_RELEASE_STILL_DX
                and abs(frame_dy) <= Menu.SWIPE_RELEASE_STILL_DY
            ):
                self._still_frames += 1
            else:
                self._still_frames = 0

            if self._still_frames >= Menu.SWIPE_RELEASE_STILL_FRAMES:
                self._swipe_locked = False
                self._still_frames = 0
                self._swipe_origin = (panel_x, panel_y, now)

            return

        if self._swipe_origin is None:
            self._swipe_origin = (panel_x, panel_y, now)
            return

        origin_x, origin_y, origin_t = self._swipe_origin
        total_dx = panel_x - origin_x
        total_dy = panel_y - origin_y
        total_dt = now - origin_t

        if total_dt <= 0:
            self._swipe_origin = (panel_x, panel_y, now)
            return

        if total_dt > Menu.SWIPE_MAX_DT:
            # Sliding window: keep looking for a fresh swipe start.
            self._swipe_origin = (panel_x, panel_y, now)
            return

        if abs(total_dy) > Menu.SWIPE_MAX_DY:
            # Excess vertical drift likely means this was not a horizontal swipe.
            self._swipe_origin = (panel_x, panel_y, now)
            return

        if now - self._last_swipe_time < Menu.SWIPE_COOLDOWN:
            return

        if abs(total_dx) >= Menu.SWIPE_MIN_DX:
            swipe_speed = abs(total_dx) / total_dt
            is_fast_swipe = (
                abs(total_dx) >= Menu.FAST_SWIPE_MIN_DX
                and swipe_speed >= Menu.FAST_SWIPE_MIN_SPEED_PX_S
                and abs(total_dy) <= Menu.FAST_SWIPE_MAX_DY
            )

            if not is_fast_swipe and abs(frame_dx) < Menu.SWIPE_MIN_FRAME_DX:
                self._swipe_origin = (panel_x, panel_y, now)
                return

            if not is_fast_swipe and abs(total_dx) < (
                abs(total_dy) * Menu.SWIPE_MIN_HORIZONTAL_RATIO
            ):
                self._swipe_origin = (panel_x, panel_y, now)
                return

            if not is_fast_swipe and swipe_speed < Menu.SWIPE_MIN_SPEED_PX_S:
                # Keep this point as a fresh origin to avoid accumulating slow drift.
                self._swipe_origin = (panel_x, panel_y, now)
                return

            direction = 1 if total_dx > 0 else -1

            if (
                self._last_swipe_direction != 0
                and direction != self._last_swipe_direction
                and now - self._last_swipe_time < Menu.SWIPE_OPPOSITE_GUARD_DT
            ):
                # Ignore immediate return motion after a successful swipe.
                self._swipe_origin = (panel_x, panel_y, now)
                return

            if direction > 0:
                self.next_page()
            else:
                self.prev_page()

            self._last_swipe_time = now
            self._last_swipe_direction = direction
            self._swipe_locked = True
            self._still_frames = 0
            self._swipe_origin = None

            # Clear hover timers so a swipe does not immediately trigger a dwell click.
            for item in self.items:
                item.hover(False)

    def _draw_page_indicator(self, frame: Frame) -> None:
        """Draw the bottom page-indicator bars, highlighting the active page."""
        page_count, indicator_y = self._get_page_indicator_layout()
        if page_count <= 0:
            return

        center_y = indicator_y + ((self.height - indicator_y) // 2)

        for idx in range(page_count):
            x0 = (idx * self.width) // page_count
            x1 = ((idx + 1) * self.width) // page_count
            if idx == self.page:
                frame[indicator_y : self.height, x0:x1] = 1
            else:
                frame[center_y : center_y + 1, x0:x1] = 1

            if idx == self._indicator_hover_page and self._indicator_hover_start is not None:
                duration = min(
                    (time.time() - self._indicator_hover_start) / MenuItem.CLICK_TIME, 1.0
                )
                progress = int((x1 - x0) * duration)
                if progress > 0:
                    frame[indicator_y : self.height, x0 : x0 + progress] = 1

    def _pointer_to_panel(self, source: str, x: float, y: float) -> tuple[int, int]:
        """Convert a normalized (0..1) pointer to clamped panel pixels (pose is mirrored)."""
        if source == "pose":
            panel_x = int(self.width - (x * self.width))
        else:
            panel_x = int(x * self.width)
        panel_y = int(y * self.height)
        panel_x = max(0, min(self.width - 1, panel_x))
        panel_y = max(0, min(self.height - 1, panel_y))
        return panel_x, panel_y

    def get_frame(self, pose_results: Any, input_hub: Any = None) -> Frame:
        """Render the menu: resolve the pointer, apply hover/click/swipe, draw items + pointer."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        now = time.time()
        controller_navigation_active = now < self._controller_navigation_until
        get_allowed_sources = getattr(self.mode_manager, "get_allowed_input_sources", None)
        allowed_sources = (
            get_allowed_sources(include_web=True)
            if callable(get_allowed_sources)
            else {"pose", "controller", "web"}
        )
        allow_pose_fallback = "pose" in allowed_sources
        pointer_source = "pose"

        panel_x = None
        panel_y = None
        active_pointer = None
        if (input_hub is not None) and (not controller_navigation_active):
            active_pointer = input_hub.get_active_pointer(
                max_age_sec=0.8, allowed_sources=allowed_sources
            )

        if active_pointer is not None:
            pointer_source = active_pointer.source
            panel_x, panel_y = self._pointer_to_panel(
                active_pointer.source, active_pointer.x, active_pointer.y
            )
        elif allow_pose_fallback:
            finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
            if finger_x is not None and finger_y is not None:
                panel_x, panel_y = self._pointer_to_panel("pose", finger_x, finger_y)

        if (input_hub is not None) and (not controller_navigation_active):
            for click in input_hub.pop_clicks(max_age_sec=1.2, allowed_sources=allowed_sources):
                click_x, click_y = self._pointer_to_panel(click.source, click.x, click.y)
                clicked_indicator_page = self._get_indicator_page(click_x, click_y)
                if clicked_indicator_page is not None:
                    if clicked_indicator_page != self.page:
                        self.page = clicked_indicator_page
                        for item in self.items:
                            item.hover(False)
                    continue

                for item in self.items:
                    if item.is_hovered(click_y):
                        item.click(click.source)
                        break

        if controller_navigation_active:
            hovered_indicator_page = None
            in_indicator_area = False
            self._reset_swipe_state()
            self._indicator_hover_page = None
            self._indicator_hover_start = None
            for item in self.items:
                item.hover(False)
        else:
            hovered_indicator_page = self._get_indicator_page(panel_x, panel_y)
            in_indicator_area = hovered_indicator_page is not None
            if hovered_indicator_page is None:
                self._update_swipe(panel_x, panel_y, now)
            else:
                # While using the indicator, suppress swipe detection entirely.
                self._reset_swipe_state()

            self._update_indicator_hover(panel_x, panel_y, now, hovered_page=hovered_indicator_page)

        for item in self.items:
            if in_indicator_area:
                item.hover(False, source=pointer_source)
            elif panel_y is not None:
                item.hover(item.is_hovered(panel_y), source=pointer_source)
            else:
                item.hover(False, source=pointer_source)

            item.draw(frame)

        if self.items:
            self._clamp_selected_row()
            selected = self.items[self._selected_row]
            hover_top = max(0, selected.y - 1)
            hover_bottom = min(frame.shape[0], selected.y + 6)
            frame[hover_top:hover_bottom, 0:1] = 1

        self._draw_page_indicator(frame)
        if controller_navigation_active:
            return frame
        if pointer_source == "pose" and allow_pose_fallback:
            frame = human_pose.draw_right_index_pointer(frame, pose_results)
        else:
            pointer_x = (panel_x / self.width) if panel_x is not None else None
            pointer_y = (panel_y / self.height) if panel_y is not None else None
            frame = human_pose.draw_pointer(frame, pointer_x, pointer_y, mirror_x=False)
        return frame
