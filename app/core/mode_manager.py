import logging
import time

logger = logging.getLogger(__name__)


class ModeManager:
    """Tracks the active display mode and the input source allowed to drive it."""

    CONTROL_GESTURE = "gesture"
    CONTROL_CONTROLLER = "controller"

    MODE_SLEEP = "sleep"
    MODE_CLOCK = "clock"
    MODE_POSE = "pose"
    MODE_MENU = "menu"
    MODE_PAINT = "paint"
    MODE_CARICATURE = "caricature"
    MODE_PERCUSSION = "percussion"
    MODE_AUTODRUM = "autodrum"
    MODE_BEATMIRROR = "beatmirror"
    MODE_TETRIS = "tetris"
    MODE_PONG = "pong"
    MODE_TANK = "tank"
    MODE_WORLDCUP = "worldcup"
    MODE_BOARD = "board"
    MODE_FONT_PREVIEW = "font_preview"
    MODE_SCRIPT = "script"
    MODE_DEFAULT = MODE_CLOCK
    MAX_FPS = {
        MODE_SLEEP: 1,
        MODE_CLOCK: 4,
        MODE_POSE: 30,
        MODE_MENU: 30,
        MODE_PAINT: 30,
        MODE_CARICATURE: 30,
        MODE_PERCUSSION: 30,
        MODE_AUTODRUM: 30,
        MODE_BEATMIRROR: 30,
        MODE_TETRIS: 30,
        MODE_PONG: 30,
        MODE_TANK: 30,
        MODE_WORLDCUP: 30,
        MODE_BOARD: 30,
        MODE_FONT_PREVIEW: 30,
        MODE_SCRIPT: 30,
    }

    def __init__(self, mode: str = MODE_DEFAULT) -> None:
        self.last_mode: str | None = None
        self.mode = mode
        self.mode_start_time = time.time()
        self.mode_update_time = time.time()
        self.menu_click_start: float | None = None
        self.pose_enabled = True
        self.control_source = self.CONTROL_GESTURE
        self.controller_connected = False
        self._manual_clock_selection = False

    def set_mode(self, mode: str, entered_via: str | None = None) -> None:
        """Switch to ``mode`` (falling back to clock if pose is disabled) and note the source."""
        requested_mode = mode
        if mode == self.MODE_POSE and not self.pose_enabled:
            mode = self.MODE_CLOCK

        normalized_source = self.normalize_control_source(entered_via)
        if normalized_source is not None:
            self.control_source = normalized_source

        if mode != self.mode:
            previous_mode = self.mode
            self.last_mode = self.mode
            self.mode_start_time = time.time()
            logger.info(
                "Mode changed from %s to %s (requested=%s)", previous_mode, mode, requested_mode
            )

        self.mode = mode
        self.mode_update_time = time.time()

    def click_menu(self, entered_via: str | None = None) -> None:
        """Toggle the menu on a dwell: open/close once the click is held for 2s."""
        if self.menu_click_start is None:
            self.menu_click_start = time.time()
        elif time.time() - self.menu_click_start > 2:
            if self.mode != self.MODE_MENU:
                self.set_mode(self.MODE_MENU, entered_via=entered_via)
            else:
                self.set_mode(self._previous_non_menu_mode(), entered_via=entered_via)
            self.menu_click_start = None

    def toggle_menu(self, entered_via: str | None = None) -> None:
        """Open the menu, or return to the previous mode if it is already open."""
        if self.mode != self.MODE_MENU:
            self.set_mode(self.MODE_MENU, entered_via=entered_via)
            return

        self.set_mode(self._previous_non_menu_mode(), entered_via=entered_via)
        self.menu_click_start = None

    def _previous_non_menu_mode(self) -> str:
        """Return the mode to restore when leaving the menu (pose if none recorded)."""
        if self.last_mode and self.last_mode != self.MODE_MENU:
            return self.last_mode
        return self.MODE_POSE

    def reset_menu_click(self):
        """Cancel an in-progress menu dwell."""
        self.menu_click_start = None

    def note_manual_clock_selection(self) -> None:
        """Record that the user explicitly picked clock from the menu.

        The transition policy consumes this to suppress the live-World-Cup
        auto-switch until a match that is not already live goes live.
        """
        self._manual_clock_selection = True

    def consume_manual_clock_selection(self) -> bool:
        """Return whether clock was just manually selected, clearing the flag."""
        flag = self._manual_clock_selection
        self._manual_clock_selection = False
        return flag

    def get_mode_time(self) -> float:
        """Return seconds elapsed since the current mode was entered."""
        return time.time() - self.mode_start_time

    def get_time_since_last_mode_update(self) -> float:
        """Return seconds since the last ``set_mode`` call (including same-mode re-sets)."""
        return time.time() - self.mode_update_time

    def get_fps_limit(self) -> int:
        """Return the FPS cap for the active mode (30 for the first 5s, for snappy transitions)."""
        if self.get_mode_time() > 5:
            return self.MAX_FPS.get(self.mode, 1)
        else:
            return 30

    def toggle_pose_enabled(self) -> None:
        """Flip whether pose mode may be entered."""
        self.pose_enabled = not self.pose_enabled
        logger.info("Pose mode enabled=%s", self.pose_enabled)

    @classmethod
    def normalize_control_source(cls, source: str | None) -> str | None:
        """Map a raw source label to a canonical control source, or ``None`` if unrecognized."""
        if source in (cls.CONTROL_GESTURE, "pose"):
            return cls.CONTROL_GESTURE
        if source == cls.CONTROL_CONTROLLER:
            return cls.CONTROL_CONTROLLER
        return None

    def update_controller_connected(self, connected: bool) -> bool:
        """Update controller-connected state; return True if it just became connected."""
        connected = bool(connected)
        was_connected = self.controller_connected
        self.controller_connected = connected
        if connected and not was_connected:
            self.control_source = self.CONTROL_CONTROLLER
            return True
        return False

    def get_effective_control_source(self) -> str:
        """Return the active control source, forcing gesture when no controller is connected."""
        if not self.controller_connected:
            return self.CONTROL_GESTURE
        return self.control_source

    def get_allowed_input_sources(self, *, include_web: bool = True) -> set[str]:
        """Return the input-source labels permitted to drive the display right now."""
        effective = self.get_effective_control_source()
        allowed = {"pose"} if effective == self.CONTROL_GESTURE else {"controller"}
        if include_web:
            allowed.add("web")
        return allowed
