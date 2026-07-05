import logging
import time
from collections.abc import Callable

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
    MODE_LIFE = "life"
    MODE_SANDFALL = "sandfall"
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
        MODE_LIFE: 15,
        MODE_SANDFALL: 30,
    }

    def __init__(self, mode: str = MODE_DEFAULT) -> None:
        self.last_mode: str | None = None
        self.mode = mode
        self.mode_start_time = time.time()
        self.mode_update_time = time.time()
        self.menu_click_start: float | None = None
        self.pose_enabled = True
        # Notified after every set_pose_enabled, whatever the source (panel
        # menu, web UI, ...); the main loop hooks persistence in here.
        self.on_pose_enabled_changed: Callable[[bool], None] | None = None
        # True while the last hook invocation raised: the live toggle applied
        # but was not persisted. Read by the web API to report the failure;
        # a failed persist is retried even on a same-value set.
        self.pose_persist_failed = False
        # One-shot: the last mode change was a menu-close restore (as opposed
        # to an explicit selection); consumed by the transition policy to
        # tell a resumed gesture chain from a fresh menu launch.
        self._restored_from_menu = False
        self.control_source = self.CONTROL_GESTURE
        self.controller_connected = False
        self._manual_clock_selection = False

    def set_mode(self, mode: str, entered_via: str | None = None) -> None:
        """Switch to ``mode`` and note the source."""
        normalized_source = self.normalize_control_source(entered_via)
        if normalized_source is not None:
            self.control_source = normalized_source

        if mode != self.mode:
            previous_mode = self.mode
            self.last_mode = self.mode
            self.mode_start_time = time.time()
            self._restored_from_menu = False
            logger.info("Mode changed from %s to %s", previous_mode, mode)

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
                self._leave_menu(entered_via)
            self.menu_click_start = None

    def toggle_menu(self, entered_via: str | None = None) -> None:
        """Open the menu, or return to the previous mode if it is already open."""
        if self.mode != self.MODE_MENU:
            self.set_mode(self.MODE_MENU, entered_via=entered_via)
            return

        self._leave_menu(entered_via)
        self.menu_click_start = None

    def _leave_menu(self, entered_via: str | None) -> None:
        """Close the menu, restoring the previous mode and marking it as a restore."""
        # Order matters: set_mode clears the restore flag on every change,
        # so it must be raised after the restore switch, not before.
        self.set_mode(self._previous_non_menu_mode(), entered_via=entered_via)
        self._restored_from_menu = True

    def consume_menu_restore(self) -> bool:
        """Return whether the last mode change was a menu-close restore, clearing the flag.

        One-shot like :meth:`consume_manual_clock_selection`, so the signal
        cannot outlive the policy frame that observes it.
        """
        restored = self._restored_from_menu
        self._restored_from_menu = False
        return restored

    def retarget_menu_restore(self, mode: str) -> None:
        """Point the menu's close-restore target at ``mode``.

        Used by the transition policy when the mode the menu would restore
        stopped being valid while the menu was open (e.g. a chain-entered
        mode whose chain was ended by the POSE toggle).
        """
        if self.mode == self.MODE_MENU:
            self.last_mode = mode

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

    def set_pose_enabled(self, enabled: bool) -> None:
        """Set whether the person-driven auto chain (sandfall/caricature) may run.

        No-op sets (same value) do not fire the change hook, so redundant
        web/MCP requests never rewrite the settings file — unless the last
        persist attempt failed, in which case a same-value set retries it.
        """
        enabled = bool(enabled)
        if enabled == self.pose_enabled and not self.pose_persist_failed:
            return
        if enabled != self.pose_enabled:
            self.pose_enabled = enabled
            logger.info("Pose mode enabled=%s", self.pose_enabled)
        self._notify_pose_enabled_changed()

    def _notify_pose_enabled_changed(self) -> None:
        """Fire the persistence hook, recording failure instead of raising.

        The hook runs on whatever thread toggled (render loop, web, MCP), so
        a persistence error must not crash the caller; ``pose_persist_failed``
        lets the web API surface it instead of silently returning success.
        """
        if self.on_pose_enabled_changed is None:
            return
        try:
            self.on_pose_enabled_changed(self.pose_enabled)
        except Exception:
            logger.exception("pose_enabled change hook failed")
            self.pose_persist_failed = True
        else:
            self.pose_persist_failed = False

    def toggle_pose_enabled(self) -> None:
        """Flip whether the person-driven auto chain (sandfall/caricature) may run."""
        self.set_pose_enabled(not self.pose_enabled)

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
