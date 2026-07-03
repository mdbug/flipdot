import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import app.services.human_pose as human_pose
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame
from app.services.worldcup import get_worldcup_scorecard


@dataclass
class TransitionState:
    """Per-frame state produced by mode transition rules."""

    face_mesh_results: Any
    eyes_visible: bool
    reason: str
    estimated_distance: float | None
    angle: float | None


class TransitionPolicy:
    """Centralized mode transition logic for the main loop."""

    WORLDCUP_LIVE_CHECK_INTERVAL = 30.0
    HOURLY_SCRIPT_DURATION = 60.0

    def __init__(
        self,
        *,
        pose_timeout: float,
        sleep_start_hour: int,
        sleep_end_hour: int,
        pose_distance_threshold: float = 1.3,
        face_mesh_max_fps: float = 12.0,
    ) -> None:
        self._sleep_lock = threading.Lock()
        self.pose_timeout = pose_timeout
        self.sleep_enabled = True
        self.sleep_start_hour = sleep_start_hour
        self.sleep_end_hour = sleep_end_hour
        self.pose_distance_threshold = pose_distance_threshold
        self.face_mesh_submit_interval = (
            0.0 if face_mesh_max_fps <= 0 else (1.0 / face_mesh_max_fps)
        )
        self._last_face_mesh_submit = 0.0
        self._cached_face_mesh_results = None
        self._last_worldcup_live_check = 0.0
        self._cached_worldcup_live = False
        self._cached_worldcup_live_keys: set[Any] = set()
        self._worldcup_lock = threading.Lock()
        self._worldcup_refresh_in_flight = False
        # Live matches the user has already seen when manually choosing clock;
        # they no longer trigger the auto-switch back to World Cup.
        self._acknowledged_live_keys: set[Any] = set()
        # Hourly random-script interlude on the idle clock.
        self._last_hourly_script_hour: int | None = None
        self._hourly_script_active = False
        self._last_shuffle_date: Any = None

    @staticmethod
    def _worldcup_event_key(event: dict[str, Any]) -> Any:
        """Return a stable identity for a live match for new-match detection."""
        event_id = event.get("event_id")
        if event_id is not None:
            return ("id", str(event_id))
        return ("teams", event.get("home_team"), event.get("away_team"))

    def _refresh_worldcup_live_cache(self) -> None:
        """Refresh World Cup live state off the main render thread."""
        live = None
        live_keys: set[Any] | None = None
        try:
            payload = get_worldcup_scorecard()
            events = payload.get("events") or []
            live_events = [event for event in events if event.get("status_bucket") == "live"]
            live = bool(live_events)
            live_keys = {self._worldcup_event_key(event) for event in live_events}
        except Exception:
            # Keep the previous cached value on transient API/network errors.
            live = None
            live_keys = None

        with self._worldcup_lock:
            if live is not None:
                self._cached_worldcup_live = live
                self._cached_worldcup_live_keys = live_keys or set()
            self._last_worldcup_live_check = time.monotonic()
            self._worldcup_refresh_in_flight = False

    def get_sleep_settings(self) -> dict[str, int | bool]:
        """Return the current sleep-window settings as a JSON-friendly dict."""
        with self._sleep_lock:
            return {
                "enabled": self.sleep_enabled,
                "start_hour": self.sleep_start_hour,
                "end_hour": self.sleep_end_hour,
            }

    def set_sleep_settings(
        self, *, enabled: bool, start_hour: int, end_hour: int
    ) -> dict[str, int | bool]:
        """Update the sleep window (hours clamped to 0..23) and return the new settings."""
        with self._sleep_lock:
            self.sleep_enabled = bool(enabled)
            self.sleep_start_hour = max(0, min(23, int(start_hour)))
            self.sleep_end_hour = max(0, min(23, int(end_hour)))
            return {
                "enabled": self.sleep_enabled,
                "start_hour": self.sleep_start_hour,
                "end_hour": self.sleep_end_hour,
            }

    def _is_worldcup_live(self) -> bool:
        """Return the cached World Cup live flag, kicking off a throttled background refresh."""
        now_mono = time.monotonic()
        should_refresh = False
        with self._worldcup_lock:
            cached_live = self._cached_worldcup_live
            if now_mono - self._last_worldcup_live_check >= self.WORLDCUP_LIVE_CHECK_INTERVAL:
                if not self._worldcup_refresh_in_flight:
                    self._worldcup_refresh_in_flight = True
                    # Throttle refresh attempts even before the worker returns.
                    self._last_worldcup_live_check = now_mono
                    should_refresh = True

        if should_refresh:
            try:
                threading.Thread(target=self._refresh_worldcup_live_cache, daemon=True).start()
            except Exception:
                with self._worldcup_lock:
                    self._worldcup_refresh_in_flight = False

        return cached_live

    def _should_autoswitch_to_worldcup(self, mode_manager: ModeManager) -> bool:
        """Decide whether idle clock mode should hand off to live World Cup.

        Returns True only when a match that is not already acknowledged is live.
        A manual clock selection from the menu acknowledges every currently-live
        match, so the board stays on clock until a *new* match goes live.
        """
        # Kick the throttled refresh; read liveness from the keys it maintains so
        # the live flag and the key set always agree on the same snapshot.
        self._is_worldcup_live()
        with self._worldcup_lock:
            live_keys = set(self._cached_worldcup_live_keys)

        # Drop acknowledgements for matches that have since finished.
        self._acknowledged_live_keys &= live_keys

        if mode_manager.consume_manual_clock_selection():
            # The user explicitly chose clock; treat current live matches as seen.
            self._acknowledged_live_keys = set(live_keys)

        if not live_keys:
            return False

        new_live_keys = live_keys - self._acknowledged_live_keys
        if new_live_keys:
            # Hand off and reset so returning to clock starts a fresh window.
            self._acknowledged_live_keys = set()
            return True
        return False

    def _maybe_reshuffle_scripts(self, now: datetime, script_mode: Any) -> None:
        """Rebuild the day's script order once per day, overnight.

        Prefer to do it while asleep; if sleep is disabled, do it from 5 a.m. on.
        """
        if self._last_shuffle_date == now.date():
            return
        if self.is_sleep_hour(now) or (not self.get_sleep_settings()["enabled"] and now.hour >= 5):
            script_mode.reshuffle_day()
            self._last_shuffle_date = now.date()

    def _should_start_hourly_script(self, now: datetime) -> bool:
        """Return True at most once per clock hour, at the top of the hour."""
        if now.minute != 0:
            return False
        if now.hour == self._last_hourly_script_hour:
            return False
        self._last_hourly_script_hour = now.hour
        return True

    def is_sleep_hour(self, now: datetime | None = None) -> bool:
        """Return whether ``now`` (default: local now) falls inside the sleep window."""
        with self._sleep_lock:
            sleep_enabled = self.sleep_enabled
            sleep_start_hour = self.sleep_start_hour
            sleep_end_hour = self.sleep_end_hour

        if not sleep_enabled:
            return False

        if now is None:
            now = datetime.now()

        # Equal bounds mean an empty sleep window.
        if sleep_start_hour == sleep_end_hour:
            return False

        # Non-wrapping window, e.g. 01 -> 07.
        if sleep_start_hour < sleep_end_hour:
            return sleep_start_hour <= now.hour < sleep_end_hour

        # Wrapping window, e.g. 23 -> 07.
        return now.hour >= sleep_start_hour or now.hour < sleep_end_hour

    def apply(
        self,
        *,
        frame: Frame,
        pose_results: Any,
        mode_manager: ModeManager,
        paint_mode: Any,
        script_mode: Any,
    ) -> TransitionState:
        """Drive ``mode_manager`` from pose/clock/sleep rules and return the frame's pose state."""
        state = TransitionState(
            face_mesh_results=None,
            eyes_visible=False,
            reason="",
            estimated_distance=None,
            angle=None,
        )

        now = datetime.now()
        self._maybe_reshuffle_scripts(now, script_mode)
        if self.is_sleep_hour(now):
            mode_manager.set_mode(ModeManager.MODE_SLEEP)
            return state

        current_mode = mode_manager.mode

        # Sleep mode is policy-driven, so leave it as soon as the window no
        # longer applies (for example after runtime settings changes via Web UI).
        if current_mode == ModeManager.MODE_SLEEP:
            mode_manager.set_mode(ModeManager.MODE_CLOCK)
            current_mode = mode_manager.mode

        # Prioritize live World Cup information when idle on clock mode, unless
        # the user manually selected clock and no new match has gone live since.
        if current_mode == ModeManager.MODE_CLOCK and self._should_autoswitch_to_worldcup(
            mode_manager
        ):
            mode_manager.set_mode(ModeManager.MODE_WORLDCUP)
            return state

        # Clear the interlude flag once we are no longer in the auto-started script
        # (e.g. the user navigated away manually).
        if self._hourly_script_active and current_mode != ModeManager.MODE_SCRIPT:
            self._hourly_script_active = False

        # Once an hour, while idle on the clock, play a short random script as an
        # interlude, then return to the clock (dissolve handled by the registry).
        if current_mode == ModeManager.MODE_CLOCK and self._should_start_hourly_script(now):
            if script_mode.start_next():
                self._hourly_script_active = True
                mode_manager.set_mode(ModeManager.MODE_SCRIPT)
            return state

        if current_mode == ModeManager.MODE_SCRIPT and self._hourly_script_active:
            if mode_manager.get_mode_time() >= self.HOURLY_SCRIPT_DURATION:
                script_mode.stop_script()
                self._hourly_script_active = False
                mode_manager.set_mode(ModeManager.MODE_CLOCK)
            return state

        if current_mode in (
            ModeManager.MODE_MENU,
            ModeManager.MODE_PAINT,
            ModeManager.MODE_PERCUSSION,
            ModeManager.MODE_AUTODRUM,
            ModeManager.MODE_BEATMIRROR,
            ModeManager.MODE_TETRIS,
            ModeManager.MODE_PONG,
            ModeManager.MODE_TANK,
            ModeManager.MODE_WORLDCUP,
            ModeManager.MODE_FONT_PREVIEW,
        ):
            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_CARICATURE:
            # No pose processing needed; caricature handles its own state.
            pass

        elif current_mode == ModeManager.MODE_POSE:
            state.eyes_visible, state.reason, state.angle = (
                human_pose.eyes_visible_and_facing_camera(pose_results)
            )
            state.estimated_distance, _ = human_pose.estimate_distance(pose_results)
            if human_pose.should_draw_face_features(state.estimated_distance):
                now_mono = time.monotonic()
                if (
                    self.face_mesh_submit_interval == 0.0
                    or now_mono - self._last_face_mesh_submit >= self.face_mesh_submit_interval
                ):
                    self._cached_face_mesh_results = human_pose.get_face_mesh(frame)
                    self._last_face_mesh_submit = now_mono
                state.face_mesh_results = self._cached_face_mesh_results
            else:
                self._cached_face_mesh_results = None

            if pose_results and pose_results.pose_landmarks:
                mode_manager.set_mode(ModeManager.MODE_POSE)
            elif mode_manager.get_time_since_last_mode_update() > self.pose_timeout:
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

            if human_pose.is_arms_crossed(pose_results) and state.eyes_visible:
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        else:
            # Clock and fallback modes.
            if mode_manager.pose_enabled:
                state.eyes_visible, state.reason, state.angle = (
                    human_pose.eyes_visible_and_facing_camera(pose_results)
                )
                state.estimated_distance, _ = human_pose.estimate_distance(pose_results)
                if (
                    pose_results
                    and pose_results.pose_landmarks
                    and state.eyes_visible
                    and state.estimated_distance is not None
                    and state.estimated_distance < self.pose_distance_threshold
                ):
                    if mode_manager.mode not in (
                        ModeManager.MODE_MENU,
                        ModeManager.MODE_PAINT,
                        ModeManager.MODE_CARICATURE,
                    ):
                        mode_manager.set_mode(ModeManager.MODE_POSE)

            if human_pose.is_arms_crossed(pose_results) and (
                state.eyes_visible or not mode_manager.pose_enabled
            ):
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        if current_mode == ModeManager.MODE_PAINT and mode_manager.mode != ModeManager.MODE_PAINT:
            paint_mode.clear()

        return state
