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
    # 0..1 progress of the auto-caricature exit hold; None when not exiting.
    caricature_exit_progress: float | None = None


class TransitionPolicy:
    """Centralized mode transition logic for the main loop.

    Besides sleep/clock housekeeping, this drives the gesture chain:
    clock -> sandfall (person present) -> caricature (very close, with
    hysteresis) and back.
    """

    WORLDCUP_LIVE_CHECK_INTERVAL = 30.0
    HOURLY_SCRIPT_DURATION = 60.0
    # Seconds without a detected face before caricature mode returns to clock.
    CARICATURE_NO_FACE_TIMEOUT = 8.0
    # Very-close hysteresis: enter caricature below ENTER, leave it above EXIT.
    # ENTER mirrors human_pose.VERY_CLOSE_FACE_DISTANCE's default (0.5).
    CARICATURE_ENTER_DISTANCE = 0.5
    CARICATURE_EXIT_DISTANCE = 0.65
    # The distance estimate swings wildly on single frames (especially while
    # the viewer turns away), so both caricature edges require the reading to
    # hold continuously before they fire.
    CARICATURE_ENTER_HOLD_SECONDS = 1.0
    CARICATURE_EXIT_HOLD_SECONDS = 2.0

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
        # Mode to return to when a policy-entered ("auto") caricature ends
        # because the viewer backed away. None => caricature (if active) was
        # menu/MCP-launched and ignores distance.
        self._caricature_return_mode: str | None = None
        # True only while sandfall was auto-entered from the clock; menu/web-
        # launched sandfall keeps its idle-forever behavior.
        self._sandfall_via_chain = False
        # Hold timers (time.monotonic) smoothing the noisy distance estimate
        # on the caricature enter/exit edges.
        self._very_close_since: float | None = None
        self._backing_away_since: float | None = None

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

    def _is_very_close(self, distance: float | None) -> bool:
        """Whether the viewer is inside the caricature enter threshold."""
        return distance is not None and distance < self.CARICATURE_ENTER_DISTANCE

    def _sustained_very_close(self, eyes_visible: bool, distance: float | None) -> bool:
        """Whether the viewer has stayed very close, facing the camera, long enough.

        The raw distance estimate produces wild single-frame readings while the
        viewer is turned away, so caricature entry requires the facing gate plus
        an uninterrupted ``CARICATURE_ENTER_HOLD_SECONDS`` hold.
        """
        if eyes_visible and self._is_very_close(distance):
            now_mono = time.monotonic()
            if self._very_close_since is None:
                self._very_close_since = now_mono
            return now_mono - self._very_close_since >= self.CARICATURE_ENTER_HOLD_SECONDS
        self._very_close_since = None
        return False

    def _enter_auto_caricature(self, mode_manager: ModeManager, return_mode: str) -> None:
        """Switch to caricature, remembering the chain mode to return to."""
        self._caricature_return_mode = return_mode
        self._very_close_since = None
        self._backing_away_since = None
        mode_manager.set_mode(ModeManager.MODE_CARICATURE)

    def _submit_face_mesh(self, frame: Frame) -> Any:
        """Submit ``frame`` for face-mesh inference (throttled) and return the latest results."""
        now_mono = time.monotonic()
        if (
            self.face_mesh_submit_interval == 0.0
            or now_mono - self._last_face_mesh_submit >= self.face_mesh_submit_interval
        ):
            self._cached_face_mesh_results = human_pose.get_face_mesh(frame)
            self._last_face_mesh_submit = now_mono
        return self._cached_face_mesh_results

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

        # Any external mode change (menu, MCP, web, worldcup, sleep) ends the
        # gesture chain: the caricature return mode is only meaningful while in
        # caricature, and the sandfall chain flag only while in sandfall or the
        # caricature it handed off to.
        if current_mode != ModeManager.MODE_CARICATURE:
            self._caricature_return_mode = None
            self._backing_away_since = None
        if current_mode not in (ModeManager.MODE_SANDFALL, ModeManager.MODE_CARICATURE):
            self._sandfall_via_chain = False
        if current_mode != ModeManager.MODE_SANDFALL:
            self._very_close_since = None

        # The menu's POSE toggle governs the whole auto gesture chain: turning
        # it off ends any chain-entered mode immediately (menu/MCP-launched
        # sandfall and caricature are unaffected).
        if not mode_manager.pose_enabled and (
            self._sandfall_via_chain or self._caricature_return_mode is not None
        ):
            self._sandfall_via_chain = False
            self._caricature_return_mode = None
            self._backing_away_since = None
            self._very_close_since = None
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
            ModeManager.MODE_LIFE,
        ):
            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_CARICATURE:
            # The live caricature always wants face-mesh landmarks (no distance
            # gate); the mode itself is a pure renderer over the results.
            state.face_mesh_results = self._submit_face_mesh(frame)
            face_present = bool(
                state.face_mesh_results is not None
                and getattr(state.face_mesh_results, "multi_face_landmarks", None)
            )
            state.estimated_distance, _ = human_pose.estimate_distance(pose_results)
            backing_away = (
                state.estimated_distance is not None
                and state.estimated_distance > self.CARICATURE_EXIT_DISTANCE
            )
            now_mono = time.monotonic()
            if self._caricature_return_mode is not None and backing_away:
                if self._backing_away_since is None:
                    self._backing_away_since = now_mono
                # Lets the caricature shrink back onto the viewer's head
                # while the exit hold runs down.
                state.caricature_exit_progress = min(
                    1.0, (now_mono - self._backing_away_since) / self.CARICATURE_EXIT_HOLD_SECONDS
                )
            else:
                self._backing_away_since = None

            if (
                self._caricature_return_mode is not None
                and self._backing_away_since is not None
                and now_mono - self._backing_away_since >= self.CARICATURE_EXIT_HOLD_SECONDS
            ):
                # Auto-entered from the gesture chain: sustained backing past
                # the hysteresis exit returns to the mode we came from.
                return_mode = self._caricature_return_mode
                self._caricature_return_mode = None
                self._backing_away_since = None
                mode_manager.set_mode(return_mode)
            elif face_present:
                # Keepalive: refreshes mode_update_time without resetting the mode.
                mode_manager.set_mode(ModeManager.MODE_CARICATURE)
            elif mode_manager.get_time_since_last_mode_update() > self.CARICATURE_NO_FACE_TIMEOUT:
                self._caricature_return_mode = None
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_POSE:
            state.eyes_visible, state.reason, state.angle = (
                human_pose.eyes_visible_and_facing_camera(pose_results)
            )
            state.estimated_distance, _ = human_pose.estimate_distance(pose_results)
            if human_pose.should_draw_face_features(state.estimated_distance):
                state.face_mesh_results = self._submit_face_mesh(frame)
            else:
                self._cached_face_mesh_results = None

            if pose_results and pose_results.pose_landmarks:
                # Keepalive: refreshes mode_update_time without resetting the mode.
                mode_manager.set_mode(ModeManager.MODE_POSE)
            elif mode_manager.get_time_since_last_mode_update() > self.pose_timeout:
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

            if human_pose.is_arms_crossed(pose_results) and state.eyes_visible:
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_SANDFALL:
            state.eyes_visible, state.reason, state.angle = (
                human_pose.eyes_visible_and_facing_camera(pose_results)
            )
            state.estimated_distance, _ = human_pose.estimate_distance(pose_results)
            # Face mesh feeds the eyes/mouth overlay on the silhouette.
            if human_pose.should_draw_face_features(state.estimated_distance):
                state.face_mesh_results = self._submit_face_mesh(frame)
            else:
                self._cached_face_mesh_results = None
            if self._sandfall_via_chain:
                # Chain-entered sandfall mirrors pose mode's presence rules;
                # menu/web-launched sandfall idles indefinitely.
                person_present = bool(pose_results and pose_results.pose_landmarks)
                if person_present and self._sustained_very_close(
                    state.eyes_visible, state.estimated_distance
                ):
                    self._enter_auto_caricature(mode_manager, ModeManager.MODE_SANDFALL)
                elif person_present:
                    mode_manager.set_mode(ModeManager.MODE_SANDFALL)
                elif mode_manager.get_time_since_last_mode_update() > self.pose_timeout:
                    mode_manager.set_mode(ModeManager.MODE_CLOCK)

            if human_pose.is_arms_crossed(pose_results):
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
                        self._sandfall_via_chain = True
                        mode_manager.set_mode(ModeManager.MODE_SANDFALL)

            if human_pose.is_arms_crossed(pose_results) and (
                state.eyes_visible or not mode_manager.pose_enabled
            ):
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                mode_manager.reset_menu_click()

        if current_mode == ModeManager.MODE_PAINT and mode_manager.mode != ModeManager.MODE_PAINT:
            paint_mode.clear()

        return state
