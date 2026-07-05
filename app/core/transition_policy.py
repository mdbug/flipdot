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
    # Face-mesh poll interval for a manual caricature abandoned with nobody
    # in frame: enough to notice a returning face, without running the
    # landmarker at full rate on an empty scene forever.
    ABANDONED_FACE_MESH_INTERVAL = 2.0
    # A manual (menu/MCP-launched) caricature is exempt from the chain's
    # presence rules, but an empty scene this long means it was abandoned:
    # return to clock so pose inference stops running at the mode's full
    # rate forever and clock-anchored behaviors (worldcup, hourly scripts,
    # chain re-entry) can resume.
    MANUAL_CARICATURE_ABANDON_TIMEOUT = 300.0
    # Modes the gesture chain may preempt with sandfall when a person shows
    # up. An allow-list so new modes are safe from passer-by hijack by
    # default (the board and running scripts must not be taken over).
    CHAIN_PREEMPTIBLE_MODES = (ModeManager.MODE_CLOCK,)
    # Very-close hysteresis: enter caricature below ENTER, leave it above EXIT.
    # ENTER mirrors human_pose.VERY_CLOSE_FACE_DISTANCE's default (0.5).
    CARICATURE_ENTER_DISTANCE = 0.5
    CARICATURE_EXIT_DISTANCE = 0.65
    # The distance estimate swings wildly on single frames (especially while
    # the viewer turns away), so both caricature edges require the reading to
    # hold continuously before they fire.
    CARICATURE_ENTER_HOLD_SECONDS = 1.0
    CARICATURE_EXIT_HOLD_SECONDS = 2.0
    # A total dropout (no distance, no face) may keep the exit hold running
    # only after this much definite backing-away was observed first; a hold
    # seeded by a single wild far reading is cancelled instead of silently
    # completing against a still-present viewer.
    CARICATURE_EXIT_CONFIRM_SECONDS = 0.3
    # Brief facing-check flickers hold an in-progress menu dwell, but a
    # sustained untrusted streak resets it so turned-away arms-crossed
    # misfires can never bank dwell time toward opening the menu.
    MENU_GESTURE_UNTRUST_GRACE_SECONDS = 0.5

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
        # Stage of the auto gesture chain (clock -> sandfall -> caricature):
        # the chain-entered mode this policy expects to be active, or None
        # when the active mode is menu/MCP-launched and therefore exempt from
        # the chain's presence/distance rules. The menu only parks the chain
        # (closing it restores the previous mode); any other external mode
        # change — including an explicit menu selection — ends it.
        self._chain_stage: str | None = None
        # True while the chain is parked because the menu is open on top of it.
        self._chain_parked_in_menu = False
        # Hold timers (time.monotonic) smoothing the noisy distance estimate
        # on the caricature enter/exit edges. The enter hold pauses (rather
        # than resets) across pose dropouts, which are common at very close
        # range; ``_very_close_paused_at`` marks where the pause began.
        self._very_close_since: float | None = None
        self._very_close_paused_at: float | None = None
        self._backing_away_since: float | None = None
        # Last time a definite far reading extended the exit hold; a total
        # dropout consults it to tell confirmed backing-away from one wild
        # frame (see CARICATURE_EXIT_CONFIRM_SECONDS).
        self._backing_away_last_definite: float | None = None
        # Start of the current untrusted arms-crossed streak (facing gate
        # down while the gesture reads crossed); bounds how long a menu
        # dwell may be held across untrusted frames.
        self._menu_gesture_untrusted_since: float | None = None

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

    def _sustained_very_close(
        self, person_present: bool, eyes_visible: bool, distance: float | None
    ) -> bool:
        """Whether the viewer has stayed very close, facing the camera, long enough.

        The raw distance estimate produces wild single-frame readings while the
        viewer is turned away, so caricature entry requires the facing gate plus
        an uninterrupted ``CARICATURE_ENTER_HOLD_SECONDS`` hold. Pose dropouts
        (no landmarks at all — common at very close range where the body leaves
        the frame) pause the hold rather than reset it; only a viewer who is
        demonstrably present but far or turned away resets it.
        """
        now_mono = time.monotonic()
        if person_present and eyes_visible and self._is_very_close(distance):
            if self._very_close_since is None:
                self._very_close_since = now_mono
            elif self._very_close_paused_at is not None:
                # Resume: absent time does not count toward the hold.
                self._very_close_since += now_mono - self._very_close_paused_at
            self._very_close_paused_at = None
            return now_mono - self._very_close_since >= self.CARICATURE_ENTER_HOLD_SECONDS
        if not person_present:
            if self._very_close_since is not None and self._very_close_paused_at is None:
                self._very_close_paused_at = now_mono
            return False
        self._very_close_since = None
        self._very_close_paused_at = None
        return False

    def _reset_hold_timers(self) -> None:
        """Clear the caricature enter/exit hold timers."""
        self._very_close_since = None
        self._very_close_paused_at = None
        self._backing_away_since = None
        self._backing_away_last_definite = None

    def _set_chain_stage(self, stage: str | None) -> None:
        """Move the gesture chain to ``stage`` (None ends it) and clear its hold timers."""
        self._chain_stage = stage
        self._chain_parked_in_menu = False
        self._reset_hold_timers()

    def _end_chain(self) -> None:
        """Forget the gesture chain and its hold timers."""
        self._set_chain_stage(None)

    def _enter_auto_caricature(self, mode_manager: ModeManager) -> None:
        """Hand the chain from sandfall to caricature."""
        self._set_chain_stage(ModeManager.MODE_CARICATURE)
        mode_manager.set_mode(ModeManager.MODE_CARICATURE)

    def _fill_facing_and_distance(self, state: TransitionState, pose_results: Any) -> None:
        """Fill the frame's facing/angle/distance readings from pose landmarks."""
        state.eyes_visible, state.reason, state.angle = human_pose.eyes_visible_and_facing_camera(
            pose_results
        )
        state.estimated_distance, _ = human_pose.estimate_distance(pose_results)

    def _update_presence_state(
        self, state: TransitionState, frame: Frame, pose_results: Any
    ) -> None:
        """Fill facing/distance state and, when close enough, face-mesh results."""
        self._fill_facing_and_distance(state, pose_results)
        if human_pose.should_draw_face_features(state.estimated_distance):
            state.face_mesh_results = self._submit_face_mesh(frame)
        else:
            self._cached_face_mesh_results = None

    def _update_exit_hold(self, state: TransitionState, face_present: bool) -> None:
        """Advance the chain caricature's backing-away exit hold.

        Sets ``state.caricature_exit_progress`` while the hold runs (1.0 means
        the hold is complete). A definite close reading cancels the hold, and
        so does a dropped distance estimate while the face mesh still tracks a
        face — pose distance routinely drops at very close range, where a
        tracked face proves the viewer has not left. A distance dropout with
        no tracked face (the viewer turned and walked off) keeps the hold
        running so a fast walk-away still hands off — but only when at least
        ``CARICATURE_EXIT_CONFIRM_SECONDS`` of definite far readings preceded
        the dropout; one wild far frame followed by a close-range dropout is
        noise, not a departure.
        """
        now_mono = time.monotonic()
        distance = state.estimated_distance
        if distance is not None:
            if distance > self.CARICATURE_EXIT_DISTANCE:
                if self._backing_away_since is None:
                    self._backing_away_since = now_mono
                self._backing_away_last_definite = now_mono
            else:
                self._backing_away_since = None
        elif face_present:
            self._backing_away_since = None
        elif self._backing_away_since is not None and (
            self._backing_away_last_definite is None
            or self._backing_away_last_definite - self._backing_away_since
            < self.CARICATURE_EXIT_CONFIRM_SECONDS
        ):
            self._backing_away_since = None
        if self._backing_away_since is not None:
            # Lets the caricature shrink back onto the viewer's head while
            # the exit hold runs down.
            state.caricature_exit_progress = min(
                1.0, (now_mono - self._backing_away_since) / self.CARICATURE_EXIT_HOLD_SECONDS
            )

    def _submit_face_mesh(self, frame: Frame, interval: float | None = None) -> Any:
        """Submit ``frame`` for face-mesh inference (throttled) and return the latest results.

        ``interval`` overrides the default submit throttle (used to slow-poll
        an abandoned manual caricature).
        """
        if interval is None:
            interval = self.face_mesh_submit_interval
        now_mono = time.monotonic()
        if interval == 0.0 or now_mono - self._last_face_mesh_submit >= interval:
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

        # The chain owns the mode it last set. Any external change (MCP, web,
        # worldcup, sleep, a menu selection of a different mode) ends it; the
        # menu itself only parks the chain, since closing the menu restores
        # the previous mode and the chain then resumes. Leaving the menu into
        # the chain mode *without* a restore means the user explicitly
        # selected it — a manual launch, which also ends the chain.
        if self._chain_stage is not None:
            if current_mode == ModeManager.MODE_MENU:
                self._chain_parked_in_menu = True
            elif current_mode != self._chain_stage:
                self._end_chain()
            elif self._chain_parked_in_menu and not mode_manager.consume_menu_restore():
                self._end_chain()
            else:
                self._chain_parked_in_menu = False
        if current_mode != self._chain_stage:
            # Hold timers only run while actually observing the chain mode.
            self._reset_hold_timers()

        # The menu's POSE toggle governs the whole auto gesture chain: turning
        # it off ends any chain-entered mode immediately (menu/MCP-launched
        # sandfall and caricature are unaffected).
        if not mode_manager.pose_enabled and self._chain_stage is not None:
            self._end_chain()
            if current_mode != ModeManager.MODE_MENU:
                mode_manager.set_mode(ModeManager.MODE_CLOCK)
                current_mode = mode_manager.mode
            else:
                # The chain mode the menu parked over is dead; closing the
                # menu must not resurrect it as an unowned manual mode.
                mode_manager.retarget_menu_restore(ModeManager.MODE_CLOCK)

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

        # Each branch decides whether the arms-crossed menu gesture is trusted
        # this frame; the shared handler runs once after the branch chain.
        # Presence modes require the facing gate because is_arms_crossed
        # misfires on a viewer who is turned away.
        menu_gesture_allowed = False

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
            menu_gesture_allowed = True

        elif current_mode == ModeManager.MODE_CARICATURE:
            # The live caricature always wants face-mesh landmarks (no distance
            # gate); the mode itself is a pure renderer over the results. A
            # manual caricature abandoned past the no-face timeout drops to a
            # slow face-mesh poll, so an empty scene does not run the
            # landmarker at full rate forever while a returning face (even
            # one pose misses) is still noticed within the poll interval.
            person_present = bool(pose_results and pose_results.pose_landmarks)
            abandoned_manual = (
                self._chain_stage != ModeManager.MODE_CARICATURE
                and not person_present
                and mode_manager.get_time_since_last_mode_update() > self.CARICATURE_NO_FACE_TIMEOUT
            )
            state.face_mesh_results = self._submit_face_mesh(
                frame,
                interval=self.ABANDONED_FACE_MESH_INTERVAL if abandoned_manual else None,
            )
            face_present = bool(
                state.face_mesh_results is not None
                and getattr(state.face_mesh_results, "multi_face_landmarks", None)
            )
            self._fill_facing_and_distance(state, pose_results)
            # A tracked face is as strong a facing signal as the pose gate —
            # and more reliable at caricature range, where pose landmarks
            # routinely drop out; without it a very close viewer could be
            # unable to open the menu at all.
            menu_gesture_allowed = state.eyes_visible or face_present

            if face_present:
                # Keepalive: refreshes mode_update_time without resetting the mode.
                mode_manager.set_mode(ModeManager.MODE_CARICATURE)

            if self._chain_stage == ModeManager.MODE_CARICATURE:
                # Chain-entered only: menu/MCP-launched caricature is exempt
                # from presence rules and idles on the invite face instead.
                self._update_exit_hold(state, face_present)
                if (
                    state.caricature_exit_progress is not None
                    and state.caricature_exit_progress >= 1.0
                ):
                    # Sustained backing past the hysteresis exit hands the
                    # chain back to sandfall.
                    self._set_chain_stage(ModeManager.MODE_SANDFALL)
                    mode_manager.set_mode(ModeManager.MODE_SANDFALL)
                elif (
                    not face_present
                    and mode_manager.get_time_since_last_mode_update()
                    > self.CARICATURE_NO_FACE_TIMEOUT
                ):
                    self._end_chain()
                    mode_manager.set_mode(ModeManager.MODE_CLOCK)
            elif (
                not person_present
                and not face_present
                and mode_manager.get_time_since_last_mode_update()
                > self.MANUAL_CARICATURE_ABANDON_TIMEOUT
            ):
                # A manual caricature is presence-exempt while anyone might
                # engage, but a scene empty for minutes was abandoned: return
                # to clock instead of running pose at full rate forever.
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

        elif current_mode == ModeManager.MODE_POSE:
            self._update_presence_state(state, frame, pose_results)
            menu_gesture_allowed = state.eyes_visible

            if pose_results and pose_results.pose_landmarks:
                # Keepalive: refreshes mode_update_time without resetting the mode.
                mode_manager.set_mode(ModeManager.MODE_POSE)
            elif mode_manager.get_time_since_last_mode_update() > self.pose_timeout:
                mode_manager.set_mode(ModeManager.MODE_CLOCK)

        elif current_mode == ModeManager.MODE_SANDFALL:
            # Face mesh feeds the eyes/mouth overlay on the silhouette.
            self._update_presence_state(state, frame, pose_results)
            menu_gesture_allowed = state.eyes_visible

            if self._chain_stage == ModeManager.MODE_SANDFALL:
                # Chain-entered sandfall mirrors pose mode's presence rules;
                # menu/web-launched sandfall idles indefinitely. The hold is
                # evaluated every frame; a pose dropout pauses it (the chain's
                # own timeout below bounds how long a pause can last).
                person_present = bool(pose_results and pose_results.pose_landmarks)
                if self._sustained_very_close(
                    person_present, state.eyes_visible, state.estimated_distance
                ):
                    self._enter_auto_caricature(mode_manager)
                elif person_present:
                    mode_manager.set_mode(ModeManager.MODE_SANDFALL)
                elif mode_manager.get_time_since_last_mode_update() > self.pose_timeout:
                    self._end_chain()
                    mode_manager.set_mode(ModeManager.MODE_CLOCK)

        else:
            # Clock and fallback modes.
            menu_gesture_allowed = not mode_manager.pose_enabled
            if mode_manager.pose_enabled:
                self._fill_facing_and_distance(state, pose_results)
                menu_gesture_allowed = state.eyes_visible
                if (
                    pose_results
                    and pose_results.pose_landmarks
                    and state.eyes_visible
                    and state.estimated_distance is not None
                    and state.estimated_distance < self.pose_distance_threshold
                ):
                    if mode_manager.mode in self.CHAIN_PREEMPTIBLE_MODES:
                        self._set_chain_stage(ModeManager.MODE_SANDFALL)
                        mode_manager.set_mode(ModeManager.MODE_SANDFALL)

        if human_pose.is_arms_crossed(pose_results):
            if menu_gesture_allowed:
                self._menu_gesture_untrusted_since = None
                mode_manager.click_menu(entered_via=ModeManager.CONTROL_GESTURE)
            else:
                # A brief untrusted flicker (e.g. the facing check dropping
                # mid-dwell) holds an in-progress dwell, but only within the
                # grace window: click_menu credits raw wall-clock time, so an
                # unbounded hold would let a long turned-away misfire streak
                # complete the dwell on the first trusted frame.
                now_mono = time.monotonic()
                if self._menu_gesture_untrusted_since is None:
                    self._menu_gesture_untrusted_since = now_mono
                elif (
                    now_mono - self._menu_gesture_untrusted_since
                    > self.MENU_GESTURE_UNTRUST_GRACE_SECONDS
                ):
                    mode_manager.reset_menu_click()
        else:
            self._menu_gesture_untrusted_since = None
            mode_manager.reset_menu_click()

        if current_mode == ModeManager.MODE_PAINT and mode_manager.mode != ModeManager.MODE_PAINT:
            paint_mode.clear()

        return state
