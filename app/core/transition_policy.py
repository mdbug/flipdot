from dataclasses import dataclass
from datetime import datetime
import threading
import time

import app.services.human_pose as human_pose
from app.services.worldcup import get_worldcup_scorecard
from app.core.mode_manager import ModeManager


@dataclass
class TransitionState:
    """Per-frame state produced by mode transition rules."""

    face_mesh_results: object
    eyes_visible: bool
    reason: str
    estimated_distance: float | None
    angle: float | None


class TransitionPolicy:
    """Centralized mode transition logic for the main loop."""

    WORLDCUP_LIVE_CHECK_INTERVAL = 30.0

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
        self.face_mesh_submit_interval = 0.0 if face_mesh_max_fps <= 0 else (1.0 / face_mesh_max_fps)
        self._last_face_mesh_submit = 0.0
        self._cached_face_mesh_results = None
        self._last_worldcup_live_check = 0.0
        self._cached_worldcup_live = False

    def get_sleep_settings(self) -> dict[str, int | bool]:
        with self._sleep_lock:
            return {
                "enabled": self.sleep_enabled,
                "start_hour": self.sleep_start_hour,
                "end_hour": self.sleep_end_hour,
            }

    def set_sleep_settings(self, *, enabled: bool, start_hour: int, end_hour: int) -> dict[str, int | bool]:
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
        now_mono = time.monotonic()
        if now_mono - self._last_worldcup_live_check < self.WORLDCUP_LIVE_CHECK_INTERVAL:
            return self._cached_worldcup_live

        self._last_worldcup_live_check = now_mono
        payload = get_worldcup_scorecard()
        events = payload.get("events") or []
        self._cached_worldcup_live = any(event.get("status_bucket") == "live" for event in events)
        return self._cached_worldcup_live

    def is_sleep_hour(self, now: datetime | None = None) -> bool:
        with self._sleep_lock:
            sleep_enabled = self.sleep_enabled
            sleep_start_hour = self.sleep_start_hour
            sleep_end_hour = self.sleep_end_hour

        if not sleep_enabled:
            return False

        if now is None:
            now = datetime.now()

        # Preserve existing behavior: only support non-wrapping ranges.
        return (
            sleep_end_hour > sleep_start_hour
            and sleep_start_hour <= now.hour < sleep_end_hour
        )

    def apply(self, *, frame, pose_results, mode_manager: ModeManager, paint_mode) -> TransitionState:
        state = TransitionState(
            face_mesh_results=None,
            eyes_visible=False,
            reason="",
            estimated_distance=None,
            angle=None,
        )

        now = datetime.now()
        if self.is_sleep_hour(now):
            mode_manager.set_mode(ModeManager.MODE_SLEEP)
            return state

        current_mode = mode_manager.mode

        # Prioritize live World Cup information when idle on clock mode.
        if current_mode == ModeManager.MODE_CLOCK and self._is_worldcup_live():
            mode_manager.set_mode(ModeManager.MODE_WORLDCUP)
            return state

        if current_mode in (
            ModeManager.MODE_MENU,
            ModeManager.MODE_PAINT,
            ModeManager.MODE_PERCUSSION,
            ModeManager.MODE_AUTODRUM,
            ModeManager.MODE_BEATMIRROR,
            ModeManager.MODE_TETRIS,
            ModeManager.MODE_PONG,
            ModeManager.MODE_WORLDCUP,
        ):
            if human_pose.is_arms_crossed(pose_results):
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        elif current_mode == ModeManager.MODE_CARICATURE:
            # No pose processing needed; caricature handles its own state.
            pass

        elif current_mode == ModeManager.MODE_POSE:
            state.eyes_visible, state.reason, state.angle = human_pose.eyes_visible_and_facing_camera(pose_results)
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
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        else:
            # Clock and fallback modes.
            if mode_manager.pose_enabled:
                state.eyes_visible, state.reason, state.angle = human_pose.eyes_visible_and_facing_camera(pose_results)
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

            if human_pose.is_arms_crossed(pose_results) and (state.eyes_visible or not mode_manager.pose_enabled):
                mode_manager.click_menu()
            else:
                mode_manager.reset_menu_click()

        if current_mode == ModeManager.MODE_PAINT and mode_manager.mode != ModeManager.MODE_PAINT:
            paint_mode.clear()

        return state
