from dataclasses import dataclass
from datetime import datetime

import app.services.human_pose as human_pose
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

    def __init__(
        self,
        *,
        pose_timeout: float,
        sleep_start_hour: int,
        sleep_end_hour: int,
        pose_distance_threshold: float = 1.3,
    ) -> None:
        self.pose_timeout = pose_timeout
        self.sleep_start_hour = sleep_start_hour
        self.sleep_end_hour = sleep_end_hour
        self.pose_distance_threshold = pose_distance_threshold

    def is_sleep_hour(self, now: datetime | None = None) -> bool:
        if now is None:
            now = datetime.now()

        # Preserve existing behavior: only support non-wrapping ranges.
        return (
            self.sleep_end_hour > self.sleep_start_hour
            and self.sleep_start_hour <= now.hour < self.sleep_end_hour
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

        if current_mode in (
            ModeManager.MODE_MENU,
            ModeManager.MODE_PAINT,
            ModeManager.MODE_PERCUSSION,
            ModeManager.MODE_AUTODRUM,
            ModeManager.MODE_BEATMIRROR,
            ModeManager.MODE_TETRIS,
            ModeManager.MODE_PONG,
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
                state.face_mesh_results = human_pose.get_face_mesh(frame)

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
