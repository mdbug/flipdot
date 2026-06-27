from typing import Any

import cv2
import numpy as np

import app.services.human_pose as human_pose
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame


class Paint:
    """Free-draw canvas: dwell to toggle drawing (pose) or hold a button (web/controller)."""

    CLICK_TIME = 20  # Number of frames to hold position to toggle drawing

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.last_pointer_position: tuple[int, int] | None = None
        self.last_pointer_duration = 0
        self.drawing = False

    def clear(self) -> None:
        """Wipe the canvas and reset pointer/drawing state."""
        self.canvas[:, :] = 0
        self.drawing = False
        self.last_pointer_position = None
        self.last_pointer_duration = 0

    def _pointer_to_pixel(self, source: str, x: float, y: float) -> tuple[int, int]:
        """Convert a normalized (0..1) pointer to clamped canvas pixels (pose is mirrored)."""
        if source == "pose":
            pixel_x = int(self.width - (x * self.width))
        else:
            pixel_x = int(x * self.width)
        pixel_y = int(y * self.height)
        pixel_x = max(0, min(self.width - 1, pixel_x))
        pixel_y = max(0, min(self.height - 1, pixel_y))
        return pixel_x, pixel_y

    def get_frame(self, pose_results: Any, input_hub: Any = None) -> Frame:
        """Apply the active pointer to the canvas and return it with a pointer + status bar."""
        get_allowed_sources = getattr(self.mode_manager, "get_allowed_input_sources", None)
        allowed_sources = (
            get_allowed_sources(include_web=True)
            if callable(get_allowed_sources)
            else {"pose", "controller", "web"}
        )
        allow_pose_fallback = "pose" in allowed_sources
        pointer_source = "pose" if allow_pose_fallback else "web"
        web_button_down = input_hub.is_button_down(source="web") if input_hub is not None else False
        controller_button_down = (
            input_hub.is_button_down(source="controller")
            if input_hub is not None and "controller" in allowed_sources
            else False
        )
        pointer_button_down = web_button_down or controller_button_down
        pointer_sample = (
            input_hub.get_active_pointer(max_age_sec=0.8, allowed_sources=allowed_sources)
            if input_hub is not None
            else None
        )
        if pointer_sample is not None:
            pointer_source = pointer_sample.source
            pixel_x, pixel_y = self._pointer_to_pixel(
                pointer_sample.source, pointer_sample.x, pointer_sample.y
            )
            has_pointer = True
        elif allow_pose_fallback:
            finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
            has_pointer = finger_x is not None and finger_y is not None
            if finger_x is not None and finger_y is not None:
                pixel_x, pixel_y = self._pointer_to_pixel("pose", finger_x, finger_y)
        else:
            has_pointer = False

        if has_pointer:
            if pointer_source in ("web", "controller"):
                self.last_pointer_duration = 0
                if self.last_pointer_position is not None and pointer_button_down:
                    cv2.line(
                        self.canvas, self.last_pointer_position, (pixel_x, pixel_y), 1, thickness=1
                    )
            else:
                if self.last_pointer_position is not None:
                    if (pixel_x, pixel_y) == self.last_pointer_position:
                        self.last_pointer_duration += 1
                    else:
                        self.last_pointer_duration = 0

                    if self.last_pointer_duration == self.CLICK_TIME:
                        if pixel_x == self.width - 1 and pixel_y == 0:
                            self.canvas[:, :] = 0
                            self.drawing = False
                        else:
                            self.drawing = not self.drawing

                    if self.drawing:
                        cv2.line(
                            self.canvas,
                            self.last_pointer_position,
                            (pixel_x, pixel_y),
                            1,
                            thickness=1,
                        )

            self.last_pointer_position = (pixel_x, pixel_y)
        else:
            self.last_pointer_position = None

        frame = self.canvas.copy()
        if pointer_source == "pose":
            frame = human_pose.draw_right_index_pointer(frame, pose_results, size=2)
        else:
            if self.last_pointer_position is not None:
                pointer_x = self.last_pointer_position[0] / self.width
                pointer_y = self.last_pointer_position[1] / self.height
            else:
                pointer_x = None
                pointer_y = None
            frame = human_pose.draw_pointer(frame, pointer_x, pointer_y, size=2, mirror_x=False)
        if pointer_source in ("web", "controller"):
            if pointer_button_down:
                frame[self.height - 1, 0 : self.width] = 1
            else:
                frame[self.height - 1, 0 : self.width] = 0
        elif not self.drawing:
            if self.last_pointer_duration > self.CLICK_TIME:
                frame[self.height - 1, 0 : self.width] = 0
            else:
                frame[
                    self.height - 1,
                    0 : min(
                        int(self.last_pointer_duration / self.CLICK_TIME * self.width), self.width
                    ),
                ] = 1
        else:
            if self.last_pointer_duration > self.CLICK_TIME:
                frame[self.height - 1, 0 : self.width] = 1
            else:
                frame[
                    self.height - 1,
                    0 : self.width
                    - min(
                        int(self.last_pointer_duration / self.CLICK_TIME * self.width), self.width
                    ),
                ] = 1

        return frame
