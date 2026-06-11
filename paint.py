import numpy as np
import human_pose
import cv2

class Paint:
    CLICK_TIME = 20  # Number of frames to hold position to toggle drawing

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.canvas = np.zeros((height, width), dtype=np.uint8)
        self.last_pointer_position = None
        self.last_pointer_duration = 0
        self.drawing = False

    def clear(self):
        self.canvas[:, :] = 0
        self.drawing = False
        self.last_pointer_position = None
        self.last_pointer_duration = 0

    def get_frame(self, pose_results):
        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_x is not None and finger_y is not None:
            pixel_x = min(int(self.width - (finger_x*self.width)), self.width - 1)
            pixel_y = min(int(finger_y * self.height), self.height - 1)

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
                    cv2.line(self.canvas, self.last_pointer_position, (pixel_x, pixel_y), 1, thickness=1)

            self.last_pointer_position = (pixel_x, pixel_y)
        else:
            self.last_pointer_position = None

        frame = self.canvas.copy()
        frame = human_pose.draw_right_index_pointer(frame, pose_results, size=2)
        if not self.drawing:
            if self.last_pointer_duration > self.CLICK_TIME:
                frame[self.height-1, 0:self.width] = 0
            else:
                frame[self.height-1, 0:min(int(self.last_pointer_duration/self.CLICK_TIME*self.width), self.width)] = 1
        else:
            if self.last_pointer_duration > self.CLICK_TIME:
                frame[self.height-1, 0:self.width] = 1
            else:
                frame[self.height-1, 0:self.width - min(int(self.last_pointer_duration/self.CLICK_TIME*self.width), self.width)] = 1

        return frame
