
from text import write
import numpy as np
import time
import human_pose
from mode_manager import ModeManager

try:
    from mediapipe.python.solutions.pose import PoseLandmark
except ModuleNotFoundError:
    import mediapipe as mp

    PoseLandmark = mp.solutions.pose.PoseLandmark

class MenuItem:
    CLICK_TIME = 2

    def __init__(self, label, row, width, on_click=None):
        self.label = label
        self.row = row
        self.hovered = False
        self.width = width
        self.hover_start_time = None
        self.on_click = on_click
        self.page = 0

    def is_hovered(self, y):
        return (self.row*8 <= y < (self.row+1)*8)

    def hover(self, hovering):
        if hovering and not self.hovered:
            self.hover_start_time = time.time()
        elif not hovering:
            self.hover_start_time = None
        else:
            hover_duration = self.get_hover_duration()
            if hover_duration >= MenuItem.CLICK_TIME and self.on_click:
                self.click()
                self.hover_start_time = None

        self.hovered = hovering

    def click(self):
        if self.on_click:
            self.on_click()
    
    def get_hover_duration(self):
        if self.hover_start_time:
            return time.time() - self.hover_start_time
        return 0

    def draw_hover(self, frame):
        if self.hovered:
            duration = self.get_hover_duration()
            slice = min(int(self.width*duration/MenuItem.CLICK_TIME), 28)
            frame[self.row*8:self.row*8+7, 0:slice] = frame[self.row*8:self.row*8+7, 0:slice] ^ 1

class Button(MenuItem):
    def __init__(self, label, row, width, on_click=None):
        super().__init__(label, row, width, on_click)

    def draw(self, frame):
        write(frame, self.label, x=1, y=self.row*8+1, size=5)
        sep = min(self.row*8+7, frame.shape[0]-1)
        frame[sep, 0:self.width] = 1
        self.draw_hover(frame)

class Checkbox(MenuItem):
    def __init__(self, label, row, width, checked=False, on_click=None):
        super().__init__(label, row, width, on_click=on_click)
        self.checked = checked

    def draw(self, frame):
        write(frame, self.label, x=7, y=self.row*8+1, size=5)
        sep = min(self.row*8+7, frame.shape[0]-1)
        frame[sep, 0:self.width] = 1
        self.draw_hover(frame)
        frame[self.row*8:self.row*8+7, 0:6] = 0
        frame[self.row*8+1:self.row*8+6, 0:5] = 1
        frame[self.row*8+2:self.row*8+5, 1:4] = 0
        if self.checked:
            frame[self.row*8+3, 2] = 1

        # blink while hovered
        if self.hovered and self.hover_start_time:
            frame[self.row*8+3, 2] = int(time.time()*2) % 2
    
    def hover(self, hovering):
        super().hover(hovering)

    def click(self):
        self.checked = not self.checked
        if self.on_click:
            self.on_click()

class Menu:
    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.items = []
        self.mode_manager = mode_manager
        self.add_item(Button("CLOCK", 0, width, on_click=lambda: mode_manager.set_mode(ModeManager.MODE_CLOCK)))
        self.add_item(Button("PAINT", 1, width, on_click=lambda: mode_manager.set_mode(ModeManager.MODE_PAINT)))
        self.add_item(Checkbox("POSE", 2, width, checked=mode_manager.pose_enabled, on_click=mode_manager.toggle_pose_enabled))
        self.add_item(Button("CARIC", 3, width, on_click=lambda: mode_manager.set_mode(ModeManager.MODE_CARICATURE)))

    def add_item(self, item):
        self.items.append(item)

    def get_frame(self, pose_results):
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)

        for item in self.items:
            if finger_y is not None:
                item.hover(item.is_hovered(finger_y * self.height))

            item.draw(frame)

        frame = human_pose.draw_right_index_pointer(frame, pose_results)
        return frame