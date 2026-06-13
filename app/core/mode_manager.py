import time
import logging


logger = logging.getLogger(__name__)

class ModeManager:
    MODE_SLEEP = 'sleep'
    MODE_CLOCK = 'clock'
    MODE_POSE = 'pose'
    MODE_MENU = 'menu'
    MODE_PAINT = 'paint'
    MODE_CARICATURE = 'caricature'
    MODE_PERCUSSION = 'percussion'
    MODE_AUTODRUM = 'autodrum'
    MODE_BEATMIRROR = 'beatmirror'
    MODE_TETRIS = 'tetris'
    MODE_PONG = 'pong'
    MODE_WORLDCUP = 'worldcup'
    MODE_BOARD = 'board'
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
        MODE_WORLDCUP: 4,
        MODE_BOARD: 30,
    }

    def __init__(self, mode=MODE_DEFAULT):
        self.last_mode = None
        self.mode = mode
        self.mode_start_time = time.time()
        self.mode_update_time = time.time()
        self.menu_click_start = None
        self.pose_enabled = True

    def set_mode(self, mode):
        requested_mode = mode
        if mode == self.MODE_POSE and not self.pose_enabled:
            mode = self.MODE_CLOCK

        if mode != self.mode:
            previous_mode = self.mode
            self.last_mode = self.mode
            self.mode_start_time = time.time()
            logger.info("Mode changed from %s to %s (requested=%s)", previous_mode, mode, requested_mode)

        self.mode = mode
        self.mode_update_time = time.time()

    def click_menu(self):
        if self.menu_click_start is None:
            self.menu_click_start = time.time()
        elif time.time() - self.menu_click_start > 2:
            if self.mode != self.MODE_MENU:
                self.set_mode(self.MODE_MENU)
            else:
                previous_mode = self.last_mode if self.last_mode not in (None, self.MODE_MENU) else self.MODE_POSE
                self.set_mode(previous_mode)
            self.menu_click_start = None

    def toggle_menu(self):
        if self.mode != self.MODE_MENU:
            self.set_mode(self.MODE_MENU)
            return

        previous_mode = self.last_mode if self.last_mode not in (None, self.MODE_MENU) else self.MODE_POSE
        self.set_mode(previous_mode)
        self.menu_click_start = None

    def reset_menu_click(self):
        self.menu_click_start = None

    def get_mode_time(self):
        return time.time() - self.mode_start_time

    def get_time_since_last_mode_update(self):
        return time.time() - self.mode_update_time

    def get_fps_limit(self):
        if self.get_mode_time() > 5:
            return self.MAX_FPS.get(self.mode, 1)
        else:
            return 30 
    
    def toggle_pose_enabled(self):
        self.pose_enabled = not self.pose_enabled
        logger.info("Pose mode enabled=%s", self.pose_enabled)
