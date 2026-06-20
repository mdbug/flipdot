import time
import logging


logger = logging.getLogger(__name__)

class ModeManager:
    CONTROL_GESTURE = 'gesture'
    CONTROL_CONTROLLER = 'controller'

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
    MODE_FONT_PREVIEW = 'font_preview'
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
        MODE_WORLDCUP: 30,
        MODE_BOARD: 30,
        MODE_FONT_PREVIEW: 30,
    }

    def __init__(self, mode=MODE_DEFAULT):
        self.last_mode = None
        self.mode = mode
        self.mode_start_time = time.time()
        self.mode_update_time = time.time()
        self.menu_click_start = None
        self.pose_enabled = True
        self.control_source = self.CONTROL_GESTURE
        self.controller_connected = False

    def set_mode(self, mode, entered_via=None):
        requested_mode = mode
        if mode == self.MODE_POSE and not self.pose_enabled:
            mode = self.MODE_CLOCK

        normalized_source = self.normalize_control_source(entered_via)
        if normalized_source is not None:
            self.control_source = normalized_source

        if mode != self.mode:
            previous_mode = self.mode
            self.last_mode = self.mode
            self.mode_start_time = time.time()
            logger.info("Mode changed from %s to %s (requested=%s)", previous_mode, mode, requested_mode)

        self.mode = mode
        self.mode_update_time = time.time()

    def click_menu(self, entered_via=None):
        if self.menu_click_start is None:
            self.menu_click_start = time.time()
        elif time.time() - self.menu_click_start > 2:
            if self.mode != self.MODE_MENU:
                self.set_mode(self.MODE_MENU, entered_via=entered_via)
            else:
                previous_mode = self.last_mode if self.last_mode not in (None, self.MODE_MENU) else self.MODE_POSE
                self.set_mode(previous_mode, entered_via=entered_via)
            self.menu_click_start = None

    def toggle_menu(self, entered_via=None):
        if self.mode != self.MODE_MENU:
            self.set_mode(self.MODE_MENU, entered_via=entered_via)
            return

        previous_mode = self.last_mode if self.last_mode not in (None, self.MODE_MENU) else self.MODE_POSE
        self.set_mode(previous_mode, entered_via=entered_via)
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

    @classmethod
    def normalize_control_source(cls, source):
        if source in (cls.CONTROL_GESTURE, 'pose'):
            return cls.CONTROL_GESTURE
        if source == cls.CONTROL_CONTROLLER:
            return cls.CONTROL_CONTROLLER
        return None

    def update_controller_connected(self, connected):
        connected = bool(connected)
        was_connected = self.controller_connected
        self.controller_connected = connected
        if connected and not was_connected:
            self.control_source = self.CONTROL_CONTROLLER
            return True
        return False

    def get_effective_control_source(self):
        if not self.controller_connected:
            return self.CONTROL_GESTURE
        return self.control_source

    def get_allowed_input_sources(self, *, include_web=True):
        effective = self.get_effective_control_source()
        allowed = {'pose'} if effective == self.CONTROL_GESTURE else {'controller'}
        if include_web:
            allowed.add('web')
        return allowed
