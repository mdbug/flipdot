import time

class ModeManager:
    MODE_SLEEP = 'sleep'
    MODE_CLOCK = 'clock'
    MODE_POSE = 'pose'
    MODE_DEFAULT = MODE_CLOCK
    MAX_FPS = {
        MODE_SLEEP: 1,
        MODE_CLOCK: 4,
        MODE_POSE: 30,
    }

    def __init__(self, mode=MODE_DEFAULT):
        self.last_mode = None
        self.mode = mode
        self.mode_start_time = time.time()
        self.mode_update_time = time.time()

    def set_mode(self, mode):
        if mode != self.mode:
            self.last_mode = self.mode
            self.mode_start_time = time.time()
        if mode in (self.MODE_SLEEP, self.MODE_CLOCK, self.MODE_POSE):
            self.mode = mode
            self.mode_update_time = time.time()
        else:
            raise ValueError("Invalid display mode")

    def get_mode_time(self):
        return time.time() - self.mode_start_time

    def get_time_since_last_mode_update(self):
        return time.time() - self.mode_update_time

    def get_fps_limit(self):
        if self.get_mode_time() > 5:
            return self.MAX_FPS.get(self.mode, 1)
        else:
            return 30 