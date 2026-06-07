import base64
import json
import threading
import time

import anthropic
import cv2
import numpy as np

import text as text_module
from mode_manager import ModeManager


class Caricature:
    STATE_COUNTDOWN = 'countdown'
    STATE_CAPTURING = 'capturing'
    STATE_LOADING = 'loading'
    STATE_DISPLAYING = 'displaying'
    STATE_ERROR = 'error'

    COUNTDOWN_DURATION = 3  # seconds of countdown before capture
    DISPLAY_DURATION = 30   # seconds before auto-returning to clock

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self._lock = threading.Lock()
        self._generation = 0
        self._last_mode_start_time = None
        self.state = None
        self.caricature_dots = None
        self.countdown_start = None
        self.display_start_time = None
        self.error_msg = None

    def _reset(self):
        with self._lock:
            self._generation += 1
            self.state = self.STATE_COUNTDOWN
            self.caricature_dots = None
            self.countdown_start = time.time()
            self.display_start_time = None
            self.error_msg = None

    def _call_claude_api(self, image_b64, generation):
        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a pixel artist. Analyze the face in this image and create a 28x28 monochrome caricature."
                                "Return ONLY a JSON array of 28 arrays, each containing exactly 28 integers "
                                "(0=white/background, 1=black/drawn). "
                                "The caricature should fill most of the 28x28 grid. "
                                "No explanations, no markdown, just the raw JSON array."
                            ),
                        },
                    ],
                }],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if '```' in raw:
                for part in raw.split('```'):
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('['):
                        raw = part
                        break

            grid = json.loads(raw)
            dots = np.array(grid, dtype=np.uint8)
            if dots.shape != (self.height, self.width):
                raise ValueError(f"Grid shape {dots.shape} != ({self.height}, {self.width})")

            with self._lock:
                if self._generation == generation:
                    self.caricature_dots = np.clip(dots, 0, 1)
                    self.state = self.STATE_DISPLAYING
                    self.display_start_time = time.time()
                    cv2.imwrite('caricature_result.png', self.caricature_dots * 255)
                    print("Caricature: saved result to caricature_result.png")

        except Exception as e:
            print(f"Caricature API error: {e}")
            with self._lock:
                if self._generation == generation:
                    self.state = self.STATE_ERROR
                    self.error_msg = str(e)[:24]

    def _start_capture(self, camera_frame):
        if camera_frame is None:
            with self._lock:
                self.state = self.STATE_ERROR
                self.error_msg = "No frame"
            return

        success, buf = cv2.imencode('.jpg', camera_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            with self._lock:
                self.state = self.STATE_ERROR
                self.error_msg = "Encode failed"
            return

        cv2.imwrite('caricature_debug.jpg', camera_frame)
        print("Caricature: saved capture to caricature_debug.jpg")

        image_b64 = base64.standard_b64encode(buf).decode('utf-8')

        with self._lock:
            generation = self._generation
            self.state = self.STATE_LOADING

        thread = threading.Thread(
            target=self._call_claude_api,
            args=(image_b64, generation),
            daemon=True,
        )
        thread.start()

    def get_frame(self, camera_frame):
        # Auto-reset when the mode is freshly entered
        if self._last_mode_start_time != self.mode_manager.mode_start_time:
            self._last_mode_start_time = self.mode_manager.mode_start_time
            self._reset()

        with self._lock:
            state = self.state
            countdown_start = self.countdown_start

        if state == self.STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            seconds_left = self.COUNTDOWN_DURATION - int(elapsed)
            if elapsed >= self.COUNTDOWN_DURATION:
                with self._lock:
                    self.state = self.STATE_CAPTURING
            return self._make_countdown_frame(max(1, seconds_left))

        if state == self.STATE_CAPTURING:
            self._start_capture(camera_frame)
            return self._make_loading_frame()

        if state == self.STATE_LOADING:
            return self._make_loading_frame()

        if state == self.STATE_DISPLAYING:
            if time.time() - self.display_start_time > self.DISPLAY_DURATION:
                self.mode_manager.set_mode(ModeManager.MODE_CLOCK)
                return np.zeros((self.height, self.width), dtype=np.uint8)
            with self._lock:
                return self.caricature_dots.copy()

        if state == self.STATE_ERROR:
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
            text_module.write(frame, "ERR", x=1, y=10, size=5)
            return frame

        return np.zeros((self.height, self.width), dtype=np.uint8)

    def _make_countdown_frame(self, seconds_left):
        """Show the countdown digit centred and scaled up 3x."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        # Render digit into a 3x5 buffer then scale 3x → 9x15
        small = np.zeros((5, 3), dtype=np.uint8)
        text_module.write(small, str(seconds_left), x=0, y=0, size=5)
        big = np.kron(small, np.ones((3, 3), dtype=np.uint8))
        y_off = (self.height - big.shape[0]) // 2
        x_off = (self.width - big.shape[1]) // 2
        frame[y_off:y_off + big.shape[0], x_off:x_off + big.shape[1]] = big
        return frame

    def _make_loading_frame(self):
        """Ping-pong vertical bar sweeping left-right to indicate processing."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        # One full left-right-left cycle every 1.2 seconds
        cycle = (time.time() % 1.2) / 1.2  # 0.0 → 1.0
        pos = cycle * 2  # 0.0 → 2.0
        if pos > 1:
            pos = 2 - pos  # bounce back
        col = int(pos * (self.width - 2))
        frame[:, col:col + 2] = 1
        return frame
