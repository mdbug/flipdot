import time
from typing import Any

import numpy as np

import app.services.human_pose as human_pose
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame


class Percussion:
    """Play the flip-dot panel as a drum machine.

    Each instrument is synthesised from the panel's physics: how many dots
    flip (loudness), solid block vs random scatter (texture), and a decaying
    rattle tail (sustain). Hand x controls tempo, hand y selects pattern.
    """

    MIN_BPM = 60
    MAX_BPM = 240
    DECAY_TICK = 0.05  # seconds between decay-tail flips

    # 8-step patterns; each step lists the regions to flip (empty = rest).
    PATTERNS: list[list[tuple[str, ...]]] = [
        # Four on the floor
        [("kick",), ("hat",), ("kick",), ("hat",), ("kick",), ("hat",), ("kick",), ("hat",)],
        # Rock backbeat
        [("kick",), ("hat",), ("snare",), ("hat",), ("kick",), ("kick",), ("snare",), ("hat",)],
        # Syncopated breakbeat
        [("kick",), ("hat",), ("snare",), ("kick",), (), ("kick",), ("snare",), ("tom",)],
        # Roll around the kit
        [
            ("kick",),
            ("snare",),
            ("tom",),
            ("hat",),
            ("kick", "snare"),
            ("tom",),
            ("hat",),
            ("kick", "tom"),
        ],
    ]

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.state = np.zeros((height, width), dtype=np.uint8)
        self.rng = np.random.default_rng()
        # Instruments: area, density (loudness/texture), decay tail densities.
        h, w = height, width
        self.instruments: dict[str, dict[str, Any]] = {
            # solid thump, no tail – the loudest single clack
            "kick": {"area": (h // 2, h, 0, w), "density": 1.0, "decay": []},
            # scattered crack with a short rattle
            "snare": {"area": (0, h // 2, 0, w // 2), "density": 0.6, "decay": [0.25]},
            # sparse tick – the quietest instrument
            "hat": {"area": (0, h // 4, w // 2, w), "density": 0.2, "decay": []},
            # dense punch with a slightly longer ring
            "tom": {"area": (h // 4, h // 2, w // 2, w), "density": 0.9, "decay": [0.35, 0.15]},
        }
        self.pattern_index = 0
        self.step = 0
        self.bpm: float = 120
        self.next_step_time = time.time()
        self._decay_events: list[tuple[float, str, float]] = []  # (due, instrument, density)

    def _scatter_flip(self, name: str, density: float) -> None:
        """XOR a random subset of the instrument's area; density 1.0 = solid."""
        r0, r1, c0, c1 = self.instruments[name]["area"]
        if density >= 1.0:
            self.state[r0:r1, c0:c1] ^= 1
        else:
            mask = self.rng.random((r1 - r0, c1 - c0)) < density
            self.state[r0:r1, c0:c1] ^= mask.astype(np.uint8)

    def _hit(self, name: str, now: float) -> None:
        """Trigger an instrument now and schedule its decay-tail flips."""
        inst = self.instruments[name]
        self._scatter_flip(name, inst["density"])
        for i, tail_density in enumerate(inst["decay"]):
            self._decay_events.append((now + (i + 1) * self.DECAY_TICK, name, tail_density))

    def adjust_bpm(self, delta: float) -> None:
        """Nudge the tempo by ``delta`` BPM, clamped to [MIN_BPM, MAX_BPM]."""
        self.bpm = max(self.MIN_BPM, min(self.MAX_BPM, float(self.bpm) + float(delta)))

    def cycle_pattern(self, delta: int) -> None:
        """Advance the active pattern by ``delta`` steps (wrapping)."""
        if not self.PATTERNS:
            return
        self.pattern_index = (self.pattern_index + int(delta)) % len(self.PATTERNS)

    def trigger_accent(self) -> None:
        """Fire a kick+snare accent immediately."""
        now = time.time()
        self._hit("kick", now)
        self._hit("snare", now)

    def get_frame(self, pose_results: Any) -> Frame:
        """Advance the sequencer (hand x = tempo, hand y = pattern) and render the panel."""
        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_x is not None and finger_y is not None:
            # Display is mirrored, so flip x for intuitive control.
            mirrored_x = min(max(1.0 - finger_x, 0.0), 1.0)
            self.bpm = self.MIN_BPM + mirrored_x * (self.MAX_BPM - self.MIN_BPM)
            self.pattern_index = min(int(finger_y * len(self.PATTERNS)), len(self.PATTERNS) - 1)

        pattern = self.PATTERNS[self.pattern_index]
        step_interval = 60.0 / self.bpm / 2  # eighth notes

        now = time.time()

        # Fire any due decay-tail flips (rattle / ring-out)
        if self._decay_events:
            due = [e for e in self._decay_events if e[0] <= now]
            self._decay_events = [e for e in self._decay_events if e[0] > now]
            for _, name, density in due:
                self._scatter_flip(name, density)

        while now >= self.next_step_time:
            self.step = (self.step + 1) % len(pattern)
            for name in pattern[self.step]:
                self._hit(name, now)
            self.next_step_time += step_interval
            # Don't try to catch up after long pauses (e.g. mode switch).
            if now - self.next_step_time > 1.0:
                self.next_step_time = now + step_interval

        frame = self.state.copy()

        # Step cursor along the bottom row.
        frame[-1, :] = 0
        seg = self.width // len(pattern)
        frame[-1, self.step * seg : (self.step + 1) * seg] = 1

        # Pattern indicator: one block per pattern on the left edge.
        frame[:, 0] = 0
        block = self.height // (len(self.PATTERNS) + 1)
        for i in range(self.pattern_index + 1):
            frame[i * block + 1 : i * block + block, 0] = 1

        frame = human_pose.draw_right_index_pointer(frame, pose_results, size=2)
        return frame
