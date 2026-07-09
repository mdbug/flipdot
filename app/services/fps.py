import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class FPSTracker:
    """Rolling-window frame-rate and per-stage timing tracker for the main loop."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.times: deque[float] = deque(maxlen=window_size)
        self.capture_times: deque[float] = deque(maxlen=window_size)
        self.process_times: deque[float] = deque(maxlen=window_size)
        self.panel_times: deque[float] = deque(maxlen=window_size)
        self.sleep_times: deque[float] = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()

    def add_frame(
        self,
        capture_time: float = 0,
        process_time: float = 0,
        panel_time: float = 0,
        sleep_time: float = 0,
    ) -> None:
        """Record one frame's wall-clock time and per-stage durations (seconds)."""
        current_time = time.time()
        self.times.append(current_time)
        self.capture_times.append(capture_time)
        self.process_times.append(process_time)
        self.panel_times.append(panel_time)
        self.sleep_times.append(sleep_time)
        self.total_frames += 1

    def get_fps(self) -> float:
        """Return the instantaneous FPS over the current window (0 if too few frames)."""
        if len(self.times) < 2:
            return 0
        time_diff = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / time_diff if time_diff > 0 else 0

    def get_average_fps(self) -> float:
        """Return the average FPS since the tracker was created."""
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0

    def get_timing_stats(self) -> dict[str, float]:
        """Return mean per-stage timings (ms) over the window, or empty if no frames yet."""
        if not self.capture_times:
            return {}
        return {
            "capture_ms": float(np.mean(self.capture_times)) * 1000,
            "process_ms": float(np.mean(self.process_times)) * 1000,
            "panel_ms": float(np.mean(self.panel_times)) * 1000,
            "sleep_ms": float(np.mean(self.sleep_times)) * 1000,
            "total_ms": float(
                np.mean(self.capture_times)
                + np.mean(self.process_times)
                + np.mean(self.panel_times)
                + np.mean(self.sleep_times)
            )
            * 1000,
        }

    def get_stats(self) -> dict[str, float]:
        """Return timing stats plus instantaneous and average FPS."""
        stats = self.get_timing_stats()
        stats["fps"] = self.get_fps()
        stats["avg_fps"] = self.get_average_fps()
        return stats

    def print_stats(self, last_print_time: float) -> None:
        """Log a one-line FPS/timing summary at INFO level."""
        fps = self.get_fps()
        avg_fps = self.get_average_fps()
        stats = self.get_timing_stats()
        logger.info(
            "fps=%.1f avg_fps=%.1f cap_ms=%.1f proc_ms=%.1f total_ms=%.1f",
            fps,
            avg_fps,
            stats.get("capture_ms", 0),
            stats.get("process_ms", 0),
            stats.get("total_ms", 0),
        )
