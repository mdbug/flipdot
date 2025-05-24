from collections import deque
import numpy as np
import time

class FPSTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.capture_times = deque(maxlen=window_size)
        self.process_times = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()
    
    def add_frame(self, capture_time=0, process_time=0):
        current_time = time.time()
        self.times.append(current_time)
        self.capture_times.append(capture_time)
        self.process_times.append(process_time)
        self.total_frames += 1
    
    def get_fps(self):
        if len(self.times) < 2:
            return 0
        time_diff = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / time_diff if time_diff > 0 else 0
    
    def get_average_fps(self):
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0
    
    def get_timing_stats(self):
        if not self.capture_times:
            return {}
        return {
            'capture_ms': np.mean(self.capture_times) * 1000,
            'process_ms': np.mean(self.process_times) * 1000,
            'total_ms': (np.mean(self.capture_times) + 
                        np.mean(self.process_times)) * 1000
        }

    def get_stats(self):
        stats = self.get_timing_stats()
        stats['fps'] = self.get_fps()
        stats['avg_fps'] = self.get_average_fps()
        return stats

    def print_stats(self, last_print_time):
        fps = self.get_fps()
        avg_fps = self.get_average_fps()
        stats = self.get_timing_stats()
        print(f"FPS: {fps:.1f} | Avg: {avg_fps:.1f} | "
                f"Cap: {stats.get('capture_ms', 0):.1f}ms | "
                f"Proc: {stats.get('process_ms', 0):.1f}ms | "
                f"Total: {stats.get('total_ms', 0):.1f}ms | ")
            