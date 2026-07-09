import app.services.fps as fps_module


class FakeClock:
    def __init__(self, start=0.0):
        self.now = float(start)

    def time(self):
        return self.now


def test_get_fps_is_zero_with_fewer_than_two_frames(monkeypatch):
    fake = FakeClock(start=0.0)
    monkeypatch.setattr(fps_module.time, "time", fake.time)
    tracker = fps_module.FPSTracker(window_size=5)

    tracker.add_frame()
    assert tracker.get_fps() == 0


def test_get_fps_and_average_fps(monkeypatch):
    fake = FakeClock(start=0.0)
    monkeypatch.setattr(fps_module.time, "time", fake.time)
    tracker = fps_module.FPSTracker(window_size=10)

    for t in (0.0, 0.1, 0.2, 0.3):
        fake.now = t
        tracker.add_frame()

    assert tracker.get_fps() == 10

    fake.now = 2.0
    assert tracker.get_average_fps() == 2.0


def test_timing_stats_are_reported_in_milliseconds(monkeypatch):
    fake = FakeClock(start=1.0)
    monkeypatch.setattr(fps_module.time, "time", fake.time)
    tracker = fps_module.FPSTracker(window_size=10)

    tracker.add_frame(capture_time=0.010, process_time=0.020, panel_time=0.003, sleep_time=0.005)
    tracker.add_frame(capture_time=0.030, process_time=0.040, panel_time=0.001, sleep_time=0.007)

    stats = tracker.get_timing_stats()

    assert round(stats["capture_ms"], 3) == 20.0
    assert round(stats["process_ms"], 3) == 30.0
    assert round(stats["panel_ms"], 3) == 2.0
    assert round(stats["sleep_ms"], 3) == 6.0
    # Total covers every stage: capture + process + panel + sleep.
    assert round(stats["total_ms"], 3) == 58.0
