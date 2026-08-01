import numpy as np
import pytest

from app.infrastructure import camera as camera_module


class _FakeCapture:
    """Stand-in for cv2.VideoCapture with scriptable read() results."""

    def __init__(self, reads):
        # Each entry is (ret, frame); the last one repeats once exhausted.
        self._reads = list(reads)
        self.released = False

    def set(self, *_args):
        return True

    def read(self):
        if len(self._reads) > 1:
            return self._reads.pop(0)
        return self._reads[0]

    def release(self):
        self.released = True


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    """Keep the reader thread and v4l2-ctl out of these tests."""
    monkeypatch.setattr(camera_module.Camera, "_apply_contrast", lambda self: None)
    # The reader thread is exercised by calling _reader's collaborators directly;
    # letting a real one run would race the assertions.
    monkeypatch.setattr(
        camera_module.threading,
        "Thread",
        lambda *a, **k: type("T", (), {"start": lambda s: None})(),
    )


def _frame(value):
    return np.full((480, 640, 3), value, dtype=np.uint8)


def test_missing_camera_does_not_raise(monkeypatch):
    """A camera that never opens must not take the app down (regression)."""
    monkeypatch.setattr(
        camera_module.cv2, "VideoCapture", lambda *a, **k: _FakeCapture([(False, None)])
    )

    cam = camera_module.Camera(camera_index=0)

    assert cam.is_available() is False
    frame = cam.read_frame()
    assert frame.shape == (480, 640, 3)
    assert not frame.any(), "an unavailable camera should hand back a black placeholder"


def test_failed_open_releases_the_capture(monkeypatch):
    captures = []

    def make(*_a, **_k):
        cap = _FakeCapture([(False, None)])
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", make)

    camera_module.Camera(camera_index=0)

    assert captures[0].released, "a capture that yields no frame should be released, not leaked"


def test_working_camera_reports_available(monkeypatch):
    monkeypatch.setattr(
        camera_module.cv2, "VideoCapture", lambda *a, **k: _FakeCapture([(True, _frame(7))])
    )

    cam = camera_module.Camera(camera_index=0)

    assert cam.is_available() is True
    assert cam.read_frame()[0, 0, 0] == 7


def test_release_blanks_the_frame_and_marks_unavailable(monkeypatch):
    monkeypatch.setattr(
        camera_module.cv2, "VideoCapture", lambda *a, **k: _FakeCapture([(True, _frame(7))])
    )
    cam = camera_module.Camera(camera_index=0)

    cam._release()

    assert cam.is_available() is False
    # A stale frame would render as a frozen mirror in the camera-fed modes.
    assert not cam.read_frame().any()


def test_camera_recovers_after_being_unplugged(monkeypatch):
    """A replugged camera is picked back up without restarting the app."""
    states = {"open": True}

    def make(*_a, **_k):
        if not states["open"]:
            return _FakeCapture([(False, None)])
        return _FakeCapture([(True, _frame(3))])

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", make)
    cam = camera_module.Camera(camera_index=0)
    assert cam.is_available() is True

    # Unplugged: the device stops delivering and the reader gives up on it.
    states["open"] = False
    cam._release()
    assert cam.is_available() is False
    assert cam._open() is False

    # Replugged: the next reopen attempt succeeds, with no restart involved.
    states["open"] = True
    assert cam._open() is True
    assert cam.is_available() is True
    assert cam.read_frame()[0, 0, 0] == 3
