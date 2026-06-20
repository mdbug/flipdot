import importlib
import sys
import types

import numpy as np


def _load_paint_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        get_right_index_finger_position=lambda pose_results: (None, None),
        draw_right_index_pointer=lambda frame, pose_results, size=2: frame,
        draw_pointer=lambda frame, x, y, size=2, mirror_x=False: frame,
    )
    cv2_stub = types.SimpleNamespace(line=lambda canvas, p0, p1, color, thickness=1: canvas)

    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setitem(sys.modules, "cv2", cv2_stub)
    sys.modules.pop("app.modes.paint", None)
    return importlib.import_module("app.modes.paint")


class DummyModeManager:
    def __init__(self):
        self.allowed_sources = {"pose", "controller", "web"}

    def get_allowed_input_sources(self, include_web=True):
        if include_web:
            return set(self.allowed_sources)
        return {source for source in self.allowed_sources if source != "web"}


class DummyInputHub:
    def __init__(self, pointer, down):
        self.pointer = pointer
        self.down = down

    def get_active_pointer(self, max_age_sec=0.8, allowed_sources=None):
        if self.pointer is None:
            return None
        if allowed_sources is not None and self.pointer.source not in allowed_sources:
            return None
        return self.pointer

    def is_button_down(self, source="web"):
        return self.down if source in ("web", "controller") else False


def test_pointer_to_pixel_mirrors_pose_but_not_web(monkeypatch):
    paint_module = _load_paint_module(monkeypatch)
    paint = paint_module.Paint(28, 28, DummyModeManager())

    pose_x, pose_y = paint._pointer_to_pixel("pose", 0.1, 0.2)
    web_x, web_y = paint._pointer_to_pixel("web", 0.1, 0.2)

    assert pose_x > web_x
    assert pose_y == web_y


def test_web_pointer_draws_line_when_button_held(monkeypatch):
    paint_module = _load_paint_module(monkeypatch)
    paint = paint_module.Paint(28, 28, DummyModeManager())

    line_calls = {"count": 0}

    def fake_line(canvas, p0, p1, color, thickness=1):
        line_calls["count"] += 1
        return canvas

    monkeypatch.setattr(paint_module.cv2, "line", fake_line)

    hub1 = DummyInputHub(pointer=types.SimpleNamespace(source="web", x=0.1, y=0.1), down=True)
    hub2 = DummyInputHub(pointer=types.SimpleNamespace(source="web", x=0.2, y=0.2), down=True)

    paint.get_frame(None, input_hub=hub1)
    paint.get_frame(None, input_hub=hub2)

    assert line_calls["count"] == 1


def test_pose_hold_toggles_drawing(monkeypatch):
    paint_module = _load_paint_module(monkeypatch)
    monkeypatch.setattr(
        paint_module.human_pose,
        "get_right_index_finger_position",
        lambda pose_results: (0.5, 0.5),
    )
    paint = paint_module.Paint(28, 28, DummyModeManager())

    for _ in range(paint.CLICK_TIME + 1):
        paint.get_frame(None)

    assert paint.drawing is True


def test_pose_hold_on_top_right_clears_canvas(monkeypatch):
    paint_module = _load_paint_module(monkeypatch)
    monkeypatch.setattr(
        paint_module.human_pose,
        "get_right_index_finger_position",
        lambda pose_results: (0.03, 0.0),
    )
    paint = paint_module.Paint(28, 28, DummyModeManager())
    paint.canvas[:, :] = 1
    paint.drawing = True

    for _ in range(paint.CLICK_TIME + 1):
        paint.get_frame(None)

    assert paint.canvas.sum() == 0
    assert paint.drawing is False


def test_controller_pointer_is_ignored_when_gesture_controls_mode(monkeypatch):
    paint_module = _load_paint_module(monkeypatch)
    manager = DummyModeManager()
    manager.allowed_sources = {"pose", "web"}
    paint = paint_module.Paint(28, 28, manager)

    line_calls = {"count": 0}

    def fake_line(canvas, p0, p1, color, thickness=1):
        line_calls["count"] += 1
        return canvas

    monkeypatch.setattr(paint_module.cv2, "line", fake_line)

    hub1 = DummyInputHub(pointer=types.SimpleNamespace(source="controller", x=0.1, y=0.1), down=True)
    hub2 = DummyInputHub(pointer=types.SimpleNamespace(source="controller", x=0.2, y=0.2), down=True)

    paint.get_frame(None, input_hub=hub1)
    paint.get_frame(None, input_hub=hub2)

    assert line_calls["count"] == 0
