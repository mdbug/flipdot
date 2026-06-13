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
    pass


class DummyInputHub:
    def __init__(self, pointer, down):
        self.pointer = pointer
        self.down = down

    def get_active_pointer(self, max_age_sec=0.8):
        return self.pointer

    def is_button_down(self, source="web"):
        return self.down


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
