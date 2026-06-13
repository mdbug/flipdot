import importlib
import sys
import types

import numpy as np


def _load_percussion_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        get_right_index_finger_position=lambda pose_results: (None, None),
        draw_right_index_pointer=lambda frame, pose_results, size=2: frame,
    )
    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    sys.modules.pop("app.modes.percussion", None)
    return importlib.import_module("app.modes.percussion")


class DummyModeManager:
    pass


def test_hit_schedules_decay_events(monkeypatch):
    percussion_module = _load_percussion_module(monkeypatch)
    monkeypatch.setattr(percussion_module.time, "time", lambda: 0.0)
    mode = percussion_module.Percussion(28, 28, DummyModeManager())

    mode._hit("tom", 1.0)

    assert len(mode._decay_events) == 2
    due_times = [event[0] for event in mode._decay_events]
    assert due_times == [1.05, 1.1]


def test_pose_controls_bpm_and_pattern(monkeypatch):
    percussion_module = _load_percussion_module(monkeypatch)
    now = {"value": 0.0}
    monkeypatch.setattr(percussion_module.time, "time", lambda: now["value"])
    monkeypatch.setattr(
        percussion_module.human_pose,
        "get_right_index_finger_position",
        lambda pose_results: (0.0, 0.75),
    )

    mode = percussion_module.Percussion(28, 28, DummyModeManager())
    frame = mode.get_frame(None)

    assert isinstance(frame, np.ndarray)
    assert mode.bpm == mode.MAX_BPM
    assert mode.pattern_index == 3


def test_due_decay_events_are_consumed(monkeypatch):
    percussion_module = _load_percussion_module(monkeypatch)
    now = {"value": 0.0}
    monkeypatch.setattr(percussion_module.time, "time", lambda: now["value"])
    mode = percussion_module.Percussion(28, 28, DummyModeManager())
    mode._decay_events = [(0.5, "kick", 1.0)]

    now["value"] = 1.0
    mode.get_frame(None)

    assert mode._decay_events == []


def test_bottom_row_step_cursor_updates(monkeypatch):
    percussion_module = _load_percussion_module(monkeypatch)
    monkeypatch.setattr(percussion_module.time, "time", lambda: 0.0)
    mode = percussion_module.Percussion(28, 28, DummyModeManager())

    frame = mode.get_frame(None)
    seg = mode.width // len(mode.PATTERNS[mode.pattern_index])

    assert (frame[-1, seg : 2 * seg] == 1).all()
