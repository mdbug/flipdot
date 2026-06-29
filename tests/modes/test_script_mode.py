import random

import numpy as np
import pytest

from app.modes.script_mode import ScriptMode

_VALID_SCRIPT = "def step(state, t, width, height):\n    return None, np.zeros((height, width))\n"


class _FakeStore:
    """In-memory stand-in for ScriptStore exposing a mutable ``list_names``."""

    def __init__(self, names):
        self.names = list(names)

    def list_names(self):
        return list(self.names)


def test_run_script_rejects_invalid_name_before_starting_worker():
    # An invalid save name must be caught up front so no sandbox worker is
    # spawned and left running after the call raises.
    mode = ScriptMode(28, 28)
    with pytest.raises(ValueError):
        mode.run_script(_VALID_SCRIPT, name="not a valid name!")
    assert mode._script is None  # nothing was started or left active


def test_error_frame_renders_message_without_raising():
    mode = ScriptMode(28, 28)
    # A long message with characters the panel font may not support must still
    # produce a valid frame (it is rendered/scrolled, not silently dropped).
    frame = mode._error_frame("ValueError: frame shape (5, 5) must be (28, 28)")
    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert frame.sum() > 0  # at least the "ERROR" header is lit


def test_error_frame_handles_empty_message():
    mode = ScriptMode(28, 28)
    frame = mode._error_frame("")
    assert frame.shape == (28, 28)
    assert frame.sum() > 0  # bare "ERROR" still shown


def test_get_frame_blank_when_no_script():
    mode = ScriptMode(28, 28)
    frame = mode.get_frame()
    assert frame.shape == (28, 28)
    assert frame.sum() == 0


def test_start_next_false_on_empty_store():
    mode = ScriptMode(28, 28, store=_FakeStore([]))
    assert mode.start_next() is False


def test_start_next_repeats_daily_order(monkeypatch):
    # Two full passes use the identical order, so every script recurs exactly
    # every N plays (uniform gap = library size).
    random.seed(0)
    names = ["a", "b", "c", "d"]
    mode = ScriptMode(28, 28, store=_FakeStore(names))
    played: list[str] = []
    monkeypatch.setattr(mode, "load_script", lambda name: played.append(name))

    for _ in range(8):
        assert mode.start_next() is True

    assert sorted(played[:4]) == names  # whole library before any repeat
    assert played[:4] == played[4:8]  # same order repeated through the day


def test_start_next_drops_deleted_script(monkeypatch):
    random.seed(1)
    store = _FakeStore(["a", "b", "c"])
    mode = ScriptMode(28, 28, store=store)
    played: list[str] = []
    monkeypatch.setattr(mode, "load_script", lambda name: played.append(name))

    mode.reshuffle_day()
    store.names.remove("b")  # deleted before it gets played
    for _ in range(4):
        mode.start_next()

    assert "b" not in played


def test_start_next_includes_script_saved_midpass(monkeypatch):
    random.seed(2)
    store = _FakeStore(["a", "b", "c"])
    mode = ScriptMode(28, 28, store=store)
    played: list[str] = []
    monkeypatch.setattr(mode, "load_script", lambda name: played.append(name))

    mode.start_next()  # builds the day's order over a, b, c
    store.names.append("d")  # saved mid-pass
    for _ in range(4):
        mode.start_next()

    assert "d" in played  # spliced into the remaining order, played the same day


def test_reshuffle_day_sets_full_order():
    random.seed(3)
    names = ["a", "b", "c", "d"]
    mode = ScriptMode(28, 28, store=_FakeStore(names))
    mode.reshuffle_day()

    assert sorted(mode._order) == names
    assert mode._queue == mode._order


def test_excluded_scripts_never_play(monkeypatch):
    random.seed(4)
    mode = ScriptMode(28, 28, store=_FakeStore(["a", "b", "c"]))
    played: list[str] = []
    monkeypatch.setattr(mode, "load_script", lambda name: played.append(name))

    mode.update_interlude_settings(excluded=["b"])
    for _ in range(6):
        mode.start_next()

    assert "b" not in played
    assert set(played) == {"a", "c"}


def test_excluding_active_script_drops_it_from_remaining_order(monkeypatch):
    random.seed(5)
    mode = ScriptMode(28, 28, store=_FakeStore(["a", "b", "c"]))
    played: list[str] = []
    monkeypatch.setattr(mode, "load_script", lambda name: played.append(name))

    mode.start_next()  # builds the order over a, b, c
    mode.update_interlude_settings(excluded=["a", "b", "c"])
    # With everything excluded, there is nothing eligible to start.
    assert mode.start_next() is False


def test_interlude_settings_round_trip():
    mode = ScriptMode(28, 28, store=_FakeStore(["a", "b"]))
    assert mode.get_interlude_settings() == {"excluded": []}

    result = mode.update_interlude_settings(excluded=["b", "b"])
    assert result == {"excluded": ["b"]}
    assert mode.list_scripts()["excluded"] == ["b"]
