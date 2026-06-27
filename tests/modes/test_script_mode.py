import numpy as np

from app.modes.script_mode import ScriptMode


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
