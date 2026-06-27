import numpy as np
import pytest

from app.modes.script_mode import ScriptMode

_VALID_SCRIPT = "def step(state, t, width, height):\n    return None, np.zeros((height, width))\n"


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
