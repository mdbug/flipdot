import numpy as np
import pytest

import app.services.text as text_module
from app.services.fonts import get_font_definition


def test_width_empty_is_zero():
    assert text_module.width("", size=6, style="regular") == 0
    assert text_module.width("", size=6, style="monospace") == 0


def test_width_monospace_uses_fixed_cell_width():
    cell_w = get_font_definition("classic", 6, "monospace").cell_width
    assert cell_w is not None
    assert text_module.width("1", size=6, style="monospace") == cell_w
    assert text_module.width(":", size=6, style="monospace") == cell_w
    assert text_module.width("1:0", size=6, style="monospace", spacing=1) == 3 * cell_w + 2


def test_width_and_center_x_follow_regular_font_geometry():
    glyphs = get_font_definition("classic", 6, "regular").glyphs
    expected = glyphs["1"].shape[1] + glyphs["2"].shape[1] + 1

    value = "12"
    measured_width = text_module.width(value, size=6, style="regular", spacing=1)
    centered = text_module.center_x(28, value, size=6, style="regular", spacing=1)

    assert measured_width == expected
    assert centered == max(0, (28 - expected) // 2)


def test_supported_characters_contains_known_glyphs():
    chars = text_module.supported_characters(sizes=(5, 6), styles=("regular", "monospace"))
    assert "0" in chars
    assert ":" in chars
    assert "A" in chars


def test_size6_has_full_uppercase_alphabet_in_regular_style():
    chars = text_module.supported_characters(sizes=(6,), styles=("regular",))
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    present = {ch for ch in chars if "A" <= ch <= "Z"}
    assert present == expected


def test_width_supports_goal_text_in_size6_regular():
    assert text_module.width("GOAL", size=6, style="regular", spacing=1) > 0


def test_write_clips_when_text_goes_out_of_bounds():
    frame = np.zeros((3, 3), dtype=np.uint8)
    text_module.write(frame, "1", x=2, y=1, size=6, style="regular")

    assert frame.shape == (3, 3)
    assert frame.sum() >= 0


def test_unknown_style_raises_key_error():
    with pytest.raises(KeyError):
        text_module.width("1", size=6, style="bold")


def test_unsupported_character_raises_key_error():
    with pytest.raises(KeyError):
        text_module.width("@", size=6, style="regular")


def test_scoreline_family_is_size6_regular_only():
    scoreline = get_font_definition("scoreline", 6, "regular")
    assert scoreline.glyphs

    with pytest.raises(KeyError):
        get_font_definition("scoreline", 5, "regular")

    with pytest.raises(KeyError):
        get_font_definition("scoreline", 6, "monospace")


def test_scoreline_uses_monospace_digits_and_regular_letters():
    scoreline = get_font_definition("scoreline", 6, "regular").glyphs
    mono = get_font_definition("classic", 6, "monospace").glyphs
    regular = get_font_definition("classic", 6, "regular").glyphs

    for digit in "0123456789":
        assert np.array_equal(scoreline[digit], mono[digit])

    for ch in ("A", "G", ":", "°"):
        assert np.array_equal(scoreline[ch], regular[ch])


def test_classic_size6_monospace_glyphs_are_fixed_width():
    mono = get_font_definition("classic", 6, "monospace")
    assert mono.cell_width == 5
    assert all(glyph.shape[1] == mono.cell_width for glyph in mono.glyphs.values())
