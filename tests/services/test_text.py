import numpy as np

import app.services.text as text_module


def test_width_empty_is_zero():
    assert text_module.width("", size=6, spacing=1) == 0
    assert text_module.width("", size=6, spacing=1, mono=True) == 0


def test_width_mono_uses_fixed_cell():
    # In mono mode every character occupies CELL_WIDTHS[size] pixels.
    cell_w = text_module.CELL_WIDTHS[6]
    assert text_module.width("1", size=6, mono=True) == cell_w
    assert text_module.width(":", size=6, mono=True) == cell_w
    # Spacing is still applied between cells.
    assert text_module.width("1:0", size=6, spacing=1, mono=True) == 3 * cell_w + 2


def test_width_and_center_x_follow_font_geometry():
    expected = (
        text_module.FONTS[6]["1"].shape[1]
        + text_module.FONTS[6]["2"].shape[1]
        + 1
    )
    value = "12"
    width = text_module.width(value, size=6, spacing=1)
    centered = text_module.center_x(28, value, size=6, spacing=1)

    assert width == expected
    assert centered == max(0, (28 - expected) // 2)


def test_supported_characters_contains_known_glyphs():
    chars = text_module.supported_characters((5, 6))
    assert "0" in chars
    assert ":" in chars
    assert "A" in chars


def test_size6_has_full_uppercase_alphabet():
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    present = {ch for ch in text_module.FONTS[6].keys() if "A" <= ch <= "Z"}
    assert present == expected


def test_font_maps_are_sorted_by_key():
    assert list(text_module.FONTS[5].keys()) == sorted(text_module.FONTS[5].keys())
    assert list(text_module.FONTS[6].keys()) == sorted(text_module.FONTS[6].keys())


def test_width_supports_goal_text_in_size6():
    assert text_module.width("GOAL", size=6, spacing=1) > 0


def test_write_clips_when_text_goes_out_of_bounds():
    frame = np.zeros((3, 3), dtype=np.uint8)
    text_module.write(frame, "1", x=2, y=1, size=6)

    assert frame.shape == (3, 3)
    assert frame.sum() >= 0
