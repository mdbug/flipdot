import numpy as np

from app.modes.font_preview import FontPreview


class DummyPointer:
    def __init__(self, source, x, y):
        self.source = source
        self.x = x
        self.y = y


class DummyInputHub:
    def __init__(self, pointer=None):
        self._pointer = pointer

    def get_active_pointer(self, max_age_sec=0.8):
        return self._pointer


def test_get_frame_returns_panel_sized_uint8():
    mode = FontPreview(28, 28, mode_manager=object())

    frame = mode.get_frame(pose_results=None, input_hub=None)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8


def test_next_and_previous_variant_wraps():
    mode = FontPreview(28, 28, mode_manager=object())

    mode._configured_variants = list(mode._variants[:4])
    start = mode._window_start
    mode.next_variant()
    assert mode._window_start == start

    mode._configured_variants = list(mode._variants[:5])
    mode.next_variant()
    assert mode._window_start == 1
    mode.previous_variant()
    assert mode._window_start == 0


def test_update_settings_normalizes_phrase():
    mode = FontPreview(28, 28, mode_manager=object())

    settings = mode.update_settings(phrase="   HELLO    FLIPDOT   ")

    assert settings["phrase"] == "HELLO FLIPDOT"
    assert settings["spacing"] == 0
    assert isinstance(settings["variants"], list)


def test_update_settings_limits_variants_to_four():
    mode = FontPreview(28, 28, mode_manager=object())

    variants = [
        {"family": family, "size": size, "style": style}
        for family, size, style in mode._variants[:6]
    ]
    settings = mode.update_settings(phrase="FLIPDOT", variants=variants)

    assert len(settings["variants"]) == 4


def test_update_settings_clamps_spacing():
    mode = FontPreview(28, 28, mode_manager=object())

    settings = mode.update_settings(phrase="FLIPDOT", spacing=99)

    assert settings["spacing"] == 6


def test_hover_zone_prev_triggers_variant_change(monkeypatch):
    mode = FontPreview(28, 28, mode_manager=object())
    mode._configured_variants = list(mode._variants[:5])
    mode._window_start = 2
    before = mode._window_start

    fake_time = {"value": 10.0}
    monkeypatch.setattr("app.modes.font_preview.time.time", lambda: fake_time["value"])

    # x=0.05 and y=0.5 should map to the left third in web pointer space.
    hub = DummyInputHub(pointer=DummyPointer(source="web", x=0.05, y=0.5))
    mode.get_frame(input_hub=hub)

    fake_time["value"] = 11.0
    mode.get_frame(input_hub=hub)

    assert mode._window_start == (before - 1) % len(mode._configured_variants)
