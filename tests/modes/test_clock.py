import importlib
import sys
import types


def _load_clock_module(monkeypatch):
    weather_stub = types.SimpleNamespace(
        get_weather_forecast=lambda: {
            "current_temperature": 20,
            "max_temperature_today": 25,
            "hourly_rain_forecast": [],
        }
    )
    monkeypatch.setitem(sys.modules, "app.services.weather", weather_stub)
    sys.modules.pop("app.modes.clock", None)
    return importlib.import_module("app.modes.clock")


class FakeDateTime:
    @classmethod
    def now(cls):
        class Now:
            hour = 5

            @staticmethod
            def strftime(fmt):
                if fmt == "%d.%m.%y":
                    return "13.06.26"
                if fmt == "%H:%M":
                    return "05:42"
                raise AssertionError("unexpected format")

        return Now()


def test_get_weather_uses_hourly_cache(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    now = {"value": 0.0}

    def fake_time():
        return now["value"]

    calls = {"count": 0}

    def fake_weather():
        calls["count"] += 1
        return {
            "current_temperature": 20,
            "max_temperature_today": 25,
            "hourly_rain_forecast": [],
        }

    monkeypatch.setattr(clock_module.time, "time", fake_time)
    monkeypatch.setattr(clock_module, "get_weather_forecast", fake_weather)

    clock = clock_module.Clock(width=28, height=28)
    now["value"] = 0.1
    clock.get_weather()
    assert calls["count"] == 1

    now["value"] = 120.0
    clock.get_weather()
    assert calls["count"] == 1

    now["value"] = 3601.0
    clock.get_weather()
    assert calls["count"] == 2


def test_get_frame_respects_clock_interval(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    now = {"value": 0.0}

    def fake_time():
        return now["value"]

    monkeypatch.setattr(clock_module.time, "time", fake_time)
    clock = clock_module.Clock(width=28, height=28)

    calls = {"count": 0}

    def fake_update():
        calls["count"] += 1
        clock.last_frame_update = now["value"]

    monkeypatch.setattr(clock, "update_frame", fake_update)

    now["value"] = 0.1
    clock.get_frame()
    assert calls["count"] == 1

    now["value"] = 0.5
    clock.get_frame()
    assert calls["count"] == 1

    now["value"] = 1.2
    clock.get_frame()
    assert calls["count"] == 2


def test_update_frame_draws_hour_progress_bar(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    monkeypatch.setattr(clock_module, "datetime", FakeDateTime)
    monkeypatch.setattr(clock_module.Clock, "get_weather", lambda self: None)

    clock = clock_module.Clock(width=28, height=28)
    clock.update_frame()

    assert (clock.frame[25, 1:27] == 1).all()
    assert (clock.frame[27, 1:27] == 1).all()
    assert (clock.frame[26, 2 : 2 + 5] == 1).all()


def test_update_settings_validates_style(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    clock = clock_module.Clock(width=28, height=28)

    assert clock.get_settings() == {"style": "digital", "seconds": False}

    assert clock.update_settings(style="analog") == {"style": "analog", "seconds": False}
    assert clock.style == "analog"

    # Unknown styles are ignored, leaving the previous value intact.
    assert clock.update_settings(style="bogus") == {"style": "analog", "seconds": False}
    assert clock.style == "analog"


def test_update_settings_toggles_second_hand(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    clock = clock_module.Clock(width=28, height=28)

    assert clock.seconds is False
    assert clock.update_settings(style="analog", seconds=True) == {
        "style": "analog",
        "seconds": True,
    }
    assert clock.seconds is True

    # Omitting ``seconds`` leaves the toggle untouched.
    assert clock.update_settings(style="digital")["seconds"] is True
    assert clock.update_settings(style="digital", seconds=False)["seconds"] is False


def test_second_hand_only_rendered_when_enabled(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    monkeypatch.setattr(clock_module, "datetime", FakeNoonDateTime)
    monkeypatch.setattr(clock_module.Clock, "get_weather", lambda self: None)

    clock = clock_module.Clock(width=28, height=28)
    clock.update_settings(style="analog")
    clock.update_frame()
    without_seconds = clock.frame.copy()

    clock.update_settings(style="analog", seconds=True)
    clock.update_frame()
    with_seconds = clock.frame.copy()

    # At 12:00:30 the second hand sweeps straight down, carving dark pixels the
    # second-less face leaves white in the lower half of the dial.
    assert int(without_seconds.sum()) > int(with_seconds.sum())


def test_analog_render_lights_pixels(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    monkeypatch.setattr(clock_module.Clock, "get_weather", lambda self: None)

    clock = clock_module.Clock(width=28, height=28)
    clock.update_settings(style="analog")
    clock.update_frame()

    frame = clock.frame
    assert frame.shape == (28, 28)
    assert frame.dtype.name == "uint8"
    # The white face should light a large block of pixels.
    assert int(frame.sum()) > 200
    # Black ticks/hands carve dark pixels out of the white face interior.
    assert (frame[8:20, 8:20] == 0).any()
    # The dial touches the left/top edges and leaves a 1px margin right/bottom.
    assert int(frame[:, 0].sum()) > 0
    assert int(frame[0, :].sum()) > 0
    assert int(frame[:, 27].sum()) == 0
    assert int(frame[27, :].sum()) == 0


class FakeNoonDateTime:
    @classmethod
    def now(cls):
        class Now:
            hour = 12
            minute = 0
            second = 30

            @staticmethod
            def strftime(fmt):
                raise AssertionError("strftime not used by the analog face")

        return Now()


def test_analog_face_is_mirror_symmetric_at_noon(monkeypatch):
    clock_module = _load_clock_module(monkeypatch)
    monkeypatch.setattr(clock_module, "datetime", FakeNoonDateTime)
    monkeypatch.setattr(clock_module.Clock, "get_weather", lambda self: None)

    clock = clock_module.Clock(width=28, height=28)
    clock.update_settings(style="analog")
    clock.update_frame()

    # At noon both hands point straight up, so the whole dial is symmetric about
    # the integer center column (13). Compare mirrored column pairs.
    frame = clock.frame
    for k in range(1, 13):
        assert (frame[:, 13 - k] == frame[:, 13 + k]).all()
