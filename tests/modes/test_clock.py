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
