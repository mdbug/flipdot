from app.services.settings_store import RuntimeSettingsStore


def test_clock_settings_round_trip(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "settings.json")

    assert store.load_clock_settings() is None

    store.save_clock_settings(style="analog")
    assert store.load_clock_settings() == {"style": "analog", "seconds": False}

    store.save_clock_settings(style="digital")
    assert store.load_clock_settings() == {"style": "digital", "seconds": False}

    store.save_clock_settings(style="analog", seconds=True)
    assert store.load_clock_settings() == {"style": "analog", "seconds": True}


def test_clock_settings_reject_invalid_style(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "settings.json")

    # Saving an unknown style falls back to digital.
    store.save_clock_settings(style="bogus")
    assert store.load_clock_settings() == {"style": "digital", "seconds": False}


def test_clock_settings_loaded_invalid_falls_back(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"clock": {"style": "wobble"}}', encoding="utf-8")

    store = RuntimeSettingsStore(settings_path)
    assert store.load_clock_settings() == {"style": "digital", "seconds": False}


def test_clock_settings_preserves_other_sections(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    store.save_sleep_settings(enabled=True, start_hour=22, end_hour=6)
    store.save_clock_settings(style="analog")

    assert store.load_sleep_settings() == {
        "enabled": True,
        "start_hour": 22,
        "end_hour": 6,
    }
    assert store.load_clock_settings() == {"style": "analog", "seconds": False}


def test_script_settings_round_trip(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "settings.json")

    assert store.load_script_settings() is None

    # Stored sorted and de-duplicated.
    store.save_script_settings(excluded=["birthday", "birthday", "alarm"])
    assert store.load_script_settings() == {"excluded": ["alarm", "birthday"]}

    store.save_script_settings(excluded=[])
    assert store.load_script_settings() == {"excluded": []}


def test_script_settings_loaded_invalid_returns_none(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"scripts": {"excluded": "nope"}}', encoding="utf-8")

    store = RuntimeSettingsStore(settings_path)
    assert store.load_script_settings() is None
