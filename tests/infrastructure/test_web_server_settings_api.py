"""Tests for the pose (person-detection) settings API."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.core.mode_manager import ModeManager  # noqa: E402
from app.infrastructure.web_server import WebServer  # noqa: E402
from app.services.settings_store import RuntimeSettingsStore  # noqa: E402


class DummyInputHub:
    def submit_pointer(self, source, x, y):
        pass

    def submit_click(self, source, x, y):
        pass

    def submit_action(self, source, action):
        pass

    def set_button_down(self, source, is_down):
        pass


def _make_server(port, settings_path):
    return WebServer(
        input_hub=DummyInputHub(), host="127.0.0.1", port=port, settings_path=settings_path
    )


def test_pose_settings_require_mode_manager(tmp_path):
    server = _make_server(9300, tmp_path / "settings.json")
    client = TestClient(server._app)

    assert client.get("/api/settings/pose").status_code == 409
    assert client.post("/api/settings/pose", json={"enabled": False}).status_code == 409


def test_get_pose_settings_reflects_mode_manager(tmp_path):
    server = _make_server(9301, tmp_path / "settings.json")
    manager = ModeManager()
    manager.pose_enabled = False
    server.attach_mode_manager(manager)
    client = TestClient(server._app)

    res = client.get("/api/settings/pose")

    assert res.status_code == 200
    assert res.json() == {"enabled": False}


def test_post_pose_settings_applies_and_persists(tmp_path):
    settings_path = tmp_path / "settings.json"
    store = RuntimeSettingsStore(settings_path)
    server = _make_server(9302, settings_path)
    manager = ModeManager()
    # Persistence is the on_pose_enabled_changed hook's job (one path for
    # every source); mirror the main loop's wiring.
    manager.on_pose_enabled_changed = lambda enabled: store.save_pose_settings(enabled=enabled)
    server.attach_mode_manager(manager)
    client = TestClient(server._app)

    res = client.post("/api/settings/pose", json={"enabled": False})

    assert res.status_code == 200
    assert res.json() == {"status": "ok", "enabled": False}
    assert manager.pose_enabled is False
    assert store.load_pose_settings() == {"enabled": False}


def test_post_pose_settings_reports_persist_failure(tmp_path):
    # The live toggle applies, but a failed settings write must surface as a
    # 500 — a silent 200 would revert the user's choice on the next restart.
    server = _make_server(9305, tmp_path / "settings.json")
    manager = ModeManager()

    def broken_hook(enabled):
        raise OSError("disk full")

    manager.on_pose_enabled_changed = broken_hook
    server.attach_mode_manager(manager)
    client = TestClient(server._app)

    res = client.post("/api/settings/pose", json={"enabled": False})

    assert res.status_code == 500
    assert manager.pose_enabled is False  # applied live despite the failure


def test_web_server_uses_injected_settings_store(tmp_path):
    # A shared store instance is what serializes writes against the main
    # loop's persistence hook (the store's lock is per-instance).
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=9304, settings_store=store)

    assert server._settings_store is store


def test_web_server_rejects_both_settings_path_and_store(tmp_path):
    # A silently ignored settings_path would recreate the split-store race
    # the injected store exists to prevent.
    store = RuntimeSettingsStore(tmp_path / "settings.json")

    with pytest.raises(ValueError):
        WebServer(
            input_hub=DummyInputHub(),
            host="127.0.0.1",
            port=9306,
            settings_path=tmp_path / "other.json",
            settings_store=store,
        )


def test_attach_mode_manager_does_not_clobber_live_pose_setting(tmp_path):
    # The persisted toggle is applied at startup by the main loop; attaching
    # the (lazily started) web server must not overwrite a live value that
    # was changed since — e.g. via the panel menu.
    settings_path = tmp_path / "settings.json"
    RuntimeSettingsStore(settings_path).save_pose_settings(enabled=False)
    server = _make_server(9303, settings_path)
    manager = ModeManager()
    manager.pose_enabled = True

    server.attach_mode_manager(manager)

    assert manager.pose_enabled is True
