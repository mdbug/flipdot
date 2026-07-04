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
    server = _make_server(9302, settings_path)
    manager = ModeManager()
    server.attach_mode_manager(manager)
    client = TestClient(server._app)

    res = client.post("/api/settings/pose", json={"enabled": False})

    assert res.status_code == 200
    assert res.json() == {"status": "ok", "enabled": False}
    assert manager.pose_enabled is False
    store = RuntimeSettingsStore(settings_path)
    assert store.load_pose_settings() == {"enabled": False}


def test_attach_mode_manager_applies_persisted_pose_setting(tmp_path):
    settings_path = tmp_path / "settings.json"
    RuntimeSettingsStore(settings_path).save_pose_settings(enabled=False)
    server = _make_server(9303, settings_path)
    manager = ModeManager()

    server.attach_mode_manager(manager)

    assert manager.pose_enabled is False
