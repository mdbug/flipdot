import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.infrastructure.web_server import WebServer
from app.services.settings_store import RuntimeSettingsStore


class DummyInputHub:
    def submit_pointer(self, source, x, y):
        pass

    def submit_click(self, source, x, y):
        pass

    def submit_action(self, source, action):
        pass

    def set_button_down(self, source, is_down):
        pass


class DummyScriptMode:
    def __init__(self, scripts=None, active="", code_map=None):
        self._scripts = scripts or []
        self._active = active
        self._code_map = code_map or {}
        self._excluded = set()
        self.loaded = None
        self.deleted = None

    def list_scripts(self):
        return {
            "scripts": self._scripts,
            "active": self._active,
            "excluded": sorted(self._excluded),
        }

    def get_interlude_settings(self):
        return {"excluded": sorted(self._excluded)}

    def update_interlude_settings(self, *, excluded):
        self._excluded = {str(name) for name in excluded}
        return self.get_interlude_settings()

    def get_code(self, name):
        return self._code_map.get(name)

    def load_script(self, name):
        if name not in self._scripts:
            raise ValueError(f"script '{name}' not found")
        self.loaded = name
        self._active = name
        return {"running": True, "name": name}

    def delete_script(self, name):
        if name not in self._scripts:
            return False
        self._scripts.remove(name)
        self.deleted = name
        if self._active == name:
            self._active = ""
        return True


class DummyModeManager:
    def __init__(self):
        self.mode = None

    def set_mode(self, mode):
        self.mode = mode


def _make_server(port):
    return WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=port)


def test_scripts_endpoint_requires_attachment():
    server = _make_server(9200)
    client = TestClient(server._app)

    assert client.get("/api/scripts").status_code == 409
    assert client.get("/api/scripts/foo/code").status_code == 409
    assert client.post("/api/scripts/foo/play").status_code == 409
    assert client.delete("/api/scripts/foo").status_code == 409


def test_list_scripts_returns_names():
    server = _make_server(9201)
    server.attach_script_mode(DummyScriptMode(scripts=["wave", "rain"], active="wave"))
    client = TestClient(server._app)

    res = client.get("/api/scripts")
    assert res.status_code == 200
    data = res.json()
    assert data["scripts"] == ["wave", "rain"]
    assert data["active"] == "wave"


def test_get_script_code_found():
    script_mode = DummyScriptMode(
        scripts=["wave"],
        code_map={"wave": "def step(state, t, w, h):\n    pass"},
    )
    server = _make_server(9202)
    server.attach_script_mode(script_mode)
    client = TestClient(server._app)

    res = client.get("/api/scripts/wave/code")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "wave"
    assert "step" in data["code"]


def test_get_script_code_not_found():
    server = _make_server(9203)
    server.attach_script_mode(DummyScriptMode())
    client = TestClient(server._app)

    res = client.get("/api/scripts/missing/code")
    assert res.status_code == 404


def test_play_calls_load_and_switches_mode():
    script_mode = DummyScriptMode(scripts=["rain"])
    mode_manager = DummyModeManager()
    server = _make_server(9204)
    server.attach_script_mode(script_mode)
    server.attach_mode_manager(mode_manager)
    client = TestClient(server._app)

    res = client.post("/api/scripts/rain/play")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["running"] is True
    assert script_mode.loaded == "rain"
    assert mode_manager.mode == "script"


def test_play_400_on_value_error():
    server = _make_server(9205)
    server.attach_script_mode(DummyScriptMode(scripts=[]))
    client = TestClient(server._app)

    res = client.post("/api/scripts/ghost/play")
    assert res.status_code == 400
    assert "not found" in res.json()["detail"]


def test_delete_success():
    script_mode = DummyScriptMode(scripts=["wave", "rain"])
    server = _make_server(9206)
    server.attach_script_mode(script_mode)
    client = TestClient(server._app)

    res = client.delete("/api/scripts/wave")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "deleted": True}
    assert "wave" not in script_mode._scripts


def test_delete_not_found():
    server = _make_server(9207)
    server.attach_script_mode(DummyScriptMode(scripts=[]))
    client = TestClient(server._app)

    res = client.delete("/api/scripts/ghost")
    assert res.status_code == 404


def test_list_scripts_includes_excluded():
    script_mode = DummyScriptMode(scripts=["wave", "birthday"])
    script_mode.update_interlude_settings(excluded=["birthday"])
    server = _make_server(9209)
    server.attach_script_mode(script_mode)
    client = TestClient(server._app)

    res = client.get("/api/scripts")
    assert res.status_code == 200
    assert res.json()["excluded"] == ["birthday"]


def test_post_script_settings_updates_and_persists(tmp_path):
    script_mode = DummyScriptMode(scripts=["wave", "birthday"])
    server = _make_server(9210)
    server._settings_store = RuntimeSettingsStore(tmp_path / "settings.json")
    server.attach_script_mode(script_mode)
    client = TestClient(server._app)

    res = client.post("/api/settings/scripts", json={"excluded": ["birthday"]})
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "excluded": ["birthday"]}
    assert script_mode.get_interlude_settings() == {"excluded": ["birthday"]}
    # Persisted so it survives a reattach.
    assert server._settings_store.load_script_settings() == {"excluded": ["birthday"]}


def test_attach_script_mode_applies_persisted_exclusions(tmp_path):
    store = RuntimeSettingsStore(tmp_path / "settings.json")
    store.save_script_settings(excluded=["birthday"])
    script_mode = DummyScriptMode(scripts=["wave", "birthday"])
    server = _make_server(9211)
    server._settings_store = store
    server.attach_script_mode(script_mode)

    assert script_mode.get_interlude_settings() == {"excluded": ["birthday"]}


def test_script_settings_endpoint_requires_attachment():
    server = _make_server(9212)
    client = TestClient(server._app)

    assert client.get("/api/settings/scripts").status_code == 409
    assert client.post("/api/settings/scripts", json={"excluded": []}).status_code == 409


def test_scripts_page_route_serves_html():
    server = _make_server(9208)
    client = TestClient(server._app)

    res = client.get("/scripts")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
