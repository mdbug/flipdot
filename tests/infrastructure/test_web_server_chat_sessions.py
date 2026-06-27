import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.infrastructure.web_server import WebServer


class DummyInputHub:
    def submit_pointer(self, *a, **k):
        pass

    def submit_click(self, *a, **k):
        pass

    def submit_action(self, *a, **k):
        pass

    def set_button_down(self, *a, **k):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate the on-disk session store to a temp dir for the test.
    monkeypatch.setenv("CHAT_SESSIONS_DIR", str(tmp_path / "chat_sessions"))
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8127)
    with TestClient(server._app) as client:
        yield client, server


def _seed_session(server, *, title="Write HELLO", messages=None):
    """Create a saved session directly through the store (no API key needed)."""
    record = server._chat_sessions.create(title=title, model="claude-opus-4-8")
    server._chat_sessions.save(
        record["id"],
        messages=messages
        or [
            {"role": "user", "content": "write hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "Done!"}]},
        ],
        title=title,
    )
    return record["id"]


def test_list_sessions_empty(client):
    c, _server = client
    data = c.get("/api/chat/sessions").json()
    assert data == {"active_id": None, "sessions": []}


def test_list_and_get_session(client):
    c, server = client
    session_id = _seed_session(server)

    listing = c.get("/api/chat/sessions").json()
    assert len(listing["sessions"]) == 1
    assert listing["sessions"][0]["id"] == session_id
    assert "messages" not in listing["sessions"][0]

    full = c.get(f"/api/chat/sessions/{session_id}").json()
    assert full["id"] == session_id
    assert full["messages"][0]["content"] == "write hello"


def test_resume_loads_into_active_buffer(client):
    c, server = client
    session_id = _seed_session(server)

    resumed = c.post(f"/api/chat/sessions/{session_id}/resume").json()
    assert resumed["id"] == session_id
    assert server._chat_session_id == session_id
    assert server._chat_messages[0]["content"] == "write hello"


def test_rename_session(client):
    c, server = client
    session_id = _seed_session(server)

    resp = c.patch(f"/api/chat/sessions/{session_id}", json={"title": "Renamed chat"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Renamed chat"
    assert server._chat_sessions.load(session_id)["title"] == "Renamed chat"


def test_delete_active_session_clears_buffer(client):
    c, server = client
    session_id = _seed_session(server)
    c.post(f"/api/chat/sessions/{session_id}/resume")
    assert server._chat_session_id == session_id

    assert c.delete(f"/api/chat/sessions/{session_id}").json() == {"status": "ok"}
    assert server._chat_sessions.load(session_id) is None
    assert server._chat_session_id is None
    assert server._chat_messages == []

    # Deleting again is a 404.
    assert c.delete(f"/api/chat/sessions/{session_id}").status_code == 404


def test_reset_clears_active_session(client):
    c, server = client
    session_id = _seed_session(server)
    c.post(f"/api/chat/sessions/{session_id}/resume")

    assert c.post("/api/chat/reset").json() == {"status": "ok"}
    assert server._chat_session_id is None
    assert server._chat_messages == []
    # The session itself stays on disk.
    assert server._chat_sessions.load(session_id) is not None


def test_invalid_session_id_rejected(client):
    c, _server = client
    assert c.get("/api/chat/sessions/not-a-valid-id").status_code == 400
    assert c.delete("/api/chat/sessions/..%2f..%2fetc").status_code in (400, 404)


def test_missing_session_is_404(client):
    c, _server = client
    missing = "0" * 32
    assert c.get(f"/api/chat/sessions/{missing}").status_code == 404
    assert c.post(f"/api/chat/sessions/{missing}/resume").status_code == 404
    assert c.patch(f"/api/chat/sessions/{missing}", json={"title": "x"}).status_code == 404
