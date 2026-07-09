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


def test_models_endpoint_lists_registry_with_availability(client, monkeypatch):
    c, server = client
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    data = c.get("/api/chat/models").json()
    assert data["locked"] is None
    by_id = {m["id"]: m for m in data["models"]}
    assert by_id["claude-haiku-4-5"]["provider"] == "anthropic"
    assert by_id["claude-haiku-4-5"]["available"] is False
    assert by_id["gpt-5.4-mini"]["available"] is True
    assert by_id["z-ai/glm-5.2"]["provider"] == "openrouter"
    assert by_id["z-ai/glm-5.2"]["available"] is False
    assert data["default"] in by_id


def test_model_locks_on_first_message_and_ignores_later_switches(client, monkeypatch):
    c, server = client
    from app.infrastructure import chat as chat_backend

    seen_models = []

    async def fake_run_chat(mcp, messages, *, model=None):
        seen_models.append(model)
        messages.append({"role": "assistant", "content": "ok"})
        yield chat_backend._event({"type": "text", "text": "ok"})
        yield chat_backend._event({"type": "done"})

    monkeypatch.setattr(chat_backend, "run_chat", fake_run_chat)

    c.post("/api/chat", json={"message": "hi", "model": "gpt-5.4-mini"})
    assert seen_models == ["gpt-5.4-mini"]
    assert server._chat_model == "gpt-5.4-mini"
    # A different model on a later turn is ignored — the lock wins.
    c.post("/api/chat", json={"message": "again", "model": "claude-fable-5"})
    assert seen_models == ["gpt-5.4-mini", "gpt-5.4-mini"]
    # The saved session records the locked model.
    assert server._chat_sessions.load(server._chat_session_id)["model"] == "gpt-5.4-mini"

    # Reset unlocks; the next message can pick a new model.
    c.post("/api/chat/reset")
    assert server._chat_model is None
    c.post("/api/chat", json={"message": "fresh", "model": "claude-fable-5"})
    assert seen_models[-1] == "claude-fable-5"


def test_resume_restores_model_lock(client):
    c, server = client
    session_id = _seed_session(server)

    c.post(f"/api/chat/sessions/{session_id}/resume")
    assert server._chat_model == "claude-opus-4-8"

    data = c.get("/api/chat/models").json()
    assert data["locked"] == "claude-opus-4-8"

    # Deleting the active session clears the lock again.
    c.delete(f"/api/chat/sessions/{session_id}")
    assert server._chat_model is None


def test_failed_turn_drops_unanswered_user_message(client, monkeypatch):
    c, server = client
    from app.infrastructure import chat as chat_backend

    async def failing_run_chat(mcp, messages, *, model=None):
        # Errored out before any assistant reply was appended.
        yield chat_backend._event({"type": "error", "message": "no key"})

    monkeypatch.setattr(chat_backend, "run_chat", failing_run_chat)

    c.post("/api/chat", json={"message": "hi", "model": "gpt-5.4-mini"})

    # The unanswered user message is dropped, nothing is persisted, and the
    # fresh conversation's model lock is released again.
    assert server._chat_messages == []
    assert server._chat_session_id is None
    assert server._chat_model is None
    assert c.get("/api/chat/sessions").json()["sessions"] == []


def test_usage_event_is_parsed_not_sniffed(client, monkeypatch):
    c, server = client
    from app.infrastructure import chat as chat_backend

    async def fake_run_chat(mcp, messages, *, model=None):
        # Assistant text that merely CONTAINS the usage marker string must not
        # be mistaken for the usage event itself.
        yield chat_backend._event({"type": "text", "text": 'the marker is "type": "usage"'})
        messages.append({"role": "assistant", "content": "ok"})
        yield chat_backend._event(
            {
                "type": "usage",
                "input": 7,
                "output": 3,
                "cache_write": 0,
                "cache_read": 0,
                "cost": 0.5,
            }
        )
        yield chat_backend._event({"type": "done"})

    monkeypatch.setattr(chat_backend, "run_chat", fake_run_chat)

    c.post("/api/chat", json={"message": "hi", "model": "gpt-5.4-mini"})

    usage = server._chat_sessions.load(server._chat_session_id)["usage"]
    assert usage["input"] == 7
    assert usage["output"] == 3
    assert usage["cost"] == 0.5
