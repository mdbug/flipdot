import asyncio
import json

import pytest

pytest.importorskip("mcp")

from app.infrastructure import chat as chat_backend
from app.infrastructure.mcp_server import build_flipdot_mcp


class DummyModeManager:
    def __init__(self):
        self.mode = "clock"

    def set_mode(self, mode, entered_via=None):
        self.mode = mode

    def get_mode_time(self):
        return 1.0


class DummyBoard:
    def __init__(self):
        self.cleared = 0
        self.text_objects = []

    def clear(self):
        self.cleared += 1

    def add_text_object(self, payload):
        item = {"id": f"txt_{len(self.text_objects) + 1}", **payload}
        self.text_objects.append(item)
        return item


def _build_mcp(mode_manager=None, board=None):
    frame = [[0] * 28 for _ in range(28)]
    return build_flipdot_mcp(
        input_hub=object(),
        snapshot_frame=lambda: ([list(r) for r in frame], "clock", 28, 28),
        get_mode_manager=lambda: mode_manager,
        get_board=lambda: board,
        get_transition_policy=lambda: None,
        settings_store=object(),
    )


def test_chat_available_requires_key_and_mcp(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert chat_backend.chat_available(True) is False
    assert chat_backend.chat_available(False) is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert chat_backend.chat_available(True) is True
    assert chat_backend.chat_available(False) is False  # MCP disabled


def test_tool_schemas_match_mcp_tools():
    mcp = _build_mcp()
    schemas = asyncio.run(chat_backend._mcp_tool_schemas(mcp))
    names = {s["name"] for s in schemas}
    assert {"get_display", "set_mode", "show_message"}.issubset(names)
    # Every schema is in Anthropic tool shape.
    for schema in schemas:
        assert set(schema) >= {"name", "description", "input_schema"}


def test_call_mcp_tool_executes_against_server():
    mode_manager = DummyModeManager()
    mcp = _build_mcp(mode_manager=mode_manager)

    text, is_error = asyncio.run(chat_backend._call_mcp_tool(mcp, "set_mode", {"mode": "board"}))

    assert is_error is False
    assert mode_manager.mode == "board"
    assert "board" in text


def test_call_mcp_tool_reports_errors():
    mcp = _build_mcp()
    text, is_error = asyncio.run(
        chat_backend._call_mcp_tool(mcp, "set_mode", {"mode": "not-a-real-mode"})
    )
    assert is_error is True
    assert "Error" in text


def test_run_chat_without_credentials_streams_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    chat_backend._client = None  # reset cached client
    mcp = _build_mcp()

    async def collect():
        events = []
        async for line in chat_backend.run_chat(mcp, [{"role": "user", "content": "hi"}]):
            events.append(json.loads(line))
        return events

    events = asyncio.run(collect())
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "credentials" in events[0]["message"].lower()


class _FakeBlock:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class _FakeFinal:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeDelta:
    def __init__(self, type, **kwargs):
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeEvent:
    def __init__(self, type, delta=None):
        self.type = type
        self.delta = delta


class _FakeStream:
    def __init__(self, texts, final, thoughts=None):
        self._texts = texts
        self._thoughts = thoughts or []
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for chunk in self._thoughts:
            yield _FakeEvent("content_block_delta", _FakeDelta("thinking_delta", thinking=chunk))
        for chunk in self._texts:
            yield _FakeEvent("content_block_delta", _FakeDelta("text_delta", text=chunk))

    async def get_final_message(self):
        return self._final


class _FakeMessages:
    def __init__(self, script):
        self._script = script
        self._turn = 0

    def stream(self, **kwargs):
        turn = self._script[self._turn]
        self._turn += 1
        return _FakeStream(turn["texts"], turn["final"], turn.get("thoughts"))


class _FakeClient:
    def __init__(self, script):
        self.messages = _FakeMessages(script)


def test_run_chat_full_loop_executes_tool(monkeypatch):
    mode_manager = DummyModeManager()
    board = DummyBoard()
    mcp = _build_mcp(mode_manager=mode_manager, board=board)

    script = [
        {
            "thoughts": ["I should ", "call show_message."],
            "texts": ["Showing it. "],
            "final": _FakeFinal(
                content=[
                    _FakeBlock("text", text="Showing it. "),
                    _FakeBlock(
                        "tool_use",
                        name="show_message",
                        input={"text": "HELLO"},
                        id="toolu_1",
                    ),
                ],
                stop_reason="tool_use",
            ),
        },
        {
            "texts": ["Done!"],
            "final": _FakeFinal(
                content=[_FakeBlock("text", text="Done!")],
                stop_reason="end_turn",
            ),
        },
    ]
    monkeypatch.setattr(chat_backend, "get_async_client", lambda: _FakeClient(script))

    messages = [{"role": "user", "content": "show hello"}]

    async def collect():
        return [json.loads(line) async for line in chat_backend.run_chat(mcp, messages)]

    events = asyncio.run(collect())
    types = [e["type"] for e in events]

    assert types == ["thinking", "thinking", "text", "tool", "text", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "thinking") == "I should call show_message."
    assert events[3]["name"] == "show_message"
    # The tool actually ran against the MCP server.
    assert mode_manager.mode == "board"
    assert board.text_objects[-1]["text"] == "HELLO"
    # History captured the assistant turn, the tool result, and the final turn.
    assert any(m["role"] == "user" and isinstance(m["content"], list) for m in messages)


def test_run_chat_without_mcp_streams_error():
    async def collect():
        events = []
        async for line in chat_backend.run_chat(None, [{"role": "user", "content": "hi"}]):
            events.append(json.loads(line))
        return events

    events = asyncio.run(collect())
    assert events[0]["type"] == "error"


def test_chat_routes_registered_and_guarded(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.infrastructure.web_server import WebServer

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    class Hub:
        def submit_pointer(self, *a, **k):
            pass

        def submit_click(self, *a, **k):
            pass

        def submit_action(self, *a, **k):
            pass

        def set_button_down(self, *a, **k):
            pass

    server = WebServer(input_hub=Hub(), host="127.0.0.1", port=8126)
    server.attach_mode_manager(DummyModeManager())

    with TestClient(server._app) as client:
        status = client.get("/api/chat/status")
        assert status.status_code == 200
        assert status.json() == {"available": False}

        # Streams a single error event (no key) rather than crashing.
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        assert lines[-1]["type"] == "error"

        assert client.post("/api/chat/reset").json() == {"status": "ok"}
