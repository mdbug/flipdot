import asyncio
import json
from datetime import date

import pytest

pytest.importorskip("mcp")

from app.infrastructure import chat as chat_backend
from app.infrastructure.mcp_server import build_flipdot_mcp


class DummyModeManager:
    def __init__(self):
        self.mode = "clock"
        self.pose_enabled = True

    def set_mode(self, mode, entered_via=None):
        self.mode = mode

    def set_pose_enabled(self, enabled):
        self.pose_enabled = bool(enabled)

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


class _FakeStopDetails:
    def __init__(self, category=None):
        self.category = category


class _FakeTo:
    def __init__(self, model):
        self.model = model


class _FakeFallbackBlock:
    """Mimics the ``fallback`` content block the API emits at a switch point."""

    def __init__(self, to_model):
        self.type = "fallback"
        self.to = _FakeTo(to_model)


class _FakeUsage:
    def __init__(self, input=0, output=0, cache_write=0, cache_read=0):
        self.input_tokens = input
        self.output_tokens = output
        self.cache_creation_input_tokens = cache_write
        self.cache_read_input_tokens = cache_read


class _FakeFinal:
    def __init__(self, content, stop_reason, stop_details=None, usage=None, model=None):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = usage
        self.model = model


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


class _FakeBeta:
    def __init__(self, script):
        self.messages = _FakeMessages(script)


class _FakeClient:
    def __init__(self, script):
        self.messages = _FakeMessages(script)
        # The Fable 5 path uses the beta endpoint; give it its own turn counter.
        self.beta = _FakeBeta(script)


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

    assert types == ["thinking", "thinking", "text", "tool", "text", "usage", "done"]
    assert (
        "".join(e["text"] for e in events if e["type"] == "thinking")
        == "I should call show_message."
    )
    assert events[3]["name"] == "show_message"
    # The tool actually ran against the MCP server.
    assert mode_manager.mode == "board"
    assert board.text_objects[-1]["text"] == "HELLO"
    # History captured the assistant turn, the tool result, and the final turn.
    assert any(m["role"] == "user" and isinstance(m["content"], list) for m in messages)


def test_run_chat_surfaces_refusal(monkeypatch):
    mcp = _build_mcp()
    script = [
        {
            "texts": [],
            "final": _FakeFinal(
                content=[],
                stop_reason="refusal",
                stop_details=_FakeStopDetails("cyber"),
            ),
        },
    ]
    monkeypatch.setattr(chat_backend, "get_async_client", lambda: _FakeClient(script))

    messages = [{"role": "user", "content": "..."}]

    async def collect():
        return [json.loads(line) async for line in chat_backend.run_chat(mcp, messages)]

    events = asyncio.run(collect())

    assert events[-1]["type"] == "error"
    assert "declined" in events[-1]["message"].lower()
    assert "cyber" in events[-1]["message"]
    # Refusal ends the stream without a "done" and without persisting a turn.
    assert all(e["type"] != "done" for e in events)
    assert all(m["role"] != "assistant" for m in messages)


def test_run_chat_reports_fallback_served_model(monkeypatch):
    mcp = _build_mcp()
    script = [
        {
            "texts": ["Sure."],
            "final": _FakeFinal(
                content=[
                    _FakeFallbackBlock("claude-opus-4-8"),
                    _FakeBlock("text", text="Sure."),
                ],
                stop_reason="end_turn",
            ),
        },
    ]
    monkeypatch.setattr(chat_backend, "get_async_client", lambda: _FakeClient(script))

    messages = [{"role": "user", "content": "hi"}]

    async def collect():
        return [
            json.loads(line)
            async for line in chat_backend.run_chat(mcp, messages, model="claude-fable-5")
        ]

    events = asyncio.run(collect())
    types = [e["type"] for e in events]

    # Fable 5 declined and Opus 4.8 served — the user is told which, then it completes.
    assert "notice" in types
    notice = next(e for e in events if e["type"] == "notice")
    assert "claude-opus-4-8" in notice["text"]
    assert types[-1] == "done"


def test_usage_dict_prices_known_model_and_skips_unknown():
    tokens = {"input": 1_000_000, "output": 1_000_000, "cache_write": 0, "cache_read": 1_000_000}
    priced = chat_backend._usage_dict("claude-opus-4-8", tokens)
    # 5 (input) + 25 (output) + 0.5 (cache read) per the PRICING table.
    assert priced["cost"] == pytest.approx(30.5)
    assert priced["input"] == 1_000_000

    unpriced = chat_backend._usage_dict("claude-not-a-model", tokens)
    assert unpriced["cost"] is None
    assert unpriced["output"] == 1_000_000


def test_sonnet_5_switches_from_intro_to_standard_pricing(monkeypatch):
    tokens = {"input": 1_000_000, "output": 1_000_000, "cache_write": 0, "cache_read": 1_000_000}

    class _IntroDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 8, 31)

    monkeypatch.setattr(chat_backend, "date", _IntroDate)
    # 2 (input) + 10 (output) + 0.2 (cache read) at the introductory rate.
    intro = chat_backend._usage_dict("claude-sonnet-5", tokens)
    assert intro["cost"] == pytest.approx(12.2)

    class _StandardDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 1)

    monkeypatch.setattr(chat_backend, "date", _StandardDate)
    # 3 (input) + 15 (output) + 0.3 (cache read) at the standard rate.
    standard = chat_backend._usage_dict("claude-sonnet-5", tokens)
    assert standard["cost"] == pytest.approx(18.3)


def test_run_chat_emits_summed_usage_across_turns(monkeypatch):
    mode_manager = DummyModeManager()
    board = DummyBoard()
    mcp = _build_mcp(mode_manager=mode_manager, board=board)

    script = [
        {
            "texts": ["Showing it. "],
            "final": _FakeFinal(
                content=[
                    _FakeBlock("tool_use", name="show_message", input={"text": "HI"}, id="t1"),
                ],
                stop_reason="tool_use",
                usage=_FakeUsage(input=100, output=50),
                model="claude-opus-4-8",
            ),
        },
        {
            "texts": ["Done!"],
            "final": _FakeFinal(
                content=[_FakeBlock("text", text="Done!")],
                stop_reason="end_turn",
                usage=_FakeUsage(input=200, output=30, cache_read=1000),
                model="claude-opus-4-8",
            ),
        },
    ]
    monkeypatch.setattr(chat_backend, "get_async_client", lambda: _FakeClient(script))

    async def collect():
        return [
            json.loads(line)
            async for line in chat_backend.run_chat(
                mcp, [{"role": "user", "content": "hi"}], model="claude-opus-4-8"
            )
        ]

    events = asyncio.run(collect())
    usage = next(e for e in events if e["type"] == "usage")
    # Tokens summed across both turns.
    assert usage["input"] == 300
    assert usage["output"] == 80
    assert usage["cache_read"] == 1000
    # cost = (300*5 + 80*25 + 1000*0.5) / 1e6
    assert usage["cost"] == pytest.approx(0.004)


def test_run_chat_usage_prices_by_served_model(monkeypatch):
    """After a fallback, the served model (final.model) drives pricing."""
    mcp = _build_mcp()
    script = [
        {
            "texts": ["Sure."],
            "final": _FakeFinal(
                content=[_FakeFallbackBlock("claude-opus-4-8"), _FakeBlock("text", text="Sure.")],
                stop_reason="end_turn",
                usage=_FakeUsage(input=1_000_000),
                model="claude-opus-4-8",
            ),
        },
    ]
    monkeypatch.setattr(chat_backend, "get_async_client", lambda: _FakeClient(script))

    async def collect():
        return [
            json.loads(line)
            async for line in chat_backend.run_chat(
                mcp, [{"role": "user", "content": "hi"}], model="claude-fable-5"
            )
        ]

    events = asyncio.run(collect())
    usage = next(e for e in events if e["type"] == "usage")
    # Priced at Opus ($5/1M input), not Fable ($10/1M) — the model that served.
    assert usage["cost"] == pytest.approx(5.0)


def test_replayable_content_strips_pre_fallback_internals():
    content = [
        _FakeBlock("thinking"),
        _FakeBlock("text", text="partial"),
        _FakeFallbackBlock("claude-opus-4-8"),
        _FakeBlock("thinking"),
        _FakeBlock("text", text="answer"),
        _FakeBlock("tool_use", name="show_message", input={}, id="t1"),
    ]
    kept = [b.type for b in chat_backend._replayable_content(content)]
    # Pre-boundary thinking dropped; pre-boundary text kept; boundary and after kept.
    assert kept == ["text", "fallback", "thinking", "text", "tool_use"]


def test_replayable_content_noop_without_fallback():
    content = [_FakeBlock("thinking"), _FakeBlock("text", text="hi")]
    assert chat_backend._replayable_content(content) is content


def test_serialize_messages_converts_blocks_to_json():
    import json as _json

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                _FakeBlock("text", text="Showing it."),
                _FakeBlock("tool_use", name="show_message", input={"text": "HI"}, id="t1"),
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    ]
    serialized = chat_backend.serialize_messages(messages)
    # Round-trips through JSON without error and preserves structure.
    _json.dumps(serialized)
    assert serialized[0] == {"role": "user", "content": "hi"}
    assert serialized[1]["content"][0] == {"type": "text", "text": "Showing it."}
    assert serialized[1]["content"][1]["name"] == "show_message"
    assert serialized[1]["content"][1]["input"] == {"text": "HI"}
    assert serialized[2]["content"][0]["type"] == "tool_result"


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
