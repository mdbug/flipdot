"""Tests for the OpenAI-compatible chat loop (official OpenAI API + OpenRouter)."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from app.infrastructure import chat as chat_backend
from app.infrastructure import chat_openai
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


def _build_mcp(mode_manager=None):
    frame = [[0] * 28 for _ in range(28)]
    return build_flipdot_mcp(
        snapshot_frame=lambda: ([list(r) for r in frame], "clock", 28, 28),
        get_mode_manager=lambda: mode_manager,
        get_board=lambda: None,
        get_transition_policy=lambda: None,
        settings_store=object(),
    )


class _FakeFunction:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeCallDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeTokenDetails:
    def __init__(self, cached_tokens=0):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, cached=0, cost=None):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _FakeTokenDetails(cached)
        if cost is not None:
            self.cost = cost


class _FakeChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


class _FakeChunkStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


class _FakeCompletions:
    def __init__(self, script):
        self._script = script
        self._turn = 0
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self._script[self._turn]
        self._turn += 1
        return _FakeChunkStream(chunks)


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class _FakeOpenAIClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)


def _collect(mcp, messages, model):
    async def run():
        return [
            json.loads(line)
            async for line in chat_openai.run_openai_chat(mcp, messages, model=model)
        ]

    return asyncio.run(run())


def test_openai_loop_executes_tool_and_prices_usage(monkeypatch):
    mode_manager = DummyModeManager()
    mcp = _build_mcp(mode_manager=mode_manager)

    script = [
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="Switching. "))]),
            _FakeChunk(
                choices=[
                    _FakeChoice(
                        delta=_FakeDelta(
                            tool_calls=[_FakeCallDelta(0, id="call_1", name="set_mode")]
                        )
                    )
                ]
            ),
            _FakeChunk(
                choices=[
                    _FakeChoice(
                        delta=_FakeDelta(
                            tool_calls=[_FakeCallDelta(0, arguments='{"mode": "board"}')]
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            _FakeChunk(usage=_FakeUsage(prompt=1_000_000, completion=500_000)),
        ],
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="Done!"))]),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(), finish_reason="stop")]),
            _FakeChunk(usage=_FakeUsage(prompt=1_000_000, completion=500_000, cached=1_000_000)),
        ],
    ]
    fake = _FakeOpenAIClient(script)
    monkeypatch.setattr(chat_openai, "get_async_client", lambda provider: fake)

    messages = [{"role": "user", "content": "switch to board"}]
    events = _collect(mcp, messages, "gpt-5.4-mini")

    assert [e["type"] for e in events] == ["text", "tool", "text", "usage", "done"]
    assert mode_manager.mode == "board"
    assert events[1] == {"type": "tool", "name": "set_mode", "input": {"mode": "board"}}

    # History gained: assistant w/ tool_calls, tool result, final assistant.
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "set_mode"
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[3]["content"] == "Done!"

    # Usage: 1M input + 1M cached-read input + 1M output at gpt-5.4-mini rates.
    usage = events[3]
    assert usage["input"] == 1_000_000
    assert usage["cache_read"] == 1_000_000
    assert usage["output"] == 1_000_000
    assert usage["cost"] == pytest.approx(0.75 + 0.075 + 4.5)

    # Official-API requests include stream usage options and OpenAI tool shape.
    call = fake.chat.completions.calls[0]
    assert call["stream_options"] == {"include_usage": True}
    assert "extra_body" not in call
    assert call["messages"][0] == {"role": "system", "content": chat_backend.SYSTEM_PROMPT}
    assert all(tool["type"] == "function" for tool in call["tools"])


def test_openrouter_uses_api_reported_cost(monkeypatch):
    mcp = _build_mcp()
    script = [
        [
            _FakeChunk(
                choices=[_FakeChoice(delta=_FakeDelta(content="Hi!"), finish_reason="stop")]
            ),
            _FakeChunk(usage=_FakeUsage(prompt=1000, completion=100, cost=0.00123)),
        ],
    ]
    fake = _FakeOpenAIClient(script)
    monkeypatch.setattr(chat_openai, "get_async_client", lambda provider: fake)

    events = _collect(mcp, [{"role": "user", "content": "hi"}], "z-ai/glm-5.2")

    assert [e["type"] for e in events] == ["text", "usage", "done"]
    usage = events[1]
    # No static rate table for OpenRouter — cost is the API-reported figure.
    assert usage["cost"] == pytest.approx(0.00123)
    assert fake.chat.completions.calls[0]["extra_body"] == {"usage": {"include": True}}


def test_vision_model_appends_image_user_message_after_get_display(monkeypatch):
    mcp = _build_mcp()
    script = [
        [
            _FakeChunk(
                choices=[
                    _FakeChoice(
                        delta=_FakeDelta(
                            tool_calls=[_FakeCallDelta(0, id="call_1", name="get_display")]
                        )
                    )
                ]
            ),
            _FakeChunk(
                choices=[
                    _FakeChoice(
                        delta=_FakeDelta(tool_calls=[_FakeCallDelta(0, arguments="{}")]),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            _FakeChunk(usage=_FakeUsage(prompt=1000, completion=100)),
        ],
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="I see it."))]),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(), finish_reason="stop")]),
            _FakeChunk(usage=_FakeUsage(prompt=1000, completion=100)),
        ],
    ]
    fake = _FakeOpenAIClient(script)
    monkeypatch.setattr(chat_openai, "get_async_client", lambda provider: fake)

    messages = [{"role": "user", "content": "what's on the display?"}]
    _collect(mcp, messages, "gpt-5.5")

    # The get_display image rides a follow-up user message (tool messages are text-only).
    image_messages = [
        m
        for m in messages
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and any(part.get("type") == "image_url" for part in m["content"])
    ]
    assert len(image_messages) == 1
    url = image_messages[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_openai_loop_without_key_streams_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chat_openai._clients.clear()
    mcp = _build_mcp()

    events = _collect(mcp, [{"role": "user", "content": "hi"}], "gpt-5.5")
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "OPENAI_API_KEY" in events[0]["message"]
