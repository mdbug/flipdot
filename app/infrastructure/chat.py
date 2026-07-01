"""Claude chat backend that drives the flip-dot display through the MCP server.

This is "Option B": the FastAPI backend itself runs the agentic tool-use loop and
acts as the MCP client. Anthropic only ever sees tool *definitions* and sends back
tool-use *requests*; the tools execute locally against the same in-process
``FastMCP`` instance that is mounted at ``/mcp`` for external agents. Nothing about
the display is exposed to the internet — only this backend talks to the API.

The loop streams assistant text back to the browser as newline-delimited JSON
(NDJSON) events so the chat panel can render tokens as they arrive.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

# Default model. Override with the ANTHROPIC_MODEL env var.
DEFAULT_MODEL = "claude-opus-4-8"
# Models the in-UI selector may choose. All support the adaptive-thinking
# request shape used below, so no per-model thinking handling is needed
# (Fable 5 has thinking always on, which the same shape covers).
ALLOWED_MODELS = ("claude-opus-4-8", "claude-sonnet-5", "claude-fable-5")
MAX_TOKENS = 32768

# Claude Fable 5's safety classifiers can decline a request (a successful
# response with stop_reason "refusal"). Server-side fallbacks transparently
# re-serve a declined request on a broader-availability model within the same
# call; enabled only for the models that need it (the beta endpoint is required).
FALLBACK_MODEL = "claude-opus-4-8"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-06-01"
FALLBACK_ENABLED_MODELS = ("claude-fable-5",)

# USD per 1M tokens, per model. ``cache_write`` is the 5-minute-TTL rate (1.25x
# input) and ``cache_read`` the cache-hit rate (0.1x input) — the standard
# Anthropic cache economics. A model missing from this table renders tokens
# without a dollar figure (cost is None) rather than a guessed number.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_write": 6.25, "cache_read": 0.5},
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_write": 12.5, "cache_read": 1.0},
    # TODO: claude-sonnet-5 pricing is unconfirmed — fill in the real rates. Until
    # then it is intentionally absent so the UI shows tokens without a cost.
}

SYSTEM_PROMPT = (
    "You control a 28x28 monochrome flip-dot display through "
    "the provided tools. Pixels are either lit or dark — there is no colour or "
    "greyscale, and text must be short to fit. Use the tools to show messages, "
    "draw, switch modes, read back what is currently on the panel, and inspect or "
    "change system state. "
    "For dynamic or animated effects that the static drawing tools can't express "
    "(Game of Life, a bouncing ball, plasma, falling rain, fire), use the "
    "run_script tool: write a small self-contained Python frame generator with "
    "def setup(width, height) and def step(state, t, width, height) that returns "
    "(new_state, frame), where frame is a (height, width) numpy array of 0/1 "
    "values (width and height are both 28) and t is elapsed seconds since the "
    "animation started — base motion on t so speed is steady regardless of frame "
    "rate. The code runs sandboxed (no "
    "filesystem or network); only numpy "
    "(as np), math and random are available — do not import anything else. If "
    "run_script returns an error, read it and fix the code, then retry. Save good "
    "animations with a name so you can re-run them later. To edit an existing "
    "saved animation, call get_script to read its source first, then run_script "
    "the modified code under the same name to overwrite it. "
    "After drawing, writing or starting a script, you can call get_display to "
    "verify the result and refine it. Be concise and friendly: briefly confirm "
    "what you did rather than narrating every step."
)


class ChatUnavailable(RuntimeError):
    """Raised when chat cannot run (missing API key or MCP disabled)."""


def _has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def chat_available(mcp_enabled: bool) -> bool:
    """Whether chat can run: MCP enabled and API credentials present."""
    return bool(mcp_enabled) and _has_credentials()


_client = None


def get_async_client() -> Any:
    """Lazily build a cached AsyncAnthropic client.

    Raises ChatUnavailable with a clear message when no API credentials are set,
    so the route can return a friendly error instead of a stack trace.
    """
    global _client
    if _client is not None:
        return _client
    if not _has_credentials():
        raise ChatUnavailable(
            "No Anthropic API credentials found. Set ANTHROPIC_API_KEY in the "
            "environment (.env) to enable chat."
        )
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ChatUnavailable("The 'anthropic' package is not installed.") from exc

    _client = AsyncAnthropic()
    return _client


async def _mcp_tool_schemas(mcp: Any) -> list[dict]:
    """Convert the MCP server's tools into Anthropic tool definitions."""
    tools = await mcp.list_tools()
    schemas = []
    for tool in tools:
        schemas.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
        )
    return schemas


async def _call_mcp_tool(mcp: Any, name: str, arguments: dict | None) -> tuple[str, bool]:
    """Execute one MCP tool and return (text, is_error)."""
    try:
        result = await mcp.call_tool(name, arguments or {})
    except Exception as exc:  # noqa: BLE001 - surface tool failures back to Claude
        return f"Error calling {name}: {exc}", True

    # FastMCP.call_tool returns either a sequence of content blocks or a
    # (content, structured_result) tuple depending on SDK version.
    content = result[0] if isinstance(result, tuple) else result
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return ("\n".join(parts) or "(no output)", False)


def _event(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _serialize_block(block: Any) -> Any:
    """Convert one assistant content block into a JSON-serializable dict.

    Real Anthropic SDK blocks are pydantic models with ``model_dump``; we fall
    back to picking the known fields off plain objects so the conversation can be
    written to disk and replayed unchanged (``thinking`` blocks keep their
    ``signature``, which the API requires when replaying extended thinking).
    """
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    result: dict[str, Any] = {}
    for field in ("type", "text", "thinking", "signature", "name", "input", "id"):
        value = getattr(block, field, None)
        if value is not None:
            result[field] = value
    return result


def serialize_messages(messages: list[dict]) -> list[dict]:
    """Return a JSON-serializable deep copy of the chat history for persistence.

    User turns hold plain strings or already-dict tool-result blocks; assistant
    turns hold raw SDK content blocks, which are converted via
    :func:`_serialize_block`.
    """
    serialized: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [_serialize_block(block) for block in content]
        serialized.append({"role": message.get("role"), "content": content})
    return serialized


def _add_usage(acc: dict[str, int], usage: Any) -> None:
    """Fold one turn's ``final.usage`` into a running token accumulator."""
    for key in ("input", "output", "cache_write", "cache_read"):
        acc.setdefault(key, 0)
    acc["input"] += getattr(usage, "input_tokens", 0) or 0
    acc["output"] += getattr(usage, "output_tokens", 0) or 0
    acc["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    acc["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0


def _cost(model: str | None, tokens: dict[str, int]) -> float | None:
    """Return the USD cost of ``tokens`` for ``model``, or None if unpriced."""
    rates = PRICING.get(model or "")
    if rates is None:
        return None
    return round(
        sum(tokens.get(k, 0) * rates[k] for k in ("input", "output", "cache_write", "cache_read"))
        / 1_000_000,
        6,
    )


def _usage_dict(model: str | None, tokens: dict[str, int]) -> dict[str, Any]:
    """Build the ``usage`` event payload: token counts plus computed cost."""
    return {
        "input": tokens.get("input", 0),
        "output": tokens.get("output", 0),
        "cache_write": tokens.get("cache_write", 0),
        "cache_read": tokens.get("cache_read", 0),
        "cost": _cost(model, tokens),
    }


def _fallback_served_by(final: Any) -> str | None:
    """Return the model that took over if a server-side fallback fired this turn.

    Each switch point is a ``fallback`` content block whose ``to.model`` names the
    model that continued. Returns None when no fallback occurred.
    """
    served = None
    for block in getattr(final, "content", None) or []:
        if getattr(block, "type", None) == "fallback":
            to = getattr(block, "to", None)
            served = getattr(to, "model", None) or served
    return served


def _refusal_message(final: Any) -> str:
    """Build a user-facing message for a response the safety system declined."""
    msg = "The request was declined by the model's safety system"
    details = getattr(final, "stop_details", None)
    category = getattr(details, "category", None) if details is not None else None
    if category:
        msg += f" ({category})"
    return msg + ". Try rephrasing your request."


def _replayable_content(content: list) -> list:
    """Strip model-internal blocks emitted before a server-side fallback boundary.

    When the primary model is declined mid-turn, the discarded partial may carry
    thinking/tool_use blocks that the API rejects on replay. Blocks at and after
    the last ``fallback`` marker replay unchanged; text blocks always do.
    """
    last_fallback = -1
    for i, block in enumerate(content):
        if getattr(block, "type", None) == "fallback":
            last_fallback = i
    if last_fallback < 0:
        return content
    internal = ("thinking", "redacted_thinking", "tool_use", "server_tool_use")
    return [
        block
        for i, block in enumerate(content)
        if i >= last_fallback or getattr(block, "type", None) not in internal
    ]


async def run_chat(
    mcp: Any, messages: list[dict], *, model: str | None = None
) -> AsyncIterator[str]:
    """Run the streaming agentic loop, yielding NDJSON event strings.

    ``messages`` is mutated in place: the assistant turns and tool-result turns are
    appended so the caller can persist the conversation across requests.
    """
    if mcp is None:
        yield _event({"type": "error", "message": "MCP server is disabled (ENABLE_MCP=false)."})
        return

    try:
        client = get_async_client()
    except ChatUnavailable as exc:
        yield _event({"type": "error", "message": str(exc)})
        return

    model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL
    tools = await _mcp_tool_schemas(mcp)

    # ``messages`` is mutated in place each turn; the stream reads the live list.
    stream_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "tools": tools,
        "messages": messages,
    }
    if model in FALLBACK_ENABLED_MODELS:
        # Opt into server-side refusal fallbacks (requires the beta endpoint).
        stream_factory = client.beta.messages.stream
        stream_kwargs["betas"] = [SERVER_SIDE_FALLBACK_BETA]
        stream_kwargs["fallbacks"] = [{"model": FALLBACK_MODEL}]
    else:
        stream_factory = client.messages.stream

    # Token usage accrues across every turn of the loop (one user message can
    # produce several ``final`` messages via tool use); ``served_model`` follows
    # the model that actually billed the turn, which differs after a fallback.
    tokens: dict[str, int] = {}
    served_model = model

    try:
        while True:
            async with stream_factory(**stream_kwargs) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield _event({"type": "text", "text": delta.text})
                    elif delta.type == "thinking_delta":
                        yield _event({"type": "thinking", "text": delta.thinking})
                final = await stream.get_final_message()

            _add_usage(tokens, getattr(final, "usage", None))
            served_model = getattr(final, "model", None) or served_model

            # A fallback block means the primary model was declined and another
            # model served (part of) this turn — tell the user which one.
            served_by = _fallback_served_by(final)
            if served_by:
                yield _event(
                    {
                        "type": "notice",
                        "text": f"Fable declined this request — continued on {served_by}.",
                    }
                )

            # Whole chain refused: surface it instead of ending silently, and
            # don't persist the empty/partial assistant turn.
            if final.stop_reason == "refusal":
                yield _event({"type": "usage", **_usage_dict(served_model, tokens)})
                yield _event({"type": "error", "message": _refusal_message(final)})
                return

            # Preserve the assistant content so the next turn replays it (text +
            # thinking + tool_use); after a fallback, drop pre-boundary internals.
            assistant_content = _replayable_content(final.content)
            messages.append({"role": "assistant", "content": assistant_content})

            if final.stop_reason != "tool_use":
                break

            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue
                yield _event({"type": "tool", "name": block.name, "input": block.input})
                output, is_error = await _call_mcp_tool(mcp, block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})
    except Exception as exc:  # noqa: BLE001 - report API/stream failures to the UI
        yield _event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return

    yield _event({"type": "usage", **_usage_dict(served_model, tokens)})
    yield _event({"type": "done"})
