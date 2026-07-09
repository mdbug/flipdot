"""OpenAI-compatible chat backend (official OpenAI API and OpenRouter).

Mirrors the Anthropic loop in :mod:`app.infrastructure.chat`: the backend runs
the agentic tool-use loop as the MCP client and yields the same NDJSON event
stream to the browser. OpenAI and OpenRouter both speak the Chat Completions
wire format, so one loop serves both — only the base URL, API key, and usage
accounting differ.

History note: a conversation is locked to one model (and therefore one
provider), so ``messages`` for a session driven by this module only ever holds
OpenAI-format entries — user strings, assistant dicts with ``tool_calls``, and
``role: "tool"`` results — all plain dicts, which the session store persists
unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from app.infrastructure import chat as chat_backend

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = chat_backend.MAX_TOKENS

_clients: dict[str, Any] = {}


def get_async_client(provider: str) -> Any:
    """Lazily build a cached AsyncOpenAI client for ``provider``.

    Raises ChatUnavailable with a clear message when the provider's API key is
    missing, so the route can return a friendly error instead of a stack trace.
    """
    cached = _clients.get(provider)
    if cached is not None:
        return cached
    env_var = (
        "OPENROUTER_API_KEY" if provider == chat_backend.PROVIDER_OPENROUTER else "OPENAI_API_KEY"
    )
    api_key = os.getenv(env_var)
    if not api_key:
        raise chat_backend.ChatUnavailable(
            f"No {provider} API credentials found. Set {env_var} in the environment "
            "(.env) to enable this model."
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise chat_backend.ChatUnavailable("The 'openai' package is not installed.") from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if provider == chat_backend.PROVIDER_OPENROUTER:
        kwargs["base_url"] = OPENROUTER_BASE_URL
    client = AsyncOpenAI(**kwargs)
    _clients[provider] = client
    return client


async def _openai_tool_schemas(mcp: Any) -> list[dict]:
    """Convert the MCP server's tools into OpenAI function-tool definitions."""
    tools = await mcp.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]


class _ToolCallAccumulator:
    """Assemble streamed tool-call deltas into complete calls, keyed by index."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def add(self, delta_calls: list[Any] | None) -> None:
        """Fold one chunk's ``delta.tool_calls`` into the accumulator."""
        for call in delta_calls or []:
            entry = self._calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
            if call.id:
                entry["id"] = call.id
            function = getattr(call, "function", None)
            if function is not None:
                if function.name:
                    entry["name"] += function.name
                if function.arguments:
                    entry["arguments"] += function.arguments

    def finalized(self) -> list[dict[str, Any]]:
        """Return the assembled calls in the OpenAI assistant-message shape."""
        return [
            {
                "id": entry["id"],
                "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            }
            for _, entry in sorted(self._calls.items())
        ]


def _parse_arguments(raw: str) -> tuple[dict | None, str | None]:
    """Parse a tool call's JSON arguments, returning (args, error)."""
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Invalid tool arguments JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Tool arguments must be a JSON object."
    return parsed, None


def _usage_cost(usage: Any) -> float | None:
    """Extract OpenRouter's API-reported USD cost from a usage object, if any."""
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    return float(cost) if cost is not None else None


def _add_usage(acc: dict[str, int], usage: Any) -> None:
    """Fold one turn's usage into the accumulator using the shared key names."""
    for key in ("input", "output", "cache_write", "cache_read"):
        acc.setdefault(key, 0)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    acc["input"] += max(prompt - cached, 0)
    acc["cache_read"] += cached
    acc["output"] += getattr(usage, "completion_tokens", 0) or 0


async def run_openai_chat(mcp: Any, messages: list[dict], *, model: str) -> AsyncIterator[str]:
    """Run the streaming agentic loop against an OpenAI-compatible provider.

    Yields the same NDJSON events as the Anthropic loop; ``messages`` is mutated
    in place with assistant turns and tool results for persistence.
    """
    _event = chat_backend._event
    provider = chat_backend.MODELS[model]["provider"]
    supports_vision = chat_backend.MODELS.get(model, {}).get("supports_vision", False)
    try:
        client = get_async_client(provider)
    except chat_backend.ChatUnavailable as exc:
        yield _event({"type": "error", "message": str(exc)})
        return

    tools = await _openai_tool_schemas(mcp)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "max_completion_tokens": MAX_TOKENS,
        "tools": tools,
        "stream": True,
    }
    if provider == chat_backend.PROVIDER_OPENROUTER:
        # OpenRouter reports the actual USD cost when usage accounting is on —
        # its per-token rates vary by upstream provider, so this beats a table.
        request_kwargs["extra_body"] = {"usage": {"include": True}}
    else:
        request_kwargs["stream_options"] = {"include_usage": True}

    tokens: dict[str, int] = {}
    api_cost = 0.0
    api_cost_seen = False

    try:
        while True:
            request_kwargs["messages"] = [
                {"role": "system", "content": chat_backend.SYSTEM_PROMPT},
                *messages,
            ]
            stream = await client.chat.completions.create(**request_kwargs)

            text_parts: list[str] = []
            tool_calls = _ToolCallAccumulator()
            finish_reason: str | None = None
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    _add_usage(tokens, usage)
                    cost = _usage_cost(usage)
                    if cost is not None:
                        api_cost += cost
                        api_cost_seen = True
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    text_parts.append(delta.content)
                    yield _event({"type": "text", "text": delta.content})
                tool_calls.add(getattr(delta, "tool_calls", None))

            calls = tool_calls.finalized()
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }
            if calls:
                assistant_message["tool_calls"] = calls
            messages.append(assistant_message)

            if finish_reason != "tool_calls" or not calls:
                break

            # A role:"tool" message is text-only, so any image a tool returns is
            # collected here and appended as a follow-up user message below.
            pending_images: list[dict] = []
            for call in calls:
                name = call["function"]["name"]
                arguments, parse_error = _parse_arguments(call["function"]["arguments"])
                yield _event({"type": "tool", "name": name, "input": arguments or {}})
                if parse_error is not None:
                    output = parse_error
                else:
                    blocks, _ = await chat_backend._call_mcp_tool(mcp, name, arguments)
                    output, image_blocks = chat_backend.split_tool_blocks(blocks, supports_vision)
                    pending_images.extend(image_blocks)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            if pending_images:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{img['mime']};base64,{img['data']}"},
                            }
                            for img in pending_images
                        ],
                    }
                )
    except Exception as exc:  # noqa: BLE001 - report API/stream failures to the UI
        yield _event({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return

    usage_payload = chat_backend._usage_dict(model, tokens)
    if api_cost_seen:
        usage_payload["cost"] = round(api_cost, 6)
    yield _event({"type": "usage", **usage_payload})
    yield _event({"type": "done"})
