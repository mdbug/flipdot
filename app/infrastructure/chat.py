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
MAX_TOKENS = 8192

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
    "animations with a name so you can re-run them later. "
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
    tools = await _mcp_tool_schemas(mcp)

    try:
        while True:
            async with client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=tools,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield _event({"type": "text", "text": delta.text})
                    elif delta.type == "thinking_delta":
                        yield _event({"type": "thinking", "text": delta.thinking})
                final = await stream.get_final_message()

            # Preserve the full assistant content (text + thinking + tool_use) so
            # the next turn replays it unchanged on the same model.
            messages.append({"role": "assistant", "content": final.content})

            if final.stop_reason != "tool_use":
                break

            tool_results = []
            for block in final.content:
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

    yield _event({"type": "done"})
