"""MCP server exposing the flip-dot display to AI agents.

The tools here are thin wrappers over the same live objects the web UI already
drives (``InputHub``, the ``Board`` mode, ``ModeManager``, the
``TransitionPolicy`` sleep schedule). The resulting ``FastMCP`` instance is
mounted onto the existing FastAPI app by :class:`WebServer`, so agents reach it
over the same host/port as the browser console.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.core.mode_manager import ModeManager

# Mode ids an agent is allowed to switch to. Derived from ModeManager so it stays
# in sync with the constants defined there.
KNOWN_MODES = tuple(
    value
    for name, value in vars(ModeManager).items()
    if name.startswith("MODE_") and isinstance(value, str)
)

# Cap decoded image payloads so an agent can't exhaust memory with a huge upload
# (the panel is only 28x28; this is generous headroom).
_MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Bound a freehand stroke so a single call can't pin CPU/memory with a giant
# point list (the panel is only 28x28; this is wide headroom).
_MAX_STROKE_POINTS = 10000

# Accept friendly aliases for the Board shape tool names.
_SHAPE_ALIASES = {
    "line": "line",
    "rect": "rectangle",
    "rectangle": "rectangle",
    "box": "rectangle",
    "circle": "circle",
    "ellipse": "circle",
}


def _frame_to_ascii(pixels: list[list[int]]) -> str:
    """Render a binary frame as ASCII so an agent can 'see' the display."""
    return "\n".join("".join("█" if cell else "·" for cell in row) for row in pixels)


def build_flipdot_mcp(
    *,
    snapshot_frame: Callable[[], tuple[list[list[int]], str, int, int]],
    get_mode_manager: Callable[[], Any | None],
    get_board: Callable[[], Any | None],
    get_script_mode: Callable[[], Any | None] | None = None,
    get_transition_policy: Callable[[], Any | None],
    settings_store: Any,
    get_controller_status: Callable[[], Any] | None = None,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
) -> FastMCP:
    """Build the flip-dot MCP server.

    All collaborators are passed as getters (or already-constructed handles) so
    that objects attached to the web server *after* construction — the board,
    transition policy and mode manager — resolve at call time rather than being
    captured as ``None``.
    """

    # streamable_http_path="/" so that mounting the app at "/mcp" yields the
    # endpoint at "/mcp" (rather than the default "/mcp" *inside* the sub-app,
    # which would double to "/mcp/mcp").
    mcp = FastMCP(
        "flipdot",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        # DNS-rebinding protection on: only the configured Host/Origin values are
        # accepted (localhost by default). Access is additionally gated by the
        # MCP_AUTH_TOKEN bearer check the web server wraps around this mount.
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts or ["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=allowed_origins or [],
        ),
    )

    def _require_board() -> Any:
        board = get_board()
        if board is None:
            raise ValueError("board mode is not available yet")
        return board

    def _require_mode_manager() -> Any:
        mode_manager = get_mode_manager()
        if mode_manager is None:
            raise ValueError("mode manager is not available yet")
        return mode_manager

    def _require_script_mode() -> Any:
        script_mode = get_script_mode() if get_script_mode is not None else None
        if script_mode is None:
            raise ValueError("script mode is not available yet")
        return script_mode

    def _require_transition_policy() -> Any:
        policy = get_transition_policy()
        if policy is None:
            raise ValueError("transition policy is not available yet")
        return policy

    # --- Read the display -------------------------------------------------

    @mcp.tool()
    def get_display() -> dict:
        """Return the current 28x28 frame as ASCII art plus the active mode.

        Off pixels are '·' and lit pixels are '█'. Call this after drawing to
        check the result and iterate.
        """
        pixels, mode, width, height = snapshot_frame()
        return {
            "mode": mode,
            "width": width,
            "height": height,
            "ascii": _frame_to_ascii(pixels),
        }

    @mcp.tool()
    def list_modes() -> dict:
        """List the display modes an agent can switch between."""
        return {"modes": list(KNOWN_MODES)}

    @mcp.tool()
    def get_status() -> dict:
        """Report the current mode, time in mode, and sleep schedule."""
        mode_manager = _require_mode_manager()
        status: dict = {
            "mode": mode_manager.mode,
            "mode_time_sec": round(mode_manager.get_mode_time(), 1),
        }
        policy = get_transition_policy()
        if policy is not None:
            status["sleep"] = policy.get_sleep_settings()
        if get_controller_status is not None:
            try:
                status["controllers"] = get_controller_status()
            except Exception:
                status["controllers"] = None
        if get_script_mode is not None:
            script_mode = get_script_mode()
            if script_mode is not None:
                status["script"] = script_mode.status()
        return status

    # --- Mode & system control -------------------------------------------

    @mcp.tool()
    def set_mode(mode: str) -> dict:
        """Switch the display to a given mode (see list_modes for valid ids)."""
        normalized = str(mode or "").strip().lower()
        if normalized not in KNOWN_MODES:
            raise ValueError(f"unknown mode '{mode}'; valid modes: {', '.join(KNOWN_MODES)}")
        _require_mode_manager().set_mode(normalized)
        return {"status": "ok", "mode": normalized}

    @mcp.tool()
    def get_sleep_settings() -> dict:
        """Return the sleep (blank-screen) schedule."""
        return _require_transition_policy().get_sleep_settings()

    @mcp.tool()
    def set_sleep_settings(enabled: bool, start_hour: int, end_hour: int) -> dict:
        """Set the sleep schedule. Hours are 0-23. Persists across restarts."""
        settings = _require_transition_policy().set_sleep_settings(
            enabled=bool(enabled),
            start_hour=int(start_hour),
            end_hour=int(end_hour),
        )
        settings_store.save_sleep_settings(
            enabled=bool(settings["enabled"]),
            start_hour=int(settings["start_hour"]),
            end_hour=int(settings["end_hour"]),
        )
        return {"status": "ok", **settings}

    # --- Messages & notifications ----------------------------------------

    @mcp.tool()
    def show_message(text: str, scroll: bool = False) -> dict:
        """Show a short text message on the display.

        Clears the board, switches to board mode and adds the message as a text
        object. Set scroll=True for a horizontally scrolling marquee.
        """
        board = _require_board()
        _require_mode_manager().set_mode(ModeManager.MODE_BOARD)
        board.clear()
        created = board.add_text_object(
            {
                "text": str(text),
                "x": 0,
                "y": 11,
                "font": "classic",
                "size": 5,
                "style": "regular",
                "spacing": 1,
                "scroll": bool(scroll),
            }
        )
        return {"status": "ok", "text_object": created}

    @mcp.tool()
    def clear_board() -> dict:
        """Erase everything on the board (drawing, text and images)."""
        _require_board().clear()
        return {"status": "ok"}

    # --- Board drawing & pixel art ---------------------------------------

    @mcp.tool()
    def add_text(
        text: str,
        x: int = 0,
        y: int = 11,
        font: str = "classic",
        size: int = 5,
        style: str = "regular",
        spacing: int = 1,
        scroll: bool = False,
    ) -> dict:
        """Add a text object to the board at pixel position (x, y)."""
        created = _require_board().add_text_object(
            {
                "text": str(text),
                "x": int(x),
                "y": int(y),
                "font": str(font),
                "size": int(size),
                "style": str(style),
                "spacing": int(spacing),
                "scroll": bool(scroll),
            }
        )
        return {"status": "ok", "text_object": created}

    @mcp.tool()
    def draw_shape(
        tool: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        line_width: int = 1,
        color: str = "on",
    ) -> dict:
        """Draw a shape on the board.

        tool is one of 'line', 'rectangle', 'circle'. Coordinates are normalized
        in [0, 1] (0,0 = top-left, 1,1 = bottom-right). color is 'on' or 'off'.
        """
        shape = _SHAPE_ALIASES.get(str(tool or "").strip().lower())
        if shape is None:
            raise ValueError("tool must be one of: line, rectangle, circle")
        _require_board().draw_shape(
            shape,
            {"x": float(start_x), "y": float(start_y)},
            {"x": float(end_x), "y": float(end_y)},
            line_width=int(line_width),
            color=str(color),
        )
        return {"status": "ok"}

    @mcp.tool()
    def draw_stroke(
        points: list[dict],
        line_width: int = 1,
        color: str = "on",
    ) -> dict:
        """Draw a freehand stroke connecting a list of points.

        Each point is {"x": float, "y": float} with coordinates normalized in
        [0, 1]. A single point paints a dot. color is 'on' or 'off'.
        """
        if len(points or []) > _MAX_STROKE_POINTS:
            raise ValueError(f"too many points; maximum is {_MAX_STROKE_POINTS}")
        normalized = [{"x": float(p["x"]), "y": float(p["y"])} for p in (points or [])]
        if not normalized:
            raise ValueError("points are required")
        _require_board().apply_stroke(
            normalized,
            line_width=int(line_width),
            color=str(color),
        )
        return {"status": "ok"}

    @mcp.tool()
    def place_image(
        image_base64: str,
        mode: str = "stamp",
        x: int = 0,
        y: int = 0,
        threshold: int = 128,
    ) -> dict:
        """Place an image on the board from base64-encoded image bytes.

        The image is converted to 1-bit and fitted to the panel. mode is 'stamp'
        (burn into the drawing layer) or 'object' (movable layer). threshold is
        the 0-255 brightness cutoff.
        """
        if mode not in {"stamp", "object"}:
            raise ValueError("mode must be 'stamp' or 'object'")
        # Reject before decoding: base64 inflates by 4/3, so cap the encoded
        # length up front rather than materialising a huge bytes object first.
        if len(image_base64) > _MAX_IMAGE_BYTES // 3 * 4 + 4:
            raise ValueError(f"image is too large; maximum is {_MAX_IMAGE_BYTES} decoded bytes")
        try:
            raw = base64.b64decode(image_base64, validate=True)
        except Exception as exc:  # noqa: BLE001 - surface decode failures to the agent
            raise ValueError(f"image_base64 is not valid base64: {exc}") from exc
        if not raw:
            raise ValueError("image payload is empty")
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"image is too large ({len(raw)} bytes); maximum is {_MAX_IMAGE_BYTES}"
            )
        result = _require_board().place_uploaded_image(
            raw, mode=mode, x=int(x), y=int(y), threshold=int(threshold)
        )
        return {"status": "ok", **result}

    @mcp.tool()
    def undo() -> dict:
        """Undo the last board change."""
        return {"status": "ok", "applied": _require_board().undo()}

    @mcp.tool()
    def get_board_state() -> dict:
        """Return the full board state (text objects, image objects, metadata)."""
        return _require_board().export_state()

    @mcp.tool()
    def get_fonts() -> dict:
        """Return the available font families, sizes and styles for text."""
        return _require_board().get_font_catalog()

    @mcp.tool()
    def list_boards() -> dict:
        """List saved board names and the active board."""
        return _require_board().list_boards()

    @mcp.tool()
    def save_board(name: str) -> dict:
        """Save the current board under a name ([A-Za-z0-9_-], up to 64 chars)."""
        return {"status": "ok", **_require_board().save_board(name)}

    @mcp.tool()
    def load_board(name: str) -> dict:
        """Load a previously saved board by name."""
        loaded = _require_board().load_board(name)
        if not loaded:
            raise ValueError(f"board '{name}' not found")
        return {"status": "ok", "loaded": True}

    # --- Scripted animations (sandboxed code) ----------------------------

    @mcp.tool()
    def run_script(code: str, name: str = "") -> dict:
        """Run a sandboxed Python animation on the display.

        Use this for dynamic/animated effects (Game of Life, bouncing ball,
        plasma, rain) that the drawing tools can't express. The code runs in an
        isolated process with no filesystem or network access; it must be
        self-contained and define a frame generator:

            def setup(width: int, height: int) -> Any:
                # return the initial state (optional)
            def step(state: Any, t: float, width: int, height: int
                     ) -> tuple[Any, np.ndarray]:
                # return (new_state, frame); frame is a (height, width)
                # numpy array of 0/1 values. t is elapsed seconds since the
                # animation started (use it for time-based motion).

        The display is 28 wide x 28 tall. `numpy` (as `np`), `math` and
        `random` are available; no other imports are allowed. If `name` is
        given the script is also saved for re-running later. On invalid or
        unsafe code this returns an error explaining what to fix.
        """
        script_mode = _require_script_mode()
        try:
            result = script_mode.run_script(str(code), name=str(name or "").strip())
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc
        _require_mode_manager().set_mode(ModeManager.MODE_SCRIPT)
        return {"status": "ok", **result}

    @mcp.tool()
    def stop_script() -> dict:
        """Stop the running animation and return the display to the clock."""
        stopped = _require_script_mode().stop_script()
        _require_mode_manager().set_mode(ModeManager.MODE_CLOCK)
        return {"status": "ok", "stopped": stopped}

    @mcp.tool()
    def list_scripts() -> dict:
        """List saved animation scripts and the active one."""
        return _require_script_mode().list_scripts()

    @mcp.tool()
    def get_script(name: str) -> dict:
        """Return the source code of a saved animation script by name.

        Use this to read an existing script before editing it: fetch the code,
        modify it, then run_script the new version under the same name to
        overwrite the saved copy.
        """
        code = _require_script_mode().get_code(name)
        if code is None:
            raise ValueError(f"script '{name}' not found")
        return {"name": name, "code": code}

    @mcp.tool()
    def save_script(name: str) -> dict:
        """Save the running animation under a name ([A-Za-z0-9_-], up to 64)."""
        saved = _require_script_mode().save_script(name)
        return {"status": "ok", "name": saved}

    @mcp.tool()
    def load_script(name: str) -> dict:
        """Load and run a previously saved animation by name."""
        script_mode = _require_script_mode()
        try:
            result = script_mode.load_script(name)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(str(exc)) from exc
        _require_mode_manager().set_mode(ModeManager.MODE_SCRIPT)
        return {"status": "ok", **result}

    @mcp.tool()
    def delete_script(name: str) -> dict:
        """Delete a saved animation script by name."""
        deleted = _require_script_mode().delete_script(name)
        if not deleted:
            raise ValueError(f"script '{name}' not found")
        return {"status": "ok", "deleted": True}

    return mcp
