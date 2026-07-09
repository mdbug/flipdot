from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.infrastructure import chat as chat_backend
from app.infrastructure.controller_metrics import ControllerMetrics
from app.infrastructure.mcp_server import build_flipdot_mcp
from app.modes.contracts import Frame
from app.services.chat_session_store import ChatSessionStore
from app.services.settings_store import DEFAULT_SETTINGS_PATH, RuntimeSettingsStore

if TYPE_CHECKING:
    from app.core.mode_manager import ModeManager

logger = logging.getLogger(__name__)

# Cap raw image uploads so a single request can't exhaust memory before the
# image layer ever sees it (the panel is only 28x28; this is generous headroom).
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Controller status is polled at up to 30 Hz per WebSocket client (plus REST
# calls); a short-lived shared cache keeps provider calls and metrics samples
# at that rate in total instead of multiplying per client.
_CONTROLLER_STATUS_CACHE_SEC = 1.0 / 30.0


def _mcp_enabled() -> bool:
    return os.getenv("ENABLE_MCP", "true").strip().lower() not in {"0", "false", "no", "off"}


def _mcp_auth_token() -> str:
    """The bearer token required on ``/mcp`` (empty string if unconfigured)."""
    return os.getenv("MCP_AUTH_TOKEN", "").strip()


def _cors_settings() -> tuple[list[str], str | None]:
    """CORS (allow_origins, allow_origin_regex) for the web UI.

    The UI is served same-origin, so cross-origin access is not required by
    default. ``WEB_UI_ALLOWED_ORIGINS`` (comma-separated exact origins) overrides
    this; otherwise only localhost on any port is permitted via a regex — a
    tightening of the previous ``allow_origins=["*"]``.
    """
    raw = os.getenv("WEB_UI_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()], None
    return [], r"https?://(localhost|127\.0\.0\.1)(:\d+)?"


def _mcp_security_lists(host: str, port: int) -> tuple[list[str], list[str]]:
    """Allowed ``Host``/``Origin`` values for MCP DNS-rebinding protection.

    Defaults to localhost only. Operators exposing the server to a LAN (by
    binding a non-loopback host) can widen the Host allow-list via
    ``MCP_ALLOWED_HOSTS`` (comma-separated ``host:*`` patterns); the bearer
    token remains the primary control.
    """
    explicit = os.getenv("MCP_ALLOWED_HOSTS", "").strip()
    if explicit:
        hosts = [h.strip() for h in explicit.split(",") if h.strip()]
    else:
        hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
        if host not in ("127.0.0.1", "localhost", "0.0.0.0", "::", ""):
            hosts.append(f"{host}:*")
    origins: list[str] = []
    for h in hosts:
        base = h[:-2] if h.endswith(":*") else h
        origins.append(f"http://{base}:*")
        origins.append(f"https://{base}:*")
    return hosts, origins


def _bearer_guard(app: Any, token: str) -> Any:
    """Wrap an ASGI ``app`` so every HTTP request needs ``Bearer <token>``."""

    async def guarded(scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            headers = dict(scope.get("headers") or [])
            provided = headers.get(b"authorization", b"").decode("latin-1")
            expected = f"Bearer {token}"
            if not (provided and secrets.compare_digest(provided, expected)):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return guarded


def _compute_asset_version(static_dir: Path) -> str:
    h = hashlib.sha1()
    skip_dirs = {"node_modules", ".mypy_cache", "tests"}
    paths = [
        p
        for p in static_dir.rglob("*")
        if p.suffix in {".js", ".css", ".html"}
        and not any(part in skip_dirs for part in p.relative_to(static_dir).parts)
    ]
    for p in sorted(paths):
        h.update(p.relative_to(static_dir).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


_ASSET_RE = re.compile(r'((?:href|src)="/static/[^"]+\.(?:js|css))(\?[^"]*)?(")')


def _inject_version(html: str, version: str) -> str:
    return _ASSET_RE.sub(lambda m: f"{m.group(1)}?v={version}{m.group(3)}", html)


class _VersionedStaticFiles(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


class PointerEventPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class ActionPayload(BaseModel):
    action: str


class ButtonPayload(BaseModel):
    down: bool


class BoardTextPayload(BaseModel):
    text: str = Field(default="", max_length=32)


class BoardPointPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    all_hits: bool = False
    select: bool = True


class BoardDrawPayload(BaseModel):
    # A freehand stroke on a 28x28 panel needs at most a few hundred points; cap
    # it generously so a single request can't pin CPU/memory with a giant list.
    points: list[BoardPointPayload] = Field(max_length=10000)
    line_width: int = Field(default=1, ge=1, le=8)
    color: str = Field(default="on")


class BoardShapePayload(BaseModel):
    tool: str
    start: BoardPointPayload
    end: BoardPointPayload
    line_width: int = Field(default=1, ge=1, le=8)
    color: str = Field(default="on")


class BoardTextObjectCreatePayload(BaseModel):
    text: str = Field(default="", max_length=64)
    x: int = 0
    y: int = 11
    font: str = "classic"
    size: int = 5
    style: str = "regular"
    spacing: int = Field(default=1, ge=0, le=6)
    scroll: bool = False
    scroll_speed: float = 7.0


class BoardTextObjectUpdatePayload(BaseModel):
    text: str | None = Field(default=None, max_length=64)
    x: int | None = None
    y: int | None = None
    font: str | None = None
    size: int | None = None
    style: str | None = None
    spacing: int | None = Field(default=None, ge=0, le=6)
    scroll: bool | None = None
    scroll_speed: float | None = None


class BoardImageMovePayload(BaseModel):
    x: int
    y: int


class BoardDragItemPayload(BaseModel):
    kind: str
    id: str
    x: int
    y: int


class BoardDragPayload(BaseModel):
    kind: str | None = None
    id: str | None = None
    x: int | None = None
    y: int | None = None
    ids: list[BoardDragItemPayload] | None = None


class BoardSavePayload(BaseModel):
    name: str


class BoardRenamePayload(BaseModel):
    old_name: str
    new_name: str


class SleepSettingsPayload(BaseModel):
    enabled: bool
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=23)


class PoseSettingsPayload(BaseModel):
    enabled: bool


class ClockSettingsPayload(BaseModel):
    style: Literal["digital", "analog"] = "digital"
    seconds: bool = False


class ScriptInterludePayload(BaseModel):
    excluded: list[str] = Field(default_factory=list)


class ChatPayload(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=64)


class ChatRenamePayload(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class FontPreviewVariantPayload(BaseModel):
    family: str
    size: int
    style: str


class FontPreviewSettingsPayload(BaseModel):
    phrase: str = Field(default="FLIPDOT", max_length=32)
    spacing: int = Field(default=0, ge=0, le=6)
    variants: list[FontPreviewVariantPayload] | None = None


class WebServer:
    """Built-in FastAPI server for frame mirroring and browser input."""

    def __init__(
        self,
        *,
        input_hub: Any,
        host: str,
        port: int,
        settings_path: Path | None = None,
        settings_store: RuntimeSettingsStore | None = None,
    ) -> None:
        self._input_hub = input_hub
        self._host = host
        self._port = port
        if settings_store is not None and settings_path is not None:
            raise ValueError("pass either settings_store or settings_path, not both")
        # Sharing the main loop's store matters: RuntimeSettingsStore's lock
        # is per-instance, so a second instance on the same file would let
        # concurrent read-modify-writes silently drop each other's sections.
        self._settings_store = settings_store or RuntimeSettingsStore(
            settings_path or DEFAULT_SETTINGS_PATH
        )

        self._frame_lock = threading.Lock()
        self._latest_frame = [[0 for _ in range(28)] for _ in range(28)]
        self._frame_width = 28
        self._frame_height = 28
        self._frame_version = 0
        self._current_mode = ""
        self._controls: list[dict] = []
        self._controller_status_provider: Callable[[], dict | list[dict]] | None = None
        self._controller_metrics = ControllerMetrics()
        self._controller_status_cache: tuple[dict, list[dict]] | None = None
        self._controller_status_cached_monotonic = 0.0
        self._controller_status_cache_lock = threading.Lock()
        self._board = None
        self._script_mode = None
        self._transition_policy = None
        self._font_preview = None
        self._clock = None
        self._mode_manager: ModeManager | None = None

        # Single shared conversation for the in-UI Claude chat (one physical
        # display = one conversation). Serialized with a lock so overlapping
        # requests can't interleave turns into the shared history. The active
        # conversation is auto-saved to the ChatSessionStore after every turn;
        # ``_chat_session_id`` is None until the first message opens a session.
        self._chat_messages: list[dict] = []
        self._chat_lock = asyncio.Lock()
        self._chat_sessions = ChatSessionStore()
        self._chat_session_id: str | None = None
        # The model is locked for the lifetime of a conversation (histories are
        # provider-native, so switching providers mid-chat would corrupt them).
        # None until the first message of a fresh conversation locks it.
        self._chat_model: str | None = None

        # Build the MCP server (if enabled) before the FastAPI app. The same
        # object backs two consumers: the in-UI Claude chat, which calls its
        # tools in-process, and the external HTTP /mcp endpoint. The in-UI chat
        # works whenever the object exists; the HTTP endpoint, which grants
        # remote tool access (including sandboxed code execution), is only
        # mounted when an MCP_AUTH_TOKEN is configured and is gated by that
        # bearer token on every request.
        self._mcp = None
        self._mcp_token = _mcp_auth_token()
        self._mcp_http_mounted = False
        if _mcp_enabled():
            allowed_hosts, allowed_origins = _mcp_security_lists(host, port)
            self._mcp = build_flipdot_mcp(
                snapshot_frame=self.snapshot_frame,
                get_mode_manager=lambda: self._mode_manager,
                get_board=lambda: self._board,
                get_script_mode=lambda: self._script_mode,
                get_transition_policy=lambda: self._transition_policy,
                settings_store=self._settings_store,
                get_controller_status=self._mcp_controller_status,
                allowed_hosts=allowed_hosts,
                allowed_origins=allowed_origins,
            )

        self._app = FastAPI(lifespan=self._build_lifespan())
        cors_origins, cors_origin_regex = _cors_settings()
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_origin_regex=cors_origin_regex,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self._mcp is not None and self._mcp_token:
            self._app.mount("/mcp", _bearer_guard(self._mcp.streamable_http_app(), self._mcp_token))
            self._mcp_http_mounted = True
        elif self._mcp is not None:
            logger.warning(
                "MCP_AUTH_TOKEN is not set; the external /mcp HTTP endpoint is disabled "
                "(the in-UI chat still works). Set MCP_AUTH_TOKEN to expose /mcp."
            )

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._wire_routes()

    def _build_lifespan(self) -> Callable[[FastAPI], Any]:
        server = self

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            # Read state via ``server`` (not captured locals): this runs at app
            # startup, after __init__ has decided whether the HTTP endpoint is
            # mounted. The session manager only exists once streamable_http_app()
            # has been called, i.e. when /mcp is mounted; the in-UI chat uses the
            # mcp object directly and needs no session manager.
            mcp = server._mcp
            if mcp is None or not server._mcp_http_mounted:
                yield
                return
            async with mcp.session_manager.run():
                yield

        return lifespan

    def snapshot_frame(self) -> tuple[list[list[int]], str, int, int]:
        """Return a cheap immutable snapshot of the latest published frame."""
        with self._frame_lock:
            return (
                [list(row) for row in self._latest_frame],
                self._current_mode,
                self._frame_width,
                self._frame_height,
            )

    def _mcp_controller_status(self) -> dict:
        status, _statuses = self._controller_status_payload()
        return status

    def _wire_routes(self) -> None:
        static_dir = Path(__file__).resolve().parents[2] / "web_ui"
        asset_version = _compute_asset_version(static_dir)
        logger.info("Web UI asset version: %s", asset_version)

        _html_pages: dict[str, bytes] = {
            name: _inject_version(
                (static_dir / f"{name}.html").read_text(encoding="utf-8"), asset_version
            ).encode("utf-8")
            for name in ("index", "font_grid", "controller_metrics", "chat", "scripts")
        }
        _HTML_HEADERS = {"Cache-Control": "no-cache"}

        self._app.mount(
            "/static",
            _VersionedStaticFiles(directory=str(static_dir)),
            name="static",
        )

        @self._app.get("/")
        def ui_root() -> HTMLResponse:
            return HTMLResponse(_html_pages["index"], headers=_HTML_HEADERS)

        @self._app.get("/font-grid")
        def font_grid_page() -> HTMLResponse:
            return HTMLResponse(_html_pages["font_grid"], headers=_HTML_HEADERS)

        @self._app.get("/controller-metrics")
        def controller_metrics_page() -> HTMLResponse:
            return HTMLResponse(_html_pages["controller_metrics"], headers=_HTML_HEADERS)

        @self._app.get("/chat")
        def chat_page() -> HTMLResponse:
            return HTMLResponse(_html_pages["chat"], headers=_HTML_HEADERS)

        @self._app.get("/scripts")
        def scripts_page() -> HTMLResponse:
            return HTMLResponse(_html_pages["scripts"], headers=_HTML_HEADERS)

        @self._app.get("/favicon.ico")
        def favicon() -> Response:
            # 204 must not carry a body (a JSON body here is malformed HTTP).
            return Response(status_code=204)

        @self._app.get("/api/frame")
        def get_frame() -> JSONResponse:
            with self._frame_lock:
                payload = {
                    "width": self._frame_width,
                    "height": self._frame_height,
                    "version": self._frame_version,
                    "pixels": self._latest_frame,
                    "mode": self._current_mode,
                    "controls": self._controls,
                }
            controller_status, controller_statuses = self._controller_status_payload()
            payload["controller_status"] = controller_status
            payload["controller_statuses"] = controller_statuses
            return JSONResponse(payload)

        @self._app.get("/api/controller/status")
        def get_controller_status() -> JSONResponse:
            controller_status, controller_statuses = self._controller_status_payload()
            return JSONResponse(
                {
                    "controller_status": controller_status,
                    "controller_statuses": controller_statuses,
                }
            )

        @self._app.get("/api/controller/metrics")
        def get_controller_metrics() -> JSONResponse:
            self._controller_status_payload()
            return JSONResponse(self._controller_metrics.payload())

        @self._app.post("/api/input/pointer")
        def post_pointer(payload: PointerEventPayload) -> dict[str, str]:
            self._input_hub.submit_pointer(source="web", x=payload.x, y=payload.y)
            return {"status": "ok"}

        @self._app.post("/api/input/click")
        def post_click(payload: PointerEventPayload) -> dict[str, str]:
            self._input_hub.submit_pointer(source="web", x=payload.x, y=payload.y)
            self._input_hub.submit_click(source="web", x=payload.x, y=payload.y)
            return {"status": "ok"}

        @self._app.post("/api/input/action")
        def post_action(payload: ActionPayload) -> dict[str, str]:
            action = payload.action.strip().lower()
            if not action:
                raise HTTPException(status_code=400, detail="action is required")
            self._input_hub.submit_action(source="web", action=action)
            return {"status": "ok"}

        @self._app.post("/api/input/button")
        def post_button(payload: ButtonPayload) -> dict[str, str]:
            self._input_hub.set_button_down(source="web", is_down=payload.down)
            return {"status": "ok"}

        @self._app.get("/api/board/state")
        def get_board_state() -> JSONResponse:
            board = self._require_board()
            return JSONResponse(board.export_state())

        @self._app.post("/api/board/text")
        def post_board_text(payload: BoardTextPayload) -> dict[str, str]:
            board = self._require_board()
            board.set_text(payload.text)
            return {"status": "ok"}

        @self._app.post("/api/board/draw")
        def post_board_draw(payload: BoardDrawPayload) -> dict[str, str]:
            board = self._require_board()
            if not payload.points:
                raise HTTPException(status_code=400, detail="points are required")
            board.apply_stroke(
                [{"x": p.x, "y": p.y} for p in payload.points],
                line_width=payload.line_width,
                color=payload.color,
            )
            return {"status": "ok"}

        @self._app.post("/api/board/clear")
        def post_board_clear() -> dict[str, str]:
            board = self._require_board()
            board.clear()
            return {"status": "ok"}

        @self._app.post("/api/board/undo")
        def post_board_undo() -> dict[str, object]:
            board = self._require_board()
            return {"status": "ok", "applied": board.undo()}

        @self._app.get("/api/board/fonts")
        def get_board_fonts() -> JSONResponse:
            board = self._require_board()
            return JSONResponse(board.get_font_catalog())

        @self._app.post("/api/board/hit-test")
        def post_board_hit_test(payload: BoardPointPayload) -> JSONResponse:
            board = self._require_board()
            hit_result = board.hit_test(
                payload.x,
                payload.y,
                select=payload.select,
                all_hits=payload.all_hits,
            )
            if payload.all_hits:
                hits = hit_result if isinstance(hit_result, list) else []
                return JSONResponse(
                    {"status": "ok", "hit": hits[0] if hits else None, "hits": hits}
                )
            return JSONResponse({"status": "ok", "hit": hit_result})

        @self._app.get("/api/board/text-objects")
        def get_board_text_objects() -> JSONResponse:
            board = self._require_board()
            payload = board.export_state()
            return JSONResponse(
                {
                    "text_objects": payload.get("text_objects", []),
                    "selected_text_id": payload.get("selected_text_id", ""),
                }
            )

        @self._app.post("/api/board/text-objects")
        def post_board_text_object(payload: BoardTextObjectCreatePayload) -> JSONResponse:
            board = self._require_board()
            created = board.add_text_object(payload.model_dump())
            return JSONResponse({"status": "ok", "text_object": created})

        @self._app.patch("/api/board/text-objects/{object_id}")
        def patch_board_text_object(
            object_id: str, payload: BoardTextObjectUpdatePayload
        ) -> JSONResponse:
            board = self._require_board()
            updated = board.update_text_object(
                object_id,
                payload.model_dump(exclude_none=True),
            )
            if updated is None:
                raise HTTPException(status_code=404, detail="text object not found")
            return JSONResponse({"status": "ok", "text_object": updated})

        @self._app.delete("/api/board/text-objects/{object_id}")
        def delete_board_text_object(object_id: str) -> dict[str, object]:
            board = self._require_board()
            return {"status": "ok", "deleted": board.delete_text_object(object_id)}

        @self._app.post("/api/board/shapes")
        def post_board_shape(payload: BoardShapePayload) -> dict[str, str]:
            board = self._require_board()
            board.draw_shape(
                payload.tool,
                {"x": payload.start.x, "y": payload.start.y},
                {"x": payload.end.x, "y": payload.end.y},
                line_width=payload.line_width,
                color=payload.color,
            )
            return {"status": "ok"}

        @self._app.post("/api/board/image/upload")
        async def post_board_image_upload(
            file: UploadFile = File(...),
            mode: str = Form("stamp"),
            x: int = Form(0),
            y: int = Form(0),
            threshold: int = Form(128),
        ) -> JSONResponse:
            board = self._require_board()
            # Reject oversized uploads by Content-Length first, then bound the
            # actual read so a lying/chunked client still can't blow up memory.
            if file.size is not None and file.size > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="image is too large")
            raw = await file.read(_MAX_UPLOAD_BYTES + 1)
            if len(raw) > _MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="image is too large")
            if not raw:
                raise HTTPException(status_code=400, detail="image payload is empty")
            if mode not in {"stamp", "object"}:
                raise HTTPException(status_code=400, detail="mode must be stamp or object")
            payload = board.place_uploaded_image(
                raw,
                mode=mode,
                x=x,
                y=y,
                threshold=threshold,
            )
            return JSONResponse({"status": "ok", **payload})

        @self._app.patch("/api/board/image-objects/{object_id}")
        def patch_board_image_object(
            object_id: str, payload: BoardImageMovePayload
        ) -> JSONResponse:
            board = self._require_board()
            updated = board.move_image_object(object_id, x=payload.x, y=payload.y)
            if updated is None:
                raise HTTPException(status_code=404, detail="image object not found")
            return JSONResponse({"status": "ok", "image_object": updated})

        @self._app.post("/api/board/drag/move")
        def post_board_drag_move(payload: BoardDragPayload) -> JSONResponse:
            board = self._require_board()
            try:
                if payload.ids:
                    moved = board.move_objects(
                        [item.model_dump() for item in payload.ids], persist=False
                    )
                    if moved is None:
                        raise HTTPException(status_code=404, detail="object not found")
                    return JSONResponse({"status": "ok", "objects": moved})

                if (
                    payload.kind is None
                    or payload.id is None
                    or payload.x is None
                    or payload.y is None
                ):
                    raise HTTPException(
                        status_code=400, detail="drag payload requires kind/id/x/y or ids"
                    )

                moved = board.move_object(
                    payload.kind,
                    payload.id,
                    payload.x,
                    payload.y,
                    persist=False,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if moved is None:
                raise HTTPException(status_code=404, detail="object not found")
            return JSONResponse({"status": "ok", "object": moved})

        @self._app.post("/api/board/drag/commit")
        def post_board_drag_commit(payload: BoardDragPayload) -> JSONResponse:
            board = self._require_board()
            try:
                if payload.ids:
                    moved = board.move_objects(
                        [item.model_dump() for item in payload.ids], persist=True
                    )
                    if moved is None:
                        raise HTTPException(status_code=404, detail="object not found")
                    return JSONResponse({"status": "ok", "objects": moved})

                if (
                    payload.kind is None
                    or payload.id is None
                    or payload.x is None
                    or payload.y is None
                ):
                    raise HTTPException(
                        status_code=400, detail="drag payload requires kind/id/x/y or ids"
                    )

                moved = board.move_object(
                    payload.kind,
                    payload.id,
                    payload.x,
                    payload.y,
                    persist=True,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if moved is None:
                raise HTTPException(status_code=404, detail="object not found")
            return JSONResponse({"status": "ok", "object": moved})

        @self._app.delete("/api/board/image-objects/{object_id}")
        def delete_board_image_object(object_id: str) -> dict[str, object]:
            board = self._require_board()
            return {"status": "ok", "deleted": board.delete_image_object(object_id)}

        @self._app.get("/api/boards")
        def get_boards() -> JSONResponse:
            board = self._require_board()
            return JSONResponse(board.list_boards())

        @self._app.post("/api/boards/save")
        def post_boards_save(payload: BoardSavePayload) -> JSONResponse:
            board = self._require_board()
            try:
                result = board.save_board(payload.name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return JSONResponse({"status": "ok", **result})

        @self._app.post("/api/boards/load")
        def post_boards_load(payload: BoardSavePayload) -> dict[str, object]:
            board = self._require_board()
            try:
                loaded = board.load_board(payload.name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not loaded:
                raise HTTPException(status_code=404, detail="board not found")
            return {"status": "ok", "loaded": True}

        @self._app.post("/api/boards/delete")
        def post_boards_delete(payload: BoardSavePayload) -> dict[str, object]:
            board = self._require_board()
            try:
                deleted = board.delete_board(payload.name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"status": "ok", "deleted": deleted}

        @self._app.post("/api/boards/rename")
        def post_boards_rename(payload: BoardRenamePayload) -> dict[str, object]:
            board = self._require_board()
            try:
                renamed = board.rename_board(payload.old_name, payload.new_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"status": "ok", "renamed": renamed}

        @self._app.get("/api/settings/sleep")
        def get_sleep_settings() -> JSONResponse:
            transition_policy = self._require_transition_policy()
            return JSONResponse(transition_policy.get_sleep_settings())

        @self._app.post("/api/settings/sleep")
        def post_sleep_settings(payload: SleepSettingsPayload) -> JSONResponse:
            transition_policy = self._require_transition_policy()
            settings = transition_policy.set_sleep_settings(
                enabled=payload.enabled,
                start_hour=payload.start_hour,
                end_hour=payload.end_hour,
            )
            self._settings_store.save_sleep_settings(
                enabled=bool(settings["enabled"]),
                start_hour=int(settings["start_hour"]),
                end_hour=int(settings["end_hour"]),
            )
            return JSONResponse({"status": "ok", **settings})

        @self._app.get("/api/settings/pose")
        def get_pose_settings() -> JSONResponse:
            mode_manager = self._require_mode_manager()
            return JSONResponse({"enabled": bool(mode_manager.pose_enabled)})

        @self._app.post("/api/settings/pose")
        def post_pose_settings(payload: PoseSettingsPayload) -> JSONResponse:
            mode_manager = self._require_mode_manager()
            # Persistence happens in ModeManager's on_pose_enabled_changed
            # hook (wired by the main loop) — the single path shared by every
            # source (panel menu, web UI, MCP).
            mode_manager.set_pose_enabled(payload.enabled)
            if mode_manager.pose_persist_failed:
                # The live toggle applied, but the settings write failed: a
                # silent 200 would revert the choice on the next restart.
                raise HTTPException(
                    status_code=500, detail="pose setting applied but could not be persisted"
                )
            return JSONResponse({"status": "ok", "enabled": bool(mode_manager.pose_enabled)})

        @self._app.get("/api/settings/clock")
        def get_clock_settings() -> JSONResponse:
            clock = self._require_clock()
            return JSONResponse(clock.get_settings())

        @self._app.post("/api/settings/clock")
        def post_clock_settings(payload: ClockSettingsPayload) -> JSONResponse:
            clock = self._require_clock()
            settings = clock.update_settings(style=payload.style, seconds=payload.seconds)
            self._settings_store.save_clock_settings(
                style=str(settings["style"]), seconds=bool(settings["seconds"])
            )
            return JSONResponse({"status": "ok", **settings})

        @self._app.get("/api/settings/font-preview")
        def get_font_preview_settings() -> JSONResponse:
            font_preview = self._require_font_preview()
            return JSONResponse(font_preview.get_settings())

        @self._app.get("/api/font-preview/variants")
        def get_font_preview_variants() -> JSONResponse:
            font_preview = self._require_font_preview()
            return JSONResponse(font_preview.get_variant_catalog())

        @self._app.get("/api/font-preview/glyph-grid")
        def get_font_preview_glyph_grid() -> JSONResponse:
            font_preview = self._require_font_preview()
            return JSONResponse(font_preview.get_glyph_grid())

        @self._app.post("/api/settings/font-preview")
        def post_font_preview_settings(payload: FontPreviewSettingsPayload) -> JSONResponse:
            font_preview = self._require_font_preview()
            variants_payload = None
            if payload.variants is not None:
                variants_payload = [item.model_dump() for item in payload.variants]
            settings = font_preview.update_settings(
                phrase=payload.phrase,
                variants=variants_payload,
                spacing=payload.spacing,
            )
            self._settings_store.save_font_preview_settings(
                phrase=str(settings["phrase"]),
                spacing=int(settings.get("spacing", 0)),
                variants=list(settings.get("variants", [])),
            )
            return JSONResponse({"status": "ok", **settings})

        @self._app.get("/api/scripts")
        def get_scripts() -> JSONResponse:
            return JSONResponse(self._require_script_mode().list_scripts())

        @self._app.get("/api/settings/scripts")
        def get_script_settings() -> JSONResponse:
            return JSONResponse(self._require_script_mode().get_interlude_settings())

        @self._app.post("/api/settings/scripts")
        def post_script_settings(payload: ScriptInterludePayload) -> JSONResponse:
            script_mode = self._require_script_mode()
            settings = script_mode.update_interlude_settings(excluded=payload.excluded)
            self._settings_store.save_script_settings(excluded=list(settings["excluded"]))
            return JSONResponse({"status": "ok", **settings})

        @self._app.get("/api/scripts/{name}/code")
        def get_script_code(name: str) -> JSONResponse:
            try:
                code = self._require_script_mode().get_code(name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if code is None:
                raise HTTPException(status_code=404, detail="script not found")
            return JSONResponse({"name": name, "code": code})

        @self._app.post("/api/scripts/{name}/play")
        def post_script_play(name: str) -> JSONResponse:
            try:
                result = self._require_script_mode().load_script(name)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if self._mode_manager is not None:
                self._mode_manager.set_mode("script")
            return JSONResponse({"status": "ok", **result})

        @self._app.delete("/api/scripts/{name}")
        def delete_script_endpoint(name: str) -> JSONResponse:
            try:
                deleted = self._require_script_mode().delete_script(name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not deleted:
                raise HTTPException(status_code=404, detail="script not found")
            return JSONResponse({"status": "ok", "deleted": True})

        @self._app.get("/api/ping")
        def ping() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.get("/api/chat/status")
        def get_chat_status() -> dict[str, bool]:
            return {"available": chat_backend.chat_available(self._mcp is not None)}

        @self._app.get("/api/chat/models")
        async def get_chat_models() -> dict[str, Any]:
            return {
                "default": chat_backend.resolve_model(None),
                "locked": self._chat_model,
                "models": [
                    {
                        "id": model_id,
                        "label": entry["label"],
                        "provider": entry["provider"],
                        "available": chat_backend.provider_available(entry["provider"]),
                    }
                    for model_id, entry in chat_backend.MODELS.items()
                ],
            }

        @self._app.post("/api/chat")
        async def post_chat(payload: ChatPayload) -> StreamingResponse:
            message = payload.message.strip()
            if not message:
                raise HTTPException(status_code=400, detail="message is required")

            async def generate() -> AsyncIterator[str]:
                # Hold the lock for the whole turn so the shared history stays
                # consistent if a second request arrives mid-stream.
                async with self._chat_lock:
                    new_session = self._chat_session_id is None
                    # Lock the model on the first message; later turns ignore the
                    # payload's model so the history stays provider-native.
                    if self._chat_model is None:
                        self._chat_model = chat_backend.resolve_model(payload.model)
                    history_len_before = len(self._chat_messages)
                    self._chat_messages.append({"role": "user", "content": message})
                    turn_usage: dict | None = None
                    async for event in chat_backend.run_chat(
                        self._mcp, self._chat_messages, model=self._chat_model
                    ):
                        # Snatch this turn's token/cost totals off the stream so we
                        # can fold them into the persisted session total below.
                        # Each event is one NDJSON line; parse it rather than
                        # substring-sniffing, which a text event containing the
                        # marker string would fool.
                        try:
                            parsed = json.loads(event)
                        except json.JSONDecodeError:
                            parsed = None
                        if isinstance(parsed, dict) and parsed.get("type") == "usage":
                            turn_usage = parsed
                        yield event

                    # A turn that errored before any assistant reply (no key,
                    # provider down) appended nothing after the user message:
                    # drop it so the history never carries an unanswered turn,
                    # and skip persisting. A fresh conversation left empty also
                    # releases the model lock — nothing is provider-native yet.
                    if len(self._chat_messages) == history_len_before + 1:
                        self._chat_messages.pop()
                        if not self._chat_messages and self._chat_session_id is None:
                            self._chat_model = None
                        return
                    if self._chat_session_id is None:
                        record = self._chat_sessions.create(title=message, model=self._chat_model)
                        self._chat_session_id = record["id"]
                    summary = self._chat_sessions.save(
                        self._chat_session_id,
                        messages=chat_backend.serialize_messages(self._chat_messages),
                        title=message if new_session else None,
                        model=self._chat_model,
                        usage=turn_usage,
                    )
                    yield (
                        json.dumps(
                            {"type": "session_saved", "session": summary},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        @self._app.post("/api/chat/reset")
        async def post_chat_reset() -> dict[str, str]:
            # Start a fresh conversation. The previous session stays on disk.
            async with self._chat_lock:
                self._chat_messages.clear()
                self._chat_session_id = None
                self._chat_model = None
            return {"status": "ok"}

        @self._app.get("/api/chat/sessions")
        async def list_chat_sessions() -> dict[str, Any]:
            return {
                "active_id": self._chat_session_id,
                "sessions": self._chat_sessions.list_summaries(),
            }

        @self._app.get("/api/chat/sessions/{session_id}")
        async def get_chat_session(session_id: str) -> dict[str, Any]:
            try:
                session_id = self._chat_sessions.sanitize_id(session_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            record = self._chat_sessions.load(session_id)
            if record is None:
                raise HTTPException(status_code=404, detail="session not found")
            return record

        @self._app.post("/api/chat/sessions/{session_id}/resume")
        async def resume_chat_session(session_id: str) -> dict[str, Any]:
            try:
                session_id = self._chat_sessions.sanitize_id(session_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            async with self._chat_lock:
                record = self._chat_sessions.load(session_id)
                if record is None:
                    raise HTTPException(status_code=404, detail="session not found")
                self._chat_messages.clear()
                self._chat_messages.extend(record.get("messages") or [])
                self._chat_session_id = session_id
                # Restore the session's model lock; an unknown (e.g. removed)
                # model leaves it unlocked so the next message re-resolves.
                stored_model = record.get("model")
                self._chat_model = stored_model if stored_model in chat_backend.MODELS else None
            return record

        @self._app.patch("/api/chat/sessions/{session_id}")
        async def rename_chat_session(
            session_id: str, payload: ChatRenamePayload
        ) -> dict[str, Any]:
            try:
                session_id = self._chat_sessions.sanitize_id(session_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            summary = self._chat_sessions.rename(session_id, payload.title)
            if summary is None:
                raise HTTPException(status_code=404, detail="session not found")
            return summary

        @self._app.delete("/api/chat/sessions/{session_id}")
        async def delete_chat_session(session_id: str) -> dict[str, str]:
            try:
                session_id = self._chat_sessions.sanitize_id(session_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            async with self._chat_lock:
                existed = self._chat_sessions.delete(session_id)
                if not existed:
                    raise HTTPException(status_code=404, detail="session not found")
                if self._chat_session_id == session_id:
                    self._chat_messages.clear()
                    self._chat_session_id = None
                    self._chat_model = None
            return {"status": "ok"}

        @self._app.websocket("/ws")
        async def ws_frames(websocket: WebSocket) -> None:
            await websocket.accept()
            sent_version = -1
            try:
                while True:
                    with self._frame_lock:
                        version = self._frame_version
                        payload = {
                            "width": self._frame_width,
                            "height": self._frame_height,
                            "version": version,
                            "pixels": self._latest_frame,
                            "mode": self._current_mode,
                            "controls": self._controls,
                        }
                    controller_status, controller_statuses = self._controller_status_payload()
                    payload["controller_status"] = controller_status
                    payload["controller_statuses"] = controller_statuses
                    if version != sent_version:
                        await websocket.send_json(payload)
                        sent_version = version
                    await asyncio.sleep(1 / 30)
            except WebSocketDisconnect:
                return

        @self._app.websocket("/ws/controller-status")
        async def ws_controller_status(websocket: WebSocket) -> None:
            await websocket.accept()
            last_signature = None
            try:
                while True:
                    status, statuses = self._controller_status_payload()
                    signature = ControllerMetrics.status_signature(statuses)
                    if signature != last_signature:
                        await websocket.send_json(
                            {
                                "controller_status": status,
                                "controller_statuses": statuses,
                            }
                        )
                        last_signature = signature
                    await asyncio.sleep(1 / 30)
            except WebSocketDisconnect:
                return

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def publish_frame(
        self,
        frame: Frame,
        *,
        mode: str = "",
        controls: list[dict] | None = None,
        panel_updated_monotonic: float | None = None,
    ) -> None:
        """Store the latest frame and metadata for API/WebSocket consumers."""
        # Convert once here so API handlers can return cheap immutable snapshots.
        pixels = frame.astype("uint8").tolist()
        with self._frame_lock:
            self._frame_height = len(pixels)
            self._frame_width = len(pixels[0]) if pixels else 0
            self._latest_frame = pixels
            self._current_mode = mode
            self._controls = controls or []
            self._frame_version += 1
        if panel_updated_monotonic is not None:
            self._controller_metrics.record_panel_latency(float(panel_updated_monotonic))

    def attach_board(self, board) -> None:
        self._board = board

    def attach_script_mode(self, script_mode) -> None:
        self._script_mode = script_mode
        persisted = self._settings_store.load_script_settings()
        if persisted is not None:
            script_mode.update_interlude_settings(excluded=list(persisted["excluded"]))

    def attach_mode_manager(self, mode_manager: ModeManager) -> None:
        """Wire the mode manager the settings/mode APIs drive.

        The persisted pose toggle is applied at startup by the main loop
        (which also persists changes from every source via the
        ``on_pose_enabled_changed`` hook); attaching must not clobber the
        live value with a stale one.
        """
        self._mode_manager = mode_manager

    def attach_transition_policy(self, transition_policy) -> None:
        self._transition_policy = transition_policy
        persisted = self._settings_store.load_sleep_settings()
        if persisted is not None:
            transition_policy.set_sleep_settings(
                enabled=bool(persisted["enabled"]),
                start_hour=int(persisted["start_hour"]),
                end_hour=int(persisted["end_hour"]),
            )

    def attach_clock(self, clock) -> None:
        self._clock = clock
        persisted = self._settings_store.load_clock_settings()
        if persisted is not None:
            clock.update_settings(
                style=str(persisted["style"]), seconds=bool(persisted.get("seconds", False))
            )

    def attach_font_preview(self, font_preview) -> None:
        self._font_preview = font_preview
        persisted: Any = self._settings_store.load_font_preview_settings()
        if persisted is not None:
            font_preview.update_settings(
                phrase=str(persisted["phrase"]),
                spacing=int(persisted.get("spacing", 0)),
                variants=list(persisted.get("variants", [])),
            )

    def attach_controller_status_provider(
        self, provider: Callable[[], dict | list[dict]] | None
    ) -> None:
        """Wire the controller-status source, dropping any cached payload.

        Invalidation matters: the provider is attached after startup, and a
        freshly attached source must take effect on the next request instead
        of waiting out a cached empty status.
        """
        with self._controller_status_cache_lock:
            self._controller_status_provider = provider
            self._controller_status_cache = None

    def _controller_status_payload(self) -> tuple[dict, list[dict]]:
        """Return (merged status, per-hub statuses), cached for a frame interval.

        Every WebSocket client and REST poller funnels through here; the cache
        bounds provider calls and metrics recording to ~30 Hz in total no
        matter how many clients are connected.
        """
        with self._controller_status_cache_lock:
            now = time.monotonic()
            if (
                self._controller_status_cache is not None
                and now - self._controller_status_cached_monotonic < _CONTROLLER_STATUS_CACHE_SEC
            ):
                return self._controller_status_cache
            fresh = self._controller_status_payload_uncached()
            self._controller_status_cache = fresh
            self._controller_status_cached_monotonic = now
            return fresh

    def _controller_status_payload_uncached(self) -> tuple[dict, list[dict]]:
        provider = self._controller_status_provider
        if provider is None:
            empty = ControllerMetrics.empty_status()
            return empty, [empty]

        try:
            raw_status = provider()
        except Exception:
            empty = ControllerMetrics.empty_status()
            return empty, [empty]

        status_list: list[dict] = []
        if isinstance(raw_status, dict):
            status_list = [self._controller_metrics.normalize_status(raw_status)]
        elif isinstance(raw_status, list):
            for item in raw_status:
                if isinstance(item, dict):
                    status_list.append(self._controller_metrics.normalize_status(item))

        if not status_list:
            empty = ControllerMetrics.empty_status()
            self._controller_metrics.record([empty])
            return empty, [empty]

        self._controller_metrics.record(status_list)
        return status_list[0], status_list

    def _require_board(self) -> Any:
        if self._board is None:
            raise HTTPException(status_code=409, detail="board mode is not attached")
        return self._board

    def _require_mode_manager(self) -> Any:
        if self._mode_manager is None:
            raise HTTPException(status_code=409, detail="mode manager is not attached")
        return self._mode_manager

    def _require_transition_policy(self) -> Any:
        if self._transition_policy is None:
            raise HTTPException(status_code=409, detail="transition policy is not attached")
        return self._transition_policy

    def _require_font_preview(self) -> Any:
        if self._font_preview is None:
            raise HTTPException(status_code=409, detail="font preview mode is not attached")
        return self._font_preview

    def _require_clock(self) -> Any:
        if self._clock is None:
            raise HTTPException(status_code=409, detail="clock mode is not attached")
        return self._clock

    def _require_script_mode(self) -> Any:
        if self._script_mode is None:
            raise HTTPException(status_code=409, detail="script mode is not attached")
        return self._script_mode
