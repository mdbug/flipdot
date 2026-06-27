from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.infrastructure import chat as chat_backend
from app.infrastructure.mcp_server import build_flipdot_mcp
from app.modes.contracts import Frame
from app.services.settings_store import RuntimeSettingsStore


def _mcp_enabled() -> bool:
    return os.getenv("ENABLE_MCP", "true").strip().lower() not in {"0", "false", "no", "off"}


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
    points: list[BoardPointPayload]
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


class ChatPayload(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


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
        self, *, input_hub: Any, host: str, port: int, settings_path: Path | None = None
    ) -> None:
        self._input_hub = input_hub
        self._host = host
        self._port = port
        self._settings_store = RuntimeSettingsStore(
            settings_path or (Path(__file__).resolve().parents[2] / "state" / "settings.json")
        )

        self._frame_lock = threading.Lock()
        self._latest_frame = [[0 for _ in range(28)] for _ in range(28)]
        self._frame_width = 28
        self._frame_height = 28
        self._frame_version = 0
        self._current_mode = ""
        self._controls: list[dict] = []
        self._controller_status_provider: Callable[[], dict | list[dict]] | None = None
        self._controller_metrics_lock = threading.Lock()
        self._controller_metrics_samples: list[dict] = []
        self._controller_metrics_events: list[dict] = []
        self._controller_metrics_button_events: list[dict] = []
        self._controller_metrics_panel_latency_events: list[dict] = []
        self._controller_metrics_counters: dict[str, dict] = {}
        self._controller_metrics_prev_connected: dict[str, bool] = {}
        self._controller_metrics_last_button_sequence: dict[str, int] = {}
        self._controller_metrics_last_latency_sequence: dict[str, int] = {}
        self._controller_metrics_last_sample_monotonic = 0.0
        self._board = None
        self._script_mode = None
        self._transition_policy = None
        self._font_preview = None
        self._mode_manager = None

        # Single shared conversation for the in-UI Claude chat (one physical
        # display = one conversation). Serialized with a lock so overlapping
        # requests can't interleave turns into the shared history.
        self._chat_messages: list[dict] = []
        self._chat_lock = asyncio.Lock()

        # Build the MCP server (if enabled) before the FastAPI app so its
        # streamable-HTTP session manager can be driven from the app lifespan.
        self._mcp = None
        if _mcp_enabled():
            self._mcp = build_flipdot_mcp(
                input_hub=self._input_hub,
                snapshot_frame=self.snapshot_frame,
                get_mode_manager=lambda: self._mode_manager,
                get_board=lambda: self._board,
                get_script_mode=lambda: self._script_mode,
                get_transition_policy=lambda: self._transition_policy,
                settings_store=self._settings_store,
                get_controller_status=self._mcp_controller_status,
            )

        self._app = FastAPI(lifespan=self._build_lifespan())
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        if self._mcp is not None:
            self._app.mount("/mcp", self._mcp.streamable_http_app())

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._wire_routes()

    def _build_lifespan(self) -> Callable[[FastAPI], Any]:
        mcp = self._mcp

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            if mcp is None:
                yield
                return
            # Run the MCP streamable-HTTP session manager for the app's lifetime.
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
        self._app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @self._app.get("/")
        def ui_root() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        @self._app.get("/font-grid")
        def font_grid_page() -> FileResponse:
            return FileResponse(static_dir / "font_grid.html")

        @self._app.get("/controller-metrics")
        def controller_metrics_page() -> FileResponse:
            return FileResponse(static_dir / "controller_metrics.html")

        @self._app.get("/chat")
        def chat_page() -> FileResponse:
            return FileResponse(static_dir / "chat.html")

        @self._app.get("/favicon.ico")
        def favicon() -> JSONResponse:
            return JSONResponse({}, status_code=204)

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
            return JSONResponse(self._controller_metrics_payload())

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
            raw = await file.read()
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

        @self._app.get("/api/ping")
        def ping() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.get("/api/chat/status")
        def get_chat_status() -> dict[str, bool]:
            return {"available": chat_backend.chat_available(self._mcp is not None)}

        @self._app.post("/api/chat")
        async def post_chat(payload: ChatPayload) -> StreamingResponse:
            message = payload.message.strip()
            if not message:
                raise HTTPException(status_code=400, detail="message is required")

            async def generate() -> AsyncIterator[str]:
                # Hold the lock for the whole turn so the shared history stays
                # consistent if a second request arrives mid-stream.
                async with self._chat_lock:
                    self._chat_messages.append({"role": "user", "content": message})
                    async for event in chat_backend.run_chat(self._mcp, self._chat_messages):
                        yield event

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        @self._app.post("/api/chat/reset")
        async def post_chat_reset() -> dict[str, str]:
            async with self._chat_lock:
                self._chat_messages.clear()
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
                    signature = self._controller_status_signature(statuses)
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
            self._record_panel_latency_metrics(float(panel_updated_monotonic))

    def _record_panel_latency_metrics(self, panel_updated_monotonic: float) -> None:
        now_wall = time.time()
        with self._controller_metrics_lock:
            latest_per_key: dict[str, dict] = {}
            for event in self._controller_metrics_button_events:
                key = str(event.get("key", "") or "")
                if not key:
                    continue
                latest_per_key[key] = event

            for key, event in latest_per_key.items():
                try:
                    sequence = int(event.get("sequence", 0))
                except (TypeError, ValueError):
                    continue
                last_sequence = self._controller_metrics_last_latency_sequence.get(key, 0)
                if sequence <= last_sequence:
                    continue
                monotonic_value: Any = event.get("monotonic")
                try:
                    event_monotonic = float(monotonic_value)
                except (TypeError, ValueError):
                    continue
                latency_ms = max(0.0, (panel_updated_monotonic - event_monotonic) * 1000.0)
                self._controller_metrics_last_latency_sequence[key] = sequence
                self._controller_metrics_panel_latency_events.append(
                    {
                        "timestamp": now_wall,
                        "key": key,
                        "sequence": sequence,
                        "latency_ms": latency_ms,
                    }
                )

            cutoff = now_wall - 3600.0
            self._controller_metrics_panel_latency_events = [
                event
                for event in self._controller_metrics_panel_latency_events
                if float(event.get("timestamp", 0.0)) >= cutoff
            ][-5000:]

    def attach_board(self, board) -> None:
        self._board = board

    def attach_script_mode(self, script_mode) -> None:
        self._script_mode = script_mode

    def attach_mode_manager(self, mode_manager) -> None:
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
        self._controller_status_provider = provider

    def _controller_status_payload(self) -> tuple[dict, list[dict]]:
        provider = self._controller_status_provider
        if provider is None:
            empty = self._empty_controller_status()
            return empty, [empty]

        try:
            raw_status = provider()
        except Exception:
            empty = self._empty_controller_status()
            return empty, [empty]

        status_list: list[dict] = []
        if isinstance(raw_status, dict):
            status_list = [self._normalize_controller_status(raw_status)]
        elif isinstance(raw_status, list):
            for item in raw_status:
                if isinstance(item, dict):
                    status_list.append(self._normalize_controller_status(item))

        if not status_list:
            empty = self._empty_controller_status()
            self._record_controller_metrics([empty])
            return empty, [empty]

        self._record_controller_metrics(status_list)
        return status_list[0], status_list

    def _record_controller_metrics(self, statuses: list[dict]) -> None:
        now_monotonic = time.monotonic()
        now_wall = time.time()
        normalized_statuses = list(statuses)

        with self._controller_metrics_lock:
            changed = False
            sample_statuses = []
            for index, status in enumerate(normalized_statuses):
                key = self._controller_metric_key(index, status)
                label = f"P{index + 1}"
                connected = bool(status.get("connected", False))
                previous_connected = self._controller_metrics_prev_connected.get(key)

                counter = self._controller_metrics_counters.setdefault(
                    key,
                    {
                        "label": label,
                        "address": str(status.get("address", "") or ""),
                        "device_name": str(status.get("device_name", "") or ""),
                        "disconnects": 0,
                        "reconnects": 0,
                        "disconnect_reason_counts": {},
                    },
                )
                counter["label"] = label
                counter["address"] = str(status.get("address", "") or "")
                counter["device_name"] = str(status.get("device_name", "") or "")

                if previous_connected is not None and previous_connected != connected:
                    changed = True
                    event_type = "connected" if connected else "disconnected"
                    reason_code = (
                        str(status.get("last_disconnect_reason_code", "") or "")
                        if not connected
                        else ""
                    )
                    if connected:
                        counter["reconnects"] += 1
                    else:
                        counter["disconnects"] += 1
                        if reason_code:
                            reason_counts = counter.setdefault("disconnect_reason_counts", {})
                            reason_counts[reason_code] = int(reason_counts.get(reason_code, 0)) + 1
                    self._controller_metrics_events.append(
                        {
                            "timestamp": now_wall,
                            "key": key,
                            "label": label,
                            "address": counter["address"],
                            "event": event_type,
                            "reason_code": reason_code or None,
                        }
                    )

                self._controller_metrics_prev_connected[key] = connected
                pressed_buttons = status.get("pressed_buttons", [])
                if not isinstance(pressed_buttons, list):
                    pressed_buttons = []
                button_events = status.get("recent_button_events", [])
                if not isinstance(button_events, list):
                    button_events = []
                for button_event in button_events:
                    if not isinstance(button_event, dict):
                        continue
                    raw_sequence: Any = button_event.get("sequence")
                    try:
                        sequence = int(raw_sequence)
                    except (TypeError, ValueError):
                        continue
                    last_sequence = self._controller_metrics_last_button_sequence.get(key, 0)
                    if sequence <= last_sequence:
                        continue
                    self._controller_metrics_last_button_sequence[key] = sequence
                    event_monotonic: Any = button_event.get("monotonic")
                    event_monotonic_value = None
                    try:
                        event_monotonic_value = float(event_monotonic)
                    except (TypeError, ValueError):
                        event_timestamp = now_wall
                    else:
                        event_timestamp = now_wall - max(0.0, now_monotonic - event_monotonic_value)
                    self._controller_metrics_button_events.append(
                        {
                            "timestamp": event_timestamp,
                            "key": key,
                            "label": label,
                            "address": counter["address"],
                            "sequence": sequence,
                            "button": str(button_event.get("button", "") or ""),
                            "event": str(button_event.get("event", "") or ""),
                            "monotonic": event_monotonic_value
                            if isinstance(event_monotonic_value, float)
                            else None,
                        }
                    )
                sample_statuses.append(
                    {
                        "key": key,
                        "label": label,
                        "address": counter["address"],
                        "connected": connected,
                        "last_event_age_ms": status.get("last_event_age_ms"),
                        "bluetooth_connect_attempts": status.get("bluetooth_connect_attempts"),
                        "bluetooth_connect_failures": status.get("bluetooth_connect_failures"),
                        "last_bluetooth_connect_attempt_age_ms": status.get(
                            "last_bluetooth_connect_attempt_age_ms"
                        ),
                        "battery_percentage": status.get("battery_percentage"),
                        "battery_source": status.get("battery_source"),
                        "battery_age_ms": status.get("battery_age_ms"),
                        "battery_poll_duration_ms": status.get("battery_poll_duration_ms"),
                        "rssi_dbm": status.get("rssi_dbm"),
                        "tx_power_dbm": status.get("tx_power_dbm"),
                        "link_quality": status.get("link_quality"),
                        "signal_source": status.get("signal_source"),
                        "connection_interval_ms": status.get("connection_interval_ms"),
                        "connection_latency": status.get("connection_latency"),
                        "supervision_timeout_ms": status.get("supervision_timeout_ms"),
                        "connection_params_source": status.get("connection_params_source"),
                        "last_disconnect_reason_code": status.get("last_disconnect_reason_code"),
                        "disconnect_reason_counts": status.get("disconnect_reason_counts"),
                        "bluetooth_metrics_age_ms": status.get("bluetooth_metrics_age_ms"),
                        "bluetooth_metrics_poll_duration_ms": status.get(
                            "bluetooth_metrics_poll_duration_ms"
                        ),
                        "pressed_count": len(pressed_buttons),
                        "pressed_buttons": [str(item) for item in pressed_buttons],
                    }
                )

            should_sample = changed or (
                now_monotonic - self._controller_metrics_last_sample_monotonic >= 0.5
            )
            if not should_sample:
                return

            self._controller_metrics_samples.append(
                {
                    "timestamp": now_wall,
                    "controllers": sample_statuses,
                }
            )
            self._controller_metrics_last_sample_monotonic = now_monotonic

            cutoff = now_wall - 3600.0
            self._controller_metrics_samples = [
                sample
                for sample in self._controller_metrics_samples
                if float(sample.get("timestamp", 0.0)) >= cutoff
            ][-7200:]
            self._controller_metrics_events = [
                event
                for event in self._controller_metrics_events
                if float(event.get("timestamp", 0.0)) >= cutoff
            ][-1000:]
            self._controller_metrics_button_events = [
                event
                for event in self._controller_metrics_button_events
                if float(event.get("timestamp", 0.0)) >= cutoff
            ][-5000:]

    def _controller_metrics_payload(self) -> dict:
        with self._controller_metrics_lock:
            samples = list(self._controller_metrics_samples)
            events = list(self._controller_metrics_events)
            button_events = list(self._controller_metrics_button_events)
            panel_latency_events = list(self._controller_metrics_panel_latency_events)
            counters = {
                key: dict(value) for key, value in self._controller_metrics_counters.items()
            }

        now_wall = time.time()
        window_sec = 3600
        window_hours = window_sec / 3600.0
        summaries = []
        for key, counter in counters.items():
            controller_samples = []
            connected_samples = 0
            freshness_values = []
            rssi_values = []
            interval_values = []
            supervision_values = []
            connection_latency_values = []
            button_event_count = 0
            for sample in samples:
                for status in sample.get("controllers", []):
                    if status.get("key") != key:
                        continue
                    controller_samples.append(status)
                    if status.get("connected"):
                        connected_samples += 1
                    age_ms = status.get("last_event_age_ms")
                    if isinstance(age_ms, (int, float)):
                        freshness_values.append(float(age_ms))
                    rssi_dbm = status.get("rssi_dbm")
                    if isinstance(rssi_dbm, (int, float)):
                        rssi_values.append(float(rssi_dbm))
                    interval_ms = status.get("connection_interval_ms")
                    if isinstance(interval_ms, (int, float)):
                        interval_values.append(float(interval_ms))
                    supervision_ms = status.get("supervision_timeout_ms")
                    if isinstance(supervision_ms, (int, float)):
                        supervision_values.append(float(supervision_ms))
                    conn_latency = status.get("connection_latency")
                    if isinstance(conn_latency, (int, float)):
                        connection_latency_values.append(float(conn_latency))

            for button_event in button_events:
                if button_event.get("key") == key:
                    button_event_count += 1

            sample_count = len(controller_samples)
            connected_ratio = (connected_samples / sample_count) if sample_count else 0.0
            average_event_age_ms = (
                (sum(freshness_values) / len(freshness_values)) if freshness_values else None
            )
            average_rssi_dbm = (sum(rssi_values) / len(rssi_values)) if rssi_values else None
            average_connection_interval_ms = (
                (sum(interval_values) / len(interval_values)) if interval_values else None
            )
            average_supervision_timeout_ms = (
                (sum(supervision_values) / len(supervision_values)) if supervision_values else None
            )
            average_connection_latency = (
                (sum(connection_latency_values) / len(connection_latency_values))
                if connection_latency_values
                else None
            )
            disconnect_count = int(counter.get("disconnects", 0))
            controller_events = sorted(
                [event for event in events if event.get("key") == key],
                key=lambda item: float(item.get("timestamp", 0.0)),
            )
            reconnect_durations_sec = []
            last_disconnect_ts = None
            for controller_event in controller_events:
                event_type = str(controller_event.get("event", "") or "")
                event_ts = float(controller_event.get("timestamp", 0.0))
                if event_type == "disconnected":
                    last_disconnect_ts = event_ts
                elif (
                    event_type == "connected"
                    and last_disconnect_ts is not None
                    and event_ts >= last_disconnect_ts
                ):
                    reconnect_durations_sec.append(event_ts - last_disconnect_ts)
                    last_disconnect_ts = None

            mttr_sec = (
                sum(reconnect_durations_sec) / len(reconnect_durations_sec)
                if reconnect_durations_sec
                else None
            )
            latency_values = [
                float(item.get("latency_ms", 0.0))
                for item in panel_latency_events
                if item.get("key") == key and isinstance(item.get("latency_ms"), (int, float))
            ]

            def percentile(values: list[float], p: float) -> float | None:
                if not values:
                    return None
                ordered = sorted(values)
                if len(ordered) == 1:
                    return ordered[0]
                rank = (len(ordered) - 1) * p
                low = int(rank)
                high = min(low + 1, len(ordered) - 1)
                weight = rank - low
                return ordered[low] * (1.0 - weight) + ordered[high] * weight

            latency_p50_ms = percentile(latency_values, 0.50)
            latency_p95_ms = percentile(latency_values, 0.95)
            latency_p99_ms = percentile(latency_values, 0.99)
            latest_status = controller_samples[-1] if controller_samples else None
            summaries.append(
                {
                    "key": key,
                    "label": counter.get("label", ""),
                    "address": counter.get("address", ""),
                    "device_name": counter.get("device_name", ""),
                    "disconnects": disconnect_count,
                    "reconnects": int(counter.get("reconnects", 0)),
                    "disconnects_per_hour": (disconnect_count / window_hours)
                    if window_hours > 0
                    else 0.0,
                    "mttr_sec": mttr_sec,
                    "connected_ratio": connected_ratio,
                    "average_event_age_ms": average_event_age_ms,
                    "average_rssi_dbm": average_rssi_dbm,
                    "average_connection_interval_ms": average_connection_interval_ms,
                    "average_supervision_timeout_ms": average_supervision_timeout_ms,
                    "average_connection_latency": average_connection_latency,
                    "button_event_count": button_event_count,
                    "disconnect_reason_counts": dict(counter.get("disconnect_reason_counts", {})),
                    "panel_latency_p50_ms": latency_p50_ms,
                    "panel_latency_p95_ms": latency_p95_ms,
                    "panel_latency_p99_ms": latency_p99_ms,
                    "panel_latency_samples": len(latency_values),
                    "latest": latest_status,
                }
            )

        return {
            "generated_at": now_wall,
            "window_sec": window_sec,
            "controllers": summaries,
            "samples": samples,
            "events": events,
            "button_events": button_events,
            "panel_latency_events": panel_latency_events,
        }

    @staticmethod
    def _controller_metric_key(index: int, status: dict) -> str:
        address = str(status.get("address", "") or "").strip().lower()
        return address if address else f"index:{index}"

    def _normalize_controller_status(self, status: dict) -> dict:
        if not isinstance(status, dict):
            return self._empty_controller_status()

        pressed_buttons = status.get("pressed_buttons", [])
        if not isinstance(pressed_buttons, list):
            pressed_buttons = []

        battery_percentage = status.get("battery_percentage")
        if battery_percentage is None:
            normalized_battery = None
        else:
            try:
                parsed_battery = int(battery_percentage)
            except (TypeError, ValueError):
                normalized_battery = None
            else:
                normalized_battery = parsed_battery if 0 <= parsed_battery <= 100 else None

        last_event_monotonic: Any = status.get("last_event_monotonic")
        try:
            last_event_monotonic_value = float(last_event_monotonic)
        except (TypeError, ValueError):
            last_event_monotonic_value = None

        if last_event_monotonic_value is None:
            last_event_age_ms = None
        else:
            last_event_age_ms = max(
                0, int(round((time.monotonic() - last_event_monotonic_value) * 1000))
            )

        battery_updated_monotonic: Any = status.get("battery_updated_monotonic")
        try:
            battery_updated_monotonic_value = float(battery_updated_monotonic)
        except (TypeError, ValueError):
            battery_updated_monotonic_value = None
        if battery_updated_monotonic_value is None:
            battery_age_ms = None
        else:
            battery_age_ms = max(
                0, int(round((time.monotonic() - battery_updated_monotonic_value) * 1000))
            )

        bluetooth_metrics_updated_monotonic: Any = status.get("bluetooth_metrics_updated_monotonic")
        try:
            bluetooth_metrics_updated_monotonic_value = float(bluetooth_metrics_updated_monotonic)
        except (TypeError, ValueError):
            bluetooth_metrics_updated_monotonic_value = None
        if bluetooth_metrics_updated_monotonic_value is None:
            bluetooth_metrics_age_ms = None
        else:
            bluetooth_metrics_age_ms = max(
                0,
                int(round((time.monotonic() - bluetooth_metrics_updated_monotonic_value) * 1000)),
            )

        last_bluetooth_connect_attempt_monotonic: Any = status.get(
            "last_bluetooth_connect_attempt_monotonic"
        )
        try:
            last_bluetooth_connect_attempt_monotonic_value = float(
                last_bluetooth_connect_attempt_monotonic
            )
        except (TypeError, ValueError):
            last_bluetooth_connect_attempt_monotonic_value = None
        if last_bluetooth_connect_attempt_monotonic_value is None:
            last_bluetooth_connect_attempt_age_ms = None
        else:
            last_bluetooth_connect_attempt_age_ms = max(
                0,
                int(
                    round(
                        (time.monotonic() - last_bluetooth_connect_attempt_monotonic_value) * 1000
                    )
                ),
            )

        return {
            "enabled": bool(status.get("enabled", False)),
            "connected": bool(status.get("connected", False)),
            "address": str(status.get("address", "") or ""),
            "device_name": str(status.get("device_name", "") or ""),
            "pressed_buttons": [str(item) for item in pressed_buttons],
            "last_event_monotonic": last_event_monotonic_value,
            "last_event_age_ms": last_event_age_ms,
            "bluetooth_connect_attempts": self._normalize_int_or_none(
                status.get("bluetooth_connect_attempts")
            ),
            "bluetooth_connect_failures": self._normalize_int_or_none(
                status.get("bluetooth_connect_failures")
            ),
            "last_bluetooth_connect_attempt_monotonic": last_bluetooth_connect_attempt_monotonic_value,
            "last_bluetooth_connect_attempt_age_ms": last_bluetooth_connect_attempt_age_ms,
            "battery_percentage": normalized_battery,
            "battery_source": str(status.get("battery_source", "") or ""),
            "battery_updated_monotonic": battery_updated_monotonic_value,
            "battery_age_ms": battery_age_ms,
            "battery_poll_duration_ms": self._normalize_int_or_none(
                status.get("battery_poll_duration_ms")
            ),
            "rssi_dbm": self._normalize_int_or_none(status.get("rssi_dbm")),
            "tx_power_dbm": self._normalize_int_or_none(status.get("tx_power_dbm")),
            "link_quality": self._normalize_int_or_none(status.get("link_quality")),
            "signal_source": str(status.get("signal_source", "") or ""),
            "connection_interval_ms": self._normalize_int_or_none(
                status.get("connection_interval_ms")
            ),
            "connection_latency": self._normalize_int_or_none(status.get("connection_latency")),
            "supervision_timeout_ms": self._normalize_int_or_none(
                status.get("supervision_timeout_ms")
            ),
            "connection_params_source": str(status.get("connection_params_source", "") or ""),
            "last_disconnect_reason_code": str(status.get("last_disconnect_reason_code", "") or ""),
            "disconnect_reason_counts": self._normalize_reason_counts(
                status.get("disconnect_reason_counts", {})
            ),
            "bluetooth_metrics_updated_monotonic": bluetooth_metrics_updated_monotonic_value,
            "bluetooth_metrics_age_ms": bluetooth_metrics_age_ms,
            "bluetooth_metrics_poll_duration_ms": self._normalize_int_or_none(
                status.get("bluetooth_metrics_poll_duration_ms")
            ),
            "recent_button_events": self._normalize_button_events(
                status.get("recent_button_events", [])
            ),
        }

    @staticmethod
    def _normalize_int_or_none(value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_button_events(value) -> list[dict]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value[-50:]:
            if not isinstance(item, dict):
                continue
            raw_sequence: Any = item.get("sequence")
            try:
                sequence = int(raw_sequence)
            except (TypeError, ValueError):
                continue
            raw_monotonic: Any = item.get("monotonic")
            try:
                monotonic = float(raw_monotonic)
            except (TypeError, ValueError):
                monotonic = None
            out.append(
                {
                    "sequence": sequence,
                    "button": str(item.get("button", "") or ""),
                    "event": str(item.get("event", "") or ""),
                    "monotonic": monotonic,
                }
            )
        return out

    @staticmethod
    def _normalize_reason_counts(value) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, int] = {}
        for key, raw_count in value.items():
            reason = str(key or "").strip().lower()
            if not reason:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                out[reason] = count
        return out

    @staticmethod
    def _controller_status_signature(statuses: list[dict]) -> tuple:
        out: list[tuple] = []
        for status in statuses:
            buttons = status.get("pressed_buttons", [])
            if not isinstance(buttons, list):
                buttons = []
            out.append(
                (
                    bool(status.get("enabled", False)),
                    bool(status.get("connected", False)),
                    str(status.get("address", "") or ""),
                    str(status.get("device_name", "") or ""),
                    tuple(str(item) for item in buttons),
                    status.get("last_event_monotonic"),
                    status.get("last_event_age_ms"),
                    status.get("bluetooth_connect_attempts"),
                    status.get("bluetooth_connect_failures"),
                    status.get("last_bluetooth_connect_attempt_monotonic"),
                    status.get("last_bluetooth_connect_attempt_age_ms"),
                    status.get("battery_percentage"),
                    status.get("battery_source"),
                    status.get("battery_updated_monotonic"),
                    status.get("battery_age_ms"),
                    status.get("battery_poll_duration_ms"),
                    status.get("rssi_dbm"),
                    status.get("tx_power_dbm"),
                    status.get("link_quality"),
                    status.get("signal_source"),
                    status.get("connection_interval_ms"),
                    status.get("connection_latency"),
                    status.get("supervision_timeout_ms"),
                    status.get("connection_params_source"),
                    status.get("last_disconnect_reason_code"),
                    tuple(sorted((status.get("disconnect_reason_counts") or {}).items())),
                    status.get("bluetooth_metrics_updated_monotonic"),
                    status.get("bluetooth_metrics_age_ms"),
                    status.get("bluetooth_metrics_poll_duration_ms"),
                    tuple(
                        (
                            event.get("sequence"),
                            event.get("button"),
                            event.get("event"),
                            event.get("monotonic"),
                        )
                        for event in status.get("recent_button_events", [])
                        if isinstance(event, dict)
                    ),
                )
            )
        return tuple(out)

    @staticmethod
    def _empty_controller_status() -> dict:
        return {
            "enabled": False,
            "connected": False,
            "address": "",
            "device_name": "",
            "pressed_buttons": [],
            "last_event_monotonic": None,
            "last_event_age_ms": None,
            "bluetooth_connect_attempts": 0,
            "bluetooth_connect_failures": 0,
            "last_bluetooth_connect_attempt_monotonic": None,
            "last_bluetooth_connect_attempt_age_ms": None,
            "battery_percentage": None,
            "battery_source": "",
            "battery_updated_monotonic": None,
            "battery_age_ms": None,
            "battery_poll_duration_ms": None,
            "rssi_dbm": None,
            "tx_power_dbm": None,
            "link_quality": None,
            "signal_source": "",
            "connection_interval_ms": None,
            "connection_latency": None,
            "supervision_timeout_ms": None,
            "connection_params_source": "",
            "last_disconnect_reason_code": "",
            "disconnect_reason_counts": {},
            "bluetooth_metrics_updated_monotonic": None,
            "bluetooth_metrics_age_ms": None,
            "bluetooth_metrics_poll_duration_ms": None,
            "recent_button_events": [],
        }

    def _require_board(self) -> Any:
        if self._board is None:
            raise HTTPException(status_code=409, detail="board mode is not attached")
        return self._board

    def _require_transition_policy(self) -> Any:
        if self._transition_policy is None:
            raise HTTPException(status_code=409, detail="transition policy is not attached")
        return self._transition_policy

    def _require_font_preview(self) -> Any:
        if self._font_preview is None:
            raise HTTPException(status_code=409, detail="font preview mode is not attached")
        return self._font_preview
