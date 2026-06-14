from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from app.services.settings_store import RuntimeSettingsStore


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

    def __init__(self, *, input_hub, host: str, port: int, settings_path: Path | None = None) -> None:
        self._input_hub = input_hub
        self._host = host
        self._port = port
        self._settings_store = RuntimeSettingsStore(
            settings_path or (Path(__file__).resolve().parents[2] / "state" / "settings.json")
        )
        self._app = FastAPI()
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._frame_lock = threading.Lock()
        self._latest_frame = [[0 for _ in range(28)] for _ in range(28)]
        self._frame_width = 28
        self._frame_height = 28
        self._frame_version = 0
        self._current_mode = ""
        self._controls = []
        self._board = None
        self._transition_policy = None
        self._font_preview = None

        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._wire_routes()

    def _wire_routes(self) -> None:
        static_dir = Path(__file__).resolve().parents[2] / "web_ui"
        self._app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @self._app.get("/")
        def ui_root() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        @self._app.get("/font-grid")
        def font_grid_page() -> FileResponse:
            return FileResponse(static_dir / "font_grid.html")

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
            return JSONResponse(payload)

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
                return JSONResponse({"status": "ok", "hit": hits[0] if hits else None, "hits": hits})
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
        def patch_board_text_object(object_id: str, payload: BoardTextObjectUpdatePayload) -> JSONResponse:
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
        def patch_board_image_object(object_id: str, payload: BoardImageMovePayload) -> JSONResponse:
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
                    moved = board.move_objects([item.model_dump() for item in payload.ids], persist=False)
                    if moved is None:
                        raise HTTPException(status_code=404, detail="object not found")
                    return JSONResponse({"status": "ok", "objects": moved})

                if payload.kind is None or payload.id is None or payload.x is None or payload.y is None:
                    raise HTTPException(status_code=400, detail="drag payload requires kind/id/x/y or ids")

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
                    moved = board.move_objects([item.model_dump() for item in payload.ids], persist=True)
                    if moved is None:
                        raise HTTPException(status_code=404, detail="object not found")
                    return JSONResponse({"status": "ok", "objects": moved})

                if payload.kind is None or payload.id is None or payload.x is None or payload.y is None:
                    raise HTTPException(status_code=400, detail="drag payload requires kind/id/x/y or ids")

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
                    if version != sent_version:
                        await websocket.send_json(payload)
                        sent_version = version
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

    def publish_frame(self, frame, *, mode: str = "", controls=None) -> None:
        # Convert once here so API handlers can return cheap immutable snapshots.
        pixels = frame.astype("uint8").tolist()
        with self._frame_lock:
            self._frame_height = len(pixels)
            self._frame_width = len(pixels[0]) if pixels else 0
            self._latest_frame = pixels
            self._current_mode = mode
            self._controls = controls or []
            self._frame_version += 1

    def attach_board(self, board) -> None:
        self._board = board

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
        persisted = self._settings_store.load_font_preview_settings()
        if persisted is not None:
            font_preview.update_settings(
                phrase=str(persisted["phrase"]),
                spacing=int(persisted.get("spacing", 0)),
                variants=list(persisted.get("variants", [])),
            )

    def _require_board(self):
        if self._board is None:
            raise HTTPException(status_code=409, detail="board mode is not attached")
        return self._board

    def _require_transition_policy(self):
        if self._transition_policy is None:
            raise HTTPException(status_code=409, detail="transition policy is not attached")
        return self._transition_policy

    def _require_font_preview(self):
        if self._font_preview is None:
            raise HTTPException(status_code=409, detail="font preview mode is not attached")
        return self._font_preview
