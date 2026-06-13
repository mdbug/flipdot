from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


class PointerEventPayload(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class ActionPayload(BaseModel):
    action: str


class ButtonPayload(BaseModel):
    down: bool


class WebServer:
    """Built-in FastAPI server for frame mirroring and browser input."""

    def __init__(self, *, input_hub, host: str, port: int) -> None:
        self._input_hub = input_hub
        self._host = host
        self._port = port
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

        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self._wire_routes()

    def _wire_routes(self) -> None:
        static_dir = Path(__file__).resolve().parents[2] / "web_ui"
        self._app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @self._app.get("/")
        def ui_root() -> FileResponse:
            return FileResponse(static_dir / "index.html")

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
