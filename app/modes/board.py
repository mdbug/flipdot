import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

import app.services.fonts as fonts
import app.services.image as image_service
import app.services.text as text
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame


class Board:
    """Persistent editable board: a draw layer plus movable text and image objects.

    The web UI and MCP tools mutate it through the public methods (thread-safe via
    an internal lock); ``get_frame`` composites the draw layer, images, and text
    (including scrolling) into a 1-bit frame each tick.
    """

    DEFAULT_STATE_PATH = Path(__file__).resolve().parents[2] / "state" / "board_state.json"
    DEFAULT_BOARDS_DIR = Path(__file__).resolve().parents[2] / "state" / "boards"
    MAX_UNDO = 20
    SCHEMA_VERSION = 2
    TEXT_SIZE = 5
    TEXT_STYLE = "regular"
    TEXT_FONT = "classic"
    TEXT_SPACING = 1
    MAX_TEXT_SPACING = 6
    MAX_TEXT_LENGTH = 64
    NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self._lock = threading.Lock()
        self._draw_layer = np.zeros((height, width), dtype=np.uint8)
        self._text_objects: list[dict[str, Any]] = []
        self._image_objects: list[dict[str, Any]] = []
        self._selected_text_id = ""
        self._selected_image_id = ""
        self._selected_text_ids: set[str] = set()
        self._selected_image_ids: set[str] = set()
        self._scroll_offsets: dict[str, float] = {}
        self._last_frame_time = time.monotonic()
        self._legacy_text_id = ""
        self._next_object_seq = 1
        self._active_board = "default"
        self._undo_stack: list[dict[str, Any]] = []
        self._state_path = Path(os.getenv("BOARD_STATE_PATH", str(Board.DEFAULT_STATE_PATH)))
        default_boards_dir = self._state_path.parent / "boards"
        self._boards_dir = Path(os.getenv("BOARD_STATES_DIR", str(default_boards_dir)))
        self._supported_chars_cache: dict[tuple[str, int, str], Any] = {}
        self._load_state()

    def _new_id(self, prefix: str) -> str:
        object_id = f"{prefix}_{self._next_object_seq}"
        self._next_object_seq += 1
        return object_id

    def _sanitize_name(self, value: Any) -> str:
        name = str(value or "").strip() or "default"
        if not Board.NAME_RE.match(name):
            raise ValueError("board name must match [A-Za-z0-9_-]{1,64}")
        return name

    def _supported_chars(self, font: str, size: int, style: str) -> Any:
        key = (font, int(size), style)
        if key not in self._supported_chars_cache:
            self._supported_chars_cache[key] = text.supported_characters(
                font=font,
                sizes=(int(size),),
                styles=(style,),
            )
        return self._supported_chars_cache[key]

    def _sanitize_text(self, value, *, font=None, size=None, style=None, max_length=None):
        font = font or Board.TEXT_FONT
        size = int(size if size is not None else Board.TEXT_SIZE)
        style = style or Board.TEXT_STYLE
        raw = str(value or "")[: int(max_length or Board.MAX_TEXT_LENGTH)]
        supported = self._supported_chars(font, size, style)
        return "".join(ch for ch in raw if ch in supported)

    def _normalize_text_spec(self, spec):
        family = str(spec.get("font", Board.TEXT_FONT) or Board.TEXT_FONT)
        if family not in fonts.available_families():
            family = Board.TEXT_FONT

        requested_size = int(spec.get("size", Board.TEXT_SIZE))
        available_sizes = fonts.available_sizes(family)
        size = requested_size if requested_size in available_sizes else available_sizes[0]

        requested_style = str(spec.get("style", Board.TEXT_STYLE) or Board.TEXT_STYLE)
        available_styles = fonts.available_styles(family, size)
        style = requested_style if requested_style in available_styles else available_styles[0]

        return family, size, style

    def _normalize_text_object(self, data, *, object_id=None):
        family, size, style = self._normalize_text_spec(data)
        try:
            spacing = int(data.get("spacing", Board.TEXT_SPACING))
        except Exception:
            spacing = Board.TEXT_SPACING
        spacing = max(0, min(Board.MAX_TEXT_SPACING, spacing))
        content = self._sanitize_text(
            data.get("text", ""),
            font=family,
            size=size,
            style=style,
            max_length=data.get("max_length", Board.MAX_TEXT_LENGTH),
        )
        return {
            "id": object_id or data.get("id") or self._new_id("txt"),
            "text": content,
            "x": int(data.get("x", 0)),
            "y": int(data.get("y", 11)),
            "font": family,
            "size": size,
            "style": style,
            "spacing": spacing,
            "scroll": bool(data.get("scroll", False)),
            "scroll_speed": float(data.get("scroll_speed", 7.0)),
        }

    def _normalize_image_pixels(self, pixels):
        matrix = np.array(pixels, dtype=np.uint8)
        if matrix.ndim != 2:
            raise ValueError("image pixels must be a 2D array")
        if matrix.shape[0] > self.height or matrix.shape[1] > self.width:
            raise ValueError("image matrix is larger than panel")
        return np.where(matrix > 0, 1, 0).astype(np.uint8)

    def _normalize_image_object(self, data, *, object_id=None):
        pixels = self._normalize_image_pixels(data.get("pixels", []))
        return {
            "id": object_id or data.get("id") or self._new_id("img"),
            "x": int(data.get("x", 0)),
            "y": int(data.get("y", 0)),
            "pixels": pixels,
        }

    def _snapshot(self):
        return {
            "draw_layer": self._draw_layer.copy(),
            "text_objects": [dict(item) for item in self._text_objects],
            "image_objects": [
                {
                    "id": item["id"],
                    "x": item["x"],
                    "y": item["y"],
                    "pixels": item["pixels"].copy(),
                }
                for item in self._image_objects
            ],
            "selected_text_id": self._selected_text_id,
            "selected_image_id": self._selected_image_id,
            "selected_text_ids": list(self._selected_text_ids),
            "selected_image_ids": list(self._selected_image_ids),
            "legacy_text_id": self._legacy_text_id,
            "active_board": self._active_board,
            "next_object_seq": self._next_object_seq,
            "scroll_offsets": dict(self._scroll_offsets),
        }

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > Board.MAX_UNDO:
            self._undo_stack.pop(0)

    def _restore_snapshot(self, snapshot):
        self._draw_layer = snapshot["draw_layer"].copy()
        self._text_objects = [dict(item) for item in snapshot.get("text_objects", [])]
        self._image_objects = [
            {
                "id": item["id"],
                "x": int(item["x"]),
                "y": int(item["y"]),
                "pixels": item["pixels"].copy(),
            }
            for item in snapshot.get("image_objects", [])
        ]
        self._selected_text_id = snapshot.get("selected_text_id", "")
        self._selected_image_id = snapshot.get("selected_image_id", "")
        self._selected_text_ids = set(
            snapshot.get("selected_text_ids")
            or ([] if not self._selected_text_id else [self._selected_text_id])
        )
        self._selected_image_ids = set(
            snapshot.get("selected_image_ids")
            or ([] if not self._selected_image_id else [self._selected_image_id])
        )
        self._legacy_text_id = snapshot.get("legacy_text_id", "")
        self._active_board = snapshot.get("active_board", "default")
        self._next_object_seq = int(snapshot.get("next_object_seq", self._next_object_seq))
        self._scroll_offsets = dict(snapshot.get("scroll_offsets", {}))

    def _legacy_text_value(self):
        if not self._legacy_text_id:
            return ""
        for item in self._text_objects:
            if item["id"] == self._legacy_text_id:
                return item["text"]
        return ""

    def _board_payload(self):
        return {
            "schema_version": Board.SCHEMA_VERSION,
            "active_board": self._active_board,
            "legacy_text_id": self._legacy_text_id,
            "draw_layer": self._draw_layer.astype(np.uint8).tolist(),
            "text_objects": [dict(item) for item in self._text_objects],
            "image_objects": [
                {
                    "id": item["id"],
                    "x": item["x"],
                    "y": item["y"],
                    "pixels": item["pixels"].astype(np.uint8).tolist(),
                }
                for item in self._image_objects
            ],
            "text": self._legacy_text_value(),
            "pixels": self._draw_layer.astype(np.uint8).tolist(),
        }

    def _atomic_write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)

    def _board_path(self, name):
        return self._boards_dir / f"{name}.json"

    def _save_named_board(self, name):
        board_name = self._sanitize_name(name)
        payload = self._board_payload()
        payload["active_board"] = board_name
        self._atomic_write_json(self._board_path(board_name), payload)

    def _save_state(self):
        payload = self._board_payload()
        self._atomic_write_json(self._state_path, payload)
        self._save_named_board(self._active_board)

    def _reset_empty(self):
        self._draw_layer[:, :] = 0
        self._text_objects = []
        self._image_objects = []
        self._selected_text_id = ""
        self._selected_image_id = ""
        self._selected_text_ids = set()
        self._selected_image_ids = set()
        self._legacy_text_id = ""
        self._scroll_offsets = {}
        self._next_object_seq = 1

    def _load_board_payload(self, payload):
        self._reset_empty()

        if not isinstance(payload, dict):
            return

        if "active_board" in payload:
            try:
                self._active_board = self._sanitize_name(payload.get("active_board"))
            except ValueError:
                self._active_board = "default"

        pixel_source = payload.get("draw_layer", payload.get("pixels", []))
        loaded = np.array(pixel_source, dtype=np.uint8)
        if loaded.shape == (self.height, self.width):
            self._draw_layer[:, :] = np.where(loaded > 0, 1, 0).astype(np.uint8)

        text_objects = payload.get("text_objects", [])
        if isinstance(text_objects, list):
            for raw in text_objects:
                if not isinstance(raw, dict):
                    continue
                normalized = self._normalize_text_object(raw)
                self._text_objects.append(normalized)

        image_objects = payload.get("image_objects", [])
        if isinstance(image_objects, list):
            for raw in image_objects:
                if not isinstance(raw, dict):
                    continue
                try:
                    normalized = self._normalize_image_object(raw)
                except Exception:
                    continue
                self._image_objects.append(normalized)

        legacy_text = payload.get("text", "")
        if not self._text_objects and isinstance(legacy_text, str) and legacy_text:
            self._legacy_text_id = self._new_id("txt")
            self._text_objects.append(
                self._normalize_text_object(
                    {
                        "id": self._legacy_text_id,
                        "text": legacy_text,
                        "x": 0,
                        "y": 11,
                        "font": Board.TEXT_FONT,
                        "size": Board.TEXT_SIZE,
                        "style": Board.TEXT_STYLE,
                        "scroll": False,
                    }
                )
            )

        if not self._legacy_text_id and self._text_objects:
            self._legacy_text_id = self._text_objects[0]["id"]

        max_id = 0
        for item in self._text_objects + self._image_objects:
            suffix = str(item.get("id", "")).split("_")[-1]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
        self._next_object_seq = max(1, max_id + 1)

    def _load_state(self):
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, encoding="utf-8") as f:
                    payload = json.load(f)
                self._load_board_payload(payload)
                return

            default_board = self._board_path(self._active_board)
            if os.path.exists(default_board):
                with open(default_board, encoding="utf-8") as f:
                    payload = json.load(f)
                self._load_board_payload(payload)
                return

            self._reset_empty()
        except Exception:
            # Keep empty default state when persisted file is invalid.
            self._reset_empty()

    def _to_pixel(self, x, y):
        px = min(self.width - 1, int(max(0.0, min(1.0, float(x))) * self.width))
        py = min(self.height - 1, int(max(0.0, min(1.0, float(y))) * self.height))
        return px, py

    def _clip_pixel(self, x, y):
        return max(0, min(self.width - 1, int(x))), max(0, min(self.height - 1, int(y)))

    def _normalize_line_width(self, line_width):
        try:
            width = int(line_width)
        except Exception:
            width = 1
        return max(1, min(8, width))

    def _normalize_color(self, color):
        value = str(color or "on").strip().lower()
        if value in {"off", "erase", "0", "false", "black"}:
            return 0
        return 1

    def _paint_point(self, x, y, *, line_width=1, color=1):
        width = self._normalize_line_width(line_width)
        value = 0 if int(color) == 0 else 1
        start_x = int(x) - (width // 2)
        start_y = int(y) - (width // 2)
        end_x = start_x + width
        end_y = start_y + width

        for py in range(start_y, end_y):
            if py < 0 or py >= self.height:
                continue
            for px in range(start_x, end_x):
                if px < 0 or px >= self.width:
                    continue
                self._draw_layer[py, px] = value

    def _draw_line(self, p0, p1, *, line_width=1, color=1):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + (dx * t)))
            y = int(round(y0 + (dy * t)))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._paint_point(x, y, line_width=line_width, color=color)

    def _draw_rect(self, p0, p1, *, line_width=1, color=1):
        x0, y0 = p0
        x1, y1 = p1
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        self._draw_line((min_x, min_y), (max_x, min_y), line_width=line_width, color=color)
        self._draw_line((max_x, min_y), (max_x, max_y), line_width=line_width, color=color)
        self._draw_line((max_x, max_y), (min_x, max_y), line_width=line_width, color=color)
        self._draw_line((min_x, max_y), (min_x, min_y), line_width=line_width, color=color)

    def _draw_circle(self, p0, p1, *, line_width=1, color=1):
        cx = int(round((p0[0] + p1[0]) / 2.0))
        cy = int(round((p0[1] + p1[1]) / 2.0))
        radius = max(1, int(round(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) / 2.0)))
        for angle in range(0, 360, 2):
            rad = math.radians(angle)
            x = int(round(cx + radius * math.cos(rad)))
            y = int(round(cy + radius * math.sin(rad)))
            if 0 <= x < self.width and 0 <= y < self.height:
                self._paint_point(x, y, line_width=line_width, color=color)

    def _blit(self, frame, pixels, x, y):
        height, width = pixels.shape
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, int(x) + width)
        y1 = min(self.height, int(y) + height)
        if x1 <= x0 or y1 <= y0:
            return
        src_x0 = x0 - int(x)
        src_y0 = y0 - int(y)
        src_x1 = src_x0 + (x1 - x0)
        src_y1 = src_y0 + (y1 - y0)
        frame[y0:y1, x0:x1] = np.where(
            pixels[src_y0:src_y1, src_x0:src_x1] > 0,
            1,
            frame[y0:y1, x0:x1],
        ).astype(np.uint8)

    def _find_text(self, object_id):
        for index, item in enumerate(self._text_objects):
            if item["id"] == object_id:
                return index, item
        return -1, None

    def _find_image(self, object_id):
        for index, item in enumerate(self._image_objects):
            if item["id"] == object_id:
                return index, item
        return -1, None

    def _text_bounds(self, item):
        width = text.width(
            item["text"],
            font=item["font"],
            size=item["size"],
            style=item["style"],
            spacing=int(item.get("spacing", Board.TEXT_SPACING)),
        )
        return {
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(max(0, width)),
            "height": int(item["size"]),
        }

    def _serialize_text_object(self, item):
        payload = dict(item)
        payload["bounds"] = self._text_bounds(item)
        return payload

    def _serialize_image_object(self, item):
        return {
            "id": item["id"],
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(item["pixels"].shape[1]),
            "height": int(item["pixels"].shape[0]),
            "pixels": item["pixels"].astype(np.uint8).tolist(),
        }

    def _image_bounds(self, item):
        return {
            "x": int(item["x"]),
            "y": int(item["y"]),
            "width": int(item["pixels"].shape[1]),
            "height": int(item["pixels"].shape[0]),
        }

    def _ordered_selected_ids(self, kind):
        if kind == "text":
            selected = self._selected_text_ids
            source = self._text_objects
        else:
            selected = self._selected_image_ids
            source = self._image_objects
        return [item["id"] for item in source if item["id"] in selected]

    def _sync_selection_scalars(self):
        ordered_text = self._ordered_selected_ids("text")
        ordered_image = self._ordered_selected_ids("image")
        self._selected_text_id = ordered_text[0] if ordered_text else ""
        self._selected_image_id = ordered_image[0] if ordered_image else ""

    def _select(self, *, text_id=None, image_id=None, text_ids=None, image_ids=None):
        if text_ids is not None:
            self._selected_text_ids = {
                str(item) for item in text_ids if self._find_text(str(item))[1] is not None
            }
        elif text_id is not None:
            text_value = str(text_id)
            self._selected_text_ids = (
                {text_value} if self._find_text(text_value)[1] is not None else set()
            )
        else:
            self._selected_text_ids = set()

        if image_ids is not None:
            self._selected_image_ids = {
                str(item) for item in image_ids if self._find_image(str(item))[1] is not None
            }
        elif image_id is not None:
            image_value = str(image_id)
            self._selected_image_ids = (
                {image_value} if self._find_image(image_value)[1] is not None else set()
            )
        else:
            self._selected_image_ids = set()

        self._sync_selection_scalars()

    def _all_hits_unlocked(self, x, y):
        px, py = self._to_pixel(x, y)
        hits = []

        for item in reversed(self._text_objects):
            bounds = self._text_bounds(item)
            if bounds["width"] <= 0 or bounds["height"] <= 0:
                continue
            if (
                bounds["x"] <= px < bounds["x"] + bounds["width"]
                and bounds["y"] <= py < bounds["y"] + bounds["height"]
            ):
                hits.append(
                    {
                        "kind": "text",
                        "id": item["id"],
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                        "pixel": {"x": int(px), "y": int(py)},
                        "bounds": bounds,
                    }
                )

        for item in reversed(self._image_objects):
            bounds = self._image_bounds(item)
            if bounds["width"] <= 0 or bounds["height"] <= 0:
                continue
            if (
                bounds["x"] <= px < bounds["x"] + bounds["width"]
                and bounds["y"] <= py < bounds["y"] + bounds["height"]
            ):
                hits.append(
                    {
                        "kind": "image",
                        "id": item["id"],
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                        "pixel": {"x": int(px), "y": int(py)},
                        "bounds": bounds,
                    }
                )

        return hits

    def _hit_test_unlocked(self, x, y):
        hits = self._all_hits_unlocked(x, y)
        return hits[0] if hits else None

    def get_font_catalog(self) -> dict[str, dict[str, list[str]]]:
        """Return available font families mapped to their sizes and styles."""
        catalog: dict[str, dict[str, list[str]]] = {}
        for family in fonts.available_families():
            catalog[family] = {}
            for size in fonts.available_sizes(family):
                catalog[family][str(size)] = list(fonts.available_styles(family, size))
        return catalog

    def list_boards(self) -> dict[str, Any]:
        """Return saved board names and the active board."""
        with self._lock:
            self._boards_dir.mkdir(parents=True, exist_ok=True)
            names = []
            for entry in self._boards_dir.glob("*.json"):
                names.append(entry.stem)
            if "default" not in names:
                names.append("default")
            return {
                "boards": sorted(set(names)),
                "active": self._active_board,
            }

    def save_board(self, name: str) -> dict[str, str]:
        """Persist the current board under ``name`` and make it active."""
        with self._lock:
            board_name = self._sanitize_name(name)
            self._active_board = board_name
            self._save_state()
            return {"name": board_name, "active": self._active_board}

    def load_board(self, name: str) -> bool:
        """Load the named board (pushing an undo point); return False if it doesn't exist."""
        with self._lock:
            board_name = self._sanitize_name(name)
            path = self._board_path(board_name)
            if not os.path.exists(path):
                return False
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            self._push_undo()
            self._active_board = board_name
            self._load_board_payload(payload)
            self._active_board = board_name
            self._save_state()
            return True

    def delete_board(self, name: str) -> bool:
        """Delete a named board (never ``default``); return whether it was removed."""
        with self._lock:
            board_name = self._sanitize_name(name)
            if board_name == "default":
                return False
            path = self._board_path(board_name)
            if not os.path.exists(path):
                return False
            os.remove(path)
            if self._active_board == board_name:
                self._active_board = "default"
                self._reset_empty()
                self._save_state()
            return True

    def rename_board(self, old_name: str, new_name: str) -> bool:
        """Rename a saved board; return False if the source is missing or target exists."""
        with self._lock:
            source = self._sanitize_name(old_name)
            target = self._sanitize_name(new_name)
            source_path = self._board_path(source)
            target_path = self._board_path(target)
            if not os.path.exists(source_path):
                return False
            if os.path.exists(target_path):
                return False
            source_path.rename(target_path)
            if self._active_board == source:
                self._active_board = target
                self._save_state()
            return True

    def add_text_object(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Add a text object from ``payload``, select it, and return its serialized form."""
        with self._lock:
            self._push_undo()
            item = self._normalize_text_object(payload)
            self._text_objects.append(item)
            self._select(text_id=item["id"])
            self._save_state()
            return self._serialize_text_object(item)

    def update_text_object(
        self, object_id: str, payload: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Update a text object in place; return its serialized form or None if missing."""
        with self._lock:
            index, existing = self._find_text(object_id)
            if existing is None:
                return None

            self._push_undo()
            candidate = dict(existing)
            candidate.update(payload or {})
            normalized = self._normalize_text_object(candidate, object_id=object_id)
            self._text_objects[index] = normalized
            self._select(text_id=normalized["id"])
            self._save_state()
            return self._serialize_text_object(normalized)

    def move_text_object(
        self, object_id: str, x: int, y: int, *, persist: bool = True
    ) -> dict[str, Any] | None:
        """Move a text object to ``(x, y)``; return its serialized form or None if missing."""
        with self._lock:
            index, existing = self._find_text(object_id)
            if existing is None:
                return None
            if persist:
                self._push_undo()
            self._text_objects[index]["x"] = int(x)
            self._text_objects[index]["y"] = int(y)
            self._select(text_id=object_id)
            if persist:
                self._save_state()
            return self._serialize_text_object(self._text_objects[index])

    def delete_text_object(self, object_id: str) -> bool:
        """Delete a text object; return whether it existed."""
        with self._lock:
            index, _existing = self._find_text(object_id)
            if index < 0:
                return False
            self._push_undo()
            deleted = self._text_objects.pop(index)
            self._scroll_offsets.pop(deleted["id"], None)
            if self._legacy_text_id == deleted["id"]:
                self._legacy_text_id = ""
            self._selected_text_ids.discard(deleted["id"])
            self._sync_selection_scalars()
            self._save_state()
            return True

    def add_image_object(self, pixels: Any, x: int = 0, y: int = 0) -> dict[str, Any]:
        """Add an image object at ``(x, y)``, select it, and return its serialized form."""
        with self._lock:
            self._push_undo()
            item = self._normalize_image_object(
                {
                    "x": int(x),
                    "y": int(y),
                    "pixels": pixels,
                }
            )
            self._image_objects.append(item)
            self._select(image_id=item["id"])
            self._save_state()
            return self._serialize_image_object(item)

    def move_image_object(
        self, object_id: str, x: int, y: int, *, persist: bool = True
    ) -> dict[str, Any] | None:
        """Move an image object to ``(x, y)``; return its serialized form or None if missing."""
        with self._lock:
            index, existing = self._find_image(object_id)
            if existing is None:
                return None
            if persist:
                self._push_undo()
            self._image_objects[index]["x"] = int(x)
            self._image_objects[index]["y"] = int(y)
            self._select(image_id=object_id)
            if persist:
                self._save_state()
            return self._serialize_image_object(self._image_objects[index])

    def delete_image_object(self, object_id: str) -> bool:
        """Delete an image object; return whether it existed."""
        with self._lock:
            index, _existing = self._find_image(object_id)
            if index < 0:
                return False
            self._push_undo()
            deleted = self._image_objects.pop(index)
            self._selected_image_ids.discard(deleted["id"])
            self._sync_selection_scalars()
            self._save_state()
            return True

    def hit_test(self, x: float, y: float, *, select: bool = True, all_hits: bool = False) -> Any:
        """Return the object(s) under normalized point ``(x, y)``, optionally selecting the top hit."""
        with self._lock:
            hits = self._all_hits_unlocked(x, y)
            hit = hits[0] if hits else None

            if select:
                if hit is None:
                    self._select()
                elif hit["kind"] == "text":
                    self._select(text_id=hit["id"])
                else:
                    self._select(image_id=hit["id"])

            if all_hits:
                return [dict(item) for item in hits]
            return dict(hit) if hit is not None else None

    def move_object(
        self, kind: str, object_id: str, x: int, y: int, *, persist: bool = True
    ) -> dict[str, Any] | None:
        """Move a text or image object by ``kind``; raise on an unknown kind."""
        kind_value = str(kind or "").strip().lower()
        if kind_value == "text":
            return self.move_text_object(object_id, x, y, persist=persist)
        if kind_value == "image":
            return self.move_image_object(object_id, x, y, persist=persist)
        raise ValueError("unsupported object kind")

    def move_objects(
        self, moves: list[dict[str, Any]] | None, *, persist: bool = True
    ) -> list[dict[str, Any]] | None:
        """Move several objects atomically; return None if any target is missing."""
        normalized: list[dict[str, Any]] = []
        for move in moves or []:
            if not isinstance(move, dict):
                continue
            kind = str(move.get("kind", "")).strip().lower()
            object_id = str(move.get("id", "")).strip()
            if not object_id or kind not in {"text", "image"}:
                continue
            try:
                x = int(move.get("x", 0))
                y = int(move.get("y", 0))
            except Exception as exc:
                raise ValueError("invalid move payload") from exc
            normalized.append({"kind": kind, "id": object_id, "x": x, "y": y})

        if not normalized:
            return []

        with self._lock:
            for item in normalized:
                if item["kind"] == "text" and self._find_text(item["id"])[1] is None:
                    return None
                if item["kind"] == "image" and self._find_image(item["id"])[1] is None:
                    return None

            if persist:
                self._push_undo()

            moved = []
            selected_text = []
            selected_image = []
            for item in normalized:
                if item["kind"] == "text":
                    index, _existing = self._find_text(item["id"])
                    self._text_objects[index]["x"] = item["x"]
                    self._text_objects[index]["y"] = item["y"]
                    moved.append(self._serialize_text_object(self._text_objects[index]))
                    selected_text.append(item["id"])
                else:
                    index, _existing = self._find_image(item["id"])
                    self._image_objects[index]["x"] = item["x"]
                    self._image_objects[index]["y"] = item["y"]
                    moved.append(self._serialize_image_object(self._image_objects[index]))
                    selected_image.append(item["id"])

            self._select(text_ids=selected_text, image_ids=selected_image)
            if persist:
                self._save_state()
            return moved

    def place_uploaded_image(
        self,
        image_bytes: bytes,
        *,
        mode: str = "stamp",
        x: int = 0,
        y: int = 0,
        threshold: int = 128,
    ) -> dict[str, Any]:
        """Decode and threshold uploaded image bytes, then stamp it or add it as an object."""
        matrix = image_service.binary_from_bytes(
            image_bytes,
            max_width=self.width,
            max_height=self.height,
            threshold=int(threshold),
        )
        x = int(x)
        y = int(y)
        if mode == "object":
            return {
                "mode": "object",
                "object": self.add_image_object(matrix, x=x, y=y),
            }

        with self._lock:
            self._push_undo()
            self._blit(self._draw_layer, matrix, x, y)
            self._save_state()
        return {
            "mode": "stamp",
            "width": int(matrix.shape[1]),
            "height": int(matrix.shape[0]),
        }

    def draw_shape(
        self,
        tool: str,
        start: dict[str, float],
        end: dict[str, float],
        *,
        line_width: int = 1,
        color: str = "on",
    ) -> None:
        """Draw a line, rectangle, or circle onto the draw layer between two points."""
        tool_name = str(tool or "line").strip().lower()
        if tool_name not in {"line", "rectangle", "circle"}:
            raise ValueError("unsupported shape tool")
        with self._lock:
            self._push_undo()
            p0 = self._to_pixel(start["x"], start["y"])
            p1 = self._to_pixel(end["x"], end["y"])
            width = self._normalize_line_width(line_width)
            draw_color = self._normalize_color(color)
            if tool_name == "line":
                self._draw_line(p0, p1, line_width=width, color=draw_color)
            elif tool_name == "rectangle":
                self._draw_rect(p0, p1, line_width=width, color=draw_color)
            elif tool_name == "circle":
                self._draw_circle(p0, p1, line_width=width, color=draw_color)
            self._save_state()

    def set_text(self, value: str) -> None:
        """Set the single legacy text object's content (creating it if needed)."""
        with self._lock:
            self._push_undo()
            if not self._legacy_text_id:
                self._legacy_text_id = self._new_id("txt")
                self._text_objects.append(
                    self._normalize_text_object(
                        {
                            "id": self._legacy_text_id,
                            "text": value,
                            "x": 0,
                            "y": 11,
                            "font": Board.TEXT_FONT,
                            "size": Board.TEXT_SIZE,
                            "style": Board.TEXT_STYLE,
                            "scroll": False,
                        }
                    )
                )
            else:
                index, item = self._find_text(self._legacy_text_id)
                if item is None:
                    self._legacy_text_id = self._new_id("txt")
                    self._text_objects.append(
                        self._normalize_text_object(
                            {
                                "id": self._legacy_text_id,
                                "text": value,
                                "x": 0,
                                "y": 11,
                                "font": Board.TEXT_FONT,
                                "size": Board.TEXT_SIZE,
                                "style": Board.TEXT_STYLE,
                                "scroll": False,
                            }
                        )
                    )
                else:
                    item["text"] = self._sanitize_text(
                        value,
                        font=item["font"],
                        size=item["size"],
                        style=item["style"],
                        max_length=32,
                    )
                    self._text_objects[index] = item
            self._save_state()

    def apply_stroke(
        self, points: list[dict[str, float]] | None, *, line_width: int = 1, color: str = "on"
    ) -> None:
        """Paint a freehand stroke through ``points`` onto the draw layer."""
        normalized_points = list(points or [])
        if not normalized_points:
            return

        with self._lock:
            self._push_undo()
            width = self._normalize_line_width(line_width)
            draw_color = self._normalize_color(color)
            pixel_points = []
            for point in normalized_points:
                px, py = self._to_pixel(point["x"], point["y"])
                pixel_points.append((px, py))

            if len(pixel_points) == 1:
                x, y = pixel_points[0]
                self._paint_point(x, y, line_width=width, color=draw_color)
            else:
                for idx in range(1, len(pixel_points)):
                    self._draw_line(
                        pixel_points[idx - 1], pixel_points[idx], line_width=width, color=draw_color
                    )

            self._save_state()

    def clear(self) -> None:
        """Wipe the draw layer and all objects (pushing an undo point)."""
        with self._lock:
            self._push_undo()
            self._draw_layer[:, :] = 0
            self._text_objects = []
            self._image_objects = []
            self._selected_text_id = ""
            self._selected_image_id = ""
            self._selected_text_ids = set()
            self._selected_image_ids = set()
            self._legacy_text_id = ""
            self._scroll_offsets = {}
            self._save_state()

    def undo(self) -> bool:
        """Restore the most recent snapshot; return False if the undo stack is empty."""
        with self._lock:
            if not self._undo_stack:
                return False
            previous = self._undo_stack.pop()
            self._restore_snapshot(previous)
            self._save_state()
            return True

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the full board for the web UI."""
        with self._lock:
            return {
                "schema_version": Board.SCHEMA_VERSION,
                "text": self._legacy_text_value(),
                "width": self.width,
                "height": self.height,
                "pixels": self._draw_layer.astype(np.uint8).tolist(),
                "active_board": self._active_board,
                "text_objects": [self._serialize_text_object(item) for item in self._text_objects],
                "image_objects": [
                    self._serialize_image_object(item) for item in self._image_objects
                ],
                "selected_text_id": self._selected_text_id,
                "selected_image_id": self._selected_image_id,
                "selected_text_ids": self._ordered_selected_ids("text"),
                "selected_image_ids": self._ordered_selected_ids("image"),
            }

    def get_frame(self, pose_results: Any = None, input_hub: Any = None) -> Frame:
        """Composite the draw layer, image objects, and (scrolling) text into a 1-bit frame."""
        del pose_results
        del input_hub
        with self._lock:
            now = time.monotonic()
            dt = max(0.0, min(0.2, now - self._last_frame_time))
            self._last_frame_time = now
            frame = self._draw_layer.copy()

            for image_item in self._image_objects:
                self._blit(frame, image_item["pixels"], image_item["x"], image_item["y"])

            for item in self._text_objects:
                if not item["text"]:
                    continue
                text_width = text.width(
                    item["text"],
                    font=item["font"],
                    size=item["size"],
                    style=item["style"],
                    spacing=int(item.get("spacing", Board.TEXT_SPACING)),
                )
                render_x = int(item["x"])
                if item.get("scroll", False) and text_width > self.width:
                    cycle = float(self.width + text_width + 2)
                    next_offset = (
                        self._scroll_offsets.get(item["id"], 0.0) + dt * item["scroll_speed"]
                    ) % cycle
                    self._scroll_offsets[item["id"]] = next_offset
                    render_x = self.width - int(next_offset)

                render_y = int(item["y"])
                text_height = int(item["size"])

                # Skip objects that are entirely outside the visible panel.
                if render_x >= self.width or render_x + text_width <= 0:
                    continue
                if render_y >= self.height or render_y + text_height <= 0:
                    continue

                text.write(
                    frame,
                    item["text"],
                    x=render_x,
                    y=render_y,
                    font=item["font"],
                    size=item["size"],
                    style=item["style"],
                    spacing=int(item.get("spacing", Board.TEXT_SPACING)),
                )
            frame[:, :] = np.where(frame > 0, 1, 0).astype(np.uint8)
            return frame
