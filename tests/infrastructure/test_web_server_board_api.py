import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.infrastructure.web_server import WebServer


class DummyInputHub:
    def submit_pointer(self, source, x, y):
        pass

    def submit_click(self, source, x, y):
        pass

    def submit_action(self, source, action):
        pass

    def set_button_down(self, source, is_down):
        pass


class DummyTransitionPolicy:
    def __init__(self):
        self._settings = {
            "enabled": True,
            "start_hour": 0,
            "end_hour": 7,
        }

    def get_sleep_settings(self):
        return dict(self._settings)


class DummyFontPreview:
    def __init__(self):
        self._settings = {"phrase": "FLIPDOT", "spacing": 0, "variants": []}

    def get_settings(self):
        return dict(self._settings)

    def get_variant_catalog(self):
        return {"classic": {"5": ["regular"], "6": ["regular", "monospace"]}}

    def get_glyph_grid(self):
        return {
            "variants": [
                {
                    "family": "classic",
                    "size": 5,
                    "style": "regular",
                    "cell_width": None,
                    "glyphs": {
                        "A": [[1, 0, 1], [1, 1, 1]],
                        "B": [[1, 1], [1, 1]],
                    },
                }
            ],
            "characters": ["A", "B"],
        }

    def update_settings(self, *, phrase, variants=None, spacing=0):
        cleaned = " ".join(str(phrase).split())
        if not cleaned:
            cleaned = "FLIPDOT"
        normalized_variants = []
        if isinstance(variants, list):
            for item in variants:
                if not isinstance(item, dict):
                    continue
                family = item.get("family")
                size = item.get("size")
                style = item.get("style")
                if not isinstance(family, str) or not isinstance(style, str):
                    continue
                normalized_variants.append(
                    {
                        "family": family,
                        "size": int(size),
                        "style": style,
                    }
                )
        self._settings = {
            "phrase": cleaned[:32],
            "spacing": max(0, min(6, int(spacing))),
            "variants": normalized_variants[:4],
        }
        return dict(self._settings)

    def set_sleep_settings(self, *, enabled, start_hour, end_hour):
        self._settings = {
            "enabled": bool(enabled),
            "start_hour": int(start_hour),
            "end_hour": int(end_hour),
        }
        return dict(self._settings)


class DummyBoard:
    def __init__(self):
        self.text = ""
        self.draw_calls = 0
        self.clear_calls = 0
        self.undo_calls = 0
        self.text_objects = []
        self._next_text_id = 1
        self.boards = ["default"]
        self.active = "default"
        self.image_objects = []

    def export_state(self):
        return {
            "text": self.text,
            "width": 28,
            "height": 28,
            "pixels": [[0] * 28 for _ in range(28)],
            "selected_text_id": "",
            "selected_image_id": "",
            "selected_text_ids": [],
            "selected_image_ids": [],
        }

    def set_text(self, value):
        self.text = value

    def apply_stroke(self, points, *, line_width=1, color="on"):
        self.draw_calls += 1

    def draw_shape(self, tool, start, end, *, line_width=1, color="on"):
        self.draw_calls += 1

    def clear(self):
        self.clear_calls += 1

    def undo(self):
        self.undo_calls += 1
        return True

    def get_font_catalog(self):
        return {"classic": {"5": ["regular", "monospace"]}}

    def add_text_object(self, payload):
        object_id = f"txt_{self._next_text_id}"
        self._next_text_id += 1
        item = {"id": object_id, **payload}
        self.text_objects.append(item)
        return item

    def update_text_object(self, object_id, payload):
        for index, item in enumerate(self.text_objects):
            if item["id"] == object_id:
                self.text_objects[index] = {**item, **payload}
                return self.text_objects[index]
        return None

    def delete_text_object(self, object_id):
        for index, item in enumerate(self.text_objects):
            if item["id"] == object_id:
                self.text_objects.pop(index)
                return True
        return False

    def place_uploaded_image(self, image_bytes, mode, x, y, threshold):
        if mode == "object":
            payload = {"id": "img_1", "x": x, "y": y, "width": 2, "height": 2, "pixels": [[1, 1], [1, 1]]}
            self.image_objects.append(payload)
            return {"mode": "object", "object": payload}
        return {"mode": "stamp", "width": 2, "height": 2}

    def move_image_object(self, object_id, x, y):
        for item in self.image_objects:
            if item["id"] == object_id:
                item["x"] = x
                item["y"] = y
                return item
        return None

    def delete_image_object(self, object_id):
        for index, item in enumerate(self.image_objects):
            if item["id"] == object_id:
                self.image_objects.pop(index)
                return True
        return False

    def list_boards(self):
        return {"boards": list(self.boards), "active": self.active}

    def save_board(self, name):
        if name not in self.boards:
            self.boards.append(name)
        self.active = name
        return {"name": name, "active": self.active}

    def load_board(self, name):
        if name in self.boards:
            self.active = name
            return True
        return False

    def delete_board(self, name):
        if name in self.boards and name != "default":
            self.boards.remove(name)
            if self.active == name:
                self.active = "default"
            return True
        return False

    def rename_board(self, old_name, new_name):
        if old_name not in self.boards or new_name in self.boards:
            return False
        self.boards = [new_name if item == old_name else item for item in self.boards]
        if self.active == old_name:
            self.active = new_name
        return True

    def hit_test(self, x, y, *, select=True, all_hits=False):
        hit = {"kind": "text", "id": "txt_1", "x": 1, "y": 2, "pixel": {"x": 1, "y": 2}}
        if all_hits:
            return [hit]
        return hit

    def move_object(self, kind, object_id, x, y, persist=True):
        if kind not in {"text", "image"}:
            raise ValueError("unsupported object kind")
        if object_id == "missing":
            return None
        return {"id": object_id, "x": x, "y": y, "persist": persist}

    def move_objects(self, moves, persist=True):
        normalized = []
        for item in moves:
            if item.get("id") == "missing":
                return None
            normalized.append({"id": item.get("id"), "x": item.get("x"), "y": item.get("y"), "persist": persist})
        return normalized


def test_board_endpoints_require_attachment():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8123)
    client = TestClient(server._app)

    response = client.get("/api/board/state")

    assert response.status_code == 409

    response = client.get("/api/settings/sleep")

    assert response.status_code == 409

    response = client.get("/api/settings/font-preview")

    assert response.status_code == 409

    response = client.get("/api/font-preview/glyph-grid")

    assert response.status_code == 409


def test_board_endpoints_mutate_attached_board():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8124)
    board = DummyBoard()
    server.attach_board(board)
    client = TestClient(server._app)

    response = client.post("/api/board/text", json={"text": "HELLO"})
    assert response.status_code == 200
    assert board.text == "HELLO"

    response = client.post("/api/board/draw", json={"points": [{"x": 0.1, "y": 0.2}]})
    assert response.status_code == 200
    assert board.draw_calls == 1

    response = client.post(
        "/api/board/draw",
        json={
            "points": [{"x": 0.3, "y": 0.4}],
            "line_width": 3,
            "color": "off",
        },
    )
    assert response.status_code == 200
    assert board.draw_calls == 2

    response = client.post("/api/board/clear", json={})
    assert response.status_code == 200
    assert board.clear_calls == 1

    response = client.post("/api/board/undo", json={})
    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert board.undo_calls == 1

    response = client.get("/api/board/state")
    assert response.status_code == 200
    assert response.json()["text"] == "HELLO"


def test_font_grid_page_route_serves_html():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8126)
    client = TestClient(server._app)

    response = client.get("/font-grid")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_sleep_settings_endpoints_with_attached_transition_policy():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8130)
    server.attach_transition_policy(DummyTransitionPolicy())
    client = TestClient(server._app)

    response = client.get("/api/settings/sleep")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "start_hour": 0, "end_hour": 7}

    response = client.post(
        "/api/settings/sleep",
        json={"enabled": False, "start_hour": 22, "end_hour": 6},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["enabled"] is False
    assert response.json()["start_hour"] == 22
    assert response.json()["end_hour"] == 6

    response = client.get("/api/settings/sleep")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "start_hour": 22, "end_hour": 6}


def test_font_preview_settings_endpoints_with_attached_mode():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8133)
    server.attach_font_preview(DummyFontPreview())
    client = TestClient(server._app)

    response = client.get("/api/settings/font-preview")
    assert response.status_code == 200
    assert response.json() == {"phrase": "FLIPDOT", "spacing": 0, "variants": []}

    response = client.get("/api/font-preview/variants")
    assert response.status_code == 200
    assert "classic" in response.json()

    response = client.get("/api/font-preview/glyph-grid")
    assert response.status_code == 200
    assert response.json()["characters"] == ["A", "B"]
    assert response.json()["variants"][0]["family"] == "classic"

    response = client.post(
        "/api/settings/font-preview",
        json={
            "phrase": "   HELLO   WORLD   ",
            "spacing": 3,
            "variants": [{"family": "classic", "size": 6, "style": "monospace"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["phrase"] == "HELLO WORLD"
    assert response.json()["spacing"] == 3
    assert response.json()["variants"] == [{"family": "classic", "size": 6, "style": "monospace"}]

    response = client.get("/api/settings/font-preview")
    assert response.status_code == 200
    assert response.json() == {
        "phrase": "HELLO WORLD",
        "spacing": 3,
        "variants": [{"family": "classic", "size": 6, "style": "monospace"}],
    }


def test_sleep_settings_persist_to_json_file(tmp_path):
    settings_path = tmp_path / "settings.json"
    server = WebServer(
        input_hub=DummyInputHub(),
        host="127.0.0.1",
        port=8131,
        settings_path=settings_path,
    )
    server.attach_transition_policy(DummyTransitionPolicy())
    client = TestClient(server._app)

    response = client.post(
        "/api/settings/sleep",
        json={"enabled": False, "start_hour": 21, "end_hour": 5},
    )
    assert response.status_code == 200
    assert settings_path.exists()
    raw = settings_path.read_text(encoding="utf-8")
    assert '"sleep"' in raw
    assert '"enabled": false' in raw
    assert '"start_hour": 21' in raw
    assert '"end_hour": 5' in raw


def test_font_preview_settings_persist_to_json_file(tmp_path):
    settings_path = tmp_path / "settings.json"
    server = WebServer(
        input_hub=DummyInputHub(),
        host="127.0.0.1",
        port=8134,
        settings_path=settings_path,
    )
    server.attach_font_preview(DummyFontPreview())
    client = TestClient(server._app)

    response = client.post(
        "/api/settings/font-preview",
        json={"phrase": "FONT LAB"},
    )
    assert response.status_code == 200
    assert settings_path.exists()
    raw = settings_path.read_text(encoding="utf-8")
    assert '"font_preview"' in raw
    assert '"phrase": "FONT LAB"' in raw
    assert '"spacing": 0' in raw
    assert '"variants": []' in raw


def test_sleep_settings_load_from_json_file_on_attach(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        '{\n  "sleep": {\n    "enabled": false,\n    "start_hour": 20,\n    "end_hour": 6\n  }\n}\n',
        encoding="utf-8",
    )

    policy = DummyTransitionPolicy()
    server = WebServer(
        input_hub=DummyInputHub(),
        host="127.0.0.1",
        port=8132,
        settings_path=settings_path,
    )
    server.attach_transition_policy(policy)
    client = TestClient(server._app)

    response = client.get("/api/settings/sleep")
    assert response.status_code == 200
    assert response.json() == {"enabled": False, "start_hour": 20, "end_hour": 6}


def test_font_preview_settings_load_from_json_file_on_attach(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        (
            '{\n  "font_preview": {\n    "phrase": "LAB MODE",\n'
            '    "spacing": 2,\n'
            '    "variants": [{"family": "classic", "size": 5, "style": "regular"}]\n  }\n}\n'
        ),
        encoding="utf-8",
    )

    preview = DummyFontPreview()
    server = WebServer(
        input_hub=DummyInputHub(),
        host="127.0.0.1",
        port=8135,
        settings_path=settings_path,
    )
    server.attach_font_preview(preview)
    client = TestClient(server._app)

    response = client.get("/api/settings/font-preview")
    assert response.status_code == 200
    assert response.json() == {
        "phrase": "LAB MODE",
        "spacing": 2,
        "variants": [{"family": "classic", "size": 5, "style": "regular"}],
    }


def test_board_new_endpoints_work_with_attached_board():
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8125)
    board = DummyBoard()
    server.attach_board(board)
    client = TestClient(server._app)

    response = client.get("/api/board/fonts")
    assert response.status_code == 200
    assert "classic" in response.json()

    response = client.post(
        "/api/board/text-objects",
        json={
            "text": "HELLO",
            "x": 1,
            "y": 2,
            "font": "classic",
            "size": 5,
            "style": "regular",
            "scroll": True,
            "scroll_speed": 8,
        },
    )
    assert response.status_code == 200
    object_id = response.json()["text_object"]["id"]

    response = client.patch(
        f"/api/board/text-objects/{object_id}",
        json={"text": "WORLD", "x": 3},
    )
    assert response.status_code == 200
    assert response.json()["text_object"]["text"] == "WORLD"

    response = client.post(
        "/api/board/shapes",
        json={
            "tool": "line",
            "start": {"x": 0.1, "y": 0.1},
            "end": {"x": 0.9, "y": 0.9},
        },
    )
    assert response.status_code == 200

    response = client.post("/api/board/hit-test", json={"x": 0.1, "y": 0.2})
    assert response.status_code == 200
    assert response.json()["hit"]["id"] == "txt_1"

    response = client.post("/api/board/hit-test", json={"x": 0.1, "y": 0.2, "all_hits": True, "select": False})
    assert response.status_code == 200
    assert response.json()["hits"][0]["id"] == "txt_1"

    response = client.post(
        "/api/board/drag/move",
        json={"kind": "text", "id": "txt_1", "x": 4, "y": 5},
    )
    assert response.status_code == 200
    assert response.json()["object"]["persist"] is False

    response = client.post(
        "/api/board/drag/commit",
        json={"kind": "text", "id": "txt_1", "x": 6, "y": 7},
    )
    assert response.status_code == 200
    assert response.json()["object"]["persist"] is True

    response = client.post(
        "/api/board/drag/move",
        json={"ids": [{"kind": "text", "id": "txt_1", "x": 8, "y": 9}]},
    )
    assert response.status_code == 200
    assert response.json()["objects"][0]["id"] == "txt_1"

    response = client.post(
        "/api/board/image/upload",
        data={"mode": "stamp", "x": "0", "y": "0", "threshold": "128"},
        files={"file": ("tiny.png", b"abc", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "stamp"

    response = client.post("/api/boards/save", json={"name": "scene1"})
    assert response.status_code == 200

    response = client.get("/api/boards")
    assert response.status_code == 200
    assert "scene1" in response.json()["boards"]

    response = client.post("/api/boards/load", json={"name": "scene1"})
    assert response.status_code == 200
    assert response.json()["loaded"] is True

    response = client.post("/api/boards/rename", json={"old_name": "scene1", "new_name": "scene2"})
    assert response.status_code == 200
    assert response.json()["renamed"] is True

    response = client.post("/api/boards/delete", json={"name": "scene2"})
    assert response.status_code == 200
    assert response.json()["deleted"] is True
