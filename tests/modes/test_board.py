import importlib
import os
import sys

import numpy as np


def _load_board_module(monkeypatch, tmp_path):
    monkeypatch.setenv("BOARD_STATE_PATH", str(tmp_path / "board_state.json"))
    sys.modules.pop("app.modes.board", None)
    return importlib.import_module("app.modes.board")


class DummyModeManager:
    pass


def test_board_initializes_with_empty_state(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    frame = board.get_frame(None)

    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert int(frame.sum()) == 0


def test_board_set_text_persists_and_exports(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.set_text("HELLO")
    state = board.export_state()

    assert state["text"] == "HELLO"
    assert os.path.exists(str(tmp_path / "board_state.json"))


def test_board_apply_stroke_sets_pixels(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.apply_stroke(
        [
            {"x": 0.1, "y": 0.1},
            {"x": 0.9, "y": 0.9},
        ]
    )
    frame = board.get_frame(None)

    assert frame.shape == (28, 28)
    assert int(frame.sum()) > 0


def test_board_clear_and_undo(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.apply_stroke([{"x": 0.5, "y": 0.5}])
    assert int(board.get_frame(None).sum()) > 0

    board.clear()
    assert int(board.get_frame(None).sum()) == 0

    applied = board.undo()
    assert applied is True
    assert int(board.get_frame(None).sum()) > 0


def test_board_loads_previous_state_on_init(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())
    board.set_text("STATE")
    board.apply_stroke([{"x": 0.4, "y": 0.4}])

    board2 = board_module.Board(28, 28, DummyModeManager())
    state = board2.export_state()

    assert state["text"] == "STATE"
    assert np.array(state["pixels"]).sum() > 0


def test_board_filters_unsupported_text_characters(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.set_text("A☃B")
    state = board.export_state()
    frame = board.get_frame(None)

    assert state["text"] == "AB"
    assert frame.shape == (28, 28)


def test_board_text_object_crud(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    created = board.add_text_object(
        {
            "text": "HELLO",
            "x": 2,
            "y": 10,
            "font": "classic",
            "size": 5,
            "style": "regular",
            "spacing": 2,
            "scroll": True,
            "scroll_speed": 9,
        }
    )
    assert created["text"] == "HELLO"
    assert created["spacing"] == 2

    updated = board.update_text_object(
        created["id"],
        {
            "text": "WORLD",
            "x": 4,
            "spacing": 0,
            "scroll": False,
        },
    )
    assert updated is not None
    assert updated["text"] == "WORLD"
    assert updated["x"] == 4
    assert updated["spacing"] == 0
    assert updated["scroll"] is False

    deleted = board.delete_text_object(created["id"])
    assert deleted is True
    assert board.delete_text_object(created["id"]) is False


def test_board_draw_shape_writes_pixels(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.draw_shape(
        "rectangle",
        {"x": 0.2, "y": 0.2},
        {"x": 0.8, "y": 0.8},
    )

    frame = board.get_frame(None)
    assert int(frame.sum()) > 0


def test_board_draw_shape_line_width_changes_more_pixels(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)

    thin = board_module.Board(28, 28, DummyModeManager())
    thin.draw_shape(
        "line",
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.1},
        line_width=1,
        color="on",
    )

    thick = board_module.Board(28, 28, DummyModeManager())
    thick.draw_shape(
        "line",
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.1},
        line_width=4,
        color="on",
    )

    assert int(thick.get_frame(None).sum()) > int(thin.get_frame(None).sum())


def test_board_apply_stroke_off_color_erases_pixels(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.apply_stroke(
        [
            {"x": 0.5, "y": 0.5},
        ]
    )
    before_erase = int(board.get_frame(None).sum())
    assert before_erase > 0

    board.apply_stroke(
        [{"x": 0.5, "y": 0.5}],
        line_width=1,
        color="off",
    )

    after_erase = int(board.get_frame(None).sum())
    assert after_erase < before_erase


def test_board_save_and_load_named_board(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    board.set_text("FIRST")
    save_result = board.save_board("demo")
    assert save_result["active"] == "demo"

    board.save_board("scratch")
    board.clear()
    assert board.export_state()["text"] == ""

    loaded = board.load_board("demo")
    assert loaded is True
    assert board.export_state()["text"] == "FIRST"

    boards_payload = board.list_boards()
    assert "demo" in boards_payload["boards"]


def test_board_hit_test_selects_top_object(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    text_object = board.add_text_object(
        {
            "text": "HI",
            "x": 2,
            "y": 2,
            "font": "classic",
            "size": 5,
            "style": "regular",
        }
    )
    board.add_image_object([[1, 1], [1, 1]], x=2, y=2)

    hit = board.hit_test(2 / 27, 2 / 27)
    assert hit is not None
    assert hit["kind"] == "text"
    assert hit["id"] == text_object["id"]

    state = board.export_state()
    assert state["selected_text_id"] == text_object["id"]


def test_board_drag_move_and_commit(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    text_object = board.add_text_object(
        {
            "text": "MOVE",
            "x": 1,
            "y": 1,
            "font": "classic",
            "size": 5,
            "style": "regular",
        }
    )

    moved = board.move_object("text", text_object["id"], 5, 6, persist=False)
    assert moved is not None
    assert moved["x"] == 5
    assert moved["y"] == 6

    committed = board.move_object("text", text_object["id"], 7, 8, persist=True)
    assert committed is not None
    assert committed["x"] == 7
    assert committed["y"] == 8


def test_board_export_state_includes_selection_arrays(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    text_object = board.add_text_object(
        {
            "text": "SEL",
            "x": 2,
            "y": 3,
            "font": "classic",
            "size": 5,
            "style": "regular",
        }
    )

    board.hit_test(2 / 27, 3 / 27, select=True, all_hits=False)
    state = board.export_state()

    assert state["selected_text_id"] == text_object["id"]
    assert state["selected_text_ids"] == [text_object["id"]]
    assert state["selected_image_ids"] == []


def test_board_move_objects_moves_group(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    text_object = board.add_text_object(
        {
            "text": "A",
            "x": 1,
            "y": 1,
            "font": "classic",
            "size": 5,
            "style": "regular",
        }
    )
    image_object = board.add_image_object([[1, 1], [1, 1]], x=0, y=0)

    moved = board.move_objects(
        [
            {"kind": "text", "id": text_object["id"], "x": 4, "y": 5},
            {"kind": "image", "id": image_object["id"], "x": 6, "y": 7},
        ],
        persist=True,
    )

    assert moved is not None
    assert len(moved) == 2
    state = board.export_state()
    text_state = next(item for item in state["text_objects"] if item["id"] == text_object["id"])
    image_state = next(item for item in state["image_objects"] if item["id"] == image_object["id"])
    assert text_state["x"] == 4
    assert text_state["y"] == 5
    assert image_state["x"] == 6
    assert image_state["y"] == 7


def test_board_frame_render_skips_offscreen_text_without_crash(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    created = board.add_text_object(
        {
            "text": "WIDE",
            "x": 0,
            "y": 0,
            "font": "classic",
            "size": 6,
            "style": "regular",
        }
    )

    moved = board.move_object("text", created["id"], -30, 0, persist=False)
    assert moved is not None

    frame = board.get_frame(None)
    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8


def test_board_text_spacing_changes_bounds(monkeypatch, tmp_path):
    board_module = _load_board_module(monkeypatch, tmp_path)
    board = board_module.Board(28, 28, DummyModeManager())

    narrow = board.add_text_object(
        {
            "text": "AA",
            "x": 0,
            "y": 0,
            "font": "classic",
            "size": 5,
            "style": "regular",
            "spacing": 0,
        }
    )
    wide = board.update_text_object(narrow["id"], {"spacing": 3})

    assert wide is not None
    assert wide["bounds"]["width"] > narrow["bounds"]["width"]
