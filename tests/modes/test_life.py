import numpy as np

import app.modes.life as life_module
from app.modes.life import LifeMirror


def _empty_world(width: int = 28, height: int = 28) -> LifeMirror:
    life = LifeMirror(width, height)
    life.grid = np.zeros((height, width), dtype=bool)
    life._history.clear()
    return life


def test_blinker_oscillates_with_period_two():
    life = _empty_world()
    life.grid[13:16, 14] = True  # vertical blinker

    life._step_generation()
    assert life.grid[14, 13:16].all()  # now horizontal
    assert life.grid.sum() == 3

    life._step_generation()
    assert life.grid[13:16, 14].all()  # back to vertical
    assert life.grid.sum() == 3


def test_block_still_life_is_stable():
    life = _empty_world()
    life.grid[10:12, 10:12] = True

    life._step_generation()

    assert life.grid[10:12, 10:12].all()
    assert life.grid.sum() == 4


def test_silhouette_seeding_ors_cells_in(monkeypatch):
    life = _empty_world()
    mask = np.zeros((28, 28), dtype=bool)
    mask[5:8, 5:8] = True
    monkeypatch.setattr(life_module.silhouette, "pose_to_mask", lambda pose, w, h: mask)

    life.grid[20, 20] = True
    life._seed_silhouette(object())

    assert life.grid[5:8, 5:8].all()
    assert life.grid[20, 20]  # existing cells survive


def test_no_person_leaves_world_untouched(monkeypatch):
    life = _empty_world()
    monkeypatch.setattr(life_module.silhouette, "pose_to_mask", lambda pose, w, h: None)

    life._seed_silhouette(None)

    assert not life.grid.any()


def test_stagnation_detects_empty_and_short_cycles():
    life = _empty_world()
    assert life._is_stagnant()  # empty world

    life = _empty_world()
    life.grid[10:12, 10:12] = True  # still life repeats immediately
    assert not life._is_stagnant()  # first sighting
    assert life._is_stagnant()  # repeat


def test_stagnant_world_reseeds_after_timeout(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(life_module.time, "time", lambda: now["value"])
    monkeypatch.setattr(life_module.silhouette, "pose_to_mask", lambda pose, w, h: None)

    life = _empty_world()
    for _ in range(200):
        now["value"] += LifeMirror.GENERATION_INTERVAL
        life.get_frame(None)
        if life.grid.any():
            break

    assert life.grid.any()


def test_get_frame_returns_binary_uint8_frame(monkeypatch):
    monkeypatch.setattr(life_module.silhouette, "pose_to_mask", lambda pose, w, h: None)
    life = LifeMirror(28, 28)

    frame = life.get_frame(None)

    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame)).issubset({0, 1})
