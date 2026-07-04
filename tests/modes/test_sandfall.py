import types

import numpy as np

import app.modes.sandfall as sandfall_module
from app.modes.sandfall import Sandfall


def _make(width: int = 28, height: int = 28) -> Sandfall:
    return Sandfall(width, height)


def _ctx(pose_results=None, face_mesh_results=None):
    """Minimal RenderContext stand-in with the fields sandfall reads."""
    return types.SimpleNamespace(pose_results=pose_results, face_mesh_results=face_mesh_results)


def test_grain_falls_one_cell_per_tick():
    sf = _make()
    sf.sand[0, 5] = True

    sf._physics_tick()

    assert not sf.sand[0, 5]
    assert sf.sand[1, 5]


def test_grain_rests_on_the_floor():
    sf = _make()
    sf.sand[27, 5] = True

    sf._physics_tick()

    assert sf.sand[27, 5]


def test_blocked_grain_slides_diagonally():
    sf = _make()
    sf.sand[27, 5] = True
    sf.sand[26, 5] = True

    sf._physics_tick()

    assert not sf.sand[26, 5]
    assert sf.sand[27, 4] or sf.sand[27, 6]
    assert sf.sand.sum() == 2


def test_fully_blocked_grain_rests():
    sf = _make()
    sf.sand[27, 4:7] = True
    sf.sand[26, 5] = True

    sf._physics_tick()

    assert sf.sand[26, 5]


def test_free_fall_distance_accumulates():
    sf = _make()
    sf.sand[0, 5] = True

    for _ in range(3):
        sf._physics_tick()

    assert sf.sand[3, 5]
    assert sf._fall_dist[3, 5] == 3


def test_hard_landing_bounces_grain_up():
    sf = _make()
    sf.BOUNCE_CHANCE = 1.0
    sf.sand[27, 5] = True
    sf._fall_dist[27, 5] = sf.BOUNCE_MIN_FALL

    sf._physics_tick()

    assert not sf.sand[27, 5]
    assert sf.sand[26, 4:7].any()
    assert sf.sand.sum() == 1


def test_gentle_landing_does_not_bounce():
    sf = _make()
    sf.BOUNCE_CHANCE = 1.0
    sf.sand[27, 5] = True
    sf._fall_dist[27, 5] = sf.BOUNCE_MIN_FALL - 1

    sf._physics_tick()

    assert sf.sand[27, 5]


def test_bounce_climbs_one_cell_per_tick_to_its_peak():
    sf = _make()
    sf.BOUNCE_CHANCE = 1.0
    sf.sand[27, 5] = True
    sf._fall_dist[27, 5] = sf.BOUNCE_MIN_FALL

    sf._physics_tick()
    assert sf.sand[26, 4:7].any()  # immediate hop
    sf._physics_tick()
    assert sf.sand[25, 4:7].any()  # keeps climbing to BOUNCE_HEIGHT
    assert sf.sand.sum() == 1


def test_long_fall_bounces_higher():
    sf = _make()
    sf.BOUNCE_CHANCE = 1.0
    sf.sand[27, 5] = True
    sf._fall_dist[27, 5] = sf.BOUNCE_HIGH_FALL

    for _ in range(sf.BOUNCE_HEIGHT_HIGH):
        sf._physics_tick()

    assert sf.sand[27 - sf.BOUNCE_HEIGHT_HIGH, 4:7].any()
    assert sf.sand.sum() == 1


def test_bounces_decay_and_grain_rests():
    sf = _make()
    sf.BOUNCE_CHANCE = 1.0
    sf.sand[27, 5] = True
    sf._fall_dist[27, 5] = sf.BOUNCE_HIGH_FALL

    for _ in range(20):
        sf._physics_tick()

    assert sf.sand.sum() == 1
    assert sf.sand[27].any()
    assert not sf._rise.any()


def test_diagonal_slide_resets_fall_distance():
    sf = _make()
    sf.sand[27, 5] = True
    sf.sand[26, 5] = True
    sf._fall_dist[26, 5] = 2  # below the bounce threshold, so it slides

    sf._physics_tick()

    assert sf._fall_dist[27, 4] == 0 and sf._fall_dist[27, 6] == 0


def test_grain_piles_on_obstacle_mask():
    sf = _make()
    sf.mask[10, 4:7] = True
    sf.sand[9, 5] = True

    sf._physics_tick()

    assert sf.sand[9, 5]


def test_new_mask_displaces_overlapping_grains_upward():
    sf = _make()
    sf.sand[10, 5] = True
    mask = np.zeros((28, 28), dtype=bool)
    mask[10, 5] = True

    sf._set_mask(mask)

    assert not sf.sand[10, 5]
    assert sf.sand[9, 5]


def test_clearing_mask_lets_piled_sand_fall():
    sf = _make()
    sf.mask[10, 5] = True
    sf.sand[9, 5] = True

    sf._set_mask(None)
    sf._physics_tick()

    assert not sf.sand[9, 5]
    assert sf.sand[10, 5]


def test_draining_removes_bottom_row_grains():
    sf = _make()
    sf.sand[27, 5] = True

    sf._physics_tick(draining=True)

    assert not sf.sand.any()


def test_get_frame_is_binary_and_draws_mask_outline(monkeypatch):
    mask = np.zeros((28, 28), dtype=bool)
    mask[10:17, 10:17] = True
    monkeypatch.setattr(sandfall_module.silhouette, "pose_to_mask", lambda pose, w, h: mask)
    sf = _make()

    frame = sf.get_frame(_ctx(pose_results=object()))

    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame)).issubset({0, 1})
    # Only the silhouette's edge is drawn; its interior stays dark.
    assert frame[10, 10:17].all() and frame[16, 10:17].all()
    assert frame[10:17, 10].all() and frame[10:17, 16].all()
    assert not frame[11:16, 11:16].any()


def test_get_frame_applies_face_renderer(monkeypatch):
    monkeypatch.setattr(sandfall_module.silhouette, "pose_to_mask", lambda pose, w, h: None)
    calls = {}

    def renderer(frame, face_mesh_results, width, height):
        calls["args"] = (face_mesh_results, width, height)
        frame[0, 0] = 1
        return frame

    sf = Sandfall(28, 28, face_renderer=renderer)
    results = object()

    frame = sf.get_frame(_ctx(face_mesh_results=results))

    assert calls["args"] == (results, 28, 28)
    assert frame[0, 0] == 1


def test_overfull_panel_triggers_drain(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(sandfall_module.time, "time", lambda: now["value"])
    monkeypatch.setattr(sandfall_module.silhouette, "pose_to_mask", lambda pose, w, h: None)
    sf = _make()
    sf.sand[14:, :] = True  # half the panel — over MAX_FILL_FRACTION

    sf.get_frame(_ctx())

    assert sf._draining_until > now["value"]
