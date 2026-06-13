import importlib
import sys
import types

import numpy as np


def _load_registry_module(monkeypatch):
    human_pose_stub = types.SimpleNamespace(
        display_human_pose=lambda pose, w, h, dist, face: np.ones((h, w), dtype=np.uint8)
    )
    transition_stub = types.SimpleNamespace(
        blend=lambda a, b, alpha: b,
        resolve=lambda dots, alpha: dots,
    )

    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setitem(sys.modules, "app.services.transition", transition_stub)
    sys.modules.pop("app.modes.registry", None)
    return importlib.import_module("app.modes.registry")


def _ctx(width=28, height=28, mode_time=0.0):
    contracts = importlib.import_module("app.modes.contracts")
    return contracts.RenderContext(
        frame=np.zeros((height, width), dtype=np.uint8),
        pose_results=None,
        face_mesh_results=None,
        estimated_distance=None,
        mode_time=mode_time,
        panel_width=width,
        panel_height=height,
        input_hub=None,
    )


class _FakeMode:
    def __init__(self, frame):
        self._frame = frame

    def get_frame(self, *args, **kwargs):
        return self._frame.copy()


def test_build_mode_registry_maps_core_renderers(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    clock = _FakeMode(np.ones((28, 28), dtype=np.uint8))
    menu = _FakeMode(np.full((28, 28), 2, dtype=np.uint8))
    paint = _FakeMode(np.full((28, 28), 3, dtype=np.uint8))
    caricature = _FakeMode(np.full((28, 28), 4, dtype=np.uint8))
    percussion = _FakeMode(np.full((28, 28), 5, dtype=np.uint8))
    autodrum = _FakeMode(np.full((28, 28), 6, dtype=np.uint8))
    beatmirror = _FakeMode(np.full((28, 28), 7, dtype=np.uint8))
    tetris = _FakeMode(np.full((28, 28), 8, dtype=np.uint8))
    pong = _FakeMode(np.full((28, 28), 9, dtype=np.uint8))
    worldcup = _FakeMode(np.full((28, 28), 10, dtype=np.uint8))
    board = _FakeMode(np.full((28, 28), 11, dtype=np.uint8))
    sleep = np.zeros((28, 28), dtype=np.uint8)

    registry = registry_module.build_mode_registry(
        clock=clock,
        menu=menu,
        paint=paint,
        caricature=caricature,
        percussion=percussion,
        autodrum=autodrum,
        beatmirror=beatmirror,
        tetris_game=tetris,
        pong_game=pong,
        worldcup=worldcup,
        board=board,
        img_sleep=sleep,
        clock_resolve_time=0.5,
        clock_disolve_time=0.5,
    )

    c = _ctx(mode_time=1.0)
    assert np.array_equal(registry.render(mm.ModeManager.MODE_SLEEP, c), sleep)
    assert registry.render(mm.ModeManager.MODE_MENU, c)[0, 0] == 2
    assert registry.render(mm.ModeManager.MODE_PAINT, c)[0, 0] == 3
    assert registry.render(mm.ModeManager.MODE_WORLDCUP, c)[0, 0] == 10
    assert registry.render(mm.ModeManager.MODE_BOARD, c)[0, 0] == 11


def test_pose_and_clock_transition_paths(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    clock = _FakeMode(np.zeros((28, 28), dtype=np.uint8))
    blank = _FakeMode(np.zeros((28, 28), dtype=np.uint8))
    registry = registry_module.build_mode_registry(
        clock=clock,
        menu=blank,
        paint=blank,
        caricature=blank,
        percussion=blank,
        autodrum=blank,
        beatmirror=blank,
        tetris_game=blank,
        pong_game=blank,
        worldcup=blank,
        board=blank,
        img_sleep=np.zeros((28, 28), dtype=np.uint8),
        clock_resolve_time=1.0,
        clock_disolve_time=1.0,
    )

    pose_out = registry.render(mm.ModeManager.MODE_POSE, _ctx(mode_time=0.1))
    clock_out = registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=0.1))

    assert pose_out.shape == (28, 28)
    assert clock_out.shape == (28, 28)
