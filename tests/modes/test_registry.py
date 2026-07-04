import importlib
import sys
import types

import numpy as np


def _load_registry_module(monkeypatch):
    # registry binds these via `import app.services.x as x`, i.e. attribute access on
    # the package, so stub both sys.modules and the package attributes. Patching only
    # sys.modules leaks the real module whenever it was imported earlier in the suite.
    services_pkg = importlib.import_module("app.services")

    human_pose_stub = types.SimpleNamespace(
        display_human_pose=lambda pose, w, h, dist, face: np.ones((h, w), dtype=np.uint8)
    )
    transition_stub = types.SimpleNamespace(
        blend=lambda a, b, alpha: b,
    )

    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setitem(sys.modules, "app.services.transition", transition_stub)
    monkeypatch.setattr(services_pkg, "human_pose", human_pose_stub, raising=False)
    monkeypatch.setattr(services_pkg, "transition", transition_stub, raising=False)
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


def _build_registry(registry_module, *, mode_blend_seconds=1.0, **mode_overrides):
    """Build a registry with blank fakes for every mode not overridden."""
    blank = _FakeMode(np.zeros((28, 28), dtype=np.uint8))
    kwargs = {
        name: blank
        for name in (
            "clock",
            "menu",
            "paint",
            "caricature",
            "percussion",
            "autodrum",
            "beatmirror",
            "tetris_game",
            "pong_game",
            "tank_game",
            "worldcup",
            "board",
            "font_preview",
            "script_mode",
            "life",
            "sandfall",
        )
    }
    kwargs.update(mode_overrides)
    return registry_module.build_mode_registry(
        img_sleep=np.zeros((28, 28), dtype=np.uint8),
        mode_blend_seconds=mode_blend_seconds,
        **kwargs,
    )


def test_build_mode_registry_maps_core_renderers(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    sleep = np.zeros((28, 28), dtype=np.uint8)
    registry = _build_registry(
        registry_module,
        mode_blend_seconds=0.5,
        menu=_FakeMode(np.full((28, 28), 2, dtype=np.uint8)),
        paint=_FakeMode(np.full((28, 28), 3, dtype=np.uint8)),
        worldcup=_FakeMode(np.full((28, 28), 10, dtype=np.uint8)),
        board=_FakeMode(np.full((28, 28), 11, dtype=np.uint8)),
        font_preview=_FakeMode(np.full((28, 28), 12, dtype=np.uint8)),
        script_mode=_FakeMode(np.full((28, 28), 13, dtype=np.uint8)),
        life=_FakeMode(np.full((28, 28), 15, dtype=np.uint8)),
        sandfall=_FakeMode(np.full((28, 28), 16, dtype=np.uint8)),
    )

    # mode_time past the blend window, so every render shows the raw frame.
    c = _ctx(mode_time=1.0)
    assert np.array_equal(registry.render(mm.ModeManager.MODE_SLEEP, c), sleep)
    assert registry.render(mm.ModeManager.MODE_MENU, c)[0, 0] == 2
    assert registry.render(mm.ModeManager.MODE_PAINT, c)[0, 0] == 3
    assert registry.render(mm.ModeManager.MODE_WORLDCUP, c)[0, 0] == 10
    assert registry.render(mm.ModeManager.MODE_BOARD, c)[0, 0] == 11
    assert registry.render(mm.ModeManager.MODE_FONT_PREVIEW, c)[0, 0] == 12
    assert registry.render(mm.ModeManager.MODE_SCRIPT, c)[0, 0] == 13
    assert registry.render(mm.ModeManager.MODE_LIFE, c)[0, 0] == 15
    assert registry.render(mm.ModeManager.MODE_SANDFALL, c)[0, 0] == 16


def test_pose_and_clock_transition_paths(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    registry = _build_registry(registry_module)

    pose_out = registry.render(mm.ModeManager.MODE_POSE, _ctx(mode_time=0.1))
    clock_out = registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=0.1))

    assert pose_out.shape == (28, 28)
    assert clock_out.shape == (28, 28)


def test_cross_blend_engaged_on_mode_change(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    # Make the blend identifiable: return the previous frame (the first argument).
    registry_module.transition.blend = lambda a, b, alpha: a

    registry = _build_registry(
        registry_module,
        clock=_FakeMode(np.full((28, 28), 1, dtype=np.uint8)),
        script_mode=_FakeMode(np.full((28, 28), 13, dtype=np.uint8)),
    )

    # Settle on the clock (first-ever render, shown raw).
    assert registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=2.0))[0, 0] == 1
    # Inside the blend window the script blends with the previous clock frame.
    during = registry.render(mm.ModeManager.MODE_SCRIPT, _ctx(mode_time=0.1))
    assert during[0, 0] == 1
    # After it, the raw script frame is shown.
    after = registry.render(mm.ModeManager.MODE_SCRIPT, _ctx(mode_time=2.0))
    assert after[0, 0] == 13


def test_first_render_has_no_blend(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    calls = []
    registry_module.transition.blend = lambda a, b, alpha: calls.append(alpha) or b

    registry = _build_registry(
        registry_module, clock=_FakeMode(np.full((28, 28), 1, dtype=np.uint8))
    )

    out = registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=0.0))
    assert out[0, 0] == 1
    assert calls == []


def test_no_blend_when_mode_unchanged(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    calls = []
    registry_module.transition.blend = lambda a, b, alpha: calls.append(alpha) or b

    registry = _build_registry(registry_module)

    registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=0.1))
    registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=0.2))
    assert calls == []


def test_blend_alpha_matches_mode_time_fraction(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    alphas = []
    registry_module.transition.blend = lambda a, b, alpha: alphas.append(alpha) or b

    registry = _build_registry(registry_module, mode_blend_seconds=2.0)

    registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=5.0))
    registry.render(mm.ModeManager.MODE_SCRIPT, _ctx(mode_time=0.5))
    assert alphas == [0.25]


def test_mode_change_mid_blend_blends_from_displayed_frame(monkeypatch):
    registry_module = _load_registry_module(monkeypatch)
    mm = importlib.import_module("app.core.mode_manager")

    # Blend keeps the previous frame, so the displayed frame stays whatever
    # was on screen before the change.
    registry_module.transition.blend = lambda a, b, alpha: a

    registry = _build_registry(
        registry_module,
        clock=_FakeMode(np.full((28, 28), 1, dtype=np.uint8)),
        script_mode=_FakeMode(np.full((28, 28), 13, dtype=np.uint8)),
        menu=_FakeMode(np.full((28, 28), 2, dtype=np.uint8)),
    )

    registry.render(mm.ModeManager.MODE_CLOCK, _ctx(mode_time=5.0))
    # Clock -> script: displays the clock frame mid-blend.
    assert registry.render(mm.ModeManager.MODE_SCRIPT, _ctx(mode_time=0.1))[0, 0] == 1
    # Script -> menu mid-blend: the new blend source is the *displayed*
    # (still clock-valued) frame, not the raw script frame.
    assert registry.render(mm.ModeManager.MODE_MENU, _ctx(mode_time=0.1))[0, 0] == 1
