import numpy as np

from app.modes.contracts import ModeRegistry, RenderContext


def _context(width=28, height=28):
    return RenderContext(
        frame=np.zeros((height, width), dtype=np.uint8),
        pose_results=None,
        face_mesh_results=None,
        estimated_distance=None,
        mode_time=0.0,
        panel_width=width,
        panel_height=height,
    )


def test_render_unknown_mode_returns_empty_frame():
    registry = ModeRegistry()
    context = _context(10, 6)

    out = registry.render("missing", context)

    assert out.shape == (6, 10)
    assert out.dtype == np.uint8
    assert out.sum() == 0


def test_register_and_render_invokes_renderer():
    registry = ModeRegistry()
    context = _context(8, 8)

    def renderer(ctx):
        frame = np.zeros((ctx.panel_height, ctx.panel_width), dtype=np.uint8)
        frame[0, 0] = 1
        return frame

    registry.register("clock", renderer)
    out = registry.render("clock", context)

    assert out.shape == (8, 8)
    assert out[0, 0] == 1
