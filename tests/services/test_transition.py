import numpy as np
import pytest

from app.services.transition import blend, disolve, resolve


def test_disolve_rejects_invalid_alpha():
    frame = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        disolve(frame.copy(), -0.1)
    with pytest.raises(ValueError):
        disolve(frame.copy(), 1.1)


def test_blend_rejects_invalid_alpha():
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        blend(a, b, -0.2)
    with pytest.raises(ValueError):
        blend(a, b, 1.2)


def test_blend_with_alpha_extremes():
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.ones((3, 3), dtype=np.uint8)

    out0 = blend(a, b, 0.0)
    out1 = blend(a, b, 1.0)

    assert np.array_equal(out0, a)
    assert np.array_equal(out1, b)


def test_resolve_matches_disolve_with_complementary_alpha():
    a = np.ones((6, 6), dtype=np.uint8)
    b = np.ones((6, 6), dtype=np.uint8)

    np.random.seed(42)
    out_resolve = resolve(a, 0.3)

    np.random.seed(42)
    out_disolve = disolve(b, 0.7)

    assert np.array_equal(out_resolve, out_disolve)
    assert out_resolve.shape == (6, 6)
