import io

import numpy as np
import pytest

from app.services import image as image_service
from app.services.image import binary_from_bytes, crop


def test_crop_returns_same_array_for_square_image():
    image = np.arange(25).reshape(5, 5)
    out = crop(image)
    assert out.shape == (5, 5)
    assert np.array_equal(out, image)


def test_crop_center_crops_wide_image():
    image = np.arange(24).reshape(4, 6)
    out = crop(image)
    assert out.shape == (4, 4)
    assert np.array_equal(out, image[:, 1:5])


def test_crop_center_crops_tall_image():
    image = np.arange(24).reshape(6, 4)
    out = crop(image)
    assert out.shape == (4, 4)
    assert np.array_equal(out, image[1:5, :])


def _png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (width, height), color=200).save(buf, format="PNG")
    return buf.getvalue()


def test_binary_from_bytes_fits_small_image():
    out = binary_from_bytes(_png_bytes(64, 64), max_width=28, max_height=28, threshold=128)
    assert out.shape[0] <= 28 and out.shape[1] <= 28
    assert set(np.unique(out)).issubset({0, 1})


def test_binary_from_bytes_rejects_decompression_bomb(monkeypatch):
    # A modestly large image must be rejected before its pixels are decoded.
    monkeypatch.setattr(image_service, "_MAX_SOURCE_PIXELS", 16)
    with pytest.raises(ValueError, match="too large"):
        binary_from_bytes(_png_bytes(64, 64), max_width=28, max_height=28)
