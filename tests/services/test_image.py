import numpy as np

from app.services.image import crop


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
