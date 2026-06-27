from io import BytesIO

import numpy as np
from PIL import Image

# Upper bound on source image dimensions accepted by ``binary_from_bytes``.
# Generous (the panel is 28x28) but small enough to reject decompression bombs.
_MAX_SOURCE_PIXELS = 8000 * 8000


def load(file: str) -> np.ndarray:
    """Load an image from the ``imgs/`` directory as a numpy array."""
    image = Image.open("imgs/" + file)
    return np.asarray(image)


def crop(image: np.ndarray) -> np.ndarray:
    """Center-crop ``image`` to a square using its shorter side."""
    if image.shape[1] > image.shape[0]:
        crop_size = image.shape[0]
        start_x = (image.shape[1] - crop_size) // 2
        croped_image = image[:, start_x : start_x + crop_size]
    elif image.shape[0] > image.shape[1]:
        crop_size = image.shape[1]
        start_y = (image.shape[0] - crop_size) // 2
        croped_image = image[start_y : start_y + crop_size, :]
    else:
        croped_image = image

    return croped_image


def binary_from_bytes(
    data: bytes, *, max_width: int, max_height: int, threshold: int = 128
) -> np.ndarray:
    """Decode image bytes, fit within max dimensions, and threshold to a 1-bit uint8 array."""
    if not data:
        raise ValueError("image payload is empty")

    with Image.open(BytesIO(data)) as raw:
        # Image.open only reads the header, so check declared dimensions before
        # decoding pixels — guards against a decompression bomb (a tiny file that
        # expands to gigabytes once decoded). The panel is only 28x28.
        source_pixels = (raw.width or 0) * (raw.height or 0)
        if source_pixels > _MAX_SOURCE_PIXELS:
            raise ValueError(
                f"image is too large to decode ({raw.width}x{raw.height} px); "
                f"maximum is {_MAX_SOURCE_PIXELS} pixels"
            )
        grayscale = raw.convert("L")
        grayscale.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.uint8)

    cut = max(0, min(255, int(threshold)))
    return np.where(pixels >= cut, 1, 0).astype(np.uint8)
