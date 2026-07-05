"""Tests for pure helpers in the human pose service."""

import numpy as np

from app.services.human_pose import _normalize_segmentation_mask


def test_cpu_float_mask_passes_through_as_copy():
    seg = np.random.rand(60, 60).astype(np.float32)

    out = _normalize_segmentation_mask(seg)

    assert out.shape == (60, 60)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, seg)
    assert out is not seg  # detached from MediaPipe's internal buffer


def test_gpu_rgba_uint8_mask_is_reduced_and_scaled():
    """The GPU delegate returns (H, W, 4) uint8 with confidence in channel 0."""
    seg = np.zeros((60, 60, 4), dtype=np.uint8)
    seg[10:20, 10:20, 0] = 255
    seg[10:20, 10:20, 3] = 255  # alpha mirrors the confidence

    out = _normalize_segmentation_mask(seg)

    assert out.shape == (60, 60)
    assert out.dtype == np.float32
    assert out.max() == 1.0
    assert out[15, 15] == 1.0
    assert out[0, 0] == 0.0
    # A 0.5 threshold (silhouette.SEGMENTATION_THRESHOLD) must not fire on
    # faint soft-edge pixels that uint8 comparison would have let through.
    seg[30, 30, 0] = 10
    assert _normalize_segmentation_mask(seg)[30, 30] < 0.5
