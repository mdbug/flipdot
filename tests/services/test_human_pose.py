"""Tests for pure helpers in the human pose service."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import app.services.human_pose as human_pose
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


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.skipif(
    human_pose._USE_TASKS_API,
    reason="the app's real face-mesh worker would race on the shared module state",
)
def test_face_mesh_worker_failure_handling(monkeypatch):
    """Transient inference errors are survived; a persistent streak stops the
    worker and clears the cached result so consumers see "no face" rather than
    a stale face frozen forever."""
    calls = []
    fail = {"active": True}

    def fake_process(frame, ts):
        calls.append(ts)
        if fail["active"]:
            raise RuntimeError("boom")
        return SimpleNamespace(face_landmarks=[])

    monkeypatch.setattr(human_pose, "_face_mesh_process", fake_process)
    with human_pose._face_bg_lock:
        human_pose._face_bg_frame[0] = None
        human_pose._face_bg_result[0] = None

    worker = threading.Thread(target=human_pose._face_mesh_bg_worker, daemon=True)
    worker.start()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def submit():
        with human_pose._face_bg_lock:
            human_pose._face_bg_frame[0] = frame
        human_pose._face_bg_event.set()

    # One transient failure does not kill the worker; the next frame succeeds.
    submit()
    assert _wait_until(lambda: len(calls) == 1)
    fail["active"] = False
    submit()
    assert _wait_until(lambda: len(calls) == 2)
    assert _wait_until(lambda: human_pose._face_bg_result[0] is not None)

    # A full consecutive-failure streak stops the worker and clears the result.
    fail["active"] = True
    for i in range(human_pose._FACE_MESH_MAX_CONSECUTIVE_FAILURES):
        submit()
        expected = 3 + i
        assert _wait_until(lambda n=expected: len(calls) >= n)
    assert _wait_until(lambda: not worker.is_alive())
    with human_pose._face_bg_lock:
        assert human_pose._face_bg_result[0] is None
