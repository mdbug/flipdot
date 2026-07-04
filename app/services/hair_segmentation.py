"""Hair segmentation via MediaPipe's multiclass selfie segmenter.

Provides a single entry point, :func:`get_hair_mask`, that returns the latest
boolean hair mask for the camera frame. Inference runs on a background daemon
thread (mirroring the face-mesh worker in :mod:`app.services.human_pose`) and
submissions are throttled, so callers may invoke it every rendered frame.

The segmenter initializes lazily on first use: importing this module never
imports mediapipe, so it stays cheap and import-safe on machines without it.
When mediapipe or the model file (``selfie_multiclass_256x256.tflite``) is
unavailable the service logs one warning, permanently disables itself, and
returns None — the caricature then simply renders without hair.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HAIR_SEGMENT_MAX_FPS = float(os.getenv("HAIR_SEGMENT_MAX_FPS", "7"))

_HAIR_CLASS_INDEX = 1  # selfie_multiclass classes: 0=bg, 1=hair, 2=body, 3=face, 4=clothes
_MODEL_INPUT_SIZE = 256
_MODEL_FILENAME = "selfie_multiclass_256x256.tflite"

# Model directory resolution mirrors app.services.human_pose (not imported here:
# importing it would trigger its import-time pose-model initialization).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR_CANDIDATES = [
    os.getenv("MEDIAPIPE_MODELS_DIR"),
    os.path.join(_REPO_ROOT, "models"),
    os.path.join(os.path.dirname(__file__), "models"),
]
_MODELS_DIR = next(
    (p for p in _MODELS_DIR_CANDIDATES if p and os.path.isdir(p)),
    os.path.join(_REPO_ROOT, "models"),
)
_MODEL_PATH = os.path.join(_MODELS_DIR, _MODEL_FILENAME)

_state_lock = threading.Lock()
_started = False
_disabled = False
_segmenter: Any = None

# Latest-frame slot shared with the worker (same pattern as the face-mesh thread).
_bg_lock = threading.Lock()
_bg_frame: list[np.ndarray | None] = [None]
_bg_result: list[np.ndarray | None] = [None]
_bg_event = threading.Event()

_submit_interval = 0.0 if HAIR_SEGMENT_MAX_FPS <= 0 else 1.0 / HAIR_SEGMENT_MAX_FPS
_last_submit = 0.0


def _should_submit(now_mono: float, last_submit: float, interval: float) -> bool:
    """Return whether enough time has passed since ``last_submit`` to submit again."""
    return interval == 0.0 or now_mono - last_submit >= interval


def _disable(reason: str) -> None:
    """Permanently disable the service, warning once."""
    global _disabled
    if not _disabled:
        _disabled = True
        logger.warning(
            "Hair segmentation unavailable (%s); caricature renders without hair", reason
        )


def _hair_bg_worker() -> None:
    """Background thread: segment the latest submitted frame without blocking the loop."""
    ts = 0
    import mediapipe as mp  # resolved by _ensure_started before the thread launches

    while True:
        _bg_event.wait()
        _bg_event.clear()
        with _bg_lock:
            frame = _bg_frame[0]
            _bg_frame[0] = None
        if frame is None:
            continue
        try:
            small = cv2.resize(
                frame, (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA
            )
            inp = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inp)
            ts = max(ts + 1, int(time.monotonic() * 1000))
            result = _segmenter.segment_for_video(mp_image, ts)
            mask = result.category_mask.numpy_view() == _HAIR_CLASS_INDEX
        except Exception as err:  # a wedged model must never crash the loop
            _disable(f"segmentation failed: {err}")
            return
        with _bg_lock:
            _bg_result[0] = mask


def _ensure_started() -> bool:
    """Lazily create the segmenter and worker thread; return whether the service is usable."""
    global _started, _segmenter
    if _started:
        return not _disabled
    with _state_lock:
        if _started:
            return not _disabled
        try:
            if not os.path.isfile(_MODEL_PATH):
                raise FileNotFoundError(f"model not found: {_MODEL_PATH}")
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                ImageSegmenter,
                ImageSegmenterOptions,
                RunningMode,
            )

            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                output_category_mask=True,
            )
            _segmenter = ImageSegmenter.create_from_options(options)
            threading.Thread(target=_hair_bg_worker, daemon=True).start()
            logger.info("Hair segmentation ready (model=%s)", _MODEL_PATH)
        except Exception as err:
            _disable(str(err))
        finally:
            _started = True
    return not _disabled


def get_hair_mask(frame: np.ndarray | None) -> np.ndarray | None:
    """Submit ``frame`` for hair segmentation (throttled) and return the latest mask.

    Returns a ``(256, 256)`` boolean mask in the raw (unmirrored) camera raster
    orientation, or None until the first result arrives or when the service is
    disabled. Safe to call every rendered frame.
    """
    global _last_submit
    if frame is None or not _ensure_started():
        return None
    now_mono = time.monotonic()
    if _should_submit(now_mono, _last_submit, _submit_interval):
        _last_submit = now_mono
        with _bg_lock:
            _bg_frame[0] = frame
        _bg_event.set()
    with _bg_lock:
        return _bg_result[0]
