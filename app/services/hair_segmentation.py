"""Hair segmentation via MediaPipe's dedicated hair segmenter model.

Provides a single entry point, :func:`get_hair_mask`, that returns the latest
boolean hair mask for the camera frame. Inference runs on a background daemon
thread (mirroring the face-mesh worker in :mod:`app.services.human_pose`) and
submissions are throttled, so callers may invoke it every rendered frame.

The segmenter prefers the GPU delegate (falling back to CPU on the stock pip
wheel) and initializes lazily on first use: importing this module never
imports mediapipe, so it stays cheap and import-safe on machines without it.
Creation happens on the worker thread because the GPU delegate's one-time
shader compilation takes seconds and must not stall the render loop. When
mediapipe or the model file (``hair_segmenter.tflite``) is unavailable the
service logs one warning, permanently disables itself, and returns None — the
caricature then simply renders without hair. Transient per-frame inference
errors are logged and survived; only a persistent run of consecutive failures
(a genuinely wedged model) disables the service.
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

_HAIR_CLASS_INDEX = 1  # hair_segmenter classes: 0=background, 1=hair
_MODEL_INPUT_SIZE = 512
_MODEL_FILENAME = "hair_segmenter.tflite"
# A single bad frame must not kill hair for the process lifetime; only this
# many consecutive inference failures count as a wedged model.
_MAX_CONSECUTIVE_FAILURES = 10
# An intermittently failing segmenter (never 10 in a row) must not flood the
# log at the worker rate: per-frame failures are aggregated into one warning
# per interval.
_FAILURE_WARN_INTERVAL_SEC = 30.0

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
_bg_result_time: list[float] = [0.0]
_bg_event = threading.Event()

_submit_interval = 0.0 if HAIR_SEGMENT_MAX_FPS <= 0 else 1.0 / HAIR_SEGMENT_MAX_FPS
# Results older than this are stale (previous viewer/session) and not served.
# Scaled to the configured cadence so a slow (sub-1-Hz) HAIR_SEGMENT_MAX_FPS
# does not expire every healthy result between submissions.
_RESULT_MAX_AGE_SEC = max(1.0, 2.0 * _submit_interval)
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


def _hair_bg_worker(create_segmenter: Any, delegates: Any) -> None:
    """Background thread: create the segmenter, then segment submitted frames.

    Creation lives here (not in :func:`_ensure_started`) because the GPU
    delegate's one-time shader compilation takes seconds and must not stall
    the render loop; until it finishes, :func:`get_hair_mask` returns None.

    Args:
        create_segmenter: callable building an ImageSegmenter for a delegate.
        delegates: the ``BaseOptions.Delegate`` enum namespace.
    """
    global _segmenter
    ts = 0
    consecutive_failures = 0
    failures_since_warn = 0
    last_failure_warn: float | None = None
    import mediapipe as mp  # resolved by _ensure_started before the thread launches

    try:
        # GPU-first with CPU fallback, mirroring app.services.human_pose:
        # the stock pip wheel has no GPU delegate, so dev machines land on CPU.
        try:
            _segmenter = create_segmenter(delegates.GPU)
            delegate_name = "GPU"
        except Exception:
            logger.warning("Hair segmentation GPU delegate unavailable, falling back to CPU")
            _segmenter = create_segmenter(delegates.CPU)
            delegate_name = "CPU"
    except Exception as err:
        _disable(f"segmenter creation failed: {err}")
        return
    logger.info("Hair segmentation ready (%s delegate, model=%s)", delegate_name, _MODEL_PATH)

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
        except Exception as err:
            # One malformed frame must not kill hair for the process
            # lifetime; only a persistent run of failures (a wedged model)
            # disables the service.
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                _disable(f"segmentation failed {consecutive_failures}x consecutively: {err}")
                return
            failures_since_warn += 1
            now_mono = time.monotonic()
            if last_failure_warn is None or now_mono - last_failure_warn >= (
                _FAILURE_WARN_INTERVAL_SEC
            ):
                logger.warning(
                    "Hair segmentation failed on %d frame(s) since the last warning (%s);"
                    " continuing",
                    failures_since_warn,
                    err,
                )
                last_failure_warn = now_mono
                failures_since_warn = 0
            continue
        consecutive_failures = 0
        with _bg_lock:
            _bg_result[0] = mask
            _bg_result_time[0] = time.monotonic()


def _ensure_started() -> bool:
    """Lazily launch the segmenter worker thread; return whether the service is usable.

    Only cheap checks (model file, mediapipe import) run here; the slow
    segmenter creation happens on the worker thread (see :func:`_hair_bg_worker`).
    """
    global _started
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

            def create_segmenter(delegate: Any) -> Any:
                return ImageSegmenter.create_from_options(
                    ImageSegmenterOptions(
                        base_options=BaseOptions(model_asset_path=_MODEL_PATH, delegate=delegate),
                        running_mode=RunningMode.VIDEO,
                        output_category_mask=True,
                    )
                )

            threading.Thread(
                target=_hair_bg_worker,
                args=(create_segmenter, BaseOptions.Delegate),
                daemon=True,
            ).start()
        except Exception as err:
            _disable(str(err))
        finally:
            _started = True
    return not _disabled


def get_hair_mask(frame: np.ndarray | None) -> np.ndarray | None:
    """Submit ``frame`` for hair segmentation (throttled) and return the latest mask.

    Returns a square boolean mask (the model's native ``_MODEL_INPUT_SIZE``
    resolution) in the raw (unmirrored) camera raster orientation, or None
    until the first result arrives, when the latest result is stale (older
    than ``_RESULT_MAX_AGE_SEC`` — e.g. from a previous caricature session),
    or when the service is disabled. Safe to call every rendered frame.
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
        if now_mono - _bg_result_time[0] > _RESULT_MAX_AGE_SEC:
            return None
        return _bg_result[0]
