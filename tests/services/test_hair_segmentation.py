"""Tests for the hair segmentation service's mediapipe-free behavior."""

import importlib
import logging
import sys
import types

import numpy as np


def _load_module(monkeypatch, tmp_path):
    """Import a fresh service instance whose models dir is an (empty) tmp dir."""
    monkeypatch.setenv("MEDIAPIPE_MODELS_DIR", str(tmp_path))
    sys.modules.pop("app.services.hair_segmentation", None)
    return importlib.import_module("app.services.hair_segmentation")


def test_should_submit_throttle_logic(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    interval = 1.0 / 7.0

    assert module._should_submit(10.0, 0.0, interval)
    assert not module._should_submit(10.0, 10.0, interval)
    assert not module._should_submit(10.1, 10.0, interval)
    assert module._should_submit(10.2, 10.0, interval)
    assert module._should_submit(10.0, 10.0, 0.0)  # 0 interval disables the throttle


def test_missing_model_disables_with_single_warning(monkeypatch, tmp_path, caplog):
    module = _load_module(monkeypatch, tmp_path)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    with caplog.at_level(logging.WARNING):
        assert module.get_hair_mask(frame) is None
        assert module.get_hair_mask(frame) is None

    warnings = [r for r in caplog.records if "Hair segmentation unavailable" in r.getMessage()]
    assert len(warnings) == 1


def test_mediapipe_missing_disables(monkeypatch, tmp_path):
    (tmp_path / "hair_segmenter.tflite").write_bytes(b"stub")
    for name in ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load_module(monkeypatch, tmp_path)

    assert module.get_hair_mask(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    assert module._disabled is True


def test_none_frame_returns_none_without_initializing(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)

    assert module.get_hair_mask(None) is None
    assert module._started is False  # lazy: a None frame must not trigger model init


class _StopAfterCreation:
    """Fake worker event whose first wait aborts the loop (creation already ran)."""

    def wait(self):
        raise StopIteration

    def clear(self):
        pass


def _run_worker_creation(module, monkeypatch, fail_gpu):
    """Run the worker's segmenter-creation phase synchronously; return delegates used."""
    monkeypatch.setitem(sys.modules, "mediapipe", types.SimpleNamespace())
    monkeypatch.setattr(module, "_bg_event", _StopAfterCreation())
    created = []
    delegates = types.SimpleNamespace(GPU="gpu", CPU="cpu")

    def create_segmenter(delegate):
        if fail_gpu and delegate == "gpu":
            raise RuntimeError("no GPU delegate in this wheel")
        created.append(delegate)
        return types.SimpleNamespace(segment_for_video=lambda image, ts: None)

    try:
        module._hair_bg_worker(create_segmenter, delegates)
    except StopIteration:
        pass
    return created


def test_gpu_delegate_preferred(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)

    created = _run_worker_creation(module, monkeypatch, fail_gpu=False)

    assert created == ["gpu"]
    assert module._disabled is False


def test_gpu_delegate_falls_back_to_cpu(monkeypatch, tmp_path, caplog):
    module = _load_module(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        created = _run_worker_creation(module, monkeypatch, fail_gpu=True)

    assert created == ["cpu"]
    assert module._disabled is False
    assert any("falling back to CPU" in r.getMessage() for r in caplog.records)


def test_creation_failure_on_both_delegates_disables(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "mediapipe", types.SimpleNamespace())
    delegates = types.SimpleNamespace(GPU="gpu", CPU="cpu")

    def create_segmenter(delegate):
        raise RuntimeError("model rejected")

    module._hair_bg_worker(create_segmenter, delegates)

    assert module._disabled is True


def test_worker_rate_limits_failure_warnings(monkeypatch, tmp_path, caplog):
    """An intermittently failing segmenter must not warn once per failed frame."""
    module = _load_module(monkeypatch, tmp_path)
    fake_mp = types.SimpleNamespace(
        Image=lambda **kwargs: object(),
        ImageFormat=types.SimpleNamespace(SRGB=1),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    class _FailingSegmenter:
        def segment_for_video(self, image, ts):
            raise RuntimeError("boom")

    # Keep the disable threshold out of the way: this test is about the
    # warning cadence, not the wedged-model shutdown.
    monkeypatch.setattr(module, "_MAX_CONSECUTIVE_FAILURES", 100)

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    feeds = {"count": 0}

    class _FeedingEvent:
        """Drives the worker loop synchronously: one frame per wait, then stop."""

        def wait(self):
            if feeds["count"] >= 5:
                raise StopIteration
            feeds["count"] += 1
            module._bg_frame[0] = frame

        def clear(self):
            pass

    monkeypatch.setattr(module, "_bg_event", _FeedingEvent())

    with caplog.at_level(logging.WARNING):
        try:
            module._hair_bg_worker(
                lambda delegate: _FailingSegmenter(),
                types.SimpleNamespace(GPU="gpu", CPU="cpu"),
            )
        except StopIteration:
            pass

    warnings = [r for r in caplog.records if "Hair segmentation failed" in r.getMessage()]
    # Five consecutive failed frames inside one warn interval produce exactly
    # one warning (the first); the rest aggregate into the next interval.
    assert len(warnings) == 1
