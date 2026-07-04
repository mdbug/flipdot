"""Tests for the hair segmentation service's mediapipe-free behavior."""

import importlib
import logging
import sys

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
    (tmp_path / "selfie_multiclass_256x256.tflite").write_bytes(b"stub")
    for name in ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python"):
        monkeypatch.setitem(sys.modules, name, None)
    module = _load_module(monkeypatch, tmp_path)

    assert module.get_hair_mask(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    assert module._disabled is True


def test_none_frame_returns_none_without_initializing(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)

    assert module.get_hair_mask(None) is None
    assert module._started is False  # lazy: a None frame must not trigger model init
