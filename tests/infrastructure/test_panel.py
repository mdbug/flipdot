"""Tests for the panel's bounded serial write queue and diff resync."""

import queue

import numpy as np
import pytest

pytest.importorskip("serial")

from flippydot import flippydot  # noqa: E402

import app.infrastructure.panel as panel_module  # noqa: E402


def _bare_panel(max_queue: int) -> panel_module.Panel:
    """Build a hardware-mode Panel without opening a serial port or thread."""
    p = panel_module.Panel.__new__(panel_module.Panel)
    p.preview = False
    p.serial = None
    p.panel = flippydot.Panel([[1], [2], [3], [4]], 28, 7, module_rotation=0, screen_preview=False)
    p.WIDTH = p.panel.get_total_width()
    p.HEIGHT = p.panel.get_total_height()
    p._prev_frame = np.full((p.HEIGHT, p.WIDTH), 255, dtype=np.uint8)
    p._write_queue = queue.Queue(maxsize=max_queue)
    p._last_drop_warn_monotonic = 0.0
    return p


def test_update_enqueues_only_changed_modules():
    p = _bare_panel(max_queue=4)
    frame = np.zeros((p.HEIGHT, p.WIDTH), dtype=np.uint8)

    p.update(frame)  # first frame differs from the poisoned baseline
    assert p._write_queue.qsize() == 1

    p.update(frame)  # unchanged frame -> nothing queued
    assert p._write_queue.qsize() == 1


def test_dropped_write_resyncs_diff_baseline():
    p = _bare_panel(max_queue=1)
    frame = np.zeros((p.HEIGHT, p.WIDTH), dtype=np.uint8)
    p.update(frame)
    assert p._write_queue.full()

    changed = frame.copy()
    changed[0, 0] = 1
    p.update(changed)  # queue full -> write dropped, baseline poisoned
    assert (p._prev_frame == 255).all()

    p._write_queue.get_nowait()  # the port drains
    p.update(changed)  # full resend against the poisoned baseline
    assert p._write_queue.qsize() == 1
    assert np.array_equal(p._prev_frame, changed)
