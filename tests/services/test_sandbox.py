import os
import sys

import numpy as np
import pytest

from app.services.sandbox import (
    SandboxedScript,
    ScriptValidationError,
    bwrap_available,
    validate_source,
)

requires_bwrap = pytest.mark.skipif(
    not bwrap_available(), reason="bubblewrap (bwrap) is required to run sandboxed scripts"
)

GAME_OF_LIFE = """
def setup(width, height):
    rng = np.random.default_rng(0)
    return (rng.random((height, width)) < 0.3).astype(np.uint8)

def step(state, t, width, height):
    n = sum(np.roll(np.roll(state, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    new = ((n == 3) | ((state == 1) & (n == 2))).astype(np.uint8)
    return new, new
"""


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef step(s, t, w, h):\n    return np.zeros((h, w))",
        "def step(s, t, w, h):\n    return open('/etc/passwd')",
        "def step(s, t, w, h):\n    return __import__('os')",
        "def step(s, t, w, h):\n    return ().__class__.__bases__",
        "def step(s, t, w, h):\n    return eval('1')",
    ],
)
def test_validate_rejects_unsafe_code(code):
    with pytest.raises(ScriptValidationError):
        validate_source(code)


def test_validate_requires_step_function():
    with pytest.raises(ScriptValidationError):
        validate_source("def setup(w, h):\n    return 0")


def test_validate_rejects_nested_step():
    # A step() nested inside another function passes ast.walk but is invisible at
    # runtime, so it must be rejected up front.
    code = "def outer():\n    def step(s, t, w, h):\n        return 0\n    return step"
    with pytest.raises(ScriptValidationError):
        validate_source(code)


def test_validate_accepts_game_of_life():
    validate_source(GAME_OF_LIFE)  # should not raise


@requires_bwrap
def test_worker_produces_binary_frames():
    script = SandboxedScript(GAME_OF_LIFE, 28, 28)
    script.start()
    try:
        frame = script.get_frame(0)
        assert frame is not None
        assert frame.shape == (28, 28)
        assert frame.dtype == np.uint8
        assert set(np.unique(frame)).issubset({0, 1})
        assert script.get_frame(1) is not None
        assert not script.failed
    finally:
        script.stop()


@requires_bwrap
def test_infinite_loop_is_killed_by_timeout():
    code = "def step(s, t, w, h):\n    while True:\n        pass"
    script = SandboxedScript(code, 28, 28, frame_timeout=0.2)
    script.start()
    try:
        assert script.get_frame(0) is None
        assert script.failed
        assert "timed out" in (script.error or "")
    finally:
        script.stop()


@requires_bwrap
def test_wrong_shape_frame_is_rejected():
    code = "def step(s, t, w, h):\n    return np.zeros((5, 5))"
    script = SandboxedScript(code, 28, 28)
    script.start()
    try:
        assert script.get_frame(0) is None
        assert script.failed
        assert "shape" in (script.error or "")
    finally:
        script.stop()


@requires_bwrap
def test_t_is_passed_as_float_seconds():
    code = (
        "def step(state, t, w, h):\n"
        "    f = np.zeros((h, w))\n"
        "    f[0, 0] = 1 if t > 1.0 else 0\n"
        "    return f"
    )
    script = SandboxedScript(code, 28, 28)
    script.start()
    try:
        assert script.get_frame(0.5)[0, 0] == 0
        assert script.get_frame(2.0)[0, 0] == 1
    finally:
        script.stop()


@requires_bwrap
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_worker_does_not_inherit_heavy_app_imports():
    # Regression: the worker must be a clean subprocess that imports only numpy.
    # If it re-imported the host app (mediapipe/cv2/serial) its virtual memory
    # would balloon past ~900MB and clamp against the rlimit, making every
    # allocation in an otherwise-fine script fail with MemoryError.
    script = SandboxedScript(GAME_OF_LIFE, 28, 28)
    script.start()
    try:
        with open(f"/proc/{script._proc.pid}/statm") as f:
            pages = int(f.read().split()[0])
        vms_mb = pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
        assert vms_mb < 700, f"worker VMS {vms_mb:.0f}MB suggests the app stack leaked in"
    finally:
        script.stop()


@requires_bwrap
def test_runtime_error_is_reported_not_raised():
    code = "def step(s, t, w, h):\n    return 1 / 0"
    script = SandboxedScript(code, 28, 28)
    script.start()
    try:
        assert script.get_frame(0) is None
        assert script.failed
        assert script.error
    finally:
        script.stop()


@requires_bwrap
def test_sandbox_blocks_filesystem_write(tmp_path):
    # numpy's I/O surface (memmap mode="w+") is fully exposed to script code; the
    # bubblewrap jail must make the write impossible rather than relying on the
    # AST allow-list, which cannot constrain numpy.
    target = tmp_path / "escape.bin"
    code = (
        "def step(s, t, w, h):\n"
        f"    m = np.memmap({str(target)!r}, dtype=np.uint8, mode='w+', shape=(4,))\n"
        "    m[:] = 1\n"
        "    m.flush()\n"
        "    return np.zeros((h, w), dtype=np.uint8)"
    )
    script = SandboxedScript(code, 28, 28)
    script.start()
    try:
        assert script.get_frame(0) is None
        assert script.failed
    finally:
        script.stop()
    assert not target.exists()


@requires_bwrap
def test_sandbox_blocks_network(tmp_path):
    # The worker runs in an unshared (disconnected) network namespace, so even
    # numpy's DataSource fetch cannot reach the network.
    code = (
        "def step(s, t, w, h):\n"
        "    np.lib.npyio.DataSource('/tmp').open('http://127.0.0.1:9/')\n"
        "    return np.zeros((h, w), dtype=np.uint8)"
    )
    script = SandboxedScript(code, 28, 28)
    script.start()
    try:
        assert script.get_frame(0) is None
        assert script.failed
    finally:
        script.stop()
