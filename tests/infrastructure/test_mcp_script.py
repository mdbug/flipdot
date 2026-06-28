import asyncio
import json

import pytest

pytest.importorskip("mcp")

from app.core.mode_manager import ModeManager
from app.infrastructure.mcp_server import build_flipdot_mcp
from app.modes.script_mode import ScriptMode
from app.services.sandbox import bwrap_available
from app.services.script_store import ScriptStore

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


class DummyModeManager:
    def __init__(self):
        self.mode = "clock"

    def set_mode(self, mode, entered_via=None):
        self.mode = mode

    def get_mode_time(self):
        return 1.0


class DummySettingsStore:
    def save_sleep_settings(self, **_kwargs):
        pass


def _build(mode_manager, script_mode):
    return build_flipdot_mcp(
        snapshot_frame=lambda: ([[0]], mode_manager.mode, 28, 28),
        get_mode_manager=lambda: mode_manager,
        get_board=lambda: None,
        get_script_mode=lambda: script_mode,
        get_transition_policy=lambda: None,
        settings_store=DummySettingsStore(),
    )


def _script_mode(tmp_path):
    return ScriptMode(28, 28, store=ScriptStore(tmp_path))


@requires_bwrap
def test_run_script_switches_to_script_mode(tmp_path):
    mode_manager = DummyModeManager()
    script_mode = _script_mode(tmp_path)
    mcp = _build(mode_manager, script_mode)
    try:
        asyncio.run(mcp.call_tool("run_script", {"code": GAME_OF_LIFE}))
        assert mode_manager.mode == ModeManager.MODE_SCRIPT
        assert script_mode.get_frame().shape == (28, 28)
        assert script_mode.status()["running"]
    finally:
        script_mode.stop_script()


def test_run_script_rejects_unsafe_code(tmp_path):
    script_mode = _script_mode(tmp_path)
    mcp = _build(DummyModeManager(), script_mode)
    # MCP wraps the rejection; only that it fails is the contract, not the wrapped type.
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(
            mcp.call_tool("run_script", {"code": "import os\ndef step(s,t,w,h):\n    return 0"})
        )


def test_get_script_returns_saved_source(tmp_path):
    script_mode = _script_mode(tmp_path)
    script_mode._store.save("life", GAME_OF_LIFE)
    mcp = _build(DummyModeManager(), script_mode)
    result = asyncio.run(mcp.call_tool("get_script", {"name": "life"}))
    payload = json.loads(result[0].text)
    assert payload["code"] == GAME_OF_LIFE
    assert payload["name"] == "life"


def test_get_script_missing_raises(tmp_path):
    script_mode = _script_mode(tmp_path)
    mcp = _build(DummyModeManager(), script_mode)
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(mcp.call_tool("get_script", {"name": "nope"}))


@requires_bwrap
def test_save_load_list_round_trip(tmp_path):
    mode_manager = DummyModeManager()
    script_mode = _script_mode(tmp_path)
    mcp = _build(mode_manager, script_mode)
    try:
        asyncio.run(mcp.call_tool("run_script", {"code": GAME_OF_LIFE, "name": "life"}))
        listing = script_mode.list_scripts()
        assert "life" in listing["scripts"]

        script_mode.stop_script()
        asyncio.run(mcp.call_tool("load_script", {"name": "life"}))
        assert mode_manager.mode == ModeManager.MODE_SCRIPT
        assert script_mode.status()["running"]
    finally:
        script_mode.stop_script()
