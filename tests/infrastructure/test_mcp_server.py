import asyncio

import pytest

pytest.importorskip("mcp")

from app.infrastructure.mcp_server import build_flipdot_mcp


class DummyModeManager:
    def __init__(self):
        self.mode = "clock"

    def set_mode(self, mode, entered_via=None):
        self.mode = mode

    def get_mode_time(self):
        return 1.5


class DummyTransitionPolicy:
    def __init__(self):
        self._settings = {"enabled": True, "start_hour": 0, "end_hour": 7}

    def get_sleep_settings(self):
        return dict(self._settings)

    def set_sleep_settings(self, *, enabled, start_hour, end_hour):
        self._settings = {
            "enabled": bool(enabled),
            "start_hour": int(start_hour),
            "end_hour": int(end_hour),
        }
        return dict(self._settings)


class DummyBoard:
    def __init__(self):
        self.cleared = 0
        self.text_objects = []

    def clear(self):
        self.cleared += 1

    def add_text_object(self, payload):
        item = {"id": f"txt_{len(self.text_objects) + 1}", **payload}
        self.text_objects.append(item)
        return item


class DummySettingsStore:
    def __init__(self):
        self.saved = None

    def save_sleep_settings(self, *, enabled, start_hour, end_hour):
        self.saved = {"enabled": enabled, "start_hour": start_hour, "end_hour": end_hour}


class DummyInputHub:
    def submit_action(self, source, action):
        pass


def _build(mode_manager, board, policy, settings_store):
    frame = [[1 if (r + c) % 2 == 0 else 0 for c in range(28)] for r in range(28)]
    return build_flipdot_mcp(
        input_hub=DummyInputHub(),
        snapshot_frame=lambda: ([list(row) for row in frame], mode_manager.mode, 28, 28),
        get_mode_manager=lambda: mode_manager,
        get_board=lambda: board,
        get_transition_policy=lambda: policy,
        settings_store=settings_store,
    )


def test_mcp_registers_expected_tools():
    mcp = _build(DummyModeManager(), DummyBoard(), DummyTransitionPolicy(), DummySettingsStore())

    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}

    expected = {
        "get_display",
        "list_modes",
        "get_status",
        "set_mode",
        "get_sleep_settings",
        "set_sleep_settings",
        "show_message",
        "clear_board",
        "add_text",
        "draw_shape",
        "draw_stroke",
        "place_image",
        "undo",
        "get_board_state",
        "get_fonts",
        "list_boards",
        "save_board",
        "load_board",
    }
    assert expected.issubset(names)


def test_set_mode_tool_drives_mode_manager():
    mode_manager = DummyModeManager()
    mcp = _build(mode_manager, DummyBoard(), DummyTransitionPolicy(), DummySettingsStore())

    asyncio.run(mcp.call_tool("set_mode", {"mode": "board"}))

    assert mode_manager.mode == "board"


def test_set_mode_rejects_unknown_mode():
    mcp = _build(DummyModeManager(), DummyBoard(), DummyTransitionPolicy(), DummySettingsStore())

    with pytest.raises(Exception):
        asyncio.run(mcp.call_tool("set_mode", {"mode": "not-a-real-mode"}))


def test_show_message_clears_and_adds_text_on_board():
    mode_manager = DummyModeManager()
    board = DummyBoard()
    mcp = _build(mode_manager, board, DummyTransitionPolicy(), DummySettingsStore())

    asyncio.run(mcp.call_tool("show_message", {"text": "HELLO", "scroll": True}))

    assert mode_manager.mode == "board"
    assert board.cleared == 1
    assert board.text_objects[-1]["text"] == "HELLO"
    assert board.text_objects[-1]["scroll"] is True


def test_set_sleep_settings_tool_persists():
    policy = DummyTransitionPolicy()
    store = DummySettingsStore()
    mcp = _build(DummyModeManager(), DummyBoard(), policy, store)

    asyncio.run(
        mcp.call_tool("set_sleep_settings", {"enabled": False, "start_hour": 1, "end_hour": 9})
    )

    assert policy.get_sleep_settings() == {"enabled": False, "start_hour": 1, "end_hour": 9}
    assert store.saved == {"enabled": False, "start_hour": 1, "end_hour": 9}


def test_web_server_mounts_mcp_endpoint():
    pytest.importorskip("fastapi")
    from starlette.routing import Mount

    from app.infrastructure.web_server import WebServer

    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8124)

    mount_paths = [route.path for route in server._app.routes if isinstance(route, Mount)]

    assert "/mcp" in mount_paths


def test_mcp_endpoint_handshake_and_tool_call():
    """End-to-end: handshake over the mounted /mcp endpoint and drive a tool."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import json

    from fastapi.testclient import TestClient

    from app.infrastructure.web_server import WebServer

    mode_manager = DummyModeManager()
    server = WebServer(input_hub=DummyInputHub(), host="127.0.0.1", port=8125)
    server.attach_mode_manager(mode_manager)

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    def rpc(client, payload, hdrs):
        return client.post("/mcp", json=payload, headers=hdrs)

    with TestClient(server._app) as client:
        init = rpc(
            client,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            headers,
        )
        assert init.status_code == 200

        session_headers = dict(headers)
        session_id = init.headers.get("mcp-session-id")
        if session_id:
            session_headers["mcp-session-id"] = session_id
        rpc(client, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_headers)

        listed = rpc(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session_headers)
        assert listed.status_code == 200
        names = {tool["name"] for tool in json.loads(listed.text)["result"]["tools"]}
        assert {"get_display", "set_mode", "show_message"}.issubset(names)

        called = rpc(
            client,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "set_mode", "arguments": {"mode": "board"}},
            },
            session_headers,
        )
        assert called.status_code == 200
        assert mode_manager.mode == "board"
