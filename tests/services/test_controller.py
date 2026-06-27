from __future__ import annotations

from app.services.controller import ControllerHub


class FakeInputDevice:
    def __init__(
        self,
        *,
        path: str,
        name: str,
        uniq: str = "",
        phys: str = "",
        capabilities: dict[int, list[int]] | None = None,
    ) -> None:
        self.path = path
        self.name = name
        self.uniq = uniq
        self.phys = phys
        self._capabilities = capabilities or {}

    def read_loop(self):
        return iter([])

    def capabilities(self, absinfo=False):
        del absinfo
        return self._capabilities

    def close(self):
        return None


class FakeEvdev:
    class ecodes:
        EV_KEY = 1
        EV_ABS = 3
        ABS_X = 0
        ABS_Y = 1
        ABS_RX = 3
        ABS_RY = 4
        BTN_SOUTH = 304
        BTN_EAST = 305
        BTN_WEST = 307
        BTN_NORTH = 308
        BTN_TL = 310
        BTN_TR = 311
        BTN_TL2 = 312
        BTN_TR2 = 313
        BTN_START = 315
        BTN_SELECT = 314
        BTN_MODE = 316
        BTN_THUMBL = 317
        BTN_THUMBR = 318
        BTN_DPAD_UP = 544
        BTN_DPAD_DOWN = 545
        BTN_DPAD_LEFT = 546
        BTN_DPAD_RIGHT = 547
        ABS_HAT0X = 16
        ABS_HAT0Y = 17
        KEY_ENTER = 28
        KEY_SPACE = 57
        KEY_ESC = 1
        KEY_BACKSPACE = 14
        KEY_B = 48

    def __init__(self, devices: dict[str, FakeInputDevice]) -> None:
        self._devices = devices

    def list_devices(self):
        return list(self._devices.keys())

    def InputDevice(self, path: str):
        return self._devices[path]


def test_controller_hub_disabled_without_evdev():
    hub = ControllerHub(evdev_module=None, auto_start=False)

    snapshot = hub.get_status_snapshot()
    assert snapshot["enabled"] is False
    assert snapshot["connected"] is False
    assert snapshot["pressed_buttons"] == []


def test_controller_matching_uses_target_bluetooth_address():
    target = "AA:BB:CC:DD:EE:01"
    fake_evdev = FakeEvdev(
        {
            "/dev/input/event10": FakeInputDevice(
                path="/dev/input/event10",
                name="Wireless Controller",
                uniq="aa:bb:cc:dd:ee:01",
            )
        }
    )
    hub = ControllerHub(target_address=target, evdev_module=fake_evdev, auto_start=False)

    device = hub._find_matching_device()

    assert device is not None
    assert device.path == "/dev/input/event10"


def test_controller_matching_does_not_fallback_to_name_when_address_set():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(
        target_address="AA:BB:CC:DD:EE:02",
        target_name_hint="IINE_keyboard",
        evdev_module=fake_evdev,
        auto_start=False,
    )

    wrong = FakeInputDevice(
        path="/dev/input/event7",
        name="IINE_keyboard",
        uniq="a2:54:84:c5:9a:92",
    )
    assert hub._device_matches(wrong) is False

    right = FakeInputDevice(
        path="/dev/input/event6",
        name="IINE_keyboard",
        uniq="aa:bb:cc:dd:ee:02",
    )
    assert hub._device_matches(right) is True


def test_controller_matching_can_use_name_hint_without_address():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(
        target_address="",
        target_name_hint="iine_keyboard",
        evdev_module=fake_evdev,
        auto_start=False,
    )
    by_name = FakeInputDevice(path="/dev/input/event7", name="IINE_keyboard", uniq="")
    assert hub._device_matches(by_name) is True


def test_button_press_and_release_updates_snapshot(monkeypatch):
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    clock = {"value": 100.0}

    def fake_monotonic():
        return clock["value"]

    monkeypatch.setattr("app.services.controller.time.monotonic", fake_monotonic)

    hub._set_connected(
        FakeInputDevice(
            path="/dev/input/event10",
            name="Wireless Controller",
            uniq="aa:bb:cc:dd:ee:01",
        )
    )
    hub._apply_button_state("A", 1)

    snapshot = hub.get_status_snapshot()
    assert snapshot["connected"] is True
    assert snapshot["pressed_buttons"] == ["A"]
    assert snapshot["last_event_monotonic"] == 100.0

    clock["value"] = 101.0
    hub._apply_button_state("A", 0)
    snapshot = hub.get_status_snapshot()
    assert snapshot["pressed_buttons"] == []
    assert snapshot["last_event_monotonic"] == 101.0


def test_button_events_are_included_in_status_snapshot(monkeypatch):
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)
    clock = {"value": 200.0}
    monkeypatch.setattr("app.services.controller.time.monotonic", lambda: clock["value"])

    hub._apply_button_state("A", 1)
    clock["value"] = 200.2
    hub._apply_button_state("A", 0)

    events = hub.get_status_snapshot()["recent_button_events"]
    assert events == [
        {"sequence": 1, "button": "A", "event": "pressed", "monotonic": 200.0},
        {"sequence": 2, "button": "A", "event": "released", "monotonic": 200.2},
    ]


def test_button_label_mapping_uses_friendly_names():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    assert hub._map_button_label(fake_evdev.ecodes.BTN_SOUTH) == "A"
    assert hub._map_button_label(fake_evdev.ecodes.BTN_NORTH) == "Y"
    assert hub._map_button_label(999999) is None


def test_keyboard_profile_maps_a_and_b_distinctly():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    assert hub._map_button_label(fake_evdev.ecodes.KEY_ENTER) == "A"
    assert hub._map_button_label(fake_evdev.ecodes.KEY_SPACE) == "B"


def test_hat_axis_updates_dpad_buttons():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0X, -1)
    assert "D-Left" in hub.get_status_snapshot()["pressed_buttons"]

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0X, 0)
    assert "D-Left" not in hub.get_status_snapshot()["pressed_buttons"]

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0Y, 1)
    assert "D-Down" in hub.get_status_snapshot()["pressed_buttons"]


def test_quick_tap_is_latched_until_drained():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    # A press and release that happens entirely between drains must still be
    # reported as a single down-edge.
    hub._apply_button_state("A", 1)
    hub._apply_button_state("A", 0)
    assert hub.get_status_snapshot()["pressed_buttons"] == []

    assert hub.drain_pressed_events() == {"A"}
    # Edges are cleared after draining.
    assert hub.drain_pressed_events() == set()


def test_held_button_latches_single_edge():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    hub._apply_button_state("B", 1)
    hub._apply_button_state("B", 1)  # repeated press event while held
    assert hub.drain_pressed_events() == {"B"}
    assert hub.drain_pressed_events() == set()


def test_controller_matching_prefers_highest_capability_score():
    ec = FakeEvdev.ecodes
    target = "AA:BB:CC:DD:EE:01"
    fake_evdev = FakeEvdev(
        {
            "/dev/input/event1": FakeInputDevice(
                path="/dev/input/event1",
                name="Wireless Controller Sensor",
                uniq="aa:bb:cc:dd:ee:01",
                capabilities={
                    ec.EV_ABS: [ec.ABS_X],
                },
            ),
            "/dev/input/event2": FakeInputDevice(
                path="/dev/input/event2",
                name="Wireless Controller",
                uniq="aa:bb:cc:dd:ee:01",
                capabilities={
                    ec.EV_KEY: [
                        ec.BTN_SOUTH,
                        ec.BTN_EAST,
                        ec.BTN_WEST,
                        ec.BTN_NORTH,
                        ec.BTN_START,
                        ec.BTN_SELECT,
                    ],
                    ec.EV_ABS: [ec.ABS_HAT0X, ec.ABS_HAT0Y, ec.ABS_X, ec.ABS_Y],
                },
            ),
        }
    )

    hub = ControllerHub(target_address=target, evdev_module=fake_evdev, auto_start=False)
    device = hub._find_matching_device()

    assert device is not None
    assert device.path == "/dev/input/event2"


def test_button_state_is_merged_across_devices():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    hub._set_connected_devices(
        [
            FakeInputDevice(path="/dev/input/event1", name="Wireless Controller"),
            FakeInputDevice(path="/dev/input/event2", name="Wireless Controller"),
        ]
    )

    hub._apply_button_state("A", 1, device_path="/dev/input/event1")
    hub._apply_button_state("B", 1, device_path="/dev/input/event2")
    snapshot = hub.get_status_snapshot()
    assert snapshot["pressed_buttons"] == ["A", "B"]

    hub._apply_button_state("A", 0, device_path="/dev/input/event1")
    snapshot = hub.get_status_snapshot()
    assert snapshot["pressed_buttons"] == ["B"]

    hub._apply_button_state("B", 0, device_path="/dev/input/event2")
    snapshot = hub.get_status_snapshot()
    assert snapshot["pressed_buttons"] == []


def test_parse_bluetoothctl_battery_percentage_prefers_decimal_parenthesis():
    output = "Battery Percentage: 0x5c (92)"
    assert ControllerHub._parse_bluetoothctl_battery_percentage(output) == 92


def test_parse_bluetoothctl_battery_percentage_accepts_plain_decimal():
    output = "Battery Percentage: 76"
    assert ControllerHub._parse_bluetoothctl_battery_percentage(output) == 76


def test_parse_bluetoothctl_battery_percentage_rejects_missing_value():
    output = "Battery Service present, percentage unavailable"
    assert ControllerHub._parse_bluetoothctl_battery_percentage(output) is None


def test_parse_bluetoothctl_link_metrics_accepts_rssi_and_tx_power():
    output = "Device AA:BB:CC:DD:EE:01\n\tRSSI: 0xffffffc4 (-60)\n\tTxPower: 0x04 (4)\n"
    assert ControllerHub._parse_bluetoothctl_link_metrics(output) == {
        "rssi_dbm": -60,
        "tx_power_dbm": 4,
        "link_quality": None,
    }


def test_parse_bluetoothctl_link_metrics_accepts_plain_values():
    output = "RSSI: -72\nTxPower: -3\nLink Quality: 54\n"
    assert ControllerHub._parse_bluetoothctl_link_metrics(output) == {
        "rssi_dbm": -72,
        "tx_power_dbm": -3,
        "link_quality": 54,
    }


def test_parse_upower_percentage_accepts_decimal_value():
    output = "gaming-input\n  percentage:          92%\n"
    assert ControllerHub._parse_upower_percentage(output) == 92


def test_parse_upower_percentage_rounds_fractional_value():
    output = "gaming-input\n  percentage:          91.6%\n"
    assert ControllerHub._parse_upower_percentage(output) == 92


def test_parse_upower_percentage_rejects_invalid_value():
    output = "gaming-input\n  percentage:          unknown\n"
    assert ControllerHub._parse_upower_percentage(output) is None


def test_upower_info_matches_address_from_serial():
    hub = ControllerHub(evdev_module=FakeEvdev({}), auto_start=False)
    info = "serial:               AA:BB:CC:DD:EE:01\n"
    assert hub._upower_info_matches_address(info, "aa:bb:cc:dd:ee:01") is True


def test_upower_info_matches_address_from_native_path():
    hub = ControllerHub(evdev_module=FakeEvdev({}), auto_start=False)
    info = "native-path:          /org/bluez/hci0/dev_AA_BB_CC_DD_EE_01\n"
    assert hub._upower_info_matches_address(info, "aa:bb:cc:dd:ee:01") is True


def test_bluetooth_reconnect_attempt_is_throttled(monkeypatch):
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(
        target_address="AA:BB:CC:DD:EE:01",
        bluetooth_connect_interval_sec=5.0,
        evdev_module=fake_evdev,
        auto_start=False,
    )

    now = {"value": 10.0}
    calls = []

    def fake_monotonic():
        return now["value"]

    class _Result:
        returncode = 0
        stdout = "Connection successful"
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _Result()

    monkeypatch.setattr("app.services.controller.time.monotonic", fake_monotonic)
    monkeypatch.setattr("app.services.controller.subprocess.run", fake_run)

    hub._request_bluetooth_connect()
    hub._request_bluetooth_connect()
    now["value"] = 14.9
    hub._request_bluetooth_connect()
    now["value"] = 15.0
    hub._request_bluetooth_connect()

    assert [call[0] for call in calls] == [
        ["bluetoothctl", "connect", "aa:bb:cc:dd:ee:01"],
        ["bluetoothctl", "connect", "aa:bb:cc:dd:ee:01"],
    ]
    assert calls[0][1]["timeout"] == 2.0
    assert calls[0][1]["check"] is False
