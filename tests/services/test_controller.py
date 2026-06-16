from __future__ import annotations

from app.services.controller import ControllerHub


class FakeInputDevice:
    def __init__(self, *, path: str, name: str, uniq: str = "", phys: str = "") -> None:
        self.path = path
        self.name = name
        self.uniq = uniq
        self.phys = phys

    def read_loop(self):
        return iter([])


class FakeEvdev:
    class ecodes:
        EV_KEY = 1
        EV_ABS = 3
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


def test_button_label_mapping_uses_friendly_names():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    assert hub._map_button_label(fake_evdev.ecodes.BTN_SOUTH) == "A"
    assert hub._map_button_label(fake_evdev.ecodes.BTN_NORTH) == "Y"
    assert hub._map_button_label(999999) is None


def test_hat_axis_updates_dpad_buttons():
    fake_evdev = FakeEvdev({})
    hub = ControllerHub(evdev_module=fake_evdev, auto_start=False)

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0X, -1)
    assert "D-Left" in hub.get_status_snapshot()["pressed_buttons"]

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0X, 0)
    assert "D-Left" not in hub.get_status_snapshot()["pressed_buttons"]

    hub._apply_abs_state(fake_evdev.ecodes.ABS_HAT0Y, 1)
    assert "D-Down" in hub.get_status_snapshot()["pressed_buttons"]
