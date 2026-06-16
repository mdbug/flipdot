from __future__ import annotations

import logging
import threading
import time
import errno
from typing import Any

try:
    import evdev  # type: ignore
except Exception:  # pragma: no cover - optional dependency on some dev machines
    evdev = None


logger = logging.getLogger(__name__)

TARGET_CONTROLLER_ADDRESS = "AA:BB:CC:DD:EE:01"
_UNSET = object()


class ControllerHub:
    """Track a Bluetooth controller via evdev and expose status snapshots."""

    def __init__(
        self,
        *,
        target_address: str = TARGET_CONTROLLER_ADDRESS,
        scan_interval_sec: float = 0.2,
        reconnect_delay_sec: float = 0.05,
        auto_start: bool = True,
        evdev_module=_UNSET,
    ) -> None:
        self._evdev = evdev if evdev_module is _UNSET else evdev_module
        self._target_address = self._normalize_address(target_address)
        self._scan_interval_sec = max(0.1, float(scan_interval_sec))
        self._reconnect_delay_sec = max(0.05, float(reconnect_delay_sec))

        self._lock = threading.Lock()
        self._enabled = self._evdev is not None
        self._connected = False
        self._device_name = ""
        self._device_path = ""
        self._device_address = self._target_address
        self._pressed_buttons: set[str] = set()
        self._last_event_monotonic: float | None = None

        self._running = False
        self._thread: threading.Thread | None = None

        if not self._enabled:
            logger.warning("Controller disabled: evdev is not available")
            return

        if auto_start:
            self.start()

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def get_status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": bool(self._enabled),
                "connected": bool(self._connected),
                "address": self._device_address or self._target_address,
                "device_name": self._device_name,
                "pressed_buttons": sorted(self._pressed_buttons),
                "last_event_monotonic": self._last_event_monotonic,
            }

    def _worker(self) -> None:
        while self._running:
            device = self._find_matching_device()
            if device is None:
                self._set_disconnected()
                time.sleep(self._scan_interval_sec)
                continue

            self._set_connected(device)
            try:
                for event in device.read_loop():
                    if not self._running:
                        break
                    if event.type == self._evdev.ecodes.EV_KEY:
                        label = self._map_button_label(event.code)
                        if label is None:
                            continue
                        self._apply_button_state(label, event.value)
                        continue

                    if event.type == self._evdev.ecodes.EV_ABS:
                        self._apply_abs_state(event.code, event.value)
                        continue
            except OSError as exc:
                if exc.errno == errno.ENODEV:
                    logger.info("Controller disconnected, waiting for reconnect: %s", exc)
                else:
                    logger.warning("Controller input read error: %s", exc)
            except Exception as exc:
                logger.warning("Controller input read error: %s", exc)
            finally:
                self._set_disconnected()
                try:
                    device.close()
                except Exception:
                    pass
                # Keep reconnect latency low after transient BT device drops.
                time.sleep(self._reconnect_delay_sec)

    def _find_matching_device(self):
        try:
            for path in self._evdev.list_devices():
                try:
                    device = self._evdev.InputDevice(path)
                except Exception:
                    continue
                if self._device_matches(device):
                    logger.info(
                        "Controller connected: name=%s path=%s address=%s",
                        getattr(device, "name", ""),
                        getattr(device, "path", ""),
                        self._extract_device_address(device),
                    )
                    return device
        except Exception as exc:
            logger.warning("Controller scan failed: %s", exc)
        return None

    def _device_matches(self, device) -> bool:
        candidates = [
            self._normalize_address(getattr(device, "uniq", "")),
            self._normalize_address(getattr(device, "phys", "")),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            if self._target_address in candidate:
                return True
        return False

    def _extract_device_address(self, device) -> str:
        for raw in (getattr(device, "uniq", ""), getattr(device, "phys", "")):
            candidate = self._normalize_address(raw)
            if self._target_address in candidate:
                return self._target_address
        return self._target_address

    def _set_connected(self, device) -> None:
        with self._lock:
            self._connected = True
            self._device_name = str(getattr(device, "name", ""))
            self._device_path = str(getattr(device, "path", ""))
            self._device_address = self._extract_device_address(device)
            self._pressed_buttons.clear()

    def _set_disconnected(self) -> None:
        with self._lock:
            self._connected = False
            self._pressed_buttons.clear()

    def _apply_button_state(self, label: str, value: int) -> None:
        ts = time.monotonic()
        with self._lock:
            if int(value) == 0:
                self._pressed_buttons.discard(label)
            else:
                self._pressed_buttons.add(label)
            self._last_event_monotonic = ts

    def _map_button_label(self, code: int) -> str | None:
        ec = self._evdev.ecodes
        label_map = {
            ec.BTN_SOUTH: "A",
            ec.BTN_EAST: "B",
            ec.BTN_WEST: "X",
            ec.BTN_NORTH: "Y",
            ec.BTN_TL: "L1",
            ec.BTN_TR: "R1",
            ec.BTN_TL2: "L2",
            ec.BTN_TR2: "R2",
            ec.BTN_START: "Start",
            ec.BTN_SELECT: "Select",
            ec.BTN_MODE: "Home",
            ec.BTN_THUMBL: "L3",
            ec.BTN_THUMBR: "R3",
            ec.BTN_DPAD_UP: "D-Up",
            ec.BTN_DPAD_DOWN: "D-Down",
            ec.BTN_DPAD_LEFT: "D-Left",
            ec.BTN_DPAD_RIGHT: "D-Right",
        }
        return label_map.get(code)

    def _apply_abs_state(self, code: int, value: int) -> None:
        ec = self._evdev.ecodes

        if code == ec.ABS_HAT0X:
            # Hat X axis: -1 left, 0 neutral, +1 right
            self._apply_button_state("D-Left", 1 if int(value) < 0 else 0)
            self._apply_button_state("D-Right", 1 if int(value) > 0 else 0)
            return

        if code == ec.ABS_HAT0Y:
            # Hat Y axis: -1 up, 0 neutral, +1 down
            self._apply_button_state("D-Up", 1 if int(value) < 0 else 0)
            self._apply_button_state("D-Down", 1 if int(value) > 0 else 0)
            return

    @staticmethod
    def _normalize_address(value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("-", ":")
        return normalized
