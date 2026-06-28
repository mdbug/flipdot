from __future__ import annotations

import errno
import logging
import re
import select
import subprocess
import threading
import time
from typing import Any

try:
    import evdev
except Exception:  # pragma: no cover - optional dependency on some dev machines
    evdev = None


logger = logging.getLogger(__name__)

TARGET_CONTROLLER_ADDRESS = "AA:BB:CC:DD:EE:01"
_UNSET = object()
_BATTERY_PERCENTAGE_HEX_WITH_DEC_RE = re.compile(
    r"Battery Percentage:\s*0x[0-9a-fA-F]+\s*\((\d{1,3})\)",
    re.IGNORECASE,
)
_BATTERY_PERCENTAGE_DEC_RE = re.compile(r"Battery Percentage:\s*(\d{1,3})\b", re.IGNORECASE)
_BLUETOOTHCTL_RSSI_RE = re.compile(r"RSSI:\s*(?:0x[0-9a-fA-F]+\s*\()?(-?\d+)\)?", re.IGNORECASE)
_BLUETOOTHCTL_TX_POWER_RE = re.compile(
    r"TxPower:\s*(?:0x[0-9a-fA-F]+\s*\()?(-?\d+)\)?", re.IGNORECASE
)
_BLUETOOTHCTL_LINK_QUALITY_RE = re.compile(r"Link Quality:\s*(\d+)", re.IGNORECASE)
# btmgmt conn-info prints "RSSI -91" / "TX power 0" without colons; keep the colon
# optional so both colon- and space-separated bluez versions parse.
_BTMGMT_RSSI_RE = re.compile(r"RSSI\s*:?\s*(-?\d+)", re.IGNORECASE)
_BTMGMT_TX_POWER_RE = re.compile(r"TX\s*Power\s*:?\s*(-?\d+)", re.IGNORECASE)
_BTMGMT_LINK_QUALITY_RE = re.compile(r"Link\s*quality\s*:\s*(\d+)", re.IGNORECASE)
_BTMGMT_CONN_INTERVAL_MS_RE = re.compile(
    r"Connection\s*interval\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*msec", re.IGNORECASE
)
_BTMGMT_CONN_LATENCY_RE = re.compile(r"(?:Peripheral\s*)?latency\s*:\s*(\d+)", re.IGNORECASE)
_BTMGMT_SUPERVISION_TIMEOUT_MS_RE = re.compile(
    r"Supervision\s*timeout\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*msec", re.IGNORECASE
)
# btmgmt conn-info defaults to BR/EDR; BLE peripherals (most controllers/keyboards
# here connect over BLE) need an explicit address type or the query fails with
# "Not Connected". Try BR/EDR first, then LE public/random.
_BTMGMT_ADDRESS_TYPE_ARGS: tuple[tuple[str, ...], ...] = ((), ("-t", "1"), ("-t", "2"))
_BLUETOOTHCTL_CONN_INTERVAL_MS_RE = re.compile(
    r"Connection\s*interval\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*msec", re.IGNORECASE
)
_BLUETOOTHCTL_CONN_LATENCY_RE = re.compile(r"(?:Peripheral\s*)?latency\s*:\s*(\d+)", re.IGNORECASE)
_BLUETOOTHCTL_SUPERVISION_TIMEOUT_MS_RE = re.compile(
    r"Supervision\s*timeout\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*msec", re.IGNORECASE
)
_DISCONNECT_REASON_RE = re.compile(r"reason\s*0x([0-9a-fA-F]{2})", re.IGNORECASE)
_UPOWER_PERCENTAGE_RE = re.compile(r"percentage:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)
_UPOWER_SERIAL_RE = re.compile(r"serial:\s*([^\n]+)", re.IGNORECASE)
_UPOWER_NATIVE_PATH_RE = re.compile(r"native-path:\s*([^\n]+)", re.IGNORECASE)


class ControllerHub:
    """Track a Bluetooth controller via evdev and expose status snapshots."""

    def __init__(
        self,
        *,
        target_address: str = TARGET_CONTROLLER_ADDRESS,
        target_name_hint: str | None = None,
        scan_interval_sec: float = 0.2,
        reconnect_delay_sec: float = 0.05,
        bluetooth_connect_interval_sec: float = 5.0,
        battery_refresh_interval_sec: float = 20.0,
        battery_unknown_retry_interval_sec: float = 2.0,
        auto_start: bool = True,
        evdev_module=_UNSET,
    ) -> None:
        self._evdev = evdev if evdev_module is _UNSET else evdev_module
        self._target_address = self._normalize_address(target_address)
        self._target_name_hint = str(target_name_hint or "").strip().lower()
        self._scan_interval_sec = max(0.1, float(scan_interval_sec))
        self._reconnect_delay_sec = max(0.05, float(reconnect_delay_sec))
        self._bluetooth_connect_interval_sec = max(1.0, float(bluetooth_connect_interval_sec))
        self._battery_refresh_interval_sec = max(5.0, float(battery_refresh_interval_sec))
        # While connected but the OS has not yet published a battery reading
        # (common in the first seconds after a Bluetooth connect), poll on this
        # shorter cadence so the status appears promptly instead of waiting a
        # full refresh interval.
        self._battery_unknown_retry_interval_sec = max(
            0.5, min(float(battery_unknown_retry_interval_sec), self._battery_refresh_interval_sec)
        )

        self._lock = threading.Lock()
        self._enabled = self._evdev is not None
        self._connected = False
        self._device_name = ""
        self._device_path = ""
        self._device_address = self._target_address
        self._pressed_buttons: set[str] = set()
        self._pressed_buttons_by_device: dict[str, set[str]] = {}
        # Press (down) edges captured by the input thread since the last drain.
        # Latching edges here means a quick tap is never missed even if it
        # happens entirely between two main-loop input samples.
        self._just_pressed: set[str] = set()
        self._last_event_monotonic: float | None = None
        self._last_bluetooth_connect_attempt_monotonic: float = 0.0
        self._bluetooth_connect_attempts = 0
        self._bluetooth_connect_failures = 0
        self._battery_percentage: int | None = None
        self._battery_updated_monotonic: float | None = None
        self._battery_source: str | None = None
        self._battery_poll_duration_ms: int | None = None
        # Link metrics come from a mixed int/str/None Bluetooth metrics dict and are
        # only ever surfaced in reporting payloads, so they keep that broad type.
        self._rssi_dbm: int | str | None = None
        self._tx_power_dbm: int | str | None = None
        self._link_quality: int | str | None = None
        self._signal_source: str | None = None
        self._connection_interval_ms: int | str | None = None
        self._connection_latency: int | str | None = None
        self._supervision_timeout_ms: int | str | None = None
        self._connection_params_source: str | None = None
        self._bluetooth_metrics_updated_monotonic: float | None = None
        self._bluetooth_metrics_poll_duration_ms: int | None = None
        self._last_disconnect_reason_code: str | None = None
        self._disconnect_reason_counts: dict[str, int] = {}
        self._button_event_sequence = 0
        self._button_events: list[dict[str, Any]] = []

        self._running = False
        self._thread: threading.Thread | None = None
        self._battery_thread: threading.Thread | None = None
        # Signals the battery poller to read immediately (e.g. on connect).
        self._battery_wakeup = threading.Event()

        if not self._enabled:
            logger.warning("Controller disabled: evdev is not available")
            return

        if auto_start:
            self.start()

    def start(self) -> None:
        """Start the input-reading and battery-polling background threads."""
        if not self._enabled:
            return
        self._running = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        # Battery polling runs on a dedicated thread so its blocking
        # subprocess calls never stall input event reading.
        if self._battery_thread is None or not self._battery_thread.is_alive():
            self._battery_thread = threading.Thread(target=self._battery_worker, daemon=True)
            self._battery_thread.start()

    def get_status_snapshot(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of connection and button state."""
        with self._lock:
            return {
                "enabled": bool(self._enabled),
                "connected": bool(self._connected),
                "address": self._device_address or self._target_address,
                "device_name": self._device_name,
                "pressed_buttons": sorted(self._pressed_buttons),
                "last_event_monotonic": self._last_event_monotonic,
                "bluetooth_connect_attempts": int(self._bluetooth_connect_attempts),
                "bluetooth_connect_failures": int(self._bluetooth_connect_failures),
                "last_bluetooth_connect_attempt_monotonic": self._last_bluetooth_connect_attempt_monotonic,
                "battery_percentage": self._battery_percentage,
                "battery_updated_monotonic": self._battery_updated_monotonic,
                "battery_source": self._battery_source,
                "battery_poll_duration_ms": self._battery_poll_duration_ms,
                "rssi_dbm": self._rssi_dbm,
                "tx_power_dbm": self._tx_power_dbm,
                "link_quality": self._link_quality,
                "signal_source": self._signal_source,
                "connection_interval_ms": self._connection_interval_ms,
                "connection_latency": self._connection_latency,
                "supervision_timeout_ms": self._supervision_timeout_ms,
                "connection_params_source": self._connection_params_source,
                "bluetooth_metrics_updated_monotonic": self._bluetooth_metrics_updated_monotonic,
                "bluetooth_metrics_poll_duration_ms": self._bluetooth_metrics_poll_duration_ms,
                "last_disconnect_reason_code": self._last_disconnect_reason_code,
                "disconnect_reason_counts": dict(self._disconnect_reason_counts),
                "recent_button_events": list(self._button_events[-50:]),
            }

    def drain_pressed_events(self) -> set[str]:
        """Return and clear button down-edges captured since the last call.

        Edges are recorded by the input thread the instant a button is pressed,
        so consumers that poll at a lower rate (e.g. the render loop) never miss
        a quick tap that goes down and up between two samples.
        """
        with self._lock:
            edges = self._just_pressed
            self._just_pressed = set()
            return edges

    def _worker(self) -> None:
        while self._running:
            devices = self._find_matching_devices()
            if not devices:
                self._set_disconnected()
                self._request_bluetooth_connect()
                time.sleep(self._scan_interval_sec)
                continue

            self._set_connected_devices(devices)
            reconnect_needed = False
            try:
                for device in devices:
                    try:
                        device.set_blocking(False)
                    except Exception:
                        pass

                while self._running:
                    ready, _, _ = select.select(devices, [], [], self._scan_interval_sec)
                    if not ready:
                        continue

                    for device in ready:
                        try:
                            events = device.read()
                        except BlockingIOError:
                            continue
                        except OSError as exc:
                            if exc.errno == errno.ENODEV:
                                reconnect_needed = True
                                break
                            logger.warning("Controller input read error: %s", exc)
                            continue

                        for event in events:
                            if not self._running:
                                break
                            if event.type == self._evdev.ecodes.EV_KEY:
                                label = self._map_button_label(event.code)
                                if label is None:
                                    continue
                                self._apply_button_state(
                                    label, event.value, device_path=getattr(device, "path", "")
                                )
                                continue

                            if event.type == self._evdev.ecodes.EV_ABS:
                                self._apply_abs_state(
                                    event.code, event.value, device_path=getattr(device, "path", "")
                                )
                                continue

                    if not self._running:
                        break
                    if reconnect_needed:
                        break
            except OSError as exc:
                if exc.errno == errno.ENODEV:
                    logger.info("Controller disconnected, waiting for reconnect: %s", exc)
                else:
                    logger.warning("Controller input read error: %s", exc)
            except Exception as exc:
                logger.warning("Controller input read error: %s", exc)
            finally:
                self._set_disconnected()
                for device in devices:
                    try:
                        device.close()
                    except Exception:
                        pass
                # Keep reconnect latency low after transient BT device drops.
                time.sleep(self._reconnect_delay_sec)

    def _request_bluetooth_connect(self) -> None:
        if not self._target_address:
            return

        now = time.monotonic()
        if (
            now - self._last_bluetooth_connect_attempt_monotonic
            < self._bluetooth_connect_interval_sec
        ):
            return
        with self._lock:
            self._last_bluetooth_connect_attempt_monotonic = now
            self._bluetooth_connect_attempts += 1

        try:
            result = subprocess.run(
                ["bluetoothctl", "connect", self._target_address],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception as exc:
            with self._lock:
                self._bluetooth_connect_failures += 1
            logger.debug(
                "Controller reconnect attempt failed: address=%s error=%s",
                self._target_address,
                exc,
            )
            return

        if result.returncode == 0:
            logger.info("Controller reconnect requested: address=%s", self._target_address)
        else:
            with self._lock:
                self._bluetooth_connect_failures += 1
            logger.debug(
                "Controller reconnect not ready: address=%s returncode=%s output=%s",
                self._target_address,
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )

    def _find_matching_devices(self):
        candidates: list[tuple[int, Any]] = []
        try:
            for path in self._evdev.list_devices():
                try:
                    device = self._evdev.InputDevice(path)
                except Exception:
                    continue
                if not self._device_matches(device):
                    try:
                        device.close()
                    except Exception:
                        pass
                    continue

                score = self._device_score(device)
                candidates.append((score, device))
        except Exception as exc:
            logger.warning("Controller scan failed: %s", exc)

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [device for _, device in candidates]

    def _find_matching_device(self):
        devices = self._find_matching_devices()
        if not devices:
            return None

        selected_device = devices[0]
        selected_score = self._device_score(selected_device)
        for device in devices[1:]:
            try:
                device.close()
            except Exception:
                pass

        logger.info(
            "Controller connected: name=%s path=%s address=%s score=%s candidates=%s",
            getattr(selected_device, "name", ""),
            getattr(selected_device, "path", ""),
            self._extract_device_address(selected_device),
            selected_score,
            len(devices),
        )
        return selected_device

    def _device_matches(self, device) -> bool:
        candidates = [
            self._normalize_address(getattr(device, "uniq", "")),
            self._normalize_address(getattr(device, "phys", "")),
        ]
        # If an explicit target address is configured, keep matching strict to
        # that address so hubs for similarly named controllers do not overlap.
        if self._target_address:
            for candidate in candidates:
                if candidate and self._target_address in candidate:
                    return True
            return False

        if self._target_name_hint:
            name = str(getattr(device, "name", "") or "").strip().lower()
            if self._target_name_hint in name:
                return True
        return False

    def _extract_device_address(self, device) -> str:
        for raw in (getattr(device, "uniq", ""), getattr(device, "phys", "")):
            candidate = self._normalize_address(raw)
            if self._target_address in candidate:
                return self._target_address
        return self._target_address

    def _device_score(self, device) -> int:
        """Score how likely a matching evdev node is the primary gamepad input."""
        ec = self._evdev.ecodes
        score = 0

        key_codes = self._capability_codes(device, ec.EV_KEY)
        abs_codes = self._capability_codes(device, ec.EV_ABS)

        primary_keys = (
            getattr(ec, "BTN_SOUTH", None),
            getattr(ec, "BTN_EAST", None),
            getattr(ec, "BTN_WEST", None),
            getattr(ec, "BTN_NORTH", None),
            getattr(ec, "BTN_START", None),
            getattr(ec, "BTN_SELECT", None),
        )
        dpad_keys = (
            getattr(ec, "BTN_DPAD_UP", None),
            getattr(ec, "BTN_DPAD_DOWN", None),
            getattr(ec, "BTN_DPAD_LEFT", None),
            getattr(ec, "BTN_DPAD_RIGHT", None),
        )
        analog_axes = (
            getattr(ec, "ABS_X", None),
            getattr(ec, "ABS_Y", None),
            getattr(ec, "ABS_RX", None),
            getattr(ec, "ABS_RY", None),
            getattr(ec, "ABS_HAT0X", None),
            getattr(ec, "ABS_HAT0Y", None),
        )

        for code in primary_keys:
            if code is not None and code in key_codes:
                score += 3

        for code in dpad_keys:
            if code is not None and code in key_codes:
                score += 2

        for code in analog_axes:
            if code is not None and code in abs_codes:
                score += 2

        # Prefer explicit hat axes because many D-pads report through ABS.
        if getattr(ec, "ABS_HAT0X", None) in abs_codes:
            score += 3
        if getattr(ec, "ABS_HAT0Y", None) in abs_codes:
            score += 3

        if score == 0:
            # Keep address-only matches eligible as a last resort.
            score = 1
        return score

    def _capability_codes(self, device, event_type: int) -> set[int]:
        try:
            capabilities = device.capabilities(absinfo=False)
        except Exception:
            return set()

        raw_codes = capabilities.get(event_type, [])
        out: set[int] = set()
        for item in raw_codes:
            # Some evdev backends may include tuples with metadata.
            code = item[0] if isinstance(item, tuple) else item
            try:
                out.add(int(code))
            except Exception:
                continue
        return out

    def _set_connected(self, device) -> None:
        with self._lock:
            self._connected = True
            self._device_name = str(getattr(device, "name", ""))
            self._device_path = str(getattr(device, "path", ""))
            self._device_address = self._extract_device_address(device)
            self._pressed_buttons.clear()
            self._pressed_buttons_by_device = {}
            self._just_pressed.clear()
            self._battery_percentage = None
            self._battery_updated_monotonic = None
            self._battery_source = None
            self._battery_poll_duration_ms = None
            self._rssi_dbm = None
            self._tx_power_dbm = None
            self._link_quality = None
            self._signal_source = None
            self._connection_interval_ms = None
            self._connection_latency = None
            self._supervision_timeout_ms = None
            self._connection_params_source = None
            self._bluetooth_metrics_updated_monotonic = None
            self._bluetooth_metrics_poll_duration_ms = None

    def _set_connected_devices(self, devices) -> None:
        if not devices:
            self._set_disconnected()
            return

        primary = devices[0]
        names = sorted(
            {str(getattr(device, "name", "") or "") for device in devices if device is not None}
        )
        paths = [str(getattr(device, "path", "") or "") for device in devices if device is not None]
        with self._lock:
            self._connected = True
            self._device_name = ", ".join(name for name in names if name)
            self._device_path = ",".join(path for path in paths if path)
            self._device_address = self._extract_device_address(primary)
            self._pressed_buttons.clear()
            self._pressed_buttons_by_device = {path: set() for path in paths if path}
            self._just_pressed.clear()
            self._battery_percentage = None
            self._battery_updated_monotonic = None
            self._battery_source = None
            self._battery_poll_duration_ms = None
            self._rssi_dbm = None
            self._tx_power_dbm = None
            self._link_quality = None
            self._signal_source = None
            self._connection_interval_ms = None
            self._connection_latency = None
            self._supervision_timeout_ms = None
            self._connection_params_source = None
            self._bluetooth_metrics_updated_monotonic = None
            self._bluetooth_metrics_poll_duration_ms = None

        logger.info(
            "Controller connected: address=%s devices=%s names=%s",
            self._device_address,
            len(paths),
            self._device_name,
        )
        # Trigger a prompt battery read on the dedicated poller thread.
        self._battery_wakeup.set()

    def _set_disconnected(self) -> None:
        with self._lock:
            was_connected = bool(self._connected)
            address = self._device_address or self._target_address

        reason_code = self._read_recent_disconnect_reason(address) if was_connected else None

        with self._lock:
            self._connected = False
            self._pressed_buttons.clear()
            self._pressed_buttons_by_device = {}
            self._just_pressed.clear()
            self._battery_percentage = None
            self._battery_updated_monotonic = None
            self._battery_source = None
            self._battery_poll_duration_ms = None
            self._rssi_dbm = None
            self._tx_power_dbm = None
            self._link_quality = None
            self._signal_source = None
            self._connection_interval_ms = None
            self._connection_latency = None
            self._supervision_timeout_ms = None
            self._connection_params_source = None
            self._bluetooth_metrics_updated_monotonic = None
            self._bluetooth_metrics_poll_duration_ms = None
            if reason_code:
                self._last_disconnect_reason_code = reason_code
                self._disconnect_reason_counts[reason_code] = (
                    int(self._disconnect_reason_counts.get(reason_code, 0)) + 1
                )

    def _apply_button_state(self, label: str, value: int, device_path: str = "") -> None:
        ts = time.monotonic()
        path_key = device_path or "__default__"
        with self._lock:
            was_pressed = label in self._pressed_buttons
            path_buttons = self._pressed_buttons_by_device.setdefault(path_key, set())
            if int(value) == 0:
                path_buttons.discard(label)
            else:
                path_buttons.add(label)
            aggregate = set()
            for buttons in self._pressed_buttons_by_device.values():
                aggregate.update(buttons)
            self._pressed_buttons = aggregate
            if (label in aggregate) and not was_pressed:
                # Latch the down edge so consumers polling at a lower rate than
                # the input thread never miss a quick press/release.
                self._just_pressed.add(label)
            is_pressed = label in aggregate
            if is_pressed != was_pressed:
                self._button_event_sequence += 1
                self._button_events.append(
                    {
                        "sequence": self._button_event_sequence,
                        "button": label,
                        "event": "pressed" if is_pressed else "released",
                        "monotonic": ts,
                    }
                )
                self._button_events = self._button_events[-100:]
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

        # Keyboard-profile fallbacks for controllers that expose key events
        # instead of gamepad buttons in alternate switch modes.
        keyboard_map = {
            "KEY_UP": "D-Up",
            "KEY_DOWN": "D-Down",
            "KEY_LEFT": "D-Left",
            "KEY_RIGHT": "D-Right",
            "KEY_W": "D-Up",
            "KEY_S": "D-Down",
            "KEY_A": "D-Left",
            "KEY_D": "D-Right",
            "KEY_ENTER": "A",
            "KEY_SPACE": "B",
            "KEY_B": "B",
            "KEY_ESC": "B",
            "KEY_BACKSPACE": "B",
            "KEY_Z": "X",
            "KEY_X": "Y",
            "KEY_Q": "L1",
            "KEY_E": "R1",
            "KEY_1": "L2",
            "KEY_3": "R2",
            "KEY_TAB": "Select",
            "KEY_F1": "Start",
            "KEY_HOME": "Home",
        }
        for key_name, mapped_label in keyboard_map.items():
            key_code = getattr(ec, key_name, None)
            if key_code is not None:
                label_map[key_code] = mapped_label
        return label_map.get(code)

    def _apply_abs_state(self, code: int, value: int, device_path: str = "") -> None:
        ec = self._evdev.ecodes

        if code == ec.ABS_HAT0X:
            # Hat X axis: -1 left, 0 neutral, +1 right
            self._apply_button_state("D-Left", 1 if int(value) < 0 else 0, device_path=device_path)
            self._apply_button_state("D-Right", 1 if int(value) > 0 else 0, device_path=device_path)
            return

        if code == ec.ABS_HAT0Y:
            # Hat Y axis: -1 up, 0 neutral, +1 down
            self._apply_button_state("D-Up", 1 if int(value) < 0 else 0, device_path=device_path)
            self._apply_button_state("D-Down", 1 if int(value) > 0 else 0, device_path=device_path)
            return

    @staticmethod
    def _normalize_address(value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("-", ":")
        return normalized

    def _battery_worker(self) -> None:
        """Poll controller battery off the input thread.

        Battery reads shell out to ``upower``/``bluetoothctl`` with multi-second
        timeouts. Running them on a dedicated thread keeps those blocking calls
        away from the input event loop, so button presses are never delayed
        waiting on a battery refresh.
        """
        while self._running:
            with self._lock:
                connected = self._connected
                address = self._device_address or self._target_address

            if connected:
                battery_poll_start = time.monotonic()
                percentage, battery_source = self._read_battery_percentage(address)
                battery_poll_duration_ms = int(
                    round((time.monotonic() - battery_poll_start) * 1000)
                )

                bluetooth_poll_start = time.monotonic()
                bluetooth_metrics = self._read_bluetooth_link_metrics(address)
                bluetooth_poll_duration_ms = int(
                    round((time.monotonic() - bluetooth_poll_start) * 1000)
                )
                with self._lock:
                    if self._connected:
                        now = time.monotonic()
                        self._battery_percentage = percentage
                        self._battery_updated_monotonic = now
                        self._battery_source = battery_source
                        self._battery_poll_duration_ms = battery_poll_duration_ms
                        self._rssi_dbm = bluetooth_metrics.get("rssi_dbm")
                        self._tx_power_dbm = bluetooth_metrics.get("tx_power_dbm")
                        self._link_quality = bluetooth_metrics.get("link_quality")
                        self._signal_source = (
                            str(bluetooth_metrics.get("signal_source") or "") or None
                        )
                        self._connection_interval_ms = bluetooth_metrics.get(
                            "connection_interval_ms"
                        )
                        self._connection_latency = bluetooth_metrics.get("connection_latency")
                        self._supervision_timeout_ms = bluetooth_metrics.get(
                            "supervision_timeout_ms"
                        )
                        self._connection_params_source = (
                            str(bluetooth_metrics.get("connection_params_source") or "") or None
                        )
                        self._bluetooth_metrics_updated_monotonic = now
                        self._bluetooth_metrics_poll_duration_ms = bluetooth_poll_duration_ms
                if percentage is None:
                    # The battery level is often not published the instant a
                    # controller connects; retry quickly until a value appears
                    # rather than waiting the full refresh interval.
                    wait_seconds = self._battery_unknown_retry_interval_sec
                else:
                    wait_seconds = self._battery_refresh_interval_sec
            else:
                # Idle cheaply until a controller connects and wakes us.
                wait_seconds = self._scan_interval_sec

            # Sleep until the next refresh, but return early when a connect
            # event requests a fresh read.
            if self._battery_wakeup.wait(wait_seconds):
                self._battery_wakeup.clear()

    def _read_bluetooth_battery_percentage(self, address: str) -> int | None:
        output = self._read_bluetoothctl_info(address)
        if output is None:
            return None
        return self._parse_bluetoothctl_battery_percentage(output)

    def _read_bluetooth_link_metrics(self, address: str) -> dict[str, int | str | None]:
        dbus_metrics = self._read_dbus_link_metrics(address)
        if self._link_metrics_available(dbus_metrics):
            return dbus_metrics

        btmgmt_metrics = self._read_btmgmt_link_metrics(address)
        if self._link_metrics_available(btmgmt_metrics):
            return btmgmt_metrics

        output = self._read_bluetoothctl_info(address)
        if output is None:
            return {
                "rssi_dbm": None,
                "tx_power_dbm": None,
                "link_quality": None,
                "signal_source": "none",
                "connection_interval_ms": None,
                "connection_latency": None,
                "supervision_timeout_ms": None,
                "connection_params_source": "none",
            }
        metrics = self._parse_bluetoothctl_link_metrics(output)
        metrics.update(self._parse_bluetoothctl_connection_params(output))
        metrics["signal_source"] = "bluetoothctl"
        metrics["connection_params_source"] = "bluetoothctl"
        return metrics

    @staticmethod
    def _address_to_bluez_path(address: str) -> str:
        normalized_address = str(address or "").strip().upper().replace(":", "_")
        return f"/org/bluez/hci0/dev_{normalized_address}"

    @staticmethod
    def _parse_busctl_int(output: str) -> int | None:
        parts = (output or "").strip().split(maxsplit=1)
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except (TypeError, ValueError):
            return None

    def _read_bluez_device_property_int(self, address: str, prop_name: str) -> int | None:
        if not address:
            return None
        object_path = self._address_to_bluez_path(address)
        try:
            result = subprocess.run(
                [
                    "busctl",
                    "get-property",
                    "org.bluez",
                    object_path,
                    "org.bluez.Device1",
                    prop_name,
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None
        return self._parse_busctl_int(result.stdout)

    def _read_dbus_link_metrics(self, address: str) -> dict[str, int | str | None]:
        metrics = {
            "rssi_dbm": self._read_bluez_device_property_int(address, "RSSI"),
            "tx_power_dbm": self._read_bluez_device_property_int(address, "TxPower"),
            "link_quality": None,
            "signal_source": "dbus",
            "connection_interval_ms": None,
            "connection_latency": None,
            "supervision_timeout_ms": None,
            "connection_params_source": "none",
        }
        return metrics

    def _read_btmgmt_link_metrics(self, address: str) -> dict[str, int | str | None]:
        none_metrics: dict[str, int | str | None] = {
            "rssi_dbm": None,
            "tx_power_dbm": None,
            "link_quality": None,
            "signal_source": "btmgmt",
            "connection_interval_ms": None,
            "connection_latency": None,
            "supervision_timeout_ms": None,
            "connection_params_source": "btmgmt",
        }
        if not address:
            return none_metrics

        output = self._run_btmgmt_conn_info(address)
        if output is None:
            return none_metrics

        def parse_int(regex: re.Pattern[str]) -> int | None:
            match = regex.search(output)
            if match is None:
                return None
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None

        return {
            "rssi_dbm": parse_int(_BTMGMT_RSSI_RE),
            "tx_power_dbm": parse_int(_BTMGMT_TX_POWER_RE),
            "link_quality": parse_int(_BTMGMT_LINK_QUALITY_RE),
            "signal_source": "btmgmt",
            "connection_interval_ms": self._parse_rounded_ms(_BTMGMT_CONN_INTERVAL_MS_RE, output),
            "connection_latency": parse_int(_BTMGMT_CONN_LATENCY_RE),
            "supervision_timeout_ms": self._parse_rounded_ms(
                _BTMGMT_SUPERVISION_TIMEOUT_MS_RE, output
            ),
            "connection_params_source": "btmgmt",
        }

    def _run_btmgmt_conn_info(self, address: str) -> str | None:
        """Return ``btmgmt conn-info`` stdout, trying BR/EDR then LE address types.

        btmgmt queries BR/EDR by default; BLE peripherals require an explicit
        ``-t 1``/``-t 2`` or the command fails with "Not Connected". Return the
        output for the first address type that yields a successful read.
        """
        for type_args in _BTMGMT_ADDRESS_TYPE_ARGS:
            try:
                result = subprocess.run(
                    ["btmgmt", "conn-info", *type_args, address],
                    capture_output=True,
                    text=True,
                    # btmgmt emits no output when its stdin is /dev/null (the systemd
                    # default, StandardInput=null). Hand it an empty stdin pipe so it
                    # runs the command and prints the result instead of nothing.
                    input="",
                    timeout=1.5,
                    check=False,
                )
            except Exception:
                return None
            if result.returncode == 0 and "Connection Information" in (result.stdout or ""):
                return result.stdout
        return None

    @staticmethod
    def _link_metrics_available(metrics: dict[str, Any]) -> bool:
        return any(
            metrics.get(key) is not None
            for key in (
                "rssi_dbm",
                "tx_power_dbm",
                "link_quality",
                "connection_interval_ms",
                "connection_latency",
                "supervision_timeout_ms",
            )
        )

    @staticmethod
    def _parse_rounded_ms(regex: re.Pattern[str], output: str) -> int | None:
        match = regex.search(output or "")
        if match is None:
            return None
        try:
            return int(round(float(match.group(1))))
        except (TypeError, ValueError):
            return None

    def _read_recent_disconnect_reason(self, address: str) -> str | None:
        if not address:
            return None
        address_upper = address.upper()
        try:
            result = subprocess.run(
                ["journalctl", "-k", "-n", "200", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=1.2,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        lines = result.stdout.splitlines()
        for line in reversed(lines):
            if address_upper not in line.upper():
                continue
            match = _DISCONNECT_REASON_RE.search(line)
            if match is not None:
                return f"0x{match.group(1).lower()}"

        for line in reversed(lines):
            match = _DISCONNECT_REASON_RE.search(line)
            if match is not None:
                return f"0x{match.group(1).lower()}"
        return None

    def _read_bluetoothctl_info(self, address: str) -> str | None:
        if not address:
            return None

        try:
            result = subprocess.run(
                ["bluetoothctl", "info", address],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        return result.stdout

    def _read_battery_percentage(self, address: str) -> tuple[int | None, str | None]:
        # Prefer UPower (structured data over D-Bus) for more stable reads.
        percentage = self._read_upower_battery_percentage(address)
        if percentage is not None:
            return percentage, "upower"
        return self._read_bluetooth_battery_percentage(address), "bluetoothctl"

    def _read_upower_battery_percentage(self, address: str) -> int | None:
        normalized_address = self._normalize_address(address)
        if not normalized_address:
            return None

        token = normalized_address.upper().replace(":", "_")
        candidate_paths = [f"/org/freedesktop/UPower/devices/gaming_input_dev_{token}"]

        try:
            enum_result = subprocess.run(
                ["upower", "-e"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            enum_result = None

        if enum_result is not None and enum_result.returncode == 0:
            for raw_line in enum_result.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if (
                    line.startswith("/org/freedesktop/UPower/devices/gaming_input_dev_")
                    and line not in candidate_paths
                ):
                    candidate_paths.append(line)

        for object_path in candidate_paths:
            info = self._read_upower_object_info(object_path)
            if info is None:
                continue
            if not self._upower_info_matches_address(info, normalized_address):
                continue
            percentage = self._parse_upower_percentage(info)
            if percentage is not None:
                return percentage

        return None

    def _read_upower_object_info(self, object_path: str) -> str | None:
        try:
            result = subprocess.run(
                ["upower", "-i", object_path],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None
        return result.stdout

    def _upower_info_matches_address(self, info: str, normalized_address: str) -> bool:
        serial_match = _UPOWER_SERIAL_RE.search(info)
        if serial_match is not None:
            serial = self._normalize_address(serial_match.group(1))
            if serial and serial == normalized_address:
                return True

        native_path_match = _UPOWER_NATIVE_PATH_RE.search(info)
        if native_path_match is not None:
            native_path = self._normalize_address(native_path_match.group(1))
            address_token = normalized_address.replace(":", "_")
            if address_token and address_token in native_path:
                return True

        return False

    @staticmethod
    def _parse_upower_percentage(output: str) -> int | None:
        if not output:
            return None

        match = _UPOWER_PERCENTAGE_RE.search(output)
        if match is None:
            return None

        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            return None

        rounded = int(round(value))
        if 0 <= rounded <= 100:
            return rounded
        return None

    @staticmethod
    def _parse_bluetoothctl_battery_percentage(output: str) -> int | None:
        if not output:
            return None

        # Prefer decimal value in parenthesis when bluetoothctl reports
        # both forms, e.g. "Battery Percentage: 0x5c (92)".
        match = _BATTERY_PERCENTAGE_HEX_WITH_DEC_RE.search(output)
        if match is None:
            match = _BATTERY_PERCENTAGE_DEC_RE.search(output)
        if match is None:
            return None

        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            return None

        if 0 <= value <= 100:
            return value
        return None

    @staticmethod
    def _parse_bluetoothctl_link_metrics(output: str) -> dict[str, int | str | None]:
        def parse_int(regex: re.Pattern[str]) -> int | None:
            match = regex.search(output or "")
            if match is None:
                return None
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None

        return {
            "rssi_dbm": parse_int(_BLUETOOTHCTL_RSSI_RE),
            "tx_power_dbm": parse_int(_BLUETOOTHCTL_TX_POWER_RE),
            "link_quality": parse_int(_BLUETOOTHCTL_LINK_QUALITY_RE),
        }

    @staticmethod
    def _parse_bluetoothctl_connection_params(output: str) -> dict[str, int | None]:
        def parse_int(regex: re.Pattern[str]) -> int | None:
            match = regex.search(output or "")
            if match is None:
                return None
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None

        return {
            "connection_interval_ms": ControllerHub._parse_rounded_ms(
                _BLUETOOTHCTL_CONN_INTERVAL_MS_RE, output
            ),
            "connection_latency": parse_int(_BLUETOOTHCTL_CONN_LATENCY_RE),
            "supervision_timeout_ms": ControllerHub._parse_rounded_ms(
                _BLUETOOTHCTL_SUPERVISION_TIMEOUT_MS_RE, output
            ),
        }
