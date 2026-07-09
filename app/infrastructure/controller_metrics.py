"""Records and aggregates Bluetooth controller connection/latency metrics.

Extracted from :class:`~app.infrastructure.web_server.WebServer` so that module
stays focused on HTTP/WebSocket wiring. The web server normalizes each raw
controller status snapshot (:meth:`normalize_status`), feeds the batch in via
:meth:`record`, reports panel-update timestamps via :meth:`record_panel_latency`,
and serves the aggregated view from :meth:`payload` at ``/api/controller/metrics``.
All buffers are hour-windowed and size-capped so memory stays bounded.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ControllerMetrics:
    """Bounded, hour-windowed recorder/aggregator for controller status samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[dict] = []
        self._events: list[dict] = []
        self._button_events: list[dict] = []
        self._panel_latency_events: list[dict] = []
        self._counters: dict[str, dict] = {}
        self._prev_connected: dict[str, bool] = {}
        self._last_button_sequence: dict[str, int] = {}
        self._last_latency_sequence: dict[str, int] = {}
        self._last_sample_monotonic = 0.0

    def record_panel_latency(self, panel_updated_monotonic: float) -> None:
        """Fold a panel-update timestamp into the button→panel latency series."""
        now_wall = time.time()
        with self._lock:
            latest_per_key: dict[str, dict] = {}
            for event in self._button_events:
                key = str(event.get("key", "") or "")
                if not key:
                    continue
                latest_per_key[key] = event

            for key, event in latest_per_key.items():
                try:
                    sequence = int(event.get("sequence", 0))
                except (TypeError, ValueError):
                    continue
                last_sequence = self._last_latency_sequence.get(key, 0)
                if sequence <= last_sequence:
                    continue
                monotonic_value: Any = event.get("monotonic")
                try:
                    event_monotonic = float(monotonic_value)
                except (TypeError, ValueError):
                    continue
                latency_ms = max(0.0, (panel_updated_monotonic - event_monotonic) * 1000.0)
                self._last_latency_sequence[key] = sequence
                self._panel_latency_events.append(
                    {
                        "timestamp": now_wall,
                        "key": key,
                        "sequence": sequence,
                        "latency_ms": latency_ms,
                    }
                )

            cutoff = now_wall - 3600.0
            self._panel_latency_events = [
                event
                for event in self._panel_latency_events
                if float(event.get("timestamp", 0.0)) >= cutoff
            ][-5000:]

    def record(self, statuses: list[dict]) -> None:
        """Record one round of normalized controller statuses (samples + events)."""
        now_monotonic = time.monotonic()
        now_wall = time.time()
        normalized_statuses = list(statuses)

        with self._lock:
            changed = False
            sample_statuses = []
            for index, status in enumerate(normalized_statuses):
                key = self._metric_key(index, status)
                label = f"P{index + 1}"
                connected = bool(status.get("connected", False))
                previous_connected = self._prev_connected.get(key)

                counter = self._counters.setdefault(
                    key,
                    {
                        "label": label,
                        "address": str(status.get("address", "") or ""),
                        "device_name": str(status.get("device_name", "") or ""),
                        "disconnects": 0,
                        "reconnects": 0,
                        "disconnect_reason_counts": {},
                    },
                )
                counter["label"] = label
                counter["address"] = str(status.get("address", "") or "")
                counter["device_name"] = str(status.get("device_name", "") or "")

                if previous_connected is not None and previous_connected != connected:
                    changed = True
                    event_type = "connected" if connected else "disconnected"
                    reason_code = (
                        str(status.get("last_disconnect_reason_code", "") or "")
                        if not connected
                        else ""
                    )
                    if connected:
                        counter["reconnects"] += 1
                    else:
                        counter["disconnects"] += 1
                        if reason_code:
                            reason_counts = counter.setdefault("disconnect_reason_counts", {})
                            reason_counts[reason_code] = int(reason_counts.get(reason_code, 0)) + 1
                    self._events.append(
                        {
                            "timestamp": now_wall,
                            "key": key,
                            "label": label,
                            "address": counter["address"],
                            "event": event_type,
                            "reason_code": reason_code or None,
                        }
                    )

                self._prev_connected[key] = connected
                pressed_buttons = status.get("pressed_buttons", [])
                if not isinstance(pressed_buttons, list):
                    pressed_buttons = []
                button_events = status.get("recent_button_events", [])
                if not isinstance(button_events, list):
                    button_events = []
                for button_event in button_events:
                    if not isinstance(button_event, dict):
                        continue
                    raw_sequence: Any = button_event.get("sequence")
                    try:
                        sequence = int(raw_sequence)
                    except (TypeError, ValueError):
                        continue
                    last_sequence = self._last_button_sequence.get(key, 0)
                    if sequence <= last_sequence:
                        continue
                    self._last_button_sequence[key] = sequence
                    event_monotonic: Any = button_event.get("monotonic")
                    event_monotonic_value = None
                    try:
                        event_monotonic_value = float(event_monotonic)
                    except (TypeError, ValueError):
                        event_timestamp = now_wall
                    else:
                        event_timestamp = now_wall - max(0.0, now_monotonic - event_monotonic_value)
                    self._button_events.append(
                        {
                            "timestamp": event_timestamp,
                            "key": key,
                            "label": label,
                            "address": counter["address"],
                            "sequence": sequence,
                            "button": str(button_event.get("button", "") or ""),
                            "event": str(button_event.get("event", "") or ""),
                            "monotonic": event_monotonic_value
                            if isinstance(event_monotonic_value, float)
                            else None,
                        }
                    )
                sample_statuses.append(
                    {
                        "key": key,
                        "label": label,
                        "address": counter["address"],
                        "connected": connected,
                        "last_event_age_ms": status.get("last_event_age_ms"),
                        "bluetooth_connect_attempts": status.get("bluetooth_connect_attempts"),
                        "bluetooth_connect_failures": status.get("bluetooth_connect_failures"),
                        "last_bluetooth_connect_attempt_age_ms": status.get(
                            "last_bluetooth_connect_attempt_age_ms"
                        ),
                        "battery_percentage": status.get("battery_percentage"),
                        "battery_source": status.get("battery_source"),
                        "battery_age_ms": status.get("battery_age_ms"),
                        "battery_poll_duration_ms": status.get("battery_poll_duration_ms"),
                        "rssi_dbm": status.get("rssi_dbm"),
                        "tx_power_dbm": status.get("tx_power_dbm"),
                        "link_quality": status.get("link_quality"),
                        "signal_source": status.get("signal_source"),
                        "connection_interval_ms": status.get("connection_interval_ms"),
                        "connection_latency": status.get("connection_latency"),
                        "supervision_timeout_ms": status.get("supervision_timeout_ms"),
                        "connection_params_source": status.get("connection_params_source"),
                        "last_disconnect_reason_code": status.get("last_disconnect_reason_code"),
                        "disconnect_reason_counts": status.get("disconnect_reason_counts"),
                        "bluetooth_metrics_age_ms": status.get("bluetooth_metrics_age_ms"),
                        "bluetooth_metrics_poll_duration_ms": status.get(
                            "bluetooth_metrics_poll_duration_ms"
                        ),
                        "pressed_count": len(pressed_buttons),
                        "pressed_buttons": [str(item) for item in pressed_buttons],
                    }
                )

            should_sample = changed or (now_monotonic - self._last_sample_monotonic >= 0.5)
            if not should_sample:
                return

            self._samples.append(
                {
                    "timestamp": now_wall,
                    "controllers": sample_statuses,
                }
            )
            self._last_sample_monotonic = now_monotonic

            cutoff = now_wall - 3600.0
            self._samples = [
                sample for sample in self._samples if float(sample.get("timestamp", 0.0)) >= cutoff
            ][-7200:]
            self._events = [
                event for event in self._events if float(event.get("timestamp", 0.0)) >= cutoff
            ][-1000:]
            self._button_events = [
                event
                for event in self._button_events
                if float(event.get("timestamp", 0.0)) >= cutoff
            ][-5000:]

    def payload(self) -> dict:
        """Return the hour-windowed metrics summary for the metrics dashboard."""
        with self._lock:
            samples = list(self._samples)
            events = list(self._events)
            button_events = list(self._button_events)
            panel_latency_events = list(self._panel_latency_events)
            counters = {key: dict(value) for key, value in self._counters.items()}

        now_wall = time.time()
        window_sec = 3600
        window_hours = window_sec / 3600.0
        summaries = []
        for key, counter in counters.items():
            controller_samples = []
            connected_samples = 0
            freshness_values = []
            rssi_values = []
            interval_values = []
            supervision_values = []
            connection_latency_values = []
            button_event_count = 0
            for sample in samples:
                for status in sample.get("controllers", []):
                    if status.get("key") != key:
                        continue
                    controller_samples.append(status)
                    if status.get("connected"):
                        connected_samples += 1
                    age_ms = status.get("last_event_age_ms")
                    if isinstance(age_ms, (int, float)):
                        freshness_values.append(float(age_ms))
                    rssi_dbm = status.get("rssi_dbm")
                    if isinstance(rssi_dbm, (int, float)):
                        rssi_values.append(float(rssi_dbm))
                    interval_ms = status.get("connection_interval_ms")
                    if isinstance(interval_ms, (int, float)):
                        interval_values.append(float(interval_ms))
                    supervision_ms = status.get("supervision_timeout_ms")
                    if isinstance(supervision_ms, (int, float)):
                        supervision_values.append(float(supervision_ms))
                    conn_latency = status.get("connection_latency")
                    if isinstance(conn_latency, (int, float)):
                        connection_latency_values.append(float(conn_latency))

            for button_event in button_events:
                if button_event.get("key") == key:
                    button_event_count += 1

            sample_count = len(controller_samples)
            connected_ratio = (connected_samples / sample_count) if sample_count else 0.0
            average_event_age_ms = (
                (sum(freshness_values) / len(freshness_values)) if freshness_values else None
            )
            average_rssi_dbm = (sum(rssi_values) / len(rssi_values)) if rssi_values else None
            average_connection_interval_ms = (
                (sum(interval_values) / len(interval_values)) if interval_values else None
            )
            average_supervision_timeout_ms = (
                (sum(supervision_values) / len(supervision_values)) if supervision_values else None
            )
            average_connection_latency = (
                (sum(connection_latency_values) / len(connection_latency_values))
                if connection_latency_values
                else None
            )
            disconnect_count = int(counter.get("disconnects", 0))
            controller_events = sorted(
                [event for event in events if event.get("key") == key],
                key=lambda item: float(item.get("timestamp", 0.0)),
            )
            reconnect_durations_sec = []
            last_disconnect_ts = None
            for controller_event in controller_events:
                event_type = str(controller_event.get("event", "") or "")
                event_ts = float(controller_event.get("timestamp", 0.0))
                if event_type == "disconnected":
                    last_disconnect_ts = event_ts
                elif (
                    event_type == "connected"
                    and last_disconnect_ts is not None
                    and event_ts >= last_disconnect_ts
                ):
                    reconnect_durations_sec.append(event_ts - last_disconnect_ts)
                    last_disconnect_ts = None

            mttr_sec = (
                sum(reconnect_durations_sec) / len(reconnect_durations_sec)
                if reconnect_durations_sec
                else None
            )
            latency_values = [
                float(item.get("latency_ms", 0.0))
                for item in panel_latency_events
                if item.get("key") == key and isinstance(item.get("latency_ms"), (int, float))
            ]

            def percentile(values: list[float], p: float) -> float | None:
                if not values:
                    return None
                ordered = sorted(values)
                if len(ordered) == 1:
                    return ordered[0]
                rank = (len(ordered) - 1) * p
                low = int(rank)
                high = min(low + 1, len(ordered) - 1)
                weight = rank - low
                return ordered[low] * (1.0 - weight) + ordered[high] * weight

            latency_p50_ms = percentile(latency_values, 0.50)
            latency_p95_ms = percentile(latency_values, 0.95)
            latency_p99_ms = percentile(latency_values, 0.99)
            latest_status = controller_samples[-1] if controller_samples else None
            summaries.append(
                {
                    "key": key,
                    "label": counter.get("label", ""),
                    "address": counter.get("address", ""),
                    "device_name": counter.get("device_name", ""),
                    "disconnects": disconnect_count,
                    "reconnects": int(counter.get("reconnects", 0)),
                    "disconnects_per_hour": (disconnect_count / window_hours)
                    if window_hours > 0
                    else 0.0,
                    "mttr_sec": mttr_sec,
                    "connected_ratio": connected_ratio,
                    "average_event_age_ms": average_event_age_ms,
                    "average_rssi_dbm": average_rssi_dbm,
                    "average_connection_interval_ms": average_connection_interval_ms,
                    "average_supervision_timeout_ms": average_supervision_timeout_ms,
                    "average_connection_latency": average_connection_latency,
                    "button_event_count": button_event_count,
                    "disconnect_reason_counts": dict(counter.get("disconnect_reason_counts", {})),
                    "panel_latency_p50_ms": latency_p50_ms,
                    "panel_latency_p95_ms": latency_p95_ms,
                    "panel_latency_p99_ms": latency_p99_ms,
                    "panel_latency_samples": len(latency_values),
                    "latest": latest_status,
                }
            )

        return {
            "generated_at": now_wall,
            "window_sec": window_sec,
            "controllers": summaries,
            "samples": samples,
            "events": events,
            "button_events": button_events,
            "panel_latency_events": panel_latency_events,
        }

    @staticmethod
    def _metric_key(index: int, status: dict) -> str:
        address = str(status.get("address", "") or "").strip().lower()
        return address if address else f"index:{index}"

    def normalize_status(self, status: dict) -> dict:
        """Coerce a raw provider status dict into the canonical status shape."""
        if not isinstance(status, dict):
            return self.empty_status()

        pressed_buttons = status.get("pressed_buttons", [])
        if not isinstance(pressed_buttons, list):
            pressed_buttons = []

        battery_percentage = status.get("battery_percentage")
        if battery_percentage is None:
            normalized_battery = None
        else:
            try:
                parsed_battery = int(battery_percentage)
            except (TypeError, ValueError):
                normalized_battery = None
            else:
                normalized_battery = parsed_battery if 0 <= parsed_battery <= 100 else None

        last_event_monotonic: Any = status.get("last_event_monotonic")
        try:
            last_event_monotonic_value = float(last_event_monotonic)
        except (TypeError, ValueError):
            last_event_monotonic_value = None

        if last_event_monotonic_value is None:
            last_event_age_ms = None
        else:
            last_event_age_ms = max(
                0, int(round((time.monotonic() - last_event_monotonic_value) * 1000))
            )

        battery_updated_monotonic: Any = status.get("battery_updated_monotonic")
        try:
            battery_updated_monotonic_value = float(battery_updated_monotonic)
        except (TypeError, ValueError):
            battery_updated_monotonic_value = None
        if battery_updated_monotonic_value is None:
            battery_age_ms = None
        else:
            battery_age_ms = max(
                0, int(round((time.monotonic() - battery_updated_monotonic_value) * 1000))
            )

        bluetooth_metrics_updated_monotonic: Any = status.get("bluetooth_metrics_updated_monotonic")
        try:
            bluetooth_metrics_updated_monotonic_value = float(bluetooth_metrics_updated_monotonic)
        except (TypeError, ValueError):
            bluetooth_metrics_updated_monotonic_value = None
        if bluetooth_metrics_updated_monotonic_value is None:
            bluetooth_metrics_age_ms = None
        else:
            bluetooth_metrics_age_ms = max(
                0,
                int(round((time.monotonic() - bluetooth_metrics_updated_monotonic_value) * 1000)),
            )

        last_bluetooth_connect_attempt_monotonic: Any = status.get(
            "last_bluetooth_connect_attempt_monotonic"
        )
        try:
            last_bluetooth_connect_attempt_monotonic_value = float(
                last_bluetooth_connect_attempt_monotonic
            )
        except (TypeError, ValueError):
            last_bluetooth_connect_attempt_monotonic_value = None
        if last_bluetooth_connect_attempt_monotonic_value is None:
            last_bluetooth_connect_attempt_age_ms = None
        else:
            last_bluetooth_connect_attempt_age_ms = max(
                0,
                int(
                    round(
                        (time.monotonic() - last_bluetooth_connect_attempt_monotonic_value) * 1000
                    )
                ),
            )

        return {
            "enabled": bool(status.get("enabled", False)),
            "connected": bool(status.get("connected", False)),
            "address": str(status.get("address", "") or ""),
            "device_name": str(status.get("device_name", "") or ""),
            "pressed_buttons": [str(item) for item in pressed_buttons],
            "last_event_monotonic": last_event_monotonic_value,
            "last_event_age_ms": last_event_age_ms,
            "bluetooth_connect_attempts": self._normalize_int_or_none(
                status.get("bluetooth_connect_attempts")
            ),
            "bluetooth_connect_failures": self._normalize_int_or_none(
                status.get("bluetooth_connect_failures")
            ),
            "last_bluetooth_connect_attempt_monotonic": last_bluetooth_connect_attempt_monotonic_value,
            "last_bluetooth_connect_attempt_age_ms": last_bluetooth_connect_attempt_age_ms,
            "battery_percentage": normalized_battery,
            "battery_source": str(status.get("battery_source", "") or ""),
            "battery_updated_monotonic": battery_updated_monotonic_value,
            "battery_age_ms": battery_age_ms,
            "battery_poll_duration_ms": self._normalize_int_or_none(
                status.get("battery_poll_duration_ms")
            ),
            "rssi_dbm": self._normalize_int_or_none(status.get("rssi_dbm")),
            "tx_power_dbm": self._normalize_int_or_none(status.get("tx_power_dbm")),
            "link_quality": self._normalize_int_or_none(status.get("link_quality")),
            "signal_source": str(status.get("signal_source", "") or ""),
            "connection_interval_ms": self._normalize_int_or_none(
                status.get("connection_interval_ms")
            ),
            "connection_latency": self._normalize_int_or_none(status.get("connection_latency")),
            "supervision_timeout_ms": self._normalize_int_or_none(
                status.get("supervision_timeout_ms")
            ),
            "connection_params_source": str(status.get("connection_params_source", "") or ""),
            "last_disconnect_reason_code": str(status.get("last_disconnect_reason_code", "") or ""),
            "disconnect_reason_counts": self._normalize_reason_counts(
                status.get("disconnect_reason_counts", {})
            ),
            "bluetooth_metrics_updated_monotonic": bluetooth_metrics_updated_monotonic_value,
            "bluetooth_metrics_age_ms": bluetooth_metrics_age_ms,
            "bluetooth_metrics_poll_duration_ms": self._normalize_int_or_none(
                status.get("bluetooth_metrics_poll_duration_ms")
            ),
            "recent_button_events": self._normalize_button_events(
                status.get("recent_button_events", [])
            ),
        }

    @staticmethod
    def _normalize_int_or_none(value) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_button_events(value) -> list[dict]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value[-50:]:
            if not isinstance(item, dict):
                continue
            raw_sequence: Any = item.get("sequence")
            try:
                sequence = int(raw_sequence)
            except (TypeError, ValueError):
                continue
            raw_monotonic: Any = item.get("monotonic")
            try:
                monotonic = float(raw_monotonic)
            except (TypeError, ValueError):
                monotonic = None
            out.append(
                {
                    "sequence": sequence,
                    "button": str(item.get("button", "") or ""),
                    "event": str(item.get("event", "") or ""),
                    "monotonic": monotonic,
                }
            )
        return out

    @staticmethod
    def _normalize_reason_counts(value) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, int] = {}
        for key, raw_count in value.items():
            reason = str(key or "").strip().lower()
            if not reason:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                out[reason] = count
        return out

    @staticmethod
    def status_signature(statuses: list[dict]) -> tuple:
        """Return a change-detection signature over a list of controller statuses."""
        out: list[tuple] = []
        for status in statuses:
            buttons = status.get("pressed_buttons", [])
            if not isinstance(buttons, list):
                buttons = []
            out.append(
                (
                    bool(status.get("enabled", False)),
                    bool(status.get("connected", False)),
                    str(status.get("address", "") or ""),
                    str(status.get("device_name", "") or ""),
                    tuple(str(item) for item in buttons),
                    status.get("last_event_monotonic"),
                    status.get("last_event_age_ms"),
                    status.get("bluetooth_connect_attempts"),
                    status.get("bluetooth_connect_failures"),
                    status.get("last_bluetooth_connect_attempt_monotonic"),
                    status.get("last_bluetooth_connect_attempt_age_ms"),
                    status.get("battery_percentage"),
                    status.get("battery_source"),
                    status.get("battery_updated_monotonic"),
                    status.get("battery_age_ms"),
                    status.get("battery_poll_duration_ms"),
                    status.get("rssi_dbm"),
                    status.get("tx_power_dbm"),
                    status.get("link_quality"),
                    status.get("signal_source"),
                    status.get("connection_interval_ms"),
                    status.get("connection_latency"),
                    status.get("supervision_timeout_ms"),
                    status.get("connection_params_source"),
                    status.get("last_disconnect_reason_code"),
                    tuple(sorted((status.get("disconnect_reason_counts") or {}).items())),
                    status.get("bluetooth_metrics_updated_monotonic"),
                    status.get("bluetooth_metrics_age_ms"),
                    status.get("bluetooth_metrics_poll_duration_ms"),
                    tuple(
                        (
                            event.get("sequence"),
                            event.get("button"),
                            event.get("event"),
                            event.get("monotonic"),
                        )
                        for event in status.get("recent_button_events", [])
                        if isinstance(event, dict)
                    ),
                )
            )
        return tuple(out)

    @staticmethod
    def empty_status() -> dict:
        """Return the canonical status dict for a controller that is not present."""
        return {
            "enabled": False,
            "connected": False,
            "address": "",
            "device_name": "",
            "pressed_buttons": [],
            "last_event_monotonic": None,
            "last_event_age_ms": None,
            "bluetooth_connect_attempts": 0,
            "bluetooth_connect_failures": 0,
            "last_bluetooth_connect_attempt_monotonic": None,
            "last_bluetooth_connect_attempt_age_ms": None,
            "battery_percentage": None,
            "battery_source": "",
            "battery_updated_monotonic": None,
            "battery_age_ms": None,
            "battery_poll_duration_ms": None,
            "rssi_dbm": None,
            "tx_power_dbm": None,
            "link_quality": None,
            "signal_source": "",
            "connection_interval_ms": None,
            "connection_latency": None,
            "supervision_timeout_ms": None,
            "connection_params_source": "",
            "last_disconnect_reason_code": "",
            "disconnect_reason_counts": {},
            "bluetooth_metrics_updated_monotonic": None,
            "bluetooth_metrics_age_ms": None,
            "bluetooth_metrics_poll_duration_ms": None,
            "recent_button_events": [],
        }
