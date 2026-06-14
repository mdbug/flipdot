from __future__ import annotations

import json
from pathlib import Path
import threading


class RuntimeSettingsStore:
    """Persists lightweight runtime settings to a JSON file."""

    def __init__(self, settings_path: Path) -> None:
        self._settings_path = settings_path
        self._lock = threading.Lock()

    def load_sleep_settings(self) -> dict[str, int | bool] | None:
        with self._lock:
            try:
                payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return None

        sleep_payload = payload.get("sleep") if isinstance(payload, dict) else None
        if not isinstance(sleep_payload, dict):
            return None

        try:
            return {
                "enabled": bool(sleep_payload.get("enabled", True)),
                "start_hour": max(0, min(23, int(sleep_payload.get("start_hour", 0)))),
                "end_hour": max(0, min(23, int(sleep_payload.get("end_hour", 7)))),
            }
        except (TypeError, ValueError):
            return None

    def save_sleep_settings(self, *, enabled: bool, start_hour: int, end_hour: int) -> None:
        clamped = {
            "enabled": bool(enabled),
            "start_hour": max(0, min(23, int(start_hour))),
            "end_hour": max(0, min(23, int(end_hour))),
        }

        with self._lock:
            payload: dict[str, object]
            try:
                current = json.loads(self._settings_path.read_text(encoding="utf-8"))
                payload = current if isinstance(current, dict) else {}
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                payload = {}

            payload["sleep"] = clamped

            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
