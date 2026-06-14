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
        payload = self._load_payload()
        if payload is None:
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
        self._save_section("sleep", clamped)

    def load_font_preview_settings(self) -> dict[str, object] | None:
        payload = self._load_payload()
        if payload is None:
            return None

        preview_payload = payload.get("font_preview") if isinstance(payload, dict) else None
        if not isinstance(preview_payload, dict):
            return None

        phrase = preview_payload.get("phrase", "")
        if not isinstance(phrase, str):
            return None

        cleaned = " ".join(phrase.split())
        if not cleaned:
            return None

        spacing_raw = preview_payload.get("spacing", 0)
        try:
            spacing = max(0, min(6, int(spacing_raw)))
        except (TypeError, ValueError):
            spacing = 0

        return {
            "phrase": cleaned[:32],
            "spacing": spacing,
            "variants": self._normalize_font_preview_variants(preview_payload.get("variants")),
        }

    def save_font_preview_settings(self, *, phrase: str, spacing: int, variants: list[dict[str, object]]) -> None:
        cleaned = " ".join(str(phrase).split())
        if not cleaned:
            cleaned = "FLIPDOT"

        clamped_spacing = max(0, min(6, int(spacing)))
        self._save_section(
            "font_preview",
            {
                "phrase": cleaned[:32],
                "spacing": clamped_spacing,
                "variants": self._normalize_font_preview_variants(variants),
            },
        )

    def _normalize_font_preview_variants(self, raw_variants: object) -> list[dict[str, object]]:
        if not isinstance(raw_variants, list):
            return []

        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, int, str]] = set()
        for item in raw_variants:
            if not isinstance(item, dict):
                continue

            family = item.get("family")
            size = item.get("size")
            style = item.get("style")
            if not isinstance(family, str) or not isinstance(style, str):
                continue

            try:
                size_int = int(size)
            except (TypeError, ValueError):
                continue

            key = (family, size_int, style)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"family": family, "size": size_int, "style": style})
            if len(normalized) >= 4:
                break

        return normalized

    def _load_payload(self) -> dict[str, object] | None:
        with self._lock:
            try:
                payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return None
        if not isinstance(payload, dict):
            return {}
        return payload

    def _save_section(self, section: str, value: object) -> None:
        with self._lock:
            try:
                current = json.loads(self._settings_path.read_text(encoding="utf-8"))
                payload = current if isinstance(current, dict) else {}
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                payload = {}

            payload[section] = value

            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
