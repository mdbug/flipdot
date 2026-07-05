from __future__ import annotations

import json
import threading
from pathlib import Path

# Single source of truth for where runtime settings live: every composition
# root (main loop, web server default) must resolve to the same file, or two
# per-instance store locks end up racing on it.
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "state" / "settings.json"


class RuntimeSettingsStore:
    """Persists lightweight runtime settings to a JSON file."""

    def __init__(self, settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
        self._settings_path = settings_path
        self._lock = threading.Lock()

    def load_sleep_settings(self) -> dict[str, int | bool] | None:
        """Return the persisted sleep schedule, or None if unset/invalid."""
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
        """Persist the sleep schedule, clamping hours to 0-23."""
        clamped = {
            "enabled": bool(enabled),
            "start_hour": max(0, min(23, int(start_hour))),
            "end_hour": max(0, min(23, int(end_hour))),
        }
        self._save_section("sleep", clamped)

    def load_pose_settings(self) -> dict[str, bool] | None:
        """Return the persisted person-detection (auto chain) toggle, or None if unset."""
        payload = self._load_payload()
        if payload is None:
            return None

        pose_payload = payload.get("pose") if isinstance(payload, dict) else None
        if not isinstance(pose_payload, dict):
            return None

        return {"enabled": bool(pose_payload.get("enabled", True))}

    def save_pose_settings(self, *, enabled: bool) -> None:
        """Persist whether the person-driven auto chain may run."""
        self._save_section("pose", {"enabled": bool(enabled)})

    def load_clock_settings(self) -> dict[str, object] | None:
        """Return the persisted clock face style and second-hand toggle, or None."""
        payload = self._load_payload()
        if payload is None:
            return None

        clock_payload = payload.get("clock") if isinstance(payload, dict) else None
        if not isinstance(clock_payload, dict):
            return None

        style = clock_payload.get("style", "digital")
        if style not in ("digital", "analog"):
            style = "digital"
        return {"style": style, "seconds": bool(clock_payload.get("seconds", False))}

    def save_clock_settings(self, *, style: str, seconds: bool = False) -> None:
        """Persist the clock face style and second-hand toggle, defaulting unknown styles."""
        if style not in ("digital", "analog"):
            style = "digital"
        self._save_section("clock", {"style": style, "seconds": bool(seconds)})

    def load_font_preview_settings(self) -> dict[str, object] | None:
        """Return persisted font-preview settings, or None if unset/invalid."""
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

    def save_font_preview_settings(
        self, *, phrase: str, spacing: int, variants: list[dict[str, object]]
    ) -> None:
        """Persist font-preview phrase, spacing, and variants (deduped, clamped)."""
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

    def load_script_settings(self) -> dict[str, list[str]] | None:
        """Return persisted script-interlude exclusions, or None if unset/invalid."""
        payload = self._load_payload()
        if payload is None:
            return None

        script_payload = payload.get("scripts") if isinstance(payload, dict) else None
        if not isinstance(script_payload, dict):
            return None

        excluded = script_payload.get("excluded")
        if not isinstance(excluded, list):
            return None

        return {"excluded": sorted({item for item in excluded if isinstance(item, str)})}

    def save_script_settings(self, *, excluded: list[str]) -> None:
        """Persist the set of scripts excluded from the hourly clock interlude."""
        cleaned = sorted({str(name) for name in excluded})
        self._save_section("scripts", {"excluded": cleaned})

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
            if not isinstance(family, str) or not isinstance(style, str) or size is None:
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
            self._settings_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
