from __future__ import annotations

import time

import numpy as np

from app.services.fonts import available_families, available_sizes, available_styles, get_font_definition
from app.services.text import width, write


class FontPreview:
    CLICK_TIME_SEC = 0.8
    SCROLL_SPEED_PX_PER_SEC = 10.0
    MAX_COMPARE_VARIANTS = 4
    MAX_SPACING_PX = 6

    def __init__(self, width: int, height: int, mode_manager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self._phrase = "FLIPDOT"
        self._variants = self._build_variants()
        self._configured_variants = list(self._variants[: self.MAX_COMPARE_VARIANTS])
        self._spacing_px = 0
        self._window_start = 0
        self._hover_zone: str | None = None
        self._hover_started_at: float | None = None

    def _build_variants(self) -> list[tuple[str, int, str]]:
        variants: list[tuple[str, int, str]] = []
        for family in available_families():
            for size in available_sizes(family):
                for style in available_styles(family, size):
                    variants.append((family, size, style))
        if not variants:
            return [("classic", 5, "regular")]
        return variants

    def _normalize_variants(self, variants: list[dict[str, object]]) -> list[tuple[str, int, str]]:
        normalized: list[tuple[str, int, str]] = []
        seen: set[tuple[str, int, str]] = set()

        for raw in variants:
            if not isinstance(raw, dict):
                continue
            family = raw.get("family")
            size = raw.get("size")
            style = raw.get("style")
            if not isinstance(family, str) or not isinstance(style, str):
                continue
            try:
                size_int = int(size)
            except (TypeError, ValueError):
                continue

            if family not in available_families():
                continue
            if size_int not in available_sizes(family):
                continue
            if style not in available_styles(family, size_int):
                continue

            variant = (family, size_int, style)
            if variant in seen:
                continue
            seen.add(variant)
            normalized.append(variant)
            if len(normalized) >= self.MAX_COMPARE_VARIANTS:
                break

        if normalized:
            return normalized
        return list(self._variants[: self.MAX_COMPARE_VARIANTS])

    def get_settings(self) -> dict[str, object]:
        return {
            "phrase": self._phrase,
            "spacing": self._spacing_px,
            "variants": [
                {"family": family, "size": size, "style": style}
                for family, size, style in self._configured_variants
            ],
        }

    def update_settings(
        self,
        *,
        phrase: str,
        variants: list[dict[str, object]] | None = None,
        spacing: int | None = None,
    ) -> dict[str, object]:
        cleaned = " ".join(str(phrase).split())
        if not cleaned:
            cleaned = "FLIPDOT"
        self._phrase = cleaned[:32]
        if spacing is not None:
            self._spacing_px = max(0, min(self.MAX_SPACING_PX, int(spacing)))
        if variants is not None:
            self._configured_variants = self._normalize_variants(variants)
            self._window_start = 0
        return self.get_settings()

    def get_variant_catalog(self) -> dict[str, dict[str, list[str]]]:
        catalog: dict[str, dict[str, list[str]]] = {}
        for family in available_families():
            catalog[family] = {}
            for size in available_sizes(family):
                catalog[family][str(size)] = list(available_styles(family, size))
        return catalog

    def get_glyph_grid(self) -> dict[str, object]:
        variants_payload: list[dict[str, object]] = []
        all_chars: set[str] = set()

        for family in available_families():
            for size in available_sizes(family):
                for style in available_styles(family, size):
                    definition = get_font_definition(family, size, style)
                    glyph_payload: dict[str, list[list[int]]] = {}
                    for char, glyph in definition.glyphs.items():
                        serialized = glyph.astype(np.uint8).tolist()
                        glyph_payload[char] = [[int(value) for value in row] for row in serialized]
                        all_chars.add(char)

                    variants_payload.append(
                        {
                            "family": family,
                            "size": size,
                            "style": style,
                            "cell_width": definition.cell_width,
                            "glyphs": glyph_payload,
                        }
                    )

        variants_payload.sort(key=lambda item: (str(item["family"]), int(item["size"]), str(item["style"])))
        characters = sorted(all_chars, key=lambda value: (ord(value), value))
        return {
            "variants": variants_payload,
            "characters": characters,
        }

    def _visible_variants(self) -> list[tuple[str, int, str]]:
        variants = self._configured_variants
        if len(variants) <= self.MAX_COMPARE_VARIANTS:
            return list(variants)

        out: list[tuple[str, int, str]] = []
        total = len(variants)
        for offset in range(self.MAX_COMPARE_VARIANTS):
            out.append(variants[(self._window_start + offset) % total])
        return out

    def previous_variant(self) -> None:
        total = len(self._configured_variants)
        if total <= self.MAX_COMPARE_VARIANTS:
            return
        self._window_start = (self._window_start - 1) % total

    def next_variant(self) -> None:
        total = len(self._configured_variants)
        if total <= self.MAX_COMPARE_VARIANTS:
            return
        self._window_start = (self._window_start + 1) % total

    def adjust_spacing(self, delta: int) -> None:
        self._spacing_px = max(0, min(self.MAX_SPACING_PX, self._spacing_px + int(delta)))

    def _pointer_to_panel(self, source: str | None, x: float, y: float) -> tuple[int, int]:
        if source == "pose":
            panel_x = int(self.width - (x * self.width))
        else:
            panel_x = int(x * self.width)
        panel_y = int(y * self.height)
        panel_x = max(0, min(self.width - 1, panel_x))
        panel_y = max(0, min(self.height - 1, panel_y))
        return panel_x, panel_y

    def _safe_phrase_for_variant(self, phrase: str, family: str, size: int, style: str) -> str:
        glyphs = get_font_definition(family, size, style).glyphs
        fallback = "A" if "A" in glyphs else next(iter(glyphs.keys()))
        out = []
        for ch in phrase:
            out.append(ch if ch in glyphs else " ")
        safe = "".join(out).strip()
        return safe if safe else fallback

    def _detect_hover_zone(self, panel_x: int | None, panel_y: int | None) -> str | None:
        if panel_x is None or panel_y is None:
            return None
        if panel_y < 2:
            return None
        third = max(1, self.width // 3)
        if panel_x < third:
            return "prev"
        if panel_x >= (self.width - third):
            return "next"
        return None

    def _update_zone_interaction(self, zone: str | None) -> float:
        now = time.time()
        if zone is None:
            self._hover_zone = None
            self._hover_started_at = None
            return 0.0

        if zone != self._hover_zone:
            self._hover_zone = zone
            self._hover_started_at = now
            return 0.0

        if self._hover_started_at is None:
            self._hover_started_at = now
            return 0.0

        elapsed = now - self._hover_started_at
        progress = min(1.0, elapsed / self.CLICK_TIME_SEC)
        if elapsed >= self.CLICK_TIME_SEC:
            if zone == "prev":
                self.previous_variant()
            elif zone == "next":
                self.next_variant()
            self._hover_started_at = now
            progress = 0.0

        return progress

    def _draw_progress(self, frame: np.ndarray, zone: str | None, progress: float) -> None:
        frame[self.height - 1, :] = 0
        if zone is None:
            return
        bar = min(self.width, max(0, int(self.width * progress)))
        if zone == "prev":
            frame[self.height - 1, 0:bar] = 1
        elif zone == "next":
            frame[self.height - 1, self.width - bar : self.width] = 1

    def get_frame(self, pose_results=None, input_hub=None) -> np.ndarray:
        del pose_results
        frame = np.zeros((self.height, self.width), dtype=np.uint8)

        if input_hub is not None:
            pointer = input_hub.get_active_pointer(max_age_sec=0.8)
        else:
            pointer = None

        panel_x = None
        panel_y = None
        if pointer is not None:
            panel_x, panel_y = self._pointer_to_panel(pointer.source, pointer.x, pointer.y)

        zone = self._detect_hover_zone(panel_x, panel_y)
        progress = self._update_zone_interaction(zone)

        visible_variants = self._visible_variants()
        if not visible_variants:
            visible_variants = [("classic", 5, "regular")]
        band_count = len(visible_variants)

        for index, (family, size, style) in enumerate(visible_variants):
            safe_phrase = self._safe_phrase_for_variant(self._phrase, family, size, style)
            band_top = (index * self.height) // band_count
            band_bottom = ((index + 1) * self.height) // band_count
            band_height = max(1, band_bottom - band_top)

            phrase_width = width(
                safe_phrase,
                font=family,
                size=size,
                style=style,
                spacing=self._spacing_px,
            )
            if phrase_width <= self.width:
                phrase_x = max(0, (self.width - phrase_width) // 2)
            else:
                span = phrase_width + self.width
                t = (time.time() + (index * 0.2)) * self.SCROLL_SPEED_PX_PER_SEC
                phrase_x = self.width - int(t % span)

            phrase_y = band_top + max(0, (band_height - size) // 2)
            write(
                frame,
                safe_phrase,
                x=phrase_x,
                y=phrase_y,
                font=family,
                size=size,
                style=style,
                spacing=self._spacing_px,
            )

        self._draw_progress(frame, zone, progress)
        return frame
