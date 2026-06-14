from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FontDefinition:
    glyphs: dict[str, np.ndarray]
    cell_width: int | None = None


def _font_data_dir() -> Path:
    return Path(__file__).with_name("data")


def _load_font_families() -> dict[str, dict[int, dict[str, FontDefinition]]]:
    families: dict[str, dict[int, dict[str, FontDefinition]]] = {}
    for path in sorted(_font_data_dir().glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata = payload.get("metadata", {})
        family = metadata.get("family")
        size = metadata.get("size")
        style = metadata.get("style")
        cell_width = metadata.get("cell_width")

        if not isinstance(family, str) or not family:
            raise ValueError(f"Invalid or missing 'family' metadata in {path.name}")
        if not isinstance(size, int):
            raise ValueError(f"Invalid or missing 'size' metadata in {path.name}")
        if not isinstance(style, str) or not style:
            raise ValueError(f"Invalid or missing 'style' metadata in {path.name}")
        if cell_width is not None and not isinstance(cell_width, int):
            raise ValueError(f"Invalid 'cell_width' metadata in {path.name}")

        raw_glyphs = payload.get("glyphs")
        if not isinstance(raw_glyphs, dict) or not raw_glyphs:
            raise ValueError(f"Invalid or empty 'glyphs' in {path.name}")

        glyphs = {
            char: np.array(rows, dtype=np.uint8)
            for char, rows in raw_glyphs.items()
        }

        size_fonts = families.setdefault(family, {}).setdefault(size, {})
        if style in size_fonts:
            raise ValueError(
                f"Duplicate font definition for family='{family}' size={size} style='{style}'"
            )
        size_fonts[style] = FontDefinition(glyphs=glyphs, cell_width=cell_width)

    return families


FONT_FAMILIES = _load_font_families()


def available_families() -> tuple[str, ...]:
    return tuple(sorted(FONT_FAMILIES.keys()))


def available_sizes(family: str) -> tuple[int, ...]:
    return tuple(sorted(FONT_FAMILIES[family].keys()))


def available_styles(family: str, size: int) -> tuple[str, ...]:
    return tuple(sorted(FONT_FAMILIES[family][size].keys()))


def get_font_definition(family: str, size: int, style: str) -> FontDefinition:
    if family not in FONT_FAMILIES:
        raise KeyError(f"Unknown font family '{family}'. Available: {available_families()}")

    family_fonts = FONT_FAMILIES[family]
    if size not in family_fonts:
        raise KeyError(f"Unknown font size {size} for family '{family}'. Available: {tuple(sorted(family_fonts.keys()))}")

    size_fonts = family_fonts[size]
    if style not in size_fonts:
        raise KeyError(f"Unknown font style '{style}' for family '{family}' size {size}. Available: {tuple(sorted(size_fonts.keys()))}")

    return size_fonts[style]
