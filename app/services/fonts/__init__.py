from app.services.fonts.classic import CLASSIC_FONT, FontDefinition
from app.services.fonts.scoreline import SCORELINE_FONT


FONT_FAMILIES = {
    "classic": CLASSIC_FONT,
    "scoreline": SCORELINE_FONT,
}


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
