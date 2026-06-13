from app.services.fonts.classic import FontDefinition
from app.services.fonts.classic_size6_monospace import GLYPHS as MONOSPACE_SIZE6_GLYPHS
from app.services.fonts.classic_size6_regular import GLYPHS as REGULAR_SIZE6_GLYPHS


_DIGITS = set("0123456789")


# Scoreline keeps the general size-6 regular look, but uses monospace-derived
# digit bitmaps for score/time readability.
SCORELINE_SIZE6_REGULAR_GLYPHS = {
    ch: (MONOSPACE_SIZE6_GLYPHS[ch] if ch in _DIGITS else glyph)
    for ch, glyph in REGULAR_SIZE6_GLYPHS.items()
}


SCORELINE_FONT = {
    6: {
        "regular": FontDefinition(glyphs=SCORELINE_SIZE6_REGULAR_GLYPHS),
    },
}
