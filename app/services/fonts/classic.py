from dataclasses import dataclass
from typing import Dict

from app.services.fonts.classic_size5_monospace import CELL_WIDTH as SIZE5_MONO_CELL_WIDTH
from app.services.fonts.classic_size5_monospace import GLYPHS as SIZE5_MONO_GLYPHS
from app.services.fonts.classic_size5_regular import GLYPHS as SIZE5_REGULAR_GLYPHS
from app.services.fonts.classic_size6_monospace import CELL_WIDTH as SIZE6_MONO_CELL_WIDTH
from app.services.fonts.classic_size6_monospace import GLYPHS as SIZE6_MONO_GLYPHS
from app.services.fonts.classic_size6_regular import GLYPHS as SIZE6_REGULAR_GLYPHS


@dataclass(frozen=True)
class FontDefinition:
    glyphs: Dict[str, object]
    cell_width: int | None = None


CLASSIC_FONT = {
    5: {
        "regular": FontDefinition(glyphs=SIZE5_REGULAR_GLYPHS),
        "monospace": FontDefinition(
            glyphs=SIZE5_MONO_GLYPHS,
            cell_width=SIZE5_MONO_CELL_WIDTH,
        ),
    },
    6: {
        "regular": FontDefinition(glyphs=SIZE6_REGULAR_GLYPHS),
        "monospace": FontDefinition(
            glyphs=SIZE6_MONO_GLYPHS,
            cell_width=SIZE6_MONO_CELL_WIDTH,
        ),
    },
}
