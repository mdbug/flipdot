import numpy as np

from app.services.fonts import available_sizes, available_styles, get_font_definition

DEFAULT_FONT_FAMILY = "classic"
DEFAULT_FONT_SIZE = 6
DEFAULT_FONT_STYLE = "regular"


def _get_font(font: str, size: int, style: str):
    return get_font_definition(font, size, style)


def _get_glyph(font: str, size: int, style: str, char: str) -> np.ndarray:
    glyphs = _get_font(font, size, style).glyphs
    if char not in glyphs:
        raise KeyError(
            f"Unsupported character '{char}' for font '{font}' size {size} style '{style}'"
        )
    return glyphs[char]


def write(
    frame: np.ndarray,
    text: str,
    x=0,
    y=0,
    *,
    font=DEFAULT_FONT_FAMILY,
    size=DEFAULT_FONT_SIZE,
    style=DEFAULT_FONT_STYLE,
    spacing=1,
    color=1,
):
    font_def = _get_font(font, size, style)
    cell_w = font_def.cell_width
    cursor_x = int(x)
    cursor_y = int(y)
    for char in text:
        glyph = _get_glyph(font, size, style, char)
        if cell_w is not None:
            glyph_w = glyph.shape[1]
            if glyph_w > cell_w:
                raise ValueError(
                    f"Glyph '{char}' width {glyph_w} exceeds monospace cell width {cell_w}"
                )
            pad_left = (cell_w - glyph_w) // 2
            pad_right = cell_w - glyph_w - pad_left
            glyph = np.pad(glyph, ((0, 0), (pad_left, pad_right)))

        glyph_h = glyph.shape[0]
        glyph_w = glyph.shape[1]

        dst_y0 = max(cursor_y, 0)
        dst_x0 = max(cursor_x, 0)
        dst_y1 = min(cursor_y + glyph_h, frame.shape[0])
        dst_x1 = min(cursor_x + glyph_w, frame.shape[1])

        if dst_y1 > dst_y0 and dst_x1 > dst_x0:
            src_y0 = dst_y0 - cursor_y
            src_x0 = dst_x0 - cursor_x
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            src_x1 = src_x0 + (dst_x1 - dst_x0)
            src = glyph[src_y0:src_y1, src_x0:src_x1]

            if color == 1:
                frame[dst_y0:dst_y1, dst_x0:dst_x1] = src
            else:
                frame[dst_y0:dst_y1, dst_x0:dst_x1] = np.logical_not(src)

        cursor_x += glyph_w + spacing
        if cursor_x >= frame.shape[1]:
            break


def width(
    value: str,
    *,
    font=DEFAULT_FONT_FAMILY,
    size=DEFAULT_FONT_SIZE,
    style=DEFAULT_FONT_STYLE,
    spacing=1,
):
    if not value:
        return 0

    font_def = _get_font(font, size, style)
    if font_def.cell_width is not None:
        return len(value) * font_def.cell_width + spacing * (len(value) - 1)

    return (
        sum(_get_glyph(font, size, style, ch).shape[1] for ch in value)
        + spacing * (len(value) - 1)
    )


def center_x(
    frame_width: int,
    value: str,
    *,
    font=DEFAULT_FONT_FAMILY,
    size=DEFAULT_FONT_SIZE,
    style=DEFAULT_FONT_STYLE,
    spacing=1,
):
    return max(
        0,
        (
            frame_width
            - width(value, font=font, size=size, style=style, spacing=spacing)
        )
        // 2,
    )


def write_centered(
    frame: np.ndarray,
    value: str,
    *,
    y=0,
    font=DEFAULT_FONT_FAMILY,
    size=DEFAULT_FONT_SIZE,
    style=DEFAULT_FONT_STYLE,
    spacing=1,
    color=1,
):
    x = center_x(
        frame.shape[1],
        value,
        font=font,
        size=size,
        style=style,
        spacing=spacing,
    )
    write(
        frame,
        value,
        x=x,
        y=y,
        font=font,
        size=size,
        style=style,
        spacing=spacing,
        color=color,
    )


def supported_characters(
    *,
    font=DEFAULT_FONT_FAMILY,
    sizes=None,
    styles=None,
):
    if sizes is None:
        sizes = available_sizes(font)

    chars = set()
    for size in sizes:
        requested_styles = styles if styles is not None else available_styles(font, size)
        for style in requested_styles:
            chars.update(_get_font(font, size, style).glyphs.keys())

    return frozenset(chars)
