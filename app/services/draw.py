"""Generic 1-bit drawing primitives that operate on a passed numpy frame.

Mirrors the "operate on a frame" style of :mod:`app.services.text`. These are
deliberately framework-free helpers (line, circle, point) so any mode can draw
geometry onto its own ``(height, width)`` uint8 frame.
"""

from __future__ import annotations

import math

import numpy as np

Point = tuple[float, float]


def draw_point(frame: np.ndarray, x: float, y: float, *, color: int = 1) -> None:
    """Set the single pixel at ``(x, y)`` if it lies within ``frame``."""
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= yi < frame.shape[0] and 0 <= xi < frame.shape[1]:
        frame[yi, xi] = 0 if int(color) == 0 else 1


def draw_line(frame: np.ndarray, p0: Point, p1: Point, *, color: int = 1) -> None:
    """Draw a 1px line between ``p0`` and ``p1``, clipping at the frame edges.

    Standard integer (8-connected) Bresenham: a single pixel per major-axis
    step, taking diagonal steps where needed. No per-pixel rounding or temporary
    allocations.
    """
    x0 = int(round(p0[0]))
    y0 = int(round(p0[1]))
    x1 = int(round(p1[0]))
    y1 = int(round(p1[1]))
    height, width = frame.shape
    value = 0 if int(color) == 0 else 1

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 <= x1 else -1
    sy = 1 if y0 <= y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < width and 0 <= y < height:
            frame[y, x] = value
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def draw_circle(frame: np.ndarray, center: Point, radius: float, *, color: int = 1) -> None:
    """Draw a 1px circle outline of ``radius`` around ``center`` (sampled every 2°)."""
    cx, cy = center
    radius = max(1.0, float(radius))
    for angle in range(0, 360, 2):
        rad = math.radians(angle)
        draw_point(frame, cx + radius * math.cos(rad), cy + radius * math.sin(rad), color=color)


def fill_circle(frame: np.ndarray, center: Point, radius: float, *, color: int = 1) -> None:
    """Fill a solid disc of ``radius`` around ``center``.

    Uses a squared-distance mask, so the result is exactly symmetric about
    ``center`` (e.g. a half-integer center yields a disc with clean mirror
    symmetry across both axes).
    """
    cx, cy = center
    height, width = frame.shape
    yy, xx = np.ogrid[0:height, 0:width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius) ** 2
    frame[mask] = 0 if int(color) == 0 else 1


def thick_line(
    frame: np.ndarray, p0: Point, p1: Point, *, width: float = 1.0, color: int = 1
) -> None:
    """Draw a gap-free line from ``p0`` to ``p1`` with the given pixel ``width``.

    Fills the oriented rectangle of the segment: a pixel is lit when its
    perpendicular distance to the centerline is within ``width / 2`` *and* its
    projection along the segment falls between the endpoints. This is gap-free at
    any angle and has flat (square) caps — unlike offsetting parallel Bresenham
    passes, which interleave into a checkerboard on diagonals.
    """
    if width <= 1.0:
        draw_line(frame, p0, p1, color=color)
        return

    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0.0:
        draw_line(frame, p0, p1, color=color)
        return

    value = 0 if int(color) == 0 else 1
    radius = width / 2.0
    height, width_px = frame.shape

    # Unit vectors along the segment and perpendicular to it.
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length
    px, py = -uy, ux

    # Bounding box of the segment, expanded by the half-width.
    xlo = max(int(math.floor(min(x0, x1) - radius)), 0)
    xhi = min(int(math.ceil(max(x0, x1) + radius)), width_px - 1)
    ylo = max(int(math.floor(min(y0, y1) - radius)), 0)
    yhi = min(int(math.ceil(max(y0, y1) + radius)), height - 1)
    if xlo > xhi or ylo > yhi:
        return

    ys = np.arange(ylo, yhi + 1).reshape(-1, 1)
    xs = np.arange(xlo, xhi + 1).reshape(1, -1)
    dx = xs - x0
    dy = ys - y0
    along = dx * ux + dy * uy
    perp = dx * px + dy * py
    eps = 1e-9
    mask = (np.abs(perp) <= radius) & (along >= -eps) & (along <= length + eps)
    frame[ylo : yhi + 1, xlo : xhi + 1][mask] = value
