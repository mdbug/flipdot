import numpy as np

from app.modes.contracts import Frame


def disolve(dots: Frame, alpha: float) -> Frame:
    """Clear a random ``alpha`` fraction of lit pixels in place (1-bit dissolve)."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")

    mask = np.random.rand(*dots.shape) < alpha
    dots[mask] = 0
    return dots


def resolve(dots: Frame, alpha: float) -> Frame:
    """Inverse of :func:`disolve`: reveal pixels as ``alpha`` rises from 0 to 1."""
    return disolve(dots, 1 - alpha)


def blend(dots1: Frame, dots2: Frame, alpha: float) -> Frame:
    """Randomly mix two frames, taking each pixel from ``dots2`` with probability ``alpha``."""
    result = np.zeros_like(dots1)

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("Alpha must be between 0.0 and 1.0")

    mask = np.random.rand(*dots1.shape) < alpha
    result[mask] = dots2[mask]
    result[~mask] = dots1[~mask]

    return result
