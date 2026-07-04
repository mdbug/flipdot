"""Conway's Game of Life continuously seeded by the viewer's silhouette."""

import time
from typing import Any

import numpy as np

import app.services.silhouette as silhouette
from app.modes.contracts import Frame


class LifeMirror:
    """Game of Life mirror: the viewer's silhouette is stamped in as live cells.

    The world evolves at a fixed generation cadence on a torus (gliders
    that leave one edge re-enter on the opposite one). Every few seconds
    the current silhouette is OR-ed into the grid; the solid body blob's
    interior dies of overpopulation on the next generation, leaving a
    lively boundary froth. Stand still and you crystallize into gliders
    and oscillators; walk away and your last shape evolves into nothing.

    When the world goes empty or settles into a short-period cycle for a
    while, a sparse random soup is reseeded so the panel never stays dark.

    There are no gesture controls; arms-crossed exit is handled by the
    main loop as usual.
    """

    GENERATION_INTERVAL = 0.15  # seconds between generations (~7 per second)
    SEED_INTERVAL = 2.0  # seconds between silhouette stamps
    STAGNATION_TIMEOUT = 6.0  # seconds of empty/cycling world before reseeding
    HISTORY_LENGTH = 4  # recent generations compared to detect short cycles
    SOUP_DENSITY = 0.12  # live-cell fraction of a random soup reseed

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.rng = np.random.default_rng()
        self.grid = np.zeros((height, width), dtype=bool)
        self._history: list[bytes] = []
        self._stagnant_since: float | None = None
        self._last_generation_time = 0.0
        self._last_seed_time = 0.0
        self._reseed_soup()

    def _step_generation(self) -> None:
        """Advance one B3/S23 generation on a toroidal grid."""
        g = self.grid
        neighbors = sum(
            np.roll(np.roll(g, dr, axis=0), dc, axis=1)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )
        self.grid = (neighbors == 3) | (g & (neighbors == 2))

    def _is_stagnant(self) -> bool:
        """Return True when the world is empty or repeating with a short period."""
        key = self.grid.tobytes()
        stagnant = not self.grid.any() or key in self._history
        self._history.append(key)
        if len(self._history) > self.HISTORY_LENGTH:
            self._history.pop(0)
        return stagnant

    def _reseed_soup(self) -> None:
        """Replace the world with a sparse random soup."""
        self.grid = self.rng.random((self.height, self.width)) < self.SOUP_DENSITY
        self._history.clear()
        self._stagnant_since = None

    def _seed_silhouette(self, pose_results: Any) -> None:
        """Stamp the current silhouette (if any) into the world as live cells."""
        mask = silhouette.pose_to_mask(pose_results, self.width, self.height)
        if mask is not None:
            self.grid |= mask

    def get_frame(self, pose_results: Any) -> Frame:
        """Evolve the world (seeding it from the viewer) and render it."""
        now = time.time()
        if now - self._last_seed_time >= self.SEED_INTERVAL:
            self._seed_silhouette(pose_results)
            self._last_seed_time = now
        if now - self._last_generation_time >= self.GENERATION_INTERVAL:
            self._step_generation()
            self._last_generation_time = now
            if self._is_stagnant():
                if self._stagnant_since is None:
                    self._stagnant_since = now
                elif now - self._stagnant_since >= self.STAGNATION_TIMEOUT:
                    self._reseed_soup()
            else:
                self._stagnant_since = None
        return self.grid.astype(np.uint8)
