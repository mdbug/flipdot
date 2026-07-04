"""Falling-sand toy where the viewer's silhouette is a collision obstacle."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

import app.services.silhouette as silhouette
from app.modes.contracts import Frame, RenderContext

# Draws face features onto a frame: (frame, face_mesh_results, width, height).
FaceRenderer = Callable[[Frame, Any, int, int], Frame]


class Sandfall:
    """Sand rains from the top edge and piles up on the viewer's silhouette.

    A grain falls straight down when the cell below is free, otherwise
    slides diagonally down-left/down-right, otherwise rests. A grain that
    lands hard after a few cells of free fall can bounce: an arc one to
    three cells high with a random sideways kick, climbing one cell per
    tick and hanging at the apex, decaying over successive hops until it
    settles.
    The viewer's silhouette (see :mod:`app.services.silhouette`) is a
    solid obstacle rendered as just its outline, so sand accumulates on
    outstretched arms and heads and cascades off when the person moves;
    grains a moving limb sweeps into are pushed up above it rather than
    deleted. When the viewer is close enough for face-mesh landmarks, eyes
    and a mouth are lit inside the silhouette's head via ``face_renderer``.
    When the panel fills up, the floor opens for a few seconds and
    everything drains out with a full-panel crackle. With nobody in frame
    the mode idles like an hourglass — unless it was auto-entered from the
    clock, in which case the ``TransitionPolicy`` gesture chain returns to
    the clock shortly after the person leaves (and hands off to caricature
    when they come very close).

    There are no gesture controls; arms-crossed exit is handled by the
    main loop as usual.
    """

    TICK_INTERVAL = 1.0 / 30.0  # seconds per physics tick
    MAX_CATCHUP_TICKS = 5  # cap physics catch-up after a slow frame
    SPAWN_RATE = 8.0  # average grains spawned per second
    MAX_FILL_FRACTION = 0.45  # of panel area; beyond this the floor opens
    DRAIN_SECONDS = 3.0  # how long the floor stays open
    BOUNCE_MIN_FALL = 3  # cells of uninterrupted free fall before a landing can bounce
    BOUNCE_HIGH_FALL = 6  # free fall this long earns the taller bounce arc
    BOUNCE_HEIGHT = 2  # cells a normal bounce climbs before falling back
    BOUNCE_HEIGHT_HIGH = 3  # cells climbed after a BOUNCE_HIGH_FALL-length drop
    BOUNCE_CHANCE = 0.6  # probability a hard-landing grain actually bounces

    def __init__(self, width: int, height: int, face_renderer: FaceRenderer | None = None) -> None:
        self.width = width
        self.height = height
        self.face_renderer = face_renderer
        self.rng = np.random.default_rng()
        self.sand = np.zeros((height, width), dtype=bool)
        self.mask = np.zeros((height, width), dtype=bool)
        # Per-grain count of consecutive cells fallen straight down; fuels bounces.
        self._fall_dist = np.zeros((height, width), dtype=np.uint8)
        # Per-grain cells left to climb on a bounce arc (nonzero only under sand).
        self._rise = np.zeros((height, width), dtype=np.uint8)
        self._last_tick_time = time.time()
        self._draining_until = 0.0
        self._tick_parity = False

    # ------------------------------------------------------------------
    # Obstacle handling
    # ------------------------------------------------------------------

    def _set_mask(self, mask: np.ndarray | None) -> None:
        """Install the new obstacle mask, pushing swept-up grains above it."""
        self.mask = mask if mask is not None else np.zeros((self.height, self.width), dtype=bool)
        overlap = self.sand & self.mask
        if not overlap.any():
            return
        occupied = self.sand | self.mask
        for row, col in np.argwhere(overlap):
            self.sand[row, col] = False
            self._fall_dist[row, col] = 0
            self._rise[row, col] = 0
            for free_row in range(row - 1, -1, -1):
                if not occupied[free_row, col]:
                    self.sand[free_row, col] = True
                    self._fall_dist[free_row, col] = 0
                    occupied[free_row, col] = True
                    break
            # No free cell above: the grain is squeezed out of existence.

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _physics_tick(self, draining: bool = False) -> None:
        """Advance every grain one step: rise/bounce, else fall, else slide, else rest."""
        if draining:
            self.sand[-1, :] = False
            self._fall_dist[-1, :] = 0
        occupied = self.sand | self.mask
        airborne = self._rise_pass(occupied)
        airborne |= self._bounce_pass(occupied)
        for row in range(self.height - 2, -1, -1):
            movers = self.sand[row] & ~self.mask[row] & ~airborne[row]
            fall = movers & ~occupied[row + 1]
            self.sand[row] &= ~fall
            self.sand[row + 1] |= fall
            self._fall_dist[row + 1][fall] = self._fall_dist[row][fall] + 1
            self._fall_dist[row][fall] = 0
            occupied[row] = self.sand[row] | self.mask[row]
            occupied[row + 1] |= fall
            # Blocked grains try the diagonals; the preferred side
            # alternates each tick so piles don't drift one way. Grains
            # still carrying free-fall momentum (blocked mid-tick by a
            # grain arriving under them) hold in place instead, so next
            # tick's bounce pass can splash them.
            first, second = (-1, 1) if self._tick_parity else (1, -1)
            slidable = movers & ~fall & (self._fall_dist[row] < self.BOUNCE_MIN_FALL)
            blocked_cols = np.flatnonzero(slidable)
            for col in blocked_cols if first < 0 else blocked_cols[::-1]:
                for dc in (first, second):
                    new_col = col + dc
                    if 0 <= new_col < self.width and not occupied[row + 1, new_col]:
                        self.sand[row, col] = False
                        self.sand[row + 1, new_col] = True
                        # Rolling down a slope dissipates the free-fall streak.
                        self._fall_dist[row + 1, new_col] = 0
                        self._fall_dist[row, col] = 0
                        occupied[row, col] = self.mask[row, col]
                        occupied[row + 1, new_col] = True
                        break
        self._tick_parity = not self._tick_parity

    def _rise_pass(self, occupied: np.ndarray) -> np.ndarray:
        """Carry still-climbing bounced grains up one cell; return the airborne cells.

        A grain whose climb just ended is still returned as airborne, so
        it hangs at the apex for one tick before falling back — the arc
        stays visible even when render frames skip physics ticks. Climbs
        cancel against ceilings. Updates ``occupied`` in place.
        """
        airborne = np.zeros_like(self.sand)
        self._rise[0, :] = 0  # nowhere to climb from the top row
        for row in range(1, self.height):
            for col in np.flatnonzero(self.sand[row] & (self._rise[row] > 0)):
                rise = int(self._rise[row, col])
                self._rise[row, col] = 0
                if occupied[row - 1, col]:
                    continue  # ceiling: the climb ends early
                self.sand[row, col] = False
                self.sand[row - 1, col] = True
                self._rise[row - 1, col] = rise - 1
                self._fall_dist[row - 1, col] = 0
                occupied[row, col] = self.mask[row, col]
                occupied[row - 1, col] = True
                airborne[row - 1, col] = True
        return airborne

    def _bounce_pass(self, occupied: np.ndarray) -> np.ndarray:
        """Kick grains that just landed hard back up; return the cells that hopped.

        A grain lands hard when it comes to rest after at least
        ``BOUNCE_MIN_FALL`` cells of uninterrupted free fall. It gets one
        chance to bounce: an immediate 1-cell hop with a random sideways
        kick, continuing upward on later ticks via ``_rise`` to a peak of
        ``BOUNCE_HEIGHT`` (or ``BOUNCE_HEIGHT_HIGH`` after a long drop).
        Hopped cells are exempt from movement for the rest of the tick so
        the hop survives into the rendered frame. Updates ``occupied`` in
        place.
        """
        bounced = np.zeros_like(self.sand)
        supported = np.empty_like(self.sand)
        supported[-1] = True
        supported[:-1] = occupied[1:]
        landed = self.sand & ~self.mask & supported & (self._fall_dist >= self.BOUNCE_MIN_FALL)
        for row, col in np.argwhere(landed):
            fall_dist = int(self._fall_dist[row, col])
            self._fall_dist[row, col] = 0  # at most one bounce per landing
            if row == 0 or self.rng.random() >= self.BOUNCE_CHANCE:
                continue
            high = fall_dist >= self.BOUNCE_HIGH_FALL
            height = self.BOUNCE_HEIGHT_HIGH if high else self.BOUNCE_HEIGHT
            for dc in self.rng.permutation(3) - 1:
                new_col = col + dc
                if 0 <= new_col < self.width and not occupied[row - 1, new_col]:
                    self.sand[row, col] = False
                    self.sand[row - 1, new_col] = True
                    self._rise[row - 1, new_col] = height - 1
                    bounced[row - 1, new_col] = True
                    occupied[row, col] = False
                    occupied[row - 1, new_col] = True
                    break
        return bounced

    def _maybe_spawn(self) -> None:
        """Sometimes drop a new grain at a random free top-row column."""
        if self.rng.random() >= self.SPAWN_RATE * self.TICK_INTERVAL:
            return
        free_cols = np.flatnonzero(~(self.sand[0] | self.mask[0]))
        if free_cols.size:
            col = self.rng.choice(free_cols)
            self.sand[0, col] = True
            self._fall_dist[0, col] = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def get_frame(self, context: RenderContext) -> Frame:
        """Refresh the obstacle, run pending physics ticks, and render."""
        now = time.time()
        self._set_mask(silhouette.pose_to_mask(context.pose_results, self.width, self.height))

        ticks = int((now - self._last_tick_time) / self.TICK_INTERVAL)
        if ticks > self.MAX_CATCHUP_TICKS:
            ticks = self.MAX_CATCHUP_TICKS
            self._last_tick_time = now
        else:
            self._last_tick_time += ticks * self.TICK_INTERVAL
        for _ in range(ticks):
            draining = now < self._draining_until
            self._physics_tick(draining=draining)
            if not draining:
                self._maybe_spawn()

        fill_fraction = self.sand.sum() / (self.width * self.height)
        if now >= self._draining_until and fill_fraction > self.MAX_FILL_FRACTION:
            self._draining_until = now + self.DRAIN_SECONDS

        frame = (self.sand | silhouette.mask_outline(self.mask)).astype(np.uint8)
        if self.face_renderer is not None:
            frame = self.face_renderer(frame, context.face_mesh_results, self.width, self.height)
        return frame
