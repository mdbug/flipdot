import math
import time
from collections import deque
from typing import Any

import numpy as np

import app.services.text as text
from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame

# Sixteen discrete headings; unit vectors precomputed (x = col, y = row, y down).
HEADINGS = 16
_HEADING_VECS = [
    (math.cos(2 * math.pi * h / HEADINGS), math.sin(2 * math.pi * h / HEADINGS))
    for h in range(HEADINGS)
]

# The eight move directions (cardinals + diagonals) map to a subset of the
# headings above.  Keys are unit (dx, dy) vectors with y pointing down.
_MOVE_HEADINGS = {
    (1, 0): 0,  # right
    (1, 1): 2,  # down-right
    (0, 1): 4,  # down
    (-1, 1): 6,  # down-left
    (-1, 0): 8,  # left
    (-1, -1): 10,  # up-left
    (0, -1): 12,  # up
    (1, -1): 14,  # up-right
}


class Tank:
    """Two-tank combat in the style of Atari *Combat*.

    Each tank moves and aims in one of eight directions with the D-pad (hold
    two arrows for a 45-degree diagonal) and fires shells with Button A.
    Shells ricochet off the arena walls and border twice before expiring, so
    bank shots score -- and shells do not discriminate: a wild ricochet can
    come back around and take out its own shooter (gifting the foe the point).

    Controls
    --------
    * Primary controller  → right tank (side ``1``)
    * Secondary controller → left tank (side ``0``)
    * D-pad (incl. two arrows at once) → drive and aim in that direction
    * Button A             → fire (and restart once a match is won)

    A held D-pad direction snaps the tank's heading to that direction and
    drives it forward; the gun always points the way the tank last moved.
    Tanks are solid: neither can drive through the other, so they never overlap.

    When a side provides no input for ``AI_TAKEOVER_DELAY`` seconds, an AI takes
    it over so the mode is playable solo and works as an attract loop.  Each
    frame the AI scores nine candidate moves (eight directions plus hold) on
    shell danger (its own ricochets included), clear-firing-lane seeking in
    all eight directions, alignment and standoff comfort, and fires only down
    lanes a real shell would clear -- including deliberate bank shots off the
    walls, validated with the real ricochet physics so it never banks a shell
    into itself.  Reaction delays, the odd ignored threat and random shot
    holds keep it deliberately beatable; a stalemate roam and occasional
    flanking repositioning keep it lively and AI-vs-AI attract mode alive.

    Tanks are told apart on the monochrome panel by shape: the left tank is a
    solid 3x3 block, the right tank a hollow 3x3 ring.

    Impacts are dressed up with small effects, both designed so they cannot be
    mistaken for a shell (a lone travelling pixel): a ricocheting shell flashes
    a stationary plus-shaped spark at the bounce point, and a tank that slams
    into a wall or the other tank at speed blinks a dent *off* the surface --
    grinding slowly against a wall stays quiet.
    """

    AI_TAKEOVER_DELAY = 15.0
    WIN_SCORE = 5
    MAX_DT = 0.05  # clamp dt to avoid tunneling fast shells

    TANK_HALF = 1  # tank is (2*TANK_HALF+1) px square -> 3x3
    # A 3x3 tank needs this much clearance to drive between two wall pieces;
    # narrower slots look like passages but block tanks (and stray shots).
    MIN_GAP = 2 * TANK_HALF + 1
    THRUST_ACCEL = 28.0  # px/s^2
    FRICTION = 22.0  # px/s^2 deceleration when coasting
    MAX_SPEED = 12.0  # px/s

    AI_STANDOFF = 12.0  # px base distance to the foe; jittered per spawn
    AI_ALIGN_TOL = 1  # px slack for counting as sitting on a firing line
    AI_HORIZON = 0.9  # s of shell projection feeding the danger term
    AI_LOOKAHEAD = 2.5  # px along a candidate move at which it is scored
    AI_REACT_MIN = 0.12  # s reaction delay floor (sharp AI) -- fire and dodge
    AI_REACT_MAX = 0.45  # s reaction delay ceiling (sloppy AI)
    AI_REACQUIRE_GAP = 0.4  # s a shot must be lost before the reflex delay re-arms
    AI_FIRE_CHANCE = 0.22  # per-eligible-frame chance to take an open shot
    AI_DIAG_FIRE_FACTOR = 0.6  # diagonal shots are taken less eagerly (beatability)
    AI_BANK_CHANCE = 0.05  # per-frame chance (no direct shot) to hunt a bank shot
    AI_BANK_HORIZON = 2.5  # s a candidate bank shot is simulated forward
    AI_DODGE_MISS = 0.12  # chance a fresh threat wave is never noticed at all
    AI_STALL_TIME = 3.5  # s without any clear shot before roaming engages
    AI_ROAM_TIME = 2.5  # s a roam waypoint is pursued before re-picking
    AI_FLANK_CHANCE = 0.003  # per-frame chance of a spontaneous repositioning roam
    AI_NOISE_TIME = 0.5  # s the per-direction noise biases are held (anti-jitter)
    # Utility weights: danger dominates everything; a clear shot beats alignment,
    # alignment beats standoff comfort; commit + noise stop dither and lockstep.
    AI_W_DANGER = 100.0
    AI_W_SHOT = 8.0
    AI_W_ALIGN = 1.0
    AI_W_STANDOFF = 0.4
    AI_W_COMMIT = 3.0
    AI_W_REVERSE = 4.0
    AI_W_NOISE = 0.5
    AI_W_ROAM = 2.0

    SHELL_SPEED = 22.0  # px/s
    SHELL_BOUNCES = 2  # two ricochets, then the shell expires
    FIRE_COOLDOWN = 0.6  # seconds between a tank's shots
    MAX_SHELLS_PER_TANK = 2

    # Impact effects.  A ricochet throws a short-lived spark at the bounce
    # point; a tank hitting a wall (or the other tank) above BUMP_MIN_SPEED
    # flashes a dent into the surface.  The speed floor keeps the dent for
    # genuine impacts -- grinding slowly against a wall stays quiet.
    SPARK_LIFE = 0.18  # s a ricochet spark lives
    BUMP_LIFE = 0.25  # s an impact dent lives
    BUMP_MIN_SPEED = 6.0  # px/s minimum impact speed that flashes a dent

    # Between-round sequence, staged in order: the hit blast (a double
    # shockwave that rattles the field) plays over the old walls, the score
    # flashes up as an inverted banner, the fresh field then wipes in behind a
    # diagonal frontier, and finally the tanks materialise at their corners.
    ROUND_EXPLODE = 0.5  # s the hit blast plays (matches the per-hit fx life)
    ROUND_SCORE = 1.0  # s the score is displayed over the old field
    ROUND_SWAP = 0.6  # s the old field dither-dissolves into the new one
    ROUND_SPAWN = 0.35  # s the tanks take to materialise once respawned
    RESPAWN_DELAY = ROUND_EXPLODE + ROUND_SCORE + ROUND_SWAP  # total downtime
    MODE_NAME_TIME = 2.0
    WIN_RESTART_TIME = 8.0

    WIN_BLAST_GROW = 0.4  # s the match-winning blast disc expands to fill the panel
    WIN_BLAST_FADE = 1.0  # s the blast then dissolves into the Game Over screen
    WIN_BLAST_MAX_RADIUS = 40  # px -> covers 28x28 from any origin (corner ~40)
    WIN_FIRE_PERIOD = 1.5  # s between the winner's celebratory salute shots

    def __init__(self, width: int, height: int, mode_manager: ModeManager) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.rng = np.random.default_rng()
        now = time.time()
        self._start_time = now
        self._last_frame_time = now
        self._reset_match(now)

    # ------------------------------------------------------------------
    # Match setup
    # ------------------------------------------------------------------

    def _default_walls(self) -> list[tuple[int, int, int, int]]:
        """The classic fixed layout: two horizontal bars and a centre block.

        Interior wall blocks are inclusive ``(r0, r1, c0, c1)`` rects.  Used as
        the generator's fallback and by tests that want a known centre block.
        """
        h, w = self.height, self.width
        return [
            (6, 7, w // 2 - 3, w // 2 + 2),  # top bar
            (h - 8, h - 7, w // 2 - 3, w // 2 + 2),  # bottom bar
            (h // 2 - 2, h // 2 + 1, w // 2 - 2, w // 2 + 1),  # centre block
        ]

    def _make_walls(self) -> list[tuple[int, int, int, int]]:
        """A random, fair interior layout so each round looks different.

        Every layout is built with 180-degree rotational symmetry (each tank
        sees the same arena rotated half a turn -> provably fair) from a small
        set of hand-shaped templates, confined to an interior box that leaves a
        3px-wide corridor open around the whole border.  Templates place their
        pieces toward the walls and corners rather than the centre; a spread
        check rejects anything clumped in the middle of the field and a gap
        check rejects slots between pieces too narrow for a tank to drive (or
        shoot) through.  A safety check additionally enforces clear spawn
        pockets and spawn-to-spawn connectivity; on the rare miss it retries,
        falling back to the classic layout.
        """
        templates = (
            self._tpl_classic,
            self._tpl_windmill,
            self._tpl_bunkers,
            self._tpl_corridors,
            self._tpl_corners,
            self._tpl_stagger,
        )
        for _ in range(20):
            seed = templates[int(self.rng.integers(len(templates)))]()
            rects = self._symmetrize(seed)
            if (
                self._layout_is_safe(rects)
                and self._layout_is_spread(rects)
                and self._gaps_are_passable(rects)
            ):
                return rects
        return self._default_walls()

    def _gaps_are_passable(self, rects: list[tuple[int, int, int, int]]) -> bool:
        """True if every gap between two wall pieces is either closed (the
        pieces touch, reading as one solid shape) or at least ``MIN_GAP`` wide,
        so a tank can always drive -- and shoot -- through any opening."""
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                dr = max(0, max(a[0], b[0]) - min(a[1], b[1]) - 1)
                dc = max(0, max(a[2], b[2]) - min(a[3], b[3]) - 1)
                if dr == 0 and dc == 0:
                    continue  # touching/overlapping -> no gap at all
                if max(dr, dc) < self.MIN_GAP:
                    return False
        return True

    def _layout_is_spread(self, rects: list[tuple[int, int, int, int]]) -> bool:
        """True if cover is spread across the field rather than clumped in the
        middle: enough total wall to fight around (but room left to maneuver),
        no more than half of it in the central box, and at least two separate
        wall pieces -- never a single connected clump."""
        mask = self._walls_mask(rects)
        area = int(mask.sum())
        if not 20 <= area <= 80:
            return False
        h, w = self.height, self.width
        central = int(mask[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5].sum())
        return central * 2 <= area and self._wall_clumps(mask) >= 2

    @staticmethod
    def _wall_clumps(mask: np.ndarray) -> int:
        """Count 4-connected components of set pixels in ``mask``."""
        seen = np.zeros(mask.shape, dtype=bool)
        count = 0
        for r, c in zip(*np.nonzero(mask)):
            if seen[r, c]:
                continue
            count += 1
            seen[r, c] = True
            stack = [(int(r), int(c))]
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (
                        0 <= ny < mask.shape[0]
                        and 0 <= nx < mask.shape[1]
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        return count

    def _mirror_rect(
        self, rect: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """The 180-degree rotation of ``rect`` about the panel centre."""
        r0, r1, c0, c1 = rect
        h, w = self.height, self.width
        return (h - 1 - r1, h - 1 - r0, w - 1 - c1, w - 1 - c0)

    def _symmetrize(
        self, rects: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Add each rect's 180-degree mirror (skipping self-symmetric ones)."""
        out: list[tuple[int, int, int, int]] = []
        for rect in rects:
            out.append(rect)
            mirror = self._mirror_rect(rect)
            if mirror != rect:
                out.append(mirror)
        return out

    def _centred_block(self, hs: int) -> tuple[int, int, int, int]:
        """A square block centred on the panel, so it is self-symmetric.

        Bounds sum to ``h - 1`` / ``w - 1`` about the reflection axis; ``hs`` is
        the half-extent (``hs=1`` reproduces the classic 4x4 centre block).
        """
        h, w = self.height, self.width
        return (h // 2 - 1 - hs, h // 2 + hs, w // 2 - 1 - hs, w // 2 + hs)

    def _tpl_classic(self) -> list[tuple[int, int, int, int]]:
        """A horizontal bar near the top (mirror adds the bottom one) plus a
        small centre block -- the shape of the fixed classic field."""
        w = self.width
        row = int(self.rng.integers(5, 8))
        half = int(self.rng.integers(2, 4))
        bar = (row, row + 1, w // 2 - half - 1, w // 2 + half)  # centred -> mirrors cleanly
        return [bar, self._centred_block(1)]

    def _tpl_windmill(self) -> list[tuple[int, int, int, int]]:
        """Two long bars pinwheeled about the centre; the mirror completes a
        field-spanning four-armed windmill (the classic *Combat* arena) whose
        arms reach toward the border, keeping the middle itself open."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        gap = int(self.rng.integers(4, 6))
        arm = int(self.rng.integers(8, 11))
        hbar = (cy - gap - 1, cy - gap, cx - arm, cx - 1)
        vbar = (cy - arm, cy - 1, cx + gap - 1, cx + gap)
        return [hbar, vbar]

    def _tpl_bunkers(self) -> list[tuple[int, int, int, int]]:
        """A bunker above the centre and one to its right; the mirror adds the
        opposing pair -> four compass-point bunkers around an open middle."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        off = int(self.rng.integers(5, 7))
        size = int(self.rng.integers(2, 4))
        north = (cy - off - size, cy - off, cx - 2, cx + 1)
        east = (cy - 2, cy + 1, cx + off, cx + off + size)
        return [north, east]

    def _tpl_corridors(self) -> list[tuple[int, int, int, int]]:
        """A tall bar toward the left wall (the mirror adds the right one),
        carving three wide north-south lanes with the middle left open."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        gap = int(self.rng.integers(5, 7))
        half = int(self.rng.integers(3, 5))
        return [(cy - half - 1, cy + half, cx - gap - 1, cx - gap)]

    def _tpl_corners(self) -> list[tuple[int, int, int, int]]:
        """A block in the upper-right corner region plus a short bar off the
        left wall at mid-height; the mirror opposes both, spreading cover to
        the corners and flanks with nothing in the middle."""
        h, w = self.height, self.width
        cy = h // 2
        roff = int(self.rng.integers(5, 7))
        coff = int(self.rng.integers(w - 12, w - 8))
        block = (roff, roff + 3, coff, coff + 3)
        length = int(self.rng.integers(3, 6))
        wing = (cy - 1, cy, 5, 4 + length)
        return [block, wing]

    def _tpl_stagger(self) -> list[tuple[int, int, int, int]]:
        """A long bar in the upper half (the mirror makes a staggered Z),
        pushed away from the centre so the middle rows stay open."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        off = int(self.rng.integers(5, 8))
        length = int(self.rng.integers(5, 8))
        return [(cy - off - 1, cy - off, cx - length, cx - 1)]

    def _layout_is_safe(self, rects: list[tuple[int, int, int, int]]) -> bool:
        """True if every rect stays in the interior box, clears both spawn
        pockets, and leaves the two spawn corners connected for a 3x3 tank."""
        h, w = self.height, self.width
        for r0, r1, c0, c1 in rects:
            if r0 < 4 or c0 < 4 or r1 > h - 5 or c1 > w - 5:
                return False  # would break the open perimeter corridor
            if self._rect_hits_box(r0, r1, c0, c1, 1, 6, 1, 6):
                return False  # left/top spawn pocket
            if self._rect_hits_box(r0, r1, c0, c1, h - 7, h - 2, w - 7, w - 2):
                return False  # right/bottom spawn pocket
        return self._spawns_connected(rects)

    @staticmethod
    def _rect_hits_box(r0, r1, c0, c1, br0, br1, bc0, bc1) -> bool:
        return not (r1 < br0 or r0 > br1 or c1 < bc0 or c0 > bc1)

    def _spawns_connected(self, rects: list[tuple[int, int, int, int]]) -> bool:
        """True if a 3x3 tank can travel from one spawn centre to the other,
        stepping one pixel at a time in the four cardinal directions."""
        start = (3, 3)
        goal = (self.width - 4, self.height - 4)
        seen = {start}
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt in seen or self._tank_blocked(nxt[0], nxt[1], rects):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False

    def _spawn(self, side: int) -> dict:
        h, w = self.height, self.width
        if side == 0:
            pos = [3.0, 3.0]
            target = (w - 4.0, h - 4.0)
        else:
            pos = [float(w - 4), float(h - 4)]
            target = (3.0, 3.0)
        return {
            "pos": pos,
            "heading": self._heading_to(pos, target),
            "vel": 0.0,
            "alive": True,
            "fire_cd_until": 0.0,
            # Per-spawn competence factor (0=sharp, 1=sloppy): scales reaction
            # delay and miss chance so the AI is not perfect and varies by round.
            "ai_skill": float(self.rng.uniform(0.0, 1.0)),
            # Per-spawn comfort range so each round's AI fights differently.
            "ai_standoff": float(self.rng.uniform(self.AI_STANDOFF - 3, self.AI_STANDOFF + 4)),
            "ai_react_at": 0.0,  # earliest time a lined-up shot may be taken
            "ai_shot_seen": 0.0,  # last time a clear shot existed (re-arm tracking)
            "ai_threat_react_at": 0.0,  # earliest time the danger term is perceived
            "ai_threat_ignored": False,  # this threat wave was never noticed
            "ai_had_threat": False,  # edge detector for fresh threat waves
            "ai_last_move": (0, 0),  # commit bonus target (anti-dither)
            "ai_noise": {},  # per-direction bias, held for AI_NOISE_TIME (anti-jitter)
            "ai_noise_until": 0.0,  # re-roll the noise biases after this
            "ai_roam_target": None,  # (x, y) waypoint while stalemate-roaming
            "ai_roam_until": 0.0,  # re-pick the roam waypoint after this
            "intent": {"move": (0, 0)},
            "last_input_time": None,
            "wall_contact": False,  # edge detector: dent only on fresh impacts
            "side": side,
        }

    def _reset_match(self, now: float) -> None:
        self.walls = self._make_walls()
        self.tanks = [self._spawn(0), self._spawn(1)]
        self.shells: list[dict] = []
        self.fx: list[dict] = []
        self.score = [0, 0]
        self.winner: int | None = None
        self.win_time: float | None = None
        self.win_x: float | None = None  # blast origin (downed tank) for the win fx
        self.win_y: float | None = None
        self._win_noise: np.ndarray | None = None  # stable dither for the fade
        # Between-round transition state (None while a round is being played).
        self.round_over_at: float | None = None  # when the current round ended
        self.next_walls: list[tuple[int, int, int, int]] | None = None  # field to swap in
        self._round_noise: np.ndarray | None = None  # stable dither for score/field fades
        self.tanks_spawned_at: float | None = now  # drives the spawn-in animation
        self._start_time = now
        self._last_frame_time = now

    def _restart_match(self) -> None:
        self._reset_match(time.time())

    @staticmethod
    def _heading_to(origin, target) -> int:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        angle = math.atan2(dy, dx)
        return int(round(angle / (2 * math.pi / HEADINGS))) % HEADINGS

    # ------------------------------------------------------------------
    # Collision helpers
    # ------------------------------------------------------------------

    def _point_blocked(self, x: float, y: float) -> bool:
        """True if pixel ``(x, y)`` is outside the inner arena or in a wall."""
        xi, yi = int(round(x)), int(round(y))
        if xi < 1 or xi > self.width - 2 or yi < 1 or yi > self.height - 2:
            return True
        for r0, r1, c0, c1 in self.walls:
            if r0 <= yi <= r1 and c0 <= xi <= c1:
                return True
        return False

    def _tank_blocked(self, cx: float, cy: float, walls=None) -> bool:
        """True if a tank centred at ``(cx, cy)`` would overlap a wall/border.

        ``walls`` defaults to the live arena but can be a candidate rect list so
        the layout generator can validate a layout before installing it.
        """
        walls = self.walls if walls is None else walls
        xi, yi = int(round(cx)), int(round(cy))
        fr0, fr1 = yi - self.TANK_HALF, yi + self.TANK_HALF
        fc0, fc1 = xi - self.TANK_HALF, xi + self.TANK_HALF
        if fc0 < 1 or fc1 > self.width - 2 or fr0 < 1 or fr1 > self.height - 2:
            return True
        for r0, r1, c0, c1 in walls:
            if not (fr1 < r0 or fr0 > r1 or fc1 < c0 or fc0 > c1):
                return True
        return False

    def _move_blocked(self, tank: dict, cx: float, cy: float) -> bool:
        """True if ``tank`` may not occupy ``(cx, cy)``.

        Combines the wall/border test with a tank-vs-tank check so the two
        tanks can never share any pixel: their 3x3 footprints overlap whenever
        their centres are within ``2 * TANK_HALF`` on both axes.
        """
        if self._tank_blocked(cx, cy):
            return True
        other = self.tanks[1 - tank["side"]]
        if other is tank or not other["alive"]:
            return False
        oxi, oyi = int(round(other["pos"][0])), int(round(other["pos"][1]))
        xi, yi = int(round(cx)), int(round(cy))
        reach = 2 * self.TANK_HALF
        return abs(xi - oxi) <= reach and abs(yi - oyi) <= reach

    # ------------------------------------------------------------------
    # Input + AI
    # ------------------------------------------------------------------

    def _side_index(self, side) -> int:
        if isinstance(side, str):
            return 0 if side == "left" else 1
        return int(side)

    def set_controller_input(self, side, *, move_x: int, move_y: int) -> None:
        """Set a tank's drive intent from a controller (called every frame).

        ``move_x`` and ``move_y`` are each in ``{-1, 0, 1}`` (y points down).
        Two held D-pad arrows combine into a 45-degree diagonal, so a tank can
        drive and aim in any of eight directions.  The AI takeover timer only
        resets while there is actual input, so an idle controller still hands
        the tank to the AI after the delay.
        """
        mx = int(max(-1, min(1, move_x)))
        my = int(max(-1, min(1, move_y)))
        tank = self.tanks[self._side_index(side)]
        tank["intent"] = {"move": (mx, my)}
        if mx or my:
            tank["last_input_time"] = time.time()

    def fire(self, side, now: float | None = None) -> None:
        """Fire a shell for ``side`` if its tank is alive and off cooldown."""
        idx = self._side_index(side)
        now = time.time() if now is None else now
        self.tanks[idx]["last_input_time"] = now
        self._spawn_shell(idx, now)

    def restart_if_game_over(self) -> None:
        if self.winner is not None:
            self._restart_match()

    def _controller_active(self, tank, now: float) -> bool:
        return (
            tank["last_input_time"] is not None
            and now - tank["last_input_time"] < self.AI_TAKEOVER_DELAY
        )

    def _lane_clear(
        self, x: float, y: float, direction: tuple[int, int], tcx: int, tcy: int
    ) -> bool:
        """True if a shell fired from ``(x, y)`` straight along ``direction``
        reaches the footprint at ``(tcx, tcy)`` with no wall/border in between.

        Direct shots only -- the trace stops where the shell would first
        bounce, so the AI never relies on a ricochet to land a hit.  For the
        diagonal directions the two axis-adjacent cells are checked as well,
        mirroring the per-axis reflection in :meth:`_advance_shell`: a real
        diagonal shell bounces at a corner crossing even when the diagonal
        cell itself is open.
        """
        dx, dy = direction
        cx = x + dx * (self.TANK_HALF + 1)
        cy = y + dy * (self.TANK_HALF + 1)
        for _ in range(self.width + self.height):
            xi, yi = int(round(cx)), int(round(cy))
            if abs(xi - tcx) <= self.TANK_HALF and abs(yi - tcy) <= self.TANK_HALF:
                return True
            if self._point_blocked(cx, cy):
                return False
            if dx and dy and (self._point_blocked(cx + dx, cy) or self._point_blocked(cx, cy + dy)):
                return False  # corner crossing: a real shell would bounce here
            cx += dx
            cy += dy
        return False

    def _aligned_dirs(self, x: float, y: float, foe: dict) -> list[tuple[int, int]]:
        """The unit directions whose firing line through the foe the position
        ``(x, y)`` sits on within ``AI_ALIGN_TOL``: the foe's row and column
        first (a clean cardinal shot is preferred), then the 45-degree
        diagonal."""
        dx = foe["pos"][0] - x
        dy = foe["pos"][1] - y
        adx, ady = abs(dx), abs(dy)
        dirs: list[tuple[int, int]] = []
        if ady <= self.AI_ALIGN_TOL and adx > self.AI_ALIGN_TOL:
            dirs.append((1 if dx > 0 else -1, 0))
        if adx <= self.AI_ALIGN_TOL and ady > self.AI_ALIGN_TOL:
            dirs.append((0, 1 if dy > 0 else -1))
        if abs(adx - ady) <= self.AI_ALIGN_TOL and adx > self.AI_ALIGN_TOL and ady > self.AI_ALIGN_TOL:
            dirs.append((1 if dx > 0 else -1, 1 if dy > 0 else -1))
        return dirs

    def _line_misalignment(self, x: float, y: float, foe: dict) -> float:
        """Cell distance to the nearest firing-line family through the foe:
        its row, its column, or the 45-degree diagonal passing on this side."""
        adx = abs(foe["pos"][0] - x)
        ady = abs(foe["pos"][1] - y)
        return min(adx, ady, abs(adx - ady))

    def _ai_danger(
        self, shell_cells: list[list[tuple[int, int, float]]], cx: float, cy: float
    ) -> float:
        """Penalty for standing at ``(cx, cy)`` given the projected cells of
        every threatening shell (:meth:`_project_shell_cells`): a projected
        cell within the footprint plus a 1px halo adds more the sooner the
        shell gets there and the closer it passes, so the penalty falls off
        across the halo and the AI always has a gradient out of the lane."""
        reach = self.TANK_HALF + 1
        xi, yi = int(round(cx)), int(round(cy))
        danger = 0.0
        for cells in shell_cells:
            for sx, sy, t in cells:
                d = max(abs(sx - xi), abs(sy - yi))
                if d <= reach:
                    danger += (self.AI_HORIZON - t) * (1 + reach - d)
        return danger

    def _shot_lands(self, idx: int, direction: tuple[int, int]) -> bool:
        """True if a shell fired now by ``idx`` along ``direction`` would strike
        the foe -- ricochets included -- before expiring or coming back around
        into the shooter itself.

        Runs the real shell physics (:meth:`_advance_shell` plus the bounce
        budget) against a stationary foe, so whatever this approves is exactly
        what a fired shell does: this is how the AI finds deliberate bank shots
        without ever banking one into its own tank.
        """
        tank = self.tanks[idx]
        foe = self.tanks[1 - idx]
        dx, dy = direction
        sx = tank["pos"][0] + dx * (self.TANK_HALF + 1)
        sy = tank["pos"][1] + dy * (self.TANK_HALF + 1)
        if self._point_blocked(sx, sy):
            return False
        norm = math.hypot(dx, dy)
        vx = dx / norm * self.SHELL_SPEED
        vy = dy / norm * self.SHELL_SPEED
        fcx, fcy = int(round(foe["pos"][0])), int(round(foe["pos"][1]))
        ocx, ocy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
        bounces = self.SHELL_BOUNCES
        t, step = 0.0, 0.04
        while t < self.AI_BANK_HORIZON:
            sx, sy, vx, vy, bounced = self._advance_shell(sx, sy, vx, vy, step)
            if bounced:
                bounces -= 1
                if bounces < 0:
                    return False
            xi, yi = int(round(sx)), int(round(sy))
            if abs(xi - fcx) <= self.TANK_HALF and abs(yi - fcy) <= self.TANK_HALF:
                return True
            if abs(xi - ocx) <= self.TANK_HALF and abs(yi - ocy) <= self.TANK_HALF:
                return False  # it would ricochet back into ourselves
            t += step
        return False

    def _ai_intent(self, idx: int, now: float) -> dict:
        """Beatable utility AI, built for eight-direction play.

        Each frame the nine candidate moves (eight directions plus hold) are
        scored on shell danger (its own ricochets included), clear-firing-lane
        seeking, alignment with the foe's firing lines, and standoff comfort;
        the best move wins.  Firing is decided separately: a direct shot down a
        lane a real shell would clear, or -- when no direct shot exists -- an
        occasional deliberate bank shot validated with the real ricochet
        physics (never one that would come back into itself).  Human-like
        imperfection comes from a per-spawn reaction delay, the odd ignored
        threat wave, and a random hold on open shots; the per-direction noise
        biases are held for ``AI_NOISE_TIME`` so the tank drives smoothly
        instead of jittering, and a roam waypoint engages on stalemates (or as
        the odd spontaneous flank) so AI-vs-AI attract mode never deadlocks.
        """
        tank = self.tanks[idx]
        foe = self.tanks[1 - idx]
        if not foe["alive"]:
            return {"move": (0, 0)}

        react = self.AI_REACT_MIN + tank["ai_skill"] * (self.AI_REACT_MAX - self.AI_REACT_MIN)
        x, y = tank["pos"]
        fcx, fcy = int(round(foe["pos"][0])), int(round(foe["pos"][1]))
        if tank["ai_shot_seen"] == 0.0:
            tank["ai_shot_seen"] = now  # lazy init so the stall clock starts at spawn
            # The spawn corners sit on each other's diagonal, so arm the reflex
            # here too -- no instant snap-shot the moment a round begins.
            tank["ai_react_at"] = now + react

        # Threat perception: project threatening shells once, gated by a human
        # reaction delay on each fresh threat wave -- and the occasional wave
        # that is never noticed at all, so the AI is not an unhittable wall.
        # A tank's own shell becomes a threat too once it has ricocheted
        # (outbound it only ever flies away faster than the tank can drive).
        threats = [
            s
            for s in self.shells
            if s["owner"] != idx or s["bounces"] < self.SHELL_BOUNCES
        ]
        if threats:
            if not tank["ai_had_threat"]:
                tank["ai_threat_react_at"] = now + react
                tank["ai_threat_ignored"] = (
                    self.rng.random() < self.AI_DODGE_MISS * (0.5 + tank["ai_skill"])
                )
            tank["ai_had_threat"] = True
        else:
            tank["ai_had_threat"] = False
            tank["ai_threat_ignored"] = False
        shell_cells: list[list[tuple[int, int, float]]] = []
        if threats and now >= tank["ai_threat_react_at"] and not tank["ai_threat_ignored"]:
            shell_cells = [self._project_shell_cells(s, self.AI_HORIZON) for s in threats]

        # Work out whether a shot exists right now; this also drives the stall
        # clock behind the roam below.
        shot = None
        for d in self._aligned_dirs(x, y, foe):
            if self._lane_clear(x, y, d, fcx, fcy):
                shot = d
                break

        # Stalemate roam: with no shot available for a while -- or now and then
        # on a spontaneous flanking whim -- pursue a random waypoint so the AI
        # drifts off dead (wall-blocked) firing lines instead of orbiting them
        # forever; the per-move noise below breaks what mirror symmetry remains
        # between two AIs.
        if shot is not None:
            tank["ai_roam_target"] = None
        elif now - tank["ai_shot_seen"] > self.AI_STALL_TIME and (
            tank["ai_roam_target"] is None or now >= tank["ai_roam_until"]
        ):
            tank["ai_roam_target"] = (
                float(self.rng.uniform(3, self.width - 4)),
                float(self.rng.uniform(3, self.height - 4)),
            )
            tank["ai_roam_until"] = now + self.AI_ROAM_TIME
        elif tank["ai_roam_target"] is None and self.rng.random() < self.AI_FLANK_CHANCE:
            tank["ai_roam_target"] = (
                float(self.rng.uniform(3, self.width - 4)),
                float(self.rng.uniform(3, self.height - 4)),
            )
            tank["ai_roam_until"] = now + self.AI_ROAM_TIME
        elif tank["ai_roam_target"] is not None and now >= tank["ai_roam_until"]:
            tank["ai_roam_target"] = None  # a flank whim expires quietly

        # Per-direction noise biases, held for a stretch: fresh randomness every
        # frame would flip near-tied moves back and forth and read as jitter on
        # the panel, so the biases only re-roll every AI_NOISE_TIME seconds.
        if not tank["ai_noise"] or now >= tank["ai_noise_until"]:
            tank["ai_noise"] = {
                m: float(self.rng.uniform(0.0, self.AI_W_NOISE)) for m in ((0, 0), *_MOVE_HEADINGS)
            }
            tank["ai_noise_until"] = now + self.AI_NOISE_TIME

        # Score the nine candidate moves and take the best.  A driving move is
        # scored a lookahead step along its direction; the hold move is scored
        # where the tank would actually come to rest after coasting out its
        # current speed -- otherwise the AI never stops *before* its target
        # line, overshoots, and oscillates back and forth across it forever.
        hvx, hvy = _HEADING_VECS[tank["heading"]]
        coast = tank["vel"] ** 2 / (2 * self.FRICTION)
        best_move, best_score = (0, 0), -math.inf
        for move in ((0, 0), *_MOVE_HEADINGS):
            mx, my = move
            if move != (0, 0) and self._move_blocked(tank, x + mx, y + my):
                continue
            if move == (0, 0):
                px, py = x + hvx * coast, y + hvy * coast
            else:
                px, py = x + mx * self.AI_LOOKAHEAD, y + my * self.AI_LOOKAHEAD
            s = -self.AI_W_DANGER * self._ai_danger(shell_cells, px, py)
            if any(
                self._lane_clear(px, py, d, fcx, fcy) for d in self._aligned_dirs(px, py, foe)
            ):
                s += self.AI_W_SHOT
            # While a shell is inbound the alignment pull is muted, otherwise it
            # yanks the tank straight back onto the lane it just dodged out of
            # and the two terms fight in a visible back-and-forth limit cycle.
            w_align = self.AI_W_ALIGN * (0.25 if shell_cells else 1.0)
            s -= w_align * self._line_misalignment(px, py, foe)
            dist = math.hypot(foe["pos"][0] - px, foe["pos"][1] - py)
            s -= self.AI_W_STANDOFF * abs(dist - tank["ai_standoff"])
            if move == tank["ai_last_move"]:
                s += self.AI_W_COMMIT
            elif move == (-tank["ai_last_move"][0], -tank["ai_last_move"][1]) and move != (0, 0):
                # Heading snaps redirect the full speed instantly, so an about-turn
                # has no damping phase at all -- penalize it or near-tied scores
                # vibrate the tank in place at full speed.  Danger still overrides.
                s -= self.AI_W_REVERSE
            s += tank["ai_noise"][move]
            if tank["ai_roam_target"] is not None:
                rx, ry = tank["ai_roam_target"]
                s -= self.AI_W_ROAM * math.hypot(rx - px, ry - py)
            if s > best_score:
                best_move, best_score = move, s
        tank["ai_last_move"] = best_move

        # Fire: face the shot and pull the trigger after a reaction delay and a
        # random hold, so the AI misses some open shots and stays beatable. The
        # reflex timer only re-arms when the target was genuinely lost (not on
        # the frame-to-frame flicker of the lane as both tanks drift), otherwise
        # the AI could almost never satisfy the delay and would rarely fire.
        # Diagonal shots are held back a little extra: eight firing lines must
        # not double the pressure on a human player.
        if shot is not None:
            tank["heading"] = _MOVE_HEADINGS[shot]
            if now - tank["ai_shot_seen"] > self.AI_REACQUIRE_GAP:
                tank["ai_react_at"] = now + react
            tank["ai_shot_seen"] = now
            chance = self.AI_FIRE_CHANCE * (
                self.AI_DIAG_FIRE_FACTOR if shot[0] and shot[1] else 1.0
            )
            if now >= tank["ai_react_at"] and self.rng.random() < chance:
                self._spawn_shell(idx, now)
        elif now >= tank["ai_react_at"] and self.rng.random() < self.AI_BANK_CHANCE:
            # No direct shot: occasionally hunt for a deliberate bank shot --
            # the first of the eight directions whose real simulated ricochet
            # path lands on the foe (and not back on us).  The scan chance is
            # the throttle: banks stay a flourish, not a barrage.
            for d in _MOVE_HEADINGS:
                if self._shot_lands(idx, d):
                    tank["heading"] = _MOVE_HEADINGS[d]
                    tank["ai_shot_seen"] = now
                    self._spawn_shell(idx, now)
                    break
        return {"move": best_move}

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _update_match(self, now: float, dt: float) -> None:
        self.fx = [f for f in self.fx if f["until"] > now]
        if self.winner is not None:
            if self.win_time is not None and now - self.win_time >= self.WIN_RESTART_TIME:
                self._restart_match()
            return

        # Between rounds the tanks are down while the transition plays out
        # (explosion -> score -> field swap); they respawn only once it finishes.
        if self.round_over_at is not None:
            if now - self.round_over_at >= self.RESPAWN_DELAY:
                self._begin_new_round(now)
            return

        for idx, tank in enumerate(self.tanks):
            intent = (
                tank["intent"] if self._controller_active(tank, now) else self._ai_intent(idx, now)
            )
            self._drive_tank(tank, intent, now, dt)

        self._update_shells(now, dt)

    def _begin_new_round(self, now: float) -> None:
        """End the between-round transition: install the fresh field and respawn
        both tanks at their corners (preserving each side's controller timer)."""
        self.walls = self.next_walls or self._make_walls()
        self.next_walls = None
        self.round_over_at = None
        self._round_noise = None
        self.tanks_spawned_at = now
        last_inputs = [tank["last_input_time"] for tank in self.tanks]
        self.tanks = [self._spawn(0), self._spawn(1)]
        for tank, last_input in zip(self.tanks, last_inputs):
            tank["last_input_time"] = last_input

    def _drive_tank(self, tank: dict, intent: dict, now: float, dt: float) -> None:
        move = intent["move"]
        if move != (0, 0):
            # Snap heading to the pressed direction and drive forward.
            tank["heading"] = _MOVE_HEADINGS[move]
            tank["vel"] += self.THRUST_ACCEL * dt
        else:
            tank["vel"] = max(0.0, tank["vel"] - self.FRICTION * dt)
        tank["vel"] = min(self.MAX_SPEED, tank["vel"])

        dx, dy = _HEADING_VECS[tank["heading"]]
        x, y = tank["pos"]
        nx, ny = x + dx * tank["vel"] * dt, y + dy * tank["vel"] * dt
        # A fresh contact at speed flashes an impact dent.  Edge-detected on
        # the contact flag because a tank pressed against a wall stays blocked
        # every frame (the axis-slide below keeps it "moving" in place).
        contact = self._move_blocked(tank, nx, ny)
        if contact and not tank["wall_contact"] and tank["vel"] >= self.BUMP_MIN_SPEED:
            self._spawn_bump_fx(tank, now)
        tank["wall_contact"] = contact
        moved = False
        if not contact:
            tank["pos"] = [nx, ny]
            moved = True
        else:  # try sliding along one axis before giving up
            if not self._move_blocked(tank, nx, y):
                tank["pos"] = [nx, y]
                moved = True
            elif not self._move_blocked(tank, x, ny):
                tank["pos"] = [x, ny]
                moved = True
        if not moved:
            tank["vel"] = 0.0

    def _spawn_bump_fx(self, tank: dict, now: float) -> None:
        """Flash an impact dent one cell past the tank's nose -- the wall or
        tank-body pixel it just slammed into."""
        dx, dy = _HEADING_VECS[tank["heading"]]
        # Chebyshev-normalize (like the muzzle offset) so diagonal impacts
        # also land the dent in the first cell outside the 3x3 footprint.
        m = max(abs(dx), abs(dy))
        ndx, ndy = round(dx / m), round(dy / m)
        self.fx.append(
            {
                "kind": "bump",
                "x": tank["pos"][0] + ndx * (self.TANK_HALF + 1),
                "y": tank["pos"][1] + ndy * (self.TANK_HALF + 1),
                "start": now,
                "until": now + self.BUMP_LIFE,
            }
        )

    def _spawn_spark_fx(self, x: float, y: float, now: float) -> None:
        """Flash a ricochet spark where a shell just bounced (or expired)."""
        self.fx.append(
            {"kind": "spark", "x": x, "y": y, "start": now, "until": now + self.SPARK_LIFE}
        )

    def _spawn_shell(self, idx: int, now: float) -> None:
        tank = self.tanks[idx]
        if not tank["alive"] or now < tank["fire_cd_until"]:
            return
        if sum(1 for s in self.shells if s["owner"] == idx) >= self.MAX_SHELLS_PER_TANK:
            return
        dx, dy = _HEADING_VECS[tank["heading"]]
        x, y = tank["pos"]
        # Chebyshev-normalize the muzzle offset so a diagonal shell spawns in
        # the first cell outside the footprint instead of on the body corner.
        k = (self.TANK_HALF + 1) / max(abs(dx), abs(dy))
        tip = [x + dx * k, y + dy * k]
        if self._point_blocked(tip[0], tip[1]):
            return
        self.shells.append(
            {
                "pos": tip,
                "vel": [dx * self.SHELL_SPEED, dy * self.SHELL_SPEED],
                "bounces": self.SHELL_BOUNCES,
                "owner": idx,
            }
        )
        tank["fire_cd_until"] = now + self.FIRE_COOLDOWN

    def _advance_shell(
        self, x: float, y: float, vx: float, vy: float, dt: float
    ) -> tuple[float, float, float, float, bool]:
        """Step a shell one frame, reflecting off walls/border.

        Pure (reads only ``self.walls``/dimensions), so the AI can reuse it to
        project shell paths with exactly the same physics as the live update.
        """
        nx, ny = x + vx * dt, y + vy * dt
        bounced = False
        if self._point_blocked(nx, y):
            vx = -vx
            nx = x
            bounced = True
        if self._point_blocked(x, ny):
            vy = -vy
            ny = y
            bounced = True
        if not bounced and self._point_blocked(nx, ny):
            vx, vy = -vx, -vy
            nx, ny = x, y
            bounced = True
        return nx, ny, vx, vy, bounced

    def _project_shell_cells(
        self, shell: dict, horizon: float, step: float = 0.02
    ) -> list[tuple[int, int, float]]:
        """Predict the integer cells a shell will occupy over ``horizon`` seconds.

        Steps a copy of ``shell`` forward with :meth:`_advance_shell`, honoring
        its remaining ``bounces`` (stops once the shell would expire).  Returns
        ``(col, row, t)`` triples — the AI uses these to dodge and to avoid
        steering into a shell's path.
        """
        x, y = shell["pos"]
        vx, vy = shell["vel"]
        bounces = shell["bounces"]
        cells: list[tuple[int, int, float]] = []
        t = 0.0
        while t < horizon:
            x, y, vx, vy, bounced = self._advance_shell(x, y, vx, vy, step)
            if bounced:
                bounces -= 1
                if bounces < 0:
                    break
            t += step
            cells.append((int(round(x)), int(round(y)), t))
        return cells

    def _update_shells(self, now: float, dt: float) -> None:
        survivors: list[dict] = []
        for shell in self.shells:
            x, y = shell["pos"]
            vx, vy = shell["vel"]
            nx, ny, vx, vy, bounced = self._advance_shell(x, y, vx, vy, dt)
            if bounced:
                shell["bounces"] -= 1
                # Every ricochet sparks -- including the final one, where the
                # spark doubles as the expiring shell's fizzle.
                self._spawn_spark_fx(nx, ny, now)
                if shell["bounces"] < 0:
                    continue
            shell["pos"] = [nx, ny]
            shell["vel"] = [vx, vy]
            if self._resolve_hit(shell, now):
                # A hit ends the round and resets both tanks, so drop every
                # remaining shell rather than carrying it into the next round.
                self.shells = []
                return
            survivors.append(shell)
        self.shells = survivors

    def _resolve_hit(self, shell: dict, now: float) -> bool:
        # Shells do not discriminate: a ricochet strikes whichever tank it
        # meets, the shooter included (the muzzle offset keeps a fresh shell
        # outside its own tank's footprint).
        xi, yi = int(round(shell["pos"][0])), int(round(shell["pos"][1]))
        for idx, tank in enumerate(self.tanks):
            if not tank["alive"]:
                continue
            cx, cy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
            if abs(xi - cx) <= self.TANK_HALF and abs(yi - cy) <= self.TANK_HALF:
                self._register_hit(idx, now)
                return True
        return False

    def _register_hit(self, victim: int, now: float) -> None:
        """A shell struck ``victim``: the point always goes to the other side,
        so getting clipped by your own ricochet gifts your foe the point."""
        shooter = 1 - victim
        self.score[shooter] += 1
        vtank = self.tanks[victim]
        won = self.score[shooter] >= self.WIN_SCORE
        # A scored point ends the round: both tanks go down (the caller clears
        # any in-flight shells).
        for tank in self.tanks:
            tank["alive"] = False
        if won:
            # The match-winning hit gets a huge blast that fades into the Game
            # Over screen instead of the small per-hit ring.
            self.winner = shooter
            self.win_time = now
            self.win_x, self.win_y = vtank["pos"]
            self._win_noise = self.rng.random((self.height, self.width))
        else:
            # Start the between-round sequence: explosion now, then score, then a
            # dither-dissolve into a freshly generated fair field, then respawn.
            # The new field is generated up front but only swapped in later.
            self.round_over_at = now
            self.next_walls = self._make_walls()
            self._round_noise = self.rng.random((self.height, self.width))
            self.fx.append(
                {
                    "kind": "boom",
                    "x": vtank["pos"][0],
                    "y": vtank["pos"][1],
                    "start": now,
                    "until": now + 0.5,
                }
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_border(self, frame: np.ndarray) -> None:
        frame[0, :] = 1
        frame[-1, :] = 1
        frame[:, 0] = 1
        frame[:, -1] = 1

    def _draw_walls(self, frame: np.ndarray) -> None:
        for r0, r1, c0, c1 in self.walls:
            frame[r0 : r1 + 1, c0 : c1 + 1] = 1

    def _draw_tank(self, frame: np.ndarray, tank: dict) -> None:
        cx, cy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
        h = self.TANK_HALF
        r0, r1 = max(0, cy - h), min(self.height - 1, cy + h)
        c0, c1 = max(0, cx - h), min(self.width - 1, cx + h)
        frame[r0 : r1 + 1, c0 : c1 + 1] = 1
        if tank["side"] == 1:  # hollow ring tells the right tank apart
            if 0 <= cy < self.height and 0 <= cx < self.width:
                frame[cy, cx] = 0
        dx, dy = _HEADING_VECS[tank["heading"]]
        # Chebyshev-normalize the step so diagonal barrels also land outside
        # the 3x3 body (two visible pixels for every heading).
        k = 1.0 / max(abs(dx), abs(dy))
        for step in (h + 1, h + 2):
            bx = int(round(cx + dx * step * k))
            by = int(round(cy + dy * step * k))
            if 0 < bx < self.width - 1 and 0 < by < self.height - 1:
                frame[by, bx] = 1

    def _draw_shells(self, frame: np.ndarray) -> None:
        for shell in self.shells:
            xi, yi = int(round(shell["pos"][0])), int(round(shell["pos"][1]))
            if 0 <= yi < self.height and 0 <= xi < self.width:
                frame[yi, xi] = 1

    def _draw_fx(self, frame: np.ndarray, now: float) -> None:
        for f in self.fx:
            progress = (now - f["start"]) / max(1e-6, f["until"] - f["start"])
            kind = f.get("kind", "boom")
            if kind == "spark":
                self._draw_spark(frame, f, progress)
            elif kind == "bump":
                self._draw_bump(frame, f, progress)
            else:
                self._draw_boom(frame, f, progress)

    def _draw_boom(self, frame: np.ndarray, f: dict, progress: float) -> None:
        """The per-hit blast: a double shockwave -- a leading ring racing out
        with a thinner trailing ring behind it -- over an interior of embers
        that dither away as the blast cools."""
        cx, cy = int(round(f["x"])), int(round(f["y"]))
        yy, xx = np.ogrid[: self.height, : self.width]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        r1 = 1 + progress * 7
        frame[(d2 >= (r1 - 1) ** 2) & (d2 <= (r1 + 1) ** 2)] = 1
        r2 = progress * 4
        frame[(d2 >= (r2 - 0.5) ** 2) & (d2 <= (r2 + 0.5) ** 2)] = 1
        if self._round_noise is not None:
            embers = (d2 < (r2 - 0.5) ** 2) & (self._round_noise < (1.0 - progress) * 0.6)
            frame[embers] = 1

    def _draw_spark(self, frame: np.ndarray, f: dict, progress: float) -> None:
        """A ricochet flash: a plus-shaped burst pinned at the bounce point
        that collapses to a dot.  Deliberately a stationary connected cluster
        -- lone travelling pixels would read as more shells."""
        cx, cy = int(round(f["x"])), int(round(f["y"]))
        pts = [(cx, cy)]
        if progress < 0.6:
            pts += [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
        for x, y in pts:
            if 0 <= y < self.height and 0 <= x < self.width:
                frame[y, x] = 1

    def _draw_bump(self, frame: np.ndarray, f: dict, progress: float) -> None:
        """An impact dent: pixels at the contact point blink *off* -- visible
        against the lit wall or tank body on the monochrome panel -- healing
        to a single pixel as it fades.  Blink-off only: lit debris pixels
        would read as shells."""
        cx, cy = int(round(f["x"])), int(round(f["y"]))
        hole = 1 if progress < 0.5 else 0
        frame[
            max(0, cy - hole) : min(self.height, cy + hole + 1),
            max(0, cx - hole) : min(self.width, cx + hole + 1),
        ] = 0

    def _walls_mask(self, walls: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Rasterize a list of wall rects into a 0/1 pixel mask."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for r0, r1, c0, c1 in walls:
            mask[r0 : r1 + 1, c0 : c1 + 1] = 1
        return mask

    def _draw_round_transition(self, frame: np.ndarray, now: float) -> None:
        """Render the staged between-round sequence: the blast rattles the old
        field, the score flashes up as an inverted banner, then the fresh field
        wipes in behind a diagonal frontier.  The tanks stay hidden until the
        sequence completes and they respawn (:meth:`_begin_new_round`)."""
        t = now - (self.round_over_at or now)
        if t < self.ROUND_EXPLODE:
            self._draw_walls_shaken(frame, t)  # blast fx is drawn on top later
        elif t < self.ROUND_EXPLODE + self.ROUND_SCORE:
            self._draw_walls(frame)
            self._draw_score(frame, (t - self.ROUND_EXPLODE) / self.ROUND_SCORE)
        else:
            p = (t - self.ROUND_EXPLODE - self.ROUND_SCORE) / self.ROUND_SWAP
            self._draw_field_morph(frame, min(1.0, p))

    def _draw_walls_shaken(self, frame: np.ndarray, t: float) -> None:
        """The old field rattling under the blast: the wall layer jitters by up
        to a pixel, with an amplitude that decays as the explosion plays out.
        Walls never reach the panel edge, so the 1px roll cannot wrap them."""
        amp = 1.0 - t / self.ROUND_EXPLODE
        dx = round(math.sin(t * 60.0) * amp)
        dy = round(math.cos(t * 47.0) * amp)
        if dx == 0 and dy == 0:
            self._draw_walls(frame)
            return
        layer = np.zeros_like(frame)
        self._draw_walls(layer)
        frame |= np.roll(layer, (dy, dx), axis=(0, 1))

    def _draw_score(self, frame: np.ndarray, progress: float) -> None:
        """Flash the score as an inverted banner: a solid lit band with the
        digits punched out dark, dither-revealed as it sweeps in.  The mass
        pixel flip reads far stronger on the panel than lit digits on black."""
        frame[10:18, :] = 0
        band = np.zeros_like(frame)
        band[10:18, :] = 1
        text_frame = np.zeros_like(frame)
        text.write_centered(text_frame, self._score_text(), y=11, size=6, style="regular")
        band[text_frame == 1] = 0
        reveal = min(1.0, progress / 0.25)  # fade in over the first quarter, then hold
        if self._round_noise is None or reveal >= 1.0:
            frame[band == 1] = 1
        else:
            frame[(band == 1) & (self._round_noise < reveal)] = 1

    def _draw_field_morph(self, frame: np.ndarray, p: float) -> None:
        """Wipe the freshly generated field in behind a lit diagonal frontier
        that races across the panel, the old field still standing ahead of it."""
        old = self._walls_mask(self.walls)
        new = self._walls_mask(self.next_walls or self.walls)
        yy, xx = np.ogrid[: self.height, : self.width]
        diag = xx + yy
        edge = p * (self.width + self.height - 2)
        shown = np.where(diag < edge, new, old)
        frame[shown == 1] = 1
        if p < 1.0:
            frame[np.abs(diag - edge) < 1.0] = 1  # the sweeping frontier beam

    def _draw_tank_maybe_spawning(self, frame: np.ndarray, tank: dict, now: float) -> None:
        """Draw a tank, materialising it from a spark for a moment after respawn."""
        spawned = self.tanks_spawned_at
        if spawned is not None and 0.0 <= now - spawned < self.ROUND_SPAWN:
            self._draw_tank_spawn(frame, tank, (now - spawned) / self.ROUND_SPAWN)
        else:
            self._draw_tank(frame, tank)

    def _draw_tank_spawn(self, frame: np.ndarray, tank: dict, q: float) -> None:
        """A quick materialise: a centre spark grows into the full tank body."""
        if q >= 0.7:
            self._draw_tank(frame, tank)
            return
        cx, cy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
        if 0 <= cy < self.height and 0 <= cx < self.width:
            frame[cy, cx] = 1
        if q >= 0.35:  # grow to a plus shape before filling out into the body
            for bx, by in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if 0 < bx < self.width - 1 and 0 < by < self.height - 1:
                    frame[by, bx] = 1

    def _score_text(self) -> str:
        return f"{self.score[0]}:{self.score[1]}"

    def get_frame(self, pose_results: Any = None) -> Frame:
        """Step the match and render the arena. ``pose_results`` is unused."""
        now = time.time()
        dt = min(self.MAX_DT, max(0.0, now - self._last_frame_time))
        self._last_frame_time = now

        self._update_match(now, dt)

        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        self._draw_border(frame)
        if self.round_over_at is not None:
            # Between-round sequence: explosion -> score -> field swap.
            self._draw_round_transition(frame, now)
        else:
            self._draw_walls(frame)
            for tank in self.tanks:
                if tank["alive"]:
                    self._draw_tank_maybe_spawning(frame, tank, now)
            self._draw_shells(frame)
        self._draw_fx(frame, now)

        if self.winner is not None:
            self._draw_game_over(frame, now)
            return frame

        if now - self._start_time < self.MODE_NAME_TIME:
            frame[:7, :] = 0
            text.write(frame, "TANK", x=1, y=1, size=5, style="regular")

        return frame

    WIN_ICON_HALF = 2  # bigger than the in-play TANK_HALF -- this glyph stands alone

    def _draw_win_icon(self, frame: np.ndarray, cx: int, cy: int, filled: bool, half: int) -> None:
        """Draw the winner glyph in the style tanks are drawn in play
        (:meth:`_draw_tank`), scaled to ``half``: a solid body (left) or a
        one-pixel outline with a hollow interior (right) -- with a 3px
        cannon, here pointing up."""
        r0, r1, c0, c1 = cy - half, cy + half, cx - half, cx + half
        if filled:
            frame[r0 : r1 + 1, c0 : c1 + 1] = 1
        else:
            frame[r0, c0 : c1 + 1] = 1
            frame[r1, c0 : c1 + 1] = 1
            frame[r0 : r1 + 1, c0] = 1
            frame[r0 : r1 + 1, c1] = 1
        for step in (half + 1, half + 2, half + 3):
            by = cy - step
            if 0 <= by < self.height:
                frame[by, cx] = 1

    def _win_layout(self) -> tuple[int, int]:
        """X of the winner icon's centre and of the WINS text on the settled
        Game Over screen (shared by the screen and the salute animation)."""
        half = self.WIN_ICON_HALF
        icon_w, gap = 2 * half + 1, 2
        text_w = text.width("WINS", size=6, style="regular")
        start_x = max(0, (self.width - (icon_w + gap + text_w)) // 2)
        return start_x + half, start_x + icon_w + gap

    def _render_game_over_screen(self, frame: np.ndarray) -> None:
        """Compose the settled Game Over screen: ``[icon] WINS`` over the score."""
        frame[:, :] = 0
        icon_cx, text_x = self._win_layout()
        self._draw_win_icon(frame, icon_cx, 10, filled=(self.winner == 0), half=self.WIN_ICON_HALF)
        text.write(frame, "WINS", x=text_x, y=6, size=6, style="regular")
        text.write_centered(frame, self._score_text(), y=15, size=6, style="regular")

    def _draw_win_salute(self, frame: np.ndarray, t: float) -> None:
        """The winner takes a victory lap: its icon fires a salute up from its
        cannon on a steady period, each shot rising to just under the top
        border and popping into the plus-shaped ricochet-spark burst."""
        icon_cx, _ = self._win_layout()
        q = (t % self.WIN_FIRE_PERIOD) / self.WIN_FIRE_PERIOD
        muzzle_y, apex_y = 4, 2  # from just above the cannon to just under the border
        if q < 0.25:  # the shot rises from the muzzle to the apex...
            y = muzzle_y - round(q / 0.25 * (muzzle_y - apex_y))
            frame[y, icon_cx] = 1
        elif q < 0.5:  # ...and pops: a plus-shaped burst collapsing to a dot
            frame[apex_y, icon_cx] = 1
            if q < 0.4:
                frame[apex_y - 1, icon_cx] = 1
                frame[apex_y + 1, icon_cx] = 1
                frame[apex_y, icon_cx - 1] = 1
                frame[apex_y, icon_cx + 1] = 1

    def _draw_game_over(self, frame: np.ndarray, now: float) -> None:
        """The winning sequence: an expanding blast at the deciding hit that
        fades (dithered dissolve) into the settled Game Over screen."""
        elapsed = now - (self.win_time if self.win_time is not None else now)
        if elapsed < self.WIN_BLAST_GROW:
            # Grow phase: a filled disc engulfs the frozen arena behind a
            # leading shockwave ring -- the round blast's signature, writ large.
            rad = self.WIN_BLAST_MAX_RADIUS * (elapsed / self.WIN_BLAST_GROW)
            cx, cy = int(round(self.win_x or 0)), int(round(self.win_y or 0))
            yy, xx = np.ogrid[: self.height, : self.width]
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            frame[d2 <= rad**2] = 1
            r1 = rad + 3
            frame[(d2 >= (r1 - 1) ** 2) & (d2 <= (r1 + 1) ** 2)] = 1
        elif elapsed < self.WIN_BLAST_GROW + self.WIN_BLAST_FADE and self._win_noise is not None:
            # Fade phase: burn the white blast away to reveal the Game Over screen.
            target = np.zeros_like(frame)
            self._render_game_over_screen(target)
            p = (elapsed - self.WIN_BLAST_GROW) / self.WIN_BLAST_FADE
            frame[:, :] = np.where(self._win_noise < p, target, 1)
        else:
            self._render_game_over_screen(frame)
            self._draw_win_salute(frame, elapsed - self.WIN_BLAST_GROW - self.WIN_BLAST_FADE)
