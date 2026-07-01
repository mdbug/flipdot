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

# The four cardinal move directions map to a subset of the headings above.
# Keys are unit (dx, dy) vectors with y pointing down.
_MOVE_HEADINGS = {
    (1, 0): 0,  # right
    (0, 1): 4,  # down
    (-1, 0): 8,  # left
    (0, -1): 12,  # up
}


class Tank:
    """Two-tank combat in the style of Atari *Combat*.

    Each tank moves and aims in one of four cardinal directions with the D-pad
    and fires shells with Button A.  Shells ricochet off the arena walls and
    border a limited number of times before expiring, so bank shots score.

    Controls
    --------
    * Primary controller  → right tank (side ``1``)
    * Secondary controller → left tank (side ``0``)
    * D-Up/Down/Left/Right → drive and aim in that direction
    * Button A             → fire (and restart once a match is won)

    A held D-pad direction snaps the tank's heading to that cardinal direction
    and drives it forward; the gun always points the way the tank last moved.
    Tanks are solid: neither can drive through the other, so they never overlap.

    When a side provides no input for ``AI_TAKEOVER_DELAY`` seconds, an AI takes
    it over so the mode is playable solo and works as an attract loop.  Each
    frame the AI prioritises dodging an incoming shell, then maneuvers into a
    clear firing slot at standoff range (without driving into a shell's path),
    then fires only down an unobstructed lane -- occasionally leading a moving
    foe.  Reaction delay, missed shots and the odd fluffed dodge keep it
    deliberately beatable.

    Tanks are told apart on the monochrome panel by shape: the left tank is a
    solid 3x3 block, the right tank a hollow 3x3 ring.
    """

    AI_TAKEOVER_DELAY = 15.0
    WIN_SCORE = 5
    MAX_DT = 0.05  # clamp dt to avoid tunneling fast shells

    TANK_HALF = 1  # tank is (2*TANK_HALF+1) px square -> 3x3
    THRUST_ACCEL = 28.0  # px/s^2
    FRICTION = 22.0  # px/s^2 deceleration when coasting
    MAX_SPEED = 12.0  # px/s

    AI_STANDOFF = 12.0  # px the AI keeps between itself and its foe
    AI_STANDOFF_BAND = 3.0  # px dead band around standoff so the AI settles
    AI_AXIS_MARGIN = 3.0  # px the other axis must win by before the AI reorients
    AI_ALIGN_TOL = 1  # px slack for counting as sharing a row/column with foe
    AI_DODGE_HORIZON = 0.9  # s of shell look-ahead used to detect a threat
    AI_DODGE_COMMIT = 0.25  # s a chosen dodge is held before re-deciding
    AI_REACT_MIN = 0.12  # s reaction delay floor (sharp AI)
    AI_REACT_MAX = 0.45  # s reaction delay ceiling (sloppy AI)
    AI_FIRE_CHANCE = 0.22  # per-eligible-frame chance to take an open shot
    AI_REACQUIRE_GAP = 0.4  # s a shot must be lost before the reflex delay re-arms
    AI_DODGE_MISS = 0.12  # chance the AI fails to react to a given threat
    AI_LEAD_CHANCE = 0.25  # chance to lead a moving foe instead of direct aim
    AI_STALL_TIME = 3.5  # s with no available shot before the AI wanders to break a stalemate
    AI_WANDER_TIME = 1.0  # s a random-wander burst lasts once a stalemate is detected
    AI_WANDER_COMMIT = 0.3  # s each random wander direction is held before repicking

    SHELL_SPEED = 22.0  # px/s
    SHELL_BOUNCES = 1  # one ricochet, then the shell expires (tight arena)
    FIRE_COOLDOWN = 0.6  # seconds between a tank's shots
    MAX_SHELLS_PER_TANK = 2

    # Between-round sequence, staged in order: the hit blast plays over the old
    # field, the score is shown, the old field then dissolves into a fresh one,
    # and finally the tanks materialise at their corners.
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
        3px-wide corridor open around the whole border.  A safety check enforces
        clear spawn pockets and spawn-to-spawn connectivity; on the rare miss it
        retries, falling back to the classic layout.
        """
        templates = (
            self._tpl_classic,
            self._tpl_pillar,
            self._tpl_diagonal,
            self._tpl_cross,
            self._tpl_stagger,
        )
        for _ in range(20):
            seed = templates[int(self.rng.integers(len(templates)))]()
            rects = self._symmetrize(seed)
            if self._layout_is_safe(rects):
                return rects
        return self._default_walls()

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
        """A horizontal bar (mirror adds the opposite one) plus a centre block."""
        w = self.width
        row = int(self.rng.integers(5, 8))
        half = int(self.rng.integers(2, 4))
        bar = (row, row + 1, w // 2 - half - 1, w // 2 + half)  # centred -> mirrors cleanly
        core = self._centred_block(int(self.rng.integers(1, 3)))
        return [bar, core]

    def _tpl_pillar(self) -> list[tuple[int, int, int, int]]:
        """A single jittered central pillar."""
        return [self._centred_block(int(self.rng.integers(1, 4)))]

    def _tpl_diagonal(self) -> list[tuple[int, int, int, int]]:
        """A block off the upper-left of centre; the mirror opposes it."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        off = int(self.rng.integers(3, 5))
        size = int(self.rng.integers(2, 4))
        return [(cy - off - size, cy - off, cx - off - size, cx - off)]

    def _tpl_cross(self) -> list[tuple[int, int, int, int]]:
        """A centred plus/cross with open ends so lanes stay clear."""
        h, w = self.height, self.width
        arm = int(self.rng.integers(4, 8))
        vbar = (h // 2 - arm, h // 2 - 1 + arm, w // 2 - 1, w // 2)
        hbar = (h // 2 - 1, h // 2, w // 2 - arm, w // 2 - 1 + arm)
        return [vbar, hbar]

    def _tpl_stagger(self) -> list[tuple[int, int, int, int]]:
        """A short bar above-left of centre; the mirror makes a staggered Z."""
        h, w = self.height, self.width
        cy, cx = h // 2, w // 2
        off = int(self.rng.integers(2, 5))
        length = int(self.rng.integers(4, 8))
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
            "ai_strafe": 1,  # AI sidestep direction when its shot is walled off
            "ai_react_at": 0.0,  # earliest time the AI may take a lined-up shot
            "ai_shot_seen": 0.0,  # last time a shot was available (target tracking)
            "ai_face_axis": 0,  # firing axis (0=x, 1=y) held with hysteresis
            "ai_dodge_until": 0.0,  # commit a dodge for a few frames once chosen
            "ai_dodge_move": (0, 0),
            "ai_dodging": False,  # keep clearing the lane until the shell passes
            "ai_stall_since": None,  # first time (lazy) since a shot was last available
            "ai_wander_until": 0.0,  # commit a stalemate-breaking wander for a burst
            "ai_wander_move": (0, 0),
            "ai_wander_repick_at": 0.0,  # repick the random wander direction after this
            # Per-spawn competence factor (0=sharp, 1=sloppy): scales reaction
            # delay and miss chance so the AI is not perfect and varies by round.
            "ai_skill": float(self.rng.uniform(0.0, 1.0)),
            "intent": {"move": (0, 0)},
            "last_input_time": None,
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
        Movement is restricted to the four cardinal directions, so a diagonal
        press is resolved to the horizontal axis.  The AI takeover timer only
        resets while there is actual input, so an idle controller still hands
        the tank to the AI after the delay.
        """
        mx = int(max(-1, min(1, move_x)))
        my = int(max(-1, min(1, move_y)))
        if mx != 0:
            my = 0  # keep movement to a single cardinal direction
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

    def _clear_lane(self, idx: int, direction: tuple[int, int], tcx: int, tcy: int) -> bool:
        """True if a shell fired now from ``idx`` straight along ``direction``
        reaches the footprint at ``(tcx, tcy)`` with no wall/border in between.

        Direct shots only -- the trace stops at the first blocked cell, so the
        AI never relies on a ricochet to land a hit.
        """
        tank = self.tanks[idx]
        dx, dy = direction
        x = tank["pos"][0] + dx * (self.TANK_HALF + 1)
        y = tank["pos"][1] + dy * (self.TANK_HALF + 1)
        for _ in range(self.width + self.height):
            xi, yi = int(round(x)), int(round(y))
            if abs(xi - tcx) <= self.TANK_HALF and abs(yi - tcy) <= self.TANK_HALF:
                return True
            if self._point_blocked(x, y):
                return False
            x += dx
            y += dy
        return False

    def _intercepts(self, tank: dict, direction: tuple[int, int], foe: dict) -> bool:
        """True if a shell fired now along ``direction`` would strike the foe
        *while it keeps moving* -- i.e. a lead shot into the foe's path.

        The foe is extrapolated along its current heading; the shell is stepped
        with the real physics and the trace bails at the first bounce so this,
        too, stays a direct (un-banked) shot.
        """
        if foe["vel"] < 2.0:
            return False
        dx, dy = direction
        sx = tank["pos"][0] + dx * (self.TANK_HALF + 1)
        sy = tank["pos"][1] + dy * (self.TANK_HALF + 1)
        if self._point_blocked(sx, sy):
            return False
        vx, vy = dx * self.SHELL_SPEED, dy * self.SHELL_SPEED
        fhx, fhy = _HEADING_VECS[foe["heading"]]
        fvx, fvy = fhx * foe["vel"], fhy * foe["vel"]
        fx, fy = foe["pos"]
        t, step = 0.0, 0.04
        while t < 1.5:
            sx, sy, vx, vy, bounced = self._advance_shell(sx, sy, vx, vy, step)
            if bounced:
                return False
            fx += fvx * step
            fy += fvy * step
            if (
                abs(round(sx) - round(fx)) <= self.TANK_HALF
                and abs(round(sy) - round(fy)) <= self.TANK_HALF
            ):
                return True
            t += step
        return False

    def _move_lands_in_danger(self, tank: dict, move: tuple[int, int], now: float) -> bool:
        """True if stepping ``tank`` by ``move`` puts its footprint onto the
        near-future path of any enemy shell -- used so the AI won't *drive into*
        a bullet's trajectory while repositioning."""
        nx = int(round(tank["pos"][0] + move[0]))
        ny = int(round(tank["pos"][1] + move[1]))
        reach = self.TANK_HALF + 1
        for shell in self.shells:
            if shell["owner"] == tank["side"]:
                continue
            for sx, sy, _t in self._project_shell_cells(shell, self.AI_DODGE_HORIZON):
                if abs(sx - nx) <= reach and abs(sy - ny) <= reach:
                    return True
        return False

    def _incoming_threat(self, tank: dict) -> tuple[dict | None, float]:
        """The first enemy shell whose predicted path strikes ``tank``'s
        footprint within the dodge horizon, plus its time-to-impact."""
        cx, cy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
        for shell in self.shells:
            if shell["owner"] == tank["side"]:
                continue
            for sx, sy, t in self._project_shell_cells(shell, self.AI_DODGE_HORIZON):
                if abs(sx - cx) <= self.TANK_HALF and abs(sy - cy) <= self.TANK_HALF:
                    return shell, t
        return None, 0.0

    def _ai_dodge(self, tank: dict, now: float, react: float) -> tuple[int, int] | None:
        """Return a dodge move if an enemy shell is about to hit us, else None.

        The reaction delay (and the occasional missed dodge) gate only the
        *first* reaction to a fresh threat -- once committed, the AI keeps
        clearing the lane until the shell no longer threatens it, rather than
        drifting back into its path when the shell gets too close to "react" to.
        """
        threat, impact_t = self._incoming_threat(tank)
        if threat is None:
            tank["ai_dodging"] = False
            return None
        if not tank["ai_dodging"]:
            if impact_t < react:
                return None  # spotted too late to do anything about it
            if self.rng.random() < self.AI_DODGE_MISS * (0.5 + tank["ai_skill"]):
                return None  # occasional lapse so the AI is not an unhittable wall
            tank["ai_dodging"] = True

        # Hold the committed sidestep until it expires, then pick a fresh one.
        move = tank["ai_dodge_move"]
        if (
            now < tank["ai_dodge_until"]
            and move != (0, 0)
            and not self._move_blocked(tank, tank["pos"][0] + move[0], tank["pos"][1] + move[1])
        ):
            return move
        vx, vy = threat["vel"]
        if abs(vx) >= abs(vy):  # shell runs horizontally -> sidestep vertically
            options = [(0, tank["ai_strafe"]), (0, -tank["ai_strafe"])]
        else:
            options = [(tank["ai_strafe"], 0), (-tank["ai_strafe"], 0)]
        for mv in options:
            if not self._move_blocked(tank, tank["pos"][0] + mv[0], tank["pos"][1] + mv[1]):
                tank["ai_strafe"] = mv[0] + mv[1]  # remember the side that worked
                tank["ai_dodge_move"] = mv
                tank["ai_dodge_until"] = now + self.AI_DODGE_COMMIT
                return mv
        return None

    def _ai_wander_move(self, tank: dict, now: float) -> tuple[int, int]:
        """A random, validated cardinal step for a stalemate-breaking wander.

        Holds the committed direction for ``AI_WANDER_COMMIT`` then repicks, like
        the dodge sidestep.  Only ever chosen moves that clear walls, the foe and
        any live shell path, so the wander can't drive into trouble.
        """
        move = tank["ai_wander_move"]
        if (
            now < tank["ai_wander_repick_at"]
            and move != (0, 0)
            and not self._move_blocked(tank, tank["pos"][0] + move[0], tank["pos"][1] + move[1])
            and not self._move_lands_in_danger(tank, move, now)
        ):
            return move
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.rng.shuffle(dirs)
        for mv in dirs:
            if not self._move_blocked(
                tank, tank["pos"][0] + mv[0], tank["pos"][1] + mv[1]
            ) and not self._move_lands_in_danger(tank, mv, now):
                tank["ai_wander_move"] = mv
                tank["ai_wander_repick_at"] = now + self.AI_WANDER_COMMIT
                return mv
        return (0, 0)

    def _ai_intent(self, idx: int, now: float) -> dict:
        """Threat-aware, beatable AI.

        Priorities each frame: (A) dodge an incoming shell, otherwise
        (B) maneuver into a clear firing slot at standoff range without driving
        into bullet paths, and (C) fire only down a clear lane -- occasionally
        leading a moving foe -- after a human-like reaction delay.
        """
        tank = self.tanks[idx]
        foe = self.tanks[1 - idx]
        if not foe["alive"]:
            return {"move": (0, 0)}

        react = self.AI_REACT_MIN + tank["ai_skill"] * (self.AI_REACT_MAX - self.AI_REACT_MIN)

        # (A) Dodge takes precedence over everything else.
        dodge = self._ai_dodge(tank, now, react)
        if dodge is not None:
            return {"move": dodge}

        dx = foe["pos"][0] - tank["pos"][0]
        dy = foe["pos"][1] - tank["pos"][1]
        adx, ady = abs(dx), abs(dy)
        fcx, fcy = int(round(foe["pos"][0])), int(round(foe["pos"][1]))

        # (C) Work out whether we have a shot. A direct shot needs the foe on
        # our row/column with a clear lane; a lead shot fires into its path.
        shot = None
        if ady <= self.AI_ALIGN_TOL:
            d = (1 if dx > 0 else -1, 0)
            if self._clear_lane(idx, d, fcx, fcy):
                shot = d
        if shot is None and adx <= self.AI_ALIGN_TOL:
            d = (0, 1 if dy > 0 else -1)
            if self._clear_lane(idx, d, fcx, fcy):
                shot = d
        # A lead shot is only considered when there is no clean direct shot, so
        # the gun doesn't flicker off a good aim onto a speculative one.
        if shot is None and self.rng.random() < self.AI_LEAD_CHANCE:
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if self._intercepts(tank, d, foe):
                    shot = d
                    break

        # (B) Positioning. Pick the firing axis with hysteresis so the AI does
        # not flip 90 degrees every frame when the foe is near the diagonal.
        if adx >= ady + self.AI_AXIS_MARGIN:
            tank["ai_face_axis"] = 0
        elif ady >= adx + self.AI_AXIS_MARGIN:
            tank["ai_face_axis"] = 1
        if tank["ai_face_axis"] == 0:
            face = (1 if dx > 0 else -1, 0)
            along, perp_offset = adx, ady
            align_move = (0, 1 if dy > 0 else -1)
        else:
            face = (0, 1 if dy > 0 else -1)
            along, perp_offset = ady, adx
            align_move = (1 if dx > 0 else -1, 0)

        # Coast onto the firing line and into standoff range rather than thrusting
        # past and oscillating: stop driving once braking distance will carry us
        # the rest of the way, and hold within a dead band around the standoff.
        brake = tank["vel"] ** 2 / (2 * self.FRICTION)
        if perp_offset - brake > self.AI_ALIGN_TOL:
            move = align_move  # still getting onto the foe's line
        elif along - brake > self.AI_STANDOFF + self.AI_STANDOFF_BAND:
            move = face  # lined up but too far -> close to standoff range
        elif along + brake < self.AI_STANDOFF - self.AI_STANDOFF_BAND:
            move = (-face[0], -face[1])  # crowding the foe -> back off
        else:
            move = (0, 0)  # in the slot -> hold and look for a shot

        # Aligned but no clear lane -> a wall is between us; sidestep to hunt for
        # an opening instead of freezing in a stalemate.
        if perp_offset <= self.AI_ALIGN_TOL and shot is None:
            strafe = (0, tank["ai_strafe"]) if face[1] == 0 else (tank["ai_strafe"], 0)
            if self._move_blocked(tank, tank["pos"][0] + strafe[0], tank["pos"][1] + strafe[1]):
                tank["ai_strafe"] = -tank["ai_strafe"]
                strafe = (-strafe[0], -strafe[1])
            move = strafe

        # Don't drive into the path of an in-flight shell.
        if move != (0, 0) and self._move_lands_in_danger(tank, move, now):
            for alt in ((-move[0], -move[1]), (move[1], move[0]), (-move[1], -move[0]), (0, 0)):
                blocked = alt != (0, 0) and self._move_blocked(
                    tank, tank["pos"][0] + alt[0], tank["pos"][1] + alt[1]
                )
                if not blocked and not self._move_lands_in_danger(tank, alt, now):
                    move = alt
                    break

        # Stalemate breaker (mainly for AI vs AI): two mirrored tanks can strafe
        # in lockstep around a wall forever, never opening a lane. If no shot has
        # been available for a while, wander in a random direction for a short
        # burst to break the symmetry so one tank slips off the deadlock.
        if tank["ai_stall_since"] is None:
            tank["ai_stall_since"] = now
        if shot is not None:
            tank["ai_stall_since"] = now
            tank["ai_wander_until"] = 0.0
        elif now < tank["ai_wander_until"]:
            move = self._ai_wander_move(tank, now)
        elif now - tank["ai_stall_since"] > self.AI_STALL_TIME:
            tank["ai_wander_until"] = now + self.AI_WANDER_TIME
            tank["ai_stall_since"] = now
            move = self._ai_wander_move(tank, now)

        # (C) Fire: face the shot and pull the trigger after a reaction delay and
        # a random hold, so the AI misses some open shots and is beatable. The
        # reflex timer only re-arms when the target was genuinely lost (not on
        # the frame-to-frame flicker of the lane as both tanks drift), otherwise
        # the AI could almost never satisfy the delay and would rarely fire.
        if shot is not None:
            tank["heading"] = _MOVE_HEADINGS[shot]
            if now - tank["ai_shot_seen"] > self.AI_REACQUIRE_GAP:
                tank["ai_react_at"] = now + react
            tank["ai_shot_seen"] = now
            if now >= tank["ai_react_at"] and self.rng.random() < self.AI_FIRE_CHANCE:
                self._spawn_shell(idx, now)
        return {"move": move}

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
            # Snap heading to the pressed cardinal direction and drive forward.
            tank["heading"] = _MOVE_HEADINGS[move]
            tank["vel"] += self.THRUST_ACCEL * dt
        else:
            tank["vel"] = max(0.0, tank["vel"] - self.FRICTION * dt)
        tank["vel"] = min(self.MAX_SPEED, tank["vel"])

        dx, dy = _HEADING_VECS[tank["heading"]]
        x, y = tank["pos"]
        nx, ny = x + dx * tank["vel"] * dt, y + dy * tank["vel"] * dt
        moved = False
        if not self._move_blocked(tank, nx, ny):
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

    def _spawn_shell(self, idx: int, now: float) -> None:
        tank = self.tanks[idx]
        if not tank["alive"] or now < tank["fire_cd_until"]:
            return
        if sum(1 for s in self.shells if s["owner"] == idx) >= self.MAX_SHELLS_PER_TANK:
            return
        dx, dy = _HEADING_VECS[tank["heading"]]
        x, y = tank["pos"]
        tip = [x + dx * (self.TANK_HALF + 1), y + dy * (self.TANK_HALF + 1)]
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
        xi, yi = int(round(shell["pos"][0])), int(round(shell["pos"][1]))
        for idx, tank in enumerate(self.tanks):
            if idx == shell["owner"] or not tank["alive"]:
                continue
            cx, cy = int(round(tank["pos"][0])), int(round(tank["pos"][1]))
            if abs(xi - cx) <= self.TANK_HALF and abs(yi - cy) <= self.TANK_HALF:
                self._register_hit(shell["owner"], idx, now)
                return True
        return False

    def _register_hit(self, shooter: int, victim: int, now: float) -> None:
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
                {"x": vtank["pos"][0], "y": vtank["pos"][1], "start": now, "until": now + 0.5}
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
        for step in (h + 1, h + 2):
            bx, by = int(round(cx + dx * step)), int(round(cy + dy * step))
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
            rad = 1 + int(progress * 3)
            cx, cy = int(round(f["x"])), int(round(f["y"]))
            yy, xx = np.ogrid[: self.height, : self.width]
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            frame[(d2 >= (rad - 1) ** 2) & (d2 <= (rad + 1) ** 2)] = 1

    def _walls_mask(self, walls: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Rasterize a list of wall rects into a 0/1 pixel mask."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for r0, r1, c0, c1 in walls:
            mask[r0 : r1 + 1, c0 : c1 + 1] = 1
        return mask

    def _draw_round_transition(self, frame: np.ndarray, now: float) -> None:
        """Render the staged between-round sequence: the blast plays over the
        old field, then the score, then the old field dither-dissolves into the
        new one.  The tanks stay hidden until the sequence completes and they
        respawn (:meth:`_begin_new_round`)."""
        t = now - (self.round_over_at or now)
        if t < self.ROUND_EXPLODE:
            self._draw_walls(frame)  # old field under the blast (drawn on top later)
        elif t < self.ROUND_EXPLODE + self.ROUND_SCORE:
            self._draw_walls(frame)
            self._draw_score(frame, (t - self.ROUND_EXPLODE) / self.ROUND_SCORE)
        else:
            p = (t - self.ROUND_EXPLODE - self.ROUND_SCORE) / self.ROUND_SWAP
            self._draw_field_morph(frame, min(1.0, p))

    def _draw_score(self, frame: np.ndarray, progress: float) -> None:
        """Flash the score in a cleared band, dither-revealing it as it fades in."""
        frame[10:18, :] = 0
        text_frame = np.zeros_like(frame)
        text.write_centered(text_frame, self._score_text(), y=11, size=6, style="regular")
        reveal = min(1.0, progress / 0.25)  # fade in over the first quarter, then hold
        if self._round_noise is None:
            frame[text_frame == 1] = 1
        else:
            frame[(text_frame == 1) & (self._round_noise < reveal)] = 1

    def _draw_field_morph(self, frame: np.ndarray, p: float) -> None:
        """Dither-dissolve the old field into the freshly generated one."""
        old = self._walls_mask(self.walls)
        new = self._walls_mask(self.next_walls or self.walls)
        shown = new if self._round_noise is None else np.where(self._round_noise < p, new, old)
        frame[shown == 1] = 1

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

    def _draw_win_icon(self, frame: np.ndarray, cx: int, cy: int, filled: bool) -> None:
        """Draw the winner glyph exactly as tanks are drawn in play
        (:meth:`_draw_tank`): a 3x3 body -- solid (left) or a ring with the
        centre pixel off (right) -- with a 2px cannon, here pointing up."""
        h = self.TANK_HALF
        frame[cy - h : cy + h + 1, cx - h : cx + h + 1] = 1
        if not filled:
            frame[cy, cx] = 0
        for step in (h + 1, h + 2):
            by = cy - step
            if 0 <= by < self.height:
                frame[by, cx] = 1

    def _render_game_over_screen(self, frame: np.ndarray) -> None:
        """Compose the settled Game Over screen: ``[icon] WINS`` over the score."""
        frame[:, :] = 0
        icon_w, gap = 2 * self.TANK_HALF + 1, 2
        text_w = text.width("WINS", size=5, style="regular")
        start_x = max(0, (self.width - (icon_w + gap + text_w)) // 2)
        self._draw_win_icon(frame, start_x + self.TANK_HALF, 9, filled=(self.winner == 0))
        text.write(frame, "WINS", x=start_x + icon_w + gap, y=6, size=5, style="regular")
        text.write_centered(frame, self._score_text(), y=15, size=6, style="regular")

    def _draw_game_over(self, frame: np.ndarray, now: float) -> None:
        """The winning sequence: an expanding blast at the deciding hit that
        fades (dithered dissolve) into the settled Game Over screen."""
        elapsed = now - (self.win_time if self.win_time is not None else now)
        if elapsed < self.WIN_BLAST_GROW:
            # Grow phase: a filled disc engulfs the frozen arena.
            rad = self.WIN_BLAST_MAX_RADIUS * (elapsed / self.WIN_BLAST_GROW)
            cx, cy = int(round(self.win_x or 0)), int(round(self.win_y or 0))
            yy, xx = np.ogrid[: self.height, : self.width]
            frame[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2] = 1
        elif elapsed < self.WIN_BLAST_GROW + self.WIN_BLAST_FADE and self._win_noise is not None:
            # Fade phase: burn the white blast away to reveal the Game Over screen.
            target = np.zeros_like(frame)
            self._render_game_over_screen(target)
            p = (elapsed - self.WIN_BLAST_GROW) / self.WIN_BLAST_FADE
            frame[:, :] = np.where(self._win_noise < p, target, 1)
        else:
            self._render_game_over_screen(frame)
