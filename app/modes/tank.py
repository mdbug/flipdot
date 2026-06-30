import math
import time
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
    it over so the mode is playable solo and works as an attract loop.  The AI
    steers toward its opponent along the dominant axis and fires when roughly
    aligned; it is deliberately beatable.

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

    AI_STANDOFF = 18.0  # px the AI keeps between itself and its foe

    SHELL_SPEED = 22.0  # px/s
    SHELL_BOUNCES = 1  # one ricochet, then the shell expires (tight arena)
    FIRE_COOLDOWN = 0.6  # seconds between a tank's shots
    MAX_SHELLS_PER_TANK = 2

    RESPAWN_DELAY = 1.2  # seconds both tanks stay down between rounds
    MODE_NAME_TIME = 2.0
    WIN_RESTART_TIME = 8.0

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

    def _make_walls(self) -> list[tuple[int, int, int, int]]:
        """Interior wall blocks as inclusive ``(r0, r1, c0, c1)`` rects."""
        h, w = self.height, self.width
        return [
            (6, 7, w // 2 - 3, w // 2 + 2),  # top bar
            (h - 8, h - 7, w // 2 - 3, w // 2 + 2),  # bottom bar
            (h // 2 - 2, h // 2 + 1, w // 2 - 2, w // 2 + 1),  # centre block
        ]

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
            "respawn_at": None,
            "fire_cd_until": 0.0,
            "ai_strafe": 1,  # AI sidestep direction when its shot is walled off
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

    def _tank_blocked(self, cx: float, cy: float) -> bool:
        """True if a tank centred at ``(cx, cy)`` would overlap a wall/border."""
        xi, yi = int(round(cx)), int(round(cy))
        fr0, fr1 = yi - self.TANK_HALF, yi + self.TANK_HALF
        fc0, fc1 = xi - self.TANK_HALF, xi + self.TANK_HALF
        if fc0 < 1 or fc1 > self.width - 2 or fr0 < 1 or fr1 > self.height - 2:
            return True
        for r0, r1, c0, c1 in self.walls:
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

    def _ai_intent(self, idx: int, now: float) -> dict:
        """Beatable AI: line up on the opponent's row/column, then hold range.

        The AI closes the gap on the axis it is *nearest* aligned on (so it
        ends up sharing a row or column with its foe) while keeping at least
        ``AI_STANDOFF`` pixels on the firing axis, so it no longer drives
        straight in and overlaps the opponent.
        """
        tank = self.tanks[idx]
        foe = self.tanks[1 - idx]
        if not foe["alive"]:
            return {"move": (0, 0)}
        dx = foe["pos"][0] - tank["pos"][0]
        dy = foe["pos"][1] - tank["pos"][1]
        adx, ady = abs(dx), abs(dy)

        # Pick the firing axis: the one the foe is further away on. The other
        # (smaller) offset is what we close to share a row/column.
        if adx >= ady:
            face = (1 if dx > 0 else -1, 0)
            along, perp_offset = adx, ady
            align_move = (0, 1 if dy > 0 else -1)
        else:
            face = (0, 1 if dy > 0 else -1)
            along, perp_offset = ady, adx
            align_move = (1 if dx > 0 else -1, 0)

        # Stop advancing early enough that coasting halts at the standoff
        # rather than ploughing through it into the foe.
        brake_dist = tank["vel"] ** 2 / (2 * self.FRICTION)
        if perp_offset > 1:
            move = align_move  # still getting onto the foe's line
        elif along - brake_dist > self.AI_STANDOFF:
            move = face  # aligned but too far -> close to standoff range
        else:
            move = (0, 0)  # at range and lined up -> hold and shoot

        # When lined up, aim straight at the foe. Fire on a random hold if the
        # lane is open; otherwise a wall is between them, so sidestep to hunt
        # for a clear shot instead of freezing in a stalemate.
        if perp_offset <= 1:
            tank["heading"] = _MOVE_HEADINGS[face]
            tip_x = tank["pos"][0] + face[0] * (self.TANK_HALF + 1)
            tip_y = tank["pos"][1] + face[1] * (self.TANK_HALF + 1)
            if self._point_blocked(tip_x, tip_y):
                strafe = (
                    (0, tank["ai_strafe"]) if face[1] == 0 else (tank["ai_strafe"], 0)
                )
                nx = tank["pos"][0] + strafe[0]
                ny = tank["pos"][1] + strafe[1]
                if self._tank_blocked(nx, ny):  # wall ahead -> reverse the sidestep
                    tank["ai_strafe"] = -tank["ai_strafe"]
                    strafe = (-strafe[0], -strafe[1])
                move = strafe
            elif self.rng.random() < 0.25:
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

        for idx, tank in enumerate(self.tanks):
            if not tank["alive"]:
                if tank["respawn_at"] is not None and now >= tank["respawn_at"]:
                    self.tanks[idx] = self._spawn(idx)
                    self.tanks[idx]["last_input_time"] = tank["last_input_time"]
                continue
            intent = (
                tank["intent"] if self._controller_active(tank, now) else self._ai_intent(idx, now)
            )
            self._drive_tank(tank, intent, now, dt)

        self._update_shells(now, dt)

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

    def _update_shells(self, now: float, dt: float) -> None:
        survivors: list[dict] = []
        for shell in self.shells:
            x, y = shell["pos"]
            vx, vy = shell["vel"]
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
        self.fx.append(
            {"x": vtank["pos"][0], "y": vtank["pos"][1], "start": now, "until": now + 0.5}
        )
        # A scored point ends the round: both tanks go down briefly and respawn
        # at their starting positions (the caller clears any in-flight shells).
        for tank in self.tanks:
            tank["alive"] = False
            tank["respawn_at"] = now + self.RESPAWN_DELAY
        if self.score[shooter] >= self.WIN_SCORE:
            self.winner = shooter
            self.win_time = now

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
        self._draw_walls(frame)
        for tank in self.tanks:
            if tank["alive"]:
                self._draw_tank(frame, tank)
        self._draw_shells(frame)
        self._draw_fx(frame, now)

        if self.winner is None and now - self._start_time < self.MODE_NAME_TIME:
            frame[:7, :] = 0
            text.write(frame, "TANK", x=1, y=1, size=5, style="regular")

        if self.winner is not None:
            frame[:, :] = 0
            msg = "LEFT WINS" if self.winner == 0 else "RIGHT WINS"
            text.write_centered(frame, msg, y=6, size=5, style="regular")
            text.write_centered(frame, self._score_text(), y=15, size=6, style="regular")

        return frame
