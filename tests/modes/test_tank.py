import math

import numpy as np
import pytest

from app.core.mode_manager import ModeManager
from app.modes.contracts import ModeRegistry, RenderContext
from app.modes.tank import HEADINGS, Tank


@pytest.fixture
def tank():
    return Tank(28, 28, ModeManager())


def _aim(tank_obj, side, target):
    """Point a tank's heading straight at a pixel target."""
    t = tank_obj.tanks[side]
    t["heading"] = Tank._heading_to(t["pos"], target)


def test_get_frame_is_valid_bitmap(tank):
    frame = tank.get_frame(None)
    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame)).issubset({0, 1})


def test_drive_moves_tank_along_pressed_direction(tank):
    t = tank.tanks[0]
    t["pos"] = [4.0, 4.0]  # open corner, clear of every wall block
    t["last_input_time"] = 1000.0  # mark controller-active so AI stays off
    tank.set_controller_input("left", move_x=1, move_y=0)  # drive right
    start_x = t["pos"][0]
    now = 1000.0
    for _ in range(20):
        now += 0.05
        t["last_input_time"] = now
        tank.set_controller_input("left", move_x=1, move_y=0)
        tank._update_match(now, 0.05)
    assert t["heading"] == 0  # snapped to the +x cardinal direction
    assert t["pos"][0] > start_x + 1.0
    assert abs(t["pos"][1] - 4.0) < 0.5  # stayed on its row


def test_tanks_cannot_overlap(tank):
    # Two tanks on the same open row; drive the left one straight at the right.
    mover, blocker = tank.tanks[0], tank.tanks[1]
    mover["pos"] = [10.0, 24.0]
    blocker["pos"] = [16.0, 24.0]
    now = 2500.0
    for _ in range(60):
        now += 0.05
        # Mover pushes right; blocker stays put (idle but controller-active).
        mover["last_input_time"] = now
        blocker["last_input_time"] = now
        tank.set_controller_input("left", move_x=1, move_y=0)
        tank.set_controller_input("right", move_x=0, move_y=0)
        tank._update_match(now, 0.05)
        # The two 3x3 footprints must never share a pixel.
        dx = abs(round(mover["pos"][0]) - round(blocker["pos"][0]))
        dy = abs(round(mover["pos"][1]) - round(blocker["pos"][1]))
        assert dx > 2 * tank.TANK_HALF or dy > 2 * tank.TANK_HALF
    # It drove up against the blocker but stopped short of it.
    assert mover["pos"][0] < blocker["pos"][0]


def test_wall_blocks_movement(tank):
    # Place a tank just left of the centre wall block and drive into it.
    r0, r1, c0, c1 = tank.walls[2]
    t = tank.tanks[0]
    t["pos"] = [float(c0 - 2), float((r0 + r1) / 2)]
    t["heading"] = 0  # +x, toward the wall
    now = 2000.0
    for _ in range(40):
        now += 0.05
        t["last_input_time"] = now
        tank.set_controller_input("left", move_x=1, move_y=0)
        tank._update_match(now, 0.05)
        # The tank footprint must never enter the wall.
        assert not tank._tank_blocked(t["pos"][0], t["pos"][1])
    # It pressed up against the wall but did not pass through it.
    assert t["pos"][0] < c0


def test_shell_hits_opponent_and_scores(tank):
    now = 3000.0
    shooter, victim = tank.tanks[1], tank.tanks[0]
    # Open row clear of every wall; freeze both tanks so only the shell moves.
    victim["pos"] = [6.0, 24.0]
    shooter["pos"] = [12.0, 24.0]
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controller -> AI stays off, tanks hold still
    # A shell two pixels right of the victim, moving left into it.
    tank.shells = [{"pos": [8.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 3, "owner": 1}]
    tank._update_match(now, 0.05)
    assert tank.score[1] == 1
    assert victim["alive"] is False
    assert tank.shells == []  # the shell was consumed by the hit


def test_point_resets_both_tanks_to_their_start_positions(tank):
    now = 3000.0
    # Record where each tank starts a round, then drag both well away from it
    # onto an open row clear of every wall block.
    starts = [list(t["pos"]) for t in tank.tanks]
    tank.tanks[0]["pos"] = [8.0, 24.0]
    tank.tanks[1]["pos"] = [20.0, 24.0]
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controller -> AI stays off
    tank.shells = [
        # Shell from tank 1 about to strike tank 0, plus a second loose shell.
        {"pos": [10.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 3, "owner": 1},
        {"pos": [4.0, 4.0], "vel": [0.0, tank.SHELL_SPEED], "bounces": 3, "owner": 0},
    ]
    tank._update_match(now, 0.05)

    # The point downs both tanks and clears every shell from the arena.
    assert all(not t["alive"] for t in tank.tanks)
    assert tank.shells == []

    # After the respawn delay both tanks are back at their initial positions.
    later = now + tank.RESPAWN_DELAY + 0.05
    for t in tank.tanks:
        t["last_input_time"] = later
    tank._update_match(later, 0.05)
    assert all(t["alive"] for t in tank.tanks)
    assert [list(t["pos"]) for t in tank.tanks] == starts


def test_shell_bounces_then_expires(tank):
    now = 4000.0
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controller -> AI never fires its own shells
    # Aim a shell straight at the left border so it keeps reflecting in place.
    tank.shells = [{"pos": [1.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 1, "owner": 0}]
    # First reflection: heading flips, one bounce consumed.
    tank._update_match(now, 0.05)
    assert len(tank.shells) == 1
    assert tank.shells[0]["vel"][0] > 0  # now travelling +x
    assert tank.shells[0]["bounces"] == 0
    # Send it back into the border; the next reflection drops below zero -> gone.
    tank.shells[0]["pos"] = [1.0, 24.0]
    tank.shells[0]["vel"] = [-tank.SHELL_SPEED, 0.0]
    tank._update_match(now + 0.05, 0.05)
    assert tank.shells == []


def test_ai_takes_over_only_after_delay(tank):
    t = tank.tanks[0]
    t["pos"] = [14.0, 14.0]
    t["heading"] = 0
    t["intent"] = {"move": (0, 0)}  # idle controller intent

    # Recent input -> controller intent (idle) wins, AI does not steer.
    now = 5000.0
    t["last_input_time"] = now
    assert tank._controller_active(t, now) is True

    # No input past the delay -> AI is allowed to take over.
    later = now + tank.AI_TAKEOVER_DELAY + 1
    assert tank._controller_active(t, later) is False


def test_ai_keeps_its_distance_instead_of_overlapping(tank):
    ai = tank.tanks[0]
    foe_pos = (22.0, 4.0)  # a row clear of every wall block
    ai["pos"] = [4.0, 4.0]
    now = 6000.0
    closest = float("inf")
    for _ in range(200):  # ~10 s of AI driving toward a stationary foe
        # Pin the foe alive and in place each frame so the AI always has the
        # same target and the measured distance stays meaningful.
        foe = tank.tanks[1]
        foe["alive"] = True
        foe["respawn_at"] = None
        foe["pos"] = list(foe_pos)
        foe["last_input_time"] = now  # idle controller -> foe never moves
        # Block the AI's shots so a hit never resets the round mid-approach.
        ai["fire_cd_until"] = float("inf")
        now += 0.05
        tank._update_match(now, 0.05)
        dist = math.hypot(ai["pos"][0] - foe_pos[0], ai["pos"][1] - foe_pos[1])
        closest = min(closest, dist)
    # It approached but never closed in far enough to overlap the foe.
    assert closest >= tank.AI_STANDOFF - 1
    assert closest <= tank.AI_STANDOFF + 3


def test_reaching_win_score_sets_winner_and_restarts(tank):
    now = 6000.0
    tank.score = [tank.WIN_SCORE - 1, 0]
    tank._register_hit(0, 1, now)
    assert tank.winner == 0
    assert tank.win_time == now

    tank.restart_if_game_over()
    assert tank.winner is None
    assert tank.score == [0, 0]


def test_satisfies_renderer_contract(tank):
    registry = ModeRegistry()
    registry.register(ModeManager.MODE_TANK, lambda c: tank.get_frame(c.pose_results))
    ctx = RenderContext(
        frame=np.zeros((28, 28), dtype=np.uint8),
        pose_results=None,
        face_mesh_results=None,
        estimated_distance=None,
        mode_time=0.0,
        panel_width=28,
        panel_height=28,
    )
    out = registry.render(ModeManager.MODE_TANK, ctx)
    assert out.shape == (28, 28)
    assert out.dtype == np.uint8


def test_heading_to_is_in_range(tank):
    for target in [(0, 0), (27, 0), (0, 27), (27, 27), (14, 14)]:
        h = Tank._heading_to((14, 14), target)
        assert 0 <= h < HEADINGS
