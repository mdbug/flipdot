import math

import numpy as np
import pytest

import app.services.text as text
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


def _wall_mask(tank_obj):
    """Rasterize the arena's wall rects into a 0/1 pixel mask."""
    mask = np.zeros((tank_obj.height, tank_obj.width), dtype=np.uint8)
    for r0, r1, c0, c1 in tank_obj.walls:
        mask[r0 : r1 + 1, c0 : c1 + 1] = 1
    return mask


def test_get_frame_is_valid_bitmap(tank):
    frame = tank.get_frame(None)
    assert frame.shape == (28, 28)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame)).issubset({0, 1})


def test_drive_moves_tank_along_pressed_direction(tank):
    tank.walls = []  # empty arena: isolate movement physics from random walls
    t = tank.tanks[0]
    t["pos"] = [4.0, 4.0]  # open corner, clear of every wall block
    t["last_input_time"] = 1000.0  # mark controller-active so AI stays off
    tank.set_controller_input("left", move_x=1, move_y=0)  # drive right
    start_x = t["pos"][0]
    now = 1000.0
    for _ in range(20):
        now += 0.05
        t["last_input_time"] = now
        tank.tanks[1]["last_input_time"] = now  # idle controller -> foe AI off
        tank.set_controller_input("left", move_x=1, move_y=0)
        tank._update_match(now, 0.05)
    assert t["heading"] == 0  # snapped to the +x cardinal direction
    assert t["pos"][0] > start_x + 1.0
    assert abs(t["pos"][1] - 4.0) < 0.5  # stayed on its row


def test_tanks_cannot_overlap(tank):
    tank.walls = []  # empty arena so only the tank-vs-tank rule is under test
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
    tank.walls = tank._default_walls()  # known layout: walls[2] is the centre block
    # Place a tank just left of the centre wall block and drive into it.
    r0, r1, c0, c1 = tank.walls[2]
    t = tank.tanks[0]
    t["pos"] = [float(c0 - 2), float((r0 + r1) / 2)]
    t["heading"] = 0  # +x, toward the wall
    now = 2000.0
    for _ in range(40):
        now += 0.05
        t["last_input_time"] = now
        tank.tanks[1]["last_input_time"] = now  # idle controller -> foe AI off
        tank.set_controller_input("left", move_x=1, move_y=0)
        tank._update_match(now, 0.05)
        # The tank footprint must never enter the wall.
        assert not tank._tank_blocked(t["pos"][0], t["pos"][1])
    # It pressed up against the wall but did not pass through it.
    assert t["pos"][0] < c0


def test_shell_hits_opponent_and_scores(tank):
    now = 3000.0
    tank.walls = []  # empty arena so the shell reaches the victim unobstructed
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
    tank.walls = []  # empty arena so the setup row stays clear
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
    tank.walls = []  # empty arena so the shell only reflects off the border
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


class _FakeRng:
    """Deterministic stand-in for the AI's RNG: ``random()`` returns a fixed
    value so the random fire/bank/dodge rolls become predictable in tests."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value

    def uniform(self, lo=0.0, hi=1.0) -> float:
        return lo

    def integers(self, lo, hi=None):
        # Layout generation (after an unexpected hit) picks lows deterministically.
        return 0 if hi is None else lo


def test_lane_clear_sees_open_row_but_not_through_a_wall(tank):
    tank.walls = tank._default_walls()  # known layout: walls[2] is the centre block
    # Open row clear of every wall block: a straight shot reaches the target.
    assert tank._lane_clear(4.0, 4.0, (1, 0), 22, 4) is True

    # Same row, but the centre wall block now sits between the two.
    r0, r1, c0, c1 = tank.walls[2]
    row = (r0 + r1) // 2
    assert tank._lane_clear(float(c0 - 4), float(row), (1, 0), c1 + 4, row) is False


def test_ai_never_fires_through_a_wall(tank):
    now = 6000.0
    tank.walls = tank._default_walls()  # known layout: walls[2] is the centre block
    tank.AI_BANK_CHANCE = 0.0  # direct-shot discipline is under test, not banks
    ai, foe = tank.tanks[0], tank.tanks[1]
    tank.rng = _FakeRng(0.0)  # every fire/lead roll passes -> only the lane gates it
    ai["ai_skill"] = 0.0
    ai["ai_react_at"] = 0.0
    foe["vel"] = 0.0

    # Aligned on a row with the centre wall block directly between them.
    r0, r1, c0, c1 = tank.walls[2]
    row = (r0 + r1) // 2
    ai["pos"] = [float(c0 - 4), float(row)]
    foe["pos"] = [float(c1 + 4), float(row)]
    for _ in range(50):
        now += 0.05
        tank._ai_intent(0, now)
        assert tank.shells == []  # wall in the way -> it holds its fire

    # Clear the lane (open row) and, after the brief reaction delay, it takes the
    # shot it was denied.
    ai["pos"] = [4.0, 4.0]
    foe["pos"] = [22.0, 4.0]
    for _ in range(10):
        now += 0.05
        tank._ai_intent(0, now)
        if tank.shells:
            break
    assert len(tank.shells) == 1


def test_ai_dodges_an_incoming_shell(tank):
    now = 6000.0
    tank.walls = []  # empty arena so the dodge has room and the shell runs clean
    ai, foe = tank.tanks[0], tank.tanks[1]
    tank.rng = _FakeRng(0.99)  # never fluff the dodge, never fire
    ai["pos"] = [8.0, 10.0]  # open area with room to sidestep on both axes
    ai["ai_skill"] = 0.0  # quickest reflexes
    ai["ai_standoff"] = float(tank.AI_STANDOFF)  # pin the per-spawn jitter
    foe["pos"] = [22.0, 4.0]
    foe["last_input_time"] = now  # idle controller -> foe AI off, foe holds still
    # An enemy shell barrelling straight down the AI's row toward it.
    tank.shells = [{"pos": [22.0, 10.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 1, "owner": 1}]
    for _ in range(20):
        now += 0.05
        foe["last_input_time"] = now
        tank._update_match(now, 0.05)
    assert tank.score[1] == 0  # the shot never connected
    assert abs(ai["pos"][1] - 10.0) >= 1.0  # it sidestepped off the lane


def test_ai_will_not_drive_into_a_shell_path(tank):
    now = 6000.0
    tank.walls = []  # empty arena so only the shell path drives the decision
    tank.rng = _FakeRng(0.99)  # never miss the threat, never fire back
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [6.0, 10.0]
    ai["ai_skill"] = 0.0
    ai["ai_standoff"] = float(tank.AI_STANDOFF)  # pin the per-spawn jitter
    foe["pos"] = [22.0, 22.0]  # far away and unaligned -> no shot distracts the AI
    # An enemy shell crossing down column 8 (clear of every wall block), just to
    # the AI's right.
    shell = {"pos": [8.0, 4.0], "vel": [0.0, tank.SHELL_SPEED], "bounces": 1, "owner": 1}
    tank.shells = [dict(shell)]
    for _ in range(20):
        now += 0.05
        foe["last_input_time"] = now  # idle controller -> foe AI off
        tank._update_match(now, 0.05)
    assert tank.score[1] == 0  # it repositioned without wandering into the shell

    # Once a threat is perceived, the chosen move never steps onto the path.
    tank.shells = [dict(shell)]
    ai["ai_had_threat"] = True
    ai["ai_threat_react_at"] = 0.0
    ai["ai_threat_ignored"] = False
    move = tank._ai_intent(0, now)["move"]
    nx = round(ai["pos"][0]) + move[0]
    ny = round(ai["pos"][1]) + move[1]
    assert all(
        max(abs(sx - nx), abs(sy - ny)) > tank.TANK_HALF
        for sx, sy, _t in tank._project_shell_cells(tank.shells[0], tank.AI_HORIZON)
    )


def test_ai_approaches_but_never_overlaps_the_foe(tank):
    tank.walls = []  # empty arena so the AI can close on a clear line
    ai = tank.tanks[0]
    foe_pos = (22.0, 4.0)  # a row clear of every wall block
    ai["pos"] = [4.0, 4.0]
    now = 6000.0
    closest = float("inf")
    for _ in range(200):  # ~10 s of AI maneuvering toward a stationary foe
        foe = tank.tanks[1]
        foe["alive"] = True
        foe["pos"] = list(foe_pos)
        foe["last_input_time"] = now  # idle controller -> foe never moves
        ai["fire_cd_until"] = float("inf")  # block shots so no hit resets the round
        now += 0.05
        tank._update_match(now, 0.05)
        dist = math.hypot(ai["pos"][0] - foe_pos[0], ai["pos"][1] - foe_pos[1])
        closest = min(closest, dist)
    # It closed to firing range but the 3x3 footprints never overlapped.
    assert closest > 2 * tank.TANK_HALF
    assert closest <= tank.AI_STANDOFF + 6


def test_ai_roams_when_every_lane_is_blocked(tank):
    # A persistent no-shot stalemate: a full-width bar splits the arena and the
    # foe is pinned on the far side, so no lane -- cardinal, diagonal or banked
    # off a wall -- ever reaches it and no maneuvering opens a shot.  Left to
    # the deterministic terms the AI would orbit the dead firing lines forever;
    # the stalemate roam must eventually engage and keep it moving.
    tank.rng = np.random.default_rng(0)
    tank.walls = [(13, 14, 1, 26)]
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [13.0, 18.0]  # below the bar, column-aligned with the foe
    foe["pos"] = [13.0, 9.0]  # above the bar, held fixed
    foe["vel"] = 0.0

    now = 6000.0
    positions = set()
    roam_seen = False
    for _ in range(150):  # ~7.5 s, comfortably past AI_STALL_TIME
        now += 0.05
        intent = tank._ai_intent(0, now)  # drive only the AI; the foe stays put
        tank._drive_tank(ai, intent, now, 0.05)
        roam_seen = roam_seen or ai["ai_roam_target"] is not None
        positions.add((round(ai["pos"][0]), round(ai["pos"][1])))
        assert tank.shells == []  # the wall denies every shot -> a true stalemate

    # The roam engaged and moved the tank around, rather than freezing.
    assert roam_seen
    assert len(positions) > 1


def test_drive_moves_tank_diagonally(tank):
    tank.walls = []  # empty arena: isolate movement physics from random walls
    t = tank.tanks[0]
    t["pos"] = [4.0, 4.0]
    now = 1000.0
    for _ in range(20):
        now += 0.05
        t["last_input_time"] = now
        tank.tanks[1]["last_input_time"] = now  # idle controller -> foe AI off
        tank.set_controller_input("left", move_x=1, move_y=1)  # two arrows held
        tank._update_match(now, 0.05)
    assert t["heading"] == 2  # snapped to the down-right diagonal
    dx = t["pos"][0] - 4.0
    dy = t["pos"][1] - 4.0
    assert dx > 1.0 and dy > 1.0
    assert abs(dx - dy) < 0.5  # a true 45-degree track


def test_diagonal_press_is_not_clamped(tank):
    tank.set_controller_input("left", move_x=1, move_y=-1)
    assert tank.tanks[0]["intent"]["move"] == (1, -1)


def test_diagonal_muzzle_and_shell_speed(tank):
    now = 5000.0
    tank.walls = []
    t = tank.tanks[0]
    t["pos"] = [14.0, 14.0]
    t["heading"] = 2  # down-right diagonal
    tank.fire("left", now)
    assert len(tank.shells) == 1
    shell = tank.shells[0]
    # The muzzle sits in the first cell outside the 3x3 body along the diagonal.
    assert [round(shell["pos"][0]), round(shell["pos"][1])] == [16, 16]
    # Diagonal shells fly at the same speed as cardinal ones.
    assert math.hypot(*shell["vel"]) == pytest.approx(tank.SHELL_SPEED)


def test_diagonal_shell_reflects_off_border(tank):
    now = 4000.0
    tank.walls = []  # empty arena so the shell only reflects off the border
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers -> AI never interferes
    s = tank.SHELL_SPEED / math.sqrt(2.0)
    # A shell flying down-left into the left border reflects on x only.
    tank.shells = [{"pos": [1.0, 14.0], "vel": [-s, s], "bounces": 1, "owner": 0}]
    tank._update_match(now, 0.05)
    assert len(tank.shells) == 1
    assert tank.shells[0]["vel"][0] > 0  # x component reflected
    assert tank.shells[0]["vel"][1] > 0  # y component untouched
    assert tank.shells[0]["bounces"] == 0


def test_your_own_shell_can_hit_you(tank):
    now = 3000.0
    tank.walls = []  # empty arena so the shell reaches the victim unobstructed
    victim = tank.tanks[0]
    victim["pos"] = [6.0, 24.0]
    tank.tanks[1]["pos"] = [20.0, 24.0]
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controller -> AI stays off, tanks hold still
    # The victim's OWN ricocheted shell, two pixels right of it, moving left.
    tank.shells = [{"pos": [8.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 0, "owner": 0}]
    tank._update_match(now, 0.05)
    assert victim["alive"] is False
    assert tank.score == [0, 1]  # a self-hit gifts the point to the other side


def test_shell_expires_after_two_bounces(tank):
    assert tank.SHELL_BOUNCES == 2
    now = 4000.0
    tank.walls = []  # empty arena so the shell only reflects off the border
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers -> AI never interferes
    # Throw a full-budget shell at the left border, re-arming it after each
    # reflection: it survives two bounces and expires on the third.
    tank.shells = [
        {"pos": [1.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": tank.SHELL_BOUNCES, "owner": 0}
    ]
    for remaining in (1, 0):
        tank._update_match(now, 0.05)
        assert len(tank.shells) == 1
        assert tank.shells[0]["bounces"] == remaining
        tank.shells[0]["pos"] = [1.0, 24.0]
        tank.shells[0]["vel"] = [-tank.SHELL_SPEED, 0.0]
        now += 0.05
    tank._update_match(now, 0.05)
    assert tank.shells == []


def test_shell_bounce_spawns_a_spark(tank):
    now = 4000.0
    tank.walls = []  # empty arena so the shell only reflects off the border
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers -> AI never interferes
    tank.shells = [{"pos": [1.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 1, "owner": 0}]
    tank._update_match(now, 0.05)
    sparks = [f for f in tank.fx if f.get("kind") == "spark"]
    assert len(sparks) == 1
    # The spark sits where the shell reflected, right against the border.
    assert (round(sparks[0]["x"]), round(sparks[0]["y"])) == (1, 24)


def test_expiring_shell_still_sparks(tank):
    now = 4000.0
    tank.walls = []  # empty arena so the shell only reflects off the border
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers -> AI never interferes
    # A shell with no bounce budget left: this reflection expires it.
    tank.shells = [{"pos": [1.0, 24.0], "vel": [-tank.SHELL_SPEED, 0.0], "bounces": 0, "owner": 0}]
    tank._update_match(now, 0.05)
    assert tank.shells == []  # expired on the bounce...
    assert any(f.get("kind") == "spark" for f in tank.fx)  # ...but fizzled visibly


def test_fast_wall_impact_flashes_a_dent(tank):
    tank.walls = []  # empty arena: the tank slams into the right border
    t = tank.tanks[0]
    t["pos"] = [25.4, 14.0]
    t["heading"] = 0  # +x, toward the border
    t["vel"] = tank.MAX_SPEED
    tank._drive_tank(t, {"move": (1, 0)}, 2000.0, 0.05)
    bumps = [f for f in tank.fx if f.get("kind") == "bump"]
    assert len(bumps) == 1
    # The dent sits one cell past the tank's nose, in the surface it struck.
    assert bumps[0]["x"] == pytest.approx(t["pos"][0] + tank.TANK_HALF + 1)
    # Grinding on against the wall does not keep re-flashing the dent.
    for _ in range(10):
        tank._drive_tank(t, {"move": (1, 0)}, 2000.0, 0.05)
    assert len([f for f in tank.fx if f.get("kind") == "bump"]) == 1


def test_slow_push_against_wall_stays_quiet(tank):
    tank.walls = []  # empty arena: the tank creeps into the right border
    t = tank.tanks[0]
    t["pos"] = [25.4, 14.0]
    t["heading"] = 0  # +x, toward the border
    t["vel"] = 0.0  # from rest, contact comes well below BUMP_MIN_SPEED
    for _ in range(20):
        tank._drive_tank(t, {"move": (1, 0)}, 2000.0, 0.05)
    assert not any(f.get("kind") == "bump" for f in tank.fx)


def test_tank_on_tank_collision_flashes_a_dent(tank):
    tank.walls = []  # empty arena so only the tank-vs-tank block is under test
    mover, blocker = tank.tanks[0], tank.tanks[1]
    mover["pos"] = [10.0, 24.0]
    blocker["pos"] = [14.0, 24.0]
    mover["heading"] = 0  # +x, straight at the blocker
    mover["vel"] = tank.MAX_SPEED
    for _ in range(10):
        tank._drive_tank(mover, {"move": (1, 0)}, 2500.0, 0.05)
    # One dent on first contact, none while it keeps pressing into the foe.
    assert len([f for f in tank.fx if f.get("kind") == "bump"]) == 1


def test_spark_and_dent_render(tank):
    now = 1000.0
    # A young spark is a stationary plus-shaped cluster -- nothing like the
    # lone travelling pixel of a shell -- that later collapses to a dot.
    frame = np.zeros((28, 28), dtype=np.uint8)
    tank.fx = [{"kind": "spark", "x": 14.0, "y": 14.0, "start": now, "until": now + tank.SPARK_LIFE}]
    tank._draw_fx(frame, now + 0.01)
    assert frame[14, 14] == 1
    assert frame[13, 14] == 1 and frame[15, 14] == 1  # vertical arms
    assert frame[14, 13] == 1 and frame[14, 15] == 1  # horizontal arms
    assert frame[13, 13] == 0  # a plus, not a solid blob
    late = np.zeros((28, 28), dtype=np.uint8)
    tank._draw_fx(late, now + tank.SPARK_LIFE * 0.9)
    assert late.sum() == 1  # collapsed to the centre dot

    # A young dent blinks the contact pixels off even on a fully lit surface,
    # and lights nothing anywhere (no lit debris that could read as shells).
    frame = np.ones((28, 28), dtype=np.uint8)
    tank.fx = [{"kind": "bump", "x": 14.0, "y": 14.0, "start": now, "until": now + tank.BUMP_LIFE}]
    tank._draw_fx(frame, now + 0.01)
    assert frame[13:16, 13:16].sum() == 0
    assert frame.sum() == 28 * 28 - 9  # exactly the 3x3 hole, nothing else changed


def test_ai_takes_a_bank_shot(tank):
    now = 6000.0
    # A wall block denies the direct column shot, but a shell fired down-left
    # ricochets off the left border and comes back in on the foe.
    tank.walls = [(12, 13, 4, 8)]
    tank.rng = _FakeRng(0.0)  # every bank-hunt and fire roll passes
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [6.0, 8.0]
    ai["ai_skill"] = 0.0
    foe["pos"] = [6.0, 18.0]
    foe["vel"] = 0.0
    # No direct lane exists (the wall blocks the shared column)...
    assert tank._lane_clear(6.0, 8.0, (0, 1), 6, 18) is False
    # ...but the physics sim approves the down-left bank and rejects the shot
    # straight down that would bounce off the wall back into the shooter.
    assert tank._shot_lands(0, (-1, 1)) is True
    assert tank._shot_lands(0, (0, 1)) is False
    for _ in range(10):
        now += 0.05
        tank._ai_intent(0, now)
        if tank.shells:
            break
    assert len(tank.shells) == 1
    assert ai["heading"] == 6  # gun snapped to the down-left bank direction
    vx, vy = tank.shells[0]["vel"]
    assert vx < 0 and vy > 0


def test_lane_clear_diagonal_blocked_at_corner_crossing(tank):
    # Two wall cells touching only at a corner: every cell on the diagonal
    # itself is open, but a real shell reflects at the corner crossing
    # (per-axis physics), so the AI's lane trace must read blocked too.
    tank.walls = [(11, 11, 12, 12), (12, 12, 11, 11)]
    assert tank._lane_clear(8.0, 8.0, (1, 1), 16, 16) is False
    tank.walls = []
    assert tank._lane_clear(8.0, 8.0, (1, 1), 16, 16) is True


def test_ai_takes_a_diagonal_shot(tank):
    now = 6000.0
    tank.walls = []  # empty arena: the diagonal lane to the foe is open
    tank.rng = _FakeRng(0.0)  # every fire roll passes -> only alignment gates it
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [6.0, 6.0]
    ai["ai_skill"] = 0.0
    foe["pos"] = [16.0, 16.0]  # only the down-right diagonal lines up
    foe["vel"] = 0.0
    for _ in range(10):
        now += 0.05
        tank._ai_intent(0, now)
        if tank.shells:
            break
    assert len(tank.shells) == 1
    assert ai["heading"] == 2  # gun snapped to the diagonal
    vx, vy = tank.shells[0]["vel"]
    assert vx > 0 and vy > 0 and abs(vx) == pytest.approx(abs(vy))


def test_ai_never_fires_diagonally_through_corner_gap(tank):
    now = 6000.0
    # The corner-touching walls sit exactly on the diagonal between the tanks.
    tank.walls = [(11, 11, 12, 12), (12, 12, 11, 11)]
    tank.AI_BANK_CHANCE = 0.0  # the corner-crossing lane rule is under test, not banks
    tank.rng = _FakeRng(0.0)  # every fire roll passes -> only the lane gates it
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [8.0, 8.0]
    ai["ai_skill"] = 0.0
    foe["pos"] = [16.0, 16.0]
    foe["vel"] = 0.0
    for _ in range(50):
        now += 0.05
        tank._ai_intent(0, now)
        assert tank.shells == []  # corner in the way -> it holds its fire


def test_ai_dodges_a_diagonal_shell(tank):
    now = 6000.0
    tank.walls = []  # empty arena so the dodge has room and the shell runs clean
    tank.rng = _FakeRng(0.99)  # never miss the threat, never fire back
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [10.0, 10.0]
    ai["ai_skill"] = 0.0  # quickest reflexes
    ai["ai_standoff"] = float(tank.AI_STANDOFF)  # pin the per-spawn jitter
    foe["pos"] = [22.0, 4.0]
    s = tank.SHELL_SPEED / math.sqrt(2.0)
    # An enemy shell barrelling up the x==y diagonal straight at the AI.
    tank.shells = [{"pos": [18.0, 18.0], "vel": [-s, -s], "bounces": 1, "owner": 1}]
    for _ in range(20):
        now += 0.05
        foe["last_input_time"] = now  # idle controller -> foe AI off
        tank._update_match(now, 0.05)
    assert tank.score[1] == 0  # the shot never connected
    assert abs(ai["pos"][0] - ai["pos"][1]) >= 1.5  # it left the diagonal lane


def test_ai_maneuvers_onto_a_diagonal_firing_line(tank):
    tank.walls = []  # empty arena so alignment is purely the AI's doing
    ai, foe = tank.tanks[0], tank.tanks[1]
    ai["pos"] = [6.0, 8.0]  # nearest firing line through the foe is the diagonal
    foe_pos = (20.0, 18.0)
    now = 6000.0
    aligned = False
    for _ in range(200):  # ~10 s of AI maneuvering toward a pinned foe
        foe["alive"] = True
        foe["pos"] = list(foe_pos)
        foe["vel"] = 0.0
        foe["last_input_time"] = now  # idle controller -> foe never moves
        ai["fire_cd_until"] = float("inf")  # block shots so no hit resets the round
        now += 0.05
        tank._update_match(now, 0.05)
        adx = abs(ai["pos"][0] - foe_pos[0])
        ady = abs(ai["pos"][1] - foe_pos[1])
        aligned = aligned or abs(adx - ady) <= tank.AI_ALIGN_TOL + 1
        # The 3x3 footprints never overlap while it maneuvers.
        assert max(adx, ady) > 2 * tank.TANK_HALF
    assert aligned


def test_ai_vs_ai_attract_mode_progresses(tank):
    # Unattended attract mode: both controllers idle, both tanks AI-driven.
    # Over a long stretch the match must make progress (a point is scored)
    # rather than deadlocking, and never crash.
    tank.rng = np.random.default_rng(1)
    now = 8000.0
    progressed = False
    for _ in range(1800):  # ~90 simulated seconds
        now += 0.05
        tank._update_match(now, 0.05)
        progressed = progressed or sum(tank.score) >= 1
    assert progressed


def test_diagonal_barrel_renders_two_pixels(tank):
    t = tank.tanks[0]
    t["pos"] = [14.0, 14.0]
    t["heading"] = 2  # down-right diagonal
    frame = np.zeros((28, 28), dtype=np.uint8)
    tank._draw_tank(frame, t)
    assert frame[16, 16] == 1 and frame[17, 17] == 1  # both outside the 3x3 body

    t["heading"] = 0  # right: the cardinal barrel is unchanged
    frame = np.zeros((28, 28), dtype=np.uint8)
    tank._draw_tank(frame, t)
    assert frame[14, 16] == 1 and frame[14, 17] == 1


def test_reaching_win_score_sets_winner_and_restarts(tank):
    now = 6000.0
    tank.score = [tank.WIN_SCORE - 1, 0]
    tank._register_hit(1, now)
    assert tank.winner == 0
    assert tank.win_time == now

    tank.restart_if_game_over()
    assert tank.winner is None
    assert tank.score == [0, 0]


def test_win_blast_captures_origin_and_resets(tank):
    now = 6000.0
    tank.score = [tank.WIN_SCORE - 1, 0]
    victim = tank.tanks[1]
    victim["pos"] = [20.0, 12.0]
    tank._register_hit(1, now)
    # The deciding hit records the blast origin and a dither field for the fade.
    assert tank.winner == 0
    assert (tank.win_x, tank.win_y) == (20.0, 12.0)
    assert tank._win_noise is not None and tank._win_noise.shape == (28, 28)

    tank.restart_if_game_over()
    assert tank.win_x is None and tank.win_y is None and tank._win_noise is None


def test_win_blast_then_game_over_screen(tank):
    now = 6000.0
    tank.score = [0, tank.WIN_SCORE - 1]
    tank.tanks[0]["pos"] = [14.0, 14.0]
    tank._register_hit(0, now)  # tank 1 (right, hollow) wins

    def draw_at(t):
        frame = np.zeros((28, 28), dtype=np.uint8)
        tank._draw_game_over(frame, t)
        return frame

    # Grow phase: the blast lights up a large swathe of the panel.
    blast = draw_at(tank.win_time + 0.1)
    assert blast.dtype == np.uint8 and set(np.unique(blast)).issubset({0, 1})
    assert blast.sum() > 100

    # Settled phase: a valid bitmap with the winner icon + text lit.
    settled = draw_at(tank.win_time + tank.WIN_BLAST_GROW + tank.WIN_BLAST_FADE + 0.1)
    assert set(np.unique(settled)).issubset({0, 1})
    assert 0 < settled.sum() < blast.sum()  # text screen, not a full-white blast


def test_game_over_screen_fires_a_salute(tank):
    now = 6000.0
    tank.score = [0, tank.WIN_SCORE - 1]
    tank.tanks[0]["pos"] = [14.0, 14.0]
    tank._register_hit(0, now)  # tank 1 wins
    settled_at = tank.win_time + tank.WIN_BLAST_GROW + tank.WIN_BLAST_FADE

    def draw_at(t):
        frame = np.zeros((28, 28), dtype=np.uint8)
        tank._draw_game_over(frame, t)
        return frame

    # The salute cycle: a rising shot, then a burst, then a quiet stretch --
    # each phase lighting more pixels than the bare settled screen.
    quiet = draw_at(settled_at + tank.WIN_FIRE_PERIOD * 0.9)
    shot = draw_at(settled_at + tank.WIN_FIRE_PERIOD * 0.1)
    burst = draw_at(settled_at + tank.WIN_FIRE_PERIOD * 0.3)
    assert shot.sum() == quiet.sum() + 1  # the single rising shot pixel
    assert burst.sum() == quiet.sum() + 5  # the plus-shaped burst
    # It keeps saluting for the whole Game Over hold, one shot per period.
    later = draw_at(settled_at + tank.WIN_FIRE_PERIOD * 3.1)
    assert later.sum() == quiet.sum() + 1


def test_score_shown_between_rounds(tank, monkeypatch):
    now = 3000.0
    monkeypatch.setattr("app.modes.tank.time.time", lambda: now)
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers -> AI stays off
    tank._start_time = now - tank.MODE_NAME_TIME - 1  # past the intro banner
    tank._last_frame_time = now
    tank.score = [2, 1]
    tank._register_hit(0, now)  # a non-winning point downs both tanks
    assert tank.winner is None and all(not t["alive"] for t in tank.tanks)

    ref = np.zeros((28, 28), dtype=np.uint8)
    text.write_centered(ref, tank._score_text(), y=11, size=6, style="regular")
    assert ref.sum() > 0

    # While the hit explosion still plays the score banner is withheld.
    during_blast = tank.get_frame(None)
    assert tank.fx and not during_blast[10:18, :].all()

    # After the explosion, during the score phase, the score shows as an
    # inverted banner (past the brief dither sweep-in): the band is lit solid
    # with the digits punched out dark.
    scoring = now + tank.ROUND_EXPLODE + 0.4
    monkeypatch.setattr("app.modes.tank.time.time", lambda: scoring)
    settled = tank.get_frame(None)
    assert not tank.fx
    assert not settled[ref == 1].any()  # digits are the dark holes...
    band = settled[10:18, :].astype(bool) | ref[10:18, :].astype(bool)
    assert band.all()  # ...in an otherwise fully lit band

    # Once both respawn (past the full transition) the overlay no longer runs.
    later = now + tank.RESPAWN_DELAY + 0.05
    monkeypatch.setattr("app.modes.tank.time.time", lambda: later)
    for t in tank.tanks:
        t["last_input_time"] = later
    tank.get_frame(None)
    assert all(t["alive"] for t in tank.tanks)


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


def test_generated_layout_is_180_symmetric(tank):
    # A layout mirrored about the panel centre equals itself, so each tank sees
    # the same arena rotated half a turn -> neither corner is advantaged.
    for seed in range(200):
        tank.rng = np.random.default_rng(seed)
        tank.walls = tank._make_walls()
        mask = _wall_mask(tank)
        assert np.array_equal(mask, np.rot90(mask, 2))


def test_generated_layout_keeps_spawns_clear_and_connected(tank):
    for seed in range(200):
        tank.rng = np.random.default_rng(seed)
        tank.walls = tank._make_walls()
        # Both spawn footprints are free of walls...
        assert not tank._tank_blocked(3, 3)
        assert not tank._tank_blocked(tank.width - 4, tank.height - 4)
        # ...and a 3x3 tank can travel between the two spawn corners.
        assert tank._spawns_connected(tank.walls)


def test_generated_layout_has_distributed_cover(tank):
    # Every layout carries enough wall to fight around (never a lone small box
    # in an empty field) while staying sparse enough to maneuver, keeps at
    # most half its mass out of the central box, and comes as at least two
    # separate pieces -- never one connected clump in the middle.
    for seed in range(200):
        tank.rng = np.random.default_rng(seed)
        tank.walls = tank._make_walls()
        mask = _wall_mask(tank)
        area = int(mask.sum())
        assert 20 <= area <= 80
        central = int(mask[9:19, 9:19].sum())
        assert central * 2 <= area
        assert tank._wall_clumps(mask) >= 2


def test_generated_layout_gaps_admit_a_tank(tank):
    # Any opening between two wall pieces is wide enough for a 3x3 tank (and
    # a shell) to pass -- no teasing 1-2px slots that play like solid wall.
    for seed in range(200):
        tank.rng = np.random.default_rng(seed)
        assert tank._gaps_are_passable(tank._make_walls())


def test_new_field_generated_after_each_point(tank):
    now = 6000.0
    tank.walls = tank._default_walls()
    old = tank.walls
    tank.tanks[0]["pos"] = [14.0, 14.0]
    tank._register_hit(0, now)  # a non-winning point starts the round transition
    assert tank.winner is None
    assert tank.round_over_at == now
    # The field is generated up front but held back -- the old field stays put
    # through the explosion and score phases so the swap comes last.
    assert tank.walls is old
    assert tank.next_walls is not None
    expected = tank.next_walls
    tank._update_match(now + tank.ROUND_EXPLODE + 0.1, 0.05)
    assert tank.walls is old and tank.round_over_at is not None

    # Once the whole sequence finishes the new field is installed and both tanks
    # respawn at their corners.
    later = now + tank.RESPAWN_DELAY + 0.05
    for t in tank.tanks:
        t["last_input_time"] = later
    tank._update_match(later, 0.05)
    assert tank.round_over_at is None
    assert tank.walls is expected
    assert all(t["alive"] for t in tank.tanks)
    # The installed field is still fair, spawn-clear and connected.
    mask = _wall_mask(tank)
    assert np.array_equal(mask, np.rot90(mask, 2))
    assert not tank._tank_blocked(3, 3)
    assert tank._spawns_connected(tank.walls)


def test_field_swap_wipes_in_diagonally(tank):
    # The new field sweeps in behind a diagonal frontier: ahead of it the old
    # walls still stand, behind it only the new ones show.
    tank.walls = [(5, 6, 5, 10)]  # old piece, top-left (diagonal 10..16)
    tank.next_walls = [(8, 9, 5, 10)]  # new piece, same corner (diagonal 13..19)

    frame = np.zeros((28, 28), dtype=np.uint8)
    tank._draw_field_morph(frame, 0.1)  # frontier barely off the top-left corner
    assert frame[5:7, 5:11].all()  # old piece still standing ahead of the wipe
    assert not frame[8:10, 5:11].any()  # new piece not yet revealed

    frame = np.zeros((28, 28), dtype=np.uint8)
    tank._draw_field_morph(frame, 0.6)  # frontier well past the top-left corner
    assert frame[8:10, 5:11].all()  # the new piece has wiped in
    assert not frame[5:7, 5:11].any()  # the old piece there is gone
    assert frame.sum() > 12  # the lit frontier beam itself is on the panel


def test_round_transition_renders_and_hides_tanks(tank, monkeypatch):
    now = 7000.0
    monkeypatch.setattr("app.modes.tank.time.time", lambda: now)
    for t in tank.tanks:
        t["last_input_time"] = now  # idle controllers
    tank._start_time = now - tank.MODE_NAME_TIME - 1  # past the intro banner
    tank._last_frame_time = now
    tank._register_hit(0, now)  # start the transition

    # Sample across the whole sequence: every frame is a valid bitmap and the
    # tanks stay down until it completes.
    for frac in (0.0, 0.3, 0.6, 0.9):
        moment = now + tank.RESPAWN_DELAY * frac
        monkeypatch.setattr("app.modes.tank.time.time", lambda m=moment: m)
        tank._last_frame_time = moment
        frame = tank.get_frame(None)
        assert set(np.unique(frame)).issubset({0, 1})
        assert tank.round_over_at is not None  # mid-transition -> tanks hidden

    # Past the full transition the tanks respawn and start materialising.
    after = now + tank.RESPAWN_DELAY + 0.01
    monkeypatch.setattr("app.modes.tank.time.time", lambda: after)
    tank._last_frame_time = after
    tank.get_frame(None)
    assert tank.round_over_at is None
    assert all(t["alive"] for t in tank.tanks)
    assert tank.tanks_spawned_at == after
