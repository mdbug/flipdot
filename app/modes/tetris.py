import numpy as np
import time
import app.services.text as text
import app.services.human_pose as human_pose
from app.modes.autodrum import AutoDrum

# Index of the TETRIS song in AutoDrum.SONGS, looked up by name so it
# stays correct if the song list is ever reordered.
_TETRIS_SONG_IDX = next(
    i for i, s in enumerate(AutoDrum.SONGS) if s['name'] == 'TETRIS'
)


class Tetris(AutoDrum):
    """Playable Tetris mode with gesture controls.

    Inherits AutoDrum's Korobeiniki sequencer and flip-dot audio engine.
    Replaces the AI Tetris demo with a player-controlled game.

    Layout (28×28 panel)
    --------------------
    * Melody band: columns 0–6 (7 px), shown inverted — lit background,
      note stripes dark.  The score (top) and next-piece preview (below
      it) are XOR'd over the band, so they invert whatever lies beneath
      and stay readable under note stripes.
    * Separator : column 7, always lit.
    * Game board: columns 8–27, 10 cells × 2 px wide (20 px), 14 rows tall.

    Gesture controls
    ----------------
    * Right index finger X  → steer piece left/right (auto-repeat 150 ms)
    * Left hand raised      → rotate piece (rising-edge trigger)
    * Arms crossed          → exit to menu (handled by main loop as usual)

    Game-over screen
    ----------------
    Death jingle, panel flash, then "GAME / OVER" + lines + high score.
    Raise left hand (or wait 10 s) to restart; every new game also
    restarts the song from the top of the theme.

    AI attract mode
    ---------------
    When no person is detected, an AI takes over after AI_TAKEOVER_DELAY
    seconds.  The AI evaluates every (rotation, column) placement of the
    current piece AND of the known next piece (one-ply lookahead via the
    preview) with a classic heuristic, then steers the falling piece
    using the same buffered-input path as human gestures.  The tempo
    ramp doubles as a natural kill screen: as lines accumulate, the
    music — and therefore beat-locked gravity — eventually outruns the
    AI's fixed input cadence, so even the attract mode dies in the end.
    """

    AI_TAKEOVER_DELAY = 5.0   # seconds of absence before AI takes over

    def __init__(self, width, height, mode_manager):
        # These must exist before super().__init__: the init chain
        # (_load_song → _tetris_background → … → _player_spawn) can in
        # principle reach _trigger_game_over, which uses them.
        self._best = 0                  # high score — survives restarts
        self._jingle_events = []        # (due_time, note_name) death-jingle queue
        # super().__init__ → _load_song(0) → our override → super()._load_song(idx)
        # which calls _tetris_background() → _init_player_state() → _player_spawn().
        # All game state is ready by the time super() returns.
        super().__init__(width, height, mode_manager)
        # Gesture edge-detection state (set after super so init chain is clean)
        self._left_raise_armed = True   # True = ready to trigger on next raise
        self._last_move_time = 0.0      # timestamp of last horizontal step
        self._pending = {'move': 0, 'rotate': False}  # buffered per-step intent

    # ------------------------------------------------------------------
    # _load_song override — always loads TETRIS song, then re-pins the
    # melody band to cols 0–6 so col 7 (separator) is never masked.
    # ------------------------------------------------------------------

    def _load_song(self, index):
        super()._load_song(_TETRIS_SONG_IDX)
        # Melody band: cols 0–6 (7 px); col 7 is the separator, not part of
        # the band, so state flips there always reach the panel as sound.
        mc0 = 0
        mc1 = 7
        for name in self._melody_names:
            inst = self.instruments[name]
            r0, r1, _, _ = inst['area']
            inst['area'] = (r0, r1, mc0, mc1)
        self._voice_span = (mc0, mc1)
        # Restrict crash to the melody column band so line-clear hits don't
        # scatter pixels into the game board area (cols 8–27).
        r0, r1, _, _ = self.instruments['crash']['area']
        self.instruments['crash']['area'] = (r0, r1, mc0, mc1)

    # ------------------------------------------------------------------
    # _tetris_background override — fixed 10-column player board layout
    # ------------------------------------------------------------------

    def _tetris_background(self):
        """Initialise the player board; return its first rendered frame."""
        h = self.height
        c = 2
        ncols = 10
        nrows = h // c   # 14 for a 28-px-tall panel (uses full height)
        x0 = self.width - ncols * c   # right-align: col 8 for a 28-wide panel
        # Reuse self._tetris for tick-counting (board/piece fields unused).
        self._tetris = {
            'c': c, 'ncols': ncols, 'nrows': nrows, 'x0': x0,
            'board': None,   # not used — player board lives in self._player
            'piece': None,
            'flash': [], 'flash_ticks': 0,
            'ticks': -1,
        }
        self._init_player_state()
        return self._render_player_board()

    # ------------------------------------------------------------------
    # Player board helpers
    # ------------------------------------------------------------------

    def _init_player_state(self):
        t = self._tetris
        self._player = {
            'board':          np.zeros((t['nrows'], t['ncols']), dtype=bool),
            'piece':          None,
            'piece_seq':      0,     # monotonic spawn counter (AI retarget key)
            'next_rots':      None,  # rotations of the upcoming piece (preview)
            'flash':          [],
            'flash_ticks':    0,
            'lines':          0,
            'game_over':      False,
            'game_over_time': None,
        }
        self._ai_target = None          # (target_col, target_rot_idx)
        self._ai_piece_id = None        # piece_seq when _ai_target was computed
        self._last_person_time = None   # timestamp of last detected person
        self._ai_rotate_cooldown = 0.0  # timestamp of last AI-requested rotation
        self._player_spawn(time.time())

    def _player_tet_fits(self, cells, row, col):
        """Collision test against the player's board (not the AI board)."""
        t = self._tetris
        board = self._player['board']
        for dr, dc in cells:
            r, cc = row + dr, col + dc
            if cc < 0 or cc >= t['ncols'] or r >= t['nrows']:
                return False
            if r >= 0 and board[r, cc]:
                return False
        return True

    def _trigger_game_over(self, now):
        """Set game-over state and schedule the descending death jingle."""
        self._player['game_over'] = True
        self._player['game_over_time'] = now
        self._best = max(self._best, self._player['lines'])
        notes = ['a5', 'g5', 'f5', 'e5', 'd5', 'c5', 'b4', 'a4']
        self._jingle_events = [
            (now + i * 0.08, note) for i, note in enumerate(notes)
        ]

    def _player_spawn(self, now):
        """Promote the previewed piece to falling; draw a new next piece."""
        t = self._tetris
        shapes = list(self.TETROMINOES.values())
        if self._player['next_rots'] is None:   # very first spawn of a game
            shape = shapes[int(self.rng.integers(len(shapes)))]
            self._player['next_rots'] = self._rotations(shape)
        all_rots = self._player['next_rots']
        # Draw the upcoming piece now: it feeds both the preview render
        # and the AI's one-ply lookahead.
        shape = shapes[int(self.rng.integers(len(shapes)))]
        self._player['next_rots'] = self._rotations(shape)

        cells = all_rots[0]
        piece_width = max(dc for _, dc in cells) + 1
        col = (t['ncols'] - piece_width) // 2
        row = -(max(dr for dr, _ in cells) + 1)
        self._player['piece_seq'] += 1
        piece = {
            'cells':    cells,
            'row':      row,
            'col':      col,
            'all_rots': all_rots,
            'rot_idx':  0,
            'seq':      self._player['piece_seq'],
        }
        self._player['piece'] = piece
        # A fresh piece must not inherit input buffered for the old one.
        self._pending = {'move': 0, 'rotate': False}
        # Game over when spawn position is already blocked
        if not self._player_tet_fits(cells, row, col):
            self._trigger_game_over(now)
            self._player['piece'] = None

    def _player_rotate(self):
        """Try the next rotation with wall-kick ±1, ±2."""
        p = self._player['piece']
        if p is None:
            return
        all_rots = p['all_rots']
        next_idx = (p['rot_idx'] + 1) % len(all_rots)
        next_cells = all_rots[next_idx]
        for kick in (0, -1, 1, -2, 2):
            new_col = p['col'] + kick
            if self._player_tet_fits(next_cells, p['row'], new_col):
                p['cells'] = next_cells
                p['rot_idx'] = next_idx
                p['col'] = new_col
                self._bg_frame = self._render_player_board()
                return

    def _player_lock(self, now):
        """Stamp the current piece into the board, detect full rows."""
        t = self._tetris
        p = self._player['piece']
        if p is None:
            return
        game_over = False
        for dr, dc in p['cells']:
            r, cc = p['row'] + dr, p['col'] + dc
            if r < 0:
                game_over = True   # piece locked above the visible playfield
            elif r < t['nrows']:
                self._player['board'][r, cc] = True
        self._player['piece'] = None
        if game_over:
            self._trigger_game_over(now)
            return
        full = list(np.flatnonzero(self._player['board'].all(axis=1)))
        if full:
            self._player['flash'] = full
            self._player['flash_ticks'] = 0
        else:
            self._player_spawn(now)

    def _player_hard_drop(self):
        """Drop the piece to the lowest valid row and lock it. (Unused —
        kept in case a safe gesture for it is found later.)"""
        p = self._player['piece']
        if p is None:
            return
        while self._player_tet_fits(p['cells'], p['row'] + 1, p['col']):
            p['row'] += 1
        self._player_lock(time.time())
        self._bg_frame = self._render_player_board()

    def _restart_game(self):
        """Start a fresh game AND the song from the top of the theme."""
        self._jingle_events = []
        self._left_raise_armed = False
        self._pending = {'move': 0, 'rotate': False}
        # _load_song resets the sequencer to bar 1, silences the melody
        # column, and — via the 'bg' hook — rebuilds the board
        # (_init_player_state + first spawn).  self._best survives.
        self._load_song(0)

    # ------------------------------------------------------------------
    # _tetris_step override — buffered input each step, gravity on beats
    # ------------------------------------------------------------------

    def _tetris_step(self, now):
        """Called on every sequencer step; execute buffered input, then gravity on beats."""
        t = self._tetris
        song = self.SONGS[self.song_index]
        t['ticks'] += 1

        # Execute buffered player input on every step (eighth-note resolution).
        # This runs before the beat guard so moves and rotations land as
        # grace notes even on off-beats, keeping them in sync with the grid.
        if not self._player['game_over'] and not self._player['flash']:
            changed = False
            if self._pending['rotate']:
                self._player_rotate()   # _player_rotate also updates _bg_frame
                self._pending['rotate'] = False
                changed = True
            if self._pending['move']:
                p = self._player['piece']
                if p is not None:
                    new_col = p['col'] + self._pending['move']
                    if self._player_tet_fits(p['cells'], p['row'], new_col):
                        p['col'] = new_col
                        changed = True
                self._pending['move'] = 0
            if changed:
                self._bg_frame = self._render_player_board()

        if t['ticks'] % song['subdivisions']:
            return   # only gravity/lock/flash on full beats

        if self._player['game_over']:
            return

        flash = self._player['flash']
        if flash:
            # Line-clear animation: blink for 2 beats, then collapse
            self._player['flash_ticks'] += 1
            if self._player['flash_ticks'] >= 2:
                keep = np.ones(t['nrows'], dtype=bool)
                for r in flash:
                    keep[r] = False
                self._player['board'] = np.vstack([
                    np.zeros((len(flash), t['ncols']), dtype=bool),
                    self._player['board'][keep],
                ])
                n_cleared = len(flash)
                self._player['lines'] += n_cleared
                self._player['flash'] = []
                self._player['flash_ticks'] = 0
                # Extra crash hit for multi-line clears; the flip-dot splash
                # plus shimmer tail scales naturally with line count.
                if n_cleared >= 2:
                    self._hit('crash', now)
                self._player_spawn(now)
        elif self._player['piece'] is not None:
            p = self._player['piece']
            if self._player_tet_fits(p['cells'], p['row'] + 1, p['col']):
                p['row'] += 1
            else:
                self._player_lock(now)
        else:
            self._player_spawn(now)

        self._bg_frame = self._render_player_board()

    # ------------------------------------------------------------------
    # _render_player_board — 1-bit background frame for XOR composite
    # ------------------------------------------------------------------

    def _render_player_board(self):
        t = self._tetris
        c = t['c']
        h, w = self.height, self.width
        # Align board to fill the full panel height.
        top = h - t['nrows'] * c
        x0 = t['x0']
        bg = np.zeros((h, w), dtype=np.uint8)

        flash = set(self._player['flash'])

        # Locked cells
        for r, cc in zip(*np.nonzero(self._player['board'])):
            if int(r) not in flash:
                r0 = top + int(r) * c
                c0 = x0 + int(cc) * c
                bg[r0:r0 + c, c0:c0 + c] = 1

        # Current falling piece
        p = self._player['piece']
        if p is not None:
            for dr, dc in p['cells']:
                row = p['row'] + dr
                if row >= 0:
                    r0 = top + row * c
                    c0 = x0 + (p['col'] + dc) * c
                    if 0 <= r0 < h and 0 <= c0 < w:
                        bg[r0:r0 + c, c0:c0 + c] = 1

        # Flashing rows blink (solid on even ticks)
        if flash and self._player['flash_ticks'] % 2 == 0:
            for r in flash:
                bg[top + r * c:top + (r + 1) * c,
                   x0:x0 + t['ncols'] * c] = 1

        return bg

    # ------------------------------------------------------------------
    # AI attract-mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ai_score_board(board, ncols, nrows):
        """Heuristic score for a (post-clear) board — higher is better.

        Line clears are rewarded separately by the caller, so this only
        judges the shape of what remains.
        """
        col_heights = np.zeros(ncols, dtype=int)
        for col in range(ncols):
            for row in range(nrows):
                if board[row, col]:
                    col_heights[col] = nrows - row
                    break
        aggregate_height = int(col_heights.sum())
        holes = 0
        for col in range(ncols):
            ch = col_heights[col]
            if ch == 0:
                continue
            top_row = nrows - ch
            for row in range(top_row + 1, nrows):
                if not board[row, col]:
                    holes += 1
        bumpiness = int(np.abs(np.diff(col_heights)).sum())
        # Wells: columns lower than both neighbours breed unrecoverable states;
        # penalise quadratically so deep wells are disproportionately bad.
        wells = 0
        for col in range(ncols):
            left = col_heights[col - 1] if col > 0 else nrows
            right = col_heights[col + 1] if col < ncols - 1 else nrows
            depth = min(left, right) - col_heights[col]
            if depth > 0:
                wells += depth * (depth + 1) // 2
        return (- aggregate_height * 0.51
                - holes * 6
                - bumpiness * 1.8
                - wells * 1.2)

    def _ai_drop_board(self, board, cells, col):
        """Simulate dropping cells at col onto board.

        Returns (board_after_line_clears, lines_cleared), or None if the
        placement is impossible or would lock above the playfield.
        """
        t = self._tetris

        def fits(r):
            for dr, dc in cells:
                rr, cc = r + dr, col + dc
                if cc < 0 or cc >= t['ncols'] or rr >= t['nrows']:
                    return False
                if rr >= 0 and board[rr, cc]:
                    return False
            return True

        row = -(max(dr for dr, _ in cells) + 1)
        if not fits(row):
            return None
        while fits(row + 1):
            row += 1
        if row + min(dr for dr, _ in cells) < 0:
            return None   # would lock sticking out of the top
        nb = board.copy()
        for dr, dc in cells:
            nb[row + dr, col + dc] = True
        full = nb.all(axis=1)
        cleared = int(full.sum())
        if cleared:
            nb = np.vstack([np.zeros((cleared, t['ncols']), dtype=bool),
                            nb[~full]])
        return nb, cleared

    def _ai_best_placement(self):
        """(target_col, target_rot_idx) maximising score with lookahead.

        For every placement of the current piece, the resulting board is
        re-evaluated against every placement of the KNOWN next piece (the
        one shown in the preview), and the best combined outcome wins —
        depth-2 search, exact because the next piece is not random to us.
        Runs once per spawn; on a 10×14 board this is ~1.5k leaf
        evaluations, a few ms of work.
        """
        t = self._tetris
        p = self._player['piece']
        board = self._player['board']
        next_rots = self._player['next_rots']
        best = None
        for rot_idx, cells in enumerate(p['all_rots']):
            piece_max_dc = max(dc for _, dc in cells)
            for col in range(0, t['ncols'] - piece_max_dc):
                res = self._ai_drop_board(board, cells, col)
                if res is None:
                    continue
                b1, c1 = res
                # Lookahead: best response with the previewed next piece
                best_next = None
                for ncells in next_rots:
                    n_max_dc = max(dc for _, dc in ncells)
                    for ncol in range(0, t['ncols'] - n_max_dc):
                        res2 = self._ai_drop_board(b1, ncells, ncol)
                        if res2 is None:
                            continue
                        b2, c2 = res2
                        s2 = c2 * 100 + self._ai_score_board(
                            b2, t['ncols'], t['nrows'])
                        if best_next is None or s2 > best_next:
                            best_next = s2
                if best_next is None:
                    # Next piece would have nowhere to go: near-certain death
                    score = c1 * 100 + self._ai_score_board(
                        b1, t['ncols'], t['nrows']) - 1000
                else:
                    score = c1 * 100 + best_next
                if best is None or score > best[0]:
                    best = (score, col, rot_idx)
        if best is None:
            return p['col'], p['rot_idx']
        return best[1], best[2]

    def _handle_ai(self, now):
        """Drive the current piece toward the AI's chosen placement."""
        p = self._player['piece']
        if p is None:
            return
        # Recompute target whenever a new piece spawns.  Keyed on the
        # monotonic spawn counter — NOT id(p), which CPython can reuse
        # for a new dict allocated at a freed dict's address.
        if p['seq'] != self._ai_piece_id:
            self._ai_target = self._ai_best_placement()
            self._ai_piece_id = p['seq']
            self._ai_rotate_cooldown = 0.0
        target_col, target_rot = self._ai_target
        # Rotate toward target using a dedicated cooldown — independent of
        # _left_raise_armed so human takeover is never disrupted, and multiple
        # rotations (e.g. 180°) work correctly.  Wait until the piece has
        # entered the visible board (row >= 0) so the rotation is observable.
        if (p['rot_idx'] != target_rot and p['row'] >= 0
                and now - self._ai_rotate_cooldown >= 0.15):
            self._pending['rotate'] = True
            self._ai_rotate_cooldown = now
        # Steer toward target column in parallel — don't wait for rotation.
        if p['col'] != target_col and now - self._last_move_time >= 0.15:
            self._pending['move'] = 1 if target_col > p['col'] else -1
            self._last_move_time = now

    # ------------------------------------------------------------------
    # Gesture handling
    # ------------------------------------------------------------------

    def _handle_gestures(self, pose_results, now):
        # ---- Game over: only watch for restart ----
        if self._player['game_over']:
            left_raised = human_pose.is_left_hand_raised(pose_results)
            if left_raised and self._left_raise_armed:
                self._restart_game()
            elif not left_raised:
                self._left_raise_armed = True
            # Auto-restart after 10 s
            if (self._player['game_over_time'] is not None
                    and now - self._player['game_over_time'] >= 10.0):
                self._restart_game()
            return

        # ---- Presence detection: route to AI if no person detected ----
        person_present = (
            pose_results is not None
            and getattr(pose_results, 'pose_landmarks', None) is not None
        )
        if person_present:
            self._last_person_time = now
        else:
            delay_expired = (
                self._last_person_time is None
                or now - self._last_person_time >= self.AI_TAKEOVER_DELAY
            )
            if delay_expired:
                self._handle_ai(now)
            return

        # ---- Rotate: buffer intent, execute on next step ----
        left_raised = human_pose.is_left_hand_raised(pose_results)
        if left_raised and self._left_raise_armed:
            self._pending['rotate'] = True
            self._left_raise_armed = False
        elif not left_raised:
            self._left_raise_armed = True

        # ---- Steer: buffer direction, execute on next step ----
        finger_x, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_x is not None and self._player['piece'] is not None:
            t = self._tetris
            p = self._player['piece']
            # draw_right_index_pointer mirrors: screen_x = width − finger_x × width
            screen_x = self.width - finger_x * self.width
            # Map screen_x to board column
            piece_max_dc = max(dc for _, dc in p['cells'])
            max_col = t['ncols'] - piece_max_dc - 1
            target_col = max(0, min(max_col, int((screen_x - t['x0']) / t['c'])))
            if target_col != p['col'] and now - self._last_move_time >= 0.15:
                self._pending['move'] = 1 if target_col > p['col'] else -1
                self._last_move_time = now

    def get_frame(self, pose_results):
        now = time.time()
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song['bpm'] / song['subdivisions']
        # Tempo ramp: every 5 lines the song — and beat-locked gravity —
        # speeds up.  Deliberately uncapped: at extreme line counts the
        # beat outruns the AI's fixed 0.15 s input cadence, giving the
        # attract mode an emergent kill screen.
        step_interval /= (1 + 0.08 * (self._player['lines'] // 5))

        self._handle_gestures(pose_results, now)

        if self._player['game_over']:
            # Fire any scheduled jingle notes, then drain tails via shared method
            due_jingle = [e for e in self._jingle_events if e[0] <= now]
            self._jingle_events = [e for e in self._jingle_events if e[0] > now]
            for _, note in due_jingle:
                self._hit(note, now)
            self._tick_voice(now)
            # Once jingle and all tails are spent, silence the melody column
            if not self._jingle_events and not self._voice_decay and not self._decay_events:
                if self._voice_shimmer is not None and self._voice_area is not None:
                    r0, r1, c0, c1 = self._voice_area
                    self.state[r0:r1, c0:c1] ^= self._voice_shimmer
                    self._voice_shimmer = None
                mc0, mc1 = self._voice_span
                self.state[:, mc0:mc1] = 0
        else:
            self._tick_voice(now)
            self._advance_sequencer(now, step_interval)

        # Composite: melody state XOR game board background
        frame = self.state.copy()
        if self._bg_frame is not None:
            frame ^= self._bg_frame

        # Separator line between melody band (cols 0–6) and game board
        frame[:, 7] = 1

        # Invert the melody band: lit background, note stripes dark.
        # XOR with a constant changes polarity only — frame-to-frame
        # flips (the sound) are completely unaffected.
        frame[:, :7] ^= 1

        # Score (top) and next-piece preview live in a UI layer XOR'd
        # over the melody band: the glyphs and preview cells invert
        # whatever lies beneath (dark on the lit background, lit when a
        # note stripe passes under them), so they always stay readable.
        if not self._player['game_over']:
            ui = np.zeros((self.height, 7), dtype=np.uint8)
            score_str = str(min(self._player['lines'], 99))
            text.write(ui, score_str, x=text.center_x(7, score_str, size=5),
                       y=1, size=5)
            nxt = self._player['next_rots'][0]
            pw = max(dc for _, dc in nxt) + 1
            px = max(0, (7 - pw) // 2)
            for dr, dc in nxt:           # 1 px per cell, under the score
                ui[9 + dr, px + dc] = 1
            frame[:, :7] ^= ui

        # Game-over overlay: jingle first, then flash, then static screen
        if self._player['game_over'] and self._player['game_over_time'] is not None:
            elapsed = now - self._player['game_over_time']
            if elapsed < 0.7:
                pass  # jingle plays over the live board
            elif elapsed < 1.9:
                # Flash the whole frame at 10 Hz for a dramatic crash effect
                if int((elapsed - 0.7) / 0.1) % 2:
                    frame ^= 1
            else:
                # Full-screen static display: "GAME / OVER", lines, high score
                frame[:, :] = 0
                text.write_centered(frame, 'GAME', y=2, size=5)
                text.write_centered(frame, 'OVER', y=9, size=5)
                score_str = str(self._player['lines'])
                text.write_centered(frame, score_str, y=16, size=6)
                hi_str = 'HI ' + str(self._best)
                text.write_centered(frame, hi_str, y=23, size=5)

        return frame