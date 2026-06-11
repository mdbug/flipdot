import numpy as np
import time
import text
import human_pose
from autodrum import AutoDrum

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
    * Melody band: columns 0–6 (7 px), inverted — pitch stripes appear dark
      on a lit background.
    * Separator : column 7, always lit.
    * Game board: columns 8–27, 10 cells × 2 px wide (20 px), 14 rows tall.

    Gesture controls
    ----------------
    * Right index finger X  → steer piece left/right (auto-repeat 150 ms)
    * Left hand raised      → rotate piece (rising-edge trigger)
    * Arms crossed          → exit to menu (handled by main loop as usual)

    Game-over screen
    ----------------
    Flashes the panel, then shows "GAME / OVER" + lines cleared.
    Raise left hand (or wait 10 s) to restart.

    AI attract mode
    ---------------
    When no person is detected, an AI takes over after AI_TAKEOVER_DELAY
    seconds.  The AI evaluates every possible (rotation, column) placement
    with a classic heuristic and steers the falling piece toward the best
    one using the same buffered-input path as human gestures.
    """

    AI_TAKEOVER_DELAY = 5.0   # seconds of absence before AI takes over

    def __init__(self, width, height, mode_manager):
        # super().__init__ → _load_song(0) → our override → super()._load_song(4)
        # which calls _tetris_background() → _init_player_state() → _player_spawn().
        # All game state is ready by the time super() returns.
        super().__init__(width, height, mode_manager)
        # Gesture edge-detection state (set after super so init chain is clean)
        self._left_raise_armed = True   # True  = ready to trigger on next raise
        self._drop_armed = True         # True  = ready to trigger on next low-Y
        self._last_move_time = 0.0      # timestamp of last horizontal step
        self._pending = {'move': 0, 'rotate': False}  # buffered per-step intent
        self._jingle_events = []        # (due_time, note_name) death-jingle queue
        # AI attract-mode state
        self._ai_target = None          # (target_col, target_rot_idx) for current piece
        self._ai_piece_id = None        # id() of piece when _ai_target was computed
        self._last_person_time = None   # timestamp of last frame with a detected person
        self._ai_rotate_cooldown = 0.0  # timestamp of last AI-requested rotation

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
            'flash':          [],
            'flash_ticks':    0,
            'lines':          0,
            'game_over':      False,
            'game_over_time': None,
        }
        self._ai_target = None
        self._ai_piece_id = None
        self._last_person_time = None
        self._ai_rotate_cooldown = 0.0
        self._player_spawn()

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

    def _trigger_game_over(self):
        """Set game-over state and schedule the descending death jingle."""
        now = time.time()
        self._player['game_over'] = True
        self._player['game_over_time'] = now
        notes = ['a5', 'g5', 'f5', 'e5', 'd5', 'c5', 'b4', 'a4']
        self._jingle_events = [
            (now + i * 0.08, note) for i, note in enumerate(notes)
        ]

    def _player_spawn(self):
        """Spawn a random tetromino centred at the top."""
        t = self._tetris
        shapes = list(self.TETROMINOES.values())
        shape = shapes[int(self.rng.integers(len(shapes)))]
        all_rots = self._rotations(shape)
        cells = all_rots[0]
        piece_width = max(dc for _, dc in cells) + 1
        col = (t['ncols'] - piece_width) // 2
        row = -(max(dr for dr, _ in cells) + 1)
        piece = {
            'cells':    cells,
            'row':      row,
            'col':      col,
            'all_rots': all_rots,
            'rot_idx':  0,
        }
        self._player['piece'] = piece
        # Game over when spawn position is already blocked
        if not self._player_tet_fits(cells, row, col):
            self._trigger_game_over()
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

    def _player_lock(self):
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
            self._trigger_game_over()
            return
        full = list(np.flatnonzero(self._player['board'].all(axis=1)))
        if full:
            self._player['flash'] = full
            self._player['flash_ticks'] = 0
        else:
            self._player_spawn()

    def _player_hard_drop(self):
        """Drop the piece to the lowest valid row and lock it."""
        p = self._player['piece']
        if p is None:
            return
        while self._player_tet_fits(p['cells'], p['row'] + 1, p['col']):
            p['row'] += 1
        self._player_lock()
        self._bg_frame = self._render_player_board()

    def _restart_game(self):
        """Reset board to empty state and spawn the first piece."""
        t = self._tetris
        self._player['board'][:] = False
        self._player['piece'] = None
        self._player['flash'] = []
        self._player['flash_ticks'] = 0
        self._player['lines'] = 0
        self._player['game_over'] = False
        self._player['game_over_time'] = None
        self._left_raise_armed = False
        self._drop_armed = True
        self._pending = {'move': 0, 'rotate': False}
        self._jingle_events = []
        self._ai_target = None
        self._ai_piece_id = None
        self._last_person_time = None
        self._ai_rotate_cooldown = 0.0
        self._player_spawn()
        self._bg_frame = self._render_player_board()

    # ------------------------------------------------------------------
    # _tetris_step override — gravity / line-clear on each beat
    # ------------------------------------------------------------------

    def _tetris_step(self):
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
                    self._hit('crash', time.time())
                self._player_spawn()
        elif self._player['piece'] is not None:
            p = self._player['piece']
            if self._player_tet_fits(p['cells'], p['row'] + 1, p['col']):
                p['row'] += 1
            else:
                self._player_lock()
        else:
            self._player_spawn()

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
        """Heuristic score for a board state — higher is better."""
        col_heights = np.zeros(ncols, dtype=int)
        for col in range(ncols):
            for row in range(nrows):
                if board[row, col]:
                    col_heights[col] = nrows - row
                    break
        aggregate_height = int(col_heights.sum())
        complete_lines = int(board.all(axis=1).sum())
        holes = 0
        for col in range(ncols):
            h = col_heights[col]
            if h == 0:
                continue
            top_row = nrows - h
            for row in range(top_row + 1, nrows):
                if not board[row, col]:
                    holes += 1
        bumpiness = int(np.abs(np.diff(col_heights)).sum())
        return complete_lines * 100 - aggregate_height * 0.51 - holes * 8 - bumpiness * 1.5

    def _ai_best_placement(self):
        """Return (target_col, target_rot_idx) that maximises board score."""
        t = self._tetris
        p = self._player['piece']
        board = self._player['board']
        best_score = None
        best_col = p['col']
        best_rot = p['rot_idx']
        for rot_idx, cells in enumerate(p['all_rots']):
            piece_max_dc = max(dc for _, dc in cells)
            for col in range(0, t['ncols'] - piece_max_dc):
                # Simulate drop from above the board so every column is
                # evaluated correctly regardless of where the piece currently is.
                row = -(max(dr for dr, _ in cells) + 1)
                while self._player_tet_fits(cells, row + 1, col):
                    row += 1
                # Skip placements that land entirely above the visible board
                if row + max(dr for dr, _ in cells) < 0:
                    continue
                # Stamp onto a copy and score
                temp = board.copy()
                valid = True
                for dr, dc in cells:
                    r, cc = row + dr, col + dc
                    if r < 0:
                        valid = False
                        break
                    temp[r, cc] = True
                if not valid:
                    continue
                score = self._ai_score_board(temp, t['ncols'], t['nrows'])
                if best_score is None or score > best_score:
                    best_score = score
                    best_col = col
                    best_rot = rot_idx
        return best_col, best_rot

    def _handle_ai(self, now):
        """Drive the current piece toward the AI's chosen placement."""
        p = self._player['piece']
        if p is None:
            return
        # Recompute target whenever a new piece spawns
        if id(p) != self._ai_piece_id:
            self._ai_target = self._ai_best_placement()
            self._ai_piece_id = id(p)
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
                self._left_raise_armed = False
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

        # ---- Hard drop removed (too easy to trigger accidentally) ----

    def get_frame(self, pose_results):
        now = time.time()
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song['bpm'] / song['subdivisions']
        step_interval /= (1 + 0.08 * (self._player['lines'] // 5))

        # Handle player gestures before advancing the sequencer so that
        # a hard-drop locks the piece into _bg_frame before it renders.
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

        # Separator line between melody (cols 0–7) and game board (cols 8–27)
        frame[:, 7] = 1

        # Invert the melody column so it reads as light-on-dark
        frame[:, :7] ^= 1

        # Live score at the top of the melody column, XOR'd with melody.
        if not self._player['game_over']:
            score_str = str(min(self._player['lines'], 99))
            score_w = len(score_str) * 4 - 1
            score_x = max(0, (7 - score_w) // 2)
            score_layer = np.ones((self.height, 7), dtype=np.uint8)
            text.write(score_layer, score_str, x=score_x, y=1, size=5, color=0)
            frame[:, :7] ^= score_layer

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
                # Full-screen static display: "GAME" / "OVER" + large score
                frame[:, :] = 0
                # Each word = 4 chars × (3 px + 1 px spacing) − 1 trailing = 15 px
                word_x = (self.width - 15) // 2
                text.write(frame, 'GAME', x=word_x, y=2, size=5)
                text.write(frame, 'OVER', x=word_x, y=9, size=5)
                score_str = str(self._player['lines'])
                # size-6 digits: 5 px wide + 1 px spacing per char, minus trailing
                score_width = len(score_str) * 6 - 1
                score_x = max(0, (self.width - score_width) // 2)
                text.write(frame, score_str, x=score_x, y=19, size=6)

        return frame
