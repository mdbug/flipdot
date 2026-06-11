import numpy as np
import time
import text
import human_pose
from autodrum import AutoDrum

# Index of the TETRIS song in AutoDrum.SONGS
_TETRIS_SONG_IDX = 4


class Tetris(AutoDrum):
    """Playable Tetris mode with gesture controls.

    Inherits AutoDrum's Korobeiniki sequencer and flip-dot audio engine.
    Replaces the AI Tetris demo with a player-controlled game.

    Layout (28×28 panel)
    --------------------
    * Game board : columns 0–19, 10 cells wide × 2 px each (20 px).
    * Melody band: columns 20–27 (right 8 px) — pitch stripes XOR here,
      keeping the game area free of note flashes while still clicking.
    * Bottom row (row 27): sequencer step cursor (inherited).

    Gesture controls
    ----------------
    * Right index finger X  → steer piece left/right (auto-repeat 150 ms)
    * Left hand raised      → rotate piece (rising-edge trigger)
    * Finger Y > 0.7        → hard-drop piece (rising-edge trigger)
    * Arms crossed          → exit to menu (handled by main loop as usual)

    Game-over screen
    ----------------
    Shows "OVER" + lines cleared.  Raise left hand (or wait 5 s) to restart.
    """

    def __init__(self, width, height, mode_manager):
        # super().__init__ → _load_song(0) → our override → super()._load_song(4)
        # which calls _tetris_background() → _init_player_state() → _player_spawn().
        # All game state is ready by the time super() returns.
        super().__init__(width, height, mode_manager)
        # Gesture edge-detection state (set after super so init chain is clean)
        self._left_raise_armed = True   # True  = ready to trigger on next raise
        self._drop_armed = True         # True  = ready to trigger on next low-Y
        self._last_move_time = 0.0      # timestamp of last horizontal step

    # ------------------------------------------------------------------
    # _load_song override — always loads TETRIS song, then re-pins the
    # melody band to the right 8 columns so the game board stays clean.
    # ------------------------------------------------------------------

    def _load_song(self, index):
        super()._load_song(_TETRIS_SONG_IDX)
        # Reposition every melody-voice instrument to the right 8 px.
        mc0 = 0            # melody in left 8 px
        mc1 = 8
        for name in self._melody_names:
            inst = self.instruments[name]
            r0, r1, _, _ = inst['area']
            inst['area'] = (r0, r1, mc0, mc1)
        self._voice_span = (mc0, mc1)

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
            self._player['game_over'] = True
            self._player['game_over_time'] = time.time()
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
            self._player['game_over'] = True
            self._player['game_over_time'] = time.time()
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
        self._player_spawn()
        self._bg_frame = self._render_player_board()

    # ------------------------------------------------------------------
    # _tetris_step override — gravity / line-clear on each beat
    # ------------------------------------------------------------------

    def _tetris_step(self):
        """Called on every sequencer step; acts only on full beats."""
        t = self._tetris
        song = self.SONGS[self.song_index]
        t['ticks'] += 1
        if t['ticks'] % song['subdivisions']:
            return   # only act on full beats

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
                self._player['lines'] += len(flash)
                self._player['flash'] = []
                self._player['flash_ticks'] = 0
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

        # ---- Rotate: rising edge of left hand raised ----
        left_raised = human_pose.is_left_hand_raised(pose_results)
        if left_raised and self._left_raise_armed:
            self._player_rotate()
            self._left_raise_armed = False
        elif not left_raised:
            self._left_raise_armed = True

        # ---- Steer: right index finger X → target column ----
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
                direction = 1 if target_col > p['col'] else -1
                new_col = p['col'] + direction
                if self._player_tet_fits(p['cells'], p['row'], new_col):
                    p['col'] = new_col
                    self._last_move_time = now
                    self._bg_frame = self._render_player_board()

        # ---- Hard drop removed (too easy to trigger accidentally) ----

    # ------------------------------------------------------------------
    # get_frame — sequencer + sound (from AutoDrum) + player display
    # ------------------------------------------------------------------

    def get_frame(self, pose_results):
        now = time.time()
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song['bpm'] / song['subdivisions']

        # Handle player gestures before advancing the sequencer so that
        # a hard-drop locks the piece into _bg_frame before it renders.
        self._handle_gestures(pose_results, now)

        if self._player['game_over']:
            # Silence any ringing melody note and freeze the sequencer.
            if self._voice_shimmer is not None and self._voice_area is not None:
                r0, r1, c0, c1 = self._voice_area
                self.state[r0:r1, c0:c1] ^= self._voice_shimmer
                self._voice_shimmer = None
            self._voice_decay = []
            self._decay_events = []
            mc0, mc1 = self._voice_span
            self.state[:, mc0:mc1] = 0
        else:
            # Fire any due decay-tail flips (cymbal rattle / ring-out)
            if self._decay_events:
                due = [e for e in self._decay_events if e[0] <= now]
                self._decay_events = [e for e in self._decay_events if e[0] > now]
                for _, name, density in due:
                    self._scatter_flip(name, density)

            # Melody ring-out shimmer (self-cancelling per-tick)
            while self._voice_decay and self._voice_decay[0][0] <= now:
                _, density = self._voice_decay.pop(0)
                r0, r1, c0, c1 = self._voice_area
                if self._voice_shimmer is not None:
                    self.state[r0:r1, c0:c1] ^= self._voice_shimmer
                    self._voice_shimmer = None
                if density > 0:
                    self._voice_shimmer = (
                        self.rng.random((r1 - r0, c1 - c0)) < density
                    ).astype(np.uint8)
                    self.state[r0:r1, c0:c1] ^= self._voice_shimmer

            # Advance the sequencer (calls our _tetris_step on each beat)
            while now >= self.next_step_time:
                repeats, pattern = self._section()
                prev = self.step
                self.step = (self.step + 1) % len(pattern)

                for name in pattern[self.step]:
                    self._hit(name, now)

                if 'bg_step' in song:
                    getattr(self, song['bg_step'])()

                if prev == len(pattern) - 1:
                    self.section_repeats += 1
                    if repeats > 0 and self.section_repeats >= repeats:
                        self.section_index = (
                            (self.section_index + 1) % len(song['sections'])
                        )
                        self.section_repeats = 0
                        self.step = -1

                self.next_step_time += step_interval
                if now - self.next_step_time > 1.0:
                    self.next_step_time = now + step_interval

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
            text.write(frame, score_str, x=score_x, y=1, size=5, color=0)

        # Game-over overlay
        if self._player['game_over'] and self._player['game_over_time'] is not None:
            elapsed = now - self._player['game_over_time']
            if elapsed < 1.2:
                # Flash the whole frame at 10 Hz for a dramatic crash effect
                if int(elapsed / 0.1) % 2:
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
