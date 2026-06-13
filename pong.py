import numpy as np
import time
import text
import human_pose
from autodrum import AutoDrum


class Pong(AutoDrum):
    """Pong with gesture control, an AI opponent, and beat-locked physics.

    The ball moves one tick per sequencer step, so its motion IS the
    metronome.  Game sounds are diegetic — drawn into the background at
    the spot where they happen, so they are loud (lots of dots), look
    like impacts, and clean themselves up because the background is
    recomposed from game state on every step:

    * ball movement → the steady tick of background flips
    * wall bounce   → a 2×10 flash of the wall at the impact point
    * paddle hit    → a flash hugging the paddle face (the vertical
                      mirror of the wall flash), then a semicircular
                      ripple radiating from the impact point
    * serve         → a centre burst as the ball launches
    * point scored  → the whole panel becomes the hit: a goal-mouth
                      shockwave, full-screen sweeps, score flashes, and
                      loud panel-wide percussion before the next serve
    * game won      → a full-screen fanfare that expands, sweeps, and
                      flashes across the whole display, then lands on
                      the result screen

    Rallies accelerate the tempo (and therefore the ball) the same way
    Tetris lines speed the song — beat-locked physics gets faster with
    the music for free.

    Layout (h×w panel)
    ------------------
    The playfield uses the whole panel: paddles 2 px wide at the edges,
    2×2 ball, score digits at the top.

    Controls
    --------
    * Right index finger height → your paddle (right side).  Movement is
      buffered and applied on the step grid, max PLAYER_SPEED px/step.
    * Left hand raised on the win screen → rematch (or wait 8 s).
    * Arms crossed → exit to menu (handled by the main loop as usual).

    When no person is detected for AI_TAKEOVER_DELAY seconds, the AI
    plays both paddles as an attract mode.  The AI is deliberately
    beatable: it only tracks the ball on its own half, moves slower
    than you, and aims with a small random error — sharp angles win.
    """

    AI_TAKEOVER_DELAY = 3.0
    WIN_SCORE = 5
    PAD_H = 6          # paddle height in px
    PAD_W = 2          # paddle width in px
    BALL = 2           # ball is BALL×BALL px
    PLAYER_SPEED = 3   # px per step
    AI_SPEED = 2       # px per step
    POINT_CELEBRATION_TIME = 1.25
    WIN_FANFARE_TIME = 2.0

    SONGS = [
        {
            'name': 'PONG',
            'bpm': 116,
            'subdivisions': 4,   # ball ticks on 16th notes
            'sections': [
                (0, [set() for _ in range(16)]),   # no backing drums —
            ],                                     # the ball is the beat
            'bg': '_pong_background',
            'bg_step': '_pong_step',
            # Voice stripes are retained for AutoDrum compatibility;
            # Pong's score/win moments use full-screen events below.
            'melody': {
                'pitches': (('g5', 79), ('e5', 76), ('c5', 72),
                            ('g4', 67), ('e4', 64), ('c4', 60)),
                'long': ('g5', 'c5', 'c4'),
            },
        },
    ]

    def __init__(self, width, height, mode_manager):
        self._celebration_hits = []  # (due_time, instrument_name) queue;
        #                             exists before the super() init chain
        super().__init__(width, height, mode_manager)
        self._left_raise_armed = True
        self._human_target = None      # latest finger-derived paddle top
        self._last_person_time = None

    # ------------------------------------------------------------------
    # Match setup
    # ------------------------------------------------------------------

    def _pong_background(self):
        """Initialise the match; return the first rendered frame."""
        h, w = self.height, self.width
        self._pong = {
            'ft': 0, 'fb': h,            # playfield top/bottom (excl.)
            'ball': [w // 2 - 1, h // 2],
            'v': [0, 0],
            'lpad': (h - self.PAD_H) // 2,
            'rpad': (h - self.PAD_H) // 2,
            'aim_l': 0, 'aim_r': 0,      # AI aiming error, resampled per rally
            'score': [0, 0],             # [left, right]
            'rally': 0,
            'fx': [],                    # transient impact effects
            'serve_wait': 8,             # steps until serve (ball blinks)
            'serve_dir': 1 if self.rng.random() < 0.5 else -1,
            'serve_blink': 0,
            'celebration': None,
            'winner': None, 'win_time': None, 'win_text': '',
        }
        return self._compose_pong_bg()

    # ------------------------------------------------------------------
    # Per-step game update — every flip lands on the musical grid
    # ------------------------------------------------------------------

    def _human_active(self, now):
        return (self._last_person_time is not None
                and now - self._last_person_time < self.AI_TAKEOVER_DELAY)

    def _ai_paddle_target(self, side):
        """Where the AI wants its paddle top.  side: 0 = left, 1 = right."""
        p = self._pong
        x, y = p['ball']
        w = self.width
        centre = (p['ft'] + p['fb'] - self.PAD_H) // 2
        approaching = p['v'][0] < 0 if side == 0 else p['v'][0] > 0
        on_my_half = x < w // 2 if side == 0 else x + self.BALL > w // 2
        if p['v'][0] != 0 and approaching and on_my_half:
            aim = p['aim_l'] if side == 0 else p['aim_r']
            return y + self.BALL // 2 - self.PAD_H // 2 + aim
        return centre

    def _move_paddle(self, key, target, speed):
        p = self._pong
        target = max(p['ft'], min(p['fb'] - self.PAD_H, int(round(target))))
        p[key] += max(-speed, min(speed, target - p[key]))

    def _pong_step(self, now):
        """Advance the match one sequencer step."""
        p = self._pong
        if p['winner'] is not None:
            # Keep aging the impact effects so the winning point's
            # shockwave finishes rolling behind the fanfare.
            for f in p['fx']:
                f['ttl'] -= 1
            p['fx'] = [f for f in p['fx'] if f['ttl'] > 0]
            self._bg_frame = self._compose_pong_bg()
            return

        # Age impact effects.  Every transition they cause lands on the
        # grid, and expiry cleans up automatically because the
        # background is recomposed from game state each step.
        for f in p['fx']:
            f['ttl'] -= 1
        p['fx'] = [f for f in p['fx'] if f['ttl'] > 0]

        # Paddles chase their targets at limited speed — quantised to
        # the grid, so paddle motion clicks in rhythm too.  The human
        # (right-hand tracking) plays the RIGHT paddle; the left paddle
        # is always the AI.
        self._move_paddle('lpad', self._ai_paddle_target(0), self.AI_SPEED)
        if self._human_active(now) and self._human_target is not None:
            self._move_paddle('rpad', self._human_target, self.PLAYER_SPEED)
        else:
            self._move_paddle('rpad', self._ai_paddle_target(1), self.AI_SPEED)

        if p['serve_wait'] > 0:
            p['serve_wait'] -= 1
            p['serve_blink'] += 1
            if p['serve_wait'] == 0:
                # Wipe celebration/drum residue in the SAME frame as
                # the serve burst, so cleanup is absorbed by the launch.
                mc0, mc1 = self._voice_span
                self.state[:, mc0:mc1] = 0
                self._decay_events = []
                self._voice_decay = []
                self._voice_shimmer = None
                p['celebration'] = None
                p['ball'] = [self.width // 2 - 1,
                             (p['ft'] + p['fb'] - self.BALL) // 2]
                p['v'] = [2 * p['serve_dir'],
                          int(self.rng.integers(-1, 2))]
                p['aim_l'], p['aim_r'] = self._new_aim(), self._new_aim()
                p['fx'].append({'kind': 'serve', 'ttl': 1, 'ttl0': 1})
            self._bg_frame = self._compose_pong_bg()
            return

        # Ball flight
        x, y = p['ball']
        vx, vy = p['v']
        x_old = x
        x += vx
        y += vy
        # Walls
        if y < p['ft']:
            y = 2 * p['ft'] - y
            vy = -vy
            p['fx'].append({'kind': 'wall', 'ttl': 1, 'ttl0': 1,
                            'x': x, 'top': True})
        elif y + self.BALL > p['fb']:
            y = 2 * (p['fb'] - self.BALL) - y
            vy = -vy
            p['fx'].append({'kind': 'wall', 'ttl': 1, 'ttl0': 1,
                            'x': x, 'top': False})
        # Left paddle plane (only on the crossing step, never behind it)
        lplane = self.PAD_W
        if vx < 0 and x_old > lplane >= x:
            if y + self.BALL > p['lpad'] and y < p['lpad'] + self.PAD_H:
                x = lplane
                vx = -vx
                vy = self._english(y, p['lpad'])
                p['rally'] += 1
                p['aim_r'] = self._new_aim()
                p['fx'].append({'kind': 'pad', 'ttl': 2, 'ttl0': 2,
                                'side': 0, 'y': p['lpad'],
                                'cy': y + self.BALL // 2})
        # Right paddle plane
        rplane = self.width - self.PAD_W
        if vx > 0 and x_old + self.BALL < rplane <= x + self.BALL:
            if y + self.BALL > p['rpad'] and y < p['rpad'] + self.PAD_H:
                x = rplane - self.BALL
                vx = -vx
                vy = self._english(y, p['rpad'])
                p['rally'] += 1
                p['aim_l'] = self._new_aim()
                p['fx'].append({'kind': 'pad', 'ttl': 2, 'ttl0': 2,
                                'side': 1, 'y': p['rpad'],
                                'cy': y + self.BALL // 2})
        # Goals
        if x + self.BALL <= 0:
            self._point(1, now, 0, y + self.BALL // 2)
        elif x >= self.width:
            self._point(0, now, self.width - 1, y + self.BALL // 2)
        else:
            p['ball'] = [x, y]
            p['v'] = [vx, vy]

        self._bg_frame = self._compose_pong_bg()

    def _english(self, ball_y, pad_y):
        """Bounce angle from where the ball met the paddle."""
        rel = (ball_y + self.BALL / 2) - (pad_y + self.PAD_H / 2)
        return max(-2, min(2, int(round(rel / (self.PAD_H / 2) * 2))))

    def _new_aim(self):
        """AI aiming error: small enough to rally, big enough to lose."""
        return int(self.rng.integers(-3, 4))

    def _point(self, side, now, exit_x, exit_y):
        """side scored: shockwave rings roll out of the goal mouth."""
        p = self._pong
        p['score'][side] += 1
        p['rally'] = 0
        p['fx'].append({'kind': 'burst', 'ttl': 8, 'ttl0': 8,
                        'cx': exit_x, 'cy': exit_y})
        self._start_celebration('point', side, now, exit_x, exit_y)
        if p['score'][side] >= self.WIN_SCORE:
            self._trigger_win(side, now)
            return
        p['serve_wait'] = 12
        p['serve_blink'] = 0
        # Serve toward the player who just lost the point
        p['serve_dir'] = -1 if side == 1 else 1
        p['ball'] = [self.width // 2 - 1, (p['ft'] + p['fb'] - self.BALL) // 2]
        p['v'] = [0, 0]

    def _trigger_win(self, side, now):
        p = self._pong
        p['winner'] = side
        p['win_time'] = now
        human_won = side == 1 and self._human_active(now)
        p['win_text'] = 'YOU WIN' if human_won else 'AI WINS'
        exit_x = 0 if side == 1 else self.width - 1
        exit_y = p['ball'][1] + self.BALL // 2
        self._start_celebration('win', side, now, exit_x, exit_y)

    def _restart_match(self):
        self._celebration_hits = []
        self._left_raise_armed = False
        # _load_song resets the sequencer and, via the 'bg' hook,
        # rebuilds the whole match state.
        self._load_song(0)

    def _start_celebration(self, kind, side, now, exit_x, exit_y):
        duration = (self.WIN_FANFARE_TIME if kind == 'win'
                    else self.POINT_CELEBRATION_TIME)
        self._pong['celebration'] = {
            'kind': kind,
            'side': side,
            'start': now,
            'duration': duration,
            'cx': int(exit_x),
            'cy': int(max(0, min(self.height - 1, exit_y))),
            'score': tuple(self._pong['score']),
        }
        if kind == 'win':
            names = (['crash', 'clap', 'tom', 'kick', 'clap', 'crash']
                     if side == 1
                     else ['crash', 'kick', 'tom', 'clap', 'kick', 'crash'])
            offsets = [0.0, 0.16, 0.32, 0.50, 0.72, 1.02]
        else:
            names = (['crash', 'clap', 'tom', 'crash'] if side == 1
                     else ['crash', 'kick', 'tom', 'clap'])
            offsets = [0.0, 0.14, 0.30, 0.48]
        self._celebration_hits = [(now + offset, name)
                                  for offset, name in zip(offsets, names)]

    def _render_celebration(self, now):
        celebration = self._pong.get('celebration')
        if celebration is None:
            return None
        elapsed = now - celebration['start']
        if elapsed >= celebration['duration']:
            if celebration['kind'] != 'win':
                self._pong['celebration'] = None
            return None

        height, width = self.height, self.width
        frame = np.zeros((height, width), dtype=np.uint8)
        yy, xx = np.ogrid[:height, :width]
        origin_x, origin_y = celebration['cx'], celebration['cy']
        progress = max(0.0, min(1.0, elapsed / celebration['duration']))
        score_side = celebration['side']
        human_scored = score_side == 1

        distance2 = (yy - origin_y) ** 2 + (xx - origin_x) ** 2
        lead = 3 + int(progress * (width + height))
        for radius in (lead, lead - 7, lead - 14):
            if radius > 1:
                frame[(distance2 >= (radius - 1) ** 2)
                      & (distance2 <= (radius + 1) ** 2)] = 1

        if celebration['kind'] == 'point':
            self._draw_point_celebration(frame, celebration, elapsed,
                                         progress, human_scored)
        else:
            self._draw_win_celebration(frame, celebration, elapsed,
                                       progress, human_scored)
        return frame

    def _draw_point_celebration(self, frame, celebration, elapsed, progress,
                                human_scored):
        height, width = self.height, self.width
        yy, xx = np.indices((height, width))
        sweep = int(progress * (height + 8))
        if human_scored:
            frame[height - min(height, sweep):height, :] = 1
            frame[((yy + xx + int(elapsed * 18)) % 6) < 2] ^= 1
        else:
            frame[:min(height, sweep), :] = 1
            frame[((yy - xx + int(elapsed * 18)) % 6) < 2] ^= 1

        band_width = max(3, int(width * (1.0 - progress)))
        if human_scored:
            frame[:, max(0, width - band_width):width] = 1
        else:
            frame[:, 0:min(width, band_width)] = 1

        if 0.28 <= elapsed <= 0.75:
            self._draw_big_score(frame, celebration['score'])
        if elapsed >= 0.75 and int(elapsed * 12) % 2 == 0:
            frame[height // 2 - 2:height // 2 + 2, :] = 1

    def _draw_win_celebration(self, frame, celebration, elapsed, progress,
                              human_won):
        height, width = self.height, self.width
        yy, xx = np.indices((height, width))
        center_y, center_x = height // 2, width // 2
        fan = np.abs(yy - center_y) <= (np.abs(xx - center_x) * progress + 1)
        frame[fan] = 1

        sweep = int(progress * (height + width))
        if human_won:
            frame[(height - 1 - yy) + np.abs(xx - center_x) < sweep] ^= 1
        else:
            frame[yy + np.abs(xx - center_x) < sweep] ^= 1

        beat = int(elapsed / 0.16)
        if beat % 2 == 0:
            frame[(yy + xx + beat) % 4 == 0] = 1
        else:
            frame[(yy - xx + beat) % 4 == 0] = 1

        if elapsed >= 1.05:
            msg = self._pong['win_text']
            msg_w = self._text_width(msg, size=5)
            text.write(frame, msg, x=max(0, (width - msg_w) // 2),
                       y=6, size=5)
        if elapsed >= 1.45 and int(elapsed * 12) % 2 == 0:
            frame[:, :] ^= 1

    def _draw_big_score(self, frame, scores):
        width = self.width
        score_str = self._score_text(scores)
        score_w = self._text_width(score_str, size=6)
        text.write(frame, score_str, x=max(0, (width - score_w) // 2),
                   y=10, size=6)

    @staticmethod
    def _score_text(scores):
        return '%d:%d' % (scores[0], scores[1])

    @staticmethod
    def _text_width(value, size=5, spacing=1):
        font = text.FONTS[size]
        if not value:
            return 0
        return sum(font[ch].shape[1] for ch in value) + spacing * (len(value) - 1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _compose_pong_bg(self):
        h, w = self.height, self.width
        p = self._pong
        bg = np.zeros((h, w), dtype=np.uint8)
        # Paddles
        bg[p['lpad']:p['lpad'] + self.PAD_H, 0:self.PAD_W] = 1
        bg[p['rpad']:p['rpad'] + self.PAD_H, w - self.PAD_W:w] = 1
        # Ball (blinks at centre during the serve count-in)
        draw_ball = (p['serve_wait'] == 0 or p['serve_blink'] % 4 < 2)
        if p['winner'] is None and draw_ball:
            x, y = p['ball']
            bg[max(0, y):y + self.BALL, max(0, x):max(0, x + self.BALL)] = 1
        # Impact effects, drawn where they happened
        for f in p['fx']:
            if f['kind'] == 'pad':
                face = self.PAD_W if f['side'] == 0 else w - self.PAD_W - 1
                direction = 1 if f['side'] == 0 else -1
                if f['ttl'] == f['ttl0']:          # first step: face flash
                    r0 = max(p['ft'], f['y'] - 2)
                    r1 = min(p['fb'], f['y'] + self.PAD_H + 2)
                    c0 = face if direction == 1 else face - 1
                    bg[r0:r1, max(0, c0):c0 + 2] = 1
                else:                              # second step: ripple
                    rad = 5
                    for dr in range(-rad, rad + 1):
                        dc = int(round((rad * rad - dr * dr) ** 0.5))
                        rr = f['cy'] + dr
                        for cc in (face + direction * dc,
                                   face + direction * max(0, dc - 1)):
                            if p['ft'] <= rr < p['fb'] and 0 <= cc < w:
                                bg[rr, cc] = 1
            elif f['kind'] == 'wall':
                c0 = max(0, f['x'] - 4)
                if f['top']:
                    bg[p['ft']:p['ft'] + 2, c0:c0 + 10] = 1
                else:
                    bg[p['fb'] - 2:p['fb'], c0:c0 + 10] = 1
            elif f['kind'] == 'burst':
                # Double shockwave from the goal mouth: a leading and a
                # trailing wavefront, rolling clear across the panel
                age = f['ttl0'] - f['ttl']
                yy, xx = np.ogrid[:h, :w]
                d2 = (yy - f['cy']) ** 2 + (xx - f['cx']) ** 2
                lead = age * 4 + 4
                for rad in (lead, lead - 9):
                    if rad > 2:
                        bg[(d2 >= (rad - 1) ** 2)
                           & (d2 <= (rad + 1) ** 2)] = 1
            elif f['kind'] == 'serve':
                mr = (p['ft'] + p['fb']) // 2
                bg[mr - 3:mr + 3, w // 2 - 3:w // 2 + 3] = 1
        # Score digits at the top (drawn last, over the ball — classic)
        score_str = self._score_text(p['score'])
        score_w = self._text_width(score_str, size=5)
        text.write(bg, score_str, x=max(0, (w - score_w) // 2), y=1, size=5)
        return bg

    # ------------------------------------------------------------------
    # Gestures + frame assembly
    # ------------------------------------------------------------------

    def _handle_gestures(self, pose_results, now):
        p = self._pong
        if p['winner'] is not None:
            left_raised = human_pose.is_left_hand_raised(pose_results)
            if left_raised and self._left_raise_armed:
                self._restart_match()
            elif not left_raised:
                self._left_raise_armed = True
            if p['win_time'] is not None and now - p['win_time'] >= 8.0:
                self._restart_match()
            return
        person_present = (
            pose_results is not None
            and getattr(pose_results, 'pose_landmarks', None) is not None
        )
        if not person_present:
            return
        self._last_person_time = now
        finger_x, finger_y = human_pose.get_right_index_finger_position(
            pose_results)
        if finger_y is not None:
            # Finger height → paddle CENTRE; buffered, applied on the grid
            self._human_target = (finger_y * (p['fb'] - p['ft'])
                                  + p['ft'] - self.PAD_H / 2)

    def get_frame(self, pose_results):
        now = time.time()
        p = self._pong
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song['bpm'] / song['subdivisions']
        # Rallies accelerate the tick — and therefore the ball
        step_interval /= (1 + 0.05 * min(p['rally'], 10))

        self._handle_gestures(pose_results, now)

        due = [e for e in self._celebration_hits if e[0] <= now]
        self._celebration_hits = [e for e in self._celebration_hits
                                  if e[0] > now]
        for _, instrument_name in due:
            self._hit(instrument_name, now)

        self._tick_voice(now)
        # The sequencer keeps ticking after a win too: the pattern is
        # empty, but bg_step still ages the final shockwave.
        self._advance_sequencer(now, step_interval)

        frame = self.state.copy()
        if self._bg_frame is not None:
            frame ^= self._bg_frame

        celebration_frame = self._render_celebration(now)
        if celebration_frame is not None:
            frame = celebration_frame

        # Mode name overlay for the first 2 s
        if celebration_frame is None and now - self.song_start_time < 2.0:
            frame[:6, :] = 0
            text.write(frame, song['name'], x=1, y=0, size=5)

        # Win screen: fanfare plays over the final court for a moment,
        # then the static result
        if p['winner'] is not None and now - p['win_time'] >= self.WIN_FANFARE_TIME:
            frame[:, :] = 0
            msg = p['win_text']
            msg_w = self._text_width(msg, size=5)
            text.write(frame, msg, x=max(0, (self.width - msg_w) // 2),
                       y=6, size=5)
            score_str = self._score_text(p['score'])
            score_w = self._text_width(score_str, size=6)
            text.write(frame, score_str,
                       x=max(0, (self.width - score_w) // 2), y=15, size=6)

        return frame