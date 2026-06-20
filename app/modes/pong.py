import numpy as np
import time
import app.services.text as text
import app.services.human_pose as human_pose


class Pong:
    """Pong with gesture control and smooth continuous motion.

    Physics runs every rendered frame (time-based), so paddle and ball
    movement stay responsive and fluid even at varying frame rates.

    Percussion-like panel effects are still drawn at impact locations and
    naturally clean up over time.

    Effects:

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

    Rallies accelerate ball speed to intensify play.

    Layout (h×w panel)
    ------------------
    The playfield uses the whole panel: paddles 2 px wide at the edges,
    2×2 ball, score digits at the top.

    Controls
    --------
    * Right index finger height → your paddle (right side).  Movement is
            continuous and speed-limited, max PLAYER_SPEED px/s.
    * Left hand raised on the win screen → rematch (or wait 8 s).
    * Arms crossed → exit to menu (handled by the main loop as usual).

    When no person is detected for AI_TAKEOVER_DELAY seconds, the AI
    plays both paddles as an attract mode.  The AI is deliberately
    beatable: it only tracks the ball on its own half, moves slower
    than you, and aims with a small random error — sharp angles win.
    """

    AI_TAKEOVER_DELAY = 15.0
    WIN_SCORE = 5
    PAD_H = 6          # paddle height in px
    PAD_W = 2          # paddle width in px
    BALL = 2           # ball is BALL×BALL px
    PLAYER_SPEED = 24.0      # px/s
    CONTROLLER_SPEED = 100.0  # px/s when driven by controller
    AI_SPEED = 16.0          # px/s
    BASE_BALL_SPEED = 15.5   # px/s
    MAX_DT = 0.05            # clamp dt to avoid tunneling on hitches
    INITIAL_SERVE_DELAY = 1.0
    POINT_SERVE_DELAY = 1.5
    POINT_CELEBRATION_TIME = 1.25
    WIN_FANFARE_TIME = 2.0
    MODE_NAME_TIME = 2.0
    WIN_RESTART_TIME = 8.0

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.state = np.zeros((height, width), dtype=np.uint8)
        self.rng = np.random.default_rng()

        # Drum-like panel hits used by score/win celebrations.
        h, w = height, width
        self._instruments = {
            'kick':  {'area': (h // 2, h, 0, w),           'density': 1.0,
                      'decay': []},
            'tom':   {'area': (h // 4, h // 2, w // 2, w), 'density': 0.9,
                      'decay': [0.35, 0.15]},
            'crash': {'area': (0, h, 0, w),                'density': 0.8,
                      'decay': [0.45, 0.3, 0.18, 0.1, 0.05]},
            'clap':  {'area': (0, h // 2, 0, w),           'density': 0.5,
                      'decay': [0.3, 0.18, 0.1]},
        }
        self._decay_events = []

        self._celebration_hits = []
        self._left_raise_armed = True
        self._human_target = None
        self._last_person_time = None
        self._last_controller_input_time = None
        now = time.time()
        self.song_start_time = now
        self._last_frame_time = now
        self._reset_match(now, initial=True)

    # ------------------------------------------------------------------
    # Match setup
    # ------------------------------------------------------------------

    def _reset_match(self, now, initial=False):
        """Initialize or restart the match state."""
        h, w = self.height, self.width
        self._last_person_time = now
        self._decay_events = []
        self._celebration_hits = []
        self.state[:, :] = 0
        center_pad = (h - self.PAD_H) / 2.0
        self._pong = {
            'ft': 0,
            'fb': h,
            'ball': [w / 2.0 - self.BALL / 2.0, (h - self.BALL) / 2.0],
            'v': [0.0, 0.0],
            'lpad': center_pad,
            'rpad': center_pad,
            'aim_l': 0,
            'aim_r': 0,
            'score': [0, 0],
            'rally': 0,
            'fx': [],
            'serve_until': now + self.INITIAL_SERVE_DELAY,
            'serve_dir': 1 if self.rng.random() < 0.5 else -1,
            'celebration': None,
            'winner': None,
            'win_time': None,
            'win_text': '',
        }
        self._center_paddles()
        if initial:
            self.song_start_time = now
        self._last_frame_time = now

    def _center_paddles(self):
        p = self._pong
        center_pad = (p['ft'] + p['fb'] - self.PAD_H) / 2.0
        p['lpad'] = center_pad
        p['rpad'] = center_pad

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def _human_active(self, now):
        person_active = (
            self._last_person_time is not None
            and now - self._last_person_time < self.AI_TAKEOVER_DELAY
        )
        controller_active = (
            self._last_controller_input_time is not None
            and now - self._last_controller_input_time < self.AI_TAKEOVER_DELAY
        )
        return person_active or controller_active

    def _ai_paddle_target(self, side):
        """Where the AI wants its paddle top.  side: 0 = left, 1 = right."""
        p = self._pong
        x, y = p['ball']
        w = self.width
        centre = (p['ft'] + p['fb'] - self.PAD_H) / 2.0
        approaching = p['v'][0] < 0 if side == 0 else p['v'][0] > 0
        on_my_half = x < w // 2 if side == 0 else x + self.BALL > w // 2
        if p['v'][0] != 0 and approaching and on_my_half:
            aim = p['aim_l'] if side == 0 else p['aim_r']
            return y + self.BALL // 2 - self.PAD_H // 2 + aim
        return centre

    def _move_paddle(self, key, target, speed, dt):
        p = self._pong
        target = max(p['ft'], min(p['fb'] - self.PAD_H, float(target)))
        max_move = speed * dt
        delta = max(-max_move, min(max_move, target - p[key]))
        p[key] += delta

    def _tick_decay(self, now):
        if not self._decay_events:
            return
        due = [e for e in self._decay_events if e[0] <= now]
        self._decay_events = [e for e in self._decay_events if e[0] > now]
        for _, name, density in due:
            self._scatter_flip(name, density)

    def _scatter_flip(self, name, density):
        inst = self._instruments[name]
        r0, r1, c0, c1 = inst['area']
        if density >= 1.0:
            self.state[r0:r1, c0:c1] ^= 1
            return
        mask = self.rng.random((r1 - r0, c1 - c0)) < density
        self.state[r0:r1, c0:c1] ^= mask.astype(np.uint8)

    def _hit(self, name, now):
        inst = self._instruments.get(name)
        if inst is None:
            return
        self._scatter_flip(name, inst['density'])
        for i, tail_density in enumerate(inst['decay']):
            self._decay_events.append((now + (i + 1) * 0.05, name, tail_density))

    def _update_match(self, now, dt):
        p = self._pong

        self._tick_decay(now)

        # Age and prune impact effects.
        p['fx'] = [f for f in p['fx'] if f['until'] > now]

        if p['winner'] is not None:
            return

        if p['serve_until'] is not None:
            # Keep both paddles centered while waiting for the next serve.
            self._center_paddles()
            if now >= p['serve_until'] and p['v'][0] == 0.0 and p['v'][1] == 0.0:
                # Clear celebration/percussion residue in the launch frame.
                self.state[:, :] = 0
                self._decay_events = []
                p['celebration'] = None
                p['ball'] = [self.width / 2.0 - self.BALL / 2.0,
                             (p['ft'] + p['fb'] - self.BALL) / 2.0]
                speed = self.BASE_BALL_SPEED * (1.0 + 0.05 * min(p['rally'], 10))
                p['v'] = [speed * p['serve_dir'],
                          float(self.rng.integers(-1, 2)) * speed * 0.5]
                p['aim_l'], p['aim_r'] = self._new_aim(), self._new_aim()
                p['fx'].append({'kind': 'serve', 'start': now, 'until': now + 0.12})
                p['serve_until'] = None
            return

        self._move_paddle('lpad', self._ai_paddle_target(0), self.AI_SPEED, dt)
        if self._human_active(now) and self._human_target is not None:
            controller_active = (
                self._last_controller_input_time is not None
                and now - self._last_controller_input_time < self.AI_TAKEOVER_DELAY
            )
            speed = self.CONTROLLER_SPEED if controller_active else self.PLAYER_SPEED
            self._move_paddle('rpad', self._human_target, speed, dt)
        else:
            self._move_paddle('rpad', self._ai_paddle_target(1), self.AI_SPEED, dt)

        # Ball flight (continuous).
        x, y = p['ball']
        vx, vy = p['v']
        x_old = x
        y += vy * dt
        x = x_old + vx * dt

        # Walls
        if y < p['ft']:
            y = 2 * p['ft'] - y
            vy = -vy
            p['fx'].append({'kind': 'wall', 'x': x, 'top': True,
                            'start': now, 'until': now + 0.08})
        elif y + self.BALL > p['fb']:
            y = 2 * (p['fb'] - self.BALL) - y
            vy = -vy
            p['fx'].append({'kind': 'wall', 'x': x, 'top': False,
                            'start': now, 'until': now + 0.08})

        # Left paddle plane.
        lplane = self.PAD_W
        if vx < 0 and x_old > lplane >= x:
            if y + self.BALL > p['lpad'] and y < p['lpad'] + self.PAD_H:
                x = lplane
                vx = -vx
                p['rally'] += 1
                speed = self.BASE_BALL_SPEED * (1.0 + 0.05 * min(p['rally'], 10))
                vx = speed
                vy = self._english(y, p['lpad'], speed)
                p['aim_r'] = self._new_aim()
                p['fx'].append({'kind': 'pad', 'start': now, 'until': now + 0.18,
                                'side': 0, 'y': p['lpad'],
                                'cy': y + self.BALL // 2})

        # Right paddle plane.
        rplane = self.width - self.PAD_W
        if vx > 0 and x_old + self.BALL < rplane <= x + self.BALL:
            if y + self.BALL > p['rpad'] and y < p['rpad'] + self.PAD_H:
                x = rplane - self.BALL
                vx = -vx
                p['rally'] += 1
                speed = self.BASE_BALL_SPEED * (1.0 + 0.05 * min(p['rally'], 10))
                vx = -speed
                vy = self._english(y, p['rpad'], speed)
                p['aim_l'] = self._new_aim()
                p['fx'].append({'kind': 'pad', 'start': now, 'until': now + 0.18,
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

    def _english(self, ball_y, pad_y, speed):
        """Bounce angle from where the ball met the paddle."""
        rel = (ball_y + self.BALL / 2.0) - (pad_y + self.PAD_H / 2.0)
        rel_norm = max(-1.0, min(1.0, rel / max(1.0, self.PAD_H / 2.0)))
        return rel_norm * speed

    def _new_aim(self):
        """AI aiming error: small enough to rally, big enough to lose."""
        return int(self.rng.integers(-3, 4))

    def _point(self, side, now, exit_x, exit_y):
        """side scored: shockwave rings roll out of the goal mouth."""
        p = self._pong
        p['score'][side] += 1
        p['rally'] = 0
        p['fx'].append({'kind': 'burst', 'start': now, 'until': now + 0.95,
                        'cx': exit_x, 'cy': exit_y})
        self._start_celebration('point', side, now, exit_x, exit_y)
        if p['score'][side] >= self.WIN_SCORE:
            self._trigger_win(side, now)
            return

        # Serve toward the player who just lost the point
        p['serve_dir'] = -1 if side == 1 else 1
        self._center_paddles()
        p['ball'] = [self.width / 2.0 - self.BALL / 2.0,
                     (p['ft'] + p['fb'] - self.BALL) / 2.0]
        p['v'] = [0.0, 0.0]
        p['serve_until'] = now + self.POINT_SERVE_DELAY

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
        self._left_raise_armed = False
        self.song_start_time = time.time()
        self._reset_match(self.song_start_time)

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
            text.write_centered(frame, msg, y=6, size=5, style="regular")
        if elapsed >= 1.45 and int(elapsed * 12) % 2 == 0:
            frame[:, :] ^= 1

    def _draw_big_score(self, frame, scores):
        score_str = self._score_text(scores)
        text.write_centered(frame, score_str, y=10, size=6, style="regular")

    @staticmethod
    def _score_text(scores):
        return '%d:%d' % (scores[0], scores[1])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _compose_pong_bg(self, now):
        h, w = self.height, self.width
        p = self._pong
        bg = np.zeros((h, w), dtype=np.uint8)

        lpad = int(round(p['lpad']))
        rpad = int(round(p['rpad']))

        # Paddles
        bg[lpad:lpad + self.PAD_H, 0:self.PAD_W] = 1
        bg[rpad:rpad + self.PAD_H, w - self.PAD_W:w] = 1

        # Ball blinks at centre during serve count-in.
        draw_ball = True
        if p['serve_until'] is not None and now < p['serve_until']:
            draw_ball = int(now * 8.0) % 2 == 0
        if p['winner'] is None and draw_ball:
            x, y = p['ball']
            xi = int(round(x))
            yi = int(round(y))
            bg[max(0, yi):yi + self.BALL, max(0, xi):max(0, xi + self.BALL)] = 1

        # Impact effects, drawn where they happened.
        for f in p['fx']:
            if f['kind'] == 'pad':
                face = self.PAD_W if f['side'] == 0 else w - self.PAD_W - 1
                direction = 1 if f['side'] == 0 else -1
                elapsed = now - f['start']
                if elapsed < 0.08:
                    r0 = max(p['ft'], int(round(f['y'])) - 2)
                    r1 = min(p['fb'], int(round(f['y'])) + self.PAD_H + 2)
                    c0 = face if direction == 1 else face - 1
                    bg[r0:r1, max(0, c0):c0 + 2] = 1
                else:
                    rad = 5
                    for dr in range(-rad, rad + 1):
                        dc = int(round((rad * rad - dr * dr) ** 0.5))
                        rr = int(round(f['cy'])) + dr
                        for cc in (face + direction * dc,
                                   face + direction * max(0, dc - 1)):
                            if p['ft'] <= rr < p['fb'] and 0 <= cc < w:
                                bg[rr, cc] = 1
            elif f['kind'] == 'wall':
                c0 = max(0, int(round(f['x'])) - 4)
                if f['top']:
                    bg[p['ft']:p['ft'] + 2, c0:c0 + 10] = 1
                else:
                    bg[p['fb'] - 2:p['fb'], c0:c0 + 10] = 1
            elif f['kind'] == 'burst':
                # Double shockwave from the goal mouth.
                progress = max(0.0, min(1.0, (now - f['start']) / (f['until'] - f['start'])))
                yy, xx = np.ogrid[:h, :w]
                d2 = (yy - f['cy']) ** 2 + (xx - f['cx']) ** 2
                lead = 4 + int(progress * (w + h))
                for rad in (lead, lead - 9):
                    if rad > 2:
                        bg[(d2 >= (rad - 1) ** 2)
                           & (d2 <= (rad + 1) ** 2)] = 1
            elif f['kind'] == 'serve':
                mr = (p['ft'] + p['fb']) // 2
                bg[mr - 3:mr + 3, w // 2 - 3:w // 2 + 3] = 1

        # Score digits at the top (drawn last, over the ball — classic)
        score_str = self._score_text(p['score'])
        text.write_centered(bg, score_str, y=1, size=5, style="regular")
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
            if p['win_time'] is not None and now - p['win_time'] >= self.WIN_RESTART_TIME:
                self._restart_match()
            return

        person_present = (
            pose_results is not None
            and getattr(pose_results, 'pose_landmarks', None) is not None
        )
        if not person_present:
            return
        self._last_person_time = now
        _, finger_y = human_pose.get_right_index_finger_position(pose_results)
        if finger_y is not None:
            # Finger height -> paddle center target.
            self._human_target = (finger_y * (p['fb'] - p['ft'])
                                  + p['ft'] - self.PAD_H / 2)

    def get_frame(self, pose_results):
        now = time.time()
        dt = min(self.MAX_DT, max(0.0, now - self._last_frame_time))
        self._last_frame_time = now

        p = self._pong

        self._handle_gestures(pose_results, now)
        self._update_match(now, dt)

        due = [e for e in self._celebration_hits if e[0] <= now]
        self._celebration_hits = [e for e in self._celebration_hits
                                  if e[0] > now]
        for _, instrument_name in due:
            self._hit(instrument_name, now)

        frame = self.state.copy()
        frame ^= self._compose_pong_bg(now)

        celebration_frame = self._render_celebration(now)
        if celebration_frame is not None:
            frame = celebration_frame

        # Mode name overlay for the first 2 s
        if celebration_frame is None and now - self.song_start_time < self.MODE_NAME_TIME:
            frame[:6, :] = 0
            text.write(frame, 'PONG', x=1, y=0, size=5, style="regular")

        # Win screen: fanfare plays over the final court for a moment,
        # then the static result
        if p['winner'] is not None and now - p['win_time'] >= self.WIN_FANFARE_TIME:
            frame[:, :] = 0
            msg = p['win_text']
            text.write_centered(frame, msg, y=6, size=5, style="regular")
            score_str = self._score_text(p['score'])
            text.write_centered(frame, score_str, y=15, size=6, style="regular")

        return frame

    def set_controller_target(self, norm_y):
        p = self._pong
        normalized = max(0.0, min(1.0, float(norm_y)))
        playable_span = p['fb'] - p['ft']
        self._human_target = (normalized * playable_span) + p['ft'] - self.PAD_H / 2
        self._last_controller_input_time = time.time()

    def restart_if_game_over(self):
        if self._pong.get('winner') is not None:
            self._restart_match()