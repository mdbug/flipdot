import numpy as np
import time
from PIL import Image
import text
import human_pose


class AutoDrum:
    """Autonomous drum sequencer playing recognisable song patterns.

    The flip-dot click IS the sound.  Each instrument is synthesised from
    three physical parameters of the panel:

    * loudness – how many dots flip at once (density × region area)
    * texture  – solid block (thump) vs random scatter (crack/sizzle)
    * sustain  – decay tail: diminishing random subsets re-flip over the
                 next ~50-250 ms, like a rattling cymbal

    Raise your left hand above your head and hold for 1 s to skip to the
    next song.  Arms-crossed exits to the menu (handled by the main loop).

    Pattern encoding
    ----------------
    Each song is a list of sections: (repeats, [step0, step1, ...])
    repeats=0 means loop the section (and therefore the song) forever.
    Each step is a set of instrument names to strike on that step.
    """

    SKIP_HOLD_TIME = 1.0  # seconds to hold left hand raised to skip to next song
    DECAY_TICK = 0.05     # seconds between decay-tail flips

    # MARCH pitches as (name, MIDI number), highest first.  Stripes are
    # placed semitone-proportionally so the panel contour matches the
    # actual melodic intervals.
    MARCH_PITCHES = (
        ('g5', 79), ('gb5', 78), ('f5', 77), ('e5', 76), ('eb5', 75),
        ('d5', 74), ('db5', 73), ('c5', 72), ('b4', 71), ('bb4', 70),
        ('a4', 69), ('ab4', 68), ('g4', 67), ('gb4', 66), ('eb4', 63),
        ('bb3', 58),
    )
    # Pitches that occur as half notes get a '<name>_long' variant with
    # this shimmering decay tail (~0.6 s of diminishing re-flips).
    MARCH_LONG_NOTES = ('g4', 'd5')
    MARCH_LONG_DECAY = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07,
                        0.06, 0.05, 0.05, 0.04, 0.04, 0.03]

    SONGS = [
        {
            'name': 'ROCK',
            'bpm': 82,
            'subdivisions': 2,  # 8th notes; one bar = 8 steps
            'sections': [
                (0, [                        # loop forever
                    # stomp stomp CLAP rest | stomp stomp CLAP rest
                    {'kick'}, {'kick'}, {'snare'}, set(),
                    {'kick'}, {'kick'}, {'snare'}, set(),
                ]),
            ],
        },
        {
            'name': 'SEVEN',
            'bpm': 124,
            'subdivisions': 2,  # 8th notes; 16 steps = full 2-bar riff
            'sections': [
                (0, [
                    # Seven Nation Army bass riff rhythm: E E G. E D C B
                    # E(q)  E(q)  G(d.q)        E(q)  D(e)  C(q)   B(h)
                    {'kick'}, set(), {'kick'}, set(),
                    {'kick'}, set(), set(),    {'kick'},
                    set(),    {'kick'}, {'kick'}, set(),
                    {'kick'}, set(), set(),    set(),
                ]),
            ],
        },
        {
            'name': 'TIGER',
            'bpm': 108,
            'subdivisions': 2,  # 8th notes; 16 steps = 2 bars
            'sections': [
                (2, [
                    # Eye of the Tiger intro: three groups of 3 toms + kick punch
                    {'tom'}, {'tom'}, {'tom'}, set(),
                    {'tom'}, {'tom'}, {'tom'}, set(),
                    {'tom'}, {'tom'}, {'tom'}, set(),
                    {'kick'}, set(),  set(),   set(),
                ]),
                (0, [
                    # Verse groove with ticking hats on the off-beats
                    {'kick'}, {'hat'},  {'kick'}, {'hat'},
                    {'snare'}, {'hat'}, {'kick'}, {'hat'},
                    {'kick'}, {'kick'}, {'hat'},  set(),
                    {'snare'}, {'hat'}, set(),    {'hat'},
                ]),
            ],
        },
        {
            'name': 'MARCH',
            'bpm': 103,
            'subdivisions': 4,  # 16th notes; 192 steps = full 12-bar theme
            'sections': [
                (0, [
                    # One full-width stripe per pitch, placed semitone-
                    # proportionally, top = highest (see _load_song).
                    # ---- Antecedent ----
                    # Bar 1: G4(q) G4(q) G4(q) Eb4(d.e) Bb3(16th)
                    {'g4'}, set(), set(), set(),
                    {'g4'}, set(), set(), set(),
                    {'g4'}, set(), set(), set(),
                    {'eb4'}, set(), set(), {'bb3'},
                    # Bar 2: G4(q) Eb4(d.e) Bb3(16th) G4(HALF — rings out)
                    {'g4'}, set(), set(), set(),
                    {'eb4'}, set(), set(), {'bb3'},
                    {'g4_long'}, set(), set(), set(),
                    set(), set(), set(), set(),
                    # Bar 3: D5(q) D5(q) D5(q) Eb5(d.e) Bb4(16th) — high phrase
                    {'d5'}, set(), set(), set(),
                    {'d5'}, set(), set(), set(),
                    {'d5'}, set(), set(), set(),
                    {'eb5'}, set(), set(), {'bb4'},
                    # Bar 4: Gb4(q) Eb5(d.e) Bb4(16th) G4(HALF)
                    {'gb4'}, set(), set(), set(),
                    {'eb5'}, set(), set(), {'bb4'},
                    {'g4_long'}, set(), set(), set(),
                    set(), set(), set(), set(),
                    # ---- Consequent (the "answer") ----
                    # Bar 5: G5(q) G4(d.e) G4(16) G5(q) Gb5(d.e) F5(16)
                    {'g5'}, set(), set(), set(),
                    {'g4'}, set(), set(), {'g4'},
                    {'g5'}, set(), set(), set(),
                    {'gb5'}, set(), set(), {'f5'},
                    # Bar 6: E5(16) Eb5(16) E5(8) rest Ab4(8)
                    #        Db5(q) C5(d.e) B4(16)   — chromatic flourish
                    {'e5'}, {'eb5'}, {'e5'}, set(),
                    set(), set(), {'ab4'}, set(),
                    {'db5'}, set(), set(), set(),
                    {'c5'}, set(), set(), {'b4'},
                    # Bar 7: Bb4(16) A4(16) Bb4(8) rest Eb4(8)
                    #        Gb4(q) Eb4(d.e) Gb4(16)
                    {'bb4'}, {'a4'}, {'bb4'}, set(),
                    set(), set(), {'eb4'}, set(),
                    {'gb4'}, set(), set(), set(),
                    {'eb4'}, set(), set(), {'gb4'},
                    # Bar 8: Bb4(q) G4(d.e) Bb4(16) D5(HALF) — first ending, up
                    {'bb4'}, set(), set(), set(),
                    {'g4'}, set(), set(), {'bb4'},
                    {'d5_long'}, set(), set(), set(),
                    set(), set(), set(), set(),
                    # ---- Consequent again, second ending ----
                    # Bar 9 = Bar 5
                    {'g5'}, set(), set(), set(),
                    {'g4'}, set(), set(), {'g4'},
                    {'g5'}, set(), set(), set(),
                    {'gb5'}, set(), set(), {'f5'},
                    # Bar 10 = Bar 6
                    {'e5'}, {'eb5'}, {'e5'}, set(),
                    set(), set(), {'ab4'}, set(),
                    {'db5'}, set(), set(), set(),
                    {'c5'}, set(), set(), {'b4'},
                    # Bar 11: like Bar 7 but turns DOWN at the end (Bb3)
                    {'bb4'}, {'a4'}, {'bb4'}, set(),
                    set(), set(), {'eb4'}, set(),
                    {'gb4'}, set(), set(), set(),
                    {'eb4'}, set(), set(), {'bb3'},
                    # Bar 12: G4(q) Eb4(d.e) Bb3(16) G4(HALF) — home, recaps bar 2
                    {'g4'}, set(), set(), set(),
                    {'eb4'}, set(), set(), {'bb3'},
                    {'g4_long'}, set(), set(), set(),
                    set(), set(), set(), set(),
                ]),
            ],
        },
        {
            'name': 'STORM',
            'bpm': 136,
            'subdivisions': 2,  # 8th notes; 8 steps = 1 bar
            'sections': [
                (1, [
                    # Stage 1: just a kick on beat 1 (sparse open)
                    {'kick'}, set(), set(), set(),
                    set(),    set(), set(), set(),
                ]),
                (2, [
                    # Stage 2: kick + snare enter on 2 & 4
                    {'kick'}, set(), {'snare'}, set(),
                    {'kick'}, set(), {'snare'}, set(),
                ]),
                (2, [
                    # Stage 3: snare roll builds tension
                    {'snare'}, {'snare'}, {'snare'}, {'snare'},
                    {'snare'}, {'snare'}, {'snare'}, {'snare'},
                ]),
                (1, [
                    # DROP impact bar: crash splash on the downbeat
                    {'crash', 'kick'}, {'hat'}, {'kick', 'snare'}, {'hat'},
                    {'kick'}, {'hat'}, {'kick', 'snare'}, {'hat'},
                ]),
                (0, [
                    # DROP groove: 4-on-the-floor + ticking hats
                    {'kick'}, {'hat'}, {'kick', 'snare'}, {'hat'},
                    {'kick'}, {'hat'}, {'kick', 'snare'}, {'hat'},
                ]),
            ],
        },
    ]

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.state = np.zeros((height, width), dtype=np.uint8)
        self.rng = np.random.default_rng()
        # Instruments: area (r0,r1,c0,c1), density (loudness/texture),
        # decay (densities of the rattle tail, one per DECAY_TICK).
        h, w = height, width
        self._default_densities = {'snare': 0.6, 'tom': 0.9, 'hat': 0.2}
        self._default_decays = {'snare': [0.25], 'tom': [0.35, 0.15], 'hat': []}
        self._default_areas = {
            'kick':  (h // 2, h, 0, w),
            'snare': (0, h // 2, 0, w // 2),
            'tom':   (h // 4, h // 2, w // 2, w),
            'hat':   (0, h // 4, w // 2, w),
            'crash': (0, h, 0, w),
        }
        self.instruments = {
            # solid full-width thump, no tail – the loudest single clack
            'kick':  {'area': (h // 2, h, 0, w),           'density': 1.0,
                      'decay': []},
            # scattered crack with a short rattle
            'snare': {'area': (0, h // 2, 0, w // 2),      'density': 0.6,
                      'decay': [0.25]},
            # dense punch with a slightly longer ring
            'tom':   {'area': (h // 4, h // 2, w // 2, w), 'density': 0.9,
                      'decay': [0.35, 0.15]},
            # tiny sparse tick – the quietest instrument
            'hat':   {'area': (0, h // 4, w // 2, w),      'density': 0.2,
                      'decay': []},
            # whole-panel splash with a long shimmering tail
            'crash': {'area': (0, h, 0, w),                'density': 0.8,
                      'decay': [0.45, 0.3, 0.18, 0.1, 0.05]},
        }
        self._load_song(0)

    def _load_song(self, index):
        self.song_index = index % len(self.SONGS)
        self.section_index = 0
        self.section_repeats = 0
        self.step = -1
        self.next_step_time = time.time()
        self.song_start_time = time.time()
        self.state[:, :] = 0
        self._skip_hold_start = None
        self._decay_events = []  # (due_time, instrument_name, density)
        song_name = self.SONGS[self.song_index]['name']
        h, w = self.height, self.width
        if song_name == 'MARCH':
            img = Image.open('imgs/darthvader.png').convert('L').resize(
                (self.width, self.height), Image.NEAREST)
            self._bg_frame = (np.asarray(img) < 128).astype(np.uint8)
            # Full-width stripe per pitch, placed proportionally to its
            # MIDI number (top = highest), so leaps look like leaps and
            # the bar-6/7 chromatic flourishes wiggle in place.
            hi = self.MARCH_PITCHES[0][1]
            lo = self.MARCH_PITCHES[-1][1]
            thickness = max(2, (h - 1) // (hi - lo))
            for name, midi in self.MARCH_PITCHES:
                r0 = round((hi - midi) / (hi - lo) * (h - 1 - thickness))
                self.instruments[name] = {'area': (r0, r0 + thickness, 0, w),
                                          'density': 1.0, 'decay': []}
            # Half notes ring out instead of dying as a single click.
            for name in self.MARCH_LONG_NOTES:
                self.instruments[name + '_long'] = {
                    'area': self.instruments[name]['area'], 'density': 1.0,
                    'decay': list(self.MARCH_LONG_DECAY)}
        else:
            self._bg_frame = None
            # Drop MARCH-only pitch instruments and restore drum defaults.
            for name, _ in self.MARCH_PITCHES:
                self.instruments.pop(name, None)
            for name in self.MARCH_LONG_NOTES:
                self.instruments.pop(name + '_long', None)
            for name, area in self._default_areas.items():
                self.instruments[name]['area'] = area
            for name, d in self._default_densities.items():
                self.instruments[name]['density'] = d
                self.instruments[name]['decay'] = list(self._default_decays[name])

    def _scatter_flip(self, name, density):
        """XOR a random subset of the instrument's area; density 1.0 = solid."""
        r0, r1, c0, c1 = self.instruments[name]['area']
        if density >= 1.0:
            self.state[r0:r1, c0:c1] ^= 1
        else:
            mask = self.rng.random((r1 - r0, c1 - c0)) < density
            self.state[r0:r1, c0:c1] ^= mask.astype(np.uint8)

    def _hit(self, name, now):
        inst = self.instruments[name]
        self._scatter_flip(name, inst['density'])
        for i, tail_density in enumerate(inst['decay']):
            self._decay_events.append(
                (now + (i + 1) * self.DECAY_TICK, name, tail_density))

    def _section(self):
        return self.SONGS[self.song_index]['sections'][self.section_index]

    def get_frame(self, pose_results):
        now = time.time()
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song['bpm'] / song['subdivisions']

        # Fire any due decay-tail flips (cymbal rattle / ring-out)
        if self._decay_events:
            due = [e for e in self._decay_events if e[0] <= now]
            self._decay_events = [e for e in self._decay_events if e[0] > now]
            for _, name, density in due:
                self._scatter_flip(name, density)

        # Advance the sequencer, catching up if we fell behind
        while now >= self.next_step_time:
            repeats, pattern = self._section()
            prev = self.step
            self.step = (self.step + 1) % len(pattern)

            for name in pattern[self.step]:
                self._hit(name, now)

            # Completed one full pass through the pattern?
            if prev == len(pattern) - 1:
                self.section_repeats += 1
                if repeats > 0 and self.section_repeats >= repeats:
                    self.section_index = (self.section_index + 1) % len(song['sections'])
                    self.section_repeats = 0
                    self.step = -1  # next iteration will start the new section at 0

            self.next_step_time += step_interval
            # After a long pause (mode switch etc.) don't try to catch up
            if now - self.next_step_time > 1.0:
                self.next_step_time = now + step_interval

        # Raise left hand above head and hold for SKIP_HOLD_TIME → next song
        if human_pose.is_left_hand_raised(pose_results):
            if self._skip_hold_start is None:
                self._skip_hold_start = now
            elif now - self._skip_hold_start >= self.SKIP_HOLD_TIME:
                self._load_song(self.song_index + 1)
                return self.state.copy()
        else:
            self._skip_hold_start = None

        frame = self.state.copy()

        # XOR background image (e.g. Darth Vader for MARCH).
        # Full XOR preserves correct dot-flip sound everywhere; the
        # note stripes momentarily invert Vader's silhouette as they pass.
        if self._bg_frame is not None:
            frame ^= self._bg_frame

        # Step cursor along the bottom row
        _, pattern = self._section()
        if self.step >= 0:
            frame[-1, :] = 0
            n = len(pattern)
            if n <= self.width:
                seg = self.width // n
                frame[-1, self.step * seg:min((self.step + 1) * seg, self.width)] = 1
            else:
                # Pattern longer than display width: single-pixel position indicator
                frame[-1, self.step * self.width // n] = 1

        # Song name overlay for first 2 s after a load
        if now - self.song_start_time < 2.0:
            frame[:6, :] = 0
            text.write(frame, song['name'], x=1, y=0, size=5)

        # Skip progress bar on top row while left hand is raised
        if self._skip_hold_start is not None:
            progress = min(int((now - self._skip_hold_start) / self.SKIP_HOLD_TIME * self.width), self.width)
            frame[0, :progress] ^= 1

        frame = human_pose.draw_right_index_pointer(frame, pose_results, size=2)
        return frame