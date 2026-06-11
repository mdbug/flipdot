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
    repeats=0 means hold that section (and therefore the song) forever;
    if every section has repeats>0 the song cycles through its whole
    arc endlessly (used by ROCK's chant bar and STORM's build→drop).
    Each step is a set of instrument names to strike on that step.

    Songs may also define:
    * 'melody' – pitch-named monophonic voice stripes, see _voice_hit;
      pitches are (name, MIDI) highest-first, 'long' lists pitches that
      get a ringing '<name>_long' variant
    * 'image'  – a picture XOR'd under everything (Vader for MARCH)
    * 'bg'     – name of a method drawing a procedural background
                 (the Tetris well for TETRIS)
    """

    SKIP_HOLD_TIME = 1.0  # seconds to hold left hand raised to skip to next song
    DECAY_TICK = 0.05     # seconds between decay-tail flips

    # Songs may define a 'melody': pitch-named monophonic voice stripes
    # (see _voice_hit).  Because the voice is monophonic, every attack
    # wipes the previous note's leftover dots in the SAME panel refresh
    # that flips the new stripe on — the panel only clicks when its
    # state changes, so cleanup never adds its own transient.  Pitches
    # listed under 'long' get a '<name>_long' variant with this
    # shimmering ring-out tail (self-cancelling: each tick undoes the
    # previous tick's scatter in the same frame it adds a smaller one).
    MELODY_LONG_DECAY = [0.15, 0.12, 0.10, 0.09, 0.08, 0.07,
                         0.06, 0.05, 0.05, 0.04, 0.04, 0.03]

    SONGS = [
        {
            'name': 'ROCK',
            'bpm': 82,
            'subdivisions': 2,  # 8th notes; one bar = 8 steps
            'sections': [
                (3, [
                    # stomp stomp CLAP rest | stomp stomp CLAP rest
                    # ('clap' = wide scatter with a stadium-echo tail)
                    {'kick'}, {'kick'}, {'clap'}, set(),
                    {'kick'}, {'kick'}, {'clap'}, set(),
                ]),
                (1, [
                    # Every 4th bar the chant rides on top of the stomps:
                    # WE(1) WILL(1&) WE(2) WILL(2&) ROCK(3) YOU(4)
                    {'kick', 'tom'}, {'kick', 'tom'},
                    {'clap', 'tom'}, {'tom'},
                    {'kick', 'tom'}, {'kick'},
                    {'clap', 'tom'}, set(),
                ]),
            ],
        },
        {
            'name': 'SEVEN',
            'bpm': 124,
            'subdivisions': 2,  # 8th notes; 16 steps = full 2-bar riff
            'sections': [
                (0, [
                    # Seven Nation Army riff as pitch stripes.  Rhythm:
                    # E(dotted q) E(@2&) G E D (8ths) C(half) B(half) —
                    # both long notes land on offbeats, hence the lurch.
                    {'e'}, set(), set(), {'e'},
                    {'g'}, {'e'}, {'d'}, {'c_long'},
                    set(), set(), set(), {'b_long'},
                    set(), set(), set(), set(),
                ]),
            ],
            'melody': {
                'pitches': (('g', 67), ('e', 64), ('d', 62),
                            ('c', 60), ('b', 59)),
                'long': ('c', 'b'),
            },
        },
        {
            'name': 'TIGER',
            'bpm': 108,
            'subdivisions': 2,  # 8th notes; 32 steps = 4-bar stab figure
            'sections': [
                (3, [
                    # Eye of the Tiger stabs: C . . C-Bb-C | . C-Bb-C |
                    # . C-G-Ab(rings) | (silence over the pulse).
                    # Groups hit beat 3, 4, and land ringing on 4&.
                    {'c5_long'}, set(), set(), set(),
                    {'c5'}, set(), {'bb4'}, {'c5_long'},
                    set(), set(), set(), set(),
                    {'c5'}, set(), {'bb4'}, {'c5_long'},
                    set(), set(), set(), set(),
                    {'c5'}, set(), {'g4'}, {'ab4_long'},
                    set(), set(), set(), set(),
                    set(), set(), set(), set(),
                ]),
                (0, [
                    # Verse groove with ticking hats on the off-beats
                    {'kick'}, {'hat'},  {'kick'}, {'hat'},
                    {'snare'}, {'hat'}, {'kick'}, {'hat'},
                    {'kick'}, {'kick'}, {'hat'},  set(),
                    {'snare'}, {'hat'}, set(),    {'hat'},
                ]),
            ],
            'melody': {
                'pitches': (('c5', 72), ('bb4', 70),
                            ('ab4', 68), ('g4', 67)),
                'long': ('c5', 'ab4'),
            },
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
            'image': 'imgs/darthvader.png',
            'melody': {
                'pitches': (
                    ('g5', 79), ('gb5', 78), ('f5', 77), ('e5', 76),
                    ('eb5', 75), ('d5', 74), ('db5', 73), ('c5', 72),
                    ('b4', 71), ('bb4', 70), ('a4', 69), ('ab4', 68),
                    ('g4', 67), ('gb4', 66), ('eb4', 63), ('bb3', 58),
                ),
                'long': ('g4', 'd5'),
            },
        },
        {
            'name': 'TETRIS',
            'bpm': 144,
            'subdivisions': 2,  # 8th notes; 64 steps = Korobeiniki Theme A
            'sections': [
                (0, [
                    # Bar 1: E5(q) B4(8) C5(8) D5(q) C5(8) B4(8)
                    {'e5'}, set(), {'b4'}, {'c5'},
                    {'d5'}, set(), {'c5'}, {'b4'},
                    # Bar 2: A4(q) A4(8) C5(8) E5(q) D5(8) C5(8)
                    {'a4'}, set(), {'a4'}, {'c5'},
                    {'e5'}, set(), {'d5'}, {'c5'},
                    # Bar 3: B4(q.) C5(8) D5(q) E5(q)
                    {'b4'}, set(), set(), {'c5'},
                    {'d5'}, set(), {'e5'}, set(),
                    # Bar 4: C5(q) A4(q) A4(HALF — rings out)
                    {'c5'}, set(), {'a4'}, set(),
                    {'a4_long'}, set(), set(), set(),
                    # Bar 5: D5(q.) F5(8) A5(q) G5(8) F5(8) — the high turn
                    {'d5'}, set(), set(), {'f5'},
                    {'a5'}, set(), {'g5'}, {'f5'},
                    # Bar 6: E5(q.) C5(8) E5(q) D5(8) C5(8)
                    {'e5'}, set(), set(), {'c5'},
                    {'e5'}, set(), {'d5'}, {'c5'},
                    # Bar 7: B4(q) B4(8) C5(8) D5(q) E5(q)
                    {'b4'}, set(), {'b4'}, {'c5'},
                    {'d5'}, set(), {'e5'}, set(),
                    # Bar 8: C5(q) A4(q) A4(HALF) — home
                    {'c5'}, set(), {'a4'}, set(),
                    {'a4_long'}, set(), set(), set(),
                ]),
            ],
            'bg': '_tetris_background',
            'melody': {
                'pitches': (('a5', 81), ('g5', 79), ('f5', 77), ('e5', 76),
                            ('d5', 74), ('c5', 72), ('b4', 71), ('a4', 69)),
                'long': ('a4',),
            },
        },
        {
            'name': 'STORM',
            'bpm': 136,
            'subdivisions': 4,  # 16th notes; every section all repeats>0,
            'sections': [      # so the arc cycles: build→drop→breakdown→…
                (2, [
                    # Stage 1: just a kick on beat 1 (sparse open)
                    {'kick'}, set(), set(), set(),
                    set(), set(), set(), set(),
                    set(), set(), set(), set(),
                    set(), set(), set(), set(),
                ]),
                (2, [
                    # Stage 2: backbeat enters
                    {'kick'}, set(), set(), set(),
                    {'snare'}, set(), set(), set(),
                    {'kick'}, set(), set(), set(),
                    {'snare'}, set(), set(), set(),
                ]),
                (1, [
                    # Stage 3: snare roll at 8ths
                    {'snare'}, set(), {'snare'}, set(),
                    {'snare'}, set(), {'snare'}, set(),
                    {'snare'}, set(), {'snare'}, set(),
                    {'snare'}, set(), {'snare'}, set(),
                ]),
                (1, [
                    # Stage 4: roll doubles to 16ths, stacks tom for the
                    # crescendo — then ONE FULL BEAT OF SILENCE before
                    # the drop (the most important beat in the song)
                    {'snare'}, {'snare'}, {'snare'}, {'snare'},
                    {'snare'}, {'snare'}, {'snare'}, {'snare'},
                    {'snare', 'tom'}, {'snare', 'tom'},
                    {'snare', 'tom'}, {'snare', 'tom'},
                    set(), set(), set(), set(),
                ]),
                (1, [
                    # DROP impact bar: whole-panel crash splash on 1
                    {'crash', 'kick'}, set(), {'hat'}, set(),
                    {'kick', 'snare'}, set(), {'hat'}, set(),
                    {'kick'}, set(), {'hat'}, set(),
                    {'kick', 'snare'}, set(), {'hat'}, set(),
                ]),
                (7, [
                    # DROP groove: 4-on-the-floor + ticking hats, then
                    # the cycle wraps back to the sparse breakdown
                    {'kick'}, set(), {'hat'}, set(),
                    {'kick', 'snare'}, set(), {'hat'}, set(),
                    {'kick'}, set(), {'hat'}, set(),
                    {'kick', 'snare'}, set(), {'hat'}, set(),
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
        self._default_densities = {'kick': 1.0, 'snare': 0.6, 'tom': 0.9,
                                   'hat': 0.2, 'crash': 0.8, 'clap': 0.5}
        self._default_decays = {'kick': [], 'snare': [0.25],
                                'tom': [0.35, 0.15], 'hat': [],
                                'crash': [0.45, 0.3, 0.18, 0.1, 0.05],
                                'clap': [0.3, 0.18, 0.1]}
        self._default_areas = {
            'kick':  (h // 2, h, 0, w),
            'snare': (0, h // 2, 0, w // 2),
            'tom':   (h // 4, h // 2, w // 2, w),
            'hat':   (0, h // 4, w // 2, w),
            'crash': (0, h, 0, w),
            'clap':  (0, h // 2, 0, w),
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
            # wide hand-clap with a stadium echo
            'clap':  {'area': (0, h // 2, 0, w),           'density': 0.5,
                      'decay': [0.3, 0.18, 0.1]},
        }
        self._melody_names = []
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
        self._decay_events = []   # (due_time, instrument_name, density)
        self._voice_decay = []    # (due_time, density) melody ring-out ticks
        self._voice_shimmer = None  # previous shimmer mask, undone next tick
        self._voice_area = None   # area of the currently sounding note
        song = self.SONGS[self.song_index]
        h, w = self.height, self.width
        # Background, XOR'd under everything at render time: either an
        # image file ('image') or a procedural drawing method ('bg')
        if 'image' in song:
            img = Image.open(song['image']).convert('L').resize(
                (self.width, self.height), Image.NEAREST)
            self._bg_frame = (np.asarray(img) < 128).astype(np.uint8)
        elif 'bg' in song:
            self._bg_frame = getattr(self, song['bg'])()
        else:
            self._bg_frame = None
        # Drop the previous song's melody instruments, restore drum defaults
        for name in self._melody_names:
            self.instruments.pop(name, None)
        self._melody_names = []
        for name, area in self._default_areas.items():
            self.instruments[name]['area'] = area
        for name, d in self._default_densities.items():
            self.instruments[name]['density'] = d
            self.instruments[name]['decay'] = list(self._default_decays[name])
        # Build monophonic voice stripes for songs with a melody:
        # full-width stripe per pitch, placed proportionally to its MIDI
        # number (top = highest), so leaps look like leaps and chromatic
        # runs wiggle in place.
        if 'melody' in song:
            pitches = song['melody']['pitches']
            hi, lo = pitches[0][1], pitches[-1][1]
            thickness = max(2, (h - 1) // (hi - lo))
            for name, midi in pitches:
                r0 = round((hi - midi) / (hi - lo) * (h - 1 - thickness))
                self.instruments[name] = {
                    'area': (r0, r0 + thickness, 0, w), 'density': 1.0,
                    'voice': True, 'decay': []}
                self._melody_names.append(name)
            # Held notes ring out with the shimmering tail.
            for name in song['melody'].get('long', ()):
                self.instruments[name + '_long'] = {
                    'area': self.instruments[name]['area'], 'density': 1.0,
                    'voice': True, 'decay': list(self.MELODY_LONG_DECAY)}
                self._melody_names.append(name + '_long')

    def _tetris_background(self):
        """Procedural 1-bit Tetris well: settled stack + falling T-piece.

        Blocks are (c-1)×(c-1) filled squares on a c-grid, so a 1-dot
        gap separates them and they read as tetromino cells.  One
        column is left empty — the classic well waiting for an I-piece.
        The bottom panel row is kept clear for the step cursor.
        """
        h, w = self.height, self.width
        bg = np.zeros((h, w), dtype=np.uint8)
        # Cell size incl. 1-dot gap; never below 3 so blocks are at
        # least 2×2 dots and actually read as tetromino cells.
        c = max(3, min(h, w) // 8)
        ncols = w // c

        def block(row_up, col):
            """Draw one cell; row_up counts upward from the floor."""
            r1 = (h - 1) - row_up * c          # floor = panel row h-2
            r0, c0 = max(r1 - (c - 1), 0), col * c
            bg[r0:r1, c0:c0 + c - 1] = 1

        # Settled stack: deterministic jagged skyline, one empty well
        skyline = [2, 3, 1, 3, 3, 2, 1, 2]
        heights = [skyline[i % len(skyline)] for i in range(ncols)]
        heights[max(0, ncols - 2)] = 0     # the empty I-piece well
        for col, stack_h in enumerate(heights):
            for row_up in range(stack_h):
                block(row_up, col)

        # Falling T-piece near the top, left of centre
        top, mid = 1, max(1, ncols // 2 - 2)
        for col in (mid, mid + 1, mid + 2):
            r0, c0 = top * c, col * c
            bg[r0:r0 + c - 1, c0:c0 + c - 1] = 1
        r0, c0 = (top + 1) * c, (mid + 1) * c
        bg[r0:r0 + c - 1, c0:c0 + c - 1] = 1
        return bg

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
        if inst.get('voice'):
            self._voice_hit(name, now)
            return
        self._scatter_flip(name, inst['density'])
        for i, tail_density in enumerate(inst['decay']):
            self._decay_events.append(
                (now + (i + 1) * self.DECAY_TICK, name, tail_density))

    def _voice_hit(self, name, now):
        """Monophonic melody attack with same-frame cleanup.

        Wipes every leftover melody dot OUTSIDE this note's stripe in
        the same panel refresh that fires the attack, so cleanup never
        produces its own click — the wipe and the clack are one flip
        event, and only the active stripe overlays the background.
        The stripe itself is XOR-flipped, so a repeated pitch blinks
        but always clacks at full loudness.
        """
        inst = self.instruments[name]
        r0, r1, c0, c1 = inst['area']
        # Choke the previous note's ring; its shimmer dots either lie
        # outside the new stripe (wiped below) or get absorbed by it.
        self._voice_decay = []
        self._voice_shimmer = None
        region = np.zeros_like(self.state, dtype=bool)
        region[r0:r1, c0:c1] = True
        self.state[~region] = 0          # cleanup, hidden inside the attack
        self.state[r0:r1, c0:c1] ^= 1    # the attack clack
        self._voice_area = (r0, r1, c0, c1)
        if inst['decay']:
            self._voice_decay = [
                (now + (i + 1) * self.DECAY_TICK, d)
                for i, d in enumerate(inst['decay'])]
            # Final zero-density tick sweeps up the last shimmer dots.
            self._voice_decay.append(
                (now + (len(inst['decay']) + 1) * self.DECAY_TICK, 0.0))

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

        # Melody ring-out shimmer (self-cancelling: each tick undoes the
        # previous tick's dots and scatters a smaller fresh set in the
        # same frame — one rattle click per tick, zero residue).
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
        # The melody voice cleans up after itself, so at most one stripe
        # (the sounding note) overlays Vader at any moment — it reads
        # like a pitch indicator jumping around the silhouette.
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

        return frame