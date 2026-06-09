import numpy as np
import time
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
            'subdivisions': 4,  # 16th notes; 32 steps = full 2-bar theme
            'sections': [
                (0, [
                    # Imperial March: G G G Eb-Bb G Eb-Bb G
                    # kick=main note, snare=Eb (first anacrusis), tom=Bb (grace note)
                    # Bar 1: G(q) G(q) G(q) Eb(d.e) Bb(16th)
                    {'kick'}, set(), set(), set(),
                    {'kick'}, set(), set(), set(),
                    {'kick'}, set(), set(), set(),
                    {'snare'}, set(), set(), {'tom'},
                    # Bar 2: G(half) Eb(d.e) Bb(16th) G(q)
                    {'kick'}, set(), set(), set(), set(), set(), set(), set(),
                    {'snare'}, set(), set(), {'tom'},
                    {'kick'}, set(), set(), set(),
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
