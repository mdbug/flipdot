import time
from typing import Any

import numpy as np

import app.services.silhouette as silhouette
import app.services.text as text
from app.modes.autodrum import AutoDrum
from app.modes.contracts import Frame


class BeatMirror(AutoDrum):
    """Dance mode: a mirror that only looks at you on the beat.

    A sparse backing groove plays from the AutoDrum engine.  Your
    silhouette is sampled ONCE PER SEQUENCER STEP (eighth notes) and
    held between steps, so the dots that flip on each step are exactly
    the dots you moved: stand still and the mirror is silent, dance
    hard and it crunches in time with the music.  Movement is the lead
    instrument; the drums are just the metronome.

    Layout (h×w panel)
    ------------------
    * rows 0–1     : percussion band (hat + snare) — their XOR residue
                     frames the mirror instead of scrambling it
    * rows 2..h-4  : the mirror (cols 0..w-2) + a 1-px motion VU bar
                     on the right edge (col w-1)
    * rows h-3,h-2 : kick band
    * row  h-1     : sequencer step cursor (inherited convention)

    When nobody is in frame, the held silhouette dissolves over a few
    steps — a rhythmic decrescendo as you walk away — and then "DANCE"
    blinks on the beat as an invitation that itself clicks in rhythm.

    There are no gesture controls: raised hands are dancing, not
    commands.  Arms-crossed exit is handled by the main loop as usual.

    All pose-API assumptions live in :mod:`app.services.silhouette`,
    which prefers pose_results.segmentation_mask (enable_segmentation=True
    in the mediapipe Pose constructor) for a true silhouette and falls
    back to drawing a thick-limbed skeleton from pose_landmarks.
    """

    MIRROR_TOP = 2  # first mirror row (below the percussion band)
    VU_FULL = 150  # flips per step that light the full VU bar
    DISSOLVE = 0.3  # fraction of held dots cleared per step when absent

    SONGS = [
        {
            "name": "MIRROR",
            "bpm": 104,
            "subdivisions": 2,  # silhouette updates on 8th notes
            "sections": [  # all repeats>0 → the groove cycles
                (
                    4,
                    [
                        # Heartbeat: kick on 1 & 3 — room to hear yourself
                        {"kick"},
                        set(),
                        set(),
                        set(),
                        {"kick"},
                        set(),
                        set(),
                        set(),
                    ],
                ),
                (
                    4,
                    [
                        # Backbeat joins
                        {"kick"},
                        set(),
                        {"snare"},
                        set(),
                        {"kick"},
                        set(),
                        {"snare"},
                        set(),
                    ],
                ),
                (
                    4,
                    [
                        # Full groove with ticking hats
                        {"kick"},
                        {"hat"},
                        {"snare"},
                        {"hat"},
                        {"kick"},
                        {"hat"},
                        {"snare"},
                        {"hat"},
                    ],
                ),
            ],
            "bg": "_mirror_background",
            "bg_step": "_mirror_step",
        },
    ]

    # ------------------------------------------------------------------
    # Song / instrument setup
    # ------------------------------------------------------------------

    def _load_song(self, index: int) -> None:
        super()._load_song(index)
        h, w = self.height, self.width
        # Pin percussion into thin border bands so drum residue frames
        # the mirror instead of scrambling the silhouette.
        self.instruments["hat"]["area"] = (0, 2, 0, w)
        self.instruments["snare"]["area"] = (0, 2, 0, w)
        self.instruments["kick"]["area"] = (h - 3, h - 1, 0, w)

    def _mirror_background(self) -> Frame:
        """Initialise the mirror state; return the first (empty) frame."""
        h, w = self.height, self.width
        self._last_pose = None
        mh = (h - 3) - self.MIRROR_TOP  # mirror rows MIRROR_TOP .. h-4
        mw = w - 1  # col w-1 is the motion VU bar
        self._mirror: dict[str, Any] = {
            "mh": mh,
            "mw": mw,
            "mask": np.zeros((mh, mw), dtype=bool),  # held silhouette
            "flips": 0,  # last step's movement
            "beat": 0,
        }
        return self._compose_mirror_bg()

    # ------------------------------------------------------------------
    # Per-step update — this is where the beat quantisation happens
    # ------------------------------------------------------------------

    def _mirror_step(self, now: float) -> None:
        """Sample the latest pose ONCE per sequencer step and hold it.

        Because the held silhouette only changes here — inside the same
        frame as the drum hits — every movement-flip lands exactly on
        the musical grid, and the number of flipped dots is literally
        how much the dancer moved since the last step.
        """
        m = self._mirror
        m["beat"] += 1
        new = self._pose_to_mask(self._last_pose)
        if new is not None:
            m["flips"] = int((new ^ m["mask"]).sum())
            m["mask"] = new
        elif m["mask"].any():
            # Person gone: dissolve the held silhouette over a few
            # steps — a rhythmic decrescendo instead of one loud wipe.
            idx = np.argwhere(m["mask"])
            n = max(1, int(round(len(idx) * self.DISSOLVE)))
            sel = self.rng.choice(len(idx), size=n, replace=False)
            m["mask"][idx[sel, 0], idx[sel, 1]] = False
            m["flips"] = n
        else:
            m["flips"] = 0
        self._bg_frame = self._compose_mirror_bg()

    def _compose_mirror_bg(self) -> Frame:
        """Held silhouette + motion VU (+ blinking invitation when empty)."""
        h, w = self.height, self.width
        m = self._mirror
        bg = np.zeros((h, w), dtype=np.uint8)
        bg[self.MIRROR_TOP : self.MIRROR_TOP + m["mh"], : m["mw"]] = m["mask"]
        # Motion VU on the right edge: bar height = last step's flips
        vu = min(m["mh"], m["flips"] * m["mh"] // self.VU_FULL)
        if vu:
            r1 = self.MIRROR_TOP + m["mh"]
            bg[r1 - vu : r1, w - 1] = 1
        # Nobody here, nothing held: blink "DANCE" one beat on, one off.
        # The blink toggles inside this step update, so even the
        # invitation clicks in rhythm.
        if not m["mask"].any() and m["flips"] == 0 and m["beat"] % 4 < 2:
            text.write_centered(bg, "DANCE", y=h // 2 - 3, size=5, style="regular")
        return bg

    # ------------------------------------------------------------------
    # Pose → silhouette.  The ONLY method that touches the pose API.
    # ------------------------------------------------------------------

    def _pose_to_mask(self, pose_results: Any) -> np.ndarray | None:
        """Latest pose → boolean silhouette mask at mirror size, or None if nobody."""
        m = self._mirror
        return silhouette.pose_to_mask(pose_results, m["mw"], m["mh"])

    # ------------------------------------------------------------------
    # get_frame — sequencer + compositing; no gesture controls
    # ------------------------------------------------------------------

    def get_frame(self, pose_results: Any) -> Frame:
        """Run the sequencer, XOR the beat-quantised mirror, and draw the step cursor."""
        now = time.time()
        # Stash the newest pose; _mirror_step samples it on the grid.
        self._last_pose = pose_results
        song = self.SONGS[self.song_index]
        step_interval = 60.0 / song["bpm"] / song["subdivisions"]

        self._tick_voice(now)  # drum decay tails (snare crack etc.)
        self._advance_sequencer(now, step_interval)

        frame = self.state.copy()
        if self._bg_frame is not None:
            frame ^= self._bg_frame

        # Step cursor along the bottom row
        _, pattern = self._section()
        if self.step >= 0:
            frame[-1, :] = 0
            seg = self.width // len(pattern)
            frame[-1, self.step * seg : min((self.step + 1) * seg, self.width)] = 1

        # Mode name overlay for the first 2 s
        if now - self.song_start_time < 2.0:
            frame[:6, :] = 0
            text.write(frame, song["name"], x=1, y=0, size=5, style="regular")

        return frame
