# Demo video audio — cleanup notes

Working notes on converting the four demo screen-recordings and trying to remove the
background fan whir from their audio. The flip-dot **click** sound is the thing to
preserve; the steady **fan/whir** is the thing to remove.

Source clips (kept locally, gitignored via `docs/videos/*.mov`): `march.mov`, `pong.mov`,
`tank.mov`, `tetris.mov` — all 1080×1080 H.264, stereo AAC 48 kHz.

## Status

**Solved (2026-07-08).** The `resynth` variant of `denoise_audio.py` (approach #8 below)
was chosen by ear; all four committed `*.mp4` files were regenerated with it and the
intermediate A/B candidate files were deleted. The approach: a per-frequency-bin
percentile noise profile + soft spectral gate (#6) removes the fan without a click-free
noise window; a whole-frame gate on a broadband transient score (#7) and a constant
synthetic "comfort noise" fan bed (#8) eliminate the "gurgling" musical-noise artifact
the earlier variants had. Click levels stay identical to the untouched reference on
every measurement.

One caveat: `pong` gets less fan reduction than the others (~17 dB click/floor
separation vs 40+ dB for `march`/`tank`) because its click activity is nearly
continuous, so the frame gate is open most of the time — the algorithm errs on the side
of preserving clicks. Listening verdict: acceptable.

## The core problem (diagnosed from spectrograms + level analysis)

The flip-dot clicks and the fan are hard to separate because they overlap on **both** axes:

- **Level:** measured per-100 ms RMS on `tetris.mov` — fan floor ≈ **−62 dB**, clicks
  ≈ **−45 dB**, median ≈ −52 dB. Only ~**17 dB** of separation, and the clicks are quiet
  (whole-clip mean −48 dB; only occasional peaks reach −23 dB).
- **Frequency:** spectrograms (`showspectrumpic`) show the clicks are **broadband
  transient bursts** (energy from ~DC up to a ~19 kHz cutoff). The fan is a fainter
  broadband haze in the same range, slightly concentrated in the lows (below ~1.5 kHz),
  plus a faint tonal line near ~14 kHz. There is **no** click-free window in the music to
  sample cleanly (the clips play continuously), and the "quiet" regions still contain the
  quiet clicks.

Because the two sounds share frequency content and are close in level, level-based tools
(gate) and spectral-subtraction tools either leave the fan or damage the clicks.

## What was tried (on `tetris` as the test clip)

All chains re-derive audio from `tetris.mov` and end with
`loudnorm=I=-14:TP=-1.5:LRA=11` to keep it loud. Video is copied (`-c:v copy`) from the
already-encoded mp4 during audio experiments.

| # | Approach | Filter (audio) | Result |
|---|----------|----------------|--------|
| 1 | Gentle generic denoise (**"opt1"**, the version originally liked) | `highpass=f=90, afftdn=nf=-25, loudnorm` | Clicks perfect; **fan clearly present**. Safe fallback. |
| 2 | Profiled spectral subtraction (aggressive) | `highpass=f=90, afftdn (sample_noise over a "quiet" window) nr=30, anlmdn, loudnorm` | Fan gone but **clicks completely wrong** — the sampled "noise" window actually contained the quiet clicks, so afftdn learned and subtracted the clicks themselves. |
| 3 | Noise gate — first attempt | `agate threshold=0.0316(−30 dB) …` | **Muted everything** — threshold was set above the click level (clicks are ~−45 dB), so the gate stayed closed. |
| 4 | Noise gate — corrected threshold | `agate threshold=0.0018(−55 dB), range=0.06, release=350 …` | Clicks preserved but fan barely reduced (~1 dB at the quietest moments). Continuous music has too few pure-fan gaps longer than the release for a gate to grab; `loudnorm` re-boosts what's left. |
| 5 | **RNN denoiser** (`arnndn`) with downloaded rnnoise models | `highpass=f=90, arnndn=m=<model>.rnnn, loudnorm` | Preserves the transient clicks while lowering the fan floor. Models `mp` and `cb` reduced the quietest-moment floor by ~5–7 dB (−35 → −43 dB `mp`) with click peaks preserved (≈ −0.5 dB). Models `sh`/`bd` were worse. **Still not satisfactory by ear** — and post-loudnorm analysis later showed it also squashes click peaks ~3 dB (p99 −21.3 vs −18.6 dB opt1). |
| 6 | **Per-bin percentile noise profile + soft spectral gate** (`denoise_audio.py` `mild`/`strong`, librosa/scipy) | Python: per-frequency-bin 10th-percentile of \|STFT\| over the whole clip = fan profile (works *because* the fan is stationary — no click-free window needed, which is what sank #2); smoothed soft mask attenuates bins at/below the floor up to 40 dB; then `highpass=f=90, loudnorm` | Fan floor −62 → −78 dB pre-loudnorm (−50.8 vs opt1's −33.5 dB post-loudnorm) with click p90/p99 identical to opt1 to 0.1 dB. By ear: *good for the most part, but occasional "gurgling"* — musical noise from isolated fan blobs fluttering the per-bin gate. |
| 7 | **#6 + frame-coherent gate** (`denoise_audio.py` `framegate`) | As #6, plus a per-frame gate driven by the fraction of bins above the fan floor (clicks are broadband → open the whole frame; fan flutter is isolated bins → stays closed; ~30 ms tail hold + ~100 ms smoothing against pumping) | Floor −80.5 dB pre-loudnorm (−53.6 post), clicks still identical to opt1. By ear: *slightly better than #6, still a bit gurgling* — near-silence between clicks makes even small residue/flutter stand out. |
| 8 | **#7 + comfort-noise fill** (`denoise_audio.py` `resynth`) | As #7 but gate depth 80 dB (real signal → silence between clicks), then add a constant synthetic fan bed: random-phase STFT shaped by the measured fan profile at −12 dB below original fan level (decorrelated per stereo channel) | **Chosen.** Inter-click audio is pure stationary noise — structurally nothing left to flutter; spectrogram shows a uniform faint bed where #7 had hard silence gaps. Floor −53.3 dB post-loudnorm, clicks still identical to opt1. By ear: good. |

### rnnoise models
Downloaded to the scratchpad from
`https://github.com/GregorR/rnnoise-models` (`sh`, `bd`, `mp`, `cb` — general broadband
models, ~300 KB each; `arnndn` is speech-oriented so it doesn't perfectly fit flip-dot
clicks). Re-fetch if needed; `arnndn` needs the `.rnnn` file at run time.

(The A/B candidate files that lived in this folder during the investigation —
`tetris_opt1`, `tetris_rnn_*`, `tetris_gate_*` — were deleted after `resynth` was
chosen; every variant is reproducible from the `.mov` sources via `denoise_audio.py`.)

## Ideas not yet tried

- A **purpose-built noise-reduction tool** rather than ffmpeg built-ins: iZotope RX
  (Spectral De-noise / Voice De-noise), Audacity's Noise Reduction (learn a profile from a
  genuine silent tail if one can be recorded), or Adobe Audition. These give interactive,
  much finer control than `afftdn`/`arnndn`.
- **Record a few seconds of pure fan** (room tone, panel idle, no clicks) to use as a
  clean noise profile for spectral subtraction — the single biggest missing input; every
  automated attempt to find a click-free window in the clips failed because there isn't one.
- **Re-record** the demos in a quieter environment, or with the mic closer to the panel so
  the clicks sit much higher above the fan (more level separation makes any method work).
- Train/obtain an **`arnndn` model** on flip-dot-specific noise, or try other community
  rnnoise models.
- Combine mild `arnndn` + a surgical **notch** at the ~14 kHz tonal line and a gentle
  low-shelf cut for the sub-1.5 kHz fan emphasis, tuned by ear.

## Reproduce

Approach #6/#7 (once a variant is chosen by ear, run per clip; overwrites `<clip>.mp4`
in place, reusing its already-encoded video stream and re-deriving audio from the `.mov`):

```bash
cd docs/videos
python3 -m venv /tmp/audioenv && /tmp/audioenv/bin/pip install numpy scipy librosa soundfile
for c in march pong tank tetris; do
  /tmp/audioenv/bin/python denoise_audio.py $c.mov --variant resynth
done
```

Approach #5 for reference (RNN `mp` model):

```bash
cd docs/videos
ffmpeg -y -i tetris.mov \
  -c:v libx264 -crf 26 -preset veryslow -pix_fmt yuv420p \
  -filter:a "highpass=f=90,arnndn=m=/path/to/mp.rnnn,loudnorm=I=-14:TP=-1.5:LRA=11" \
  -c:a aac -b:a 128k -movflags +faststart tetris.mp4
```

Diagnostics used:
- Per-100 ms RMS distribution:
  `ffmpeg -i x.mov -af "asetnsamples=n=4800,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-" -f null -`
- Spectrogram image: `ffmpeg -ss T -to T2 -i x.mov -lavfi "showspectrumpic=s=1200x600:legend=1:scale=log" out.png`
