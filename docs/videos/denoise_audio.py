"""Remove the stationary fan noise from a demo clip while preserving flip-dot clicks.

The fan is stationary, so a low percentile of the STFT magnitude per frequency
bin — taken over the whole clip — estimates the fan spectrum without needing a
click-free window (there isn't one). A smoothed soft mask then attenuates
everything at or below that floor while leaving the broadband click transients
untouched.

The "framegate" variant additionally gates whole STFT frames by a broadband
transient score (fraction of bins above the fan floor): clicks light up the
whole spectrum at once, whereas fan flutter only opens isolated bins, so this
suppresses the watery "musical noise" artifact of purely per-bin gating —
between clicks the attenuation is spectrally uniform and cannot gurgle.

The "resynth" variant goes further: it gates the real signal fully to silence
between clicks and adds back a constant *synthetic* fan bed (random-phase
noise shaped by the measured fan profile, ``comfort_db`` below the original
fan level). Inter-click audio is then perfectly stationary noise — nothing
left that can flutter — and any per-bin residue only exists while a click is
masking it.

Usage:
    python denoise_audio.py tetris.mov [--variant resynth] [-o tetris_out.mp4]

Reads the audio from ``<clip>.mov``, processes it, and muxes it with the video
stream of the matching ``<clip>.mp4`` (which must already exist — the video
encode is fine, only the audio is re-derived).

Requires ffmpeg on PATH and: pip install numpy scipy librosa soundfile
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

SR = 48000
N_FFT = 2048
HOP = 512

#: variant name -> (over_sub, max_atten_db, frame_coherent, comfort_db).
#: comfort_db is the synthetic fan bed level relative to the measured fan
#: (None = no comfort noise); higher attenuation + comfort = more fan removal.
VARIANTS: dict[str, tuple[float, float, bool, float | None]] = {
    "mild": (2.0, 18.0, False, None),
    "strong": (3.0, 40.0, False, None),
    "framegate": (3.0, 40.0, True, None),
    "resynth": (3.0, 80.0, True, -12.0),
}

OUTPUT_AUDIO_FILTER = "highpass=f=90,loudnorm=I=-14:TP=-1.5:LRA=11"


def spectral_gate(
    y: np.ndarray,
    noise_percentile: float = 10.0,
    over_sub: float = 2.0,
    max_atten_db: float = 18.0,
    time_smooth: int = 3,
    freq_smooth: int = 5,
    frame_coherent: bool = False,
    frame_open_lo: float = 0.05,
    frame_open_hi: float = 0.25,
    comfort_db: float | None = None,
    comfort_seed: int = 0,
) -> np.ndarray:
    """Attenuate the stationary noise floor of a mono signal, sparing transients.

    Args:
        y: Mono float signal at ``SR``.
        noise_percentile: Percentile of per-bin magnitude used as the fan profile.
        over_sub: Oversubtraction factor — how far above the measured floor the
            mask starts opening.
        max_atten_db: Attenuation applied where the signal sits at the floor.
        time_smooth / freq_smooth: Half-widths of the Hann kernel that smooths
            the mask to avoid musical noise.
        frame_coherent: Also gate whole frames by broadband transient score,
            suppressing per-bin flutter ("gurgling") between clicks.
        frame_open_lo / frame_open_hi: Fraction-of-bins-above-floor scores at
            which the frame gate is fully closed / fully open.
        comfort_db: If set, add a constant synthetic fan bed this many dB below
            the measured fan level (masks any residual flutter; steady noise
            cannot gurgle).
        comfort_seed: RNG seed for the comfort-noise phases (vary per channel
            so stereo noise stays decorrelated).
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(S)
    noise = np.percentile(mag, noise_percentile, axis=1, keepdims=True)
    ratio = mag / np.maximum(noise * over_sub, 1e-10)
    mask = np.clip(ratio - 1.0, 0.0, 1.0)
    kernel = np.outer(
        scipy.signal.windows.hann(freq_smooth * 2 + 1),
        scipy.signal.windows.hann(time_smooth * 2 + 1),
    )
    kernel /= kernel.sum()
    mask = scipy.signal.fftconvolve(mask, kernel, mode="same")
    if frame_coherent:
        frac = np.mean(ratio > 1.0, axis=0)
        frame_open = np.clip((frac - frame_open_lo) / (frame_open_hi - frame_open_lo), 0.0, 1.0)
        # hold click tails open (~30 ms), then smooth (~100 ms) to avoid pumping
        frame_open = scipy.ndimage.maximum_filter1d(frame_open, size=3)
        w = scipy.signal.windows.hann(9)
        frame_open = np.convolve(frame_open, w / w.sum(), mode="same")
        mask = mask * frame_open[np.newaxis, :]
    floor_gain = 10 ** (-max_atten_db / 20)
    gain = floor_gain + (1 - floor_gain) * mask
    out = librosa.istft(S * gain, hop_length=HOP, length=len(y))
    if comfort_db is not None:
        rng = np.random.default_rng(comfort_seed)
        phase = np.exp(2j * np.pi * rng.random(S.shape))
        bed = librosa.istft(noise * phase * 10 ** (comfort_db / 20), hop_length=HOP, length=len(y))
        out = out + bed
    return out


def process_clip(mov: Path, video_mp4: Path, out: Path, variant: str) -> None:
    """Denoise ``mov``'s audio and mux it with ``video_mp4``'s video into ``out``."""
    over_sub, max_atten_db, frame_coherent, comfort_db = VARIANTS[variant]
    y, _ = librosa.load(str(mov), sr=SR, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    processed = np.stack(
        [
            spectral_gate(
                ch,
                over_sub=over_sub,
                max_atten_db=max_atten_db,
                frame_coherent=frame_coherent,
                comfort_db=comfort_db,
                comfort_seed=i,
            )
            for i, ch in enumerate(y)
        ],
        axis=1,
    )
    # write to a sibling temp file so out may safely equal video_mp4
    tmp_out = out.with_name(out.stem + ".denoise-tmp.mp4")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, processed, SR)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_mp4),
                "-i",
                tmp.name,
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-filter:a",
                OUTPUT_AUDIO_FILTER,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(tmp_out),
            ],
            check=True,
        )
    tmp_out.replace(out)


def main() -> None:
    """Parse arguments and process one clip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mov", type=Path, help="source .mov clip")
    parser.add_argument("--variant", choices=VARIANTS, default="resynth")
    parser.add_argument(
        "-o", "--out", type=Path, default=None, help="output mp4 (default: overwrite <clip>.mp4)"
    )
    args = parser.parse_args()
    video_mp4 = args.mov.with_suffix(".mp4")
    out = args.out or video_mp4
    process_clip(args.mov, video_mp4, out, args.variant)
    print(f"wrote {out} ({args.variant})")


if __name__ == "__main__":
    main()
