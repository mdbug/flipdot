"""Convert MediaPipe pose results into a 1-bit silhouette mask.

Extracted from ``BeatMirror`` so every mode that wants a body silhouette
(beat mirror, game of life, sandfall, ...) shares one implementation.

The mask prefers ``pose_results.segmentation_mask`` (a true silhouette,
centre-cropped to a square so the person keeps their proportions) and
falls back to a thick-limbed skeleton drawn from ``pose_landmarks``.
Both paths are mirrored horizontally — the same convention as the finger
pointer — so the panel behaves like a mirror.
"""

from typing import Any

import numpy as np
from PIL import Image

# mediapipe 33-landmark model indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
LIMBS = (
    (11, 13),
    (13, 15),  # left arm : shoulder→elbow→wrist
    (12, 14),
    (14, 16),  # right arm
    (23, 25),
    (25, 27),  # left leg : hip→knee→ankle
    (24, 26),
    (26, 28),  # right leg
    (11, 12),
    (23, 24),  # shoulder line, hip line
)

MIN_VISIBILITY = 0.5
SEGMENTATION_THRESHOLD = 0.5


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Return only the 1-px edge of a boolean mask (its 4-connected interior removed).

    Off-panel neighbours count as filled, so a body cropped by the frame
    border is not traced along that border.
    """
    interior = mask.copy()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    return mask & ~interior


def pose_to_mask(pose_results: Any, width: int, height: int) -> np.ndarray | None:
    """Return a ``(height, width)`` boolean silhouette mask, or None if nobody."""
    if pose_results is None:
        return None

    seg = getattr(pose_results, "segmentation_mask", None)
    if seg is not None:
        return _segmentation_to_mask(seg, width, height)

    lms = getattr(pose_results, "pose_landmarks", None)
    if lms is None:
        return None
    return _skeleton_to_mask(lms, width, height)


def _segmentation_to_mask(seg: Any, width: int, height: int) -> np.ndarray | None:
    """Centre-crop the segmentation mask square, threshold, resize, and mirror."""
    arr = np.asarray(seg)
    hh, ww = arr.shape[:2]
    s = min(hh, ww)
    r0, c0 = (hh - s) // 2, (ww - s) // 2
    img = Image.fromarray(
        ((arr[r0 : r0 + s, c0 : c0 + s] > SEGMENTATION_THRESHOLD) * 255).astype(np.uint8)
    )
    img = img.resize((width, height), Image.NEAREST)
    mask = np.asarray(img)[:, ::-1] > 127
    return mask if mask.any() else None


def _skeleton_to_mask(lms: Any, width: int, height: int) -> np.ndarray | None:
    """Draw a thick-limbed skeleton (with filled torso and head) from landmarks."""
    wanted = set(j for limb in LIMBS for j in limb) | {NOSE}
    pts = {}
    for i in wanted:
        lm = lms.landmark[i]
        if getattr(lm, "visibility", 1.0) < MIN_VISIBILITY:
            continue
        # mirrored x, matching the finger-pointer convention
        pts[i] = ((1.0 - lm.x) * (width - 1), lm.y * (height - 1))
    if not pts:
        return None
    mask = np.zeros((height, width), dtype=bool)

    def stamp(x: float, y: float, r: int = 1) -> None:
        xi, yi = int(round(x)), int(round(y))
        if xi < -r or yi < -r or xi >= width + r or yi >= height + r:
            return  # landmark outside the panel (partially off-camera)
        mask[max(0, yi - r) : max(0, yi + r), max(0, xi - r) : max(0, xi + r)] = True

    def line(a: tuple[float, ...], b: tuple[float, ...], r: int = 1) -> None:
        (x0, y0), (x1, y1) = a, b
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in np.linspace(0.0, 1.0, n + 1):
            stamp(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r)

    for a, b in LIMBS:
        if a in pts and b in pts:
            line(pts[a], pts[b])
    # Filled torso: sweep lines from the shoulder line to the hip line
    quad = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if all(i in pts for i in quad):
        for t in np.linspace(0.0, 1.0, max(2, height // 2)):
            left = tuple(
                pts[L_SHOULDER][k] + (pts[L_HIP][k] - pts[L_SHOULDER][k]) * t for k in (0, 1)
            )
            right = tuple(
                pts[R_SHOULDER][k] + (pts[R_HIP][k] - pts[R_SHOULDER][k]) * t for k in (0, 1)
            )
            line(left, right)
    if NOSE in pts:
        stamp(*pts[NOSE], r=2)  # head
        # neck: connect the head to the shoulder midpoint
        if L_SHOULDER in pts and R_SHOULDER in pts:
            midx = (pts[L_SHOULDER][0] + pts[R_SHOULDER][0]) / 2
            midy = (pts[L_SHOULDER][1] + pts[R_SHOULDER][1]) / 2
            line(pts[NOSE], (midx, midy))
    return mask if mask.any() else None
