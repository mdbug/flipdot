"""Live line-art caricature mirror driven by MediaPipe face-mesh landmarks.

Renders the viewer's face as 1-bit vector strokes at native panel resolution:
landmarks are normalized into a face-local frame, expression/structure metrics
are measured, amplified relative to fixed neutral values, and drawn with the
clipping-safe primitives from :mod:`app.services.draw`. Everything is local and
deterministic — no network calls and no raster scaling or thresholding.

Side names ("left"/"right") follow the face-mesh index convention used in
:mod:`app.services.human_pose` (the 33/133 eye family is "left"). The extracted
x coordinates are mirrored, so a "left" feature renders on the panel's right
half — the display behaves like a mirror, matching pose mode.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.mode_manager import ModeManager
from app.modes.contracts import Frame, RenderContext
from app.services.draw import draw_circle, draw_line, draw_point, thick_line

HairMaskProvider = Callable[[np.ndarray], "np.ndarray | None"]
# Maps a mirrored-normalized face point to the float panel position where the
# sandfall face renderer would draw it: (x_norm, y_norm, width, height) -> (x_px, y_px).
RealFaceAnchor = Callable[[float, float, int, int], tuple[float, float]]

# Displayed exit progress unwinds at most this fast (fraction/second) when the
# policy's backing-away hold aborts, instead of snapping to full size.
EXIT_RECOVERY_PER_SEC = 1.5

logger = logging.getLogger(__name__)

# --- Face-mesh landmark indices (MediaPipe 478-point topology) ---
# 16-point subsample of FACEMESH_FACE_OVAL, ordered clockwise from the forehead.
FACE_OVAL_INDICES = [10, 297, 284, 389, 454, 361, 397, 379, 152, 150, 172, 132, 234, 162, 54, 67]
LEFT_BROW_INDICES = [46, 52, 55]
RIGHT_BROW_INDICES = [276, 282, 285]
LEFT_BROW_MID = 52
RIGHT_BROW_MID = 282
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
NOSE_BRIDGE, NOSE_TIP = 168, 4
NOSE_BASE_LEFT, NOSE_BASE_RIGHT = 98, 327
MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT = 61, 291
UPPER_LIP_TOP, LOWER_LIP_BOTTOM = 0, 17
UPPER_LIP_INNER, LOWER_LIP_INNER = 13, 14
MOUTH_UPPER_INDICES = [61, 40, 0, 270, 291]
MOUTH_LOWER_INDICES = [61, 91, 17, 321, 291]
FACE_TOP, CHIN = 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454
JAW_LEFT, JAW_RIGHT = 172, 397

_MOUTH_ALL = sorted({*MOUTH_UPPER_INDICES, *MOUTH_LOWER_INDICES})
_MOUTH_UPPER_SET = frozenset(MOUTH_UPPER_INDICES)
# Smile lifts the corners most; opening separates the lip midpoints most.
_MOUTH_SMILE_WEIGHT = {61: 1.0, 291: 1.0, 40: 0.5, 270: 0.5, 91: 0.5, 321: 0.5, 0: 0.0, 17: 0.0}
_MOUTH_OPEN_WEIGHT = {0: 1.0, 17: 1.0, 40: 0.7, 270: 0.7, 91: 0.7, 321: 0.7, 61: 0.0, 291: 0.0}

USED_INDICES: list[int] = sorted(
    {
        *FACE_OVAL_INDICES,
        *LEFT_BROW_INDICES,
        *RIGHT_BROW_INDICES,
        LEFT_EYE_OUTER,
        LEFT_EYE_INNER,
        RIGHT_EYE_INNER,
        RIGHT_EYE_OUTER,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        LEFT_IRIS_CENTER,
        RIGHT_IRIS_CENTER,
        NOSE_BRIDGE,
        NOSE_TIP,
        NOSE_BASE_LEFT,
        NOSE_BASE_RIGHT,
        UPPER_LIP_INNER,
        LOWER_LIP_INNER,
        *_MOUTH_ALL,
    }
)
_INDEX_TO_ROW = {index: row for row, index in enumerate(USED_INDICES)}
_MAX_MESH_INDEX = max(index for index in USED_INDICES if index < LEFT_IRIS_CENTER)

# --- Geometry: how the face-local frame maps onto the panel ---
# Sized to favor a large face; tall hair may clip at the top edge.
PANEL_IOD_FRAC = 0.28  # inter-ocular distance as a fraction of panel width
FACE_ANCHOR_Y_FRAC = 0.42  # eye line sits at this fraction of panel height
FACE_MIN_IOD_NORM = 0.03  # below this normalized IOD the face is too far to trust
LOWER_FACE_Y_GAIN = 1.0  # on-device knob for camera-above foreshortening
# Entrance animation: the caricature first appears where (and as small as)
# the viewer's real face sits on the panel — matching the face sandfall drew
# on the silhouette — then grows into the canonical projection.
ENTRY_ZOOM_SECONDS = 1.5

# --- Exaggeration ---
ROLL_GAIN = 1.6
ROLL_MAX_RAD = 0.61

# --- Smoothing & rendering ---
LANDMARK_EMA_ALPHA = 0.35
FACE_HOLD_SEC = 1.0  # keep drawing the last face this long after detection drops
EYE_OPEN_MIN_IOD = 0.055  # hysteresis: closed eye opens above this exaggerated lid gap
EYE_CLOSE_MIN_IOD = 0.035  # hysteresis: open eye closes below this exaggerated lid gap
EYE_GAP_DISPLAY_GAIN = 2.5  # lid-gap px multiplier so openness reads at 28 px
EYE_MIN_HALF_GAP_PX = 1.0
MOUTH_OPEN_MIN_PX = 1.6  # exaggerated inner-lip gap needed to draw an open mouth
MOUTH_OPEN_UPPER_SHARE = 0.3  # opening shifts the lower lip more than the upper
# One row keeps brow and lid distinct while still letting lowered (angry) brows
# drop below their neutral row; two rows would push them back above neutral.
MIN_BROW_EYE_GAP_ROWS = 1
MIN_NOSE_LIP_GAP_ROWS = 1

# --- Hair (sampled from the selfie-multiclass hair segmentation mask) ---
HAIR_COL_XS = np.linspace(-1.8, 1.8, 23)  # sample columns in face-local IOD units
HAIR_RAY_TOP_IOD = -3.2  # rays scan from this far above the eye line...
HAIR_RAY_BOTTOM_IOD = 2.2  # ...down past the chin (side curtains, long hair)
HAIR_RAY_STEP_IOD = 0.08
_HAIR_RAY_YS = np.arange(HAIR_RAY_TOP_IOD, HAIR_RAY_BOTTOM_IOD, HAIR_RAY_STEP_IOD)
# Face-local (x, y) sample grid for the hair rays; contents are constant, only
# the per-call rotation and eye-mid offset vary.
_HAIR_SAMPLE_GRID = np.empty((HAIR_COL_XS.size * _HAIR_RAY_YS.size, 2))
_HAIR_SAMPLE_GRID[:, 0] = np.repeat(HAIR_COL_XS, _HAIR_RAY_YS.size)
_HAIR_SAMPLE_GRID[:, 1] = np.tile(_HAIR_RAY_YS, HAIR_COL_XS.size)
HAIR_MIN_COVERAGE = 0.02  # fraction of ray samples that must hit hair to accept a profile
HAIR_EMA_ALPHA = 0.3  # slower than landmarks: masks arrive at the segmenter's ~7 Hz
HAIR_ABSENT_HEIGHT_IOD = 0.02  # decayed columns below this collapse to "no hair"
HAIR_MIN_HEIGHT_IOD = 0.15  # skip skinnier columns (speckle gate)
HAIR_STROKE_WIDTH_PX = 1.8  # > column spacing in px, so the fill comb is gap-free
HAIR_BROW_GUARD_MAX_X_IOD = 0.9
HAIR_BROW_GUARD_MIN_HEIGHT_IOD = 0.75  # fill floor over the brows: >= 1 row above a raised brow
FOREHEAD_ARC_INDICES = [162, 54, 67, 10, 297, 284, 389]

# --- Ears (brackets attached to the oval's widest points, mesh 234/454) ---
EAR_HALF_HEIGHT_IOD = 0.3
EAR_WIDTH_IOD = 0.32
EAR_COVERED_MIN_HANG_Y_IOD = 0.1  # side hair hanging below this local y hides the ear

# --- Idle placeholder face ---
IDLE_FACE_MARGIN_PX = 2.0
IDLE_EYE_OFFSET_X_FRAC = 0.35
IDLE_EYE_OFFSET_Y_FRAC = 0.30
IDLE_MOUTH_OFFSET_Y_FRAC = 0.45
IDLE_MOUTH_HALF_WIDTH_FRAC = 0.35
IDLE_BLINK_PERIOD_SEC = 2.4
IDLE_BLINK_CLOSED_SEC = 0.25

_EPS = 1e-6


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return min(hi, max(lo, value))


def _smoothstep(t: float) -> float:
    """Ease ``t`` in [0, 1] with the classic smoothstep curve."""
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class ExaggerationSpec:
    """Neutral value, amplification gain, and clamp range for one face metric."""

    neutral: float
    gain: float
    lo: float
    hi: float

    def apply(self, measured: float) -> float:
        """Amplify the deviation of ``measured`` from neutral, clamped to the range."""
        return _clamp(self.neutral + self.gain * (measured - self.neutral), self.lo, self.hi)


# Metric units are inter-ocular distances; values are on-device tuning starting points.
EXAGGERATION: dict[str, ExaggerationSpec] = {
    "mouth_width": ExaggerationSpec(neutral=0.80, gain=1.8, lo=0.55, hi=1.35),
    "mouth_open": ExaggerationSpec(neutral=0.03, gain=2.2, lo=0.0, hi=0.65),
    "smile": ExaggerationSpec(neutral=0.0, gain=2.5, lo=-0.28, hi=0.28),
    "brow": ExaggerationSpec(neutral=0.36, gain=2.2, lo=0.18, hi=0.62),
    "eye_open": ExaggerationSpec(neutral=0.105, gain=2.0, lo=0.0, hi=0.28),
    "jaw_width": ExaggerationSpec(neutral=1.30, gain=1.4, lo=0.95, hi=1.70),
    "face_aspect": ExaggerationSpec(neutral=1.32, gain=1.3, lo=1.05, hi=1.65),
    # Mean hair height above the forehead arc, in IOD units.
    "hair_volume": ExaggerationSpec(neutral=0.22, gain=1.7, lo=0.0, hi=0.9),
}


@dataclass(frozen=True)
class FaceMetrics:
    """Dimensionless face measurements in face-local, IOD-normalized coordinates."""

    mouth_width: float
    mouth_open: float
    smile: float
    brow_left: float
    brow_right: float
    eye_open_left: float
    eye_open_right: float
    jaw_width: float
    face_aspect: float
    roll: float


def _extract_points(landmark_list: Any) -> np.ndarray | None:
    """Gather the used landmarks into an ``(N, 2)`` array of mirrored normalized coords.

    Returns None when the mesh is too short to contain the required landmarks.
    Missing iris landmarks (a 468-point mesh without iris refinement) fall back
    to the corresponding eye-corner midpoint.
    """
    count = len(landmark_list)
    if count <= _MAX_MESH_INDEX:
        return None
    points = np.empty((len(USED_INDICES), 2), dtype=np.float64)
    for row, index in enumerate(USED_INDICES):
        if index < count:
            landmark = landmark_list[index]
            points[row, 0] = 1.0 - landmark.x
            points[row, 1] = landmark.y
        else:
            points[row] = np.nan
    for iris, (corner_a, corner_b) in (
        (LEFT_IRIS_CENTER, (LEFT_EYE_OUTER, LEFT_EYE_INNER)),
        (RIGHT_IRIS_CENTER, (RIGHT_EYE_INNER, RIGHT_EYE_OUTER)),
    ):
        row = _INDEX_TO_ROW[iris]
        if np.isnan(points[row, 0]):
            points[row] = (points[_INDEX_TO_ROW[corner_a]] + points[_INDEX_TO_ROW[corner_b]]) / 2.0
    return points


def _face_basis(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return ``(eye_mid, iod, roll)`` computed from the eye-corner landmarks."""
    eye_left = (points[_INDEX_TO_ROW[LEFT_EYE_OUTER]] + points[_INDEX_TO_ROW[LEFT_EYE_INNER]]) / 2.0
    eye_right = (
        points[_INDEX_TO_ROW[RIGHT_EYE_INNER]] + points[_INDEX_TO_ROW[RIGHT_EYE_OUTER]]
    ) / 2.0
    eye_mid = (eye_left + eye_right) / 2.0
    # The mesh-"left" eye sits on the panel's right after mirroring, so this
    # delta points in +x for an upright face and roll is ~0.
    delta = eye_left - eye_right
    iod = float(math.hypot(delta[0], delta[1]))
    roll = math.atan2(delta[1], delta[0])
    return eye_mid, iod, roll


def _to_face_local(points: np.ndarray, eye_mid: np.ndarray, iod: float, roll: float) -> np.ndarray:
    """Map points into the face-local frame: eye-mid origin, roll removed, IOD units."""
    cos_r, sin_r = math.cos(-roll), math.sin(-roll)
    rotation = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    return ((points - eye_mid) / iod) @ rotation.T


def _measure_metrics(local: np.ndarray, roll: float) -> FaceMetrics:
    """Measure the expression/structure metrics from face-local points."""

    def x(index: int) -> float:
        return float(local[_INDEX_TO_ROW[index], 0])

    def y(index: int) -> float:
        return float(local[_INDEX_TO_ROW[index], 1])

    face_width = abs(x(FACE_RIGHT) - x(FACE_LEFT))
    return FaceMetrics(
        mouth_width=abs(x(MOUTH_CORNER_RIGHT) - x(MOUTH_CORNER_LEFT)),
        mouth_open=abs(y(LOWER_LIP_INNER) - y(UPPER_LIP_INNER)),
        smile=(y(UPPER_LIP_TOP) + y(LOWER_LIP_BOTTOM)) / 2.0
        - (y(MOUTH_CORNER_LEFT) + y(MOUTH_CORNER_RIGHT)) / 2.0,
        brow_left=(y(LEFT_EYE_OUTER) + y(LEFT_EYE_INNER)) / 2.0 - y(LEFT_BROW_MID),
        brow_right=(y(RIGHT_EYE_INNER) + y(RIGHT_EYE_OUTER)) / 2.0 - y(RIGHT_BROW_MID),
        eye_open_left=abs(y(LEFT_EYE_BOTTOM) - y(LEFT_EYE_TOP)),
        eye_open_right=abs(y(RIGHT_EYE_BOTTOM) - y(RIGHT_EYE_TOP)),
        jaw_width=abs(x(JAW_RIGHT) - x(JAW_LEFT)),
        face_aspect=abs(y(CHIN) - y(FACE_TOP)) / max(face_width, _EPS),
        roll=roll,
    )


def _exaggerate(measured: FaceMetrics) -> FaceMetrics:
    """Amplify each metric's deviation from its neutral value."""
    spec = EXAGGERATION
    return FaceMetrics(
        mouth_width=spec["mouth_width"].apply(measured.mouth_width),
        mouth_open=spec["mouth_open"].apply(measured.mouth_open),
        smile=spec["smile"].apply(measured.smile),
        brow_left=spec["brow"].apply(measured.brow_left),
        brow_right=spec["brow"].apply(measured.brow_right),
        eye_open_left=spec["eye_open"].apply(measured.eye_open_left),
        eye_open_right=spec["eye_open"].apply(measured.eye_open_right),
        jaw_width=spec["jaw_width"].apply(measured.jaw_width),
        face_aspect=spec["face_aspect"].apply(measured.face_aspect),
        roll=_clamp(ROLL_GAIN * measured.roll, -ROLL_MAX_RAD, ROLL_MAX_RAD),
    )


def _apply_exaggeration(local: np.ndarray, measured: FaceMetrics, exag: FaceMetrics) -> np.ndarray:
    """Warp face-local points so they exhibit the exaggerated metrics.

    Width-like metrics scale multiplicatively about a feature center; gap-like
    metrics shift points additively, so near-zero measurements (e.g. a closed
    mouth) stay numerically stable. Eye openness is intentionally not applied
    here — eyes are drawn parametrically from the exaggerated metric.
    """
    out = local.copy()

    # Lower-face vertical stretch (long/short chin) below the eye line.
    aspect_scale = (exag.face_aspect / max(measured.face_aspect, _EPS)) * LOWER_FACE_Y_GAIN
    lower = out[:, 1] > 0.0
    out[lower, 1] *= aspect_scale

    # Jaw width, tapering from full effect at the chin to none at the eye line.
    chin_y = float(out[_INDEX_TO_ROW[CHIN], 1])
    if chin_y > _EPS:
        jaw_scale = exag.jaw_width / max(measured.jaw_width, _EPS)
        taper = np.clip(out[:, 1] / chin_y, 0.0, 1.0)
        out[:, 0] *= 1.0 + (jaw_scale - 1.0) * taper

    for indices, delta in (
        (LEFT_BROW_INDICES, exag.brow_left - measured.brow_left),
        (RIGHT_BROW_INDICES, exag.brow_right - measured.brow_right),
    ):
        for index in indices:
            out[_INDEX_TO_ROW[index], 1] -= delta

    # Mouth: width about its center, smile on the corners, opening on the lips.
    anchor_rows = [
        _INDEX_TO_ROW[index]
        for index in (MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, UPPER_LIP_TOP, LOWER_LIP_BOTTOM)
    ]
    center_x = float(np.mean(out[anchor_rows, 0]))
    width_scale = exag.mouth_width / max(measured.mouth_width, _EPS)
    smile_delta = exag.smile - measured.smile
    open_delta = exag.mouth_open - measured.mouth_open
    for index in _MOUTH_ALL:
        row = _INDEX_TO_ROW[index]
        out[row, 0] = center_x + width_scale * (out[row, 0] - center_x)
        out[row, 1] -= smile_delta * _MOUTH_SMILE_WEIGHT[index]
        open_shift = open_delta * _MOUTH_OPEN_WEIGHT[index]
        if index in _MOUTH_UPPER_SET:
            out[row, 1] -= open_shift * MOUTH_OPEN_UPPER_SHARE
        else:
            out[row, 1] += open_shift * (1.0 - MOUTH_OPEN_UPPER_SHARE)
    return out


def _project_to_panel(
    local: np.ndarray,
    roll: float,
    width: int,
    height: int,
    anchor: np.ndarray | None = None,
    iod_px: float | None = None,
) -> np.ndarray:
    """Project face-local points onto panel pixels, re-applying the (exaggerated) roll.

    ``anchor``/``iod_px`` override the canonical center/scale (used by the
    entry zoom); by default the face is centered with the standard size.
    """
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    rotation = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    if anchor is None:
        anchor = np.array([(width - 1) / 2.0, height * FACE_ANCHOR_Y_FRAC])
    if iod_px is None:
        iod_px = width * PANEL_IOD_FRAC
    return anchor + iod_px * (local @ rotation.T)


def _hair_profile_from_mask(
    mask: np.ndarray, eye_mid: np.ndarray, iod: float, roll: float
) -> np.ndarray:
    """Sample the hair mask into per-column top/bottom profiles in face-local units.

    Casts a vertical ray per ``HAIR_COL_XS`` column through the mask (raw,
    unmirrored camera raster) and returns a ``(2, n_cols)`` array: row 0 is the
    topmost and row 1 the bottommost hair hit, both as heights above the eye
    line (positive up, IOD units), NaN where the column has no hair. Returns
    all-NaN when total coverage is below ``HAIR_MIN_COVERAGE``.
    """
    n_cols = HAIR_COL_XS.size
    n_rays = _HAIR_RAY_YS.size
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    rotation = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    # mirrored normalized coords
    normalized = eye_mid + iod * (_HAIR_SAMPLE_GRID @ rotation.T)
    height, width = mask.shape
    cols = np.round((1.0 - normalized[:, 0]) * (width - 1)).astype(int)  # unmirror
    rows = np.round(normalized[:, 1] * (height - 1)).astype(int)
    in_bounds = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    hits = np.zeros(n_cols * n_rays, dtype=bool)
    hits[in_bounds] = mask[rows[in_bounds], cols[in_bounds]]
    hits = hits.reshape(n_cols, n_rays)
    profile = np.full((2, n_cols), np.nan)
    if hits.mean() < HAIR_MIN_COVERAGE:
        return profile
    has_hit = hits.any(axis=1)
    topmost = hits.argmax(axis=1)
    bottommost = n_rays - 1 - hits[:, ::-1].argmax(axis=1)
    profile[0, has_hit] = -_HAIR_RAY_YS[topmost[has_hit]]
    profile[1, has_hit] = -_HAIR_RAY_YS[bottommost[has_hit]]
    return profile


def _forehead_arc_heights(local: np.ndarray) -> np.ndarray:
    """Interpolate the upper face-oval arc height (positive up) at each hair column."""
    xs = np.array([local[_INDEX_TO_ROW[index], 0] for index in FOREHEAD_ARC_INDICES])
    heights = np.array([-local[_INDEX_TO_ROW[index], 1] for index in FOREHEAD_ARC_INDICES])
    order = np.argsort(xs)
    return np.interp(HAIR_COL_XS, xs[order], heights[order])


def _smooth_hair_profile(previous: np.ndarray | None, measured: np.ndarray) -> np.ndarray:
    """Blend the measured ``(2, n_cols)`` hair profile into the previous one.

    Element-wise EMA where both are present; newly appearing columns are
    adopted directly, vanished ones decay toward zero. A column whose *top*
    has decayed to (near) nothing is collapsed to fully absent — bottoms may
    legitimately sit near zero (the eye line), so they never trigger collapse.
    """
    if previous is None:
        return measured.copy()
    out = previous.copy()
    prev_absent = np.isnan(previous)
    meas_absent = np.isnan(measured)
    both = ~prev_absent & ~meas_absent
    out[both] = (1.0 - HAIR_EMA_ALPHA) * previous[both] + HAIR_EMA_ALPHA * measured[both]
    adopt = prev_absent & ~meas_absent
    out[adopt] = measured[adopt]
    decay = ~prev_absent & meas_absent
    out[decay] = (1.0 - HAIR_EMA_ALPHA) * previous[decay]
    out[:, np.abs(out[0]) < HAIR_ABSENT_HEIGHT_IOD] = np.nan
    return out


def _exaggerate_hair(heights_above_arc: np.ndarray) -> np.ndarray:
    """Scale per-column hair heights so the mean volume matches its exaggerated value."""
    if bool(np.all(np.isnan(heights_above_arc))):
        return heights_above_arc
    volume = float(np.nanmean(np.clip(heights_above_arc, 0.0, None)))
    if volume <= _EPS:
        return heights_above_arc
    target = EXAGGERATION["hair_volume"].apply(volume)
    return heights_above_arc * (target / volume)


HairStroke = tuple[int, tuple[float, float], tuple[float, float]]


def _hair_strokes(profile: np.ndarray, local: np.ndarray) -> list[HairStroke]:
    """Turn the ``(2, n_cols)`` hair profile into face-local fill strokes.

    Returns ``(column_index, top_point, bottom_point)`` per drawable column.
    Columns over the face fill from the forehead arc up (the brow guard keeps
    them clear of a fully raised brow); columns beside the face hang down to
    their measured bottom hit, rendering side curtains and long hair.
    """
    arc = _forehead_arc_heights(local)
    exaggerated = _exaggerate_hair(profile[0] - arc)
    strokes: list[HairStroke] = []
    for index, (x, arc_height, extra, hang) in enumerate(
        zip(HAIR_COL_XS, arc, exaggerated, profile[1], strict=True)
    ):
        if np.isnan(extra):
            continue
        if abs(x) <= HAIR_BROW_GUARD_MAX_X_IOD:
            bottom = max(arc_height, HAIR_BROW_GUARD_MIN_HEIGHT_IOD)
        elif not np.isnan(hang):
            bottom = float(hang)
        else:
            bottom = arc_height
        top = arc_height + extra
        if top - bottom < HAIR_MIN_HEIGHT_IOD:
            continue
        strokes.append((index, (float(x), -top), (float(x), -bottom)))
    return strokes


def _ear_polylines(warped: np.ndarray) -> list[np.ndarray]:
    """Build face-local ear brackets attached to the oval's widest points.

    Each ear is a 3-point ``(top, tip, bottom)`` polyline bulging outward from
    the warped oval anchor, so ears follow jaw-width exaggeration and roll.
    """
    ears = []
    for index in (FACE_LEFT, FACE_RIGHT):
        anchor = warped[_INDEX_TO_ROW[index]]
        direction = 1.0 if anchor[0] >= 0 else -1.0
        ears.append(
            np.array(
                [
                    [anchor[0], anchor[1] - EAR_HALF_HEIGHT_IOD],
                    [anchor[0] + direction * EAR_WIDTH_IOD, anchor[1]],
                    [anchor[0], anchor[1] + EAR_HALF_HEIGHT_IOD],
                ]
            )
        )
    return ears


def _ear_hidden_by_hair(strokes: list[HairStroke], side_sign: float) -> bool:
    """Whether side-hair strokes on the given side hang low enough to cover the ear."""
    for _, _, (x, bottom_y) in strokes:
        if (
            abs(x) > HAIR_BROW_GUARD_MAX_X_IOD
            and x * side_sign > 0
            and bottom_y > EAR_COVERED_MIN_HANG_Y_IOD
        ):
            return True
    return False


def _latest_points(face_mesh_results: Any) -> np.ndarray | None:
    """Extract mirrored landmark points from face-mesh results, or None when unusable."""
    if face_mesh_results is None:
        return None
    faces = getattr(face_mesh_results, "multi_face_landmarks", None)
    if not faces:
        return None
    points = _extract_points(faces[0].landmark)
    if points is None:
        return None
    _, iod, _ = _face_basis(points)
    if iod < FACE_MIN_IOD_NORM:
        return None
    return points


def _pt(panel: np.ndarray, index: int) -> tuple[float, float]:
    """Return the panel-space ``(x, y)`` of a landmark by its mesh index."""
    row = _INDEX_TO_ROW[index]
    return float(panel[row, 0]), float(panel[row, 1])


def _draw_polyline(
    frame: Frame, points: list[tuple[float, float]], *, closed: bool = False
) -> None:
    """Draw connected line segments through ``points``, optionally closing the loop."""
    for start, end in zip(points, points[1:], strict=False):
        draw_line(frame, start, end)
    if closed and len(points) > 2:
        draw_line(frame, points[-1], points[0])


class Caricature:
    """Renders the viewer as a live, exaggerated line-art face."""

    def __init__(
        self,
        width: int,
        height: int,
        mode_manager: ModeManager,
        hair_mask_provider: HairMaskProvider | None = None,
        real_face_anchor: RealFaceAnchor | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self.hair_mask_provider = hair_mask_provider
        # Projects the real eye midpoint to where the sandfall renderer drew
        # the face, so entry/exit zooms land on the silhouette's face; falls
        # back to raw panel scaling when not provided.
        self.real_face_anchor = real_face_anchor
        self._panel_iod_px = width * PANEL_IOD_FRAC
        self._last_mode_start_time: float | None = None
        self._smoothed: np.ndarray | None = None
        self._hair_profile: np.ndarray | None = None
        # Identity of the last ingested hair mask: masks arrive at ~7 Hz while
        # frames render at up to 30, so re-smoothing the same mask would both
        # waste work and quadruple the EMA's effective rate.
        self._last_hair_mask: np.ndarray | None = None
        self._last_face_time: float | None = None
        self._last_render_time: float | None = None
        self._eye_open: dict[str, bool] = {"left": True, "right": True}
        self._showing_face = False
        # Where the entry zoom starts: (anchor px, iod px) of the viewer's
        # real face, captured on the first faced frame after mode entry.
        self._entry_start: tuple[np.ndarray, float] | None = None
        # Exit progress actually rendered; trails the policy's value so an
        # aborted exit hold eases back instead of snapping to full size.
        self._displayed_exit_progress = 0.0
        # Projection used for the current frame (canonical outside the zoom).
        self._proj_anchor: np.ndarray = np.array([(width - 1) / 2.0, height * FACE_ANCHOR_Y_FRAC])
        self._proj_iod_px: float = self._panel_iod_px

    def _reset(self) -> None:
        """Clear smoothing and per-eye state on mode entry."""
        self._smoothed = None
        self._hair_profile = None
        self._last_hair_mask = None
        self._last_face_time = None
        self._last_render_time = None
        self._eye_open = {"left": True, "right": True}
        self._showing_face = False
        self._entry_start = None
        self._displayed_exit_progress = 0.0

    def get_frame(self, context: RenderContext) -> Frame:
        """Render the caricature from the context's face-mesh results."""
        if self._last_mode_start_time != self.mode_manager.mode_start_time:
            self._last_mode_start_time = self.mode_manager.mode_start_time
            self._reset()

        now = time.time()
        points = _latest_points(context.face_mesh_results)
        if points is not None:
            if self._smoothed is None:
                self._smoothed = points
            else:
                self._smoothed = (
                    1.0 - LANDMARK_EMA_ALPHA
                ) * self._smoothed + LANDMARK_EMA_ALPHA * points
            self._last_face_time = now

        exit_progress = self._smooth_exit_progress(context.caricature_exit_progress, now)

        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        smoothed = self._smoothed
        # While an exit hold runs, keep rendering the (stale) face past
        # FACE_HOLD_SEC: the hold outlives the face-hold window on a fast
        # walk-away, and flipping to the full-size idle invite face would
        # break the shrink-onto-head handoff animation.
        has_face = (
            smoothed is not None
            and self._last_face_time is not None
            and (now - self._last_face_time <= FACE_HOLD_SEC or exit_progress is not None)
        )
        if smoothed is not None and has_face:
            hair_mask = None
            if self.hair_mask_provider is not None:
                hair_mask = self.hair_mask_provider(context.frame)
            self._render_face(frame, smoothed, hair_mask, context.mode_time, exit_progress)
        else:
            self._render_idle(frame, now)
        if has_face != self._showing_face:
            self._showing_face = has_face
            logger.debug("Caricature %s", "tracking a face" if has_face else "idle (no face)")
        return frame

    def _smooth_exit_progress(self, exit_progress: float | None, now: float) -> float | None:
        """Track the policy's exit progress, easing back when the hold aborts.

        The policy's value can drop from ~1.0 to None in one frame (the viewer
        stepped forward again); unwinding at ``EXIT_RECOVERY_PER_SEC`` instead
        of jumping keeps the face from popping back to full size.
        """
        dt = 0.0
        if self._last_render_time is not None:
            dt = max(0.0, now - self._last_render_time)
        self._last_render_time = now

        target = exit_progress if exit_progress is not None else 0.0
        if target >= self._displayed_exit_progress:
            self._displayed_exit_progress = target
        else:
            self._displayed_exit_progress = max(
                target, self._displayed_exit_progress - dt * EXIT_RECOVERY_PER_SEC
            )
        return self._displayed_exit_progress if self._displayed_exit_progress > 0.0 else None

    def _real_face_projection(self, eye_mid: np.ndarray, iod: float) -> tuple[np.ndarray, float]:
        """Return (anchor px, iod px) of the viewer's real on-panel face.

        Uses ``real_face_anchor`` (the sandfall face renderer's projection,
        with its perspective correction and row offsets) when available, so
        entry/exit zooms land exactly where sandfall drew the face.
        """
        if self.real_face_anchor is not None:
            anchor = np.array(
                self.real_face_anchor(float(eye_mid[0]), float(eye_mid[1]), self.width, self.height)
            )
        else:
            anchor = np.array([eye_mid[0] * self.width, eye_mid[1] * self.height])
        return anchor, iod * self.width

    def _projection_params(
        self,
        eye_mid: np.ndarray,
        iod: float,
        mode_time: float,
        exit_progress: float | None,
    ) -> tuple[np.ndarray, float]:
        """Return the (anchor, iod px) for this frame's projection.

        For the first ``ENTRY_ZOOM_SECONDS`` after mode entry the projection is
        interpolated from the viewer's real on-panel face position and size
        (where sandfall drew its eyes/mouth) toward the canonical projection,
        so the caricature appears to grow out of the silhouette's head. While
        ``exit_progress`` runs (the policy's backing-away hold) the projection
        shrinks back onto the viewer's live face, so the handoff to sandfall
        lands where sandfall will draw its face. The exit blend starts from
        whatever the entry/canonical projection currently is, so an exit hold
        that begins mid-entry-zoom continues from the partially-grown face
        instead of snapping to full size.
        """
        target_anchor = np.array([(self.width - 1) / 2.0, self.height * FACE_ANCHOR_Y_FRAC])
        if mode_time >= ENTRY_ZOOM_SECONDS:
            base_anchor, base_iod = target_anchor, self._panel_iod_px
        else:
            if self._entry_start is None:
                self._entry_start = self._real_face_projection(eye_mid, iod)
            start_anchor, start_iod = self._entry_start
            ease = _smoothstep(mode_time / ENTRY_ZOOM_SECONDS)
            base_anchor = start_anchor + (target_anchor - start_anchor) * ease
            base_iod = start_iod + (self._panel_iod_px - start_iod) * ease
        if exit_progress is None:
            return base_anchor, base_iod
        real_anchor, real_iod = self._real_face_projection(eye_mid, iod)
        ease = _smoothstep(exit_progress)
        return (
            base_anchor + (real_anchor - base_anchor) * ease,
            base_iod + (real_iod - base_iod) * ease,
        )

    def _project(self, local: np.ndarray, roll: float) -> np.ndarray:
        """Project face-local points with the current (possibly zooming) anchor/scale."""
        return _project_to_panel(
            local,
            roll,
            self.width,
            self.height,
            anchor=self._proj_anchor,
            iod_px=self._proj_iod_px,
        )

    def _render_face(
        self,
        frame: Frame,
        points: np.ndarray,
        hair_mask: np.ndarray | None,
        mode_time: float,
        exit_progress: float | None,
    ) -> None:
        """Run the measure → exaggerate → project pipeline and draw all strokes."""
        eye_mid, iod, roll = _face_basis(points)
        if iod < FACE_MIN_IOD_NORM:
            return
        self._proj_anchor, self._proj_iod_px = self._projection_params(
            eye_mid, iod, mode_time, exit_progress
        )
        local = _to_face_local(points, eye_mid, iod, roll)
        measured = _measure_metrics(local, roll)
        exag = _exaggerate(measured)
        warped = _apply_exaggeration(local, measured, exag)
        panel = self._project(warped, exag.roll)

        if hair_mask is not None and hair_mask is not self._last_hair_mask:
            # Masks arrive at ~7 Hz (the provider returns the same array until
            # the worker finishes a new one); resampling and EMA-smoothing only
            # new masks keeps HAIR_EMA_ALPHA at its tuned per-mask rate.
            self._last_hair_mask = hair_mask
            measured_profile = _hair_profile_from_mask(hair_mask, eye_mid, iod, roll)
            self._hair_profile = _smooth_hair_profile(self._hair_profile, measured_profile)
        hair_strokes: list[HairStroke] = []
        if self._hair_profile is not None:
            hair_strokes = _hair_strokes(self._hair_profile, warped)
            self._draw_hair(frame, hair_strokes, exag.roll)

        _draw_polyline(frame, [_pt(panel, index) for index in FACE_OVAL_INDICES], closed=True)
        self._draw_ears(frame, warped, hair_strokes, exag.roll)
        left_eye_top = self._draw_eye(
            frame,
            panel,
            side="left",
            outer=LEFT_EYE_OUTER,
            inner=LEFT_EYE_INNER,
            iris=LEFT_IRIS_CENTER,
            gap_iod=exag.eye_open_left,
        )
        right_eye_top = self._draw_eye(
            frame,
            panel,
            side="right",
            outer=RIGHT_EYE_OUTER,
            inner=RIGHT_EYE_INNER,
            iris=RIGHT_IRIS_CENTER,
            gap_iod=exag.eye_open_right,
        )
        self._draw_brow(frame, panel, LEFT_BROW_INDICES, left_eye_top)
        self._draw_brow(frame, panel, RIGHT_BROW_INDICES, right_eye_top)
        self._draw_nose(frame, panel)
        self._draw_mouth(frame, panel, exag)

    def _update_eye_state(self, side: str, gap_iod: float) -> bool:
        """Apply open/close hysteresis for one eye and return whether it is open."""
        threshold = EYE_CLOSE_MIN_IOD if self._eye_open[side] else EYE_OPEN_MIN_IOD
        self._eye_open[side] = gap_iod >= threshold
        return self._eye_open[side]

    def _draw_eye(
        self,
        frame: Frame,
        panel: np.ndarray,
        *,
        side: str,
        outer: int,
        inner: int,
        iris: int,
        gap_iod: float,
    ) -> float:
        """Draw one eye (lids + iris, or a closed line); return its topmost stroke y."""
        p_outer = _pt(panel, outer)
        p_inner = _pt(panel, inner)
        center_y = (p_outer[1] + p_inner[1]) / 2.0
        if not self._update_eye_state(side, gap_iod):
            draw_line(frame, p_outer, p_inner)
            return center_y
        mid_x = (p_outer[0] + p_inner[0]) / 2.0
        half_gap = max(
            EYE_MIN_HALF_GAP_PX, gap_iod * self._proj_iod_px * EYE_GAP_DISPLAY_GAIN / 2.0
        )
        _draw_polyline(frame, [p_outer, (mid_x, center_y - half_gap), p_inner])
        _draw_polyline(frame, [p_outer, (mid_x, center_y + half_gap), p_inner])
        iris_x, iris_y = _pt(panel, iris)
        low_x, high_x = sorted((p_outer[0], p_inner[0]))
        iris_x = _clamp(iris_x, low_x + 1.0, high_x - 1.0)
        iris_y = _clamp(iris_y, center_y - half_gap + 1.0, center_y + half_gap - 1.0)
        draw_point(frame, iris_x, iris_y)
        return center_y - half_gap

    def _draw_brow(
        self, frame: Frame, panel: np.ndarray, indices: list[int], eye_top_y: float
    ) -> None:
        """Draw one brow, shifted up if needed to stay clear of the eye below it."""
        points = [_pt(panel, index) for index in indices]
        lowest = max(point[1] for point in points)
        overshoot = lowest - (eye_top_y - MIN_BROW_EYE_GAP_ROWS)
        if overshoot > 0:
            points = [(x, y - overshoot) for x, y in points]
        _draw_polyline(frame, points)

    def _draw_nose(self, frame: Frame, panel: np.ndarray) -> None:
        """Draw the nose bridge and base, shifted up if needed to clear the upper lip."""
        bridge = _pt(panel, NOSE_BRIDGE)
        tip = _pt(panel, NOSE_TIP)
        base_left = _pt(panel, NOSE_BASE_LEFT)
        base_right = _pt(panel, NOSE_BASE_RIGHT)
        lip_y = _pt(panel, UPPER_LIP_TOP)[1]
        lowest = max(tip[1], base_left[1], base_right[1])
        overshoot = lowest - (lip_y - MIN_NOSE_LIP_GAP_ROWS)
        if overshoot > 0:
            tip = (tip[0], tip[1] - overshoot)
            base_left = (base_left[0], base_left[1] - overshoot)
            base_right = (base_right[0], base_right[1] - overshoot)
        draw_line(frame, bridge, tip)
        draw_line(frame, base_left, base_right)

    def _draw_mouth(self, frame: Frame, panel: np.ndarray, exag: FaceMetrics) -> None:
        """Draw the mouth: both lip polylines when open, a single mid polyline when closed."""
        upper = [_pt(panel, index) for index in MOUTH_UPPER_INDICES]
        lower = [_pt(panel, index) for index in MOUTH_LOWER_INDICES]
        if exag.mouth_open * self._proj_iod_px >= MOUTH_OPEN_MIN_PX:
            _draw_polyline(frame, upper)
            _draw_polyline(frame, lower)
        else:
            mid = [
                ((ux + lx) / 2.0, (uy + ly) / 2.0)
                for (ux, uy), (lx, ly) in zip(upper, lower, strict=True)
            ]
            _draw_polyline(frame, mid)

    def _draw_hair(self, frame: Frame, strokes: list[HairStroke], roll: float) -> None:
        """Fill the hair mass with a comb of thick strokes plus a silhouette polyline."""
        if not strokes:
            return
        endpoints = np.array([point for _, top, bottom in strokes for point in (top, bottom)])
        panel = self._project(endpoints, roll)
        stroke_width = max(1.0, HAIR_STROKE_WIDTH_PX * self._proj_iod_px / self._panel_iod_px)
        tops: dict[int, tuple[float, float]] = {}
        for k, (column, _, _) in enumerate(strokes):
            top = (float(panel[2 * k, 0]), float(panel[2 * k, 1]))
            bottom = (float(panel[2 * k + 1, 0]), float(panel[2 * k + 1, 1]))
            thick_line(frame, top, bottom, width=stroke_width)
            tops[column] = top
        columns = sorted(tops)
        for left, right in zip(columns, columns[1:], strict=False):
            if right - left == 1:  # break the silhouette across no-hair gaps
                draw_line(frame, tops[left], tops[right])

    def _draw_ears(
        self, frame: Frame, warped: np.ndarray, hair_strokes: list[HairStroke], roll: float
    ) -> None:
        """Draw ear brackets on the oval sides unless side hair covers them."""
        for ear in _ear_polylines(warped):
            side_sign = 1.0 if ear[1, 0] >= 0 else -1.0
            if _ear_hidden_by_hair(hair_strokes, side_sign):
                continue
            panel = self._project(ear, roll)
            _draw_polyline(frame, [(float(x), float(y)) for x, y in panel])

    def _render_idle(self, frame: Frame, now: float) -> None:
        """Draw a simple blinking placeholder face that invites viewers closer."""
        center_x = (self.width - 1) / 2.0
        center_y = (self.height - 1) / 2.0
        radius = min(self.width, self.height) / 2.0 - IDLE_FACE_MARGIN_PX
        draw_circle(frame, (center_x, center_y), radius)
        blinking = now % IDLE_BLINK_PERIOD_SEC < IDLE_BLINK_CLOSED_SEC
        eye_y = center_y - radius * IDLE_EYE_OFFSET_Y_FRAC
        for direction in (-1.0, 1.0):
            eye_x = center_x + direction * radius * IDLE_EYE_OFFSET_X_FRAC
            if blinking:
                draw_line(frame, (eye_x - 1.0, eye_y), (eye_x + 1.0, eye_y))
            else:
                draw_point(frame, eye_x, eye_y)
                draw_point(frame, eye_x, eye_y - 1.0)
        mouth_y = center_y + radius * IDLE_MOUTH_OFFSET_Y_FRAC
        half_width = radius * IDLE_MOUTH_HALF_WIDTH_FRAC
        draw_line(frame, (center_x - half_width, mouth_y), (center_x + half_width, mouth_y))
