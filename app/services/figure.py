"""Construct the pose-mode character geometrically from pose landmarks.

Builds a filled torso polygon, head disc, neck and tapered limbs directly from
MediaPipe pose landmarks. Because the torso corners and the arm roots are the
same shoulder landmarks, the limbs attach seamlessly by construction — no
stitching heuristics against a segmentation mask.

The module is deliberately importable without mediapipe: landmark indices are
plain ints and the landmark→panel mapping is injected by the caller, so tests
can drive it with simple fakes.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.modes.contracts import Frame
from app.services.draw import fill_circle, tapered_capsule

logger = logging.getLogger(__name__)

PanelPoint = tuple[float, float]
LandmarkMapper = Callable[[Any], PanelPoint]

# MediaPipe pose landmark indices (plain ints, see module docstring).
NOSE = 0
L_EAR = 7
R_EAR = 8
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_INDEX = 19
R_INDEX = 20
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28

ARM_MIN_VISIBILITY = float(os.getenv("ARM_MIN_VISIBILITY", "0.45"))
FOREARM_MIN_VISIBILITY = float(os.getenv("FOREARM_MIN_VISIBILITY", "0.20"))
ARM_SHOULDER_RADIUS_PX = float(os.getenv("ARM_SHOULDER_RADIUS_PX", "2.0"))
ARM_ELBOW_RADIUS_PX = float(os.getenv("ARM_ELBOW_RADIUS_PX", "1.4"))
ARM_WRIST_RADIUS_PX = float(os.getenv("ARM_WRIST_RADIUS_PX", "1.0"))
ARM_OUTLINE_RADIUS_PX = int(os.getenv("ARM_OUTLINE_RADIUS_PX", "1"))
ARM_SHOULDER_MERGE_RADIUS_PX = float(os.getenv("ARM_SHOULDER_MERGE_RADIUS_PX", "3.0"))
ARM_SHOULDER_INSET_PX = float(os.getenv("ARM_SHOULDER_INSET_PX", "1.5"))
HAND_RADIUS_PX = float(os.getenv("HAND_RADIUS_PX", "1.5"))
HAND_EXTEND_PX = float(os.getenv("HAND_EXTEND_PX", "1.5"))
TORSO_MIN_VISIBILITY = float(os.getenv("TORSO_MIN_VISIBILITY", "0.45"))
TORSO_SHOULDER_EXTEND_FRAC = float(os.getenv("TORSO_SHOULDER_EXTEND_FRAC", "0.15"))
TORSO_HIP_EXTEND_FRAC = float(os.getenv("TORSO_HIP_EXTEND_FRAC", "0.10"))
HIP_MIN_VISIBILITY = float(os.getenv("HIP_MIN_VISIBILITY", "0.45"))
HIP_SYNTH_MARGIN_PX = float(os.getenv("HIP_SYNTH_MARGIN_PX", "2.0"))
LEG_MIN_VISIBILITY = float(os.getenv("LEG_MIN_VISIBILITY", "0.45"))
LEG_HIP_RADIUS_PX = float(os.getenv("LEG_HIP_RADIUS_PX", "2.0"))
LEG_KNEE_RADIUS_PX = float(os.getenv("LEG_KNEE_RADIUS_PX", "1.4"))
LEG_ANKLE_RADIUS_PX = float(os.getenv("LEG_ANKLE_RADIUS_PX", "1.0"))
HEAD_MIN_VISIBILITY = float(os.getenv("HEAD_MIN_VISIBILITY", "0.5"))
HEAD_RADIUS_EAR_SCALE = float(os.getenv("HEAD_RADIUS_EAR_SCALE", "1.3"))
HEAD_RADIUS_SHOULDER_FRAC = float(os.getenv("HEAD_RADIUS_SHOULDER_FRAC", "0.30"))
HEAD_MIN_RADIUS_PX = float(os.getenv("HEAD_MIN_RADIUS_PX", "2.0"))
HEAD_MAX_RADIUS_FRAC = float(os.getenv("HEAD_MAX_RADIUS_FRAC", "0.25"))
HEAD_FEATURE_MARGIN_PX = float(os.getenv("HEAD_FEATURE_MARGIN_PX", "1.0"))
HEAD_CENTER_SMOOTHING_ALPHA = float(os.getenv("HEAD_CENTER_SMOOTHING_ALPHA", "0.6"))
HEAD_RADIUS_SMOOTHING_ALPHA = float(os.getenv("HEAD_RADIUS_SMOOTHING_ALPHA", "0.3"))
HEAD_SMOOTHING_RESET_SECONDS = float(os.getenv("HEAD_SMOOTHING_RESET_SECONDS", "1.0"))
NECK_MIN_RADIUS_PX = float(os.getenv("NECK_MIN_RADIUS_PX", "1.3"))
NECK_SHOULDER_FRAC = float(os.getenv("NECK_SHOULDER_FRAC", "0.16"))


@dataclass
class HeadState:
    """EMA-smoothed head geometry carried across frames.

    Landmark distances jitter frame to frame, and the head radius is derived
    from them, so an unsmoothed head visibly pulses. The caller keeps one
    instance per viewer and passes it to :func:`draw_figure`; after
    ``HEAD_SMOOTHING_RESET_SECONDS`` without an update the state resets so a
    new viewer doesn't inherit the previous head.
    """

    center: PanelPoint | None = None
    radius: float | None = None
    updated_at: float = 0.0

    def smooth(self, center: PanelPoint, radius: float) -> tuple[PanelPoint, float]:
        """Blend new head geometry into the running average and return it."""
        now = time.monotonic()
        stale = (now - self.updated_at) > HEAD_SMOOTHING_RESET_SECONDS
        if self.center is None or self.radius is None or stale:
            self.center = center
            self.radius = radius
        else:
            center_alpha = HEAD_CENTER_SMOOTHING_ALPHA
            self.center = (
                self.center[0] + center_alpha * (center[0] - self.center[0]),
                self.center[1] + center_alpha * (center[1] - self.center[1]),
            )
            self.radius += HEAD_RADIUS_SMOOTHING_ALPHA * (radius - self.radius)
        self.updated_at = now
        return self.center, self.radius


_ARM_SEGMENTS = (
    (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX),
    (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX),
)
_LEG_SEGMENTS = (
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
)


def _point(
    landmarks: Sequence[Any],
    idx: int,
    to_panel: LandmarkMapper,
    min_visibility: float,
    fallback_idx: int | None = None,
) -> PanelPoint | None:
    """Map landmark ``idx`` to panel coords, or ``None`` when not visible enough."""
    primary = landmarks[idx]
    if primary.visibility >= min_visibility:
        return to_panel(primary)
    if fallback_idx is None:
        return None
    fallback = landmarks[fallback_idx]
    if fallback.visibility >= min_visibility:
        return to_panel(fallback)
    return None


def _midpoint(a: PanelPoint, b: PanelPoint) -> PanelPoint:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _distance(a: PanelPoint, b: PanelPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extend_from(origin: PanelPoint, point: PanelPoint, frac: float) -> PanelPoint:
    """Push ``point`` away from ``origin`` by ``frac`` of their distance."""
    scale = 1.0 + frac
    return (
        origin[0] + (point[0] - origin[0]) * scale,
        origin[1] + (point[1] - origin[1]) * scale,
    )


def _draw_torso(
    frame: Frame,
    landmarks: Sequence[Any],
    to_panel: LandmarkMapper,
    l_shoulder: PanelPoint,
    r_shoulder: PanelPoint,
) -> None:
    """Fill the shoulders→hips torso quad, extending off-panel on waist-up framing."""
    height = frame.shape[0]
    l_hip = _point(landmarks, L_HIP, to_panel, HIP_MIN_VISIBILITY)
    r_hip = _point(landmarks, R_HIP, to_panel, HIP_MIN_VISIBILITY)
    if l_hip is None or r_hip is None:
        # Waist-up framing: run the torso off the bottom edge instead of
        # ending it mid-chest at an unreliable hip estimate.
        synth_y = float(height) + HIP_SYNTH_MARGIN_PX
        l_hip = (l_shoulder[0], synth_y)
        r_hip = (r_shoulder[0], synth_y)

    # Joint centers sit inside the real silhouette, so widen the quad corners
    # slightly away from their midlines.
    shoulder_mid = _midpoint(l_shoulder, r_shoulder)
    hip_mid = _midpoint(l_hip, r_hip)
    quad = [
        _extend_from(shoulder_mid, l_shoulder, TORSO_SHOULDER_EXTEND_FRAC),
        _extend_from(shoulder_mid, r_shoulder, TORSO_SHOULDER_EXTEND_FRAC),
        _extend_from(hip_mid, r_hip, TORSO_HIP_EXTEND_FRAC),
        _extend_from(hip_mid, l_hip, TORSO_HIP_EXTEND_FRAC),
    ]
    points = np.round(np.array(quad, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(frame, [points], 1)


def _draw_legs(frame: Frame, landmarks: Sequence[Any], to_panel: LandmarkMapper) -> None:
    """Draw tapered hip→knee→ankle capsules for each sufficiently visible leg."""
    for hip_idx, knee_idx, ankle_idx in _LEG_SEGMENTS:
        hip = _point(landmarks, hip_idx, to_panel, LEG_MIN_VISIBILITY)
        knee = _point(landmarks, knee_idx, to_panel, LEG_MIN_VISIBILITY)
        if hip is None or knee is None:
            continue
        tapered_capsule(frame, hip, knee, r0=LEG_HIP_RADIUS_PX, r1=LEG_KNEE_RADIUS_PX)
        ankle = _point(landmarks, ankle_idx, to_panel, LEG_MIN_VISIBILITY)
        if ankle is not None:
            tapered_capsule(frame, knee, ankle, r0=LEG_KNEE_RADIUS_PX, r1=LEG_ANKLE_RADIUS_PX)


def _head_geometry(
    landmarks: Sequence[Any],
    to_panel: LandmarkMapper,
    l_shoulder: PanelPoint | None,
    r_shoulder: PanelPoint | None,
    head_cover_points: Sequence[PanelPoint] | None,
    width: int,
) -> tuple[PanelPoint, float] | None:
    """Compute the head disc center and radius, or ``None`` when no head is visible.

    With face-feature cover points the disc is fitted snugly around their
    bounding box: centering on the features (rather than the ear line) keeps
    the disc small and immune to pose/face-mesh misalignment. Without them the
    size comes from ear spacing, falling back to shoulder width.
    """
    if head_cover_points:
        xs = [p[0] for p in head_cover_points]
        ys = [p[1] for p in head_cover_points]
        center: PanelPoint = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        radius = max(_distance(center, p) for p in head_cover_points) + HEAD_FEATURE_MARGIN_PX
    else:
        l_ear = _point(landmarks, L_EAR, to_panel, HEAD_MIN_VISIBILITY)
        r_ear = _point(landmarks, R_EAR, to_panel, HEAD_MIN_VISIBILITY)
        if l_ear is not None and r_ear is not None:
            center = _midpoint(l_ear, r_ear)
            radius = HEAD_RADIUS_EAR_SCALE * _distance(l_ear, r_ear) / 2.0
        else:
            nose = _point(landmarks, NOSE, to_panel, HEAD_MIN_VISIBILITY)
            if nose is None:
                return None
            center = nose
            radius = 0.0
        if radius <= 0.0 and l_shoulder is not None and r_shoulder is not None:
            radius = HEAD_RADIUS_SHOULDER_FRAC * _distance(l_shoulder, r_shoulder)

    radius = min(max(radius, HEAD_MIN_RADIUS_PX), HEAD_MAX_RADIUS_FRAC * width)
    return center, radius


def _draw_head(
    frame: Frame,
    landmarks: Sequence[Any],
    to_panel: LandmarkMapper,
    l_shoulder: PanelPoint | None,
    r_shoulder: PanelPoint | None,
    head_cover_points: Sequence[PanelPoint] | None,
    head_state: HeadState | None,
) -> None:
    """Draw the neck and head disc, smoothing size/position across frames."""
    geometry = _head_geometry(
        landmarks, to_panel, l_shoulder, r_shoulder, head_cover_points, frame.shape[1]
    )
    if geometry is None:
        return
    center, radius = geometry
    if head_state is not None:
        center, radius = head_state.smooth(center, radius)

    if l_shoulder is not None and r_shoulder is not None:
        neck_base = _midpoint(l_shoulder, r_shoulder)
        # Neck thickness scales with shoulder width so it stays proportional
        # as the viewer moves closer or farther.
        neck_radius = max(
            NECK_MIN_RADIUS_PX, NECK_SHOULDER_FRAC * _distance(l_shoulder, r_shoulder)
        )
        tapered_capsule(frame, center, neck_base, r0=neck_radius, r1=neck_radius)
    fill_circle(frame, center, radius)


def _offset_toward(point: PanelPoint, toward: PanelPoint, distance: float) -> PanelPoint:
    """Move ``point`` by ``distance`` px toward ``toward`` (negative moves away)."""
    dx = toward[0] - point[0]
    dy = toward[1] - point[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return point
    return point[0] + dx / length * distance, point[1] + dy / length * distance


def _draw_hand(
    mask: Frame,
    landmarks: Sequence[Any],
    index_idx: int,
    to_panel: LandmarkMapper,
    elbow: PanelPoint,
    wrist: PanelPoint,
) -> None:
    """Stamp a hand disc at the end of the forearm.

    Centered between wrist and index landmark when the index is visible,
    otherwise extended past the wrist along the forearm direction.
    """
    index_tip = _point(landmarks, index_idx, to_panel, FOREARM_MIN_VISIBILITY)
    if index_tip is not None:
        center = _midpoint(wrist, index_tip)
    else:
        center = _offset_toward(wrist, elbow, -HAND_EXTEND_PX)
    fill_circle(mask, center, HAND_RADIUS_PX)


def _overlay_arm(
    frame: Frame,
    landmarks: Sequence[Any],
    segment: tuple[int, int, int, int],
    to_panel: LandmarkMapper,
) -> None:
    """Draw one tapered arm with a hand, outlining it where it crosses drawn body."""
    shoulder_idx, elbow_idx, wrist_idx, index_idx = segment
    shoulder = _point(landmarks, shoulder_idx, to_panel, ARM_MIN_VISIBILITY)
    elbow = _point(landmarks, elbow_idx, to_panel, ARM_MIN_VISIBILITY)
    if shoulder is None or elbow is None:
        return

    # Inset the upper-arm start toward the elbow so the capsule's round cap
    # tops out at the shoulder landmark instead of bulging above the torso's
    # shoulder line (which read as high, rounded shoulders).
    upper_arm_len = _distance(shoulder, elbow)
    inset = min(ARM_SHOULDER_INSET_PX, upper_arm_len / 2.0)
    arm_root = _offset_toward(shoulder, elbow, inset)

    arm_mask = np.zeros_like(frame)
    tapered_capsule(arm_mask, arm_root, elbow, r0=ARM_SHOULDER_RADIUS_PX, r1=ARM_ELBOW_RADIUS_PX)
    wrist = _point(landmarks, wrist_idx, to_panel, FOREARM_MIN_VISIBILITY, fallback_idx=index_idx)
    if wrist is not None:
        tapered_capsule(arm_mask, elbow, wrist, r0=ARM_ELBOW_RADIUS_PX, r1=ARM_WRIST_RADIUS_PX)
        _draw_hand(arm_mask, landmarks, index_idx, to_panel, elbow, wrist)

    if ARM_OUTLINE_RADIUS_PX > 0:
        kernel_size = (ARM_OUTLINE_RADIUS_PX * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(arm_mask, kernel, iterations=1)
        outline = (dilated == 1) & (arm_mask == 0) & (frame == 1)
        # Keep the shoulder attachment seamless: no dark ring in the merge zone.
        merge_zone = np.zeros_like(frame)
        fill_circle(merge_zone, shoulder, ARM_SHOULDER_MERGE_RADIUS_PX)
        outline &= merge_zone == 0
        frame[outline] = 0

    frame[arm_mask == 1] = 1


def draw_figure(
    frame: Frame,
    landmarks: Sequence[Any],
    to_panel: LandmarkMapper,
    head_cover_points: Sequence[PanelPoint] | None = None,
    head_state: HeadState | None = None,
) -> None:
    """Draw the constructed character (torso, legs, head, arms) onto ``frame``.

    Args:
        frame: 1-bit ``uint8`` panel frame, modified in place.
        landmarks: 33 pose landmarks exposing ``.x``/``.y``/``.visibility``.
        to_panel: maps a landmark to (possibly off-panel) float panel coords.
        head_cover_points: panel points the head disc should fit around (face
            features drawn later), or ``None``.
        head_state: cross-frame smoothing state for the head, or ``None`` to
            render each frame independently.
    """
    l_shoulder = _point(landmarks, L_SHOULDER, to_panel, TORSO_MIN_VISIBILITY)
    r_shoulder = _point(landmarks, R_SHOULDER, to_panel, TORSO_MIN_VISIBILITY)
    if l_shoulder is not None and r_shoulder is not None:
        _draw_torso(frame, landmarks, to_panel, l_shoulder, r_shoulder)
    _draw_legs(frame, landmarks, to_panel)
    _draw_head(frame, landmarks, to_panel, l_shoulder, r_shoulder, head_cover_points, head_state)
    # Arms last, sequentially, so each outlines against body and prior arm.
    for segment in _ARM_SEGMENTS:
        _overlay_arm(frame, landmarks, segment, to_panel)
