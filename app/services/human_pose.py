import logging
import os
import threading
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.services import figure

logger = logging.getLogger(__name__)

CLOSE_FACE_DISTANCE = float(os.getenv("CLOSE_FACE_DISTANCE", "0.9"))
VERY_CLOSE_FACE_DISTANCE = float(os.getenv("VERY_CLOSE_FACE_DISTANCE", "0.5"))

# ---------------------------------------------------------------------------
# MediaPipe Tasks API (>= 0.10) with GPU delegate, falling back to legacy API
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODELS_DIR_ENV = os.getenv("MEDIAPIPE_MODELS_DIR")
_MODELS_DIR_CANDIDATES = [
    _MODELS_DIR_ENV,
    os.path.join(_REPO_ROOT, "models"),
    os.path.join(os.path.dirname(__file__), "models"),
]
_MODELS_DIR = next(
    (p for p in _MODELS_DIR_CANDIDATES if p and os.path.isdir(p)),
    os.path.join(_REPO_ROOT, "models"),
)
_POSE_MODEL_NAME = os.getenv("POSE_MODEL", "pose_landmarker_lite")
_POSE_MODEL = os.path.join(_MODELS_DIR, f"{_POSE_MODEL_NAME}.task")
_FACE_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")

_USE_TASKS_API = False  # set True below if Tasks API + model files are present

# Face-mesh background thread: runs inference without blocking the main loop.
# Written from the bg thread, read from the main thread (GIL protects assignments).
_face_bg_lock = threading.Lock()
_face_bg_frame = [None]  # latest frame to process (overwritten, not queued)
_face_bg_result = [None]  # latest completed result
_face_bg_event = threading.Event()
_face_bg_ts = [0]  # monotonic timestamp counter (bg thread only)


# ---------------------------------------------------------------------------
# Backward-compat wrappers so every existing caller keeps working unchanged.
# ---------------------------------------------------------------------------


class _LandmarkList:
    """Wraps a flat list of landmarks as .landmark[idx] — mimics the legacy API."""

    __slots__ = ("landmark",)

    def __init__(self, landmarks):
        self.landmark = landmarks  # list of NormalizedLandmark / Landmark objects


class _PoseResultsWrapper:
    """Adapts PoseLandmarkerResult to look like mp.solutions.pose.Pose.process()."""

    __slots__ = ("pose_landmarks", "pose_world_landmarks", "segmentation_mask")

    def __init__(self, result, seg_mask=None):
        if result and result.pose_landmarks:
            self.pose_landmarks = _LandmarkList(result.pose_landmarks[0])
            self.pose_world_landmarks = (
                _LandmarkList(result.pose_world_landmarks[0])
                if result.pose_world_landmarks
                else None
            )
        else:
            self.pose_landmarks = None
            self.pose_world_landmarks = None
        self.segmentation_mask = seg_mask


class _FaceLandmarkList:
    """Single face entry for multi_face_landmarks[0].landmark[idx]."""

    __slots__ = ("landmark",)

    def __init__(self, landmarks):
        self.landmark = landmarks


class _FaceMeshResultsWrapper:
    """Adapts FaceLandmarkerResult to look like mp.solutions.face_mesh result."""

    __slots__ = ("multi_face_landmarks",)

    def __init__(self, result):
        if result and result.face_landmarks:
            self.multi_face_landmarks = [_FaceLandmarkList(result.face_landmarks[0])]
        else:
            self.multi_face_landmarks = None


# ---------------------------------------------------------------------------
# Initialise detectors — Tasks API preferred, legacy fallback
# ---------------------------------------------------------------------------


def _face_mesh_bg_worker():
    """Background thread: run face landmarker inference without blocking main loop."""
    ts = 0
    while True:
        _face_bg_event.wait()
        _face_bg_event.clear()
        with _face_bg_lock:
            frame = _face_bg_frame[0]
            _face_bg_frame[0] = None
        if frame is None:
            continue
        h, w = frame.shape[:2]
        target_h = int(FACE_MESH_INPUT_WIDTH * h / w)
        small = cv2.resize(frame, (FACE_MESH_INPUT_WIDTH, target_h), interpolation=cv2.INTER_AREA)
        inp = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inp)
        ts = max(ts + 1, int(time.monotonic() * 1000))
        result = _face_landmarker.detect_for_video(mp_image, ts)
        with _face_bg_lock:
            _face_bg_result[0] = _FaceMeshResultsWrapper(result)


try:
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    logger.info(
        "MediaPipe model resolution models_dir=%s pose_model=%s face_model=%s",
        _MODELS_DIR,
        _POSE_MODEL,
        _FACE_MODEL,
    )

    if not os.path.isfile(_POSE_MODEL):
        raise FileNotFoundError(f"Pose model not found: {_POSE_MODEL}")
    if not os.path.isfile(_FACE_MODEL):
        raise FileNotFoundError(f"Face model not found: {_FACE_MODEL}")

    try:
        _delegate = BaseOptions.Delegate.GPU
        _pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_POSE_MODEL, delegate=_delegate),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_segmentation_masks=True,
            )
        )
    except Exception:
        # GPU delegate unavailable — fall back to CPU
        logger.warning("MediaPipe GPU delegate unavailable, falling back to CPU")
        _delegate = BaseOptions.Delegate.CPU
        _pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_POSE_MODEL, delegate=_delegate),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_segmentation_masks=True,
            )
        )

    _face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_MODEL, delegate=_delegate),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    # Start face mesh background thread (pose inference stays in the main thread).
    _face_bg_thread = threading.Thread(target=_face_mesh_bg_worker, daemon=True)
    _face_bg_thread.start()

    _USE_TASKS_API = True
    _delegate_name = "GPU" if _delegate == BaseOptions.Delegate.GPU else "CPU"
    logger.info("Using MediaPipe Tasks API (%s delegate)", _delegate_name)

except Exception as _tasks_err:
    logger.warning("Tasks API unavailable (%s), using legacy API", _tasks_err)
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh
    except AttributeError:
        from mediapipe.python import solutions as _mp_solutions

        mp_drawing = _mp_solutions.drawing_utils
        mp_pose = _mp_solutions.pose
        mp_face_mesh = _mp_solutions.face_mesh

    _pose_legacy = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        enable_segmentation=True,
    )
    _face_legacy = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# Keep mp_pose available for landmark enum constants used throughout the file.
# With Tasks API, mp_pose is not imported above, so import it separately here.
if "mp_pose" not in globals():
    mp_pose = mp.solutions.pose
    logger.debug("Loaded mp.solutions.pose constants for Tasks API compatibility")


# ---------------------------------------------------------------------------
# Public detector functions
# ---------------------------------------------------------------------------


def get_human_pose(frame: np.ndarray) -> Any:
    """Run pose detection on a BGR frame, returning a pose-results wrapper."""
    input_image = cv2.resize(frame, (60, 60))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    if _USE_TASKS_API:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_image)
        ts_ms = int(time.monotonic() * 1000)
        result = _pose_landmarker.detect_for_video(mp_image, ts_ms)
        seg_mask = None
        if result.segmentation_masks:
            seg_mask = result.segmentation_masks[0].numpy_view().copy()
        return _PoseResultsWrapper(result, seg_mask)
    else:
        return _pose_legacy.process(input_image)


FACE_MESH_INPUT_WIDTH = int(os.getenv("FACE_MESH_INPUT_WIDTH", "256"))


def get_face_mesh(frame):
    """Submit frame to face mesh background thread; return latest available result."""
    if _USE_TASKS_API:
        with _face_bg_lock:
            _face_bg_frame[0] = frame
        _face_bg_event.set()
        with _face_bg_lock:
            return _face_bg_result[0]
    else:
        h, w = frame.shape[:2]
        target_h = int(FACE_MESH_INPUT_WIDTH * h / w)
        small = cv2.resize(frame, (FACE_MESH_INPUT_WIDTH, target_h), interpolation=cv2.INTER_AREA)
        inp = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return _face_legacy.process(inp)


LEFT_EYE_INDICES = [33, 133]
RIGHT_EYE_INDICES = [362, 263]
# Three vertical landmark pairs per eye for a more robust EAR
LEFT_EYE_VERTICAL_INDICES = [(159, 145), (160, 144), (161, 163)]
RIGHT_EYE_VERTICAL_INDICES = [(386, 374), (385, 380), (384, 381)]
EYE_OPEN_RATIO = float(os.getenv("EYE_OPEN_RATIO", "0.2"))
LEFT_EYEBROW_INDICES = [46, 53, 52, 65, 55]
RIGHT_EYEBROW_INDICES = [276, 283, 282, 295, 285]
MOUTH_UPPER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
MOUTH_LOWER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_CLOSED_GAP_PX = float(os.getenv("MOUTH_CLOSED_GAP_PX", "1.1"))
FACE_PERSPECTIVE_STRENGTH = float(os.getenv("FACE_PERSPECTIVE_STRENGTH", "0.32"))
FACE_PERSPECTIVE_PIVOT_Y = float(os.getenv("FACE_PERSPECTIVE_PIVOT_Y", "0.30"))
MOUTH_Y_OFFSET_PX = int(os.getenv("MOUTH_Y_OFFSET_PX", "1"))
FACE_FEATURE_Y_OFFSET_PX = int(os.getenv("FACE_FEATURE_Y_OFFSET_PX", "-1"))
SILHOUETTE_Y_OFFSET_PX = int(os.getenv("SILHOUETTE_Y_OFFSET_PX", "1"))


def should_draw_face_features(estimated_distance: float | None) -> bool:
    """Whether the viewer is close enough to render basic face features."""
    return estimated_distance is not None and estimated_distance < CLOSE_FACE_DISTANCE


def should_draw_detailed_face_features(estimated_distance: float | None) -> bool:
    """Whether the viewer is close enough to render detailed face features."""
    return estimated_distance is not None and estimated_distance < VERY_CLOSE_FACE_DISTANCE


def _apply_face_perspective_correction(x, y):
    # Camera is mounted above the screen, so pull lower-face landmarks upward.
    strength = max(0.0, FACE_PERSPECTIVE_STRENGTH)
    pivot_y = min(1.0, max(0.0, FACE_PERSPECTIVE_PIVOT_Y))
    y_corrected = y - strength * max(0.0, y - pivot_y)
    return x, min(1.0, max(0.0, y_corrected))


def _face_point_to_panel_xy(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """Float panel coords of a raw normalized face point: perspective-corrected, mirrored.

    The single projection primitive shared by the face-feature renderer and
    ``face_feature_anchor``, so the two cannot drift apart.
    """
    x, y = _apply_face_perspective_correction(x, y)
    return width - x * width, y * height


def face_feature_anchor(
    x_mirrored_norm: float, y_norm: float, width: int, height: int
) -> tuple[float, float]:
    """Float panel point where ``draw_face_features`` anchors a landmark.

    Takes mirrored normalized coordinates (x already flipped, matching the
    caricature's face basis) and applies the same projection and eye-row
    offset as the face-feature renderer, so the caricature's entry and exit
    zooms land where the sandfall silhouette's face was drawn.
    """
    x, y = _face_point_to_panel_xy(1.0 - x_mirrored_norm, y_norm, width, height)
    return x, y + FACE_FEATURE_Y_OFFSET_PX


def _draw_line(dots, x0, y0, x1, y1, value=0):
    h, w = dots.shape
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        if 0 <= x0 < w and 0 <= y0 < h:
            dots[y0, x0] = value
        return

    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    dots[ys[valid], xs[valid]] = value


def _is_eye_open(landmarks, corner_indices, vertical_pairs):
    """Return True if the eye aspect ratio (EAR) exceeds EYE_OPEN_RATIO.

    vertical_pairs is a list of (top_idx, bottom_idx) tuples.
    EAR = mean(vertical eyelid gap) / horizontal eye width
    Only the y-axis is used for the vertical component since that is
    what changes with a blink; Euclidean distance would be inflated by
    any lateral x-jitter in the landmarks.
    """
    lx = landmarks[corner_indices[0]]
    rx = landmarks[corner_indices[1]]
    horizontal = abs(lx.x - rx.x)
    if horizontal == 0:
        return True
    vertical = np.mean([abs(landmarks[t].y - landmarks[b].y) for t, b in vertical_pairs])
    return (vertical / horizontal) >= EYE_OPEN_RATIO


def _landmarks_to_panel_points(landmarks, indices, width, height):
    points = []
    for idx in indices:
        x, y = _face_point_to_panel_xy(landmarks[idx].x, landmarks[idx].y, width, height)
        points.append((int(x), int(y)))
    return points


def _draw_points_polyline(dots, points, value=0):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        _draw_line(
            dots, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], value=value
        )


def _offset_points(points, dx, dy, width, height):
    shifted = []
    for x, y in points:
        nx = min(width - 1, max(0, x + dx))
        ny = min(height - 1, max(0, y + dy))
        shifted.append((nx, ny))
    return shifted


def _pose_landmark_to_panel_xy(landmark: Any, width: int, height: int) -> tuple[float, float]:
    """Map a pose landmark to unclamped float panel coords (may lie off-panel)."""
    x, y = _apply_face_perspective_correction(landmark.x, landmark.y)
    return width - (x * width), (y * height) + SILHOUETTE_Y_OFFSET_PX


def draw_face_features(
    dots: np.ndarray,
    face_mesh_results: Any,
    width: int,
    height: int,
    draw_details: bool = False,
    value: int = 0,
) -> np.ndarray:
    """Overlay eyes and mouth (plus brows when ``draw_details``) from face-mesh landmarks.

    ``value`` is the pixel value drawn: 0 punches features into a filled head
    (pose mode), 1 lights them inside an outlined silhouette (sandfall mode).
    """
    if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
        return dots

    landmarks = face_mesh_results.multi_face_landmarks[0].landmark

    left_eye_points = _landmarks_to_panel_points(landmarks, LEFT_EYE_INDICES, width, height)
    right_eye_points = _landmarks_to_panel_points(landmarks, RIGHT_EYE_INDICES, width, height)
    left_eye = (
        int(np.mean([p[0] for p in left_eye_points])),
        int(np.mean([p[1] for p in left_eye_points])),
    )
    right_eye = (
        int(np.mean([p[0] for p in right_eye_points])),
        int(np.mean([p[1] for p in right_eye_points])),
    )
    left_eye = (left_eye[0], left_eye[1] + FACE_FEATURE_Y_OFFSET_PX)
    right_eye = (right_eye[0], right_eye[1] + FACE_FEATURE_Y_OFFSET_PX)

    if 0 <= left_eye[0] < width and 0 <= left_eye[1] < height:
        if _is_eye_open(landmarks, LEFT_EYE_INDICES, LEFT_EYE_VERTICAL_INDICES):
            dots[left_eye[1], left_eye[0]] = value
    if 0 <= right_eye[0] < width and 0 <= right_eye[1] < height:
        if _is_eye_open(landmarks, RIGHT_EYE_INDICES, RIGHT_EYE_VERTICAL_INDICES):
            dots[right_eye[1], right_eye[0]] = value

    upper_points = _landmarks_to_panel_points(landmarks, MOUTH_UPPER_INDICES, width, height)
    lower_points = _landmarks_to_panel_points(landmarks, MOUTH_LOWER_INDICES, width, height)
    mouth_total_offset = MOUTH_Y_OFFSET_PX + FACE_FEATURE_Y_OFFSET_PX
    upper_points = _offset_points(upper_points, 0, mouth_total_offset, width, height)
    lower_points = _offset_points(lower_points, 0, mouth_total_offset, width, height)

    mouth_gap = np.mean(
        [abs(upper_points[i][1] - lower_points[i][1]) for i in range(len(upper_points))]
    )
    if mouth_gap <= MOUTH_CLOSED_GAP_PX:
        center_points = [
            (
                (upper_points[i][0] + lower_points[i][0]) // 2,
                (upper_points[i][1] + lower_points[i][1]) // 2,
            )
            for i in range(len(upper_points))
        ]
        _draw_points_polyline(dots, center_points, value=value)
    else:
        _draw_points_polyline(dots, upper_points, value=value)
        _draw_points_polyline(dots, lower_points, value=value)

    if draw_details:
        left_brow_points = _landmarks_to_panel_points(
            landmarks, LEFT_EYEBROW_INDICES, width, height
        )
        right_brow_points = _landmarks_to_panel_points(
            landmarks, RIGHT_EYEBROW_INDICES, width, height
        )
        left_brow_points = _offset_points(
            left_brow_points, 0, FACE_FEATURE_Y_OFFSET_PX, width, height
        )
        right_brow_points = _offset_points(
            right_brow_points, 0, FACE_FEATURE_Y_OFFSET_PX, width, height
        )

        # Shift each eyebrow up so its lowest point is at least 1 pixel above the eye.
        # "At least 1 pixel away" = at least one blank row between them, so gap >= 2.
        for brow_points, eye in ((left_brow_points, left_eye), (right_brow_points, right_eye)):
            max_brow_y = max(p[1] for p in brow_points)
            gap = eye[1] - max_brow_y
            if gap < 2:
                dy = 2 - gap
                brow_points[:] = _offset_points(brow_points, 0, -dy, width, height)

        _draw_points_polyline(dots, left_brow_points, value=value)
        _draw_points_polyline(dots, right_brow_points, value=value)

    return dots


def _face_feature_panel_points(
    face_mesh_results: Any, width: int, height: int
) -> list[tuple[float, float]]:
    """Panel points that `draw_face_features` will draw, for sizing the head disc."""
    if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
        return []

    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    points: list[tuple[float, float]] = []

    for eye_indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        eye_points = _landmarks_to_panel_points(landmarks, eye_indices, width, height)
        points.append(
            (
                float(np.mean([p[0] for p in eye_points])),
                float(np.mean([p[1] for p in eye_points])) + FACE_FEATURE_Y_OFFSET_PX,
            )
        )

    mouth_total_offset = MOUTH_Y_OFFSET_PX + FACE_FEATURE_Y_OFFSET_PX
    for mouth_indices in (MOUTH_UPPER_INDICES, MOUTH_LOWER_INDICES):
        mouth_points = _landmarks_to_panel_points(landmarks, mouth_indices, width, height)
        points.extend(_offset_points(mouth_points, 0, mouth_total_offset, width, height))

    for brow_indices in (LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES):
        brow_points = _landmarks_to_panel_points(landmarks, brow_indices, width, height)
        points.extend(_offset_points(brow_points, 0, FACE_FEATURE_Y_OFFSET_PX, width, height))

    return points


# Cross-frame head smoothing for the single tracked viewer; resets itself
# after a pause, so a new viewer doesn't inherit the previous head.
_head_state = figure.HeadState()


def display_human_pose(
    pose_results: Any,
    width: int,
    height: int,
    estimated_distance: float | None = None,
    face_mesh_results: Any = None,
) -> np.ndarray:
    """Render the constructed character (torso, head, limbs, face) to a panel-sized frame."""
    dots = np.zeros((height, width), dtype=np.uint8)
    if pose_results is None or pose_results.pose_landmarks is None:
        return dots

    def to_panel(landmark: Any) -> tuple[float, float]:
        return _pose_landmark_to_panel_xy(landmark, width, height)

    draw_faces = should_draw_face_features(estimated_distance)
    cover_points = (
        _face_feature_panel_points(face_mesh_results, width, height) if draw_faces else None
    )
    figure.draw_figure(
        dots,
        pose_results.pose_landmarks.landmark,
        to_panel,
        head_cover_points=cover_points,
        head_state=_head_state,
    )

    if draw_faces:
        draw_details = should_draw_detailed_face_features(estimated_distance)
        dots = draw_face_features(dots, face_mesh_results, width, height, draw_details=draw_details)

    return dots


def eyes_visible_and_facing_camera(pose_results: Any) -> tuple[bool, str, float | None]:
    """Return (facing, reason, angle_deg): whether the viewer faces the camera."""
    # Check if all relevant landmarks are detected with reasonable confidence
    # - Left eye: 2 (inner), 3 (outer)
    # - Right eye: 5 (inner), 4 (outer)
    if pose_results is None or pose_results.pose_landmarks is None:
        return False, "Pose landmarks not detected", None

    landmarks = pose_results.pose_landmarks.landmark
    eye_landmarks = [landmarks[0], landmarks[2], landmarks[3], landmarks[4], landmarks[5]]

    # Visibility threshold
    confidence_threshold = 0.7

    # Check if all eye landmarks are visible with high confidence
    all_eye_landmarks_visible = all(
        landmark.visibility > confidence_threshold for landmark in eye_landmarks
    )
    if not all_eye_landmarks_visible:
        return False, "Eye landmarks not clearly visible", None

    # Use the 2D normalised pose_landmarks for the angle calculation.
    # pose_landmarks.z has the same scale as x and measures depth relative to
    # the hip midpoint (smaller = closer to camera).  Unlike pose_world_landmarks.z
    # it is not affected by the gravity-aligned body frame used by GHUM/Tasks API,
    # so it correctly reads ~0 depth-difference between the eyes when the person
    # faces the camera directly, regardless of camera tilt.
    left_eye = landmarks[2]  # left eye centre
    right_eye = landmarks[5]  # right eye centre

    dx = right_eye.x - left_eye.x  # lateral (negative when facing camera)
    dz = right_eye.z - left_eye.z  # depth difference

    # Rotate 90° around y-axis to get the face normal direction
    face_direction = np.array([dz, 0.0, -dx])
    norm = np.linalg.norm(face_direction)
    if norm == 0:
        return False, "Degenerate eye vector", None

    face_direction_normalized = face_direction / norm
    dot_product = np.dot(face_direction_normalized, np.array([0.0, 0.0, 1.0]))
    angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    # face_direction[2] = -dx; positive when right_eye.x < left_eye.x,
    # i.e. when the person faces the camera (left eye is on the camera's right).
    facing_forward = face_direction[2] > 0

    # Define threshold for the angle
    max_angle_threshold = 30  # degrees

    # Check if the face is looking at the camera within the threshold
    facing_camera = facing_forward and angle_deg < max_angle_threshold

    if facing_camera:
        return True, f"Facing camera: {angle_deg:.1f}°", angle_deg
    else:
        if not facing_forward:
            return False, f"Face turned away from camera: {angle_deg:.1f}°", angle_deg
        else:
            return False, f"Not directly facing camera: {angle_deg:.1f}°", angle_deg


def estimate_distance(pose_results: Any) -> tuple[float | None, list]:
    """Return (distance, per-landmark estimates) from apparent body size."""
    FOCAL_SCALE = float(os.getenv("FOCAL_SCALE", "1.0"))

    # Estimate distance based on the size of the person in the frame
    if pose_results is None or pose_results.pose_landmarks is None:
        return None, []

    KNOWN_DISTANCES: list[dict[str, Any]] = [
        {"landmark0": 11, "landmark1": 12, "value": 0.45, "name": "shoulder width"},
        {"landmark0": 23, "landmark1": 24, "value": 0.37, "name": "hip width"},
        {"landmark0": 11, "landmark1": 23, "value": 0.4, "name": "left shoulder to hip"},
        {"landmark0": 12, "landmark1": 24, "value": 0.4, "name": "right shoulder to hip"},
        {"landmark0": 2, "landmark1": 5, "value": 0.06, "name": "eye distance"},
        {"landmark0": 7, "landmark1": 8, "value": 0.14, "name": "ear distance"},
    ]
    landmarks = pose_results.pose_landmarks.landmark

    distance_estimates = []
    for known_distance in KNOWN_DISTANCES:
        landmark0 = landmarks[known_distance["landmark0"]]
        landmark1 = landmarks[known_distance["landmark1"]]

        # Check if both landmarks are visible
        if landmark0.visibility < 0.5 or landmark1.visibility < 0.5:
            continue

        # Calculate the distance between the two landmarks
        point0 = np.array([landmark0.x, landmark0.y, landmark0.z])
        point1 = np.array([landmark1.x, landmark1.y, landmark1.z])
        distance = np.linalg.norm(point0 - point1)

        # Discard distances with large z component
        # because they are likely not very accurate
        # z_distance = abs(point0[2] - point1[2])
        # if z_distance > 0.5 * distance:
        #    continue

        distance_estimates.append(
            ((known_distance["value"] * FOCAL_SCALE) / distance, distance, known_distance["name"])
        )

    if len(distance_estimates) == 0:
        return None, []

    return sum(value for value, _, _ in distance_estimates) / len(
        distance_estimates
    ), distance_estimates


def get_right_index_finger_position(
    pose_results: Any = None,
) -> tuple[float | None, float | None]:
    """Return the right index finger's normalized (x, y), or (None, None)."""
    if pose_results is not None and pose_results.pose_landmarks is not None:
        landmarks = pose_results.pose_landmarks.landmark
        # Get right index finger if it is clearly visible otherwise use right thumb if it is more visible otherwise estimate from both
        right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
        if right_index.visibility > 0.7:
            return right_index.x, right_index.y
        if right_thumb.visibility > 0.7:
            return right_thumb.x, right_thumb.y
        if right_index.visibility > 0.3 and right_thumb.visibility > 0.3:
            # Estimate position as average of both
            x = (right_index.x + right_thumb.x) / 2
            y = (right_index.y + right_thumb.y) / 2
            return x, y

    return None, None


def is_right_index_in_top_right_corner(
    pose_results: Any = None,
) -> tuple[bool, float | None, float | None]:
    """Return (in_corner, x, y) for the right index finger's panel position."""
    x, y = get_right_index_finger_position(pose_results)

    # Check if the index finger is in the top right corner (e.g., top 20% and right 20%)
    if x is not None and y is not None and x < 0.2 and y < 0.2:
        return True, x, y

    return False, x, y


def is_arms_crossed(pose_results: Any = None) -> bool:
    """Return True if both wrists are crossed at chest height (self-hug pose)."""
    if pose_results is None or pose_results.pose_landmarks is None:
        return False

    landmarks = pose_results.pose_landmarks.landmark
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

    if right_wrist.visibility < 0.5 or left_wrist.visibility < 0.5:
        return False

    # In image coords, person's right side is at lower x than left side.
    # Crossed: right wrist has moved past left wrist (x values swap).
    if right_wrist.x <= left_wrist.x + 0.05:
        return False

    # Both wrists must be at torso height (between shoulders and hips).
    torso_top = min(right_shoulder.y, left_shoulder.y)
    torso_bottom = max(right_hip.y, left_hip.y)
    if not (torso_top < right_wrist.y < torso_bottom):
        return False
    if not (torso_top < left_wrist.y < torso_bottom):
        return False

    return True


def is_left_hand_raised(pose_results: Any = None) -> bool:
    """Return True if the left wrist is raised above the nose (hand above head)."""
    if pose_results is None or pose_results.pose_landmarks is None:
        return False
    landmarks = pose_results.pose_landmarks.landmark
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    if left_wrist.visibility < 0.5:
        return False
    # In image coords y increases downward; wrist above nose means wrist.y < nose.y
    return left_wrist.y < nose.y - 0.05


def draw_right_index_pointer(
    frame: np.ndarray, pose_results: Any = None, size: int = 1
) -> np.ndarray:
    """Draw a pointer dot at the right index finger position on ``frame``."""
    finger_x, finger_y = get_right_index_finger_position(pose_results)
    return draw_pointer(frame, finger_x, finger_y, size=size, mirror_x=True)


def draw_pointer(
    frame: np.ndarray,
    finger_x: float | None,
    finger_y: float | None,
    size: int = 1,
    mirror_x: bool = True,
) -> np.ndarray:
    """Draw a pointer dot at normalized (finger_x, finger_y) on ``frame``."""
    height, width = frame.shape

    if finger_x is not None and finger_y is not None:
        if mirror_x:
            x = int(width - (finger_x * width))
        else:
            x = int(finger_x * width)
        y = int(finger_y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), size // 2, 0, -1)
            cv2.circle(frame, (x, y), size // 2, 1, 1)

    return frame
