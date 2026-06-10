import mediapipe as mp
import cv2
import numpy as np
import os
import time
import threading

CLOSE_FACE_DISTANCE = float(os.getenv('CLOSE_FACE_DISTANCE', '0.9'))
VERY_CLOSE_FACE_DISTANCE = float(os.getenv('VERY_CLOSE_FACE_DISTANCE', '0.5'))

# ---------------------------------------------------------------------------
# MediaPipe Tasks API (>= 0.10) with GPU delegate, falling back to legacy API
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
_POSE_MODEL_NAME = os.getenv('POSE_MODEL', 'pose_landmarker_lite')
_POSE_MODEL   = os.path.join(_MODELS_DIR, f'{_POSE_MODEL_NAME}.task')
_FACE_MODEL   = os.path.join(_MODELS_DIR, 'face_landmarker.task')

_USE_TASKS_API = False  # set True below if Tasks API + model files are present

# Face-mesh background thread: runs inference without blocking the main loop.
# Written from the bg thread, read from the main thread (GIL protects assignments).
_face_bg_lock   = threading.Lock()
_face_bg_frame  = [None]   # latest frame to process (overwritten, not queued)
_face_bg_result = [None]   # latest completed result
_face_bg_event  = threading.Event()
_face_bg_ts     = [0]      # monotonic timestamp counter (bg thread only)


# ---------------------------------------------------------------------------
# Backward-compat wrappers so every existing caller keeps working unchanged.
# ---------------------------------------------------------------------------

class _LandmarkList:
    """Wraps a flat list of landmarks as .landmark[idx] — mimics the legacy API."""
    __slots__ = ('landmark',)
    def __init__(self, landmarks):
        self.landmark = landmarks  # list of NormalizedLandmark / Landmark objects


class _PoseResultsWrapper:
    """Adapts PoseLandmarkerResult to look like mp.solutions.pose.Pose.process()."""
    __slots__ = ('pose_landmarks', 'pose_world_landmarks', 'segmentation_mask')

    def __init__(self, result, seg_mask=None):
        if result and result.pose_landmarks:
            self.pose_landmarks = _LandmarkList(result.pose_landmarks[0])
            self.pose_world_landmarks = (
                _LandmarkList(result.pose_world_landmarks[0])
                if result.pose_world_landmarks else None
            )
        else:
            self.pose_landmarks = None
            self.pose_world_landmarks = None
        self.segmentation_mask = seg_mask


class _FaceLandmarkList:
    """Single face entry for multi_face_landmarks[0].landmark[idx]."""
    __slots__ = ('landmark',)
    def __init__(self, landmarks):
        self.landmark = landmarks


class _FaceMeshResultsWrapper:
    """Adapts FaceLandmarkerResult to look like mp.solutions.face_mesh result."""
    __slots__ = ('multi_face_landmarks',)

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
        PoseLandmarker, PoseLandmarkerOptions,
        FaceLandmarker, FaceLandmarkerOptions,
        RunningMode,
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
    _delegate_name = 'GPU' if _delegate == BaseOptions.Delegate.GPU else 'CPU'
    print(f"[human_pose] Using MediaPipe Tasks API ({_delegate_name} delegate)")

except Exception as _tasks_err:
    print(f"[human_pose] Tasks API unavailable ({_tasks_err}), using legacy API")
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
try:
    mp_pose  # already defined (legacy path)
except NameError:
    mp_pose = mp.solutions.pose


# ---------------------------------------------------------------------------
# Public detector functions
# ---------------------------------------------------------------------------

def get_human_pose(frame):
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


FACE_MESH_INPUT_WIDTH = int(os.getenv('FACE_MESH_INPUT_WIDTH', '256'))

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
EYE_OPEN_RATIO = float(os.getenv('EYE_OPEN_RATIO', '0.2'))
LEFT_EYEBROW_INDICES = [46, 53, 52, 65, 55]
RIGHT_EYEBROW_INDICES = [276, 283, 282, 295, 285]
MOUTH_UPPER_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
MOUTH_LOWER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_CLOSED_GAP_PX = float(os.getenv('MOUTH_CLOSED_GAP_PX', '1.1'))
FACE_PERSPECTIVE_STRENGTH = float(os.getenv('FACE_PERSPECTIVE_STRENGTH', '0.32'))
FACE_PERSPECTIVE_PIVOT_Y = float(os.getenv('FACE_PERSPECTIVE_PIVOT_Y', '0.30'))
MOUTH_Y_OFFSET_PX = int(os.getenv('MOUTH_Y_OFFSET_PX', '1'))
FACE_FEATURE_Y_OFFSET_PX = int(os.getenv('FACE_FEATURE_Y_OFFSET_PX', '-1'))
SILHOUETTE_Y_OFFSET_PX = int(os.getenv('SILHOUETTE_Y_OFFSET_PX', '1'))
ARM_OUTLINE_RADIUS_PX = int(os.getenv('ARM_OUTLINE_RADIUS_PX', '1'))
ARM_MIN_VISIBILITY = float(os.getenv('ARM_MIN_VISIBILITY', '0.45'))
ARM_CORE_RADIUS_PX = int(os.getenv('ARM_CORE_RADIUS_PX', '1'))
FOREARM_MIN_VISIBILITY = float(os.getenv('FOREARM_MIN_VISIBILITY', '0.20'))
ARM_HAND_JOIN_CLEAR_RADIUS_PX = int(os.getenv('ARM_HAND_JOIN_CLEAR_RADIUS_PX', '2'))
ARM_SHOULDER_JOIN_CLEAR_RADIUS_PX = int(os.getenv('ARM_SHOULDER_JOIN_CLEAR_RADIUS_PX', '1'))
ARM_SHOULDER_TORSO_CLEAR_LEN_PX = int(os.getenv('ARM_SHOULDER_TORSO_CLEAR_LEN_PX', '2'))
ARM_SHOULDER_TORSO_BRIDGE_LEN_PX = int(os.getenv('ARM_SHOULDER_TORSO_BRIDGE_LEN_PX', '4'))
ARM_SHOULDER_TORSO_CONNECT_RADIUS_PX = int(os.getenv('ARM_SHOULDER_TORSO_CONNECT_RADIUS_PX', '2'))


def should_draw_face_features(estimated_distance):
    return estimated_distance is not None and estimated_distance < CLOSE_FACE_DISTANCE

def should_draw_detailed_face_features(estimated_distance):
    return estimated_distance is not None and estimated_distance < VERY_CLOSE_FACE_DISTANCE

def _xy_to_panel_xy(x, y, width, height):
    panel_x = int(width - (x * width))
    panel_y = int(y * height)
    return panel_x, panel_y


def _apply_face_perspective_correction(x, y):
    # Camera is mounted above the screen, so pull lower-face landmarks upward.
    strength = max(0.0, FACE_PERSPECTIVE_STRENGTH)
    pivot_y = min(1.0, max(0.0, FACE_PERSPECTIVE_PIVOT_Y))
    y_corrected = y - strength * max(0.0, y - pivot_y)
    return x, min(1.0, max(0.0, y_corrected))


def _apply_perspective_to_mask(mask):
    height, width = mask.shape
    strength = min(0.95, max(0.0, FACE_PERSPECTIVE_STRENGTH))
    pivot_y = min(1.0, max(0.0, FACE_PERSPECTIVE_PIVOT_Y))

    if strength == 0.0:
        return mask

    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    src_y_norm = np.where(
        ys <= pivot_y,
        ys,
        (ys - (strength * pivot_y)) / (1.0 - strength),
    )
    src_y = np.clip(src_y_norm * (height - 1), 0.0, float(height - 1)).astype(np.float32)

    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = np.tile(src_y.reshape(height, 1), (1, width))

    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _shift_mask_y(mask, y_offset_px):
    if y_offset_px == 0:
        return mask

    shifted = np.zeros_like(mask)
    height = mask.shape[0]

    if y_offset_px > 0:
        shifted[y_offset_px:, :] = mask[:height - y_offset_px, :]
    else:
        up = -y_offset_px
        shifted[:height - up, :] = mask[up:, :]

    return shifted


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


def _draw_disk(dots, cx, cy, radius, value):
    h, w = dots.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return
    if radius <= 0:
        dots[cy, cx] = value
        return

    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)

    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    region = dots[y0:y1 + 1, x0:x1 + 1]
    region[mask] = value


def _draw_brush_line(dots, x0, y0, x1, y1, radius, value):
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        _draw_disk(dots, x0, y0, radius, value)
        return

    xs = np.linspace(x0, x1, steps + 1).astype(int)
    ys = np.linspace(y0, y1, steps + 1).astype(int)
    for x, y in zip(xs, ys):
        _draw_disk(dots, x, y, radius, value)


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
    vertical = np.mean([
        abs(landmarks[t].y - landmarks[b].y)
        for t, b in vertical_pairs
    ])
    return (vertical / horizontal) >= EYE_OPEN_RATIO


def _landmarks_to_panel_points(landmarks, indices, width, height):
    points = []
    for idx in indices:
        x, y = _apply_face_perspective_correction(landmarks[idx].x, landmarks[idx].y)
        points.append(_xy_to_panel_xy(x, y, width, height))
    return points


def _draw_points_polyline(dots, points, value=0):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        _draw_line(dots, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], value=value)


def _offset_points(points, dx, dy, width, height):
    shifted = []
    for x, y in points:
        nx = min(width - 1, max(0, x + dx))
        ny = min(height - 1, max(0, y + dy))
        shifted.append((nx, ny))
    return shifted


def _pose_landmark_to_panel_xy(landmark, width, height):
    x, y = _apply_face_perspective_correction(landmark.x, landmark.y)
    px, py = _xy_to_panel_xy(x, y, width, height)
    py = min(height - 1, max(0, py + SILHOUETTE_Y_OFFSET_PX))
    return px, py


def _get_arm_point(landmarks, primary_idx, fallback_idx, min_visibility):
    primary = landmarks[primary_idx]
    if primary.visibility >= min_visibility:
        return primary

    if fallback_idx is None:
        return None

    fallback = landmarks[fallback_idx]
    if fallback.visibility >= min_visibility:
        return fallback

    return None


def _draw_arm_outlines(dots, pose_results, width, height):
    if pose_results is None or pose_results.pose_landmarks is None:
        return

    landmarks = pose_results.pose_landmarks.landmark
    arm_segments = [
        {
            "start": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "end": mp_pose.PoseLandmark.LEFT_ELBOW,
            "end_fallback": None,
            "min_visibility": ARM_MIN_VISIBILITY,
            "torso_anchor": mp_pose.PoseLandmark.LEFT_HIP,
        },
        {
            "start": mp_pose.PoseLandmark.LEFT_ELBOW,
            "end": mp_pose.PoseLandmark.LEFT_WRIST,
            "end_fallback": mp_pose.PoseLandmark.LEFT_INDEX,
            "min_visibility": FOREARM_MIN_VISIBILITY,
            "torso_anchor": None,
        },
        {
            "start": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "end": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "end_fallback": None,
            "min_visibility": ARM_MIN_VISIBILITY,
            "torso_anchor": mp_pose.PoseLandmark.RIGHT_HIP,
        },
        {
            "start": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "end": mp_pose.PoseLandmark.RIGHT_WRIST,
            "end_fallback": mp_pose.PoseLandmark.RIGHT_INDEX,
            "min_visibility": FOREARM_MIN_VISIBILITY,
            "torso_anchor": None,
        },
    ]

    outline_radius = max(0, ARM_OUTLINE_RADIUS_PX)
    core_radius = max(0, ARM_CORE_RADIUS_PX)
    arm_mask = np.zeros_like(dots, dtype=np.uint8)
    hand_join_points = []
    shoulder_join_points = []
    shoulder_torso_clear_segments = []

    for segment in arm_segments:
        min_visibility = segment["min_visibility"]
        start = _get_arm_point(landmarks, segment["start"], None, min_visibility)
        end = _get_arm_point(landmarks, segment["end"], segment["end_fallback"], min_visibility)
        if start is None or end is None:
            continue

        x0, y0 = _pose_landmark_to_panel_xy(start, width, height)
        x1, y1 = _pose_landmark_to_panel_xy(end, width, height)

        _draw_brush_line(arm_mask, x0, y0, x1, y1, core_radius, 1)
        if segment["end_fallback"] is not None:
            hand_join_points.append((x1, y1))
        if segment["start"] in (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER):
            shoulder_join_points.append((x0, y0))

            torso_anchor_idx = segment.get("torso_anchor")
            if torso_anchor_idx is not None:
                torso_anchor = _get_arm_point(landmarks, torso_anchor_idx, None, ARM_MIN_VISIBILITY)
                if torso_anchor is not None:
                    tx, ty = _pose_landmark_to_panel_xy(torso_anchor, width, height)
                    vec_x = tx - x0
                    vec_y = ty - y0
                else:
                    # Fallback for partial-body framing where hip landmarks are unreliable.
                    vec_x = 0
                    vec_y = 1

                dist = np.hypot(vec_x, vec_y)
                if dist > 0:
                    clear_len = max(1, ARM_SHOULDER_TORSO_CLEAR_LEN_PX)
                    clear_scale = min(1.0, clear_len / dist)
                    cx = int(round(x0 + (vec_x * clear_scale)))
                    cy = int(round(y0 + (vec_y * clear_scale)))
                    shoulder_torso_clear_segments.append((x0, y0, cx, cy))

                    bridge_len = max(1, ARM_SHOULDER_TORSO_BRIDGE_LEN_PX)
                    bridge_scale = min(1.0, bridge_len / dist)
                    bx = int(round(x0 + (vec_x * bridge_scale)))
                    by = int(round(y0 + (vec_y * bridge_scale)))
                    _draw_brush_line(arm_mask, x0, y0, bx, by, max(1, core_radius), 1)

    if not np.any(arm_mask):
        return

    if outline_radius > 0:
        kernel_size = (outline_radius * 2) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated_mask = cv2.dilate(arm_mask, kernel, iterations=1)
        outline_mask = (dilated_mask == 1) & (arm_mask == 0)

        join_clear_mask = np.zeros_like(dots, dtype=np.uint8)

        if hand_join_points:
            clear_radius = max(0, ARM_HAND_JOIN_CLEAR_RADIUS_PX)
            for x, y in hand_join_points:
                _draw_disk(join_clear_mask, x, y, clear_radius, 1)

        if shoulder_join_points:
            clear_radius = max(0, ARM_SHOULDER_JOIN_CLEAR_RADIUS_PX)
            for x, y in shoulder_join_points:
                _draw_disk(join_clear_mask, x, y, clear_radius, 1)

            for sx, sy, ex, ey in shoulder_torso_clear_segments:
                _draw_brush_line(join_clear_mask, sx, sy, ex, ey, clear_radius, 1)

        outline_mask &= (join_clear_mask == 0)

        dots[outline_mask] = 0

    dots[arm_mask == 1] = 1


def _draw_shoulder_torso_connectors(dots, pose_results, width, height):
    if pose_results is None or pose_results.pose_landmarks is None:
        return

    landmarks = pose_results.pose_landmarks.landmark
    connectors = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    ]

    radius = max(1, ARM_SHOULDER_TORSO_CONNECT_RADIUS_PX)
    bridge_len = max(1, ARM_SHOULDER_TORSO_BRIDGE_LEN_PX)

    for shoulder_idx, hip_idx in connectors:
        shoulder = _get_arm_point(landmarks, shoulder_idx, None, ARM_MIN_VISIBILITY)
        if shoulder is None:
            continue

        sx, sy = _pose_landmark_to_panel_xy(shoulder, width, height)
        hip = _get_arm_point(landmarks, hip_idx, None, ARM_MIN_VISIBILITY)

        if hip is not None:
            hx, hy = _pose_landmark_to_panel_xy(hip, width, height)
            vec_x = hx - sx
            vec_y = hy - sy
        else:
            vec_x = 0
            vec_y = 1

        dist = np.hypot(vec_x, vec_y)
        if dist == 0:
            continue

        scale = min(1.0, bridge_len / dist)
        ex = int(round(sx + (vec_x * scale)))
        ey = int(round(sy + (vec_y * scale)))
        _draw_brush_line(dots, sx, sy, ex, ey, radius, 1)


def _draw_face_features(dots, face_mesh_results, width, height, draw_details=False):
    if face_mesh_results is None or not face_mesh_results.multi_face_landmarks:
        return dots

    landmarks = face_mesh_results.multi_face_landmarks[0].landmark

    left_eye_points = _landmarks_to_panel_points(landmarks, LEFT_EYE_INDICES, width, height)
    right_eye_points = _landmarks_to_panel_points(landmarks, RIGHT_EYE_INDICES, width, height)
    left_eye = (int(np.mean([p[0] for p in left_eye_points])), int(np.mean([p[1] for p in left_eye_points])))
    right_eye = (int(np.mean([p[0] for p in right_eye_points])), int(np.mean([p[1] for p in right_eye_points])))
    left_eye = (left_eye[0], left_eye[1] + FACE_FEATURE_Y_OFFSET_PX)
    right_eye = (right_eye[0], right_eye[1] + FACE_FEATURE_Y_OFFSET_PX)

    if 0 <= left_eye[0] < width and 0 <= left_eye[1] < height:
        if _is_eye_open(landmarks, LEFT_EYE_INDICES, LEFT_EYE_VERTICAL_INDICES):
            dots[left_eye[1], left_eye[0]] = 0
    if 0 <= right_eye[0] < width and 0 <= right_eye[1] < height:
        if _is_eye_open(landmarks, RIGHT_EYE_INDICES, RIGHT_EYE_VERTICAL_INDICES):
            dots[right_eye[1], right_eye[0]] = 0

    upper_points = _landmarks_to_panel_points(landmarks, MOUTH_UPPER_INDICES, width, height)
    lower_points = _landmarks_to_panel_points(landmarks, MOUTH_LOWER_INDICES, width, height)
    mouth_total_offset = MOUTH_Y_OFFSET_PX + FACE_FEATURE_Y_OFFSET_PX
    upper_points = _offset_points(upper_points, 0, mouth_total_offset, width, height)
    lower_points = _offset_points(lower_points, 0, mouth_total_offset, width, height)

    mouth_gap = np.mean([abs(upper_points[i][1] - lower_points[i][1]) for i in range(len(upper_points))])
    if mouth_gap <= MOUTH_CLOSED_GAP_PX:
        center_points = [
            ((upper_points[i][0] + lower_points[i][0]) // 2, (upper_points[i][1] + lower_points[i][1]) // 2)
            for i in range(len(upper_points))
        ]
        _draw_points_polyline(dots, center_points, value=0)
    else:
        _draw_points_polyline(dots, upper_points, value=0)
        _draw_points_polyline(dots, lower_points, value=0)

    if draw_details:
        left_brow_points = _landmarks_to_panel_points(landmarks, LEFT_EYEBROW_INDICES, width, height)
        right_brow_points = _landmarks_to_panel_points(landmarks, RIGHT_EYEBROW_INDICES, width, height)
        left_brow_points = _offset_points(left_brow_points, 0, FACE_FEATURE_Y_OFFSET_PX, width, height)
        right_brow_points = _offset_points(right_brow_points, 0, FACE_FEATURE_Y_OFFSET_PX, width, height)

        # Shift each eyebrow up so its lowest point is at least 1 pixel above the eye.
        # "At least 1 pixel away" = at least one blank row between them, so gap >= 2.
        for brow_points, eye in ((left_brow_points, left_eye), (right_brow_points, right_eye)):
            max_brow_y = max(p[1] for p in brow_points)
            gap = eye[1] - max_brow_y
            if gap < 2:
                dy = 2 - gap
                brow_points[:] = _offset_points(brow_points, 0, -dy, width, height)

        _draw_points_polyline(dots, left_brow_points, value=0)
        _draw_points_polyline(dots, right_brow_points, value=0)

    return dots


def display_human_pose(pose_results, width, height, estimated_distance=None, face_mesh_results=None):
    dots = np.zeros((width, height), dtype=np.uint8)
    if pose_results.segmentation_mask is not None:
        dots = cv2.resize(pose_results.segmentation_mask, (width, height), cv2.INTER_AREA)
        dots = (dots > 0.5).astype(np.uint8)
        dots = np.fliplr(dots)
        dots = _apply_perspective_to_mask(dots)
        dots = _shift_mask_y(dots, SILHOUETTE_Y_OFFSET_PX)
        _draw_arm_outlines(dots, pose_results, width, height)
        _draw_shoulder_torso_connectors(dots, pose_results, width, height)

        if should_draw_face_features(estimated_distance):
            draw_details = should_draw_detailed_face_features(estimated_distance)
            dots = _draw_face_features(dots, face_mesh_results, width, height, draw_details=draw_details)
    
    return dots

def eyes_visible_and_facing_camera(pose_results):
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
    all_eye_landmarks_visible = all(landmark.visibility > confidence_threshold for landmark in eye_landmarks)
    if not all_eye_landmarks_visible:
        return False, "Eye landmarks not clearly visible", None

    # Use the 2D normalised pose_landmarks for the angle calculation.
    # pose_landmarks.z has the same scale as x and measures depth relative to
    # the hip midpoint (smaller = closer to camera).  Unlike pose_world_landmarks.z
    # it is not affected by the gravity-aligned body frame used by GHUM/Tasks API,
    # so it correctly reads ~0 depth-difference between the eyes when the person
    # faces the camera directly, regardless of camera tilt.
    left_eye  = landmarks[2]   # left eye centre
    right_eye = landmarks[5]   # right eye centre

    dx = right_eye.x - left_eye.x   # lateral (negative when facing camera)
    dz = right_eye.z - left_eye.z   # depth difference

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

def estimate_distance(pose_results):
    FOCAL_SCALE = float(os.getenv('FOCAL_SCALE', '1.0'))

    # Estimate distance based on the size of the person in the frame
    if pose_results is None or pose_results.pose_landmarks is None:
        return None, []

    KNOWN_DISTANCES = [
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
        #z_distance = abs(point0[2] - point1[2])
        #if z_distance > 0.5 * distance:
        #    continue

        distance_estimates.append(((known_distance["value"] * FOCAL_SCALE) / distance, distance, known_distance["name"]))


    if len(distance_estimates) == 0:
        return None, []

    return sum(value for value, _, _ in distance_estimates) / len(distance_estimates), distance_estimates

def get_right_index_finger_position(pose_results=None):
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


def is_right_index_in_top_right_corner(pose_results=None):
    x, y = get_right_index_finger_position(pose_results)

    # Check if the index finger is in the top right corner (e.g., top 20% and right 20%)
    if x is not None and y is not None and x < 0.2 and y < 0.2:
        return True, x, y

    return False, x, y


def is_arms_crossed(pose_results=None):
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

def is_left_hand_raised(pose_results=None):
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


def draw_right_index_pointer(frame, pose_results=None, size=1):
    finger_x, finger_y = get_right_index_finger_position(pose_results)
    height, width = frame.shape

    if finger_x is not None and finger_y is not None:
        x = int(width - (finger_x * width))
        y = int(finger_y * height)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), size//2, 0, -1)
            cv2.circle(frame, (x, y), size//2, 1, 1)

    return frame
