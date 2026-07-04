"""Tests for the live landmark-driven caricature mode."""

import dataclasses
import math

import numpy as np
import pytest

import app.modes.caricature as caricature
from app.modes.contracts import RenderContext

WIDTH = 28
HEIGHT = 28


class _Lm:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 0.0


class _FaceList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _Results:
    def __init__(self, faces):
        self.multi_face_landmarks = faces


class _FakeModeManager:
    def __init__(self):
        self.mode_start_time = 100.0


def _face_offsets(
    *,
    eye_open_left=0.105,
    eye_open_right=0.105,
    brow_left=0.36,
    brow_right=0.36,
    mouth_width=0.80,
    mouth_open=0.03,
    smile=0.0,
):
    """Landmark offsets in IOD units (unmirrored, x right / y down) around the eye midpoint.

    The neutral defaults are chosen to match the EXAGGERATION neutrals so a
    default face measures (approximately) neutral on every metric.
    """
    half_mouth = mouth_width / 2.0
    mouth_y = 1.1
    lip = 0.14  # outer-lip midpoints sit this far beyond the inner-lip gap
    return {
        # Face oval: jaw width 1.30, aspect ~1.31.
        10: (0.00, -0.95),
        297: (0.40, -0.90),
        284: (0.75, -0.70),
        389: (1.00, -0.25),
        454: (1.03, 0.20),
        361: (0.95, 0.75),
        397: (0.65, 1.20),
        379: (0.38, 1.55),
        152: (0.00, 1.75),
        150: (-0.38, 1.55),
        172: (-0.65, 1.20),
        132: (-0.95, 0.75),
        234: (-1.03, 0.20),
        162: (-1.00, -0.25),
        54: (-0.75, -0.70),
        67: (-0.40, -0.90),
        # Mesh-"left" eye (33 family) is on the image's left, the panel's right.
        33: (-0.68, 0.0),
        133: (-0.32, 0.0),
        159: (-0.5, -eye_open_left / 2),
        145: (-0.5, eye_open_left / 2),
        468: (-0.5, 0.0),
        362: (0.32, 0.0),
        263: (0.68, 0.0),
        386: (0.5, -eye_open_right / 2),
        374: (0.5, eye_open_right / 2),
        473: (0.5, 0.0),
        # Brows.
        46: (-0.75, -brow_left),
        52: (-0.5, -brow_left),
        55: (-0.25, -brow_left),
        276: (0.25, -brow_right),
        282: (0.5, -brow_right),
        285: (0.75, -brow_right),
        # Nose.
        168: (0.0, 0.10),
        4: (0.0, 0.55),
        98: (-0.15, 0.65),
        327: (0.15, 0.65),
        # Mouth (inner lips 13/14 carry the opening; 0/17 are the outer mids).
        13: (0.0, mouth_y - mouth_open / 2),
        14: (0.0, mouth_y + mouth_open / 2),
        0: (0.0, mouth_y - mouth_open / 2 - lip),
        17: (0.0, mouth_y + mouth_open / 2 + lip),
        61: (-half_mouth, mouth_y - smile),
        291: (half_mouth, mouth_y - smile),
        40: (-half_mouth / 2, mouth_y - mouth_open / 2 - lip * 0.8 - smile / 2),
        270: (half_mouth / 2, mouth_y - mouth_open / 2 - lip * 0.8 - smile / 2),
        91: (-half_mouth / 2, mouth_y + mouth_open / 2 + lip * 0.8 - smile / 2),
        321: (half_mouth / 2, mouth_y + mouth_open / 2 + lip * 0.8 - smile / 2),
    }


def _make_face(*, center=(0.5, 0.45), iod=0.15, roll=0.0, n_landmarks=478, **offset_kwargs):
    offsets = _face_offsets(**offset_kwargs)
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    landmarks = [_Lm(0.5, 0.5) for _ in range(n_landmarks)]
    for index, (ox, oy) in offsets.items():
        if index >= n_landmarks:
            continue
        rx = ox * cos_r - oy * sin_r
        ry = ox * sin_r + oy * cos_r
        landmarks[index] = _Lm(center[0] + iod * rx, center[1] + iod * ry)
    return _Results([_FaceList(landmarks)])


def _ctx(results, mode_time=None, exit_progress=None):
    # Default to a time past the entry zoom so geometry tests see the
    # canonical (centered, full-size) projection.
    if mode_time is None:
        mode_time = caricature.ENTRY_ZOOM_SECONDS + 1.0
    return RenderContext(
        frame=np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
        pose_results=None,
        face_mesh_results=results,
        estimated_distance=None,
        mode_time=mode_time,
        panel_width=WIDTH,
        panel_height=HEIGHT,
        caricature_exit_progress=exit_progress,
    )


def _make_mode():
    return caricature.Caricature(WIDTH, HEIGHT, _FakeModeManager())


def _pipeline_panel(face):
    """Run the pure geometry pipeline (no drawing) and return panel-space points."""
    points = caricature._latest_points(face)
    assert points is not None
    eye_mid, iod, roll = caricature._face_basis(points)
    local = caricature._to_face_local(points, eye_mid, iod, roll)
    measured = caricature._measure_metrics(local, roll)
    exag = caricature._exaggerate(measured)
    warped = caricature._apply_exaggeration(local, measured, exag)
    return caricature._project_to_panel(warped, exag.roll, WIDTH, HEIGHT)


def test_frame_contract_with_face():
    frame = _make_mode().get_frame(_ctx(_make_face()))

    assert frame.shape == (HEIGHT, WIDTH)
    assert frame.dtype == np.uint8
    assert set(np.unique(frame)) <= {0, 1}
    assert frame.sum() > 0


def test_no_results_renders_idle_face():
    frame = _make_mode().get_frame(_ctx(None))

    assert frame.sum() > 0
    assert set(np.unique(frame)) <= {0, 1}


def test_empty_face_list_is_treated_as_no_face():
    mode = _make_mode()

    assert mode.get_frame(_ctx(_Results(None))).sum() > 0
    assert mode.get_frame(_ctx(_Results([]))).sum() > 0


def test_neutral_face_regions():
    frame = _make_mode().get_frame(_ctx(_make_face()))

    assert frame[0:5, :].sum() == 0  # headroom above the oval stays empty (hair space)
    assert frame[8:11, :].sum() > 0  # brow band
    assert frame[10:14, :].sum() > 0  # eye band
    assert frame[17:24, :].sum() > 0  # mouth band
    cols = np.where(frame.any(axis=0))[0]
    assert cols.min() <= 7  # face oval reaches the panel's left edge region
    assert cols.max() >= 20  # ... and the right edge region


def test_scale_invariance_across_distance():
    near = _make_mode().get_frame(_ctx(_make_face(iod=0.20)))
    far = _make_mode().get_frame(_ctx(_make_face(iod=0.10)))

    rows_near = np.where(near.any(axis=1))[0]
    rows_far = np.where(far.any(axis=1))[0]
    near_height = rows_near.max() - rows_near.min()
    far_height = rows_far.max() - rows_far.min()
    assert abs(int(near_height) - int(far_height)) <= 2


def test_exaggeration_spec_amplifies_and_clamps():
    spec = caricature.ExaggerationSpec(neutral=0.1, gain=2.0, lo=0.0, hi=0.3)

    assert spec.apply(0.1) == pytest.approx(0.1)
    assert spec.apply(0.15) == pytest.approx(0.2)
    assert spec.apply(0.05) == pytest.approx(0.0)
    assert spec.apply(1.0) == pytest.approx(0.3)
    assert spec.apply(-1.0) == pytest.approx(0.0)


def test_roll_exaggeration_is_clamped():
    points = caricature._latest_points(_make_face())
    eye_mid, iod, roll = caricature._face_basis(points)
    local = caricature._to_face_local(points, eye_mid, iod, roll)
    measured = caricature._measure_metrics(local, roll)

    exag = caricature._exaggerate(dataclasses.replace(measured, roll=1.0))

    assert exag.roll == pytest.approx(caricature.ROLL_MAX_RAD)


def test_roll_is_detected_from_eye_line():
    points = caricature._latest_points(_make_face(roll=0.2))

    _, _, roll = caricature._face_basis(points)

    assert abs(abs(roll) - 0.2) < 0.02


def test_mouth_open_increases_mouth_extent():
    closed = _make_mode().get_frame(_ctx(_make_face(mouth_open=0.0)))
    opened = _make_mode().get_frame(_ctx(_make_face(mouth_open=0.2)))

    def extent(frame):
        # Rows 18..22 / columns 11..16 cover the mouth but exclude the nose
        # base (row 17), the chin arc (row 23+), and the face-oval sides.
        band = frame[18:23, 11:17]
        rows = np.where(band.any(axis=1))[0]
        return int(rows.max() - rows.min()) if rows.size else 0

    assert extent(opened) >= extent(closed) + 3


def test_brow_raise_moves_brows_up():
    neutral = _pipeline_panel(_make_face())
    raised = _pipeline_panel(_make_face(brow_left=0.5, brow_right=0.5))

    row = caricature._INDEX_TO_ROW[caricature.LEFT_BROW_MID]
    assert raised[row, 1] < neutral[row, 1] - 1.0


def test_smile_lifts_mouth_corners():
    smiling = _pipeline_panel(_make_face(smile=0.10))
    frowning = _pipeline_panel(_make_face(smile=-0.10))

    corner = caricature._INDEX_TO_ROW[caricature.MOUTH_CORNER_LEFT]
    assert smiling[corner, 1] < frowning[corner, 1] - 2.0


def test_subject_left_mouth_corner_renders_on_panel_left():
    panel = _pipeline_panel(_make_face())

    subject_left = caricature._INDEX_TO_ROW[caricature.MOUTH_CORNER_RIGHT]  # mesh 291
    subject_right = caricature._INDEX_TO_ROW[caricature.MOUTH_CORNER_LEFT]  # mesh 61
    assert panel[subject_left, 0] < panel[subject_right, 0]


def test_wink_closes_one_eye_on_the_mirrored_side():
    # The mesh-"right" eye (362/263 family) renders on the panel's left; when it
    # closes, that side of the eye band collapses to a single line.
    frame = _make_mode().get_frame(_ctx(_make_face(eye_open_right=0.01)))

    eye_band = frame[9:14, :]
    left_count = int(eye_band[:, 8:13].sum())
    right_count = int(eye_band[:, 15:20].sum())
    assert right_count > left_count


def test_landmark_smoothing_lags_and_converges():
    mode = _make_mode()
    mode.get_frame(_ctx(_make_face(center=(0.4, 0.45))))
    before = mode._smoothed.copy()

    mode.get_frame(_ctx(_make_face(center=(0.6, 0.45))))
    moved = np.abs(mode._smoothed - before)[:, 0].max()
    assert moved == pytest.approx(0.2 * caricature.LANDMARK_EMA_ALPHA, abs=0.01)

    for _ in range(30):
        mode.get_frame(_ctx(_make_face(center=(0.6, 0.45))))
    fresh = _make_mode()
    fresh.get_frame(_ctx(_make_face(center=(0.6, 0.45))))
    assert np.abs(mode._smoothed - fresh._smoothed).max() < 0.01


def test_face_hold_then_idle(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(caricature.time, "time", lambda: now["value"])
    mode = _make_mode()
    face_frame = mode.get_frame(_ctx(_make_face()))

    now["value"] = 1000.5
    held = mode.get_frame(_ctx(None))
    assert np.array_equal(held, face_frame)

    now["value"] = 1000.0 + caricature.FACE_HOLD_SEC + 1.0
    idle = mode.get_frame(_ctx(None))
    assert idle.sum() > 0
    assert not np.array_equal(idle, face_frame)


def test_idle_blink_toggles(monkeypatch):
    now = {"value": 0.1}  # inside the blink window
    monkeypatch.setattr(caricature.time, "time", lambda: now["value"])
    mode = _make_mode()
    blinking = mode.get_frame(_ctx(None))

    now["value"] = 1.2  # outside the blink window
    eyes_open = mode.get_frame(_ctx(None))

    assert not np.array_equal(blinking, eyes_open)


def test_mode_reentry_resets_smoothing():
    manager = _FakeModeManager()
    mode = caricature.Caricature(WIDTH, HEIGHT, manager)
    mode.get_frame(_ctx(_make_face(center=(0.4, 0.45))))

    manager.mode_start_time += 10.0
    mode.get_frame(_ctx(_make_face(center=(0.6, 0.45))))

    fresh = _make_mode()
    fresh.get_frame(_ctx(_make_face(center=(0.6, 0.45))))
    assert np.array_equal(mode._smoothed, fresh._smoothed)


def test_absurd_landmarks_stay_in_bounds():
    face = _make_face(
        center=(0.9, 0.9), iod=0.5, mouth_open=3.0, smile=2.0, brow_left=2.0, brow_right=-1.0
    )
    frame = _make_mode().get_frame(_ctx(face))

    assert frame.shape == (HEIGHT, WIDTH)
    assert set(np.unique(frame)) <= {0, 1}


def test_tiny_iod_treated_as_no_face(monkeypatch):
    monkeypatch.setattr(caricature.time, "time", lambda: 1.0)

    frame = _make_mode().get_frame(_ctx(_make_face(iod=0.01)))
    idle = _make_mode().get_frame(_ctx(None))

    assert np.array_equal(frame, idle)


def test_mesh_without_iris_landmarks_still_renders():
    frame = _make_mode().get_frame(_ctx(_make_face(n_landmarks=468)))

    assert frame.sum() > 0


def _make_hair_mask(
    *,
    height_iod=0.5,
    half_width_iod=1.0,
    x_offset_iod=0.0,
    side_drop_iod=None,
    center=(0.5, 0.45),
    iod=0.15,
    size=64,
):
    """Paint a rectangular hair cap above the synthetic face's head.

    The mask is in raw (unmirrored) raster coordinates, sharing its geometry
    with ``_make_face``: the cap top sits ``0.95 + height_iod`` IOD above the
    eye line, matching the synthetic forehead-top landmark at -0.95. With
    ``side_drop_iod`` two hair bands additionally hang beside the face, down
    to that depth below the eye line (long hair / side curtains).
    """
    mask = np.zeros((size, size), dtype=bool)
    center_x = center[0] + x_offset_iod * iod
    x0 = int((center_x - half_width_iod * iod) * (size - 1))
    x1 = int((center_x + half_width_iod * iod) * (size - 1))
    y0 = int((center[1] - (0.95 + height_iod) * iod) * (size - 1))
    y1 = int((center[1] - 0.5 * iod) * (size - 1))
    mask[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
    if side_drop_iod is not None:
        drop_y = int((center[1] + side_drop_iod * iod) * (size - 1))
        for sign in (-1, 1):
            band_x0 = int((center_x + (sign * 1.2 - 0.3) * iod) * (size - 1))
            band_x1 = int((center_x + (sign * 1.2 + 0.3) * iod) * (size - 1))
            mask[max(y0, 0) : max(drop_y, 0), max(band_x0, 0) : max(band_x1, 0)] = True
    return mask


def _make_hair_mode(mask):
    return caricature.Caricature(
        WIDTH, HEIGHT, _FakeModeManager(), hair_mask_provider=lambda frame: mask
    )


def test_hair_renders_above_face_oval():
    mode = _make_hair_mode(_make_hair_mask(height_iod=0.6))

    frame = mode.get_frame(_ctx(_make_face()))

    assert frame[0:6, :].sum() > 0  # hair mass in the headroom rows
    assert frame[10:14, :].sum() > 0  # face still drawn beneath it


def test_bigger_hair_mask_lights_more_pixels():
    small = _make_hair_mode(_make_hair_mask(height_iod=0.2)).get_frame(_ctx(_make_face()))
    big = _make_hair_mode(_make_hair_mask(height_iod=0.8)).get_frame(_ctx(_make_face()))

    assert big[0:7, :].sum() > small[0:7, :].sum()


def test_bald_mask_matches_render_without_provider():
    bald = _make_hair_mode(np.zeros((64, 64), dtype=bool)).get_frame(_ctx(_make_face()))
    plain = _make_mode().get_frame(_ctx(_make_face()))

    assert np.array_equal(bald, plain)


def test_provider_returning_none_matches_render_without_provider():
    no_mask = _make_hair_mode(None).get_frame(_ctx(_make_face()))
    plain = _make_mode().get_frame(_ctx(_make_face()))

    assert np.array_equal(no_mask, plain)


def test_asymmetric_hair_renders_mirrored():
    # Hair on the subject's left (unmirrored raster x > center) must land on
    # the panel's left half, like a mirror.
    mask = _make_hair_mask(height_iod=0.6, half_width_iod=0.5, x_offset_iod=0.7)

    frame = _make_hair_mode(mask).get_frame(_ctx(_make_face()))

    hair_rows = frame[0:6, :]
    assert hair_rows[:, 0:14].sum() > hair_rows[:, 14:28].sum()


def test_ears_render_beside_the_oval():
    frame = _make_mode().get_frame(_ctx(_make_face()))

    # The ear brackets bulge outward past the oval sides (which sit at cols 7/20).
    assert frame[11:17, 0:7].sum() > 0
    assert frame[11:17, 21:28].sum() > 0


def test_ear_hidden_by_hair_detection():
    points = caricature._latest_points(_make_face())
    eye_mid, iod, roll = caricature._face_basis(points)
    local = caricature._to_face_local(points, eye_mid, iod, roll)

    # A side curtain hanging below the eye line on one side hides only that ear.
    side_col = int(np.argmin(np.abs(caricature.HAIR_COL_XS - 1.3)))
    profile = _empty_hair_profile()
    profile[:, side_col] = (1.2, -1.4)
    strokes = caricature._hair_strokes(profile, local)
    assert caricature._ear_hidden_by_hair(strokes, 1.0) is True
    assert caricature._ear_hidden_by_hair(strokes, -1.0) is False

    # A short crown/sideburn stroke ending above the ear hides nothing.
    guard_col = int(np.argmin(np.abs(caricature.HAIR_COL_XS - 0.82)))
    profile = _empty_hair_profile()
    profile[:, guard_col] = (1.5, 0.3)
    strokes = caricature._hair_strokes(profile, local)
    assert caricature._ear_hidden_by_hair(strokes, 1.0) is False


def test_side_hair_renders_beside_the_face():
    crown_only = _make_hair_mode(_make_hair_mask(height_iod=0.5)).get_frame(_ctx(_make_face()))
    with_sides = _make_hair_mode(_make_hair_mask(height_iod=0.5, side_drop_iod=1.5)).get_frame(
        _ctx(_make_face())
    )

    # Rows below the ear brackets (which end near row 16), beside the face.
    left = np.s_[17:23, 0:5]
    right = np.s_[17:23, 23:28]
    assert crown_only[left].sum() == 0
    assert crown_only[right].sum() == 0
    assert with_sides[left].sum() > 0
    assert with_sides[right].sum() > 0


def test_hair_profile_resets_on_mode_reentry():
    manager = _FakeModeManager()
    mask = _make_hair_mask()
    mode = caricature.Caricature(WIDTH, HEIGHT, manager, hair_mask_provider=lambda frame: mask)
    mode.get_frame(_ctx(_make_face()))
    assert mode._hair_profile is not None

    manager.mode_start_time += 5.0
    mode.get_frame(_ctx(None))

    assert mode._hair_profile is None


def test_hair_profile_from_mask_measures_cap_height():
    mask = _make_hair_mask(height_iod=0.5)
    eye_mid = np.array([0.5, 0.45])  # centered face: mirrored == unmirrored

    profile = caricature._hair_profile_from_mask(mask, eye_mid, 0.15, 0.0)

    center_col = len(caricature.HAIR_COL_XS) // 2
    assert profile[0, center_col] == pytest.approx(1.45, abs=0.1)  # cap top
    # Coarse tolerance: one 64px-mask pixel spans ~0.1 IOD at iod=0.15.
    assert profile[1, center_col] == pytest.approx(0.5, abs=0.2)  # cap bottom
    assert np.isnan(profile[0, 0])  # column at |x|=1.8 lies beyond the cap


def test_hair_profile_ignores_sparse_noise():
    mask = np.zeros((64, 64), dtype=bool)
    for x in (5, 20, 35, 50, 60):
        mask[10, x] = True

    profile = caricature._hair_profile_from_mask(mask, np.array([0.5, 0.45]), 0.15, 0.0)

    assert np.all(np.isnan(profile))


def test_smooth_hair_profile_handles_absent_columns():
    previous = np.array([[1.0, np.nan, 2.0, 0.02], [0.5, np.nan, -1.0, 0.01]])
    measured = np.array([[2.0, 3.0, np.nan, np.nan], [0.0, 1.0, np.nan, np.nan]])

    out = caricature._smooth_hair_profile(previous, measured)

    alpha = caricature.HAIR_EMA_ALPHA
    assert out[0, 0] == pytest.approx((1 - alpha) * 1.0 + alpha * 2.0)
    assert out[1, 0] == pytest.approx((1 - alpha) * 0.5)  # bottoms near 0 stay valid
    assert out[0, 1] == pytest.approx(3.0)  # newly appearing column adopted directly
    assert out[0, 2] == pytest.approx((1 - alpha) * 2.0)  # vanished column decays
    assert out[1, 2] == pytest.approx((1 - alpha) * -1.0)
    assert np.all(np.isnan(out[:, 3]))  # top decayed below the absence threshold

    first = caricature._smooth_hair_profile(None, measured)
    assert np.array_equal(first, measured, equal_nan=True)


def test_hair_exaggeration_clamps_volume():
    spec = caricature.EXAGGERATION["hair_volume"]

    high = caricature._exaggerate_hair(np.full(19, 2.0))
    assert np.allclose(high, spec.hi)

    low = caricature._exaggerate_hair(np.full(19, 0.05))
    assert np.allclose(low, 0.0)


def _empty_hair_profile():
    return np.full((2, len(caricature.HAIR_COL_XS)), np.nan)


def test_hair_strokes_respect_brow_guard_and_skip_short():
    points = caricature._latest_points(_make_face())
    eye_mid, iod, roll = caricature._face_basis(points)
    local = caricature._to_face_local(points, eye_mid, iod, roll)
    center = len(caricature.HAIR_COL_XS) // 2

    # A tall column over the face where the forehead arc has dropped below the
    # brow-guard floor: the fill bottom must be raised to the guard.
    guard_col = int(np.argmin(np.abs(caricature.HAIR_COL_XS - 0.82)))  # arc ~ 0.58
    profile = _empty_hair_profile()
    profile[:, guard_col] = (1.5, 0.3)
    strokes = caricature._hair_strokes(profile, local)
    assert len(strokes) == 1
    _, _, bottom = strokes[0]
    assert bottom[1] == pytest.approx(-caricature.HAIR_BROW_GUARD_MIN_HEIGHT_IOD)

    # A column barely above the arc collapses under exaggeration and is skipped.
    profile = _empty_hair_profile()
    profile[:, center] = (1.0, 0.5)  # central column, arc ~ 0.95
    assert caricature._hair_strokes(profile, local) == []


def test_hair_strokes_hang_to_measured_bottom_beside_the_face():
    points = caricature._latest_points(_make_face())
    eye_mid, iod, roll = caricature._face_basis(points)
    local = caricature._to_face_local(points, eye_mid, iod, roll)

    side_col = int(np.argmin(np.abs(caricature.HAIR_COL_XS - 1.3)))  # beyond the guard span
    profile = _empty_hair_profile()
    profile[:, side_col] = (1.2, -1.4)  # hair from above the head to below the eye line
    strokes = caricature._hair_strokes(profile, local)

    assert len(strokes) == 1
    _, top, bottom = strokes[0]
    assert bottom[1] == pytest.approx(1.4)  # face-local y is down: hangs below the eye line
    assert top[1] < 0


def test_entry_zoom_starts_at_real_face_position_and_size():
    # Face offset toward the camera's left (mirrored to panel right), small IOD.
    face = _make_face(center=(0.3, 0.3), iod=0.08)
    small = _make_mode().get_frame(_ctx(face, mode_time=0.0))
    full = _make_mode().get_frame(_ctx(face))

    ys_s, xs_s = np.nonzero(small)
    ys_f, xs_f = np.nonzero(full)
    assert len(xs_s) > 0
    # At t=0 the caricature sits where the real (mirrored) face is on the panel.
    assert abs(xs_s.mean() - (1.0 - 0.3) * WIDTH) < 4.0
    assert abs(ys_s.mean() - 0.3 * HEIGHT) < 4.0
    # It is markedly smaller than the settled projection.
    small_extent = (xs_s.max() - xs_s.min()) + (ys_s.max() - ys_s.min())
    full_extent = (xs_f.max() - xs_f.min()) + (ys_f.max() - ys_f.min())
    assert small_extent < full_extent * 0.7
    # After the zoom window the projection is canonical (horizontally centered).
    assert abs(xs_f.mean() - (WIDTH - 1) / 2.0) < 3.0


def test_entry_zoom_skipped_when_face_appears_late():
    face = _make_face(center=(0.3, 0.3), iod=0.08)
    frame = _make_mode().get_frame(_ctx(face, mode_time=caricature.ENTRY_ZOOM_SECONDS + 0.5))

    ys, xs = np.nonzero(frame)
    # No zoom: rendered straight at the canonical centered projection.
    assert abs(xs.mean() - (WIDTH - 1) / 2.0) < 3.0


def test_entry_zoom_anchor_is_captured_once():
    mode = _make_mode()
    mode.get_frame(_ctx(_make_face(center=(0.3, 0.3), iod=0.08), mode_time=0.0))
    start_anchor, start_iod = mode._entry_start

    # The viewer moves mid-zoom; the zoom keeps interpolating from the
    # originally captured anchor rather than chasing the new position.
    mode.get_frame(_ctx(_make_face(center=(0.7, 0.6), iod=0.15), mode_time=0.5))
    anchor_after, iod_after = mode._entry_start
    assert np.array_equal(anchor_after, start_anchor)
    assert iod_after == start_iod


def test_exit_shrink_moves_face_onto_real_position():
    face = _make_face(center=(0.3, 0.3), iod=0.08)
    mode = _make_mode()
    full = mode.get_frame(_ctx(face))
    shrunk = mode.get_frame(_ctx(face, exit_progress=1.0))

    ys_f, xs_f = np.nonzero(full)
    ys_s, xs_s = np.nonzero(shrunk)
    # At full exit progress the face sits at the real (mirrored) position...
    assert abs(xs_s.mean() - (1.0 - 0.3) * WIDTH) < 4.0
    assert abs(ys_s.mean() - 0.3 * HEIGHT) < 4.0
    # ...and is markedly smaller than the canonical projection.
    shrunk_extent = (xs_s.max() - xs_s.min()) + (ys_s.max() - ys_s.min())
    full_extent = (xs_f.max() - xs_f.min()) + (ys_f.max() - ys_f.min())
    assert shrunk_extent < full_extent * 0.7


def test_exit_progress_zero_keeps_canonical_projection():
    face = _make_face(center=(0.3, 0.3), iod=0.08)
    canonical = _make_mode().get_frame(_ctx(face))
    at_zero = _make_mode().get_frame(_ctx(face, exit_progress=0.0))

    assert np.array_equal(at_zero, canonical)
