"""Tests for the constructed-figure renderer (pose-mode character)."""

import types

import cv2
import numpy as np

import app.services.figure as figure

SIZE = 28


def _landmarks(positions: dict[int, tuple[float, float]], visibility: float = 1.0) -> list:
    """Build 33 fake landmarks from {index: (x, y)}; unlisted indices are invisible."""
    landmarks = [types.SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for i, (x, y) in positions.items():
        landmarks[i] = types.SimpleNamespace(x=x, y=y, visibility=visibility)
    return landmarks


def _identity(landmark) -> tuple[float, float]:
    """Mapper for landmarks whose x/y are already panel pixel coordinates."""
    return landmark.x, landmark.y


def _full_body_panel() -> dict[int, tuple[float, float]]:
    """An upright full-body figure in panel pixel coordinates (28x28)."""
    return {
        figure.NOSE: (14.0, 3.0),
        figure.L_EAR: (16.0, 3.0),
        figure.R_EAR: (12.0, 3.0),
        figure.L_SHOULDER: (18.0, 9.0),
        figure.R_SHOULDER: (10.0, 9.0),
        figure.L_ELBOW: (20.0, 13.0),
        figure.R_ELBOW: (8.0, 13.0),
        figure.L_WRIST: (21.0, 17.0),
        figure.R_WRIST: (7.0, 17.0),
        figure.L_HIP: (16.5, 17.0),
        figure.R_HIP: (11.5, 17.0),
        figure.L_KNEE: (16.5, 21.0),
        figure.R_KNEE: (11.5, 21.0),
        figure.L_ANKLE: (16.5, 26.0),
        figure.R_ANKLE: (11.5, 26.0),
    }


def _draw(positions: dict[int, tuple[float, float]], **kwargs) -> np.ndarray:
    frame = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame, _landmarks(positions), _identity, **kwargs)
    return frame


def _foreground_components(frame: np.ndarray) -> int:
    num_labels, _ = cv2.connectedComponents(frame, connectivity=8)
    return num_labels - 1  # drop the background label


def test_full_body_is_single_connected_component():
    frame = _draw(_full_body_panel())
    assert frame.any()
    assert _foreground_components(frame) == 1


def test_torso_filled():
    frame = _draw(_full_body_panel())
    # Centroid of the shoulders/hips quad.
    assert frame[13, 14] == 1


def test_head_disc_from_ear_spacing():
    frame = _draw(_full_body_panel())
    # Ear distance 4 => radius 1.3 * 2 = 2.6 around (14, 3).
    assert frame[3, 14] == 1
    assert frame[3, 12] == 1 and frame[3, 16] == 1  # inside the radius
    assert frame[3, 10] == 0 and frame[3, 18] == 0  # outside the radius


def test_head_falls_back_to_shoulder_width_without_ears():
    positions = _full_body_panel()
    landmarks = _landmarks(positions)
    landmarks[figure.L_EAR].visibility = 0.0
    landmarks[figure.R_EAR].visibility = 0.0
    frame = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame, landmarks, _identity)
    # Shoulder distance 8 => radius 0.3 * 8 = 2.4 around the nose (14, 3).
    assert frame[3, 14] == 1
    assert frame[3, 12] == 1
    assert frame[3, 18] == 0


def test_head_disc_fits_around_cover_points():
    without = _draw(_full_body_panel())
    assert without[3, 20] == 0
    # With face-feature points the disc re-centers on them and encloses them.
    with_cover = _draw(_full_body_panel(), head_cover_points=[(20.0, 3.0)])
    assert with_cover[3, 20] == 1


def test_head_cover_radius_is_capped():
    # Wildly spread cover points (e.g. a stale face mesh) must not balloon the
    # head past the max radius (0.25 * width = 7).
    cover = [(4.0, 3.0), (24.0, 3.0)]
    frame = _draw(_full_body_panel(), head_cover_points=cover)
    assert frame[3, 20] == 1  # 6 px from the bbox center (14, 3): inside
    assert frame[3, 23] == 0  # 9 px out: clipped by the cap


def test_head_state_smooths_radius_changes():
    state = figure.HeadState()
    frame1 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame1, _landmarks(_full_body_panel()), _identity, head_state=state)

    wide = _full_body_panel()
    wide[figure.L_EAR] = (20.0, 3.0)
    wide[figure.R_EAR] = (8.0, 3.0)  # ear spacing now targets the max radius (7)
    frame2 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame2, _landmarks(wide), _identity, head_state=state)

    # The radius moved toward the new target without jumping to it.
    assert frame2[3, 17] == 1  # 3 px out: beyond the previous ~2.6 radius
    assert frame2[3, 20] == 0  # 6 px out: well short of the instant target


def test_head_state_resets_after_gap():
    state = figure.HeadState()
    frame1 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame1, _landmarks(_full_body_panel()), _identity, head_state=state)

    # Simulate the viewer leaving and a new one appearing later.
    state.updated_at -= figure.HEAD_SMOOTHING_RESET_SECONDS + 1.0

    wide = _full_body_panel()
    wide[figure.L_EAR] = (20.0, 3.0)
    wide[figure.R_EAR] = (8.0, 3.0)
    frame2 = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame2, _landmarks(wide), _identity, head_state=state)

    assert frame2[3, 20] == 1  # full target radius applied immediately


def test_no_shoulder_cap_above_shoulder_line():
    frame = _draw(_full_body_panel())
    # The arm capsules' round caps are inset toward the elbow, so nothing may
    # poke above the torso top row (y=9) outside the neck/head columns.
    assert not frame[7:9, 18:].any()
    assert not frame[7:9, :11].any()


def test_neck_scales_with_shoulder_width():
    positions = {
        figure.NOSE: (14.5, 3.0),
        figure.L_EAR: (16.5, 3.0),
        figure.R_EAR: (12.5, 3.0),
        figure.L_SHOULDER: (20.5, 9.0),
        figure.R_SHOULDER: (8.5, 9.0),
    }
    frame = _draw(positions)
    # Shoulder width 12 => neck radius 0.16 * 12 = 1.92 => 4 columns wide,
    # still narrower than the head disc.
    neck_cols = np.flatnonzero(frame[7])
    assert neck_cols.size == 4
    head_cols = np.flatnonzero(frame[3])
    assert head_cols.size > neck_cols.size


def test_hand_disc_extends_past_wrist():
    positions = {
        figure.L_SHOULDER: (3.0, 7.5),
        figure.L_ELBOW: (13.0, 7.5),
        figure.L_WRIST: (23.0, 7.5),
    }
    frame = _draw(positions)
    # No index landmark: the hand disc sits past the wrist along the forearm.
    assert frame[7, 25] == 1
    assert frame[7, 27] == 0


def test_hand_follows_index_landmark():
    positions = {
        figure.L_SHOULDER: (3.0, 7.5),
        figure.L_ELBOW: (13.0, 7.5),
        figure.L_WRIST: (23.0, 7.5),
        figure.L_INDEX: (24.0, 10.5),
    }
    frame = _draw(positions)
    # Hand disc centered between wrist and index tip: (23.5, 9).
    assert frame[10, 23] == 1
    assert frame[7, 25] == 0  # no longer extended straight along the forearm


def test_arm_tapers_from_shoulder_to_wrist():
    # Only one horizontal arm on a half-integer row for symmetric columns.
    positions = {
        figure.L_SHOULDER: (3.0, 7.5),
        figure.L_ELBOW: (13.0, 7.5),
        figure.L_WRIST: (23.0, 7.5),
    }
    frame = _draw(positions)
    heights = [int(frame[:, col].sum()) for col in range(4, 23)]
    assert heights[0] == 4  # near the shoulder
    assert heights[-1] == 2  # near the wrist
    assert all(a >= b for a, b in zip(heights, heights[1:], strict=False))


def test_arm_over_chest_gets_outline_but_stays_attached():
    positions = _full_body_panel()
    # Cross the right forearm horizontally over the chest.
    positions[figure.R_ELBOW] = (8.0, 13.0)
    positions[figure.R_WRIST] = (13.0, 13.0)
    frame = _draw(positions)
    # At a forearm column outside the shoulder merge zone the capsule is lit
    # and separated from the torso by a dark ring above and below.
    assert frame[13, 13] == 1
    assert frame[11, 13] == 0
    assert frame[15, 13] == 0
    # The merge zone keeps the arm attached: wrist and head share a component.
    _, labels = cv2.connectedComponents(frame, connectivity=8)
    assert labels[13, 13] == labels[3, 14]


def test_no_dark_ring_at_shoulder_attachment():
    frame = _draw(_full_body_panel())
    # Torso pixels within the merge radius of the left shoulder (18, 9) are
    # never carved by the arm outline, so the attachment reads seamless.
    region = frame[9:12, 16:19]
    assert region.all(), f"hole at the shoulder attachment: {region}"


def test_waist_up_extends_torso_to_bottom():
    landmarks = _landmarks(_full_body_panel())
    for idx in (
        figure.L_HIP,
        figure.R_HIP,
        figure.L_KNEE,
        figure.R_KNEE,
        figure.L_ANKLE,
        figure.R_ANKLE,
    ):
        landmarks[idx].visibility = 0.1
    frame = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame, landmarks, _identity)
    assert frame[SIZE - 1, 12:16].all(), "torso should run off the bottom edge"
    assert _foreground_components(frame) == 1


def test_legs_drawn_when_visible():
    frame = _draw(_full_body_panel())
    assert frame[24, 16] == 1  # shin pixel between knee and ankle


def test_legs_skipped_when_knees_invisible():
    landmarks = _landmarks(_full_body_panel())
    for idx in (figure.L_KNEE, figure.R_KNEE, figure.L_ANKLE, figure.R_ANKLE):
        landmarks[idx].visibility = 0.1
    frame = np.zeros((SIZE, SIZE), dtype=np.uint8)
    figure.draw_figure(frame, landmarks, _identity)
    assert not frame[20:, :].any()


def test_mirrors_x_with_normalized_mapper():
    # Normalized camera coords through the mirroring mapper used in production.
    x = 0.85
    positions = {
        figure.NOSE: (x, 0.08),
        figure.L_EAR: (x + 0.05, 0.08),
        figure.R_EAR: (x - 0.05, 0.08),
        figure.L_SHOULDER: (x + 0.12, 0.3),
        figure.R_SHOULDER: (x - 0.12, 0.3),
        figure.L_ELBOW: (x + 0.18, 0.45),
        figure.R_ELBOW: (x - 0.18, 0.45),
        figure.L_WRIST: (x + 0.22, 0.6),
        figure.R_WRIST: (x - 0.22, 0.6),
        figure.L_HIP: (x + 0.08, 0.6),
        figure.R_HIP: (x - 0.08, 0.6),
    }
    frame = np.zeros((SIZE, SIZE), dtype=np.uint8)

    def mirror(landmark) -> tuple[float, float]:
        return SIZE - (landmark.x * SIZE), landmark.y * SIZE

    figure.draw_figure(frame, _landmarks(positions), mirror)
    # A body on the camera's left lands on the panel's left half.
    assert frame[:, :14].any()
    assert not frame[:, 14:].any()


def test_all_invisible_draws_nothing():
    frame = _draw({})
    assert int(frame.sum()) == 0
