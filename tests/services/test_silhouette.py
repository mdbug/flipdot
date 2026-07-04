import types

import numpy as np

import app.services.silhouette as silhouette


def _landmarks(positions: dict[int, tuple[float, float]], visibility: float = 1.0):
    """Build a fake pose_landmarks object from {index: (x, y)} in [0, 1] coords."""
    landmark = [types.SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
    for i, (x, y) in positions.items():
        landmark[i] = types.SimpleNamespace(x=x, y=y, visibility=visibility)
    return types.SimpleNamespace(landmark=landmark)


def _full_body_positions(x: float = 0.5) -> dict[int, tuple[float, float]]:
    """Landmark positions for a simple upright body centred at camera x."""
    return {
        silhouette.NOSE: (x, 0.1),
        silhouette.L_SHOULDER: (x + 0.1, 0.3),
        silhouette.R_SHOULDER: (x - 0.1, 0.3),
        13: (x + 0.15, 0.45),
        14: (x - 0.15, 0.45),
        15: (x + 0.2, 0.6),
        16: (x - 0.2, 0.6),
        silhouette.L_HIP: (x + 0.08, 0.6),
        silhouette.R_HIP: (x - 0.08, 0.6),
        25: (x + 0.08, 0.75),
        26: (x - 0.08, 0.75),
        27: (x + 0.08, 0.9),
        28: (x - 0.08, 0.9),
    }


def test_none_pose_returns_none():
    assert silhouette.pose_to_mask(None, 28, 28) is None


def test_segmentation_mask_is_thresholded_resized_and_mirrored():
    seg = np.zeros((8, 8), dtype=np.float32)
    seg[:, :4] = 1.0  # person fills the left half of the camera image
    pose_results = types.SimpleNamespace(segmentation_mask=seg)

    mask = silhouette.pose_to_mask(pose_results, 28, 28)

    assert mask is not None
    assert mask.shape == (28, 28)
    assert mask.dtype == bool
    # Mirrored: the camera's left half lands on the panel's right half.
    assert mask[:, 14:].all()
    assert not mask[:, :14].any()


def test_segmentation_center_crops_non_square_input():
    seg = np.zeros((8, 16), dtype=np.float32)
    seg[:, :4] = 1.0  # outside the central 8x8 crop
    pose_results = types.SimpleNamespace(segmentation_mask=seg)

    assert silhouette.pose_to_mask(pose_results, 28, 28) is None


def test_empty_segmentation_returns_none():
    pose_results = types.SimpleNamespace(segmentation_mask=np.zeros((8, 8), dtype=np.float32))
    assert silhouette.pose_to_mask(pose_results, 28, 28) is None


def test_skeleton_fallback_draws_body():
    pose_results = types.SimpleNamespace(
        segmentation_mask=None, pose_landmarks=_landmarks(_full_body_positions())
    )

    mask = silhouette.pose_to_mask(pose_results, 28, 28)

    assert mask is not None
    assert mask.shape == (28, 28)
    assert mask.any()


def test_skeleton_fallback_mirrors_x():
    # Body on the camera's left (x near 1 after mirroring is near 0).
    pose_results = types.SimpleNamespace(
        segmentation_mask=None, pose_landmarks=_landmarks(_full_body_positions(x=0.85))
    )

    mask = silhouette.pose_to_mask(pose_results, 28, 28)

    assert mask is not None
    # Mirrored to the panel's left half; nothing on the far right.
    assert mask[:, :14].any()
    assert not mask[:, 20:].any()


def test_mask_outline_keeps_edge_and_clears_interior():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True

    outline = silhouette.mask_outline(mask)

    assert outline[2, 2:7].all() and outline[6, 2:7].all()
    assert outline[2:7, 2].all() and outline[2:7, 6].all()
    assert not outline[3:6, 3:6].any()


def test_mask_outline_skips_panel_border():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:, 3:7] = True  # body cropped by the bottom of the frame

    outline = silhouette.mask_outline(mask)

    assert not outline[9, 4:6].any()  # no line traced along the panel border
    assert outline[5, 3:7].all()  # the blob's top edge is still outlined


def test_skeleton_ignores_low_visibility_landmarks():
    pose_results = types.SimpleNamespace(
        segmentation_mask=None,
        pose_landmarks=_landmarks(_full_body_positions(), visibility=0.1),
    )

    assert silhouette.pose_to_mask(pose_results, 28, 28) is None
