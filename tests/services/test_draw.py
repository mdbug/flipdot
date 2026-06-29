import numpy as np

from app.services.draw import (
    draw_circle,
    draw_line,
    draw_point,
    fill_circle,
    thick_line,
)


def test_draw_point_sets_and_clips():
    frame = np.zeros((10, 10), dtype=np.uint8)
    draw_point(frame, 3, 4)
    assert frame[4, 3] == 1

    # Out-of-bounds points are silently dropped.
    draw_point(frame, -1, 5)
    draw_point(frame, 20, 20)
    assert int(frame.sum()) == 1


def test_draw_point_color_zero_clears():
    frame = np.ones((5, 5), dtype=np.uint8)
    draw_point(frame, 2, 2, color=0)
    assert frame[2, 2] == 0


def test_draw_line_horizontal():
    frame = np.zeros((10, 10), dtype=np.uint8)
    draw_line(frame, (1, 5), (8, 5))
    assert (frame[5, 1:9] == 1).all()


def test_draw_line_diagonal_endpoints():
    frame = np.zeros((10, 10), dtype=np.uint8)
    draw_line(frame, (0, 0), (9, 9))
    assert frame[0, 0] == 1
    assert frame[9, 9] == 1


def test_draw_line_is_thin_8_connected():
    # Standard Bresenham lights one pixel per major-axis step (no doubled rows).
    frame = np.zeros((20, 20), dtype=np.uint8)
    draw_line(frame, (2, 2), (17, 9))  # shallow slope, x is the major axis
    for col in range(2, 18):
        assert int(frame[:, col].sum()) == 1


def test_draw_line_color_zero_carves():
    frame = np.ones((6, 6), dtype=np.uint8)
    draw_line(frame, (0, 0), (5, 5), color=0)
    assert frame[0, 0] == 0
    assert frame[5, 5] == 0


def test_draw_circle_outline():
    frame = np.zeros((20, 20), dtype=np.uint8)
    draw_circle(frame, (10, 10), 6)
    assert int(frame.sum()) > 0
    # The center is not part of the outline.
    assert frame[10, 10] == 0
    # A point near the rim along the x axis should be lit.
    assert frame[10, 16] == 1


def test_fill_circle_is_filled_and_symmetric():
    frame = np.zeros((28, 28), dtype=np.uint8)
    # Half-integer center => clean mirror symmetry across both axes.
    fill_circle(frame, (13.5, 13.5), 13.5)
    assert frame[13, 13] == 1
    assert int(frame.sum()) > 200
    assert (frame == frame[:, ::-1]).all()
    assert (frame == frame[::-1, :]).all()


def test_fill_circle_color_zero_carves():
    frame = np.ones((10, 10), dtype=np.uint8)
    fill_circle(frame, (4.5, 4.5), 2, color=0)
    assert frame[4, 4] == 0
    assert frame[0, 0] == 1


def test_thick_line_width_three_includes_centerline():
    frame = np.zeros((12, 12), dtype=np.uint8)
    thick_line(frame, (1, 1), (10, 10), width=3)
    # An odd width includes the exact centerline, so each diagonal pixel is lit.
    for i in range(1, 11):
        assert frame[i, i] == 1
    # A thicker stroke lights more pixels than the bare 1px centerline.
    assert int(frame.sum()) > 10
