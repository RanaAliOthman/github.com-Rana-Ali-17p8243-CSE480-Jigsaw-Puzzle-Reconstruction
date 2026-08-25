import numpy as np

from src.contour_extraction import (boundary_from_mask, extract_contour,
                                    extract_contour_cv2, polygon_area,
                                    trace_boundary)
from src.synthetic import BLANK, TAB, piece_mask


def _point_set(contour):
    return set(map(tuple, np.asarray(contour).tolist()))


def test_traced_boundary_matches_opencv_on_a_rectangle():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 4:16] = 255
    traced = trace_boundary(mask)
    assert _point_set(traced) == _point_set(extract_contour_cv2(mask))
    assert len(traced) == 2 * (10 + 12) - 4       # perimeter pixels


def test_traced_boundary_matches_opencv_on_a_jigsaw_silhouette():
    mask = piece_mask((TAB, BLANK, TAB, BLANK), size=64, margin=24)
    traced = trace_boundary(mask)
    assert _point_set(traced) == _point_set(extract_contour_cv2(mask))


def test_traced_boundary_handles_a_narrow_neck():
    # A tab joined by a thin neck must be visited and left again correctly.
    mask = np.zeros((40, 46), np.uint8)
    mask[10:30, 10:30] = 255
    mask[18:22, 30:38] = 255
    assert _point_set(trace_boundary(mask)) == _point_set(extract_contour_cv2(mask))


def test_traversal_is_clockwise_in_image_coordinates():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 4:16] = 255
    assert polygon_area(trace_boundary(mask)) > 0
    assert polygon_area(trace_boundary(mask, clockwise=False)) < 0
    # OpenCV uses the opposite sense for an external contour.
    assert polygon_area(extract_contour_cv2(mask)) < 0


def test_extract_contour_picks_the_largest_region():
    mask = np.zeros((40, 40), np.uint8)
    mask[2:6, 2:6] = 255       # small decoy, first in raster order
    mask[15:35, 15:35] = 255   # the real region
    contour = extract_contour(mask)
    assert contour[:, 0].min() >= 15 and contour[:, 1].min() >= 15


def test_degenerate_inputs():
    assert len(trace_boundary(np.zeros((5, 5), np.uint8))) == 0
    single = np.zeros((5, 5), np.uint8)
    single[2, 2] = 255
    assert len(trace_boundary(single)) == 1


def test_boundary_from_mask_is_one_pixel_wide():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255
    boundary = boundary_from_mask(mask)
    assert boundary[5, 5] == 255 and boundary[10, 10] == 0
