"""Tests for :mod:`src.edge_detection`.

The gradient operators are checked on synthetic step edges whose true edge
position and orientation are known exactly, and the Canny stages are checked
individually — the brief requires each stage to be inspectable, so each one is
exercised on its own as well as through the full routine.
"""
import numpy as np
import pytest

from src.edge_detection import (canny, double_threshold, hysteresis,
                                non_maximum_suppression, prewitt, sobel)


def _vertical_edge(size=20, at=10):
    image = np.zeros((size, size))
    image[:, at:] = 255.0
    return image


def _horizontal_edge(size=20, at=10):
    image = np.zeros((size, size))
    image[at:, :] = 255.0
    return image


@pytest.mark.parametrize("operator", [sobel, prewitt])
def test_gradient_operators_return_magnitude_and_orientation(operator):
    magnitude, orientation = operator(_vertical_edge())
    assert magnitude.shape == orientation.shape == (20, 20)
    assert magnitude.max() > 0


@pytest.mark.parametrize("operator", [sobel, prewitt])
def test_a_vertical_edge_produces_a_horizontal_gradient(operator):
    magnitude, orientation = operator(_vertical_edge())
    strong = magnitude > 0.5 * magnitude.max()
    # The gradient points across the edge: 0 (dark to bright) or +-pi.
    angles = np.abs(orientation[strong])
    assert np.all((angles < 0.2) | (np.abs(angles - np.pi) < 0.2))
    # And the response sits on the step, not away from it.
    columns = np.unique(np.nonzero(strong)[1])
    assert set(columns) <= {9, 10}


@pytest.mark.parametrize("operator", [sobel, prewitt])
def test_a_horizontal_edge_produces_a_vertical_gradient(operator):
    magnitude, orientation = operator(_horizontal_edge())
    strong = magnitude > 0.5 * magnitude.max()
    angles = np.abs(orientation[strong])
    assert np.all(np.abs(angles - np.pi / 2) < 0.2)
    rows = np.unique(np.nonzero(strong)[0])
    assert set(rows) <= {9, 10}


def test_a_flat_image_has_no_gradient():
    flat = np.full((15, 15), 77.0)
    for operator in (sobel, prewitt):
        magnitude, _ = operator(flat)
        assert magnitude.max() == pytest.approx(0.0)


def test_sobel_weights_the_centre_row_more_heavily_than_prewitt():
    """Same edge, different smoothing: the two operators must not be identical."""
    image = _vertical_edge()
    assert sobel(image)[0].max() > prewitt(image)[0].max()


def test_non_maximum_suppression_thins_a_ridge_to_one_pixel():
    magnitude = np.zeros((9, 9))
    magnitude[:, 3] = 4.0
    magnitude[:, 4] = 9.0      # the ridge
    magnitude[:, 5] = 4.0
    orientation = np.zeros_like(magnitude)   # gradient points along +x
    thinned = non_maximum_suppression(magnitude, orientation)
    interior = thinned[1:-1, 1:-1]
    assert np.all(interior[:, 3] == 9.0)     # column 4 of the full array survives
    assert np.count_nonzero(interior) == interior.shape[0]


def test_non_maximum_suppression_never_adds_energy():
    rng = np.random.default_rng(9)
    magnitude = rng.random((25, 25)) * 100
    orientation = rng.random((25, 25)) * np.pi - np.pi / 2
    thinned = non_maximum_suppression(magnitude, orientation)
    assert np.all((thinned == 0) | (thinned == magnitude))
    assert np.count_nonzero(thinned) < np.count_nonzero(magnitude)


def test_double_threshold_labels_strong_weak_and_suppressed():
    values = np.array([[10.0, 30.0, 80.0]])
    labelled = double_threshold(values, low=20, high=50)
    assert labelled.tolist() == [[0, 75, 255]]
    assert labelled.dtype == np.uint8


def test_hysteresis_keeps_weak_pixels_only_when_they_touch_a_strong_one():
    labelled = np.zeros((7, 7), np.uint8)
    labelled[1, 1] = 255          # a strong seed ...
    labelled[1, 2] = 75           # ... with a weak pixel touching it
    labelled[1, 3] = 75           # ... and a weak pixel touching that
    labelled[5, 5] = 75           # an isolated weak pixel, far away
    linked = hysteresis(labelled)
    assert linked[1, 1] == linked[1, 2] == linked[1, 3] == 255
    assert linked[5, 5] == 0
    assert set(np.unique(linked)) <= {0, 255}


def test_canny_exposes_every_stage_and_returns_a_binary_map():
    edges, stages = canny(_vertical_edge(40, 20), sigma=1.2, low=20, high=50)
    assert set(np.unique(edges)) <= {0, 255}
    for stage in ("smoothed", "magnitude", "orientation", "nms", "double_threshold"):
        assert stage in stages, 'missing stage %s' % stage
    assert stages["smoothed"].shape == edges.shape
    assert set(np.unique(stages["double_threshold"])) <= {0, 75, 255}
    assert np.count_nonzero(stages["nms"]) < np.count_nonzero(stages["magnitude"])


def test_canny_finds_the_edge_where_it_actually_is():
    edges, _ = canny(_vertical_edge(40, 20), sigma=1.2, low=20, high=50)
    interior = edges[5:-5]
    for row in interior:
        found = np.nonzero(row)[0]
        assert len(found) and np.all(np.abs(found - 19.5) <= 1.5)


def test_canny_thresholds_control_how_much_survives():
    rng = np.random.default_rng(10)
    image = rng.random((40, 40)) * 40
    image[:, 20:] += 120
    permissive, _ = canny(image, low=5, high=15)
    strict, _ = canny(image, low=60, high=120)
    assert np.count_nonzero(permissive) > np.count_nonzero(strict)


def test_canny_smoothing_precedes_differentiation():
    """The stored 'smoothed' stage must be blurrier than the input."""
    rng = np.random.default_rng(11)
    image = rng.random((30, 30)) * 255
    _, stages = canny(image, sigma=2.0)
    assert np.std(np.diff(stages["smoothed"], axis=1)) < np.std(np.diff(image, axis=1))
