"""Tests for :mod:`src.thresholding`.

The three required methods are checked on images built so that each one has a
known correct answer: a clean bimodal image for the global and Otsu methods, and
an image with a strong illumination gradient — where no single global threshold
can succeed — for the adaptive method.
"""
import numpy as np

from src.thresholding import adaptive_threshold, global_threshold, otsu_threshold


def _bimodal(dark=40, bright=200, size=10):
    """Half the pixels dark, half bright, so the correct threshold is any value between."""
    image = np.full((size, size), dark, np.uint8)
    image[size // 2:] = bright
    return image


def test_global_threshold_is_binary_and_splits_at_the_given_value():
    image = _bimodal()
    mask = global_threshold(image, 100)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[:5].max() == 0 and mask[5:].min() == 255


def test_global_threshold_inversion_is_the_complement():
    image = _bimodal()
    mask = global_threshold(image, 100)
    inverted = global_threshold(image, 100, invert=True)
    assert np.array_equal(inverted, 255 - mask)


def test_otsu_finds_a_threshold_between_the_two_modes():
    image = _bimodal(dark=40, bright=200)
    mask, threshold = otsu_threshold(image)
    assert 40 <= threshold < 200
    assert np.array_equal(mask, global_threshold(image, threshold))
    assert mask[:5].max() == 0 and mask[5:].min() == 255


def test_otsu_tracks_the_modes_when_they_move():
    low_contrast = _bimodal(dark=100, bright=140)
    high_contrast = _bimodal(dark=10, bright=250)
    assert otsu_threshold(low_contrast)[1] > otsu_threshold(high_contrast)[1]


def test_otsu_separates_the_foreground_of_a_noisy_bimodal_image():
    rng = np.random.default_rng(5)
    truth = np.zeros((60, 60), bool)
    truth[15:45, 15:45] = True
    image = np.where(truth, 190, 60) + rng.normal(0, 12, truth.shape)
    mask, _ = otsu_threshold(np.clip(image, 0, 255).astype(np.uint8))
    agreement = ((mask > 0) == truth).mean()
    assert agreement > 0.99


def test_otsu_inversion_selects_the_other_side():
    image = _bimodal()
    mask, threshold = otsu_threshold(image)
    inverted, inverted_threshold = otsu_threshold(image, invert=True)
    assert inverted_threshold == threshold
    assert np.array_equal(inverted, 255 - mask)


def test_adaptive_threshold_succeeds_where_a_global_one_cannot():
    """A bright square on a strong illumination ramp: no global value works.

    The square is 45 grey levels above its surroundings, but the ramp spans 200
    levels, so every global threshold either swallows the bright end of the ramp
    or misses the square.  The local comparison sees only the 45-level step.
    """
    ramp = np.tile(np.linspace(0, 200, 80), (80, 1))
    truth = np.zeros((80, 80), bool)
    truth[30:50, 30:50] = True
    image = np.clip(ramp + 45 * truth, 0, 255).astype(np.uint8)

    best_global = max(((global_threshold(image, t) > 0) == truth).mean()
                      for t in range(0, 256, 2))
    # c shifts the threshold below the local mean, so a negative c demands that a
    # foreground pixel stand clearly above its neighbourhood.
    adaptive = adaptive_threshold(image, size=15, c=-10) > 0
    assert best_global < 0.95, 'the ramp must actually defeat a global threshold'
    assert (adaptive == truth).mean() > 0.97
    assert (adaptive == truth).mean() > best_global


def test_adaptive_threshold_supports_both_neighbourhood_shapes():
    rng = np.random.default_rng(6)
    image = (rng.random((40, 40)) * 255).astype(np.uint8)
    for method in ("mean", "gaussian"):
        mask = adaptive_threshold(image, 7, 5, method=method)
        assert mask.shape == image.shape
        assert set(np.unique(mask)) <= {0, 255}


def test_adaptive_offset_controls_how_much_foreground_survives():
    """The threshold is the local mean minus c, so raising c admits more pixels."""
    rng = np.random.default_rng(7)
    image = (rng.random((40, 40)) * 255).astype(np.uint8)
    strict = (adaptive_threshold(image, 9, -40) > 0).mean()
    neutral = (adaptive_threshold(image, 9, 0) > 0).mean()
    generous = (adaptive_threshold(image, 9, 40) > 0).mean()
    assert strict < neutral < generous


def test_adaptive_inversion_is_the_complement():
    rng = np.random.default_rng(8)
    image = (rng.random((30, 30)) * 255).astype(np.uint8)
    mask = adaptive_threshold(image, 7, 5)
    assert np.array_equal(adaptive_threshold(image, 7, 5, invert=True), 255 - mask)


def test_a_constant_image_has_no_local_structure_to_threshold():
    """With no local variation the offset alone decides, uniformly."""
    flat = np.full((20, 20), 128, np.uint8)
    assert global_threshold(flat, 200).max() == 0
    assert global_threshold(flat, 100).min() == 255
    assert adaptive_threshold(flat, 7, 5).min() == 255      # local mean minus 5
    assert adaptive_threshold(flat, 7, -5).max() == 0       # local mean plus 5
