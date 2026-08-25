"""Tests for the from-scratch enhancement primitives in :mod:`src.enhancement`.

The brief requires these operations to be implemented without OpenCV, so the
tests check them against explicit textbook definitions rather than against a
library reference: the convolution is compared with a per-pixel double loop, the
Gaussian kernel with its analytic form, and the median filter with a plain
Python neighbourhood median.
"""
import numpy as np
import pytest

from src.enhancement import (contrast_stretch, convolution, gaussian_filter,
                             gaussian_kernel, histogram,
                             histogram_equalization, laplacian_kernel,
                             laplacian_sharpen, mean_filter, median_filter,
                             unsharp_mask)


def _reference_convolution(image, kernel):
    """The textbook per-pixel double loop, with the same reflect padding."""
    kh, kw = kernel.shape
    padded = np.pad(np.asarray(image, float), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
                    mode="reflect")
    out = np.zeros(np.shape(image), float)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = (padded[y:y + kh, x:x + kw] * kernel).sum()
    return out


def test_convolution_matches_an_explicit_per_pixel_loop():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (12, 15)).astype(float)
    kernel = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    assert np.allclose(convolution(image, kernel), _reference_convolution(image, kernel))


def test_convolution_preserves_shape_and_handles_non_square_kernels():
    image = np.arange(30.).reshape(5, 6)
    assert np.allclose(convolution(image, np.array([[1.]])), image)
    assert convolution(image, np.ones((1, 5)) / 5).shape == image.shape
    assert convolution(image, np.ones((5, 1)) / 5).shape == image.shape


def test_convolution_filters_each_channel_independently():
    rng = np.random.default_rng(1)
    colour = rng.integers(0, 256, (9, 9, 3)).astype(float)
    kernel = gaussian_kernel(3, 0.8)
    blurred = convolution(colour, kernel)
    assert blurred.shape == colour.shape
    for channel in range(3):
        assert np.allclose(blurred[:, :, channel], convolution(colour[:, :, channel], kernel))


def test_convolution_rejects_even_kernels():
    with pytest.raises(ValueError):
        convolution(np.zeros((5, 5)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        convolution(np.zeros((5, 5)), np.ones((3, 4)))


def test_mean_filter_averages_its_neighbourhood():
    image = np.zeros((9, 9))
    image[4, 4] = 9.0
    smoothed = mean_filter(image, 3)
    assert smoothed[4, 4] == pytest.approx(1.0)
    assert smoothed.sum() == pytest.approx(image.sum())


def test_gaussian_kernel_is_normalised_symmetric_and_analytic():
    kernel = gaussian_kernel(5, 1.0)
    assert kernel.shape == (5, 5)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel[::-1]) and np.allclose(kernel, kernel[:, ::-1])
    assert kernel[2, 2] == kernel.max()

    offsets = np.arange(5) - 2
    x, y = np.meshgrid(offsets, offsets)
    analytic = np.exp(-(x * x + y * y) / 2.0)
    assert np.allclose(kernel, analytic / analytic.sum())


def test_gaussian_kernel_spreads_with_sigma_and_validates_its_arguments():
    narrow, wide = gaussian_kernel(7, 0.8), gaussian_kernel(7, 3.0)
    assert wide[0, 0] > narrow[0, 0]
    assert wide[3, 3] < narrow[3, 3]
    for bad in ((4, 1.0), (5, 0.0), (5, -1.0)):
        with pytest.raises(ValueError):
            gaussian_kernel(*bad)


def test_gaussian_filter_smooths_without_changing_total_intensity():
    image = np.zeros((21, 21))
    image[10, 10] = 100.0
    smoothed = gaussian_filter(image, 5, 1.0)
    assert smoothed[10, 10] < 100.0
    assert smoothed.sum() == pytest.approx(100.0, rel=1e-6)


def test_median_filter_removes_impulses_that_smoothing_only_spreads():
    image = np.zeros((9, 9), np.uint8)
    image[4, 4] = 255
    assert median_filter(image, 3)[4, 4] == 0
    assert mean_filter(image, 3)[4, 4] > 0


def test_median_filter_matches_a_plain_neighbourhood_median():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (7, 7)).astype(np.uint8)
    padded = np.pad(image, 1, mode="reflect")
    reference = np.array([[np.median(padded[y:y + 3, x:x + 3]) for x in range(7)]
                          for y in range(7)]).astype(np.uint8)
    assert np.array_equal(median_filter(image, 3), reference)


def test_median_filter_banding_does_not_change_the_result():
    """The filter processes horizontal bands; band size must not be observable."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (40, 11, 3)).astype(np.uint8)
    whole = median_filter(image, 5, chunk_rows=1000)
    banded = median_filter(image, 5, chunk_rows=7)
    assert np.array_equal(whole, banded)


def test_median_filter_rejects_even_sizes():
    with pytest.raises(ValueError):
        median_filter(np.zeros((5, 5), np.uint8), 4)


def test_histogram_counts_every_pixel():
    image = np.array([[0, 0], [255, 128]], np.uint8)
    counts = histogram(image)
    assert counts.shape == (256,)
    assert counts.sum() == image.size
    assert counts[0] == 2 and counts[128] == 1 and counts[255] == 1


def test_histogram_equalisation_spans_the_full_range():
    rng = np.random.default_rng(4)
    image = (rng.integers(80, 140, (32, 32))).astype(np.uint8)
    equalised = histogram_equalization(image)
    assert equalised.dtype == np.uint8
    assert equalised.max() == 255 and equalised.min() == 0
    assert equalised.std() > image.std()


def test_histogram_equalisation_leaves_a_constant_image_alone():
    flat = np.full((8, 8), 42, np.uint8)
    assert np.array_equal(histogram_equalization(flat), flat)


def test_contrast_stretch_maps_the_given_limits_to_the_full_range():
    image = np.array([[10, 10], [20, 20]], np.uint8)
    assert contrast_stretch(image, 10, 20).tolist() == [[0, 0], [255, 255]]


def test_contrast_stretch_defaults_ignore_a_few_outliers():
    """The percentile limits must come from the body of the image, not its specks."""
    image = np.tile(np.linspace(80, 120, 20).astype(np.uint8), (20, 1))
    image[0, 0] = 0          # a single dark speck ...
    image[0, 1] = 255        # ... and a single specular highlight
    stretched = contrast_stretch(image)
    body = stretched[1:]     # the ramp, untouched by the two outliers
    assert body.max() - body.min() > 200, 'the ramp must use most of the range'
    assert stretched[0, 0] == 0 and stretched[0, 1] == 255
    # Taking the limits from the extremes instead would compress the ramp badly.
    assert np.ptp(contrast_stretch(image, 0, 255)[1:]) < 60


def test_unsharp_mask_raises_a_peak_above_its_surroundings():
    image = np.full((15, 15), 100.0)
    image[7, 7] = 180.0
    sharpened = unsharp_mask(image, 5, 1.0, 1.0)
    assert sharpened.dtype == np.uint8
    assert sharpened[7, 7] >= 180
    assert int(sharpened[7, 7]) - int(sharpened[7, 4]) > 180 - 100


def test_laplacian_kernel_sums_to_zero_in_both_forms():
    assert laplacian_kernel(diagonal=True).sum() == 0
    assert laplacian_kernel(diagonal=False).sum() == 0
    assert laplacian_kernel(diagonal=False)[0, 0] == 0


def test_laplacian_sharpening_increases_local_contrast():
    image = np.full((15, 15), 100.0)
    image[7, 7] = 160.0
    sharpened = laplacian_sharpen(image, 1.0)
    assert sharpened.dtype == np.uint8
    assert sharpened[7, 7] > 160
    assert sharpened[0, 0] == 100
