"""Image enhancement primitives implemented from scratch.

Every routine in this module is written directly on NumPy arrays; no OpenCV
filtering, histogram or convolution function is used.  All filters share a
single :func:`convolution` implementation so that the report can state one
padding and one accumulation convention for the whole library.

Convention: the routines here are *correlation*-style (the kernel is not
flipped).  All kernels used by the library are either symmetric (mean,
Gaussian, Laplacian) or defined directly in correlation form (Sobel, Prewitt),
so this choice only fixes the sign convention of the gradient operators.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "convolution", "mean_filter", "gaussian_kernel", "gaussian_filter",
    "median_filter", "histogram", "histogram_equalization", "contrast_stretch",
    "unsharp_mask", "laplacian_kernel", "laplacian_sharpen",
]


def convolution(image, kernel, padding="reflect"):
    """Correlate ``image`` with ``kernel``, preserving the input shape.

    The accumulation loops over the *kernel taps* rather than over the image
    pixels.  Each tap contributes one shifted, scaled copy of the padded image,
    which is mathematically identical to the textbook per-pixel double loop but
    costs ``k*k`` vectorised array operations instead of ``H*W`` Python
    iterations.  That difference is what makes the operators in this library
    usable on the 1920x1080 puzzle photographs.

    Parameters
    ----------
    image : array_like
        2-D greyscale or 3-D multi-channel image.  The same kernel is applied
        independently to every channel.
    kernel : array_like
        2-D kernel with odd height and width.
    padding : str
        Any mode accepted by :func:`numpy.pad`; ``"reflect"`` avoids the dark
        border that zero padding introduces at image edges.

    Returns
    -------
    numpy.ndarray
        Float64 array with the same shape as ``image``.
    """
    array = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or any(n % 2 == 0 for n in kernel.shape):
        raise ValueError("kernel dimensions must be odd")

    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    single_channel = array.ndim == 2
    if single_channel:
        array = array[..., None]

    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=padding)
    height, width = array.shape[:2]
    out = np.zeros_like(array)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight:
                out += weight * padded[i:i + height, j:j + width]
    return out[..., 0] if single_channel else out


def mean_filter(image, size=3):
    """Box (arithmetic mean) smoothing with a ``size x size`` neighbourhood."""
    return convolution(image, np.ones((size, size)) / (size * size))


def gaussian_kernel(size=5, sigma=1.0):
    """Normalised 2-D Gaussian kernel derived from ``size`` and ``sigma``."""
    if size % 2 == 0 or size < 1 or sigma <= 0:
        raise ValueError("positive sigma and odd size required")
    offsets = np.arange(size) - size // 2
    x, y = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_filter(image, size=5, sigma=1.0):
    """Gaussian smoothing using a kernel built by :func:`gaussian_kernel`."""
    return convolution(image, gaussian_kernel(size, sigma))


def median_filter(image, size=3, chunk_rows=128):
    """Median filtering over a ``size x size`` neighbourhood.

    The median is a rank statistic, not a weighted sum, so it cannot be
    expressed as a convolution and some form of neighbourhood enumeration is
    unavoidable -- this is the loop the project brief asks to be justified.
    The enumeration is done by materialising the sliding windows and reducing
    them with :func:`numpy.median`, which keeps the per-pixel work in compiled
    code.  Windows are built one horizontal band at a time because a full-frame
    window stack for a 1920x1080 colour image would need several hundred
    megabytes at once.
    """
    if size % 2 == 0:
        raise ValueError("odd size required")
    array = np.asarray(image)
    radius = size // 2
    single_channel = array.ndim == 2
    work = array[..., None] if single_channel else array

    padded = np.pad(work, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    height = work.shape[0]
    out = np.empty(work.shape, dtype=np.float64)
    for start in range(0, height, chunk_rows):
        stop = min(start + chunk_rows, height)
        band = padded[start:stop + 2 * radius]
        windows = sliding_window_view(band, (size, size), axis=(0, 1))
        out[start:stop] = np.median(windows, axis=(-2, -1))

    out = out.astype(array.dtype, copy=False)
    return out[..., 0] if single_channel else out


def histogram(image, bins=256):
    """Intensity histogram computed by the library (no OpenCV/NumPy helper)."""
    values = np.clip(np.asarray(image), 0, bins - 1).astype(np.int64).ravel()
    counts = np.zeros(bins, dtype=np.int64)
    np.add.at(counts, values, 1)
    return counts


def histogram_equalization(image):
    """Global histogram equalisation via the normalised cumulative histogram."""
    array = np.asarray(image, dtype=np.uint8)
    cumulative = histogram(array).cumsum()
    non_zero = cumulative[cumulative > 0]
    if not len(non_zero) or cumulative[-1] == non_zero[0]:
        return array.copy()
    lut = np.round((cumulative - non_zero[0]) * 255 /
                   (cumulative[-1] - non_zero[0])).clip(0, 255).astype(np.uint8)
    return lut[array]


def contrast_stretch(image, low=None, high=None):
    """Linear contrast stretch between ``low`` and ``high`` intensities.

    When the limits are omitted the 2nd and 98th percentiles are used, so a
    handful of specular highlights cannot collapse the output range.
    """
    array = np.asarray(image, dtype=float)
    lo = np.percentile(array, 2) if low is None else low
    hi = np.percentile(array, 98) if high is None else high
    return np.clip((array - lo) * 255 / max(hi - lo, 1e-9), 0, 255).astype(np.uint8)


def unsharp_mask(image, size=5, sigma=1.0, amount=1.0):
    """Sharpen by adding a scaled high-pass (original minus Gaussian) residual."""
    array = np.asarray(image, dtype=float)
    blurred = gaussian_filter(array, size, sigma)
    return np.clip(array + amount * (array - blurred), 0, 255).astype(np.uint8)


def laplacian_kernel(diagonal=True):
    """Discrete Laplacian, with or without the diagonal neighbours."""
    if diagonal:
        return np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    return np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])


def laplacian_sharpen(image, amount=1.0, diagonal=True):
    """Sharpen with a Laplacian operator built on :func:`convolution`."""
    array = np.asarray(image, dtype=float)
    detail = convolution(array, laplacian_kernel(diagonal))
    return np.clip(array - amount * detail, 0, 255).astype(np.uint8)
