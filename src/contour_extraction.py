"""Boundary tracing and contour utilities.

:func:`trace_boundary` is a from-scratch Moore-neighbourhood contour tracer; it
is the routine the reconstruction pipeline uses.  :func:`extract_contour_cv2`
is kept only so the unit tests can cross-check the two against each other.
"""
import cv2
import numpy as np

__all__ = ["trace_boundary", "extract_contour", "extract_contour_cv2",
           "boundary_from_mask", "polygon_area"]

# Eight neighbours in clockwise order for image coordinates (x right, y down),
# starting at East.  Clockwise on screen is the traversal convention used by the
# whole library, so every piece side is described in the same rotational sense.
_RING = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def polygon_area(points):
    """Signed shoelace area; positive for a clockwise loop in image coordinates.

    Image coordinates put ``y`` downwards, which flips the usual sign convention:
    a loop that appears clockwise on screen encloses positive shoelace area here.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def trace_boundary(mask, clockwise=True):
    """Trace one region's outer boundary with Moore-neighbourhood tracing.

    The traced region is the one containing the topmost-leftmost foreground
    pixel; use :func:`extract_contour` when the mask may hold several regions
    and the largest one is wanted.

    Starting from the topmost-leftmost foreground pixel, the tracer repeatedly
    walks the eight-neighbourhood ring of the current boundary pixel, beginning
    just after the pixel it arrived from, and steps onto the first foreground
    neighbour it meets.  Jacob's stopping criterion terminates the walk when the
    start pixel is re-entered from the same direction, which correctly handles
    boundaries that must be visited twice (one-pixel-wide necks between a jigsaw
    tab and the piece body, for example).

    Parameters
    ----------
    mask : array_like
        Non-zero pixels are foreground.
    clockwise : bool
        Traversal sense in image coordinates; the default matches the rest of
        the library.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` int32 array of ``(x, y)`` boundary points in order.
    """
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.empty((0, 2), np.int32)

    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    first = int(np.argmin(ys * width + xs))
    start = (int(xs[first]), int(ys[first]))

    def is_foreground(point):
        x, y = point
        return 0 <= x < width and 0 <= y < height and binary[y, x]

    # The pixel west of the start is background by construction of `start`.
    backtrack = (start[0] - 1, start[1])
    points = [start]
    current = start
    state = None
    guard = 8 * int(binary.sum()) + 16
    for _ in range(guard):
        offset = (backtrack[0] - current[0], backtrack[1] - current[1])
        index = _RING.index(offset)
        found = None
        for step in range(1, 9):
            candidate_offset = _RING[(index + step) % 8]
            candidate = (current[0] + candidate_offset[0],
                         current[1] + candidate_offset[1])
            if is_foreground(candidate):
                found = candidate
                break
            backtrack = candidate
        if found is None:          # isolated pixel
            break
        current = found
        if state is None:
            state = (current, backtrack)
        elif (current, backtrack) == state:
            break
        points.append(current)
    else:                          # pragma: no cover - guard should never trip
        raise RuntimeError("boundary tracing did not terminate")

    contour = np.asarray(points[:-1] if len(points) > 1 else points, np.int32)
    if len(contour) > 2:
        is_clockwise = polygon_area(contour) > 0
        if is_clockwise != clockwise:
            contour = contour[::-1].copy()
    return contour


def extract_contour(mask, method="trace"):
    """Outer contour of the largest region as an ``(N, 2)`` array of ``(x, y)``."""
    if method != "trace":
        return extract_contour_cv2(mask)
    from .segmentation import connected_components
    binary = np.asarray(mask) > 0
    if not binary.any():
        return np.empty((0, 2), np.int32)
    labels, components = connected_components(binary)
    if len(components) > 1:
        largest = max(components, key=lambda c: c.area)
        binary = labels == largest.label
    return trace_boundary(binary)


def extract_contour_cv2(mask):
    """Reference contour from OpenCV, used to validate :func:`trace_boundary`."""
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), np.int32)
    return max(contours, key=cv2.contourArea)[:, 0, :]


def boundary_from_mask(mask):
    """One-pixel-wide boundary image (mask minus its erosion)."""
    binary = np.asarray(mask) > 0
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return (binary & ~eroded).astype(np.uint8) * 255
