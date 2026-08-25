"""Extraction of individual puzzle pieces from a photograph of a scrambled puzzle.

The input this module expects is the natural one for the project: a single
top-down photograph containing all the pieces of one puzzle, laid out on a
contrasting surface in arbitrary positions and orientations.  Working from one
frame -- rather than from a separate photograph per piece -- is what keeps
illumination, white balance and scale consistent across pieces, which the
colour half of the edge-matching score depends on.

The stage sequence is: enhancement -> Otsu threshold -> morphological cleanup ->
connected-component labelling -> separation of touching pieces -> per-piece
deskew, boundary tracing and corner detection.  Every one of those operations
comes from this library.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .contour_extraction import trace_boundary
from .enhancement import gaussian_filter
from .segmentation import (binary_close, binary_open, connected_components,
                           fill_holes, separate_touching)
from .thresholding import otsu_threshold

__all__ = ["ExtractedPiece", "frame_mask", "candidate_components",
           "dominant_angle", "detect_corners", "piece_quality", "extract_frame_pieces"]


@dataclass
class ExtractedPiece:
    """One puzzle piece isolated from a frame and rotated upright.

    ``image`` and ``mask`` are the deskewed crop of the piece; ``corners`` are
    the four body corners in that crop, ordered top-left, top-right,
    bottom-right, bottom-left.  ``angle`` is the rotation (degrees, positive
    counter-clockwise) that was removed to make the body axis-aligned, and is
    only defined modulo 90 degrees -- resolving the remaining four-fold
    ambiguity is the assembly algorithm's job.
    """
    index: int
    image: np.ndarray
    mask: np.ndarray
    contour: np.ndarray
    corners: np.ndarray
    angle: float
    source_centroid: tuple
    quality: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def pitch(self):
        """Mean body-edge length in pixels (the puzzle's grid spacing)."""
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def frame_mask(image, blur_size=5, blur_sigma=1.4, morph_size=5):
    """Binary foreground mask for a whole frame, using the library's own stages.

    Thresholding is applied to the *value* channel -- the per-pixel maximum of
    the three colour channels -- rather than to luminance.  Luminance weights
    red at 0.299, so the saturated red artwork printed on this puzzle
    (BGR of roughly 40, 40, 200) has a luminance near 88 and falls on the
    background side of an Otsu threshold of about 91: whole red-printed regions
    would be carved out of the silhouette.  The value channel of the same pixel
    is 200 and separates cleanly from the dark mat.
    """
    if image.ndim == 3:
        gray = image.max(axis=2)
    else:
        gray = image
    smoothed = gaussian_filter(gray, blur_size, blur_sigma)
    binary, threshold = otsu_threshold(smoothed)
    # Keep the polarity whose image border is least occupied: the pieces are the
    # objects, the mat is the background, whichever of the two is brighter.
    if np.mean(np.r_[binary[0], binary[-1], binary[:, 0], binary[:, -1]] > 0) > 0.5:
        binary = 255 - binary
    binary = binary_open(binary, morph_size)
    binary = binary_close(binary, morph_size, iterations=2)
    binary = fill_holes(binary)
    return binary, threshold


def candidate_components(binary, area_ratio=(0.72, 1.55)):
    """Label the mask, split touching blobs, and keep piece-sized regions."""
    labels, components = separate_touching(binary)
    if not components:
        return labels, []
    areas = np.array([c.area for c in components], dtype=float)
    reference = float(np.median(areas[areas >= 0.25 * areas.max()]))
    low, high = area_ratio[0] * reference, area_ratio[1] * reference
    kept = [c for c in components if low <= c.area <= high]
    return labels, kept


def dominant_angle(contour, span=9):
    """Estimate a silhouette's orientation modulo 90 degrees from its tangents.

    Each boundary point contributes the direction of the chord spanning
    ``2*span`` points around it.  Those directions are mapped through a
    four-fold angle transform so that the four straight body edges of a jigsaw
    piece -- which are ninety degrees apart -- reinforce one another instead of
    cancelling, and the circular mean of the result is the body orientation.
    Tabs and blanks contribute directions spread over all angles and therefore
    average out.

    Returns
    -------
    float
        Angle in degrees within ``(-45, 45]``.
    """
    points = np.asarray(contour, dtype=np.float64)
    if len(points) < 2 * span + 1:
        return 0.0
    delta = np.roll(points, -span, axis=0) - np.roll(points, span, axis=0)
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    resultant = np.exp(4j * angles).mean()
    return float(np.degrees(np.angle(resultant) / 4))


def detect_corners(contour):
    """Locate the four body corners of a deskewed piece.

    With the body already axis-aligned, the corners are the boundary points
    that are extreme along the two diagonals: a tab protrudes from the middle of
    a side and so never reaches a diagonal extreme, whereas a corner is far from
    the centre along both axes at once.  This replaces per-quadrant heuristics,
    which pick tab tips whenever a piece has two tabs meeting near one corner.
    """
    c = np.asarray(contour, dtype=np.float64)
    if len(c) < 4:
        return np.empty((0, 2), np.float32)
    total, difference = c[:, 0] + c[:, 1], c[:, 0] - c[:, 1]
    order = [int(np.argmin(total)), int(np.argmax(difference)),
             int(np.argmax(total)), int(np.argmin(difference))]
    return c[order].astype(np.float32)


def piece_quality(mask, corners):
    """Score how much a silhouette looks like a jigsaw piece (higher is better).

    A genuine piece has four corners forming a near-square whose side length is
    the puzzle pitch, and its area is close to that square's area: the material
    added by tabs is very nearly the material removed by blanks, so the ratio of
    silhouette area to pitch squared sits near one regardless of how many tabs
    the piece happens to have.  Photographic clutter -- clamps, hex keys, a
    lattice bracket -- fails at least one of those tests.
    """
    if len(corners) != 4:
        return 0.0, {}
    sides = np.array([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
    diagonals = np.array([np.linalg.norm(corners[0] - corners[2]),
                          np.linalg.norm(corners[1] - corners[3])])
    if sides.min() <= 1 or diagonals.min() <= 1:
        return 0.0, {}

    squareness = float(sides.std() / sides.mean())
    diagonal_ratio = float(abs(diagonals[0] - diagonals[1]) / diagonals.mean())
    # A square's diagonal is sqrt(2) times its side; deviation detects shear.
    shear = float(abs(diagonals.mean() / (sides.mean() * np.sqrt(2)) - 1.0))
    fill = float(np.count_nonzero(mask) / (sides.mean() ** 2))

    penalty = (4.0 * squareness + 4.0 * diagonal_ratio + 4.0 * shear
               + 6.0 * max(0.0, abs(fill - 1.0) - 0.15))
    return float(np.exp(-penalty)), {
        "squareness": squareness, "diagonal_ratio": diagonal_ratio,
        "shear": shear, "fill": fill, "pitch": float(sides.mean()),
    }


def _deskew(image, mask, angle, margin=8):
    """Rotate a crop by ``-angle`` about its centre, expanding the canvas."""
    h, w = mask.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]
    size = (new_w, new_h)
    rotated_image = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC,
                                   borderValue=(0, 0, 0))
    rotated_mask = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(rotated_mask)
    if not len(xs):
        return rotated_image, rotated_mask
    x0, x1 = max(0, xs.min() - margin), min(size[0], xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(size[1], ys.max() + margin + 1)
    return rotated_image[y0:y1, x0:x1], rotated_mask[y0:y1, x0:x1]


def extract_frame_pieces(image, expected=None, min_quality=0.05, pad=14):
    """Isolate every puzzle piece in a frame and return them upright.

    Parameters
    ----------
    image : numpy.ndarray
        BGR photograph of the scrambled puzzle.
    expected : int or None
        If given, the ``expected`` highest-scoring candidates are kept; this is
        how photographic clutter of piece-like area is discarded.  If ``None``
        every candidate above ``min_quality`` is returned.
    min_quality : float
        Rejection threshold for :func:`piece_quality`.

    Returns
    -------
    list of ExtractedPiece
        Ordered by position in the frame (top to bottom, then left to right).
    """
    binary, _ = frame_mask(image)
    labels, components = candidate_components(binary)

    pieces = []
    for component in components:
        x, y, w, h = component.bbox
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        crop_mask = ((labels[y0:y1, x0:x1] == component.label) * 255).astype(np.uint8)
        crop_image = image[y0:y1, x0:x1].copy()

        contour = trace_boundary(crop_mask)
        if len(contour) < 32:
            continue
        angle = dominant_angle(contour)
        upright_image, upright_mask = _deskew(crop_image, crop_mask, angle)

        # Rotation resampling can shed a few pixels; keep the largest region.
        sub_labels, sub_components = connected_components(upright_mask)
        if not sub_components:
            continue
        largest = max(sub_components, key=lambda c: c.area)
        upright_mask = ((sub_labels == largest.label) * 255).astype(np.uint8)

        upright_contour = trace_boundary(upright_mask)
        if len(upright_contour) < 32:
            continue
        corners = detect_corners(upright_contour)
        quality, diagnostics = piece_quality(upright_mask, corners)
        if quality < min_quality:
            continue

        pieces.append(ExtractedPiece(
            index=len(pieces), image=upright_image, mask=upright_mask,
            contour=upright_contour, corners=corners, angle=angle,
            source_centroid=component.centroid, quality=quality,
            diagnostics=diagnostics,
        ))

    if expected is not None and len(pieces) > expected:
        pieces = sorted(pieces, key=lambda p: -p.quality)[:expected]

    pieces.sort(key=lambda p: (round(p.source_centroid[1] / 50), p.source_centroid[0]))
    for position, piece in enumerate(pieces):
        piece.index = position
    return pieces
