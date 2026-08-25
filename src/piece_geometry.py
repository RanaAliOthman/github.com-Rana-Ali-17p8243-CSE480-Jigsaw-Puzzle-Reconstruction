"""Canonical piece representation and four-sided edge description.

A *canonical* piece is one that has been warped so that its four body corners
land on a fixed square of side ``BODY`` centred in a ``CANVAS`` x ``CANVAS``
image.  Every piece therefore has the same scale and the same body geometry,
and the only remaining degree of freedom is which of the four quarter turns is
the "upright" one -- exactly the ambiguity the assembly algorithm resolves.

Side conventions
----------------
The outer boundary is traced clockwise on screen (see
:mod:`src.contour_extraction`).  Sides are the four boundary arcs between
consecutive body corners, named ``top``, ``right``, ``bottom``, ``left`` and
stored in that order.  Consequently ``top`` runs left-to-right, ``right`` runs
top-to-bottom, ``bottom`` runs right-to-left and ``left`` runs bottom-to-top.

Two sides that physically mate are traversed in *opposite* physical directions,
so one of the two descriptors must always be reversed before comparison.  That
holds for both adjacency orientations -- ``A.right`` against ``B.left`` and
``A.bottom`` against ``B.top`` -- which is why the matching code can reverse
unconditionally instead of case-splitting.
"""
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .contour_extraction import trace_boundary

__all__ = ["SideType", "Side", "CanonicalPiece", "BODY", "CANVAS",
           "body_rect", "canonicalize", "describe_piece", "rotate_piece",
           "classify_side", "side_relief"]

BODY = 128      # body-square side length in canonical pixels
CANVAS = 224    # canonical canvas; the margin holds tabs on all four sides
SIDE_NAMES = ("top", "right", "bottom", "left")


class SideType(str, Enum):
    """Physical type of one piece side."""
    FLAT = "FLAT"    # straight border edge
    TAB = "TAB"      # protrusion
    BLANK = "BLANK"  # indentation


@dataclass
class Side:
    """One described side of a piece."""
    name: str
    kind: SideType
    profile: np.ndarray   # signed outward deviation from the chord, in body units
    colors: np.ndarray    # (n, 3 * len(depths)) interior colour strip, 0..1
    points: np.ndarray    # resampled boundary points, (n, 2)

    @property
    def relief(self):
        """Signed mean deviation of the central half of the side."""
        n = len(self.profile)
        return float(np.mean(self.profile[n // 4:3 * n // 4]))


@dataclass
class CanonicalPiece:
    """A piece warped onto the canonical canvas, with its four sides described."""
    id: int
    image: np.ndarray
    mask: np.ndarray
    sides: list
    contour: np.ndarray
    rotation: int = 0     # quarter turns applied relative to the source observation
    depths: tuple = ()    # colour-strip sampling depths used to build `sides`

    @property
    def kinds(self):
        return tuple(side.kind for side in self.sides)

    @property
    def flat_count(self):
        return sum(kind is SideType.FLAT for kind in self.kinds)


def body_rect(body=BODY, canvas=CANVAS):
    """Corner coordinates of the canonical body square, clockwise from top-left."""
    pad = (canvas - body) / 2.0
    return np.float32([[pad, pad], [pad + body, pad],
                       [pad + body, pad + body], [pad, pad + body]])


def canonicalize(image, mask, corners, body=BODY, canvas=CANVAS):
    """Warp a piece so its body corners land on the canonical square."""
    transform = cv2.getPerspectiveTransform(
        np.asarray(corners, np.float32), body_rect(body, canvas))
    warped_image = cv2.warpPerspective(image, transform, (canvas, canvas),
                                       flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    warped_mask = cv2.warpPerspective(mask, transform, (canvas, canvas),
                                      flags=cv2.INTER_NEAREST)
    return warped_image, (warped_mask > 127).astype(np.uint8) * 255


def _resample(points, n):
    """Resample a polyline to ``n`` points evenly spaced by arc length."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.zeros((n, 2))
    steps = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if steps[-1] <= 0:
        return np.repeat(points[:1], n, axis=0)
    target = np.linspace(0.0, steps[-1], n)
    return np.column_stack([np.interp(target, steps, points[:, 0]),
                            np.interp(target, steps, points[:, 1])])


def _side_arcs(contour, body=BODY, canvas=CANVAS):
    """Split a clockwise contour into the four arcs between body corners."""
    contour = np.asarray(contour, dtype=np.float64)
    corners = body_rect(body, canvas)
    indices = [int(np.argmin(np.sum((contour - corner) ** 2, axis=1)))
               for corner in corners]
    arcs = []
    for start, stop in zip(indices, indices[1:] + indices[:1]):
        if stop >= start:
            arc = contour[start:stop + 1]
        else:
            arc = np.vstack([contour[start:], contour[:stop + 1]])
        arcs.append(arc)
    return arcs, indices


def _signed_profile(arc_points, start_corner, end_corner, body=BODY):
    """Signed distance of each point from the corner-to-corner chord.

    Positive means outside the body square (material added, i.e. towards a tab);
    negative means inside it (material removed, i.e. towards a blank).  Values
    are expressed as a fraction of the body side so they are scale free.
    """
    chord = np.asarray(end_corner, np.float64) - np.asarray(start_corner, np.float64)
    length = np.linalg.norm(chord)
    if length < 1e-9:
        return np.zeros(len(arc_points))
    # Clockwise traversal puts the piece interior to the left of travel, so the
    # outward normal is the right-hand normal of the chord direction.
    normal = np.array([chord[1], -chord[0]]) / length
    offsets = np.asarray(arc_points, np.float64) - np.asarray(start_corner, np.float64)
    return (offsets @ normal) / body


def _sample_colors(image, mask, arc_points, centre, depths):
    """Sample interior colour strips at several depths beneath a side."""
    points = np.asarray(arc_points, np.float64)
    inward = np.asarray(centre, np.float64) - points
    norms = np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-6)
    inward /= norms

    height, width = mask.shape[:2]
    bands = []
    for depth in depths:
        coords = np.rint(points + depth * inward).astype(int)
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        sample = image[coords[:, 1], coords[:, 0]].astype(np.float32) / 255.0
        # Where the sample fell outside the silhouette the colour is background;
        # carry the previous valid band inwards instead of injecting black.
        outside = mask[coords[:, 1], coords[:, 0]] == 0
        if outside.any() and bands:
            sample[outside] = bands[-1][outside]
        bands.append(sample)
    return np.concatenate(bands, axis=1).astype(np.float32)


def side_relief(profile):
    """Mean signed deviation of a side's central half, in body-side units."""
    profile = np.asarray(profile, dtype=np.float64)
    if not len(profile):
        return 0.0
    n = len(profile)
    return float(np.mean(profile[n // 4:3 * n // 4]))


def classify_side(profile, flat_tolerance=0.075):
    """Label a side FLAT, TAB or BLANK from its signed profile.

    The decision uses the *relief* -- the mean signed deviation over the central
    half of the side -- rather than the peak deviation.  A border edge is
    straight along its whole length, so its relief is near zero, while a tab or
    a blank displaces roughly 0.11 to 0.18 body units at its centre.  Peak
    deviation was tried first and proved unusable: a single nick in the
    cardboard, or a few pixels of segmentation noise near a corner, pushes the
    peak of a genuinely straight edge past any threshold that still separates
    tabs from flats.  Measured over this puzzle's 140 sides the relief is
    cleanly trimodal, with an empty band between 0.068 and 0.095.
    """
    relief = side_relief(profile)
    if abs(relief) < flat_tolerance:
        return SideType.FLAT
    return SideType.TAB if relief > 0 else SideType.BLANK


def describe_piece(piece_id, image, mask, samples=96, depths=(3, 8, 14),
                   body=BODY, canvas=CANVAS, rotation=0):
    """Describe a canonical piece's four sides.

    Parameters
    ----------
    image, mask : numpy.ndarray
        Canonical canvas image and binary mask.
    samples : int
        Number of arc-length-uniform samples per side.
    depths : tuple of int
        Depths, in canonical pixels, at which the interior colour strips are
        sampled.  Several depths make the photometric signature tolerant of the
        soft, slightly shadowed edge of a physical cardboard piece.
    """
    contour = trace_boundary(mask)
    arcs, _ = _side_arcs(contour, body, canvas)
    corners = body_rect(body, canvas)
    ys, xs = np.nonzero(mask)
    centre = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([canvas / 2, canvas / 2])

    sides = []
    for index, name in enumerate(SIDE_NAMES):
        points = _resample(arcs[index], samples)
        profile = _signed_profile(points, corners[index], corners[(index + 1) % 4], body)
        colors = _sample_colors(image, mask, points, centre, depths)
        sides.append(Side(name=name, kind=classify_side(profile),
                          profile=profile, colors=colors, points=points))
    return CanonicalPiece(id=piece_id, image=image, mask=mask, sides=sides,
                          contour=contour, rotation=rotation,
                          depths=tuple(depths))


def rotate_piece(piece, k):
    """Return the piece rotated by ``k`` counter-clockwise quarter turns.

    The piece is re-described from the rotated pixels rather than by permuting
    the existing side objects, so every descriptor is measured in the same
    canonical traversal direction it will be compared in.
    """
    k %= 4
    if k == 0:
        return piece
    return describe_piece(piece.id, np.rot90(piece.image, k).copy(),
                          np.rot90(piece.mask, k).copy(),
                          samples=len(piece.sides[0].profile),
                          depths=piece.depths or (3, 8, 14),
                          rotation=(piece.rotation + k) % 4)
