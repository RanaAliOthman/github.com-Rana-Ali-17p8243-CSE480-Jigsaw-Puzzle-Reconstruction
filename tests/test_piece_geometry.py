"""Tests for the canonical geometry in :mod:`src.piece_geometry`.

This file covers the geometric frame a piece is placed in — the body square, the
perspective warp onto the canonical canvas, and the arc-length resampling of the
boundary.  The descriptors built on top of that frame (side types, profiles,
colour strips, rotation) are covered by ``tests/test_piece_description.py``.
"""
import cv2
import numpy as np

from src.piece_geometry import (BODY, CANVAS, body_rect, canonicalize,
                                describe_piece, rotate_piece)
from src.synthetic import BLANK, FLAT, TAB, piece_mask


def _canvas_piece(kinds=(TAB, BLANK, FLAT, TAB), colour=(80, 140, 210)):
    mask = piece_mask(kinds, size=BODY, margin=(CANVAS - BODY) // 2)
    image = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    image[mask > 0] = colour
    return image, mask


def test_body_rect_is_centred_on_the_canvas():
    corners = body_rect(BODY, CANVAS)
    assert corners.shape == (4, 2)
    pad = (CANVAS - BODY) / 2.0
    assert np.allclose(corners, [[pad, pad], [pad + BODY, pad],
                                 [pad + BODY, pad + BODY], [pad, pad + BODY]])
    # The margin on every side is what holds the tabs.
    assert pad > 0


def test_canonicalize_places_the_body_square():
    mask = piece_mask((TAB, BLANK, FLAT, TAB), size=90, margin=40)
    image = np.zeros(mask.shape + (3,), np.uint8)
    image[mask > 0] = (200, 200, 200)
    corners = np.float32([[40, 40], [130, 40], [130, 130], [40, 130]])
    warped_image, warped_mask = canonicalize(image, mask, corners)
    assert warped_mask.shape == (CANVAS, CANVAS)
    assert warped_image.shape == (CANVAS, CANVAS, 3)
    pad = (CANVAS - BODY) // 2
    # The body interior must be filled and the canvas corners must be empty.
    assert warped_mask[pad + BODY // 2, pad + BODY // 2] > 0
    assert warped_mask[2, 2] == 0


def test_canonicalize_undoes_a_perspective_distortion():
    """A piece photographed at an angle must land on the same canonical square."""
    image, mask = _canvas_piece()
    pad = (CANVAS - BODY) // 2
    square = body_rect(BODY, CANVAS)
    skewed = np.float32([[pad - 12, pad + 5], [pad + BODY + 9, pad - 7],
                         [pad + BODY + 4, pad + BODY + 14], [pad + 3, pad + BODY - 6]])
    warp = cv2.getPerspectiveTransform(square, skewed)
    tilted_image = cv2.warpPerspective(image, warp, (CANVAS, CANVAS))
    tilted_mask = cv2.warpPerspective(mask, warp, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST)

    _, restored = canonicalize(tilted_image, tilted_mask, skewed)
    overlap = ((restored > 0) & (mask > 0)).sum() / max((mask > 0).sum(), 1)
    assert overlap > 0.95


def test_canonical_mask_is_binary():
    image, mask = _canvas_piece()
    _, warped_mask = canonicalize(image, mask, body_rect(BODY, CANVAS))
    assert set(np.unique(warped_mask)) <= {0, 255}


def test_the_boundary_is_resampled_uniformly_along_its_arc():
    image, mask = _canvas_piece()
    piece = describe_piece(1, image, mask, samples=96)
    for side in piece.sides:
        steps = np.linalg.norm(np.diff(side.points, axis=0), axis=1)
        assert steps.max() < 3 * steps.mean()
        assert steps.min() > 0


def test_rotating_the_pixels_rotates_the_canvas_not_its_size():
    image, mask = _canvas_piece()
    piece = describe_piece(1, image, mask)
    turned = rotate_piece(piece, 1)
    assert turned.image.shape == piece.image.shape
    assert turned.mask.shape == piece.mask.shape
    assert np.array_equal(turned.mask, np.rot90(piece.mask, 1))
