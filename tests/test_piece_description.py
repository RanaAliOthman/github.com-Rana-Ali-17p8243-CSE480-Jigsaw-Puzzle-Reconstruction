"""Tests for the piece-edge description required by §4 of the brief.

Description is :func:`src.piece_geometry.describe_piece`: it turns a canonical
piece into four named sides, each carrying a signed silhouette profile, a
TAB/BLANK/FLAT label and an interior colour strip.  The geometry that feeds it —
corner placement and the canonical warp — is covered by
``tests/test_piece_geometry.py``; this file covers the descriptors themselves,
on synthetic pieces whose sides are known by construction and on the 35 real
pieces extracted from the photographs.
"""
import cv2
import numpy as np
import pytest
from pathlib import Path

from src.piece_geometry import (BODY, CANVAS, SIDE_NAMES, SideType,
                                classify_side, describe_piece, rotate_piece,
                                side_relief)
from src.synthetic import BLANK, FLAT, TAB, piece_mask, side_curve

PIECES = Path('data/pieces')


def _piece(kinds, colour=(80, 140, 210), pid=1, **kwargs):
    mask = piece_mask(kinds, size=BODY, margin=(CANVAS - BODY) // 2)
    image = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    image[mask > 0] = colour
    return describe_piece(pid, image, mask, **kwargs)


@pytest.fixture(scope="module")
def real_pieces():
    """Every canonical piece extracted from the photographs, described."""
    paths = sorted(PIECES.glob('piece_*.png'))
    if len(paths) != 35:
        pytest.skip('canonical pieces not built; run scripts.build_ground_truth')
    described = []
    for path in paths:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        described.append(describe_piece(int(path.stem.split('_')[1]),
                                        rgba[:, :, :3], rgba[:, :, 3]))
    return described


# ----------------------------------------------------------------- structure

def test_a_piece_has_four_sides_named_clockwise_from_the_top():
    piece = _piece((TAB, BLANK, FLAT, TAB))
    assert [side.name for side in piece.sides] == list(SIDE_NAMES)
    assert [side.name for side in piece.sides] == ['top', 'right', 'bottom', 'left']


def test_side_types_are_recovered_from_the_silhouette():
    piece = _piece((TAB, BLANK, FLAT, TAB))
    assert piece.kinds == (SideType.TAB, SideType.BLANK, SideType.FLAT, SideType.TAB)
    assert piece.flat_count == 1


def test_every_combination_of_side_types_is_described_correctly():
    expected = {TAB: SideType.TAB, BLANK: SideType.BLANK, FLAT: SideType.FLAT}
    for kinds in ((FLAT, FLAT, TAB, BLANK), (BLANK, BLANK, BLANK, BLANK),
                  (TAB, TAB, TAB, TAB), (FLAT, TAB, FLAT, BLANK)):
        piece = _piece(kinds)
        assert piece.kinds == tuple(expected[k] for k in kinds), kinds


# ------------------------------------------------------------------ profiles

def test_relief_sign_distinguishes_tab_from_blank():
    assert side_relief(side_curve(TAB)) > 0
    assert side_relief(side_curve(BLANK)) < 0
    assert abs(side_relief(side_curve(FLAT))) < 1e-9
    assert classify_side(side_curve(TAB)) is SideType.TAB
    assert classify_side(side_curve(BLANK)) is SideType.BLANK
    assert classify_side(side_curve(FLAT)) is SideType.FLAT


def test_classification_follows_the_stated_tolerance():
    """A profile just inside the tolerance is FLAT; just outside it is not."""
    almost_flat = np.full(96, 0.07)
    assert classify_side(almost_flat) is SideType.FLAT
    assert classify_side(np.full(96, 0.08)) is SideType.TAB
    assert classify_side(np.full(96, -0.08)) is SideType.BLANK


def test_profile_starts_and_ends_at_the_corners():
    piece = _piece((TAB, BLANK, FLAT, TAB))
    for side in piece.sides:
        assert abs(side.profile[0]) < 0.03
        assert abs(side.profile[-1]) < 0.03


def test_profiles_are_sampled_at_the_requested_resolution():
    for samples in (48, 96, 192):
        piece = _piece((TAB, BLANK, FLAT, TAB), samples=samples)
        for side in piece.sides:
            assert len(side.profile) == samples
            assert side.points.shape == (samples, 2)


def test_the_side_relief_property_agrees_with_the_free_function():
    piece = _piece((TAB, BLANK, FLAT, TAB))
    for side in piece.sides:
        assert side.relief == pytest.approx(side_relief(side.profile))


# -------------------------------------------------------------- colour strip

def test_colour_strips_carry_one_triple_per_depth():
    depths = (3, 8, 14)
    piece = _piece((TAB, BLANK, FLAT, TAB), samples=96, depths=depths)
    assert piece.depths == depths
    for side in piece.sides:
        assert side.colors.shape == (96, 3 * len(depths))
        assert side.colors.min() >= 0.0 and side.colors.max() <= 1.0


def test_the_number_of_depths_is_configurable():
    piece = _piece((TAB, BLANK, FLAT, TAB), depths=(4, 10))
    assert piece.depths == (4, 10)
    assert all(side.colors.shape[1] == 6 for side in piece.sides)


def test_colour_strips_are_sampled_inside_the_piece_not_on_the_background():
    """Sampling steps inward from the boundary, so it must land on the piece.

    A handful of samples per side still miss: the inward direction points at the
    centroid, which near a tab's flanks can leave the silhouette again.  The
    measured worst case on this shape is 97.9% of samples on the piece.
    """
    colour = (80, 140, 210)
    piece = _piece((TAB, BLANK, FLAT, TAB), colour=colour)
    expected = np.array(colour, np.float32) / 255.0
    for side in piece.sides:
        for depth_index in range(side.colors.shape[1] // 3):
            band = side.colors[:, 3 * depth_index:3 * depth_index + 3]
            on_piece = np.isclose(band, expected, atol=0.02).all(axis=1)
            assert on_piece.mean() > 0.95, '%s at depth %d' % (side.name, depth_index)


def test_colour_strips_follow_the_piece_colour():
    blue = _piece((FLAT, FLAT, FLAT, FLAT), colour=(200, 60, 60), pid=1)
    red = _piece((FLAT, FLAT, FLAT, FLAT), colour=(60, 60, 200), pid=2)
    assert not np.allclose(blue.sides[0].colors, red.sides[0].colors)


# --------------------------------------------------------------- orientation

def test_rotation_permutes_sides_predictably():
    piece = _piece((TAB, BLANK, FLAT, TAB))
    turned = rotate_piece(piece, 1)
    # One counter-clockwise quarter turn moves the old right side to the top.
    assert turned.kinds == tuple(piece.kinds[(j + 1) % 4] for j in range(4))
    assert rotate_piece(piece, 0) is piece
    assert rotate_piece(piece, 4).kinds == piece.kinds


def test_rotation_preserves_the_descriptor_shape_and_depths():
    piece = _piece((TAB, BLANK, FLAT, TAB), samples=64, depths=(4, 10))
    turned = rotate_piece(piece, 3)
    assert turned.depths == piece.depths
    for side in turned.sides:
        assert len(side.profile) == 64
        assert side.colors.shape == (64, 6)


# -------------------------------------------------------- the 35 real pieces

def test_every_real_piece_yields_four_described_sides(real_pieces):
    assert len(real_pieces) == 35
    for piece in real_pieces:
        assert len(piece.sides) == 4
        for side in piece.sides:
            assert np.isfinite(side.profile).all()
            assert np.isfinite(side.colors).all()
            assert 0.0 <= side.colors.min() and side.colors.max() <= 1.0


def test_real_classifications_are_consistent_with_the_relief_they_are_derived_from(real_pieces):
    for piece in real_pieces:
        for side in piece.sides:
            relief = side_relief(side.profile)
            if side.kind is SideType.TAB:
                assert relief > 0.075
            elif side.kind is SideType.BLANK:
                assert relief < -0.075
            else:
                assert abs(relief) <= 0.075


def test_real_side_types_are_approximately_but_not_exactly_the_truth(real_pieces):
    """The 5x7 truth is 24 flats and 58 tab/blank pairs; the classifier misses some.

    Measured on this puzzle: 56 tabs, 59 blanks, 25 flats.  Per-piece flat
    classification is known to be unreliable — three pieces get the wrong flat
    count under every profile statistic tried — which is why the solver charges a
    cost for silhouette disagreement instead of trusting these labels.
    """
    kinds = [side.kind for piece in real_pieces for side in piece.sides]
    assert len(kinds) == 140
    flats = kinds.count(SideType.FLAT)
    tabs = kinds.count(SideType.TAB)
    blanks = kinds.count(SideType.BLANK)
    assert 22 <= flats <= 28, 'flat count %d is far from the true 24' % flats
    assert abs(tabs - blanks) <= 6, 'tabs %d vs blanks %d' % (tabs, blanks)
    assert flats != 24 or tabs == blanks == 58, \
        'if the counts ever become exact, the solver may trust them again'
