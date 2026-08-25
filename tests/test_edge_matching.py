"""Tests for the compatibility measure in :mod:`src.edge_compatibility`.

The measure is Mahalanobis gradient compatibility: it extrapolates a side's own
inward colour gradient across the seam and penalises the discrepancy.  These
tests cut a known image into real interlocking pieces, so the true partner of
each side is known exactly.
"""
import numpy as np
import pytest

from src.edge_compatibility import (CHANNELS, INCOMPATIBLE, colour_bands,
                                    colour_dissimilarity, compatibility,
                                    compatibility_matrices, complementary,
                                    mgc_dissimilarity, shape_dissimilarity)
from src.piece_geometry import (BODY, CANVAS, SideType, describe_piece,
                                rotate_piece)
from src.synthetic import BLANK, FLAT, TAB, cut_image, piece_mask


@pytest.fixture(scope="module")
def quartet():
    """A 2x2 puzzle cut from a smooth image, in row-major order."""
    ys, xs = np.mgrid[0:256, 0:256].astype(np.float64)
    image = np.clip(np.dstack([128 + 110 * np.sin(xs / 29.0),
                               128 + 110 * np.cos(ys / 33.0),
                               128 + 110 * np.sin((xs + ys) / 41.0)]),
                    0, 255).astype(np.uint8)
    cut, _ = cut_image(image, 2, 2, margin=48, seed=3)
    return [describe_piece(i, patch, mask) for i, (patch, mask) in enumerate(cut)]


def test_the_gate_admits_only_a_tab_against_a_blank():
    assert complementary(SideType.TAB, SideType.BLANK)
    assert complementary(SideType.BLANK, SideType.TAB)
    assert not complementary(SideType.TAB, SideType.TAB)
    assert not complementary(SideType.BLANK, SideType.BLANK)
    assert not complementary(SideType.FLAT, SideType.TAB)
    assert not complementary(SideType.FLAT, SideType.FLAT)


def test_colour_bands_reshapes_by_depth_and_reverses(quartet):
    side = quartet[0].sides[1]
    bands = colour_bands(side)
    depths = side.colors.shape[1] // CHANNELS
    assert bands.shape == (depths, side.colors.shape[0], CHANNELS)
    reversed_bands = colour_bands(side, reverse=True)
    assert np.allclose(reversed_bands, bands[:, ::-1, :])


def test_the_true_partner_scores_better_than_the_wrong_one(quartet):
    top_left, top_right, bottom_left, bottom_right = quartet
    source = top_left.sides[1]                 # right side of the top-left piece
    true_partner = top_right.sides[3]          # its actual neighbour
    wrong_partner = bottom_right.sides[3]      # a side from elsewhere in the image
    assert mgc_dissimilarity(source, true_partner) < mgc_dissimilarity(source, wrong_partner)

    source = top_left.sides[2]                 # bottom side
    assert (mgc_dissimilarity(source, bottom_left.sides[0])
            < mgc_dissimilarity(source, bottom_right.sides[0]))


def test_the_measure_is_symmetric(quartet):
    a = quartet[0].sides[1]
    b = quartet[1].sides[3]
    assert mgc_dissimilarity(a, b) == pytest.approx(mgc_dissimilarity(b, a), rel=1e-9)


def test_compatibility_rejects_pairs_that_cannot_mate(quartet):
    piece = quartet[0]
    tab_sides = [s for s in piece.sides if s.kind is SideType.TAB]
    if len(tab_sides) >= 2:
        assert compatibility(tab_sides[0], tab_sides[1]) == INCOMPATIBLE
    for side in piece.sides:
        if side.kind is SideType.FLAT:
            assert compatibility(side, piece.sides[0]) == INCOMPATIBLE
    # The gate can be lifted deliberately; the number then still comes out finite.
    assert np.isfinite(compatibility(piece.sides[0], piece.sides[0], gate=False))


def test_compatibility_returns_a_single_finite_number(quartet):
    value = compatibility(quartet[0].sides[1], quartet[1].sides[3])
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert value >= 0.0


def test_shape_is_not_part_of_the_score(quartet):
    """Shape is a gate only, so the score must not move when a profile changes."""
    a = quartet[0].sides[1]
    b = quartet[1].sides[3]
    before = compatibility(a, b)
    shape_before = shape_dissimilarity(a, b)
    original = b.profile.copy()
    try:
        b.profile = b.profile * 0.5      # a badly mismatched silhouette
        assert compatibility(a, b) == pytest.approx(before)
        assert shape_dissimilarity(a, b) != pytest.approx(shape_before)
    finally:
        b.profile = original


def test_the_matrices_cover_every_pair_and_rotation(quartet):
    variants = [[rotate_piece(piece, k) for k in range(4)] for piece in quartet]
    horizontal, vertical = compatibility_matrices(variants)
    count = len(quartet)
    assert horizontal.shape == (count, 4, count, 4) == vertical.shape
    for i in range(count):
        assert np.all(np.isinf(horizontal[i, :, i, :]))
        assert np.all(np.isinf(vertical[i, :, i, :]))
    ungated_h, ungated_v = compatibility_matrices(variants, gate=False)
    assert np.isfinite(ungated_h).sum() > np.isfinite(horizontal).sum()


def test_colour_ssd_baseline_still_works(quartet):
    """The SSD baseline is retained for the comparisons in the report."""
    source = quartet[0].sides[1]
    assert (colour_dissimilarity(source, quartet[1].sides[3])
            < colour_dissimilarity(source, quartet[3].sides[3]))


def _flat_coloured_piece(kinds, colour, pid):
    """A piece of one uniform colour, so only the seam geometry can differ."""
    mask = piece_mask(kinds, size=BODY, margin=(CANVAS - BODY) // 2)
    image = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    image[mask > 0] = colour
    return describe_piece(pid, image, mask)


def test_a_mating_pair_beats_a_wrongly_coloured_one_of_the_same_shape():
    left = _flat_coloured_piece((FLAT, TAB, FLAT, FLAT), (60, 120, 200), 1)
    right = _flat_coloured_piece((FLAT, FLAT, FLAT, BLANK), (60, 120, 200), 2)
    other = _flat_coloured_piece((FLAT, FLAT, FLAT, BLANK), (200, 60, 60), 3)

    source, good, bad = left.sides[1], right.sides[3], other.sides[3]
    assert complementary(source.kind, good.kind)
    assert shape_dissimilarity(source, good) < 1e-3, 'a cut tab fits its own blank'
    assert colour_dissimilarity(source, good) < colour_dissimilarity(source, bad)
    assert compatibility(source, good) < compatibility(source, bad)


def test_two_tabs_on_the_same_piece_are_rejected_outright():
    piece = _flat_coloured_piece((TAB, TAB, FLAT, FLAT), (90, 150, 210), 4)
    assert compatibility(piece.sides[0], piece.sides[1]) == INCOMPATIBLE
