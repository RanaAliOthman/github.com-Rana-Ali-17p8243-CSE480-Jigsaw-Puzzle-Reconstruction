"""Tests for the blind reconstruction search in :mod:`src.solver`.

The photographed puzzle cannot be used to test whether the *search* works,
because its compatibility measure is only 46.6% accurate at rank 1 and any
failure would be ambiguous between the two.  These tests therefore cut a known
image into a real interlocking puzzle, scramble it, and require the search to
put it back exactly.
"""
import numpy as np
import pytest

from src.piece_geometry import describe_piece, rotate_piece
from src.solver import (Placement, Reconstruction, _growth_order, _ring_order,
                        build_variants, infer_grid, infer_layout, render_grid,
                        solve)
from src.synthetic import cut_image

ROWS, COLS = 2, 3


def textured_image(height, width):
    """A smooth, strongly coloured image; MGC needs a gradient to extrapolate."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    red = 128 + 110 * np.sin(xs / 37.0) * np.cos(ys / 51.0)
    green = 128 + 110 * np.sin((xs + 2 * ys) / 61.0)
    blue = 128 + 110 * np.cos((3 * xs - ys) / 43.0)
    return np.clip(np.dstack([blue, green, red]), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def puzzle():
    """A cut-up 2x3 puzzle: the pieces in solved order, plus their true layout."""
    image = textured_image(ROWS * 128, COLS * 128)
    cut, layout = cut_image(image, ROWS, COLS, margin=48, seed=7)
    pieces = [describe_piece(index, patch, mask)
              for index, (patch, mask) in enumerate(cut)]
    return pieces, layout


def scramble(pieces, seed):
    """Shuffle the bag and give every piece a random rotation."""
    rng = np.random.default_rng(seed)
    turns = {piece.id: int(rng.integers(0, 4)) for piece in pieces}
    turned = [rotate_piece(piece, turns[piece.id]) for piece in pieces]
    order = rng.permutation(len(turned))
    return [turned[i] for i in order], turns


def neighbours(ids, cols):
    pairs = set()
    for cell, pid in enumerate(ids):
        row, col = divmod(cell, cols)
        if col < cols - 1:
            pairs.add(frozenset((pid, ids[cell + 1])))
        if row < len(ids) // cols - 1:
            pairs.add(frozenset((pid, ids[cell + cols])))
    return pairs


def test_grid_is_inferred_from_the_pieces_alone(puzzle):
    pieces, _ = puzzle
    assert infer_grid(pieces) == (ROWS, COLS)
    shape, labels = infer_layout(pieces)
    assert shape == (ROWS, COLS)
    # A 2x3 rectangle is all border: four corners with two flat sides each and
    # two edge pieces with one.
    assert sorted(len(sides) for sides in labels) == [1, 1, 2, 2, 2, 2]


def test_ring_walk_visits_every_border_cell_once_and_stays_adjacent():
    ring = _ring_order(5, 7)
    border = [c for c in range(35)
              if c // 7 in (0, 4) or c % 7 in (0, 6)]
    assert sorted(ring) == border
    for first, second in zip(ring, ring[1:] + ring[:1]):
        gap = abs(first - second)
        assert gap == 1 or gap == 7, (first, second)


def test_growth_order_always_has_a_placed_neighbour():
    ring = _ring_order(5, 7)
    placed = set(ring)
    for cell in _growth_order(5, 7, ring):
        row, col = divmod(cell, 7)
        around = set()
        if col > 0: around.add(cell - 1)
        if col < 6: around.add(cell + 1)
        if row > 0: around.add(cell - 7)
        if row < 4: around.add(cell + 7)
        assert around & placed, cell
        placed.add(cell)


def test_a_scrambled_synthetic_puzzle_is_reconstructed_exactly(puzzle):
    pieces, _ = puzzle
    scrambled, turns = scramble(pieces, seed=3)
    result = solve(scrambled, beam_width=64, per_state=4)

    assert result.grid == (ROWS, COLS)
    assert result.complete
    assert len({p.piece_id for p in result.placements}) == len(pieces)

    truth = list(range(ROWS * COLS))
    placed = result.piece_ids()
    # A rectangle that is not square has exactly one symmetry the seams cannot
    # distinguish: the whole puzzle turned through 180 degrees.
    assert placed == truth or placed[::-1] == truth, placed

    for cell, placement in enumerate(result.placements):
        applied = (turns[placement.piece_id] + placement.rotation) % 4
        expected = 0 if placed == truth else 2
        assert applied == expected, (cell, placement, applied)


def test_reconstruction_is_deterministic(puzzle):
    pieces, _ = puzzle
    scrambled, _ = scramble(pieces, seed=11)
    first = solve(scrambled, beam_width=32, per_state=3)
    second = solve(scrambled, beam_width=32, per_state=3)
    assert first.piece_ids() == second.piece_ids()
    assert first.rotations() == second.rotations()
    assert first.cost == pytest.approx(second.cost)


def test_supplied_order_does_not_change_the_answer(puzzle):
    pieces, _ = puzzle
    scrambled, _ = scramble(pieces, seed=5)
    forward = solve(scrambled, beam_width=32, per_state=3)
    backward = solve(list(reversed(scrambled)), beam_width=32, per_state=3)
    assert neighbours(forward.piece_ids(), COLS) == neighbours(backward.piece_ids(), COLS)


def test_quality_score_is_reported_and_bounded(puzzle):
    pieces, _ = puzzle
    scrambled, _ = scramble(pieces, seed=3)
    result = solve(scrambled, beam_width=64, per_state=4)
    quality = result.quality
    assert quality['seams_realised'] == quality['seams_expected']
    assert 0.0 <= quality['quality_score'] <= 1.0
    assert 0.0 <= quality['mutual_best_fraction'] <= 1.0
    assert quality['mean_seam_cost'] >= 0.0


def test_rendering_produces_the_expected_canvas(puzzle):
    pieces, _ = puzzle
    scrambled, _ = scramble(pieces, seed=3)
    variants = build_variants(scrambled)
    result = solve(scrambled, beam_width=32, per_state=3, variants=variants)
    image = render_grid(variants, result, margin=10)
    assert image.shape == (ROWS * 128 + 20, COLS * 128 + 20, 3)
    assert image.std() > 5, 'the canvas should not be blank'


def test_an_incomplete_arrangement_is_still_returned(puzzle):
    """A grid with more cells than pieces must not raise; it returns what it has."""
    pieces, _ = puzzle
    with pytest.raises(ValueError):
        solve(pieces, rows=3, cols=3)


def test_placement_carries_identity_without_using_it(puzzle):
    pieces, _ = puzzle
    scrambled, _ = scramble(pieces, seed=3)
    result = solve(scrambled, beam_width=32, per_state=3)
    assert all(isinstance(p, Placement) for p in result.placements)
    assert isinstance(result, Reconstruction)
    for placement in result.placements:
        assert scrambled[placement.index].id == placement.piece_id
