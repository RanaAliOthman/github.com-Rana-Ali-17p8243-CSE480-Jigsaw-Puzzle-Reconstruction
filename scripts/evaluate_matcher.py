"""Measure how well a compatibility measure identifies true neighbours.

With the solved layout known, every one of the 58 internal seams gives a
(side, true partner) pair.  For each such side we rank all physically plausible
partners by the measure under test and record where the true one lands.  Rank 1
for most sides is the precondition for any assembly search to succeed.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.piece_geometry import describe_piece, rotate_piece, SideType
from src import edge_compatibility as EC

ROWS, COLS = 5, 7


def load_solved():
    """Return pieces already rotated into their solved orientation, in cell order."""
    layout = json.loads(Path('data/ground_truth/layout.json').read_text())
    rotations = layout['solved_rotations_ccw']
    pieces = []
    for position in range(ROWS * COLS):
        piece_id = layout['positions_piece_id'][position]
        rgba = cv2.imread(str(Path('data/pieces') / ('piece_%02d.png' % piece_id)),
                          cv2.IMREAD_UNCHANGED)
        base = describe_piece(piece_id, rgba[:, :, :3], rgba[:, :, 3])
        pieces.append(rotate_piece(base, int(rotations[str(piece_id)])))
    return pieces


def true_pairs(pieces):
    """(a_index, a_side, b_index, b_side) for every internal seam, solved frame."""
    pairs = []
    for position in range(ROWS * COLS):
        row, col = divmod(position, COLS)
        if col < COLS - 1:
            pairs.append((position, 1, position + 1, 3))
        if row < ROWS - 1:
            pairs.append((position, 2, position + COLS, 0))
    return pairs


def rank_report(pieces, measure, label):
    """Rank of the true partner for each seam, in both directions."""
    sides = [(index, j, piece.sides[j])
             for index, piece in enumerate(pieces) for j in range(4)]
    ranks = []
    for a_index, a_side, b_index, b_side in true_pairs(pieces):
        for (src_i, src_j), (dst_i, dst_j) in (((a_index, a_side), (b_index, b_side)),
                                               ((b_index, b_side), (a_index, a_side))):
            source = pieces[src_i].sides[src_j]
            scored = []
            for index, j, side in sides:
                if index == src_i:
                    continue
                value = measure(source, side)
                if np.isfinite(value):
                    scored.append((value, index, j))
            scored.sort()
            position = next((k + 1 for k, (_, i, j) in enumerate(scored)
                             if i == dst_i and j == dst_j), None)
            ranks.append(position if position is not None else len(scored) + 1)
    ranks = np.array(ranks)
    print('%-28s rank1 %3d/%3d (%4.1f%%)  top3 %4.1f%%  median %3.0f  mean %5.1f'
          % (label, int((ranks == 1).sum()), len(ranks),
             100 * (ranks == 1).mean(), 100 * (ranks <= 3).mean(),
             np.median(ranks), ranks.mean()))
    return ranks


def main():
    pieces = load_solved()
    print('loaded %d pieces in solved orientation; %d internal seams\n'
          % (len(pieces), len(true_pairs(pieces))))

    scales = EC.estimate_scales(pieces)

    rank_report(pieces, lambda a, b: EC.shape_dissimilarity(a, b)
                if EC.complementary(a.kind, b.kind) else np.inf, 'shape only')
    rank_report(pieces, lambda a, b: EC.colour_dissimilarity(a, b)
                if EC.complementary(a.kind, b.kind) else np.inf, 'colour only (SSD)')
    for w in (0.6, 0.3, 0.15):
        rank_report(pieces, lambda a, b, w=w: EC.compatibility(
            a, b, scales, shape_weight=w, colour_weight=1 - w),
            'shape %.2f + colour %.2f' % (w, 1 - w))


if __name__ == '__main__':
    main()
