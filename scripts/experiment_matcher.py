"""Search for a better compatibility measure than the current MGC baseline.

Exploratory.  It re-describes the solved pieces under several colour spaces and
colour-strip sampling depths, then reports, for each configuration, where the
true partner of every seam side ranks among all physically plausible partners.
Whatever wins here is what gets promoted into ``src.edge_compatibility``.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.piece_geometry import describe_piece, rotate_piece, SideType

ROWS, COLS = 5, 7
SPACES = {
    'bgr': None,
    'lab': cv2.COLOR_BGR2Lab,
    'hsv': cv2.COLOR_BGR2HSV,
}


def load_solved(depths=(3, 8, 14), space='bgr'):
    """Pieces in their solved orientation and cell order, described as asked."""
    layout = json.loads(Path('data/ground_truth/layout.json').read_text())
    rotations = layout['solved_rotations_ccw']
    pieces = []
    for position in range(ROWS * COLS):
        piece_id = layout['positions_piece_id'][position]
        rgba = cv2.imread(str(Path('data/pieces') / ('piece_%02d.png' % piece_id)),
                          cv2.IMREAD_UNCHANGED)
        image = rgba[:, :, :3]
        if SPACES[space] is not None:
            image = cv2.cvtColor(image, SPACES[space])
        base = describe_piece(piece_id, image, rgba[:, :, 3], depths=depths)
        pieces.append(rotate_piece(base, int(rotations[str(piece_id)])))
    return pieces


def true_pairs():
    """(a_cell, a_side, b_cell, b_side) for every internal seam of the solved grid."""
    pairs = []
    for position in range(ROWS * COLS):
        row, col = divmod(position, COLS)
        if col < COLS - 1:
            pairs.append((position, 1, position + 1, 3))
        if row < ROWS - 1:
            pairs.append((position, 2, position + COLS, 0))
    return pairs


def bands(side, n_depths):
    """(depth, sample, channel) view of a side's colour strip."""
    c = np.asarray(side.colors, dtype=np.float64)
    return c.reshape(c.shape[0], n_depths, 3).transpose(1, 0, 2)


def make_mgc(n_depths, near=0, far=1, regulariser=1e-4):
    """Symmetric Mahalanobis gradient compatibility over the chosen band pair."""
    def directional(x, y):
        near_a, next_a = x[near], x[far]
        near_b = y[near]
        gradient = near_a - next_a
        mu = gradient.mean(axis=0)
        covariance = np.cov(gradient, rowvar=False) + regulariser * np.eye(3)
        inverse = np.linalg.inv(covariance)
        residual = (near_b - near_a) - mu
        return float(np.einsum('ij,jk,ik->i', residual, inverse, residual).mean())

    def measure(a, b):
        x, y = bands(a, n_depths), bands(b, n_depths)[:, ::-1, :]
        return directional(x, y) + directional(y, x)
    return measure


def complementary(first, second):
    return {first, second} == {SideType.TAB, SideType.BLANK}


def evaluate(pieces, measure, label):
    """Rank of the true partner in both directions, plus the best-buddy count."""
    sides = [(index, j, piece.sides[j])
             for index, piece in enumerate(pieces) for j in range(4)]
    cost = {}
    for src_i, src_j, source in sides:
        for dst_i, dst_j, target in sides:
            if src_i == dst_i:
                continue
            if not complementary(source.kind, target.kind):
                continue
            cost[(src_i, src_j, dst_i, dst_j)] = measure(source, target)

    def best_partner(key):
        candidates = [(v, k[2], k[3]) for k, v in cost.items() if k[:2] == key]
        return min(candidates)[1:] if candidates else None

    ranks = []
    for a_i, a_j, b_i, b_j in true_pairs():
        for (src, dst) in (((a_i, a_j), (b_i, b_j)), ((b_i, b_j), (a_i, a_j))):
            scored = sorted((v, k[2], k[3]) for k, v in cost.items() if k[:2] == src)
            position = next((n + 1 for n, (_, i, j) in enumerate(scored)
                             if (i, j) == dst), len(scored) + 1)
            ranks.append(position)

    keys = {k[:2] for k in cost}
    best = {k: best_partner(k) for k in keys}
    buddies = sum(1 for k, v in best.items()
                  if v is not None and best.get(v) == k) // 2
    true_set = {frozenset(((a_i, a_j), (b_i, b_j))) for a_i, a_j, b_i, b_j in true_pairs()}
    correct_buddies = sum(1 for k, v in best.items()
                          if v is not None and best.get(v) == k
                          and frozenset((k, v)) in true_set) // 2

    ranks = np.array(ranks)
    print('%-34s rank1 %3d/%3d (%4.1f%%)  top3 %4.1f%%  median %3.0f  '
          'buddies %2d (%2d correct)'
          % (label, int((ranks == 1).sum()), len(ranks), 100 * (ranks == 1).mean(),
             100 * (ranks <= 3).mean(), np.median(ranks), buddies, correct_buddies))
    return ranks


CONFIGS = [
    ((3, 8, 14), 'bgr', 0, 1),
    ((3, 8, 14), 'lab', 0, 1),
    ((3, 8, 14), 'hsv', 0, 1),
    ((2, 5, 9), 'lab', 0, 1),
    ((2, 4, 7, 11), 'lab', 0, 1),
    ((2, 4, 7, 11), 'bgr', 0, 1),
    ((3, 6, 10, 15), 'lab', 0, 1),
    ((2, 3, 5, 8, 12), 'lab', 0, 1),
]


def main():
    if '--measures' in sys.argv:
        sweep_measures()
        return
    for depths, space, near, far in CONFIGS:
        pieces = load_solved(depths, space)
        evaluate(pieces, make_mgc(len(depths), near, far),
                 'MGC %s depths=%s' % (space, ','.join(map(str, depths))))


# --------------------------------------------------------------------------
# Second sweep: keep the winning colour space and depths, vary the measure.
# --------------------------------------------------------------------------
def make_mgc_robust(n_depths, near=0, far=1, regulariser=1e-4):
    """MGC whose per-sample residuals are combined by median, not mean."""
    def directional(x, y):
        gradient = x[near] - x[far]
        mu = gradient.mean(axis=0)
        covariance = np.cov(gradient, rowvar=False) + regulariser * np.eye(3)
        inverse = np.linalg.inv(covariance)
        residual = (y[near] - x[near]) - mu
        return float(np.median(np.einsum('ij,jk,ik->i', residual, inverse, residual)))

    def measure(a, b):
        x, y = bands(a, n_depths), bands(b, n_depths)[:, ::-1, :]
        return directional(x, y) + directional(y, x)
    return measure


def make_ssd_dc(n_depths):
    """SSD after removing each strip's mean colour."""
    def measure(a, b):
        x, y = bands(a, n_depths), bands(b, n_depths)[:, ::-1, :]
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        return float(np.mean((x - y) ** 2))
    return measure


def make_blend(n_depths, weight):
    """MGC plus a weighted, roughly unit-scaled DC-removed SSD term."""
    mgc, ssd = make_mgc(n_depths), make_ssd_dc(n_depths)
    return lambda a, b: mgc(a, b) + weight * 400.0 * ssd(a, b)


def sweep_measures():
    depths = (3, 8, 14)
    pieces = load_solved(depths, 'bgr')
    n = len(depths)
    evaluate(pieces, make_mgc(n, 0, 1), 'MGC gradient depths 3->8')
    evaluate(pieces, make_mgc(n, 0, 2), 'MGC gradient depths 3->14')
    evaluate(pieces, make_mgc(n, 1, 2), 'MGC gradient depths 8->14')
    evaluate(pieces, make_mgc_robust(n, 0, 1), 'MGC median-combined residuals')
    for w in (0.25, 0.5, 1.0, 2.0):
        evaluate(pieces, make_blend(n, w), 'MGC + %.2f * SSD-DC' % w)


if __name__ == '__main__':
    main()
