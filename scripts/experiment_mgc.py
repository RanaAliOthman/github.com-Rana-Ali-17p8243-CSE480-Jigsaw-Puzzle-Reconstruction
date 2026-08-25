"""Compare candidate compatibility measures by true-neighbour rank.

WORK IN PROGRESS -- written but never executed. See NEXT_STEPS.md.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_matcher import load_solved, rank_report
from src import edge_compatibility as EC

DEPTHS = 3          # strips sampled at three depths
CHANNELS = 3


def bands(side):
    """(depth, sample, channel) view of a side's colour strip."""
    c = np.asarray(side.colors, dtype=np.float64)
    n = c.shape[0]
    return c.reshape(n, DEPTHS, CHANNELS).transpose(1, 0, 2)


def ssd_dc(a, b):
    """SSD after removing each strip's mean colour (illumination offset)."""
    x, y = bands(a), bands(b)[:, ::-1, :]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return float(np.mean((x - y) ** 2))


def _directional_mgc(a, b, regulariser=1e-4):
    """Cost of continuing piece A's colour gradient across the seam into B."""
    x, y = bands(a), bands(b)[:, ::-1, :]
    near_a, next_a = x[0], x[1]            # depth 3 and depth 8 under A's edge
    near_b = y[0]                          # depth 3 under B's edge, aligned to A

    gradient = near_a - next_a             # A's own outward colour change
    mu = gradient.mean(axis=0)
    covariance = np.cov(gradient, rowvar=False) + regulariser * np.eye(CHANNELS)
    inverse = np.linalg.inv(covariance)

    crossing = near_b - near_a             # observed change across the seam
    residual = crossing - mu
    return float(np.einsum('ij,jk,ik->i', residual, inverse, residual).mean())


def mgc(a, b):
    """Symmetric Mahalanobis gradient compatibility."""
    return _directional_mgc(a, b) + _directional_mgc(b, a)


def main():
    pieces = load_solved()
    inf = float('inf')

    def gate(fn):
        return lambda a, b: fn(a, b) if EC.complementary(a.kind, b.kind) else inf

    rank_report(pieces, gate(EC.colour_dissimilarity), 'colour SSD (baseline)')
    rank_report(pieces, gate(ssd_dc), 'colour SSD, DC removed')
    rank_report(pieces, gate(mgc), 'MGC')

    scales = EC.estimate_scales(pieces)
    shape_scale = scales[0]
    for w in (0.05, 0.10, 0.20):
        rank_report(pieces, gate(
            lambda a, b, w=w: (1 - w) * mgc(a, b)
            + w * 40.0 * EC.shape_dissimilarity(a, b) / shape_scale),
            'MGC %.2f + shape %.2f' % (1 - w, w))


if __name__ == '__main__':
    main()
