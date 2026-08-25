"""The one evaluation protocol both models and the classical measure are scored by.

Every measure — Siamese, GNN, or classical MGC — is handed to :func:`rank_report`
as the same kind of callable, so the numbers they produce are comparable by
construction rather than by assertion.  Keeping this in one module is the point:
the moment each training script writes its own ranking loop, the comparison the
brief asks for stops being a comparison.

Rank-1 here is quoted over the held-out block only.  It is *not* comparable to
the 46.6% MGC figure quoted in the Milestone 1 report, which ranks against all
140 sides of the puzzle rather than the 32 sides of the block.
"""
import numpy as np

from src.edge_compatibility import complementary

__all__ = ["rank_report", "block_seams"]


def block_seams(block_cells, cols=7, rows=5):
    """``(cell_a, side_a, cell_b, side_b)`` for the true seams inside a block."""
    seams = []
    for cell in sorted(block_cells):
        row, col = divmod(cell, cols)
        if col < cols - 1 and cell + 1 in block_cells:
            seams.append((cell, 1, cell + 1, 3))
        if row < rows - 1 and cell + cols in block_cells:
            seams.append((cell, 2, cell + cols, 0))
    return seams


def rank_report(score, pieces, block_cells, cols=7, rows=5):
    """Rank of the true partner among gated candidates inside the held-out block.

    Parameters
    ----------
    score : callable
        ``score(a_cell, a_side, b_cell, b_side)`` returning a *dissimilarity*:
        lower means a better match.  A model whose head emits a similarity must
        negate it before passing it here.
    pieces : sequence
        The solved pieces, indexed by cell.
    block_cells : set of int
        The held-out cells.  Only sides of these cells are candidates, and only
        seams wholly inside the block are scored.

    Returns
    -------
    dict
        ``pairs_scored``, ``rank1``, ``top3`` and ``median_rank``.
    """
    block = sorted(block_cells)
    sides = [(cell, j) for cell in block for j in range(4)]

    ranks = []
    for a_cell, a_side, b_cell, b_side in block_seams(block_cells, cols, rows):
        for src, dst in (((a_cell, a_side), (b_cell, b_side)),
                         ((b_cell, b_side), (a_cell, a_side))):
            scored = []
            for cell, j in sides:
                if cell == src[0]:
                    continue
                # Only pairs the tab/blank gate admits: the assembly search
                # never asks about the others, so ranking them would flatter
                # every measure equally and for free.
                if not complementary(pieces[src[0]].sides[src[1]].kind,
                                     pieces[cell].sides[j].kind):
                    continue
                scored.append((score(src[0], src[1], cell, j), cell, j))
            if not scored:
                continue
            scored.sort()
            position = next((k + 1 for k, (_, cell, j) in enumerate(scored)
                             if (cell, j) == dst), len(scored) + 1)
            ranks.append(position)

    ranks = np.array(ranks) if ranks else np.array([0])
    return {"pairs_scored": int(len(ranks)),
            "rank1": float((ranks == 1).mean()),
            "top3": float((ranks <= 3).mean()),
            "median_rank": float(np.median(ranks))}
