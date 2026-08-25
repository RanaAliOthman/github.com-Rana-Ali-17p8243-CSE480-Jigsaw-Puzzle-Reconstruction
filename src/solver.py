"""Blind reconstruction search: scrambled pieces in, an arrangement out.

Nothing in this module reads ``data/ground_truth``.  It is given an unordered
bag of :class:`~src.piece_geometry.CanonicalPiece` objects with no identity,
position or orientation information, and it searches over both position and
rotation for the arrangement whose seams are most compatible.

How the search works
--------------------
Rank-1 accuracy of the compatibility measure on this puzzle is 46.6% (see
:mod:`src.edge_compatibility`), so a single greedy pass in raster order fails:
one early mistake propagates along the whole row.  The search is therefore
structured to spend its certainty first and its guesses last.

1. **Infer the grid.**  The number of pieces gives the candidate factor pairs;
   the number of pieces with exactly one flat side picks between them, because
   a rows x cols rectangle has exactly ``2 * (rows + cols - 4)`` such pieces.
2. **Solve the border ring first.**  Border pieces are the most constrained
   objects in the puzzle: a corner piece has two adjacent flat sides and so has
   exactly one admissible rotation in each corner cell, and an edge piece has
   exactly one admissible rotation in each edge cell.  The ring is walked
   clockwise from the top-left corner as a beam search, every step scoring the
   new cell against its already-placed neighbours.
3. **Grow inwards.**  Interior cells are then filled in order of how many
   already-placed neighbours they have, most-constrained first, again as a beam
   search.  Filling a cell that already has two placed neighbours is a far
   better-conditioned decision than filling one that has a single neighbour,
   which is exactly what raster order gets wrong.

Beam retention and backtracking
-------------------------------
Every stage keeps the ``beam_width`` cheapest partial arrangements rather than a
single one, and each retained arrangement contributes its ``per_state`` cheapest
extensions.  A wrong placement is therefore not fatal: the correct alternative
survives in a sibling beam entry and overtakes the mistake as soon as the
mistake's later seams start costing more.  This is backtracking in the sense the
brief asks for -- the search is never committed to a decision it has not yet
paid for -- while staying a bounded, non-exponential search.

The width matters, and it was measured rather than guessed.  On the photographed
puzzle the objective the search reaches falls monotonically as the beam widens
-- 1735 at width 3000, 1713 at 10000, 1688 at 30000, 1671 at 60000 -- so the
search is genuinely finding better arrangements, not merely spending time.  What
it is *not* doing is finding more correct ones, for the reason set out next.

What actually limits this puzzle
--------------------------------
The true arrangement scores **1943** under this objective, which is worse than
every arrangement the search finds at width 3000 or above.  The search is
therefore not the limitation: it is already returning arrangements the
compatibility measure prefers to the correct one.  Position accuracy accordingly
moves around between 54% and 69% as the beam widens instead of improving with
it.  Closing that gap needs a better measure, not a bigger search; the honest
statement of the result is that a blind reconstruction of this particular puzzle
does not reach 100%, and the evidence for why is reproducible with
``python3 -m scripts.evaluate_solver``.

Tie-breaking rule
-----------------
Within a cell, candidates are ordered by ``(cost, piece index, rotation)``
ascending.  Partial arrangements are ordered by cost, and equal costs are broken
by generation order, which is itself fully determined: parent arrangements are
expanded in their own retained order, and each parent contributes its candidates
in the order just stated.  The result therefore does not depend on dictionary
iteration order, on the order the pieces were supplied, or on anything else
incidental -- a property the test suite asserts directly.

Flat sides are a cost, not a gate
---------------------------------
Which sides of a piece are straight border edges cannot be decided piece by
piece on this puzzle.  Measured over all 140 sides the flat/curved decision
overlaps for three pieces: piece 2 is occluded by a spring clamp in all fifty
photographs, and pieces 5 and 29 cannot be told apart by any profile statistic
tried -- relief, RMS, span, 90th percentile and several combinations all leave
at least one of them misclassified.  Hard-classifying each side therefore leaves
thirteen edge pieces for sixteen edge cells, and the search dead-ends before it
starts.

So the flat pattern enters as a *cost* instead.  A cell dictates which of a
piece's sides would face outwards, and putting a piece there costs, for each of
those sides, however far its relief is from straight, plus, for each inward
side, however far its relief falls short of a real tab or blank.  A piece whose
silhouette is unambiguous is effectively barred from the wrong kind of cell,
while an ambiguous one can be overruled by colour evidence.  The counting
constraint that a hard classifier was needed for -- exactly four corners and
``2 * (rows + cols - 4)`` edges -- comes free, because the search places each
piece in exactly one cell.

The tab/blank gate stays hard, but is applied by the *sign* of a side's relief
rather than by its flat/curved label, so it never depends on the decision that
was just shown to be unreliable.  Only two inward-facing sides are ever
compared, and their signs are unambiguous: curved sides displace 0.12 to 0.27
body units, far from zero.

Dead ends
---------
A cell can still run out of candidates, because the tab/blank gate is hard and
piece 2's silhouette is genuinely damaged.  When that happens the search drops
the gate for that cell and scores every remaining pair, with a penalty large
enough that a gate-breaking placement never outranks a legal one.  If even that
leaves nothing the cell is left empty and the search continues, so an incomplete
arrangement is still returned, scored, and reported as incomplete -- the brief
requires the best arrangement found to be returned either way.
"""
from dataclasses import dataclass, field

import numpy as np

from .edge_compatibility import INCOMPATIBLE, compatibility_matrices
from .piece_geometry import BODY, CANVAS, rotate_piece, side_relief

__all__ = ["Placement", "Reconstruction", "infer_grid", "infer_layout",
           "build_variants", "side_reliefs", "cost_tensors", "flat_costs",
           "cell_order", "build_objective", "arrangement_cost",
           "arrangement_report", "solve", "render_grid", "FLAT_THRESHOLD"]

FLAT_THRESHOLD = 0.09   # body units; the empty band between flat and curved sides


@dataclass(frozen=True)
class Placement:
    """One piece placed in one cell at one rotation."""
    index: int        # position of the piece in the supplied list
    piece_id: int     # the piece's own identifier, carried through for reporting
    rotation: int     # counter-clockwise quarter turns applied to the piece


@dataclass
class Reconstruction:
    """The arrangement the search settled on, with its blind quality figures."""
    grid: tuple
    placements: tuple           # one Placement or None per cell, in cell order
    cost: float
    quality: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def complete(self):
        return all(p is not None for p in self.placements)

    def piece_ids(self):
        return [None if p is None else p.piece_id for p in self.placements]

    def rotations(self):
        return [None if p is None else p.rotation for p in self.placements]


# ---------------------------------------------------------------- grid and setup

def infer_grid(pieces):
    """Infer ``(rows, cols)`` from the pieces alone, without being told.

    Every factor pair of the piece count is a candidate rectangle.  A
    ``rows x cols`` rectangle has four corner pieces (two flat sides) and
    ``2 * (rows + cols - 4)`` edge pieces (one flat side), so the observed flat
    counts select among the candidates.  ``rows <= cols`` is returned, which
    fixes the otherwise free choice between a shape and its transpose.
    """
    count = len(pieces)
    edges = sum(1 for piece in pieces if piece.flat_count == 1)
    candidates = [(r, count // r) for r in range(1, int(count ** 0.5) + 1)
                  if count % r == 0]
    if not candidates:
        return 1, count

    def mismatch(shape):
        rows, cols = shape
        predicted = 2 * (rows + cols - 4) if rows > 1 and cols > 1 else count - 2
        return abs(predicted - edges), abs(rows - cols)

    return min(candidates, key=mismatch)


def _role_costs(piece, threshold=FLAT_THRESHOLD):
    """What it would cost to call this piece a corner, an edge or an interior piece.

    Calling a side flat costs its measured relief, because a genuinely straight
    border edge has a relief of nearly zero.  Calling a side curved costs
    whatever the relief falls short of ``threshold``, because a genuine tab or
    blank displaces well beyond it.  Both terms are zero for a side that is
    unambiguous, so the totals only ever reflect the sides the silhouette is
    equivocal about.
    """
    reliefs = [side_relief(side.profile) for side in piece.sides]
    flat = [abs(r) for r in reliefs]
    curved = [max(0.0, threshold - abs(r)) for r in reliefs]
    total_curved = sum(curved)

    interior = (total_curved, ())
    edge = min(((flat[j] + total_curved - curved[j], (j,)) for j in range(4)))
    corner = min(((flat[j] + flat[(j + 1) % 4]
                   + total_curved - curved[j] - curved[(j + 1) % 4],
                   (j, (j + 1) % 4)) for j in range(4)))
    return {'interior': interior, 'edge': edge, 'corner': corner}, reliefs


def _assign_roles(pieces, rows, cols, threshold=FLAT_THRESHOLD):
    """Choose which pieces are corners, edges and interior pieces, globally.

    Classifying each side on its own threshold does not work on this puzzle:
    three of the thirty-five pieces come out with the wrong number of flat
    sides (piece 2 is occluded by a clamp in every photograph, and pieces 5 and
    33 each have one side whose relief lands in the ambiguous band), which
    leaves fewer edge pieces than there are edge cells and dead-ends the search
    before it starts.

    A rectangle fixes the counts exactly -- four corners, ``2 * (rows + cols -
    4)`` edges, the rest interior -- so the labelling is chosen as the
    assignment of pieces to those roles with the lowest total cost.  The
    optimum is found exactly by a small dynamic program over
    ``(corners used, edges used)``, which needs no external solver.  Only the
    piece count and the assumption that the puzzle is a rectangle go into this;
    no ground truth does.

    Returns ``(total_cost, labels)`` where ``labels[i]`` is the tuple of side
    indices that piece ``i`` should treat as flat.
    """
    count = len(pieces)
    corner_slots = 4
    edge_slots = 2 * (rows + cols - 4)
    if rows < 2 or cols < 2 or edge_slots < 0 or corner_slots + edge_slots > count:
        return float('inf'), [()] * count

    costs = [_role_costs(piece, threshold)[0] for piece in pieces]
    infinity = float('inf')

    # table[c][e] = cheapest cost of having used c corners and e edges so far
    table = [[infinity] * (edge_slots + 1) for _ in range(corner_slots + 1)]
    table[0][0] = 0.0
    choice = {}
    for index in range(count):
        nxt = [[infinity] * (edge_slots + 1) for _ in range(corner_slots + 1)]
        for c in range(corner_slots + 1):
            for e in range(edge_slots + 1):
                base = table[c][e]
                if base == infinity:
                    continue
                for role, (delta, _sides), dc, de in (
                        ('interior', costs[index]['interior'], 0, 0),
                        ('edge', costs[index]['edge'], 0, 1),
                        ('corner', costs[index]['corner'], 1, 0)):
                    nc, ne = c + dc, e + de
                    if nc > corner_slots or ne > edge_slots:
                        continue
                    value = base + delta
                    if value < nxt[nc][ne] - 1e-12:
                        nxt[nc][ne] = value
                        choice[(index, nc, ne)] = (role, c, e)
        table = nxt

    total = table[corner_slots][edge_slots]
    if total == infinity:
        return infinity, [()] * count

    labels = [()] * count
    c, e = corner_slots, edge_slots
    for index in range(count - 1, -1, -1):
        role, c, e = choice[(index, c, e)]
        labels[index] = costs[index][role][1]
    return total, labels


def infer_layout(pieces, threshold=FLAT_THRESHOLD):
    """Infer the grid shape and every piece's flat sides together.

    The two questions are not separable -- which rectangle the pieces form
    decides how many border pieces there must be, and the border pieces are
    identified by their flat sides -- so every factor pair of the piece count is
    scored by the total cost of its best labelling and the cheapest wins.
    ``rows <= cols`` is returned, which fixes the free choice between a shape
    and its transpose.

    Returns ``(rows, cols), labels``.
    """
    count = len(pieces)
    shapes = [(r, count // r) for r in range(2, int(count ** 0.5) + 1)
              if count % r == 0]
    if not shapes:
        return (1, count), [()] * count
    scored = [( _assign_roles(pieces, r, c, threshold), (r, c)) for r, c in shapes]
    (total, labels), shape = min(scored, key=lambda item: (item[0][0], item[1]))
    return shape, labels


def build_variants(pieces):
    """``variants[i][k]`` is piece ``i`` turned by ``k`` quarter turns."""
    return [[rotate_piece(piece, k) for k in range(4)] for piece in pieces]


def _required_flats(row, col, rows, cols):
    """Which of ``(top, right, bottom, left)`` face outwards in this cell."""
    return (row == 0, col == cols - 1, row == rows - 1, col == 0)


def side_reliefs(variants):
    """``relief[i, k, s]``: signed relief of side ``s`` of piece ``i`` turned ``k``."""
    count = len(variants)
    relief = np.zeros((count, 4, 4))
    for i in range(count):
        for k in range(4):
            for j, side in enumerate(variants[i][k].sides):
                relief[i, k, j] = side_relief(side.profile)
    return relief


def cost_tensors(variants, relief=None):
    """Gated and ungated horizontal/vertical join costs.

    The ungated tensors score every pair whether or not it can physically mate;
    the gated ones are the same numbers with impossible pairs set to
    :data:`~src.edge_compatibility.INCOMPATIBLE`.  A pair may mate when one side
    is a tab and the other a blank, which is read off the *sign* of each side's
    relief -- see the module docstring on why the flat/curved label is
    deliberately not consulted here.
    """
    raw_h, raw_v = compatibility_matrices(variants, gate=False)
    if relief is None:
        relief = side_reliefs(variants)
    count = len(variants)
    sign = np.where(relief > 0, 1, -1)

    mates_h = sign[:, :, 1][:, :, None, None] * sign[:, :, 3][None, None, :, :] == -1
    mates_v = sign[:, :, 2][:, :, None, None] * sign[:, :, 0][None, None, :, :] == -1
    self_pair = np.zeros((count, 4, count, 4), dtype=bool)
    self_pair[np.arange(count), :, np.arange(count), :] = True

    gated_h = np.where(mates_h & ~self_pair, raw_h, INCOMPATIBLE)
    gated_v = np.where(mates_v & ~self_pair, raw_v, INCOMPATIBLE)
    raw_h = np.where(self_pair, INCOMPATIBLE, raw_h)
    raw_v = np.where(self_pair, INCOMPATIBLE, raw_v)
    return gated_h, gated_v, raw_h, raw_v


def flat_costs(relief, rows, cols, weight, threshold=FLAT_THRESHOLD):
    """``cost[pattern][i, k]``: what the silhouette says about a piece in a cell.

    ``pattern`` is the tuple of four booleans saying which sides face outwards,
    so there are at most nine distinct patterns in any rectangle.  For a side
    that must face outwards the cost is its relief -- zero for a genuinely
    straight edge.  For a side that must face inwards the cost is however far
    its relief falls short of ``threshold`` -- zero for a genuine tab or blank.
    ``weight`` converts body units into the units of the seam costs.
    """
    magnitude = np.abs(relief)
    outward = magnitude
    inward = np.maximum(0.0, threshold - magnitude)
    costs = {}
    for row in (0, 1, rows - 1):
        for col in (0, 1, cols - 1):
            pattern = _required_flats(row, col, rows, cols)
            if pattern in costs:
                continue
            total = np.zeros(relief.shape[:2])
            for side, is_outward in enumerate(pattern):
                total += outward[:, :, side] if is_outward else inward[:, :, side]
            costs[pattern] = weight * total
    return costs


def build_objective(variants, rows, cols, flat_weight=1.0, relief=None,
                    tensors=None, gate_penalty_factor=10.0):
    """Everything the search scores an arrangement with, in one place.

    Both the flat-side weight and the dead-end penalty are expressed as
    multiples of the puzzle's own median seam cost, so neither depends on the
    arbitrary units of the compatibility measure.

    ``gate_penalty_factor`` needs care.  It has to be big enough that a
    gate-breaking placement never beats a legal one, and small enough that it
    does not dominate the other 57 seams: this puzzle's true arrangement
    contains one seam that the gate rejects, because piece 2's silhouette is
    damaged, so a penalty of a thousand typical seams would make the correct
    answer cost twenty-five times the objective of a wrong one.  Ten is the
    default.

    Returns ``(tensors, label_costs, patterns, gate_penalty, typical)``, where
    ``tensors`` is ``(gated_h, gated_v, raw_h, raw_v, soft_h, soft_v)``.  The
    soft tensors are what the search actually optimises: a pair the gate rejects
    is not removed but charged ``gate_penalty`` on top of its raw cost, so the
    true arrangement stays reachable even though one of its seams is
    gate-breaking.  The gated tensors are kept for scoring and diagnosis.
    """
    if relief is None:
        relief = side_reliefs(variants)
    if tensors is None:
        tensors = cost_tensors(variants, relief)
    gated_h, gated_v = tensors[0], tensors[1]
    finite = np.concatenate([gated_h[np.isfinite(gated_h)],
                             gated_v[np.isfinite(gated_v)]])
    typical = max(float(np.percentile(finite, 50)) if finite.size else 1.0, 1e-9)
    label_costs = flat_costs(relief, rows, cols,
                             weight=flat_weight * typical / FLAT_THRESHOLD)
    patterns = {cell: _required_flats(cell // cols, cell % cols, rows, cols)
                for cell in range(rows * cols)}
    gate_penalty = gate_penalty_factor * typical
    gated_h, gated_v, raw_h, raw_v = tensors[:4]
    soft_h = np.where(np.isfinite(gated_h), gated_h, raw_h + gate_penalty)
    soft_v = np.where(np.isfinite(gated_v), gated_v, raw_v + gate_penalty)
    tensors = (gated_h, gated_v, raw_h, raw_v, soft_h, soft_v)
    return tensors, label_costs, patterns, gate_penalty, typical


def arrangement_report(cells, tensors, label_costs, patterns, rows, cols,
                       gate_penalty=0.0):
    """Score a complete or partial arrangement under the search's own objective.

    ``cells[c]`` is ``(piece index, rotation)`` or ``None``.  The total is the
    sum of every realised seam cost plus every placed piece's flat-side cost,
    which is exactly what :func:`solve` accumulates as it fills cells, so the
    two are directly comparable.  The parts are returned separately as well,
    because a single gate-breaking seam can otherwise hide the fit of the other
    fifty-seven.
    """
    gated_h, gated_v = tensors[0], tensors[1]
    flat_total, seam_total, violations, seams = 0.0, 0.0, 0, []
    for cell, entry in enumerate(cells):
        if entry is None:
            continue
        index, rotation = entry
        flat_total += float(label_costs[patterns[cell]][index, rotation])
        row, col = divmod(cell, cols)
        for tensor, other in ((gated_h, cell + 1 if col < cols - 1 else None),
                              (gated_v, cell + cols if row < rows - 1 else None)):
            if other is None or cells[other] is None:
                continue
            j, b = cells[other]
            value = float(tensor[index, rotation, j, b])
            if np.isfinite(value):
                seam_total += value
                seams.append(value)
            else:
                violations += 1
                seam_total += gate_penalty
    return {
        'total': flat_total + seam_total,
        'flat_cost': flat_total,
        'seam_cost': seam_total,
        'gate_violations': violations,
        'seams_scored': len(seams),
        'mean_legal_seam': float(np.mean(seams)) if seams else float('nan'),
    }


def arrangement_cost(cells, tensors, label_costs, patterns, rows, cols,
                     gate_penalty=0.0):
    """The scalar objective; see :func:`arrangement_report` for the breakdown."""
    return arrangement_report(cells, tensors, label_costs, patterns, rows, cols,
                              gate_penalty)['total']


# ---------------------------------------------------------------- search machinery

def cell_order(rows, cols, strategy='border'):
    """The sequence of cells the search fills, under the named strategy.

    ``border``
        The perimeter clockwise from the top-left corner, then inwards.  Border
        pieces are the most constrained, but each step of the ring is decided on
        a single seam.
    ``raster``
        Left to right, top to bottom.  Only the first row and the first column
        are decided on one seam; every other cell is decided on two.
    ``growth``
        Purely most-constrained-first from the top-left cell.
    """
    if strategy == 'raster':
        return list(range(rows * cols))
    if strategy == 'growth':
        return [0] + _growth_order(rows, cols, [0])
    if strategy == 'border':
        ring = _ring_order(rows, cols)
        return ring + _growth_order(rows, cols, ring)
    raise ValueError('unknown cell order strategy: %r' % (strategy,))


def _ring_order(rows, cols):
    """Perimeter cells, clockwise from the top-left corner.

    Consecutive cells in this walk are always physically adjacent, so every
    step of the border beam scores against a neighbour that is already placed.
    """
    if rows == 1 or cols == 1:
        return list(range(rows * cols))
    top = [c for c in range(cols)]
    right = [r * cols + (cols - 1) for r in range(1, rows)]
    bottom = [(rows - 1) * cols + c for c in range(cols - 2, -1, -1)]
    left = [r * cols for r in range(rows - 2, 0, -1)]
    return top + right + bottom + left


def _growth_order(rows, cols, already):
    """Remaining cells, most-constrained first.

    At each step the next cell is the one with the most already-chosen
    neighbours, ties broken by cell index.  Starting from a placed border this
    spirals inwards, so no cell is ever decided on the evidence of a single
    seam while a two-seam decision was available.
    """
    placed = set(already)
    remaining = [c for c in range(rows * cols) if c not in placed]
    order = []

    def around(cell):
        row, col = divmod(cell, cols)
        out = []
        if col > 0: out.append(cell - 1)
        if col < cols - 1: out.append(cell + 1)
        if row > 0: out.append(cell - cols)
        if row < rows - 1: out.append(cell + cols)
        return out

    while remaining:
        best = min(remaining,
                   key=lambda c: (-sum(n in placed for n in around(c)), c))
        order.append(best)
        placed.add(best)
        remaining.remove(best)
    return order


def _neighbour_costs(cells, cell, horizontal, vertical, rows, cols):
    """Seam cost of every ``(piece, rotation)`` in ``cell``, as a ``(count, 4)`` array.

    Only already-placed neighbours contribute, so the same routine serves the
    border walk (one neighbour) and the inward growth (two or more).
    """
    row, col = divmod(cell, cols)
    total = np.zeros(horizontal.shape[:2])
    if col > 0 and cells[cell - 1] is not None:
        j, b = cells[cell - 1]
        total = total + horizontal[j, b]
    if col < cols - 1 and cells[cell + 1] is not None:
        j, b = cells[cell + 1]
        total = total + horizontal[:, :, j, b]
    if row > 0 and cells[cell - cols] is not None:
        j, b = cells[cell - cols]
        total = total + vertical[j, b]
    if row < rows - 1 and cells[cell + cols] is not None:
        j, b = cells[cell + cols]
        total = total + vertical[:, :, j, b]
    return total


def _used_indices(mask):
    """The piece indices set in a bitmask.

    Arrangements track which pieces they have consumed as an integer bitmask
    rather than a set: the search creates one successor per candidate per cell,
    and ``mask | (1 << index)`` is a constant-time operation where rebuilding a
    frozenset is not.
    """
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _best_options(delta, used, per_state):
    """The ``per_state`` cheapest ``(cost, piece, rotation)`` still available.

    Ties are resolved by piece index then rotation, because the flattened index
    is ``piece * 4 + rotation`` and the sort is stable.
    """
    if used:
        delta = delta.copy()
        delta[_used_indices(used)] = INCOMPATIBLE
    flat = delta.ravel()
    finite = np.isfinite(flat)
    count = int(finite.sum())
    if not count:
        return []
    take = min(per_state, count)
    order = np.argpartition(flat, take - 1)[:take] if take < flat.size else np.arange(flat.size)
    order = order[np.argsort(flat[order], kind='stable')]
    return [(float(flat[k]), int(k) // 4, int(k) % 4) for k in order
            if np.isfinite(flat[k])]


def _retain(candidates, beam_width):
    """Keep the cheapest ``beam_width`` arrangements, deterministically.

    A stable sort on cost alone is enough: candidates are generated in a fixed
    order, so equal costs keep that order rather than falling back on an
    arbitrary comparison of whole arrangements.
    """
    if len(candidates) <= beam_width:
        costs = np.fromiter((c[0] for c in candidates), float, len(candidates))
        return [candidates[i] for i in np.argsort(costs, kind='stable')]
    costs = np.fromiter((c[0] for c in candidates), float, len(candidates))
    keep = np.argpartition(costs, beam_width - 1)[:beam_width]
    keep = keep[np.argsort(costs[keep], kind='stable')]
    return [candidates[i] for i in keep]


def _fill(states, order, tensors, label_costs, patterns, rows, cols,
          beam_width, per_state, gate_penalty, diagnostics):
    """Beam-fill the cells in ``order`` against the soft objective."""
    soft_h, soft_v = tensors[4], tensors[5]

    for cell in order:
        label = label_costs[patterns[cell]]
        nxt = []
        for cost, cells, used in states:
            delta = _neighbour_costs(cells, cell, soft_h, soft_v, rows, cols) + label
            options = _best_options(delta, used, per_state)

            if not options:
                # Nothing at all fits: leave the cell empty and carry on, so the
                # best incomplete arrangement is still available to return.
                diagnostics['unfilled_cells'] += 1
                nxt.append((cost + gate_penalty, cells, used))
                continue

            for delta_cost, index, rotation in options:
                filled = list(cells)
                filled[cell] = (index, rotation)
                nxt.append((cost + delta_cost, tuple(filled), used | (1 << index)))

        if not nxt:
            break
        states = _retain(nxt, beam_width)
    return states


# ---------------------------------------------------------------- quality scoring

def _quality(cells, tensors, rows, cols):
    """Confidence figures computable without any ground truth.

    ``best_partner_fraction`` asks, for each directed seam actually realised,
    whether the neighbour placed there is the cheapest partner that side had
    anywhere in the puzzle.  ``mutual_best_fraction`` is the stricter "best
    buddy" version: both sides of the seam must prefer each other over every
    alternative.  Best-buddy agreement is the standard blind proxy for jigsaw
    reconstruction quality, and on this puzzle it is an informative one: of the
    21 best-buddy pairs the measure finds, 17 are true neighbours.
    """
    gated_h, gated_v = tensors[0], tensors[1]
    seams, directed_best, mutual, gate_breaks = [], 0, 0, 0
    directed_total = 0

    def check(tensor, left, right):
        nonlocal directed_best, mutual, gate_breaks, directed_total
        (i, a), (j, b) = left, right
        value = tensor[i, a, j, b]
        seams.append(float(value))
        directed_total += 2
        if not np.isfinite(value):
            gate_breaks += 1
            return
        forward = bool(np.isclose(value, np.min(tensor[i, a])))
        backward = bool(np.isclose(value, np.min(tensor[:, :, j, b])))
        directed_best += int(forward) + int(backward)
        mutual += int(forward and backward)

    pairs = 0
    for cell in range(rows * cols):
        row, col = divmod(cell, cols)
        if col < cols - 1 and cells[cell] and cells[cell + 1]:
            check(gated_h, cells[cell], cells[cell + 1]); pairs += 1
        if row < rows - 1 and cells[cell] and cells[cell + cols]:
            check(gated_v, cells[cell], cells[cell + cols]); pairs += 1

    finite = [v for v in seams if np.isfinite(v)]
    expected = (rows - 1) * cols + rows * (cols - 1)
    return {
        'seams_realised': pairs,
        'seams_expected': expected,
        'seams_violating_the_shape_gate': gate_breaks,
        'mean_seam_cost': float(np.mean(finite)) if finite else None,
        'median_seam_cost': float(np.median(finite)) if finite else None,
        'best_partner_fraction': directed_best / directed_total if directed_total else 0.0,
        'mutual_best_fraction': mutual / pairs if pairs else 0.0,
        'quality_score': mutual / expected if expected else 0.0,
    }


# ---------------------------------------------------------------- entry point

def solve(pieces, rows=None, cols=None, beam_width=10000, per_state=16,
          flat_weight=1.0, order='raster', gate_penalty_factor=10.0,
          variants=None, tensors=None, relief=None, progress=None):
    """Reconstruct a scrambled puzzle blind.

    Parameters
    ----------
    pieces : sequence of CanonicalPiece
        The scrambled pieces, in any order.  Their ``id`` is carried through to
        the result for reporting but is never used to decide a placement.
    rows, cols : int, optional
        The grid shape.  Inferred from the pieces when not given.
    beam_width : int
        How many partial arrangements each stage retains.  The default costs
        about thirty seconds on the 5x7 puzzle; see the module docstring for
        what widening it does and does not buy.
    per_state : int
        How many extensions each retained arrangement contributes per cell.
    variants, tensors, relief : optional
        Pre-computed descriptions and join costs.  Supplying them lets a caller
        try several search settings without paying for the descriptions twice;
        they are derived from ``pieces`` when omitted.
    flat_weight : float
        How heavily the silhouette's opinion about which sides are straight
        counts against the colour evidence.  It is expressed as a multiple of a
        typical seam cost per body unit of relief, so it is scale free.

    Returns
    -------
    Reconstruction
        The cheapest arrangement found, complete or not, with its blind quality
        figures and search diagnostics.
    """
    if rows is None or cols is None:
        (rows, cols), _ = infer_layout(pieces)
    count = len(pieces)
    if rows * cols != count:
        raise ValueError('grid %dx%d does not hold %d pieces' % (rows, cols, count))

    if variants is None:
        variants = build_variants(pieces)
    tensors, label_costs, patterns, gate_penalty, typical = build_objective(
        variants, rows, cols, flat_weight, relief, tensors, gate_penalty_factor)
    diagnostics = {'unfilled_cells': 0,
                   'grid_inferred': (rows, cols), 'beam_width': beam_width,
                   'per_state': per_state, 'flat_weight': flat_weight,
                   'cell_order': order, 'typical_seam_cost': typical,
                   'gate_penalty_factor': gate_penalty_factor}

    empty = (0.0, tuple([None] * (rows * cols)), 0)
    order = cell_order(rows, cols, order)
    if progress:
        progress('filling %d cells' % len(order))
    states = _fill([empty], order, tensors, label_costs, patterns, rows, cols,
                   beam_width, per_state, gate_penalty, diagnostics)

    if not states:
        return Reconstruction((rows, cols), tuple([None] * count),
                              float('inf'), {}, diagnostics)

    # Prefer the most complete arrangement, then the cheapest, then the
    # deterministic tie-break.
    best = min(states, key=lambda s: (-sum(c is not None for c in s[1]), s[0]))
    cost, cells, _ = best
    placements = tuple(None if c is None else Placement(c[0], pieces[c[0]].id, c[1])
                       for c in cells)
    quality = _quality(cells, tensors, rows, cols)
    return Reconstruction((rows, cols), placements, float(cost), quality, diagnostics)


# ---------------------------------------------------------------- rendering

def render_grid(variants, reconstruction, margin=60, background=(245, 245, 245),
                body=BODY, canvas=CANVAS):
    """Composite a reconstruction so each piece's tabs enter its neighbour's blanks.

    Pieces are laid one body-square apart, which is exactly the pitch at which
    the canonical representation makes interlocking pieces meet.
    """
    rows, cols = reconstruction.grid
    pad = (canvas - body) // 2
    height, width = rows * body + 2 * margin, cols * body + 2 * margin
    image = np.full((height, width, 3), background, np.uint8)
    for cell, placement in enumerate(reconstruction.placements):
        if placement is None:
            continue
        piece = variants[placement.index][placement.rotation]
        row, col = divmod(cell, cols)
        y0, x0 = margin + row * body - pad, margin + col * body - pad
        ys, xs = np.nonzero(piece.mask > 0)
        ty, tx = y0 + ys, x0 + xs
        inside = (ty >= 0) & (ty < height) & (tx >= 0) & (tx < width)
        image[ty[inside], tx[inside]] = piece.image[ys[inside], xs[inside]]
    return image
