"""Compatibility between two candidate piece sides.

The score answers one question: if these two sides were pressed together, how
badly would the join fit?  Lower is better.

The exact formula
-----------------
Let ``a`` and ``b`` be the two sides.  ``b`` is traversed in the opposite
physical direction to ``a``, so every descriptor of ``b`` is reversed before
comparison (see :mod:`src.piece_geometry`).  Write ``a0`` for the colour strip
sampled just under ``a`` and ``a1`` for the strip sampled one step deeper, and
likewise ``b0``, ``b1`` for the reversed strips of ``b``.  Then

    D(a, b) = INCOMPATIBLE                     if not complementary(a, b)
    D(a, b) = M(a -> b) + M(b -> a)            otherwise

    M(x -> y) = mean_i (r_i - mu)' inv(S) (r_i - mu)
        g_i   = x0_i - x1_i          (x's own inward-to-outward colour gradient)
        mu    = mean_i g_i
        S     = cov(g) + 1e-4 * I    (3x3, over the colour channels)
        r_i   = y0_i - x0_i          (the colour change actually observed
                                      across the seam at sample i)

This is *Mahalanobis gradient compatibility* (MGC).  It predicts the colour
just beyond a side by extrapolating that side's own gradient, and penalises the
discrepancy under the metric of that gradient's covariance, so a seam that
continues an existing colour trend is cheap while an abrupt unexplained change
is expensive.  It is used because it is invariant to the constant brightness
offsets that dominate this puzzle's large, low-texture white regions.

The weights
-----------
    w_colour = 1.0      MGC is the entire numeric score
    w_shape  = 0.0      shape is a hard gate only, never a weighted term

Shape is *not* mixed into the number.  This puzzle is die cut: measured over
all 140 sides every tab peaks at about +0.27 body units and every blank at
about -0.25, so the silhouette says reliably *whether* two sides may mate but
almost nothing about *which* tab belongs in *which* blank.  Adding shape to the
score was measured to make matching monotonically worse -- rank-1 accuracy over
the 116 directed true-neighbour pairs falls from 46.6% (MGC alone) to 44.0% at
5% shape weight, 43.1% at 10% and 34.5% at 20%; the previous default of 0.6
shape / 0.4 colour scored 7.8%.  Reproduce with ``python3 -m
scripts.experiment_mgc``.  Shape therefore appears only in
:func:`complementary`, which rejects tab-against-tab, blank-against-blank and
anything involving a flat border edge.

:func:`shape_dissimilarity` and :func:`colour_dissimilarity` are retained for
diagnostics and for the baseline comparisons in ``scripts/evaluate_matcher.py``.
"""
import numpy as np

from .piece_geometry import SideType

__all__ = ["INCOMPATIBLE", "REGULARISER", "complementary", "colour_bands",
           "shape_dissimilarity", "colour_dissimilarity", "mgc_dissimilarity",
           "estimate_scales", "compatibility", "compatibility_matrices"]

INCOMPATIBLE = float("inf")
REGULARISER = 1e-4     # keeps the 3x3 channel covariance invertible
CHANNELS = 3


def complementary(first, second):
    """True when two side types can physically mate."""
    return {first, second} == {SideType.TAB, SideType.BLANK}


def colour_bands(side, reverse=False):
    """``(depth, sample, channel)`` view of a side's interior colour strip.

    The number of depths is read from the array itself, so this stays correct
    whatever ``depths`` was passed to :func:`~src.piece_geometry.describe_piece`.
    """
    colors = np.asarray(side.colors, dtype=np.float64)
    samples, width = colors.shape
    bands = colors.reshape(samples, width // CHANNELS, CHANNELS).transpose(1, 0, 2)
    return bands[:, ::-1, :] if reverse else bands


def shape_dissimilarity(a, b):
    """Mean squared residual of the mating condition ``a + reverse(b) = 0``.

    Diagnostic only -- see the module docstring for why it is not scored.
    """
    return float(np.mean((a.profile + b.profile[::-1]) ** 2))


def colour_dissimilarity(a, b):
    """Mean squared difference between the two interior colour strips.

    The plain sum-of-squared-differences baseline that MGC replaced.
    """
    other = b.colors[::-1]
    if a.colors.shape != other.shape:
        return 1.0
    return float(np.mean((a.colors - other) ** 2))


def _directional_mgc(near_x, next_x, near_y, regulariser=REGULARISER):
    """``M(x -> y)`` of the module docstring: cost of continuing x's gradient."""
    gradient = near_x - next_x
    mu = gradient.mean(axis=0)
    covariance = np.cov(gradient, rowvar=False) + regulariser * np.eye(CHANNELS)
    residual = (near_y - near_x) - mu
    inverse = np.linalg.inv(covariance)
    return float(np.einsum('ij,jk,ik->i', residual, inverse, residual).mean())


def mgc_dissimilarity(a, b, regulariser=REGULARISER):
    """Symmetric Mahalanobis gradient compatibility of two sides.

    ``b`` is reversed internally; callers pass both sides in their own natural
    traversal direction.  No physical gate is applied here -- use
    :func:`compatibility` for the gated score.
    """
    x = colour_bands(a)
    y = colour_bands(b, reverse=True)
    if x.shape != y.shape or x.shape[0] < 2:
        return INCOMPATIBLE
    return (_directional_mgc(x[0], x[1], y[0], regulariser)
            + _directional_mgc(y[0], y[1], x[0], regulariser))


def estimate_scales(pieces, sample_limit=4000, seed=480):
    """Typical shape and colour dissimilarity for a puzzle, for diagnostics.

    The scales are the medians over a random sample of *physically plausible*
    pairs (tab against blank).  They are no longer needed by
    :func:`compatibility`, which is a single unnormalised term, but the
    baseline comparisons in ``scripts/evaluate_matcher.py`` still use them to
    put shape and colour on a common footing.
    """
    rng = np.random.default_rng(seed)
    sides = [side for piece in pieces for side in piece.sides
             if side.kind is not SideType.FLAT]
    shape_values, colour_values = [], []
    if len(sides) < 2:
        return 1.0, 1.0
    for _ in range(sample_limit):
        i, j = rng.integers(0, len(sides), 2)
        if i == j or not complementary(sides[i].kind, sides[j].kind):
            continue
        shape_values.append(shape_dissimilarity(sides[i], sides[j]))
        colour_values.append(colour_dissimilarity(sides[i], sides[j]))
    shape_scale = float(np.median(shape_values)) if shape_values else 1.0
    colour_scale = float(np.median(colour_values)) if colour_values else 1.0
    return max(shape_scale, 1e-9), max(colour_scale, 1e-9)


def compatibility(a, b, gate=True):
    """Dissimilarity of the join between side ``a`` and side ``b``.

    ``b`` is reversed internally.  Returns :data:`INCOMPATIBLE` for pairs that
    cannot physically mate.  Pass ``gate=False`` to score a pair anyway, which
    the solver does when a cell has no gated candidate left -- piece 2 is
    occluded in every photograph and two of its side labels are known to be
    unreliable, so a hard gate can otherwise dead-end the search.
    """
    if gate and not complementary(a.kind, b.kind):
        return INCOMPATIBLE
    return mgc_dissimilarity(a, b)


def compatibility_matrices(variants, gate=True):
    """Pre-compute every horizontal and vertical join cost.

    Parameters
    ----------
    variants : list of list of CanonicalPiece
        ``variants[i][k]`` is piece ``i`` rotated by ``k`` quarter turns.
    gate : bool
        When False the tab/blank gate is skipped and every pair is scored, so
        the caller can fall back to a soft comparison.

    Returns
    -------
    (horizontal, vertical)
        ``horizontal[i, a, j, b]`` is the cost of placing piece ``j`` in
        rotation ``b`` immediately to the right of piece ``i`` in rotation
        ``a``; ``vertical`` is the same for ``j`` immediately below ``i``.
        Both are ``INCOMPATIBLE`` where the sides cannot mate, and on the
        diagonal ``i == j``.
    """
    count = len(variants)
    horizontal = np.full((count, 4, count, 4), INCOMPATIBLE)
    vertical = np.full((count, 4, count, 4), INCOMPATIBLE)
    for i in range(count):
        for a in range(4):
            right = variants[i][a].sides[1]
            bottom = variants[i][a].sides[2]
            for j in range(count):
                if i == j:
                    continue
                for b in range(4):
                    horizontal[i, a, j, b] = compatibility(
                        right, variants[j][b].sides[3], gate)
                    vertical[i, a, j, b] = compatibility(
                        bottom, variants[j][b].sides[0], gate)
    return horizontal, vertical
