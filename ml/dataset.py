"""Turn the verified ground truth into labelled side-pair examples.

The learning milestone needs pairs of piece sides labelled "these two are
actually adjacent" or "these two are not".  Phase 0 produced everything needed
to build them: ``data/ground_truth/layout.json`` gives the solved 5x7 layout,
which fixes all 58 internal seams and therefore every true adjacency, and
``data/pieces/piece_NN.png`` gives a clean, occlusion-free image of each piece.

What a training example is
--------------------------
Each side is represented by the *colour strip* beneath it: the artwork sampled
at a fixed number of depths inwards from the boundary, resampled to a fixed
number of points along it.  That is a small RGB image, ``(3, depths, samples)``,
and it is deliberately the same information the classical MGC measure reads --
so a learned score can be compared against it fairly.

Two mating sides run in *opposite* physical directions, so the second strip of
every pair is reversed along its length.  After that reversal, column ``i`` of
strip A meets column ``i`` of strip B at the seam, for positive and negative
pairs alike; the convention carries no signal the model could exploit.

Negatives are drawn only from pairs the tab/blank gate would allow, because
those are the only pairs the solver ever asks about.  Scoring pairs that the
silhouette already rejects would inflate the numbers for free.

The split is by piece, not by seam
----------------------------------
There is one puzzle.  Splitting its 58 seams at random would put the same
physical piece on both sides of the split, and a model can then recognise a
piece it has already seen rather than judge whether two edges continue each
other.  Instead a *contiguous block* of cells is held out; a pair is a training
pair only when both its pieces are outside the block and a validation pair only
when both are inside it, and pairs straddling the boundary are dropped
entirely.  The block is contiguous rather than scattered because seams only
survive the split when both their pieces fall on the same side of it: eight
randomly chosen cells are rarely adjacent and leave almost no validation
positives, whereas an eight-cell block contains ten seams.  This still costs
training data, and is why the numbers in the training report are as small as
they are.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.edge_compatibility import complementary
from src.piece_geometry import describe_piece, rotate_piece

__all__ = ["MissingAdjacencyGroundTruth", "load_adjacency_manifest",
           "split_manifest", "DEPTHS", "SAMPLES", "load_solved_pieces",
           "side_strip", "true_seams", "build_pairs", "NODE_DIM", "EDGE_DIM",
           "side_features", "pair_features", "build_graph"]

DEPTHS = tuple(range(1, 17))   # 16 depths inwards from the boundary, in canonical px
SAMPLES = 96                   # points along each side
LAYOUT = Path("data/ground_truth/layout.json")
PIECES = Path("data/pieces")


class MissingAdjacencyGroundTruth(RuntimeError):
    pass


def load_adjacency_manifest(path):
    """Load an externally supplied adjacency manifest, if one is being used.

    Retained for the case where adjacencies come from a file rather than from
    the derived layout; :func:`true_seams` is what the training script uses.
    """
    p = Path(path)
    if not p.exists():
        raise MissingAdjacencyGroundTruth(
            f"{p} is required: columns puzzle,piece_a,side_a,piece_b,side_b,"
            "relative_rotation,split")
    with p.open() as f:
        rows = list(csv.DictReader(f))
    required = {"puzzle", "piece_a", "side_a", "piece_b", "side_b",
                "relative_rotation", "split"}
    if not rows or not required.issubset(rows[0]):
        raise MissingAdjacencyGroundTruth("invalid adjacency manifest")
    return rows


def split_manifest(rows):
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}


def load_solved_pieces(layout_path=LAYOUT, pieces_dir=PIECES, depths=DEPTHS):
    """Every piece described with deep colour strips and turned into its solved pose.

    Returns ``(pieces, rows, cols)`` with ``pieces[cell]`` the piece occupying
    that cell of the solved grid, so grid adjacency is simply cell adjacency.
    """
    layout = json.loads(Path(layout_path).read_text())
    rows, cols = layout["grid"]
    rotations = layout["solved_rotations_ccw"]
    pieces = []
    for cell in range(rows * cols):
        piece_id = layout["positions_piece_id"][cell]
        rgba = cv2.imread(str(Path(pieces_dir) / ("piece_%02d.png" % piece_id)),
                          cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise MissingAdjacencyGroundTruth("missing piece image for id %d" % piece_id)
        base = describe_piece(piece_id, rgba[:, :, :3], rgba[:, :, 3], depths=depths)
        pieces.append(rotate_piece(base, int(rotations[str(piece_id)])))
    return pieces, rows, cols


def side_strip(side, reverse=False):
    """A side's colour strip as a ``(3, depths, samples)`` float32 array in 0..1."""
    colors = np.asarray(side.colors, dtype=np.float32)      # (samples, 3 * depths)
    samples, width = colors.shape
    strip = colors.reshape(samples, width // 3, 3)          # (samples, depths, 3)
    strip = strip.transpose(2, 1, 0)                        # (3, depths, samples)
    return np.ascontiguousarray(strip[:, :, ::-1] if reverse else strip)


def true_seams(rows, cols):
    """``(cell_a, side_a, cell_b, side_b)`` for every internal seam of the solved grid.

    Side order is top, right, bottom, left, so a piece's right side (1) meets its
    right-hand neighbour's left side (3), and its bottom (2) meets the piece
    below's top (0).
    """
    seams = []
    for cell in range(rows * cols):
        row, col = divmod(cell, cols)
        if col < cols - 1:
            seams.append((cell, 1, cell + 1, 3))
        if row < rows - 1:
            seams.append((cell, 2, cell + cols, 0))
    return seams


def build_pairs(pieces, rows, cols, block=(2, 4), negatives_per_positive=8, seed=480):
    """Labelled side pairs, split by piece.

    ``block`` is the ``(rows, cols)`` size of the held-out corner block, taken
    from the bottom-right of the grid.

    Returns ``(train, valid, info)``.  Each example is
    ``(strip_a, strip_b, label, (cell_a, side_a, cell_b, side_b))`` with
    ``strip_b`` already reversed.

    Both directions of every seam are emitted as positives, because the solver
    asks the question in both directions too.
    """
    rng = np.random.default_rng(seed)
    cells = rows * cols
    block_rows, block_cols = block
    validation_cells = {r * cols + c
                        for r in range(rows - block_rows, rows)
                        for c in range(cols - block_cols, cols)}

    sides = [(cell, j) for cell in range(cells) for j in range(4)]
    strips = {(cell, j): side_strip(pieces[cell].sides[j]) for cell, j in sides}
    reversed_strips = {(cell, j): side_strip(pieces[cell].sides[j], reverse=True)
                       for cell, j in sides}

    seams = true_seams(rows, cols)
    positive_keys = set()
    for a_cell, a_side, b_cell, b_side in seams:
        positive_keys.add((a_cell, a_side, b_cell, b_side))
        positive_keys.add((b_cell, b_side, a_cell, a_side))

    def bucket(a_cell, b_cell):
        if a_cell in validation_cells and b_cell in validation_cells:
            return "valid"
        if a_cell not in validation_cells and b_cell not in validation_cells:
            return "train"
        return "dropped"          # straddles the split; using it would leak

    examples = {"train": [], "valid": [], "dropped": []}
    for key in sorted(positive_keys):
        a_cell, a_side, b_cell, b_side = key
        examples[bucket(a_cell, b_cell)].append(
            (strips[(a_cell, a_side)], reversed_strips[(b_cell, b_side)], 1.0, key))

    # Negatives: only pairs the tab/blank gate would let the solver consider.
    candidates = []
    for a_cell, a_side in sides:
        for b_cell, b_side in sides:
            if a_cell == b_cell:
                continue
            if (a_cell, a_side, b_cell, b_side) in positive_keys:
                continue
            if not complementary(pieces[a_cell].sides[a_side].kind,
                                 pieces[b_cell].sides[b_side].kind):
                continue
            if bucket(a_cell, b_cell) == "dropped":
                continue
            candidates.append((a_cell, a_side, b_cell, b_side))

    wanted = {"train": len(examples["train"]) * negatives_per_positive,
              "valid": len(examples["valid"]) * negatives_per_positive}
    pools = {"train": [], "valid": []}
    for key in candidates:
        pools[bucket(key[0], key[2])].append(key)
    for name in ("train", "valid"):
        pool = pools[name]
        take = min(wanted[name], len(pool))
        chosen = rng.choice(len(pool), size=take, replace=False) if take else []
        for index in np.sort(np.asarray(chosen, dtype=int)):
            a_cell, a_side, b_cell, b_side = pool[int(index)]
            examples[name].append(
                (strips[(a_cell, a_side)], reversed_strips[(b_cell, b_side)], 0.0,
                 pool[int(index)]))

    info = {
        "held_out_block": list(block),
        "held_out_cells": sorted(validation_cells),
        "strip_shape": list(strips[(0, 0)].shape),
        "train_positive": sum(1 for e in examples["train"] if e[2] == 1.0),
        "train_negative": sum(1 for e in examples["train"] if e[2] == 0.0),
        "valid_positive": sum(1 for e in examples["valid"] if e[2] == 1.0),
        "valid_negative": sum(1 for e in examples["valid"] if e[2] == 0.0),
        "positives_dropped_at_split_boundary": len(examples["dropped"]),
        "gated_negative_pool": len(candidates),
    }
    return examples["train"], examples["valid"], info


# --------------------------------------------------------------------- graphs
#
# The graph neural network is trained on *exactly* the examples the Siamese sees
# -- the same held-out block, the same positives, the same sampled negatives --
# so the brief's "same dataset split for both models" holds by construction
# rather than by coincidence.  What differs is the model's view of them: the
# Siamese judges a pair in isolation, while the GNN makes each side a node and
# each candidate pair an edge, so message passing lets a side's representation
# be shaped by every other partner competing for it.

BANDS = 4          # the 16 sampling depths are summarised in four bands
NODE_DIM = 31      # see side_features
EDGE_DIM = 16      # see pair_features
_KINDS = ("TAB", "BLANK", "FLAT")


def _band_stats(strip):
    """Per-channel mean and standard deviation in each of four depth bands."""
    channels, depths, _ = strip.shape
    banded = strip.reshape(channels, BANDS, depths // BANDS, -1)
    return (banded.mean(axis=(2, 3)).ravel(),      # 3 x 4
            banded.std(axis=(2, 3)).ravel())       # 3 x 4


def side_features(side, strip):
    """A side as a fixed-length vector: what its artwork looks like near the edge.

    Twelve band means, twelve band standard deviations, the mean inward colour
    gradient per channel, the side's relief, and a one-hot of its tab/blank/flat
    label -- 31 numbers, none of them the classical compatibility score, so the
    learned measures are not handed the answer they are being compared against.
    """
    means, stds = _band_stats(strip)
    inward = (strip[:, 0, :].mean(axis=1) - strip[:, -1, :].mean(axis=1))
    kind = np.array([float(side.kind.value == name) for name in _KINDS],
                    dtype=np.float32)
    relief = np.array([side.relief], dtype=np.float32)
    return np.concatenate([means, stds, inward, relief, kind]).astype(np.float32)


def pair_features(strip_a, strip_b_reversed, side_a, side_b):
    """A candidate pair as a fixed-length vector, symmetric in what it measures.

    Twelve per-band mean absolute colour differences across the seam, the two
    reliefs, whether the tab/blank gate admits the pair, and the overall mean
    absolute difference -- 16 numbers.
    """
    a_means, _ = _band_stats(strip_a)
    b_means, _ = _band_stats(strip_b_reversed)
    difference = np.abs(a_means - b_means)                       # 12
    seam = np.abs(strip_a[:, 0, :] - strip_b_reversed[:, 0, :]).mean()
    extra = np.array([side_a.relief, side_b.relief,
                      float(complementary(side_a.kind, side_b.kind)), seam],
                     dtype=np.float32)
    return np.concatenate([difference, extra]).astype(np.float32)


def build_graph(examples, pieces, extra_edges=()):
    """Turn labelled side pairs into one graph the GNN can be run on.

    ``examples`` are the ``(strip_a, strip_b, label, key)`` tuples returned by
    :func:`build_pairs`; each becomes one directed edge from side ``a`` to side
    ``b``.  ``extra_edges`` are additional ``(a_cell, a_side, b_cell, b_side)``
    keys to include unlabelled, which is how every gated candidate inside the
    held-out block gets a score at evaluation time.

    Returns a dict of numpy arrays plus ``index``, mapping an edge key to its
    row, so a caller can read one pair's prediction back out.
    """
    keys = [example[3] for example in examples]
    labels = [example[2] for example in examples]
    for key in extra_edges:
        if key not in set(keys):
            keys.append(key)
            labels.append(float('nan'))          # scored, never trained on

    nodes = sorted({(cell, side) for key in keys
                    for cell, side in ((key[0], key[1]), (key[2], key[3]))})
    node_index = {node: i for i, node in enumerate(nodes)}

    strips = {node: side_strip(pieces[node[0]].sides[node[1]]) for node in nodes}
    reversed_strips = {node: side_strip(pieces[node[0]].sides[node[1]], reverse=True)
                       for node in nodes}

    node_features = np.stack([side_features(pieces[cell].sides[side], strips[(cell, side)])
                              for cell, side in nodes])
    edge_index = np.array([[node_index[(k[0], k[1])] for k in keys],
                           [node_index[(k[2], k[3])] for k in keys]], dtype=np.int64)
    edge_attr = np.stack([pair_features(strips[(k[0], k[1])],
                                        reversed_strips[(k[2], k[3])],
                                        pieces[k[0]].sides[k[1]],
                                        pieces[k[2]].sides[k[3]]) for k in keys])
    return {"nodes": node_features.astype(np.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr.astype(np.float32),
            "labels": np.array(labels, dtype=np.float32).reshape(-1, 1),
            "index": {key: row for row, key in enumerate(keys)},
            "node_count": len(nodes)}
