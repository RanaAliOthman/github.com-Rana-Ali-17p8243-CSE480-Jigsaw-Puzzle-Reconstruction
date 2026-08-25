"""Cross-frame piece registry and ground-truth derivation.

The dataset provides fifty top-down photographs that each contain all thirty-five
pieces of the puzzle, plus YOLO boxes naming which physical piece is which.  This
module turns that redundancy into three things the rest of the project needs:

1. a clean, occlusion-free canonical image of every piece, obtained by aligning
   all of a piece's observations and taking a per-pixel median;
2. the solved layout -- which piece belongs in which cell of the 5x7 grid, and
   at which orientation -- derived and then verified, not assumed;
3. per-frame annotations giving each piece's position and orientation in every
   photograph, which are the supervised labels for the learning milestone and
   the reference for honest accuracy figures.

What the identity labels are and are not used for
-------------------------------------------------
The YOLO class of a box names a physical piece.  That is used here only to know
that a piece seen in frame A is the same object as a piece seen in frame B, and
to index the derived ground truth.  It is never used to decide where a piece
belongs: the layout is established by the evidence in
:func:`verify_layout` and :func:`solve_orientations`, and the reconstruction
pipeline itself never reads any of it.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .edge_compatibility import colour_dissimilarity
from .frame_extraction import extract_frame_pieces
from .piece_geometry import (BODY, CANVAS, SideType, canonicalize,
                             describe_piece, rotate_piece)

__all__ = ["ROWS", "COLS", "full_frame_paths", "identity_boxes",
           "collect_observations", "build_registry", "verify_layout",
           "solve_orientations", "render_solution"]

ROWS, COLS = 5, 7
PAD = (CANVAS - BODY) // 2
_CORE = slice(PAD + 12, PAD + BODY - 12)
MATCH_RADIUS = 60.0     # px between a YOLO box centre and a segmented centroid


@dataclass
class Observation:
    """One piece as seen in one photograph, warped to the canonical canvas."""
    piece_id: int
    frame: int
    image: np.ndarray
    mask: np.ndarray
    quality: float
    centroid: tuple


def full_frame_paths(dataset_root):
    """Every photograph whose label file contains all 35 pieces."""
    root = Path(dataset_root)
    out = []
    for split in ("train", "valid", "test"):
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            continue
        for label in sorted(label_dir.glob("*.txt")):
            lines = [l for l in label.read_text().splitlines() if l.strip()]
            if len(lines) != ROWS * COLS:
                continue
            images = list((root / "images" / split).glob(label.stem + ".*"))
            if images:
                out.append((split, label.stem, images[0]))
    return out


def identity_boxes(dataset_root, split, stem, shape):
    """Piece identity and box centre, in pixels, for one photograph."""
    root = Path(dataset_root)
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    height, width = shape[:2]
    out = []
    for line in (root / "labels" / split / (stem + ".txt")).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        out.append((int(names[int(fields[0])]),
                    float(fields[1]) * width, float(fields[2]) * height))
    return out


def collect_observations(dataset_root, progress=None):
    """Segment every full frame and canonicalise each identified piece.

    Candidates are matched to identity boxes by centroid proximity rather than
    by requiring a perfect count, so a frame that also yields a clamp or a hex
    key still contributes its genuine pieces, and a frame in which one piece is
    lost to occlusion still contributes the rest.
    """
    frames = full_frame_paths(dataset_root)
    observations, frame_names = [], []
    for index, (split, stem, path) in enumerate(frames):
        image = cv2.imread(str(path))
        if image is None:
            continue
        try:
            pieces = extract_frame_pieces(image)
        except Exception:
            continue
        boxes = identity_boxes(dataset_root, split, stem, image.shape)

        pairs = []
        for piece_id, box_x, box_y in boxes:
            for piece in pieces:
                cx, cy = piece.source_centroid
                distance = float(np.hypot(cx - box_x, cy - box_y))
                if distance <= MATCH_RADIUS:
                    pairs.append((distance, piece_id, id(piece), piece))
        pairs.sort(key=lambda item: item[0])

        assigned, claimed = {}, set()
        for _, piece_id, key, piece in pairs:
            if piece_id in assigned or key in claimed:
                continue
            assigned[piece_id] = piece
            claimed.add(key)

        frame_index = len(frame_names)
        frame_names.append(stem)
        for piece_id, piece in sorted(assigned.items()):
            canonical_image, canonical_mask = canonicalize(
                piece.image, piece.mask, piece.corners)
            observations.append(Observation(
                piece_id=piece_id, frame=frame_index, image=canonical_image,
                mask=(canonical_mask > 0).astype(np.uint8),
                quality=piece.quality, centroid=piece.source_centroid))
        if progress:
            progress(index + 1, len(frames), stem, len(assigned))
    return observations, frame_names


def _alignment_cost(image, mask, reference_image, reference_mask, k):
    """Cost of aligning an observation to a reference under ``k`` quarter turns.

    Artwork inside the body square drives the decision.  The body square is
    interior to the piece, so it survives boundary occlusions such as a gripper
    finger resting on an edge, and printed artwork -- unlike many jigsaw
    silhouettes -- is never symmetric under a quarter turn.  Silhouette overlap
    is kept as a weak tie-breaker for pieces whose artwork is nearly blank.
    """
    rotated_image = np.rot90(image, k)
    rotated_mask = np.rot90(mask, k)
    colour = float(np.mean((rotated_image[_CORE, _CORE].astype(np.float32)
                            - reference_image[_CORE, _CORE].astype(np.float32)) ** 2))
    union = np.count_nonzero((rotated_mask > 0) | (reference_mask > 0))
    overlap = np.count_nonzero((rotated_mask > 0) & (reference_mask > 0))
    return colour / (255.0 ** 2) - 0.5 * (overlap / union if union else 0.0)


def build_registry(observations, passes=3):
    """Median-combine every observation of each piece into one clean canvas.

    Returns
    -------
    (registry, alignment)
        ``registry[piece_id] = (image, mask)`` and
        ``alignment[(piece_id, frame)] = quarter turns`` mapping that frame's
        observation onto the registry canvas.
    """
    by_piece = {}
    for observation in observations:
        by_piece.setdefault(observation.piece_id, []).append(observation)

    registry, alignment = {}, {}
    for piece_id, group in sorted(by_piece.items()):
        # Occlusion only ever removes material, so the largest silhouette is the
        # least occluded observation and makes the best starting reference.
        seed = max(group, key=lambda o: int(o.mask.sum()))
        reference_image, reference_mask = seed.image, seed.mask

        chosen = []
        for _ in range(passes):
            chosen, images, masks = [], [], []
            for observation in group:
                k = min(range(4), key=lambda k: _alignment_cost(
                    observation.image, observation.mask,
                    reference_image, reference_mask, k))
                chosen.append(k)
                images.append(np.rot90(observation.image, k))
                masks.append(np.rot90(observation.mask, k))
            mask_stack = np.stack(masks)
            reference_mask = (mask_stack.mean(axis=0) > 0.5).astype(np.uint8)
            image_stack = np.stack(images).astype(np.float32)
            covered = np.broadcast_to(mask_stack[..., None].astype(bool), image_stack.shape)
            image_stack[~covered] = np.nan
            with np.errstate(all="ignore"):
                reference_image = np.nan_to_num(
                    np.nanmedian(image_stack, axis=0)).astype(np.uint8)

        registry[piece_id] = (reference_image, reference_mask)
        for observation, k in zip(group, chosen):
            alignment[(piece_id, observation.frame)] = int(k)
    return registry, alignment


def registry_pieces(registry):
    """Describe every registry entry, ordered by piece identity."""
    return [describe_piece(piece_id, image, (mask > 0).astype(np.uint8) * 255)
            for piece_id, (image, mask) in sorted(registry.items())]


def _grid_requirements(position):
    row, col = divmod(position, COLS)
    return (row == 0, col == COLS - 1, row == ROWS - 1, col == 0)


def verify_layout(pieces, order=None, trials=300, seed=480):
    """Score how well an arrangement can satisfy the puzzle's shape constraints.

    The score counts violated constraints at the best rotation assignment: a
    side that is flat where it must meet a neighbour (or vice versa), and a seam
    that fails to pair a tab with a blank.  Colour is deliberately not used, so
    this test is independent of the photometric evidence.

    Returns
    -------
    dict
        The hypothesis score, the null distribution over random arrangements,
        and the fraction of random arrangements scoring at least as well.
    """
    rng = np.random.default_rng(seed)
    kinds = []
    for piece in pieces:
        base = [side.kind for side in piece.sides]
        # A piece rotated by k has side j equal to side (j + k) % 4 of the original.
        kinds.append([[base[(j + k) % 4] for j in range(4)] for k in range(4)])

    def score(arrangement, restarts):
        cells = ROWS * COLS

        def border(position, k):
            required = _grid_requirements(position)
            row = kinds[arrangement[position]][k]
            return sum((row[j] is SideType.FLAT) != required[j] for j in range(4))

        def seam(a_pos, a_k, b_pos, b_k, vertical):
            a = kinds[arrangement[a_pos]][a_k][2 if vertical else 1]
            b = kinds[arrangement[b_pos]][b_k][0 if vertical else 3]
            return 0 if {a, b} == {SideType.TAB, SideType.BLANK} else 1

        def total(rotations):
            value = sum(border(p, rotations[p]) for p in range(cells))
            for position in range(cells):
                row, col = divmod(position, COLS)
                if col < COLS - 1:
                    value += seam(position, rotations[position],
                                  position + 1, rotations[position + 1], False)
                if row < ROWS - 1:
                    value += seam(position, rotations[position],
                                  position + COLS, rotations[position + COLS], True)
            return value

        best = None
        for attempt in range(restarts):
            rotations = ([int(np.argmin([border(p, k) for k in range(4)]))
                          for p in range(cells)] if attempt == 0
                         else list(rng.integers(0, 4, cells)))
            for _ in range(40):
                changed = False
                for position in rng.permutation(cells):
                    row, col = divmod(position, COLS)
                    values = []
                    for k in range(4):
                        value = border(position, k)
                        if col > 0:
                            value += seam(position - 1, rotations[position - 1], position, k, False)
                        if col < COLS - 1:
                            value += seam(position, k, position + 1, rotations[position + 1], False)
                        if row > 0:
                            value += seam(position - COLS, rotations[position - COLS], position, k, True)
                        if row < ROWS - 1:
                            value += seam(position, k, position + COLS, rotations[position + COLS], True)
                        values.append(value)
                    pick = int(np.argmin(values))
                    if pick != rotations[position]:
                        rotations[position], changed = pick, True
                if not changed:
                    break
            value = total(rotations)
            if best is None or value < best[0]:
                best = (value, list(rotations))
        return best

    order = list(range(len(pieces))) if order is None else list(order)
    hypothesis, rotations = score(order, restarts=200)
    null = np.array([score(list(rng.permutation(len(pieces))), restarts=12)[0]
                     for _ in range(trials)])
    return {
        "hypothesis_violations": int(hypothesis),
        "hypothesis_rotations": [int(k) for k in rotations],
        "null_min": int(null.min()), "null_median": float(np.median(null)),
        "null_mean": float(null.mean()), "null_trials": int(trials),
        "fraction_null_at_least_as_good": float(np.mean(null <= hypothesis)),
    }


def solve_orientations(pieces, restarts=400, seed=480,
                       shape_penalty=1.0, border_penalty=1.0):
    """Choose each piece's orientation in the solved grid using artwork continuity.

    Shape cannot decide this.  The puzzle is die-cut, so every tab shares one
    profile and every blank its complement, and a great many rotation
    assignments satisfy the tab/blank and border constraints equally well.
    Colour continuity across seams breaks the tie; shape and border conditions
    are retained as additive penalties so the search still respects them.
    """
    count = len(pieces)
    variants = [[piece] + [rotate_piece(piece, k) for k in (1, 2, 3)] for piece in pieces]

    horizontal = np.zeros((count, 4, count, 4))
    vertical = np.zeros((count, 4, count, 4))
    for i in range(count):
        for a in range(4):
            right, bottom = variants[i][a].sides[1], variants[i][a].sides[2]
            for j in range(count):
                if i == j:
                    continue
                for b in range(4):
                    left, top = variants[j][b].sides[3], variants[j][b].sides[0]
                    cost = colour_dissimilarity(right, left)
                    if {right.kind, left.kind} != {SideType.TAB, SideType.BLANK}:
                        cost += shape_penalty
                    horizontal[i, a, j, b] = cost
                    cost = colour_dissimilarity(bottom, top)
                    if {bottom.kind, top.kind} != {SideType.TAB, SideType.BLANK}:
                        cost += shape_penalty
                    vertical[i, a, j, b] = cost

    border = np.zeros((count, 4))
    for position in range(count):
        required = _grid_requirements(position)
        for k in range(4):
            kinds = [side.kind for side in variants[position][k].sides]
            border[position, k] = border_penalty * sum(
                (kinds[j] is SideType.FLAT) != required[j] for j in range(4))

    rng = np.random.default_rng(seed)

    def total(rotations):
        value = sum(border[p, rotations[p]] for p in range(count))
        for position in range(count):
            row, col = divmod(position, COLS)
            if col < COLS - 1:
                value += horizontal[position, rotations[position],
                                    position + 1, rotations[position + 1]]
            if row < ROWS - 1:
                value += vertical[position, rotations[position],
                                  position + COLS, rotations[position + COLS]]
        return float(value)

    best = None
    for attempt in range(restarts):
        rotations = (np.argmin(border, axis=1).tolist() if attempt == 0
                     else list(rng.integers(0, 4, count)))
        for _ in range(60):
            changed = False
            for position in rng.permutation(count):
                row, col = divmod(position, COLS)
                values = border[position].astype(float).copy()
                for k in range(4):
                    if col > 0:
                        values[k] += horizontal[position - 1, rotations[position - 1], position, k]
                    if col < COLS - 1:
                        values[k] += horizontal[position, k, position + 1, rotations[position + 1]]
                    if row > 0:
                        values[k] += vertical[position - COLS, rotations[position - COLS], position, k]
                    if row < ROWS - 1:
                        values[k] += vertical[position, k, position + COLS, rotations[position + COLS]]
                pick = int(np.argmin(values))
                if pick != rotations[position]:
                    rotations[position], changed = pick, True
            if not changed:
                break
        value = total(rotations)
        if best is None or value < best[0]:
            best = (value, [int(k) for k in rotations])
    return best


def render_solution(pieces, rotations, margin=60, background=(245, 245, 245)):
    """Composite the pieces into the solved grid so tabs enter their blanks."""
    height, width = ROWS * BODY + 2 * margin, COLS * BODY + 2 * margin
    canvas = np.full((height, width, 3), background, np.uint8)
    for position, (piece, k) in enumerate(zip(pieces, rotations)):
        rotated = rotate_piece(piece, k)
        row, col = divmod(position, COLS)
        y0, x0 = margin + row * BODY - PAD, margin + col * BODY - PAD
        ys, xs = np.nonzero(rotated.mask > 0)
        ty, tx = y0 + ys, x0 + xs
        inside = (ty >= 0) & (ty < height) & (tx >= 0) & (tx < width)
        canvas[ty[inside], tx[inside]] = rotated.image[ys[inside], xs[inside]]
    return canvas
