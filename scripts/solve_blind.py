"""Reconstruct the puzzle blind and report how well it did.

    python3 -m scripts.solve_blind [--beam 200] [--per-state 6] [--seed 480]

The pieces in ``data/pieces`` are shuffled and each is given a random rotation,
so the solver receives no position and no orientation information.  It is not
told the grid shape either.  Ground truth is read only *after* the search has
finished, purely to score it.

Accuracy is reported twice, because a 5x7 grid has one exact symmetry: turning
a finished puzzle by 180 degrees maps it onto another valid arrangement with
identical seam costs, which no compatibility measure can distinguish.  The
figure quoted as the result is the better of the two orientations, and the
other is printed alongside so the choice is visible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import blind_accuracy as score
from src.piece_geometry import describe_piece, rotate_piece
from src.solver import build_variants, render_grid, solve

PIECES = Path('data/pieces')
LAYOUT = Path('data/ground_truth/layout.json')
OUTPUT = Path('results/reconstructed_images')


def load_scrambled(seed):
    """Load every piece, give it a random rotation, and shuffle the bag.

    Returns the pieces and, for scoring only, the rotation each one was given.
    """
    rng = np.random.default_rng(seed)
    paths = sorted(PIECES.glob('piece_*.png'))
    pieces, applied = [], {}
    for path in paths:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        piece_id = int(path.stem.split('_')[-1])
        turn = int(rng.integers(0, 4))
        base = describe_piece(piece_id, rgba[:, :, :3], rgba[:, :, 3])
        pieces.append(rotate_piece(base, turn))
        applied[piece_id] = turn
    order = rng.permutation(len(pieces))
    return [pieces[i] for i in order], applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beam', type=int, default=10000)
    parser.add_argument('--per-state', type=int, default=16)
    parser.add_argument('--seed', type=int, default=480)
    parser.add_argument('--rows', type=int, default=None)
    parser.add_argument('--cols', type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    pieces, applied = load_scrambled(args.seed)
    print('%d pieces loaded, shuffled and randomly rotated' % len(pieces), flush=True)

    variants = build_variants(pieces)
    reconstruction = solve(pieces, args.rows, args.cols, beam_width=args.beam,
                           per_state=args.per_state, variants=variants,
                           progress=lambda m: print('  ' + m, flush=True))
    elapsed = time.perf_counter() - started

    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = render_grid(variants, reconstruction)
    path = OUTPUT / 'blind_reconstruction.png'
    cv2.imwrite(str(path), image)

    report = {
        'grid_inferred': list(reconstruction.grid),
        'complete': reconstruction.complete,
        'search_cost': reconstruction.cost,
        'quality': reconstruction.quality,
        'diagnostics': {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in reconstruction.diagnostics.items()},
        'runtime_seconds': elapsed,
        'output': str(path),
    }
    if LAYOUT.exists():
        report['accuracy'] = score(reconstruction, applied,
                                   json.loads(LAYOUT.read_text()))
    print(json.dumps(report, indent=2))
    Path('results/evaluation_results').mkdir(parents=True, exist_ok=True)
    Path('results/evaluation_results/blind_reconstruction.json').write_text(
        json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
