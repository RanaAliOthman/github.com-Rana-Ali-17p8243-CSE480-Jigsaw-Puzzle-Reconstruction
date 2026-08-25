"""Command-line entry point: reconstruct a scrambled puzzle from piece images.

    python3 main.py --pieces data/pieces --output results/reconstructed_images/run.png

The pieces are read in whatever order the filesystem gives them and nothing
about their identity, position or orientation is passed to the search; the file
names are carried through to the report only so that a result can be read.  The
grid shape is inferred from the pieces themselves.

Ground truth, if supplied with ``--ground-truth``, is read only after the search
has finished, to score it.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from src.evaluation import blind_accuracy
from src.piece_geometry import describe_piece
from src.solver import build_variants, render_grid, solve


def load_pieces(folder):
    """Describe every RGBA piece image in a folder."""
    paths = sorted(Path(folder).glob("*.png"))
    pieces = []
    for path in paths:
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("expected an RGBA piece image: %s" % path)
        digits = "".join(c for c in path.stem if c.isdigit())
        pieces.append(describe_piece(int(digits or len(pieces) + 1),
                                     rgba[:, :, :3], rgba[:, :, 3]))
    if not pieces:
        raise ValueError("no piece images found in %s" % folder)
    return pieces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", default="data/pieces",
                        help="folder of RGBA piece images")
    parser.add_argument("--output",
                        default="results/reconstructed_images/reconstruction.png")
    parser.add_argument("--beam-width", type=int, default=10000)
    parser.add_argument("--per-state", type=int, default=16)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--ground-truth", default="",
                        help="evaluation-only layout JSON; omit to disable")
    args = parser.parse_args()

    started = time.perf_counter()
    pieces = load_pieces(args.pieces)
    print("%d pieces loaded from %s" % (len(pieces), args.pieces), flush=True)

    variants = build_variants(pieces)
    result = solve(pieces, args.rows, args.cols, beam_width=args.beam_width,
                   per_state=args.per_state, variants=variants,
                   progress=lambda message: print("  " + message, flush=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render_grid(variants, result))

    report = {
        "grid_inferred": list(result.grid),
        "complete": result.complete,
        "search_objective": result.cost,
        "quality": result.quality,
        "diagnostics": {key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in result.diagnostics.items()},
        "placements": [None if p is None else
                       {"cell": cell, "piece_id": p.piece_id, "rotation": p.rotation}
                       for cell, p in enumerate(result.placements)],
        "runtime_seconds": time.perf_counter() - started,
        "output": str(output),
    }
    if args.ground_truth:
        layout = json.loads(Path(args.ground_truth).read_text())
        # The pieces were handed over unrotated, so nothing was applied to them.
        report["accuracy"] = blind_accuracy(result, {}, layout)
    print(json.dumps(report, indent=2))
    metrics = output.with_suffix(".metrics.json")
    metrics.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
