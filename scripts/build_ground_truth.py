"""Derive and verify the puzzle's ground truth from the supplied photographs.

Run once (about four minutes); everything downstream reads the files it writes.

    python3 -m scripts.build_ground_truth

Outputs
-------
data/pieces/piece_NN.png
    Canonical, occlusion-free RGBA image of each piece, median-combined over all
    its observations.
data/ground_truth/layout.json
    The solved 5x7 layout, the solved orientation of every piece, and the
    evidence supporting both.
data/ground_truth/frame_annotations.json
    For every photograph, each piece's grid position and orientation.
results/ground_truth/solved_reference.png
    The reconstructed reference image, for visual confirmation.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATASET, ROOT
from src.registry import (COLS, ROWS, build_registry, collect_observations,
                          registry_pieces, render_solution, solve_orientations,
                          verify_layout)


def report(done, total, stem, matched):
    print("  %2d/%d  %-18s pieces matched: %2d" % (done, total, stem[:18], matched),
          flush=True)


def main():
    started = time.time()
    print("Collecting piece observations from every full-puzzle photograph...")
    observations, frame_names = collect_observations(DATASET, progress=report)
    print("%d observations from %d photographs" % (len(observations), len(frame_names)))

    print("\nMedian-combining observations into clean canonical pieces...")
    registry, alignment = build_registry(observations)
    pieces = registry_pieces(registry)
    print("registry holds %d pieces" % len(pieces))

    piece_dir = ROOT / "data" / "pieces"
    piece_dir.mkdir(parents=True, exist_ok=True)
    for piece_id, (image, mask) in sorted(registry.items()):
        rgba = np.dstack([image, (mask > 0).astype(np.uint8) * 255])
        cv2.imwrite(str(piece_dir / ("piece_%02d.png" % piece_id)), rgba)
    print("wrote %s" % piece_dir)

    print("\nTesting the layout hypothesis on shape evidence alone...")
    evidence = verify_layout(pieces)
    print("  hypothesised layout violates %d of %d constraints"
          % (evidence["hypothesis_violations"], 24 + 58))
    print("  random arrangements: min %d, median %.1f over %d trials"
          % (evidence["null_min"], evidence["null_median"], evidence["null_trials"]))
    print("  fraction of random arrangements at least as consistent: %.4f"
          % evidence["fraction_null_at_least_as_good"])

    print("\nResolving orientations from artwork continuity...")
    cost, rotations = solve_orientations(pieces)
    print("  best seam cost %.4f" % cost)

    reference = render_solution(pieces, rotations)
    figure_dir = ROOT / "results" / "ground_truth"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(figure_dir / "solved_reference.png"), reference)
    print("  wrote %s" % (figure_dir / "solved_reference.png"))

    layout = {
        "grid": [ROWS, COLS],
        "positions_piece_id": [position + 1 for position in range(ROWS * COLS)],
        "solved_rotations_ccw": {str(position + 1): rotations[position]
                                 for position in range(ROWS * COLS)},
        "rotation_convention": (
            "Quarter turns counter-clockwise applied to the canonical piece in "
            "data/pieces to bring it into its solved orientation."),
        "derivation": (
            "Pieces were segmented from every full-puzzle photograph with the "
            "project's own library, warped to a canonical body square, and "
            "median-combined across all observations of the same piece. The "
            "hypothesis that dataset identity k occupies row-major cell k was "
            "then tested on shape evidence alone (flat borders and tab/blank "
            "complementarity), and orientations were resolved from artwork "
            "continuity. The resulting reference image is at "
            "results/ground_truth/solved_reference.png."),
        "shape_evidence": evidence,
        "orientation_seam_cost": cost,
        "solver_must_not_read_this_file": True,
        "known_limitations": [
            "Piece 2 is partially occluded by a spring clamp resting on it in "
            "every one of the fifty photographs, so its silhouette cannot be "
            "fully recovered from this dataset and two of its side labels are "
            "unreliable.",
            "The puzzle is die-cut: every tab shares one profile and every blank "
            "its complement, so side shape constrains but cannot by itself "
            "determine orientation.",
        ],
    }
    truth_dir = ROOT / "data" / "ground_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "layout.json").write_text(json.dumps(layout, indent=2))
    print("wrote %s" % (truth_dir / "layout.json"))

    # Per-frame annotations: a piece's orientation in a photograph is the turn
    # that aligns it with the registry, plus the registry piece's solved turn.
    annotations = {}
    for (piece_id, frame), align in sorted(alignment.items()):
        position = piece_id - 1
        annotations.setdefault(frame_names[frame], {})[str(piece_id)] = {
            "row": position // COLS,
            "column": position % COLS,
            "rotation_ccw": int((align + rotations[position]) % 4),
        }
    (truth_dir / "frame_annotations.json").write_text(json.dumps({
        "grid": [ROWS, COLS],
        "description": (
            "Per photograph, the solved grid cell and orientation of every piece "
            "that could be segmented and identified in it. Orientation is the "
            "number of counter-clockwise quarter turns taking the piece as "
            "extracted and deskewed from that photograph into its solved pose."),
        "frames": annotations,
    }, indent=2))
    print("wrote %s (%d frames)" % (truth_dir / "frame_annotations.json", len(annotations)))
    print("\ndone in %.1f s" % (time.time() - started))


if __name__ == "__main__":
    main()
