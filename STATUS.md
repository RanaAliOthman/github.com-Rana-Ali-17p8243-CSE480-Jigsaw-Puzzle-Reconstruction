# Project status

Last verified: 2026-08-22. This is the single status file for the project. It
replaces `FINAL_STATUS.md`, `HOW_TO_RUN.md`, `DISCUSSION_GUIDE.md`,
`SOLVING_REQUIREMENTS.md` and `RECONSTRUCTION_ANALYSIS.md`, all of which predated
the derived ground truth and contradicted both each other and the evidence.

- **How to run it** → [`README.md`](README.md)
- **Requirement-by-requirement evidence** → [`report/milestone_1_report.md`](report/milestone_1_report.md)
- **Milestone 2, and the output contract** → [`report/milestone_2_report.md`](report/milestone_2_report.md)
- **What to do next** → [`NEXT_STEPS.md`](NEXT_STEPS.md)

---

## In one line

Milestone 1 is complete as a library and as an end-to-end routine; the blind
reconstruction places **24 of 35** pieces correctly, which is **not** a complete
reconstruction and is not presented as one.

| | |
|---|---|
| Conformance checks (`python3 -m scripts.audit_requirements`) | **24 / 24** |
| Unit tests (`python3 -m pytest -q`) | **108**, ~15 s |
| Demonstration notebook | runs top to bottom, ~70 s |
| Blind position accuracy | 24 / 35 cells (68.6%) |
| Blind orientation accuracy | 23 / 35 cells (65.7%) |
| Blind neighbour accuracy | 34 / 58 seams (58.6%) |
| Complete reconstruction | **no** |
| Milestone 2 (learned measure) | **substantially outstanding** |

## What works

- The from-scratch library: convolution, mean/Gaussian/median filtering,
  histogram computation and equalisation, contrast stretching, sharpening,
  global/Otsu/adaptive thresholding, Sobel/Prewitt with orientation, a
  stage-by-stage Canny, connected components, Moore boundary tracing and binary
  morphology. OpenCV is used only for I/O, colour conversion and geometric warps.
- Piece extraction from the photographs, cross-frame combination into 35 clean
  piece images, and a layout and per-piece orientation derived and verified two
  independent ways (`data/ground_truth/layout.json` records the evidence).
- Piece-edge description: four corners, four sides, TAB/BLANK/FLAT labels,
  signed silhouette profiles and interior colour strips at three depths.
- Piece-edge matching: one number per candidate pair, Mahalanobis gradient
  compatibility, with shape as a gate only. 46.6% rank-1 over the 116 directed
  true-neighbour pairs, against 22.4% for colour SSD.
- Blind assembly: an unordered bag of pieces in, grid shape inferred, position
  and rotation searched, a rendered reconstruction and a quality score out. On a
  synthetic puzzle cut from a known image it reconstructs **exactly**.
- `notebooks/full_project_demo.ipynb` runs the whole of the above on the real
  photographs and reproduces every headline figure.

## What limits the result

Measured, not assumed. `python3 -m scripts.evaluate_solver` scores the *true*
arrangement under the solver's own objective: from beam width 3 000 upwards the
search finds arrangements scoring **better** than the truth (1735 against 1943),
and widening the beam lowers the objective monotonically while accuracy wanders
between 54% and 69%.

So the search works and **the compatibility measure is the ceiling**. More search
cannot help. Three properties of this puzzle are why:

1. It is **die-cut** — every tab shares one profile, so shape carries almost no
   identifying information and can only gate.
2. **Flat sides cannot be classified reliably per piece**; three of the 35 come
   out with the wrong flat count under every profile statistic swept, so the
   solver charges a cost for silhouette disagreement rather than trusting a label.
3. One true seam **breaks the tab/blank gate**, because a spring clamp damages
   piece 2's silhouette in all fifty photographs, so the gate must be a penalty
   or the correct answer is unreachable.

## What is outstanding

- **Milestone 2.** Both models are trained on the same split for 500 epochs each.
  At the checkpoints they keep, **neither beats the classical measure**: 40.0%
  rank-1 each against MGC's 55.0% on the identical held-out candidate set. The
  GNN's best-validation checkpoint (epoch 225) reaches 60.0%, but that is 12
  correct pairs against 11 out of 20 — not a meaningful lead, and not the
  checkpoint the run selects. Both overfit 84 training positives hard: the GNN's
  training loss ends at 0.0006 with validation loss at 8.84 and recall on true
  seams down to 0.20. Only the `neighbor_logit` head is trained in either model;
  `orientation_logits` and `compatibility` remain at their random initialisation
  and must not be reported. Nothing model-driven is wired into the solver.
  Summary: `python3 -m ml.evaluate_models`; full state:
  [`report/milestone_2_report.md`](report/milestone_2_report.md).
- **A better measure is the open problem.** `NEXT_STEPS.md` §3 ranks the options.

## Corrections to the superseded documents

Claims that appeared in the files this one replaces, and what is now established:

| Superseded claim | Established |
|---|---|
| "Positive true-neighbour pairs cannot be generated from the supplied labels." | They can, and were. `scripts/build_ground_truth.py` derives the layout and orientations from the photographs; `data/ground_truth/` holds them with their evidence. |
| "A reference photograph or layout manifest is required for any verifiable result." | Not required. The layout was derived from shape evidence alone (5 violated constraints out of 82, against a minimum of 34 over 300 random arrangements) and orientations from artwork continuity. |
| "Why are shape and colour both needed?" | They are not. Shape is a gate only; every weighted shape term measured made matching monotonically worse on this die-cut puzzle. |
| "Tests passed: 11 / 11." | 108 tests pass. |
| "The 35-piece images are diagnostic arrangements only." | `data/pieces/piece_01..35.png` are clean per-piece images, median-combined over ~44 views each, and are the input the solver actually uses. |
