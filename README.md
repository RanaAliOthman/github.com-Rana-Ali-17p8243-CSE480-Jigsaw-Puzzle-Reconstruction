# CSE480 Jigsaw Puzzle Reconstruction

Machine Vision project, Ain Shams University. A from-scratch image-processing
library that segments a photograph of a scrambled jigsaw puzzle, describes each
piece, and reconstructs the puzzle.

The puzzle is a 5 x 7 (35 piece) HIWIN promotional jigsaw. The dataset is a
Roboflow YOLO detection set containing 4,684 photographs, 50 of which show all
35 pieces scattered on a black mat in a single top-down frame.

---

## 1. Requirements

Tested with **Python 3.10.12** on Linux.

```bash
python3 -m pip install -r requirements.txt
```

| Package | Needed for |
|---|---|
| `numpy`, `opencv-python`, `PyYAML` | the classical pipeline (required) |
| `pytest` | the test suite |
| `matplotlib`, `nbformat`, `nbclient`, `ipykernel` | the demonstration notebook and its figures |
| `torch` | the Milestone 2 models (see §6) |

OpenCV is used only for image I/O, colour conversion, and geometric warps. The
operations the brief requires to be written from scratch — convolution, the
mean/Gaussian/median filters, histogram computation and equalisation, contrast
stretching, sharpening, global/Otsu/adaptive thresholding, Sobel/Prewitt/Canny,
connected-component labelling, boundary tracing and binary morphology — are all
implemented in `src/` on plain NumPy.

### Dataset

The dataset must already be unpacked at `data/raw/detection/`:

```
data/raw/detection/
├── data.yaml
├── images/{train,valid,test}/
└── labels/{train,valid,test}/
```

Check it is in place:

```bash
python3 -c "from config import DATASET; from src.registry import full_frame_paths; \
print(DATASET.exists(), len(full_frame_paths(DATASET)), 'full-puzzle frames')"
```

Expected output: `True 50 full-puzzle frames`.

---

## 2. Quick start

Run these three commands from the repository root, in order.

```bash
python3 -m pip install -r requirements.txt   # once
python3 -m scripts.build_ground_truth        # ~3 minutes
python3 -m pytest -q                         # ~15 seconds
python3 -m scripts.solve_blind               # ~32 seconds
```

Or run `notebooks/full_project_demo.ipynb` top to bottom (~70 s), which does the
whole of §3 in one pass and shows every intermediate.

---

## 3. What each command does

### `python3 -m scripts.build_ground_truth`

The main pipeline. Runtime **≈170 s**. It reads only the dataset and writes
everything the rest of the project depends on.

For each of the 50 full-puzzle photographs it runs the library end to end —
Gaussian smoothing, Otsu thresholding on the value channel, morphological
cleanup, from-scratch connected-component labelling, separation of pieces that
touch, boundary tracing, deskewing and corner detection. It then combines all
observations of the same physical piece into one clean, occlusion-free image,
establishes the solved layout, and resolves every piece's orientation.

Console output ends with:

```
1531 observations from 50 photographs
registry holds 35 pieces

Testing the layout hypothesis on shape evidence alone...
  hypothesised layout violates 5 of 82 constraints
  random arrangements: min 34, median 42.0 over 300 trials
  fraction of random arrangements at least as consistent: 0.0000

Resolving orientations from artwork continuity...
  best seam cost 6.2309
```

Files written:

| Path | Contents |
|---|---|
| `results/ground_truth/solved_reference.png` | **The reconstructed puzzle.** Open this first. |
| `data/pieces/piece_01..35.png` | Clean RGBA image of each piece, median-combined over ~44 views |
| `data/ground_truth/layout.json` | Solved grid, per-piece orientation, and the evidence for both |
| `data/ground_truth/frame_annotations.json` | 1,531 per-piece position + orientation labels across 46 frames |

The reconstruction is verified two independent ways. A purely geometric test
(flat borders and tab/blank complementarity, no colour) scores the layout at 5
violated constraints out of 82, against a minimum of 34 for 300 random
arrangements. Orientations are then fixed by artwork continuity, and
`solved_reference.png` confirms the result visually: the HIWIN logo and the red
飛躍新30+ mark read correctly and the artwork is continuous across seams.

### `python3 -m scripts.solve_blind`

Shuffles the 35 pieces, gives each a random rotation, and reconstructs the
puzzle blind — no identity, no position, no orientation, and the grid shape is
inferred rather than given. Runtime **≈32 s**. Writes
`results/reconstructed_images/blind_reconstruction.png` and
`results/evaluation_results/blind_reconstruction.json`.

Ground truth is read only after the search has finished, to score it. See §6 for
what it achieves and `report/milestone_1_report.md` for how it works.

### `notebooks/full_project_demo.ipynb`

The demonstration notebook. Runtime **≈70 s**, most of it the final assembly
search. It walks the brief in order — enhancement, thresholding, edge detection,
segmentation and contours, description, matching, assembly — computing every
figure from the real photographs rather than loading a saved one, and writing its
outputs into `results/enhanced_images/`, `results/masks/`, `results/contours/`,
`results/edge_visualisations/`, `results/reconstructed_images/` and
`results/evaluation_results/`. It reproduces the 46.6% matcher figure and the
24/35 reconstruction figure quoted below. It can be opened from the repository
root or from `notebooks/`; the first cell finds the root either way.

### `python3 -m pytest -q`

108 tests, ~15 s, no dataset required. They cover the from-scratch operators
against known results — including the boundary tracer checked point-for-point
against `cv2.findContours`, and connected components checked for 4- versus
8-connectivity and union-find merging — plus the piece descriptors, the
compatibility score, and the assembly search — the last of these on a real
interlocking puzzle cut from a known image, which the search must put back
together exactly.

---

## 4. Repository map

```
├── main.py                     CLI: piece images in, reconstruction out
├── config.py                   paths and the random seed
├── src/
│   ├── enhancement.py          convolution, mean/Gaussian/median, histogram,
│   │                           equalisation, contrast stretch, unsharp, Laplacian
│   ├── thresholding.py         global, Otsu, adaptive
│   ├── edge_detection.py       Sobel, Prewitt, full Canny
│   ├── segmentation.py         morphology, connected components, hole filling,
│   │                           separation of touching pieces
│   ├── contour_extraction.py   Moore-neighbourhood boundary tracing
│   ├── frame_extraction.py     photograph -> upright pieces with corners
│   ├── piece_geometry.py       canonical piece form and four-side description
│   ├── edge_compatibility.py   the pairwise side-matching score
│   ├── solver.py               the blind assembly search
│   ├── evaluation.py           accuracy against the derived ground truth
│   ├── registry.py             cross-frame combination, layout verification
│   └── synthetic.py            generates puzzles with a known solution
├── scripts/
│   ├── build_ground_truth.py   derive and verify the ground truth
│   ├── solve_blind.py          scramble the pieces and reconstruct them
│   ├── evaluate_solver.py      accuracy, and what limits it
│   ├── evaluate_matcher.py     true-neighbour rank of the matcher
│   └── audit_requirements.py   conformance against the brief
├── report/
│   ├── milestone_1_report.md   Milestone 1: requirements, evidence, results
│   ├── milestone_2_report.md   Milestone 2: the output contract and what is
│   │                           still outstanding
│   └── *.pdf                   the same documents, typeset
├── ml/
│   ├── dataset.py              pair dataset built from the derived adjacencies
│   ├── siamese.py, gnn.py      the Milestone 2 model definitions
│   ├── ranking.py              the one protocol every measure is scored by
│   ├── train_siamese.py        Siamese training + held-out rank evaluation
│   ├── train_gnn.py            GNN training on the same split
│   └── evaluate_models.py      both models vs classical, from their reports
├── notebooks/
│   └── full_project_demo.ipynb the end-to-end demonstration
├── tests/                      test_enhancement, test_thresholding,
│                               test_edge_detection, test_segmentation,
│                               test_piece_description, test_piece_geometry,
│                               test_contour_extraction, test_edge_matching,
│                               test_assembly, test_ml
├── data/
│   ├── raw/detection/          the Roboflow dataset (not in the repository)
│   ├── input/                  the photograph the notebook demonstrates on
│   ├── pieces/                 the 35 clean piece images
│   ├── sample_pieces/          six of them, for quick experiments
│   └── ground_truth/           derived layout and per-frame annotations
└── results/
    ├── enhanced_images/, masks/, contours/, edge_visualisations/
    ├── reconstructed_images/, evaluation_results/
    └── ground_truth/           solved_reference.png and friends
```

### The compatibility score

For two candidate sides, with the second traversed in reverse (two mating sides
run in opposite physical directions):

```
D(a, b) = M(a -> b) + M(b -> a)          [Mahalanobis gradient compatibility]

M(x -> y) = mean_i (r_i - mu)' inv(S) (r_i - mu)
    g_i = x0_i - x1_i     x's own colour gradient from its edge inwards
    mu  = mean_i g_i
    S   = cov(g) + 1e-4 I
    r_i = y0_i - x0_i     the colour change observed across the seam
```

`x0` is the colour strip sampled just under side `x` and `x1` the strip one step
deeper. The score asks how badly the seam contradicts the colour trend each side
was already following, measured under the covariance of that trend — which makes
it blind to the constant brightness offsets that dominate this puzzle's large
white regions.

**The weights are `w_colour = 1.0` and `w_shape = 0.0`.** Shape is a *gate*, not
a term: it decides whether two sides may mate at all and contributes nothing to
the number. This is a measured decision, not a stylistic one. The puzzle is die
cut, so every tab is the same tab; adding shape to the score makes matching
monotonically worse. Rank-1 accuracy over the 116 directed true-neighbour pairs:

| measure | rank-1 | top-3 | median rank |
|---|---|---|---|
| shape only | 5.2% | 6.0% | 28 |
| colour SSD | 22.4% | 34.5% | 8 |
| shape 0.6 + colour 0.4 (the earlier default) | 7.8% | 19.0% | 16 |
| **MGC (used)** | **46.6%** | **58.6%** | **2** |
| MGC 0.95 + shape 0.05 | 44.0% | 56.0% | 2 |
| MGC 0.80 + shape 0.20 | 34.5% | 50.0% | 4 |

Reproduce with `python3 -m scripts.experiment_mgc`. Colour space and sampling
depth were swept too (`python3 -m scripts.experiment_matcher`): BGR at depths
3/8/14 beat Lab, HSV, finer spacings and every blend tried, so those are what
the library uses. Full derivation is in the module docstring of
`src/edge_compatibility.py`.

---

## 5. Known data limitations

Both are properties of the photographs, not of the code, and both are recorded
in `data/ground_truth/layout.json`.

- **Piece 2 cannot be fully recovered.** A black spring clamp rests physically on
  its top-left corner in all 50 photographs, so part of its silhouette is never
  visible and two of its side labels are unreliable.
- **The puzzle is die-cut.** Every tab shares one profile and every blank its
  complement (measured peak 0.27 and −0.25 body units across all 140 sides), so
  side shape constrains the solution but cannot by itself identify which tab
  belongs in which blank. Artwork continuity carries that burden, and most of
  this puzzle's artwork is low-texture white machinery.

---

## 6. Current status

**Working:** everything in §2 and §3 — the from-scratch library, piece
extraction, the verified layout and orientations, the reconstructed reference
image, the per-frame annotations, the test suite, the demonstration notebook, and
the blind assembly search. All 24 conformance checks pass.

### Blind reconstruction

`src/solver.py` is handed an unordered bag of pieces with no identity, position
or orientation information, infers the grid shape itself, and searches over both
position and rotation.

```bash
python3 -m scripts.solve_blind          # shuffle + randomly rotate, then solve
python3 main.py --pieces data/pieces    # the general entry point
```

On the photographed puzzle, blind, at the default settings (~38 s):

| measure | result |
|---|---|
| neighbour accuracy | 34 / 58 seams (58.6%) |
| position accuracy | 24 / 35 cells (68.6%) |
| orientation accuracy | 23 / 35 cells (65.7%) |
| complete reconstruction | **no** |

**This is not a complete reconstruction and is not presented as one.** The grid
shape is inferred correctly and every cell is filled, but a third of the pieces
land in the wrong place. `python3 -m scripts.evaluate_solver` shows why, by
scoring the *true* arrangement under the solver's own objective: from beam width
3 000 upwards the search returns arrangements that score **better** than the
correct one (1735 against 1943). The search is therefore working and the
compatibility measure is the limitation — widening the beam lowers the objective
monotonically while accuracy wanders between 54% and 69%. Raising this figure
needs a better measure, which is what Milestone 2's learned matcher is for.

On a synthetic puzzle cut from a known image, where the measure is not the
bottleneck, the same search reconstructs the puzzle **exactly** — that is what
`tests/test_assembly.py` asserts.

**`report/milestone_1_report.md` is the Milestone 1 submission report**: every
requirement of the brief mapped to the code that fulfils it and the measured
result. `report/milestone_2_report.md` is the Milestone 2 report — the output
contract every model must satisfy, the dataset and training facts the brief asks
for, and each requirement that is still outstanding.
`report/milestone_1_report.pdf` is
the same document typeset. A web version is at
<https://claude.ai/code/artifact/a42f5dbc-e48e-4d42-9d0f-b4c8bb7a7746>.

**Outstanding.** `STATUS.md` is the single status file; the short version:

- **Milestone 2 is substantially outstanding.** Both models are now trained for
  500 epochs on the same split, and at the checkpoints they keep both land at
  40.0% rank-1 against the classical measure's 55.0% on the identical held-out
  candidate set. The GNN's best-validation checkpoint reaches 60.0%, but that is
  12 correct pairs against 11 out of 20 — inside noise, and not the checkpoint the
  run selects. Only the `neighbor_logit` head is trained in either model, and
  nothing model-driven is wired into the solver. Run
  `python3 -m ml.evaluate_models` for the comparison, and see
  `report/milestone_2_report.md` for the requirement-by-requirement state.
- **A better compatibility measure is the open problem**, for the reasons
  measured above. `NEXT_STEPS.md` §3 ranks the options.

---

## 7. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — run from the repository root.
`config.py` resolves every path relative to itself, but the dataset lookup and
the output folders assume the working directory is the project root.
Either `python3 -m scripts.build_ground_truth` or
`python3 scripts/build_ground_truth.py` works.

**`full_frame_paths` returns 0** — the dataset is missing or unpacked to the
wrong place. It must sit at `data/raw/detection/` with `data.yaml` beside the
`images/` and `labels/` folders.

**The build looks stalled** — it prints one line per photograph, roughly 2–3 s
each; the whole run is about three minutes. Progress lines that read
`pieces matched: 33` rather than 35 are normal and expected: some pieces are
occluded in some frames, and the cross-frame combination is what compensates.

**A `RuntimeWarning: All-NaN slice encountered`** during the build is harmless.
It occurs at canvas pixels that no observation of a piece ever covers.
