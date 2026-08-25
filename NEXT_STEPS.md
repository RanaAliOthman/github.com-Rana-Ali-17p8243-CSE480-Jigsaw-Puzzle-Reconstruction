# Handoff — state of work and what to do next

Last updated: 2026-08-22. Read this together with `STATUS.md` (what is and is not
done), `README.md` (how to run), `report/milestone_2_report.md` (Milestone 2
requirement by requirement, and the output contract a learned measure must plug
into) and `data/ground_truth/layout.json` (derived ground truth and its
evidence).

---

## 1. Where the project stands

The agreed plan has five phases. **Phases 0, 1, 2 and 4 are complete and
verified. Phase 3 — Milestone 2 — is outstanding.**

| Phase | Scope | State |
|---|---|---|
| 0 | Honest ground truth | **Done** |
| 1 | Make the classical solver actually solve, blind | **Done** |
| 2 | Wire the library into the end-to-end routine, close §1–§3 gaps | **Done** |
| 3 | Milestone 2: Siamese CNN + GNN | **Outstanding** |
| 4 | Reports, notebook, docs cleanup | **Done** |

Conformance against the project brief: **24 of 24 checks pass** (was 22). Run
`python3 -m scripts.audit_requirements`.

`python3 -m pytest -q` — **108 tests pass** (was 55), about 15 s.

`notebooks/full_project_demo.ipynb` runs top to bottom in about 70 s and
reproduces every headline figure from the real photographs.

---

## 2. The result on the photographed puzzle, stated plainly

Blind, at the default settings (~38 s):

| measure | result |
|---|---|
| neighbour accuracy | 34 / 58 seams (58.6%) |
| position accuracy | 24 / 35 cells (68.6%) |
| orientation accuracy | 23 / 35 cells (65.7%) |
| complete reconstruction | **no** |

The grid shape (5×7) is inferred correctly and every cell is filled. The
reconstruction is reported after a 180° rotation, which is the one symmetry a
non-square grid's seams genuinely cannot distinguish; both readings are printed
so the choice is visible.

**This is not a complete reconstruction and should not be presented as one.**

    python3 -m scripts.solve_blind          # shuffles + randomly rotates, then solves
    python3 main.py --pieces data/pieces    # the general command-line entry point

### The matcher, and what was ruled out

MGC (Mahalanobis gradient compatibility) on BGR at depths 3/8/14, with shape as a
hard gate and no weighted shape term. Rank-1 over the 116 directed
true-neighbour pairs: **46.6%**, against 22.4% for colour SSD and 7.8% for the
project's earlier 0.6-shape/0.4-colour default. Lab (38.8%), HSV (29.3%), finer
and deeper colour sampling, a longer-baseline gradient, median-combined residuals
and DC-removed SSD blends were all swept and all lose. Reproduce with
`python3 -m scripts.experiment_matcher` and `... --measures`.

### Three solver decisions that mattered, in order of size

1. **The flat-side pattern had to become a cost, not a classification.**
   Classifying each side independently gives the wrong flat count for three of
   the thirty-five pieces, leaving thirteen edge pieces for sixteen edge cells,
   so the search dead-ended before it started. No profile statistic fixes this:
   relief, RMS, span, 90th percentile and several combinations were swept, and
   the best of them still misclassifies pieces 5 and 29. The solver now charges a
   piece for how badly its silhouette disagrees with the cell it is being put
   in, and lets colour overrule an ambiguous silhouette. The counting constraint
   comes free, because the search places each piece exactly once.
2. **The tab/blank gate had to become soft.** The true arrangement contains one
   seam the gate rejects, because piece 2's silhouette is damaged — so with a
   hard gate the correct answer was literally unreachable. It is now a penalty
   of ten typical seam costs.
3. **The beam had to be much wider.** Width 200 → 10 000 took position accuracy
   from 45.7% to 68.6%.

---

## 3. What now limits this puzzle — measured, not assumed

Run `python3 -m scripts.evaluate_solver`. It scores the *true* arrangement under
the solver's own objective and compares it with what the search returns:

| beam width | objective found | mean seam cost | position accuracy |
|---|---|---|---|
| 3 000 | 1735 | 23.83 | 62.9% |
| 10 000 | 1713 | 23.40 | **68.6%** |
| 30 000 | 1688 | 23.22 | 54.3% |
| 60 000 | 1671 | 22.03 | 54.3% |
| *the true arrangement* | *1943* | *19.47* | *100%* |

Two things follow, and the report says both.

- **The search works.** The objective it reaches falls monotonically as the beam
  widens, and from width 3 000 upwards it finds arrangements that score *better*
  than the correct one.
- **The compatibility measure is what is wrong.** Because the truth is not the
  optimum of the objective, minimising the objective harder does not make the
  answer more correct — position accuracy wanders between 54% and 69% instead of
  improving. No amount of extra search fixes this.

Note the tension in the truth's own numbers: its mean *legal* seam cost (19.47)
is clearly better than anything the search finds (22–24), but it pays for one
gate-breaking seam. The measure ranks individual seams well and ranks whole
arrangements badly.

**So: a better measure is the only thing that will raise this figure.** Worth
trying, in order of expected value:

1. A learned pairwise measure — which is exactly Milestone 2 (Phase 3), and the
   labels for it already exist in `data/ground_truth/frame_annotations.json`.
   This is the natural next step and it addresses the actual bottleneck. See §4
   for why the first attempt at it did not work and what to change.
2. Best-buddy driven assembly: build rigid blocks out of mutually-best pairs
   (17 of the 21 the measure finds are true neighbours, 81% precision) and place
   blocks rather than pieces. This changes the search space rather than the
   measure, and is the standard remedy in the literature.
3. Re-deriving the piece images. `src/registry.py` median-combines ~44 views per
   piece; seam colours are the input the measure depends on, and pieces 2, 5, 29
   and 33 are where both the silhouette and the matching go wrong.

---

## 4. Phase 3 — the actual next task

Both models are now trained on the same split for **500 epochs** each, and at
the checkpoints they keep **neither beats the classical measure**: 40.0% rank-1
each against MGC's 55.0% on the identical held-out candidate set.

| Measure | rank-1 | top-3 | median rank |
|---|---|---|---|
| Siamese CNN, epoch 500 | 40.0% | 70.0% | 2 |
| GNN, epoch 500 | 40.0% | 70.0% | 2 |
| **classical MGC** | **55.0%** | **80.0%** | **1** |
| GNN, epoch 225 (best validation) | 60.0% | 80.0% | 1 |

The last row is 12 correct pairs against MGC's 11, out of 20. Treat it as a
reason to add early stopping, not as a result.

Both overfit hard — best validation loss at epoch 223 (Siamese, 0.52) and 225
(GNN, 0.66), then an order-of-magnitude degradation while training loss collapses
to 0.13 and 0.0006. **84 training positives is the problem**, not the
architecture or the schedule — one puzzle cannot supply more. Three things to do,
in order:

1. **Grow the training set synthetically.** `src/synthetic.py` cuts a real
   interlocking puzzle of any size from any image, with the true adjacencies
   known by construction. Thousands of positives are available this way, and the
   held-out evaluation on the real puzzle stays honest.
2. **Stop at the best validation loss.** Both runs keep the last epoch. The GNN
   shows exactly what that costs: 60.0% rank-1 at epoch 225 against 40.0% at
   epoch 500. Both trainers now *save* the best-validation weights; neither
   selects them.
3. **Only then wire it into the solver**, and only if it beats MGC on the real
   held-out block. `src/solver.py` takes its costs from
   `edge_compatibility.compatibility_matrices`, so a learned measure substitutes
   there.

`report/milestone_2_report.md` states all of this requirement by requirement,
including the two facts that most need fixing: the run keeps the last epoch
rather than the best, and no unseen *testing* set exists — the held-out block
serves as validation and as the reported evaluation.

Also outstanding in Phase 3: `orientation_logits` and `compatibility` are still
at their random initialisation and **must not be reported**; no model score has
ever been passed to the assembly algorithm; the Siamese has no best-validation
checkpoint because its best epoch predates best-state tracking; and no unseen
*testing* set exists. `python3 -m ml.evaluate_models` writes the
comparison as it currently stands.

---

## 5. Scripts

| Command | Purpose | Runtime |
|---|---|---|
| `python3 -m scripts.solve_blind` | Scramble the pieces and reconstruct them blind | ~40 s |
| `python3 main.py --pieces data/pieces` | The general command-line entry point | ~40 s |
| `python3 -m pytest -q` | 108 unit tests | ~15 s |
| `python3 -m scripts.audit_requirements` | Conformance against the brief (24/24) | ~40 s |
| `python3 -m ml.train_siamese --epochs 500` | Train the Siamese pair classifier (`--resume` continues) | ~140 s |
| `python3 -m ml.train_gnn --epochs 500` | Train the GNN on the same split | ~94 s |
| `python3 -m ml.evaluate_models` | Both models vs classical, from their reports | instant |
| `python3 -m scripts.evaluate_solver` | Accuracy, and whether search or measure is the limit | ~6 min |
| `python3 -m scripts.tune_solver` | How the search responds to its parameters | ~5 min |
| `python3 -m scripts.build_ground_truth` | Rebuild all ground truth from the dataset | ~170 s |
| `python3 -m scripts.evaluate_matcher` | True-neighbour rank of the current matcher | ~60 s |
| `python3 -m scripts.experiment_mgc` | Compare SSD / DC-removed / MGC variants | ~90 s |
| `python3 -m scripts.experiment_matcher` | Sweep colour space, depth and measure variants | ~10 min |
| `python3 -m scripts.build_report_pdf` | Typeset every `report/*_report.md` | ~10 s |

---

## 6. What Phase 4 removed

Deleted, because nothing in the current path
(`frame_extraction` → `piece_geometry` → `edge_compatibility` → `solver`) used
them: `src/pipeline.py`, `src/dataset_reconstruction.py`, `src/solver_v2.py`,
`src/assembly.py`, `src/piece_description.py`, `src/edge_matching.py`,
`src/rendering.py`, `scripts/export_piece_review.py`,
`scripts/export_photoshop_pieces.py`, `scripts/assemble_review_candidate.py`,
`tests/test_core.py`, and the dead `internal_metrics` / `save_metrics` /
`layout_metrics` helpers in `src/evaluation.py`.

`FINAL_STATUS.md`, `HOW_TO_RUN.md`, `DISCUSSION_GUIDE.md`,
`SOLVING_REQUIREMENTS.md` and `RECONSTRUCTION_ANALYSIS.md` were folded into
`STATUS.md`, which lists the specific claims they made that the evidence has
since contradicted.

Left alone deliberately: `results/photoshop_numbered_35*`,
`results/photoshop_35_transparent/` and `results/piece_review/` are output
artefacts of the superseded manual-cutout approach. They are not referenced by
any code and can be deleted whenever you want the space back.

---

## 7. Disposable

Everything under the session scratchpad in `/tmp` is regenerable and safe to
lose. `scripts/build_ground_truth.py` recreates all derived data from the
dataset in one deterministic run. Nothing outside the repository is needed to
resume.

---

## 8. Open questions for the user

- Phase 4 is finished and the repository now conforms to the brief in full.
  The remaining work is Milestone 2, and §4 says what to change about it.
- The blind reconstruction plateaus at 68.6% position accuracy and the evidence
  in §3 says the compatibility measure, not the search, is the cause. Reporting
  it as-is with that analysis, and putting the remaining effort into a learned
  measure trained on synthetic puzzles, is the recommendation.
