"""Measure the blind solver honestly, and locate what limits it.

    python3 -m scripts.evaluate_solver

Two things are reported.

1. **Accuracy over several scrambles.**  The puzzle is fixed, but the order the
   pieces arrive in and the rotation each arrives at are not, and they change
   the search's tie-breaks.  Reporting one scramble would be reporting noise.

2. **The decisive diagnostic: is the search or the objective at fault?**  The
   true arrangement is scored under the solver's own objective and compared
   with the arrangement the solver returned.  If the solver's answer costs
   *less* than the truth, the search is working and the compatibility measure
   is wrong; no amount of extra search effort can then help.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.solve_blind import load_scrambled, score
from src.solver import (arrangement_report, build_objective, build_variants,
                        side_reliefs, solve)

ROWS, COLS = 5, 7
SEEDS = (480, 1, 2)
SETTINGS = ([dict(order=o, flat_weight=w, gate_penalty_factor=g,
                  beam_width=200, per_state=6)
             for o in ('border', 'raster')
             for w in (1.0, 2.0)
             for g in (3.0, 10.0)]
            + [dict(order='raster', flat_weight=1.0, gate_penalty_factor=10.0,
                    beam_width=b, per_state=p)
               for b, p in ((50, 4), (800, 8), (3000, 12))])


def truth_cells(pieces, applied, layout):
    """The true arrangement, expressed as ``(piece index, rotation)`` per cell."""
    index_of = {piece.id: i for i, piece in enumerate(pieces)}
    cells = []
    for cell in range(ROWS * COLS):
        pid = layout['positions_piece_id'][cell]
        solved = int(layout['solved_rotations_ccw'][str(pid)])
        cells.append((index_of[pid], (solved - applied[pid]) % 4))
    return tuple(cells)


def main():
    layout = json.loads(Path('data/ground_truth/layout.json').read_text())
    rows = {tuple(sorted(s.items())): [] for s in SETTINGS}
    verdicts = []

    for seed in SEEDS:
        pieces, applied = load_scrambled(seed)
        variants = build_variants(pieces)
        relief = side_reliefs(variants)
        truth = truth_cells(pieces, applied, layout)

        for setting in SETTINGS:
            objective = build_objective(variants, ROWS, COLS,
                                        setting['flat_weight'], relief,
                                        gate_penalty_factor=setting['gate_penalty_factor'])
            tensors, label_costs, patterns, gate_penalty, _ = objective
            result = solve(pieces, ROWS, COLS, variants=variants,
                           tensors=tensors, relief=relief, **setting)
            accuracy = score(result, applied, layout)['reported']
            rows[tuple(sorted(setting.items()))].append(accuracy)

            found = tuple(None if p is None else (p.index, p.rotation)
                          for p in result.placements)
            args = (tensors, label_costs, patterns, ROWS, COLS, gate_penalty)
            verdicts.append((seed, setting,
                             arrangement_report(found, *args),
                             arrangement_report(truth, *args)))

    print('Accuracy over %d scrambles (mean +- spread)\n' % len(SEEDS))
    print('%-46s %-16s %-16s' % ('setting', 'neighbour', 'position'))
    for setting in SETTINGS:
        got = rows[tuple(sorted(setting.items()))]
        neighbour = np.array([a['neighbour_accuracy'] for a in got])
        position = np.array([a['position_accuracy'] for a in got])
        print('%-46s %5.1f%% +- %4.1f   %5.1f%% +- %4.1f'
              % (', '.join('%s=%s' % kv for kv in setting.items()),
                 100 * neighbour.mean(), 100 * neighbour.std(),
                 100 * position.mean(), 100 * position.std()))

    print('\nIs the search or the objective at fault?')
    print('The true arrangement is scored under the solver\'s own objective.')
    print('%-30s %10s %10s %8s %8s'
          % ('setting (seed 480)', 'solver', 'truth', 'solver', 'truth'))
    print('%-30s %10s %10s %8s %8s'
          % ('', 'objective', 'objective', 'mean seam', 'mean seam'))
    beaten = 0
    for seed, setting, found, truth_report in verdicts:
        if seed != SEEDS[0]:
            beaten += found['total'] < truth_report['total']
            continue
        beaten += found['total'] < truth_report['total']
        print('%-30s %10.1f %10.1f %8.2f %8.2f   gate breaks %d vs %d'
              % ('order=%s w=%g g=%g' % (setting['order'], setting['flat_weight'],
                                         setting['gate_penalty_factor']),
                 found['total'], truth_report['total'],
                 found['mean_legal_seam'], truth_report['mean_legal_seam'],
                 found['gate_violations'], truth_report['gate_violations']))
    print('\nthe search beat the true arrangement in %d of %d runs'
          % (beaten, len(verdicts)))
    if beaten == len(verdicts):
        print('=> the search is not the limitation; the compatibility measure is.')


if __name__ == '__main__':
    main()
