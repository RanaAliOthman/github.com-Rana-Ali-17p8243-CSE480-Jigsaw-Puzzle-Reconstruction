"""How the blind search responds to its parameters.

    python3 -m scripts.tune_solver

The piece descriptions and the join-cost tensors are computed once and shared
across every setting, so this measures the search alone.  Ground truth is read
only to score the results.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.solve_blind import load_scrambled, score
from src.solver import build_variants, cost_tensors, side_reliefs, solve

SETTINGS = [dict(beam_width=b, per_state=p, flat_weight=w, order=o)
            for o in ('border', 'raster', 'growth')
            for w in (1.0, 2.0, 5.0)
            for b, p in ((200, 6), (800, 8))]


def main():
    layout = json.loads(Path('data/ground_truth/layout.json').read_text())
    pieces, applied = load_scrambled(480)
    variants = build_variants(pieces)
    relief = side_reliefs(variants)
    tensors = cost_tensors(variants, relief)
    print('%-52s %-13s %-13s %-8s'
          % ('setting', 'neighbour', 'position', 'quality'))
    for setting in SETTINGS:
        result = solve(pieces, variants=variants, tensors=tensors, relief=relief,
                       **setting)
        accuracy = score(result, applied, layout)['reported']
        print('%-52s %2d/58 %5.1f%%  %2d/35 %5.1f%%  %5.3f'
              % (', '.join('%s=%s' % kv for kv in setting.items()),
                 accuracy['neighbour_hits'], 100 * accuracy['neighbour_accuracy'],
                 accuracy['position_hits'], 100 * accuracy['position_accuracy'],
                 result.quality['quality_score']))


if __name__ == '__main__':
    main()
