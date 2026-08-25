"""Compare both learned measures with the classical one, honestly.

    python3 -m ml.evaluate_models

This reads the reports written by ``python3 -m ml.train_siamese`` and
``python3 -m ml.train_gnn`` and restates them side by side, so one file in
``results/evaluation_results`` carries the comparison the brief asks for. It
computes nothing new: every rank was produced by a training run on its own
held-out block, where the classical measure was scored over the identical
candidate set by the single protocol in :mod:`ml.ranking`.

An earlier version of this script wrote ``not_trainable_from_supplied_labels``.
That was true only before ``data/ground_truth/`` existed; the adjacencies are now
derived and both models train on them.
"""
import json
from pathlib import Path

RESULTS = Path('results/evaluation_results')
REPORTS = {'siamese': RESULTS / 'siamese_training.json',
           'gnn': RESULTS / 'gnn_training.json'}
OUTPUT = RESULTS / 'model_comparison.json'


def _model_entry(report):
    """Pull the comparable figures out of one training report."""
    ranks = report['held_out_block_rank']
    return {
        'epochs': report['epochs'],
        'parameters': report.get('parameters'),
        'training_seconds': round(report['runtime_seconds'], 1),
        'device': report['device'],
        'hyperparameters': report.get('hyperparameters'),
        'best_epoch': report.get('best_epoch'),
        'best_valid_loss': report.get('best_valid_loss'),
        'final_valid_loss': report['history'][-1]['valid_loss'],
        'final_recall_on_true_seams': report['history'][-1]['valid_recall_on_true_seams'],
        'rank_final_epoch': ranks.get('learned_final_epoch'),
        'rank_best_valid': ranks.get('learned_best_valid'),
        'trained_heads': report['trained_heads'],
        'untrained_heads': report['untrained_heads'],
    }


def compare(reports=None, output=OUTPUT):
    """Summarise every trained model against the classical measure."""
    reports = reports or REPORTS
    loaded = {}
    for name, path in reports.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                '%s is missing; run `python3 -m ml.train_%s` first' % (path, name))
        loaded[name] = json.loads(Path(path).read_text())

    classical = [r['held_out_block_rank']['classical_mgc'] for r in loaded.values()]
    if any(c != classical[0] for c in classical):
        raise ValueError('the classical baseline differs between reports; the two '
                         'models were not evaluated on the same candidate set')

    models = {name: _model_entry(report) for name, report in loaded.items()}
    splits = {name: report['dataset']['held_out_cells'] for name, report in loaded.items()}
    if any(cells != next(iter(splits.values())) for cells in splits.values()):
        raise ValueError('the models were trained on different splits')

    best = max(
        [('classical MGC', classical[0]['rank1'])] +
        [('%s (%s)' % (name, 'best-validation checkpoint' if entry['rank_best_valid']
                       and entry['rank_best_valid']['rank1'] >= entry['rank_final_epoch']['rank1']
                       else 'final epoch'),
          max(entry['rank_final_epoch']['rank1'],
              (entry['rank_best_valid'] or {}).get('rank1', 0.0)))
         for name, entry in models.items()],
        key=lambda item: item[1])

    summary = {
        'sources': {name: str(path) for name, path in reports.items()},
        'protocol': (
            'One 5x7 puzzle. A 2x4 block of cells is held out; every true seam '
            'inside it is scored against the same gated candidate set by every '
            'measure, through ml/ranking.py. Rank 1 means the true partner '
            'scored best.'),
        'held_out_cells': next(iter(splits.values())),
        'pairs_scored': classical[0]['pairs_scored'],
        'training_set_size': {
            'positives': next(iter(loaded.values()))['dataset']['train_positive'],
            'negatives': next(iter(loaded.values()))['dataset']['train_negative'],
        },
        'models': models,
        'classical_mgc': classical[0],
        'best_measure': {'name': best[0], 'rank1': best[1]},
        'caveats': [
            next(iter(loaded.values()))['caveat'],
            'Only %d pairs are scored, so one pair is 5 percentage points. '
            'Differences of this size are not statistically meaningful.'
            % classical[0]['pairs_scored'],
            'Only the neighbour head is trained in either model; the orientation '
            'and compatibility outputs are at random initialisation and must not '
            'be reported.',
            'No model score has been passed to the assembly algorithm, so no '
            'learned reconstruction, position accuracy or quality score exists.',
        ],
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    print(json.dumps(compare(), indent=2))
