"""Posthoc paired-video uncertainty for additive counts; never recompute HOTA as a mean."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = HERE / 'tracking_uncertainty.json'
    assert not target.exists()
    source = HERE / 'final_summary.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    assert result['status'] == 'PASS'
    quality = result['quality']
    names = {dataset: sorted(quality[dataset]['original']['videos']) for dataset in ('bdd', 'ua')}
    assert all(len(roster) == 40 for roster in names.values())
    for dataset, roster in names.items():
        for mode in ('original', 'control', 'precise', 'fast'):
            group = quality[dataset][mode]
            assert sorted(group['videos']) == roster
            for metric in ('idsw', 'fn', 'fp'):
                assert sum(group['videos'][name][metric] for name in roster) == group['combined'][metric]
    rng = np.random.default_rng(167)
    pairs = [('control', 'original'), ('precise', 'original'), ('fast', 'original'),
             ('precise', 'control'), ('fast', 'control')]
    rows = {}
    for dataset in ('bdd', 'ua', 'all'):
        roster = [(label, name) for label in ('bdd', 'ua') if dataset in (label, 'all') for name in names[label]]
        if dataset == 'all':
            draws = np.concatenate([rng.integers(0, 40, size=(5000, 40)),
                                    40 + rng.integers(0, 40, size=(5000, 40))], axis=1)
        else:
            draws = rng.integers(0, 40, size=(5000, 40))
        rows[dataset] = {}
        for mode, reference in pairs:
            comparison = {}
            for metric in ('idsw', 'fn', 'fp'):
                delta = np.array([quality[label][mode]['videos'][name][metric] -
                                  quality[label][reference]['videos'][name][metric] for label, name in roster])
                assert np.isfinite(delta).all()
                bootstrap = delta[draws].sum(axis=1)
                comparison[metric] = {
                    'sum_delta': int(delta.sum()), 'CI95_bootstrap_sum_delta': np.quantile(bootstrap, [.025, .975]).tolist(),
                    'videos_increased': int((delta > 0).sum()), 'videos_decreased': int((delta < 0).sum()),
                    'videos_equal': int((delta == 0).sum()),
                    'max_video_increase': int(delta.max()), 'max_video_decrease': int(delta.min()),
                }
            rows[dataset][mode + '_minus_' + reference] = comparison
    target.write_text(json.dumps({
        'status': 'PASS', 'meaning': 'Posthoc paired-video additive-count uncertainty, not a new quality gate',
        'seed': 167, 'replicates': 5000, 'quality_frames_per_clip': 150,
        'sampling': '40 videos/dataset with replacement; all80 preserves separate 40+40 strata; same draws for all modes',
        'results': rows,
        'limitations': ['Fixed, repeatedly seen development roster; not a blind population sample.',
                        'Multiple exploratory comparisons; no family-wise significance or non-inferiority claim.',
                        'HOTA is kept as the original TrackEval aggregate; no mean-of-video substitute.'],
        'sha256': {str(path): sha(path) for path in (Path(__file__), source)},
    }, indent=2) + '\n', encoding='utf-8')
    print('R167 PAIRED-VIDEO COUNT AUDIT COMPLETE', flush=True)
    for dataset in ('bdd', 'ua'):
        print(dataset, rows[dataset]['precise_minus_original']['idsw'], flush=True)


if __name__ == '__main__':
    main()
