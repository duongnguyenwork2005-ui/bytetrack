"""Aggregate verified quality preservation and paired serial PC timings."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent/'review125_pf_bdd'


def read(path):
    return json.loads(path.read_text())


def main():
    verified = read(HERE/'verification.json')
    assert verified['status'] == 'PASS' and verified['raw_metric_parity'] == verified['units'] == 600
    timing = read(HERE/'serial_timing.json')
    assert len(timing['rows']) == 18 and all(timing['gates'].values())
    numerical, direct_c = read(HERE/'numerical.json'), read(HERE/'c_api_checks.json')
    assert numerical['status'] == direct_c['status'] == 'PASS'
    for path, digest in direct_c['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    old = read(OLD/'summary.json')
    quality = {k: float(np.mean([row[k] for row in verified['seeds']])) for k in verified['seeds'][0]}
    assert quality == old['summary']['pf100']
    rows = {(row['repeat'], row['video'], row['mode']): row for row in timing['rows']}
    pairs = {}
    for reference in ('parent100', 'classic500'):
        changes = []
        for repeat in range(3):
            for video in read(HERE/'inputs.json')['timing_videos']:
                fast, before = rows[repeat, video, 'fast100'], rows[repeat, video, reference]
                assert fast['frames'] == before['frames'] == 184
                changes.append({'repeat': repeat, 'video': video,
                                'update_time_reduction_pct': 100*(1-fast['mean_update_ms']/before['mean_update_ms'])})
        means = timing['mean_update_ms']
        pairs[reference] = {'reduction_of_mean_update_time_pct': 100*(1-means['fast100']/means[reference]),
                            'range_paired_reduction_pct': [min(r['update_time_reduction_pct'] for r in changes),
                                                           max(r['update_time_reduction_pct'] for r in changes)],
                            'faster_pairs': sum(r['update_time_reduction_pct'] > 0 for r in changes),
                            'pairs': changes}
    result = {'status': 'PASS', 'videos': old['videos'], 'frames_per_seed': old['frames'],
              'quality_units': verified['units'], 'raw_metric_parity': verified['raw_metric_parity'],
              'max_box_error': verified['max_box_error'], 'quality': quality,
              'references': old['summary'], 'mean_update_ms': timing['mean_update_ms'],
              'paired_timing': pairs, 'scope': timing['scope'], 'default_changed': False}
    (HERE/'summary.json').write_text(json.dumps(result, indent=2))
    files = [HERE/f for f in ('verification.json', 'serial_timing.json', 'numerical.json',
                             'c_api_checks.json', 'smoke.json', 'summary.json', 'inputs.json', 'finalize.py')]
    final = {'status': 'PASS', 'quality_units': 600, 'timing_runs': 18, 'compiled_update_cases': 1500,
             'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (HERE/'final_verification.json').write_text(json.dumps(final, indent=2))
    print('FINAL PASS', timing['mean_update_ms'], {k: v['reduction_of_mean_update_time_pct'] for k,v in pairs.items()}, flush=True)


if __name__ == '__main__':
    main()
