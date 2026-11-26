"""Recompose each seed, verify BDD artefacts, and pair bootstrap by video."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import run as job

HERE = job.HERE


def load(path):
    return job.boot_stats.load_raw_fields(path)


def analyze():
    frozen = json.loads((HERE / 'inputs.json').read_text())
    names = job.videos()
    for p, h in frozen['sha256'].items():
        assert job.sha(p) == h, p
    all_raw, summary, seeds = {}, {}, {}
    for mode in ('pf100', 'cart500', 'classic500', 'kf', 'ekf', 'ukf'):
        seed_values = range(3) if mode.endswith('500') or mode == 'pf100' else (None,)
        raws, scores = [], []
        for seed in seed_values:
            path = job.classic(seed) if mode == 'classic500' else HERE / f'evaluation/{job.tag(mode, seed)}/raw_fields.json'
            raw = load(path)
            assert set(raw) == set(names)
            value = job.boot_stats.combine(raw, names)
            if mode != 'classic500':
                recorded = json.loads((path.parent / 'combined.json').read_text())
                assert all(abs(recorded[k] - value[k]) < 1e-9 for k in job.boot_stats.METRICS)
            raws.append(raw)
            scores.append(value)
        all_raw[mode] = raws
        seeds[mode] = scores
        summary[mode] = {k: float(np.mean([s[k] for s in scores])) for k in job.boot_stats.METRICS}
    ref_parity = {}
    for mode, historical in (('kf', 'cv'), ('ekf', 'ekf')):
        folder = job.REFERENCE / f'r61-yolov8n_coco-c25-{historical}'
        old = load(folder / 'raw_fields.json')
        ref_parity[mode] = {v: {k: all_raw[mode][0][v][k] == old[v][k] for k in job.boot_stats.SCALAR_FIELDS} for v in names}
        old_score = job.boot_stats.combine(old, names)
        ref_parity[mode]['combined_max_difference'] = max(abs(summary[mode][k] - old_score[k]) for k in job.boot_stats.METRICS)
    job.write(HERE / 'summary.json', {'videos': 200, 'frames': 39973, 'summary': summary, 'seeds': seeds,
                                    'R61_reference_parity': ref_parity})
    units = 0
    for mode in ('pf100', 'cart500', 'kf', 'ekf', 'ukf'):
        for seed in (range(3) if mode in ('pf100', 'cart500') else (None,)):
            folder = HERE / 'tracking' / job.SPLIT / job.tag(mode, seed)
            for name in names:
                marker = json.loads((folder / f'meta/{name}.json').read_text())
                assert marker['frames'] == frozen['frames'][name]
                for kind, suffix, key in (('data', '.txt', 'mot_sha256'), ('data_cls', '.csv', 'classes_sha256'),
                                          ('timing', '.csv', 'timing_sha256')):
                    assert job.sha(folder / kind / f'{name}{suffix}') == marker[key]
                timing = pd.read_csv(folder / f'timing/{name}.csv')
                assert len(timing) == marker['frames'] and timing.n_steps.eq(1).all()
                units += 1
    job.write(HERE / 'verification.json', {'status': 'PASS', 'units': units, 'hashes': len(frozen['sha256']),
              'raw_groups': 9, 'evaluator': 'mot_crowd', 'R61_combined_difference':
              {k: v['combined_max_difference'] for k, v in ref_parity.items()}})
    # Shared paired resamples, combined separately for every seed before mean.
    stacks = {m: [job.boot_stats._stack(r, names) for r in rows] for m, rows in all_raw.items()}
    rng = np.random.default_rng(125)
    fields = ('HOTA', 'IDF1', 'IDSW', 'FN')
    pairs = ('cart500', 'classic500', 'kf', 'ekf', 'ukf')
    samples = {m: {f: [] for f in fields} for m in pairs}
    for i in range(5000):
        idx = rng.integers(0, len(names), size=len(names))
        values = {m: [job.boot_stats.combine_stacked({f: st[f][idx] for f in st}) for st in ss]
                  for m, ss in stacks.items()}
        means = {m: {f: float(np.mean([s[f] for s in v])) for f in fields} for m, v in values.items()}
        for m in pairs:
            for f in fields:
                samples[m][f].append(means['pf100'][f] - means[m][f])
    intervals = {m: {f: {'difference': summary['pf100'][f] - summary[m][f],
                        'ci95': np.quantile(a, [.025, .975]).tolist()} for f, a in fields_.items()}
                 for m, fields_ in samples.items()}
    job.write(HERE / 'bootstrap.json', {'paired_unit': 'video', 'samples': 5000,
                                      'PF100_minus_reference': intervals})
    print('SUMMARY', summary, flush=True)
    print('VERIFIED', units, flush=True)


if __name__ == '__main__':
    analyze()
