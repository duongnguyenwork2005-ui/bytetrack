"""Serial four-filter PC timing after quality completion, with R125 MOT parity."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import native_job as native
job = native.base


def main():
    assert json.loads((HERE/'verification.json').read_text())['raw_metric_parity'] == 600
    assert not (HERE/'all_filters_timing.json').exists()
    frozen = json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():
        assert job.bdd.sha(p) == h, p
    paths = [Path(__file__), HERE/'01_all_filters_timing_protocol.md']
    hashes = {str(p): job.bdd.sha(p) for p in paths}
    (HERE/'all_filters_timing_inputs.json').write_text(json.dumps({'sha256': hashes}, indent=2))
    modes = ('kf', 'ekf', 'ukf', 'fast100')
    rows, signatures = [], {}
    for repeat in range(3):
        for video in frozen['timing_videos']:
            for mode in modes if repeat%2 == 0 else tuple(reversed(modes)):
                mot, _, timing = job.replay(video, mode, 0, 200)
                values = mot.to_numpy(float)
                tag = 'pf100' if mode == 'fast100' else mode
                tag = job.bdd.tag(tag, 0 if mode == 'fast100' else None)
                reference = job.array(job.BASE/f'tracking/{job.bdd.SPLIT}/{tag}/data/{video}.txt')
                reference = reference[reference[:,0] <= 200]
                assert values.shape == reference.shape
                assert np.array_equal(values[:,:2], reference[:,:2])
                assert np.allclose(values[:,6], reference[:,6], atol=1e-9, rtol=0)
                assert np.allclose(values[:,2:6], reference[:,2:6], atol=1e-3, rtol=0)
                key = (video, mode)
                if key in signatures:
                    assert np.array_equal(values, signatures[key])
                else:
                    signatures[key] = values
                elapsed = timing.t_update_ms.to_numpy()[16:]
                assert len(elapsed) == 184
                row = {'repeat': repeat, 'video': video, 'mode': mode, 'frames': len(elapsed),
                       'mean_update_ms': float(elapsed.mean()), 'p90_update_ms': float(np.quantile(elapsed,.9))}
                rows.append(row)
                print(row, flush=True)
    means = {m: float(np.mean([r['mean_update_ms'] for r in rows if r['mode']==m])) for m in modes}
    for p,h in {**frozen['sha256'], **hashes}.items():
        assert job.bdd.sha(p) == h, p
    (HERE/'all_filters_timing.json').write_text(json.dumps({'rows': rows, 'mean_update_ms': means,
        'prefix_parity': 24, 'scope': 'two BDD clips, PC tracker.update, serial; not Nano FPS'}, indent=2))
    print('ALL FILTERS TIMING PASS', means, flush=True)


if __name__ == '__main__':
    main()
