"""Serial CPU update timing; run after quality processes have exited."""
import json
import numpy as np
import run as job


def measure():
    assert (job.HERE / 'verification.json').is_file()
    assert not (job.HERE / 'serial_timing.json').exists()
    frozen = json.loads((job.HERE / 'inputs.json').read_text())
    for p, h in frozen['sha256'].items():
        assert job.sha(p) == h, p
    modes = ('kf', 'classic500', 'cart500', 'pf100')
    rows, signatures = [], {}
    for repeat in range(3):
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for name in frozen['timing_videos']:
            length = min(200, frozen['frames'][name])
            for mode in order:
                seed = None if mode == 'kf' else 0
                mot, _, timing = job.replay(name, mode, seed, length)
                values, key = mot.to_numpy(float), (name, mode)
                if key in signatures:
                    assert np.array_equal(values, signatures[key])
                else:
                    signatures[key] = values
                if mode != 'classic500':
                    full = job.HERE / 'tracking' / job.SPLIT / job.tag(mode, seed) / f'data/{name}.txt'
                    old = np.loadtxt(full, delimiter=',', ndmin=2)
                    assert np.allclose(values, old[old[:, 0] <= length], atol=1e-9, rtol=0)
                a = timing.t_update_ms.to_numpy()[16:]
                row = {'repeat': repeat, 'video': name, 'mode': mode, 'frames': len(a),
                       'mean_update_ms': float(a.mean()), 'p90_update_ms': float(np.quantile(a, .9))}
                rows.append(row)
                print(row, flush=True)
    means = {m: float(np.mean([r['mean_update_ms'] for r in rows if r['mode'] == m])) for m in modes}
    for p, h in frozen['sha256'].items():
        assert job.sha(p) == h, p
    job.write(job.HERE / 'serial_timing.json', {'rows': rows, 'mean_update_ms': means,
              'relative_time_reduction_percent': {m: 100 * (1 - means['pf100'] / means[m]) for m in modes if m != 'pf100'},
              'scope': 'PC tracker update only, two preselected BDD clips, 16 warmup, three reversed orders; not Nano FPS'})
    print('TIMING', means, flush=True)


if __name__ == '__main__':
    measure()
