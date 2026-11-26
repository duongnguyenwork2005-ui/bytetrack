"""Reuse the R118 runner, reference the R118 Cartesian parent, isolate outputs."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'review118_pf_shape'))
import run as job
from shape_robust import installed
import numpy as np
from pf_ctrv import PFCTRV
from cartesian import CartesianPF
from shape_robust import ShapeRobustPF, inflate

PARENT = job.HERE
job.HERE = HERE
job.installed = installed
job.parent = lambda seed: PARENT / f'dev/evaluation/cartesian-s{seed}/raw_fields.json'


def preflight():
    assert not (HERE / 'inputs.json').exists()
    prior = json.loads((PARENT / 'inputs.json').read_text())
    paths = [Path(p) for p in prior['sha256']]
    paths += [Path(__file__), HERE / 'shape_robust.py', HERE / '00_protocol.md', PARENT / 'run.py']
    paths += [job.parent(s) for s in range(3)]
    summaries = [job.base.boot_stats.combine(job.base.boot_stats.load_raw_fields(job.parent(s)), job.base.DEV) for s in range(3)]
    job.write(HERE / 'inputs.json', {'frames': prior['frames'], 'baseline_seeds': summaries,
                                   'sha256': {str(p): job.base.sha(p) for p in paths}})


def checks():
    assert np.array_equal(inflate(np.array([0., 1., 10.]), np.ones(3)), [1., 1., 20.8])
    pf = PFCTRV(num_particles=20)
    mean, cov = pf.initiate(np.array([50, 70, 1.5, 80]))
    particles = np.tile(mean, (20, 1))
    def new(cls):
        return cls(pf, mean, cov, particles, np.random.default_rng(4))
    a, b = new(CartesianPF), new(ShapeRobustPF)
    a.predict(); b.predict()
    z = np.array([51., 70., 1.5, 80.])
    x, p = a.update(z); y, q = b.update(z)
    assert np.array_equal(x, y) and np.array_equal(p, q)
    assert '_measurement_noise' not in pf.__dict__
    a, b = new(CartesianPF), new(ShapeRobustPF)
    a.predict(); b.predict()
    z = np.array([51., 70., .1, 20.])
    a.update(z); b.update(z)
    assert np.array_equal(a.means, b.means) and np.array_equal(a.logw, b.logw)
    assert b.shape_mean[1, 0] > a.shape_mean[1, 0]
    assert np.linalg.eigvalsh(b.estimate()[1]).min() > -1e-8
    for s in range(3):
        for video in job.base.SMOKE:
            mot, _ = job.replay(video, s, False, 128)
            old = np.loadtxt(PARENT / f'dev/tracking/DETRAC-all/cartesian-s{s}/data/{video}.txt', delimiter=',')
            assert np.allclose(mot.to_numpy(float), old[old[:, 0] <= 128], atol=1e-9, rtol=0)
            x, _ = job.replay(video, s, True, 128)
            y, _ = job.replay(video, s, True, 128)
            assert np.array_equal(x.to_numpy(float), y.to_numpy(float))
    job.write(HERE / 'smoke.json', {'status': 'PASS', 'parent_parity': 6, 'repeat': 6,
                                 'position_unchanged': True, 'robust_shape_gain': True})
    print('CHECKS PASS', flush=True)


def verify():
    job.summary()
    frozen = json.loads((HERE / 'inputs.json').read_text())
    for path, digest in frozen['sha256'].items():
        assert job.base.sha(Path(path)) == digest, path
    units = 0
    for seed in range(3):
        raw = job.base.boot_stats.load_raw_fields(HERE / f'dev/evaluation/cartesian-s{seed}/raw_fields.json')
        assert set(raw) == set(job.base.DEV)
        for video in job.base.DEV:
            folder = HERE / f'dev/tracking/DETRAC-all/cartesian-s{seed}'
            marker = json.loads((folder / f'meta/{video}.json').read_text())
            assert marker['frames'] == frozen['frames'][video]
            assert marker['mot_sha256'] == job.base.sha(folder / f'data/{video}.txt')
            units += 1
    job.write(HERE / 'verification.json', {'status': 'PASS', 'units': units,
                                         'source_hashes': len(frozen['sha256'])})


if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        job.dev(int(sys.argv[2]))
    else:
        {'preflight': preflight, 'checks': checks, 'verify': verify}[sys.argv[1]]()
