"""Model-preserving numerical checks before any speed or tracking claims."""
from pathlib import Path
import sys
import json
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]/'src'))
from fast import FastCartesianPF, CartesianPF, solve2, installed
from pf_ctrv import PFCTRV
import rbpf_ctrv


def main():
    rng = np.random.default_rng(126)
    max_error = 0.
    for scale in (1e-8, 1., 1e8):
        b = rng.normal(size=(500, 2, 2))*np.sqrt(scale)
        s = b@b.transpose(0, 2, 1)+np.eye(2)[None]*scale*.01
        residual, cross = rng.normal(size=(500, 2)), rng.normal(size=(500, 4, 2))
        white, logc, gain = solve2(s, residual, cross)
        chol = np.linalg.cholesky(s)
        oldw = np.linalg.solve(chol, residual[..., None])[..., 0]
        oldg = np.linalg.solve(s, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
        assert np.allclose(white, oldw, atol=1e-7, rtol=1e-10)
        assert np.allclose(gain, oldg, atol=1e-5, rtol=1e-10)
        assert np.allclose(logc, np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(1), atol=1e-10)
    near = np.array([[[1., 1.], [1., 1.+1e-14]]])
    cross, residual = np.ones((1, 4, 2)), np.ones((1, 2))
    _, _, observed = solve2(near, residual, cross)
    expected = np.linalg.solve(near, cross.transpose(0, 2, 1)).transpose(0, 2, 1)
    assert np.array_equal(observed, expected), 'ill-condition fallback'
    for seed in range(3):
        pf = PFCTRV(num_particles=100)
        mean, cov = pf.initiate(np.array([50., 80., 1.5, 70.]))
        particles = pf.sample_initial(mean, cov, np.random.default_rng(seed))
        a = CartesianPF(pf, mean, cov, particles, np.random.default_rng(seed+10))
        b = FastCartesianPF(pf, mean, cov, particles, np.random.default_rng(seed+10))
        for t in range(120):
            for method, args in [('predict', (t%11 in (0, 1, 2),)),
                                 ('update', (np.array([50+2*t, 80+10*np.sin(t/10), 1.5, 70+2*np.sin(t/20)]),))]:
                x, p = getattr(a, method)(*args)
                y, q = getattr(b, method)(*args)
                max_error = max(max_error, float(np.max(np.abs(x-y))))
                assert np.allclose(x, y, atol=1e-8, rtol=1e-8), (seed,t,method,'mean')
                assert np.allclose(p, q, atol=1e-7, rtol=1e-7), (seed,t,method,'cov')
                assert np.allclose(a.means, b.means, atol=1e-8, rtol=1e-8)
                assert np.allclose(a.covariances, b.covariances, atol=1e-7, rtol=1e-7)
                assert np.array_equal(a.omega, b.omega)
                assert a.rng.bit_generator.state == b.rng.bit_generator.state
                assert np.linalg.eigvalsh(q).min() >= -1e-8
            z = b.full_particles()
            z[:] = -1000
            assert not (b.full_particles() == -1000).all()
    original = rbpf_ctrv.ConditionalPF
    with installed():assert rbpf_ctrv.ConditionalPF is FastCartesianPF
    assert rbpf_ctrv.ConditionalPF is original
    (HERE/'numerical.json').write_text(json.dumps({'status':'PASS','max_state_error':max_error,'steps':720,
          'solver_cases':1500,'RNG_parity':True,'PSD':True,'copy_ownership':True,'hook_restore':True},indent=2))
    print('NUMERICAL PASS',max_error,flush=True)


if __name__=='__main__':main()
