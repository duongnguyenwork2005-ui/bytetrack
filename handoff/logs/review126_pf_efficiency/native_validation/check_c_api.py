"""Exercise the compiled C update, including its non-mutating fallback path."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]/'src'))
from native_pf import lib, LIB


def main():
    rng = np.random.default_rng(126002)
    largest = 0.0
    for scale in (1e-8, 1.0, 1e8):
        n = 500
        b = rng.normal(size=(n, 4, 4)) * np.sqrt(scale)
        p = np.ascontiguousarray(b @ b.transpose(0, 2, 1) + np.eye(4)[None] * scale * .01)
        m = np.ascontiguousarray(rng.normal(size=(n, 4)) * np.sqrt(scale))
        z = np.ascontiguousarray(rng.normal(size=4) * np.sqrt(scale))
        r = np.ascontiguousarray(np.diag([.2, .4, .3, .5]) * scale)
        original_m, original_p = m.copy(), p.copy()
        out_m, out_p = np.empty_like(m), np.empty_like(p)
        white, diag = np.empty((n, 2)), np.empty((n, 2))
        assert lib.update(n, m, p, z, r, out_m, out_p, white, diag) == 0
        s = p[:, :2, :2] + r[:2, :2]
        residual = z[:2] - m[:, :2]
        chol = np.linalg.cholesky(s)
        expected_white = np.linalg.solve(chol, residual[..., None])[..., 0]
        gain = np.linalg.solve(s, p[:, :, :2].transpose(0, 2, 1)).transpose(0, 2, 1)
        expected_m = m + (gain @ residual[..., None])[..., 0]
        correction = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        correction[:, :, :2] -= gain
        expected_p = correction @ p @ correction.transpose(0, 2, 1) + gain @ r[:2, :2] @ gain.transpose(0, 2, 1)
        expected_p = .5 * (expected_p + expected_p.transpose(0, 2, 1))
        assert np.allclose(out_m, expected_m, atol=1e-10*np.sqrt(scale), rtol=1e-10)
        assert np.allclose(out_p, expected_p, atol=1e-10*scale, rtol=1e-10)
        assert np.allclose(white, expected_white, atol=1e-10, rtol=1e-10)
        assert np.allclose(diag, np.diagonal(chol, axis1=1, axis2=2), atol=1e-10*np.sqrt(scale), rtol=1e-10)
        assert np.array_equal(m, original_m) and np.array_equal(p, original_p)
        assert np.linalg.eigvalsh(out_p).min() >= -1e-10*scale
        largest = max(largest, float(np.max(abs(out_p-expected_p))) / scale)

    # Reject the third particle after writing scratch entries for the first two.
    # The Python fallback needs the entire input arrays to remain untouched.
    m = np.ascontiguousarray(rng.normal(size=(3, 4)))
    p = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    p[2, :2, :2] = [[1., 1.], [1., 1.+1e-14]]
    z = np.zeros(4)
    r = np.zeros((4, 4))
    old_m, old_p = m.copy(), p.copy()
    out_m, out_p = np.empty_like(m), np.empty_like(p)
    status = lib.update(3, m, p, z, r, out_m, out_p, np.empty((3, 2)), np.empty((3, 2)))
    assert status == 3
    assert np.array_equal(m, old_m) and np.array_equal(p, old_p)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'status': 'PASS', 'compiled_update_cases': 1500,
              'non_mutating_late_fallback': True, 'PSD': True,
              'max_cov_error_divided_by_scale': largest,
              'sha256': {str(p): digest(p) for p in (Path(__file__), LIB, HERE.parent/'native/kernel.c')}}
    (HERE/'c_api_checks.json').write_text(json.dumps(report, indent=2))
    print('COMPILED C API PASS', largest, flush=True)


if __name__ == '__main__':
    main()
