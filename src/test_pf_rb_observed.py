"""Analytic moment and default-parity checks for the optional PF conditional estimate."""
from __future__ import annotations

import numpy as np

from pf_ctrv import PFCTRV


def test_conditional_moments() -> None:
    n = 500
    base = PFCTRV(num_particles=n, optimal_proposal=True, resample_ess_ratio=0.0)
    candidate = PFCTRV(num_particles=n, optimal_proposal=True, rb_observed_estimate=True,
                       resample_ess_ratio=0.0)
    init = np.random.default_rng(19)
    f = np.tile(np.array([100.0, 80.0, 2.0, 0.2, 0.01, 0.8, 40.0, 0.0, 0.0]), (n, 1))
    f[:, 0] += init.normal(0, 2, n)
    f[:, 1] += init.normal(0, 2, n)
    f[:, 6] += init.normal(0, 1, n)
    std = base.process_noise_std_batch(f[:, base.H])
    x = f + init.standard_normal(f.shape) * std
    z = np.array([101.0, 79.0, 0.81, 40.5])
    logw = np.full(n, -np.log(n))
    a = base.update_particles(x.copy(), logw.copy(), z, 0.8, np.random.default_rng(37), prior=(f, std))
    b = candidate.update_particles(x.copy(), logw.copy(), z, 0.8, np.random.default_rng(37), prior=(f, std))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "proposal/weights must stay identical"

    ids = [candidate.CX, candidate.CY, candidate.A, candidate.H]
    r2 = np.diag(candidate._measurement_noise(z[3], 0.8))
    q2 = std[:, ids] ** 2
    s2 = 1.0 / (1.0 / q2 + 1.0 / r2)
    m = s2 * (f[:, ids] / q2 + z[None, :] / r2)
    tot = q2 + r2
    ll = -0.5 * np.sum((z[None, :] - f[:, ids]) ** 2 / tot + np.log(tot), axis=1)
    w = np.exp(ll - np.max(ll))
    w /= w.sum()
    expected_mean = w @ m
    assert np.allclose(b[2][ids], expected_mean, atol=1e-10)
    assert np.allclose(b[3][ids, ids], w @ (s2 + (m - expected_mean) ** 2), atol=1e-10)
    assert np.max(np.abs(a[2][ids] - expected_mean)) > 1e-4, "base must retain Monte Carlo draw noise"
    assert np.allclose(a[2][[candidate.V, candidate.OMEGA]], b[2][[candidate.V, candidate.OMEGA]])


def test_default_parity_and_validation() -> None:
    for kwargs in ({"rb_observed_estimate": True},
                   {"rb_observed_estimate": True, "optimal_proposal": True, "estimate": "map"}):
        try:
            PFCTRV(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected validation failure: {kwargs}")
    default = PFCTRV(num_particles=100, optimal_proposal=True)
    explicit_false = PFCTRV(num_particles=100, optimal_proposal=True, rb_observed_estimate=False)
    m, p = default.initiate(np.array([100.0, 80.0, 0.8, 40.0]))
    x = default.sample_initial(m, p, np.random.default_rng(2))
    logw = np.full(100, -np.log(100))
    pred_a, prior_a = default.propagate(x.copy(), np.random.default_rng(3), return_prior=True)
    pred_b, prior_b = explicit_false.propagate(x.copy(), np.random.default_rng(3), return_prior=True)
    assert np.array_equal(pred_a, pred_b)
    out_a = default.update_particles(pred_a, logw.copy(), np.array([101.0, 80.5, 0.81, 40.1]), 0.6,
                                     np.random.default_rng(4), prior=prior_a)
    out_b = explicit_false.update_particles(pred_b, logw.copy(), np.array([101.0, 80.5, 0.81, 40.1]), 0.6,
                                            np.random.default_rng(4), prior=prior_b)
    assert all(np.array_equal(a, b) for a, b in zip(out_a, out_b))


if __name__ == "__main__":
    test_conditional_moments()
    test_default_parity_and_validation()
    print("PF conditional moments and default parity: PASS")
