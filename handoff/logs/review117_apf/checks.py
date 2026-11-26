"""Numerical mixture checks independent of tracking benchmark scores."""

import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from pf_ctrv import PFCTRV
from apf import conditional_mixture, fresh_posterior, installed


def run():
    pf = PFCTRV(num_particles=200000, optimal_proposal=True, estimate="mean")
    n = pf.num_particles
    means = np.zeros((n, 9))
    means[:, pf.H], means[:, pf.A] = 80, 1.5
    means[:n//2, pf.CX] = -3
    means[n//2:, pf.CX] = 3
    means[:n//2, pf.V] = -2
    means[n//2:, pf.V] = 2
    std = np.ones((n, 9)) * .4
    observation = np.array([1., 0, 1.5, 80])
    logw = np.full(n, -np.log(n))
    m, var, w = conditional_mixture(pf, (means, std), logw, observation, np.ones(4))
    scalar_gain = .16 / 1.16
    assert np.allclose(m[:, pf.CX], means[:, pf.CX] + scalar_gain * (1-means[:, pf.CX]))
    assert np.allclose(var[:, pf.CX], .16 / 1.16)
    assert np.isclose(np.exp(w).sum(), 1)
    analytical_mean = np.exp(w) @ m
    centered = m - analytical_mean
    analytical_var = np.exp(w) @ (var + centered**2)
    sample, parents = fresh_posterior(pf, m, var, w, np.random.default_rng(117))
    error = np.abs(sample.mean(axis=0) - analytical_mean)
    assert np.all(error < 6 * np.sqrt(analytical_var/n) + .001), error
    assert np.allclose(sample.var(axis=0), analytical_var, rtol=.035, atol=.003)
    assert len(np.unique(sample[:, pf.V])) == n
    assert len(np.unique(parents)) < n
    original = PFCTRV.update_particles
    with installed("apf"):
        assert PFCTRV.update_particles is not original
    assert PFCTRV.update_particles is original
    rng = np.random.default_rng(7)
    a = fresh_posterior(pf, m, var, w, rng)[0]
    b = fresh_posterior(pf, m, var, w, np.random.default_rng(7))[0]
    assert np.array_equal(a, b)
    print("PASS mixture weight normalization, analytic posterior mean/variance, independent offspring, RNG and restoration")


if __name__ == "__main__":
    run()
