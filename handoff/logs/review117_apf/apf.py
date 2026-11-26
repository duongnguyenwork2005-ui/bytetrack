"""Auxiliary sampling for the existing diagonal-transition CTRV particle filter."""

from contextlib import contextmanager

import numpy as np
from pf_ctrv import PFCTRV, _logsumexp, systematic_resample
from ekf_ctrv import normalize_angle


def conditional_mixture(pf, prior, logw, observation, variance):
    """Exact Gaussian component moments before the existing height clamp."""
    means, process_std = prior
    observed = [pf.CX, pf.CY, pf.A, pf.H]
    total = process_std[:, observed] ** 2 + variance
    residual = observation - means[:, observed]
    log_evidence = -.5 * np.sum(residual ** 2 / total + np.log(total), axis=1)
    weights = logw + log_evidence
    weights -= _logsumexp(weights)
    posterior = means.copy()
    posterior_var = process_std ** 2
    gain = posterior_var[:, observed] / total
    posterior[:, observed] += gain * residual
    posterior_var[:, observed] *= variance / total
    return posterior, posterior_var, weights


def fresh_posterior(pf, means, variances, logw, rng):
    """Independent offspring even when a parent is selected more than once."""
    indexes = systematic_resample(np.exp(logw), float(rng.random()))
    particles = means[indexes] + rng.standard_normal(means.shape) * np.sqrt(variances[indexes])
    particles[:, pf.THETA] = normalize_angle(particles[:, pf.THETA])
    particles[:, pf.H] = np.maximum(particles[:, pf.H], 1e-3)
    return particles, indexes


@contextmanager
def installed(mode="apf"):
    if mode not in ("base", "apf", "apf_k2", "apf_ess_k2"):
        raise ValueError(mode)
    original = PFCTRV.update_particles
    counters = {"auxiliary_updates": 0, "fallback": 0, "independent_offspring": 0,
                "distinct_parents_sum": 0}
    if mode == "base":
        yield counters
        return

    def update(pf, X, logw, z, conf, rng, diag_key=None, prior=None):
        observed = [pf.CX, pf.CY, pf.A, pf.H]
        z = np.asarray(z, dtype=float)
        if not pf.optimal_proposal or prior is None or prior[0].shape != X.shape:
            counters["fallback"] += 1
            return original(pf, X, logw, z, conf, rng, diag_key, prior)
        assert pf.measurement_height == "observation" and not pf.rb_observed_estimate
        assert pf.estimate == "mean" and pf.diag_rows is None
        variance = np.diag(pf._measurement_noise(max(z[3], 1e-3), conf))
        likelihood = pf.log_likelihood(X, z, variance)
        best = float(likelihood.max())
        if not np.isfinite(best) or -2 * best > pf.degenerate_maha2:
            counters["fallback"] += 1
            return original(pf, X, logw, z, conf, rng, diag_key, prior)
        means, variances, weights = conditional_mixture(pf, prior, logw, z, variance)
        if mode == "apf_ess_k2" and pf.ess(weights) >= pf.resample_ess_ratio * len(weights):
            counters["fallback"] += 1
            return original(pf, X, logw, z, conf, rng, diag_key, prior)
        particles, indexes = fresh_posterior(pf, means, variances, weights, rng)
        new_logw = np.full(len(logw), -np.log(len(logw)))
        mean, covariance = pf.estimate_state(particles, new_logw)
        if mode in ("apf_k2", "apf_ess_k2"):
            noise = pf.process_noise_std_batch(particles[:, pf.H]) * pf.roughening
            noise[:, observed] *= pf.roughening_box_scale
            noise[:, [pf.V, pf.THETA, pf.OMEGA, pf.VA, pf.VH]] *= pf.roughening_motion_scale
            particles += rng.standard_normal(particles.shape) * noise
            particles[:, pf.THETA] = normalize_angle(particles[:, pf.THETA])
            particles[:, pf.H] = np.maximum(particles[:, pf.H], 1e-3)
        pf.last_resample_idx = indexes
        pf.n_update += 1
        pf.n_k1_update += 1
        pf.n_resample += 1
        counters["auxiliary_updates"] += 1
        counters["independent_offspring"] += len(indexes)
        counters["distinct_parents_sum"] += len(np.unique(indexes))
        return particles, new_logw, mean, covariance

    PFCTRV.update_particles = update
    try:
        yield counters
    finally:
        PFCTRV.update_particles = original
