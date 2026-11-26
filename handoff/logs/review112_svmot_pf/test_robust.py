"""Check the robust update numerically and against the Gaussian inlier case."""

import numpy as np
import test_rbpf
from pf_ctrv import PFCTRV
from rbpf import ConditionalPF
from robust_filter import RobustConditionalPF

test_rbpf.ConditionalPF = RobustConditionalPF
test_rbpf.test_conditional_kalman()
test_rbpf.test_angles_lost_and_repeat()
pf = PFCTRV(num_particles=32)
mean, covariance = pf.initiate(np.array([30., 50., 1., 40.]))
particles = np.repeat(mean[None], 32, axis=0)
gaussian = ConditionalPF(pf, mean, covariance, particles, np.random.default_rng(114))
robust = RobustConditionalPF(pf, mean, covariance, particles, np.random.default_rng(114))
gaussian.predict()
robust.predict()
outlier = np.array([1030., 50., 1., 40.])
normal_mean, _ = gaussian.update(outlier)
robust_mean, robust_cov = robust.update(outlier)
assert abs(robust_mean[pf.CX] - mean[pf.CX]) < abs(normal_mean[pf.CX] - mean[pf.CX])
assert np.isfinite(robust_mean).all() and np.linalg.eigvalsh(robust_cov).min() > -1e-8
print("PASS bounded influence under a 1000-pixel outlier")
