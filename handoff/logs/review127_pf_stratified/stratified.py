"""Randomized stratification for scalar Gaussian process innovations only."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'review126_pf_efficiency'))
from native_pf import NativeCartesianPF as FastCartesianPF
import rbpf_ctrv as rb


def gaussian_strata(rng, n):
    u = (np.arange(n)+rng.random(n))/n
    u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
    return ndtri(u)[rng.permutation(n)]


class ProcessRNG:
    def __init__(self, base, n):
        self.base, self.n = base, n
    def standard_normal(self, size):
        assert size == self.n
        return gaussian_strata(self.base, size)
    def __getattr__(self, name):
        return getattr(self.base, name)


class StratifiedPF(FastCartesianPF):
    def predict(self, lost=False):
        base = self.rng
        self.rng = ProcessRNG(base, self.n)
        try:
            return super().predict(lost)
        finally:
            self.rng = base


@contextmanager
def installed(enabled=True):
    previous = rb.ConditionalPF
    rb.ConditionalPF = StratifiedPF if enabled else FastCartesianPF
    try:
        yield
    finally:
        rb.ConditionalPF = previous
