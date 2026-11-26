"""Moment-matched deterministic Gaussian prior nodes, stochastic PF thereafter."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtri

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
import rbpf_ctrv as rb


@lru_cache(maxsize=8)
def standard_nodes(n):
    if n<2:raise ValueError('at least two nodes required')
    nodes=ndtri((np.arange(n,dtype=float)+.5)/n)
    nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    nodes.setflags(write=False)
    return nodes


class QuantilePF(PredictedNoisePF):
    def __init__(self,pf,mean,covariance,particles,rng):
        super().__init__(pf,mean,covariance,particles,rng)
        variance=float(covariance[pf.OMEGA,pf.OMEGA]);location=float(mean[pf.OMEGA])
        if variance<0 or not np.isfinite(variance+location):
            raise ValueError('invalid angular prior moments')
        self.omega=location+np.sqrt(variance)*standard_nodes(self.n)
        self._points_cache=None


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=QuantilePF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
