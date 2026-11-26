"""Coordinatewise Gaussian-mixture median as an L1 position Bayes action."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
import rbpf_ctrv as rb


def marginal_median(weights,means,variances):
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
    means=np.asarray(means,dtype=float);variances=np.asarray(variances,dtype=float)
    if not all(np.isfinite(a).all() for a in (weights,means,variances)) or np.any(variances<0):
        raise ValueError('Finite nonnegative Gaussian marginal variance required')
    low,high=float(means.min()),float(means.max())
    if low==high:return low
    positive=variances>0
    sigma=np.sqrt(variances[positive])
    def function(x):
        values=np.empty_like(means)
        values[positive]=ndtr((x-means[positive])/sigma)
        values[~positive]=(x>=means[~positive]).astype(float)
        return float(weights@values)-.5
    left,right=function(low),function(high)
    if left>=0:return low
    if right<=0:return high
    if positive.all():
        return float(brentq(function,low,high,xtol=1e-10,rtol=4*np.finfo(float).eps))
    for _ in range(64):
        middle=low+(high-low)*.5
        if middle==low or middle==high:break
        if function(middle)>=0:high=middle
        else:low=middle
    return high


class MedianPF(PredictedNoisePF):
    def estimate(self):
        mean,cov=super().estimate()
        weights=np.exp(self.logw)
        action=mean.copy()
        for axis in (0,1):
            action[axis]=marginal_median(weights,self.means[:,axis],self.covariances[:,axis,axis])
        shift=mean-action
        risk=cov+np.outer(shift,shift)
        return action,(risk+risk.T)*.5


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=MedianPF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
