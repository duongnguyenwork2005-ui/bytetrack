"""Finite-mixture moment correction after systematic selection, not an ETPF."""
from contextlib import contextmanager
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'review132_pf_predicted_noise'))
from predicted_noise import PredictedNoisePF
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb


def moments(weights,omega,means,covariances):
    weights=np.asarray(weights,dtype=float)
    weights=weights/weights.sum()
    points=np.column_stack((omega,means))
    center=weights@points
    delta=points-center
    cov=np.einsum('n,ni,nj->ij',weights,delta,delta)
    cov[1:,1:]+=np.einsum('n,nij->ij',weights,covariances)
    return center,(cov+cov.T)*.5


def corrected_cloud(weights,omega,means,covariances,selected):
    """Return private corrected arrays; never mutate the input or use RNG."""
    new_w=omega[selected].copy();new_m=means[selected].copy();new_p=covariances[selected].copy()
    target,tcov=moments(weights,omega,means,covariances)
    current,scov=moments(np.ones(len(selected)),new_w,new_m,new_p)
    scale=max(1.,float(np.max(omega*omega)))
    tolerance=64*np.finfo(float).eps*scale
    if min(tcov[0,0],scov[0,0])<=tolerance:
        return new_w,new_m,new_p,False
    beta=tcov[1:,0]/tcov[0,0];beta_s=scov[1:,0]/scov[0,0]
    schur=tcov[1:,1:]-np.outer(tcov[1:,0],tcov[1:,0])/tcov[0,0]
    schur_s=scov[1:,1:]-np.outer(scov[1:,0],scov[1:,0])/scov[0,0]
    left=np.linalg.cholesky((schur+schur.T)*.5)
    right=np.linalg.cholesky((schur_s+schur_s.T)*.5)
    transform=np.linalg.solve(right.T,left.T).T
    residual=new_m-current[1:]-(new_w-current[0])[:,None]*beta_s
    new_w=target[0]+np.sqrt(tcov[0,0]/scov[0,0])*(new_w-current[0])
    new_m=target[1:]+(new_w-target[0])[:,None]*beta+residual@transform.T
    new_p=transform@new_p@transform.T
    new_p=(new_p+new_p.transpose(0,2,1))*.5
    if not all(np.isfinite(a).all() for a in (new_w,new_m,new_p)):
        raise ValueError('Nonfinite moment transform')
    return np.ascontiguousarray(new_w),np.ascontiguousarray(new_m),np.ascontiguousarray(new_p),True


class MomentResamplePF(PredictedNoisePF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.correct_moments=True
        self.corrected_resamples=self.correction_fallbacks=0

    def update(self,observation,confidence=None):
        pf=self.pf;ratio=pf.resample_ess_ratio
        # Parent performs the identical Gaussian update/estimate but defers selection.
        pf.resample_ess_ratio=0.
        try:result=super().update(observation,confidence)
        finally:pf.resample_ess_ratio=ratio
        if pf.ess(self.logw)<ratio*self.n:
            weights=np.exp(self.logw)
            selected=systematic_resample(weights,float(self.rng.random()))
            applied=False
            if self.correct_moments:
                try:
                    omega,means,covs,applied=corrected_cloud(weights,self.omega,self.means,self.covariances,selected)
                except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                    omega=self.omega[selected].copy();means=self.means[selected].copy();covs=self.covariances[selected].copy()
                self.corrected_resamples+=int(applied);self.correction_fallbacks+=int(not applied)
            else:
                omega=self.omega[selected].copy();means=self.means[selected].copy();covs=self.covariances[selected].copy()
            self.omega,self.means,self.covariances=omega,means,covs
            self.logw.fill(-np.log(self.n));self._points_cache=None
            self.resamples+=1;pf.n_resample+=1
        return result


@contextmanager
def installed(enabled=True):
    previous=rb.ConditionalPF
    rb.ConditionalPF=MomentResamplePF if enabled else PredictedNoisePF
    try:yield
    finally:rb.ConditionalPF=previous
