"""Observation-guided incoming turn innovation with an explicit prior/q ratio."""
from contextlib import contextmanager
import numpy as np
from confidence import installed as confidence_installed
from predicted_noise import PredictedNoisePF
from native_pf import lib
from cartesian import transition
from pf_ctrv import systematic_resample
import rbpf_ctrv as rb


def position_derivative(omega,dt,means):
    angle=np.asarray(omega)*dt;small=abs(angle)<1e-3;safe=np.where(small,1.,angle)
    da=dt**2*np.where(small,-angle/3+angle**3/30-angle**5/840,
                      (angle*np.cos(angle)-np.sin(angle))/safe**2)
    db=dt**2*np.where(small,.5-angle**2/8+angle**4/144-angle**6/5760,
                      (angle*np.sin(angle)-(1-np.cos(angle)))/safe**2)
    return np.column_stack((da*means[:,2]-db*means[:,3],db*means[:,2]+da*means[:,3]))


def normal_proposal(prior_mu,prior_var,predicted_means,predicted_covariances,jacobian,z,noise):
    if prior_var<=0:raise ValueError('Positive proposal prior variance required')
    s=predicted_covariances[:,:2,:2]+noise[None,:2,:2]
    # Batched 2x2 Cholesky avoids a separate tiny LAPACK solve per particle.
    a=np.sqrt(s[:,0,0]);b=s[:,1,0]/a;d2=s[:,1,1]-b*b
    if not (d2>0).all():raise FloatingPointError('Non-SPD guided innovation covariance')
    d=np.sqrt(d2);j0=jacobian[:,0]/a;j1=(jacobian[:,1]-b*j0)/d
    var=1./(1./prior_var+j0*j0+j1*j1)
    residual=np.asarray(z)[None,:2]-predicted_means[:,:2]
    r0=residual[:,0]/a;r1=(residual[:,1]-b*r0)/d
    mean=prior_mu+var*(j0*r0+j1*r1)
    assert np.isfinite(mean).all() and np.isfinite(var).all() and (var>0).all()
    return mean,var


def log_prior_over_proposal(value,prior_mu,prior_var,proposal_mu,proposal_var):
    return .5*(np.log(proposal_var/prior_var)+(value-proposal_mu)**2/proposal_var-
                (value-prior_mu)**2/prior_var)


class GuidedTurnPF(PredictedNoisePF):
    guided=True

    def __init__(self,pf,mean,covariance,initial_particles,rng):
        super().__init__(pf,mean,covariance,initial_particles,rng)
        self.incoming_mu=np.full(self.n,mean[pf.OMEGA],dtype=np.float64)
        self.incoming_var=float(covariance[pf.OMEGA,pf.OMEGA])
        self._proposal_cache=None;self.guided_updates=0;self.skipped_guides=0
        self.last_selected=None

    def predict(self,lost=False):
        self._points_cache=None;pf=self.pf;height=max(self.shape_mean[1,0],1e-3)
        self._proposal_cache={'means':self.means.copy(),'covs':self.covariances.copy(),
           'incoming_mu':self.incoming_mu.copy(),'incoming_var':self.incoming_var,
           'shape_mean':self.shape_mean.copy(),'shape_cov':self.shape_cov.copy(),
           'height':height,'lost':lost}
        previous_omega=self.omega.copy();matrix=transition(previous_omega,pf.dt)
        lib.predict(self.n,self.means,self.covariances,matrix,
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    self.shape_mean,self.shape_cov,pf.dt,int(lost))
        eta=self.rng.standard_normal(self.n)*pf._std_omega
        self.omega+=eta
        self.incoming_mu=previous_omega;self.incoming_var=float(pf._std_omega**2)
        self._proposal_cache['eta']=eta
        return self.estimate()

    def initialize_motion(self,*args,**kwargs):
        result=super().initialize_motion(*args,**kwargs)
        if result[2]:self._proposal_cache=None
        return result

    def _predict_from_cache(self,cache,omega):
        pf=self.pf;m=cache['means'].copy();p=cache['covs'].copy()
        sm=cache['shape_mean'].copy();sp=cache['shape_cov'].copy();height=cache['height']
        lib.predict(self.n,m,p,transition(omega,pf.dt),
                    (pf._std_weight_position*height)**2,(pf._std_weight_velocity*height)**2,
                    sm,sp,pf.dt,int(cache['lost']))
        return m,p

    def update(self,observation,confidence=None):
        self._points_cache=None;pf=self.pf;cache=self._proposal_cache;self._proposal_cache=None
        z=np.ascontiguousarray(observation,dtype=np.float64)
        # Same predicted common shape height and confidence formula as R132/R142.
        noise=pf._measurement_noise(max(float(self.shape_mean[1,0]),1e-3),confidence)
        if self.guided and cache is not None and cache['incoming_var']>0:
            centre=cache['incoming_mu'];variance=cache['incoming_var']
            m,p=self._predict_from_cache(cache,centre)
            jac=position_derivative(centre,pf.dt,cache['means'])
            qm,qv=normal_proposal(centre,variance,m,p,jac,z,noise)
            omega=qm+np.sqrt(qv)*self.rng.standard_normal(self.n)
            self.means,self.covariances=self._predict_from_cache(cache,omega)
            self.logw+=log_prior_over_proposal(omega,centre,variance,qm,qv)
            # Preserve the independent NEW angular innovation's temporal role.
            self.omega=omega+cache['eta'];self.incoming_mu=omega.copy()
            self.guided_updates+=1
        elif self.guided:self.skipped_guides+=1
        status=lib.update(self.n,self.means,self.covariances,z,noise,
                          self._next_m,self._next_p,self._white,self._chol)
        if status:raise FloatingPointError('Native covariance failure in guided PF; no silent fallback')
        self.means,self._next_m=self._next_m,self.means
        self.covariances,self._next_p=self._next_p,self.covariances
        self.logw+=-.5*np.sum(self._white**2,axis=1)-np.log(self._chol).sum(axis=1)
        self.logw-=self.logw.max();self.logw-=np.log(np.exp(self.logw).sum())
        lib.shapes(self.shape_mean,self.shape_cov,z,noise)
        result=self.estimate();self.updates+=1;pf.n_update+=1;self.last_selected=None
        if pf.ess(self.logw)<pf.resample_ess_ratio*self.n:
            selected=systematic_resample(np.exp(self.logw),float(self.rng.random()))
            self.means=self.means[selected].copy();self.covariances=self.covariances[selected].copy()
            self.omega=self.omega[selected].copy();self.incoming_mu=self.incoming_mu[selected].copy()
            self.logw.fill(-np.log(self.n));self._points_cache=None
            self.resamples+=1;pf.n_resample+=1;self.last_selected=selected.copy()
        return result


@contextmanager
def installed(enabled=True):
    with confidence_installed():
        previous=rb.ConditionalPF
        rb.ConditionalPF=type('ConfiguredGuidedPF',(GuidedTurnPF,),{'guided':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
