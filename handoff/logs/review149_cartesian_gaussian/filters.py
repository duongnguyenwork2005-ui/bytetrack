"""Research Cartesian coordinated-turn Gaussian filters, isolated from defaults."""
from contextlib import contextmanager
import numpy as np
from ekf_ctrv import EKFTrackerCTRV
from ukf_ctrv import UKFTrackerCTRV
import tracker_ctrv as tc
from shared import installed as shared_installed


def coefficients(omega, dt):
    angle=omega*dt
    a=dt*np.sinc(angle/np.pi)
    b=.5*omega*dt**2*np.sinc(angle/(2*np.pi))**2
    if abs(angle)<1e-3:
        da=dt**2*(-angle/3+angle**3/30-angle**5/840)
        db=dt**2*(.5-angle**2/8+angle**4/144-angle**6/5760)
    else:
        da=dt**2*(angle*np.cos(angle)-np.sin(angle))/angle**2
        db=dt**2*(angle*np.sin(angle)-(1-np.cos(angle)))/angle**2
    return a,b,da,db


class CartesianEKF(EKFTrackerCTRV):
    VX,VY=2,3

    def initiate(self, measurement):
        z=np.asarray(measurement,dtype=float);h=max(z[3],1e-3)
        m=np.zeros(9);m[[0,1,5,6]]=z
        std=np.array([2*self._std_weight_position*h]*2+[10*self._std_weight_velocity*h]*2+
                     [self._init_std_omega,1e-2,2*self._std_weight_position*h,1e-5,10*self._std_weight_velocity*h])
        return m,np.diag(std**2)

    def f(self, x):
        x=np.asarray(x,dtype=float);y=x.copy();w=x[4];a,b,_,_=coefficients(w,self.dt)
        c,s=np.cos(w*self.dt),np.sin(w*self.dt)
        y[0]+=a*x[2]-b*x[3];y[1]+=b*x[2]+a*x[3]
        y[2]=c*x[2]-s*x[3];y[3]=s*x[2]+c*x[3]
        y[5]+=self.dt*x[7];y[6]+=self.dt*x[8]
        return y

    def jacobian(self,x):
        dt=self.dt;w=x[4];a,b,da,db=coefficients(w,dt);c,s=np.cos(w*dt),np.sin(w*dt)
        f=np.eye(9)
        f[0,2:4]=a,-b;f[1,2:4]=b,a
        f[2,2:4]=c,-s;f[3,2:4]=s,c
        f[0,4]=da*x[2]-db*x[3];f[1,4]=db*x[2]+da*x[3]
        f[2,4]=-dt*(s*x[2]+c*x[3]);f[3,4]=dt*(c*x[2]-s*x[3])
        f[5,7]=f[6,8]=dt
        return f

    def _process_noise(self,h):
        h=max(h,1e-3)
        std=np.array([self._std_weight_position*h]*2+[self._std_weight_velocity*h]*2+
                     [self._std_omega,1e-2,self._std_weight_position*h,1e-5,self._std_weight_velocity*h])
        return np.diag(std**2)

    def try_initiate_from_motion(self,mean,covariance,measurement,n_frames=1.,ref_pos=None):
        # Cartesian velocity is observable through position/velocity covariance.
        # Preserve tracker lifecycle API but do not reset the Bayesian velocity.
        return mean,covariance,True

    def update(self,mean,covariance,measurement,confidence=None):
        h=self._update_mat;r=self._measurement_noise(max(mean[self.H],1e-3),confidence)
        innovation=h@covariance@h.T+r
        gain=np.linalg.solve(innovation,(covariance@h.T).T).T
        updated=mean+gain@(np.asarray(measurement)-h@mean)
        correction=np.eye(9)-gain@h
        cov=correction@covariance@correction.T+gain@r@gain.T
        return updated,(cov+cov.T)*.5


class CartesianUKF(CartesianEKF,UKFTrackerCTRV):
    def sigma_points(self,mean,covariance):
        p=(covariance+covariance.T)*.5
        try:root=np.linalg.cholesky(self._n_plus_lambda*p)
        except np.linalg.LinAlgError:
            values,vectors=np.linalg.eigh(p)
            if values.min() < -1e-10*max(1.,float(abs(values).max())):
                raise ValueError('Indefinite Cartesian UKF covariance')
            root=vectors@np.diag(np.sqrt(np.maximum(values,0)*self._n_plus_lambda))
        return np.concatenate((mean[None],mean[None]+root.T,mean[None]-root.T))

    def _state_mean(self,points,weights):return weights@points
    def _state_residual(self,points,mean):return points-mean

    def predict(self,mean,covariance):
        points=self.sigma_points(mean,covariance)
        transformed=np.array([self.f(x) for x in points])
        m=self.Wm@transformed;d=transformed-m
        p=(d*self.Wc[:,None]).T@d+self._process_noise(mean[self.H])
        return m,(p+p.T)*.5


class CartesianTrack(tc.CTRVSTrack):
    filter_class=CartesianEKF
    shared_kalman=None


class CartesianUKFTrack(CartesianTrack):
    filter_class=CartesianUKF


@contextmanager
def installed(forward=True,audit=None):
    originals=[(cls,cls.__dict__.get('track_class')) for cls in (tc.CTRVByteTracker,tc.UKFByteTracker)]
    tc.CTRVByteTracker.track_class=CartesianTrack
    tc.UKFByteTracker.track_class=CartesianUKFTrack
    try:
        with shared_installed(forward,audit):yield
    finally:
        for cls,value in originals:
            if value is None:delattr(cls,'track_class')
            else:cls.track_class=value
