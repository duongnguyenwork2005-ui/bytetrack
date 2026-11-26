"""Match the Native PF chord initialization policy in Cartesian Gaussian CT."""
from contextlib import contextmanager
import numpy as np
from filters import CartesianEKF,CartesianUKF
from shared import installed as shared_installed
import tracker_ctrv as tc

class ChordInit:
    chord_init=True

    def try_initiate_from_motion(self,mean,covariance,measurement,n_frames=1.,ref_pos=None):
        if not self.chord_init:return super().try_initiate_from_motion(mean,covariance,measurement,n_frames,ref_pos)
        reference=np.asarray(mean[:2] if ref_pos is None else ref_pos,dtype=float)
        velocity=(np.asarray(measurement,dtype=float)[:2]-reference)/(self.dt*max(n_frames,1.))
        speed=float(np.linalg.norm(velocity))
        if speed<self._min_speed_for_heading:return mean.copy(),covariance.copy(),False
        direction=velocity/speed;across=np.array([-direction[1],direction[0]])
        m=mean.copy();p=covariance.copy();m[2:4]=velocity
        p[2:4,:]=0.;p[:,2:4]=0.
        sigma=2*self._std_weight_velocity*max(float(mean[self.H]),1e-3)*10
        p[2:4,2:4]=sigma**2*np.outer(direction,direction)+(speed*self._heading_std_after_init)**2*np.outer(across,across)
        return m,(p+p.T)*.5,True

class ChordEKF(ChordInit,CartesianEKF):pass
class ChordUKF(ChordInit,CartesianUKF):pass

@contextmanager
def installed(forward=True,audit=None):
    class ETrack(tc.CTRVSTrack):filter_class=ChordEKF;shared_kalman=None
    class UTrack(tc.UKFSTrack):filter_class=ChordUKF;shared_kalman=None
    old=[(c,c.__dict__.get('track_class')) for c in (tc.CTRVByteTracker,tc.UKFByteTracker)]
    tc.CTRVByteTracker.track_class=ETrack;tc.UKFByteTracker.track_class=UTrack
    try:
        with shared_installed(forward,audit):yield
    finally:
        for c,value in old:
            if value is None:delattr(c,'track_class')
            else:c.track_class=value
