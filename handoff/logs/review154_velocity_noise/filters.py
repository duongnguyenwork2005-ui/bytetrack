"""Isolated polar velocity-noise mapping, preserving the original state model."""
from contextlib import contextmanager
import numpy as np
from ekf_ctrv import EKFTrackerCTRV
from ukf_ctrv import UKFTrackerCTRV
import tracker_ctrv as tc
from shared import installed as shared_installed

MISSING=object()


def heading_variance(speed,height,velocity_weight,old_variance):
    sigma=velocity_weight*max(float(height),1e-3)
    return max(float(old_variance),(sigma/max(abs(float(speed)),sigma))**2)


class VelocityNoise:
    coordinate_aware=True

    def predict(self,mean,covariance):
        old=self.__dict__.get('_r154_prior_speed',MISSING)
        self._r154_prior_speed=float(mean[self.V])
        try:return super().predict(mean,covariance)
        finally:
            if old is MISSING:del self._r154_prior_speed
            else:self._r154_prior_speed=old

    def _process_noise(self,height):
        q=super()._process_noise(height)
        if self.coordinate_aware:
            if '_r154_prior_speed' not in self.__dict__:raise RuntimeError('Prior speed context required')
            q[self.THETA,self.THETA]=heading_variance(self._r154_prior_speed,height,
                                          self._std_weight_velocity,q[self.THETA,self.THETA])
        return q


class AdaptiveEKF(VelocityNoise,EKFTrackerCTRV):pass
class AdaptiveUKF(VelocityNoise,UKFTrackerCTRV):pass


@contextmanager
def installed(forward=True,audit=None,enabled=True):
    class ConfigEKF(AdaptiveEKF):coordinate_aware=enabled
    class ConfigUKF(AdaptiveUKF):coordinate_aware=enabled
    class ETrack(tc.CTRVSTrack):filter_class=ConfigEKF;shared_kalman=None
    class UTrack(tc.UKFSTrack):filter_class=ConfigUKF;shared_kalman=None
    originals=[(cls,cls.__dict__.get('track_class')) for cls in (tc.CTRVByteTracker,tc.UKFByteTracker)]
    tc.CTRVByteTracker.track_class=ETrack;tc.UKFByteTracker.track_class=UTrack
    try:
        with shared_installed(forward,audit):yield
    finally:
        for cls,value in originals:
            if value is None:delattr(cls,'track_class')
            else:cls.track_class=value
