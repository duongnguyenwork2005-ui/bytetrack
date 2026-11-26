"""Transport velocity covariance without erasing learned cross covariance."""
from contextlib import contextmanager
import numpy as np
from guided import GuidedTurnPF
from confidence import installed as confidence_installed
import rbpf_ctrv as rb

def transport(old,target):
    old_root=np.linalg.cholesky(old[:,2:,2:])
    target_root=np.linalg.cholesky(target)
    a=np.linalg.solve(old_root.transpose(0,2,1),target_root.transpose(0,2,1)).transpose(0,2,1)
    d=np.broadcast_to(np.eye(4),old.shape).copy();d[:,2:,2:]=a
    p=d@old@d.transpose(0,2,1)
    return np.ascontiguousarray((p+p.transpose(0,2,1))*.5)

class TransportPF(GuidedTurnPF):
    preserve_init_cross=True

    def initialize_motion(self,*args,**kwargs):
        if not self.preserve_init_cross:return super().initialize_motion(*args,**kwargs)
        old=self.covariances.copy()
        result=super().initialize_motion(*args,**kwargs)
        if not result[2]:return result
        self.covariances=transport(old,self.covariances[:,2:,2:])
        self._points_cache=None
        mean,covariance=self.estimate()
        return mean,covariance,True

@contextmanager
def installed(enabled=True):
    with confidence_installed():
        previous=rb.ConditionalPF
        rb.ConditionalPF=type('ConfiguredTransportPF',(TransportPF,),{'preserve_init_cross':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
