"""Retain the conditional height-trend covariance when the tracker is Lost."""
from contextlib import contextmanager
from guided import GuidedTurnPF
from confidence import installed as confidence_installed
import rbpf_ctrv as rb


class PreservedShapePF(GuidedTurnPF):
    preserve_lost_cov=True

    def predict(self,lost=False):
        if self.preserve_lost_cov and lost:
            self.shape_mean[1,1]=0.
            return super().predict(lost=False)
        return super().predict(lost=lost)


@contextmanager
def installed(enabled=True):
    with confidence_installed():
        previous=rb.ConditionalPF
        rb.ConditionalPF=type('ConfiguredPreservedShapePF',(PreservedShapePF,),{'preserve_lost_cov':bool(enabled)})
        try:yield
        finally:rb.ConditionalPF=previous
