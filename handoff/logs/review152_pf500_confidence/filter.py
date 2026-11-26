"""Same R142 kernel, with a fail-loud actual particle-count guard."""
from contextlib import contextmanager
from confidence import installed as confidence_installed
from predicted_noise import PredictedNoisePF
import rbpf_ctrv as rb


class PF500(PredictedNoisePF):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.n!=500 or self.pf.num_particles!=500:
            raise ValueError('R152 requires exactly 500 particles')


@contextmanager
def installed():
    with confidence_installed():
        previous=rb.ConditionalPF;rb.ConditionalPF=PF500
        try:yield
        finally:rb.ConditionalPF=previous
