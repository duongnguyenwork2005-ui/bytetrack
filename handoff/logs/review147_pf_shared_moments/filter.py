from contextlib import contextmanager
from pathlib import Path
import sys

sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'review135_pf_moment_resample'),
              str(Path(__file__).resolve().parent.parent/'review142_pf_confidence')]
from moment_resample import MomentResamplePF
from confidence import installed as confidence_installed
import rbpf_ctrv as rb


@contextmanager
def installed(enabled=True):
    with confidence_installed():
        previous=rb.ConditionalPF
        if enabled:rb.ConditionalPF=MomentResamplePF
        try:yield
        finally:rb.ConditionalPF=previous
