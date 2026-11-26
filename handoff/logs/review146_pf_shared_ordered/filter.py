from contextlib import contextmanager
from pathlib import Path
import sys

sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'review140_pf_ordered_resample'),
              str(Path(__file__).resolve().parent.parent/'review142_pf_confidence')]
from ordered import OrderedPF
from confidence import installed as confidence_installed
import rbpf_ctrv as rb


@contextmanager
def installed(enabled=True):
    with confidence_installed():
        original=rb.ConditionalPF
        if enabled:rb.ConditionalPF=OrderedPF
        try:yield
        finally:rb.ConditionalPF=original
