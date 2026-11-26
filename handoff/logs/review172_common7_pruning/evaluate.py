"""Only evaluate the fixed common checkpoints with the unchanged runtime-geometry scorer."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'review169_paired_cuda_resume/monitor.py'
spec = importlib.util.spec_from_file_location('r172_frozen_quality', source)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE = HERE

if __name__ == '__main__':
    assert sys.argv[1] in ('anchor', 'control', 'joint50')
    base.dev(sys.argv[1])
