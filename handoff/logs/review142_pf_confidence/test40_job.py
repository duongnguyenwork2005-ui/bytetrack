"""Reuse R133's frozen reference/evaluation logic with the R142 filter only."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
HELPER=HERE.parent/'review133_pf_scale_mixture'
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r133_test40_harness',HELPER/'test40_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from confidence import installed
base.HERE=HERE;base.OUT=HERE/'test40';base.base.HERE=HERE;base.base.OUT=base.OUT
base.base.installed=installed


def preflight():
    base.preflight()
    path=base.OUT/'inputs.json';frozen=json.loads(path.read_text())
    paths=[Path(__file__),HELPER/'mixture.py']
    frozen['sha256'].update({str(p):base.base.sha(p) for p in paths})
    frozen['position_noise']='Gaussian'
    frozen['measurement_noise_confidence']='current matched detection score, existing NSA R formula'
    base.base.write(path,frozen)


if __name__=='__main__':
    if sys.argv[1].isdigit():base.base.full('candidate100',int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':base.smoke,'summary':base.summary,'bootstrap':base.bootstrap}[sys.argv[1]]()
