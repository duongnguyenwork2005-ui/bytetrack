from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review128_pf_heading'))
spec=importlib.util.spec_from_file_location('r128_bdd_harness',HERE.parent/'review128_pf_heading/bdd_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from confidence import installed
base.TOP=HERE;base.HERE=HERE/'bdd';base.job.HERE=base.HERE;base.job.installed=installed


def preflight():
    base.preflight()
    path=base.HERE/'inputs.json';frozen=json.loads(path.read_text())
    frozen['sha256'][str(Path(__file__))]=base.job.sha(Path(__file__))
    frozen['measurement_noise_confidence']='current matched detection score, existing NSA R formula'
    frozen['effective_measurement_noise_height']='prediction'
    base.job.write(path,frozen)


if __name__=='__main__':
    if sys.argv[1].isdigit():base.job.full('pf100',int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':base.smoke,'summary':base.summary}[sys.argv[1]]()
