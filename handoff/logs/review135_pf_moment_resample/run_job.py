"""Same-input UA development gate for joint-moment resampling correction."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from moment_resample import installed

PARENT=HERE.parent/'review132_pf_predicted_noise'
base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((PARENT/'inputs.json').read_text())
    assert json.loads((HERE/'numerical.json').read_text())['status']=='PASS'
    proof=json.loads((PARENT/'final_verification.json').read_text())
    for p,h in {**prior['sha256'],**proof['sha256']}.items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'moment_resample.py',HERE/'numerical.py',
            HERE/'numerical.json',HERE/'00_protocol.md',PARENT/'final_verification.json']
    for seed in range(3):
        paths += [PARENT/f'dev/evaluation/cartesian-s{seed}/raw_fields.json']
        paths += [PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{v}.txt' for v in job.base.SMOKE]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],
             'resampling':'systematic plus RB joint mean/covariance correction, no additional RNG',
             'effective_measurement_noise_height':'prediction','sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    for seed in range(3):
        for video in job.base.SMOKE:
            observed,_=job.replay(video,seed,False,128)
            ref=np.loadtxt(PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(observed.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            a,_=job.replay(video,seed,True,128);b,_=job.replay(video,seed,True,128)
            assert np.array_equal(a.to_numpy(float),b.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','R132_controls':6,'repeat':6})
    print('SMOKE PASS',flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
