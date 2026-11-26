"""Frozen three-scale observation model and same-input UA development gate."""
from pathlib import Path
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
from mixture import installed,MixturePF,WEIGHTS,SCALES
import native_pf

PARENT=HERE.parent/'review132_pf_predicted_noise'
base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    assert json.loads((HERE/'numerical.json').read_text())['status']=='PASS'
    prior=json.loads((PARENT/'inputs.json').read_text())
    proof=json.loads((PARENT/'final_verification.json').read_text())
    assert proof['status']=='PASS' and proof['quality_video_runs']==996
    for p,h in {**prior['sha256'],**proof['sha256']}.items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]
    paths += [Path(__file__),HERE/'mixture.py',HERE/'numerical.py',HERE/'numerical.json',HERE/'00_protocol.md',
              PARENT/'final_verification.json',PARENT/'final_summary.json']
    paths += [PARENT/f'dev/evaluation/cartesian-s{s}/raw_fields.json' for s in range(3)]
    for s in range(3):
        for video in job.base.SMOKE:
            paths += [PARENT/f'dev/tracking/DETRAC-all/cartesian-s{s}/data/{video}.txt',
                      base.OLD/f'dev/tracking/DETRAC-all/cartesian-s{s}/data/{video}.txt']
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],
             'effective_measurement_noise_height':'prediction','position_noise':'covariance-matched-three-scale-Gaussian-mixture',
             'weights':WEIGHTS.tolist(),'scales':SCALES.tolist(),'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    records=[]
    for seed in range(3):
        for video in job.base.SMOKE:
            saved=job.installed;job.installed=native_pf.installed
            try:native,_=job.replay(video,seed,True,128)
            finally:job.installed=saved
            ref=np.loadtxt(base.OLD/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(native.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            control,_=job.replay(video,seed,False,128)
            ref=np.loadtxt(PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            assert np.allclose(control.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            old_weights,old_scales=MixturePF.weights,MixturePF.scales
            MixturePF.weights=np.array([1.]);MixturePF.scales=np.array([1.])
            try:one,_=job.replay(video,seed,True,128)
            finally:MixturePF.weights,MixturePF.scales=old_weights,old_scales
            assert np.allclose(one.to_numpy(float),ref[ref[:,0]<=128],atol=1e-9,rtol=0)
            x,_=job.replay(video,seed,True,128);y,_=job.replay(video,seed,True,128)
            assert np.array_equal(x.to_numpy(float),y.to_numpy(float))
            records.append({'video':video,'seed':seed,'native_control':True,'R132_control':True,'one_component_control':True,'repeat':True})
    job.write(HERE/'smoke.json',{'status':'PASS','checks':records})
    print('SMOKE PASS',len(records),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
