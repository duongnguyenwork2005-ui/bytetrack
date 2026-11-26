"""Same-input UA gate for current-score-conditioned measurement noise."""
from pathlib import Path
import importlib.util
import inspect
import json
import sys
from functools import partial

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review129_pf_soft_init'))
spec=importlib.util.spec_from_file_location('r129_ua_harness',HERE.parent/'review129_pf_soft_init/run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
import ul_backend
from confidence import installed

PARENT=HERE.parent/'review132_pf_predicted_noise'
base.HERE=HERE;base.job.HERE=HERE;base.job.installed=installed
job=base.job


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((PARENT/'inputs.json').read_text())
    assert json.loads((HERE/'numerical.json').read_text())['status']=='PASS'
    proof=json.loads((PARENT/'final_verification.json').read_text())
    for p,h in {**prior['sha256'],**proof['sha256']}.items():assert job.base.sha(Path(p))==h,p
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'confidence.py',HERE/'numerical.py',
            HERE/'numerical.json',HERE/'00_protocol.md',PARENT/'final_verification.json',Path(inspect.getfile(ul_backend.STrack))]
    for seed in range(3):
        paths += [PARENT/f'dev/evaluation/cartesian-s{seed}/raw_fields.json']
        paths += [PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{v}.txt' for v in job.base.SMOKE]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':prior['baseline_seeds'],
             'measurement_noise':'max(1-current_detection_confidence,0.05)*R0',
             'effective_measurement_noise_height':'prediction',
             'STrack_source':inspect.getfile(ul_backend.STrack),'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    off_events=[];on_events=[]
    for seed in range(3):
        for video in job.base.SMOKE:
            ref=np.loadtxt(PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',',ndmin=2)
            expected=ref[ref[:,0]<=128]
            observed,_=job.replay(video,seed,False,128)
            assert np.allclose(observed.to_numpy(float),expected,atol=1e-9,rtol=0)
            saved=job.installed
            try:
                job.installed=partial(installed,forward=False,audit=lambda p,c:off_events.append((p,c)))
                control,_=job.replay(video,seed,True,128)
                assert np.allclose(control.to_numpy(float),expected,atol=1e-9,rtol=0)
                job.installed=partial(installed,audit=lambda p,c:on_events.append((p,c)))
                a,_=job.replay(video,seed,True,128);b,_=job.replay(video,seed,True,128)
                assert np.array_equal(a.to_numpy(float),b.to_numpy(float))
            finally:job.installed=saved
    assert off_events and on_events
    assert all(p is not None and 0<=p<=1 and c is None for p,c in off_events)
    assert all(p is not None and 0<=p<=1 and p==c for p,c in on_events)
    job.write(HERE/'smoke.json',{'status':'PASS','R132_controls':6,'route_off_controls':6,'repeat':6,
              'route_off_updates':len(off_events),'current_score_updates':len(on_events),
              'all_scores_current':True,'route_off_MOT_parent_parity':True})
    print('SMOKE PASS',len(off_events),len(on_events),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'summary':base.summary}[sys.argv[1]]()
