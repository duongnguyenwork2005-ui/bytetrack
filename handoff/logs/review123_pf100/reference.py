"""Fresh deterministic KF/EKF/UKF replays on the exact PF dev inputs."""
from pathlib import Path
import json
import sys
import numpy as np
import run_job as wrapper
job=wrapper.job
HERE=wrapper.HERE
CONFIGS={'kf':'bytetrack.yaml','ekf':job.ROOT/'configs/bytetrack_ekf_ctrv.yaml','ukf':job.ROOT/'configs/bytetrack_ukf_ctrv.yaml'}


def preflight():
    assert not (HERE/'reference_inputs.json').exists()
    prior=json.loads((HERE/'inputs.json').read_text())
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),CONFIGS['ekf'],CONFIGS['ukf']]
    job.write(HERE/'reference_inputs.json',{'sha256':{str(p):job.base.sha(p) for p in paths},'configs':{k:job.base.rt.cfg_to_dict(job.base.rt.load_tracker_cfg(v,{'track_buffer':30})) for k,v in CONFIGS.items()},'seed':None})


def replay(method):
    job.CFG=CONFIGS[method]
    for video in job.base.DEV:
        folder=HERE/f'reference/tracking/DETRAC-all/{method}-s0';marker=folder/f'meta/{video}.json'
        assert not marker.exists()
        mot,timing=job.replay(video,None)
        output=folder/f'data/{video}.txt';output.parent.mkdir(parents=True,exist_ok=True)
        mot.to_csv(output,index=False,header=False,float_format='%.10f')
        job.write(marker,{'frames':len(timing),'mot_sha256':job.base.sha(output),'seed':None})
        print('REFERENCE',method,video,len(timing),flush=True)
    job.score(0,'DETRAC-all',job.base.DEV,mode=method,stage='reference')


def verify():
    frozen=json.loads((HERE/'reference_inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.base.sha(Path(p))==h,p
    results={};checks=0
    for method in CONFIGS:
        raw=job.base.boot_stats.load_raw_fields(HERE/f'reference/evaluation/{method}-s0/raw_fields.json')
        assert set(raw)==set(job.base.DEV)
        scores=job.base.boot_stats.combine(raw,job.base.DEV)
        import pandas as pd
        old=job.ROOT/f'results/trackeval/DETRAC-all/p2-kc48_r70_pf_dev-A-{"cv" if method=="kf" else method}/combined_metrics.csv'
        historical=pd.read_csv(old).iloc[0].to_dict()
        historical_parity={k:abs(scores[k]-float(historical[k])) for k in job.base.METRICS}
        for video in job.base.DEV:
            folder=HERE/f'reference/tracking/DETRAC-all/{method}-s0';meta=json.loads((folder/f'meta/{video}.json').read_text())
            assert meta['mot_sha256']==job.base.sha(folder/f'data/{video}.txt')
            checks+=1
        results[method]={'metrics':scores,'R70_abs_difference':historical_parity}
    job.write(HERE/'reference_summary.json',{'results':results,'units':checks,'status':'PASS'})
    print('REFERENCE',results,flush=True)


if __name__=='__main__':
    if sys.argv[1] in CONFIGS:replay(sys.argv[1])
    else:{'preflight':preflight,'verify':verify}[sys.argv[1]]()
