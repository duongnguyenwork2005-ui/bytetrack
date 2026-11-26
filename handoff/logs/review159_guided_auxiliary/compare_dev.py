"""Post-hoc paired-video IDSW uncertainty; does not reopen any failed gate."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;LOGS=HERE.parent
sys.path.insert(0,str(LOGS/'review144_shared_confidence'))
spec=importlib.util.spec_from_file_location('r159_dev_inputs',LOGS/'review144_shared_confidence/job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    videos=j.case('dev')['videos'];arrays={};files=[Path(__file__)]
    for name,review in [('guided','review153_guided_turn'),('lost_shape','review156_guided_shape'),
                         ('init_transport','review157_pf_init_transport'),('auxiliary','review159_guided_auxiliary')]:
        folder=LOGS/review;summary=j.read(folder/'dev/summary.json')
        assert j.read(folder/'dev/verification.json')['status']=='PASS';rows=[]
        for seed in range(3):
            p=folder/'dev/evaluation'/f'pf-s{seed}/raw_fields.json';raw=j.read(p);files.append(p)
            assert set(raw)==set(videos);rows.append([raw[v]['IDSW'] for v in videos])
        arrays[name]=np.mean(rows,axis=0)
        assert abs(arrays[name].sum()-summary['scores']['pf']['IDSW'])<1e-8
        files.extend([folder/'dev/summary.json',folder/'dev/verification.json'])
    rows=[]
    for seed in range(3):
        p=j.pf_raw(j.case('dev'),seed);raw=j.read(p);files.append(p);rows.append([raw[v]['IDSW'] for v in videos])
    arrays['R142']=np.mean(rows,axis=0)
    for name,review,mode in [('KF','review144_shared_confidence','kf'),('E144','review144_shared_confidence','ekf'),
          ('U144','review144_shared_confidence','ukf'),('E154','review154_velocity_noise','ekf'),('U154','review154_velocity_noise','ukf')]:
        p=LOGS/review/'dev/evaluation'/mode/'raw_fields.json';raw=j.read(p);files.append(p)
        arrays[name]=np.array([raw[v]['IDSW'] for v in videos],dtype=float)
    rng=np.random.default_rng(159);sample=rng.integers(0,len(videos),size=(5000,len(videos)));results={}
    pairs=[('guided',r) for r in ('R142','KF','E144','U144','E154','U154')]
    pairs += [(r,'guided') for r in ('lost_shape','init_transport','auxiliary')]
    for candidate,reference in pairs:
        delta=arrays[candidate]-arrays[reference];boot=delta[sample].sum(axis=1)
        results[candidate+' minus '+reference]={'IDSW_delta':float(delta.sum()),
            'paired_video_CI95':np.quantile(boot,[.025,.975]).tolist(),
            'videos_lower':int((delta<0).sum()),'videos_equal':int((delta==0).sum()),'videos_higher':int((delta>0).sum())}
    result={'status':'PASS','post_hoc':True,'videos':videos,'bootstrap_replicates':5000,'bootstrap_seed':159,
            'mean_seed_IDSW':{name:float(a.sum()) for name,a in arrays.items()},'comparisons':results,
            'limitation':'Conditional on already-observed dev and model search; unadjusted multiple CIs, not blind-test or selection-bias correction.',
            'scenario_gate_changed':False,'sha256':{str(p):sha(p) for p in files}}
    j.write(HERE/'dev_comparison.json',result)
    for name,row in results.items():print(name,row['IDSW_delta'],row['paired_video_CI95'],flush=True)

if __name__=='__main__':main()
