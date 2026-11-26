from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review118_pf_shape'))
import run as job
PARENT=job.HERE
OLD_CFG=job.CFG
job.HERE=HERE
job.CFG=HERE/'rbpf100.yaml'
job.parent=lambda s:PARENT/f'dev/evaluation/cartesian-s{s}/raw_fields.json'


def preflight():
    assert not (HERE/'inputs.json').exists()
    prior=json.loads((PARENT/'inputs.json').read_text())
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'00_protocol.md',job.CFG,PARENT/'run.py']
    paths += [job.parent(s) for s in range(3)]
    a=job.base.rt.cfg_to_dict(job.base.rt.load_tracker_cfg(OLD_CFG,{'track_buffer':30}))
    b=job.base.rt.cfg_to_dict(job.base.rt.load_tracker_cfg(job.CFG,{'track_buffer':30}))
    assert {k:(a[k],b[k]) for k in a if a[k]!=b[k]}=={'pf_num_particles':(500,100)}
    baseline=[job.base.boot_stats.combine(job.base.boot_stats.load_raw_fields(job.parent(s)),job.base.DEV) for s in range(3)]
    job.write(HERE/'inputs.json',{'frames':prior['frames'],'baseline_seeds':baseline,'config':b,'sha256':{str(p):job.base.sha(p) for p in paths}})


def checks():
    for seed in range(3):
        for video in job.base.SMOKE:
            job.CFG=OLD_CFG
            mot,_=job.replay(video,seed,True,128)
            old=np.loadtxt(PARENT/f'dev/tracking/DETRAC-all/cartesian-s{seed}/data/{video}.txt',delimiter=',')
            assert np.allclose(mot.to_numpy(float),old[old[:,0]<=128],atol=1e-9,rtol=0)
            job.CFG=HERE/'rbpf100.yaml'
            a,_=job.replay(video,seed,True,128);b,_=job.replay(video,seed,True,128)
            assert np.array_equal(a.to_numpy(float),b.to_numpy(float))
    job.write(HERE/'smoke.json',{'status':'PASS','parent_parity':6,'repeat':6})
    print('CHECKS PASS',flush=True)


def verify():
    frozen=json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.base.sha(Path(p))==h,p
    means={};raw_by_seed=[];units=0
    for s in range(3):
        raw=job.base.boot_stats.load_raw_fields(HERE/f'dev/evaluation/cartesian-s{s}/raw_fields.json')
        assert set(raw)==set(job.base.DEV)
        combined=job.base.boot_stats.combine(raw,job.base.DEV)
        recorded=json.loads((HERE/f'dev/evaluation/cartesian-s{s}/combined.json').read_text())
        assert all(abs(recorded[k]-combined[k])<1e-9 for k in job.base.METRICS)
        raw_by_seed.append(combined)
        for video in job.base.DEV:
            folder=HERE/f'dev/tracking/DETRAC-all/cartesian-s{s}'
            marker=json.loads((folder/f'meta/{video}.json').read_text())
            assert marker['frames']==frozen['frames'][video] and marker['mot_sha256']==job.base.sha(folder/f'data/{video}.txt')
            units+=1
    baseline={k:float(np.mean([r[k] for r in frozen['baseline_seeds']])) for k in job.base.METRICS}
    candidate={k:float(np.mean([r[k] for r in raw_by_seed])) for k in job.base.METRICS}
    conditions={'IDSW':candidate['IDSW']<=1.02*baseline['IDSW'],'HOTA':candidate['HOTA']>=baseline['HOTA']-.0025,'FN':candidate['FN']<=1.005*baseline['FN']}
    job.write(HERE/'dev_summary.json',{'baseline':baseline,'candidate':candidate,'seeds':raw_by_seed,'conditions':conditions,'pass':all(conditions.values())})
    job.write(HERE/'verification.json',{'status':'PASS','units':units,'source_hashes':len(frozen['sha256'])})
    print('SUMMARY',baseline,candidate,conditions,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='dev':job.dev(int(sys.argv[2]))
    else:{'preflight':preflight,'checks':checks,'verify':verify}[sys.argv[1]]()
