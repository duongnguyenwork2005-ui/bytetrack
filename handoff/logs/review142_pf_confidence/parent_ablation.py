"""BDD paired analysis versus the identical PF core without score-conditioned R."""
from pathlib import Path
import json
import numpy as np
import bdd_job as bdd

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'review132_pf_predicted_noise'
job=bdd.base.job


def read(path):return json.loads(path.read_text())


def main():
    assert read(HERE/'bdd/verification.json')['units']==600
    proof=read(PARENT/'final_verification.json');assert proof['status']=='PASS'
    hashes={**read(HERE/'bdd/inputs.json')['sha256'],**proof['sha256']}
    for p,h in hashes.items():assert job.sha(p)==h,p
    paths=[Path(__file__),HERE/'00b_noise_ablation.md',HERE/'bdd/summary.json',
           HERE/'bdd/verification.json',PARENT/'bdd/summary.json',PARENT/'final_verification.json']
    raw_paths={}
    for mode,root in (('candidate',HERE),('parent',PARENT)):
        raw_paths[mode]=[root/f'bdd/evaluation/pf100-s{seed}/raw_fields.json' for seed in range(3)]
        paths+=raw_paths[mode]
        if mode=='parent':
            for p in raw_paths[mode]:assert str(p) in proof['sha256'],p
    hashes.update({str(p):job.sha(p) for p in paths})
    frozen={'unit':'video','videos':200,'seeds':[0,1,2],'samples':5000,'bootstrap_seed':128,'sha256':hashes}
    input_path=HERE/'parent_ablation_inputs.json'
    if input_path.exists():assert read(input_path)==frozen
    else:job.write(input_path,frozen)
    names=job.videos();assert len(names)==200
    raws={m:[job.boot_stats.load_raw_fields(p) for p in ps] for m,ps in raw_paths.items()}
    assert all(set(r)==set(names) for rows in raws.values() for r in rows)
    scores={m:[job.boot_stats.combine(r,names) for r in rows] for m,rows in raws.items()}
    means={m:{f:float(np.mean([r[f] for r in rows])) for f in job.boot_stats.METRICS} for m,rows in scores.items()}
    for mode,root in (('candidate',HERE),('parent',PARENT)):
        expected=read(root/'bdd/summary.json')['candidate']
        assert all(abs(means[mode][f]-expected[f])<1e-9 for f in job.boot_stats.METRICS)
    stacks={m:[job.boot_stats._stack(r,names) for r in rows] for m,rows in raws.items()}
    fields=('HOTA','IDF1','IDSW','FN','FP');samples={f:[] for f in fields}
    rng=np.random.default_rng(128)
    for _ in range(5000):
        index=rng.integers(0,len(names),size=len(names))
        boot={m:[job.boot_stats.combine_stacked({f:stack[f][index] for f in stack}) for stack in rows]
              for m,rows in stacks.items()}
        for field in fields:
            samples[field].append(float(np.mean([r[field] for r in boot['candidate']]))-
                                  float(np.mean([r[field] for r in boot['parent']])))
    result={'unit':'video','videos':200,'seeds':[0,1,2],'samples':5000,'bootstrap_seed':128,
            'scores':means,'candidate_minus_parent':
            {f:{'difference':means['candidate'][f]-means['parent'][f],
                'ci95':np.quantile(v,[.025,.975]).tolist()} for f,v in samples.items()},
            'scope':'PF+current-score R versus same PF fixed R; no pure algorithm superiority claim'}
    job.write(HERE/'parent_ablation.json',result)
    for p,h in hashes.items():assert job.sha(p)==h,p
    recorded=[Path(__file__),HERE/'00b_noise_ablation.md',input_path,HERE/'parent_ablation.json']+paths
    job.write(HERE/'parent_ablation_verification.json',{'status':'PASS','new_units':600,'reused_parent_units':600,
              'sha256':{str(p):job.sha(p) for p in recorded}})
    print('PARENT ABLATION PASS',result['candidate_minus_parent'],flush=True)


if __name__=='__main__':main()
