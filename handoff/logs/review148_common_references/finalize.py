"""Close the full comparator audit without changing any frozen experiment input."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import math
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r148_full_reference_audit',HERE/'run.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
j=audit.j


def exact_seed_bootstrap(pf, refs, videos, samples=5000):
    """Same video draws for all methods; average COMBINED across PF seeds exactly."""
    stacks={m:j.boot_stats._stack(raw,videos) for m,raw in refs.items()}
    pstacks=[j.boot_stats._stack(raw,videos) for raw in pf]
    metrics=('IDSW','HOTA','FN')
    pairs=[('ekf','kf'),('ukf','kf'),('pf','kf'),('pf','ekf'),('pf','ukf')]
    observed={m:j.boot_stats.combine_stacked(s) for m,s in stacks.items()}
    pfull=[j.boot_stats.combine_stacked(s) for s in pstacks]
    observed['pf']={k:float(np.mean([v[k] for v in pfull])) for k in metrics}
    values={f'{a}_minus_{b}':{k:np.empty(samples) for k in metrics} for a,b in pairs}
    rng=np.random.default_rng(148)
    for repeat in range(samples):
        ix=rng.integers(0,len(videos),len(videos))
        combined={m:j.boot_stats.combine_stacked({k:v[ix] for k,v in st.items()}) for m,st in stacks.items()}
        pvalues=[j.boot_stats.combine_stacked({k:v[ix] for k,v in st.items()}) for st in pstacks]
        combined['pf']={k:float(np.mean([v[k] for v in pvalues])) for k in metrics}
        for a,b in pairs:
            for k in metrics:values[f'{a}_minus_{b}'][k][repeat]=combined[a][k]-combined[b][k]
    result={}
    for a,b in pairs:
        result[f'{a}_minus_{b}']={}
        for k in metrics:
            lo,hi=np.percentile(values[f'{a}_minus_{b}'][k],[2.5,97.5])
            result[f'{a}_minus_{b}'][k]={'difference':observed[a][k]-observed[b][k],
                                       'ci_low':float(lo),'ci_high':float(hi)}
    return {'samples':samples,'seed':148,'unit':'paired video','PF':'exact mean of seed COMBINED metrics',
            'conditional_exploratory_CIs':True,'comparisons':result}


def main():
    result=j.read(HERE/'summary.json');files=[Path(__file__),HERE/'summary.json'];frozen={};units=0
    for stage in ('bdd','test40'):
        c=j.case(stage);summary=j.read(HERE/stage/'summary.json');proof=j.read(HERE/stage/'verification.json')
        assert proof['status']=='PASS' and summary['pass']==all(summary['gates'].values())
        assert proof['new_units']==proof['reused_PF_units']==len(c['videos'])*3
        assert j.read(HERE/stage/'smoke.json')['status']=='PASS'
        refs={}
        for mode in j.METHODS:
            path=HERE/stage/'evaluation'/mode/'raw_fields.json'
            raw=j.boot_stats.load_raw_fields(path);refs[mode]=raw
            assert set(raw)==set(c['videos'])
            recombined=j.boot_stats.combine(raw,c['videos'])
            for k,v in recombined.items():
                assert math.isfinite(v) and math.isclose(v,summary['scores'][mode][k],abs_tol=1e-12)
                assert math.isclose(v,j.read(path.with_name('combined.json'))[k],abs_tol=1e-12)
        pf=[j.boot_stats.load_raw_fields(j.pf_raw(c,s)) for s in range(3)]
        boot=exact_seed_bootstrap(pf,refs,c['videos']);j.write(HERE/stage/'bootstrap.json',boot)
        for frozen_record in (j.read(HERE/stage/'inputs.json')['sha256'],proof['sha256']):
            for p,h in frozen_record.items():
                if p in frozen:assert frozen[p]==h,p
                frozen[p]=h
        files += [HERE/stage/p for p in ('inputs.json','summary.json','verification.json','smoke.json','bootstrap.json')]
        units+=proof['new_units']
        result['datasets'][stage]['bootstrap']=boot
        print('BOOTSTRAP COMPLETE',stage,flush=True)
    assert units==720 and result['new_PF_video_runs']==0 and result['reused_PF_video_runs']==720
    for p,h in frozen.items():assert j.sha(p)==h,p
    goal=all(v['pass'] for v in result['datasets'].values())
    assert goal==result['both_full_gates']
    result.update({'status':'PASS','goal_observed_met':goal,'scenario_gate_open':goal,
                   'new_PF_method_selected':False,'default_changed':False,'Nano_measured':False,
                   'limitations':['Repeatedly observed datasets; selection effects are not removed by these CIs.',
                                  'PF is the unchanged historical R142 candidate, not a new improvement.',
                                  'R144/R146/R147 dev failures remain failures under their own protocols.']})
    j.write(HERE/'final_summary.json',result);files.append(HERE/'final_summary.json')
    j.write(HERE/'final_verification.json',{'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'goal_observed_met':goal,'scenario_gate_open':goal,'new_reference_units':units,'new_PF_units':0,
            'reused_PF_units':720,'frozen_records':len(frozen),'sha256':{str(p):j.sha(p) for p in files}})
    print('FINAL REFERENCE INTEGRITY PASS; full gate:',goal,flush=True)


if __name__=='__main__':main()
