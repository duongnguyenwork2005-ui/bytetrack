"""Close the frozen R132 comparison after full quality, bootstrap and serial timing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def mean(rows,field):return sum(r[field] for r in rows)/len(rows)


def main():
    hashes={}
    for path in (HERE/'inputs.json',HERE/'bdd/inputs.json',HERE/'test40/inputs.json',HERE/'timing_inputs.json'):
        for p,h in read(path)['sha256'].items():
            if p in hashes:assert hashes[p]==h,p
            hashes[p]=h
    for p,h in hashes.items():assert sha(p)==h,p
    for folder,units in ((HERE,36),(HERE/'bdd',600),(HERE/'test40',360)):
        proof=read(folder/'verification.json')
        assert proof['status']=='PASS' and proof['units']==units
    bdd=read(HERE/'bdd/summary.json');ua=read(HERE/'test40/summary.json')
    assert bdd['pass']==all(bdd['gates'].values())
    assert ua['pass']==all(ua['gates'].values())
    minimum=min(ua['scores'][m]['mean']['IDSW'] for m in ('kf','ekf','ukf'))
    observed=ua['scores']['candidate100']['mean']['IDSW']<minimum
    assert observed==ua['gates']['IDSW_vs_all']
    timing=read(HERE/'serial_timing.json');rows=timing['rows']
    assert len(rows)==len({(r['repeat'],r['video'],r['mode']) for r in rows})==36
    for mode,value in timing['mean_update_ms'].items():
        matching=[r for r in rows if r['mode']==mode]
        assert len(matching)==6 and all(r['frames']==184 for r in matching)
        assert math.isclose(mean(matching,'mean_update_ms'),value,rel_tol=1e-12)
        if mode!='classic500':assert all(r['full_prefix_parity'] is True for r in matching)
    index={(r['repeat'],r['video'],r['mode']):r for r in rows}
    pairs={}
    for mode in ('native100','classic500','kf','ekf','ukf'):
        values=[]
        for r in rows:
            if r['mode']=='candidate100':
                before=index[r['repeat'],r['video'],mode]['mean_update_ms']
                values.append(100*(1-r['mean_update_ms']/before))
        pairs[mode]={'mean_time_reduction_pct':100*(1-timing['mean_update_ms']['candidate100']/timing['mean_update_ms'][mode]),
                     'paired_reductions_pct':values,'faster_pairs':sum(v>0 for v in values)}
    ua_boot=read(HERE/'test40/bootstrap.json');bdd_boot=read(HERE/'bdd/bootstrap.json')
    assert ua_boot['samples']==bdd_boot['samples']==5000
    result={'status':'PASS','date_local':'2026-11-19','goal_observed_met':bool(bdd['pass'] and ua['pass']),
            'BDD':{'videos':200,'frames':39973,'candidate':bdd['candidate'],'seeds':bdd['seeds'],
                   'references':bdd['references'],'gates':bdd['gates'],
                   'bootstrap':bdd_boot['candidate_minus_reference']},
            'UA_test40':{'videos':40,'frames':56340,'scores':ua['scores'],'gates':ua['gates'],
                         'bootstrap':ua_boot['candidate_minus_reference']},
            'PC_timing':{'mean_update_ms':timing['mean_update_ms'],'candidate_vs_reference':pairs,'scope':timing['scope']},
            'quality_video_runs':996,'timing_runs':36,'frozen_input_hashes':len(hashes),
            'effective_noise_height':{'native100':'observation','candidate100':'prediction'},
            'default_changed':False,'Nano_measured':False,
            'limitations':['Datasets observed in prior rounds; not blind tests.',
                           'Lower observed mean IDSW does not establish significance when its video-bootstrap CI crosses zero.',
                           'PC tracker timing from two selected workload clips is not whole-system or Nano FPS.']}
    (HERE/'final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs=[Path(__file__),HERE/'final_summary.json',HERE/'resume_check.json']
    for folder in (HERE,HERE/'bdd',HERE/'test40'):
        outputs += [folder/'inputs.json',folder/'smoke.json',folder/'verification.json']
    outputs += [HERE/'dev_summary.json',HERE/'bdd/summary.json',HERE/'bdd/bootstrap.json',
                HERE/'test40/summary.json',HERE/'test40/bootstrap.json',HERE/'serial_timing.json',HERE/'timing_inputs.json']
    for seed in range(3):
        outputs += [HERE/f'dev/evaluation/cartesian-s{seed}/raw_fields.json',HERE/f'bdd/evaluation/pf100-s{seed}/raw_fields.json']
    for mode in ('kf','ekf','ukf','native100','candidate100'):
        for seed in (range(3) if mode.endswith('100') else (0,)):
            tag=f'{mode}-s{seed}' if mode.endswith('100') else mode
            outputs += [HERE/f'test40/evaluation/{tag}/raw_fields.json']
    proof={'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'quality_video_runs':996,'timing_runs':36,'goal_observed_met':result['goal_observed_met'],
           'sha256':{str(p):sha(p) for p in outputs}}
    (HERE/'final_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINAL PASS; observed goal met:',result['goal_observed_met'],flush=True)
    print('BDD IDSW',bdd['candidate']['IDSW'],'UA',ua['scores']['candidate100']['mean']['IDSW'],flush=True)
    print('PC update ms',timing['mean_update_ms'],flush=True)


if __name__=='__main__':main()
