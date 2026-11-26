"""Verify frozen quality evidence and report the observed, not assumed, goal."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'review132_pf_predicted_noise'


def read(path):return json.loads(path.read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    hashes={};proof_files=[]
    for folder,units in ((HERE,36),(HERE/'bdd',600),(HERE/'test40',120)):
        frozen=read(folder/'inputs.json')
        for p,h in frozen['sha256'].items():
            if p in hashes:assert hashes[p]==h,p
            hashes[p]=h
        proof=read(folder/'verification.json')
        assert proof['status']=='PASS' and proof['units']==units
        proof_files += [folder/'inputs.json',folder/'verification.json',folder/'smoke.json']
    assert read(HERE/'test40/verification.json')['reference_units']==240
    for p,h in hashes.items():assert sha(p)==h,p
    bdd=read(HERE/'bdd/summary.json');ua=read(HERE/'test40/summary.json')
    goal=bool(bdd['pass'] and ua['pass'])
    assert bdd['pass']==all(bdd['gates'].values()) and ua['pass']==all(ua['gates'].values())
    bboot=read(HERE/'bdd/bootstrap.json');uboot=read(HERE/'test40/bootstrap.json')
    assert bboot['samples']==uboot['samples']==5000
    ablation=read(HERE/'parent_ablation.json')
    ablation_proof=read(HERE/'parent_ablation_verification.json')
    assert ablation['samples']==5000 and ablation_proof['status']=='PASS'
    assert ablation_proof['new_units']==ablation_proof['reused_parent_units']==600
    for p,h in read(HERE/'parent_ablation_inputs.json')['sha256'].items():
        if p in hashes:assert hashes[p]==h,p
        hashes[p]=h
        assert sha(p)==h,p
    for p,h in ablation_proof['sha256'].items():assert sha(p)==h,p
    proof_files += [HERE/'parent_ablation.py',HERE/'00b_noise_ablation.md',HERE/'parent_ablation_inputs.json',
                    HERE/'parent_ablation.json',HERE/'parent_ablation_verification.json']
    timing=None
    if goal:
        timing=read(HERE/'serial_timing.json');rows=timing['rows']
        assert len(rows)==len({(r['repeat'],r['video'],r['mode']) for r in rows})==42
        for mode,value in timing['mean_update_ms'].items():
            selected=[r for r in rows if r['mode']==mode]
            assert len(selected)==6 and all(r['frames']==184 for r in selected)
            assert math.isclose(sum(r['mean_update_ms'] for r in selected)/6,value,rel_tol=1e-12)
            if mode!='classic500':assert all(r['full_prefix_parity'] is True for r in selected)
        for p,h in read(HERE/'timing_inputs.json')['sha256'].items():assert sha(p)==h,p
        proof_files += [HERE/'serial_timing.json',HERE/'timing_inputs.json',HERE/'measure.py',HERE/'02_timing_protocol.md']
    data={'status':'PASS','goal_observed_met':goal,'date_local':'2026-11-19',
          'new_quality_video_runs':756,'reused_test40_reference_units':240,'reused_parent_ablation_units':720,
          'reused_UA_parent_ablation_units':120,'reused_BDD_parent_ablation_units':600,
          'BDD':{'videos':200,'frames':39973,**bdd,'bootstrap':bboot},
          'UA_test40':{'videos':40,'frames':56340,**ua,'bootstrap':uboot},
          'parent_R132':read(PARENT/'final_summary.json'),'BDD_parent_ablation':ablation,
          'comparison_scope':'Proposed PF+confidence-conditioned R versus frozen fixed-R KF/EKF/UKF; not pure inference-algorithm superiority',
          'PC_timing':timing,'default_changed':False,'Nano_measured':False,
          'limitations':['Datasets repeatedly observed during research; not independent blind tests.',
                         'Observed mean IDSW wins are not significance claims if video CIs include zero.',
                         'PF plus confidence-conditioned R is not a pure inference-algorithm ablation; classification score is not calibrated localization uncertainty.']}
    (HERE/'final_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    proof_files += [Path(__file__),HERE/'final_summary.json',HERE/'numerical.json',HERE/'dev_summary.json',
                    HERE/'bdd/summary.json',HERE/'bdd/bootstrap.json',HERE/'test40/summary.json',HERE/'test40/bootstrap.json']
    count=0
    for stage,split,tag in (('dev','DETRAC-all','cartesian'),('bdd','BDDCOCOVID-val','pf100'),('test40','DETRAC-test','candidate100')):
        for seed in range(3):
            raw_path=HERE/stage/'evaluation'/f'{tag}-s{seed}/raw_fields.json';proof_files.append(raw_path)
            meta=HERE/stage/'tracking'/split/f'{tag}-s{seed}'/'meta'
            expected=12 if stage=='dev' else (200 if stage=='bdd' else 40)
            markers=list(meta.glob('*.json'));assert len(markers)==expected
            proof_files+=markers;count+=len(markers)
    assert count==756
    proof={'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
           'new_quality_video_runs':756,'goal_observed_met':goal,'timing_runs':42 if timing else 0,
           'frozen_source_hashes':len(hashes),'frozen_markers':count,
           'sha256':{str(p):sha(p) for p in proof_files}}
    (HERE/'final_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('FINAL INTEGRITY PASS; observed goal:',goal,flush=True)
    print('IDSW BDD',bdd['candidate']['IDSW'],'UA test40',ua['scores']['candidate100']['mean']['IDSW'],flush=True)


if __name__=='__main__':main()
