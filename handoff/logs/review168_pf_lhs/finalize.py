"""Recheck the closed dev gate and report paired uncertainty, without retuning."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'review153_guided_turn'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    target=HERE/'final_verification.json';assert not target.exists()
    inputs=read(HERE/'dev/inputs.json');files={}
    for p,h in inputs['sha256'].items():assert sha(p)==h,p
    for p,h in read(HERE/'dev/verification.json')['sha256'].items():assert sha(p)==h,p
    for marker in (HERE/'dev/tracking/DETRAC-all').glob('pf-s*/meta/*.json'):
        record=read(marker);root=marker.parent.parent
        for rel,h in record['sha256'].items():
            p=root/rel;assert sha(p)==h,p;files[str(p)]=h
        files[str(marker)]=sha(marker)
    videos=list(inputs['frames']);current=[];parent=[];raw_paths=[]
    for seed in range(3):
        a=HERE/'dev/evaluation'/('pf-s%d'%seed)/'raw_fields.json'
        b=PARENT/'dev/evaluation'/('pf-s%d'%seed)/'raw_fields.json';raw_paths.extend((a,b))
        left,right=read(a),read(b);assert set(left)==set(right)==set(videos)
        current.append([left[v]['IDSW'] for v in videos]);parent.append([right[v]['IDSW'] for v in videos])
    current=np.array(current,float);parent=np.array(parent,float);delta=(current-parent).mean(axis=0)
    rng=np.random.default_rng(168);draws=rng.integers(0,len(videos),size=(5000,len(videos)))
    ci=np.quantile(delta[draws].sum(axis=1),[.025,.975]).tolist();summary=read(HERE/'dev/summary.json')
    assert float(current.sum(axis=1).mean())==summary['scores']['pf']['IDSW']
    activity=[read(HERE/'dev'/('seed%d_lhs_activity.json'%s)) for s in range(3)]
    count={key:sum(r[key] for r in activity) for key in activity[0]}
    assert count['calls']>0 and count['normal_draws']==100*count['calls']
    paths=[Path(__file__),HERE/'dev/summary.json',HERE/'dev/verification.json',HERE/'dev/inputs.json',
        HERE/'numerical.json',*raw_paths]
    files.update({str(p):sha(p) for p in paths})
    target.write_text(json.dumps({'status':'PASS','quality_gate_pass':summary['pass'],
        'seed_IDSW':current.sum(axis=1).tolist(),'parent_seed_IDSW':parent.sum(axis=1).tolist(),
        'mean_IDSW_delta_vs_parent':float(delta.sum()),'paired_video_bootstrap_CI95':ci,
        'bootstrap':'5000 paired video draws; mean three seeds per video, seed168',
        'activity':count,'marginal_sampling':'randomized LHS innovations; same Gaussian marginal, not full SQMC',
        'limitation':'Repeatedly used dev, post-hoc uncertainty and uncorrected multiple selection; not a blind win.',
        'sha256':files},indent=2)+'\n')
    print('R168 FINAL AUDIT PASS; QUALITY GATE',summary['pass'],'DELTA',float(delta.sum()),'CI95',ci,flush=True)
if __name__=='__main__':main()
