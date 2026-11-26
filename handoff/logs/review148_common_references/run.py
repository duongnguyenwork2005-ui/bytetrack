"""Full-cohort comparator audit, reusing the unchanged R142 PF only."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent;R144=HERE.parent/'review144_shared_confidence'
sys.path.insert(0,str(R144))
spec=importlib.util.spec_from_file_location('r144_full_references',R144/'job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
j.HERE=HERE


def preflight(stage):
    assert stage in ('bdd','test40') and j.read(R144/'numerical.json')['status']=='PASS'
    target=HERE/stage/'inputs.json';assert not target.exists()
    c=j.case(stage)
    prior=j.R142/('bdd/inputs.json' if stage=='bdd' else 'test40/inputs.json')
    hashes=j.read(prior)['sha256'].copy()
    for p,h in hashes.items():assert j.sha(p)==h,p
    proof=j.read(j.R142/'final_verification.json');assert proof['goal_observed_met']
    paths=[Path(__file__),HERE/'00_protocol.md',R144/'job.py',R144/'shared.py',R144/'numerical.json',
           R144/'dev/summary.json',j.R142/'final_verification.json',prior]
    for m in j.METHODS:
        paths += [j.ref_raw(c,m)]+[j.ref_mot(c,m,v) for v in c['videos']]
    for s in range(3):
        raw=j.pf_raw(c,s);assert str(raw) in proof['sha256'] and j.sha(raw)==proof['sha256'][str(raw)]
        paths += [raw]+[j.pf_mot(c,s,v) for v in c['videos']]
    hashes.update({str(p):j.sha(p) for p in paths})
    j.write(target,{'sha256':hashes,'frames':c['frames'],'scope':'Frozen comparator audit; unchanged historical R142 PF',
                    'new_PF_tracking':False})
    print('PREFLIGHT',stage,len(c['videos']),sum(c['frames'].values()),flush=True)


def prepare():
    for stage in ('bdd','test40'):
        preflight(stage);j.smoke(stage)
    print('ALL REFERENCE PREFLIGHT/SMOKE PASS',flush=True)


def worker(index):
    tasks=[(stage,mode) for stage in ('bdd','test40') for mode in j.METHODS]
    for ix,(stage,mode) in enumerate(tasks):
        if ix%2!=index:continue
        j.full(stage,mode)
    print('REFERENCE WORKER COMPLETE',index,flush=True)


def summary():
    for stage in ('bdd','test40'):j.summary(stage)
    results={stage:j.read(HERE/stage/'summary.json') for stage in ('bdd','test40')}
    j.write(HERE/'summary.json',{'new_reference_video_runs':720,'new_PF_video_runs':0,
           'reused_PF_video_runs':720,'datasets':results,'both_full_gates':all(x['pass'] for x in results.values()),
           'dev_failed_hypotheses_not_promoted':['R144','R146','R147']})
    print('FULL REFERENCE AUDIT COMPLETE',all(x['pass'] for x in results.values()),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='worker':worker(int(sys.argv[2]))
    elif sys.argv[1]=='summary':summary()
