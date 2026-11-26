"""Freeze the improved Gaussian component on both full cohorts before scoring."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r154_noise_component',HERE/'run.py')
component=importlib.util.module_from_spec(spec);spec.loader.exec_module(component)
base=component.base;j=component.j


def prepare():
    assert j.read(HERE/'dev/summary.json')['component_pass']
    assert j.read(HERE/'dev/verification.json')['status']=='PASS';j.check('dev')
    for stage in ('bdd','test40'):
        c=j.case(stage);old=base.CONTROLS/stage/'inputs.json';target=HERE/stage/'inputs.json';assert not target.exists()
        hashes=j.read(old)['sha256'].copy()
        for p,h in hashes.items():assert j.sha(p)==h,p
        paths=[Path(__file__),HERE/'01_full_audit.md',HERE/'run.py',HERE/'filters.py',HERE/'00_protocol.md',
               HERE/'numerical.json',HERE/'dev/inputs.json',HERE/'dev/summary.json',HERE/'dev/verification.json',
               component.HELPER/'run.py',old,base.CONTROLS/'final_verification.json',
               HERE.parent/'review144_shared_confidence/job.py',HERE.parent/'review144_shared_confidence/shared.py']
        for method in j.METHODS:paths += [base.reference(stage,method)]+[base.reference_mot(stage,method,v) for v in c['videos']]
        for seed in range(3):paths += [j.pf_raw(c,seed)]+[j.pf_mot(c,seed,v) for v in c['videos']]
        hashes.update({str(p):j.sha(p) for p in paths})
        j.write(target,{'sha256':hashes,'frames':c['frames'],'new_PF_tracking':False,
                        'scope':'Both full cohorts of the unchanged, dev-qualified Gaussian noise component'})
        base.smoke(stage)
    print('FULL GAUSSIAN AUDIT PREFLIGHT PASS',flush=True)


def worker():
    for stage in ('bdd','test40'):
        for method in ('ekf','ukf'):j.full(stage,method)
    print('FULL GAUSSIAN AUDIT WORKER COMPLETE',flush=True)


def summary():
    results={}
    for stage in ('bdd','test40'):
        base.summary(stage);record=j.read(HERE/stage/'summary.json')
        keys=['ekf_below_KF','ukf_below_KF','ekf_HOTA','ekf_FN','ukf_HOTA','ukf_FN']
        record['user_Gaussian_prerequisite']=all(record['gates'][k] for k in keys)
        results[stage]=record
    eligible=all(r['user_Gaussian_prerequisite'] for r in results.values())
    j.write(HERE/'full_audit_summary.json',{'new_Gaussian_video_runs':480,'new_PF_video_runs':0,
           'reused_KF_runs':240,'reused_PF_runs':720,'datasets':results,
           'Gaussian_prerequisite_both':eligible,'new_PF_goal_claimed':False,'scenario_gate_open':False})
    print('GAUSSIAN FULL PREREQUISITE',eligible,flush=True)


if __name__=='__main__':{'prepare':prepare,'worker':worker,'summary':summary}[sys.argv[1]]()
