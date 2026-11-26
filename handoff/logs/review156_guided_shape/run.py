"""One Lost-shape covariance ablation against the stronger R154 dev references."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;HELPER=HERE.parent/'review146_pf_shared_ordered';PARENT=HERE.parent/'review153_guided_turn'
GAUSS=HERE.parent/'review154_velocity_noise';R144=HERE.parent/'review144_shared_confidence'
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r156_generic_cohort',HELPER/'run_job.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('r156_preserved_shape',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
base.HERE=HERE;base.j.HERE=HERE;base.j.pf_installed=factory.installed;j=base.j
from guided import GuidedTurnPF
from pf_ctrv import PFCTRV


def parent_raw(seed):return PARENT/'dev/evaluation'/f'pf-s{seed}/raw_fields.json'
def parent_mot(video,seed):return PARENT/'dev/tracking/DETRAC-all'/f'pf-s{seed}/data/{video}.txt'
def control_raw(mode):return (R144 if mode=='kf' else GAUSS)/f'dev/evaluation/{mode}/raw_fields.json'
def make(cls,seed):
    pf=PFCTRV(num_particles=100,resample_ess_ratio=.9);m,p=pf.initiate(np.array([50.,80.,1.5,70.]));rng=np.random.default_rng(seed)
    return cls(pf,m,p,pf.sample_initial(m,p,rng),np.random.default_rng(seed+10))


def numerical():
    rng=np.random.default_rng(156);steps=0;maxerror=0.
    for seed in range(3):
        a=make(GuidedTurnPF,seed);b=make(factory.PreservedShapePF,seed);b.preserve_lost_cov=False
        for t in range(100):
            for method,args in [('predict',(t%13<4,)),('update',(np.array([50+4*t,80+8*np.sin(t/12),1.5,70.]),(t%11)/10))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                for field in ('means','covariances','omega','logw','shape_mean','shape_cov','incoming_mu'):
                    assert np.array_equal(getattr(a,field),getattr(b,field)),field
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.linalg.eigvalsh(q).min()>-1e-6;steps+=1
    for dt in (.04,.2,1.):
        for _ in range(100):
            model=make(factory.PreservedShapePF,1);model.pf.dt=dt
            a=rng.normal(size=(2,2));p=a@a.T+np.eye(2)*.1
            h=float(rng.uniform(1,300));model.shape_mean[1]=[h,rng.normal()];model.shape_cov[1]=p
            f=np.array([[1,dt],[0,1]]);expected=f@p@f.T+np.diag([(h/20)**2,(h/160)**2])
            model.predict(lost=True)
            assert np.allclose(model.shape_cov[1],expected,atol=1e-11,rtol=1e-12)
            assert np.array_equal(model.shape_mean[1],np.array([h,0.]))
            maxerror=max(maxerror,float(abs(model.shape_cov[1]-expected).max()))
    a=make(GuidedTurnPF,88);b=make(factory.PreservedShapePF,88)
    for t in range(40):
        a.predict(lost=True);b.predict(lost=True)
        for field in ('means','covariances','omega','logw','incoming_mu'):
            assert np.array_equal(getattr(a,field),getattr(b,field))
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        assert np.array_equal(a.shape_mean,b.shape_mean)
        assert b._proposal_cache['lost'] is False
    assert not np.array_equal(a.shape_cov,b.shape_cov)
    # No-Lost enabled path must also preserve the complete parent trajectory.
    a=make(GuidedTurnPF,89);b=make(factory.PreservedShapePF,89)
    for t in range(80):
        for method,args in [('predict',(False,)),('update',(np.array([50+3*t,80+t,1.5,70.]),.8))]:
            x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
            assert np.array_equal(x,y) and np.array_equal(p,q)
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'shape_oracles':300,
          'max_shape_cov_error':maxerror,'Lost_body_omega_RNG_steps':40,'tracked_parity_steps':160,'PSD':True})
    print('GUIDED SHAPE NUMERICAL PASS',flush=True)


def preflight():
    assert j.read(HERE/'numerical.json')['status']=='PASS'
    assert j.read(PARENT/'dev/summary.json')['pass'] and j.read(GAUSS/'dev/summary.json')['component_pass']
    old=PARENT/'dev/inputs.json';hashes=j.read(old)['sha256'].copy()
    for p,h in hashes.items():assert j.sha(p)==h,p
    for folder in (PARENT,GAUSS):
        for p,h in j.read(folder/'dev/inputs.json')['sha256'].items():assert j.sha(p)==h,p
        for p,h in j.read(folder/'dev/verification.json')['sha256'].items():assert j.sha(p)==h,p
    paths=[Path(__file__),HERE/'filter.py',HERE/'00_protocol.md',HERE/'numerical.json',HELPER/'run_job.py',
           PARENT/'guided.py',PARENT/'dev/summary.json',PARENT/'dev/verification.json',
           GAUSS/'filters.py',GAUSS/'dev/summary.json',GAUSS/'dev/verification.json']
    for s in range(3):paths += [parent_raw(s)]+[parent_mot(v,s) for v in j.case('dev')['videos']]
    for mode in j.METHODS:paths.append(control_raw(mode))
    hashes.update({str(p):j.sha(p) for p in paths})
    target=HERE/'dev/inputs.json';assert not target.exists()
    j.write(target,{'sha256':hashes,'frames':j.case('dev')['frames'],'reference_IDSW':{'kf':449,'ekf':435,'ukf':430},
                   'parent':'R153 guided PF N100','candidate':'R153 + mean-only Lost height trend reset'})
    print('GUIDED SHAPE PREFLIGHT PASS',flush=True)


def smoke():
    j.check('dev');c=j.case('dev');saved=j.pf_installed;j.pf_installed=lambda:factory.installed(False)
    try:
        for v in (c['videos'][0],c['videos'][-1]):
            for seed in range(3):
                mot,_,_=j.replay(c,v,'pf',seed,length=128);j.equal_prefix(mot.to_numpy(float),parent_mot(v,seed),128)
    finally:j.pf_installed=saved
    for v in (c['videos'][0],c['videos'][-1]):
        for seed in range(3):
            a,ac,_=j.replay(c,v,'pf',seed,length=128);b,bc,_=j.replay(c,v,'pf',seed,length=128)
            assert a.equals(b) and ac.equals(bc)
    j.check('dev');j.write(HERE/'dev/smoke.json',{'status':'PASS','disabled_parent_prefix':6,'candidate_repeat':6,
                                               'Gaussian_references_reused_and_hash_verified':36})
    print('GUIDED SHAPE SMOKE PASS',flush=True)


def summary():
    j.check('dev');c=j.case('dev');scores={m:j.boot_stats.combine(j.boot_stats.load_raw_fields(control_raw(m)),c['videos']) for m in j.METHODS}
    rows=[];paths=[]
    for seed in range(3):
        raw=base.candidate_raw('dev',seed);paths.append(raw)
        rows.append(j.boot_stats.combine(j.boot_stats.load_raw_fields(raw),c['videos']))
        for video in c['videos']:
            root=base.candidate_root('dev',seed);marker=root/'meta'/f'{video}.json';entry=j.read(marker)
            assert entry['frames']==c['frames'][video]
            for rel,h in entry['sha256'].items():assert j.sha(root/rel)==h
            paths.append(marker)
    scores['pf']={k:float(np.mean([r[k] for r in rows])) for k in j.boot_stats.METRICS}
    parents=[j.boot_stats.combine(j.boot_stats.load_raw_fields(parent_raw(s)),c['videos']) for s in range(3)]
    parent={k:float(np.mean([r[k] for r in parents])) for k in j.boot_stats.METRICS};target=min(scores[m]['IDSW'] for m in ('ekf','ukf'))
    gates={'E_below_KF':scores['ekf']['IDSW']<scores['kf']['IDSW'],'U_below_KF':scores['ukf']['IDSW']<scores['kf']['IDSW'],
           'PF_not_above_E_U':scores['pf']['IDSW']<=target,'two_seeds':sum(r['IDSW']<=target for r in rows)>=2,
           'HOTA':scores['pf']['HOTA']>=parent['HOTA']-.001,'FN':scores['pf']['FN']<=parent['FN']*1.002}
    j.write(HERE/'dev/summary.json',{'scores':scores,'parent':parent,'PF_seeds':rows,'gates':gates,'pass':all(gates.values())})
    j.write(HERE/'dev/verification.json',{'status':'PASS','new_units':36,'sha256':{str(p):j.sha(p) for p in paths}})
    print('GUIDED SHAPE SUMMARY',scores,gates,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for seed in range(3):base.full_pf('dev',seed)
    else:{'numerical':numerical,'preflight':preflight,'smoke':smoke,'summary':summary}[action]()
