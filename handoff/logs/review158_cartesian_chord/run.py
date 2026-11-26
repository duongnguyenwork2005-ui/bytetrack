"""One initialization ablation, retaining the stronger R154 dev controls."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'review149_cartesian_gaussian'
STRONG=HERE.parent/'review154_velocity_noise'
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('r158_cartesian_harness',PARENT/'run.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
spec=importlib.util.spec_from_file_location('r158_chord_factory',HERE/'filters.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
t.HERE=HERE;t.j.HERE=HERE;t.installed=factory.installed;t.j.installed=factory.installed;j=t.j

def numerical():
    rng=np.random.default_rng(158);off=cases=psd=0
    for parent,child in ((factory.CartesianEKF,factory.ChordEKF),(factory.CartesianUKF,factory.ChordUKF)):
        a=parent();b=child();b.chord_init=False
        m,p=a.initiate(np.array([50.,80.,1.5,70.]));x,q=b.initiate(np.array([50.,80.,1.5,70.]))
        for tick in range(80):
            m,p=a.predict(m,p);x,q=b.predict(x,q)
            assert np.array_equal(m,x) and np.array_equal(p,q);off+=1
            z=np.array([50+3*tick,80+5*np.sin(tick/11),1.5,70.])
            m,p=a.update(m,p,z,.8);x,q=b.update(x,q,z,.8)
            assert np.array_equal(m,x) and np.array_equal(p,q);off+=1
            mm,pp,ok=a.try_initiate_from_motion(m,p,z,1.,(0.,0.));xx,qq,kk=b.try_initiate_from_motion(x,q,z,1.,(0.,0.))
            assert ok==kk and np.array_equal(mm,xx) and np.array_equal(pp,qq)
        for dt in (.04,.2,1.):
            model=child(dt=dt)
            for gap in (1.,2.,4.):
                for angle in np.linspace(-np.pi,np.pi,9):
                    m=rng.normal(size=9);m[6]=70.;a=rng.normal(size=(9,9));p=a@a.T+np.eye(9)*.01
                    saved=(m.copy(),p.copy());z=np.array([70.,90.,1.5,70.]);d=20*np.array([np.cos(angle),np.sin(angle)])
                    x,q,ok=model.try_initiate_from_motion(m,p,z,gap,z[:2]-d);assert ok
                    v=d/(dt*gap);speed=np.linalg.norm(v);direction=d/20;across=np.array([-direction[1],direction[0]])
                    expected=(2*70/160*10)**2*np.outer(direction,direction)+(speed*model._heading_std_after_init)**2*np.outer(across,across)
                    assert np.allclose(x[2:4],v) and np.allclose(q[2:4,2:4],expected)
                    keep=[0,1,4,5,6,7,8]
                    assert np.array_equal(x[keep],m[keep]) and np.array_equal(q[np.ix_(keep,keep)],p[np.ix_(keep,keep)])
                    assert np.array_equal(m,saved[0]) and np.array_equal(p,saved[1]) and np.linalg.eigvalsh(q).min()>0
                    xx,qq,no=model.try_initiate_from_motion(m,p,z,gap,z[:2]);assert not no
                    assert np.array_equal(xx,m) and np.array_equal(qq,p);cases+=1
        model=child();m,p=model.initiate(np.array([50.,80.,1.5,70.]));ref=m[:2].copy()
        for tick in range(160):
            m,p=model.predict(m,p);z=np.array([50+3*tick,80+10*np.sin(tick/12),1.5,70.])
            if tick==1:m,p,ok=model.try_initiate_from_motion(m,p,z,1.,ref);assert ok
            if tick%13<9:m,p=model.update(m,p,z,.7)
            assert np.isfinite(m).all() and np.linalg.eigvalsh(p).min()>-1e-8;psd+=1
    classes=(j.tc.CTRVByteTracker,j.tc.UKFByteTracker);old=[c.track_class for c in classes]
    with factory.installed():assert j.tc.CTRVByteTracker.track_class.filter_class is factory.ChordEKF
    assert [c.track_class for c in classes]==old
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':off,'chord_covariance_cases':cases,
          'curved_missing_updates_PSD_steps':psd,'input_ownership':True,'no_motion':True,'hook_restore':True})
    print('CARTESIAN CHORD NUMERICAL PASS',flush=True)

def preflight():
    t.preflight('dev');path=HERE/'dev/inputs.json';frozen=j.read(path)
    paths=[Path(__file__),PARENT/'run.py',PARENT/'filters.py',STRONG/'dev/summary.json',STRONG/'dev/verification.json']
    for mode in ('ekf','ukf'):paths.append(STRONG/'dev/evaluation'/mode/'raw_fields.json')
    frozen['sha256'].update({str(p):j.sha(p) for p in paths})
    frozen.update(hypothesis='R149 Cartesian CT + Native PF two-observation chord initialization',
                  strong_reference_IDSW={'kf':449,'ekf':435,'ukf':430})
    j.write(path,frozen)

def summary():
    t.summary('dev');path=HERE/'dev/summary.json';result=j.read(path);gates={}
    strong=j.read(STRONG/'dev/summary.json')['scores'];scores=result['scores']
    for mode in ('ekf','ukf'):
        gates[mode+'_below_KF']=scores[mode]['IDSW']<scores['kf']['IDSW']
        gates[mode+'_IDSW_not_worse_than_strong']=scores[mode]['IDSW']<=strong[mode]['IDSW']
        gates[mode+'_HOTA']=scores[mode]['HOTA']>=strong[mode]['HOTA']-.001
        gates[mode+'_FN']=scores[mode]['FN']<=strong[mode]['FN']*1.002
    result.update(strong_controls=strong,strong_gates=gates,component_pass=all(gates.values()),
                  pass_with_strong_reference=all(gates.values()) and result['gates']['PF_not_above_EKF_UKF'] and result['gates']['PF_two_seeds'],
                  full_or_scenario_started=False)
    j.write(path,result)
    print('STRONG REFERENCE COMPONENT',result['component_pass'],scores,gates,flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for mode in ('ekf','ukf'):j.full('dev',mode)
    else:{'numerical':numerical,'preflight':preflight,'smoke':lambda:t.smoke('dev'),'summary':summary}[action]()
