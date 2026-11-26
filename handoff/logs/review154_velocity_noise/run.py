"""Component gate for a single preregistered Gaussian noise-model hypothesis."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;HELPER=HERE.parent/'review149_cartesian_gaussian'
sys.path.insert(0,str(HELPER))
spec=importlib.util.spec_from_file_location('r154_gaussian_cohort',HELPER/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('r154_noise_factory',HERE/'filters.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
base.HERE=HERE;base.j.HERE=HERE;base.installed=factory.installed;base.j.installed=factory.installed
j=base.j
from ekf_ctrv import EKFTrackerCTRV
from ukf_ctrv import UKFTrackerCTRV


def numerical():
    rng=np.random.default_rng(154);steps=0;maxrotation=0.
    for old,new in ((EKFTrackerCTRV,factory.AdaptiveEKF),(UKFTrackerCTRV,factory.AdaptiveUKF)):
        a,b=old(),new();b.coordinate_aware=False
        m,p=a.initiate(np.array([50.,80.,1.5,70.]));x,q=b.initiate(np.array([50.,80.,1.5,70.]))
        for t in range(120):
            m,p=a.predict(m,p);x,q=b.predict(x,q)
            assert np.array_equal(m,x) and np.array_equal(p,q);steps+=1
            z=np.array([50+3*t,80+20*np.sin(t/15),1.5,70.]);c=None if t%17==0 else (t%11)/10
            m,p=a.update(m,p,z,c);x,q=b.update(x,q,z,c)
            assert np.array_equal(m,x) and np.array_equal(p,q);steps+=1
            assert '_r154_prior_speed' not in b.__dict__
    for _ in range(500):
        height=float(rng.uniform(1,300));sigma=height/160;speed=float(rng.uniform(1,100))*sigma;angle=rng.uniform(-np.pi,np.pi)
        variance=factory.heading_variance(speed,height,1/160,0.)
        g=np.array([[np.cos(angle),-speed*np.sin(angle)],[np.sin(angle),speed*np.cos(angle)]])
        actual=g@np.diag([sigma**2,variance])@g.T;expected=np.eye(2)*sigma**2
        error=float(abs(actual-expected).max());maxrotation=max(maxrotation,error)
        assert np.allclose(actual,expected,atol=1e-11,rtol=1e-12)
    for cls in (factory.AdaptiveEKF,factory.AdaptiveUKF):
        model=cls();m,p=model.initiate(np.array([50.,80.,1.5,70.]))
        for t in range(150):
            m,p=model.predict(m,p);assert '_r154_prior_speed' not in model.__dict__
            if t%11>2:m,p=model.update(m,p,np.array([50+2*t,80+5*np.sin(t/12),1.5,70.]),.8)
            assert np.isfinite(m).all() and np.linalg.eigvalsh(p).min()>-1e-7
        for speed in (0.,-1e-12,1e-12,.01,1.,10.,100.):
            variance=factory.heading_variance(speed,70.,1/160,.01**2)
            assert .01**2<=variance<=1.
        model._r154_prior_speed=123.
        try:
            model.predict(np.zeros(9),np.eye(2))
        except (IndexError,ValueError):pass
        assert model._r154_prior_speed==123.
        del model._r154_prior_speed
    trackers=(j.ul_backend.BYTETracker,j.tc.CTRVByteTracker,j.tc.UKFByteTracker);original=[c.track_class for c in trackers]
    with factory.installed():pass
    assert original==[c.track_class for c in trackers]
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'rotation_cases':500,
          'max_rotation_covariance_error':maxrotation,'enabled_PSD_steps':300,'zero_speed_bounded':True,'context_restore':True})
    print('VELOCITY NOISE NUMERICAL PASS',flush=True)


def preflight(stage):
    if stage=='dev':base.preflight(stage)
    else:
        assert j.read(HERE/('dev' if stage=='bdd' else 'bdd')/'summary.json')['component_pass']
        assert j.read(HERE/'numerical.json')['status']=='PASS';c=j.case(stage)
        old=base.CONTROLS/stage/'inputs.json';target=HERE/stage/'inputs.json';assert not target.exists()
        hashes=j.read(old)['sha256'].copy()
        for p,h in hashes.items():assert j.sha(p)==h,p
        paths=[HERE/'00_protocol.md',HERE/'numerical.json',old,base.CONTROLS/'final_verification.json']
        for method in j.METHODS:paths += [base.reference(stage,method)]+[base.reference_mot(stage,method,v) for v in c['videos']]
        for seed in range(3):paths += [j.pf_raw(c,seed)]+[j.pf_mot(c,seed,v) for v in c['videos']]
        hashes.update({str(p):j.sha(p) for p in paths});j.write(target,{'sha256':hashes,'frames':c['frames'],'new_PF_tracking':False})
    path=HERE/stage/'inputs.json';frozen=j.read(path)
    paths=[Path(__file__),HERE/'filters.py',HELPER/'run.py',HERE.parent/'review144_shared_confidence/shared.py']
    frozen['sha256'].update({str(p):j.sha(p) for p in paths})
    frozen['hypothesis']='Prior-speed-dependent heading Q floor; same polar model and init';j.write(path,frozen)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='numerical':numerical()
    elif action=='full':assert sys.argv[3] in ('ekf','ukf');j.full(sys.argv[2],sys.argv[3])
    elif action=='preflight':preflight(sys.argv[2])
    else:{'smoke':base.smoke,'summary':base.summary}[action](sys.argv[2])
