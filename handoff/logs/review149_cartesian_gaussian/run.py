"""Numerical and staged quality evaluation for Cartesian EKF/UKF."""
from pathlib import Path
import importlib.util
import inspect
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;R144=HERE.parent/'review144_shared_confidence'
sys.path.insert(0,str(R144))
spec=importlib.util.spec_from_file_location('r149_shared_harness',R144/'job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
from filters import CartesianEKF,CartesianUKF,installed
legacy_installed=j.installed;j.installed=installed;j.HERE=HERE
METHODS=('ekf','ukf');CONTROLS=HERE.parent/'review148_common_references'


def reference(stage,mode):
    folder=R144 if stage=='dev' else CONTROLS
    return folder/stage/'evaluation'/mode/'raw_fields.json'


def reference_mot(stage,mode,video):
    folder=R144 if stage=='dev' else CONTROLS;c=j.case(stage)
    return folder/stage/'tracking'/c['split']/mode/'data'/f'{video}.txt'


def numerical():
    from scipy.integrate import solve_ivp
    rng=np.random.default_rng(149);maxjac=maxode=maxrotation=maxcv=maxupdate=0.
    for dt in (.04,.2,1.):
        model=CartesianEKF(dt=dt)
        for w in (0.,1e-10,-1e-7,.00099,.00101,-.1,1.,-2.):
            for _ in range(4):
                x=rng.normal(size=9);x[4]=w;x[6]=50.
                eps=1e-5
                numerical_f=np.column_stack([(model.f(x+eps*np.eye(9)[k])-model.f(x-eps*np.eye(9)[k]))/(2*eps) for k in range(9)])
                error=float(abs(model.jacobian(x)-numerical_f).max());maxjac=max(maxjac,error)
                assert np.allclose(model.jacobian(x),numerical_f,rtol=2e-6,atol=2e-8),error
                def ode(t,z):return np.array([z[2],z[3],-w*z[3],w*z[2]])
                oracle=solve_ivp(ode,(0,dt),x[:4],rtol=1e-11,atol=1e-12).y[:,-1]
                error=float(abs(oracle-model.f(x)[:4]).max());maxode=max(maxode,error)
                assert np.allclose(oracle,model.f(x)[:4],atol=1e-9,rtol=1e-9)
                angle=rng.uniform(-np.pi,np.pi);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                transform=np.eye(9);transform[:2,:2]=transform[2:4,2:4]=rot
                error=float(abs(model.f(transform@x)-transform@model.f(x)).max());maxrotation=max(maxrotation,error)
                assert error<1e-11
    mapping=[0,1,5,6,2,3,7,8]
    for cls in (CartesianEKF,CartesianUKF):
        model=cls(std_omega=0.,init_std_omega=0.);kf=j.ul_backend.KalmanFilterXYAH()
        z=np.array([100.,200.,1.2,50.]);m,p=model.initiate(z);a,b=kf.initiate(z)
        for t in range(80):
            m,p=model.predict(m,p);a,b=kf.predict(a,b)
            z=np.array([100+t*.7,200+t*.3,1.2+.01*np.sin(t/8),50.+np.sin(t/5)])
            confidence=(t%9)/10
            m,p=model.update(m,p,z,confidence);a,b=kf.update(a,b,z,confidence)
            error=max(float(abs(m[mapping]-a).max()),float(abs(p[np.ix_(mapping,mapping)]-b).max()));maxcv=max(maxcv,error)
            assert np.allclose(m[mapping],a,atol=1e-7,rtol=1e-9)
            assert np.allclose(p[np.ix_(mapping,mapping)],b,atol=1e-7,rtol=1e-9)
        assert m[2]>0 and m[3]>0
    ukf=CartesianUKF()
    for _ in range(100):
        a=rng.normal(size=(9,9));p=a@a.T+np.eye(9)*.01;m=rng.normal(size=9);m[6]=50.
        z=np.array([2.,3.,1.2,51.]);confidence=.7
        points=ukf.sigma_points(m,p);observed=points@ukf._update_mat.T;zm=ukf.Wm@observed
        dx=points-m;dz=observed-zm
        s=(dz*ukf.Wc[:,None]).T@dz+ukf._measurement_noise(m[6],confidence)
        cross=(dx*ukf.Wc[:,None]).T@dz;gain=np.linalg.solve(s,cross.T).T
        expected_m=m+gain@(z-zm);expected_p=p-gain@s@gain.T
        actual_m,actual_p=ukf.update(m,p,z,confidence)
        error=max(float(abs(expected_m-actual_m).max()),float(abs(expected_p-actual_p).max()));maxupdate=max(maxupdate,error)
        assert np.allclose(expected_m,actual_m,atol=1e-10) and np.allclose(expected_p,actual_p,atol=1e-10)
    for cls in (CartesianEKF,CartesianUKF):
        model=cls();m,p=model.initiate(np.array([50.,70.,1.5,60.]))
        for t in range(160):
            m,p=model.predict(m,p)
            if t%13<9:m,p=model.update(m,p,np.array([50+3*t,70+20*np.sin(t/20),1.5,60.]),.8)
            assert np.isfinite(m).all() and np.linalg.eigvalsh(p).min()>-1e-8
            before=(m.copy(),p.copy());x,q,ok=model.try_initiate_from_motion(m,p,np.ones(4),ref_pos=(0,0))
            assert ok and np.array_equal(x,before[0]) and np.array_equal(q,before[1])
    trackers=(j.ul_backend.BYTETracker,j.tc.CTRVByteTracker,j.tc.UKFByteTracker)
    originals=[c.track_class for c in trackers]
    try:
        with installed():
            assert j.tc.CTRVByteTracker.track_class.filter_class is CartesianEKF
            assert j.tc.UKFByteTracker.track_class.filter_class is CartesianUKF
            raise RuntimeError('restore fixture')
    except RuntimeError:pass
    assert originals==[c.track_class for c in trackers]
    j.write(HERE/'numerical.json',{'status':'PASS','jacobian_ODE_rotation_cases':96,'CV_steps':160,
             'sigma_update_cases':100,'PSD_motion_steps':320,'max_jacobian_error':maxjac,'max_ODE_error':maxode,
             'max_rotation_error':maxrotation,'max_CV_error':maxcv,'max_update_error':maxupdate,'restore':True})
    print('CARTESIAN NUMERICAL PASS',flush=True)


def preflight(stage):
    assert j.read(HERE/'numerical.json')['status']=='PASS'
    assert j.read(CONTROLS/'final_verification.json')['status']=='PASS'
    if stage!='dev':assert j.read(HERE/('dev' if stage=='bdd' else 'bdd')/'summary.json')['pass']
    target=HERE/stage/'inputs.json';assert not target.exists();c=j.case(stage)
    old=R144/'dev/inputs.json' if stage=='dev' else CONTROLS/stage/'inputs.json'
    frozen=j.read(old)['sha256'].copy()
    for p,h in frozen.items():assert j.sha(p)==h,p
    paths=[Path(__file__),HERE/'filters.py',HERE/'00_protocol.md',HERE/'numerical.json',R144/'job.py',R144/'shared.py',
           old,CONTROLS/'final_verification.json']
    for method in j.METHODS:
        paths += [reference(stage,method)]+[reference_mot(stage,method,v) for v in c['videos']]
    for seed in range(3):paths += [j.pf_raw(c,seed)]+[j.pf_mot(c,seed,v) for v in c['videos']]
    frozen.update({str(p):j.sha(p) for p in paths})
    j.write(target,{'sha256':frozen,'frames':c['frames'],'new_Gaussian_methods':['CartesianEKF','CartesianUKF'],
                   'new_PF_tracking':False,'KF_unchanged_commonR':True})
    print('PREFLIGHT PASS',stage,flush=True)


def smoke(stage):
    j.check(stage);c=j.case(stage);rows=[]
    for v in (c['videos'][0],c['videos'][-1]):
        for method in METHODS:
            j.installed=legacy_installed
            try:old,_,_=j.replay(c,v,method,length=128)
            finally:j.installed=installed
            j.equal_prefix(old.to_numpy(float),reference_mot(stage,method,v),128)
            a,ac,_=j.replay(c,v,method,length=128);b,bc,_=j.replay(c,v,method,length=128)
            assert a.equals(b) and ac.equals(bc)
            rows.append({'video':v,'method':method,'historical_control':True,'candidate_repeat':True})
        for seed in range(3):
            a,_,_=j.replay(c,v,'pf',seed,length=128)
            j.equal_prefix(a.to_numpy(float),j.pf_mot(c,seed,v),128)
    j.check(stage);j.write(HERE/stage/'smoke.json',{'status':'PASS','rows':rows,'PF_parity':6})
    print('CARTESIAN SMOKE PASS',stage,flush=True)


def summary(stage):
    j.check(stage);c=j.case(stage)
    controls={m:j.boot_stats.combine(j.boot_stats.load_raw_fields(reference(stage,m)),c['videos']) for m in j.METHODS}
    scores={'kf':controls['kf']};files=[]
    for method in METHODS:
        path=HERE/stage/'evaluation'/method/'raw_fields.json';files.append(path)
        raw=j.boot_stats.load_raw_fields(path);assert set(raw)==set(c['videos'])
        scores[method]=j.boot_stats.combine(raw,c['videos'])
        for v in c['videos']:
            root=HERE/stage/'tracking'/c['split']/method;marker=root/'meta'/f'{v}.json';entry=j.read(marker)
            assert entry['frames']==c['frames'][v]
            for rel,h in entry['sha256'].items():assert j.sha(root/rel)==h
            files.append(marker)
    pf=[j.boot_stats.combine(j.boot_stats.load_raw_fields(j.pf_raw(c,s)),c['videos']) for s in range(3)]
    scores['pf']={k:float(np.mean([s[k] for s in pf])) for k in j.boot_stats.METRICS}
    gates={}
    for m in METHODS:
        gates[m+'_below_KF']=scores[m]['IDSW']<scores['kf']['IDSW']
        gates[m+'_IDSW_not_worse_than_polar']=scores[m]['IDSW']<=controls[m]['IDSW']
        gates[m+'_HOTA']=scores[m]['HOTA']>=controls[m]['HOTA']-.001
        gates[m+'_FN']=scores[m]['FN']<=1.002*controls[m]['FN']
    component=all(gates.values());target=min(scores[m]['IDSW'] for m in METHODS)
    gates['PF_not_above_EKF_UKF']=scores['pf']['IDSW']<=target
    gates['PF_two_seeds']=sum(s['IDSW']<=target for s in pf)>=2
    result={'scores':scores,'polar_controls':controls,'PF_seeds':pf,'gates':gates,
            'component_pass':component,'pass':all(gates.values()),'new_PF_units':0}
    j.write(HERE/stage/'summary.json',result)
    j.write(HERE/stage/'verification.json',{'status':'PASS','new_units':len(c['videos'])*2,
             'sha256':{str(p):j.sha(p) for p in files}})
    print('CARTESIAN SUMMARY',stage,result,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='numerical':numerical()
    elif action=='full':
        assert sys.argv[3] in METHODS;j.full(sys.argv[2],sys.argv[3])
    else:{'preflight':preflight,'smoke':smoke,'summary':summary}[action](sys.argv[2])
