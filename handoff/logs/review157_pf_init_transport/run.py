"""Reuse the frozen R156 stronger-reference harness for one init ablation."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;TEMPLATE=HERE.parent/'review156_guided_shape/run.py'
spec=importlib.util.spec_from_file_location('r157_cohort_template',TEMPLATE)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
spec=importlib.util.spec_from_file_location('r157_transport_factory',HERE/'filter.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
t.HERE=HERE;t.factory=factory;t.base.HERE=HERE;t.base.j.HERE=HERE;t.j.pf_installed=factory.installed
j=t.j

def numerical():
    rng=np.random.default_rng(157);steps=0;initializations=0
    for seed in range(3):
        a=t.make(t.GuidedTurnPF,seed);b=t.make(factory.TransportPF,seed);b.preserve_init_cross=False
        for tick in range(80):
            z=np.array([50+6*tick,80+7*np.sin(tick/10),1.5,70.])
            for name,args in [('predict',(tick%11<3,)),('update',(z,.8))]:
                x,p=getattr(a,name)(*args);y,q=getattr(b,name)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state;steps+=1
            if tick in (2,20):
                aa=a.initialize_motion(x,p,z,1.,z[:2]-[6.,2.])
                bb=b.initialize_motion(y,q,z,1.,z[:2]-[6.,2.])
                assert aa[2]==bb[2] and np.array_equal(aa[0],bb[0]) and np.array_equal(aa[1],bb[1])
                initializations+=int(aa[2])
    for _ in range(500):
        a=rng.normal(size=(4,4));old=a@a.T+np.eye(4)*.1
        b=rng.normal(size=(2,2));target=b@b.T+np.eye(2)*.1
        actual=factory.transport(old[None],target[None])[0]
        lo=np.linalg.cholesky(old[2:,2:]);lt=np.linalg.cholesky(target)
        d=np.eye(4);d[2:,2:]=lt@np.linalg.inv(lo);expected=d@old@d.T
        assert np.allclose(actual,expected,atol=1e-11)
        assert np.allclose(actual[:2,:2],old[:2,:2]) and np.allclose(actual[2:,2:],target)
        assert np.linalg.eigvalsh(actual).min()>0
        assert np.allclose(np.linalg.solve(lt,actual[:2,2:].T),np.linalg.solve(lo,old[:2,2:].T))
    enabled=0
    for dt in (.04,.2,1.):
        for gap in (1.,2.,4.):
            for heading in np.linspace(-np.pi,np.pi,7):
                a=t.make(t.GuidedTurnPF,71);b=t.make(factory.TransportPF,71)
                a.pf.dt=b.pf.dt=dt
                x,p=a.predict();y,q=b.predict();old=b.covariances.copy()
                z=np.array([70.,90.,1.5,70.]);ref=z[:2]-20*np.array([np.cos(heading),np.sin(heading)])
                aa=a.initialize_motion(x,p,z,gap,ref);bb=b.initialize_motion(y,q,z,gap,ref)
                assert aa[2] and bb[2];enabled+=1
                assert np.array_equal(a.means,b.means) and np.array_equal(a.omega,b.omega)
                assert np.array_equal(a.logw,b.logw) and a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.allclose(b.covariances[:,:2,:2],old[:,:2,:2])
                assert np.allclose(b.covariances[:,2:,2:],a.covariances[:,2:,2:])
                assert not np.allclose(b.covariances[:,:2,2:],0)
                assert np.linalg.eigvalsh(b.covariances).min()>0 and b._proposal_cache is None
                assert np.linalg.eigvalsh(bb[1]).min()>-1e-6
                a.update(z,.7);b.update(z,.7)
                assert np.isfinite(b.covariances).all()
    a=t.make(t.GuidedTurnPF,72);b=t.make(factory.TransportPF,72);x,p=a.estimate();y,q=b.estimate()
    aa=a.initialize_motion(x,p,x[[0,1,5,6]],1.,x[:2]);bb=b.initialize_motion(y,q,y[[0,1,5,6]],1.,y[:2])
    assert aa[2] is False and bb[2] is False and np.array_equal(aa[0],bb[0]) and np.array_equal(aa[1],bb[1])
    original=t.factory.rb.ConditionalPF
    with factory.installed():assert issubclass(factory.rb.ConditionalPF,factory.TransportPF)
    assert factory.rb.ConditionalPF is original
    j.write(HERE/'numerical.json',{'status':'PASS','disabled_parent_steps':steps,'disabled_init_success':initializations,
          'SPD_oracles':500,'enabled_init_cases':enabled,'no_motion_parity':True,'hook_restore':True})
    print('INIT TRANSPORT NUMERICAL PASS',flush=True)

def preflight():
    t.preflight()
    path=HERE/'dev/inputs.json';inputs=j.read(path)
    inputs['candidate']='R153 guided + PSD init covariance transport'
    inputs['sha256'][str(Path(__file__))]=j.sha(Path(__file__))
    inputs['sha256'][str(TEMPLATE)]=j.sha(TEMPLATE)
    j.write(path,inputs)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='worker':
        for seed in range(3):t.base.full_pf('dev',seed)
    else:{'numerical':numerical,'preflight':preflight,'smoke':t.smoke,'summary':t.summary}[action]()
