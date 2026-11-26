from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from quantile import standard_nodes,QuantilePF,PredictedNoisePF
from pf_ctrv import PFCTRV


def main():
    nodes=standard_nodes(100)
    assert not nodes.flags.writeable and standard_nodes(100) is nodes
    assert abs(nodes.mean())<1e-14 and abs(np.mean(nodes*nodes)-1)<1e-14
    assert np.allclose(nodes,-nodes[::-1],atol=1e-14,rtol=0)
    for seed in range(3):
        pf=PFCTRV(num_particles=100)
        mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
        particles=pf.sample_initial(mean,cov,np.random.default_rng(seed));original=particles.copy()
        rng=np.random.default_rng(seed+10);state=rng.bit_generator.state
        b=QuantilePF(pf,mean,cov,particles,rng)
        assert state==rng.bit_generator.state and np.array_equal(particles,original)
        assert abs(b.omega.mean()-mean[pf.OMEGA])<1e-14
        assert abs(np.var(b.omega)-cov[pf.OMEGA,pf.OMEGA])<1e-14
        forced=particles.copy();forced[:,pf.OMEGA]=b.omega
        a=PredictedNoisePF(pf,mean,cov,forced,np.random.default_rng(seed+10))
        for t in range(40):
            for method,args in [('predict',(t%11<3,)),('update',(np.array([50+t,80+3*np.sin(t/10),1.5,70.+np.sin(t/9)]),))]:
                x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
                assert np.array_equal(x,y) and np.array_equal(p,q)
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                assert np.isfinite(y).all() and np.linalg.eigvalsh(q).min()>-1e-8
    pf=PFCTRV(num_particles=100,init_std_omega=0.)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(2))
    a=PredictedNoisePF(pf,mean,cov,particles,np.random.default_rng(12))
    b=QuantilePF(pf,mean,cov,particles,np.random.default_rng(12))
    for t in range(40):
        for method,args in [('predict',(False,)),('update',(np.array([50+t,80.,1.5,70.]),))]:
            x,p=getattr(a,method)(*args);y,q=getattr(b,method)(*args)
            assert np.array_equal(x,y) and np.array_equal(p,q)
    report={'status':'PASS','N':100,'node_mean':float(nodes.mean()),'node_variance':float(np.var(nodes)),
            'readonly':True,'no_constructor_draw':True,'input_ownership':True,'kernel_parity_steps':240,
            'zero_variance_steps':80,'PSD':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
