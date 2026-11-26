"""Exercise accepted and rejected native MH branches at a broad synthetic prior."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.linalg import expm
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r165_numerical_harness',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def main():
    target=HERE/'rejection_oracle.json';assert not target.exists()
    model=r.pair(181,ratio=1.)[1];model.audit=True
    # Test inputs, not an experimental tracking parameter change.
    model.pf.dt=1.;model.pf._std_omega=2.;model.incoming_var=4.;model.means[:,2:]=[12.,2.]
    accepted=rejected=steps=0;maximum=0.
    for tick in range(20):
        model.predict(lost=tick%7==2);model.update(np.array([65+12*tick,84+10*np.sin(tick/4),1.5,70.]),.8)
        audit=model.last_move
        if audit is None:continue
        steps+=1;mask=audit['accepted'];accepted+=int(mask.sum());rejected+=int((~mask).sum())
        omega=np.where(mask,audit['new'],audit['old']);cache=audit['cache'];noise=audit['noise'];z=audit['z']
        for child,w in enumerate(omega):
            a=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,-w],[0,0,w,0]],float);f=expm(a)
            h=cache['height'];qq=np.diag([(model.pf._std_weight_position*h)**2]*2+[(model.pf._std_weight_velocity*h)**2]*2)
            m=f@cache['means'][child];p=f@cache['covs'][child]@f.T+qq
            gain=np.linalg.solve(p[:2,:2]+noise[:2,:2],p[:2,:]).T
            mean=m+gain@(z[:2]-m[:2]);c=np.eye(4);c[:,:2]-=gain
            cov=c@p@c.T+gain@noise[:2,:2]@gain.T
            error=max(float(abs(mean-model.means[child]).max()),float(abs(cov-model.covariances[child]).max()))
            maximum=max(maximum,error);assert error<1e-6
    assert accepted>0 and rejected>0 and steps>0,(accepted,rejected,steps)
    r.j.write(target,{'status':'PASS','moves':steps,'accepted':accepted,'rejected':rejected,
        'posterior_oracles':accepted+rejected,'max_error':maximum,'tracking_hyperparameters_changed':False,
        'sha256':{str(p):r.j.sha(p) for p in (Path(__file__),HERE/'run.py',HERE/'rejuvenate.py')}})
    print('R165 ACTUAL REJECTION ORACLE PASS',accepted,rejected,maximum,flush=True)
if __name__=='__main__':main()
