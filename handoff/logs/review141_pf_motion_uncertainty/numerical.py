from pathlib import Path
import sys
import json
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from motion_uncertainty import covariance_floor,MotionUncertaintyPF,PredictedNoisePF
from pf_ctrv import PFCTRV


def make(cls,seed):
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return cls(pf,mean,cov,particles,np.random.default_rng(seed+10))


def parity(a,b,aa,bb):
    assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
    for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
        assert np.array_equal(getattr(a,field),getattr(b,field)),field
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def main():
    rng=np.random.default_rng(141);minimum=0.
    for _ in range(1000):
        x=rng.normal(size=(2,2));y=rng.normal(size=(2,2))
        prior=x@x.T+.01*np.eye(2);noise=y@y.T+.01*np.eye(2)
        p,n=prior.copy(),noise.copy();new=covariance_floor(prior,noise)
        eigen=min(np.linalg.eigvalsh(new).min(),np.linalg.eigvalsh(new-prior).min(),np.linalg.eigvalsh(new-noise).min())
        minimum=min(minimum,float(eigen));assert eigen>-1e-11
        assert np.array_equal(prior,p) and np.array_equal(noise,n)
        larger=noise+prior
        assert np.array_equal(covariance_floor(larger,noise),larger)
    assert np.array_equal(covariance_floor(np.eye(2)*.01,np.eye(2)),np.eye(2))
    steps=0
    for seed in range(3):
        a=make(PredictedNoisePF,seed);b=make(MotionUncertaintyPF,seed);b.floor_velocity=False
        for t in range(80):
            aa=a.predict(t%11<3);bb=b.predict(t%11<3);parity(a,b,aa,bb);steps+=1
            z=np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)])
            if t==1:
                az=a.initialize_motion(*aa,z,1,np.array([50.,80.]))
                bz=b.initialize_motion(*bb,z,1,np.array([50.,80.]))
                assert az[2]==bz[2];parity(a,b,az[:2],bz[:2]);steps+=1
            parity(a,b,a.update(z,.4),b.update(z,.4));steps+=1
    changed=0;unchanged=0
    for displacement in (1.,40.):
        a=make(PredictedNoisePF,0);b=make(MotionUncertaintyPF,0)
        aa=a.predict(False);bb=b.predict(False)
        z=np.array([50+displacement,80.,1.5,70.])
        az=a.initialize_motion(*aa,z,1,np.array([50.,80.]))
        bz=b.initialize_motion(*bb,z,1,np.array([50.,80.]))
        assert az[2] and bz[2] and np.array_equal(az[0],bz[0])
        for field in ('means','omega','logw','shape_mean','shape_cov'):
            assert np.array_equal(getattr(a,field),getattr(b,field)),field
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        expected=a.covariances.copy();expected[:,2:,2:]=b.covariances[:,2:,2:]
        assert np.array_equal(expected,b.covariances)
        if displacement==1:
            changed+=1;assert b.velocity_covariance_floors==1
            velocity_noise=2*b.pf._measurement_noise(70.,None)[:2,:2]
            assert np.linalg.eigvalsh(b.covariances[0,2:,2:]-velocity_noise).min()>-1e-10
            assert np.linalg.eigvalsh(bz[1]).min()>-1e-7
        else:
            unchanged+=1;assert b.velocity_covariance_floors==0
            parity(a,b,az[:2],bz[:2])
        height=max(float(b.shape_mean[1,0]),1e-3)
        expected_noise=b.pf._measurement_noise(height,.4)[:2,:2].copy()
        b.update(z,.4)
        recorded=b.pf._measurement_noise(b.previous_noise_height,b.previous_confidence)[:2,:2]
        assert np.array_equal(expected_noise,recorded)
        detached=b.full_particles();detached[:]=0
        assert not np.array_equal(detached,b.full_particles())
    gap_variances=[]
    for gap in (1,2,4):
        model=make(MotionUncertaintyPF,0)
        for _ in range(gap):state=model.predict(False)
        result=model.initialize_motion(*state,np.array([54.,80.,1.5,70.]),gap,np.array([50.,80.]))
        assert result[2] and model.velocity_covariance_floors==1
        gap_variances.append(float(model.covariances[0,3,3]))
    assert np.allclose(np.array(gap_variances)*np.array([1,4,16]),gap_variances[0],atol=1e-12,rtol=0)
    report={'status':'PASS','PSD_floor_cases':1000,'min_constraint_eigenvalue':minimum,
            'control_steps':steps,'low_speed_changed':changed,'high_speed_noop':unchanged,
            'point_mean_unchanged':True,'RNG_unchanged':True,'noise_context_exact':True,'ownership':True,
            'gap_variances':gap_variances,'gap_scaling_inverse_T_squared':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
