from pathlib import Path
import sys
import json
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'src'))
from confidence import installed,view_class,score_scope,PENDING,PredictedNoisePF,rb
from pf_ctrv import PFCTRV


class FakeModel:
    def __init__(self):self.seen=[]
    def update(self,observation,confidence):
        self.seen.append(confidence)
        return np.asarray(observation).copy(),np.eye(4)


def make(seed):
    pf=PFCTRV(num_particles=100)
    mean,cov=pf.initiate(np.array([50.,80.,1.5,70.]))
    particles=pf.sample_initial(mean,cov,np.random.default_rng(seed))
    return PredictedNoisePF(pf,mean,cov,particles,np.random.default_rng(seed+10))


def main():
    fake=FakeModel();track=SimpleNamespace(rbpf=fake,score=.1,sync_rbpf=lambda:None)
    view=view_class()(object(),track)
    with score_scope(track,.8):
        view.update(None,None,np.ones(4))
        view.update(None,None,np.ones(4),.3)
    view.update(None,None,np.ones(4))
    assert fake.seen==[.8,.3,None] and not hasattr(track,PENDING)
    setattr(track,PENDING,.2)
    try:
        with score_scope(track,.9):raise RuntimeError('fixture')
    except RuntimeError:pass
    assert getattr(track,PENDING)==.2
    off=view_class(False)(object(),track);off.update(None,None,np.ones(4))
    assert fake.seen[-1] is None
    old_factory=rb.ConditionalPF;old_track=rb.RBPFByteTracker.track_class
    try:
        with installed(True):
            outer=rb.RBPFByteTracker.track_class
            with installed(False):assert rb.RBPFByteTracker.track_class is rb.RBPFSTrack
            assert rb.RBPFByteTracker.track_class is outer
            raise RuntimeError('fixture')
    except RuntimeError:pass
    assert rb.ConditionalPF is old_factory and rb.RBPFByteTracker.track_class is old_track
    steps=0
    for seed in range(3):
        a=make(seed);b=make(seed)
        target=SimpleNamespace(rbpf=b,score=.1,sync_rbpf=lambda:None)
        routed=view_class()(b.pf,target)
        for t in range(60):
            aa=a.predict(t%11<3);bb=b.predict(t%11<3)
            assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
            confidence=(0.,.3,.8,1.,None)[t%5]
            z=np.array([50+6*t,80+10*np.sin(t/8),1.5,70.+np.sin(t/9)])
            aa=a.update(z,confidence)
            if confidence is not None:
                with score_scope(target,confidence):bb=routed.update(None,None,z)
            else:bb=routed.update(None,None,z)
            assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
            for field in ('means','covariances','omega','logw','shape_mean','shape_cov'):
                assert np.array_equal(getattr(a,field),getattr(b,field)),field
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            assert np.linalg.eigvalsh(bb[1]).min()>-1e-6;steps+=2
        base=a.pf._measurement_noise(70.,None)
        for confidence in (0.,.3,.8,1.):
            assert np.array_equal(a.pf._measurement_noise(70.,confidence),base*max(1-confidence,.05))
    report={'status':'PASS','current_explicit_none_off_routing':True,'no_stale_confidence':True,
            'context_exception_restore':True,'direct_confidence_parity_steps':steps,
            'RNG_unchanged':True,'PSD':True,'existing_R_formula_exact':True}
    (HERE/'numerical.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERICAL PASS',report,flush=True)


if __name__=='__main__':main()
