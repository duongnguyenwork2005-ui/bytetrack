"""Serial six-filter comparison on the unchanged R125 workload sample."""
from pathlib import Path
import hashlib
import json
import sys

import bdd_job as bdd
import numpy as np
import native_pf

HERE=Path(__file__).resolve().parent
job=bdd.base.job
OLD=bdd.base.OLD


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array(path):return np.loadtxt(path,delimiter=',',ndmin=2) if path.stat().st_size else np.empty((0,7))


def main():
    assert read(HERE/'bdd/summary.json')['pass']
    assert read(HERE/'test40/verification.json')['units']==360
    assert not (HERE/'serial_timing.json').exists()
    frozen=read(HERE/'bdd/inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    paths=[Path(__file__),HERE/'02_timing_protocol.md',HERE/'test40/summary.json']
    hashes={str(p):sha(p) for p in paths}
    job.write(HERE/'timing_inputs.json',{'sha256':hashes,'videos':frozen['timing_videos']})
    modes=('kf','ekf','ukf','classic500','native100','candidate100')
    rows=[];signatures={}
    for repeat in range(3):
        for video in frozen['timing_videos']:
            for mode in modes if repeat%2==0 else tuple(reversed(modes)):
                saved=job.installed
                if mode=='native100':job.installed=native_pf.installed
                try:mot,_,timing=job.replay(video,'pf100' if mode.endswith('100') else mode,0,200)
                finally:job.installed=saved
                values=mot.to_numpy(float);key=(mode,video)
                if key in signatures:assert np.array_equal(values,signatures[key])
                else:signatures[key]=values
                if mode=='candidate100':ref=HERE/f'bdd/tracking/{job.SPLIT}/pf100-s0/data/{video}.txt'
                elif mode=='native100':ref=OLD/f'tracking/{job.SPLIT}/pf100-s0/data/{video}.txt'
                elif mode=='classic500':ref=job.ROOT/f'data/processed/trackers/{job.SPLIT}/r74-bdd-full-motion200-s0/data/{video}.txt'
                else:ref=OLD/f'tracking/{job.SPLIT}/{mode}/data/{video}.txt'
                parity=None
                if ref.exists():
                    expected=array(ref);expected=expected[expected[:,0]<=200]
                    assert values.shape==expected.shape
                    assert np.array_equal(values[:,:2],expected[:,:2])
                    assert np.allclose(values[:,6],expected[:,6],atol=1e-9,rtol=0)
                    assert np.allclose(values[:,2:6],expected[:,2:6],atol=1e-3,rtol=0);parity=True
                else:assert mode=='classic500'
                elapsed=timing.t_update_ms.to_numpy()[16:];assert len(elapsed)==184
                row={'repeat':repeat,'video':video,'mode':mode,'frames':184,
                     'mean_update_ms':float(elapsed.mean()),'p90_update_ms':float(np.quantile(elapsed,.9)),
                     'full_prefix_parity':parity}
                rows.append(row);print(row,flush=True)
    means={m:float(np.mean([r['mean_update_ms'] for r in rows if r['mode']==m])) for m in modes}
    for p,h in {**frozen['sha256'],**hashes}.items():assert sha(p)==h,p
    job.write(HERE/'serial_timing.json',{'rows':rows,'mean_update_ms':means,
              'scope':'two BDD clips, PC tracker.update, 16 warmup, 3 reversed repeats; not Nano FPS'})
    print('TIMING PASS',means,flush=True)


if __name__=='__main__':main()
