"""Serial baseline/fast/old-PF timing, before full only when smoke is valid."""
import json
import numpy as np
from pathlib import Path
import runner as job


def main():
    assert (job.HERE/'smoke.json').is_file()
    assert not (job.HERE/'serial_timing.json').exists()
    frozen=json.loads((job.HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.bdd.sha(p)==h,p
    rows=[];signatures={}
    modes=('classic500','parent100','fast100')
    for repeat in range(3):
        for name in frozen['timing_videos']:
            for mode in modes if repeat%2==0 else tuple(reversed(modes)):
                mot,_,timing=job.replay(name,mode,0,200)
                key=(mode,name);values=mot.to_numpy(float)
                if key in signatures:assert np.array_equal(values,signatures[key])
                else:signatures[key]=values
                a=timing.t_update_ms.to_numpy()[16:]
                row={'repeat':repeat,'video':name,'mode':mode,'frames':len(a),
                     'mean_update_ms':float(a.mean()),'p90_update_ms':float(np.quantile(a,.9))}
                rows.append(row);print(row,flush=True)
    means={m:float(np.mean([r['mean_update_ms'] for r in rows if r['mode']==m])) for m in modes}
    gate={'speed20':means['fast100'] <= .8*means['parent100'],
          'vs_classic':means['fast100'] <= means['classic500']}
    for p,h in frozen['sha256'].items():assert job.bdd.sha(p)==h,p
    job.write(job.HERE/'serial_timing.json',{'rows':rows,'mean_update_ms':means,'gates':gate,
          'scope':'two BDD clips, PC tracker.update, serial, 16warmup, not Nano FPS'})
    print('TIMING',means,gate,flush=True)


if __name__=='__main__':main()
