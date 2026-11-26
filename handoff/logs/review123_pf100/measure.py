"""Serial N500/N100 PC timing after all quality runs have exited."""
import numpy as np
import json
from pathlib import Path
import run_job as wrapper
job=wrapper.job


def run():
    assert not (wrapper.HERE/'serial_timing.json').exists()
    frozen=json.loads((wrapper.HERE/'inputs.json').read_text())['sha256']
    frozen={**frozen,str(Path(__file__)):job.base.sha(Path(__file__)),
            str(wrapper.HERE/'transfer_summary.json'):job.base.sha(wrapper.HERE/'transfer_summary.json')}
    for p,h in frozen.items():assert job.base.sha(Path(p))==h,p
    job.write(wrapper.HERE/'timing_inputs.json',{'sha256':frozen})
    rows=[];signatures={}
    for repeat in range(3):
        order=('N500','N100') if repeat%2==0 else ('N100','N500')
        for video in job.base.SMOKE:
            for mode in order:
                job.CFG=wrapper.OLD_CFG if mode=='N500' else wrapper.HERE/'rbpf100.yaml'
                mot,timing=job.replay(video,0,True,length=512)
                values=mot.to_numpy(float);key=(video,mode)
                if key in signatures:assert np.array_equal(values,signatures[key])
                else:signatures[key]=values
                measured=timing.t_update_ms.to_numpy()[32:]
                row={'repeat':repeat,'video':video,'mode':mode,'mean_update_ms':float(measured.mean()),'p90_update_ms':float(np.quantile(measured,.9)),'frames':len(measured)}
                rows.append(row);print(row,flush=True)
    means={k:float(np.mean([r['mean_update_ms'] for r in rows if r['mode']==k])) for k in ('N500','N100')}
    paired=[]
    for repeat in range(3):
        for video in job.base.SMOKE:
            pair={r['mode']:r['mean_update_ms'] for r in rows if r['repeat']==repeat and r['video']==video}
            paired.append({'repeat':repeat,'video':video,'reduction_percent':100*(1-pair['N100']/pair['N500'])})
    for p,h in frozen.items():assert job.base.sha(Path(p))==h,p
    job.write(wrapper.HERE/'serial_timing.json',{'rows':rows,'mean_update_ms':means,'reduction_percent':100*(1-means['N100']/means['N500']),'paired':paired,'hashes_pass':len(frozen),'scope':'PC tracker update, two512-frame prefixes,32warmup,three reversed-order repeats; not Nano FPS'})
    print(means,flush=True)


if __name__=='__main__':run()
