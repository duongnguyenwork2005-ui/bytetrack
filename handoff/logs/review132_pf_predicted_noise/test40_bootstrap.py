"""Paired video uncertainty for the complete test40 confirmation, without retuning."""
from pathlib import Path
import json
import numpy as np
import test40_job as job


def main():
    here=job.OUT;names=job.videos();summary=job.read(here/'summary.json');raws={}
    for mode in job.CONFIGS:
        raws[mode]=[job.boot_stats.load_raw_fields(here/f'evaluation/{job.tag(mode,s)}/raw_fields.json')
                    for s in (range(3) if mode.endswith('100') else (0,))]
    stacks={m:[job.boot_stats._stack(r,names) for r in rows] for m,rows in raws.items()}
    fields=('HOTA','IDF1','IDSW','FN');refs=('kf','ekf','ukf','native100')
    samples={m:{f:[] for f in fields} for m in refs};rng=np.random.default_rng(132)
    for _ in range(5000):
        idx=rng.integers(0,len(names),size=len(names))
        estimates={m:[job.boot_stats.combine_stacked({f:st[f][idx] for f in st}) for st in rows]
                   for m,rows in stacks.items()}
        means={m:{f:float(np.mean([r[f] for r in rows])) for f in fields} for m,rows in estimates.items()}
        for mode in refs:
            for f in fields:samples[mode][f].append(means['candidate100'][f]-means[mode][f])
    result={m:{f:{'difference':summary['scores']['candidate100']['mean'][f]-summary['scores'][m]['mean'][f],
                   'ci95':np.quantile(values,[.025,.975]).tolist()} for f,values in rows.items()} for m,rows in samples.items()}
    job.write(here/'bootstrap.json',{'unit':'video','samples':5000,'candidate_minus_reference':result})
    print('TEST40 BOOTSTRAP COMPLETE',flush=True)


if __name__=='__main__':main()
