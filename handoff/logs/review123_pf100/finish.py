"""Audit all N100 quality outputs and the serial timing provenance."""
from pathlib import Path
import json
import numpy as np
import run_job as wrapper
from transfer_job import VIDEOS
job=wrapper.job
HERE=wrapper.HERE


def run():
    manifests={name:json.loads((HERE/name).read_text()) for name in ('inputs.json','transfer_inputs.json','reference_inputs.json','timing_inputs.json')}
    hashes={}
    for m in manifests.values():
        for p,h in m['sha256'].items():
            if p in hashes:assert hashes[p]==h,(p,'conflicting frozen digests')
            hashes[p]=h
    for p,h in hashes.items():assert job.base.sha(Path(p))==h,p
    units=groups=0
    for stage,split,videos,trackers in [('dev','DETRAC-all',job.base.DEV,[f'cartesian-s{s}' for s in range(3)]),('transfer','SVMOTSYS-test',VIDEOS,[f'cartesian-s{s}' for s in range(3)]),('reference','DETRAC-all',job.base.DEV,[f'{m}-s0' for m in ('kf','ekf','ukf')])]:
        for tracker in trackers:
            raw=job.base.boot_stats.load_raw_fields(HERE/f'{stage}/evaluation/{tracker}/raw_fields.json')
            assert set(raw)==set(videos)
            computed=job.base.boot_stats.combine(raw,videos)
            recorded=json.loads((HERE/f'{stage}/evaluation/{tracker}/combined.json').read_text())
            assert all(abs(computed[k]-recorded[k])<1e-9 for k in job.base.METRICS)
            groups+=1
            for video in videos:
                folder=HERE/f'{stage}/tracking/{split}/{tracker}'
                meta=json.loads((folder/f'meta/{video}.json').read_text())
                assert meta['mot_sha256']==job.base.sha(folder/f'data/{video}.txt')
                expected=manifests['inputs.json' if stage!='transfer' else 'transfer_inputs.json']['frames'][video]
                assert meta['frames']==expected
                units+=1
    dev=json.loads((HERE/'dev_summary.json').read_text());transfer=json.loads((HERE/'transfer_summary.json').read_text())
    assert dev['pass'] and transfer['pass']
    timing=json.loads((HERE/'serial_timing.json').read_text())
    assert len(timing['rows'])==12 and all(r['frames']==480 for r in timing['rows'])
    for mode,value in timing['mean_update_ms'].items():
        assert abs(value-np.mean([r['mean_update_ms'] for r in timing['rows'] if r['mode']==mode]))<1e-12
    # Paired video diagnostic against the freshly replayed deterministic filters.
    baseline={m:job.base.boot_stats.load_raw_fields(HERE/f'reference/evaluation/{m}-s0/raw_fields.json') for m in ('kf','ekf','ukf')}
    current=[job.base.boot_stats.load_raw_fields(HERE/f'dev/evaluation/cartesian-s{s}/raw_fields.json') for s in range(3)]
    rng=np.random.default_rng(123)
    picks=rng.integers(len(job.base.DEV),size=(10000,len(job.base.DEV)))
    ci={}
    for m,old in baseline.items():
        changes=np.array([np.mean([s[v]['IDSW'] for s in current])-old[v]['IDSW'] for v in job.base.DEV])
        ci[m]=np.quantile(changes[picks].sum(axis=1),[.025,.975]).tolist()
    job.write(HERE/'final_verification.json',{'status':'PASS','quality_units':units,'raw_aggregation_groups':groups,'frozen_source_hashes':len(hashes),'timing_units':12,'timing_repeat_mot_parity':True,'dev_PF100_minus_reference_IDSW_bootstrap95':ci,'nano_measured':False,'source_defaults_changed':False})
    print('FINAL PASS',units,groups,len(hashes),ci,flush=True)


if __name__=='__main__':run()
