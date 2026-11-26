"""Audit checkpointed test40 videos after the desktop update interruption."""
from pathlib import Path
from datetime import datetime,timezone
import test40_job as job

HERE=Path(__file__).resolve().parent


def main():
    frozen=job.base.read(job.OUT/'inputs.json')
    for p,h in frozen['sha256'].items():assert job.base.sha(p)==h,p
    progress=[];unmarked=[];bad=[];units=0
    for seed in range(3):
        tag=f'candidate100-s{seed}';folder=job.OUT/f'tracking/{job.base.SPLIT}/{tag}';done=0
        for video in job.base.videos():
            marker=folder/f'meta/{video}.json'
            if not marker.exists():
                for kind,suffix in (('data','.txt'),('timing','.csv'),('data_cls','.csv')):
                    path=folder/kind/f'{video}{suffix}'
                    if path.exists():unmarked.append(str(path))
                continue
            try:
                saved=job.base.read(marker)
                assert saved['frames']==frozen['frames'][video]
                for kind,suffix,key in (('data','.txt','mot_sha256'),('timing','.csv','timing_sha256'),('data_cls','.csv','classes_sha256')):
                    assert job.base.sha(folder/kind/f'{video}{suffix}')==saved[key]
            except Exception as error:
                bad.append({'marker':str(marker),'error':repr(error)});continue
            done+=1;units+=1
        scored=(job.OUT/f'evaluation/{tag}/combined.json').exists() and (job.OUT/f'evaluation/{tag}/raw_fields.json').exists()
        if scored:assert done==40
        progress.append({'seed':seed,'videos':done,'scored':scored})
    result={'status':'FAIL' if bad else 'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'source_hashes':len(frozen['sha256']),'validated_markers':units,'progress':progress,
            'invalid_markers':bad,'unmarked_artifacts_to_replay':unmarked,
            'interruption':'Codex Desktop update; continue authorized task'}
    job.base.write(HERE/'resume_check.json',result)
    assert not bad,bad
    print('RESUME PASS',units,'markers',len(frozen['sha256']),'hashes',progress,flush=True)


if __name__=='__main__':main()
