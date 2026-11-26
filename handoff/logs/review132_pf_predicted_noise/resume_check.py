"""Validate frozen inputs and completed video artefacts before resuming test40."""
from pathlib import Path
from datetime import datetime, timezone
import test40_job as job

HERE=Path(__file__).resolve().parent


def main():
    frozen=job.read(job.OUT/'inputs.json')
    for p,h in frozen['sha256'].items():
        assert job.sha(p)==h,p
    progress=[];units=0;unmarked=[]
    for mode in job.CONFIGS:
        for seed in (range(3) if mode.endswith('100') else (0,)):
            tag=job.tag(mode,seed);folder=job.OUT/f'tracking/{job.SPLIT}/{tag}'
            count=0;remaining_frames=0
            for video in job.videos():
                marker=folder/f'meta/{video}.json'
                if not marker.is_file():
                    remaining_frames+=frozen['frames'][video]
                    for kind,suffix in (('data','.txt'),('timing','.csv'),('data_cls','.csv')):
                        path=folder/kind/f'{video}{suffix}'
                        if path.exists():unmarked.append(str(path))
                    continue
                saved=job.read(marker)
                assert saved['frames']==frozen['frames'][video],str(marker)
                for kind,suffix,key in (('data','.txt','mot_sha256'),('timing','.csv','timing_sha256'),('data_cls','.csv','classes_sha256')):
                    assert job.sha(folder/kind/f'{video}{suffix}')==saved[key],(tag,video,key)
                count+=1;units+=1
            raw_path=job.OUT/f'evaluation/{tag}/raw_fields.json'
            combined_path=job.OUT/f'evaluation/{tag}/combined.json'
            scored=raw_path.is_file() and combined_path.is_file()
            if scored:
                assert count==40
                raw=job.boot_stats.load_raw_fields(raw_path)
                assert set(raw)==set(job.videos())
                combined=job.boot_stats.combine(raw,job.videos());recorded=job.read(combined_path)
                assert all(abs(combined[k]-recorded[k])<1e-9 for k in job.boot_stats.METRICS)
            progress.append({'mode':mode,'seed':seed,'videos':count,'scored':scored,'remaining_frames':remaining_frames})
    result={'status':'PASS','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'validated_markers':units,'source_hashes':len(frozen['sha256']),
            'progress':progress,'unmarked_artifacts_to_replay':unmarked}
    job.write(HERE/'resume_check.json',result)
    print('RESUME PASS',units,'markers',len(frozen['sha256']),'hashes',flush=True)
    print(progress,flush=True)


if __name__=='__main__':main()
