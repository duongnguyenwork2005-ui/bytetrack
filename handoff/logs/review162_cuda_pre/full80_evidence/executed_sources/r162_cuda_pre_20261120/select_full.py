"""Follow the prospective R162 pilot rule; preserve all quality failures."""
from pathlib import Path
import hashlib,json,sys
from statistics import mean
root=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=root/'timing_models_v3.json';config=read(path);pilot=root/'runs_absolute/pilot'
done=read(pilot/'complete.json');inputs=read(pilot/'inputs.json');rows=read(pilot/'pairs.json')
assert done['status']=='PASS' and done['runs']==8 and done['models_sha256']==sha(path)
assert inputs['models']==config['models']
means={n:mean(r['fps_steady'] for r in rows if r['mode']==n) for n in config['models']}
assert all(sum(r['mode']==n for r in rows)==4 for n in means)
result={'pilot_mean_FPS':means,'full_skipped':means['gpu_pre']<=means['original'],
        'quality_FAIL_remains_FAIL':True,'pilot_complete_sha256':sha(pilot/'complete.json')}
target=root/'selection.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
if not result['full_skipped']:
    config['source_sha256'][str(target)]=sha(target)
    (root/'full_models.json').write_text(json.dumps(config,indent=2))
print('CUDA_FULL_SELECTION',json.dumps(result),flush=True)
