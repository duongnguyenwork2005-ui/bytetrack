"""Prospective four-clip speed selection; retain original and matched control."""
from pathlib import Path
import hashlib,json,sys
from statistics import mean
root=Path(sys.argv[1]).resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=root/'timing_models_v5_bound.json';config=read(path);pilot=root/'runs_absolute/pilot'
done=read(pilot/'complete.json');inputs=read(pilot/'inputs.json');rows=read(pilot/'pairs.json')
assert done['status']=='PASS' and done['runs']==4*len(config['models']) and done['models_sha256']==sha(path)
assert inputs['models']==config['models']
means={n:mean(r['fps_steady'] for r in rows if r['mode']==n) for n in config['models']}
assert all(sum(r['mode']==n for r in rows)==4 for n in means)
selected=[n for n in means if n not in ('original','control') and means[n]>means['original']]
result={'pilot_mean_FPS':means,'full_skipped':not selected,'selected_students':selected,
    'quality_FAIL_remains_FAIL':True,'pilot_complete_sha256':sha(pilot/'complete.json')}
target=root/'selection.json';assert not target.exists();target.write_text(json.dumps(result,indent=2))
if selected:
    config['models']={n:m for n,m in config['models'].items() if n in ('original','control') or n in selected}
    config['source_sha256'][str(target)]=sha(target)
    (root/'full_models.json').write_text(json.dumps(config,indent=2))
print('R167 FULL SELECTION',json.dumps(result),flush=True)
