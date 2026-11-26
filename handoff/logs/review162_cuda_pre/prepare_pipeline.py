"""Build a minimal isolated stream patch and a multi-package timing harness."""
from pathlib import Path
import difflib
import hashlib
import json

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'review155_runtime_accuracy'
REMOTE='/home/jetson/nano_run/r162_cuda_pre_20261120'
ORIGINAL='/home/jetson/nano_run/r108_15fps_20261117/overlap_package'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(source,old,new):
    assert source.count(old)==1,(source.count(old),old)
    return source.replace(old,new)
def create(name,source):
    p=HERE/name;assert not p.exists(),str(p);p.write_text(source)

old=(HERE/'baseline_snapshot/nano_detect_stream.py').read_text()
new=replace(old,'            eng = nd.TRTEngine(a.engine, bo_dem=a.bo_dem)',
    '            from gpu_pre import GpuTRTEngine\n            from gpu_detector import GpuNanoDetector\n            eng = GpuTRTEngine(a.engine, bo_dem=a.bo_dem)')
new=replace(new,'            det = nd.NanoDetector(eng, lb_hw, a.conf, a.iou, a.max_det, None, False)',
    '            det = GpuNanoDetector(eng, lb_hw, a.conf, a.iou, a.max_det, None, False)')
create('nano_detect_stream.py',new)
create('stream.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='R108/src/nano_detect_stream.py',tofile='R162/src/nano_detect_stream.py')))

runner=(BASE/'run_models_v2.py').read_text()
runner=replace(runner,"    package=Path(config['package']);assert package==HOME/'nano_run/r108_15fps_20261117/overlap_package'",
    "    packages={Path(m['runtime_package']) for m in config['models'].values()}\n    assert packages=={HOME/'nano_run/r108_15fps_20261117/overlap_package',HOME/'nano_run/r162_cuda_pre_20261120/package'}\n    package=Path(config['models']['original']['runtime_package'])")
runner=replace(runner,"          *sorted((package/'src').glob('*.py'))]}",
    "          *[p for folder in sorted(packages) for p in sorted((folder/'src').glob('*.py'))],\n          *sorted((HOME/'nano_run/r162_cuda_pre_20261120/package/src').glob('*.so'))]}")
runner=replace(runner,"                    guard.check_guard(boot,swap,errors);model=config['models'][mode]",
    "                    guard.check_guard(boot,swap,errors);model=config['models'][mode]\n                    package=Path(model['runtime_package'])")
create('run_models_v3.py',runner)
wrapper=replace((BASE/'timing_v2.py').read_text(),"HERE/'run_models_v2.py'","HERE/'run_models_v3.py'")
create('timing_v3.py',wrapper)
create('select_full.py','''"""Follow the prospective R162 pilot rule; preserve all quality failures."""
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
''')
bindings={n:sha(HERE/n) for n in ('nano_detect_stream.py','gpu_pre.py','gpu_detector.py','fused_pre.cu','run_models_v3.py','timing_v3.py','select_full.py','00_protocol.md')}
(HERE/'pipeline_preflight.json').write_text(json.dumps({'status':'PASS','original_package':ORIGINAL,'candidate_package':REMOTE+'/package',
    'baseline_stream_sha256':sha(HERE/'baseline_snapshot/nano_detect_stream.py'),
    'baseline_detector_sha256':sha(HERE/'baseline_snapshot/nano_detector.py'),
    'source_sha256':bindings,'timing_note':'infer_ms includes CUDA preprocessing and uint8 upload; pre_ms only measures remaining CPU preparation.'},indent=2)+'\n')
print('CUDA PIPELINE PREPARED',bindings)

if __name__!='__main__':raise RuntimeError('Run this preparation once')
