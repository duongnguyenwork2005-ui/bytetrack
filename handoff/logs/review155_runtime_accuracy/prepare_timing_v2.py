"""Fingerprint imported driver paths, retaining the original failed runner."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

old=(HERE/'timing_stage/run_models.py').read_text()
fragment="SOURCE/'bench_nano.py',\n          *[(SOURCE/'drivers'/name) for name in ('r86_driver.py','r101_driver.py','r102_driver.py')],"
replacement="Path(benchmark.__file__),\n          *[Path(module.__file__) for module in (guard,tools,live)],"
assert old.count(fragment)==1
target=HERE/'run_models_v2.py';assert not target.exists();target.write_text(old.replace(fragment,replacement))
wrapper=(HERE/'timing.py').read_text();assert wrapper.count("HERE/'run_models.py'")==1
target=HERE/'timing_v2.py';assert not target.exists();target.write_text(wrapper.replace("HERE/'run_models.py'","HERE/'run_models_v2.py'"))
config=json.loads((HERE/'timing_stage/timing_models.json').read_text())
for name in ('timing.py','run_models.py'):del config['source_sha256'][REMOTE+'/'+name]
for name in ('timing_v2.py','run_models_v2.py'):config['source_sha256'][REMOTE+'/'+name]=sha(HERE/name)
write(HERE/'timing_models_v2.json',config)
paths=[Path(__file__),HERE/'02_timing_path_fix.md',HERE/'run_models_v2.py',HERE/'timing_v2.py',HERE/'timing_models_v2.json']
write(HERE/'timing_v2_preflight.json',{'status':'PASS','reason':'Fingerprint actual imported driver files',
      'failed_pilot_inference_frames':0,'sha256':{str(p):sha(p) for p in paths}})
print('TIMING PATH V2 PREFLIGHT PASS')
