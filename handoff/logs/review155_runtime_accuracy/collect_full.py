"""Nano: collect only completed benchmark outputs and their bound sources."""
from pathlib import Path
import hashlib
import json
import tarfile
import time

ROOT=Path('/home/jetson/nano_run/r155_runtime_accuracy_20261120')
BASE=Path('/home/jetson/nano_run')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    full=ROOT/'runs_absolute/full'
    while not (full/'complete.json').exists() or not (full/'after.json').exists():
        time.sleep(10)
    inputs=read(full/'inputs.json');after=read(full/'after.json');done=read(full/'complete.json')
    assert done['status']=='PASS' and inputs['boot']==after['boot'] and after['errors']=='0'
    config=read(ROOT/'full_models.json');sources=dict(inputs['code_sha256']);sources.update(config['source_sha256'])
    for path,digest in sources.items():
        real=Path(path).resolve();real.relative_to(BASE);assert sha(real)==digest,path
    manifest={'status':'PASS','input_sha256':sha(full/'inputs.json'),'complete_sha256':sha(full/'complete.json'),
              'after_sha256':sha(full/'after.json'),'source_sha256':sources,'collector_sha256':sha(Path(__file__)),
              'boot':after['boot'],'outputs':done['runs'],'archive_contains_images':False}
    (ROOT/'collection.json').write_text(json.dumps(manifest,indent=2))
    archive=ROOT/'full80_evidence_20261120.tar.gz'
    names=['runs_absolute/full','runs_absolute/pilot','quality','selection.json','selection.log','full.log',
           'pilot_absolute.log','pilot.log','pilot_v2.log','full_models.json','timing_models_v2.json',
           'timing_v2.py','run_models_v2.py','select_full.py','04_full_tradeoff.md','collection.json']
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=1) as bundle:
            for name in names:bundle.add(ROOT/name,arcname=name)
            for path in sorted(sources):
                real=Path(path).resolve();relative=real.relative_to(BASE)
                bundle.add(real,arcname='executed_sources/'+str(relative),recursive=False)
    print('FULL80_EVIDENCE_READY',archive,archive.stat().st_size,sha(archive),flush=True)

if __name__=='__main__':main()
