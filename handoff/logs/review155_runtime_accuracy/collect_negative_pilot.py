"""Archive a completed negative pilot without claiming a full80 result."""
from pathlib import Path
import argparse,hashlib,json,tarfile

BASE=Path('/home/jetson/nano_run')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();root.relative_to(BASE);logs=root/'runs_absolute/pilot'
    inputs=read(logs/'inputs.json');after=read(logs/'after.json');done=read(logs/'complete.json')
    selection=read(root/'selection.json');config=read(Path(inputs['config_path']))
    assert done['status']=='PASS' and selection['full_skipped']
    assert inputs['boot']==after['boot'] and after['errors']=='0'
    assert done['models_sha256']==sha(Path(inputs['config_path']))
    assert selection['pilot_complete_sha256']==sha(logs/'complete.json')
    assert len(inputs['videos'])==4 and done['runs']==4*len(inputs['models'])
    sources=dict(inputs['code_sha256']);sources.update(config['source_sha256'])
    for path,h in sources.items():
        p=Path(path).resolve();p.relative_to(BASE);assert sha(p)==h,path
    record={'status':'PASS','phase':'negative_pilot','clips':4,'outputs':done['runs'],
        'boot':inputs['boot'],'input_sha256':sha(logs/'inputs.json'),'after_sha256':sha(logs/'after.json'),
        'complete_sha256':sha(logs/'complete.json'),'selection_sha256':sha(root/'selection.json'),
        'source_sha256':sources,'collector_sha256':sha(Path(__file__)),'contains_source_images':False}
    manifest=root/'pilot_collection.json';assert not manifest.exists();manifest.write_text(json.dumps(record,indent=2)+'\n')
    names=['runs_absolute/pilot','runs_absolute/engine','quality','selection.json','pilot.log','selection.log',
        'engine.log','timing_models_v2.json','timing_models_v3.json','timing_v2.py','timing_v3.py',
        'run_models_v2.py','run_models_v3.py','select_full.py','scope.md','00_protocol.md','pilot_collection.json']
    archive=root/'negative_pilot_evidence_20261120.tar.gz'
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=1) as bundle:
            for name in names:
                if (root/name).exists():bundle.add(str(root/name),arcname=name)
            bundle.add(str(Path(__file__)),arcname='collect_negative_pilot.py',recursive=False)
            for path in sorted(sources):
                p=Path(path).resolve();bundle.add(str(p),arcname='executed_sources/'+str(p.relative_to(BASE)),recursive=False)
    print('NEGATIVE_PILOT_EVIDENCE_READY',str(archive),archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
