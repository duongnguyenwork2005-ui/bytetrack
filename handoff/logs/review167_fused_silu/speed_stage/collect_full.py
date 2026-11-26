"""Collect a completed extra full80 trial, preserving exact executed sources."""
from pathlib import Path
import argparse,hashlib,json,tarfile

BASE=Path('/home/jetson/nano_run')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root.resolve();root.relative_to(BASE);full=root/'runs_absolute/full'
    inputs=read(full/'inputs.json');after=read(full/'after.json');done=read(full/'complete.json')
    assert done['status']=='PASS' and inputs['boot']==after['boot'] and after['errors']=='0'
    config=read(root/'full_models.json');sources=dict(inputs['code_sha256']);sources.update(config['source_sha256'])
    for path,h in sources.items():
        real=Path(path).resolve();real.relative_to(BASE);assert sha(real)==h,path
    record={'status':'PASS','input_sha256':sha(full/'inputs.json'),'complete_sha256':sha(full/'complete.json'),
        'after_sha256':sha(full/'after.json'),'source_sha256':sources,'collector_sha256':sha(Path(__file__)),
        'boot':after['boot'],'outputs':done['runs'],'archive_contains_images':False}
    manifest=root/'collection.json';assert not manifest.exists();manifest.write_text(json.dumps(record,indent=2))
    names=['timing_models_v5.json','timing_models_v5_bound.json','timing_v5.py','run_models_v5.py','select_full_bound.py','environment.json','environment_after.json','record_environment.py','01_cuda_namespace_fix.md','02_fast_variant_scope.md','runs_absolute/full','runs_absolute/pilot','runs_absolute/engine','quality','selection.json',
        'selection.log','full.log','pilot.log','engine.log','full_models.json','timing_models_v2.json',
        'timing_models_v3.json','timing_v2.py','run_models_v2.py','timing_v3.py','run_models_v3.py',
        'select_full.py','scope.md','00_protocol.md','collection.json','micro.py','micro.json','micro.log',
        'oracle.py','oracle.json','oracle.log','candidate_parity.json','gpu_pre.py','gpu_detector.py',
        'fused_pre.cu','fused_pre.so','package_verification.json','extra_chain_commands.json']
    archive=root/'full80_evidence_20261121.tar.gz'
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',compresslevel=1) as bundle:
            for name in names:
                path=root/name
                if path.exists():bundle.add(str(path),arcname=name)
            bundle.add(str(Path(__file__)),arcname='collect_extra.py',recursive=False)
            driver=Path(__file__).parent/'run_extra_chain.py'
            if driver.exists():bundle.add(str(driver),arcname='run_extra_chain.py',recursive=False)
            for path in sorted(sources):
                real=Path(path).resolve();bundle.add(str(real),arcname='executed_sources/'+str(real.relative_to(BASE)),recursive=False)
    print('EXTRA_FULL80_EVIDENCE_READY',str(archive),archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
