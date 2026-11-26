"""Nano: copy the 2.8MB experimental package and change only its child adapter."""
from pathlib import Path
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/jetson/nano_run/r108_15fps_20261117/overlap_package')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    check=json.loads((HERE/'pipeline_preflight.json').read_text())
    for name,h in check['source_sha256'].items():assert sha(HERE/name)==h,name
    assert sha(SOURCE/'src/nano_detect_stream.py')==check['baseline_stream_sha256']
    assert sha(SOURCE/'src/nano_detector.py')==check['baseline_detector_sha256']
    destination=HERE/'package';assert not destination.exists()
    shutil.copytree(str(SOURCE),str(destination),ignore=shutil.ignore_patterns('__pycache__','runs_v2','results'))
    changed={'src/nano_detect_stream.py'}
    added={'src/'+n for n in ('gpu_pre.py','gpu_detector.py','fused_pre.so')}
    for name in ('nano_detect_stream.py','gpu_pre.py','gpu_detector.py','fused_pre.so'):
        shutil.copyfile(str(HERE/name),str(destination/'src'/name))
    original={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file()
        and not any(part in ('__pycache__','runs_v2','results') for part in p.relative_to(SOURCE).parts)}
    current={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}
    assert set(current)==set(original)|added
    differences={n for n,h in original.items() if current[n]!=h}
    assert differences==changed,differences
    (HERE/'package_verification.json').write_text(json.dumps({'status':'PASS','original_sha256':original,
        'candidate_sha256':current,'changed':sorted(changed),'added':sorted(added),
        'source_sha256':{str(HERE/n):sha(HERE/n) for n in ('stage_pipeline.py','pipeline_preflight.json')}},indent=2)+'\n')
    print('CUDA ISOLATED PACKAGE PASS',len(original),len(current),flush=True)
if __name__=='__main__':main()
