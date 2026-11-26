"""Preserve package symlinks; do not materialize its external data/model trees."""
from pathlib import Path
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/jetson/nano_run/r108_15fps_20261117/overlap_package')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entries(root):
    output={}
    for p in root.rglob('*'):
        name=str(p.relative_to(root))
        if any(part in ('__pycache__','runs_v2','results') for part in p.relative_to(root).parts):continue
        if p.is_symlink():output[name]={'symlink_resolved_target':str(p.resolve())}
        elif p.is_file():output[name]={'sha256':sha(p)}
    return output
def main():
    check=json.loads((HERE/'pipeline_preflight.json').read_text())
    for name,h in check['source_sha256'].items():assert sha(HERE/name)==h,name
    assert sha(SOURCE/'src/nano_detect_stream.py')==check['baseline_stream_sha256']
    assert sha(SOURCE/'src/nano_detector.py')==check['baseline_detector_sha256']
    destination=HERE/'package';failed=HERE/'package_stage_v1_failed'
    assert destination.resolve().parent==HERE and not failed.exists()
    assert not (HERE/'package_verification.json').exists()
    destination.rename(failed)
    shutil.copytree(str(SOURCE),str(destination),symlinks=True,ignore=shutil.ignore_patterns('__pycache__','runs_v2','results'))
    # An absolute reference preserves the same referent even for a relative
    # source link whose parent changed. No target files are modified or copied.
    for p in SOURCE.rglob('*'):
        if p.is_symlink() and not any(part in ('__pycache__','runs_v2','results') for part in p.relative_to(SOURCE).parts):
            link=destination/p.relative_to(SOURCE)
            assert link.is_symlink()
            referent=p.resolve();link.unlink();link.symlink_to(referent,target_is_directory=p.is_dir())
    changed={'src/nano_detect_stream.py'};added={'src/'+n for n in ('gpu_pre.py','gpu_detector.py','fused_pre.so')}
    for name in ('nano_detect_stream.py','gpu_pre.py','gpu_detector.py','fused_pre.so'):
        shutil.copyfile(str(HERE/name),str(destination/'src'/name))
    original=entries(SOURCE);current=entries(destination)
    assert set(current)==set(original)|added,(set(current)-set(original)-added,set(original)-set(current))
    assert {n for n,h in original.items() if current[n]!=h}==changed
    record={'status':'PASS','original_entries':original,'candidate_entries':current,
        'changed':sorted(changed),'added':sorted(added),'symlink_referents_unchanged':True,
        'prior_failed_copy_preserved':str(failed),'source_sha256':
            {str(HERE/n):sha(HERE/n) for n in ('stage_pipeline_v2.py','pipeline_preflight.json')}}
    (HERE/'package_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('CUDA ISOLATED PACKAGE PASS',len(original),len(current),flush=True)
if __name__=='__main__':main()
