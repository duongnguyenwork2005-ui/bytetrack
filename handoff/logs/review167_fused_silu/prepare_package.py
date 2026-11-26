"""Native: clone code while preserving external data/model symlink referents."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'r162_cuda_pre_20261120/package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def entries(root):
    result={}
    for p in root.rglob('*'):
        relative=p.relative_to(root)
        if any(x in ('__pycache__','runs_v2','results') for x in relative.parts):continue
        if p.is_symlink():result[str(relative)]={'symlink_resolved_target':str(p.resolve())}
        elif p.is_file():result[str(relative)]={'sha256':sha(p)}
    return result
def main():
    destination=HERE/'package';assert not destination.exists() and destination.resolve().parent==HERE
    assert not (SOURCE/'src').is_symlink()
    before=entries(SOURCE)
    shutil.copytree(str(SOURCE),str(destination),symlinks=True,ignore=shutil.ignore_patterns('__pycache__','runs_v2','results'))
    for p in SOURCE.rglob('*'):
        if p.is_symlink() and not any(x in ('__pycache__','runs_v2','results') for x in p.relative_to(SOURCE).parts):
            link=destination/p.relative_to(SOURCE);assert link.is_symlink()
            target=p.resolve();link.unlink();link.symlink_to(target,target_is_directory=p.is_dir())
    detector=destination/'src/nano_detector.py';original=(SOURCE/'src/nano_detector.py').read_text()
    prefix="\n# R167: load only the two bound experimental plugins before engine deserialization.\nimport ctypes as _r167_ct\nfrom pathlib import Path as _r167_Path\n_r167_plugin_handles = [_r167_ct.CDLL(str(_r167_Path(__file__).parent / ('fused_silu_' + kind + '.so')), mode=_r167_ct.RTLD_GLOBAL) for kind in ('precise', 'fast')]\n\n"
    marker='class TRTEngine(object):';assert original.count(marker)==1
    if detector.is_symlink():detector.unlink()
    detector.write_text(original.replace(marker,prefix+marker))
    for kind in ('precise','fast'):
        target=destination/'src'/('fused_silu_'+kind+'.so');assert not target.exists()
        shutil.copyfile(str(HERE/target.name),str(target));assert sha(target)==sha(HERE/target.name)
    after=entries(destination);changed={'src/nano_detector.py'};added={'src/fused_silu_'+k+'.so' for k in ('precise','fast')}
    assert set(after)==set(before)|added
    assert {n for n in before if before[n]!=after[n]}==changed
    assert entries(SOURCE)==before,'Original package changed unexpectedly'
    (HERE/'package_verification.json').write_text(json.dumps({'status':'PASS','source':str(SOURCE),'package':str(destination),
        'original_entries':before,'candidate_entries':after,'changed':sorted(changed),'added':sorted(added),
        'symlink_referents_unchanged':True,'source_sha256':{str(Path(__file__)):sha(Path(__file__))}},indent=2)+'\n')
    print('R167 ISOLATED PACKAGE VERIFIED',len(before),len(after),flush=True)
if __name__=='__main__':main()
