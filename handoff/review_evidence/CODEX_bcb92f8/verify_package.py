from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'review_239b9c6/tree'
PKG = HERE / 'package'
TREE = HERE / 'tree'
manifest = json.loads((PKG / 'MANIFEST.json').read_text())
entries = manifest['file_trong_goi'] + manifest['file_bo_qua']
digest = lambda b: hashlib.sha256(b).hexdigest()
errors = []
recovered = {}
patch = next((PKG / 'patch').glob('*.diff'))
for e in entries:
    p = BASE / e['path']
    if e['sha256_base'] and (not p.exists() or digest(p.read_bytes()) != e['sha256_base']):
        dest = HERE / 'recover_preimage' / e['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and digest(p.read_bytes().replace(b'\r\n', b'\n')) == e['sha256_base']:
            dest.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
            method = 'retained base normalized CRLF to LF; exact declared base SHA256'
        else:
            shutil.copyfile(PKG / e['trong_goi'], dest)
            rev = subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf',
                                  'apply', '-R', '--include=' + e['path'], str(patch)],
                                 cwd=HERE / 'recover_preimage', capture_output=True, text=True)
            assert rev.returncode == 0, rev.stderr
            method = 'target payload plus reverse patch; exact declared base SHA256'
        assert digest(dest.read_bytes()) == e['sha256_base'], e['path']
        recovered[e['path']] = method
    payload = e.get('trong_goi') or e.get('trung_noi_dung_voi')
    if payload:
        data = (PKG / payload).read_bytes()
        if digest(data) != e['sha256_target'] or len(data) != e['bytes']:
            errors.append(['payload', e['path']])
assert not errors, errors
assert not TREE.exists(), 'Refuse to overwrite existing reconstruction'
shutil.copytree(BASE, TREE, ignore=shutil.ignore_patterns('__pycache__'))
for rel in recovered:
    target = TREE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / 'recover_preimage' / rel, target)
patch = next((PKG / 'patch').glob('*.diff'))
paths = re.findall(rb'^diff --git a/(.*?) b/(.*?)$', patch.read_bytes(), re.M)
for a,b in paths:
    assert a == b
    name = a.decode()
    assert name not in {'.gitignore', 'live.bat'}
    assert not name.startswith(('src/demo/', 'outputs/', '.git/', 'data/raw/'))
    assert Path(name).suffix.lower() not in {'.npz','.mp4','.avi','.mkv','.engine','.pt','.zip'}
check = subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply','--check',str(patch)],cwd=TREE,capture_output=True,text=True)
assert check.returncode == 0, check.stderr
applied = subprocess.run(['git','-c','core.autocrlf=false','-c','core.eol=lf','apply',str(patch)],cwd=TREE,capture_output=True,text=True)
assert applied.returncode == 0, applied.stderr
for e in entries:
    p = TREE / e['path']
    if e['sha256_target']:
        assert p.exists() and digest(p.read_bytes()) == e['sha256_target'], e['path']
    else:
        assert not p.exists(), e['path']
result = dict(base=manifest['base_commit'], target=manifest['target_commit'],
              payloads_verified=len(manifest['file_trong_goi']),
              deduplicated=len(manifest['file_bo_qua']),
              base_files_verified=sum(bool(e['sha256_base']) for e in entries),
              patch_paths=len(paths), target_paths_verified=len(entries),
              patch_check_exit=check.returncode, patch_apply_exit=applied.returncode)
result['base_preimage_recovery'] = recovered
(HERE/'package_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
