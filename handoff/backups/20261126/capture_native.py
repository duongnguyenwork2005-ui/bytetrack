"""Copy only manifest-bound deployed engines/plugins from Nano; no inference."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
LOGS=HERE.parents[1]/'logs'
REMOTE='/home/jetson/nano_run/r175_native_backup_20261126'
OUT=Path('C:/Users/Mikuno/AppData/Local/Temp/bytetrack-backup-20261126/native')
OUT.mkdir(exist_ok=True)
bindings={}
for folder in ('review167_fused_silu','review172_common7_pruning'):
    summary=json.loads((LOGS/folder/'final_summary.json').read_text())
    assert summary['status']=='PASS'
    for quality in summary['detection'].values():
        engine=quality['engine']
        for key in ('engine','onnx'):
            path=engine[key];digest=engine[key+'_sha256']
            assert path.startswith('/home/jetson/nano_run/') and '..' not in Path(path).parts
            assert bindings.setdefault(path,digest)==digest
package=json.loads((LOGS/'review167_fused_silu/package_verification.json').read_text())
for name,row in package['candidate_entries'].items():
    if name.endswith('.so'):
        bindings['/home/jetson/nano_run/r167_fused_silu_20261121/package/'+name]=row['sha256']
config=OUT/'bindings.json';config.write_text(json.dumps(bindings,indent=2)+'\n')
subprocess.run(['ssh','-o','HostName=10.11.50.200','-o','BatchMode=yes','-o','ConnectTimeout=10','nano',
                'test ! -e '+REMOTE+' && mkdir '+REMOTE],check=True)
subprocess.run(['scp','-o','HostName=10.11.50.200','-o','BatchMode=yes',str(config),'nano:'+REMOTE+'/bindings.json'],check=True)
program="""from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/jetson/nano_run/r175_native_backup_20261126')
bindings=json.loads((root/'bindings.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
for p,h in bindings.items():assert sha(p)==h,p
with tarfile.open(str(root/'nano-engines-20261126.tar.gz'),'w:gz') as tar:
 for p in sorted(bindings):tar.add(p,arcname=str(Path(p).relative_to('/home/jetson/nano_run')),recursive=False)
 tar.add(str(root/'bindings.json'),arcname='bindings.json',recursive=False)
for p,h in bindings.items():assert sha(p)==h,p
result={'status':'PASS','files':len(bindings),'archive_sha256':sha(root/'nano-engines-20261126.tar.gz')}
(root/'complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
"""
source=OUT/'capture.py';source.write_text(program)
subprocess.run(['scp','-o','HostName=10.11.50.200','-o','BatchMode=yes',str(source),'nano:'+REMOTE+'/capture.py'],check=True)
result=subprocess.run(['ssh','-o','HostName=10.11.50.200','-o','BatchMode=yes','-o','ConnectTimeout=10','nano',
                        'python3 '+REMOTE+'/capture.py'],check=True,stdout=subprocess.PIPE,text=True)
complete=json.loads(result.stdout);assert complete['status']=='PASS'
(OUT/'native_complete.json').write_text(json.dumps(complete,indent=2)+'\n')
target=OUT/'nano-engines-20261126.tar.gz'
subprocess.run(['scp','-o','HostName=10.11.50.200','-o','BatchMode=yes','nano:'+REMOTE+'/nano-engines-20261126.tar.gz',str(target)],check=True)
with target.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==complete['archive_sha256']
print('NATIVE_ARCHIVE_VERIFIED',complete['files'],target.stat().st_size,complete['archive_sha256'],flush=True)
