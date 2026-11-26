"""All GPU-qualified variants, plus a matched rebuild; no model selected by AP."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r167_fused_silu_20261121'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    verification=read(HERE/'screen_verification.json');assert verification['status']=='PASS'
    for p,h in verification['sha256'].items():assert sha(Path(p))==h,p
    runtime=read(HERE/'screen_evidence/runtime_models.json');selection=read(HERE/'screen_evidence/selection.json')
    eligible=selection['eligible_for_actual_AP'];assert eligible and len(eligible)<=2
    runtime['models']={n:r for n,r in runtime['models'].items() if n=='control' or n in eligible}
    runtime['source_sha256'][REMOTE+'/capture.py']=sha(HERE/'capture.py')
    parent=HERE.parent/'review162_cuda_pre'
    for name in ('capture.py','gpu_pre.py','fused_pre.cu','oracle.json','00_protocol.md'):
        runtime['source_sha256']['/home/jetson/nano_run/r162_cuda_pre_20261120/'+name]=sha(parent/name)
    runtime['source_sha256']['/home/jetson/nano_run/r162_cuda_pre_20261120/fused_pre.so']=sha(parent/'capture_evidence/fused_pre.so')
    target=HERE/'actual_models.json';assert not target.exists();target.write_text(json.dumps(runtime,indent=2)+'\n')
    (HERE/'actual_preflight.json').write_text(json.dumps({'status':'PASS','models':list(runtime['models']),
        'raw_FAIL_remains_FAIL':True,'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'capture.py',target,HERE/'screen_verification.json')}},indent=2)+'\n')
    print('R167 ACTUAL AP PREFLIGHT PASS',list(runtime['models']),flush=True)
if __name__=='__main__':main()
