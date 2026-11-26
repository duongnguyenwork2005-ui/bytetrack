"""Audit all 7190 candidate arrays, then freeze GPU-preprocessing timing inputs."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'review155_runtime_accuracy'
REMOTE='/home/jetson/nano_run/r162_cuda_pre_20261120'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')

def main():
    cache=HERE/'capture_evidence/cache/gpu_pre';original=BASE/'cache/original'
    bound=[Path(__file__),HERE/'00_protocol.md',HERE/'oracle.json',HERE/'package_verification.json']
    for folder in (original,cache):
        verification=read(folder/'quality_verification.json');assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(Path(p))==h,p
        bound.extend(folder/n for n in ('quality.json','quality_verification.json','inputs.json','complete.json'))
    a=read(original/'quality.json');b=read(cache/'quality.json')
    assert a['dataset_sha256']==b['dataset_sha256'] and a['engine']==b['engine']
    check=[]
    for index in range(72):
        left=original/('%04d.npz'%index);right=cache/left.name
        with np.load(left,allow_pickle=False) as x,np.load(right,allow_pickle=False) as y:
            assert set(x.files)==set(y.files)
            equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
            check.append({'chunk':index,'equal':equal,'all_equal':all(equal.values())})
        bound.extend((left,right))
    equal_ap={key:a[key]==b[key] for key in ('mAP50','mAP50_95','classes')}
    write(HERE/'candidate_parity.json',{'status':'PASS','images':7190,'chunks':check,
        'all_candidate_arrays_bit_exact':all(r['all_equal'] for r in check),'AP_equal':equal_ap,
        'note':'Status verifies the comparison; individual equality fields state its outcome.'})
    output=HERE/'speed_stage';output.mkdir(exist_ok=False);(output/'quality').mkdir()
    model=read(BASE/'models.json')['models']['original'];models={}
    oracle=read(HERE/'oracle.json');assert oracle['pixel_status']==oracle['raw_status']=='PASS'
    gates={k:b[k]>=a[k]-.005 for k in ('mAP50','mAP50_95')}
    gates.update({c:r['AP50']>=a['classes'][c]['AP50']-.02 for c,r in b['classes'].items()})
    for name,quality,folder,package in (
        ('original',a,original,'/home/jetson/nano_run/r108_15fps_20261117/overlap_package'),
        ('gpu_pre',b,cache,REMOTE+'/package')):
        target=output/'quality'/(name+'.json');shutil.copyfile(folder/'quality.json',target)
        models[name]=dict(model,runtime_quality=REMOTE+'/quality/'+name+'.json',runtime_quality_sha256=sha(target),
            quality_recommended=True if name=='original' else all(gates.values()),
            historical_raw_parity='PASS',runtime_package=package)
    for name in ('timing_v3.py','run_models_v3.py','select_full.py','micro.py','00_protocol.md','package_verification.json','oracle.json','candidate_parity.json'):
        shutil.copyfile(HERE/name,output/name)
    sources={REMOTE+'/'+p.name:sha(p) for p in output.iterdir() if p.is_file()}
    sources.update({m['runtime_quality']:m['runtime_quality_sha256'] for m in models.values()})
    config={'models':models,'package':'per-model runtime_package','baseline_quality':REMOTE+'/quality/original.json',
        'dataset_sha256':a['dataset_sha256'],'allow_diagnostic_quality_failure':True,'batch':'CUDA_preprocessing',
        'source_sha256':sources}
    write(output/'timing_models_v3.json',config)
    write(HERE/'speed_preflight.json',{'status':'PASS','gates_vs_original':gates,
        'candidate_arrays_bit_exact':all(r['all_equal'] for r in check),'AP_equal':equal_ap,
        'source_sha256':{str(p):sha(p) for p in bound},'portable_sha256':
            {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}})
    print('CUDA SPEED PREFLIGHT PASS','array equality',all(r['all_equal'] for r in check),'AP',a['mAP50'],b['mAP50'],flush=True)

if __name__=='__main__':main()
