"""Freeze both EMA/optimizer restarts before opening any new monitor scores."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'review164_latency_pruning'
import torch,yaml
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def main():
    target=HERE/'training_inputs.json';assert not target.exists();diagnostic=read(HERE/'diagnostic.json');assert diagnostic['status']=='PASS'
    for p,h in diagnostic['sha256'].items():assert sha(p)==h,p
    parent=read(PARENT/'training_inputs.json');hashes=parent['sha256'].copy()
    for p,h in hashes.items():assert sha(p)==h,p
    data=HERE/'data';data.mkdir(exist_ok=False)
    for name in ('train.txt','dev300.txt'):shutil.copyfile(PARENT/'data'/name,data/name)
    names={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}
    (data/'data.yaml').write_text(yaml.safe_dump({'path':str(data),'train':str(data/'train.txt'),'val':str(data/'dev300.txt'),'names':names}))
    inputs=HERE/'resume_inputs';inputs.mkdir();resume={};accepted={};changed={}
    binding=[Path(__file__),HERE/'00_protocol.md',HERE/'diagnostic.py',HERE/'diagnostic.json',HERE/'windows_events.json',
        HERE/'resume.py',HERE/'monitor.py',HERE/'export_recovered.py',HERE/'run.py',HERE/'recover.ps1',PARENT/'training_inputs.json',
        *data.iterdir()]
    for mode in ('control','joint50'):
        source=PARENT/'runs'/mode;root=HERE/'runs'/mode;weights=root/'weights';weights.mkdir(parents=True)
        shapes=read(source/'batch_shapes.json');shapes={str(i):shapes[str(i)] for i in range(7)}
        (root/'batch_shapes.json').write_text(json.dumps(shapes,indent=2)+'\n')
        accepted[mode]={}
        for i in range(7):
            old=source/'weights'/('epoch%d.pt'%i);dest=weights/old.name;shutil.copyfile(old,dest)
            accepted[mode][str(i)]=sha(old);assert sha(dest)==sha(old);binding.extend((old,dest))
        origin=source/'weights/epoch6.pt';ckpt=torch.load(origin,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==6 and ckpt['optimizer'] is not None and all(torch.isfinite(p).all() for p in ckpt['ema'].parameters())
        before=dict(ckpt['train_args']);after=dict(before)
        after.update(data=str(data/'data.yaml'),project=str(HERE/'runs'),name=mode,save_dir=str(root))
        ckpt['train_args']=after;path=inputs/(mode+'.pt');torch.save(ckpt,path)
        copy=torch.load(path,map_location='cpu',weights_only=False)
        assert equal(ckpt['ema'].state_dict(),copy['ema'].state_dict()) and equal(ckpt['optimizer'],copy['optimizer'])
        assert copy['epoch']==6 and copy['updates']==ckpt['updates'] and copy['scaler']==ckpt['scaler']
        assert {k for k in before if before[k]!=after[k]}<={'data','project','name','save_dir'}
        changed[mode]={k:{'before':before[k],'after':after[k]} for k in before if before[k]!=after[k]}
        resume[mode]={'path':str(path),'sha256':sha(path),'parent_epoch6_sha256':sha(origin)}
        binding.extend((path,source/'results.csv'))
    hashes.update({str(p):sha(p) for p in binding})
    target.write_text(json.dumps({'status':'PASS','models':parent['models'],'resume':resume,'accepted_parent_sha256':accepted,
        'resume_IO_changes_only':changed,'sha256':hashes,'epochs':10,'seed':164,'rect':True,
        'monitor':'exact argmax/NMS384x640 AP50-95; tie AP50 then earlier epoch','CUDA_LAUNCH_BLOCKING':'1',
        'paired_restart_from_ema_not_bit_exact':True,'parent_control_tail7_9_excluded':True},indent=2)+'\n')
    print('R169 MATCHED RESUME INPUTS VERIFIED',flush=True)
if __name__=='__main__':main()
