"""Physically prune internal dependencies; independently mask the reference."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
(HERE/'settings').mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings');os.environ['OMP_NUM_THREADS']='2'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck,SPPF,Conv

torch.set_num_threads(2)
ANCHOR=ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'
EXPECTED='d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82'
OUT=ROOT/'models/r164_latency_pruning/v2';WORK=ROOT/'tmp/r164_latency_pruning/v2'
NAMES={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def tensor(value):return value[0] if isinstance(value,(tuple,list)) else value
def mask_affine(module,removed):
    with torch.no_grad():module.bn.weight[removed]=0.;module.bn.bias[removed]=0.
def output_slice(module,keep):
    c=module.conv;assert c.groups==1 and c.bias is None
    c.out_channels=len(keep);c.weight=nn.Parameter(c.weight.detach()[keep].contiguous())
    bn=module.bn;bn.num_features=len(keep)
    for key in ('weight','bias'):setattr(bn,key,nn.Parameter(getattr(bn,key).detach()[keep].contiguous()))
    for key in ('running_mean','running_var'):setattr(bn,key,getattr(bn,key).detach()[keep].contiguous())
def input_slice(module,keep):
    c=module.conv if hasattr(module,'conv') else module
    assert c.groups==1;c.in_channels=len(keep);c.weight=nn.Parameter(c.weight.detach()[:,keep].contiguous())
def indices(module):
    c=module.conv;old=c.out_channels;count=max(8,round(old*.5/8)*8)
    assert count<=old,(old,count)
    importance=c.weight.detach().flatten(1).norm(dim=1)*module.bn.weight.detach().abs()
    keep=torch.argsort(importance,descending=True,stable=True)[:count].sort().values
    removed=torch.tensor([i for i in range(old) if i not in set(keep.tolist())],dtype=torch.long)
    return old,keep,removed
def pair(student,reference,first,second,changes):
    a=student.get_submodule(first);b=student.get_submodule(second)
    old,keep,removed=indices(a)
    if not len(removed):
        changes.append({'kind':'minimum_width_noop','first':first,'width':old});return
    bc=b.conv if hasattr(b,'conv') else b
    assert bc.in_channels%old==0
    repeat=bc.in_channels//old;columns=torch.cat([keep+i*old for i in range(repeat)])
    mask_affine(reference.get_submodule(first),removed);output_slice(a,keep);input_slice(b,columns)
    changes.append({'kind':'channel_pair','first':first,'second':second,'old':old,'keep':keep.tolist(),'repeat':repeat})
def body(student,reference,changes):
    targets=[(name,type(module)) for name,module in student.named_modules() if isinstance(module,(Bottleneck,SPPF))]
    for name,_ in targets:pair(student,reference,name+'.cv1',name+'.cv2',changes)
    for j in range(len(student.model[10].m)):
        prefix='model.10.m.%d.ffn.'%j;pair(student,reference,prefix+'0',prefix+'1',changes)
def head(student,reference,changes):
    for scale in range(3):
        prefix='model.23.cv2.%d.'%scale
        pair(student,reference,prefix+'0',prefix+'1',changes)
        pair(student,reference,prefix+'1',prefix+'2',changes)
        prefix='model.23.cv3.%d.'%scale
        first=student.get_submodule(prefix+'0.1');dw=student.get_submodule(prefix+'1.0')
        old,keep,removed=indices(first)
        assert dw.conv.groups==dw.conv.in_channels==dw.conv.out_channels==old
        mask_affine(reference.get_submodule(prefix+'0.1'),removed)
        mask_affine(reference.get_submodule(prefix+'1.0'),removed)
        output_slice(first,keep)
        c=dw.conv;c.in_channels=c.out_channels=c.groups=len(keep)
        c.weight=nn.Parameter(c.weight.detach()[keep].contiguous())
        bn=dw.bn;bn.num_features=len(keep)
        for key in ('weight','bias'):setattr(bn,key,nn.Parameter(getattr(bn,key).detach()[keep].contiguous()))
        for key in ('running_mean','running_var'):setattr(bn,key,getattr(bn,key).detach()[keep].contiguous())
        input_slice(student.get_submodule(prefix+'1.1'),keep)
        changes.append({'kind':'classification_depthwise_dependency','scale':scale,'old':old,'keep':keep.tolist()})
        pair(student,reference,prefix+'1.1',prefix+'2',changes)
def depths(student,reference,targets,changes):
    for index in targets:
        module=student.model[index];ref=reference.model[index]
        count=len(module.m);assert count>=1 and module.cv2.conv.in_channels==(2+count)*module.c
        kept=module.cv2.conv.in_channels-module.c
        with torch.no_grad():ref.cv2.conv.weight[:,kept:]=0.
        module.m=nn.ModuleList(list(module.m)[:-1]);input_slice(module.cv2,torch.arange(kept))
        changes.append({'kind':'depth','block':index,'removed_last_module':count-1,'new_cv2_in':kept})
class FunctionalHardSwish(nn.Module):
    def forward(self,x):return x*F.relu6(x+3.)/6.
def activations(student,reference,changes):
    for name,module in student.named_modules():
        if isinstance(module,Conv) and isinstance(module.act,nn.SiLU):
            module.act=nn.Hardswish();reference.get_submodule(name).act=FunctionalHardSwish()
            changes.append({'kind':'activation','module':name+'.act','from':'SiLU','to':'Hardswish'})
def convolution_macs(model,x):
    total=[0];hooks=[]
    def hook(module,args,value):
        total[0]+=int(value.numel()*module.kernel_size[0]*module.kernel_size[1]*module.in_channels/module.groups)
    for module in model.modules():
        if isinstance(module,nn.Conv2d):hooks.append(module.register_forward_hook(hook))
    with torch.inference_mode():model(x)
    for h in hooks:h.remove()
    return total[0]
def profile():
    rows=[r for r in read(HERE/'profile.json') if 'name' in r];groups={};total=sum(r['averageMs'] for r in rows);mapped=0.
    for row in rows:
        ids=set(map(int,re.findall(r'/model\.(\d+)',row['name'])))
        if ids:mapped+=row['averageMs']
        for i in ids:groups[i]=groups.get(i,0.)+row['averageMs']/len(ids)
    assert mapped/total>=.8,'Profile attribution fallback must be implemented/reviewed before proceeding'
    targets=sorted((2,4,6,8,13,16,19,22),key=lambda i:(-groups.get(i,0.),i))[:2]
    return {'total_ms':total,'coverage':mapped/total,'block_ms':groups,'depth_targets':targets}
def main():
    assert not (HERE/'inputs_v2.json').exists();assert sha(ANCHOR)==EXPECTED
    OUT.mkdir(exist_ok=False);WORK.mkdir(exist_ok=False)
    binding=[Path(__file__),HERE/'02_min_width_fix.md',HERE/'00_protocol.md',HERE/'01_profile_scope.md',HERE/'profile.json',HERE/'profile.log',ANCHOR]
    parent=read(HERE.parent/'review145_yolo_recovery/inputs.json')
    for path,digest in parent['sha256'].items():assert sha(Path(path))==digest,path
    binding.append(HERE.parent/'review145_yolo_recovery/inputs.json')
    cost=profile();torch.manual_seed(164);teacher=YOLO(str(ANCHOR)).model.float().eval()
    assert teacher.names==NAMES
    train=read(HERE.parent/'review145_yolo_recovery/inputs.json')['models']
    import cv2
    sysmodule=__import__('sys');sysmodule.path.insert(0,str(HERE.parent/'review150_nano_pruning/source_snapshot'))
    from nano_detector import preprocess,letterbox
    pictures=(HERE.parent/'review145_yolo_recovery/data/dev300.txt').read_text().splitlines()
    fixtures=[torch.zeros(1,3,384,640),torch.rand(1,3,384,640)]
    for i in (0,49,149,299):
        path=Path(pictures[i]);frame=cv2.imread(str(path));assert frame is not None
        fixtures.append(torch.from_numpy(preprocess(letterbox(frame,(384,640))[0])))
    baseline={'parameters':sum(p.numel() for p in teacher.parameters()),'ConvMAC':convolution_macs(teacher,fixtures[0])}
    frozen={'sha256':{str(p):sha(p) for p in binding},'anchor':baseline,'profile':cost,'models':{}}
    write(HERE/'inputs_v2.json',frozen)
    for mode in ('head50','body50','joint50','depth2','hswish'):
        student=deepcopy(teacher);reference=deepcopy(teacher);changes=[]
        if mode in ('body50','joint50'):body(student,reference,changes)
        if mode in ('head50','joint50'):head(student,reference,changes)
        if mode=='depth2':depths(student,reference,cost['depth_targets'],changes)
        if mode=='hswish':activations(student,reference,changes)
        assert student.names==NAMES and len(student.model[-1].cv2)==len(student.model[-1].cv3)==3
        assert all(branch[-1].out_channels==5 for branch in student.model[-1].cv3)
        assert all(branch[-1].out_channels==64 for branch in student.model[-1].cv2)
        checks=[]
        with torch.inference_mode():
            for index,x in enumerate(fixtures):
                actual=tensor(student(x));expected=tensor(reference(x))
                assert actual.shape==expected.shape==(1,9,5040) and torch.isfinite(actual).all()
                box=float((actual[:,:4]-expected[:,:4]).abs().max());score=float((actual[:,4:]-expected[:,4:]).abs().max())
                checks.append({'fixture':index,'box_error':box,'score_error':score})
                assert torch.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02),(mode,checks[-1])
                assert torch.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5),(mode,checks[-1])
        path=OUT/(mode+'_init.pt')
        torch.save({'model':student,'train_args':{'task':'detect','imgsz':640,'rect':True}},path)
        result={'path':str(path),'sha256':sha(path),'parameters':sum(p.numel() for p in student.parameters()),
                'ConvMAC':convolution_macs(student,fixtures[0]),'changes':changes,'masked_reference_checks':checks,'classes':NAMES}
        if mode!='hswish':assert result['parameters']<baseline['parameters']
        frozen['models'][mode]=result;write(HERE/'inputs_v2.json',frozen)
        print('R164 STRUCTURAL ORACLE PASS',mode,result['parameters'],result['ConvMAC'],flush=True)
    frozen['status']='PASS';write(HERE/'inputs_v2.json',frozen)
    print('R164 CPU PROPOSALS COMPLETE',cost,flush=True)
if __name__=='__main__':main()
