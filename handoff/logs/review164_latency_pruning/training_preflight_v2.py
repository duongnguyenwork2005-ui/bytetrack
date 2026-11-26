"""A finite KD backward and checkpoint round-trip before any recovery epoch."""
from pathlib import Path
import hashlib,io,json,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings');os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0,str(HERE.parent/'review104_yolo_compression'))
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from fit_channels import train_with_distillation
torch.set_num_threads(2)
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    target=HERE/'training_preflight_v2.json';assert not target.exists()
    frozen=json.loads((HERE/'inputs_v2.json').read_text());records={}
    models={'control':{'path':str(ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'),
        'parameters':frozen['anchor']['parameters']},**frozen['models']}
    for name,record in models.items():
        model=YOLO(record['path'],task='detect').model.to('cuda:0').float().train()
        # The real Trainer re-enables checkpoint parameters before optimizing.
        for name_p,p in model.named_parameters():p.requires_grad_('.dfl.' not in name_p)
        model.args=get_cfg();model.zero_grad(set_to_none=True)
        assert sum(p.numel() for p in model.parameters())==record['parameters']
        batch={'img':torch.zeros(1,3,384,640,device='cuda:0'),
            'batch_idx':torch.tensor([0.],device='cuda:0'),'cls':torch.tensor([[0.]],device='cuda:0'),
            'bboxes':torch.tensor([[.5,.5,.2,.3]],device='cuda:0')}
        class Probe:
            def train(self,**kwargs):
                loss,_=model.loss(batch);assert torch.isfinite(loss).all();loss.sum().backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                self.loss=loss.detach().cpu().tolist();self.gradients=len(grads)
        probe=Probe();train_with_distillation(probe,{})
        stream=io.BytesIO();torch.save(model,stream);stream.seek(0)
        recovered=torch.load(stream,map_location='cpu',weights_only=False).eval()
        assert sum(p.numel() for p in recovered.parameters())==record['parameters']
        with torch.inference_mode():out=recovered(torch.zeros(1,3,384,640))[0]
        assert out.shape==(1,9,5040) and torch.isfinite(out).all()
        records[name]={'KD_loss':probe.loss,'finite_gradient_tensors':probe.gradients,
            'checkpoint_bytes':len(stream.getbuffer()),'parameters':record['parameters'],'input_sha256':sha(record['path'])}
        print('R164 TRAINING PREFLIGHT',name,'PASS',probe.loss,flush=True)
        del model,recovered,probe,stream;torch.cuda.empty_cache()
    target.write_text(json.dumps({'status':'PASS','models':records,'training_epochs':0,
        'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE.parent/'review104_yolo_compression/fit_channels.py')}},indent=2)+'\n')
if __name__=='__main__':main()
