"""Fresh synchronized CUDA workload; never writes a trained checkpoint."""
from pathlib import Path
import hashlib,json,os,sys
HERE=Path(__file__).resolve().parent;LOGS=HERE.parent
os.environ['CUDA_LAUNCH_BLOCKING']='1';os.environ['TORCH_SHOW_CPP_STACKTRACES']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
sys.path.insert(0,str(LOGS/'review104_yolo_compression'))
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from fit_channels import train_with_distillation
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=HERE/'diagnostic.json';assert not target.exists();rows={};torch.manual_seed(169)
    for mode in ('control','joint50'):
        path=LOGS/'review164_latency_pruning/runs'/mode/'weights/epoch6.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);assert checkpoint['epoch']==6 and checkpoint['optimizer']
        assert all(torch.isfinite(p).all() for p in checkpoint['ema'].parameters())
        model=YOLO(str(path),task='detect').model.to('cuda:0').float().train();model.args=get_cfg()
        for name,p in model.named_parameters():p.requires_grad_('.dfl.' not in name)
        class Probe:
            def train(self,**kwargs):
                for tick in range(32):
                    model.zero_grad(set_to_none=True)
                    batch={'img':torch.rand(4,3,384,640,device='cuda:0'),'batch_idx':torch.arange(4.,device='cuda:0'),
                        'cls':((torch.arange(4,device='cuda:0')+tick)%5).float()[:,None],
                        'bboxes':torch.tensor([[.5,.5,.2,.3]]*4,device='cuda:0')}
                    loss,_=model.loss(batch);assert torch.isfinite(loss).all();loss.sum().backward();torch.cuda.synchronize()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                self.loss=loss.detach().cpu().tolist()
        probe=Probe();train_with_distillation(probe,{})
        rows[mode]={'steps':32,'checkpoint_sha256':sha(path),'final_loss':probe.loss,'epochs_saved':0}
        print('R169 CUDA DIAGNOSTIC PASS',mode,flush=True);del model,probe,checkpoint;torch.cuda.empty_cache()
    target.write_text(json.dumps({'status':'PASS','CUDA_LAUNCH_BLOCKING':'1','models':rows,
        'root_cause_identified':False,'sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'00_protocol.md',HERE/'windows_events.json')}},indent=2)+'\n')
if __name__=='__main__':main()
