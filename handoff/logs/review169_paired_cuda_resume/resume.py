"""Symmetric three-epoch continuation, retaining the seven accepted parent epochs."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,sys
HERE=Path(__file__).resolve().parent;LOGS=HERE.parent;PARENT=LOGS/'review164_latency_pruning'
os.environ['CUDA_LAUNCH_BLOCKING']='1';os.environ['TORCH_SHOW_CPP_STACKTRACES']='1'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
spec=importlib.util.spec_from_file_location('r169_frozen_training_helpers',PARENT/'train.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.HERE=HERE
import torch
from ultralytics import YOLO
torch.set_num_threads(2)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def check():
    c=read(HERE/'training_inputs.json')
    for p,h in c['sha256'].items():assert sha(p)==h,p
    return c
class CrashAwareTrainer(base.LocalTrainer):
    def preprocess_batch(self,batch):
        self.r169_last_cpu_batch={key:value.detach() if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
        assert all(v.device.type=='cpu' for v in self.r169_last_cpu_batch.values() if isinstance(v,torch.Tensor))
        return super().preprocess_batch(batch)
def train(mode):
    frozen=check();record=frozen['models'][mode];root=HERE/'runs'/mode
    assert not (HERE/(mode+'_training.json')).exists()
    assert not (root/'weights/epoch7.pt').exists()
    model=YOLO(frozen['resume'][mode]['path'],task='detect')
    assert sum(p.numel() for p in model.model.parameters())==record['parameters']
    cache=HERE/'label_caches';cache.mkdir(exist_ok=True)
    original_load=base.dataset_module.load_dataset_cache_file;original_save=base.dataset_module.save_dataset_cache_file
    def cache_path(path):return cache/(hashlib.sha256(str(path).encode()).hexdigest()+'.cache')
    def load(path):return original_load(cache_path(path))
    def save(prefix,path,value,version):return original_save(prefix,cache_path(path),value,version)
    shapes=read(root/'batch_shapes.json')
    def start(trainer):
        assert trainer.start_epoch==7 and trainer.epochs==10 and trainer.args.seed==164
        assert trainer.args.rect and trainer.args.batch==4 and trainer.args.amp is False
        assert Path(trainer.save_dir).resolve()==root.resolve()
    def epoch_start(trainer):
        assert trainer.epoch in (7,8,9);trainer.r164_epoch_shapes={}
    def epoch_end(trainer):
        shapes[str(trainer.epoch)]=dict(trainer.r164_epoch_shapes);write(root/'batch_shapes.json',shapes)
    model.add_callback('on_train_start',start);model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_epoch_end',epoch_end)
    base.dataset_module.load_dataset_cache_file=load;base.dataset_module.save_dataset_cache_file=save
    try:
        base.train_with_distillation(model,dict(trainer=CrashAwareTrainer,resume=frozen['resume'][mode]['path'],
            save_dir=str(root),workers=0,device=0,cache=False,save_period=1))
    except BaseException:
        trainer=getattr(model,'trainer',None);batch=getattr(trainer,'r169_last_cpu_batch',None)
        if batch is not None:
            target=HERE/(mode+'_failed_cpu_batch.pt');assert not target.exists();torch.save(batch,target)
        raise
    finally:
        base.dataset_module.load_dataset_cache_file=original_load;base.dataset_module.save_dataset_cache_file=original_save
    paths=[root/'weights'/('epoch%d.pt'%i) for i in range(10)];assert all(p.exists() for p in paths)
    assert set(shapes)=={str(i) for i in range(10)}
    for i in range(7):assert sha(paths[i])==frozen['accepted_parent_sha256'][mode][str(i)]
    with (PARENT/'runs'/mode/'results.csv').open(newline='') as stream:old=list(csv.DictReader(stream))
    with (root/'results.csv').open(newline='') as stream:new=list(csv.DictReader(stream))
    old=[r for r in old if int(r['epoch'])<=7];new=[r for r in new if int(r['epoch'])>=8]
    assert len(old)==7 and len(new)==3 and [int(r['epoch']) for r in old+new]==list(range(1,11))
    history=root/'accepted_history.csv'
    with history.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(old[0]));writer.writeheader();writer.writerows(old+new)
    check();write(HERE/(mode+'_training.json'),{'status':'PASS','model':mode,'epochs_completed':10,
        'accepted_parent_epochs':7,'paired_resumed_epochs':3,'resume_is_not_bit_exact':True,
        'parameters':record['parameters'],'checkpoint_sha256':{str(p):sha(p) for p in paths},
        'history_sha256':sha(root/'results.csv'),'accepted_history_sha256':sha(history),
        'batch_shapes_sha256':sha(root/'batch_shapes.json'),'source_sha256':sha(Path(__file__)),'Nano_FPS_measured':False})
    print('R169 PAIRED RESUME COMPLETE',mode,flush=True)
if __name__=='__main__':train(sys.argv[1])
