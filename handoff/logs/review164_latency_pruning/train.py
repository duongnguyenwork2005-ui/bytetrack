"""Matched rectangular KD recovery, bounded by the prospective GPU screen."""
from pathlib import Path
import argparse,hashlib,json,os,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
(HERE/'settings').mkdir(exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path[:0]=[str(HERE.parent/'review104_yolo_compression')]
import torch
import yaml
from ultralytics import YOLO
import ultralytics.data.dataset as dataset_module
from fit_channels import PrunedTrainer,train_with_distillation

torch.set_num_threads(2)
ANCHOR=ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def check():
    contract=read(HERE/'training_inputs.json')
    for path,h in contract['sha256'].items():assert sha(path)==h,path
    return contract
class LocalTrainer(PrunedTrainer):
    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        shapes=getattr(self,'r164_epoch_shapes',{})
        key='x'.join(map(str,batch['img'].shape));shapes[key]=shapes.get(key,0)+1
        self.r164_epoch_shapes=shapes
        return batch
    def build_dataset(self,img_path,mode='train',batch=None):
        data=super().build_dataset(img_path,mode,batch)
        # Avoid consuming stale .npy files beside the original dataset, even
        # when the upstream loader probes them with cache=False.
        data.npy_files=[HERE/'disabled_image_cache'/(hashlib.sha256(p.encode()).hexdigest()+'.npy') for p in data.im_files]
        assert not any(p.exists() for p in data.npy_files)
        return data
def prepare():
    target=HERE/'training_inputs.json';assert not target.exists()
    selection=read(HERE/'screen_evidence/selection.json')
    assert selection['status']=='PASS' and selection['GPU_only']
    assert read(HERE/'screen_verification.json')['status']=='PASS'
    assert read(HERE/'training_preflight_v2.json')['status']=='PASS'
    selected=selection['selected_for_training'];assert len(selected)<=2
    if not selected:
        print('R164 NEGATIVE GPU SCREEN; NO TRAINING',flush=True);return
    init=read(HERE/'inputs_v2.json');models=init['models'];parent=read(HERE.parent/'review145_yolo_recovery/inputs.json')
    for path,h in init['sha256'].items():assert sha(path)==h,path
    for path,h in parent['sha256'].items():assert sha(path)==h,path
    paths=[Path(__file__),HERE/'monitor.py',HERE/'verify_screen.py',HERE/'recover.ps1',HERE/'export_recovered.py',HERE/'00_protocol.md',HERE/'01_profile_scope.md',HERE/'02_min_width_fix.md',
        HERE/'screen_verification.json',HERE/'screen_evidence/selection.json',HERE/'inputs_v2.json',
        HERE/'training_preflight_v2.py',HERE/'training_preflight_v2.json',
        HERE.parent/'review151_resolution_tradeoff/run.py',
        HERE.parent/'review150_nano_pruning/source_snapshot/nano_detector.py',
        HERE.parent/'review104_yolo_compression/fit_channels.py',HERE.parent/'review104_yolo_compression/prepare.py']
    contracts={'control':{'path':str(ANCHOR),'parameters':init['anchor']['parameters'],'ConvMAC':init['anchor']['ConvMAC']}}
    for name in selected:
        assert name in models and sha(models[name]['path'])==models[name]['sha256']
        contracts[name]={'path':models[name]['path'],'parameters':models[name]['parameters'],'ConvMAC':models[name]['ConvMAC']}
    for record in contracts.values():record['sha256']=sha(record['path']);paths.append(Path(record['path']))
    data=HERE/'data';data.mkdir(exist_ok=False)
    for name in ('train.txt','dev300.txt'):
        source=HERE.parent/'review145_yolo_recovery/data'/name
        (data/name).write_bytes(source.read_bytes());paths.append(data/name)
    names={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}
    (data/'data.yaml').write_text(yaml.safe_dump({'path':str(data),'train':str(data/'train.txt'),'val':str(data/'dev300.txt'),'names':names}),encoding='utf-8')
    paths.append(data/'data.yaml');hashes=dict(parent['sha256']);hashes.update({str(p):sha(p) for p in paths})
    assert len((data/'train.txt').read_text().splitlines())==5200
    assert len((data/'dev300.txt').read_text().splitlines())==300
    write(target,{'status':'PASS','models':contracts,'sha256':hashes,'epochs':10,'seed':164,'rect':True,
        'monitor':'exact argmax/NMS384x640 AP50-95; tie AP50 then earlier epoch',
        'cache_policy':'all label cache loads/saves redirected under R164; original image .npy probes disabled'})
    print('R164 TRAINING CONTRACT PASS',list(contracts),flush=True)
def train(mode):
    frozen=check();record=frozen['models'][mode];root=HERE/'runs'/mode
    assert not root.exists();model=YOLO(record['path'],task='detect')
    assert sum(p.numel() for p in model.model.parameters())==record['parameters']
    cache=HERE/'label_caches';cache.mkdir(exist_ok=True)
    original_load=dataset_module.load_dataset_cache_file;original_save=dataset_module.save_dataset_cache_file
    def cache_path(path):return cache/(hashlib.sha256(str(path).encode()).hexdigest()+'.cache')
    def load(path):return original_load(cache_path(path))
    def save(prefix,path,value,version):return original_save(prefix,cache_path(path),value,version)
    shapes={}
    def epoch_start(trainer):
        trainer.r164_epoch_shapes={}
    def epoch_end(trainer):
        shapes[str(trainer.epoch)]=dict(trainer.r164_epoch_shapes);write(root/'batch_shapes.json',shapes)
    model.add_callback('on_train_epoch_start',epoch_start);model.add_callback('on_train_epoch_end',epoch_end)
    dataset_module.load_dataset_cache_file=load;dataset_module.save_dataset_cache_file=save
    args=dict(trainer=LocalTrainer,data=str(HERE/'data/data.yaml'),epochs=10,imgsz=640,rect=True,
        batch=4,workers=0,optimizer='SGD',lr0=.002,warmup_epochs=.5,mosaic=0,mixup=0,amp=False,
        device=0,seed=164,deterministic=True,patience=0,plots=False,cache=False,save_period=1,
        project=str(HERE/'runs'),name=mode,exist_ok=False,verbose=False)
    try:train_with_distillation(model,args)
    finally:
        dataset_module.load_dataset_cache_file=original_load;dataset_module.save_dataset_cache_file=original_save
    checkpoints=[root/'weights'/('epoch%d.pt'%epoch) for epoch in range(10)]
    assert all(p.is_file() for p in checkpoints)
    assert set(shapes)=={str(epoch) for epoch in range(10)}
    check();write(HERE/(mode+'_training.json'),{'status':'PASS','model':mode,'epochs_completed':10,
        'parameters':record['parameters'],'checkpoint_sha256':{str(p):sha(p) for p in checkpoints},
        'history_sha256':sha(root/'results.csv'),'batch_shapes_sha256':sha(root/'batch_shapes.json'),
        'source_sha256':sha(Path(__file__)),'Nano_FPS_measured':False})
    print('R164 TRAINING COMPLETE',mode,flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','train'));parser.add_argument('mode',nargs='?')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:train(args.mode)
