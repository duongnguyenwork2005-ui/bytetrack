"""Select epochs on the frozen 300-image monitor with deployed argmax geometry."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
spec=importlib.util.spec_from_file_location('r164_frozen_argmax',HERE.parent/'review151_resolution_tradeoff/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
torch.set_num_threads(2);cv2.setNumThreads(1)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def check():
    frozen=read(HERE/'training_inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    return frozen
def label(image):return Path(image.as_posix().replace('/images/','/labels/')).with_suffix('.txt')
def evaluate(checkpoint,images,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(checkpoint),task='detect').model.to('cuda:0').float().eval()
    assert model.names==base.NAMES
    validator=DetectionValidator(args={'conf':.001,'iou':.7,'max_det':300,'plots':False},save_dir=output)
    metrics=DetMetrics(names=base.NAMES);gt=np.zeros(5,dtype=int);tp25=gt.copy();n25=gt.copy()
    for index,image in enumerate(images):
        frame=cv2.imread(str(image));assert frame is not None,image
        h,w=frame.shape[:2];x=base.det.preprocess(base.det.letterbox(frame,(384,640))[0])
        with torch.inference_mode():raw=model(torch.from_numpy(x).to('cuda:0'))[0].cpu().numpy()
        assert raw.shape==(1,9,5040) and np.isfinite(raw).all()
        predictions=base.det.decode_nms(raw,conf_thres=.001,iou_thres=.7,max_det=300)
        base.det.scale_boxes(predictions,(384,640),frame.shape)
        path=label(image);a=np.loadtxt(path,ndmin=2,dtype=np.float32) if path.stat().st_size else np.empty((0,5),np.float32)
        z=a[:,1:]*np.array([w,h,w,h],dtype=np.float32);boxes=base.det.xywh2xyxy(z);target=a[:,0].astype(int)
        pred={'bboxes':torch.from_numpy(predictions[:,:4]),'cls':torch.from_numpy(predictions[:,5])}
        batch={'bboxes':torch.from_numpy(boxes),'cls':torch.from_numpy(a[:,0])}
        correct=validator._process_batch(pred,batch)['tp']
        metrics.update_stats({'tp':correct,'conf':predictions[:,4],'pred_cls':predictions[:,5],
            'target_cls':target,'target_img':np.unique(target),'im_name':image.name})
        classes,correct25=base.operating_stats(validator,predictions,batch)
        n25+=np.bincount(classes,minlength=5);tp25+=np.bincount(classes,weights=correct25,minlength=5).astype(int)
        gt+=np.bincount(target,minlength=5)
        if (index+1)%500==0:print('R164 QUALITY',checkpoint.name,index+1,'/',len(images),flush=True)
    metrics.process(save_dir=output,plot=False);per={}
    for ix,cls in enumerate(metrics.box.ap_class_index):
        cls=int(cls);per[base.NAMES[cls]]={'AP50':float(metrics.box.ap50[ix]),'AP50_95':float(metrics.box.ap[ix]),
            'GT_boxes':int(gt[cls]),'detections_c25':int(n25[cls]),'TP_c25_IoU50':int(tp25[cls]),
            'precision_c25':float(tp25[cls]/max(n25[cls],1)),
            'recall_c25':float(tp25[cls]/gt[cls]) if gt[cls] else None}
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'images':len(images),
        'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'classes':per,
        'geometry':[384,640],'runtime':'PC PyTorch CUDA FP32; no Nano timing',
        'postprocessing':'frozen Nano single argmax/NMS conf .001 iou .7 max300',
        'Nano_FPS_measured':False,'source_sha256':sha(Path(__file__))}
    write(output/'summary.json',result);del model;torch.cuda.empty_cache();return result
def select(mode):
    frozen=check();assert mode in frozen['models'];training=read(HERE/(mode+'_training.json'));assert training['status']=='PASS'
    out=HERE/'monitor'/mode;assert not out.exists();out.mkdir(parents=True)
    images=[Path(p) for p in (HERE/'data/dev300.txt').read_text().splitlines()];assert len(images)==300
    records=[]
    for epoch in range(10):
        checkpoint=HERE/'runs'/mode/'weights'/('epoch%d.pt'%epoch)
        assert sha(checkpoint)==training['checkpoint_sha256'][str(checkpoint)]
        result=evaluate(checkpoint,images,out/('epoch%d'%epoch));result['epoch_zero_based']=epoch
        records.append(result);write(out/'partial.json',records)
        print('R164 MONITOR',mode,epoch,result['mAP50'],result['mAP50_95'],flush=True)
    best=max(records,key=lambda r:(r['mAP50_95'],r['mAP50'],-r['epoch_zero_based']))
    check();write(out/'selection.json',{'status':'PASS','rule':'max AP50-95, tie AP50, tie earlier epoch; monitor300 only',
        'selected':best,'epochs':records,'training_sha256':sha(HERE/(mode+'_training.json')),
        'source_sha256':sha(Path(__file__)),'opened_dev7190':False})
    print('R164 EPOCH SELECTED',mode,best['epoch_zero_based'],best['mAP50_95'],flush=True)
def dev(mode):
    check();images=[base.DATA/r['anh_ra'] for r in base.rows()];assert len(images)==7190
    frozen=base.read(base.HERE/'inputs.json')['sha256']
    for p,h in frozen.items():assert sha(p)==h,p
    if mode=='anchor':checkpoint=base.ANCHOR
    else:
        selection=read(HERE/'monitor'/mode/'selection.json');assert selection['status']=='PASS'
        checkpoint=Path(selection['selected']['checkpoint']);assert sha(checkpoint)==selection['selected']['checkpoint_sha256']
    result=evaluate(checkpoint,images,HERE/'quality'/mode)
    check()
    for p,h in frozen.items():assert sha(p)==h,p
    write(HERE/'quality'/mode/'verification.json',{'status':'PASS','source_and_dev_hashes':len(frozen),
        'summary_sha256':sha(HERE/'quality'/mode/'summary.json')})
    print('R164 DEV COMPLETE',mode,result['mAP50'],result['mAP50_95'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('select','dev'));parser.add_argument('mode')
    args=parser.parse_args();{'select':select,'dev':dev}[args.action](args.mode)
