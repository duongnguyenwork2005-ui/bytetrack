"""Fixed-shape exports and CPU detection validation with the deployed Nano path."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import sys
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'data/yolo_mixed5_v1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics

torch.set_num_threads(2);cv2.setNumThreads(1)
NAMES={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}
SHAPES={'baseline':(384,640),'r352x608':(352,608),'r320x576':(320,576)}
ANCHOR=ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'
SOURCE=HERE.parent/'review150_nano_pruning/source_snapshot/nano_detector.py'
spec=importlib.util.spec_from_file_location('r151_original_nano_detector',SOURCE)
det=importlib.util.module_from_spec(spec);spec.loader.exec_module(det)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def rows():
    with (DATA/'manifest/anh.csv').open(encoding='utf-8-sig',newline='') as f:
        selected=[r for r in csv.DictReader(f) if r['split']=='dev']
    assert len(selected)==7190 and len({r['anh_ra'] for r in selected})==7190
    return selected


def operating_stats(validator,predictions,batch):
    keep=predictions[:,4]>.25
    selected=predictions[keep]
    pred={'bboxes':torch.from_numpy(selected[:,:4]),'cls':torch.from_numpy(selected[:,5])}
    # Re-match after confidence filtering: a discarded low-confidence box must
    # not consume a GT match when reporting the operating point's P/R.
    correct=validator._process_batch(pred,batch)['tp'][:,0]
    return selected[:,5].astype(int),correct


def prepare():
    assert not (HERE/'inputs.json').exists()
    assert sha(ANCHOR)=='d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82'
    validator=DetectionValidator(args={'plots':False},save_dir=HERE/'sanity')
    fixture=np.array([[3,0,13,10,.9,0],[0,0,10,10,.1,0]],dtype=np.float32)
    batch={'bboxes':torch.tensor([[0.,0.,10.,10.]]),'cls':torch.tensor([0.])}
    classes,correct=operating_stats(validator,fixture,batch)
    assert classes.tolist()==[0] and correct.tolist()==[True]
    manifest=rows();hashes={str(p):sha(p) for p in (Path(__file__),HERE/'00_protocol.md',SOURCE,ANCHOR,DATA/'manifest/anh.csv')}
    counts=np.zeros(5,dtype=int)
    for row in manifest:
        image,label=DATA/row['anh_ra'],DATA/row['nhan_ra']
        assert sha(image)==row['sha256_ra'] and sha(label)==row['sha256_nhan']
        hashes[str(image)]=row['sha256_ra'];hashes[str(label)]=row['sha256_nhan']
        counts += np.array([int(row['n_'+name]) for name in NAMES.values()])
    write(HERE/'inputs.json',{'sha256':hashes,'images':7190,'classes':NAMES,'GT_boxes':counts.tolist(),
            'shapes':SHAPES,'Nano_FPS':None,'selection':'all dev manifest entries, no subset chosen by score'})
    print('RESOLUTION PREPARE PASS',counts,flush=True)


def check():
    frozen=read(HERE/'inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p


def export():
    check();monitor=(HERE.parent/'review145_yolo_recovery/data/dev300.txt').read_text().splitlines()
    chosen=[Path(monitor[i]) for i in (0,33,66,99,132,165,198,231,264,299)]
    records={};folder=HERE/'exports';folder.mkdir(exist_ok=False)
    for mode,hw in SHAPES.items():
        out=folder/mode;out.mkdir();checkpoint=out/'anchor.pt';checkpoint.write_bytes(ANCHOR.read_bytes())
        model=YOLO(str(checkpoint),task='detect');reference=YOLO(str(checkpoint),task='detect').model.cpu().float().eval()
        path=Path(model.export(format='onnx',imgsz=hw,batch=1,opset=12,dynamic=False,simplify=False,half=False,nms=False,device='cpu'))
        graph=onnx.load(path);onnx.checker.check_model(graph)
        dims=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim];assert dims==[1,3,*hw]
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        evidence=out/'parity';evidence.mkdir();checks=[]
        for ix,image in enumerate(chosen):
            frame=cv2.imread(str(image));assert frame is not None
            x=det.preprocess(det.letterbox(frame,hw)[0])
            with torch.inference_mode():expected=reference(torch.from_numpy(x))[0].numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x})[0]
            assert actual.shape==expected.shape and actual.shape[1]==9
            assert np.isfinite(actual).all() and np.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02)
            assert np.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5)
            np.savez_compressed(evidence/f'{ix:02d}.npz',input=x,expected=actual)
            checks.append({'image':str(image),'sha256':sha(image),'max_box_error':float(abs(actual[:,:4]-expected[:,:4]).max()),
                           'max_score_error':float(abs(actual[:,4:]-expected[:,4:]).max())})
        records[mode]={'path':str(path),'sha256':sha(path),'input':dims,'output':list(actual.shape),'ORT_parity':checks,
                        'parity_files':{str(p):sha(p) for p in evidence.glob('*.npz')},'Nano_FPS':None}
        write(HERE/'export_partial.json',records);print('RESOLUTION EXPORT PASS',mode,flush=True)
    check();write(HERE/'export_verification.json',{'status':'PASS','models':records})


def evaluate(mode):
    assert mode in SHAPES;check();hw=SHAPES[mode];output=HERE/'quality'/mode;output.mkdir(parents=True,exist_ok=False)
    model=YOLO(str(ANCHOR),task='detect').model.cpu().float().eval()
    validator=DetectionValidator(args={'conf':.001,'iou':.7,'max_det':300,'plots':False},save_dir=output)
    metrics=DetMetrics(names=NAMES);tp25=np.zeros(5,dtype=int);n25=np.zeros(5,dtype=int);gt=np.zeros(5,dtype=int)
    all_rows=rows()
    for index,row in enumerate(all_rows):
        image=DATA/row['anh_ra'];frame=cv2.imread(str(image));assert frame is not None
        h,w=frame.shape[:2];assert (w,h)==(int(row['w']),int(row['h']))
        x=det.preprocess(det.letterbox(frame,hw)[0])
        with torch.inference_mode():raw=model(torch.from_numpy(x))[0].numpy()
        predictions=det.decode_nms(raw,conf_thres=.001,iou_thres=.7,max_det=300)
        det.scale_boxes(predictions,hw,frame.shape)
        label=DATA/row['nhan_ra'];a=np.loadtxt(label,ndmin=2,dtype=np.float32) if label.stat().st_size else np.empty((0,5),np.float32)
        z=a[:,1:]*np.array([w,h,w,h],dtype=np.float32)
        boxes=det.xywh2xyxy(z);target=a[:,0].astype(int)
        pred={'bboxes':torch.from_numpy(predictions[:,:4]),'cls':torch.from_numpy(predictions[:,5])}
        batch={'bboxes':torch.from_numpy(boxes),'cls':torch.from_numpy(a[:,0])}
        correct=validator._process_batch(pred,batch)['tp']
        metrics.update_stats({'tp':correct,'conf':predictions[:,4],'pred_cls':predictions[:,5],
               'target_cls':target,'target_img':np.unique(target),'im_name':image.name})
        classes,correct25=operating_stats(validator,predictions,batch)
        n25+=np.bincount(classes,minlength=5);tp25+=np.bincount(classes,weights=correct25,minlength=5).astype(int)
        gt+=np.bincount(target,minlength=5)
        if (index+1)%250==0:print('DETECTION CPU',mode,index+1,'/7190',flush=True)
    metrics.process(save_dir=output,plot=False)
    per={}
    for ix,cls in enumerate(metrics.box.ap_class_index):
        cls=int(cls);per[NAMES[cls]]={'AP50':float(metrics.box.ap50[ix]),'AP50_95':float(metrics.box.ap[ix]),
          'GT_boxes':int(gt[cls]),'detections_c25':int(n25[cls]),'TP_c25_IoU50':int(tp25[cls]),
          'precision_c25':float(tp25[cls]/max(n25[cls],1)),'recall_c25':float(tp25[cls]/gt[cls]) if gt[cls] else None}
    check();write(output/'summary.json',{'images':len(all_rows),'input':list(hw),'checkpoint_sha256':sha(ANCHOR),
          'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'classes':per,
          'definition':'Deployed Nano argmax/NMS/letterbox + existing Ultralytics matching/AP; FP32 CPU dev',
          'Nano_FPS':None,'precision_recall_at_fixed_confidence':.25})
    print('RESOLUTION QUALITY COMPLETE',mode,flush=True)


def summary():
    check();scores={m:read(HERE/'quality'/m/'summary.json') for m in SHAPES};gates={}
    baseline=scores['baseline']
    for m in ('r352x608','r320x576'):
        current=scores[m]
        gates[m]={'mAP50':current['mAP50']>=baseline['mAP50']-.005,
                  'mAP50_95':current['mAP50_95']>=baseline['mAP50_95']-.005,
                  'per_class':all(current['classes'][n]['AP50']>=baseline['classes'][n]['AP50']-.02 for n in NAMES.values())}
    write(HERE/'quality.json',{'scores':scores,'gates':gates,'Nano_FPS_measured':False})
    print('RESOLUTION QUALITY SUMMARY',gates,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='evaluate':evaluate(sys.argv[2])
    else:{'prepare':prepare,'export':export,'summary':summary}[action]()
