"""Score actual Nano TensorRT candidates using the unchanged detector/AP path."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
torch.set_num_threads(2)
SOURCE=HERE.parent/'review150_nano_pruning/source_snapshot/nano_detector.py'
spec=importlib.util.spec_from_file_location('r155_deployed_detector',SOURCE)
det=importlib.util.module_from_spec(spec);spec.loader.exec_module(det)
NAMES={0:'car',1:'truck',2:'bus',3:'motorcycle',4:'bicycle'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def candidates(raw):
    p=raw[0].T;score=p[:,4:].max(axis=1);classes=p[:,4:].argmax(axis=1);keep=score>.001
    return np.column_stack((det.xywh2xyxy(p[keep,:4]),score[keep],classes[keep])).astype(np.float32)


def postprocess(data,hw,original_shape):
    x=np.array(data,dtype=np.float32,copy=True)
    assert x.ndim==2 and x.shape[1]==6 and np.isfinite(x).all()
    if len(x)>det.MAX_NMS:x=x[np.argsort(-x[:,4],kind='stable')[:det.MAX_NMS]]
    if len(x):
        offset=x[:,5:6]*det.MAX_WH
        selected=det.nms_matrix(x[:,:4]+offset,x[:,4],.7)[:300];x=x[selected]
    return det.scale_boxes(x,hw,original_shape)


def operating_stats(validator,predictions,gt):
    selected=predictions[predictions[:,4]>.25]
    pred={'bboxes':torch.from_numpy(selected[:,:4]),'cls':torch.from_numpy(selected[:,5])}
    correct=validator._process_batch(pred,gt)['tp'][:,0]
    return selected[:,5].astype(int),correct


def fixture():
    rng=np.random.default_rng(155)
    for n in (0,7,100,2000):
        raw=rng.random((1,9,n),dtype=np.float32)
        raw[:,:2]*=500;raw[:,2:4]=raw[:,2:4]*200+1
        actual=postprocess(candidates(raw),(352,608),(720,1280))
        expected=det.decode_nms(raw,.001,.7,300)
        det.scale_boxes(expected,(352,608),(720,1280))
        assert np.array_equal(actual,expected),(n,actual.shape,expected.shape)
    validator=DetectionValidator(args={'plots':False},save_dir=HERE/'sanity')
    boxes=np.array([[3,0,13,10,.9,0],[0,0,10,10,.1,0]],dtype=np.float32)
    gt={'bboxes':torch.tensor([[0.,0.,10.,10.]]),'cls':torch.tensor([0.])}
    labels,tp=operating_stats(validator,boxes,gt);assert labels.tolist()==[0] and tp.tolist()==[True]
    write(HERE/'fixture.json',{'status':'PASS','bridge_cases':[0,7,100,2000],'fixed_confidence_rematch':True,
          'sha256':{str(p):sha(p) for p in (Path(__file__),SOURCE,Path(inspect.getsourcefile(DetectionValidator)),
                                           Path(inspect.getsourcefile(DetMetrics)))}})
    print('NANO CANDIDATE BRIDGE FIXTURE PASS',flush=True)


def score(folder):
    assert read(HERE/'fixture.json')['status']=='PASS'
    for p,h in read(HERE/'fixture.json')['sha256'].items():assert sha(p)==h,p
    dataset=read(HERE/'dataset.json');inputs=read(folder/'inputs.json');complete=read(folder/'complete.json')
    assert complete['status']=='PASS' and complete['images']==7190
    assert complete['engine_sha256']==inputs['engine']['engine_sha256']
    assert inputs['dataset_sha256']==sha(HERE/'dataset.json') and inputs['detector_source_sha256']==sha(SOURCE)
    for name,h in complete['sha256'].items():assert sha(folder/name)==h,name
    validator=DetectionValidator(args={'plots':False},save_dir=folder/'metrics')
    metrics=DetMetrics(names=NAMES);seen=[];gtcounts=np.zeros(5,dtype=int);n25=np.zeros(5,dtype=int);tp25=np.zeros(5,dtype=int)
    for chunk in range(72):
        path=folder/('%04d.npz'%chunk)
        with np.load(path,allow_pickle=False) as archive:
            data=archive['data'];offsets=archive['offsets'];indices=archive['indices'];shapes=archive['orig_shapes'];hw=archive['hw']
            assert len(offsets)==len(indices)+1 and offsets[0]==0 and offsets[-1]==len(data)
            assert list(hw)==inputs['engine']['hw'] and np.isfinite(data).all()
            assert indices.tolist()==list(range(chunk*100,min(7190,(chunk+1)*100)))
            for position,index in enumerate(indices):
                row=dataset['images'][int(index)];assert row['index']==int(index) and row['shape']==list(shapes[position])
                pred=postprocess(data[offsets[position]:offsets[position+1]],tuple(hw),tuple(row['shape']))
                a=np.asarray(row['GT_xywh_normalized'],dtype=np.float32).reshape(-1,5)
                h,w=row['shape'];boxes=det.xywh2xyxy(a[:,1:]*np.array([w,h,w,h],dtype=np.float32))
                gt={'bboxes':torch.from_numpy(boxes),'cls':torch.from_numpy(a[:,0])}
                pv={'bboxes':torch.from_numpy(pred[:,:4]),'cls':torch.from_numpy(pred[:,5])}
                tp=validator._process_batch(pv,gt)['tp'];target=a[:,0].astype(int)
                metrics.update_stats({'tp':tp,'conf':pred[:,4],'pred_cls':pred[:,5],'target_cls':target,
                         'target_img':np.unique(target),'im_name':row['relative']})
                labels,correct25=operating_stats(validator,pred,gt)
                n25+=np.bincount(labels,minlength=5);tp25+=np.bincount(labels,weights=correct25,minlength=5).astype(int)
                gtcounts+=np.bincount(target,minlength=5);seen.append(int(index))
        if (chunk+1)%10==0:print('SCORE NANO DETECTION',folder.name,len(seen),flush=True)
    assert seen==list(range(7190));metrics.process(save_dir=folder/'metrics',plot=False)
    per={}
    for ix,cls in enumerate(metrics.box.ap_class_index):
        cls=int(cls);per[NAMES[cls]]={'AP50':float(metrics.box.ap50[ix]),'AP50_95':float(metrics.box.ap[ix]),
            'GT_boxes':int(gtcounts[cls]),'detections_c25':int(n25[cls]),'TP_c25_IoU50':int(tp25[cls]),
            'precision_c25':float(tp25[cls]/max(n25[cls],1)),'recall_c25':float(tp25[cls]/gtcounts[cls])}
    result={'images':7190,'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'classes':per,
        'engine':inputs['engine'],'dataset_sha256':sha(HERE/'dataset.json'),'runtime':'actual Nano TensorRT',
        'AP_definition':'R109 argmax/greedy NMS c001/iou.7/maxdet300 + existing Ultralytics AP/matching',
        'precision_recall_at_fixed_confidence':.25,'Nano_pipeline_FPS_measured':False}
    write(folder/'quality.json',result)
    write(folder/'quality_verification.json',{'status':'PASS','images':len(seen),'unique_indices':len(set(seen)),
       'sha256':{str(p):sha(p) for p in (folder/'quality.json',folder/'inputs.json',folder/'complete.json',HERE/'dataset.json',
                                        HERE/'fixture.json',Path(__file__))}})
    print('ACTUAL NANO AP COMPLETE',folder.name,result['mAP50'],result['mAP50_95'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('fixture','score'));parser.add_argument('folder',nargs='?',type=Path)
    args=parser.parse_args()
    if args.action=='fixture':fixture()
    else:score(args.folder)
