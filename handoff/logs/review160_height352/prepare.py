"""Export a single geometry-derived shape, with independent ORT/PT fixtures."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile

os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
torch.set_num_threads(1);cv2.setNumThreads(1)

R145=HERE.parent/'review145_yolo_recovery';R155=HERE.parent/'review155_runtime_accuracy'
SOURCE=HERE.parent/'review150_nano_pruning/source_snapshot/nano_detector.py'
REMOTE='/home/jetson/nano_run/r160_height352_20261120'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    assert not (HERE/'inputs.json').exists()
    anchor=ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'
    assert sha(anchor)=='d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82'
    assert read(R155/'cache/original/quality_verification.json')['status']=='PASS'
    spec=importlib.util.spec_from_file_location('r160_preprocessing',SOURCE)
    det=importlib.util.module_from_spec(spec);spec.loader.exec_module(det)
    output=HERE/'stage';output.mkdir(exist_ok=False);(output/'parity').mkdir()
    shutil.copyfile(anchor,output/'checkpoint.pt')
    model=YOLO(str(output/'checkpoint.pt'));reference=deepcopy(model.model).cpu().float().eval()
    path=Path(model.export(format='onnx',imgsz=(352,640),batch=1,opset=12,dynamic=False,
                           simplify=False,half=False,nms=False,device='cpu'))
    onnx.checker.check_model(onnx.load(path));shutil.move(path,output/'model.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(output/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    checks=[];images=read(R145/'export_verification.json')['models']['anchor']['CPU_ORT_parity']
    for ix,row in enumerate(images):
        image=Path(row['image']);frame=cv2.imread(str(image));assert frame is not None
        x=det.preprocess(det.letterbox(frame,(352,640))[0])
        with torch.inference_mode():expected=reference(torch.from_numpy(x))[0].cpu().numpy()
        actual=session.run(None,{session.get_inputs()[0].name:x})[0]
        assert actual.shape==expected.shape==(1,9,4620) and np.isfinite(actual).all()
        assert np.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02)
        assert np.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5)
        np.savez_compressed(output/'parity'/('%02d.npz'%ix),input=x,expected=actual)
        checks.append({'image':str(image),'sha256':sha(image),'box_error':float(abs(actual[:,:4]-expected[:,:4]).max()),
                       'score_error':float(abs(actual[:,4:]-expected[:,4:]).max())})
    prior=read(R155/'models.json');original=prior['models']['original']
    config={'models':{'original':original,'r352x640':{'onnx':REMOTE+'/model.onnx',
              'onnx_sha256':sha(output/'model.onnx'),'engine':REMOTE+'/model.engine',
              'hw':[352,640],'parity':REMOTE+'/parity'}},'package':prior['package']}
    write(output/'build_models.json',config)
    with tarfile.open(HERE/'stage.tar','w:') as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix!='.pt':archive.add(p,arcname=str(p.relative_to(output)).replace('\\','/'))
    paths=[Path(__file__),HERE/'00_protocol.md',HERE/'build_capture.sh',anchor,SOURCE,R155/'dataset.json',
           R155/'inputs.json',R155/'capture.py',R155/'score.py',R155/'cache/original/quality.json',
           R155/'cache/original/quality_verification.json',HERE/'stage.tar']
    write(HERE/'inputs.json',{'sha256':{str(p):sha(p) for p in paths},'CPU_ORT_PyTorch_checks':checks,
          'status':'PASS','input':[1,3,352,640],'output':[1,9,4620],
          'weights_unchanged':True,'Nano_pipeline_FPS_measured':False})
    print('HEIGHT352 EXPORT PARITY PASS',flush=True)

if __name__=='__main__':main()
