"""Prepare static ONNX for Nano and independently compare ORT with PyTorch."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE/'settings'))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

torch.set_num_threads(2)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def tensor(path):
    frame=cv2.imread(str(path));assert frame is not None
    h,w=frame.shape[:2];r=min(384/h,640/w);nw,nh=round(w*r),round(h*r)
    resized=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR)
    dw,dh=(640-nw)/2,(384-nh)/2
    padded=cv2.copyMakeBorder(resized,round(dh-.1),round(dh+.1),round(dw-.1),round(dw+.1),
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    assert padded.shape==(384,640,3)
    return np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255


def main():
    assert read(HERE/'quality.json')['scores']
    frozen=read(HERE/'inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    checkpoints={'anchor':ROOT/'models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt'}
    for mode in ('control','channel25_KD'):
        record=read(HERE/f'{mode}_training.json');assert record['status']=='PASS'
        checkpoints[mode]=Path(record['best']);assert sha(checkpoints[mode])==record['best_sha256']
    manifest=(HERE/'data/dev300.txt').read_text().splitlines()
    chosen=[Path(manifest[i]) for i in (0,33,66,99,132,165,198,231,264,299)]
    paths=[Path(__file__),HERE/'00_protocol.md',HERE/'quality.json',HERE/'inputs.json',*checkpoints.values(),*chosen]
    locked={'sha256':{str(p):sha(p) for p in paths},'input':[1,3,384,640],'opset':12,'images':10}
    lock=HERE/'export_inputs.json'
    if lock.exists():assert read(lock)==locked
    else:write(lock,locked)
    output=HERE/'onnx';output.mkdir(exist_ok=True)
    records={}
    for mode,path in checkpoints.items():
        model=YOLO(str(path),task='detect')
        reference=deepcopy(model.model).cpu().float().eval()
        macs=None
        try:
            from thop import profile
            with torch.inference_mode():macs,_=profile(deepcopy(reference),inputs=(torch.zeros(1,3,384,640),),verbose=False)
        except ImportError:pass
        exported=Path(model.export(format='onnx',imgsz=(384,640),batch=1,opset=12,dynamic=False,
                                   simplify=False,half=False,nms=False,device='cpu'))
        destination=output/f'{mode}_384x640.onnx'
        assert not destination.exists()
        destination.write_bytes(exported.read_bytes())
        graph=onnx.load(destination);onnx.checker.check_model(graph)
        dims=[d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim];assert dims==[1,3,384,640]
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(destination),sess_options=options,providers=['CPUExecutionProvider'])
        errors=[]
        for image in chosen:
            x=tensor(image)
            with torch.inference_mode():expected=reference(torch.from_numpy(x))[0].cpu().numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x})[0]
            assert actual.shape==expected.shape==(1,9,5040) and np.isfinite(actual).all()
            assert np.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02)
            assert np.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5)
            errors.append({'image':str(image),'max_box_error':float(abs(actual[:,:4]-expected[:,:4]).max()),
                           'max_class_score_error':float(abs(actual[:,4:]-expected[:,4:]).max())})
        records[mode]={'path':str(destination),'sha256':sha(destination),'bytes':destination.stat().st_size,
              'parameters':sum(p.numel() for p in reference.parameters()),'estimated_thop_MACs':macs,
              'input':dims,'output':[1,9,5040],'CPU_ORT_parity':errors,'Nano_FPS':None}
        write(HERE/'export_partial.json',records)
        print('EXPORT PARITY PASS',mode,flush=True)
    for p,h in locked['sha256'].items():assert sha(p)==h,p
    write(HERE/'export_verification.json',{'status':'PASS','models':records,'Nano_FPS_measured':False,
             'MACs_note':'THOP estimates, not measured Nano throughput; attention operations may be uncounted.'})


if __name__=='__main__':main()
