"""Export and verify every candidate; no quality/FPS selection here."""
from pathlib import Path
import hashlib,json,os,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
(HERE/'settings').mkdir(exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(1)
sys.path.insert(0,str(HERE.parent/'review150_nano_pruning/source_snapshot'))
from nano_detector import preprocess,letterbox
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def main():
    frozen=read(HERE/'inputs_v2.json');assert frozen['status']=='PASS'
    for p,h in frozen['sha256'].items():assert sha(Path(p))==h,p
    assert not (HERE/'export_verification.json').exists()
    images=(HERE.parent/'review145_yolo_recovery/data/dev300.txt').read_text().splitlines()[:10]
    sessions=ort.SessionOptions();sessions.intra_op_num_threads=2;sessions.inter_op_num_threads=1
    records={}
    for mode,record in frozen['models'].items():
        weights=Path(record['path']);assert sha(weights)==record['sha256']
        folder=HERE/'export_parity'/mode;folder.mkdir(parents=True,exist_ok=False)
        model=YOLO(str(weights));net=model.model.cpu().float().eval()
        with torch.inference_mode():assert net(torch.zeros(1,3,384,640))[0].shape==(1,9,5040)
        target=Path(model.export(format='onnx',imgsz=(384,640),batch=1,opset=12,dynamic=False,
                   simplify=False,half=False,nms=False,device='cpu'))
        graph=onnx.load(str(target));onnx.checker.check_model(graph)
        assert [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]==[1,3,384,640]
        inference=ort.InferenceSession(str(target),sess_options=sessions,providers=['CPUExecutionProvider'])
        checks=[]
        for index,name in enumerate(images):
            image=Path(name);frame=cv2.imread(str(image));assert frame is not None
            x=preprocess(letterbox(frame,(384,640))[0])
            with torch.inference_mode():
                value=net(torch.from_numpy(x));value=value[0] if isinstance(value,(tuple,list)) else value
                expected=value.numpy()
            actual=inference.run(None,{'images':x})[0]
            assert actual.shape==expected.shape==(1,9,5040) and np.isfinite(actual).all()
            assert np.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02),(mode,index,'boxes')
            assert np.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5),(mode,index,'scores')
            np.savez_compressed(folder/('%02d.npz'%index),input=x,expected=expected)
            checks.append({'image':str(image),'sha256':sha(image),'max_box_error':float(np.abs(actual[:,:4]-expected[:,:4]).max()),
                           'max_score_error':float(np.abs(actual[:,4:]-expected[:,4:]).max())})
        records[mode]={'path':str(target),'sha256':sha(target),'input':[1,3,384,640],'output':[1,9,5040],
            'weights':record,'checks':checks,'parity_sha256':{str(p):sha(p) for p in folder.glob('*.npz')}}
        write(HERE/'export_partial.json',records);print('R164 ONNX PARITY PASS',mode,flush=True)
    write(HERE/'export_verification.json',{'status':'PASS','models':records,
        'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'inputs_v2.json')}})
    print('R164 ALL FIVE ONNX EXPORTS PASS',flush=True)
if __name__=='__main__':main()
