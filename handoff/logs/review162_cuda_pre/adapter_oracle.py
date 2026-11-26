"""Verify the actual stream adapter, including masked BGR frames and NMS."""
from pathlib import Path
import hashlib,json,os,sys,tarfile
import numpy as np
os.environ['R103_EARLY_CUTOFF']='1'
HERE=Path(__file__).resolve().parent;BASE=Path('/home/jetson/nano_run/r155_runtime_accuracy_20261120')
config=json.loads((BASE/'models.json').read_text());sys.path.insert(0,str(Path(config['package'])/'src'))
from gpu_pre import GpuTRTEngine
from gpu_detector import GpuNanoDetector
from nano_detector import NanoDetector
import cv2
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=HERE/'adapter_oracle.json';assert not target.exists()
    model=config['models']['original'];assert sha(Path(model['engine']))==model['engine_sha256']
    engine=GpuTRTEngine(model['engine'])
    cpu=NanoDetector(engine,engine.hw,.25,.7,3000,None,False)
    gpu=GpuNanoDetector(engine,engine.hw,.25,.7,3000,None,False)
    dataset=json.loads((BASE/'dataset.json').read_text());checks=[]
    with tarfile.open(str(BASE/'dev_images.tar'),'r:') as archive:
        for index in np.linspace(0,7189,20).astype(int):
            row=dataset['images'][int(index)];content=archive.extractfile(row['relative']).read()
            assert hashlib.sha256(content).hexdigest()==row['sha256']
            source=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
            for masked in (False,True):
                frame=source.copy();h,w=frame.shape[:2]
                if masked:frame[h//4:h//2,w//4:w//2]=0
                a,_=cpu.detect(frame);b,_=gpu.detect(frame)
                equal=bool(np.array_equal(a,b))
                checks.append({'index':int(index),'masked_fixture':masked,'detections':len(a),'bit_exact':equal})
    result={'status':'PASS' if all(r['bit_exact'] for r in checks) else 'FAIL','pairs':checks,
        'engine':model,'max_det':3000,'confidence':.25,'iou':.7,'FPS_measured':False,
        'source_sha256':{str(HERE/n):sha(HERE/n) for n in ('adapter_oracle.py','gpu_detector.py','gpu_pre.py','fused_pre.so')}}
    target.write_text(json.dumps(result,indent=2)+'\n')
    assert result['status']=='PASS','Stream adapter parity failed'
    print('CUDA STREAM ADAPTER ORACLE PASS',len(checks),flush=True)
if __name__=='__main__':main()
