"""Native Nano oracle against its actual OpenCV path; no FPS estimate."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/jetson/nano_run/r155_runtime_accuracy_20261120')
config=json.loads((BASE/'models.json').read_text())
sys.path.insert(0,str(Path(config['package'])/'src'))
from gpu_pre import GpuTRTEngine
from nano_detector import letterbox,preprocess
import cv2
cv2.setNumThreads(1)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    destination=ROOT/'oracle.json'
    assert not destination.exists(),'Keep previous evidence; use a versioned oracle after a fix'
    model=config['models']['original']
    assert sha(model['engine'])==model['engine_sha256']
    engine=GpuTRTEngine(model['engine']);rng=np.random.RandomState(162)
    checks=[];raw_checks=[]
    def pixels(name,frame):
        cpu=preprocess(letterbox(frame,engine.hw)[0]);gpu=engine.tensor(frame)
        delta=np.abs(cpu-gpu)
        count=int(np.count_nonzero(cpu!=gpu))
        row={'name':name,'shape':list(frame.shape),'max_error':float(delta.max()),
             'different_elements':count,'elements':int(cpu.size),'bit_exact_fraction':1.-count/float(cpu.size),
             'within_one_LSB':bool(delta.max()<=1./255.+2e-7)}
        checks.append(row);print('PIXEL ORACLE',row,flush=True)
        return cpu,row
    shapes=[(540,960),(720,1280),(1080,1920),(384,640),(768,1280),(539,959),
            (401,701),(333,111),(7,11),(1,1),(1,9),(9,1)]
    for h,w in shapes:
        random=rng.randint(0,256,(h,w,3),dtype=np.uint8)
        pixels('random_%dx%d'%(h,w),random)
        edge=np.zeros((h,w,3),dtype=np.uint8)
        edge[:,::2,0]=255;edge[::2,:,1]=255;edge[:,:,2]=127
        pixels('edges_%dx%d'%(h,w),edge)
    dataset=json.loads((BASE/'dataset.json').read_text())
    archive=BASE/'dev_images.tar'
    assert sha(archive)==dataset['archive_sha256']
    actual=[]
    with tarfile.open(str(archive),'r:') as bundle:
        for index in np.linspace(0,7189,20).astype(int):
            row=dataset['images'][int(index)]
            content=bundle.extractfile(row['relative']).read()
            assert hashlib.sha256(content).hexdigest()==row['sha256']
            frame=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
            assert list(frame.shape[:2])==row['shape']
            cpu,check=pixels('actual_'+str(index),frame)
            actual.append((int(index),frame,cpu))
    pixel_pass=all(r['within_one_LSB'] for r in checks)
    if pixel_pass:
        for _ in range(10):engine(np.zeros(engine.in_shape,dtype=np.float32))
        for index,frame,cpu in actual:
            a=engine(cpu).copy();b=engine(frame).copy()
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            box=float(np.abs(a[:,:4]-b[:,:4]).max());score=float(np.abs(a[:,4:]-b[:,4:]).max())
            assert min(float(a[:,4:].min()),float(b[:,4:].min()))>=0
            assert max(float(a[:,4:].max()),float(b[:,4:].max()))<=1
            raw_checks.append({'index':index,'box_max_px':box,'score_max':score,'pass':box<=6. and score<=.02})
            print('RAW ORACLE',raw_checks[-1],flush=True)
    result={'pixel_status':'PASS' if pixel_pass else 'FAIL','raw_status':
            ('PASS' if all(r['pass'] for r in raw_checks) else 'FAIL') if raw_checks else 'NOT_RUN',
        'pixel_checks':checks,'raw_checks':raw_checks,'opencv':cv2.__version__,'numpy':np.__version__,
        'engine':model,'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'source_sha256':{str(ROOT/n):sha(ROOT/n) for n in ('oracle.py','gpu_pre.py','fused_pre.cu','fused_pre.so','00_protocol.md')},
        'dataset_sha256':sha(BASE/'dataset.json'),'microbenchmark_FPS':False}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    assert pixel_pass,'Pixel oracle exceeds the locked one-LSB limit'
    print('CUDA PIXEL ORACLE COMPLETE',result['pixel_status'],result['raw_status'],flush=True)

if __name__=='__main__':main()
