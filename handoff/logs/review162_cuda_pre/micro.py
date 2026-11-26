"""Alternate CPU/GPU preprocessing under real inference load, not pipeline FPS."""
from pathlib import Path
import hashlib,json,sys,tarfile,time
from statistics import median
import numpy as np

HERE=Path(__file__).resolve().parent;BASE=Path('/home/jetson/nano_run/r155_runtime_accuracy_20261120')
config=json.loads((BASE/'models.json').read_text());sys.path.insert(0,str(Path(config['package'])/'src'))
from gpu_pre import GpuTRTEngine
from nano_detector import preprocess,letterbox
import cv2
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def swap():
    rows={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('SwapFree:','SwapTotal:'))}
    return (rows['SwapTotal']-rows['SwapFree'])/1024.

def main():
    target=HERE/'micro.json';assert not target.exists()
    start_swap=swap();assert start_swap<100.,'Locked timing-start memory gate'
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert Path('/sys/fs/ext4/mmcblk1p1/errors_count').read_text().strip()=='0'
    oracle=json.loads((HERE/'oracle.json').read_text());assert oracle['pixel_status']==oracle['raw_status']=='PASS'
    for p,h in oracle['source_sha256'].items():assert sha(Path(p))==h,p
    dataset=json.loads((BASE/'dataset.json').read_text());frames=[]
    with tarfile.open(str(BASE/'dev_images.tar'),'r:') as archive:
        for index in (0,1797,3594,5391,7189):
            row=dataset['images'][index];content=archive.extractfile(row['relative']).read()
            assert hashlib.sha256(content).hexdigest()==row['sha256']
            frames.append((index,cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)))
    engine=GpuTRTEngine(config['models']['original']['engine']);results=[]
    for _,frame in frames:
        for _ in range(3):
            engine(preprocess(letterbox(frame,engine.hw)[0]));engine(frame)
    for repetition in range(20):
        for index,frame in frames:
            for mode in (('cpu','gpu') if repetition%2==0 else ('gpu','cpu')):
                started=time.perf_counter()
                value=preprocess(letterbox(frame,engine.hw)[0]) if mode=='cpu' else frame
                prepared=time.perf_counter();engine(value);finished=time.perf_counter()
                times=dict(engine.t_last)
                times.update(index=index,mode=mode,repetition=repetition,cpu_pre_ms=1000*(prepared-started),
                    wall_ms=1000*(finished-started))
                times['all_pre_and_upload_ms']=times['cpu_pre_ms']+times['h2d_ms']+times.get('gpu_pre_ms',0.)
                results.append(times)
        assert swap()-start_swap<=100.,'Swap grew past the locked runtime guard'
    aggregate={}
    for mode in ('cpu','gpu'):
        selected=[r for r in results if r['mode']==mode]
        aggregate[mode]={k:median(r.get(k,0.) for r in selected) for k in
            ('cpu_pre_ms','gpu_pre_ms','h2d_ms','all_pre_and_upload_ms','exec_ms','d2h_ms','wall_ms')}
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
    target.write_text(json.dumps({'status':'PASS','rows':results,'medians_ms':aggregate,'same_engine_context':True,
        'samples_per_mode':100,'boot':boot,'start_swap_mb':start_swap,'end_swap_mb':swap(),
        'pipeline_FPS_measured':False,'source_sha256':{str(HERE/n):sha(HERE/n) for n in
            ('micro.py','gpu_pre.py','fused_pre.so','oracle.json','00_protocol.md')}},indent=2)+'\n')
    print('CUDA PRE MICROBENCHMARK COMPLETE',json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
