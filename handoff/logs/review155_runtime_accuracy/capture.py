"""Nano Python3.6: cache actual TensorRT candidates, never estimate pipeline FPS."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='meta_',suffix='.tmp',dir=str(path.parent))
    with os.fdopen(fd,'w') as f:f.write(json.dumps(value,indent=2))
    os.replace(tmp,str(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('mode')
    parser.add_argument('dataset',type=Path);parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();config=read(args.config);model=config['models'][args.mode]
    assert Path(args.mode).name==args.mode
    dataset=read(args.dataset);images=dataset['images'];assert len(images)==7190
    assert sha(args.archive)==dataset['archive_sha256'] and sha(Path(model['engine']))==model['engine_sha256']
    assert sha(Path(model['onnx']))==model['onnx_sha256']
    package=Path(config['package']);sys.path.insert(0,str(package/'src'))
    from nano_detector import TRTEngine,letterbox,preprocess,xywh2xyxy
    import nano_detector
    import tensorrt as trt
    import cv2
    cv2.setNumThreads(1)
    output=args.output/args.mode;output.mkdir(parents=True,exist_ok=True)
    frozen={'engine':model,'config_sha256':sha(args.config),'dataset_sha256':sha(args.dataset),
        'archive_sha256':sha(args.archive),'capture_sha256':sha(Path(__file__)),
        'detector_source_sha256':sha(Path(nano_detector.__file__)),'images':7190,'candidate_confidence':.001,
        'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'numpy':np.__version__,'opencv':cv2.__version__,
        'tensorrt':trt.__version__}
    inputs=output/'inputs.json'
    if inputs.exists():
        previous=read(inputs)
        # A reboot changes timing conditions, not deterministic detection quality.
        assert {k:v for k,v in previous.items() if k!='boot'}=={k:v for k,v in frozen.items() if k!='boot'}
    else:write(inputs,frozen)
    engine=TRTEngine(model['engine']);assert list(engine.in_shape)==[1,3]+model['hw']
    assert engine.out_shape[0:2]==(1,9)
    for _ in range(10):engine(np.zeros(engine.in_shape,dtype=np.float32))
    with tarfile.open(args.archive,'r:') as archive:
        for first in range(0,len(images),100):
            selected=images[first:first+100];number=first//100
            target=output/('%04d.npz'%number);marker=target.with_suffix('.json')
            indices=[row['index'] for row in selected]
            if marker.exists():
                prior=read(marker);assert prior['indices']==indices and sha(target)==prior['sha256']
                continue
            assert not target.exists(),'Unmarked final chunk requires review: '+str(target)
            chunks=[];offsets=[0];shapes=[]
            for row in selected:
                member=archive.getmember(row['relative']);content=archive.extractfile(member).read()
                assert hashlib.sha256(content).hexdigest()==row['sha256']
                frame=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR)
                assert frame is not None and list(frame.shape[:2])==row['shape']
                if row['index']<10:
                    fd,path=tempfile.mkstemp(suffix='.jpg',dir='/dev/shm');os.close(fd)
                    try:
                        Path(path).write_bytes(content);assert np.array_equal(frame,cv2.imread(path))
                    finally:os.unlink(path)
                x=preprocess(letterbox(frame,tuple(model['hw']))[0]);raw=engine(x)
                assert raw.shape[0:2]==(1,9) and np.isfinite(raw).all()
                p=raw[0].T;scores=p[:,4:].max(axis=1);classes=p[:,4:].argmax(axis=1)
                assert scores.min()>=0 and scores.max()<=1
                keep=scores>.001
                candidates=np.column_stack((xywh2xyxy(p[keep,:4]),scores[keep],classes[keep])).astype(np.float32)
                chunks.append(candidates);offsets.append(offsets[-1]+len(candidates));shapes.append(row['shape'])
            fd,temp=tempfile.mkstemp(prefix='chunk_',suffix='.npz',dir=str(output));os.close(fd)
            np.savez_compressed(temp,data=np.concatenate(chunks),offsets=np.array(offsets,dtype=np.int64),
                  indices=np.array(indices,dtype=np.int32),orig_shapes=np.array(shapes,dtype=np.int32),hw=np.array(model['hw'],dtype=np.int32))
            assert not target.exists();os.replace(temp,str(target))
            write(marker,{'indices':indices,'sha256':sha(target),'candidate_count':offsets[-1]})
            print('NANO DETECTION CACHE',args.mode,first+len(selected),'/7190',flush=True)
    files={}
    for number in range(72):
        target=output/('%04d.npz'%number);marker=target.with_suffix('.json');entry=read(marker)
        assert sha(target)==entry['sha256'];files[str(target.name)]=entry['sha256'];files[str(marker.name)]=sha(marker)
    assert sha(Path(model['engine']))==model['engine_sha256'] and sha(args.archive)==dataset['archive_sha256']
    write(output/'complete.json',{'status':'PASS','images':7190,'engine_sha256':model['engine_sha256'],
             'input':list(engine.in_shape),'output':list(engine.out_shape),'sha256':files,'Nano_pipeline_FPS_measured':False})
    print('NANO DETECTION COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
