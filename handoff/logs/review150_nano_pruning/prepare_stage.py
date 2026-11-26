"""Bind the completed R145 exports and ten fixed ORT fixtures for Nano."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2
import numpy as np
import onnx
import onnxruntime as ort

HERE=Path(__file__).resolve().parent
R145=HERE.parent/'review145_yolo_recovery'
R151=HERE.parent/'review151_resolution_tradeoff'
REMOTE='/home/jetson/nano_run/r150_pruning_20261120'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')

def main():
    export=read(R145/'export_verification.json');assert export['status']=='PASS'
    output=HERE/'stage';output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('r150_detector',HERE/'source_snapshot/nano_detector.py')
    det=importlib.util.module_from_spec(spec);spec.loader.exec_module(det)
    models={};sources={}
    for name,record in export['models'].items():
        source=Path(record['path']);assert sha(source)==record['sha256']
        folder=output/name;folder.mkdir();parity=folder/'parity';parity.mkdir()
        shutil.copyfile(source,folder/'model.onnx')
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(source),sess_options=options,providers=['CPUExecutionProvider'])
        for ix,row in enumerate(record['CPU_ORT_parity']):
            image=Path(row['image']);frame=cv2.imread(str(image));assert frame is not None
            tensor=det.preprocess(det.letterbox(frame,(384,640))[0])
            expected=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            assert expected.shape==(1,9,5040) and np.isfinite(expected).all()
            np.savez_compressed(parity/('%02d.npz'%ix),input=tensor,expected=expected)
            sources[str(image)]=sha(image)
        models[name]={'onnx':REMOTE+'/'+name+'/model.onnx','onnx_sha256':record['sha256'],
                      'engine':REMOTE+'/'+name+'/model.engine','hw':[384,640],
                      'parity':REMOTE+'/'+name+'/parity'}
    # Record graph equivalence without substituting the already-built engine.
    anchor=onnx.load(R145/'onnx/anchor_384x640.onnx')
    prior=onnx.load(R151/'exports/baseline/anchor.onnx')
    graph_equal=anchor.graph.SerializeToString()==prior.graph.SerializeToString()
    write(output/'build_models.json',{'models':models,'package':'/home/jetson/nano_run/r108_15fps_20261117/overlap_package'})
    with tarfile.open(HERE/'stage.tar','w:') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.add(path,arcname=str(path.relative_to(output)).replace('\\','/'))
    paths=[Path(__file__),HERE/'00_protocol.md',HERE/'check_engine.py',R145/'export_verification.json',
           R145/'quality.json',HERE/'source_snapshot/nano_detector.py',HERE/'stage.tar']
    write(HERE/'stage_inputs.json',{'sha256':{str(p):sha(p) for p in paths},'fixture_images':sources,
          'R145_R151_anchor_graph_byte_equal':graph_equal,'reuse_engine':False,
          'archive_bytes':(HERE/'stage.tar').stat().st_size})
    print('PRUNING STAGE PASS; anchor graph equal:',graph_equal,flush=True)

if __name__=='__main__':main()
