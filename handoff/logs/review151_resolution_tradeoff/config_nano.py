"""Run on Nano to bind existing engines and parity tensors before benchmarks."""
from pathlib import Path
import hashlib
import json

HOME=Path.home();ROOT=HOME/'nano_run/r151_resolution_20261120'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    modes={'original':('baseline',(384,640)), 'anchor_rebuilt':('baseline',(384,640)),
           'r352x608':('r352x608',(352,608)), 'r320x576':('r320x576',(320,576))}
    models={}
    for name,(source,hw) in modes.items():
        if name=='original':
            engine=HOME/'nano_run/r102_interp50_20261111/yolo11n_interp50_fp16.engine'
            onnx=engine.with_name('yolo11n_interp50.onnx')
            assert sha(engine)=='13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823'
        else:engine=ROOT/'models'/(source+'.engine');onnx=engine.with_suffix('.onnx')
        models[name]={'engine':str(engine),'engine_sha256':sha(engine),'onnx':str(onnx),'onnx_sha256':sha(onnx),
                      'hw':list(hw),'parity':str(ROOT/source/'parity')}
    value={'models':models,'package':str(HOME/'nano_run/r108_15fps_20261117/overlap_package')}
    target=ROOT/'models.json';assert not target.exists();target.write_text(json.dumps(value,indent=2))
    print('NANO MODEL CONFIG LOCKED',flush=True)


if __name__=='__main__':main()
