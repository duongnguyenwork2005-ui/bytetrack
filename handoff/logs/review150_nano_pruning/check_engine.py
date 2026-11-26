"""Run on Nano Python 3.6: validate static TensorRT output against CPU ORT."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);parser.add_argument('mode');args=parser.parse_args()
    config=json.loads(args.config.read_text());model=config['models'][args.mode]
    package=Path(config['package']);sys.path.insert(0,str(package/'src'))
    from nano_detector import TRTEngine
    engine_path=Path(model['engine']);assert sha(engine_path)==model['engine_sha256']
    samples=sorted(Path(model['parity']).glob('*.npz'));assert len(samples)==10
    engine=TRTEngine(str(engine_path));assert list(engine.in_shape)==[1,3]+model['hw']
    checks=[]
    for path in samples:
        with np.load(path,allow_pickle=False) as f:x=f['input'];expected=f['expected']
        actual=engine(np.ascontiguousarray(x))
        assert actual.shape==expected.shape and actual.shape[1]==9 and np.isfinite(actual).all()
        assert actual[:,4:].min()>=0 and actual[:,4:].max()<=1
        selected=np.maximum(actual[:,4:].max(axis=1),expected[:,4:].max(axis=1))>=.05
        box=float(abs(actual[:,:4]-expected[:,:4]).transpose(0,2,1)[selected].max()) if selected.any() else 0.
        score=float(abs(actual[:,4:]-expected[:,4:]).max())
        assert box<=6. and score<=.02,(args.mode,str(path),box,score)
        checks.append({'path':str(path),'sha256':sha(path),'box_max_absdiff_active':box,'score_max_absdiff':score})
    output=args.config.parent/(args.mode+'_parity.json')
    output.write_text(json.dumps({'status':'PASS','mode':args.mode,'engine_sha256':sha(engine_path),
                      'input':list(engine.in_shape),'output':list(engine.out_shape),'checks':checks},indent=2))
    print('ENGINE PARITY PASS',args.mode,flush=True)


if __name__=='__main__':main()
