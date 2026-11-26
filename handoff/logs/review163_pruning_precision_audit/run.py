"""Reuse the frozen Nano-path FP32 evaluator for the completed R145 models."""
from pathlib import Path
import importlib.util
import sys

HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'review151_resolution_tradeoff'
spec=importlib.util.spec_from_file_location('r163_frozen_cpu_evaluator',PARENT/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.HERE=HERE;base.SHAPES={'control':(384,640),'channel25_KD':(384,640)}

def check():
    for path,digest in base.read(HERE/'inputs.json')['sha256'].items():assert base.sha(path)==digest,path
base.check=check

def prepare():
    target=HERE/'inputs.json';assert not target.exists()
    frozen=base.read(PARENT/'inputs.json')['sha256'].copy()
    for path,digest in frozen.items():assert base.sha(path)==digest,path
    paths=[Path(__file__),HERE/'00_scope.md',PARENT/'run.py'];models={}
    for mode in base.SHAPES:
        record=HERE.parent/'review145_yolo_recovery'/(mode+'_training.json')
        training=base.read(record);checkpoint=Path(training['best']);assert base.sha(checkpoint)==training['best_sha256']
        paths.extend([record,checkpoint]);models[mode]=str(checkpoint)
    frozen.update({str(p):base.sha(p) for p in paths})
    base.write(target,{'sha256':frozen,'models':models,'runtime':'PC PyTorch FP32',
          'precision_only_ablation':False,'Nano_FPS_measured':False})
    print('PRUNING FP32 AUDIT PREFLIGHT PASS',flush=True)

def evaluate(mode):
    assert mode in base.SHAPES;base.ANCHOR=Path(base.read(HERE/'inputs.json')['models'][mode]);base.evaluate(mode)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:assert sys.argv[1]=='evaluate';evaluate(sys.argv[2])
