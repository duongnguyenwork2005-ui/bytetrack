"""Resume the interrupted matched-budget control with the original KD wrapper."""
from pathlib import Path
from shutil import copy2
import importlib.util
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r145_locked_training',HERE/'train.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)


def main():
    frozen=j.check();mode='control';run=HERE/'runs'/mode;last=run/'weights/last.pt'
    assert not (HERE/'control_training.json').exists()
    c=torch.load(last,map_location='cpu',weights_only=False)
    assert c['epoch']==2 and c['optimizer'] is not None
    for key,value in {'epochs':5,'batch':4,'imgsz':640,'seed':145,'amp':False,'workers':0}.items():
        assert c['train_args'][key]==value,(key,c['train_args'][key])
    del c
    backup=HERE/'resume_before_control';backup.mkdir(exist_ok=False)
    paths=[last,run/'weights/best.pt',run/'results.csv',run/'args.yaml']
    for source in paths:copy2(source,backup/source.name)
    lock={str(p):j.sha(p) for p in [Path(__file__),HERE/'02_resume_protocol.md',*backup.iterdir()]}
    j.write(HERE/'resume_control_inputs.json',{'sha256':lock,'completed_epochs_before':3,
            'total_epochs':5,'standard_resume':True,'uninterrupted_RNG_parity_claimed':False})
    model=j.YOLO(str(last),task='detect')
    j.train_with_distillation(model,dict(trainer=j.PrunedTrainer,resume=str(last),device=0,workers=0,
                                      batch=4,imgsz=640,plots=False,patience=0))
    best=run/'weights/best.pt';trained=j.YOLO(str(best),task='detect').model
    assert sum(p.numel() for p in trained.parameters())==frozen['models'][mode]['parameters']
    history=list(j.csv.DictReader((run/'results.csv').open()))
    assert [int(r['epoch']) for r in history]==[1,2,3,4,5]
    j.check()
    for p,h in lock.items():assert j.sha(p)==h,p
    j.write(HERE/'control_training.json',{'status':'PASS','best':str(best),'best_sha256':j.sha(best),
        'last_sha256':j.sha(last),'history_sha256':j.sha(run/'results.csv'),
        'parameters':frozen['models'][mode]['parameters'],'epochs_budget':5,'Nano_FPS':None,
        'resumed_after_epochs':3,'uninterrupted_RNG_parity_claimed':False})
    print('RESUMED CONTROL TRAINING PASS',flush=True)


if __name__=='__main__':main()
