"""Verify all six actual-engine AP audits and freeze portable timing inputs."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
REMOTE='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')

def main():
    config=read(HERE/'models.json');baseline=read(HERE/'cache/original/quality.json')
    output=HERE/'timing_stage';output.mkdir(exist_ok=False);(output/'quality').mkdir()
    reports={};paths=[Path(__file__),HERE/'timing.py',HERE/'01_speed_tradeoff_scope.md']
    for name,model in config['models'].items():
        folder=HERE/'cache'/name;verification=read(folder/'quality_verification.json')
        assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(p)==h,p
        report=read(folder/'quality.json');assert report['engine']==model
        assert report['dataset_sha256']==baseline['dataset_sha256']
        gates={k:report[k]>=baseline[k]-.005 for k in ('mAP50','mAP50_95')}
        gates.update({c:report['classes'][c]['AP50']>=r['AP50']-.02 for c,r in baseline['classes'].items()})
        reports[name]={'mAP50':report['mAP50'],'mAP50_95':report['mAP50_95'],
            'delta_AP50_pp':100*(report['mAP50']-baseline['mAP50']),
            'delta_AP50_95_pp':100*(report['mAP50_95']-baseline['mAP50_95']),
            'per_class_AP50':{c:r['AP50'] for c,r in report['classes'].items()},
            'gates':gates,'quality_recommended':all(gates.values()),
            'historical_raw_parity':config['historical_raw_parity'][name]}
        target=output/'quality'/(name+'.json');shutil.copyfile(folder/'quality.json',target)
        model.update(runtime_quality=REMOTE+'/quality/'+name+'.json',runtime_quality_sha256=sha(target),
                     quality_recommended=all(gates.values()),historical_raw_parity=reports[name]['historical_raw_parity'])
        paths.extend([folder/'quality.json',folder/'quality_verification.json'])
    shutil.copyfile(HERE/'timing.py',output/'timing.py')
    shutil.copyfile(HERE.parent/'review150_nano_pruning/run_models.py',output/'run_models.py')
    config.update(baseline_quality=REMOTE+'/quality/original.json',dataset_sha256=baseline['dataset_sha256'],
                  allow_diagnostic_quality_failure=True,
                  source_sha256={REMOTE+'/'+name:sha(output/name) for name in ('timing.py','run_models.py')})
    config['source_sha256'].update({m['runtime_quality']:m['runtime_quality_sha256'] for m in config['models'].values()})
    write(output/'timing_models.json',config)
    write(HERE/'actual_AP_summary.json',{'status':'PASS','images_per_model':7190,'models':reports,
          'Nano_pipeline_FPS_measured':False,'definition':'Actual TensorRT + deployed argmax/NMS on internal dev',
          'GT_counts':{c:r['GT_boxes'] for c,r in baseline['classes'].items()}})
    write(HERE/'timing_preflight.json',{'status':'PASS','sha256':{str(p):sha(p) for p in paths},
          'portable_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}})
    for name,r in reports.items():print(name,r['mAP50'],r['mAP50_95'],r['quality_recommended'],flush=True)

if __name__=='__main__':main()
