"""Prepare immutable actual-AP-based timing inputs for pruning or height352."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('batch',choices=('pruning','height'));args=parser.parse_args()
    pruning=args.batch=='pruning'
    folder=HERE.parent/('review150_nano_pruning' if pruning else 'review160_height352')
    remote='/home/jetson/nano_run/'+('r150_pruning_20261120' if pruning else 'r160_height352_20261120')
    raw=read(folder/('runtime_models.json' if pruning else 'models.json'))
    original=read(HERE/'models.json')['models']['original']
    models={'original':original}
    for name,model in raw['models'].items():
        if name!='original':models[name]=model
    output=folder/'speed_stage';output.mkdir(exist_ok=False);(output/'quality').mkdir()
    baseline=read(HERE/'cache/original/quality.json');reports={};paths=[Path(__file__)]
    for name,model in models.items():
        cache=(HERE if name=='original' else folder)/'cache'/name
        verification=read(cache/'quality_verification.json');assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(p)==h,p
        report=read(cache/'quality.json');assert report['engine']==model
        assert report['dataset_sha256']==baseline['dataset_sha256']
        gates={k:report[k]>=baseline[k]-.005 for k in ('mAP50','mAP50_95')}
        gates.update({c:r['AP50']>=baseline['classes'][c]['AP50']-.02 for c,r in report['classes'].items()})
        if name=='original':raw_status='PASS'
        else:
            marker=folder/(name+'_parity.json')
            if marker.exists():
                parity=read(marker);assert parity['status']=='PASS' and parity['engine_sha256']==model['engine_sha256']
                assert len(parity['checks'])==10;raw_status='PASS';paths.append(marker)
            else:
                log=folder/(name+'_parity.log' if pruning else 'raw_parity.log')
                assert 'AssertionError:' in log.read_text(),('Raw check infrastructure error requires review',name)
                raw_status='FAIL';paths.append(log)
        target=output/'quality'/(name+'.json');shutil.copyfile(cache/'quality.json',target)
        model.update(runtime_quality=remote+'/quality/'+name+'.json',runtime_quality_sha256=sha(target),
                      quality_recommended=all(gates.values()),historical_raw_parity=raw_status)
        reports[name]={'mAP50':report['mAP50'],'mAP50_95':report['mAP50_95'],'classes':report['classes'],
                       'gates_vs_original':gates,'quality_recommended':all(gates.values()),'raw_parity':raw_status}
        paths.extend([cache/'quality.json',cache/'quality_verification.json'])
    for name in ('timing_v2.py','run_models_v2.py'):shutil.copyfile(HERE/name,output/name)
    shutil.copyfile(HERE/'select_extra.py',output/'select_full.py')
    scope=folder/('01_speed_screen.md' if pruning else '00_protocol.md');shutil.copyfile(scope,output/'scope.md')
    config={'models':models,'package':raw['package'],'baseline_quality':remote+'/quality/original.json',
        'dataset_sha256':baseline['dataset_sha256'],'allow_diagnostic_quality_failure':True,'batch':args.batch,
        'source_sha256':{remote+'/'+name:sha(output/name) for name in ('timing_v2.py','run_models_v2.py','select_full.py','scope.md')}}
    config['source_sha256'].update({model['runtime_quality']:model['runtime_quality_sha256'] for model in models.values()})
    write(output/'timing_models_v2.json',config)
    if pruning:
        candidate=reports['channel25_KD'];control=reports['control']
        g={k:candidate[k]>=control[k]-.005 for k in ('mAP50','mAP50_95')}
        g.update({c:r['AP50']>=control['classes'][c]['AP50']-.02 for c,r in candidate['classes'].items()})
        candidate['gates_vs_matched_control']=g
        candidate['quality_recommended_both']=candidate['quality_recommended'] and all(g.values())
    write(folder/'actual_AP_summary.json',{'status':'PASS','models':reports,'images_per_model':7190,'Nano_FPS_measured':False})
    paths.extend([HERE/'select_extra.py',scope])
    write(folder/'speed_preflight.json',{'status':'PASS','sha256':{str(p):sha(p) for p in paths},
        'portable_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}})
    print('EXTRA SPEED PREFLIGHT',args.batch,{n:r['mAP50'] for n,r in reports.items()},flush=True)

if __name__=='__main__':main()
