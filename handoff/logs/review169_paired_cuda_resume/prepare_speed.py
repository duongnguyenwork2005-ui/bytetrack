"""Freeze actual TensorRT quality and identical CUDA preprocessing for final timing."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r169_paired_cuda_resume_20261121/speed'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    out=HERE/'speed_stage';out.mkdir(exist_ok=False);(out/'quality').mkdir()
    cache=HERE/'recovered_evidence/cache';baseline=HERE.parent/'review155_runtime_accuracy/cache/original'
    records={'original':baseline,**{p.name:p for p in cache.iterdir() if p.is_dir()}}
    assert 'control' in records and len(records)>=3
    exports=read(HERE/'recovered_exports.json')['models'];built=read(HERE/'recovered_evidence/build_results.json')
    scores={};bound=[Path(__file__),HERE/'03_pipeline_details.md',HERE/'recovered_exports.json',HERE/'recovered_evidence/build_results.json',
        HERE/'continue_runtime.ps1',HERE/'finish_timing.ps1']
    for mode,folder in records.items():
        verification=read(folder/'quality_verification.json');assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(Path(p))==h,p
        score=read(folder/'quality.json');assert score['images']==7190 and score['runtime']=='actual Nano TensorRT'
        assert score['engine']['hw']==[384,640]
        if mode!='original':
            assert score['engine']['onnx_sha256']==exports[mode]['onnx_sha256']
            assert score['engine']['checkpoint_sha256']==exports[mode]['checkpoint_sha256']
            assert score['engine']['engine_sha256']==built[mode]['engine_sha256']
            inputs=read(folder/'inputs.json')
            assert inputs['config_sha256']==sha(HERE/'recovered_evidence/runtime_models.json')
            assert inputs['capture_sha256']==sha(HERE.parent/'review162_cuda_pre/capture.py')
        scores[mode]=score;shutil.copyfile(folder/'quality.json',out/'quality'/(mode+'.json'))
        bound.extend((folder/'quality.json',folder/'quality_verification.json',folder/'complete.json',folder/'inputs.json'))
    models={};gates={}
    for mode,score in scores.items():
        assert score['dataset_sha256']==scores['original']['dataset_sha256']
        refs=['original'] if mode in ('original','control') else ['original','control'];current={}
        for ref in refs:
            current.update({ref+'_'+key:score[key]>=scores[ref][key]-.005 for key in ('mAP50','mAP50_95')})
            current.update({ref+'_'+c:score['classes'][c]['AP50']>=row['AP50']-.02 for c,row in scores[ref]['classes'].items()})
        raw='PASS' if mode=='original' else read(HERE/'recovered_evidence'/(mode+'_raw_parity.json'))['status']
        models[mode]=dict(score['engine'],runtime_quality=REMOTE+'/quality/'+mode+'.json',
            runtime_quality_sha256=sha(out/'quality'/(mode+'.json')),quality_recommended=all(current.values()),
            historical_raw_parity=raw,runtime_package='/home/jetson/nano_run/r162_cuda_pre_20261120/package')
        gates[mode]=current
    for name in ('timing_v4.py','run_models_v4.py','select_full.py','00_protocol.md','01_profile_scope.md','03_pipeline_details.md','pipeline_source_derivation.json'):
        shutil.copyfile(HERE/name,out/name)
    source={REMOTE+'/'+p.name:sha(p) for p in out.iterdir() if p.is_file()}
    source.update({m['runtime_quality']:m['runtime_quality_sha256'] for m in models.values()})
    config={'models':models,'package':'same R162 CUDA package for every model',
        'baseline_quality':REMOTE+'/quality/original.json','control_quality':REMOTE+'/quality/control.json',
        'dataset_sha256':scores['original']['dataset_sha256'],'allow_diagnostic_quality_failure':True,
        'batch':'R169 recovered structure; identical CUDA preprocessing','source_sha256':source}
    (out/'timing_models_v4.json').write_text(json.dumps(config,indent=2)+'\n')
    (HERE/'speed_preflight.json').write_text(json.dumps({'status':'PASS','gates':gates,
        'raw_parity':{n:m['historical_raw_parity'] for n,m in models.items()},'quality_FAIL_remains_FAIL':True,
        'sha256':{str(p):sha(p) for p in bound},
        'portable_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}},indent=2)+'\n')
    print('R169 SPEED PREFLIGHT PASS',gates,flush=True)
if __name__=='__main__':main()
