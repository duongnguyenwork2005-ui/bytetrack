"""Freeze actual AP, both plugin binaries, and a common final runtime package."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r167_fused_silu_20261121'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    package=read(HERE/'package_verification.json');assert package['status']=='PASS'
    out=HERE/'speed_stage';out.mkdir(exist_ok=False);(out/'quality').mkdir()
    records={'original':HERE.parent/'review155_runtime_accuracy/cache/original'}
    records.update({n:HERE/'actual_evidence/cache'/n for n in read(HERE/'actual_models.json')['models']})
    scores={};bound=[Path(__file__),HERE/'package_verification.json',HERE/'screen_verification.json']
    for name,folder in records.items():
        verification=read(folder/'quality_verification.json');assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(Path(p))==h,p
        quality=read(folder/'quality.json');assert quality['images']==7190 and quality['runtime']=='actual Nano TensorRT'
        scores[name]=quality;shutil.copyfile(folder/'quality.json',out/'quality'/(name+'.json'))
        bound.extend((folder/'quality.json',folder/'quality_verification.json',folder/'inputs.json',folder/'complete.json'))
    models={};gates={}
    for name,quality in scores.items():
        assert quality['engine']['hw']==[384,640] and quality['dataset_sha256']==scores['original']['dataset_sha256']
        current={};refs=['original'] if name in ('original','control') else ['original','control']
        for ref in refs:
            current.update({ref+'_'+k:quality[k]>=scores[ref][k]-.005 for k in ('mAP50','mAP50_95')})
            current.update({ref+'_'+c:quality['classes'][c]['AP50']>=r['AP50']-.02 for c,r in scores[ref]['classes'].items()})
        raw='PASS' if name=='original' else read(HERE/'screen_evidence'/(name+'_raw_parity.json'))['status']
        models[name]=dict(quality['engine'],runtime_quality=REMOTE+'/speed/quality/'+name+'.json',
            runtime_quality_sha256=sha(out/'quality'/(name+'.json')),runtime_package=REMOTE+'/package',
            quality_recommended=all(current.values()),historical_raw_parity=raw)
        gates[name]=current
    old=HERE.parent/'review164_latency_pruning';source=(old/'run_models_v4.py').read_text()
    a='nano_run/r162_cuda_pre_20261120/package';b='nano_run/r167_fused_silu_20261121/package'
    assert source.count(a)==2;source=source.replace(a,b)
    a='        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)'
    assert source.count(a)==1;source=source.replace(a,"        if config['models'][name].get('plugin'):command.append('--plugins='+config['models'][name]['plugin'])\n"+a)
    (out/'run_models_v5.py').write_text(source)
    wrapper=(old/'timing_v4.py').read_text();assert wrapper.count('run_models_v4.py')==1
    (out/'timing_v5.py').write_text(wrapper.replace('run_models_v4.py','run_models_v5.py'))
    selector=(old/'select_full.py').read_text().replace('timing_models_v4.json','timing_models_v5.json').replace('R164','R167')
    (out/'select_full.py').write_text(selector)
    for name in ('00_protocol.md','01_cuda_namespace_fix.md','02_fast_variant_scope.md','package_verification.json'):
        shutil.copyfile(HERE/name,out/name)
    sources={REMOTE+'/speed/'+p.name:sha(p) for p in out.iterdir() if p.is_file()}
    sources.update({m['runtime_quality']:m['runtime_quality_sha256'] for m in models.values()})
    for p,h in read(HERE/'actual_models.json')['source_sha256'].items():sources[p]=h
    for name,row in package['candidate_entries'].items():
        if 'sha256' in row and name.startswith('src/'):sources[REMOTE+'/package/'+name]=row['sha256']
    config={'models':models,'package':'same R167 loader + R162 CUDA preprocessing for every model',
        'baseline_quality':REMOTE+'/speed/quality/original.json','control_quality':REMOTE+'/speed/quality/control.json',
        'dataset_sha256':scores['original']['dataset_sha256'],'allow_diagnostic_quality_failure':True,
        'batch':'R167 SiLU fusion; no weights changed; raw FAIL retained','source_sha256':sources}
    (out/'timing_models_v5.json').write_text(json.dumps(config,indent=2)+'\n')
    (HERE/'speed_preflight.json').write_text(json.dumps({'status':'PASS','AP_gates':gates,
        'raw_parity':{n:m['historical_raw_parity'] for n,m in models.items()},'raw_FAIL_remains_FAIL':True,
        'source_sha256':{str(p):sha(p) for p in [*bound,old/'run_models_v4.py',old/'timing_v4.py',old/'select_full.py']},
        'portable_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}},indent=2)+'\n')
    print('R167 SPEED PREFLIGHT PASS',gates,flush=True)
if __name__=='__main__':main()
