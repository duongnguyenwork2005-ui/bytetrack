"""Independently reparse and audit the Nano GPU-only selection, retaining raw FAILs."""
from pathlib import Path
import hashlib,json,re,statistics
HERE=Path(__file__).resolve().parent;E=HERE/'screen_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def health(value,boot):
    assert value['boot']==boot and value['SD_errors']=='0'
    assert value['available_MiB']>512 and value['swap_MiB']<512
    assert value['thermal_C'] and max(value['thermal_C'].values())<75
def main():
    target=HERE/'screen_verification.json';assert not target.exists()
    config=read(E/'screen_config.json');runtime=read(E/'runtime_models.json');selection=read(E/'selection.json')
    assert sha(E/'screen_config.json')==sha(HERE/'stage/screen_config.json')==selection['config_sha256']
    assert sha(E/'runtime_models.json')==selection['runtime_models_sha256']
    assert sha(E/'gpu_rows.json')==selection['GPU_rows_sha256']
    for p,h in config['source_sha256'].items():assert sha(HERE/'stage'/Path(p).name)==h,p
    for name,model in config['models'].items():
        assert sha(HERE/'stage'/name/'model.onnx')==model['onnx_sha256']
        for row in model['parity']:assert sha(HERE/'stage'/name/'parity'/Path(row['path']).name)==row['sha256']
    builds=read(E/'build_results.json');raw_status={}
    assert set(builds)==set(config['models'])
    for name,result in builds.items():
        assert sha(E/(name+'_build.log'))==result['log_sha256']
        if result['returncode']!=0:assert name not in runtime['models'];continue
        assert result['engine_sha256']==runtime['models'][name]['engine_sha256']
        raw=read(E/(name+'_raw_parity.json'));assert raw['engine_sha256']==result['engine_sha256']
        assert len(raw['checks'])==10
        for row in raw['checks']:assert row['pass']==(row['box_error']<=6 and row['score_error']<=.02)
        expected='PASS' if all(row['pass'] for row in raw['checks']) else 'FAIL';assert raw['status']==expected
        raw_status[name]=expected
    rows=read(E/'gpu_rows.json');models={'original':config['original'],**runtime['models']}
    expected=list(models)+list(reversed(list(models)));assert [r['mode'] for r in rows]==expected
    boot=read(E/'screen_before.json')['boot'];times={}
    for ix,row in enumerate(rows):
        assert row['index']==ix;health(row['before'],boot);health(row['after'],boot)
        log=E/'gpu_screen'/('%02d_%s.log'%(ix,row['mode']));assert sha(log)==row['log_sha256']
        text=log.read_text();assert 'PASSED' in text
        value=float(re.search(r'GPU Compute Time:.*median = ([\d.]+) ms',text).group(1));assert value==row['GPU_median_ms']
        times.setdefault(row['mode'],[]).append(value)
    report={}
    for name,values in times.items():
        assert len(values)==2;mean=statistics.mean(values);difference=100*abs(values[0]-values[1])/mean
        assert difference<=2
        assert mean==selection['models'][name]['mean_of_two_medians_ms']
        assert abs(difference-selection['models'][name]['repeat_difference_pct'])<1e-10
        report[name]={'GPU_ms':mean,'repeat_difference_pct':difference}
    eligible=[];base=report['original']['GPU_ms']
    for name in runtime['models']:
        gain=100*(1-report[name]['GPU_ms']/base);report[name]['GPU_time_reduction_pct']=gain
        assert abs(gain-selection['models'][name]['GPU_time_reduction_pct'])<1e-10
        if gain>=8:eligible.append(name)
    selected=[]
    if eligible:
        selected.append(min(eligible,key=lambda n:(report[n]['GPU_ms'],n)))
        least=min(eligible,key=lambda n:(1-runtime['models'][n]['ConvMAC']/config['anchor_statistics']['ConvMAC'],n))
        if least not in selected:selected.append(least)
    assert selected==selection['selected_for_training'] and eligible==selection['eligible']
    assert selection['status']=='PASS' and selection['GPU_only'] and not selection['Nano_pipeline_FPS_measured']
    health(selection['after'],boot)
    hashes={str(p):sha(p) for p in E.rglob('*') if p.is_file()}
    hashes[str(Path(__file__))]=sha(Path(__file__))
    target.write_text(json.dumps({'status':'PASS','GPU_only':True,'Nano_pipeline_FPS_measured':False,
        'models':report,'selected_for_training':selected,'raw_parity_status':raw_status,
        'sha256':hashes,'boot':boot},indent=2)+'\n',encoding='utf-8')
    print('R164 GPU SCREEN VERIFIED',report,'SELECTED',selected,'RAW',raw_status,flush=True)
if __name__=='__main__':main()
