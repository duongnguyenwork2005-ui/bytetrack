"""Reparse both-order GPU timings and retain every numerical failure."""
from pathlib import Path
import hashlib,json,re
from statistics import mean
HERE=Path(__file__).resolve().parent;E=HERE/'screen_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=HERE/'screen_verification.json';assert not target.exists()
    config=read(E/'config.json');runtime=read(E/'runtime_models.json');selection=read(E/'selection.json')
    assert sha(E/'config.json')==sha(HERE/'stage/config.json')==selection['config_sha256']
    assert sha(E/'runtime_models.json')==selection['runtime_models_sha256'] and sha(E/'gpu_rows.json')==selection['gpu_rows_sha256']
    for p,h in config['source_sha256'].items():
        path=HERE/'libraries'/Path(p).name if p.endswith('.so') else HERE/'stage'/Path(p).name
        if p.endswith('/r164_latency_pruning_20261121/build_screen.py'):path=HERE.parent/'review164_latency_pruning/build_screen.py'
        assert sha(path)==h,p
    for name,row in config['models'].items():assert sha(HERE/'stage'/(name+'.onnx'))==row['onnx_sha256']
    builds=read(E/'build_results.json');raws={}
    for name,row in builds.items():
        assert sha(E/(name+'_build.log'))==row['log_sha256']
        if row['returncode']!=0:assert name not in runtime['models'];continue
        assert runtime['models'][name]['engine_sha256']==row['engine_sha256']
        raw=read(E/(name+'_raw_parity.json'));assert raw['engine_sha256']==row['engine_sha256'] and len(raw['checks'])==10
        for check in raw['checks']:assert check['pass']==(check['box_error']<=6 and check['score_error']<=.02)
        expected='PASS' if all(c['pass'] for c in raw['checks']) else 'FAIL';assert expected==raw['status'];raws[name]=expected
    rows=read(E/'gpu_rows.json');names=['original',*runtime['models']];assert [r['mode'] for r in rows]==names+list(reversed(names))
    boot=read(E/'before.json')['boot'];stats={}
    for index,row in enumerate(rows):
        assert row['index']==index;log=E/'gpu_screen'/('%02d_%s.log'%(index,row['mode']))
        assert sha(log)==row['log_sha256'];text=log.read_text();assert 'PASSED' in text
        value=float(re.search(r'GPU Compute Time:.*median = ([\d.]+) ms',text).group(1));assert value==row['GPU_median_ms']
        for state in (row['before'],row['after']):
            assert state['boot']==boot and state['SD_errors']=='0' and state['available_MiB']>512 and state['swap_MiB']<512
            assert state['thermal_C'] and max(state['thermal_C'].values())<75
        stats.setdefault(row['mode'],[]).append(value)
    means={n:mean(v) for n,v in stats.items()};eligible=[]
    for name,values in stats.items():
        assert len(values)==2;difference=100*abs(values[0]-values[1])/means[name];assert difference<=2
        assert means[name]==selection['models'][name]['mean_two_medians_ms']
        assert abs(difference-selection['models'][name]['repeat_difference_pct'])<1e-10
        if name not in ('original','control'):
            gain=100*(1-means[name]/means['original'])
            assert abs(gain-selection['models'][name]['GPU_time_reduction_pct'])<1e-10
            if gain>=8:eligible.append(name)
    assert eligible==selection['eligible_for_actual_AP'] and selection['GPU_only'] and not selection['Nano_pipeline_FPS_measured']
    paths=[Path(__file__),*(p for p in E.rglob('*') if p.is_file()),*(HERE/'libraries').glob('*.so')]
    target.write_text(json.dumps({'status':'PASS','GPU_ms':means,'raw_status':raws,'eligible_for_actual_AP':eligible,
        'Nano_pipeline_FPS_measured':False,'sha256':{str(p):sha(p) for p in paths}},indent=2)+'\n')
    print('R167 GPU SCREEN VERIFIED',means,'RAW',raws,flush=True)
if __name__=='__main__':main()
