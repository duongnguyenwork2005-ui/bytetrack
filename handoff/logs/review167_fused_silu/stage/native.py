"""Nano Python3.6: independent graph builds, raw checks, then GPU-only ABBA."""
from pathlib import Path
import ctypes,importlib.util,json,re,subprocess,time
from statistics import mean
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r167_unchanged_gpu_helpers',HERE.parent/'r164_latency_pruning_20261121/build_screen.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.HERE=HERE
def main():
    config=base.read(HERE/'config.json');base.check(config)
    handles=[ctypes.CDLL(path,mode=ctypes.RTLD_GLOBAL) for path in config['libraries']]
    assert not (HERE/'before.json').exists();before=base.state();base.write(HERE/'before.json',before);boot=before['boot']
    valid={};builds={}
    for name,model in config['models'].items():
        base.check(config);assert not Path(model['engine']).exists()
        command=[base.TRT,'--onnx='+model['onnx'],'--saveEngine='+model['engine'],'--fp16','--workspace=512','--buildOnly','--profilingVerbosity=detailed']
        if model.get('plugin'):command.append('--plugins='+model['plugin'])
        log=HERE/(name+'_build.log');start=time.monotonic()
        with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record={'returncode':result.returncode,'seconds':time.monotonic()-start,'log_sha256':base.sha(log),
            'command':command,'after':base.state()}
        if result.returncode==0:
            model['engine_sha256']=base.sha(Path(model['engine']));record['engine_sha256']=model['engine_sha256'];valid[name]=model
        builds[name]=record;base.write(HERE/'build_results.json',builds)
        print('R167 BUILD',name,result.returncode,record['seconds'],flush=True)
    assert 'control' in valid,'Matched rebuild failed; retain failure, do not screen'
    config['models']=valid;base.check(config);base.write(HERE/'runtime_models.json',config)
    for name in valid:base.raw_checks(config,name)
    models={'original':config['original']};models.update(valid);order=list(models)+list(reversed(list(models)))
    (HERE/'gpu_screen').mkdir(exist_ok=False);rows=[]
    for index,name in enumerate(order):
        start=base.health(boot);model=models[name]
        command=[base.TRT,'--loadEngine='+model['engine'],'--warmUp=500','--duration=6','--iterations=100']
        if model.get('plugin'):command.append('--plugins='+model['plugin'])
        log=HERE/'gpu_screen'/('%02d_%s.log'%(index,name))
        with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text();assert result.returncode==0 and 'PASSED' in text,name
        value=float(re.search(r'GPU Compute Time:.*median = ([\d.]+) ms',text).group(1))
        rows.append({'index':index,'mode':name,'GPU_median_ms':value,'before':start,'after':base.health(boot),
            'log_sha256':base.sha(log)});base.write(HERE/'gpu_rows.json',rows)
        print('R167 GPU',name,value,flush=True)
    report={}
    for name in models:
        values=[r['GPU_median_ms'] for r in rows if r['mode']==name];assert len(values)==2
        difference=100*abs(values[0]-values[1])/mean(values);assert difference<=2,(name,values)
        report[name]={'mean_two_medians_ms':mean(values),'repeat_difference_pct':difference}
    eligible=[];anchor=report['original']['mean_two_medians_ms']
    for name in valid:
        gain=100*(1-report[name]['mean_two_medians_ms']/anchor);report[name]['GPU_time_reduction_pct']=gain
        if name!='control' and gain>=8:eligible.append(name)
    base.check(config);base.write(HERE/'selection.json',{'status':'PASS','GPU_only':True,'models':report,
        'eligible_for_actual_AP':eligible,'Nano_pipeline_FPS_measured':False,'after':base.state(),
        'config_sha256':base.sha(HERE/'config.json'),'runtime_models_sha256':base.sha(HERE/'runtime_models.json'),
        'gpu_rows_sha256':base.sha(HERE/'gpu_rows.json')})
    print('R167 GPU SCREEN COMPLETE',json.dumps(report),'AP candidates',eligible,flush=True)
if __name__=='__main__':main()
