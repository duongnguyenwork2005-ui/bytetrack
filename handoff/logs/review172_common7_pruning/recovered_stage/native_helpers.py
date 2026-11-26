"""Nano Python3.6: build all candidates, preserve raw failures, screen GPU only."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
from statistics import mean

HERE=Path(__file__).resolve().parent
TRT='/usr/src/tensorrt/bin/trtexec'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for value in iter(lambda:stream.read(1024*1024),b''):h.update(value)
    return h.hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')
def state():
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith(('SwapFree:','SwapTotal:','MemAvailable:'))}
    thermal={}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            name=(zone/'type').read_text().strip()
            if any(x in name.lower() for x in ('cpu','gpu')):thermal[name]=float((zone/'temp').read_text())/1000.
        except (OSError,ValueError):pass
    return {'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'swap_MiB':(memory['SwapTotal']-memory['SwapFree'])/1024.,'available_MiB':memory['MemAvailable']/1024.,
        'thermal_C':thermal,'SD_errors':Path('/sys/fs/ext4/mmcblk1p1/errors_count').read_text().strip()}
def check(config):
    for p,h in config['source_sha256'].items():assert sha(p)==h,p
    for model in config['models'].values():assert sha(model['onnx'])==model['onnx_sha256']
    assert sha(config['original']['engine'])==config['original']['engine_sha256']
def health(boot):
    value=state();assert value['boot']==boot and value['SD_errors']=='0'
    assert value['available_MiB']>512 and value['swap_MiB']<512,value
    assert value['thermal_C'] and max(value['thermal_C'].values())<75,value
    return value
def raw_checks(config,name):
    import numpy as np
    sys.path.insert(0,str(Path(config['package'])/'src'))
    from nano_detector import TRTEngine
    model=config['models'][name];engine=TRTEngine(model['engine']);checks=[]
    assert engine.in_shape==(1,3,384,640) and engine.out_shape==(1,9,5040)
    for _ in range(10):engine(np.zeros(engine.in_shape,dtype=np.float32))
    for row in model['parity']:
        assert sha(row['path'])==row['sha256']
        with np.load(row['path'],allow_pickle=False) as cache:x=cache['input'];expected=cache['expected']
        actual=engine(x);assert actual.shape==expected.shape and np.isfinite(actual).all()
        assert actual[:,4:].min()>=0 and actual[:,4:].max()<=1
        box=float(np.abs(actual[:,:4]-expected[:,:4]).max());score=float(np.abs(actual[:,4:]-expected[:,4:]).max())
        checks.append({'fixture':row['path'],'box_error':box,'score_error':score,'pass':box<=6 and score<=.02})
    report={'status':'PASS' if all(r['pass'] for r in checks) else 'FAIL','engine_sha256':model['engine_sha256'],'checks':checks}
    write(HERE/(name+'_raw_parity.json'),report)
    print('R164 RAW PARITY',name,report['status'],flush=True)
def main():
    source=HERE/'screen_config.json';config=read(source);check(config)
    assert not (HERE/'screen_before.json').exists()
    before=state();assert before['SD_errors']=='0';write(HERE/'screen_before.json',before)
    boot=before['boot'];valid={};builds={}
    for name,model in config['models'].items():
        check(config);assert not Path(model['engine']).exists()
        start=time.monotonic();log=HERE/(name+'_build.log')
        command=[TRT,'--onnx='+model['onnx'],'--saveEngine='+model['engine'],'--fp16','--workspace=512',
                 '--buildOnly','--profilingVerbosity=detailed']
        with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        builds[name]={'returncode':result.returncode,'seconds':time.monotonic()-start,'log_sha256':sha(log),
            'state_after':state(),'command':command}
        if result.returncode==0:
            assert Path(model['engine']).stat().st_size>0;model['engine_sha256']=sha(model['engine']);valid[name]=model
            builds[name]['engine_sha256']=model['engine_sha256']
        write(HERE/'build_results.json',builds);print('R164 BUILD',name,builds[name],flush=True)
        assert state()['boot']==boot and state()['SD_errors']=='0'
    config['models']=valid;write(HERE/'runtime_models.json',config)
    for name in valid:raw_checks(config,name)
    models={'original':config['original']};models.update(valid);order=list(models)+list(reversed(list(models)))
    rows=[];(HERE/'gpu_screen').mkdir(exist_ok=False)
    for index,name in enumerate(order):
        start=health(boot);command=[TRT,'--loadEngine='+models[name]['engine'],'--warmUp=500','--duration=6','--iterations=100']
        log=HERE/'gpu_screen'/('%02d_%s.log'%(index,name))
        with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        text=log.read_text();assert result.returncode==0 and 'PASSED' in text,(name,str(log))
        match=re.search(r'GPU Compute Time:.*median = ([\d.]+) ms',text);assert match,name
        end=health(boot);rows.append({'index':index,'mode':name,'GPU_median_ms':float(match.group(1)),
            'before':start,'after':end,'log_sha256':sha(log)})
        write(HERE/'gpu_rows.json',rows);print('R164 GPU SCREEN',name,rows[-1]['GPU_median_ms'],flush=True)
    report={}
    for name in models:
        values=[r['GPU_median_ms'] for r in rows if r['mode']==name];assert len(values)==2
        report[name]={'mean_of_two_medians_ms':mean(values),'repeat_difference_pct':100*abs(values[0]-values[1])/mean(values)}
        assert report[name]['repeat_difference_pct']<=2,('Unstable screen; do not select',name,values)
    base=report['original']['mean_of_two_medians_ms'];eligible=[]
    for name in valid:
        report[name]['GPU_time_reduction_pct']=100*(1-report[name]['mean_of_two_medians_ms']/base)
        if report[name]['GPU_time_reduction_pct']>=8:eligible.append(name)
    selected=[]
    if eligible:
        fastest=min(eligible,key=lambda n:(report[n]['mean_of_two_medians_ms'],n));selected.append(fastest)
        least=min(eligible,key=lambda n:(1-valid[n]['ConvMAC']/config['anchor_statistics']['ConvMAC'],n))
        if least not in selected:selected.append(least)
    check(config)
    selection={'status':'PASS','GPU_only':True,'Nano_pipeline_FPS_measured':False,'models':report,
        'eligible':eligible,'selected_for_training':selected,'negative_screen':not selected,
        'build_failures':[n for n,r in builds.items() if r['returncode']!=0],
        'config_sha256':sha(source),'runtime_models_sha256':sha(HERE/'runtime_models.json'),
        'GPU_rows_sha256':sha(HERE/'gpu_rows.json'),'after':state()}
    write(HERE/'selection.json',selection);print('R164 TRAINING SELECTION',json.dumps(selection),flush=True)
if __name__=='__main__':main()
