"""Nano Python3.6: build recovered graphs, retain parity failures, capture real AP inputs."""
from pathlib import Path
import importlib.util,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r164_frozen_build',HERE.parent/'build_screen.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.HERE=HERE
def main():
    config=base.read(HERE/'config.json');base.check(config)
    assert not (HERE/'before.json').exists();base.write(HERE/'before.json',base.state())
    builds={};valid={}
    for name,model in config['models'].items():
        base.check(config);assert not Path(model['engine']).exists()
        command=[base.TRT,'--onnx='+model['onnx'],'--saveEngine='+model['engine'],'--fp16','--workspace=512',
            '--buildOnly','--profilingVerbosity=detailed']
        log=HERE/(name+'_build.log');start=time.monotonic()
        with log.open('x') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record={'returncode':result.returncode,'seconds':time.monotonic()-start,'log_sha256':base.sha(log),
            'command':command,'after':base.state()}
        if result.returncode==0:
            model['engine_sha256']=base.sha(Path(model['engine']));record['engine_sha256']=model['engine_sha256'];valid[name]=model
        builds[name]=record;base.write(HERE/'build_results.json',builds)
        print('R164 RECOVERED BUILD',name,result.returncode,record['seconds'],flush=True)
    config['models']=valid;base.check(config);base.write(HERE/'runtime_models.json',config)
    for name in valid:base.raw_checks(config,name)
    for name in valid:
        capture=config['capture'];env=dict(os.environ,LD_PRELOAD='/usr/lib/aarch64-linux-gnu/libgomp.so.1',PYTHONUNBUFFERED='1')
        command=['python3',capture['script'],str(HERE/'runtime_models.json'),name,capture['dataset'],capture['archive'],str(HERE/'cache')]
        log=HERE/(name+'_capture.log')
        with log.open('x') as stream:result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert result.returncode==0,('capture failed',name,str(log))
        complete=base.read(HERE/'cache'/name/'complete.json');assert complete['status']=='PASS' and complete['images']==7190
        base.check(config);print('R164 ACTUAL CAPTURE COMPLETE',name,flush=True)
    base.write(HERE/'complete.json',{'status':'PASS','built':list(valid),
        'build_failures':[n for n,r in builds.items() if r['returncode']!=0],
        'after':base.state(),'Nano_pipeline_FPS_measured':False})
if __name__=='__main__':main()
