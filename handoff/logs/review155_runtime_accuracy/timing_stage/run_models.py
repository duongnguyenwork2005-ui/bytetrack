"""Paired, full-inference Nano measurements using the immutable R109 package."""
from pathlib import Path
from shutil import copytree,disk_usage
from tempfile import TemporaryDirectory
import argparse
import json
import os
import re
import subprocess
import sys

HOME=Path.home();SOURCE=HOME/'nano_run/r105_nano_fps'
sys.path[:0]=[str(SOURCE),str(SOURCE/'drivers')]
import bench_nano as benchmark
import r86_driver as guard
import r101_driver as tools
import r102_driver as live


def write(path,value):Path(path).write_text(json.dumps(value,indent=2))
def read(path):return json.loads(Path(path).read_text())


def check_models(config):
    for name,model in config['models'].items():
        assert guard.sha(Path(model['engine']))==model['engine_sha256']
        assert guard.sha(Path(model['onnx']))==model['onnx_sha256']
        parity=read(Path(config['_config']).parent/(name+'_parity.json'))
        assert parity['status']=='PASS' and parity['engine_sha256']==model['engine_sha256']


def engine_benchmark(config,output):
    check_models(config);output.mkdir(exist_ok=False);rows=[]
    order=list(config['models']);order+=list(reversed(order))
    for index,name in enumerate(order):
        command=['/usr/src/tensorrt/bin/trtexec','--loadEngine='+config['models'][name]['engine'],
                  '--warmUp=500','--duration=6','--iterations=100']
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
        (output/('%02d_%s.log'%(index,name))).write_text(result.stdout)
        assert result.returncode==0 and 'PASSED' in result.stdout
        found=re.search(r'GPU Compute Time:.*median = ([\d.]+) ms',result.stdout);assert found
        rows.append({'index':index,'mode':name,'gpu_median_ms':float(found.group(1))})
        write(output/'rows.json',rows);print('ENGINE BENCH',name,rows[-1],flush=True)
    check_models(config);write(output/'complete.json',{'status':'PASS','rows':rows,'end_to_end_FPS':False})


def pipeline(config,phase,output):
    check_models(config);output.mkdir(exist_ok=False);names=list(config['models'])
    manifest=read(benchmark.SAMPLES/'sample_manifest.json')['rows']
    if phase=='pilot':
        samples=[]
        for dataset in ('bdd','ua'):
            group=[r for r in manifest if r['dataset']==dataset];assert len(group)==40
            samples.extend(group[ix] for ix in (0,10))
    else:
        pilot=read(output.parent/'pilot/complete.json')
        pilot_inputs=read(output.parent/'pilot/inputs.json')
        assert pilot['status']=='PASS'
        assert pilot['models_sha256']==guard.sha(Path(pilot_inputs['config_path']))
        # A dev-qualified selection may contain fewer modes than the pilot.
        # Bind every selected engine/ONNX/shape to the unchanged pilot inputs.
        assert set(config['models']).issubset(pilot_inputs['models'])
        for name,model in config['models'].items():assert model==pilot_inputs['models'][name]
        samples=manifest;assert len(samples)==80
    boot=guard.read('/proc/sys/kernel/random/boot_id');swap=guard.swap_mb()
    errors=guard.read('/sys/fs/ext4/mmcblk1p1/errors_count');assert errors=='0' and swap<100
    package=Path(config['package']);assert package==HOME/'nano_run/r108_15fps_20261117/overlap_package'
    code={str(p):guard.sha(p) for p in [Path(__file__),Path(config['_config']),SOURCE/'bench_nano.py',
          *[(SOURCE/'drivers'/name) for name in ('r86_driver.py','r101_driver.py','r102_driver.py')],
          *sorted((package/'src').glob('*.py'))]}
    write(output/'inputs.json',{'models':config['models'],'config_path':config['_config'],'videos':samples,'mode_order':names,
            'boot':boot,'swap_mb':swap,'errors':errors,'swap_activity':tools.swap_activity(),'code_sha256':code})
    original_run=live.run_one
    def run_overlap(*args):
        live.PACKAGE=package;os.environ['R103_EARLY_CUTOFF']='1'
        return original_run(*args)
    live.run_one=run_overlap;results=[]
    try:
        for index,row in enumerate(samples):
            dataset,name=row['dataset'],row['name'];source=benchmark.SAMPLES/dataset/name
            needed=sum(p.stat().st_size for p in source.iterdir() if p.is_file())
            assert needed<disk_usage('/dev/shm').free/4
            with TemporaryDirectory(prefix='r150_',dir='/dev/shm') as temp:
                copied=Path(temp)/name;copytree(source,copied);images=sorted(copied.glob('*.jpg'))
                assert len(images)==150 and images[0].name==row['first_file'] and images[-1].name==row['last_file']
                hashes={p.name:guard.sha(p) for p in images};write(copied/'r105_images.json',hashes)
                write(output/(dataset+'_'+name+'_images.json'),hashes)
                first=index%len(names);order=names[first:]+names[:first]
                for mode in order:
                    guard.check_guard(boot,swap,errors);model=config['models'][mode]
                    live.ENGINES={'A':guard.ENGINES['B'],'B':Path(model['engine'])}
                    onnx_sha={'A':guard.sha(guard.ONNX['B']),'B':model['onnx_sha256']}
                    root=output/mode;root.mkdir(exist_ok=True)
                    mot,cls,folder,item=benchmark.run_mode(root,'full',row,copied,live,guard,onnx_sha)
                    meta=read(folder/'run_meta.json')
                    assert item['n_inferred']==150 and not item['skipped_frames']
                    assert meta['hello_con']['letterbox_hw']==model['hw']
                    assert meta['cau_hinh']['det_conf']==.25 and not meta['hien_thi']['bat']
                    assert meta['hello_con']['engine_sha256']==model['engine_sha256']
                    if mode=='original':
                        prior=HOME/'nano_run/r109_full80_20261117/results/overlap/runs/full/pipeline'/dataset/name/'B'
                        assert guard.sha(mot)==guard.sha(prior/'data'/(name+'.txt'))
                        assert guard.sha(cls)==guard.sha(prior/'data_cls'/(name+'.csv'))
                        item['R109_MOT_CLS_byte_parity']=True
                    guard.check_guard(boot,swap,errors)
                    item.update(mode=mode,pre_nms_confidence=.25)
                    results.append(item);write(output/'pairs.json',results)
                    print('PIPELINE',phase,dataset,name,mode,item['fps_steady'],flush=True)
        check_models(config)
        for p,h in code.items():assert guard.sha(Path(p))==h,p
        assert len(results)==len(samples)*len(names)
        write(output/'complete.json',{'status':'PASS','runs':len(results),'models_sha256':guard.sha(Path(config['_config']))})
    finally:
        live.run_one=original_run
        for key in benchmark.ENV_KEYS:os.environ.pop(key,None)
        write(output/'after.json',{'boot':guard.read('/proc/sys/kernel/random/boot_id'),'swap_mb':guard.swap_mb(),
                  'swap_activity':tools.swap_activity(),'errors':guard.read('/sys/fs/ext4/mmcblk1p1/errors_count')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('engine','pilot','full'))
    parser.add_argument('config',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    config=read(args.config);config['_config']=str(args.config.resolve())
    if args.action=='engine':engine_benchmark(config,args.output)
    else:pipeline(config,args.action,args.output)
