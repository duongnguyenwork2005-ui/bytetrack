"""Derive an isolated GPU-preprocessing capture from the frozen R155 path."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'review155_runtime_accuracy'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

source=(BASE/'capture.py').read_text()
def change(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)

change('    import nano_detector\n','    import nano_detector\n    from gpu_pre import GpuTRTEngine as TRTEngine\n')
change("    engine=TRTEngine(model['engine']);", "    oracle=read(Path(__file__).parent/'oracle.json')\n    assert oracle['pixel_status']==oracle['raw_status']=='PASS'\n    for p,h in oracle['source_sha256'].items():assert sha(p)==h,p\n    engine=TRTEngine(model['engine']);")
change("                x=preprocess(letterbox(frame,tuple(model['hw']))[0]);raw=engine(x)",
       "                raw=engine(frame)")
change("        'boot':Path('/proc/sys/kernel/random/boot_id')", 
       "        'preprocessing_sha256':{str(Path(__file__).parent/n):sha(Path(__file__).parent/n) for n in ('gpu_pre.py','fused_pre.cu','fused_pre.so','oracle.json','00_protocol.md')},\n        'boot':Path('/proc/sys/kernel/random/boot_id')")
change("    assert sha(Path(model['engine']))==model['engine_sha256'] and sha(args.archive)==dataset['archive_sha256']\n    write(output/", 
       "    assert sha(Path(model['engine']))==model['engine_sha256'] and sha(args.archive)==dataset['archive_sha256']\n    for p,h in frozen['preprocessing_sha256'].items():assert sha(p)==h,p\n    write(output/")
target=HERE/'capture.py';assert not target.exists();target.write_text(source)
config=json.loads((BASE/'models.json').read_text());model=config['models']['original']
(HERE/'models.json').write_text(json.dumps({'package':config['package'],'models':{'gpu_pre':model}},indent=2)+'\n')
(HERE/'capture_preflight.json').write_text(json.dumps({'status':'PASS','source_capture_sha256':sha(BASE/'capture.py'),
    'changes':['use verified CUDA preprocessor','pass raw BGR frame','bind preprocessing sources and oracle'],
    'sha256':{str(p):sha(p) for p in (Path(__file__),target,HERE/'models.json',HERE/'oracle.json',HERE/'gpu_pre.py',HERE/'fused_pre.cu',HERE/'00_protocol.md')}},indent=2)+'\n')
print('CUDA CAPTURE PREFLIGHT PASS')
