"""Package only the frozen five ONNX models and numerical parity fixtures."""
from pathlib import Path
import hashlib,json,shutil,tarfile

HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r164_latency_pruning_20261121'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    export=read(HERE/'export_verification.json');assert export['status']=='PASS'
    for p,h in export['source_sha256'].items():assert sha(Path(p))==h,p
    stage=HERE/'stage';stage.mkdir(exist_ok=False);models={}
    for name,entry in export['models'].items():
        folder=stage/name;folder.mkdir();(folder/'parity').mkdir()
        onnx=Path(entry['path']);assert sha(onnx)==entry['sha256']
        shutil.copyfile(onnx,folder/'model.onnx');parity=[]
        for p,h in entry['parity_sha256'].items():
            source=Path(p);assert sha(source)==h;target=folder/'parity'/source.name;shutil.copyfile(source,target)
            parity.append({'path':REMOTE+'/'+name+'/parity/'+source.name,'sha256':h})
        models[name]={'onnx':REMOTE+'/'+name+'/model.onnx','onnx_sha256':entry['sha256'],
            'engine':REMOTE+'/'+name+'/model.engine','parameters':entry['weights']['parameters'],
            'ConvMAC':entry['weights']['ConvMAC'],'parity':parity,'hw':[384,640]}
    for name in ('00_protocol.md','01_profile_scope.md','02_min_width_fix.md','build_screen.py'):
        shutil.copyfile(HERE/name,stage/name)
    original=read(HERE.parent/'review155_runtime_accuracy/models.json')['models']['original']
    config={'models':models,'original':original,'anchor_statistics':read(HERE/'inputs_v2.json')['anchor'],
        'package':read(HERE.parent/'review155_runtime_accuracy/models.json')['package'],
        'source_sha256':{REMOTE+'/'+p.name:sha(p) for p in stage.iterdir() if p.is_file()}}
    (stage/'screen_config.json').write_text(json.dumps(config,indent=2)+'\n')
    archive=HERE/'stage.tar'
    with tarfile.open(archive,'x:') as bundle:
        for p in stage.iterdir():bundle.add(p,arcname=p.name)
    (HERE/'stage_preflight.json').write_text(json.dumps({'status':'PASS','archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size,'models':list(models),'source_sha256':
            {str(p):sha(p) for p in (Path(__file__),HERE/'export_verification.json',HERE/'inputs_v2.json')}},indent=2)+'\n')
    print('R164 GPU STAGE PASS',archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
