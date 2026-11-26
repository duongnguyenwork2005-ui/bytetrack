"""Bind selected checkpoints, actual preprocessing, and the fixed 7190-image data."""
from pathlib import Path
import hashlib,json,shutil,tarfile
HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r169_paired_cuda_resume_20261121/recovered'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    export=read(HERE/'recovered_exports.json');assert export['status']=='PASS'
    for p,h in export['source_sha256'].items():assert sha(Path(p))==h,p
    out=HERE/'recovered_stage';out.mkdir(exist_ok=False);models={}
    for mode,row in export['models'].items():
        folder=out/mode;folder.mkdir();(folder/'parity').mkdir()
        source=Path(row['path']);assert sha(source)==row['onnx_sha256'];shutil.copyfile(source,folder/'model.onnx')
        parity=[]
        for fixture in row['parity']:
            p=Path(fixture['fixture']);assert sha(p)==fixture['fixture_sha256'];shutil.copyfile(p,folder/'parity'/p.name)
            parity.append({'path':REMOTE+'/'+mode+'/parity/'+p.name,'sha256':fixture['fixture_sha256']})
        models[mode]={'onnx':REMOTE+'/'+mode+'/model.onnx','onnx_sha256':row['onnx_sha256'],
            'engine':REMOTE+'/'+mode+'/model.engine','hw':[384,640],'parity':parity,'parameters':row['parameters'],
            'checkpoint_sha256':row['checkpoint_sha256'],'epoch_zero_based':row['epoch_zero_based']}
    for name in ('build_recovered.py','native_helpers.py','00_protocol.md','parent_00_protocol.md','01_profile_scope.md','02_min_width_fix.md','03_pipeline_details.md'):
        shutil.copyfile(HERE/name,out/name)
    r155=HERE.parent/'review155_runtime_accuracy';r162=HERE.parent/'review162_cuda_pre'
    remote155='/home/jetson/nano_run/r155_runtime_accuracy_20261120'
    remote162='/home/jetson/nano_run/r162_cuda_pre_20261120'
    sources={REMOTE+'/'+p.name:sha(p) for p in out.iterdir() if p.is_file()}
    sources[REMOTE+'/native_helpers.py']=sha(HERE/'native_helpers.py')
    for name in ('capture.py','gpu_pre.py','fused_pre.cu','oracle.json','00_protocol.md'):
        sources[remote162+'/'+name]=sha(r162/name)
    for name in ('fused_pre.so',):sources[remote162+'/'+name]=sha(r162/'capture_evidence'/name)
    sources[remote155+'/dataset.json']=sha(r155/'dataset.json')
    original=read(r155/'models.json')['models']['original']
    config={'models':models,'original':original,'source_sha256':sources,'package':read(r155/'models.json')['package'],
        'capture':{'script':remote162+'/capture.py','dataset':remote155+'/dataset.json','archive':remote155+'/dev_images.tar'}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    archive=HERE/'recovered_stage.tar'
    with tarfile.open(archive,'x:') as bundle:
        for p in out.iterdir():bundle.add(p,arcname=p.name)
    (HERE/'recovered_stage_preflight.json').write_text(json.dumps({'status':'PASS','archive_sha256':sha(archive),
        'archive_bytes':archive.stat().st_size,'sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'recovered_exports.json',out/'config.json')}},indent=2)+'\n')
    print('R169 RECOVERED STAGE PASS',archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
