"""Bind both predeclared CUDA variants to the unchanged original ONNX weights."""
from pathlib import Path
import hashlib,json,shutil,tarfile
import onnx
HERE=Path(__file__).resolve().parent;REMOTE='/home/jetson/nano_run/r167_fused_silu_20261121'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    transformation=read(HERE/'transform.json');assert transformation['status']=='PASS'
    for p,h in transformation['source_sha256'].items():assert sha(Path(p))==h,p
    for p,h in transformation['parity_sha256'].items():assert sha(Path(p))==h,p
    source=Path(transformation['fused']);assert sha(source)==transformation['fused_sha256']
    out=HERE/'stage';out.mkdir(exist_ok=False)
    shutil.copyfile(HERE/'anchor.onnx',out/'control.onnx');shutil.copyfile(source,out/'precise.onnx')
    graph=onnx.load(source)
    for node in graph.graph.node:
        if node.op_type=='FusedSiluRounded':node.op_type='FusedSiluFast'
    assert sum(n.op_type=='FusedSiluFast' for n in graph.graph.node)==77
    original=onnx.load(HERE/'anchor.onnx')
    assert {p.name:p.SerializeToString() for p in original.graph.initializer}=={p.name:p.SerializeToString() for p in graph.graph.initializer}
    onnx.checker.check_model(graph);onnx.save(graph,out/'fast.onnx')
    (out/'parity').mkdir();parity=[]
    for p in sorted((HERE/'parity').glob('*.npz')):
        shutil.copyfile(p,out/'parity'/p.name);parity.append({'path':REMOTE+'/parity/'+p.name,'sha256':sha(p)})
    names=('native.py','00_protocol.md','01_cuda_namespace_fix.md','02_fast_variant_scope.md','fused_silu_v2.cu','fused_silu_fast.cu','kernel_oracle_v2.py')
    for name in names:shutil.copyfile(HERE/name,out/name)
    for mode in ('precise','fast'):
        report=read(HERE/('kernel_oracle_'+mode+'.json'));assert report['status']=='PASS'
        shutil.copyfile(HERE/('kernel_oracle_'+mode+'.json'),out/('kernel_oracle_'+mode+'.json'))
    records={}
    for mode in ('control','precise','fast'):
        records[mode]={'onnx':REMOTE+'/'+mode+'.onnx','onnx_sha256':sha(out/(mode+'.onnx')),
            'engine':REMOTE+'/'+mode+'.engine','hw':[384,640],'parity':parity,
            'parameters':2590815,'ConvMAC':1894456320}
        if mode!='control':records[mode]['plugin']=REMOTE+'/fused_silu_'+mode+'.so'
    sources={REMOTE+'/'+p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix!='.onnx'}
    for mode in ('precise','fast'):
        for p,h in read(HERE/('kernel_oracle_'+mode+'.json'))['sha256'].items():
            if p.endswith('.so'):sources[p]=h
    parent=HERE.parent/'review164_latency_pruning/build_screen.py'
    sources['/home/jetson/nano_run/r164_latency_pruning_20261121/build_screen.py']=sha(parent)
    common=read(HERE.parent/'review155_runtime_accuracy/models.json')
    config={'models':records,'original':common['models']['original'],'package':common['package'],
        'libraries':[REMOTE+'/fused_silu_'+mode+'.so' for mode in ('precise','fast')],
        'source_sha256':sources}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    archive=HERE/'stage.tar'
    with tarfile.open(archive,'x:') as bundle:
        for p in out.iterdir():bundle.add(p,arcname=p.name)
    (HERE/'stage_preflight.json').write_text(json.dumps({'status':'PASS','archive_sha256':sha(archive),
        'sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'transform.json',out/'config.json')}},indent=2)+'\n')
    print('R167 STAGE PASS',archive.stat().st_size,sha(archive),flush=True)
if __name__=='__main__':main()
