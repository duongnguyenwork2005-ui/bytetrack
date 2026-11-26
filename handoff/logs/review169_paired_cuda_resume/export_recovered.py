"""Export selected recovery epochs without altering the frozen initialization study."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['YOLO_CONFIG_DIR']=str(HERE/'settings')
spec=importlib.util.spec_from_file_location('r164_deployed_path',HERE.parent/'review151_resolution_tradeoff/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import cv2,numpy as np,onnx,onnxruntime as ort,torch
from ultralytics import YOLO
torch.set_num_threads(2);cv2.setNumThreads(1)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    target=HERE/'recovered_exports.json';assert not target.exists()
    frozen=read(HERE/'training_inputs.json')
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    monitor=[Path(p) for p in (HERE/'data/dev300.txt').read_text().splitlines()]
    images=[monitor[i] for i in (0,33,66,99,132,165,198,231,264,299)]
    records={};folder=ROOT/'models/r169_paired_resume';folder.mkdir(exist_ok=False)
    bound=[Path(__file__),HERE/'training_inputs.json']
    for mode,entry in frozen['models'].items():
        selection=read(HERE/'monitor'/mode/'selection.json');assert selection['status']=='PASS'
        checkpoint=Path(selection['selected']['checkpoint']);assert sha(checkpoint)==selection['selected']['checkpoint_sha256']
        assert read(HERE/'quality'/mode/'verification.json')['status']=='PASS'
        out=folder/mode;out.mkdir();copy=out/'model.pt';shutil.copyfile(checkpoint,copy)
        reference=YOLO(str(copy),task='detect').model.cpu().float().eval()
        assert reference.names==base.NAMES and sum(p.numel() for p in reference.parameters())==entry['parameters']
        model=YOLO(str(copy),task='detect')
        path=Path(model.export(format='onnx',imgsz=(384,640),batch=1,opset=12,dynamic=False,
            simplify=False,half=False,nms=False,device='cpu'))
        graph=onnx.load(path);onnx.checker.check_model(graph)
        assert [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]==[1,3,384,640]
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        parity=out/'parity';parity.mkdir();checks=[]
        for ix,image in enumerate(images):
            frame=cv2.imread(str(image));assert frame is not None
            x=base.det.preprocess(base.det.letterbox(frame,(384,640))[0])
            with torch.inference_mode():expected=reference(torch.from_numpy(x))[0].numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x})[0]
            assert actual.shape==expected.shape==(1,9,5040) and np.isfinite(actual).all()
            assert np.allclose(actual[:,:4],expected[:,:4],rtol=1e-4,atol=.02)
            assert np.allclose(actual[:,4:],expected[:,4:],rtol=1e-4,atol=1e-5)
            fixture=parity/('%02d.npz'%ix);np.savez_compressed(fixture,input=x,expected=expected)
            checks.append({'image':str(image),'image_sha256':sha(image),'fixture':str(fixture),'fixture_sha256':sha(fixture),
                'max_box_error':float(abs(actual[:,:4]-expected[:,:4]).max()),
                'max_score_error':float(abs(actual[:,4:]-expected[:,4:]).max())})
        records[mode]={'path':str(path),'onnx_sha256':sha(path),'checkpoint':str(checkpoint),
            'checkpoint_sha256':sha(checkpoint),'epoch_zero_based':selection['selected']['epoch_zero_based'],
            'parameters':entry['parameters'],'parity':checks,'hw':[384,640]}
        bound.extend((HERE/'monitor'/mode/'selection.json',HERE/'quality'/mode/'summary.json',checkpoint,path))
        (HERE/'recovered_exports_partial.json').write_text(json.dumps(records,indent=2)+'\n')
        print('R164 RECOVERED EXPORT PASS',mode,flush=True)
    for p,h in frozen['sha256'].items():assert sha(p)==h,p
    target.write_text(json.dumps({'status':'PASS','models':records,'source_sha256':{str(p):sha(p) for p in bound},
        'Nano_FPS_measured':False},indent=2)+'\n')
if __name__=='__main__':main()
