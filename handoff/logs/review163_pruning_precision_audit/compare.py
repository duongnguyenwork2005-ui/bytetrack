"""Close the matched-runtime-path audit without treating backend as precision only."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    target=HERE/'comparison.json';assert not target.exists()
    inputs=read(HERE/'inputs.json')
    for p,h in inputs['sha256'].items():assert sha(Path(p))==h,p
    standard_path=HERE.parent/'review145_yolo_recovery/quality.json';standard=read(standard_path)['scores']
    references={};bound=[HERE/'inputs.json',Path(__file__),standard_path]
    for name in ('anchor','control','channel25_KD'):
        cpu_path=(HERE.parent/'review151_resolution_tradeoff/quality/baseline/summary.json' if name=='anchor'
                  else HERE/'quality'/name/'summary.json')
        nano_root=HERE.parent/'review150_nano_pruning/cache'/name
        cpu=read(cpu_path);nano=read(nano_root/'quality.json');verification=read(nano_root/'quality_verification.json')
        assert verification['status']=='PASS'
        for p,h in verification['sha256'].items():assert sha(Path(p))==h,p
        assert cpu['images']==nano['images']==7190 and cpu['input']==nano['engine']['hw']==[384,640]
        assert cpu['checkpoint_sha256']==standard[name]['sha256']
        assert cpu['classes'].keys()==nano['classes'].keys()
        for cls in cpu['classes']:assert cpu['classes'][cls]['GT_boxes']==nano['classes'][cls]['GT_boxes']
        references[name]={'standard_validation':standard[name],
            'FP32_CPU':cpu,'FP16_TensorRT_Nano':nano,
            'Nano_minus_CPU_pp':{k:100*(nano[k]-cpu[k]) for k in ('mAP50','mAP50_95')},
            'Nano_minus_CPU_class_AP50_pp':{c:100*(nano['classes'][c]['AP50']-r['AP50']) for c,r in cpu['classes'].items()}}
        bound.extend((cpu_path,nano_root/'quality.json',nano_root/'quality_verification.json'))
    effects={}
    for runtime in ('FP32_CPU','FP16_TensorRT_Nano'):
        effects[runtime]={parent:{k:100*(references['channel25_KD'][runtime][k]-references[parent][runtime][k])
            for k in ('mAP50','mAP50_95')} for parent in ('anchor','control')}
    result={'status':'PASS','images':7190,'models':references,'pruned_minus_reference_pp':effects,
        'pure_precision_ablation':False,'Nano_FPS_measured':False,
        'limitation':'CPU PyTorch/OpenCV and Nano TensorRT/OpenCV differ; standard validation also differs in geometry/multi-label postprocessing. No causal FP16-only claim.',
        'source_sha256':{str(p):sha(p) for p in bound}}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('PRUNING RUNTIME-PATH AUDIT COMPLETE',json.dumps({n:r['Nano_minus_CPU_pp'] for n,r in references.items()}),flush=True)
    print('PRUNED EFFECT PP',json.dumps(effects),flush=True)
if __name__=='__main__':main()
