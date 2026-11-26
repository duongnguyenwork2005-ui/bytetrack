"""Actual-AP-qualified or explicitly diagnostic R155 Nano speed measurements."""
from pathlib import Path
import argparse
import importlib.util
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r164_cuda_model_bench',HERE/'run_models_v4.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

def read(p):return json.loads(Path(p).read_text())

def check_models(config):
    for path,digest in config['source_sha256'].items():assert bench.guard.sha(Path(path))==digest,path
    baseline=read(config['baseline_quality'])
    for name,model in config['models'].items():
        for kind in ('engine','onnx'):
            assert bench.guard.sha(Path(model[kind]))==model[kind+'_sha256']
        quality=read(model['runtime_quality'])
        assert bench.guard.sha(Path(model['runtime_quality']))==model['runtime_quality_sha256']
        assert quality['runtime']=='actual Nano TensorRT' and quality['images']==7190
        assert quality['dataset_sha256']==config['dataset_sha256']==baseline['dataset_sha256']
        for key in ('engine_sha256','onnx_sha256','hw'):assert quality['engine'][key]==model[key]
        assert {c['GT_boxes'] for c in quality['classes'].values()}=={53204,3054,1469,226,407}
        references={'original':baseline}
        if name not in ('original','control'):references['control']=read(config['control_quality'])
        gates={}
        for ref_name,ref in references.items():
            gates.update({ref_name+'_'+k:quality[k]>=ref[k]-.005 for k in ('mAP50','mAP50_95')})
            gates.update({ref_name+'_'+c:quality['classes'][c]['AP50']>=r['AP50']-.02 for c,r in ref['classes'].items()})
        assert model['quality_recommended']==all(gates.values())
        # Raw parity FAIL remains a failure. This separate trial measures an
        # actual-AP/FPS trade-off under the prospective R155 scope amendment.
        assert model['historical_raw_parity'] in ('PASS','FAIL')
        assert config['allow_diagnostic_quality_failure'] is True
    assert bench.guard.sha(Path(config['_config']))==config['_config_sha256']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('engine','pilot','full'))
    parser.add_argument('config',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    config=read(args.config);config['_config']=str(args.config.resolve())
    config['_config_sha256']=bench.guard.sha(args.config)
    bench.check_models=check_models
    if args.action=='engine':bench.engine_benchmark(config,args.output)
    else:bench.pipeline(config,args.action,args.output)

if __name__=='__main__':main()
