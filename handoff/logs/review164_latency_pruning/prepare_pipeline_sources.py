"""Version the existing R162 runner for the same CUDA package on every model."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'review162_cuda_pre'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=(PARENT/'run_models_v3.py').read_text()
    old="assert packages=={HOME/'nano_run/r108_15fps_20261117/overlap_package',HOME/'nano_run/r162_cuda_pre_20261120/package'}"
    new="assert packages=={HOME/'nano_run/r162_cuda_pre_20261120/package'}"
    assert source.count(old)==1
    target=HERE/'run_models_v4.py';assert not target.exists();target.write_text(source.replace(old,new),newline='\n')
    wrapper=(PARENT/'timing_v3.py').read_text()
    wrapper=wrapper.replace("'r155_immutable_model_bench',HERE/'run_models_v3.py'", "'r164_cuda_model_bench',HERE/'run_models_v4.py'")
    old="        gates={k:quality[k]>=baseline[k]-.005 for k in ('mAP50','mAP50_95')}\n        gates.update({c:quality['classes'][c]['AP50']>=r['AP50']-.02 for c,r in baseline['classes'].items()})"
    new="        references={'original':baseline}\n        if name not in ('original','control'):references['control']=read(config['control_quality'])\n        gates={}\n        for ref_name,ref in references.items():\n            gates.update({ref_name+'_'+k:quality[k]>=ref[k]-.005 for k in ('mAP50','mAP50_95')})\n            gates.update({ref_name+'_'+c:quality['classes'][c]['AP50']>=r['AP50']-.02 for c,r in ref['classes'].items()})"
    assert wrapper.count(old)==1;wrapper=wrapper.replace(old,new)
    out=HERE/'timing_v4.py';assert not out.exists();out.write_text(wrapper,newline='\n')
    (HERE/'pipeline_source_derivation.json').write_text(json.dumps({'status':'PASS',
        'changes':['singleton R162 package, including original','original+control quality comparisons'],
        'sha256':{str(p):sha(p) for p in (Path(__file__),PARENT/'run_models_v3.py',PARENT/'timing_v3.py',target,out,HERE/'03_pipeline_details.md')}},indent=2)+'\n')
    print('R164 PIPELINE SOURCES GENERATED',flush=True)
if __name__=='__main__':main()
