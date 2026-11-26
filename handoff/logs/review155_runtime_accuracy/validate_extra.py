"""Read-only source/model/quality preflight for the three staged speed trials."""
from pathlib import Path
import hashlib,importlib.util,json

BASE=Path('/home/jetson/nano_run')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    results=[]
    for folder,version in (('r150_pruning_20261120','v2'),('r160_height352_20261120','v2'),('r162_cuda_pre_20261120','v3')):
        root=BASE/folder;source=root/('timing_'+version+'.py');config_path=root/('timing_models_'+version+'.json')
        spec=importlib.util.spec_from_file_location(folder,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=json.loads(config_path.read_text());config['_config']=str(config_path);config['_config_sha256']=sha(config_path)
        module.check_models(config)
        results.append({'root':str(root),'source_sha256':sha(source),'config_sha256':sha(config_path),'models':list(config['models']),'status':'PASS'})
    root=BASE/'r162_cuda_pre_20261120';package=root/'package'
    record=json.loads((root/'package_verification.json').read_text())
    for name,entry in record['candidate_entries'].items():
        p=package/name
        if 'sha256' in entry:assert sha(p)==entry['sha256'],name
        else:assert p.is_symlink() and str(p.resolve())==entry['symlink_resolved_target'],name
    result={'status':'PASS','trials':results,'candidate_package_entries_verified':len(record['candidate_entries']),
        'GPU_work_started':False,'source_sha256':sha(Path(__file__))}
    target=Path(__file__).parent/'extra_staging_verification.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print('EXTRA_STAGING_VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':main()
