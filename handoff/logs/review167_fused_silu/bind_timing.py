"""Add archive/environment provenance via a new config, preserving the first draft."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;OUT=HERE/'speed_stage';REMOTE='/home/jetson/nano_run/r167_fused_silu_20261121/speed'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=OUT/'timing_models_v5.json';config=json.loads(base.read_text())
    source=(HERE.parent/'review155_runtime_accuracy/collect_extra.py').read_text()
    old="names=['runs_absolute/full'";new="names=['timing_models_v5.json','timing_models_v5_bound.json','timing_v5.py','run_models_v5.py','select_full_bound.py','environment.json','environment_after.json','record_environment.py','01_cuda_namespace_fix.md','02_fast_variant_scope.md','runs_absolute/full'"
    assert source.count(old)==1;source=source.replace(old,new).replace('full80_evidence_20261120.tar.gz','full80_evidence_20261121.tar.gz')
    collector=OUT/'collect_full.py';assert not collector.exists();collector.write_text(source)
    environment=OUT/'record_environment.py';environment.write_bytes((HERE/'record_environment.py').read_bytes())
    selector=OUT/'select_full_bound.py';selector.write_text((OUT/'select_full.py').read_text().replace('timing_models_v5.json','timing_models_v5_bound.json'))
    for p in (collector,environment,selector):config['source_sha256'][REMOTE+'/'+p.name]=sha(p)
    bound=OUT/'timing_models_v5_bound.json';assert not bound.exists();bound.write_text(json.dumps(config,indent=2)+'\n')
    (HERE/'timing_bound_preflight.json').write_text(json.dumps({'status':'PASS','models_unchanged':True,
        'clock_policy_unchanged':True,'sha256':{str(p):sha(p) for p in (Path(__file__),base,bound,collector,environment,selector)}},indent=2)+'\n')
    print('R167 TIMING INPUTS BOUND',flush=True)
if __name__=='__main__':main()
