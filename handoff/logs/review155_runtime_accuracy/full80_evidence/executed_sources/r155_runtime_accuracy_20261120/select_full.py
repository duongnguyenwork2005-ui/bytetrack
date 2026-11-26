"""Run on Nano after the complete four-clip pilot; freeze all selections."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    config_path=root/'timing_models_v2.json';config=read(config_path)
    pilot=root/'runs_absolute/pilot';complete=read(pilot/'complete.json');inputs=read(pilot/'inputs.json');rows=read(pilot/'pairs.json')
    assert complete['status']=='PASS' and complete['runs']==24 and complete['models_sha256']==sha(config_path)
    assert inputs['config_path']==str(config_path) and inputs['models']==config['models']
    means={}
    for name in config['models']:
        group=[r for r in rows if r['mode']==name]
        assert len(group)==4 and len({(r['dataset'],r['name']) for r in group})==4
        assert all(r['frames']==r['n_inferred']==150 and not r['skipped_frames'] for r in group)
        means[name]=statistics.mean(r['fps_steady'] for r in group)
    candidates=[m for m in means if m!='original'];rank=lambda m:(-means[m],m)
    eligible=sorted((m for m in candidates if config['models'][m]['quality_recommended'] and means[m]>means['original']),key=rank)
    primary=eligible[0] if eligible else sorted(candidates,key=rank)[0]
    selected=['original',primary]
    diagnostic=sorted((m for m in candidates if not config['models'][m]['quality_recommended'] and means[m]>means['original']),key=rank)
    if diagnostic and diagnostic[0] not in selected:selected.append(diagnostic[0])
    result={'pilot_mean_FPS':means,'selected':selected,'primary_quality_recommended':config['models'][primary]['quality_recommended'],
        'selection_scope':'04_full_tradeoff.md','pilot_complete_sha256':sha(pilot/'complete.json'),
        'quality_FAIL_remains_FAIL':True,'inference_skipping':False}
    destination=root/'selection.json';assert not destination.exists()
    destination.write_text(json.dumps(result,indent=2))
    config['models']={name:config['models'][name] for name in selected}
    for path in (root/'04_full_tradeoff.md',Path(__file__).resolve(),destination):config['source_sha256'][str(path)]=sha(path)
    (root/'full_models.json').write_text(json.dumps(config,indent=2))
    print('FULL_SELECTION',json.dumps(result),flush=True)

if __name__=='__main__':main()
