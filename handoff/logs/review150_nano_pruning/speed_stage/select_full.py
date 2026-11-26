"""Nano selector for the prospectively defined pruning/height pilot screen."""
from pathlib import Path
import argparse
import hashlib
import json
from statistics import mean

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root.resolve()
    config_path=root/'timing_models_v2.json';config=read(config_path);pilot=root/'runs_absolute/pilot'
    done=read(pilot/'complete.json');inputs=read(pilot/'inputs.json');rows=read(pilot/'pairs.json')
    assert done['status']=='PASS' and done['runs']==4*len(config['models'])
    assert done['models_sha256']==sha(config_path) and inputs['models']==config['models']
    means={name:mean(r['fps_steady'] for r in rows if r['mode']==name) for name in config['models']}
    for name in means:assert len([r for r in rows if r['mode']==name])==4
    candidate='channel25_KD' if config['batch']=='pruning' else 'r352x640'
    selected=['original','control','channel25_KD'] if config['batch']=='pruning' else ['original',candidate]
    proceed=means[candidate]>means['original']
    result={'pilot_mean_FPS':means,'selected':selected if proceed else [],'full_skipped':not proceed,
        'reason':'Candidate faster than original on fixed pilot' if proceed else 'Candidate did not beat original pilot mean',
        'quality_FAIL_remains_FAIL':True,'pilot_complete_sha256':sha(pilot/'complete.json')}
    path=root/'selection.json';assert not path.exists();path.write_text(json.dumps(result,indent=2))
    if proceed:
        config['models']={name:config['models'][name] for name in selected}
        config['source_sha256'][str(path)]=sha(path)
        (root/'full_models.json').write_text(json.dumps(config,indent=2))
    print('EXTRA_FULL_SELECTION',json.dumps(result),flush=True)
if __name__=='__main__':main()
