from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(HERE.parent/'review115_new_datasets'),str(HERE.parent/'review118_pf_shape')]
import score
import detect
from cartesian import installed

DATA=Path('D:/LABOTARY/test dataset/KITTI-R119')
BASE=HERE.parent/'review119_kitti_full'
CFG=HERE.parent/'review123_pf100/rbpf100.yaml'
score.DATA=DATA
score.TYPES['rbpf100']=CFG


def names():
    a=json.loads((BASE/'inputs.json').read_text())
    return [n for n,r in a['sequences'].items() if r['include']]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    score.write_json(path,value)


def preflight():
    assert not (HERE/'inputs.json').exists()
    assert json.loads((BASE/'verification.json').read_text())['status']=='PASS'
    assert json.loads((HERE.parent/'review123_pf100/final_verification.json').read_text())['status']=='PASS'
    prior=json.loads((BASE/'inputs.json').read_text())
    sources={**prior['sha256'],**json.loads((BASE/'cache_hashes.json').read_text())}
    paths=[Path(__file__),HERE/'00_protocol.md',CFG,ROOT/'src/rbpf_ctrv.py',HERE.parent/'review118_pf_shape/cartesian.py']
    paths += [BASE/f'scores/{n}.json' for n in names()]
    for p in paths:sources[str(p)]=detect.sha256(p)
    for p,h in sources.items():assert detect.sha256(Path(p))==h,p
    write(HERE/'inputs.json',{'sequences':names(),'sha256':sources,'reference':'R119 native KITTI Car','config':score.rt.cfg_to_dict(score.rt.load_tracker_cfg(CFG,{'track_buffer':10}))})


def track(seed):
    score.WORK=DATA/f'evaluation_pf100_s{seed}'
    score.METHODS=[('rbpf100',seed)]
    output=HERE/f'scores/s{seed}';output.mkdir(parents=True,exist_ok=True)
    with installed():
        for name in names():
            assert not (output/f'{name}.json').exists()
            score.score_sequence(name,output)


def combine(records,tag):
    rows={n:r['variants'][tag] for n,r in records.items()}
    metrics={}
    for m in score.METRICS:
        key=m.get_name()
        raw={n:{k:np.asarray(v) if isinstance(v,list) else v for k,v in r['raw'][key].items()} for n,r in rows.items()}
        metrics[key]=m.combine_sequences(raw)
    groups={}
    for r in rows.values():
        for g,c in r['groups'].items():groups.setdefault(g,Counter()).update(c)
    result={'metrics':score.concise(metrics),'groups':groups,'recovery':sum((Counter(r['recovery']) for r in rows.values()),Counter())}
    assert result['metrics']['IDSW']==groups['all']['IDSW'] and result['metrics']['FN']==groups['all']['FN']
    return result


def summary():
    frozen=json.loads((HERE/'inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert detect.sha256(Path(p))==h,p
    references={n:json.loads((BASE/f'scores/{n}.json').read_text()) for n in names()}
    results={m:combine(references,m) for m in ('kf','ekf','ukf')};seeds={};units=0
    for s in range(3):
        records={n:json.loads((HERE/f'scores/s{s}/{n}.json').read_text()) for n in names()}
        tag=f'rbpf100-s{s}'
        for n,r in records.items():
            assert r['frames']==references[n]['frames'] and r['cache_sha256']==references[n]['cache_sha256'] and r['gt_sha256']==references[n]['gt_sha256']
            assert detect.sha256(DATA/f'evaluation_pf100_s{s}/trackers/{tag}/data/{n[-4:]}.txt')==r['variants'][tag]['mot_sha256']
            units+=1
        seeds[str(s)]=combine(records,tag)
    groups=list(seeds['0']['groups'])
    results['rbpf100_mean']={'metrics':{k:float(np.mean([r['metrics'][k] for r in seeds.values()])) for k in seeds['0']['metrics']},'groups':{g:{k:float(np.mean([r['groups'][g][k] for r in seeds.values()])) for k in seeds['0']['groups'][g]} for g in groups}}
    write(HERE/'summary.json',{'results':results,'PF_seeds':seeds,'frames':sum(r['frames'] for r in references.values()),'gt_boxes':sum(r['gt_boxes'] for r in references.values()),'units':units})
    write(HERE/'verification.json',{'status':'PASS','units':units,'source_hashes':len(frozen['sha256']),'baseline_cache_gt_parity':units,'CLEAR_group_parity':units})
    print('SUMMARY',results,flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():track(int(sys.argv[1]))
    else:{'preflight':preflight,'summary':summary}[sys.argv[1]]()
