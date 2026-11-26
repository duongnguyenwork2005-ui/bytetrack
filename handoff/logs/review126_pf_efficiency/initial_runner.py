"""Same frozen BDD experiment as R125 with the fast numerical implementation."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'review125_pf_bdd'))
import run as bdd
BASE = bdd.HERE
from fast import installed


def write(path, value):
    bdd.write(path, value)


def array(path):
    return np.loadtxt(path, delimiter=',', ndmin=2) if path.stat().st_size else np.empty((0,7))


def replay(name, mode='fast100', seed=0, length=None):
    previous = bdd.installed
    if mode in ('fast100', 'parent100'):
        bdd.installed = lambda: installed(mode == 'fast100')
    try:
        return bdd.replay(name, 'pf100' if mode in ('fast100', 'parent100') else mode, seed, length)
    finally:
        bdd.installed = previous


def preflight():
    assert not (HERE / 'inputs.json').exists()
    old = json.loads((BASE / 'inputs.json').read_text())
    for p, h in old['sha256'].items():
        assert bdd.sha(p) == h, p
    sources = dict(old['sha256'])
    paths = [Path(__file__), HERE / 'fast.py', HERE / 'checks.py', HERE / '00_protocol.md',
             HERE / 'measure.py', HERE / 'numerical.json', BASE / 'final_verification.json']
    paths += [BASE / f'evaluation/pf100-s{s}/raw_fields.json' for s in range(3)]
    for s in range(3):
        paths += [BASE / f'tracking/{bdd.SPLIT}/pf100-s{s}/data/{n}.txt' for n in bdd.videos()]
    sources.update({str(p): bdd.sha(p) for p in paths})
    write(HERE / 'inputs.json', {'sha256': sources, 'timing_videos': old['timing_videos'],
                                 'frames': old['frames'], 'config': old['configs']['pf100']})


def smoke():
    records = []
    names = list(dict.fromkeys(json.loads((HERE / 'inputs.json').read_text())['timing_videos'] + bdd.prior.SMOKE[:2]))
    for seed in range(3):
        for name in names:
            length = min(200, len(bdd.det_cache.open_cache('yolov8n_coco', name, .25, [1,2,3,5,7], 300)))
            mot, _, _ = replay(name, seed=seed, length=length)
            ref = array(BASE / f'tracking/{bdd.SPLIT}/pf100-s{seed}/data/{name}.txt')
            ref = ref[ref[:,0] <= length]
            actual = mot.to_numpy(float)
            assert actual.shape == ref.shape
            assert np.array_equal(actual[:,[0,1,6]], ref[:,[0,1,6]])
            error = float(np.max(abs(actual[:,2:6]-ref[:,2:6]))) if len(actual) else 0.
            assert error < 1e-3, (name, seed, error)
            records.append({'video':name,'seed':seed,'max_box_error':error})
    write(HERE / 'smoke.json', {'status':'PASS','units':records})
    print('SMOKE PASS', len(records), max(x['max_box_error'] for x in records), flush=True)


def full(seed):
    assert (HERE / 'smoke.json').is_file()
    for index, name in enumerate(bdd.videos(), 1):
        folder = HERE / f'tracking/{bdd.SPLIT}/fast100-s{seed}'
        marker = folder / f'meta/{name}.json'
        if marker.is_file():
            continue
        mot, cls, timing = replay(name, seed=seed)
        output = folder / f'data/{name}.txt'
        output.parent.mkdir(parents=True, exist_ok=True)
        mot.to_csv(output, header=False, index=False, float_format='%.10f')
        ref = array(BASE / f'tracking/{bdd.SPLIT}/pf100-s{seed}/data/{name}.txt')
        actual = mot.to_numpy(float)
        assert actual.shape == ref.shape, (name, seed, 'rows')
        assert np.array_equal(actual[:,[0,1,6]], ref[:,[0,1,6]]), (name, seed, 'identity')
        error = float(np.max(abs(actual[:,2:6]-ref[:,2:6]))) if len(actual) else 0.
        assert error < 1e-3, (name, seed, error)
        write(marker, {'frames':len(timing),'mot_sha256':bdd.sha(output),'max_box_error':error})
        if index%20==0:
            print('DONE',seed,index,name,flush=True)
    previous = bdd.HERE
    bdd.HERE = HERE
    try:
        bdd.score('fast100',seed)
    finally:
        bdd.HERE = previous


def verify():
    frozen = json.loads((HERE / 'inputs.json').read_text())
    for p,h in frozen['sha256'].items():
        assert bdd.sha(p)==h,p
    summaries, errors, units = [], [], 0
    for seed in range(3):
        raw = bdd.boot_stats.load_raw_fields(HERE / f'evaluation/fast100-s{seed}/raw_fields.json')
        reference = bdd.boot_stats.load_raw_fields(BASE / f'evaluation/pf100-s{seed}/raw_fields.json')
        assert set(raw)==set(reference)==set(bdd.videos())
        for name in bdd.videos():
            assert raw[name] == reference[name], (name,seed,'raw metrics')
            folder=HERE/f'tracking/{bdd.SPLIT}/fast100-s{seed}'
            marker=json.loads((folder/f'meta/{name}.json').read_text())
            assert marker['mot_sha256']==bdd.sha(folder/f'data/{name}.txt')
            assert marker['frames']==frozen['frames'][name]
            errors.append(marker['max_box_error']);units+=1
        summaries.append(bdd.boot_stats.combine(raw,bdd.videos()))
    write(HERE/'verification.json',{'status':'PASS','units':units,'raw_metric_parity':units,
          'max_box_error':max(errors),'hashes':len(frozen['sha256']),'seeds':summaries})
    print('FULL PARITY PASS',units,max(errors),flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():full(int(sys.argv[1]))
    else:{'preflight':preflight,'smoke':smoke,'verify':verify}[sys.argv[1]]()
