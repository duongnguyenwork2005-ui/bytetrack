"""Full vehicle-only KITTI screen; keep R115 evaluator and tracking settings."""
from collections import Counter
from pathlib import Path
import json
import os
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('D:/LABOTARY/test dataset/KITTI-R119')
os.environ.setdefault('YOLO_CONFIG_DIR', str(DATA / 'settings'))
sys.path.insert(0, str(HERE.parent / 'review115_new_datasets'))
import detect
import score
import numpy as np

OLD = detect.DATA
detect.DATA = score.DATA = DATA
score.WORK = DATA / 'evaluation'
score.METHODS = [(m, None) for m in ('kf', 'ekf', 'ukf')]
OUT = HERE / 'scores'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    score.write_json(path, value)


def audit():
    assert not (HERE / 'inputs.json').exists()
    DATA.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), HERE / '00_protocol.md', Path(detect.__file__), Path(score.__file__),
               detect.WEIGHTS]
    sources += [ROOT / 'src' / x for x in ('run_tracking.py', 'tracker_ctrv.py', 'ekf_ctrv.py',
                                          'ukf_ctrv.py', 'pf_ctrv.py', 'time_units.py', 'ul_backend.py')]
    sources += [Path(score.TYPES[k]) for k in ('ekf', 'ukf')]
    records, image_hashes = {}, {}
    for i in range(21):
        name = f'kitti-{i:04d}'
        label = detect.KITTI / f'data_tracking_label_2/training/label_02/{i:04d}.txt'
        sources.append(label)
        rows = [line.split() for line in label.read_text().splitlines()]
        cars = [r for r in rows if r[2] == 'Car' and float(r[3]) < 1 and int(r[4]) <= 2]
        folder = detect.KITTI / f'data_tracking_image_2/training/image_02/{i:04d}'
        paths = sorted(folder.glob('*.png'))
        assert [p.stem for p in paths] == [f'{j:06d}' for j in range(len(paths))]
        assert max(int(r[0]) for r in rows) + 1 == len(paths)
        records[name] = {'frames': len(paths), 'valid_car_boxes': len(cars),
                         'native_occ2': sum(int(r[4]) == 2 for r in cars),
                         'include': bool(cars), 'previous_R115': name[-4:] in detect.KITTI_SEQUENCES}
        if cars:
            image_hashes.update({str(p): detect.sha256(p) for p in paths})
        print('AUDIT', name, records[name], flush=True)
    write(HERE / 'inputs.json', {'sequences': records, 'image_sha256': image_hashes,
                                'sha256': {str(p): detect.sha256(p) for p in sources}})


def names():
    return [n for n, r in json.loads((HERE / 'inputs.json').read_text())['sequences'].items() if r['include']]


def cache():
    for name in names():
        old = OLD / f'{name}_yolov8n.npz'
        target = DATA / old.name
        if old.is_file() and not target.is_file():
            shutil.copyfile(old, target)
            assert detect.sha256(old) == detect.sha256(target)
    detect.run(names())
    write(HERE / 'cache_hashes.json', {str(DATA / f'{n}_yolov8n.npz'): detect.sha256(DATA / f'{n}_yolov8n.npz') for n in names()})


def track():
    OUT.mkdir(exist_ok=True)
    for name in names():
        path = OUT / f'{name}.json'
        if path.is_file() and set(json.loads(path.read_text())['variants']) == {'kf', 'ekf', 'ukf'}:
            continue
        score.score_sequence(name, OUT)


def combine(records):
    result = {}
    for tag in ('kf', 'ekf', 'ukf'):
        rows = {n: r['variants'][tag] for n, r in records.items()}
        scores = {}
        for metric in score.METRICS:
            key = metric.get_name()
            raw = {n: {k: np.array(v) if isinstance(v, list) else v for k, v in r['raw'][key].items()} for n, r in rows.items()}
            scores[key] = metric.combine_sequences(raw)
        groups = {}
        for r in rows.values():
            for g, values in r['groups'].items():
                groups.setdefault(g, Counter()).update(values)
        result[tag] = {'metrics': score.concise(scores), 'groups': groups,
                       'recovery': sum((Counter(r['recovery']) for r in rows.values()), Counter())}
    return result


def summarize():
    manifest = json.loads((HERE / 'inputs.json').read_text())
    records = {n: json.loads((OUT / f'{n}.json').read_text()) for n in names()}
    results = {}
    for scope, subset in [('all', records), ('new18', {n: r for n, r in records.items() if not manifest['sequences'][n]['previous_R115']})]:
        totals = combine(subset)
        intervals = {}
        rng = np.random.default_rng(119)
        choices = rng.integers(len(subset), size=(10000, len(subset)))
        for method in ('ekf', 'ukf'):
            intervals[method] = {}
            for group in ('all', 'occlusion2', 'turn25', 'turn25_occlusion2'):
                diffs = np.array([r['variants'][method]['groups'][group]['IDSW'] - r['variants']['kf']['groups'][group]['IDSW'] for r in subset.values()])
                intervals[method][group] = np.quantile(diffs[choices].sum(axis=1), [.025, .975]).tolist()
        results[scope] = {'sequences': list(subset), 'frames': sum(r['frames'] for r in subset.values()),
                          'gt_boxes': sum(r['gt_boxes'] for r in subset.values()),
                          'results': totals, 'IDSW_delta_bootstrap95': intervals}
    old = HERE.parent / 'review115_new_datasets/scores'
    parity = []
    for n, r in records.items():
        if manifest['sequences'][n]['previous_R115']:
            prior = json.loads((old / f'{n}.json').read_text())
            for tag in ('kf', 'ekf', 'ukf'):
                assert r['variants'][tag] == prior['variants'][tag], (n, tag)
                parity.append([n, tag])
    for p, h in manifest['sha256'].items():
        assert detect.sha256(Path(p)) == h, p
    frozen_cache = json.loads((HERE / 'cache_hashes.json').read_text())
    for p, h in frozen_cache.items():
        assert detect.sha256(Path(p)) == h, p
    for n, r in records.items():
        for tag, row in r['variants'].items():
            assert detect.sha256(score.WORK / f'trackers/{tag}/data/{n[-4:]}.txt') == row['mot_sha256']
    write(HERE / 'summary.json', results)
    write(HERE / 'verification.json', {'status': 'PASS', 'units': len(records)*3, 'R115_parity': parity,
                                     'sources': len(manifest['sha256']), 'caches': len(frozen_cache)})
    print('SUMMARY', results, flush=True)


if __name__ == '__main__':
    {'audit': audit, 'cache': cache, 'track': track, 'summary': summarize}[sys.argv[1]]()
