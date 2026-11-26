"""Read-only re-analysis of locked GT and existing MOT; writes only beside this file."""
import argparse
from configparser import ConfigParser
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/r66_ngoai'),
                str(OUT.parent / 'review99_roundabout_turns')]
from audit_turns import labels_for_camera, trace
import run_trackeval as rte
from sher_metric import combine
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval.metrics import HOTA, CLEAR, Identity

METHODS = ('cv', 'ekf', 'ukf')
INPUTS = {}


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = digest
    return digest


def read_json(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohorts():
    helmet = read_json(ROOT / 'handoff/logs/external_eval/r66/helmet/manifest_test36.json')
    return [
        ('SVMOT', 'SVMOTSYS-test', 'r64', [f'seq_{i}_{"morning" if i < 26 else "evening"}' for i in range(22, 31)]),
        ('RoundaboutHD', 'RoundaboutHD-prefix3000', 'r64', [f'imagesc{i:03}' for i in range(1, 5)]),
        ('HELMET', 'HELMET-test36-motorcycle-v1', 'r66helmet', [x['video_id'] for x in helmet['videos']]),
        ('Sherbrooke', 'Sherbrooke-cars-f2754-3754-roi-v1', 'r66v2', ['sherbrooke']),
    ]


def ridesafe_gt(folder):
    source = read_json(folder / 'source.json')
    xml = folder / source['labels']['file']
    digest = sha(xml)
    assert digest == source['labels']['sha256']
    tree = ET.parse(xml)
    task = tree.find('.//frame_filter').getparent() if hasattr(tree.find('.//frame_filter'), 'getparent') else tree.find('.//frame_filter/..')
    stride = int(task.findtext('frame_filter').split('=')[1])
    start, stop = int(task.findtext('start_frame')), int(task.findtext('stop_frame'))
    indices = list(range(start, stop + 1, stride))
    assert len(indices) == int(task.findtext('size')) == 300 and stride == 5
    mapping = {frame: step + 1 for step, frame in enumerate(indices)}
    rows, occ = [], 0
    for tr in tree.findall('track'):
        if tr.get('label') != 'motorcycle':
            continue
        for box in tr.findall('box'):
            if box.get('outside') != '0':
                continue
            frame = int(box.get('frame'))
            x1, y1, x2, y2 = [float(box.get(k)) for k in ('xtl', 'ytl', 'xbr', 'ybr')]
            assert x2 > x1 and y2 > y1
            rows.append([mapping[frame], int(tr.get('id')), x1, y1, x2-x1, y2-y1, 1, 1, -1])
            occ += box.get('occluded') == '1'
    return source, np.asarray(rows), occ


def gt_phase():
    sha(OUT / '00_protocol.md')
    sha(Path(inspect.getfile(labels_for_camera)))
    clips, all_episodes = [], []
    for name, split, prefix, sequences in cohorts():
        for seq in sequences:
            base = ROOT / 'data/processed' / split / seq
            cfg = ConfigParser()
            assert cfg.read(base / 'seqinfo.ini')
            sha(base / 'seqinfo.ini')
            fps = cfg.getint('Sequence', 'frameRate')
            frames = cfg.getint('Sequence', 'seqLength')
            gt = base / 'gt/gt.txt'
            sha(gt)
            rows = np.loadtxt(gt, delimiter=',', ndmin=2)
            ids, present, groups, episodes = labels_for_camera(gt, seq, frames, fps)
            save_cohort(clips, all_episodes, name, seq, split, prefix, rows, ids, present,
                        groups, episodes, fps, frames, str(gt.relative_to(ROOT)), None)
    for folder in (ROOT / 'tmp/r100_ridesafe_sample', ROOT / 'tmp/r100_ridesafe_recheck'):
        source, rows, occ = ridesafe_gt(folder)
        # Isolated derived copy, not a new benchmark or a change to source data.
        gt = OUT / f'gt_{source["clip"]}.csv'
        np.savetxt(gt, rows, delimiter=',', fmt='%.10g')
        sha(gt)
        ids, present, groups, episodes = labels_for_camera(gt, source['clip'], 300, 5)
        save_cohort(clips, all_episodes, 'RideSafe', source['clip'], None, None, rows, ids,
                    present, groups, episodes, 5, 300, str(gt.relative_to(ROOT)),
                    {'folder': str(folder.relative_to(ROOT)), 'occluded_boxes': occ})
    write_json('gt_cohort.json', {'protocol_sha256': sha(OUT / '00_protocol.md'),
                                'inputs': INPUTS, 'clips': clips, 'episodes': all_episodes})
    write_csv('gt_summary.csv', [{k: v for k, v in c.items() if k not in ('extra',)} for c in clips])
    print('GT-only cohort locked:', len(clips), 'clips;', sum(c['ids'] for c in clips), 'clip-local IDs', flush=True)


def save_cohort(clips, all_episodes, name, seq, split, prefix, rows, ids, present,
                groups, episodes, fps, frames, gt_path, extra):
    assert present.sum() == len(rows)
    assert np.isfinite(rows).all() and (rows[:, 4:6] > 0).all()
    assert (rows[:, 0] >= 1).all() and (rows[:, 0] <= frames).all()
    low, high = np.zeros_like(present), np.zeros_like(present)
    if name == 'SVMOT':
        visibility = rows[:, 8]
        assert ((visibility >= 0) & (visibility <= 1)).all()
        pos = (rows[:, 0].astype(int)-1, np.searchsorted(ids, rows[:, 1]))
        low[pos] = visibility <= .5
        high[pos] = visibility > .5
        assert (low | high).sum() == present.sum() and not (low & high).any()
    masks = {key: value >= 0 for key, value in groups.items()}
    assert not (masks['turn25'] & masks['straight8']).any()
    assert not (masks['turn35'] & ~masks['turn25']).any()
    filename = f'cohort_{name}_{seq}.npz'
    np.savez_compressed(OUT / filename, ids=ids, present=present, **masks,
                        visibility_le50=low, visibility_gt50=high)
    sha(OUT / filename)
    clips.append(dict(dataset=name, sequence=seq, split=split, prefix=prefix, gt=gt_path,
                      cohort=filename, fps=fps, frames=frames, ids=len(ids), boxes=len(rows),
                      turn25=int(masks['turn25'].sum()), turn35=int(masks['turn35'].sum()),
                      straight8=int(masks['straight8'].sum()), visibility_le50=int(low.sum()),
                      visibility_gt50=int(high.sum()), extra=extra))
    all_episodes.extend([dict(dataset=name, **ep) for ep in episodes])


def ridesafe_data(clip, method):
    import pandas as pd
    source, gt_rows, _ = ridesafe_gt(ROOT / clip['extra']['folder'])
    gt_ids = np.unique(gt_rows[:, 1]).astype(int)
    path = ROOT / clip['extra']['folder'] / f'mot_{"kf" if method == "cv" else method}_None.csv'
    sha(path)
    pred = pd.read_csv(path)
    assert np.isfinite(pred.to_numpy()).all() and not pred.duplicated(['frame', 'id']).any()
    assert pred['frame'].between(1, 300).all() and (pred[['bb_width', 'bb_height']] > 0).all().all()
    tr_ids = np.sort(pred['id'].unique())
    data = dict(num_timesteps=300, num_gt_ids=len(gt_ids), num_tracker_ids=len(tr_ids),
                num_gt_dets=len(gt_rows), num_tracker_dets=len(pred), gt_ids=[], tracker_ids=[], similarity_scores=[])
    for frame in range(1, 301):
        g = gt_rows[gt_rows[:, 0] == frame]
        p = pred[pred['frame'] == frame]
        data['gt_ids'].append(np.searchsorted(gt_ids, g[:, 1]))
        data['tracker_ids'].append(np.searchsorted(tr_ids, p['id'].to_numpy()))
        data['similarity_scores'].append(_BaseDataset._calculate_box_ious(
            g[:, 2:6], p[['bb_left', 'bb_top', 'bb_width', 'bb_height']].to_numpy(), box_format='xywh'))
    expected = next(r for r in read_json(ROOT / clip['extra']['folder'] / 'pilot.json')['tracking']
                    if r['tracker'] == ('kf' if method == 'cv' else method))
    h = HOTA().eval_sequence(data)
    c = CLEAR({'PRINT_CONFIG': False}).eval_sequence(data)
    i = Identity({'PRINT_CONFIG': False}).eval_sequence(data)
    measured = dict(HOTA=float(h['HOTA'].mean()), IDF1=float(i['IDF1']), IDSW=int(c['IDSW']),
                    FN=int(c['CLR_FN']), FP=int(c['CLR_FP']))
    for field, old in [('HOTA', 'hota'), ('IDF1', 'idf1'), ('IDSW', 'idsw'), ('FN', 'fn'), ('FP', 'fp')]:
        assert abs(measured[field] - expected[old]) <= 1e-9, (clip['sequence'], method, field)
    return data, gt_ids, measured, dict(hota=h, clear=c, identity=i)


def audit_phase():
    cohort = read_json(OUT / 'gt_cohort.json')
    assert sha(OUT / '00_protocol.md') == cohort['protocol_sha256']
    for path, digest in cohort['inputs'].items():
        assert sha(ROOT / path) == digest, path
    rows, totals, checks, ride_raw = [], [], [], {m: {} for m in METHODS}
    for name, split, prefix, sequences in cohorts():
        for method in METHODS:
            tracker = f'{prefix}-yolov8n_coco-c25-{method}'
            base = ROOT / 'results/trackeval' / split / tracker
            raw = read_json(base / 'raw_fields.json')
            assert set(raw) == set(sequences)
            combined = combine(raw)
            sha(base / 'combined_metrics.csv')
            with (base / 'combined_metrics.csv').open() as stream:
                expected = next(csv.DictReader(stream))
            for field in ('HOTA', 'IDF1', 'IDSW', 'FN', 'FP'):
                assert abs(combined[field] - float(expected[field])) <= 1e-9, (tracker, field)
            totals.append(dict(dataset=name, detector='yolov8n_coco', condition='c25', filter=method,
                               aggregation='COMBINED_raw_fields', **combined))
            _, cfg, _ = rte.build_configs(split, tracker, sequences)
            dataset = rte.dataset_class('mot')(cfg)
            for clip in [c for c in cohort['clips'] if c['dataset'] == name]:
                seq = clip['sequence']
                sha(ROOT / 'data/processed/trackers' / split / tracker / 'data' / f'{seq}.txt')
                original = dataset.get_raw_seq_data(tracker, seq)
                data = dataset.get_preprocessed_seq_data(original, 'pedestrian')
                locked = np.load(OUT / clip['cohort'])
                for actual, compact in zip(original['gt_ids'], data['gt_ids']):
                    assert np.array_equal(actual, locked['ids'][compact])
                collect(clip, method, data, locked, raw[seq]['IDSW'], raw[seq]['CLR_FN'], rows, checks)
            print(name, method, len(sequences), 'CLEAR parity; raw COMBINED parity', flush=True)
    for clip in [c for c in cohort['clips'] if c['dataset'] == 'RideSafe']:
        for method in METHODS:
            data, ids, measured, metrics = ridesafe_data(clip, method)
            locked = np.load(OUT / clip['cohort'])
            assert np.array_equal(ids, locked['ids'])
            collect(clip, method, data, locked, measured['IDSW'], measured['FN'], rows, checks)
            ride_raw[method][clip['sequence']] = metrics
        print('RideSafe', clip['sequence'], '3 score/MOT parity checks', flush=True)
    for method, samples in ride_raw.items():
        h = HOTA().combine_sequences({s: r['hota'] for s, r in samples.items()})
        c = CLEAR({'PRINT_CONFIG': False}).combine_sequences({s: r['clear'] for s, r in samples.items()})
        i = Identity({'PRINT_CONFIG': False}).combine_sequences({s: r['identity'] for s, r in samples.items()})
        totals.append(dict(dataset='RideSafe', detector='yolov8n_coco', condition='c25', filter=method,
                           aggregation='COMBINED_recomputed_MOT', HOTA=float(h['HOTA'].mean()),
                           DetA=float(h['DetA'].mean()), AssA=float(h['AssA'].mean()), IDF1=float(i['IDF1']),
                           MOTA=float(c['MOTA']), IDSW=int(c['IDSW']), FP=int(c['CLR_FP']), FN=int(c['CLR_FN'])))
    summary = []
    for dataset in ('SVMOT', 'RoundaboutHD', 'HELMET', 'Sherbrooke', 'RideSafe'):
        for method in METHODS:
            for group in ('all', 'turn25', 'turn35', 'straight8', 'visibility_le50', 'visibility_gt50'):
                selected = [r for r in rows if (r['dataset'], r['filter'], r['group']) == (dataset, method, group)]
                gt = sum(r['gt_boxes'] for r in selected)
                summary.append(dict(dataset=dataset, filter=method, group=group,
                                    clips=sum(r['gt_boxes'] > 0 for r in selected),
                                    clip_local_ids=sum(r['ids'] for r in selected), gt_boxes=gt,
                                    IDSW=sum(r['IDSW'] for r in selected), FN=sum(r['FN'] for r in selected),
                                    FN_rate=sum(r['FN'] for r in selected)/gt if gt else None))
    write_csv('combined_verified.csv', totals)
    write_csv('scenario_per_clip.csv', rows)
    write_csv('scenario_summary.csv', summary)
    modules = {'labels': labels_for_camera, 'trace': trace, 'combine': combine,
               'HOTA': HOTA, 'CLEAR': CLEAR, 'Identity': Identity, 'dataset': rte.dataset_class('mot')}
    for obj in modules.values():
        sha(Path(inspect.getfile(obj)))
    sha(Path(__file__))
    write_json('verification.json', dict(clip_filter_checks=len(checks), checks=checks,
                                        combined_checks=15, inputs=INPUTS,
                                        note='No tracker/inference rerun; RideSafe rescored from existing MOT.'))
    print('VERIFIED:', len(checks), 'clip/filter CLEAR checks; 15 COMBINED results', flush=True)


def collect(clip, method, data, locked, expected_sw, expected_fn, rows, checks):
    present = locked['present']
    assert data['num_timesteps'] == clip['frames'] and data['num_gt_ids'] == clip['ids']
    assert data['num_gt_dets'] == clip['boxes']
    for t, ids in enumerate(data['gt_ids']):
        assert np.array_equal(np.sort(ids), np.flatnonzero(present[t]))
    matched, changed = trace(data, present.shape)
    missing = present & ~matched
    assert not (matched & ~present).any()
    assert int(changed.sum()) == int(expected_sw) and int(missing.sum()) == int(expected_fn)
    checks.append(dict(dataset=clip['dataset'], sequence=clip['sequence'], filter=method,
                       IDSW=int(changed.sum()), FN=int(missing.sum()), boxes=clip['boxes']))
    for group in ('all', 'turn25', 'turn35', 'straight8', 'visibility_le50', 'visibility_gt50'):
        mask = present if group == 'all' else locked[group]
        assert not (mask & ~present).any()
        rows.append(dict(dataset=clip['dataset'], sequence=clip['sequence'], filter=method, group=group,
                         ids=int(mask.any(axis=0).sum()), gt_boxes=int(mask.sum()),
                         IDSW=int((changed & mask).sum()), FN=int((missing & mask).sum())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('gt', 'audit'))
    args = parser.parse_args()
    (gt_phase if args.phase == 'gt' else audit_phase)()
