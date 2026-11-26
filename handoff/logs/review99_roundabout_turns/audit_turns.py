"""Post-hoc RoundaboutHD R64 CLEAR events grouped by image-space GT heading."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_trackeval as rte

SPLIT = 'RoundaboutHD-prefix3000'
CAMERAS = [f'imagesc{i:03}' for i in range(1, 5)]
DETECTORS = ['yolov8n_mixed5', 'yolov8n_coco', 'yolo11n_mixed5']
FILTERS = ['cv', 'ekf', 'ukf', 'pf500-s0', 'pf500-s1', 'pf500-s2']
LEVELS = {'turn25': (25, False), 'turn35': (35, False), 'straight8': (8, True)}


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def labels_for_camera(gt_path, camera, frames=3000, fps=15):
    gt = np.loadtxt(gt_path, delimiter=',')
    assert (gt[:, 6:8] == 1).all()
    raw_ids = np.unique(gt[:, 1]).astype(int)
    compact_ids = np.searchsorted(raw_ids, gt[:, 1])
    shape = (frames, len(raw_ids))
    radius = fps
    cluster_radius = round(fps * 2 / 15)
    leg = radius - cluster_radius
    sustain = (fps * 8 + 14) // 15
    present = np.zeros(shape, bool)
    present[gt[:, 0].astype(int) - 1, compact_ids] = True
    assert present.sum() == len(gt)
    episode_map = {level: np.full(shape, -1, dtype=np.int32) for level in LEVELS}
    episodes = []
    for raw_id, compact_id in zip(raw_ids, range(len(raw_ids))):
        rows = gt[gt[:, 1] == raw_id]
        rows = rows[np.argsort(rows[:, 0])]
        for segment in np.split(rows, np.where(np.diff(rows[:, 0]) != 1)[0] + 1):
            if len(segment) < 2 * radius + sustain:
                continue
            points = segment[:, 2:4] + segment[:, 4:6] * [0.5, 1]
            # Medians around t-leg, t, t+leg (full support t-radius..t+radius).
            windows = np.lib.stride_tricks.sliding_window_view(points, 2 * cluster_radius + 1, axis=0)
            smoothed = np.median(windows, axis=-1)
            before, middle, after = smoothed[:-2 * leg], smoothed[leg:-leg], smoothed[2 * leg:]
            u, v = middle - before, after - middle
            heights = segment[radius:-radius, 5]
            displacement = np.minimum(np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1))
            moving = displacement >= np.maximum(8, 0.25 * heights)
            angles = np.degrees(np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], (u * v).sum(axis=1)))
            for level, (threshold, straight) in LEVELS.items():
                # Require a sustained bend in the same direction or sustained near-straight motion.
                masks = [moving & (abs(angles) <= threshold)] if straight else [
                    moving & (angles >= threshold), moving & (angles <= -threshold)]
                for mask in masks:
                    for start, end in runs(mask):
                        if end - start < sustain:
                            continue
                        first, last = int(segment[start + radius, 0]), int(segment[end + radius - 1, 0])
                        support_end = end + 2 * radius
                        path = points[start:support_end:max(1, round(fps * 4 / 15))].tolist()
                        path.append(points[support_end - 1].tolist())
                        episode_index = len(episodes)
                        assert (episode_map[level][first - 1:last, compact_id] == -1).all()
                        episode_map[level][first - 1:last, compact_id] = episode_index
                        episodes.append({'level': level, 'camera': camera, 'gt_id': int(raw_id),
                                         'first': first, 'last': last,
                                         'peak_deg': round(float(max(abs(angles[start:end]))), 1),
                                         'path': [[round(x, 1), round(y, 1)] for x, y in path]})
    return raw_ids, present, episode_map, episodes


def trace(data, shape):
    """Replay CLEAR at IoU .5, preserving the evaluator's empty-frame handling."""
    last = np.full(shape[1], -1, dtype=int)
    previous = np.full_like(last, -1)
    matched = np.zeros(shape, dtype=bool)
    changed = np.zeros_like(matched)
    for t, (gt, tr) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        if not len(gt) or not len(tr):
            continue
        sim = data['similarity_scores'][t]
        eps = np.finfo(float).eps
        score = 1000 * (tr == previous[gt, None]) + sim
        score[sim < 0.5 - eps] = 0
        i, j = linear_sum_assignment(-score)
        keep = score[i, j] > eps
        g, h = gt[i[keep]], tr[j[keep]]
        matched[t, g] = True
        changed[t, g] = (last[g] != -1) & (last[g] != h)
        last[g] = h
        previous.fill(-1)
        previous[g] = h
    return matched, changed


def main():
    sources = {cam: labels_for_camera(ROOT / 'data/processed' / SPLIT / cam / 'gt/gt.txt', cam)
               for cam in CAMERAS}
    episodes = {cam: sources[cam][3] for cam in CAMERAS}
    rows = []
    checks = 0
    for detector in DETECTORS:
        for condition in ('c25', 'c10'):
            config = f'{detector}/{condition}'
            for method in FILTERS:
                tracker = f'r64-{detector}-{condition}-{method}'
                _, cfg, _ = rte.build_configs(SPLIT, tracker, CAMERAS)
                dataset = rte.dataset_class('mot')(cfg)
                with (ROOT / 'results/trackeval' / SPLIT / tracker / 'per_video_metrics.csv').open() as file:
                    reference = {r['video']: r for r in csv.DictReader(file)}
                for cam, (ids, present, levels, items) in sources.items():
                    raw = dataset.get_raw_seq_data(tracker, cam)
                    data = dataset.get_preprocessed_seq_data(raw, 'pedestrian')
                    assert len(ids) == data['num_gt_ids']
                    for t, (g, compact) in enumerate(zip(raw['gt_ids'], data['gt_ids'])):
                        assert np.array_equal(g, ids[compact]) and present[t].sum() == len(g)
                    matched, changed = trace(data, present.shape)
                    assert (matched & ~present).sum() == 0
                    assert changed.sum() == int(float(reference[cam]['IDSW'])), (tracker, cam, 'IDSW')
                    assert (present & ~matched).sum() == int(float(reference[cam]['FN'])), (tracker, cam, 'FN')
                    checks += 1
                    for level, index in levels.items():
                        mask = index >= 0
                        fn = present & ~matched
                        rows.append({'camera': cam, 'config': config, 'filter': method, 'level': level,
                                     'episodes': sum(e['level'] == level for e in items), 'gt': int(mask.sum()),
                                     'idsw': int((changed & mask).sum()), 'fn': int((fn & mask).sum())})
                        for key, values in [('idsw', changed), ('fn', fn)]:
                            counts = np.bincount(index[values & mask], minlength=len(items))
                            for k in np.unique(index[mask]):
                                items[k].setdefault('scores', {}).setdefault(config, {}).setdefault(method, {})[key] = int(counts[k])
            print(config, checks, flush=True)
    audit = {'parity_checks': checks, 'summary': rows, 'episodes': [e for cam in CAMERAS for e in episodes[cam]]}
    (OUT / 'audit.json').write_text(json.dumps(audit, separators=(',', ':')), encoding='utf-8')


if __name__ == '__main__':
    main()
