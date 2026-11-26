"""Frozen BDD validation transfer of the R123 Cartesian PF100 candidate."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
os.environ['BYTETRACK_BACKEND'] = 'ultralytics'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'settings'))
sys.path[:0] = [str(ROOT / 'src'), str(HERE.parent / 'review118_pf_shape')]
import numpy as np
import pandas as pd
import boot_stats
import det_cache
import run_tracking as rt
import run_trackeval as rte
import r74_pf_bdd as prior
import trackeval
from cartesian import installed

SPLIT = prior.SPLIT
CACHE = prior.CACHE
det_cache.CACHE_ROOT = CACHE.parent
CONFIGS = {
    'pf100': HERE.parent / 'review123_pf100/rbpf100.yaml',
    'cart500': ROOT / 'configs/phase2/trackers/r112/rbpf500.yaml',
    'classic500': ROOT / 'configs/phase2/trackers/r72/motion200.yaml',
    'kf': 'bytetrack.yaml',
    'ekf': ROOT / 'configs/bytetrack_ekf_ctrv.yaml',
    'ukf': ROOT / 'configs/bytetrack_ukf_ctrv.yaml',
}
REFERENCE = ROOT / 'results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/cham'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return prior.digest(Path(path))


def videos():
    return json.loads(prior.MANIFEST.read_text(encoding='utf-8'))['videos']


def config(mode):
    cfg = rt.load_tracker_cfg(CONFIGS[mode], {'track_buffer': 6})
    assert (cfg.track_high_thresh, cfg.track_low_thresh, cfg.new_track_thresh,
            cfg.match_thresh, cfg.fuse_score) == (.25, .1, .25, .8, True)
    return cfg


def tag(mode, seed):
    return mode if seed is None else f'{mode}-s{seed}'


def classic(seed):
    return ROOT / f'results/trackeval/{SPLIT}/r74-bdd-full-motion200-s{seed}/raw_fields.json'


def preflight():
    assert not (HERE / 'inputs.json').exists()
    names = videos()
    verified = prior.preflight(names)
    paths = [Path(__file__), HERE / '00_protocol.md', HERE / 'analyze.py', HERE / 'measure.py',
             prior.MANIFEST, CACHE / 'meta.json', ROOT / 'configs/phase2/r61_bdd/giao_thuc.json',
             HERE.parent / 'review118_pf_shape/cartesian.py',
             HERE.parent / 'review123_pf100/final_verification.json']
    paths += [ROOT / 'src' / p for p in ('run_tracking.py', 'tracker_ctrv.py', 'rbpf_ctrv.py',
              'pf_ctrv.py', 'ekf_ctrv.py', 'ukf_ctrv.py', 'ul_backend.py', 'time_units.py',
              'det_cache.py', 'run_trackeval.py', 'trackeval_ext.py', 'boot_stats.py',
              'lost_association.py', 'lost_recovery.py', 'assoc_log.py', 'r74_pf_bdd.py')]
    paths += [Path(rt.resolve_tracker_yaml(p)) for p in CONFIGS.values()]
    import inspect
    paths += [Path(inspect.getfile(obj)) for obj in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
              trackeval.metrics.Identity, trackeval.datasets.MotChallenge2DBox, rte.dataset_class('mot_crowd'))]
    configs = {mode: rt.cfg_to_dict(config(mode)) for mode in CONFIGS}
    a, b = configs['pf100'], configs['cart500']
    assert {k: (a[k], b[k]) for k in a if a[k] != b[k]} == {'pf_num_particles': (100, 500)}
    baseline = {}
    for seed in range(3):
        raw = boot_stats.load_raw_fields(classic(seed))
        assert set(raw) == set(names)
        baseline[str(seed)] = boot_stats.combine(raw, names)
        recorded = pd.read_csv(classic(seed).parent / 'combined_metrics.csv').iloc[0]
        assert all(abs(baseline[str(seed)][k] - recorded[k]) < 1e-9 for k in boot_stats.METRICS)
        paths += [classic(seed), classic(seed).parent / 'combined_metrics.csv']
        paths += [ROOT / f'data/processed/trackers/{SPLIT}/r74-bdd-full-motion200-s{seed}/data/{n}.txt'
                  for n in prior.SMOKE[:2]]
    for mode in ('cv', 'ekf'):
        folder = REFERENCE / f'r61-yolov8n_coco-c25-{mode}'
        paths += [folder / 'raw_fields.json', folder / 'combined_metrics.csv', folder / 'cham.xong.json']
    frames, workload = {}, []
    for name in names:
        vd = det_cache.open_cache('yolov8n_coco', name, .25, [1, 2, 3, 5, 7], 300)
        assert not vd.unreadable.any() and not vd.truncated.any()
        assert np.array_equal(vd.frame_idx, np.arange(1, len(vd) + 1))
        assert np.allclose(vd.time_s, np.arange(len(vd)) / 5, rtol=0, atol=1e-9)
        frames[name] = len(vd)
        workload.append((float(np.mean([len(row[3]) for row in vd.iter_frames()])), name))
        paths += [CACHE / f'{name}{suffix}' for suffix in ('.npz', '.json', '.xong.json')]
        paths += [ROOT / f'data/processed/{SPLIT}/{name}' / suffix for suffix in ('gt/gt.txt', 'seqinfo.ini')]
    assert sum(frames.values()) == 39973
    workload.sort()
    selected = [workload[int(p * (len(workload) - 1))][1] for p in (.5, .9)]
    assert len(set(selected)) == 2
    import scipy, ultralytics
    write(HERE / 'inputs.json', {'videos': names, 'frames': frames, 'reference_verified': verified,
          'configs': configs, 'baseline_seeds': baseline, 'timing_videos': selected,
          'workload': workload, 'backend': rt.ul_backend.backend_name(),
          'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'ultralytics': ultralytics.__version__},
          'sha256': {str(p): sha(p) for p in paths}})
    print('PREFLIGHT', len(names), sum(frames.values()), selected, flush=True)


class Prefix:
    def __init__(self, parent, count):
        self.parent, self.count, self.orig_shape = parent, count, parent.orig_shape
    def __len__(self):
        return self.count
    def iter_frames(self):
        from itertools import islice
        return islice(self.parent.iter_frames(), self.count)


def replay(name, mode, seed, length=None):
    vd = det_cache.open_cache('yolov8n_coco', name, .25, [1, 2, 3, 5, 7], 300)
    if length is not None:
        vd = Prefix(vd, length)
    with installed() if mode in ('pf100', 'cart500') else nullcontext():
        mot, cls, timing = rt.track_video(vd, config(mode), .2, elapsed_mode='steps', seed=seed, video=name)
    assert len(timing) == len(vd) and timing['n_steps'].eq(1).all()
    assert np.isfinite(mot.to_numpy(float)).all() and not mot.duplicated(['frame', 'id']).any()
    assert (mot[['bb_width', 'bb_height']] > 0).all().all()
    return mot, cls, timing


def smoke():
    assert (HERE / 'inputs.json').is_file()
    rows = []
    for seed in range(3):
        for name in prior.SMOKE[:2]:
            old, _, _ = replay(name, 'classic500', seed)
            path = ROOT / f'data/processed/trackers/{SPLIT}/r74-bdd-full-motion200-s{seed}/data/{name}.txt'
            expected = np.loadtxt(path, delimiter=',', ndmin=2)
            assert np.allclose(old.to_numpy(float), expected, atol=1e-9, rtol=0), (name, seed)
            x, _, _ = replay(name, 'pf100', seed, 64)
            y, _, _ = replay(name, 'pf100', seed, 64)
            assert np.array_equal(x.to_numpy(float), y.to_numpy(float)), (name, seed)
            rows.append({'video': name, 'seed': seed, 'R74_parity': True, 'candidate_repeat': True})
    write(HERE / 'smoke.json', {'status': 'PASS', 'checks': rows})
    print('SMOKE PASS', len(rows), flush=True)


def score(mode, seed):
    tracker = tag(mode, seed)
    ec, dc, mc = rte.build_configs(SPLIT, tracker, videos())
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(HERE / f'error_{tracker}.log'))
    dc.update(TRACKERS_FOLDER=str(HERE / 'tracking'), OUTPUT_FOLDER=str(HERE / 'evaluation'))
    ds = rte.dataset_class('mot_crowd')(dc)
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    result, message = trackeval.Evaluator(ec).evaluate([ds], metrics)
    assert all(v == 'Success' for v in next(iter(message.values())).values())
    target = HERE / 'evaluation' / tracker
    target.mkdir(parents=True, exist_ok=True)
    raw = boot_stats.save_raw_fields(result, tracker, target / 'raw_fields.json')
    assert set(raw) == set(videos())
    combined = boot_stats.combine(raw, videos())
    direct = rte.flatten_results(result, tracker)
    row = direct[direct.video == 'COMBINED_SEQ'].iloc[0]
    assert all(abs(combined[k] - row[k]) < 1e-9 for k in boot_stats.METRICS)
    write(target / 'combined.json', combined)
    print('SCORE', tracker, combined, flush=True)


def full(mode, seed):
    assert json.loads((HERE / 'smoke.json').read_text())['status'] == 'PASS'
    frozen = json.loads((HERE / 'inputs.json').read_text())
    folder = HERE / 'tracking' / SPLIT / tag(mode, seed)
    for index, name in enumerate(videos(), 1):
        marker = folder / f'meta/{name}.json'
        if marker.is_file():
            data = json.loads(marker.read_text())
            assert data['mot_sha256'] == sha(folder / f'data/{name}.txt')
            continue
        mot, cls, timing = replay(name, mode, seed)
        assert len(timing) == frozen['frames'][name]
        output = folder / f'data/{name}.txt'
        output.parent.mkdir(parents=True, exist_ok=True)
        mot.to_csv(output, index=False, header=False, float_format='%.10f')
        cls_path = folder / f'data_cls/{name}.csv'
        cls_path.parent.mkdir(exist_ok=True)
        cls.to_csv(cls_path, index=False)
        timing_path = folder / f'timing/{name}.csv'
        timing_path.parent.mkdir(exist_ok=True)
        timing.to_csv(timing_path, index=False)
        write(marker, {'frames': len(timing), 'seed': seed, 'mode': mode, 'mot_sha256': sha(output),
              'classes_sha256': sha(cls_path), 'timing_sha256': sha(timing_path)})
        if index % 10 == 0:
            print('DONE', tag(mode, seed), index, name, flush=True)
    score(mode, seed)


if __name__ == '__main__':
    command = sys.argv[1]
    if command in ('preflight', 'smoke'):
        {'preflight': preflight, 'smoke': smoke}[command]()
    elif command == 'pf':
        seed = int(sys.argv[2])
        for mode in ('pf100', 'cart500'):
            full(mode, seed)
    elif command == 'reference':
        full(sys.argv[2], None)
