"""Summarize R85 NMS paired Nano runs and verify byte-identical MOT/class output."""
import hashlib
import json
import statistics
from pathlib import Path

R = Path('/home/jetson/nano_run/r63_20261030/r85_nms_hdmi_20261105')
rows = []


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


for pair in range(1, 5):
    for tracker in ('cv', 'ekf'):
        by_variant = {}
        for variant in ('A', 'B'):
            root = R / ('p%d_%s_%s' % (pair, tracker, variant))
            run = root / 'luot'
            result = json.loads((run / 'result.json').read_text())
            meta = json.loads((run / 'live/run_meta.json').read_text())
            guard = json.loads((root / 'giam_sat.json').read_text())
            timing = meta['thoi_gian']['tom_tat_sau_warmup']
            item = {
                'pair': pair, 'tracker': tracker, 'variant': variant,
                'fps': float(result['fps']),
                'p90_ms': float(timing['tre_ms']['p90']),
                'decode_ms': float(timing['decode_ms']['trung_vi']),
                'pre_ms': float(timing['pre_ms']['trung_vi']),
                'infer_ms': float(timing['infer_ms']['trung_vi']),
                'post_ms': float(timing['post_ms']['trung_vi']),
                'n_frames': meta['dem']['n_frame_da_xu_ly'],
                'parity_replay': result['parity_byte'],
                'mot_sha256': sha(run / 'live/data/MVI_39031.txt'),
                'class_sha256': sha(run / 'live/data_cls/MVI_39031.csv'),
                'engine_sha256': result['engine_sha256'],
                'guard': guard,
            }
            by_variant[variant] = item
            rows.append(item)
        a, b = by_variant['A'], by_variant['B']
        assert a['n_frames'] == b['n_frames'] == 300
        assert a['parity_replay'] and b['parity_replay']
        assert a['engine_sha256'] == b['engine_sha256']
        assert a['mot_sha256'] == b['mot_sha256']
        assert a['class_sha256'] == b['class_sha256']

summary = {}
for tracker in ('cv', 'ekf'):
    pairs = []
    for pair in range(1, 5):
        a = next(x for x in rows if x['pair'] == pair and x['tracker'] == tracker and x['variant'] == 'A')
        b = next(x for x in rows if x['pair'] == pair and x['tracker'] == tracker and x['variant'] == 'B')
        pairs.append({
            'pair': pair, 'a_fps': a['fps'], 'b_fps': b['fps'],
            'fps_gain_pct': 100 * (b['fps'] / a['fps'] - 1),
            'p90_change_pct': 100 * (b['p90_ms'] / a['p90_ms'] - 1),
            'decode_delta_ms': b['decode_ms'] - a['decode_ms'],
        })
    summary[tracker] = {
        'pairs': pairs,
        'median_fps_gain_pct': statistics.median(x['fps_gain_pct'] for x in pairs),
        'median_p90_change_pct': statistics.median(x['p90_change_pct'] for x in pairs),
        'median_a_fps': statistics.median(x['a_fps'] for x in pairs),
        'median_b_fps': statistics.median(x['b_fps'] for x in pairs),
    }
out = {'rows': rows, 'summary': summary, 'all_mot_class_parity': True}
(R / 'summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(summary, indent=2))
