"""Bind the displayed Nano comparison to exact verified study summaries."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
VISUAL = Path('C:/Users/Mikuno/.codex/visualizations/2026/10/29/01a18729-d721-76b0-a641-d412ac489177/nano-model-tradeoff.html')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    extra_path = LOGS / 'review155_runtime_accuracy/extra_final_summary.json'
    shape_path = LOGS / 'review155_runtime_accuracy/full80_evidence/runs_absolute/full/score_verified.json'
    silu_path = LOGS / 'review167_fused_silu/final_summary.json'
    extra, shape, silu = read(extra_path), read(shape_path), read(silu_path)
    assert all(r['status'] == 'PASS' for r in (extra, shape, silu))
    cuda = extra['trials']['review162_cuda_pre']['candidates']['gpu_pre']
    channel = extra['trials']['review150_nano_pruning']['candidates']['channel25_KD']
    height = extra['trials']['review160_height352']['candidates']['r352x640']
    expected = {
        'r162-original': (cuda['speed']['all']['original']['mean_FPS'], extra['trials']['review162_cuda_pre']['original_detection']['mAP50']),
        'r162-cuda': (cuda['speed']['all']['candidate']['mean_FPS'], cuda['detection']['mAP50']),
        'r150-channel25': (channel['speed']['all']['candidate']['mean_FPS'], channel['detection']['mAP50']),
        'r160-height352': (height['speed']['all']['candidate']['mean_FPS'], height['detection']['mAP50']),
        'r155-shape': (shape['speed']['all']['modes']['r320x576']['mean_FPS'], shape['detection']['r320x576']['mAP50']),
    }
    for mode in ('precise', 'fast'):
        expected['r167-' + mode] = (silu['speed']['all']['modes'][mode]['mean_FPS'], silu['detection'][mode]['mAP50'])
    sources = [extra_path, shape_path, silu_path]
    final = HERE / 'final_summary.json'
    if final.exists():
        pruning = read(final)
        assert pruning['status'] == 'PASS' and pruning['runs'] == 240
        expected['r172-joint50'] = (pruning['speed']['all']['modes']['joint50']['mean_FPS'], pruning['detection']['joint50']['mAP50'])
        sources.append(final)
    html = VISUAL.read_text(encoding='utf-8')
    payloads = re.findall(r'<script type="application/json" data-viz-chart>\s*(.*?)\s*</script>', html, re.S)
    assert len(payloads) == 1
    chart = json.loads(payloads[0])
    columns = chart['columns']
    rows = [dict(zip(columns, row, strict=True)) for row in chart['rows']]
    assert {r['id'] for r in rows} == set(expected) and len(rows) == len(expected)
    for row in rows:
        assert (row['fps'], row['ap50']) == expected[row['id']], row['id']
        for axis, field in (('x', 'fps'), ('y', 'ap50')):
            low, high = chart['spec'][axis + 'Domain']
            assert low <= row[field] <= high
    assert sum([53204, 3054, 1469, 226, 407]) == 58360
    output = {
        'status': 'PASS', 'points': len(rows), 'source_values_exact': True,
        'unknown_pruning_FPS_plotted': False,
        'rendering_or_interactions_checked': False,
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources, VISUAL, Path(__file__)]},
    }
    (HERE / 'comparison_visual_verified.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('COMPARISON VISUAL EXACT SOURCE VALUES PASS:', len(rows), 'points')


if __name__ == '__main__':
    main()
