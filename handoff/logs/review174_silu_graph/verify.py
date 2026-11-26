"""Reparse all twelve logs and enforce the prospective R174 screen gates."""
from pathlib import Path
import hashlib
import json
import math
import re
from statistics import mean

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs, complete, rows = [read(EVIDENCE / p) for p in ('inputs.json', 'complete.json', 'rows.json')]
    bound = read(HERE / 'bindings.json')
    assert read(EVIDENCE / 'bindings.json') == inputs['bindings'] == bound
    assert complete['status'] == 'PASS' and complete['runs'] == len(rows) == 12
    assert inputs['source_sha256'] == sha(HERE / 'screen.py') == sha(EVIDENCE / 'screen.py')
    root = '/home/jetson/nano_run/r174_silu_graph_20261126/'
    for path, digest in bound.items():
        if path.startswith(root):
            assert sha(EVIDENCE / path.removeprefix(root)) == digest, path
    assert sha(EVIDENCE / 'sampleInference.cpp') == bound['/usr/src/tensorrt/samples/common/sampleInference.cpp']
    assert sha(EVIDENCE / 'sampleDevice.h') == bound['/usr/src/tensorrt/samples/common/sampleDevice.h']
    states = [inputs['before'], complete['before'], complete['after']]
    expected_order = [(repeat, mode) for repeat, group in enumerate(inputs['orders']) for mode in group]
    assert [(r['repeat'], r['mode']) for r in rows] == expected_order
    by_mode = {}
    files = [HERE / '00_protocol.md', HERE / 'screen.py', HERE / 'bindings.json', Path(__file__),
             *[EVIDENCE / p for p in ('inputs.json', 'complete.json', 'rows.json', 'bindings.json', 'sampleInference.cpp', 'sampleDevice.h')]]
    for row in rows:
        path = EVIDENCE / row['log']
        assert sha(path) == row['log_sha256'], path
        text = path.read_text(encoding='utf-8')
        assert 'PASSED TensorRT.trtexec' in text and 'TensorRT v8201' in text
        command = row['command']
        assert command[0] == '/usr/src/tensorrt/bin/trtexec'
        assert {'--warmUp=500', '--duration=6', '--iterations=100'} <= set(command)
        graph = row['mode'].endswith('_graph')
        assert ('--useCudaGraph' in command) == graph
        assert bool(re.search(r'CUDA Graph:\s+Enabled', text)) == graph
        warnings = [s for s in text.splitlines() if '[W]' in s or '[E]' in s]
        assert warnings == row['warnings'] == [], (path, warnings)
        assert not re.search(r'graph.*(?:ignored|failed)|without using CUDA graph', text, re.I)
        engine = next(s.split('=', 1)[1] for s in command if s.startswith('--loadEngine='))
        original = row['mode'].startswith('original')
        expected_engine = ('/home/jetson/nano_run/r102_interp50_20261111/yolo11n_interp50_fp16.engine' if original
                           else '/home/jetson/nano_run/r167_fused_silu_20261121/precise.engine')
        assert engine == expected_engine and ('--plugins=' in ' '.join(command)) == (not original)
        for key, value in row['timings_ms'].items():
            found = re.search(re.escape(key) + r':.*median = ([\d.]+) ms', text)
            assert found and float(found.group(1)) == value and math.isfinite(value) and value > 0
        by_mode.setdefault(row['mode'], []).append(row)
        states.extend((row['before'], row['after']))
        files.append(path)
    first = states[0]
    for state in states:
        assert state['boot'] == first['boot'] and state['SD_errors'] == '0'
        assert state['swap_MiB'] < 100 and state['MemAvailable_MiB'] > 512
        assert max(state['temperatures_C'].values()) < 75
        for key in ('cpu_governor', 'cpu_min', 'cpu_max', 'gpu_min', 'gpu_max'):
            assert state[key] == first[key], key
    statistics = {}
    for mode, group in by_mode.items():
        assert len(group) == 3
        medians = [r['timings_ms']['GPU Compute Time'] for r in group]
        statistics[mode] = {
            'mean_of_GPU_medians_ms': mean(medians), 'individual_GPU_medians_ms': medians,
            'repeat_spread_pct': 100 * (max(medians) / min(medians) - 1),
            'mean_of_enqueue_medians_ms': mean(r['timings_ms']['Enqueue Time'] for r in group),
            'mean_of_host_medians_ms': mean(r['timings_ms']['End-to-End Host Latency'] for r in group),
        }
    gates = {}
    for base in ('original', 'precise'):
        old, new = statistics[base], statistics[base + '_graph']
        reduction = 100 * (1 - new['mean_of_GPU_medians_ms'] / old['mean_of_GPU_medians_ms'])
        stable = max(old['repeat_spread_pct'], new['repeat_spread_pct']) <= 2
        gates[base] = {
            'GPU_time_reduction_pct': reduction,
            'enqueue_time_reduction_pct': 100 * (1 - new['mean_of_enqueue_medians_ms'] / old['mean_of_enqueue_medians_ms']),
            'stable_within_2pct': stable,
            'at_least_5pct_GPU_reduction': reduction >= 5,
            'eligible_for_pipeline_implementation': stable and reduction >= 5,
        }
    summary = {
        'status': 'PASS', 'meaning': 'All twelve logs and guards verified, not an FPS/accuracy result.',
        'runs': 12, 'statistics': statistics, 'gates': gates,
        'boot': first['boot'], 'max_temperature_C': max(max(s['temperatures_C'].values()) for s in states),
        'max_swap_MiB': max(s['swap_MiB'] for s in states),
        'graph_evidence': 'Enabled logs, no fallback/errors, packaged source path, and reduced enqueue time; no Nsight trace.',
        'full_pipeline_FPS_or_detection_quality_measured': False,
        'default_or_launcher_changed': False,
        'collection_v1_failed_v2_preserves_same_twelve_runs': True,
        'sha256': {str(p): sha(p) for p in files},
    }
    (HERE / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status', 'statistics', 'gates', 'max_temperature_C', 'max_swap_MiB')}, indent=2))


if __name__ == '__main__':
    main()
