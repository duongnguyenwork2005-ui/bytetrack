"""Fixed paired trtexec screen. Native Python3.6; no source/engine mutation."""
from pathlib import Path
import hashlib
import json
import os
import re
from statistics import mean
import subprocess
import time

HERE = Path(__file__).resolve().parent
NANO = Path.home() / 'nano_run'
ORIGINAL = NANO / 'r102_interp50_20261111/yolo11n_interp50_fp16.engine'
PRECISE = NANO / 'r167_fused_silu_20261121/precise.engine'
PLUGIN = NANO / 'r167_fused_silu_20261121/fused_silu_precise.so'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + '\n')


def state():
    mem = dict((k, int(v.strip().split()[0])) for k, v in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()))
    thermals = {p.parent.name: int(p.read_text()) / 1000.
                for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    row = {
        'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'SD_errors': Path('/sys/fs/ext4/mmcblk1p1/errors_count').read_text().strip(),
        'swap_MiB': (mem['SwapTotal'] - mem['SwapFree']) / 1024.,
        'MemAvailable_MiB': mem['MemAvailable'] / 1024.,
        'temperatures_C': thermals,
        'cpu_governor': Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip(),
        'cpu_min': Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq').read_text().strip(),
        'cpu_max': Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq').read_text().strip(),
        'gpu_min': Path('/sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq').read_text().strip(),
        'gpu_max': Path('/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq').read_text().strip(),
    }
    assert row['SD_errors'] == '0' and row['swap_MiB'] < 100
    assert row['MemAvailable_MiB'] > 512 and max(thermals.values()) < 75
    return row


def main():
    assert not (HERE / 'inputs.json').exists(), 'Preserve previous screen'
    bindings = json.loads((HERE / 'bindings.json').read_text())
    for p, h in bindings.items():
        assert sha(Path(p)) == h, p
    modes = {
        'original': (ORIGINAL, False), 'original_graph': (ORIGINAL, True),
        'precise': (PRECISE, False), 'precise_graph': (PRECISE, True),
    }
    orders = [list(modes), list(reversed(list(modes))),
              ['precise_graph', 'precise', 'original_graph', 'original']]
    before = state()
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], universal_newlines=True)
    other = [line for line in processes.splitlines() if
             re.search(r'\b(trtexec|timing_v[45]\.py|capture\.py|native_helpers\.py)\b', line)]
    assert not other, other
    write('inputs.json', {'before': before, 'bindings': bindings, 'orders': orders,
                         'source_sha256': sha(Path(__file__)), 'processes': processes,
                         'GPU_only': True})
    rows = []
    for repeat, order in enumerate(orders):
        for mode in order:
            now = state()
            assert now['boot'] == before['boot']
            for key in ('cpu_governor', 'cpu_min', 'cpu_max', 'gpu_min', 'gpu_max'):
                assert now[key] == before[key], key
            engine, graph = modes[mode]
            command = ['/usr/src/tensorrt/bin/trtexec', '--loadEngine=' + str(engine),
                       '--warmUp=500', '--duration=6', '--iterations=100']
            if engine == PRECISE:
                command.append('--plugins=' + str(PLUGIN))
            if graph:
                command.append('--useCudaGraph')
            start = time.time()
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    universal_newlines=True)
            filename = '%02d_%s.log' % (repeat, mode)
            (HERE / filename).write_text(result.stdout)
            assert result.returncode == 0 and 'PASSED' in result.stdout, filename
            timings = {}
            for label in ('GPU Compute Time', 'Enqueue Time', 'End-to-End Host Latency'):
                found = re.search(re.escape(label) + r':.*median = ([\d.]+) ms', result.stdout)
                assert found, (filename, label)
                timings[label] = float(found.group(1))
            warnings = [s for s in result.stdout.splitlines() if '[W]' in s or '[E]' in s]
            rows.append({'mode': mode, 'repeat': repeat, 'command': command,
                         'timings_ms': timings, 'warnings': warnings,
                         'log': filename, 'log_sha256': sha(HERE / filename),
                         'wall_s': time.time() - start, 'before': now, 'after': state()})
            write('rows.json', rows)
            print('GPU_SCREEN', mode, repeat, timings, warnings, flush=True)
    after = state()
    assert after['boot'] == before['boot']
    for p, h in bindings.items():
        assert sha(Path(p)) == h, p
    write('complete.json', {'status': 'PASS', 'runs': len(rows), 'before': before,
                           'after': after, 'GPU_only': True})
    print('R174 GPU SCREEN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
