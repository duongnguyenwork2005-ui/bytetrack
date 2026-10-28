"""Sequential, locked R62 paired measurements; never overwrite or retry a run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jetson/nano_run/r62_20261028')


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def detection_hash(out):
    import numpy as np
    meta = json.loads((out / 'live/run_meta.json').read_text())
    files = list(Path(meta['cache']['cache_dir']).rglob('MVI_39031.npz'))
    assert len(files) == 1, files
    h = hashlib.sha256()
    with np.load(str(files[0]), allow_pickle=False) as z:
        # Fixed order and length-delimited metadata; excludes ZIP timestamps.
        for name in ['frame_idx', 'n_per_frame', 'data']:
            a = z[name]
            header = json.dumps([name, str(a.dtype), list(a.shape)], separators=(',', ':')).encode('ascii')
            h.update(len(header).to_bytes(8, 'little'))
            h.update(header)
            h.update(a.tobytes(order='C'))
    result = {'detections_array_sha256': h.hexdigest(),
              'method': 'r62-paired-v1: length-prefixed JSON [field,dtype,shape], then C bytes',
              'file': str(files[0])}
    write_json(out / 'detections_array_sha256.json', result)
    return result['detections_array_sha256']


def main():
    lock = (ROOT / 'measurement.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (ROOT / 'baseline1470_lan02/exit_code').read_text().strip() == '0'
    assert (ROOT / 'candidate_verified.json').exists()
    group = ROOT / 'paired300'
    group.mkdir()  # Exclusive: an interrupted group must be inspected, not silently restarted.
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    write_json(group / 'runner.json', {'pid': os.getpid(), 'boot_id': boot, 'utc_epoch': time.time()})
    schedule = []
    for tracker in ['cv', 'ekf']:
        for pair in range(1, 5):
            order = ['baseline', 'candidate'] if pair % 2 else ['candidate', 'baseline']
            for variant in order:
                number = len(schedule) + 1
                schedule.append({'group': 'paired300', 'pair': pair, 'tracker': tracker,
                                 'variant': variant, 'expected_frames': 300,
                                 'dir': str(group / ('%02d_%s_p%02d_%s' % (number, tracker, pair, variant)))})
    write_json(group / 'index.json', schedule)
    exit_code = 1
    try:
        for i, entry in enumerate(schedule):
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
            time.sleep(20)
            out = Path(entry['dir'])
            cwd = ROOT / entry['variant'] / 'nano_transfer'
            print('START ' + json.dumps(entry), flush=True)
            subprocess.run([sys.executable, '-u', str(ROOT / 'chay_mot_luot.py'),
                            '--tracker', entry['tracker'], '--display', 'on', '--frames', '300',
                            '--out', str(out)], cwd=str(cwd), check=True)
            detection_hash(out)
            if i % 2:
                other = Path(schedule[i-1]['dir'])
                a = json.loads((out / 'result.json').read_text())
                b = json.loads((other / 'result.json').read_text())
                assert a['hashes'] == b['hashes'], 'MOT/classes mismatch in pair'
                a = json.loads((out / 'detections_array_sha256.json').read_text())
                b = json.loads((other / 'detections_array_sha256.json').read_text())
                assert a['detections_array_sha256'] == b['detections_array_sha256'], 'detection mismatch'
                print('PAIR_PARITY_PASS %s %d' % (entry['tracker'], entry['pair']), flush=True)
            os.sync()
        exit_code = 0
    finally:
        write_json(group / 'runner_exit.json', {'exit_code': exit_code, 'utc_epoch': time.time()})
        os.sync()


if __name__ == '__main__':
    main()
