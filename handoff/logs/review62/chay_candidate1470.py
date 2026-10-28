"""Two long candidate runs after paired parity, with a fresh output and lock."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jetson/nano_run/r62_20261028')


def main():
    lock = (ROOT / 'measurement.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'paired300/runner_exit.json').read_text())['exit_code'] == 0
    assert json.loads((ROOT / 'candidate_verified.json').read_text())['status'] == 'verified_for_benchmark'
    out = ROOT / 'candidate1470_lan01'
    out.mkdir()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    (out / 'runner.json').write_text(json.dumps({'pid': os.getpid(), 'boot_id': boot, 'utc_epoch': time.time()}))
    rc = 1
    try:
        for name, tracker in [('01_cv_on', 'cv'), ('02_ekf_on', 'ekf')]:
            assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
            time.sleep(20)
            print('START %s %s' % (name, tracker), flush=True)
            subprocess.run([sys.executable, '-u', str(ROOT / 'chay_mot_luot.py'),
                            '--tracker', tracker, '--display', 'on', '--frames', '1470',
                            '--out', str(out / name)], cwd=str(ROOT / 'candidate/nano_transfer'), check=True)
            os.sync()
        rc = 0
    finally:
        (out / 'runner_exit.json').write_text(json.dumps({'exit_code': rc, 'utc_epoch': time.time()}))
        os.sync()


if __name__ == '__main__':
    main()
