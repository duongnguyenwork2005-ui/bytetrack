"""R62: one measured run + reviewed replay/parity, with all output in a fresh directory.

Run with env nano Python, from a manifest-verified scratch copy of v16.
No changes to detector/tracker arguments beyond source, tracker and display.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def replay_split(out):
    out = Path(out).resolve()
    suffix = hashlib.sha256(out.as_posix().encode('utf-8')).hexdigest()[:16]
    return 'R62-' + out.name + '-' + suffix


def snapshot():
    paths = [Path('/proc/sys/kernel/random/boot_id'), Path('/proc/meminfo'),
             Path('/sys/fs/ext4/mmcblk1p1/errors_count')]
    for pattern in ['/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor',
                    '/sys/class/thermal/thermal_zone*/temp',
                    '/sys/class/devfreq/*/cur_freq']:
        import glob
        paths += [Path(p) for p in glob.glob(pattern)]
    r = {'utc_epoch': time.time()}
    for p in paths:
        try:
            r[str(p)] = p.read_text().strip()
        except OSError as e:
            r[str(p)] = str(e)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracker', choices=['cv', 'ekf'], required=True)
    ap.add_argument('--display', choices=['on', 'off'], required=True)
    ap.add_argument('--frames', type=int, choices=[300, 1470], default=300)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit('Refuse existing output: ' + str(out))
    out.mkdir(parents=True)
    env = dict(os.environ, BYTETRACK_BACKEND='np', DISPLAY=':0',
               XAUTHORITY='/run/user/1000/gdm/Xauthority', PYTHONDONTWRITEBYTECODE='1')
    source = (Path('data/smoke/MVI_39031_f0001-0300').resolve() if a.frames == 300
              else Path('/home/jetson/nano_run/data_full/MVI_39031'))
    before = snapshot()
    write_json(out / 'before.json', before)
    cmd = [sys.executable, 'src/nano_live.py', '--source', str(source), '--fps', '25',
           '--video-name', 'MVI_39031', '--engine', '/home/jetson/nano_run/engine/yolov8n_fp16.engine',
           '--onnx-sha256', digest(Path('models/export/yolov8n_640_op12.onnx')),
           '--tracker', a.tracker, '--out', str(out / 'live'), '--save-cache', 'detections',
           '--cache-root', str(out / 'cache'), '--warmup-frames', '25', '--hang-doi', '4',
           '--tegrastats-ms', '500', '--progress-every', '100']
    if a.display == 'on':
        cmd += ['--hien-thi', '--hien-thi-kich-thuoc', '960', '540']
    write_json(out / 'command.json', {'cwd':os.getcwd(),'argv':cmd})
    with (out / 'live.log').open('w') as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    after = snapshot()
    write_json(out / 'after.json', after)
    if rc != 0:
        raise SystemExit('live failed rc=%d; see %s' % (rc, out / 'live.log'))
    meta = json.loads((out / 'live/run_meta.json').read_text())
    assert meta['hoan_tat'] and meta['dem']['n_frame_da_xu_ly'] == a.frames, meta['dem']
    assert meta['dem']['n_unreadable'] == meta['dem']['n_truncated'] == 0, meta['dem']
    boot = '/proc/sys/kernel/random/boot_id'
    assert before[boot] == after[boot], 'boot changed during measurement'
    split = replay_split(out)
    replay = [sys.executable if x == 'python' else split if x == '<split>'
              else 'replay' if x == '<ten>' else x for x in meta['replay_argv']]
    ref = Path('data/processed/trackers') / split / 'replay'
    if ref.exists():
        raise SystemExit('Refuse existing replay: '+str(ref))
    with (out / 'replay.log').open('w') as log:
        subprocess.run(replay, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / 'parity.log').open('w') as log:
        subprocess.run([sys.executable, 'src/nano_parity.py', '--ref', str(ref), '--test',
                        str(out / 'live'), '--json', str(out / 'parity.json')], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    hashes = {}
    for f in ['data/MVI_39031.txt', 'data_cls/MVI_39031.csv']:
        hashes[f] = digest(out / 'live' / f)
        assert hashes[f] == digest(ref / f), 'live/replay mismatch: '+f
    result = {'tracker':a.tracker,'display':a.display,'frames':a.frames,'live_exit':rc,
              'parity_byte':True,'hashes':hashes,'replay_dir':str(ref.resolve()),
              'fps':meta['thoi_gian']['fps_xu_ly'],'time':meta['thoi_gian'],
              'display_meta':meta.get('hien_thi'),'resources':meta['thiet_bi']}
    write_json(out / 'result.json', result)
    print(json.dumps({k:result[k] for k in ['tracker','display','frames','fps','parity_byte']}),flush=True)


if __name__ == '__main__':
    main()
