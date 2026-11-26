"""Read-only timing environment snapshot; do not change clock or power policy."""
from pathlib import Path
import json,platform,subprocess,sys
HERE=Path(__file__).resolve().parent
def read(p):
    try:return Path(p).read_text().strip()
    except OSError:return None
def main():
    output=HERE/(sys.argv[1] if len(sys.argv)>1 else 'environment.json');assert not output.exists()
    gpu=Path('/sys/devices/gpu.0/devfreq/57000000.gpu');cpu=Path('/sys/devices/system/cpu/cpu0/cpufreq')
    mode=subprocess.run(['/usr/sbin/nvpmodel','-q'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    value={'boot':read('/proc/sys/kernel/random/boot_id'),'kernel':platform.release(),
        'CPU':{n:read(cpu/n) for n in ('scaling_governor','scaling_min_freq','scaling_max_freq','scaling_cur_freq')},
        'GPU':{n:read(gpu/n) for n in ('governor','min_freq','max_freq','cur_freq')},
        'nvpmodel_returncode':mode.returncode,'nvpmodel':mode.stdout,'meminfo':read('/proc/meminfo'),
        'SD_errors':read('/sys/fs/ext4/mmcblk1p1/errors_count'),'clock_policy_changed_by_this_script':False}
    output.write_text(json.dumps(value,indent=2)+'\n');print('R167 ENVIRONMENT RECORDED',value['boot'],flush=True)
if __name__=='__main__':main()
