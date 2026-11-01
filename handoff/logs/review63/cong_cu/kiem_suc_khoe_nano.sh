#!/usr/bin/env bash
# R63 - kiem suc khoe Jetson Nano, CHI DOC. Khong sudo (chi thu 'sudo -n', that bai thi ghi lai),
# khong doi nvpmodel/jetson_clocks, khong reboot, khong ghi ngoai <thu_muc_ra>.
# Cach dung: bash kiem_suc_khoe_nano.sh <thu_muc_ra> [giay_tegrastats=15]
set -u
OUT=${1:?can thu_muc_ra}; TS=${2:-15}
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"
LOG="$OUT/kiem_suc_khoe.log"
: > "$LOG"

muc() {  # muc <ten> <lenh...>: chay lenh, ghi stdout/stderr + exit code, khong dung khi loi
  local ten=$1; shift
  {
    echo "===== $ten ($(date -Is))"
    echo "\$ $*"
    "$@" 2>&1
    echo "[exit=$?]"
  } > "$OUT/$ten.txt"
  cat "$OUT/$ten.txt" >> "$LOG"
}

muc 01_l4t            cat /etc/nv_tegra_release
muc 02_os             bash -c 'uname -a; cat /etc/os-release | head -4'
muc 03_goi_trt        bash -c 'dpkg -l | grep -Ei "nvidia-jetpack|nvidia-l4t-core|tensorrt|libnvinfer[0-9]|cuda-runtime|libopencv" | awk "{print \$2, \$3}"'
muc 04_uptime_boot    bash -c 'uptime; cat /proc/sys/kernel/random/boot_id; who -b'
muc 05_nvpmodel       bash -c 'nvpmodel -q 2>&1 || sudo -n nvpmodel -q'
muc 06_jetson_clocks  bash -c 'sudo -n jetson_clocks --show 2>&1 || echo "khong co sudo khong mat khau: bo qua"'
muc 07_xung           bash -c 'for c in /sys/devices/system/cpu/cpu[0-9]*; do echo "$(basename $c) online=$(cat $c/online 2>/dev/null || echo 1) cur=$(cat $c/cpufreq/scaling_cur_freq 2>/dev/null) max=$(cat $c/cpufreq/scaling_max_freq 2>/dev/null) gov=$(cat $c/cpufreq/scaling_governor 2>/dev/null)"; done;
  for f in /sys/devices/gpu.0/devfreq/57000000.gpu /sys/devices/57000000.gpu/devfreq/57000000.gpu; do [ -d "$f" ] && echo "gpu cur=$(cat $f/cur_freq) min=$(cat $f/min_freq) max=$(cat $f/max_freq) gov=$(cat $f/governor)"; done;
  [ -r /sys/kernel/debug/clk/emc/clk_rate ] && echo "emc=$(cat /sys/kernel/debug/clk/emc/clk_rate)" || echo "emc: khong doc duoc (can debugfs)"'
muc 08_nhiet          bash -c 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type)=$(cat $z/temp)"; done'
muc 09_ram_swap       bash -c 'free -m; swapon --show; grep -E "MemAvailable|SwapFree|CmaFree" /proc/meminfo'
muc 10_dia            bash -c 'df -h / ~ 2>/dev/null; lsblk -o NAME,SIZE,MOUNTPOINT,UUID 2>/dev/null | head -20'
muc 11_tien_trinh     bash -c 'ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | grep -Ei "python|trtexec|tegrastats|nano_|update-manager|gnome-software|packagekit|apt|dpkg" | grep -v grep | head -30; echo "--- top cpu"; ps -eo pid,pcpu,pmem,cmd --sort=-pcpu | head -8'
# 12/13: ghi RO trang thai doc log ([doc_log=ok|loi]) de khong lan "doc that bai" voi "khong co dau hieu"
muc 12_kernel_hien_tai bash -c 'T=$(mktemp); if journalctl -k -b 0 --no-pager > "$T" 2>/dev/null && [ -s "$T" ]; then echo "[doc_log=ok nguon=journal dong=$(wc -l < "$T")]"; elif dmesg > "$T" 2>/dev/null && [ -s "$T" ]; then echo "[doc_log=ok nguon=dmesg dong=$(wc -l < "$T")]"; else echo "[doc_log=loi]"; fi; grep -Ei "panic|soft ?lockup|hung_task|blocked for more than|rcu.*stall|i2c.*time|watchdog|throttl|oom|ext4.*error" "$T" | tail -60; rm -f "$T"'
muc 13_kernel_truoc   bash -c 'journalctl --list-boots --no-pager 2>/dev/null | tail -8; T=$(mktemp); if journalctl -k -b -1 --no-pager > "$T" 2>/dev/null && [ -s "$T" ]; then echo "[doc_log=ok dong=$(wc -l < "$T")]"; grep -Ei "panic|soft ?lockup|hung_task|blocked for more than|rcu.*stall|i2c.*time|watchdog|throttl|oom" "$T" | tail -60; else echo "[doc_log=loi: journal khong co boot truoc]"; fi; rm -f "$T"'
muc 14_pstore         bash -c 'ls -la /sys/fs/pstore 2>&1; for f in /sys/fs/pstore/*; do [ -r "$f" ] && { echo "--- $f"; grep -Ei "panic|lockup|hung|rcu|i2c" "$f" | tail -20; }; done; true'
muc 15_dmesg_loi      bash -c 'dmesg -l err,crit,alert,emerg 2>&1 | tail -40'
muc 16_nguon_dien     bash -c 'for d in /sys/bus/i2c/drivers/ina3221x/*/iio:device*; do [ -d "$d" ] && for f in $d/in_power*_input $d/in_voltage*_input $d/in_current*_input $d/rail_name_*; do [ -r "$f" ] && echo "$(basename $f)=$(cat $f)"; done; done; grep -i "under-voltage\|overcurrent" /var/log/syslog 2>/dev/null | tail -10'
muc 17_tegrastats     bash -c "timeout $TS tegrastats --interval 1000 2>&1 | head -$((TS+2))"

# tom tat may doc
python3 - "$OUT" <<'PY'
import json, os, re, sys
out = sys.argv[1]
def doc(ten):
    """Noi dung DAU RA cua muc: bo dong tieu de '=====' va dong lenh '$ ...' (dong lenh chua chinh cac mau
    tim kiem -> lan 01/03 ngay 30/10 tung bao gia 'hung_task', '[doc_log=ok', '--- top cpu')."""
    p = os.path.join(out, ten + ".txt")
    if not os.path.exists(p):
        return ""
    return "\n".join(l for l in open(p, errors="replace").read().splitlines() if not l.startswith(("=====", "$ ")))
t = {}
m = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", doc("01_l4t")); t["l4t"] = "R%s.%s" % m.groups() if m else None
m = re.search(r"libnvinfer8\s+(\S+)", doc("03_goi_trt")); t["tensorrt_pkg"] = m.group(1) if m else None
m = re.search(r"Power Mode: (\S+)", doc("05_nvpmodel")); t["nvpmodel"] = m.group(1) if m else None
m = re.search(r"^([0-9a-f]{8}-[0-9a-f-]{27})$", doc("04_uptime_boot"), re.M); t["boot_id"] = m.group(1) if m else None
t["nhiet_mC"] = dict(re.findall(r"^(\S+)=(\d+)$", doc("08_nhiet"), re.M))
m = re.search(r"MemAvailable:\s+(\d+)", doc("09_ram_swap")); t["mem_available_kB"] = int(m.group(1)) if m else None
k = "\n".join((doc("12_kernel_hien_tai"), doc("14_pstore"), doc("13_kernel_truoc")))
t["dau_hieu_panic"] = sorted(set(x.lower() for x in re.findall(r"kernel panic|soft ?lockup|hung_task|blocked for more than|rcu[^\n]{0,20}stall|i2c[^\n]{0,30}time", k, re.I)))
# trang thai doc tung nguon log: KHONG suy "khong co dau hieu" khi doc that bai
t["doc_log"] = {"hien_tai": "ok" if "[doc_log=ok" in doc("12_kernel_hien_tai") else "loi",
                "boot_truoc": "ok" if "[doc_log=ok" in doc("13_kernel_truoc") else "loi",
                "pstore_so_file": len([l for l in doc("14_pstore").splitlines() if re.match(r"^-r", l)])}
# tach dung dong phan cach (dong lenh "$ ..." cung chua chuoi '--- top cpu' -> lan 01 dem sai)
tt = [l for l in doc("11_tien_trinh").splitlines() if not l.startswith("[exit=")]
tt = tt[:tt.index("--- top cpu")] if "--- top cpu" in tt else tt
t["tien_trinh_lien_quan"] = [re.sub(r"\s+", " ", l.strip())[:120] for l in tt]
# bo chinh phien ssh/bash dang chay script nay; chi tinh tien trinh do that su
t["tien_trinh_do_dang_chay"] = [l for l in t["tien_trinh_lien_quan"]
                                if re.search(r"trtexec|nano_live|nano_detect|bench_engine|chay_mot_luot|chay_so_cap", l)
                                and "kiem_suc_khoe_nano" not in l]
m = re.search(r"Swap:\s+\d+\s+(\d+)", doc("09_ram_swap")); t["swap_dung_mb"] = int(m.group(1)) if m else None
nz = t["nhiet_mC"]
t["nhiet_cpu_gpu_C"] = [int(nz[x]) / 1000.0 for x in ("CPU-therm", "GPU-therm") if x in nz]
t["nhiet_pmic_C"] = int(nz["PMIC-Die"]) / 1000.0 if "PMIC-Die" in nz else None
t["exit_khac_0"] = sorted(f[:-4] for f in os.listdir(out) if f.endswith(".txt") and "[exit=0]" not in doc(f[:-4]))
t["ket_luan"] = "KHONG tu duyet: nguoi/agent doc tung muc theo tieu chi ke hoach muc 4"
json.dump(t, open(os.path.join(out, "tom_tat.json"), "w"), indent=2, ensure_ascii=False)
print(json.dumps(t, indent=2, ensure_ascii=False))
PY
echo "-> $OUT (doc README.md muc 'Tieu chi on dinh' truoc khi do tai dai)"
