#!/usr/bin/env bash
# R63 - chay MOT lenh tai tren Nano voi giam sat va cong an toan (GPT review R63 Q7: timeout cho chinh tien trinh tai,
# theo doi nhiet/swap/boot_id/kernel, dung chuoi khi loi). Khong sudo, khong reboot, khong kill tien trinh khac.
# Cach dung: bash chay_co_giam_sat.sh <thu_muc_ra_moi> <timeout_s> -- <lenh> [doi so...]
# Exit: = exit cua lenh neu khac 0 (124/137 = het gio); 10 = lenh exit 0 nhung VI PHAM CONG (nhiet CPU/GPU > 80 C,
# swap tang > 50 MB, dong kernel nghi van moi, boot_id doi); 2 = sai cach dung/thu muc khong rong.
set -u
OUT=${1:?can thu_muc_ra}; TMO=${2:?can timeout_s}; shift 2
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "thieu lenh"; exit 2; }
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT khong rong"; exit 2; fi
mkdir -p "$OUT"

nhiet() { for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type)=$(cat $z/temp)"; done | tr '\n' ' '; }
swap_mb() { free -m | awk '/^Swap:/ {print $3}'; }
{
  echo "lenh=$*"; echo "timeout_s=$TMO"; echo "bat_dau=$(date -Is)"
  echo "boot_id_truoc=$(cat /proc/sys/kernel/random/boot_id)"; echo "nhiet_truoc=$(nhiet)"
  echo "swap_mb_truoc=$(swap_mb)"; echo "mem_available_kb_truoc=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
} > "$OUT/giam_sat.env"
N_DMESG=$(dmesg 2>/dev/null | wc -l)
# su co 30/10 17:10: tegrastats ghi qua dem -> log rong khi may sap. stdbuf -oL: xa tung dong; vong sync: day xuong the
stdbuf -oL tegrastats --interval 500 > "$OUT/tegrastats.log" 2>&1 &
TEG=$!
( while sleep 2; do sync; done ) &
SYN=$!
timeout --kill-after=15 "$TMO" "$@" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
RC=$?
kill $TEG $SYN 2>/dev/null; wait $TEG $SYN 2>/dev/null
dmesg 2>/dev/null | tail -n +"$((N_DMESG + 1))" > "$OUT/dmesg_moi.txt"
{
  echo "ket_thuc=$(date -Is)"; echo "exit_lenh=$RC"
  echo "boot_id_sau=$(cat /proc/sys/kernel/random/boot_id)"; echo "nhiet_sau=$(nhiet)"; echo "swap_mb_sau=$(swap_mb)"
} >> "$OUT/giam_sat.env"

python3 - "$OUT" "$RC" <<'PY'
import json, os, re, sys
out, rc = sys.argv[1], int(sys.argv[2])
env = dict(l.rstrip("\n").split("=", 1) for l in open(os.path.join(out, "giam_sat.env")) if "=" in l)
cpu, gpu = [], []
for l in open(os.path.join(out, "tegrastats.log"), errors="replace"):
    m = re.search(r"CPU@([\d.]+)C", l); cpu += [float(m.group(1))] if m else []
    m = re.search(r"GPU@([\d.]+)C", l); gpu += [float(m.group(1))] if m else []
dm = open(os.path.join(out, "dmesg_moi.txt"), errors="replace").read()
nghi = re.findall(r"[^\n]*(?:panic|soft ?lockup|hung_task|blocked for more than|rcu_sched|rcu_preempt|stall|i2c[^\n]*time|oom|out of memory|throttl|ext4[^\n]*error)[^\n]*", dm, re.I)
swap_tang = int(env.get("swap_mb_sau", 0) or 0) - int(env.get("swap_mb_truoc", 0) or 0)
vi_pham = []
if cpu and max(cpu) > 80: vi_pham.append("nhiet CPU max %.1f > 80" % max(cpu))
if gpu and max(gpu) > 80: vi_pham.append("nhiet GPU max %.1f > 80" % max(gpu))
if not cpu: vi_pham.append("khong doc duoc tegrastats")
if swap_tang > 50: vi_pham.append("swap tang %d MB" % swap_tang)
if nghi: vi_pham.append("dong kernel nghi van moi: %d" % len(nghi))
if env.get("boot_id_truoc") != env.get("boot_id_sau"): vi_pham.append("boot_id doi")
kq = {"exit_lenh": rc, "het_gio": rc in (124, 137), "cpu_max_C": max(cpu) if cpu else None,
      "gpu_max_C": max(gpu) if gpu else None, "n_mau_tegrastats": len(cpu), "swap_tang_mb": swap_tang,
      "kernel_nghi_van": nghi[:20], "vi_pham_cong": vi_pham, "env": env}
json.dump(kq, open(os.path.join(out, "giam_sat.json"), "w"), indent=2, ensure_ascii=False)
print("[giam_sat] exit=%d cpu_max=%s gpu_max=%s swap_tang=%dMB vi_pham=%s" % (rc, kq["cpu_max_C"], kq["gpu_max_C"], swap_tang, vi_pham))
sys.exit(rc if rc != 0 else (10 if vi_pham else 0))
PY
