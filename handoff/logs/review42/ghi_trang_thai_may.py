# -*- coding: utf-8 -*-
"""Vong 42: ghi trang thai (HEAD, working tree, moi truong, nguon dien, RAM, dia, GPU, cai dat nguon -- CHI DOC).
    python v42_trang_thai.py <file log> <tieu de>"""
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\LABOTARY\bytetrack")
RC = Path(r"D:\LABOTARY\review_codex")


def chay(args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    return (r.stdout + r.stderr).strip()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    out, tieu_de = Path(sys.argv[1]), sys.argv[2]
    L = [f"# {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')} -- {tieu_de}"]
    g = lambda *a: chay(["git", "-C", str(REPO), *a])  # noqa: E731
    L += [f"nhanh: {g('rev-parse', '--abbrev-ref', 'HEAD')}", f"HEAD: {g('rev-parse', 'HEAD')}",
          f"commit sau 6c8e37a: {g('rev-list', '--count', '6c8e37a..HEAD')}"]
    try:
        rb = json.loads(g("show", "HEAD:handoff/review_base.json"))
        L.append(f"base review trong HEAD: {rb['base_commit']} vong {rb['vong']} | khai bao khong ban giao: "
                 f"{len(rb.get('khai_bao_khong_ban_giao', []))}")
    except (json.JSONDecodeError, KeyError) as e:
        L.append(f"base review trong HEAD: khong doc duoc ({e})")
    wb = json.loads((REPO / "handoff/review_base.json").read_text(encoding="utf-8"))
    L.append(f"base review working tree: {wb['base_commit']} vong {wb['vong']} | khai bao: "
             f"{len(wb.get('khai_bao_khong_ban_giao', []))} | lich_su {len(wb.get('lich_su', []))}")
    L.append("--- git status --short (bo qua results/ outputs/):")
    L += [x for x in g("status", "--short").splitlines() if not x[3:].startswith(("results/", "outputs/"))]
    L.append("--- moi truong:")
    import numpy
    import psutil
    import torch
    import ultralytics
    import ultralytics.engine.trainer as ut
    try:
        import cv2
        cvv = cv2.__version__
    except ImportError:
        cvv = None
    L.append(f"python {sys.version.split()[0]} | {platform.platform()} | ultralytics {ultralytics.__version__} | torch "
             f"{torch.__version__} | cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} | numpy "
             f"{numpy.__version__} | psutil {psutil.__version__} | opencv {cvv}")
    L.append(f"python: {sys.executable}")
    L.append(f"ultralytics engine/trainer.py sha256 {sha(Path(ut.__file__))}")
    L.append(f"torch.cuda.is_available {torch.cuda.is_available()}"
             + (f" | {torch.cuda.get_device_name(0)} | VRAM {torch.cuda.get_device_properties(0).total_memory / 2 ** 20:.0f}"
                f" MiB" if torch.cuda.is_available() else ""))
    L.append("--- nguon dien:")
    b = psutil.sensors_battery()
    L.append(f"pin {b.percent}% | cam sac (power_plugged) {b.power_plugged} | secsleft {b.secsleft}" if b else "khong co pin")
    L.append(f"goi nguon: {chay(['powercfg', '/getactivescheme'])}")
    for sub, st in (("SUB_SLEEP", "STANDBYIDLE"), ("SUB_SLEEP", "HIBERNATEIDLE")):
        q = chay(["powercfg", "/query", "SCHEME_CURRENT", sub, st])
        ac = [x.split(":")[-1].strip() for x in q.splitlines() if "Current AC Power Setting Index" in x]
        dc = [x.split(":")[-1].strip() for x in q.splitlines() if "Current DC Power Setting Index" in x]
        L.append(f"{st}: AC {ac} DC {dc} (giay; 0 = khong bao gio)")
    L.append("trang thai ngu kha dung (powercfg /a): " + " | ".join(
        x.strip() for x in chay(["powercfg", "/a"]).splitlines() if x.strip().startswith(("Standby", "Hibernate",
                                                                                           "Fast", "Hybrid"))))
    ps = ("$a = Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired';"
          "$b = Test-Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending';"
          "\"WindowsUpdate RebootRequired=$a; CBS RebootPending=$b\"")
    L.append("khoi dong lai dang cho (chi doc registry): " + chay(["powershell", "-NoProfile", "-Command", ps]))
    L.append("--- RAM / dia:")
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    L.append(f"RAM tong {vm.total / 2 ** 30:.2f} GiB, con trong {vm.available / 2 ** 30:.2f} GiB ({vm.percent}% dung) | "
             f"swap/pagefile tong {sw.total / 2 ** 30:.2f} GiB dung {sw.used / 2 ** 30:.2f} GiB")
    for o in ("C:\\", "D:\\"):
        du = shutil.disk_usage(o)
        L.append(f"dia {o} trong {du.free / 2 ** 30:.1f} GiB / {du.total / 2 ** 30:.1f} GiB")
    L.append("--- GPU (nvidia-smi):")
    L.append(chay(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,"
                   "power.draw,pstate,utilization.gpu", "--format=csv"]))
    L.append(chay(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]))
    L.append("--- tien trinh python dang chay (tranh trung luot):")
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if (p.info["name"] or "").lower().startswith("python"):
                cmd = " ".join(p.info["cmdline"] or [])
                L.append(f"  {p.info['pid']} {dt.datetime.fromtimestamp(p.info['create_time']).strftime('%H:%M:%S')} "
                         f"{cmd[:200]}")
        except psutil.Error:
            pass
    L.append("--- sha256 ma / cau hinh lien quan:")
    for f in ("src/tiep_tuc_huan_luyen.py", "src/train_detector.py", "src/chay_giam_sat.py", "src/bdd_yolo_export.py",
              "src/detector_classes.py", "configs/phase2/detector/train_bdd5.yaml"):
        L.append(f"{sha(REPO / f)} *{f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
