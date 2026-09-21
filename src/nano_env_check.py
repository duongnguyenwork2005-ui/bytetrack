# -*- coding: utf-8 -*-
"""
nano_env_check.py -- Kiem MOI TRUONG truoc khi chay tracker tu cache tren thiet bi (Jetson Nano) hoac may bat ky.

    python3 src/nano_env_check.py            # chay tu GOC goi (thu muc chua src/, third_party/, data/)
    python3 src/nano_env_check.py --json ket_qua.json

Co y viet bang cu phap Python 3.6 (KHONG f-string phuc tap, KHONG `from __future__ import annotations`,
KHONG dataclasses) de van chay duoc va BAO RO tren interpreter cu: yeu cau thuc te cua chuoi import
tracker-tu-cache la Python >= 3.8 (vermin tinh: 21 file `from __future__ import annotations` (3.7),
`dataclasses` (3.7), `importlib.metadata` (3.8)); day la ket luan tinh, khong phai gia dinh.

Kiem (moi muc DAT / KHONG DAT / BO QUA, khong doan):
  1. Python >= 3.8.
  2. Goi bat buoc cho duong tracker-tu-cache: numpy, lap (lap.lapjv), pandas, yaml  (scipy: TUY CHON --
     chi dung khi use_lap=False; opencv: chi cho buoc chay video).
  3. Backend NumPy import duoc voi torch/ultralytics BI CHAN (khong keo hai goi nay).
  4. Cache + ban tham chieu trong goi doc duoc (npz co du mang, frame rong duoc giu, time_s tang).
  5. Thong tin he thong: uname, /etc/nv_tegra_release (neu co), CPU/RAM/dia trong, nvpmodel (neu co).
Exit 0 khi 1-4 DAT; exit 1 neu co muc KHONG DAT.
"""
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KQ = []


def ghi(muc, trang_thai, chi_tiet=""):
    KQ.append({"muc": muc, "trang_thai": trang_thai, "chi_tiet": chi_tiet})
    print("  [%-9s] %s%s" % (trang_thai, muc, ("  -- " + chi_tiet) if chi_tiet else ""))


def lenh(cmd):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        out, _ = p.communicate(timeout=20)
        return out.decode("utf-8", "replace").strip()
    except Exception as e:  # noqa: BLE001
        return "(khong chay duoc: %s)" % e


def main():
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]
    print("nano_env_check -- goc goi: %s" % ROOT)
    print("python: %s (%s)" % (sys.version.split()[0], sys.executable))

    # 1. Python
    v = sys.version_info
    if v >= (3, 8):
        ghi("1. Python >= 3.8", "DAT", "%d.%d.%d" % (v.major, v.minor, v.micro))
    else:
        ghi("1. Python >= 3.8", "KHONG DAT",
            "%d.%d.%d -- chuoi import can 3.8 (from __future__ import annotations, dataclasses, importlib.metadata). "
            "Dung moi truong rieng >= 3.8 (venv tu python3.8 hoac miniforge); KHONG doi python3 he thong." % (v.major, v.minor, v.micro))

    # 2. goi bat buoc / tuy chon
    phien_ban = {}
    for ten, bat_buoc in (("numpy", True), ("lap", True), ("pandas", True), ("yaml", True), ("scipy", False), ("cv2", False)):
        try:
            m = __import__(ten)
            ver = getattr(m, "__version__", "?")
            phien_ban[ten] = ver
            if ten == "lap" and not hasattr(m, "lapjv"):
                ghi("2. %s" % ten, "KHONG DAT", "import duoc nhung khong co lap.lapjv (sai goi?)")
                continue
            ghi("2. %s%s" % (ten, "" if bat_buoc else " (tuy chon)"), "DAT", ver)
        except Exception as e:  # noqa: BLE001
            phien_ban[ten] = None
            if bat_buoc:
                ghi("2. %s" % ten, "KHONG DAT", "%s: %s" % (type(e).__name__, e))
            else:
                ghi("2. %s (tuy chon)" % ten, "BO QUA", "khong co -- %s" % ("chi can cho use_lap=False" if ten == "scipy" else "chi can cho buoc chay video"))

    # 3. backend np, chan torch/ultralytics (chi khi Python du moi de import ma nguon)
    if v >= (3, 8):
        code = (
            "import sys\n"
            "class _B:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('torch', 'ultralytics'): raise ImportError('CHAN ' + name)\n"
            "sys.meta_path.insert(0, _B())\n"
            "import os, json\n"
            "os.environ['BYTETRACK_BACKEND'] = 'np'\n"
            "sys.path.insert(0, %r)\n"
            "import ul_backend, tracker_ctrv, run_tracking\n"
            "fp = ul_backend.fingerprint()\n"
            "print(json.dumps({'backend': ul_backend.backend_name(), 'torch': 'torch' in sys.modules, 'ultralytics': 'ultralytics' in sys.modules, 'vendor': fp.get('vendor_modules')}))\n"
        ) % os.path.join(ROOT, "src")
        try:
            p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
            so, se = p.communicate(timeout=120)
            so = so.decode("utf-8", "replace").strip(); se = se.decode("utf-8", "replace").strip()
            if p.returncode == 0 and '"torch": false' in so and '"ultralytics": false' in so:
                d = json.loads(so.splitlines()[-1])
                thieu = [k for k, h in (d.get("vendor") or {}).items() if h == "MISSING"]
                if thieu:
                    ghi("3. backend np (chan torch/ultralytics)", "KHONG DAT", "thieu module vendor: %s" % thieu)
                else:
                    ghi("3. backend np (chan torch/ultralytics)", "DAT", "backend=%s, torch/ultralytics KHONG bi nap" % d.get("backend"))
            else:
                ghi("3. backend np (chan torch/ultralytics)", "KHONG DAT", (se or so)[-400:])
        except Exception as e:  # noqa: BLE001
            ghi("3. backend np (chan torch/ultralytics)", "KHONG DAT", "%s: %s" % (type(e).__name__, e))
    else:
        ghi("3. backend np (chan torch/ultralytics)", "BO QUA", "Python < 3.8, khong import duoc ma nguon")

    # 4. cache + tham chieu trong goi
    try:
        import numpy as np
        cdir = os.path.join(ROOT, "data", "interim", "det_cache")
        caches = sorted(d for d in os.listdir(cdir)) if os.path.isdir(cdir) else []
        if not caches:
            ghi("4. det cache trong goi", "KHONG DAT", "khong co %s" % cdir)
        for cid in caches:
            d = os.path.join(cdir, cid)
            meta_ok = os.path.isfile(os.path.join(d, "meta.json"))
            npzs = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
            for f in npzs:
                z = np.load(os.path.join(d, f))
                can = ("frame_idx", "n_per_frame", "unreadable", "truncated", "time_s", "data", "orig_shape")
                thieu = [k for k in can if k not in z.files]
                side = os.path.isfile(os.path.join(d, f[:-4] + ".json"))
                if thieu or not meta_ok or not side:
                    ghi("4. cache %s/%s" % (cid, f), "KHONG DAT", "thieu mang %s meta=%s sidecar=%s" % (thieu, meta_ok, side))
                    continue
                n = len(z["frame_idx"]); n_rong = int((z["n_per_frame"] == 0).sum())
                tang = bool(np.all(np.diff(z["time_s"]) > 0)) if n > 1 else True
                tong = int(z["n_per_frame"].sum()) == len(z["data"])
                ok = tang and tong
                ghi("4. cache %s/%s" % (cid, f), "DAT" if ok else "KHONG DAT",
                    "%d frame (%d frame RONG giu nguyen), %d box, time_s tang=%s, tong n_per_frame==len(data)=%s"
                    % (n, n_rong, len(z["data"]), tang, tong))
        tdir = os.path.join(ROOT, "data", "processed", "trackers")
        refs = []
        if os.path.isdir(tdir):
            for split in sorted(os.listdir(tdir)):
                for name in sorted(os.listdir(os.path.join(tdir, split))):
                    if name.startswith("pc-ref-"):
                        dd = os.path.join(tdir, split, name, "data")
                        n_txt = len([f for f in os.listdir(dd) if f.endswith(".txt")]) if os.path.isdir(dd) else 0
                        refs.append("%s/%s (%d file)" % (split, name, n_txt))
        ghi("4. ban tham chieu PC (pc-ref-*)", "DAT" if refs else "KHONG DAT", ", ".join(refs) or "khong co")
    except Exception as e:  # noqa: BLE001
        ghi("4. det cache trong goi", "KHONG DAT", "%s: %s" % (type(e).__name__, e))

    # 5. he thong (chi ghi nhan)
    ht = {"uname": " ".join(platform.uname()), "may": platform.machine()}
    if os.path.isfile("/etc/nv_tegra_release"):
        ht["nv_tegra_release"] = open("/etc/nv_tegra_release").read().strip()
    if os.path.isfile("/proc/meminfo"):
        ht["meminfo_MemTotal"] = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].strip()
    if sys.platform.startswith("linux"):
        ht["df_goi"] = lenh("df -h %s | tail -1" % ROOT)
        ht["nvpmodel"] = lenh("nvpmodel -q 2>/dev/null || echo '(khong co nvpmodel)'")
        ht["python3_8"] = lenh("command -v python3.8 || echo '(khong co python3.8 trong PATH)'")
        ht["conda"] = lenh("command -v conda || command -v mamba || echo '(khong co conda/mamba)'")
    for k, val in ht.items():
        print("  [he thong ] %s: %s" % (k, val))

    kem = [r for r in KQ if r["trang_thai"] == "KHONG DAT"]
    print("KET LUAN: %s" % ("DAT (1-4)" if not kem else "KHONG DAT: " + "; ".join(r["muc"] for r in kem)))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"python": sys.version, "executable": sys.executable, "phien_ban": phien_ban,
                       "kiem": KQ, "he_thong": ht, "dat": not kem}, f, indent=2, ensure_ascii=False)
        print("-> %s" % out_json)
    return 1 if kem else 0


if __name__ == "__main__":
    sys.exit(main())
