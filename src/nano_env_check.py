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
  4. HOP DONG DAU VAO (src/nano_preflight.py, dung chung voi nano_run_cache.py -- review 20207ec N15-P3): DUNG hai cache
     (meta.json parse duoc; MVI_39031.npz du mang, sum(n_per_frame)==len(data), frame_idx/time_s tang ngat; sidecar parse
     duoc va khop schema) va DU BON file tham chieu PC doc duoc (>= 1 dong MOTChallenge hop le); YAML + ma. "Co thu muc"
     KHONG du. Thieu/hong bat ky thu gi -> KHONG DAT.
  5. Thong tin he thong: uname, /etc/nv_tegra_release (neu co), CPU/RAM/dia trong, nvpmodel (neu co).
Exit 0 CHI khi 1-4 deu DAT; exit 1 neu co muc KHONG DAT.
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

    # 4. HOP DONG DAU VAO: validator dung chung (src/nano_preflight.py) -- khong "co thu muc la du"
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import nano_preflight
        kq4 = nano_preflight.kiem_dau_vao(ROOT)
        for dong in nano_preflight.mo_ta(kq4):
            ghi("4. " + dong, "DAT" if kq4["dat"] else "INFO")
        if kq4["dat"]:
            ghi("4. hop dong dau vao (2 cache + 4 tham chieu PC + YAML + ma)", "DAT",
                "%d/%d tham chieu doc duoc" % (sum(1 for v in kq4["pc_ref"].values() if v["ok"]), len(kq4["pc_ref"])))
        else:
            for i, l in enumerate(kq4["loi"], start=1):
                ghi("4. dau vao #%d" % i, "KHONG DAT", l)
    except Exception as e:  # noqa: BLE001
        ghi("4. hop dong dau vao", "KHONG DAT", "%s: %s" % (type(e).__name__, e))

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
