"""
nano_run_cache.py -- Chay tracker CV/EKF TU CACHE tren thiet bi (Jetson Nano) va doi chieu voi tham chieu PC.
Script RIENG cho thiet bi: CHI ghi output `nano-*`; KHONG bao gio tao lai `pc-ref-*` (review 7a2feb6, finding 1).

    python src/nano_run_cache.py [--tag nano] [--chi-smoke] [--khong-chan]        # chay tu GOC goi (cwd = thu muc goi)
    -> data/processed/trackers/NANO-smoke/<tag>-{cv,ekf}/ , NANO-full/<tag>-{cv,ekf}/
    -> reference/nano_run_<YYYYmmdd-HHMMSS>.{json,log}, reference/parity_<tag>_{smoke,full}_{cv,ekf}.json

Thu tu (finding 2: kiem DAU VAO truoc, thieu -> LOI, khong chay):
  0. Kiem: cache smoke + cache goc (meta.json, npz doc duoc va du mang, sidecar), DU BON tham chieu PC
     (NANO-smoke/pc-ref-cv|ekf, NANO-full/pc-ref-cv|ekf: data/MVI_39031.txt doc duoc), YAML tracker, backend np import
     duoc (CHAN torch/ultralytics tru khi --khong-chan).
  1. Hash SHA256 toan bo cay pc-ref-* TRUOC khi chay (ghi reference/pc_ref_sha256_truoc.txt) va doi chieu voi
     reference/pc_ref_sha256.txt do PC dong goi (neu co) -> khac => LOI.
  2. Smoke (cache cat 300 frame): CV roi EKF -> parity voi pc-ref (tolerance cong bo trong nano_parity.py) + bit-identical.
     Smoke KHONG dat -> dung, khong chay full (giu output de phan tich).
  3. Full (cache goc 1470 frame): CV, EKF -> parity + bit-identical.
  4. Hash lai cay pc-ref-* SAU khi chay -> phai y het truoc.
Exit 0 khi moi buoc dat; 2 thieu dau vao; 1 co buoc khong dat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = "MVI_39031"
CACHES = {"smoke": ("nano_smoke_MVI_39031_f0001-0300", "NANO-smoke"), "full": ("yolov8n-coco_conf0.05_all", "NANO-full")}
TRACKERS = {"cv": "bytetrack.yaml", "ekf": "configs/bytetrack_ekf_ctrv.yaml"}
TRACK_ARGS = ["--videos", VIDEO, "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7", "--max-det", "300",
              "--fps", "25", "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
BLOCK = '''
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "ultralytics"):
            raise ImportError("CHAN (nano_run_cache): " + name)
sys.meta_path.insert(0, _Block())
'''


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_cay(goc: Path, thu_muc: list[Path]) -> list[str]:
    out = []
    for d in thu_muc:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out.append(f"{sha256_file(p)}  {p.relative_to(goc).as_posix()}")
    return out


def run_py(script_rel: str, args: list[str], goc: Path, chan: bool, log: list[str]) -> subprocess.CompletedProcess:
    code = ((BLOCK if chan else "") + "import sys, runpy\nsys.path.insert(0, 'src')\n"
            f"sys.argv = [{script_rel!r}] + {args!r}\nrunpy.run_path({script_rel!r}, run_name='__main__')\n")
    env = {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(goc), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    log.append(f"$ {script_rel} {' '.join(args)}  (BYTETRACK_BACKEND=np, chan torch/ul={chan})\n  exit={r.returncode}")
    log.extend("  | " + ln for ln in (r.stdout + r.stderr).strip().splitlines()[-8:])
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nano", help="tien to ten output (KHONG duoc la 'pc-ref')")
    ap.add_argument("--chi-smoke", action="store_true")
    ap.add_argument("--khong-chan", action="store_true", help="khong chan torch/ultralytics (mac dinh: chan de chung minh khong can)")
    ap.add_argument("--goc", default=str(ROOT))
    a = ap.parse_args()
    goc = Path(a.goc).resolve()
    if a.tag.startswith("pc-ref"):
        print("[LOI] --tag khong duoc bat dau bang 'pc-ref' (tham chieu PC la bat bien)"); return 2
    ts = time.strftime("%Y%m%d-%H%M%S")
    ref_dir = goc / "reference"; ref_dir.mkdir(exist_ok=True)
    log: list[str] = [f"# nano_run_cache {ts} goc={goc} python={sys.executable} {sys.version.split()[0]} tag={a.tag}"]
    kq: dict = {"goc": str(goc), "python": sys.version, "executable": sys.executable, "tag": a.tag, "bat_dau": ts, "buoc": {}}

    # ---- 0. kiem dau vao ----
    import numpy as np
    thieu = []
    for ten, (cid, split) in CACHES.items():
        cdir = goc / "data" / "interim" / "det_cache" / cid
        for f in ("meta.json", f"{VIDEO}.npz", f"{VIDEO}.json"):
            if not (cdir / f).is_file():
                thieu.append(f"cache {ten}: {cdir / f}")
        if (cdir / f"{VIDEO}.npz").is_file():
            try:
                z = np.load(cdir / f"{VIDEO}.npz")
                can = ("frame_idx", "n_per_frame", "unreadable", "truncated", "time_s", "data", "orig_shape")
                miss = [k for k in can if k not in z.files]
                if miss:
                    thieu.append(f"cache {ten}: npz thieu mang {miss}")
                elif int(z["n_per_frame"].sum()) != len(z["data"]):
                    thieu.append(f"cache {ten}: tong n_per_frame != len(data)")
                else:
                    kq["buoc"].setdefault("0_cache", {})[ten] = {"n_frame": int(len(z["frame_idx"])), "n_box": int(len(z["data"])),
                                                                "n_frame_rong": int((z["n_per_frame"] == 0).sum())}
            except Exception as e:  # noqa: BLE001
                thieu.append(f"cache {ten}: npz khong doc duoc ({type(e).__name__}: {e})")
        if (cdir / "meta.json").is_file():
            try:
                json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                thieu.append(f"cache {ten}: meta.json khong doc duoc ({e})")
        for t in TRACKERS:
            p = goc / "data" / "processed" / "trackers" / split / f"pc-ref-{t}" / "data" / f"{VIDEO}.txt"
            if not p.is_file():
                thieu.append(f"tham chieu PC thieu: {p}")
            else:
                try:
                    n = sum(1 for _ in open(p, encoding="utf-8"))
                    kq["buoc"].setdefault("0_pc_ref", {})[f"{split}/pc-ref-{t}"] = {"n_hang": n}
                except Exception as e:  # noqa: BLE001
                    thieu.append(f"tham chieu PC khong doc duoc: {p} ({e})")
    for t, y in TRACKERS.items():
        if "/" in y and not (goc / y).is_file():
            thieu.append(f"YAML tracker thieu: {y}")
    for s in ("src/run_tracking.py", "src/nano_parity.py", "third_party/ultralytics_trackers/byte_tracker.py"):
        if not (goc / s).is_file():
            thieu.append(f"thieu ma: {s}")
    r = run_py("src/nano_env_check.py", [], goc, False, log) if (goc / "src/nano_env_check.py").is_file() else None
    kq["buoc"]["0_env_check_exit"] = (r.returncode if r else None)
    code = "import sys, json\nsys.path.insert(0, 'src')\nimport ul_backend\nprint(json.dumps({'backend': ul_backend.backend_name(), 'torch': 'torch' in sys.modules, 'ul': 'ultralytics' in sys.modules}))"
    r = subprocess.run([sys.executable, "-c", (BLOCK if not a.khong_chan else "") + code], cwd=str(goc), capture_output=True, text=True,
                       env={**os.environ, "BYTETRACK_BACKEND": "np"})
    kq["buoc"]["0_import_np"] = {"exit": r.returncode, "out": r.stdout.strip()[-200:], "err": r.stderr.strip()[-300:]}
    log.append(f"import backend np: exit={r.returncode} {r.stdout.strip()[-120:]} {r.stderr.strip()[-200:]}")
    if r.returncode != 0 or '"torch": false' not in r.stdout:
        thieu.append("backend np khong import duoc (hoac keo torch/ultralytics)")
    if thieu:
        kq["buoc"]["0_thieu"] = thieu; kq["ket_luan"] = "LOI: thieu dau vao -- khong chay"
        log += ["[LOI] thieu dau vao:"] + ["  - " + t for t in thieu]
        _ghi(ref_dir, ts, kq, log); return 2
    log.append("0. dau vao: DU (2 cache, 4 tham chieu PC, YAML, ma, backend np)")

    # ---- 1. hash cay pc-ref TRUOC ----
    pc_dirs = [goc / "data/processed/trackers" / split / f"pc-ref-{t}" for _, (_, split) in CACHES.items() for t in TRACKERS]
    truoc = hash_cay(goc, pc_dirs)
    (ref_dir / "pc_ref_sha256_truoc.txt").write_text("\n".join(truoc) + "\n", encoding="utf-8")
    goc_hash = ref_dir / "pc_ref_sha256.txt"
    if goc_hash.is_file():
        chuan = [ln.strip() for ln in goc_hash.read_text(encoding="utf-8").splitlines() if ln.strip()]
        khop = set(chuan) == set(truoc)
        kq["buoc"]["1_pc_ref_khop_ban_dong_goi"] = khop
        log.append(f"1. hash pc-ref ({len(truoc)} file) khop ban PC dong goi: {khop}")
        if not khop:
            kq["ket_luan"] = "LOI: tham chieu PC khac ban dong goi -> khong chay"; _ghi(ref_dir, ts, kq, log); return 2
    else:
        kq["buoc"]["1_pc_ref_khop_ban_dong_goi"] = None
        log.append(f"1. hash pc-ref ({len(truoc)} file) ghi reference/pc_ref_sha256_truoc.txt (khong co ban dong goi de doi chieu)")

    # ---- 2/3. chay + parity ----
    dat_all = True
    for ten in (["smoke"] if a.chi_smoke else ["smoke", "full"]):
        cid, split = CACHES[ten]
        for t, yaml in TRACKERS.items():
            name = f"{a.tag}-{t}"
            out_dir = goc / "data/processed/trackers" / split / name
            args = ["--cache-id", cid, "--split-name", split, "--tracker-yaml", yaml, "--tracker-name", name, *TRACK_ARGS]
            t0 = time.perf_counter()
            r = run_py("src/run_tracking.py", args, goc, not a.khong_chan, log)
            wall = time.perf_counter() - t0
            pj = ref_dir / f"parity_{a.tag}_{ten}_{t}.json"
            r2 = run_py("src/nano_parity.py", ["--ref", f"data/processed/trackers/{split}/pc-ref-{t}", "--test", f"data/processed/trackers/{split}/{name}",
                                               "--json", str(pj.relative_to(goc))], goc, not a.khong_chan, log)
            f_ref = goc / "data/processed/trackers" / split / f"pc-ref-{t}" / "data" / f"{VIDEO}.txt"
            f_new = out_dir / "data" / f"{VIDEO}.txt"
            bit = f_new.is_file() and sha256_file(f_ref) == sha256_file(f_new)
            pjd = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
            v = (pjd.get("ket_qua") or [{}])[0]
            rec = {"replay_exit": r.returncode, "wall_s": round(wall, 2), "parity_exit": r2.returncode, "dat": bool(v.get("dat")),
                   "n_ref": v.get("n_ref"), "n_test": v.get("n_test"), "max_d_bbox": v.get("max_d_bbox"), "max_d_conf": v.get("max_d_conf"),
                   "phan_hoach": v.get("phan_hoach"), "ly_do": v.get("ly_do"), "bit_identical_data_txt": bit, "output": str(out_dir.relative_to(goc))}
            kq["buoc"][f"2_{ten}_{t}"] = rec
            log.append(f"{ten}/{t}: replay exit {r.returncode} ({wall:.1f}s) | parity exit {r2.returncode} dat={rec['dat']} n={rec['n_ref']}/{rec['n_test']} "
                       f"| bit-identical={bit}")
            dat_all &= r.returncode == 0 and r2.returncode == 0 and rec["dat"]
        if ten == "smoke" and not dat_all:
            log.append("smoke KHONG dat -> KHONG chay full; output giu nguyen de phan tich (du lieu -> cau hinh -> dependency -> so hoc -> lien ket ID)")
            break

    # ---- 4. hash cay pc-ref SAU ----
    sau = hash_cay(goc, pc_dirs)
    khong_doi = sau == truoc
    kq["buoc"]["4_pc_ref_khong_doi_sau_chay"] = khong_doi
    log.append(f"4. cay pc-ref sau khi chay y het truoc: {khong_doi}")
    dat_all &= khong_doi
    kq["ket_luan"] = "DAT" if dat_all else "KHONG DAT"
    log.append(f"KET LUAN: {kq['ket_luan']}")
    _ghi(ref_dir, ts, kq, log)
    return 0 if dat_all else 1


def _ghi(ref_dir: Path, ts: str, kq: dict, log: list[str]) -> None:
    (ref_dir / f"nano_run_{ts}.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
    (ref_dir / f"nano_run_{ts}.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log[-14:]))
    print(f"-> {ref_dir / f'nano_run_{ts}.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
