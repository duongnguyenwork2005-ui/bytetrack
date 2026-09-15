"""
verify_shared_kalman_regression.py -- Bang chung TREN DETECTION THAT ve loi `shared_kalman`
va ve viec ban sua KHONG doi ket qua Giai doan 1.

Cach lam
    1. Ghi provenance cua baseline TRUOC KHI SUA: commit 35a560f, sha256 cac file bo loc /
       tracker tai commit do, phien ban thu vien, sha256 checkpoint yolov8n.pt, sha256 3 file
       .yaml tracker, meta.json cua cache detection dung de doi chieu.
    2. Trich ma nguon TAI COMMIT 35a560f (git show) vao thu muc tam, chay tracking bang MA CU
       va MA MOI trong hai tien trinh con RIENG (tranh trung ten module), tren CUNG cache
       detection (yolov8n-coco_conf0.05_all, loc > 0.25, lop [2,3,5,7], max_det 300) voi:
         a) cau hinh MAC DINH (EKF, UKF, EKF-clamped)  -> ky vong BIT-IDENTICAL
         b) override tren instance bo loc cua tracker (`tracker.kalman_filter._std_theta = 0.05`,
            dung cach ma script/thi nghiem se lam neu muon doi tham so luc chay)
              -> ma cu: output == mac dinh (override BI BO QUA o multi_predict)  <- LOI
              -> ma moi: output != mac dinh (override co tac dung)
    3. Luu output vao results/phase2/shared_kalman/{prefix,fixed}/<case>/<video>.txt (khong commit)
       va diff_summary.csv + baseline_provenance.json (commit).

    python src/verify_shared_kalman_regression.py [--videos MVI_39031 MVI_20052 MVI_39931] [--commit 35a560f]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402

ROOT = config.PROJECT_ROOT
OUT = config.RESULTS_DIR / "phase2" / "shared_kalman"
# config.py KHONG doi giua hai commit (kiem tra sha) va KHONG duoc chep vao thu muc tam: no tinh
# PROJECT_ROOT tu vi tri file -> phai dung ban trong src/ de duong dan data/ dung.
FILES = ["ekf_ctrv.py", "ekf_ctrv_clamped.py", "ukf_ctrv.py", "lost_association.py", "tracker_ctrv.py"]
SHA_ONLY = ["config.py"]
CASES = {  # ten -> (yaml, override _std_theta hay None)
    "ekf_default": ("configs/bytetrack_ekf_ctrv.yaml", None),
    "ukf_default": ("configs/bytetrack_ukf_ctrv.yaml", None),
    "clamped_default": ("configs/bytetrack_ekf_ctrv_clamped.yaml", None),
    "ekf_override_std_theta_0.05": ("configs/bytetrack_ekf_ctrv.yaml", 0.05),
}

DRIVER = r'''
import sys, json
src_dir, cache_id, video, yaml_path, override, out_txt = sys.argv[1:7]
sys.path.insert(0, src_dir)                 # ma tracker (cu hoac moi) - phai dung TRUOC src/
sys.path.insert(1, r"%(src)s")             # det_cache, mot_datasets (chi doc cache)
import numpy as np, pandas as pd
import tracker_ctrv                         # tu src_dir
tracker_ctrv.register()
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import IterableSimpleNamespace, YAML
import det_cache
cfg = IterableSimpleNamespace(**YAML.load(yaml_path))
tr = TRACKER_MAP[cfg.tracker_type](args=cfg); tr.reset()
if override != "none":
    tr.kalman_filter._std_theta = float(override)      # override TREN INSTANCE cua tracker
vd = det_cache.open_cache(cache_id, video, 0.25, [2, 3, 5, 7], 300)
rows = []
for frame_idx, t_s, unreadable, arr in vd.iter_frames():
    if unreadable:
        continue
    out = tr.update(det_cache.to_boxes(arr, vd.orig_shape).cpu().numpy(), None)
    if len(out) == 0:
        continue
    xyxy = out[:, :4].astype(np.float32); ids = out[:, 4].astype(int); confs = out[:, 5].astype(np.float32)
    for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
        rows.append((frame_idx, int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(cf)))
pd.DataFrame(rows).to_csv(out_txt, header=False, index=False)
print(json.dumps({"filter": type(tr.kalman_filter).__name__, "std_theta": tr.kalman_filter._std_theta,
                  "n_rows": len(rows)}))
'''


def sha256_bytes(b: bytes) -> str:
    """sha256 sau khi chuan hoa CRLF -> LF (git show tra ve LF, working tree tren Windows la CRLF)."""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="35a560f")
    ap.add_argument("--cache-id", default="yolov8n-coco_conf0.05_all")
    ap.add_argument("--videos", nargs="*", default=["MVI_39031", "MVI_20052", "MVI_39931"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitignore").write_text("prefix/\nfixed/\n", encoding="utf-8")

    # --- 1. Provenance baseline truoc khi sua ---
    import torch, ultralytics, numpy, scipy
    old_src = Path(tempfile.mkdtemp(prefix="prefix_src_"))
    prov = {"prefix_commit": a.commit, "prefix_commit_full": subprocess.check_output(
        ["git", "rev-parse", a.commit], cwd=ROOT, text=True).strip(),
        "prefix_commit_date": subprocess.check_output(["git", "show", "-s", "--format=%ci", a.commit], cwd=ROOT, text=True).strip(),
        "current_commit": det_cache.git_commit(), "prefix_files_sha256": {}, "current_files_sha256": {},
        "versions": {"python": sys.version.split()[0], "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                     "cuda": torch.version.cuda, "numpy": numpy.__version__, "scipy": scipy.__version__},
        "checkpoint": {"path": config.YOLO_DEFAULT_MODEL, "sha256": det_cache.sha256_file(Path(config.YOLO_DEFAULT_MODEL))},
        "tracker_yaml_sha256": {}, "cache_id": a.cache_id, "cache_meta": det_cache.load_meta(det_cache.CACHE_ROOT / a.cache_id),
        "videos": a.videos, "load_filter": {"conf_min": 0.25, "classes": [2, 3, 5, 7], "max_det": 300},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for f in FILES:
        b = subprocess.check_output(["git", "show", f"{a.commit}:src/{f}"], cwd=ROOT)
        (old_src / f).write_bytes(b)
        prov["prefix_files_sha256"][f] = sha256_bytes(b)
        prov["current_files_sha256"][f] = sha256_bytes((ROOT / "src" / f).read_bytes())
    for f in SHA_ONLY:
        b = subprocess.check_output(["git", "show", f"{a.commit}:src/{f}"], cwd=ROOT)
        prov["prefix_files_sha256"][f] = sha256_bytes(b)
        prov["current_files_sha256"][f] = sha256_bytes((ROOT / "src" / f).read_bytes())
        assert prov["prefix_files_sha256"][f] == prov["current_files_sha256"][f], f"{f} da doi giua hai commit"
    for y, _ in CASES.values():
        prov["tracker_yaml_sha256"][y] = sha256_bytes((ROOT / y).read_bytes())
    (OUT / "baseline_provenance.json").write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    driver = old_src / "_driver.py"
    driver.write_text(DRIVER % {"src": str(ROOT / "src")}, encoding="utf-8")

    # --- 2. Chay ma cu va ma moi ---
    rows = []
    for case, (yaml_path, override) in CASES.items():
        for v in a.videos:
            outs = {}
            for tag, src_dir in (("prefix", old_src), ("fixed", ROOT / "src")):
                d = OUT / tag / case; d.mkdir(parents=True, exist_ok=True)
                txt = d / f"{v}.txt"
                r = subprocess.run([sys.executable, str(driver), str(src_dir), a.cache_id, v, str(ROOT / yaml_path),
                                    "none" if override is None else str(override), str(txt)],
                                   cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stderr[-2000:]); raise SystemExit(f"loi chay {tag}/{case}/{v}")
                outs[tag] = (txt.read_text(), json.loads(r.stdout.strip().splitlines()[-1]))
            same = outs["prefix"][0] == outs["fixed"][0]
            rows.append({"case": case, "video": v, "filter": outs["fixed"][1]["filter"],
                         "prefix_std_theta": outs["prefix"][1]["std_theta"], "fixed_std_theta": outs["fixed"][1]["std_theta"],
                         "n_rows_prefix": outs["prefix"][1]["n_rows"], "n_rows_fixed": outs["fixed"][1]["n_rows"],
                         "bit_identical": same})
            print(f"  {case:<28} {v:<12} prefix {outs['prefix'][1]['n_rows']:>6} dong | fixed {outs['fixed'][1]['n_rows']:>6} dong | "
                  f"bit-identical={same}")
    df = pd.DataFrame(rows)

    # --- 3. Override: ma cu == mac dinh (loi), ma moi != mac dinh ---
    for v in a.videos:
        d_old = (OUT / "prefix" / "ekf_default" / f"{v}.txt").read_text()
        o_old = (OUT / "prefix" / "ekf_override_std_theta_0.05" / f"{v}.txt").read_text()
        d_new = (OUT / "fixed" / "ekf_default" / f"{v}.txt").read_text()
        o_new = (OUT / "fixed" / "ekf_override_std_theta_0.05" / f"{v}.txt").read_text()
        rows.append({"case": "override_effect", "video": v, "filter": "EKFTrackerCTRV",
                     "prefix_override_equals_default": d_old == o_old, "fixed_override_equals_default": d_new == o_new})
        print(f"  override 0.05 vs mac dinh  {v:<12} ma CU: {'GIONG HET (override bi bo qua)' if d_old == o_old else 'khac'} | "
              f"ma MOI: {'giong' if d_new == o_new else 'KHAC (override co tac dung)'}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diff_summary.csv", index=False)
    dflt = df[df.case.str.endswith("_default")]
    print(f"\nMac dinh (EKF/UKF/clamped x {len(a.videos)} video): bit-identical = {int(dflt.bit_identical.sum())}/{len(dflt)}")
    print(f"-> {OUT / 'diff_summary.csv'}, {OUT / 'baseline_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
