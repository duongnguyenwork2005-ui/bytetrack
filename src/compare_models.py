"""
compare_models.py -- Gop ket qua TrackEval cua 3 motion model thanh 1 bang so sanh.

O Giai doan 2-4, bang so sanh `results/comparison_DETRAC-sample.csv` duoc lam
thu cong. Giai doan 5 chay tren 60 video nen can tu dong hoa buoc nay.

Doc `results/trackeval/<split>/<tracker>/combined_metrics.csv` cua tung motion
model roi xuat:
  - bang chi so tong hop, kem cot chenh lech (%) so voi baseline KF + CV
  - bang chi so theo tung video (de xem phuong sai giua cac video)

CACH DUNG
    python src/compare_models.py --split-name DETRAC-all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SHORT = {"cv": "KF + CV (baseline)", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDP", "IDR"]
COUNTS = ["FP", "FN", "IDSW"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Gop ket qua TrackEval cua cac motion model")
    ap.add_argument("--split-name", default="DETRAC-all")
    args = ap.parse_args()

    stem = Path(config.YOLO_DEFAULT_MODEL).stem
    root = config.RESULTS_DIR / "trackeval" / args.split_name

    rows, per_video = [], []
    for key, spec in config.MOTION_MODELS.items():
        tracker = f"{stem}-{spec['suffix']}"
        f = root / tracker / "combined_metrics.csv"
        if not f.exists():
            print(f"  [MISS] chua co {f}")
            continue
        d = pd.read_csv(f).iloc[0].to_dict()
        d["motion_model"] = SHORT.get(key, key)
        d["tracker_dir"] = tracker
        rows.append(d)

        pv = root / tracker / "per_video_metrics.csv"
        if pv.exists():
            p = pd.read_csv(pv)
            p["motion_model"] = SHORT.get(key, key)
            per_video.append(p)

    if not rows:
        print("[ERROR] Chua co ket qua TrackEval nao. Chay src/run_trackeval.py truoc.")
        return 1

    df = pd.DataFrame(rows)
    cols = ["motion_model"] + METRICS + COUNTS + ["GT_dets", "tracker_dir"]
    df = df[[c for c in cols if c in df.columns]]

    # --- Chenh lech (%) so voi baseline ---
    base = df[df["motion_model"].str.startswith("KF + CV")]
    if len(base):
        b = base.iloc[0]
        for m in METRICS:
            if m in df.columns:
                df[f"d_{m}_%"] = ((df[m] - b[m]) / b[m] * 100).round(3)
        for c in COUNTS:
            if c in df.columns:
                df[f"d_{c}"] = (df[c] - b[c]).astype(int)

    out = config.RESULTS_DIR / f"comparison_{args.split_name}.csv"
    df.to_csv(out, index=False)

    print("=" * 78)
    print(f"SO SANH MOTION MODEL -- {args.split_name}")
    print("=" * 78)
    show = ["motion_model"] + [c for c in METRICS + COUNTS if c in df.columns]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    dcols = [c for c in df.columns if c.startswith("d_")]
    if dcols:
        print("\n  Chenh lech so voi baseline KF + CV:")
        print(df[["motion_model"] + dcols].to_string(index=False,
              float_format=lambda v: f"{v:+.3f}"))
    print(f"\n  Da luu -> {out}")

    if per_video:
        pv = pd.concat(per_video, ignore_index=True)
        out_pv = config.RESULTS_DIR / f"comparison_per_video_{args.split_name}.csv"
        pv.to_csv(out_pv, index=False)
        print(f"  Theo tung video -> {out_pv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
