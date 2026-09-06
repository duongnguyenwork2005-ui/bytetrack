"""
buffer_sweep_analysis.py -- Stage E: track_buffer co dang che mat khac biet that khong?

GIA THUYET CAN KIEM TRA
O Stage 5 va Stage C, nhom doan che khuat DAI (>1.5s) cho ca 3 motion model deu
giu duoc 0% ID. Co 2 cach giai thich hoan toan khac nhau:

  (1) "3 motion model nhu nhau"  -> ket luan ve motion model
  (2) FLOOR EFFECT: track_buffer = 30 frame (1.2s o 25fps) nen ByteTrack XOA HAN
      track truoc khi doan che ket thuc. Khong con track thi khong motion model
      nao co co hoi the hien -> phep do bi CHAN TRAN, khong noi len dieu gi ve
      motion model ca.

Neu (2) dung thi noi track_buffer phai lam khac biet lo ra. Neu noi buffer ma 3
model VAN nhu nhau thi moi ket luan duoc (1).

THIET KE
18 doan `full x long` (che >=0.90, >1.5s) nam gon trong 6 video. Do dai cua chung
tu 38 den 66 frame:
    buffer 30 -> 0/18 doan nam trong buffer  (tat ca deu bi xoa)
    buffer 60 -> 16/18
    buffer 90 -> 18/18
=> ba muc buffer nay tach bach duoc dung gia thuyet can kiem tra.

CHI doi track_buffer, moi tham so ghep cap khac giu nguyen -> khong dung toi
kien truc motion model da chot.

CACH DUNG
    python src/buffer_sweep_analysis.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from stratified_analysis import analyse_tracker, mcnemar  # noqa: E402

VIDEOS = ["MVI_40131", "MVI_40171", "MVI_40172", "MVI_40204", "MVI_40751", "MVI_63554"]
LABEL = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
BASE_SUFFIX = {"cv": "bytetrack", "ekf_ctrv": "ekf-ctrv", "ukf_ctrv": "ukf-ctrv"}


def tracker_name(mm: str, buf: int) -> str:
    """buffer 30 = lan chay chinh o Giai doan 5; 60/90 = lan chay sweep cua Stage E."""
    if buf == 30:
        return f"{Path(config.YOLO_DEFAULT_MODEL).stem}-{BASE_SUFFIX[mm]}"
    return f"sweep-{mm}-buf{buf}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage E: quet track_buffer")
    ap.add_argument("--split-name", default="DETRAC-all")
    ap.add_argument("--buffers", nargs="*", type=int, default=[30, 60, 90])
    args = ap.parse_args()

    seg_tables = {
        "partial": pd.read_csv(config.INTERIM_DIR / "occlusion_segments.csv"),
        "full": pd.read_csv(config.INTERIM_DIR / "full_occlusion_segments.csv"),
    }
    # Chi giu doan cua 6 video dang xet
    for k in seg_tables:
        seg_tables[k] = seg_tables[k][seg_tables[k].video.isin(VIDEOS)]

    rows = []
    for buf in args.buffers:
        for mm in ["cv", "ekf_ctrv", "ukf_ctrv"]:
            tk = tracker_name(mm, buf)
            d = config.PROCESSED_DIR / "trackers" / args.split_name / tk / "data"
            if not d.is_dir():
                print(f"  [MISS] {tk} - chua chay, bo qua")
                continue
            res = analyse_tracker(args.split_name, tk, seg_tables, VIDEOS)
            for lvl, s in res["segments"].items():
                if not len(s):
                    continue
                s = s.copy()
                s["occlusion_level"] = lvl
                s["track_buffer"] = buf
                s["motion_model"] = LABEL[mm]
                rows.append(s)

    if not rows:
        print("[ERROR] Khong co ket qua nao.")
        return 1
    seg = pd.concat(rows, ignore_index=True)

    out_dir = config.RESULTS_DIR / "buffer_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_dir / "segment_outcomes_sweep.csv", index=False)

    # --- Chi so sanh tren tap doan chung cua MOI (buffer x model) ---
    # Neu khong, moi o co mau so khac nhau va ti le % khong so sanh truc tiep duoc.
    key = ["occlusion_level", "video", "track_id", "seg_id"]
    n_cells = seg.groupby(["track_buffer", "motion_model"]).ngroups
    cnt = (seg[seg.status != "no_before"].groupby(key)
           .apply(lambda g: len(g.drop_duplicates(["track_buffer", "motion_model"])),
                  include_groups=False))
    common = set(cnt[cnt == n_cells].index)
    seg["in_common"] = [tuple(r) in common for r in seg[key].to_numpy()]
    cmp_ = seg[seg["in_common"]]
    print(f"[sweep] {len(common)} doan duoc ca {n_cells} to hop (buffer x model) "
          f"xac dinh duoc id_before -> dung de so sanh")

    # --- Bang chinh: ti le giu ID theo (bucket x buffer x model) ---
    for lvl, lab in [("full", "CHE KHUAT >= 0.90"), ("partial", "CHE KHUAT >= 0.10")]:
        d = cmp_[cmp_.occlusion_level == lvl]
        if not len(d):
            continue
        print("\n" + "=" * 78)
        print(f"STAGE E - {lab}: ti le giu ID theo track_buffer")
        print("=" * 78)
        g = (d.groupby(["bucket", "track_buffer", "motion_model"])
               .agg(n=("status", "size"),
                    giu_ID=("status", lambda s: round((s == "preserved").mean(), 4)))
               .reset_index())
        p = g.pivot_table(index=["bucket", "track_buffer"], columns="motion_model",
                          values="giu_ID", observed=True)
        nn = g.pivot_table(index=["bucket", "track_buffer"], columns="motion_model",
                           values="n", observed=True)
        print("\n  [ti le giu ID]"); print(p.round(4).to_string())
        print("\n  [n]"); print(nn.iloc[:, 0].astype(int).to_string())
        g.to_csv(out_dir / f"retention_{lvl}.csv", index=False)

    # --- McNemar o moi muc buffer, rieng nhom long ---
    print("\n" + "=" * 78)
    print("McNEMAR tren nhom `long` o tung muc track_buffer")
    print("=" * 78)
    out = []
    for buf in sorted(cmp_.track_buffer.unique()):
        d = cmp_[(cmp_.track_buffer == buf) & (cmp_.bucket == "long")]
        if not len(d):
            continue
        t = mcnemar(d, ["occlusion_level"], "KF + CV")
        if len(t):
            t["track_buffer"] = buf
            out.append(t)
    if out:
        t = pd.concat(out, ignore_index=True)
        t.to_csv(out_dir / "mcnemar_long_by_buffer.csv", index=False)
        print(t.to_string(index=False))
    print(f"\n  -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
