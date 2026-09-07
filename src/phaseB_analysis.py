"""
phaseB_analysis.py -- Giai doan B buoc 3: danh gia ma tran 3 motion model x 4 bien the
ham chi phi lien ket.

===========================================================================
CAU HOI TRONG TAM - VA CAI BAY PHAI TRANH
===========================================================================
Tin hieu som cho thay bat DIoU lam GIAM 24% so track id rieng biet tren nhom
cv. Nghe thi tot (track it bi phan manh hon), NHUNG day chinh la cho de tu lua:

    it track id hon co the do NOI DUNG   (preserved - tot)
    ... hoac do NOI NHAM sang xe khac    (switched  - TE HON ca khong noi)

Chi so `preserved` mot minh KHONG DU de ket luan. Bat buoc phai xem dong thoi:
  - preserved : giu dung ID cu           -> muc tieu
  - switched  : noi nham sang ID khac    -> tac hai
  - lost      : khong tim lai duoc       -> trung tinh (khong noi con hon noi sai)
  - IDSW tu TrackEval : so lan doi ID tren toan video

Mot bien the lam preserved tang 5 nhung switched tang 20 la bien the TE, du
con so preserved nhin co ve cai thien.

===========================================================================
CACH DUNG
    python src/phaseB_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from stratified_analysis import analyse_tracker  # noqa: E402

VIDEOS = ["MVI_40131", "MVI_40171", "MVI_40172", "MVI_40204", "MVI_40751", "MVI_63554"]
MODELS = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
VARIANTS = {"iou": "IoU (goc)", "diou": "DIoU", "expand": "expanded gate",
            "diouexp": "DIoU + gate"}


def main() -> int:
    seg_tables = {
        "partial": pd.read_csv(config.INTERIM_DIR / "occlusion_segments.csv"),
        "full": pd.read_csv(config.INTERIM_DIR / "full_occlusion_segments.csv"),
    }
    for k in seg_tables:
        seg_tables[k] = seg_tables[k][seg_tables[k].video.isin(VIDEOS)]

    rows = []
    for mm, mlabel in MODELS.items():
        for var, vlabel in VARIANTS.items():
            tk = f"lassoc-{mm}-{var}"
            d = config.PROCESSED_DIR / "trackers" / "DETRAC-all" / tk / "data"
            if not d.is_dir() or len(list(d.glob("*.txt"))) < len(VIDEOS):
                print(f"  [MISS] {tk} chua chay xong, bo qua")
                continue
            res = analyse_tracker("DETRAC-all", tk, seg_tables, VIDEOS)
            for lvl, s in res["segments"].items():
                if len(s):
                    s = s.copy()
                    s["occlusion_level"] = lvl
                    s["motion_model"] = mlabel
                    s["variant"] = vlabel
                    s["variant_key"] = var
                    rows.append(s)

    if not rows:
        print("[ERROR] Chua co ket qua nao.")
        return 1
    seg = pd.concat(rows, ignore_index=True)
    out = config.RESULTS_DIR / "phaseB"
    out.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out / "segment_outcomes_matrix.csv", index=False)

    d = seg[seg.status != "no_before"]

    # --- Bang chinh: preserved / switched / lost cho tung o ---
    print("=" * 88)
    print("GIAI DOAN B - MA TRAN 3 MOTION MODEL x 4 BIEN THE HAM CHI PHI")
    print("=" * 88)
    print("\nDoc bang nay theo CA BA cot, khong chi nhin preserved:")
    print("  preserved TANG + switched TANG NHIEU HON  =>  bien the TE (noi nham)")
    print("  preserved TANG + switched giam/giu nguyen =>  bien the TOT\n")

    g = (d.groupby(["motion_model", "variant"])
           .agg(n=("status", "size"),
                preserved=("status", lambda s: (s == "preserved").sum()),
                switched=("status", lambda s: (s == "switched").sum()),
                lost=("status", lambda s: (s == "lost").sum()))
           .reset_index())
    g["giu_ID_%"] = (g.preserved / g.n * 100).round(2)
    g["noi_nham_%"] = (g.switched / g.n * 100).round(2)

    order = [VARIANTS[k] for k in VARIANTS]
    for m in MODELS.values():
        sub = g[g.motion_model == m].set_index("variant").reindex(order)
        if sub.n.isna().all():
            continue
        print(f"--- {m} ---")
        print(f"  {'bien the':<16} {'n':>5} {'preserved':>10} {'switched':>9} {'lost':>6} "
              f"{'giu ID %':>9} {'noi nham %':>11}")
        base = sub.loc[VARIANTS["iou"]] if VARIANTS["iou"] in sub.index else None
        for v in order:
            if v not in sub.index or pd.isna(sub.loc[v, "n"]):
                continue
            r = sub.loc[v]
            delta = ""
            if base is not None and v != VARIANTS["iou"] and not pd.isna(base.n):
                dp = int(r.preserved - base.preserved)
                ds = int(r.switched - base.switched)
                delta = f"   (preserved {dp:+d}, switched {ds:+d})"
            print(f"  {v:<16} {int(r.n):>5} {int(r.preserved):>10} {int(r.switched):>9} "
                  f"{int(r.lost):>6} {r['giu_ID_%']:>8.2f}% {r['noi_nham_%']:>10.2f}%{delta}")
        print()

    g.to_csv(out / "matrix_summary.csv", index=False)

    # --- Danh gia loi/hai rong ---
    print("=" * 88)
    print("DANH GIA LOI - HAI RONG (so voi bien the IoU goc cua CUNG motion model)")
    print("=" * 88)
    print(f"  {'motion model':<14} {'bien the':<16} {'d_preserved':>12} {'d_switched':>11} "
          f"{'loi rong':>9}")
    print("-" * 68)
    for m in MODELS.values():
        sub = g[g.motion_model == m].set_index("variant")
        if VARIANTS["iou"] not in sub.index:
            continue
        b = sub.loc[VARIANTS["iou"]]
        for v in order[1:]:
            if v not in sub.index:
                continue
            r = sub.loc[v]
            dp, ds = int(r.preserved - b.preserved), int(r.switched - b.switched)
            net = dp - ds
            verdict = "TOT" if net > 0 else ("hoa" if net == 0 else "TE")
            print(f"  {m:<14} {v:<16} {dp:>+12d} {ds:>+11d} {net:>+8d}  {verdict}")

    print("\n  'loi rong' = d_preserved - d_switched. Duong = noi dung nhieu hon noi nham.")
    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
