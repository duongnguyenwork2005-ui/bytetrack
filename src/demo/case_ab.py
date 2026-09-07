"""
case_ab.py -- DEMO Phan 2: phan loai Case A / Case B.

CAU HOI: khuc cua xay ra trong luc bi che khuat da HINH THANH TU TRUOC (tracker
co the biet), hay PHAT SINH sau khi mat quan sat (khong the biet)?

    CASE A    : |omega_before| > nguong VA omega_during CUNG DAU
                -> khuc cua da hinh thanh truoc -> CTRV co co so ngoai suy dung
    CASE B    : |omega_before| <= nguong NHUNG |omega_during| > nguong
                -> khuc cua phat sinh sau khi mat quan sat -> KHONG mo hinh nao
                   du doan duoc: day la gioi han THONG TIN
    STRAIGHT  : ca hai <= nguong -> xe di thang, CV la du
    CASE C    : ca hai > nguong nhung NGUOC DAU -> xe doi chieu quay giua doan che

Ti le CASE B cao = ung ho ket luan "gioi han thong tin" da do o giai doan truoc
(omega GT truoc che chi du doan dung dau omega GT trong che 58.2% so lan).

CHAY VOI 3 NGUONG khac nhau de xem ket luan co nhay cam voi nguong khong.

CACH DUNG
    python src/demo/case_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Nguong |omega| tinh bang DO/FRAME (nhan 25 de ra do/giay).
# 0.10 do/f = 2.5 do/s  : gan nhu thang
# 0.25 do/f = 6.2 do/s  : cong nhe
# 0.50 do/f = 12.5 do/s : cong ro
THRESHOLDS_DEG = [0.10, 0.25, 0.50]


def classify(ob_deg: float, od_deg: float, thr: float) -> str:
    """Phan loai mot doan theo omega truoc/trong (do/frame) va mot nguong."""
    b_big, d_big = abs(ob_deg) > thr, abs(od_deg) > thr
    if b_big and d_big:
        return "CASE_A" if np.sign(ob_deg) == np.sign(od_deg) else "CASE_C"
    if not b_big and d_big:
        return "CASE_B"
    return "STRAIGHT"


def main() -> int:
    out = config.RESULTS_DIR / "demo"
    df = pd.read_csv(out / "segments_omega.csv")
    df = df.dropna(subset=["omega_before", "omega_during"]).copy()
    df["ob_deg"] = np.rad2deg(df.omega_before)
    df["od_deg"] = np.rad2deg(df.omega_during)

    # --- Ghep voi ket qua phan tang da co de lay ti le giu ID cua KF vs EKF ---
    ret = None
    so = config.RESULTS_DIR / "stratified" / "segment_outcomes_DETRAC-all.csv"
    if so.exists():
        s = pd.read_csv(so)
        s = s[s.in_common][["video", "track_id", "seg_id", "occlusion_level",
                            "motion_model", "status"]]
        # seg_id duoc danh so trong CUNG bo file occlusion_segments*.csv ma
        # gt_omega.py doc, nen khoa (video, track_id, seg_id, lvl) khop truc tiep.
        s = s.rename(columns={"occlusion_level": "lvl"})
        ret = s.pivot_table(index=["video", "track_id", "seg_id", "lvl"],
                            columns="motion_model", values="status",
                            aggfunc="first").reset_index()

    rows = []
    for thr in THRESHOLDS_DEG:
        df[f"case_{thr}"] = [classify(a, b, thr) for a, b in zip(df.ob_deg, df.od_deg)]
        c = df[f"case_{thr}"].value_counts()
        n = len(df)
        r = {"nguong_do_per_frame": thr, "n": n}
        for k in ["CASE_A", "CASE_B", "STRAIGHT", "CASE_C"]:
            r[f"%{k}"] = round(c.get(k, 0) / n * 100, 1)
            r[f"n_{k}"] = int(c.get(k, 0))
        rows.append(r)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "case_ab_summary.csv", index=False)
    df.to_csv(out / "segments_classified.csv", index=False)

    print("=" * 78)
    print("DEMO PHAN 2 - TI LE CASE A / CASE B")
    print("=" * 78)
    print(f"\nTong doan phan tich duoc: {len(df)}")
    print(f"\n{'nguong':>8} {'n':>5} {'%CASE_A':>9} {'%CASE_B':>9} {'%STRAIGHT':>11} {'%CASE_C':>9}")
    print("-" * 56)
    for r in rows:
        print(f"{r['nguong_do_per_frame']:>7.2f}o {r['n']:>5} "
              f"{r['%CASE_A']:>8.1f}% {r['%CASE_B']:>8.1f}% "
              f"{r['%STRAIGHT']:>10.1f}% {r['%CASE_C']:>8.1f}%")

    print(f"\n{'nguong':>8}  " + "  ".join(f"{k:>10}" for k in
          ["n_CASE_A", "n_CASE_B", "n_STRAIGHT", "n_CASE_C"]))
    print("-" * 56)
    for r in rows:
        print(f"{r['nguong_do_per_frame']:>7.2f}o  " + "  ".join(
            f"{r['n_' + k]:>10}" for k in ["CASE_A", "CASE_B", "STRAIGHT", "CASE_C"]))

    # --- Ti le giu ID theo nhom (neu ghep duoc) ---
    if ret is not None:
        m = df.merge(ret, on=["video", "track_id", "seg_id", "lvl"], how="left")
        have = m["KF + CV"].notna().sum()
        print(f"\n[ghep voi ket qua phan tang] {have}/{len(m)} doan ghep duoc")
        if have >= 20:
            thr = THRESHOLDS_DEG[1]
            print(f"\nTi le giu duoc ID theo nhom (nguong {thr}o/frame, "
                  f"detector COCO, buffer=30):")
            print(f"  {'nhom':<10} {'n':>5} {'KF + CV':>9} {'EKF + CTRV':>12} {'UKF + CTRV':>12}")
            print("  " + "-" * 50)
            for grp, gg in m.dropna(subset=["KF + CV"]).groupby(f"case_{thr}"):
                vals = [f"{(gg[c] == 'preserved').mean() * 100:>11.1f}%"
                        for c in ["KF + CV", "EKF + CTRV", "UKF + CTRV"]]
                print(f"  {grp:<10} {len(gg):>5} {vals[0]:>9} {vals[1]:>12} {vals[2]:>12}")
            m.to_csv(out / "case_ab_with_retention.csv", index=False)

    print(f"\n  Da luu -> {out / 'case_ab_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
