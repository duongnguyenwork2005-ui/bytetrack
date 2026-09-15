"""
ablation_matrix.py -- (MOC 2, muc 3) Gop ket qua Nhom C thanh MOT bang ablation: moi dong = mot cau hinh,
ghi ro THANH PHAN CO DINH va THANH PHAN THAY DOI, kem chi so COMBINED, giu ID qua che khuat theo tang,
ti le ghep nham, va CI 95 % so voi cau hinh tham chieu cua truc do.

Nguon: results/phase2/<exp>/<cond>/{summary.csv, occlusion_strata.csv, paired_bootstrap_combined.csv,
paired_pairs.csv}, results/phase2/failure_analysis/<exp>/refind_quality.csv.

    python src/ablation_matrix.py --out results/phase2/ablation_matrix.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: (exp, cond, truc, mo ta thanh phan thay doi theo tracker, tracker tham chieu)
AXES = [
    ("expC1_buffer_dev", "c25", "C1 track_buffer",
     lambda t: f"track_buffer = {t.split('-b')[1]} s ({'CV' if t.startswith('cv') else 'EKF'})",
     lambda t: ("cv-b1.2" if t.startswith("cv") else "ekf-b1.2")),
    ("expC3_noise_dev", "c25", "C3 tham so nhieu",
     lambda t: (f"ctrv_std_theta/omega = {t.split('-th')[1].replace('-om', ' / ')}" if t.startswith("ekf")
                else f"cv_std_weight_position/velocity = 1/{t.split('-p')[1].split('-v')[0]} / 1/{t.split('-v')[1]}"),
     lambda t: ("cv-p20-v160" if t.startswith("cv") else "ekf-th0.01-om0.005")),
    ("expC2_recover_dev", "c25", "C2 noi lai track LOST",
     lambda t: ("goc (khong co vong 3)" if t in ("cv", "ekf") else
                f"vong 3 cho track LOST, gate {'Mahalanobis theo P' if t.endswith('maha') else 'IoU 0.05 (doi chung)'}"),
     lambda t: ("cv" if t.startswith("cv") else "ekf")),
]
FIXED = ("detector yolov8n + cache conf0.05 (loc >0.25, lop [2,3,5,7], max_det 300); association ByteTrack "
         "(match_thresh 0.8, fuse_score, 2 vong); 12 video dev; elapsed_mode skip")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2/ablation_matrix.csv")
    a = ap.parse_args()
    rows = []
    for exp, cond, axis, describe, ref_of in AXES:
        d = config.RESULTS_DIR / "phase2" / exp / cond
        if not (d / "summary.csv").exists():
            print(f"[BO QUA] {d}"); continue
        summ = pd.read_csv(d / "summary.csv")
        strata = pd.read_csv(d / "occlusion_strata.csv") if (d / "occlusion_strata.csv").exists() else None
        boot = pd.read_csv(d / "paired_pairs.csv") if (d / "paired_pairs.csv").exists() else None
        occ_boot = pd.read_csv(d / f"bootstrap_DETRAC-all.csv") if (d / "bootstrap_DETRAC-all.csv").exists() else None
        rq = config.RESULTS_DIR / "phase2" / "failure_analysis" / exp / "refind_quality.csv"
        rq = pd.read_csv(rq) if rq.exists() else None
        for r in summ.itertuples():
            t = r.tracker
            ref = ref_of(t)
            row = {"truc": axis, "thi_nghiem": exp, "dieu_kien": cond, "tracker": t,
                   "thanh_phan_co_dinh": FIXED, "thanh_phan_thay_doi": describe(t),
                   "tham_chieu": ref, "la_tham_chieu": t == ref,
                   "HOTA": round(r.HOTA, 4), "DetA": round(r.DetA, 4), "AssA": round(r.AssA, 4),
                   "IDF1": round(r.IDF1, 4), "MOTA": round(r.MOTA, 4), "IDSW": int(r.IDSW),
                   "FP": int(r.FP), "FN": int(r.FN),
                   "preserved": round(float(getattr(r, "preserved", float("nan"))), 4),
                   "id_continuous": round(float(getattr(r, "id_continuous", float("nan"))), 4)}
            if strata is not None:
                st = strata[strata.tracker_short == t]
                for s in ("full", "partial"):
                    v = st[st.stratum == s]
                    row[f"preserved_{s}"] = round(float(v.preserved.iloc[0]), 4) if len(v) else None
                    row[f"n_events_{s}"] = int(v.n_events.iloc[0]) if len(v) else None
            if rq is not None:
                q = rq[(rq.tracker == t) & (rq.cond == cond)]
                row["wrong_refind_rate"] = round(float(q.wrong_refind_rate.iloc[0]), 4) if len(q) else None
                row["n_refind"] = int(q.n_refind.iloc[0]) if len(q) else None
            if boot is not None:
                b = boot[(boot.tracker == t) & (boot.baseline == ref) & (boot.method == "combined_resample_videos")]
                for m in ("HOTA", "IDF1", "IDSW"):
                    bb = b[b.metric == m]
                    if len(bb):
                        row[f"d{m}"] = round(float(bb.diff_combined.iloc[0]), 4)
                        row[f"d{m}_ci"] = f"[{bb.ci_low.iloc[0]:+.4f}, {bb.ci_high.iloc[0]:+.4f}]"
                        row[f"d{m}_ci_chua_0"] = bool(bb.ci_contains_zero.iloc[0])
            if occ_boot is not None:
                ob = occ_boot[occ_boot.tracker.str.endswith(f"-{t}")]
                if len(ob):
                    row["d_preserved"] = round(float(ob["diff"].iloc[0]), 4)
                    row["d_preserved_ci"] = f"[{ob.ci_low.iloc[0]:+.4f}, {ob.ci_high.iloc[0]:+.4f}]"
                    row["d_preserved_baseline"] = str(ob.baseline.iloc[0]).split("-")[-1]
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ("truc", "tracker", "thanh_phan_thay_doi", "HOTA", "dHOTA", "dHOTA_ci", "IDSW",
                        "preserved", "preserved_full", "wrong_refind_rate") if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\n-> {out}  ({len(df)} dong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
