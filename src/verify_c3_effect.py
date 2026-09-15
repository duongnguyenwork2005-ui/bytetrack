"""
verify_c3_effect.py -- (MOC 2, lam ro diem 1) Voi moi cau hinh nhieu cua C3, tra loi BA cau hoi tach bach:

  (A) MUC DO GIONG NHAU cua ket qua: so sanh voi cau hinh THAM CHIEU o ba muc
        1. metric sau lam tron (4 chu so) - muc yeu nhat
        2. metric DAY DU DO CHINH XAC (float64 tu raw_fields.json + combined_metrics.csv)
        3. TOAN BO dau ra tracking: file MOTChallenge co bit-identical khong; neu khong,
           so dong khac, so cap (frame,id) khac, lech toa do lon nhat, va co con la SONG ANH ID
           tren toan sequence khong (track_compare.id_bijection)
  (B) OVERRIDE CO DI VAO BO LOC KHONG: doc lai `tracker_cfg` trong run_meta.json (gia tri .yaml da vao
      cau hinh), va tao tracker tu chinh cau hinh do roi doc thuoc tinh THUC cua bo loc
      (`_std_theta/_std_omega` cho EKF; `_std_weight_position/_std_weight_velocity` cho CV).
  (C) OVERRIDE CO TAC DONG TOI TRANG THAI/COVARIANCE tren DUONG CHAY THUC NGHIEM khong: chay tracker
      tren mot video cache THAT, ghi lai quy dao covariance cua cac track (trung binh P[theta,theta],
      P[cx,cx]) va so voi cau hinh tham chieu. Neu covariance KHAC nhung dau ra tracking GIONG HET
      -> ket luan: tham so DA co hieu luc, nhung KHONG doi quyet dinh ghep.

    python src/verify_c3_effect.py --exp configs/phase2/experiments/expC3_noise_dev.yaml --video MVI_20052
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import tracker_ctrv  # noqa: E402
from run_experiment import load_exp, plan  # noqa: E402
from track_compare import id_bijection, read_mot  # noqa: E402

EKF_ATTRS = ("_std_theta", "_std_omega")
CV_ATTRS = ("_std_weight_position", "_std_weight_velocity")


def filter_params(cfg) -> dict:
    """Tao tracker tu cau hinh THAT roi doc thuoc tinh cua bo loc (khong doc lai .yaml)."""
    tr = run_tracking.make_tracker(cfg)
    kf = tr.kalman_filter
    attrs = EKF_ATTRS if hasattr(kf, "_std_theta") else CV_ATTRS
    return {a: float(getattr(kf, a)) for a in attrs}


def cov_trace(cfg, vd, n_frames: int = 400) -> dict:
    """Chay tracker tren cache that; ghi quy dao covariance cua pool sau moi buoc du doan."""
    tr = run_tracking.make_tracker(cfg)
    is_ctrv = hasattr(tr.kalman_filter, "_std_theta")
    i_th, i_cx = (3, 0) if is_ctrv else (2, 0)     # CTRV: theta=3, cx=0 ; XYAH: a=2, x=0
    p_th, p_cx, n = [], [], 0
    for frame_idx, _t, unreadable, arr in vd.iter_frames():
        if unreadable or n >= n_frames:
            if n >= n_frames:
                break
            continue
        tr.update(det_cache.to_boxes(arr, vd.orig_shape).cpu().numpy(), None)
        pool = list(tr.tracked_stracks) + list(tr.lost_stracks)
        cov = [t.covariance for t in pool if t.covariance is not None]
        if cov:
            p_th.append(float(np.mean([c[i_th, i_th] for c in cov])))
            p_cx.append(float(np.mean([c[i_cx, i_cx] for c in cov])))
        n += 1
    return {"n_frames": n, "P_theta_mean": float(np.mean(p_th)) if p_th else float("nan"),
            "P_theta_sum": float(np.sum(p_th)) if p_th else float("nan"),
            "P_cx_mean": float(np.mean(p_cx)) if p_cx else float("nan"),
            "trace_theta": np.asarray(p_th), "trace_cx": np.asarray(p_cx)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="configs/phase2/experiments/expC3_noise_dev.yaml")
    ap.add_argument("--video", default="MVI_20052", help="video cache that de do quy dao covariance")
    ap.add_argument("--n-frames", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tracker_ctrv.register()
    exp = load_exp(a.exp)
    rows = plan(exp)
    split = exp["split_name"]
    videos = json.loads((config.PROJECT_ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"]
    refs = {"cv": "cv-p20-v160", "ekf": "ekf-th0.01-om0.005"}
    out_rows = []
    cache = {}
    vd = det_cache.open_cache(exp["cache_id"], a.video, 0.25, exp["classes"], exp["max_det"])
    for r in rows:
        fam = "cv" if r["tracker"].startswith("cv") else "ekf"
        ref_name = refs[fam]
        ref = next(x for x in rows if x["tracker"] == ref_name and x["cond"] == r["cond"])
        tb = exp.get("track_buffer_sec")
        ov = dict(r["set"]); ov_ref = dict(ref["set"])
        if tb:
            from time_units import track_buffer_frames
            ov["track_buffer"] = ov_ref["track_buffer"] = track_buffer_frames(float(tb), float(exp["fps"]))
        cfg = run_tracking.load_tracker_cfg(r["yaml"], ov)
        cfg_ref = run_tracking.load_tracker_cfg(ref["yaml"], ov_ref)
        # (B) tham so THUC cua bo loc
        fp, fp_ref = filter_params(cfg), filter_params(cfg_ref)
        # (A) so sanh dau ra tracking tren TOAN BO video cua thi nghiem
        d, d_ref = (config.PROCESSED_DIR / "trackers" / split / x["dir"] for x in (r, ref))
        n_bit, n_bij, n_rows_a, n_rows_match, n_ids_a, n_ids_b = 0, 0, 0, 0, 0, 0
        n_m1 = n_m5 = 0                                  # so dong khop box trong 1 px / 5 px
        for v in videos:
            fa, fb = d / "data" / f"{v}.txt", d_ref / "data" / f"{v}.txt"
            ba = fa.read_bytes() if fa.exists() else b""
            bb = fb.read_bytes() if fb.exists() else b""
            n_bit += int(ba == bb)                       # bit-identical = BYTE cua file, khong chuan hoa gi
            A, B = read_mot(fa), read_mot(fb)
            bij = id_bijection(A, B, tol=1e-6)           # song anh ID co dinh toan sequence (ghep dong theo box)
            n_bij += int(bij.is_bijection)
            n_rows_a += bij.n_rows_a
            n_rows_match += bij.n_matched_rows           # so dong co box TRUNG KHIT (bat ke ID)
            n_ids_a += bij.n_ids_a; n_ids_b += bij.n_ids_b
            from track_compare import match_rows
            for tol, acc in ((1.0, "1"), (5.0, "5")):
                mm = match_rows(A, B, tol=tol)
                k = int(((mm.id_a >= 0) & (mm.id_b >= 0)).sum())
                if acc == "1":
                    n_m1 += k
                else:
                    n_m5 += k
        # metric day du do chinh xac
        cm = config.RESULTS_DIR / "trackeval" / split / r["dir"] / "combined_metrics.csv"
        cm_ref = config.RESULTS_DIR / "trackeval" / split / ref["dir"] / "combined_metrics.csv"
        hota = hota_ref = float("nan")
        if cm.exists() and cm_ref.exists():
            hota = float(pd.read_csv(cm).iloc[0]["HOTA"]); hota_ref = float(pd.read_csv(cm_ref).iloc[0]["HOTA"])
        # (C) quy dao covariance tren cache that
        key = tuple(sorted(fp.items()))
        if key not in cache:
            cache[key] = cov_trace(cfg, vd, a.n_frames)
        ct = cache[key]
        key_ref = tuple(sorted(fp_ref.items()))
        if key_ref not in cache:
            cache[key_ref] = cov_trace(cfg_ref, vd, a.n_frames)
        ct_ref = cache[key_ref]
        same_cov = (len(ct["trace_theta"]) == len(ct_ref["trace_theta"])
                    and np.allclose(ct["trace_theta"], ct_ref["trace_theta"], rtol=0, atol=0))
        out_rows.append({
            "tracker": r["tracker"], "tham_chieu": ref_name, "la_tham_chieu": r["tracker"] == ref_name,
            "tham_so_yaml": json.dumps(r["set"], ensure_ascii=False),
            "tham_so_thuc_cua_bo_loc": json.dumps(fp),
            "override_vao_bo_loc": fp != fp_ref or r["tracker"] == ref_name,
            "HOTA_lam_tron_4": round(hota, 4) == round(hota_ref, 4),
            "HOTA_day_du": f"{hota:.12f}", "HOTA_ref_day_du": f"{hota_ref:.12f}",
            "HOTA_bang_tuyet_doi": hota == hota_ref,
            "n_video_bit_identical": n_bit, "n_video": len(videos), "n_video_song_anh_ID": n_bij,
            "n_dong": n_rows_a, "n_dong_box_trung": n_rows_match,
            "ti_le_dong_box_trung": round(n_rows_match / max(n_rows_a, 1), 4),
            "ti_le_dong_khop_1px": round(n_m1 / max(n_rows_a, 1), 4),
            "ti_le_dong_khop_5px": round(n_m5 / max(n_rows_a, 1), 4),
            "n_track": n_ids_a, "n_track_ref": n_ids_b,
            "P_theta_mean": ct["P_theta_mean"], "P_theta_mean_ref": ct_ref["P_theta_mean"],
            "covariance_giong_het": bool(same_cov),
            "ket_luan": ("tham chieu" if r["tracker"] == ref_name else
                         ("bo loc KHAC (covariance khac) nhung quyet dinh ghep KHONG doi -> dau ra bit-identical"
                          if (not same_cov and n_bit == len(videos)) else
                          ("bo loc KHAC va dau ra KHAC" if not same_cov else
                           "bo loc GIONG HET (tham so hieu dung trung nhau) -> dau ra bit-identical")))})
    df = pd.DataFrame(out_rows)
    out = Path(a.out) if a.out else config.RESULTS_DIR / "phase2" / "expC3_noise_dev" / "c3_effect_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 260)
    print(df[["tracker", "HOTA_day_du", "HOTA_bang_tuyet_doi", "n_video_bit_identical", "ti_le_dong_box_trung",
              "ti_le_dong_khop_1px", "ti_le_dong_khop_5px", "n_track", "n_track_ref", "P_theta_mean",
              "covariance_giong_het"]].to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
