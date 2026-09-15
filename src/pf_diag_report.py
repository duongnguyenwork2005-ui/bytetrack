"""
pf_diag_report.py -- (MOC 2, muc 4) Tong hop chan doan Particle Filter tu pf_diag/<video>.csv cua cac lan
chay PF (run_experiment voi pf_diag: true).

DINH NGHIA (xem PFCTRV.enable_diag trong src/pf_ctrv.py): `ess_ratio` = ESS/N voi ESS = 1/sum(w^2), trong so
DA CHUAN HOA SAU khi nhan likelihood, do TRUOC resample; `n_unique` = so gia tri cx khac nhau, do SAU
resample; `spread_theta` la do lech chuan VONG. Ket qua chi ap dung cho bootstrap PF nay voi cau hinh da
chay (ghi trong cac cot pf_*): resample khi ESS < 0.5 N, uoc luong `mean`, roughening = 0, Q/R dung chung
voi EKF; seed 0/1/2, RNG moi track = SeedSequence([seed, hash(video), track_id]).

Dau hieu suy giam chat luong:
    ess_ratio  : ESS/N truoc resample (thap = trong so tap trung vao it particle)
    resampled  : ti le update phai resample (ESS < 0.5 N)
    n_unique   : so particle KHAC NHAU sau resample / N (thap = mat da dang - sample impoverishment)
    spread_xy  : do lech chuan vi tri cua dam may (px) - theo so buoc mat (n_lost)
    spread_theta: do lech chuan vong cua huong (rad)
    degenerate : ti le update suy bien (khong particle nao gan quan sat -> dat lai vi tri)
    init_motion: so lan khoi tao 2 khung hinh (reseed v, theta)
Bao cao theo (exp, cond, tracker N, seed) + theo n_lost (0 = dang Tracked, >0 = vua noi lai sau n buoc mat).

    python src/pf_diag_report.py --exp configs/phase2/experiments/expA_detrac_dev.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from run_experiment import load_exp, plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    exp = load_exp(a.exp)
    out = Path(a.out) if a.out else config.RESULTS_DIR / "phase2" / "pf_diag" / exp["name"]
    out.mkdir(parents=True, exist_ok=True)
    rows, by_lost = [], []
    for r in plan(exp):
        if r["tracker_type"] != "bytetrack_pf_ctrv":
            continue
        d = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"] / "pf_diag"
        files = sorted(d.glob("*.csv")) if d.is_dir() else []
        if not files:
            continue
        df = pd.concat([pd.read_csv(f).assign(video=f.stem) for f in files], ignore_index=True)
        import json as _json
        meta_f = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"] / "run_meta.json"
        cfg = _json.loads(meta_f.read_text(encoding="utf-8")).get("tracker_cfg", {}) if meta_f.exists() else {}
        upd = df[df.kind == "update"]
        n = int(upd.n_unique.max()) if len(upd) else 0
        N = int(r.get("set", {}).get("pf_num_particles", 0)) or n
        rec = {"cond": r["cond"], "tracker": r["tracker"], "seed": r["seed"], "n_update": len(upd), "N_particles": N,
               "pf_resample_ess_ratio": cfg.get("pf_resample_ess_ratio"), "pf_estimate": cfg.get("pf_estimate"),
               "pf_roughening": cfg.get("pf_roughening"), "pf_degenerate_maha2": cfg.get("pf_degenerate_maha2"),
               "track_buffer": cfg.get("track_buffer"),
               "ess_ratio_mean": upd.ess_ratio.mean(), "ess_ratio_p10": upd.ess_ratio.quantile(0.10),
               "frac_resampled": upd.resampled.mean(), "frac_ess_lt_0.2": (upd.ess_ratio < 0.2).mean(),
               "n_unique_ratio_mean": (upd.n_unique / N).mean() if N else np.nan,
               "n_unique_ratio_p10": (upd.n_unique / N).quantile(0.10) if N else np.nan,
               "frac_degenerate": upd.degenerate.mean(), "spread_xy_mean": upd.spread_xy.mean(),
               "spread_theta_mean": upd.spread_theta.mean(), "spread_v_mean": upd.spread_v.mean(),
               "n_init_motion": int((df.kind == "init_motion").sum()),
               "init_motion_spread_theta": df[df.kind == "init_motion"].spread_theta.mean()}
        rows.append(rec)
        upd = upd.assign(n_lost_bin=pd.cut(upd.n_lost, [-1, 0, 2, 5, 10, 30, 10_000],
                                           labels=["0", "1-2", "3-5", "6-10", "11-30", ">30"]))
        g = upd.groupby("n_lost_bin", observed=True).agg(n=("ess_ratio", "size"), ess_ratio=("ess_ratio", "mean"),
                                                        n_unique_ratio=("n_unique", lambda s: (s / N).mean() if N else np.nan),
                                                        spread_xy=("spread_xy", "mean"), spread_theta=("spread_theta", "mean"),
                                                        degenerate=("degenerate", "mean")).reset_index()
        g.insert(0, "seed", r["seed"]); g.insert(0, "tracker", r["tracker"]); g.insert(0, "cond", r["cond"])
        by_lost.append(g)
    if not rows:
        print("khong co pf_diag"); return 1
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    bl = pd.concat(by_lost, ignore_index=True)
    bl.to_csv(out / "by_n_lost.csv", index=False)
    # trung binh + SD giua seed
    agg = summ.groupby(["cond", "tracker"]).agg(
        N=("N_particles", "first"), n_seeds=("seed", "nunique"),
        ess_ratio=("ess_ratio_mean", "mean"), ess_ratio_sd=("ess_ratio_mean", "std"),
        frac_resampled=("frac_resampled", "mean"), n_unique_ratio=("n_unique_ratio_mean", "mean"),
        n_unique_p10=("n_unique_ratio_p10", "mean"), frac_degenerate=("frac_degenerate", "mean"),
        spread_xy=("spread_xy_mean", "mean"), spread_theta=("spread_theta_mean", "mean"),
        n_init_motion=("n_init_motion", "mean")).reset_index()
    agg.to_csv(out / "summary_by_tracker.csv", index=False)
    pd.set_option("display.width", 250)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTheo so buoc mat (n_lost) - trung binh cac seed:")
    print(bl.groupby(["cond", "tracker", "n_lost_bin"], observed=True)[["n", "ess_ratio", "n_unique_ratio", "spread_xy", "spread_theta", "degenerate"]]
          .mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
