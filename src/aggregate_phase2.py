"""
aggregate_phase2.py -- Tong hop ket qua mot thi nghiem Giai doan 2 (sau run_experiment.py).

Voi moi dieu kien (A/B):
  summary.csv        moi tracker: HOTA/DetA/AssA/IDF1/MOTA/FP/FN/IDSW COMBINED_SEQ cua TrackEval
                     (PF: trung binh +- SD GIUA CAC SEED cua chi so COMBINED), thoi gian bo loc (ms/frame),
                     ti le giu ID qua che khuat (preserved, id_continuous) tu occlusion_eval.
  per_video.csv      chi so tung video (PF: trung binh theo seed TRONG TUNG VIDEO, kem sd_seed).
  paired_bootstrap.csv
                     chenh lech so voi baseline cua TRUNG BINH THEO VIDEO (HOTA, IDF1, AssA, DetA, IDSW)
                     + khoang tin cay 95% bootstrap theo VIDEO (5000 lan, seed 0), ghep cap (cung video
                     cho ca hai tracker). PF: dung trung binh theo seed trong video (seed cua cung video
                     KHONG phai video doc lap). LUU Y: day la trung binh don gian cua chi so tung video,
                     KHAC gia tri COMBINED_SEQ (TrackEval gop thong ke) - ghi ro trong bao cao.
  timing.csv         predict / assoc / update ms/frame tung tracker (trung binh tren video, PF theo seed).
  paired_bootstrap_combined.csv   (MOC 2, BANG CHINH) chenh lech cua chi so COMBINED (tai to hop thong ke du
                     y het TrackEval - boot_stats.py) so voi baseline + CI 95 % bootstrap theo video (5000, seed 0),
                     ghep cap. NHAT QUAN voi summary.csv. PF: thong ke du trung binh theo seed trong video.
  paired_pairs.csv   (MOC 2) cac cap so sanh khai bao trong exp `pairs: [[a, b], ...]` (a - b), ca hai cach
                     (combined va mean-video); moi cap ghi ro thanh phan duoc kiem soat (cot `note` neu co).
  occlusion_strata.csv  ti le giu ID theo tang che khuat (bang chung GT).

    python src/aggregate_phase2.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml
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
import boot_stats  # noqa: E402

METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN"]
BOOT_METRICS = ["HOTA", "IDF1", "AssA", "DetA", "IDSW"]


def load_raw_by_tracker(rr: list[dict], split: str) -> dict:
    """tracker -> raw_fields (PF: trung binh thong ke du theo seed trong tung video). Thieu file -> bo qua."""
    by = {}
    for r in rr:
        f = config.RESULTS_DIR / "trackeval" / split / r["dir"] / "raw_fields.json"
        if f.exists():
            by.setdefault(r["tracker"], []).append(boot_stats.load_raw_fields(f))
    return {t: boot_stats.average_seeds(v) for t, v in by.items()}


def combined_boot_rows(raw: dict, a: str, b: str, videos: list[str], n_boot: int, cond: str) -> list[dict]:
    res = boot_stats.paired_bootstrap(raw[a], raw[b], videos, n_boot=n_boot, seed=0, metrics=BOOT_METRICS)
    return [{"cond": cond, "tracker": a, "baseline": b, "metric": m, "a_combined": v["a"], "b_combined": v["b"],
             "diff_combined": v["diff"], "ci_low": v["ci_low"], "ci_high": v["ci_high"], "n_videos": len(videos),
             "ci_contains_zero": v["ci_low"] <= 0 <= v["ci_high"], "n_boot": n_boot, "seed": 0, "method": "combined_resample_videos"}
            for m, v in res.items()]


def video_bootstrap_diff(a: pd.Series, b: pd.Series, n_boot: int = 5000, seed: int = 0):
    """a, b: chi so tung video (index = video, cung thu tu). Chenh lech mean(a) - mean(b) + CI 95%."""
    a, b = a.to_numpy(float), b.to_numpy(float)
    n = len(a)
    rng = np.random.default_rng(seed)
    d = a - b
    obs = float(d.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    exp = load_exp(a.exp)
    rows = plan(exp)
    split = exp["split_name"]
    out_root = config.RESULTS_DIR / "phase2" / exp["name"]
    base_name = exp.get("baseline")
    pd.set_option("display.width", 220)

    for cond in exp["conditions"]:
        rr = [r for r in rows if r["cond"] == cond]
        per_video, combined, timing = [], [], []
        for r in rr:
            d = config.RESULTS_DIR / "trackeval" / split / r["dir"]
            if not (d / "per_video_metrics.csv").exists():
                print(f"  [MISS] {d}"); continue
            pv = pd.read_csv(d / "per_video_metrics.csv"); pv["tracker"] = r["tracker"]; pv["seed"] = r["seed"]
            per_video.append(pv)
            c = pd.read_csv(d / "combined_metrics.csv").iloc[0].to_dict(); c.update(tracker=r["tracker"], seed=r["seed"])
            combined.append(c)
            s = config.PROCESSED_DIR / "trackers" / split / r["dir"] / "summary.csv"
            if s.exists():
                t = pd.read_csv(s); t["tracker"] = r["tracker"]; t["seed"] = r["seed"]; timing.append(t)
        if not per_video:
            print(f"[{cond}] chua co ket qua"); continue
        pv = pd.concat(per_video, ignore_index=True)
        cb = pd.DataFrame(combined)
        out = out_root / cond
        out.mkdir(parents=True, exist_ok=True)

        # --- per_video: PF trung binh theo seed trong video ---
        pv_mean = pv.groupby(["tracker", "video"])[METRICS].mean().reset_index()
        pv_sd = pv.groupby(["tracker", "video"])[["HOTA", "IDF1"]].std().reset_index().rename(
            columns={"HOTA": "HOTA_sd_seed", "IDF1": "IDF1_sd_seed"})
        pv_mean = pv_mean.merge(pv_sd, on=["tracker", "video"], how="left")
        pv_mean.to_csv(out / "per_video.csv", index=False)

        # --- summary (COMBINED, PF mean +- sd theo seed) ---
        g = cb.groupby("tracker")
        summ = g[METRICS].mean()
        for m in ("HOTA", "IDF1", "AssA"):
            summ[f"{m}_sd_seed"] = g[m].std()
        summ["n_seeds"] = g["seed"].nunique()
        if timing:
            tm = pd.concat(timing).groupby("tracker")[["predict_ms_per_frame", "assoc_ms_per_frame", "update_ms_per_frame"]].mean()
            summ = summ.join(tm)
            tm.reset_index().to_csv(out / "timing.csv", index=False)
        # occlusion outcomes (neu co)
        eo = out / f"event_outcomes_{split}.csv"
        if eo.exists():
            e = pd.read_csv(eo)
            e = e[e.eligible]
            e["tracker_short"] = e["tracker"].str.replace(f"p2-{exp['name']}-{cond}-", "", regex=False).str.replace(r"-s\d+$", "", regex=True)
            occ = e.groupby("tracker_short").agg(n_events=("outcome", "size"),
                                                 preserved=("outcome", lambda s: float((s == "preserved").mean())),
                                                 switched=("outcome", lambda s: float((s == "switched").mean())),
                                                 lost_after=("outcome", lambda s: float((s == "lost_after").mean())),
                                                 id_continuous=("id_continuous", "mean"))
            summ = summ.join(occ, how="left")
            # Tang che khuat theo BANG CHUNG GT (khong suy dien): partial (0.10 <= ratio < 0.90),
            # full (>= 0.90, chi UA-DETRAC), gap-occluded_adjacent, gap-unknown (chua xac dinh)
            if "gap_evidence" in e.columns:
                ge = e["gap_evidence"].fillna("").astype(str).replace("", "unknown")
                e["stratum"] = np.where(e["has_gap"], "gap-" + ge, np.where(e["has_full"], "full", "partial"))
            else:
                e["stratum"] = np.where(e["has_full"], "full", "partial")
            strata = (e.groupby(["tracker_short", "stratum"])
                      .agg(n_events=("outcome", "size"), preserved=("outcome", lambda s: float((s == "preserved").mean())),
                           switched=("outcome", lambda s: float((s == "switched").mean())),
                           lost_after=("outcome", lambda s: float((s == "lost_after").mean())),
                           no_match_before=("outcome", lambda s: float((s == "no_match_before").mean())),
                           id_continuous=("id_continuous", "mean")).reset_index())
            strata.to_csv(out / "occlusion_strata.csv", index=False)
        order = [t["name"] for t in exp["trackers"] if t["name"] in summ.index]
        summ = summ.loc[order].reset_index()
        summ.to_csv(out / "summary.csv", index=False)

        # --- (MOC 2) bootstrap COMBINED (bang chinh) vs baseline + cac cap khai bao ---
        raw = load_raw_by_tracker(rr, split)
        vids = sorted(pv.video.unique())
        n_boot = a.n_boot
        cb_rows = []
        if base_name in raw:
            for t in order:
                if t != base_name and t in raw:
                    cb_rows += combined_boot_rows(raw, t, base_name, vids, n_boot, cond)
        if cb_rows:
            pd.DataFrame(cb_rows).to_csv(out / "paired_bootstrap_combined.csv", index=False)
        pair_rows = []
        for pr in exp.get("pairs", []) or []:
            x, y = pr[0], pr[1]
            note = pr[2] if len(pr) > 2 else ""
            if x in raw and y in raw:
                for row in combined_boot_rows(raw, x, y, vids, n_boot, cond):
                    row["note"] = note; pair_rows.append(row)
            if x in set(pv_mean.tracker) and y in set(pv_mean.tracker):
                bx = pv_mean[pv_mean.tracker == x].set_index("video"); by_ = pv_mean[pv_mean.tracker == y].set_index("video")
                common = sorted(set(bx.index) & set(by_.index))
                for m in BOOT_METRICS:
                    obs, lo, hi = video_bootstrap_diff(bx.loc[common, m], by_.loc[common, m], n_boot)
                    pair_rows.append({"cond": cond, "tracker": x, "baseline": y, "metric": m, "diff_combined": obs,
                                      "ci_low": lo, "ci_high": hi, "n_videos": len(common), "ci_contains_zero": lo <= 0 <= hi,
                                      "n_boot": n_boot, "seed": 0, "method": "mean_of_videos", "note": note})
        if pair_rows:
            pd.DataFrame(pair_rows).to_csv(out / "paired_pairs.csv", index=False)

        # --- paired bootstrap theo video vs baseline (TRUNG BINH THEO VIDEO - tham khao) ---
        boot = []
        if base_name in set(pv_mean.tracker):
            b = pv_mean[pv_mean.tracker == base_name].set_index("video")
            for t in order:
                if t == base_name:
                    continue
                x = pv_mean[pv_mean.tracker == t].set_index("video")
                common = sorted(set(x.index) & set(b.index))
                for m in BOOT_METRICS:
                    obs, lo, hi = video_bootstrap_diff(x.loc[common, m], b.loc[common, m], a.n_boot)
                    boot.append({"cond": cond, "tracker": t, "baseline": base_name, "metric": m,
                                 "diff_video_mean": obs, "ci_low": lo, "ci_high": hi, "n_videos": len(common),
                                 "ci_contains_zero": lo <= 0 <= hi, "n_boot": a.n_boot, "seed": 0})
            pd.DataFrame(boot).to_csv(out / "paired_bootstrap.csv", index=False)

        print("\n" + "=" * 100)
        print(f"{exp['name']}  dieu kien {cond} (conf > {exp['conditions'][cond]['conf_min']})  split {split}  "
              f"{pv.video.nunique()} video")
        print("=" * 100)
        cols = ["tracker"] + METRICS + [c for c in ("HOTA_sd_seed", "n_seeds", "predict_ms_per_frame",
                                                    "n_events", "preserved", "id_continuous") if c in summ.columns]
        print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if cb_rows:
            bd = pd.DataFrame(cb_rows)
            print(f"\n  [BANG CHINH] Chenh lech chi so COMBINED so voi baseline {base_name} (bootstrap {n_boot} lan theo video, "
                  f"tai to hop nhu TrackEval; CI 95%):")
            piv = bd.pivot(index="tracker", columns="metric", values="diff_combined").loc[[t for t in order if t in set(bd.tracker)]]
            lo = bd.pivot(index="tracker", columns="metric", values="ci_low").loc[piv.index]
            hi = bd.pivot(index="tracker", columns="metric", values="ci_high").loc[piv.index]
            for t in piv.index:
                s_ = "  ".join(f"{m}: {piv.loc[t, m]:+.4f} [{lo.loc[t, m]:+.4f}, {hi.loc[t, m]:+.4f}]" for m in BOOT_METRICS)
                print(f"    {t:<10} {s_}")
        if pair_rows:
            pdp = pd.DataFrame(pair_rows); pdp = pdp[pdp.method == "combined_resample_videos"]
            print("\n  Cac cap so sanh (COMBINED, a - b):")
            for (x, y), g in pdp.groupby(["tracker", "baseline"], sort=False):
                s_ = "  ".join(f"{r.metric}: {r.diff_combined:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]" for r in g.itertuples())
                print(f"    {x} - {y}: {s_}" + (f"   ({g.note.iloc[0]})" if g.note.iloc[0] else ""))
        if boot:
            bd = pd.DataFrame(boot)
            print("\n  [tham khao] Chenh lech TRUNG BINH THEO VIDEO so voi baseline (CI 95% bootstrap theo video):")
            piv = bd.pivot(index="tracker", columns="metric", values="diff_video_mean").loc[[t for t in order if t != base_name]]
            lo = bd.pivot(index="tracker", columns="metric", values="ci_low").loc[piv.index]
            hi = bd.pivot(index="tracker", columns="metric", values="ci_high").loc[piv.index]
            for t in piv.index:
                s = "  ".join(f"{m}: {piv.loc[t, m]:+.4f} [{lo.loc[t, m]:+.4f}, {hi.loc[t, m]:+.4f}]" for m in BOOT_METRICS)
                print(f"    {t:<8} {s}")
        if (out / "occlusion_strata.csv").exists():
            st = pd.read_csv(out / "occlusion_strata.csv")
            print("\n  Giu ID (preserved) theo tang che khuat - bang chung GT; gap-unknown = chua xac dinh nguyen nhan:")
            piv = st.pivot(index="tracker_short", columns="stratum", values="preserved")
            print(piv.loc[[t for t in order if t in set(st.tracker_short)]].to_string(float_format=lambda v: f"{v:.3f}"))
            print("  n_events:", st.groupby("stratum").n_events.first().to_dict())
        print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
