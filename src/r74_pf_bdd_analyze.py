"""Audit BDD R74 raw scores and compare only BDD with completed UA R72."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats

SPLIT = "BDDCOCOVID-val"
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]
BASE = ROOT / "results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/cham"
OUT = ROOT / "results/phase2/r74_pf_bdd"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checked(folder: Path, videos: list[str]) -> tuple[dict, dict]:
    raw = boot_stats.load_raw_fields(folder / "raw_fields.json")
    if set(raw) != set(videos):
        raise ValueError(f"missing raw videos: {folder}")
    row = pd.read_csv(folder / "combined_metrics.csv").iloc[0]
    recomposed = boot_stats.combine(raw, videos)
    error = max(abs(recomposed[m] - float(row[m])) for m in METRICS)
    if error > 1e-9:
        raise ValueError(f"COMBINED mismatch {folder}: {error}")
    return raw, {"error": error, **{m: recomposed[m] for m in METRICS}}


def main() -> None:
    videos = json.loads((ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json").read_text(encoding="utf-8"))["videos"]
    rows, raws, video_rows = [], {}, []
    code_fingerprint = None
    for scale in (100, 200):
        for seed in (0, 1, 2):
            name = f"r74-bdd-full-motion{scale}-s{seed}"
            tr = ROOT / "data/processed/trackers" / SPLIT / name
            meta = json.loads((tr / "run_meta.json").read_text(encoding="utf-8"))
            yaml_hash = sha(ROOT / f"configs/phase2/trackers/r72/motion{scale}.yaml")
            if code_fingerprint is None:
                code_fingerprint = meta["code"]
            elif meta["code"] != code_fingerprint:
                raise ValueError(f"tracker code fingerprint differs across runs: {name}")
            if (meta["cache_id"] != "yolov8n_coco" or meta["seed"] != seed or
                    meta["conf_min"] != .25 or meta["classes"] != [1, 2, 3, 5, 7] or
                    meta["max_det"] != 300 or meta["fps"] != 5 or
                    meta["track_buffer_frames"] != 6 or meta["elapsed_mode"] != "steps" or
                    meta["tracker_cfg"]["pf_roughening_motion_scale"] != scale / 100 or
                    meta["tracker_yaml_sha256_12"] != yaml_hash[:12]):
                raise ValueError(f"tracking metadata mismatch: {name}")
            marker_dir = tr / "video_meta"
            if any(not (marker_dir / f"{v}.json").exists() or not (tr / "data" / f"{v}.txt").exists() for v in videos):
                raise ValueError(f"tracking incomplete: {name}")
            for video in videos:
                marker = json.loads((marker_dir / f"{video}.json").read_text(encoding="utf-8"))
                if (not marker.get("complete") or marker["video"] != video or
                        marker["seed"] != seed or marker["input_fingerprint"] != meta["input_fingerprint"]):
                    raise ValueError(f"tracking marker mismatch: {name}/{video}")
            folder = ROOT / "results/trackeval" / SPLIT / name
            raw, score = load_checked(folder, videos)
            raws[scale, seed] = raw
            rows.append({"dataset": "BDD100K val", "tracker": f"PF motion {scale/100:.1f}", "seed": seed,
                         "run": name, **score})
            detail = pd.read_csv(folder / "per_video_metrics.csv")
            if set(detail.video) != set(videos):
                raise ValueError(f"per-video mismatch: {name}")
            video_rows.extend({"video": r.video, "motion_scale": scale / 100, "seed": seed,
                               "HOTA": r.HOTA, "IDF1": r.IDF1, "IDSW": r.IDSW}
                              for r in detail.itertuples())
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    pd.DataFrame(video_rows).to_csv(OUT / "per_video.csv", index=False)
    summary = per_seed.groupby("tracker", sort=False)[METRICS].mean().reset_index()
    ref_rows = []
    for kind in ("cv", "ekf"):
        name = f"r61-yolov8n_coco-c25-{kind}"
        folder = BASE / name
        marker = json.loads((folder / "cham.xong.json").read_text(encoding="utf-8"))
        if marker["so_video"] != 200 or marker["evaluator"] != "mot_crowd":
            raise ValueError(f"R61 reference scope mismatch: {name}")
        for filename, expected in marker["artifact"].items():
            if sha(folder / filename) != expected:
                raise ValueError(f"R61 reference hash mismatch: {name}/{filename}")
        _, score = load_checked(folder, videos)
        ref_rows.append({"tracker": kind.upper(), **{m: score[m] for m in METRICS}})
    summary = pd.concat([pd.DataFrame(ref_rows), summary], ignore_index=True)
    summary.insert(0, "dataset", "BDD100K val")
    summary.to_csv(OUT / "summary_bdd.csv", index=False)

    # Shared video resamples; each seed is recomposed separately, then the three
    # COMBINED metrics are averaged. This matches the primary point estimate.
    stacks = {(scale, seed): boot_stats._stack(raws[scale, seed], videos)
              for scale in (100, 200) for seed in (0, 1, 2)}
    rng = np.random.default_rng(0)
    boots = {m: np.empty(5000) for m in METRICS}
    for i in range(5000):
        idx = rng.integers(0, len(videos), size=len(videos))
        scores = {(scale, seed): boot_stats.combine_stacked({f: st[f][idx] for f in st})
                  for (scale, seed), st in stacks.items()}
        for m in METRICS:
            boots[m][i] = np.mean([scores[200, s][m] - scores[100, s][m] for s in (0, 1, 2)])
    ci_rows = []
    for m in METRICS:
        a = float(summary.loc[summary.tracker == "PF motion 2.0", m].iloc[0])
        b = float(summary.loc[summary.tracker == "PF motion 1.0", m].iloc[0])
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        ci_rows.append({"dataset": "BDD100K val", "metric": m, "motion2": a, "motion1": b,
                        "difference": a - b, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(ci_rows).to_csv(OUT / "paired_intervals_bdd.csv", index=False)
    ua_summary = pd.read_csv(ROOT / "results/phase2/kc48_r72_pf_dev/analysis/summary.csv")
    ua_ci = pd.read_csv(ROOT / "results/phase2/kc48_r72_pf_dev/analysis/paired_intervals.csv")
    report = {"BDD": {"n_videos": len(videos), "n_runs": len(rows),
                      "max_recomposition_error": float(per_seed.error.max()),
                      "summary": summary.to_dict(orient="records"), "paired": ci_rows},
              "UA_DETRAC_R72": {"summary_motion100_motion200": ua_summary[ua_summary.tracker.isin(["motion100", "motion200"])].to_dict(orient="records"),
                                "paired_intervals_source": "results/phase2/kc48_r72_pf_dev/analysis/paired_intervals.csv",
                                "paired_intervals": ua_ci.to_dict(orient="records")}}
    (OUT / "two_dataset_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(ci_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
