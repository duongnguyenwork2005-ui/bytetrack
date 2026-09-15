"""
run_experiment.py -- Chay MOT thi nghiem so sanh (Giai doan 2) tu file YAML:
    tracking (tu cache) x dieu kien x tracker x seed  ->  TrackEval  ->  occlusion_eval  ->  aggregate

File YAML (vd configs/phase2/experiments/groupA_detrac_dev.yaml):
    name: groupA_detrac_dev
    dataset: detrac            split_name: DETRAC-all       part: train
    manifest: results/phase2/manifests/detrac_dev.json
    cache_id: yolov8n-coco_conf0.05_all
    classes: [2, 3, 5, 7]      max_det: 300      fps: 25     track_buffer_sec: 1.2    elapsed_mode: skip
    conditions: {A: {conf_min: 0.25}, B: {conf_min: 0.10}}
    trackers:
      - {name: cv,  yaml: bytetrack.yaml}
      - {name: pf100, yaml: configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml, seeds: [0, 1, 2]}
    baseline: cv
    occlusion: {part: train, context_s: 1.2, gaps: false, tag: null, n_boot: 5000}

Ten thu muc tracker: p2-<name>-<cond>-<tracker>[-s<seed>] trong data/processed/trackers/<split>/.
Ket qua: results/phase2/<name>/runs.csv, <cond>/{events,event_outcomes,paired,bootstrap}_*.csv,
         results/trackeval/<split>/p2-.../{per_video,combined}_metrics.csv, va bang tong hop
         (aggregate_phase2.py).

    python src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml
    python src/run_experiment.py --exp ... --stage track      # chi tracking
    python src/run_experiment.py --exp ... --stage eval       # trackeval + occlusion
    python src/run_experiment.py --exp ... --dry-run          # in ke hoach
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import run_trackeval  # noqa: E402
from time_units import track_buffer_frames  # noqa: E402

PY = sys.executable
ROOT = config.PROJECT_ROOT


def load_exp(path: str | Path) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.setdefault("conditions", {"A": {"conf_min": 0.25}})
    d.setdefault("classes", [2, 3, 5, 7])
    d.setdefault("max_det", 300)
    d.setdefault("elapsed_mode", "skip")
    d.setdefault("occlusion", {})
    d["_path"] = str(path)
    return d


def tracker_dir_name(exp: str, cond: str, tracker: str, seed: int | None) -> str:
    return f"p2-{exp}-{cond}-{tracker}" + (f"-s{seed}" if seed is not None else "")


def plan(exp: dict) -> list[dict]:
    """Danh sach (cond, tracker, seed, dir) can chay."""
    registry = run_tracking.tracker_registry()
    rows = []
    for cond, cc in exp["conditions"].items():
        for t in exp["trackers"]:
            cfg = run_tracking.load_tracker_cfg(t["yaml"], t.get("set"))
            stochastic = cfg.tracker_type in run_tracking.STOCHASTIC_TRACKERS
            seeds = t.get("seeds", [None]) if stochastic else [None]
            if stochastic and seeds == [None]:
                seeds = [int(getattr(cfg, "pf_seed", 0))]
            for s in seeds:
                rows.append({"cond": cond, "tracker": t["name"], "yaml": t["yaml"], "set": t.get("set") or {},
                             "seed": s, "conf_min": float(cc["conf_min"]),
                             "dir": tracker_dir_name(exp["name"], cond, t["name"], s),
                             "tracker_type": cfg.tracker_type})
    return rows


def stage_track(exp: dict, rows: list[dict], videos: list[str], skip_existing: bool):
    fps = float(exp["fps"])
    dt = 1.0 / fps
    registry = run_tracking.tracker_registry()
    out_rows = []
    for r in rows:
        overrides = dict(r["set"])
        if exp.get("track_buffer_sec") is not None:
            overrides["track_buffer"] = track_buffer_frames(float(exp["track_buffer_sec"]), fps)
        cfg = run_tracking.load_tracker_cfg(r["yaml"], overrides)
        out_root = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"]
        out_root.mkdir(parents=True, exist_ok=True)
        yaml_sha = hashlib.sha256(Path(run_tracking.resolve_tracker_yaml(r["yaml"])).read_bytes()).hexdigest()[:12]
        (out_root / "run_meta.json").write_text(json.dumps({
            "experiment": exp["name"], "cond": r["cond"], "tracker": r["tracker"], "seed": r["seed"],
            "cache_id": exp["cache_id"], "tracker_yaml": r["yaml"], "tracker_yaml_sha256_12": yaml_sha,
            "tracker_cfg": vars(cfg), "overrides": overrides, "conf_min": r["conf_min"],
            "classes": exp["classes"], "max_det": exp["max_det"], "fps": fps, "dt_nominal_s": dt,
            "track_buffer_frames": int(cfg.track_buffer), "track_buffer_sec": round(int(cfg.track_buffer) / fps, 4),
            "elapsed_mode": exp["elapsed_mode"], "git_commit": det_cache.git_commit(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2, default=str), encoding="utf-8")
        t_all = 0.0
        summary = []
        for v in videos:
            f = out_root / "data" / f"{v}.txt"
            if skip_existing and f.exists():
                continue
            vd = det_cache.open_cache(exp["cache_id"], v, r["conf_min"], exp["classes"], exp["max_det"])
            t0 = time.time()
            df, df_t = run_tracking.run_video_to_dir(out_root, v, vd, cfg, dt, exp["elapsed_mode"], r["seed"], registry)
            sec = time.time() - t0
            t_all += sec
            summary.append({"video": v, "n_frames": len(vd), "n_boxes": len(df),
                            "n_tracks": int(df["id"].nunique()) if len(df) else 0,
                            "predict_ms_per_frame": round(float(df_t.t_predict_ms.mean()), 4),
                            "assoc_ms_per_frame": round(float(df_t.t_assoc_ms.mean()), 4),
                            "update_ms_per_frame": round(float(df_t.t_update_ms.mean()), 4), "seconds": round(sec, 2)})
        if summary:
            old = out_root / "summary.csv"
            s = pd.DataFrame(summary)
            if skip_existing and old.exists():
                s = pd.concat([pd.read_csv(old), s]).drop_duplicates("video", keep="last")
            s.to_csv(old, index=False)
        print(f"  [track] {r['dir']:<40} {len(summary):>3} video  {t_all:6.1f}s")
        out_rows.append({**r, "seconds_track": round(t_all, 1)})
    return out_rows


def stage_eval(exp: dict, rows: list[dict], videos: list[str]):
    split = exp["split_name"]
    dataset_cls = "mot_crowd" if split.startswith("BDD-") else "mot"
    # --- TrackEval tung tracker ---
    for r in rows:
        out_dir = config.RESULTS_DIR / "trackeval" / split / r["dir"]
        if (out_dir / "combined_metrics.csv").exists() and (out_dir / "per_video_metrics.csv").exists():
            continue
        results, _ = run_trackeval.run(split, r["dir"], videos, dataset_cls)
        df = run_trackeval.flatten_results(results, r["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        df[df.video != "COMBINED_SEQ"].sort_values("video").to_csv(out_dir / "per_video_metrics.csv", index=False)
        df[df.video == "COMBINED_SEQ"].to_csv(out_dir / "combined_metrics.csv", index=False)
        c = df[df.video == "COMBINED_SEQ"].iloc[0]
        print(f"  [eval ] {r['dir']:<40} HOTA {c.HOTA:.4f} IDF1 {c.IDF1:.4f} AssA {c.AssA:.4f} IDSW {int(c.IDSW)}")
    # --- occlusion_eval theo dieu kien (baseline dau tien) ---
    occ = exp.get("occlusion", {})
    for cond in exp["conditions"]:
        rr = [r for r in rows if r["cond"] == cond]
        base = [r["dir"] for r in rr if r["tracker"] == exp.get("baseline")]
        others = [r["dir"] for r in rr if r["tracker"] != exp.get("baseline")]
        if not base:
            print(f"  [occl ] {cond}: khong co baseline {exp.get('baseline')!r}, bo qua"); continue
        out = config.RESULTS_DIR / "phase2" / exp["name"] / cond
        out.mkdir(parents=True, exist_ok=True)
        if (out / f"bootstrap_{split}.csv").exists():
            print(f"  [occl ] {cond}: da co, bo qua"); continue
        cmd = [PY, "src/occlusion_eval.py", "--split-name", split, "--trackers", *base, *others,
               "--out-dir", str(out), "--n-boot", str(occ.get("n_boot", 5000)), "--videos", *videos]
        if occ.get("part"):
            cmd += ["--part", occ["part"]]
        if occ.get("tag"):
            cmd += ["--tag", occ["tag"]]
        if occ.get("fps"):
            cmd += ["--fps", str(occ["fps"])]
        if occ.get("context_s") is not None:
            cmd += ["--context-s", str(occ["context_s"])]
        if occ.get("gaps"):
            cmd += ["--gaps"]
        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (out / "occlusion_eval.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
        print(f"  [occl ] {cond}: exit {res.returncode} ({time.time() - t0:.0f}s) -> {out}")
        if res.returncode != 0:
            print(res.stderr[-1500:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--stage", choices=["all", "track", "eval", "aggregate"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-skip", action="store_true", help="chay lai ca video da co")
    a = ap.parse_args()
    exp = load_exp(a.exp)
    videos = json.loads((ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"] if exp.get("manifest") \
        else exp["videos"]
    rows = plan(exp)
    print(f"[exp] {exp['name']}: split={exp['split_name']} cache={exp['cache_id']} {len(videos)} video, "
          f"{len(rows)} lan chay tracker ({len(exp['conditions'])} dieu kien)")
    out_dir = config.RESULTS_DIR / "phase2" / exp["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.yaml").write_text(Path(a.exp).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "videos.json").write_text(json.dumps(videos, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)
    if a.dry_run:
        print(pd.DataFrame(rows)[["cond", "tracker", "seed", "conf_min", "dir"]].to_string(index=False)); return 0
    if a.stage in ("all", "track"):
        stage_track(exp, rows, videos, skip_existing=not a.no_skip)
    if a.stage in ("all", "eval"):
        stage_eval(exp, rows, videos)
    if a.stage in ("all", "aggregate"):
        subprocess.run([PY, "src/aggregate_phase2.py", "--exp", a.exp], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
