"""Paired video-level statistics from the Nano R86 run_meta records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.run_root.parent / "sample/sample_manifest.json").read_text())
    pairs = []
    for row in manifest["rows"]:
        ds, name = row["dataset"], row["name"]
        for tracker in ("cv", "ekf"):
            base = args.run_root / ds / name / tracker
            ms = []
            for v in ("A", "B"):
                p = base / v / "run_meta.json"
                if not p.exists():
                    ms = []
                    break
                x = json.loads(p.read_text())
                d = x["dem"]
                if not (x["hoan_tat"] and d["n_frame_da_xu_ly"] == 150 and d["n_unreadable"] == d["n_truncated"] == 0):
                    raise ValueError(p)
                t = x["thoi_gian"]
                s = t["tom_tat_sau_warmup"]
                ms.append({"fps": t["fps_xu_ly"], "fps_steady": s["fps_on_dinh"], "decode_ms": s["decode_ms"]["trung_vi"],
                           "latency_p90_ms": s["tre_ms"]["p90"], "pre_ms": s["pre_ms"]["trung_vi"],
                           "infer_ms": s["infer_ms"]["trung_vi"], "post_ms": s["post_ms"]["trung_vi"],
                           "tracker_ms": s["t_update_ms"]["trung_vi"], "init_s": t["khoi_tao_s"],
                           "total_wall_s": t["tong_s"]})
            if not ms:
                continue
            a, b = ms
            pairs.append({"dataset": ds, "name": name, "tracker": tracker, "A": a, "B": b,
                          "fps_gain_pct": 100 * (b["fps"] / a["fps"] - 1),
                          "steady_gain_pct": 100 * (b["fps_steady"] / a["fps_steady"] - 1),
                          "wall_throughput_gain_pct": 100 * (a["total_wall_s"] / b["total_wall_s"] - 1),
                          "decode_delta_ms": b["decode_ms"] - a["decode_ms"],
                          "p90_latency_change_pct": 100 * (b["latency_p90_ms"] / a["latency_p90_ms"] - 1)})
    rng = np.random.default_rng(0)
    summary = {}
    for ds in ("bdd", "ua"):
        summary[ds] = {}
        for tracker in ("cv", "ekf"):
            q = [p for p in pairs if p["dataset"] == ds and p["tracker"] == tracker]
            if not q:
                continue
            gain = np.array([p["fps_gain_pct"] for p in q])
            steady = np.array([p["steady_gain_pct"] for p in q])
            boot = np.median(gain[rng.integers(0, len(gain), (5000, len(gain)))], axis=1)
            matched = [p for p in q if abs(p["decode_delta_ms"]) < 1]
            summary[ds][tracker] = {
                "n_videos": len(q), "frames_per_run": 150, "n_runs": len(q) * 2,
                "fps_A_median": float(np.median([p["A"]["fps"] for p in q])),
                "fps_B_median": float(np.median([p["B"]["fps"] for p in q])),
                "paired_fps_gain_pct_median": float(np.median(gain)),
                "paired_fps_gain_pct_ci95_boot_video": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                "paired_fps_gain_pct_p10_p90": [float(x) for x in np.percentile(gain, [10, 90])],
                "steady_gain_pct_median": float(np.median(steady)),
                "wall_throughput_gain_pct_median": float(np.median([p["wall_throughput_gain_pct"] for p in q])),
                "p90_latency_change_pct_median": float(np.median([p["p90_latency_change_pct"] for p in q])),
                "n_B_faster": int(np.sum(gain > 0)),
                "n_matched_decode_lt1ms": len(matched),
                "matched_decode_gain_pct_median": float(np.median([p["fps_gain_pct"] for p in matched])) if matched else None,
                "decode_delta_ms_p10_p50_p90": [float(x) for x in np.percentile([p["decode_delta_ms"] for p in q], [10, 50, 90])],
                "median_stage_A_B_ms": {k: [float(np.median([p[v][k] for p in q])) for v in "AB"]
                                        for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms", "tracker_ms", "init_s", "total_wall_s")},
            }
    result = {"summary": summary, "pairs": pairs, "n_pairs": len(pairs)}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"n_pairs": len(pairs), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
