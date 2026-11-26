"""Re-evaluate the archived Sherbrooke smoke from raw MOT without changing it."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats
import run_trackeval as rte
import trackeval
from sher_metric import combine, roi_contains

SPLIT = "Sherbrooke-cars-f2754-3754-roi-v1-smoke128"
SCORES = ROOT / "results/trackeval" / SPLIT
TRACKS = ROOT / "data/processed/trackers" / SPLIT
OUT = ROOT / "handoff/logs/external_eval/r66/sherbrooke/smoke"
VIDEO = "sherbrooke"
DETECTORS = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
FILTERS = ("cv", "ekf", "ukf", "pf500-s0", "pf500-s1", "pf500-s2")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(ok, message):
    if not ok:
        raise ValueError(message)


def filter_rows(data, mask):
    lines = data.splitlines(keepends=True)
    xywh = np.asarray([[float(x) for x in line.split(b",")[2:6]] for line in lines]).reshape(-1, 4)
    cx, cy = xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2
    outside = (cx < 0) | (cy < 0) | (cx >= mask.shape[1]) | (cy >= mask.shape[0])
    keep = roi_contains(mask, *xywh.T)
    return b"".join(row for row, yes in zip(lines, keep) if yes), int(outside.sum())


def score(tmp, name, data):
    folder = tmp / SPLIT / name / "data"
    folder.mkdir(parents=True)
    (folder / f"{VIDEO}.txt").write_bytes(data)
    ec, dc, mc = rte.build_configs(SPLIT, name, [VIDEO])
    ec.update(OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
              PRINT_RESULTS=False, TIME_PROGRESS=False, LOG_ON_ERROR=None)
    dc["TRACKERS_FOLDER"] = str(tmp)
    dataset = rte.dataset_class("mot")(dc)
    evalu = trackeval.Evaluator(ec)
    metrics = [cls(mc) for cls in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity)]
    with redirect_stdout(io.StringIO()):
        res, _ = evalu.evaluate([dataset], metrics)
    combined = rte.flatten_results(res, name).set_index("video").loc["COMBINED_SEQ"].to_dict()
    raw = boot_stats.save_raw_fields(res, name, tmp / "scratch_raw.json")
    return raw, combined


def verify_row(tmp, mask, name, original, repeat):
    raw_mot = (TRACKS / name / ("data" if repeat else "data_tho") / f"{VIDEO}.txt").read_bytes()
    roi, outside = filter_rows(raw_mot, mask)
    gate = json.loads((SCORES / original / "cong_r66.json").read_text())
    provenance = gate["roi"][VIDEO]
    verify(sha(raw_mot) == provenance["sha256_tho"], f"{name} raw MOT hash")
    verify(sha(roi) == provenance["sha256_roi"], f"{name} derived ROI hash")
    verify(roi == (TRACKS / original / "data" / f"{VIDEO}.txt").read_bytes(), f"{name} ROI content")
    raw, combined = score(tmp, name, roi)
    saved_raw = (SCORES / original / "raw_fields.json").read_bytes()
    verify(raw == json.loads(saved_raw), f"{name} raw_fields")
    with (SCORES / original / "combined_metrics.csv").open() as f:
        saved = next(csv.DictReader(f))
    rescoring_error = max(abs(v - float(saved[k])) for k, v in combined.items())
    recomposed = combine(raw)
    recomposition_error = max(abs(v - float(saved[k])) for k, v in recomposed.items())
    verify(max(rescoring_error, recomposition_error) < 1e-9, f"{name} metric mismatch")
    return {"run": name, "reference": original, "raw_mot_sha256": sha(raw_mot),
            "roi_mot_sha256": sha(roi), "reference_raw_fields_sha256": sha(saved_raw),
            "rescored_raw_fields_equal": True, "n_centres_outside": outside,
            "max_error_rescore_all_csv_fields": rescoring_error,
            "max_error_recompose_8_fields": recomposition_error, "metrics": recomposed,
            "old_minus_correct_AssA": boot_stats.combine(raw)["AssA"] - recomposed["AssA"]}


def main():
    mask = np.array(Image.open(ROOT / "data/processed/_r66_nguon/sherbrooke/sherbrooke_mask.png"))
    rows = []
    with TemporaryDirectory() as folder:
        temp = Path(folder)
        for detector, cond, filt in product(DETECTORS, ("c25", "c10"), FILTERS):
            name = f"r66smoke-{detector}-{cond}-{filt}"
            rows.append(verify_row(temp, mask, name, name, False))
        repeats = []
        for seed in [0, 1]:
            name = f"r66smoke-yolov8n_mixed5-c25-pf500-s{seed}"
            repeats.append(verify_row(temp, mask, name.replace("r66smoke-", "r66smoke-lap-"), name, True))
    summary = {"smoke_runs": len(rows), "repeat_runs": len(repeats),
               "max_rescore_error": max(r["max_error_rescore_all_csv_fields"] for r in rows + repeats),
               "max_recompose_error": max(r["max_error_recompose_8_fields"] for r in rows + repeats),
               "max_old_AssA_bias": max(r["old_minus_correct_AssA"] for r in rows),
               "centres_outside_smoke": sum(r["n_centres_outside"] for r in rows)}
    evidence = {**summary, "rows": rows, "pf_repeat": repeats,
                "script_sha256": sha(Path(__file__).read_bytes()),
                "combiner_sha256": sha(Path(__file__).with_name("sher_metric.py").read_bytes())}
    (OUT / "04_cham_lai_b3_b4_b6.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
