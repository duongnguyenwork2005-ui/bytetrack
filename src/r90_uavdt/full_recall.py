"""Measure full UAVDT vehicle detector recall by source occlusion level."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
from r90_uavdt.audit import EVIDENCE, GT_PREFIX, SOURCE
from r90_uavdt.cache import CACHE

LEVELS = {1: "none", 2: "large", 3: "medium", 4: "small"}
THRESHOLDS = (.05, .1, .25)


def match_indices(gt: np.ndarray, det: np.ndarray) -> set[int]:
    if not len(gt) or not len(det):
        return set()
    left = np.maximum(gt[:, None, :2], det[None, :, :2])
    right = np.minimum(gt[:, None, 2:4], det[None, :, 2:4])
    wh = np.maximum(0, right - left)
    inter = wh[:, :, 0] * wh[:, :, 1]
    ga = np.prod(gt[:, 2:4] - gt[:, :2], axis=1)
    da = np.prod(det[:, 2:4] - det[:, :2], axis=1)
    union = ga[:, None] + da[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    rows, cols = linear_sum_assignment(np.where(iou >= .5, -iou, 1e6))
    return {int(i) for i, j in zip(rows, cols) if iou[i, j] >= .5}


def empty() -> dict:
    return {name: {"gt": 0, "hit": 0} for name in ("all", *LEVELS.values())}


def main() -> None:
    selected = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
    result = {"cache": CACHE, "iou": .5, "matching": "Hungarian one-to-one maximizing IoU above 0.5",
              "source_occlusion_codes": LEVELS, "thresholds": {str(t): {"scenes": {}} for t in THRESHOLDS}}
    with zipfile.ZipFile(SOURCE / "UAV-benchmark-MOTD_v1.0.zip") as z:
        for seq in selected:
            gt = np.loadtxt(ROOT / "data/processed/UAVDT-R90train8" / seq / "gt/gt.txt", delimiter=",", ndmin=2)
            gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
            whole = np.loadtxt(io.BytesIO(z.read(f"{GT_PREFIX}{seq}_gt_whole.txt")), delimiter=",", ndmin=2)
            occ = {(int(r[0]), int(r[1])): int(r[7]) for r in whole}
            frame_gt = defaultdict(list)
            for r in gt:
                frame_gt[int(r[0])].append(r)
            counters = {str(t): empty() for t in THRESHOLDS}
            with np.load(det_cache.CACHE_ROOT / CACHE / f"{seq}.npz") as cache:
                counts, data = cache["n_per_frame"], cache["data"]
                offset = 0
                for frame, n in enumerate(counts, start=1):
                    all_det = data[offset:offset + n]
                    offset += int(n)
                    rows = np.asarray(frame_gt.get(frame, []), dtype=np.float32).reshape(-1, 9)
                    boxes = np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]]
                    levels = [LEVELS[occ[int(r[0]), int(r[1])]] for r in rows]
                    for t in THRESHOLDS:
                        det = all_det[(all_det[:, 4] > t) & np.isin(all_det[:, 5], [2, 5, 7])]
                        found = match_indices(boxes, det[:, :4])
                        counter = counters[str(t)]
                        for ix, level in enumerate(levels):
                            for key in ("all", level):
                                counter[key]["gt"] += 1
                                counter[key]["hit"] += ix in found
                assert offset == len(data)
            for t in THRESHOLDS:
                result["thresholds"][str(t)]["scenes"][seq] = counters[str(t)]
            print("RECALL", seq, {str(t): counters[str(t)]["all"] for t in THRESHOLDS}, flush=True)
    for t in THRESHOLDS:
        total = empty()
        for counter in result["thresholds"][str(t)]["scenes"].values():
            for level in total:
                for key in ("gt", "hit"):
                    total[level][key] += counter[level][key]
        result["thresholds"][str(t)]["total"] = total
    (EVIDENCE / "full_detector_recall.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
