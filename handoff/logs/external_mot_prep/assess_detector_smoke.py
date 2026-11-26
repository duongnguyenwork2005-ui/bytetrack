"""Class-agnostic upper-bound recall of a short UTUAV detector cache at IoU 0.5."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_count(pred: np.ndarray, truth: np.ndarray) -> int:
    if len(pred) == 0 or len(truth) == 0:
        return 0
    x1 = np.maximum(pred[:, None, 0], truth[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], truth[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], truth[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], truth[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    pa = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ta = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
    iou = inter / np.maximum(pa[:, None] + ta[None, :] - inter, 1e-9)
    rows, cols = linear_sum_assignment(np.where(iou >= 0.5, 1000 + iou, 0), maximize=True)
    return int(np.sum(iou[rows, cols] >= 0.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache_npz")
    ap.add_argument("gt_txt")
    args = ap.parse_args()
    with np.load(args.cache_npz) as archive:
        frames = archive["frame_idx"]
        counts = archive["n_per_frame"]
        data = archive["data"]
    gt = pd.read_csv(args.gt_txt, header=None)
    gt = gt[gt[7] == 1]
    out = {"frames": len(frames), "moving_gt": 0}
    for threshold in (0.05, 0.25):
        out[str(threshold)] = {"predictions": 0, "matched_gt_iou50": 0}
    offset = 0
    for frame, count in zip(frames, counts):
        raw = data[offset:offset + count]
        offset += count
        g = gt[gt[0] == frame]
        truth = g[[2, 3, 4, 5]].to_numpy(dtype=float)
        truth[:, 2] += truth[:, 0]
        truth[:, 3] += truth[:, 1]
        out["moving_gt"] += len(truth)
        for threshold in (0.05, 0.25):
            pred = raw[raw[:, 4] >= threshold]
            item = out[str(threshold)]
            item["predictions"] += len(pred)
            item["matched_gt_iou50"] += match_count(pred[:, :4], truth)
    for threshold in (0.05, 0.25):
        item = out[str(threshold)]
        item["recall_upper_bound"] = item["matched_gt_iou50"] / max(out["moving_gt"], 1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
