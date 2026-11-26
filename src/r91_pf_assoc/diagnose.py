"""R91 read-only association diagnostic on vehicle GT and a shared detection cache.

The continuity proxy below is NOT TrackEval's IDSW: it matches current boxes
independently at IoU .5 and conditions on a previously visible GT identity.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
SPLIT = "UAVDT-R90train8"
VIDEOS = ("M1306", "M0204")
RUNS = {"cv": "r91-diag-c10-cv", "pf_s0": "r91-diag-c10-pf-s0"}
OUT = ROOT / "handoff/logs/review91_pf_assoc/diagnostic.json"


def iou(a, b):
    """Pairwise IoU, tlwh arrays."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    aa = np.asarray(a, dtype=float)[:, :4]
    bb = np.asarray(b, dtype=float)[:, :4]
    left = np.maximum(aa[:, None, :2], bb[None, :, :2])
    right = np.minimum(aa[:, None, :2] + aa[:, None, 2:], bb[None, :, :2] + bb[None, :, 2:])
    wh = np.maximum(right - left, 0)
    inter = wh[..., 0] * wh[..., 1]
    union = aa[:, None, 2] * aa[:, None, 3] + bb[None, :, 2] * bb[None, :, 3] - inter
    return inter / np.maximum(union, 1e-9)


def pairs(a, b, threshold=.5):
    mat = iou(a, b)
    if not mat.size:
        return {}
    ri, ci = linear_sum_assignment(1 - mat)
    return {int(r): int(c) for r, c in zip(ri, ci) if mat[r, c] >= threshold}


def by_frame(df):
    return {int(frame): group.reset_index(drop=True) for frame, group in df.groupby("frame", sort=False)}


def summarize(rows):
    out = {"eligible": len(rows)}
    for key in ("continued", "changed", "missing_output", "old_in_pool", "old_matched", "correct_det_given_old", "correct_det_claimed_other"):
        out[key] = sum(bool(r[key]) for r in rows)
    out["changed_old_lost"] = sum(r["changed"] and r["old_state"] == 2 for r in rows)
    out["changed_old_tracked"] = sum(r["changed"] and r["old_state"] == 1 for r in rows)
    for subset, selected in (("all", rows), ("changed", [r for r in rows if r["changed"]]), ("continued", [r for r in rows if r["continued"]])):
        vals = [r["old_pred_iou"] for r in selected if r["old_pred_iou"] is not None]
        out[subset + "_old_pred_iou_median"] = float(np.median(vals)) if vals else None
        out[subset + "_old_pred_iou_below_02"] = sum(v < .2 for v in vals)
        out[subset + "_with_old_pred"] = len(vals)
    return out


def run_one(video, name, run):
    base = ROOT / "data/processed/trackers" / SPLIT / run
    gt = pd.read_csv(ROOT / "data/processed" / SPLIT / video / "gt/gt.txt", header=None)
    gt = gt[gt[7] == 1].copy()  # vehicle; class 8 is distractor/ignore
    gt.columns = ["frame", "id", "x", "y", "w", "h", "valid", "class", "visibility"]
    output = pd.read_csv(base / "data" / (video + ".txt"), header=None)
    output = output.iloc[:, :6]
    output.columns = ["frame", "id", "x", "y", "w", "h"]
    logs = base / "assoc_log"
    tr = pd.read_parquet(logs / (video + "_tracks.parquet"))
    de = pd.read_parquet(logs / (video + "_dets.parquet"))
    gt_f, out_f, tr_f, de_f = map(by_frame, (gt, output, tr, de))
    last_seen = {}
    rows = []
    for frame in sorted(gt_f):
        g = gt_f[frame]
        o = out_f.get(frame, output.iloc[:0])
        d = de_f.get(frame, de.iloc[:0])
        t = tr_f.get(frame, tr.iloc[:0])
        gom = pairs(g[["x", "y", "w", "h"]].to_numpy(), o[["x", "y", "w", "h"]].to_numpy())
        gdm = pairs(g[["x", "y", "w", "h"]].to_numpy(), d[["x", "y", "w", "h"]].to_numpy())
        track_by_id = {int(row.track_id): row for row in t.itertuples(index=False)}
        for gi, row in enumerate(g.itertuples(index=False)):
            gid = int(row.id)
            prev = last_seen.get(gid)
            curr = int(o.iloc[gom[gi]].id) if gi in gom else None
            if prev and 0 < frame - prev[0] <= 8 and gi in gdm:
                old = prev[1]
                det = d.iloc[gdm[gi]]
                old_row = track_by_id.get(old)
                pred_iou = None if old_row is None else float(iou([[old_row.x, old_row.y, old_row.w, old_row.h]], [[det.x, det.y, det.w, det.h]])[0, 0])
                rows.append({
                    "frame": frame, "gt_id": gid, "gap": frame - prev[0], "continued": curr == old,
                    "changed": curr is not None and curr != old, "missing_output": curr is None,
                    "old_in_pool": old_row is not None,
                    "old_state": int(old_row.state) if old_row is not None else None,
                    "old_matched": old_row is not None and old_row.matched_det >= 0,
                    "correct_det_given_old": int(det.matched_track) == old,
                    "correct_det_claimed_other": int(det.matched_track) >= 0 and int(det.matched_track) != old,
                    "old_pred_iou": pred_iou,
                })
            if curr is not None:
                last_seen[gid] = (frame, curr)
    return summarize(rows), rows


def main():
    result = {}
    for video in VIDEOS:
        result[video] = {}
        for name, run in RUNS.items():
            summary, rows = run_one(video, name, run)
            result[video][name] = summary
            pd.DataFrame(rows).to_csv(OUT.with_name(f"{video}_{name}_continuity.csv"), index=False)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
