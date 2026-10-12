"""Build the train-only candidate audit table for the buffer=90 thesis demo.

This script reads existing tracking results and the corrected, merged occlusion
event evaluation.  It never runs a tracker and never changes source results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from occlusion_eval import iou_matrix, match_frame  # noqa: E402


OUT_ROOT = Path("outputs/thesis_demos_buffer90")
EVAL_CSV = OUT_ROOT / "evaluation/event_outcomes_DETRAC-all.csv"
SPLIT = "DETRAC-all"
TRACKERS = {
    "cv": "sweep-cv-buf90",
    "ekf": "sweep-ekf_ctrv-buf90",
    "ukf": "sweep-ukf_ctrv-buf90",
}
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
PAD = 25

# Filled only with values measured by an exact first-stage ByteTrack replay
# after a fresh rerun.  Unmeasured events remain blank rather than receiving
# an inferred score.
MEASURED_REACQ_GAP = {
    ("MVI_40871", 1, 0): 0.1052148602460394,
}


def load_gt(video: str) -> pd.DataFrame:
    p = config.PROCESSED_DIR / SPLIT / video / "gt" / "gt.txt"
    d = pd.read_csv(p, header=None, names=GT_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def load_tracker(video: str, key: str) -> pd.DataFrame:
    p = config.PROCESSED_DIR / "trackers" / SPLIT / TRACKERS[key] / "data" / f"{video}.txt"
    return pd.read_csv(p, header=None, names=TRK_COLS)


def target_timeline(gt: pd.DataFrame, trk: pd.DataFrame, gid: int,
                    f0: int, f1: int) -> dict[int, int | None]:
    """Match the target with the same corrected whole-frame Hungarian rule."""
    out: dict[int, int | None] = {}
    gg = gt[gt.frame.between(f0, f1)]
    tt = trk[trk.frame.between(f0, f1)]
    by_frame = {int(f): x for f, x in tt.groupby("frame")}
    for fr, g in gg.groupby("frame"):
        fr = int(fr)
        p = by_frame.get(fr)
        hit = None
        if p is not None and len(p):
            mat = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float),
                             p[["x", "y", "w", "h"]].to_numpy(float))
            gids, tids = g.id.to_numpy(), p.id.to_numpy()
            for ri, ci in match_frame(mat, 0.5):
                if int(gids[ri]) == gid:
                    hit = int(tids[ci])
                    break
        if bool((g.id == gid).any()):
            out[fr] = hit
    return out


def compact_chain(timeline: dict[int, int | None]) -> str:
    runs: list[int] = []
    for value in timeline.values():
        if value is not None and (not runs or value != runs[-1]):
            runs.append(value)
    return "->".join(map(str, runs))


def nearest_reacquisition(timeline: dict[int, int | None], start: int, end: int
                          ) -> tuple[int | None, int]:
    """Nearest None->ID transition to the annotated event, plus its missing streak."""
    transitions: list[tuple[int, int]] = []
    gap = 0
    for frame, value in timeline.items():
        if value is None:
            gap += 1
        else:
            if gap:
                transitions.append((frame, gap))
            gap = 0
    if not transitions:
        return None, 0
    def distance(item: tuple[int, int]) -> tuple[int, int]:
        frame = item[0]
        d = start - frame if frame < start else frame - end if frame > end else 0
        return d, -frame
    return min(transitions, key=distance)


def turn_angle(t: pd.DataFrame, win: int = 10) -> float:
    if len(t) < 4:
        return np.nan
    n = min(win, max(2, len(t) // 2))
    a, b = t.head(n), t.tail(n)
    da = a[["cx", "cy"]].to_numpy()[-1] - a[["cx", "cy"]].to_numpy()[0]
    db = b[["cx", "cy"]].to_numpy()[-1] - b[["cx", "cy"]].to_numpy()[0]
    if np.hypot(*da) < 1.0 or np.hypot(*db) < 1.0:
        return np.nan
    h0, h1 = np.degrees(np.arctan2(da[1], da[0])), np.degrees(np.arctan2(db[1], db[0]))
    return float((h1 - h0 + 180) % 360 - 180)


def main() -> int:
    raw = pd.read_csv(EVAL_CSV)
    raw = raw[raw.eligible].copy()
    key = ["video", "track_id", "event_id", "start_frame", "end_frame",
           "length_frames", "has_full"]
    wide = raw[key].drop_duplicates(key)
    for short, tracker in TRACKERS.items():
        cols = ["outcome", "id_before", "id_after", "id_continuous", "recall_during"]
        part = raw[raw.tracker == tracker][key + cols].copy()
        part = part.rename(columns={c: f"{c}_{short}" for c in cols})
        wide = wide.merge(part, on=key, how="inner", validate="one_to_one")

    gt_cache: dict[str, pd.DataFrame] = {}
    geom = []
    for r in wide.itertuples(index=False):
        gt = gt_cache.setdefault(r.video, load_gt(r.video))
        t = gt[(gt.id == int(r.track_id)) & gt.frame.between(r.start_frame, r.end_frame)].sort_values("frame")
        xy = t[["cx", "cy"]].to_numpy(float)
        steps = np.diff(xy, axis=0) if len(xy) >= 2 else np.empty((0, 2))
        path = float(np.hypot(steps[:, 0], steps[:, 1]).sum()) if len(steps) else 0.0
        net = float(np.hypot(*(xy[-1] - xy[0]))) if len(xy) >= 2 else 0.0
        geom.append({
            "occlusion_level": "full" if bool(r.has_full) else "partial",
            "max_occlusion_ratio": float(1.0 - t.vis.min()) if len(t) else np.nan,
            "mean_occlusion_ratio": float(1.0 - t.vis.mean()) if len(t) else np.nan,
            "gt_frames_in_event": int(len(t)),
            "path_len_px": path,
            "net_displacement_px": net,
            "speed_px_per_frame": path / max(1, len(t) - 1),
            "turn_deg_gt": turn_angle(t),
        })
    wide = pd.concat([wide.reset_index(drop=True), pd.DataFrame(geom)], axis=1)

    for short in TRACKERS:
        n_gt = wide.gt_frames_in_event
        observed = np.rint(wide[f"recall_during_{short}"] * n_gt).astype(int)
        wide[f"unobserved_frames_{short}"] = n_gt - observed
        wide[f"id_match_{short}"] = wide[f"outcome_{short}"].eq("preserved")

    wide["condition_1_eligible"] = True
    wide["condition_2_ekf_id_match"] = wide.id_match_ekf
    wide["condition_3_cv_not_id_match"] = ~wide.id_match_cv
    wide["condition_4_ekf_stable_pm25"] = False
    wide["condition_5_reacq_verifiable"] = False
    wide["reacquisition_linked_to_event"] = False
    wide["ekf_reacquisition_frame"] = pd.Series(pd.NA, index=wide.index, dtype="Int64")
    wide["ekf_stability_chain_pm25"] = ""
    wide["ekf_post_reacq_matches_pm25"] = 0
    for short in TRACKERS:
        wide[f"reacquisition_frame_{short}"] = pd.Series(pd.NA, index=wide.index, dtype="Int64")
        wide[f"unobserved_streak_before_reacq_{short}"] = 0

    raw_candidate = wide.condition_2_ekf_id_match & wide.condition_3_cv_not_id_match
    for idx, r in wide[raw_candidate].iterrows():
        video, gid = str(r.video), int(r.track_id)
        gt = gt_cache.setdefault(video, load_gt(video))
        timelines = {}
        for short in TRACKERS:
            timelines[short] = target_timeline(gt, load_tracker(video, short), gid,
                                               int(r.start_frame) - PAD, int(r.end_frame) + 30)
            model_reacq, model_gap = nearest_reacquisition(
                timelines[short], int(r.start_frame), int(r.end_frame))
            if model_reacq is not None:
                wide.at[idx, f"reacquisition_frame_{short}"] = model_reacq
                wide.at[idx, f"unobserved_streak_before_reacq_{short}"] = model_gap
        reacq, gap = nearest_reacquisition(timelines["ekf"], int(r.start_frame), int(r.end_frame))
        if reacq is None:
            continue
        # A transition before the annotation starts is not a reappearance from
        # this event.  Require the immediately preceding unmatched streak to
        # overlap at least one annotated event frame.  Reappearance can occur
        # inside the event or shortly after its final annotated frame.
        missing_start, missing_end = reacq - gap, reacq - 1
        linked = bool(gap > 0 and reacq >= int(r.start_frame)
                      and missing_start <= int(r.end_frame)
                      and missing_end >= int(r.start_frame))
        timeline = target_timeline(gt, load_tracker(video, "ekf"), gid, reacq - PAD, reacq + PAD)
        ids = [tid for tid in timeline.values() if tid is not None]
        post = [tid for f, tid in timeline.items() if f >= reacq and tid is not None]
        expected = int(r.id_before_ekf)
        stable = bool(ids and all(tid == expected for tid in ids)
                      and len(post) >= min(5, sum(f >= reacq for f in timeline)))
        wide.at[idx, "ekf_reacquisition_frame"] = reacq
        wide.at[idx, "ekf_stability_chain_pm25"] = compact_chain(timeline)
        wide.at[idx, "ekf_post_reacq_matches_pm25"] = len(post)
        wide.at[idx, "condition_4_ekf_stable_pm25"] = stable
        wide.at[idx, "reacquisition_linked_to_event"] = linked
        image_ok = (config.find_images_root() / video / f"img{reacq:05d}.jpg").exists()
        pred_ok = all((config.PROCESSED_DIR / "trackers" / SPLIT / name / "data" / f"{video}.txt").exists()
                      for name in TRACKERS.values())
        wide.at[idx, "condition_5_reacq_verifiable"] = bool(linked and image_ok and pred_ok)

    wide["priority_1_ekf_cont_cv_not"] = wide.id_continuous_ekf & ~wide.id_continuous_cv
    wide["measured_reacq_fused_score_gap_ekf_minus_cv"] = [
        MEASURED_REACQ_GAP.get((str(r.video), int(r.track_id), int(r.event_id)), np.nan)
        for r in wide.itertuples(index=False)
    ]
    wide["moving"] = wide.speed_px_per_frame >= 0.3
    wide["qualifies"] = (wide.condition_1_eligible & wide.condition_2_ekf_id_match
                          & wide.condition_3_cv_not_id_match
                          & wide.condition_4_ekf_stable_pm25
                          & wide.condition_5_reacq_verifiable)

    rankable = wide[wide.qualifies].copy()
    rankable["_gap"] = rankable.measured_reacq_fused_score_gap_ekf_minus_cv.fillna(-1.0)
    rankable["_turn"] = rankable.turn_deg_gt.abs().fillna(-1.0)
    rankable = rankable.sort_values(
        ["priority_1_ekf_cont_cv_not", "_gap", "moving", "_turn",
         "length_frames", "max_occlusion_ratio", "video", "track_id", "event_id"],
        ascending=[False, False, False, False, False, False, True, True, True],
    )
    ranks = {int(idx): i + 1 for i, idx in enumerate(rankable.index)}
    wide["rank"] = [ranks.get(int(i), pd.NA) for i in wide.index]

    def reason(r) -> str:
        if not r.condition_2_ekf_id_match:
            return "loai: EKF id_match=0"
        if not r.condition_3_cv_not_id_match:
            return "loai: CV id_match=1"
        if not r.condition_4_ekf_stable_pm25:
            return "loai: EKF khong on dinh trong cua so ±25 frame"
        if not r.condition_5_reacq_verifiable:
            if not r.reacquisition_linked_to_event:
                return "loai: lan tai ghep gan nhat khong bat nguon tu su kien che khuat"
            return "loai: thieu anh/prediction de doi chieu tai xuat hien"
        if int(r.rank) == 1:
            return "giu: hang 1; tai ghep sau che khuat, xe dang re, che khuat ro"
        return f"giu: du dieu kien, hang {int(r.rank)} theo thu tu uu tien"

    wide["rank_reason"] = [reason(r) for r in wide.itertuples(index=False)]
    wide = wide.sort_values(["qualifies", "rank", "video", "track_id", "event_id"],
                            ascending=[False, True, True, True, True], na_position="last")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_ROOT / "candidates_buffer90.csv", index=False)

    print(f"eligible={len(wide)}")
    print(f"ekf_id_match={int(wide.condition_2_ekf_id_match.sum())}")
    print(f"ekf_match_cv_fail={int(raw_candidate.sum())}")
    print(f"stable_pm25={int((raw_candidate & wide.condition_4_ekf_stable_pm25).sum())}")
    print(f"verifiable={int((raw_candidate & wide.condition_4_ekf_stable_pm25 & wide.condition_5_reacq_verifiable).sum())}")
    print(f"qualifies={int(wide.qualifies.sum())}")
    show = ["rank", "video", "track_id", "event_id", "start_frame", "end_frame",
            "id_continuous_cv", "id_continuous_ekf", "speed_px_per_frame", "turn_deg_gt",
            "max_occlusion_ratio", "measured_reacq_fused_score_gap_ekf_minus_cv", "rank_reason"]
    print(wide[wide.qualifies][show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
