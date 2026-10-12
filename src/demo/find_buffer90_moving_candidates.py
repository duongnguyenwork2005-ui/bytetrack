"""Audit all merged train occlusion events for a moving buffer=90 demo.

The hard gates deliberately exclude slow/stationary targets and any
reassociation that occurs inside (or is unrelated to) the annotated event.
Existing tracker outputs are read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from occlusion_eval import iou_matrix, match_frame  # noqa: E402


OUT = Path("outputs/thesis_demos_buffer90_moving")
EVAL_CSV = OUT / "evaluation/event_outcomes_DETRAC-all.csv"
REPLAY_CSV = OUT / "candidate_replay_buffer90_moving.csv"
SPLIT = "DETRAC-all"
TRACKERS = {
    "cv": "sweep-cv-buf90",
    "ekf": "sweep-ekf_ctrv-buf90",
    "ukf": "sweep-ukf_ctrv-buf90",
}
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
PAD = 25


def load_gt(video: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / SPLIT / video / "gt" / "gt.txt"
    data = pd.read_csv(path, header=None, names=GT_COLS)
    data["cx"] = data.x + data.w / 2
    data["cy"] = data.y + data.h / 2
    return data


def load_tracker(video: str, model: str) -> pd.DataFrame:
    path = (config.PROCESSED_DIR / "trackers" / SPLIT / TRACKERS[model]
            / "data" / f"{video}.txt")
    return pd.read_csv(path, header=None, names=TRK_COLS)


def target_timeline(gt: pd.DataFrame, tracker: pd.DataFrame, gt_id: int,
                    first: int, last: int) -> dict[int, int | None]:
    """Whole-frame Hungarian matching, identical to corrected evaluation."""
    answer: dict[int, int | None] = {}
    gt_window = gt[gt.frame.between(first, last)]
    tr_window = tracker[tracker.frame.between(first, last)]
    pred_by_frame = {int(frame): rows for frame, rows in tr_window.groupby("frame")}
    for frame, gt_rows in gt_window.groupby("frame"):
        frame = int(frame)
        if not bool((gt_rows.id == gt_id).any()):
            continue
        hit = None
        pred_rows = pred_by_frame.get(frame)
        if pred_rows is not None and len(pred_rows):
            matrix = iou_matrix(
                gt_rows[["x", "y", "w", "h"]].to_numpy(float),
                pred_rows[["x", "y", "w", "h"]].to_numpy(float),
            )
            gt_ids = gt_rows.id.to_numpy()
            pred_ids = pred_rows.id.to_numpy()
            for row, col in match_frame(matrix, 0.5):
                if int(gt_ids[row]) == gt_id:
                    hit = int(pred_ids[col])
                    break
        answer[frame] = hit
    return answer


def transitions(timeline: dict[int, int | None]) -> list[tuple[int, int, int]]:
    """Return (reappearance frame, missing streak, missing-start frame)."""
    result: list[tuple[int, int, int]] = []
    gap = 0
    for frame, value in sorted(timeline.items()):
        if value is None:
            gap += 1
        else:
            if gap:
                result.append((frame, gap, frame - gap))
            gap = 0
    return result


def event_reappearance(timeline: dict[int, int | None], start: int, end: int
                       ) -> tuple[int | None, int, bool]:
    """First reacquisition whose preceding missing streak overlaps the event."""
    related = []
    for frame, gap, missing_start in transitions(timeline):
        missing_end = frame - 1
        if missing_start <= end and missing_end >= start:
            related.append((frame, gap, missing_start))
    if not related:
        return None, 0, False
    frame, gap, _ = related[0]
    return frame, gap, bool(end < frame <= end + 5)


def compact_chain(timeline: dict[int, int | None]) -> str:
    chain: list[int] = []
    for track_id in timeline.values():
        if track_id is not None and (not chain or track_id != chain[-1]):
            chain.append(track_id)
    return "->".join(map(str, chain))


def turn_angle(rows: pd.DataFrame, window: int = 10) -> float:
    if len(rows) < 4:
        return np.nan
    n = min(window, max(2, len(rows) // 2))
    first, last = rows.head(n), rows.tail(n)
    da = first[["cx", "cy"]].to_numpy()[-1] - first[["cx", "cy"]].to_numpy()[0]
    db = last[["cx", "cy"]].to_numpy()[-1] - last[["cx", "cy"]].to_numpy()[0]
    if np.hypot(*da) < 1.0 or np.hypot(*db) < 1.0:
        return np.nan
    a0 = np.degrees(np.arctan2(da[1], da[0]))
    a1 = np.degrees(np.arctan2(db[1], db[0]))
    return float((a1 - a0 + 180) % 360 - 180)


def main() -> int:
    raw = pd.read_csv(EVAL_CSV)
    raw = raw[raw.eligible].copy()
    key = ["video", "track_id", "event_id", "start_frame", "end_frame",
           "length_frames", "has_full"]
    table = raw[key].drop_duplicates(key)
    for model, tracker in TRACKERS.items():
        cols = ["outcome", "id_before", "id_after", "id_continuous", "recall_during"]
        part = raw[raw.tracker == tracker][key + cols].copy()
        part = part.rename(columns={name: f"{name}_{model}" for name in cols})
        table = table.merge(part, on=key, how="inner", validate="one_to_one")

    gt_cache: dict[str, pd.DataFrame] = {}
    geometry = []
    for row in table.itertuples(index=False):
        gt = gt_cache.setdefault(row.video, load_gt(row.video))
        target = gt[(gt.id == int(row.track_id))
                    & gt.frame.between(row.start_frame, row.end_frame)].sort_values("frame")
        xy = target[["cx", "cy"]].to_numpy(float)
        steps = np.diff(xy, axis=0) if len(xy) >= 2 else np.empty((0, 2))
        path = float(np.hypot(steps[:, 0], steps[:, 1]).sum()) if len(steps) else np.nan
        net = float(np.hypot(*(xy[-1] - xy[0]))) if len(xy) >= 2 else np.nan
        diag = np.hypot(target.w.to_numpy(float), target.h.to_numpy(float))
        median_diag = float(np.median(diag)) if len(diag) else np.nan
        expected_frames = np.arange(int(row.start_frame), int(row.end_frame) + 1)
        complete = bool(len(target) == len(expected_frames)
                        and np.array_equal(target.frame.to_numpy(int), expected_frames))
        normalized = net / median_diag if median_diag > 0 else np.nan
        displacement_ok = bool(complete and normalized >= 0.5)
        geometry.append({
            "occlusion_level": "full" if bool(row.has_full) else "partial",
            "max_occlusion_ratio": float(1.0 - target.vis.min()) if len(target) else np.nan,
            "mean_occlusion_ratio": float(1.0 - target.vis.mean()) if len(target) else np.nan,
            "gt_frames_in_event": int(len(target)),
            "gt_motion_complete": complete,
            "path_len_px": path,
            "net_displacement_px": net,
            "median_bbox_diagonal_px": median_diag,
            "normalized_net_displacement": normalized,
            "speed_px_per_frame": path / max(1, len(target) - 1) if complete else np.nan,
            "turn_deg_gt": turn_angle(target) if displacement_ok else np.nan,
            "turn_reliability": "reliable" if displacement_ok else "khong dang tin",
        })
    table = pd.concat([table.reset_index(drop=True), pd.DataFrame(geometry)], axis=1)

    for model in TRACKERS:
        observed = np.rint(table[f"recall_during_{model}"] * table.gt_frames_in_event).astype(int)
        table[f"unobserved_frames_{model}"] = table.gt_frames_in_event - observed
        table[f"id_match_{model}"] = table[f"outcome_{model}"].eq("preserved")

    table["condition_eligible"] = True
    table["condition_ekf_match_cv_fail"] = table.id_match_ekf & ~table.id_match_cv
    table["condition_motion_gt_complete"] = table.gt_motion_complete
    table["condition_speed_ge_3"] = table.speed_px_per_frame >= 3.0
    table["condition_path_ge_30"] = table.path_len_px >= 30.0
    table["condition_normalized_net_ge_0_5"] = table.normalized_net_displacement >= 0.5
    table["condition_moving"] = (table.condition_motion_gt_complete
                                  & table.condition_speed_ge_3
                                  & table.condition_path_ge_30
                                  & table.condition_normalized_net_ge_0_5)
    table["condition_ekf_stable_pm25"] = False
    table["condition_reappearance_after_event"] = False
    table["condition_dropout_linked_to_event"] = False
    table["condition_association_inputs_exist"] = False
    table["reappearance_frame"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table["unobserved_streak_before_reappearance"] = 0
    table["ekf_stability_chain_pm25"] = ""
    table["ekf_post_reappearance_matches_pm25"] = 0

    pair_mask = table.condition_ekf_match_cv_fail
    for index, row in table[pair_mask].iterrows():
        video, gt_id = str(row.video), int(row.track_id)
        start, end = int(row.start_frame), int(row.end_frame)
        gt = gt_cache.setdefault(video, load_gt(video))
        ekf = load_tracker(video, "ekf")
        broad = target_timeline(gt, ekf, gt_id, start - PAD, end + PAD + 5)
        reappearance, gap, after_ok = event_reappearance(broad, start, end)
        if reappearance is None:
            continue
        linked = gap > 0 and reappearance - gap <= end and reappearance - 1 >= start
        stable_window = target_timeline(gt, ekf, gt_id,
                                        reappearance - PAD, reappearance + PAD)
        ids = [track_id for track_id in stable_window.values() if track_id is not None]
        post = [track_id for frame, track_id in stable_window.items()
                if frame >= reappearance and track_id is not None]
        expected = int(row.id_before_ekf)
        stable = bool(ids and all(track_id == expected for track_id in ids)
                      and len(post) >= min(5, sum(frame >= reappearance
                                                 for frame in stable_window)))
        image_ok = (config.find_images_root() / video
                    / f"img{reappearance:05d}.jpg").exists()
        tracker_ok = all((config.PROCESSED_DIR / "trackers" / SPLIT / tracker
                          / "data" / f"{video}.txt").exists()
                         for tracker in TRACKERS.values())
        table.at[index, "reappearance_frame"] = reappearance
        table.at[index, "unobserved_streak_before_reappearance"] = gap
        table.at[index, "condition_reappearance_after_event"] = after_ok
        table.at[index, "condition_dropout_linked_to_event"] = linked
        table.at[index, "condition_ekf_stable_pm25"] = stable
        table.at[index, "condition_association_inputs_exist"] = bool(image_ok and tracker_ok)
        table.at[index, "ekf_stability_chain_pm25"] = compact_chain(stable_window)
        table.at[index, "ekf_post_reappearance_matches_pm25"] = len(post)

    table["pre_replay_candidate"] = (
        table.condition_ekf_match_cv_fail
        & table.condition_moving
        & table.condition_ekf_stable_pm25
        & table.condition_reappearance_after_event
        & table.condition_dropout_linked_to_event
        & table.condition_association_inputs_exist
    )

    replay_cols = [
        "video", "track_id", "event_id", "reappearance_frame",
        "fused_similarity_cv", "fused_similarity_ekf", "fused_similarity_ukf",
        "fused_gap_ekf_minus_cv", "full_lap_verified_cv", "full_lap_verified_ekf",
        "full_lap_verified_ukf", "replay_complete",
    ]
    if REPLAY_CSV.exists():
        replay = pd.read_csv(REPLAY_CSV)
        table = table.merge(replay[replay_cols],
                            on=["video", "track_id", "event_id", "reappearance_frame"],
                            how="left", validate="one_to_one")
    else:
        for col in replay_cols[4:]:
            table[col] = False if col.startswith("full_lap_") or col == "replay_complete" else np.nan

    table["qualifies"] = table.pre_replay_candidate & table.replay_complete.fillna(False)
    ranked = table[table.qualifies].copy()
    ranked = ranked.sort_values(
        ["fused_gap_ekf_minus_cv", "normalized_net_displacement", "length_frames",
         "video", "track_id", "event_id"],
        ascending=[False, False, False, True, True, True],
    )
    rank_map = {int(index): rank for rank, index in enumerate(ranked.index, 1)}
    table["rank"] = [rank_map.get(int(index), pd.NA) for index in table.index]

    def exclusion_reason(row) -> str:
        if not row.condition_ekf_match_cv_fail:
            return "loai: khong dong thoi EKF id_match=1 va CV id_match=0"
        if not row.condition_motion_gt_complete:
            return "loai: GT khong du lien tuc de tinh chuyen dong"
        failed_motion = []
        if not row.condition_speed_ge_3:
            failed_motion.append("speed<3")
        if not row.condition_path_ge_30:
            failed_motion.append("path<30")
        if not row.condition_normalized_net_ge_0_5:
            failed_motion.append("net/median_diag<0.5")
        if failed_motion:
            return "loai: " + ", ".join(failed_motion)
        if not row.condition_ekf_stable_pm25:
            return "loai: EKF khong on dinh trong cua so ±25 frame"
        if not row.condition_reappearance_after_event:
            return "loai: lan tai ghep dau khong nam trong end+1..end+5"
        if not row.condition_dropout_linked_to_event:
            return "loai: chuoi mat ghep khong giao su kien che khuat"
        if not row.condition_association_inputs_exist:
            return "loai: thieu anh/prediction de replay association"
        if not bool(row.replay_complete):
            return "cho replay full LAP"
        return f"giu: hang {int(row.rank)} theo gap, net/diag, do dai"

    table["rank_reason"] = [exclusion_reason(row) for row in table.itertuples(index=False)]
    table = table.sort_values(["qualifies", "pre_replay_candidate", "rank", "video",
                               "track_id", "event_id"],
                              ascending=[False, False, True, True, True, True],
                              na_position="last")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "candidates_buffer90_moving.csv", index=False)

    sequential_pair = table.condition_ekf_match_cv_fail
    sequential_moving = sequential_pair & table.condition_moving
    sequential_after = sequential_moving & table.condition_ekf_stable_pm25 \
        & table.condition_reappearance_after_event & table.condition_dropout_linked_to_event
    print(f"eligible={len(table)}")
    print(f"ekf_match_cv_fail={int(sequential_pair.sum())}")
    print(f"moving={int(sequential_moving.sum())}")
    print(f"stable_and_post_event={int(sequential_after.sum())}")
    print(f"pre_replay={int(table.pre_replay_candidate.sum())}")
    print(f"qualified={int(table.qualifies.sum())}")
    show = ["rank", "video", "track_id", "event_id", "start_frame", "end_frame",
            "reappearance_frame", "speed_px_per_frame", "path_len_px",
            "net_displacement_px", "median_bbox_diagonal_px",
            "normalized_net_displacement", "fused_gap_ekf_minus_cv", "rank_reason"]
    print(table[table.pre_replay_candidate][show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
