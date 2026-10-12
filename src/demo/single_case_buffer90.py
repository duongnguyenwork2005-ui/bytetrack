"""
single_case_buffer90.py -- Tim va dung MOT video minh hoa (case study) tren tap
                           train, track_buffer=90: xe DANG DI CHUYEN RO, EKF+CTRV
                           giu dung ID lau hon KF+CV.

KHAC voi find_buffer90_moving_candidates.py: KHONG gioi han trong su kien che
khuat da gop. Quet TOAN BO object GT cua 60 video train, doi chieu timeline
ghep (Hungarian IoU>=0.5 tren toan frame, giong occlusion_eval) cua CV va EKF.

DINH NGHIA "EKF THANG CV" (cung tieu chi cung, khong noi):
  - CV that bai tai frame f_fail:
      switched : CV dang ghep target bang ID a (da on dinh >= 5 frame) roi tai
                 f_fail ghep bang ID b != a;
      lost     : CV dang ghep bang ID a (>= 5 frame) roi khong ghep duoc target
                 trong >= 10 frame GT lien tiep bat dau tu f_fail.
    That bai phai KEO DAI: trong 10 frame GT ke tu f_fail, CV khong duoc ghep
    lai bang ID a (loai cac "nhap nhay" 1 frame roi quay ve ID cu - do khong
    phai that bai that, xem MVI_40162 t40, MVI_40181 t18 trong bang candidate).
  - EKF: truoc f_fail dang ghep target bang ID e (>= 5 frame). Trong doan
    [f_fail, f_fail + 5] EKF KHONG duoc ghep bang ID nao khac e (chi cho phep
    None), va ke tu frame dau tien ghep duoc (offset 0..5) phai giu DUNG ID e
    >= 10 frame GT LIEN TIEP. Truong hop EKF chi dung 1-2 frame roi doi/mat bi
    loai tu dong.
  - ON DINH TRUOC DO: trong 25 frame GT truoc f_fail, ca CV lan EKF khong duoc
    doi ID (cho phep None). Neu EKF vua doi ID ngay truoc f_fail thi "EKF van
    khop dung cung ID" khong con y nghia (vd MVI_40981 t26: EKF 159->182 o
    frame 382 roi CV that bai o 387).
  - UKF KHONG dung de loc; chi ghi lai ket cuc that de hien thi.

XE DI CHUYEN RO (do tren GT trong cua so demo [f_fail-25, f_fail+25]):
  - toc do trung vi >= 3 px/frame;
  - dich chuyen rong >= 0,5 x duong cheo bbox trung vi;
  - tong quang duong >= 30 px;
  - cua so co >= 20 frame GT.

XEP HANG: co che khuat (GT vis < 1 trong [f_fail-10, f_fail+2] hoac co doan che
trong occlusion_segments.csv giao voi cua so) > doi huong ro (|turn| >= 10 deg)
> EKF giu dung ID lau hon sau f_fail (ekf_hold_until_change: dem toi khi EKF ghep
ID KHAC, cho phep detector rot < 10 frame vi rot detection xay ra o ca ba tracker;
cot ekf_hold_after = dem lien tiep tuyet doi, luu kem de doi chieu).

CACH DUNG
    python src/demo/single_case_buffer90.py scan
    python src/demo/single_case_buffer90.py build                 # ung vien hang 1, chay lai 3 tracker (khong cache)
    python src/demo/single_case_buffer90.py build --reuse-rerun   # chi render lai tu rerun/ da co
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from occlusion_eval import iou_matrix, match_frame, match_video  # noqa: E402

OUT = Path("outputs/thesis_demos_buffer90_single_case")
SPLIT = "DETRAC-all"
PART = "train"
CONF = 0.25
TRACKERS = {
    "cv": ("sweep-cv-buf90", "configs/buffer_sweep/cv_buf90.yaml", "KF + CV", "cv"),
    "ekf": ("sweep-ekf_ctrv-buf90", "configs/buffer_sweep/ekf_ctrv_buf90.yaml", "EKF + CTRV", "ekf_ctrv"),
    "ukf": ("sweep-ukf_ctrv-buf90", "configs/buffer_sweep/ukf_ctrv_buf90.yaml", "UKF + CTRV", "ukf_ctrv"),
}
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]

# Tieu chi cung
MIN_ESTABLISHED = 5        # ID phai on dinh >= 5 frame truoc khi noi "doi/mat"
MIN_LOST_STREAK = 10       # CV khong ghep >= 10 frame GT lien tiep => "lost"
MIN_EKF_HOLD = 10          # EKF giu dung ID >= 10 frame GT lien tiep sau f_fail
MAX_EKF_START_OFFSET = 5   # EKF duoc phep None toi da 5 frame sau f_fail truoc khi bat dau chuoi
WIN = 25                   # cua so demo: [f_fail - WIN, f_fail + WIN]
MIN_WIN_FRAMES = 20
MIN_MEDIAN_SPEED = 3.0
MIN_PATH = 30.0
MIN_NET_OVER_DIAG = 0.5
TURN_MIN_DEG = 10.0


# ---------------------------------------------------------------------------
# Du lieu
# ---------------------------------------------------------------------------
def load_gt(video: str) -> pd.DataFrame:
    d = pd.read_csv(config.PROCESSED_DIR / SPLIT / video / "gt" / "gt.txt", header=None, names=GT_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def load_saved(video: str, tdir: str) -> pd.DataFrame:
    d = pd.read_csv(config.PROCESSED_DIR / "trackers" / SPLIT / tdir / "data" / f"{video}.txt",
                    header=None, names=TRK_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def timelines_for_video(gt: pd.DataFrame, trk: pd.DataFrame) -> dict[int, dict[int, int | None]]:
    """gt_id -> {frame: tracker_id | None} cho MOI frame co GT cua object do."""
    m = match_video(gt, trk)
    hit = {(int(r.frame), int(r.gt_id)): int(r.tracker_id) for r in m.itertuples(index=False)}
    out: dict[int, dict[int, int | None]] = {}
    for r in gt[["frame", "id"]].itertuples(index=False):
        out.setdefault(int(r.id), {})[int(r.frame)] = hit.get((int(r.frame), int(r.id)))
    return {k: dict(sorted(v.items())) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Phan tich timeline mot object
# ---------------------------------------------------------------------------
def run_length_before(seq: list, i: int) -> tuple[int | None, int]:
    """(ID dang ghep, so frame lien tiep ghep dung ID do) ngay truoc vi tri i (bo qua None)."""
    j = i - 1
    while j >= 0 and seq[j] is None:
        j -= 1
    if j < 0:
        return None, 0
    tid, n = seq[j], 0
    while j >= 0 and seq[j] == tid:
        n += 1
        j -= 1
    return tid, n


def hold_from(seq: list, i: int, tid: int) -> int:
    """So frame LIEN TIEP ke tu i ma seq == tid (dung ngay khi khac, ke ca None)."""
    n = 0
    while i + n < len(seq) and seq[i + n] == tid:
        n += 1
    return n


def cv_failures(seq: list) -> list[dict]:
    """Cac thoi diem CV that bai: (index, loai, id cu, id moi, do dai gap None truoc do)."""
    out = []
    i = 0
    while i < len(seq):
        cur = seq[i]
        prev_id, prev_n = run_length_before(seq, i)
        if prev_id is None or prev_n < MIN_ESTABLISHED:
            i += 1
            continue
        if cur is not None and cur != prev_id:
            gap = 0
            j = i - 1
            while j >= 0 and seq[j] is None:
                gap += 1
                j -= 1
            out.append(dict(i=i, kind="switched", old=prev_id, new=cur, gap_before=gap))
            i += 1
            continue
        if cur is None and seq[i - 1] is not None:
            streak = 0
            while i + streak < len(seq) and seq[i + streak] is None:
                streak += 1
            if streak >= MIN_LOST_STREAK:
                out.append(dict(i=i, kind="lost", old=prev_id, new=None, gap_before=0, lost_streak=streak))
            i += streak
            continue
        i += 1
    return out


def ekf_wins_at(seq: list, i_fail: int) -> dict | None:
    """Kiem tra EKF tai vi tri i_fail: giu dung ID e >= MIN_EKF_HOLD frame lien tiep."""
    e, n_before = run_length_before(seq, i_fail)
    if e is None or n_before < MIN_ESTABLISHED:
        return None
    for off in range(0, MAX_EKF_START_OFFSET + 1):
        k = i_fail + off
        if k >= len(seq):
            return None
        v = seq[k]
        if v is None:
            continue
        if v != e:
            return None                      # EKF ghep ID khac -> khong "giu dung"
        hold = hold_from(seq, k, e)
        if hold >= MIN_EKF_HOLD:
            return dict(ekf_id=e, ekf_established_before=n_before, ekf_start_offset=off, ekf_hold_after=hold)
        return None
    return None


def n_id_changes(seq: list, a: int, b: int) -> int:
    """So lan doi ID (bo qua None) trong seq[a:b]."""
    ids = [v for v in seq[max(0, a):max(0, b)] if v is not None]
    return sum(1 for x, y in zip(ids, ids[1:]) if x != y)


def hold_until_change(seq: list, i: int, tid, max_none: int = MIN_LOST_STREAK) -> int:
    """So frame ghep DUNG tid ke tu i cho toi khi ghep ID KHAC (hoac mat >= max_none
    frame lien tiep). Khac hold_from: cho phep vai frame None (detector rot, thuong
    xay ra o CA BA tracker) ma khong coi la EKF "mat" ID."""
    if tid is None:
        return 0
    n = none_run = 0
    for v in seq[i:]:
        if v == tid:
            n += 1
            none_run = 0
        elif v is None:
            none_run += 1
            if none_run >= max_none:
                break
        else:
            break
    return n


def heading_change(xy: np.ndarray, win: int = 10, min_disp: float = 5.0) -> float:
    if len(xy) < 2 * win:
        return np.nan
    da = xy[win - 1] - xy[0]
    db = xy[-1] - xy[-win]
    if np.hypot(*da) < min_disp or np.hypot(*db) < min_disp:
        return np.nan
    h0, h1 = np.degrees(np.arctan2(da[1], da[0])), np.degrees(np.arctan2(db[1], db[0]))
    return float((h1 - h0 + 180) % 360 - 180)


def motion_metrics(t: pd.DataFrame, f0: int, f1: int) -> dict:
    w = t[t.frame.between(f0, f1)].sort_values("frame")
    n = len(w)
    if n < 2:
        return dict(win_frames=n)
    xy = w[["cx", "cy"]].to_numpy(float)
    fr = w.frame.to_numpy(int)
    step = np.hypot(*np.diff(xy, axis=0).T)
    consec = np.diff(fr) == 1
    speeds = step[consec]
    diag = np.hypot(w.w.to_numpy(float), w.h.to_numpy(float))
    net = float(np.hypot(*(xy[-1] - xy[0])))
    med_diag = float(np.median(diag))
    return dict(win_frames=n, win_start=int(fr[0]), win_end=int(fr[-1]),
                median_speed_px=float(np.median(speeds)) if len(speeds) else np.nan,
                mean_speed_px=float(np.mean(speeds)) if len(speeds) else np.nan,
                path_px=float(step.sum()), net_px=net, median_bbox_diag_px=med_diag,
                net_over_diag=net / med_diag if med_diag > 0 else np.nan,
                turn_deg=heading_change(xy),
                min_vis=float(w.vis.min()), mean_vis=float(w.vis.mean()))


# ---------------------------------------------------------------------------
# SCAN
# ---------------------------------------------------------------------------
def scan(out_dir: Path) -> pd.DataFrame:
    videos = sorted(p.stem for p in (config.PROCESSED_DIR / "trackers" / SPLIT / TRACKERS["cv"][0] / "data").glob("*.txt"))
    occ = pd.read_csv(config.part_file("occlusion_segments.csv", PART))
    rows = []
    n_obj = 0
    for vi, video in enumerate(videos, 1):
        gt = load_gt(video)
        tl = {k: timelines_for_video(gt, load_saved(video, TRACKERS[k][0])) for k in TRACKERS}
        for gid, cv_t in tl["cv"].items():
            frames = list(cv_t)
            if len(frames) < 2 * MIN_ESTABLISHED:
                continue
            n_obj += 1
            cv_seq = [cv_t[f] for f in frames]
            ekf_seq = [tl["ekf"].get(gid, {}).get(f) for f in frames]
            ukf_seq = [tl["ukf"].get(gid, {}).get(f) for f in frames]
            t = gt[gt.id == gid]
            for fail in cv_failures(cv_seq):
                i = fail["i"]
                win = ekf_wins_at(ekf_seq, i)
                f_fail = frames[i]
                mm = motion_metrics(t, f_fail - WIN, f_fail + WIN)
                # che khuat quanh thoi diem that bai
                near = t[t.frame.between(f_fail - 10, f_fail + 2)]
                occ_near = float(1.0 - near.vis.min()) if len(near) else np.nan
                seg = occ[(occ.video == video) & (occ.track_id == gid)
                          & (occ.start_frame <= f_fail + 2) & (occ.end_frame >= f_fail - WIN)]
                # UKF: ket cuc that (khong dung de loc)
                u_before, _ = run_length_before(ukf_seq, i)
                u_at = ukf_seq[i]
                u_hold = hold_from(ukf_seq, i, u_before) if (u_before is not None and u_at == u_before) else 0
                # CV: that bai co KEO DAI khong (10 frame ke tu f_fail khong quay ve ID cu)?
                cv_persist = not any(v == fail["old"] for v in cv_seq[i:i + MIN_EKF_HOLD])
                cv_recover_ever = any(v == fail["old"] for v in cv_seq[i:])
                cv_new_hold = hold_from(cv_seq, i, fail["new"]) if fail["new"] is not None else 0
                # On dinh truoc do (25 frame GT truoc f_fail)
                cv_chg = n_id_changes(cv_seq, i - WIN, i)
                ekf_chg = n_id_changes(ekf_seq, i - WIN, i)
                rows.append(dict(
                    video=video, gt_id=int(gid), gt_first=frames[0], gt_last=frames[-1], gt_frames=len(frames),
                    f_fail=f_fail, cv_kind=fail["kind"], cv_id_before=fail["old"], cv_id_after=fail["new"],
                    cv_gap_before_fail=fail["gap_before"], cv_lost_streak=fail.get("lost_streak", 0),
                    cv_established_before=run_length_before(cv_seq, i)[1],
                    cv_fail_persists_10=bool(cv_persist), cv_recovers_old_id_ever=bool(cv_recover_ever),
                    cv_new_id_hold=int(cv_new_hold),
                    cv_id_changes_25_before=int(cv_chg), ekf_id_changes_25_before=int(ekf_chg),
                    ekf_wins=win is not None,
                    **(win or dict(ekf_id=run_length_before(ekf_seq, i)[0], ekf_established_before=run_length_before(ekf_seq, i)[1],
                                   ekf_start_offset=None, ekf_hold_after=hold_from(ekf_seq, i, run_length_before(ekf_seq, i)[0]) if run_length_before(ekf_seq, i)[0] is not None else 0)),
                    ekf_id_at_fail=ekf_seq[i],
                    ekf_hold_until_change=hold_until_change(ekf_seq, i, run_length_before(ekf_seq, i)[0]),
                    ukf_id_before=u_before, ukf_id_at_fail=u_at, ukf_hold_after=u_hold,
                    ukf_hold_until_change=hold_until_change(ukf_seq, i, u_before),
                    **mm,
                    occ_ratio_near_fail=occ_near,
                    occ_segment_overlaps=bool(len(seg)),
                    occ_segment_max_ratio=float(seg.max_occlusion_ratio.max()) if len(seg) else 0.0,
                ))
        print(f"[{vi:2d}/{len(videos)}] {video}: {len(tl['cv'])} object, tong ung vien tho {len(rows)}", flush=True)
    d = pd.DataFrame(rows)
    print(f"\nObject GT da xet: {n_obj} | thoi diem CV that bai (ID da on dinh >= {MIN_ESTABLISHED} frame): {len(d)}")

    d["moving_speed_ok"] = d.median_speed_px >= MIN_MEDIAN_SPEED
    d["moving_path_ok"] = d.path_px >= MIN_PATH
    d["moving_net_ok"] = d.net_over_diag >= MIN_NET_OVER_DIAG
    d["moving_win_ok"] = d.win_frames >= MIN_WIN_FRAMES
    d["moving"] = d.moving_speed_ok & d.moving_path_ok & d.moving_net_ok & d.moving_win_ok
    d["has_occlusion"] = (d.occ_ratio_near_fail > 0) | d.occ_segment_overlaps
    d["has_turn"] = d.turn_deg.abs() >= TURN_MIN_DEG
    d["stable_before"] = (d.cv_id_changes_25_before == 0) & (d.ekf_id_changes_25_before == 0)
    d["qualifies"] = d.ekf_wins & d.cv_fail_persists_10 & d.stable_before & d.moving
    # Xep hang: che khuat > doi huong > thoi gian EKF giu dung ID sau khi CV that bai.
    # "Thoi gian giu dung ID" = ekf_hold_until_change (cho phep detector rot < 10 frame,
    # vi rot detection xay ra o ca ba tracker, khong phai EKF mat ID). Cot ekf_hold_after
    # (lien tiep tuyet doi) van duoc luu de doi chieu.
    q = d[d.qualifies].sort_values(["has_occlusion", "has_turn", "ekf_hold_until_change", "ekf_hold_after", "median_speed_px"],
                                   ascending=[False, False, False, False, False])
    d["rank"] = pd.NA
    d.loc[q.index, "rank"] = range(1, len(q) + 1)

    def reason(r):
        if not r.ekf_wins:
            if pd.isna(r.ekf_id) or r.ekf_established_before < MIN_ESTABLISHED:
                return f"loai: EKF chua co ID on dinh (>= {MIN_ESTABLISHED} frame) truoc khi CV that bai"
            if r.ekf_id_at_fail is not None and not pd.isna(r.ekf_id_at_fail) and r.ekf_id_at_fail != r.ekf_id:
                return "loai: EKF cung doi ID tai f_fail"
            return f"loai: EKF khong giu dung ID >= {MIN_EKF_HOLD} frame sau khi CV that bai (giu {r.ekf_hold_after})"
        if not r.cv_fail_persists_10:
            return "loai: CV quay ve ID cu trong 10 frame (nhap nhay, khong phai that bai keo dai)"
        if not r.stable_before:
            return (f"loai: doi ID trong 25 frame truoc f_fail (CV {r.cv_id_changes_25_before}, "
                    f"EKF {r.ekf_id_changes_25_before}) - canh khong sach")
        bad = []
        if not r.moving_win_ok: bad.append(f"cua so {r.win_frames} < {MIN_WIN_FRAMES} frame")
        if not r.moving_speed_ok: bad.append(f"toc do trung vi {r.median_speed_px:.2f} < {MIN_MEDIAN_SPEED}")
        if not r.moving_path_ok: bad.append(f"path {r.path_px:.1f} < {MIN_PATH}")
        if not r.moving_net_ok: bad.append(f"net/diag {r.net_over_diag:.2f} < {MIN_NET_OVER_DIAG}")
        if bad:
            return "loai: " + "; ".join(bad)
        return f"giu: hang {int(r['rank'])}"
    d["reason"] = [reason(r) for _, r in d.iterrows()]
    d = d.sort_values(["qualifies", "rank", "ekf_wins", "ekf_hold_after"], ascending=[False, True, False, False], na_position="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / "candidates_single_case.csv", index=False)

    s1 = d.ekf_wins
    s2 = s1 & d.cv_fail_persists_10
    s3 = s2 & d.stable_before
    s4 = s3 & d.moving
    print(f"Trong do EKF thang (giu dung ID >= {MIN_EKF_HOLD} frame lien tiep): {int(s1.sum())}")
    print(f"  + CV that bai keo dai >= 10 frame (khong nhap nhay ve ID cu): {int(s2.sum())}")
    print(f"  + ca CV va EKF khong doi ID trong 25 frame truoc do: {int(s3.sum())}")
    print(f"  + xe di chuyen ro (speed>={MIN_MEDIAN_SPEED}, path>={MIN_PATH}, net/diag>={MIN_NET_OVER_DIAG}): {int(s4.sum())}")
    print(f"    trong do co che khuat: {int((d.qualifies & d.has_occlusion).sum())}, doi huong >= {TURN_MIN_DEG} deg: {int((d.qualifies & d.has_turn).sum())}")
    show = ["rank", "video", "gt_id", "f_fail", "cv_kind", "cv_id_before", "cv_id_after", "cv_gap_before_fail",
            "cv_new_id_hold", "ekf_id", "ekf_start_offset", "ekf_hold_after", "ekf_hold_until_change",
            "ukf_id_before", "ukf_id_at_fail", "ukf_hold_after", "ukf_hold_until_change",
            "median_speed_px", "path_px", "net_px", "net_over_diag", "turn_deg", "occ_ratio_near_fail",
            "occ_segment_overlaps", "has_occlusion", "has_turn"]
    top = d[d.qualifies].head(15)
    print("\nBANG XEP HANG (top 15):")
    print(top[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return d


# ---------------------------------------------------------------------------
# BUILD: chay lai, doi chieu, match cost, render, metadata
# ---------------------------------------------------------------------------
def id_mapping(rerun_boxes: pd.DataFrame, saved: pd.DataFrame, f0: int, f1: int) -> tuple[dict[int, int], dict]:
    """Anh xa ID chay lai -> ID da luu bang bo phieu IoU>=0.9 (ket qua buf90 co truoc cac sua loi)."""
    votes: dict[tuple[int, int], int] = {}
    n_geom = n_raw = 0
    for fr in range(f0, f1 + 1):
        a = rerun_boxes[rerun_boxes.frame == fr]
        b = saved[saved.frame == fr]
        if not len(a) or not len(b):
            continue
        mat = iou_matrix(a[["x", "y", "w", "h"]].to_numpy(float), b[["x", "y", "w", "h"]].to_numpy(float))
        for ri, ci in match_frame(mat, 0.9):
            ra, sb = int(a.id.iloc[ri]), int(b.id.iloc[ci])
            n_geom += 1
            n_raw += int(ra == sb)
            votes[(ra, sb)] = votes.get((ra, sb), 0) + 1
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for (ra, sb), _ in sorted(votes.items(), key=lambda x: x[1], reverse=True):
        if ra not in mapping and sb not in used:
            mapping[ra] = sb
            used.add(sb)
    n_mapped = sum(c for (ra, sb), c in votes.items() if mapping.get(ra) == sb)
    n_saved = int(saved.frame.between(f0, f1).sum())
    n_rerun = int(rerun_boxes.frame.between(f0, f1).sum())
    return mapping, dict(frames=f1 - f0 + 1, saved_boxes=n_saved, rerun_boxes=n_rerun,
                         geometry_iou90_matches=n_geom, geometry_match_pct=round(100 * n_geom / max(1, n_saved), 3),
                         raw_same_id_matches=n_raw, raw_same_id_pct=round(100 * n_raw / max(1, n_saved), 3),
                         mapped_same_id_matches=n_mapped, mapped_same_id_pct=round(100 * n_mapped / max(1, n_saved), 3),
                         rerun_to_saved_id_map={str(k): int(v) for k, v in sorted(mapping.items())})


def remap_ids(df: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["rerun_id"] = out.id.astype(int)
    out["id"] = out.id.astype(int).map(mapping).fillna(out.id).astype(int)
    return out


def target_timeline(gt: pd.DataFrame, pred: pd.DataFrame, gid: int, f0: int, f1: int) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    pb = {int(f): p for f, p in pred[pred.frame.between(f0, f1)].groupby("frame")}
    for fr, g in gt[gt.frame.between(f0, f1)].groupby("frame"):
        if not bool((g.id == gid).any()):
            continue
        p = pb.get(int(fr))
        hit = None
        if p is not None and len(p):
            mat = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
            gids, tids = g.id.to_numpy(), p.id.to_numpy()
            for ri, ci in match_frame(mat, 0.5):
                if int(gids[ri]) == gid:
                    hit = int(tids[ci])
                    break
        out[int(fr)] = hit
    return out


def replay_match_cost(video: str, gid: int, frame: int, gt: pd.DataFrame, pred_pool: dict[str, pd.DataFrame],
                      old_ids: dict[str, int], timelines: dict[str, dict[int, int | None]]) -> pd.DataFrame:
    """Replay phep ghep ByteTrack tai MOT frame: IoU, fused = IoU x score, cost = 1 - fused, LAP day du."""
    import cv2
    from ultralytics import YOLO
    from ultralytics.trackers.utils import matching
    from baseline_track import load_ignored_regions, mask_ignored_regions

    root = config.find_images_root() / video
    img = mask_ignored_regions(cv2.imread(str(root / f"img{frame:05d}.jpg")), load_ignored_regions(video, SPLIT))
    res = YOLO(str(config.MODELS_DIR / "yolov8n.pt")).predict(img, conf=CONF, classes=config.YOLO_VEHICLE_CLASSES,
                                                               device="0", verbose=False)[0]
    det_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.cpu().numpy().astype(float)
    g = gt[(gt.frame == frame) & (gt.id == gid)]
    det_tlwh = np.c_[det_xyxy[:, 0], det_xyxy[:, 1], det_xyxy[:, 2] - det_xyxy[:, 0], det_xyxy[:, 3] - det_xyxy[:, 1]]
    gt_iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), det_tlwh)[0]
    j = int(np.argmax(gt_iou))

    class Det:
        def __init__(self, s): self.score = s
    dets = [Det(float(s)) for s in scores]
    rows = []
    for key, (name, _cfg, label, _mm) in TRACKERS.items():
        pool = pred_pool[key][pred_pool[key].frame == frame].reset_index(drop=True)
        boxes = [np.array([r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2], dtype=np.float32)
                 for r in pool.itertuples()]
        raw = matching.iou_distance(boxes, list(det_xyxy))
        fused = matching.fuse_score(raw, dets)
        matches, _, _ = matching.linear_assignment(fused, thresh=0.8)
        pairs = [(int(a), int(b)) for a, b in matches]
        by_row = dict(pairs)
        winner = next((ri for ri, di in pairs if di == j), None)
        old = old_ids[key]
        idx = pool.index[pool.id == old]
        if not len(idx):
            rows.append(dict(frame=frame, model=label, tracker=name, old_track_id=old, note="track cu khong con trong pool"))
            continue
        ri = int(idx[0])
        rows.append(dict(
            frame=frame, model=label, tracker=name, old_track_id=old, old_track_state=str(pool.state.iloc[ri]),
            pred_x1=float(boxes[ri][0]), pred_y1=float(boxes[ri][1]), pred_x2=float(boxes[ri][2]), pred_y2=float(boxes[ri][3]),
            det_x1=float(det_xyxy[j, 0]), det_y1=float(det_xyxy[j, 1]), det_x2=float(det_xyxy[j, 2]), det_y2=float(det_xyxy[j, 3]),
            detection_score=float(scores[j]), detection_iou_with_gt=float(gt_iou[j]),
            predict_detection_iou=float(1.0 - raw[ri, j]),
            fused_similarity_iou_times_score=float(1.0 - fused[ri, j]),
            bytetrack_cost=float(fused[ri, j]), cost_threshold=0.8,
            direct_pair_passes=bool(fused[ri, j] < 0.8),
            full_lap_assigned_det_index=by_row.get(ri), target_det_index=j,
            old_track_wins_target_det_in_full_lap=bool(by_row.get(ri) == j),
            target_det_winner_track_id=(int(pool.id.iloc[winner]) if winner is not None else None),
            actual_target_id_after_frame=timelines[key].get(frame),
            old_id_preserved_in_saved=bool(timelines[key].get(frame) == old),
        ))
    return pd.DataFrame(rows)


# ---- ve ----
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
CANVAS = (1920, 1080)
PANEL_W, PANEL_H = 640, 720
CROP_W, CROP_H = 400, 450
OUT_FPS = 12
COL_GT = (60, 210, 60)
COL_FAIL = (40, 40, 230)
CAPTION = ("Ví dụ minh họa trên tập train với track_buffer=90: trong trường hợp này, EKF duy trì ID của xe "
           "đang di chuyển lâu hơn CV. Đây là case study được chọn, không chứng minh EKF vượt trội trên toàn bộ dữ liệu.")


def color_for_id(tid: int) -> tuple[int, int, int]:
    import cv2
    hue = (int(hashlib.md5(str(tid).encode()).hexdigest(), 16) % 150) + 15
    if 45 <= hue <= 75:
        hue += 40
    b, g, r = cv2.cvtColor(np.uint8([[[hue % 180, 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def dashed_rect(im, box, color, thickness=2, dash=8):
    import cv2
    x1, y1, x2, y2 = map(int, box)
    for xa in range(x1, x2, dash * 2):
        cv2.line(im, (xa, y1), (min(xa + dash, x2), y1), color, thickness)
        cv2.line(im, (xa, y2), (min(xa + dash, x2), y2), color, thickness)
    for ya in range(y1, y2, dash * 2):
        cv2.line(im, (x1, ya), (x1, min(ya + dash, y2)), color, thickness)
        cv2.line(im, (x2, ya), (x2, min(ya + dash, y2)), color, thickness)


def dashed_line(im, p0, p1, col, thick=2, dash=8, gap=6):
    import cv2
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    if L < 1:
        return
    u = (p1 - p0) / L
    s = 0.0
    while s < L:
        a, b = p0 + u * s, p0 + u * min(s + dash, L)
        cv2.line(im, tuple(a.astype(int)), tuple(b.astype(int)), col, thick, cv2.LINE_AA)
        s += dash + gap


def pil_text(im, text, xy, size, color=(255, 255, 255), bold=False, max_width=None):
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_B if bold else FONT, size)
    if max_width:
        lines, cur = [], ""
        for w in text.split():
            t = (cur + " " + w).strip()
            if dr.textlength(t, font=font) <= max_width:
                cur = t
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        text = "\n".join(lines)
    dr.multiline_text(xy, text, font=font, fill=(color[2], color[1], color[0]), spacing=6)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render(video: str, gid: int, clip: tuple[int, int], f_fail: int, keys: dict[str, int],
           gt: pd.DataFrame, saved: dict[str, pd.DataFrame], states: dict[str, pd.DataFrame],
           timelines: dict[str, dict[int, int | None]], cand: pd.Series, costs: pd.DataFrame | None,
           out_dir: Path, mp4_name: str = "demo_ekf_wins_cv_buffer90.mp4", caption: str = CAPTION,
           turn: dict | None = None) -> dict:
    """turn (tuy chon): dict(start, end, W, angle) -> ve toan bo quy dao GT trong clip + hai mui ten
    huong truoc/sau doan re va nhan goc, de nguoi xem thay xe doi huong that."""
    import cv2
    root = config.find_images_root() / video
    t = gt[(gt.id == gid) & gt.frame.between(*clip)]
    sample = cv2.imread(str(root / f"img{clip[0]:05d}.jpg"))
    h0, w0 = sample.shape[:2]
    # Vung crop CO DINH bao trum toan bo quy dao target trong clip (+ le), cung ti le
    # voi panel; neu lon hon anh thi cat theo anh va letterbox vao panel.
    margin = 40
    ux1, uy1 = float(t.x.min()) - margin, float(t.y.min()) - margin
    ux2, uy2 = float((t.x + t.w).max()) + margin, float((t.y + t.h).max()) + margin
    cw, ch = ux2 - ux1, uy2 - uy1
    aspect = PANEL_W / PANEL_H
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cw, ch = min(cw, w0), min(ch, h0)
    cx_, cy_ = (ux1 + ux2) / 2, (uy1 + uy2) / 2
    x0 = int(np.clip(cx_ - cw / 2, 0, w0 - cw))
    y0 = int(np.clip(cy_ - ch / 2, 0, h0 - ch))
    crop = (x0, y0, x0 + int(cw), y0 + int(ch))
    scale = min(PANEL_W / (crop[2] - crop[0]), PANEL_H / (crop[3] - crop[1]))
    draw_w, draw_h = int((crop[2] - crop[0]) * scale), int((crop[3] - crop[1]) * scale)
    off_x, off_y = (PANEL_W - draw_w) // 2, (PANEL_H - draw_h) // 2

    def P(x, y):  # toa do anh -> toa do panel
        return int((x - x0) * scale) + off_x, int((y - y0) * scale) + off_y

    raw_path = out_dir / "_raw.mp4"
    vw = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, CANVAS)
    keyframes: dict[str, np.ndarray] = {}
    ids_before = {"cv": int(cand.cv_id_before), "ekf": int(cand.ekf_id), "ukf": int(cand.ukf_id_before) if pd.notna(cand.ukf_id_before) else None}
    trails_gt: list = []
    trails = {k: [] for k in TRACKERS}
    last_id = {k: None for k in TRACKERS}
    switch_until = {k: (0, None) for k in TRACKERS}
    n_switch = {k: 0 for k in TRACKERS}
    lost_streak = {k: 0 for k in TRACKERS}
    hold_ok = {k: 0 for k in TRACKERS}       # so frame lien tiep giu ID ban dau ke tu f_fail

    for fr in range(clip[0], clip[1] + 1):
        src = cv2.imread(str(root / f"img{fr:05d}.jpg"))
        g = t[t.frame == fr]
        if len(g):
            trails_gt.append((float(g.cx.iloc[0]), float(g.cy.iloc[0])))
        if turn is None:
            trails_gt = trails_gt[-36:]
        panels, status = [], []
        for key, (name, _cfg, label, _mm) in TRACKERS.items():
            im = src.copy()
            pf = saved[key][saved[key].frame == fr]
            tid = timelines[key].get(fr)
            sw = None
            if tid is not None and last_id[key] is not None and tid != last_id[key]:
                switch_until[key] = (fr + 15, (last_id[key], tid))
                n_switch[key] += 1
            if fr < switch_until[key][0]:
                sw = switch_until[key][1]
            follow = tid if tid is not None else last_id[key]
            if tid is not None:
                last_id[key] = tid
                lost_streak[key] = 0
            else:
                lost_streak[key] += 1
            # box moi xe (mong), target (day)
            for r in pf.itertuples():
                col = color_for_id(int(r.id))
                is_t = tid is not None and int(r.id) == tid
                cv2.rectangle(im, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)),
                              COL_FAIL if (is_t and sw) else col, 3 if is_t else 1)
                if is_t:
                    trails[key].append(((float(r.cx), float(r.cy)), col))
            # du doan khi mat dau (tu chay lai, da anh xa ID)
            in_lost = None
            if tid is None and follow is not None:
                st = states[key]
                row = st[(st.frame == fr) & (st.id == follow)]
                if len(row) and row.iloc[0].state == "lost":
                    row = row.iloc[0]
                    in_lost = (follow, row.cx, row.cy, row.w, row.h)
                    trails[key].append(((float(row.cx), float(row.cy)), color_for_id(int(follow))))
            trails[key] = trails[key][-36:]
            # crop + phong to (letterbox neu ti le khong khop)
            panel = np.full((PANEL_H, PANEL_W, 3), 18, np.uint8)
            panel[off_y:off_y + draw_h, off_x:off_x + draw_w] = cv2.resize(
                im[crop[1]:crop[3], crop[0]:crop[2]], (draw_w, draw_h), interpolation=cv2.INTER_CUBIC)
            # GT: box dut + quy dao lien (xanh la)
            if len(g):
                gr = g.iloc[0]
                dashed_rect(panel, (*P(gr.x, gr.y), *P(gr.x + gr.w, gr.y + gr.h)), COL_GT, 2)
            for i in range(1, len(trails_gt)):
                cv2.line(panel, P(*trails_gt[i - 1]), P(*trails_gt[i]), COL_GT, 2, cv2.LINE_AA)
            if turn is not None:
                # mui ten huong TRUOC (cua so [start-W, start]) va SAU (cua so [end, end+W]) doan re
                tt = gt[gt.id == gid].set_index("frame")
                for (fa, fb), col in (((turn["start"] - turn["W"], turn["start"]), (255, 255, 255)),
                                      ((turn["end"], turn["end"] + turn["W"]), (0, 220, 255))):
                    if fa in tt.index and fb in tt.index:
                        cv2.arrowedLine(panel, P(tt.loc[fa].cx, tt.loc[fa].cy), P(tt.loc[fb].cx, tt.loc[fb].cy),
                                        col, 3, cv2.LINE_AA, tipLength=0.25)
                if turn["start"] - turn["W"] <= fr <= turn["end"] + turn["W"]:
                    cv2.putText(panel, f"RE {turn['angle']:.0f} deg", (14, PANEL_H - 16), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (0, 220, 255), 2, cv2.LINE_AA)
            # quy dao du doan (net dut, mau track)
            for i in range(1, len(trails[key])):
                dashed_line(panel, P(*trails[key][i - 1][0]), P(*trails[key][i][0]), trails[key][i][1], 2)
            if in_lost:
                _, cx, cy, w, h = in_lost
                dashed_rect(panel, (*P(cx - w / 2, cy - h / 2), *P(cx + w / 2, cy + h / 2)), color_for_id(int(follow)), 2)
            # nhan ID
            if tid is not None and len(pf[pf.id == tid]):
                r = pf[pf.id == tid].iloc[0]
                px, py = P(r.x, r.y)
                col = COL_FAIL if sw else color_for_id(int(tid))
                cv2.putText(panel, f"ID {tid}", (px, max(20, py - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2, cv2.LINE_AA)
                if sw:
                    cv2.rectangle(panel, (px, max(0, py - 62)), (px + 250, max(0, py - 30)), COL_FAIL, -1)
                    cv2.putText(panel, f"DOI ID: {sw[0]} -> {sw[1]}", (px + 6, max(0, py - 38)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            # box detection + predict tai frame quyet dinh (neu co replay)
            if costs is not None and fr == f_fail:
                c = costs[costs.tracker == name]
                if len(c) and "pred_x1" in c and pd.notna(c.iloc[0].get("pred_x1")):
                    c = c.iloc[0]
                    dashed_rect(panel, (*P(c.pred_x1, c.pred_y1), *P(c.pred_x2, c.pred_y2)), (255, 80, 255), 3)
                    cv2.rectangle(panel, P(c.det_x1, c.det_y1), P(c.det_x2, c.det_y2), (255, 220, 40), 3)
            # vien vang khi GT bi che
            if len(g) and float(g.vis.iloc[0]) < 1.0:
                cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 210, 255), 6)
            cv2.rectangle(panel, (0, 0), (PANEL_W, 46), (0, 0, 0), -1)
            cv2.putText(panel, label, (14, 33), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            panels.append(panel)
            if fr >= f_fail and tid == ids_before[key] and hold_ok[key] == fr - f_fail:
                hold_ok[key] += 1
            stt = (f"ID {tid}" if tid is not None else (f"MAT DAU - du doan ID {follow}" if in_lost else "khong ghep duoc"))
            status.append((stt, n_switch[key], lost_streak[key], hold_ok[key]))

        canvas = np.full((CANVAS[1], CANVAS[0], 3), 18, np.uint8)
        canvas[60:780] = np.hstack(panels)
        vis = float(g.vis.iloc[0]) if len(g) else float("nan")
        if fr < f_fail:
            phase = "DANG BI CHE" if (len(g) and vis < 0.5) else "TRUOC FRAME QUYET DINH"
            if turn is not None and turn["start"] <= fr <= turn["end"]:
                phase = "DANG RE"
        elif fr == f_fail:
            phase = "FRAME QUYET DINH - CV DOI ID"
        else:
            phase = "SAU FRAME QUYET DINH"
        cv2.putText(canvas, f"track_buffer=90  |  {video}  GT {gid}  frame {fr}  t={(fr - clip[0]) / 25:4.1f}s  |  {phase}  (GT nhin thay {vis * 100:.0f}%)",
                    (16, 40), cv2.FONT_HERSHEY_DUPLEX, 0.85, (235, 235, 235), 1, cv2.LINE_AA)
        for i, (key, (stt, ns, ls, ho)) in enumerate(zip(TRACKERS, status)):
            col = COL_FAIL if ("MAT" in stt or "khong" in stt) else (200, 255, 200)
            cv2.putText(canvas, stt, (i * PANEL_W + 14, 814), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"doi ID trong clip: {ns}  |  mat dau lien tuc: {ls}  |  giu ID {ids_before[key]} sau frame {f_fail}: {ho} frame",
                        (i * PANEL_W + 14, 846), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        canvas = pil_text(canvas, "Nét liền xanh lá = quỹ đạo & box GT  ·  Nét đứt màu = quỹ đạo/box dự đoán của tracker  ·  "
                                  "Viền vàng = GT đang bị che  ·  Đỏ = đổi ID"
                          + ("  ·  Tím/vàng tại frame quyết định = box dự đoán / detection" if costs is not None else ""),
                          (16, 872), 21, color=(200, 200, 200))
        vw.write(canvas)
        for kname, kf in keys.items():
            if fr == kf:
                keyframes[kname] = canvas.copy()

    final = pil_text(canvas.copy(), caption, (18, 912), 27, bold=True, max_width=1880)
    for _ in range(int(3.5 * OUT_FPS)):
        vw.write(final)
    vw.release()
    import imageio_ffmpeg
    mp4 = out_dir / mp4_name
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart", str(mp4)], check=True)
    raw_path.unlink()
    for kname, im in keyframes.items():
        cv2.imwrite(str(out_dir / f"{kname}.png"), im)
    cap = cv2.VideoCapture(str(mp4))
    info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), codec="H.264 (libx264)")
    cap.release()
    info["duration_s"] = round(info["frames"] / OUT_FPS, 2)
    return dict(crop=list(crop), output=info, caption=caption, hold_after_fail_in_clip=hold_ok)


def build(out_dir: Path, rank: int = 1, reuse_rerun: bool = False) -> None:
    """reuse_rerun=True: KHONG chay lai tracker, doc lai ket qua cua lan chay lai truoc
    (rerun/*.csv + rerun/verification.json do chinh script nay ghi) de render lai.
    Mac dinh False: chay lai ba tracker tu frame 1, khong dung cache."""
    from rerun_with_state_log import compare_to_saved, rerun
    cands = pd.read_csv(out_dir / "candidates_single_case.csv")
    q = cands[cands.qualifies & (cands["rank"] == rank)]
    if not len(q):
        raise SystemExit(f"Khong co ung vien hang {rank} trong candidates_single_case.csv")
    c = q.iloc[0]
    video, gid, f_fail = str(c.video), int(c.gt_id), int(c.f_fail)
    clip = (max(int(c.gt_first), f_fail - WIN), min(int(c.gt_last), f_fail + max(WIN, min(int(c.ekf_hold_until_change), 50))))
    print(f"Ung vien hang {rank}: {video} GT {gid}, CV {c.cv_kind} tai frame {f_fail} ({c.cv_id_before} -> {c.cv_id_after}), "
          f"EKF giu ID {int(c.ekf_id)} them {int(c.ekf_hold_after)} frame lien tiep / {int(c.ekf_hold_until_change)} frame toi khi doi. Clip {clip}.")
    rerun_dir = out_dir / "rerun"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    gt = load_gt(video)

    saved, states, pools, maps, verify, runtime = {}, {}, {}, {}, {}, {}
    ver_json = rerun_dir / "verification.json"
    prior = json.load(open(ver_json, encoding="utf-8")) if (reuse_rerun and ver_json.exists()) else None
    if reuse_rerun and prior is None:
        raise SystemExit("--reuse-rerun: chua co rerun/verification.json, phai chay build day du truoc")
    for key, (name, cfg, label, mm) in TRACKERS.items():
        saved[key] = load_saved(video, name)
        if prior is not None:
            if prior["clip"] != list(clip) or prior["video"] != video:
                raise SystemExit("--reuse-rerun: ket qua chay lai tren dia thuoc clip/video khac")
            runtime[key] = prior["runtime"][key]
            verify[key] = prior["verify"][key]
            maps[key] = {int(k): v for k, v in prior["verify"][key]["rerun_to_saved_id_map"].items()}
            b2 = pd.read_csv(rerun_dir / f"rerun_boxes_{name}.csv")
            s2 = pd.read_csv(rerun_dir / f"rerun_states_{name}.csv")
            p2 = pd.read_csv(rerun_dir / f"rerun_pred_{name}.csv")
            states[key], pools[key] = s2, p2
            filt, mapped = verify[key].get("filter_runtime", "?"), verify[key]
        else:
            runtime[key] = {}
            boxes, st, pool, filt = rerun(video, SPLIT, mm, CONF, clip[1], tracker_cfg=cfg, runtime_info=runtime[key])
            basic = compare_to_saved(boxes, saved[key], clip[0], clip[1])
            mapping, mapped = id_mapping(boxes, saved[key], clip[0], clip[1])
            mapped["basic_check_raw_ids"] = basic
            mapped["filter_runtime"] = filt
            maps[key] = mapping
            verify[key] = mapped
            b2, s2, p2 = remap_ids(boxes, mapping), remap_ids(st, mapping), remap_ids(pool, mapping)
            b2.to_csv(rerun_dir / f"rerun_boxes_{name}.csv", index=False)
            s2.to_csv(rerun_dir / f"rerun_states_{name}.csv", index=False)
            p2.to_csv(rerun_dir / f"rerun_pred_{name}.csv", index=False)
            states[key], pools[key] = s2, p2
        # ket cuc target trong lan chay lai (sau anh xa) phai trung voi da luu
        tl_saved = target_timeline(gt, saved[key], gid, clip[0], clip[1])
        tl_rerun = target_timeline(gt, b2, gid, clip[0], clip[1])
        same = sum(int(tl_saved.get(f) == tl_rerun.get(f)) for f in tl_saved)
        verify[key]["target_timeline_frames"] = len(tl_saved)
        verify[key]["target_timeline_same_after_mapping"] = same
        verify[key]["target_timeline_same_pct"] = round(100 * same / max(1, len(tl_saved)), 2)
        verify[key]["target_id_at_fail_saved"] = tl_saved.get(f_fail)
        verify[key]["target_id_at_fail_rerun"] = tl_rerun.get(f_fail)
        verify[key]["target_id_before_fail_saved"] = tl_saved.get(f_fail - 1)
        verify[key]["target_id_before_fail_rerun"] = tl_rerun.get(f_fail - 1)
        print(f"  [{label:<10}] {filt} buffer={runtime[key].get('track_buffer')} | hinh hoc {mapped['geometry_match_pct']}% | "
              f"ID tho {mapped['raw_same_id_pct']}% | sau anh xa {mapped['mapped_same_id_pct']}% | "
              f"timeline target trung {same}/{len(tl_saved)}")
        if runtime[key].get("track_buffer") != 90:
            raise SystemExit(f"track_buffer runtime != 90 cho {name}: {runtime[key]}")
        if mapped["geometry_match_pct"] < 98.0 or mapped["mapped_same_id_pct"] < 98.0:
            raise SystemExit(f"Lan chay lai KHONG tai lap prediction da luu cho {name}: {mapped}")
    timelines = {k: target_timeline(gt, saved[k], gid, clip[0], clip[1]) for k in TRACKERS}
    for key in ("cv", "ekf"):
        v = verify[key]
        if v["target_id_at_fail_rerun"] != v["target_id_at_fail_saved"] or v["target_id_before_fail_rerun"] != v["target_id_before_fail_saved"]:
            raise SystemExit(f"Ket cuc target tai f_fail KHONG tai lap cho {key}: {v}")
    if prior is None:
        with open(ver_json, "w", encoding="utf-8") as f:
            json.dump(dict(video=video, gt_id=gid, clip=list(clip), runtime=runtime, verify=verify), f,
                      ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- match cost: chi khi CV that bai la mot quyet dinh ghep (switched) VA co che khuat / tai lien ket ----
    gap = int(c.cv_gap_before_fail)
    reassoc = bool(c.cv_kind == "switched" and (gap > 0 or bool(c.has_occlusion)))
    costs = None
    if reassoc:
        old_ids = {"cv": int(c.cv_id_before), "ekf": int(c.ekf_id), "ukf": int(c.ukf_id_before)}
        costs = replay_match_cost(video, gid, f_fail, gt, pools, old_ids, timelines)
        costs.to_csv(out_dir / "match_cost_at_key_frame.csv", index=False)
        print(costs.to_string(index=False))
    else:
        with open(out_dir / "match_cost_at_key_frame.csv", "w", encoding="utf-8") as f:
            f.write("# KHONG AP DUNG: CV that bai kieu '%s', khong che khuat, %d frame khong ghep ngay truoc do -> "
                    "khong giai thich bang association, chi mo ta ket cuc tracking quan sat duoc.\n" % (c.cv_kind, gap))

    keys = {"before": max(clip[0], f_fail - gap - 1) if gap > 0 else max(clip[0], f_fail - 10),
            "failure_or_occlusion": (f_fail - max(1, gap // 2)) if gap > 0 else f_fail,
            "after": min(clip[1], f_fail + 10)}
    render_info = render(video, gid, clip, f_fail, keys, gt, saved, states, timelines, c, costs, out_dir)

    # id_match / id_continuous theo tinh than occlusion_eval, ap dung cho doan
    # [frame ghep cuoi TRUOC khi mat dau (f_fail - gap - 1), f_fail + 10]:
    #   id_match      : ID o frame dau doan == ID o f_fail == ID goc cua tracker
    #   id_continuous : ID goc duoc ghep o MOI frame GT trong doan (khong None, khong doi)
    f_before = f_fail - gap - 1
    def id_metrics(tl, ref):
        vals = [tl.get(f) for f in range(f_before, f_fail + 11) if f in tl]
        return dict(window=[f_before, f_fail + 10], id_match=bool(tl.get(f_before) == ref and tl.get(f_fail) == ref),
                    id_continuous=bool(vals and all(v == ref for v in vals)))
    win_metrics = {k: id_metrics(timelines[k], {"cv": int(c.cv_id_before), "ekf": int(c.ekf_id), "ukf": int(c.ukf_id_before)}[k]) for k in TRACKERS}

    meta = dict(
        demo="demo_ekf_wins_cv_buffer90",
        selection=dict(rank=rank, table=str(out_dir / "candidates_single_case.csv"),
                       criteria=dict(min_median_speed_px=MIN_MEDIAN_SPEED, min_path_px=MIN_PATH, min_net_over_diag=MIN_NET_OVER_DIAG,
                                     min_ekf_hold_frames=MIN_EKF_HOLD, min_established_frames=MIN_ESTABLISHED,
                                     max_ekf_start_offset=MAX_EKF_START_OFFSET, window=WIN)),
        source=dict(split=SPLIT, part=PART, video=video, gt_id=gid, clip_frames=list(clip), cv_failure_frame=f_fail,
                    cv_failure_kind=str(c.cv_kind), keyframes=keys),
        configuration=dict(condition="Stage E mo rong: conf=0.25, track_buffer=90 (khong phai A/B)", detector="models/yolov8n.pt",
                           weights_sha256=hashlib.sha256(open(config.MODELS_DIR / "yolov8n.pt", "rb").read()).hexdigest(),
                           conf=CONF, classes=config.YOLO_VEHICLE_CLASSES, tracker_runtime=runtime,
                           saved_results={k: str(config.PROCESSED_DIR / "trackers" / SPLIT / v[0] / "data" / f"{video}.txt") for k, v in TRACKERS.items()}),
        candidate={k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in c.to_dict().items()},
        motion=dict(median_speed_px=float(c.median_speed_px), mean_speed_px=float(c.mean_speed_px), path_px=float(c.path_px),
                    net_px=float(c.net_px), median_bbox_diag_px=float(c.median_bbox_diag_px), net_over_diag=float(c.net_over_diag),
                    turn_deg=(None if pd.isna(c.turn_deg) else float(c.turn_deg)), window=[int(c.win_start), int(c.win_end)]),
        occlusion=dict(has_occlusion=bool(c.has_occlusion), occ_ratio_near_fail=float(c.occ_ratio_near_fail),
                       occ_segment_overlaps=bool(c.occ_segment_overlaps), min_vis_in_window=float(c.min_vis)),
        outcomes={k: dict(id_before=int(ids) if ids is not None else None,
                          id_at_fail=timelines[k].get(f_fail), id_after_10=timelines[k].get(f_fail + 10),
                          hold_consecutive_after_fail=int({"cv": 0, "ekf": c.ekf_hold_after, "ukf": c.ukf_hold_after}[k]),
                          hold_until_id_change_after_fail=int({"cv": 0, "ekf": c.ekf_hold_until_change, "ukf": c.ukf_hold_until_change}[k]),
                          **win_metrics[k])
                  for k, ids in {"cv": int(c.cv_id_before), "ekf": int(c.ekf_id), "ukf": int(c.ukf_id_before)}.items()},
        rerun_verification=verify,
        rerun_source=("doc lai tu rerun/ (lan chay lai khong cache truoc do, cung phien)" if prior is not None
                      else "chay lai ba tracker tu frame 1, khong dung cache"),
        association_replay=(None if costs is None else costs.to_dict("records")),
        association_note=("Replay offline tai frame that bai tren predicted pool cua lan chay lai + detection tinh lai; "
                          "khong phai log runtime lich su." if reassoc else
                          "Khong ap dung: CV that bai khong phai tai lien ket sau doan mat dau; chi mo ta ket cuc."),
        render=render_info,
        interpretation_limit=("Mot case study chon co chu dich tren tap train theo tieu chi EKF thang CV; "
                              "khong phai bang chung hieu nang tong quat, khong dung tap test."),
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"-> {out_dir / 'demo_ekf_wins_cv_buffer90.mp4'}  {render_info['output']}")


def context(out_dir: Path) -> pd.DataFrame:
    """Phep quet DOI XUNG cho moi cap tracker (A that bai, B giu dung ID) voi CUNG
    tieu chi, de dat case study vao boi canh: neu chieu nguoc lai cung xay ra
    thuong xuyen thi case study khong phai bang chung hieu nang tong quat."""
    videos = sorted(p.stem for p in (config.PROCESSED_DIR / "trackers" / SPLIT / TRACKERS["cv"][0] / "data").glob("*.txt"))
    pairs = [("cv", "ekf"), ("ekf", "cv"), ("cv", "ukf"), ("ukf", "cv"), ("ekf", "ukf"), ("ukf", "ekf")]
    counts = {p: dict(fail=0, hold_ge10=0, fail_persists=0, stable_before=0, moving=0) for p in pairs}
    for vi, v in enumerate(videos, 1):
        gt = load_gt(v)
        tl = {k: timelines_for_video(gt, load_saved(v, TRACKERS[k][0])) for k in TRACKERS}
        for a, b in pairs:
            for gid, ft in tl[a].items():
                frames = list(ft)
                if len(frames) < 2 * MIN_ESTABLISHED:
                    continue
                fs = [ft[f] for f in frames]
                hs = [tl[b].get(gid, {}).get(f) for f in frames]
                t = None
                for fail in cv_failures(fs):
                    i = fail["i"]
                    counts[(a, b)]["fail"] += 1
                    if ekf_wins_at(hs, i) is None:
                        continue
                    counts[(a, b)]["hold_ge10"] += 1
                    if any(x == fail["old"] for x in fs[i:i + MIN_EKF_HOLD]):
                        continue
                    counts[(a, b)]["fail_persists"] += 1
                    if n_id_changes(fs, i - WIN, i) or n_id_changes(hs, i - WIN, i):
                        continue
                    counts[(a, b)]["stable_before"] += 1
                    t = gt[gt.id == gid] if t is None else t
                    mm = motion_metrics(t, frames[i] - WIN, frames[i] + WIN)
                    if (mm.get("win_frames", 0) >= MIN_WIN_FRAMES and mm["median_speed_px"] >= MIN_MEDIAN_SPEED
                            and mm["path_px"] >= MIN_PATH and mm["net_over_diag"] >= MIN_NET_OVER_DIAG):
                        counts[(a, b)]["moving"] += 1
        print(f"[{vi:2d}/{len(videos)}] {v}", flush=True)
    rows = [dict(fails=a, holds=b, **c) for (a, b), c in counts.items()]
    d = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / "symmetric_context.csv", index=False)
    print(d.to_string(index=False))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "build", "context"])
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--reuse-rerun", action="store_true",
                    help="Render lai tu ket qua chay lai da co tren dia (rerun/), khong chay lai tracker")
    a = ap.parse_args()
    out_dir = Path(a.out)
    if a.cmd == "scan":
        scan(out_dir)
    elif a.cmd == "context":
        context(out_dir)
    else:
        build(out_dir, a.rank, reuse_rerun=a.reuse_rerun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
