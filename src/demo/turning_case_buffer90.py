"""
turning_case_buffer90.py -- Tim va dung MOT demo "xe RE THAT SU" tren tap train,
                            track_buffer=90, trong do EKF+CTRV giu ID tot hon KF+CV.

Dung lai toan bo may quet cua single_case_buffer90.py (timeline ghep GT<->tracker,
dinh nghia CV that bai / EKF thang / that bai keo dai / on dinh truoc do) va them
mot bo phat hien RE THAT SU tu GT.

"XE RE THAT SU" (do tren GT, KHONG dung goc doi huong cu cua cua so demo):
  Hai cua so on dinh TRUOC va SAU thoi diem re, moi cua so W = 10 frame GT lien tiep
  (>= 5 theo yeu cau; W = 10 de heading it nhieu hon). Vector huong truoc
  v1 = pos[a] - pos[a - W], vector huong sau v2 = pos[b + W] - pos[b], voi b - a <= 40
  (doan re keo dai toi da 1,6 s). Dat dieu kien khi:
    1. toc do trung vi trong MOI cua so >= 3 px/frame;
    2. |v1|, |v2| >= 15 px (vector qua nho -> goc vo nghia);
    3. GT nhin thay >= 50 % o moi frame cua cua so (vi tri GT khi bi che hoan toan la
       noi suy cua nguoi gan nhan, heading tu do khong tin duoc - vd MVI_40991 t27);
    4. box GT khong cham bien anh trong cua so (box bi cat lam tam box lech - vd
       MVI_40191 t92 o frame 601-605);
    5. goc giua v1 va v2 >= 30 deg.
  Voi moi object lay cac dinh goc khong giao nhau. Doan re = [a, b] (tu cuoi cua so
  truoc den dau cua so sau); ca hai cua so nam trong [a - W, b + W].

"EKF THANG CV" cho demo: nhu single_case_buffer90 (CV switched/lost voi ID da on dinh,
that bai keo dai >= 10 frame, EKF giu dung ID >= 10 frame GT lien tiep, ca hai khong
doi ID trong 25 frame truoc), VA f_fail nam trong cua so quanh doan re
[a - W, b + W + 10]. Dich chuyen rong >= 0,5 x duong cheo bbox trung vi va path >= 30 px
do trong cua so demo [a - W, max(b + W, f_fail + 25)].

XEP HANG (quy tac co dinh):
  1. CV that bai TRONG hoac NGAY SAU doan re truoc (dist >= 0, dist = f_fail - b neu sau,
     0 neu trong [a, b]); that bai TRUOC doan re (dist < 0) xep sau; trong moi nhom gan nhat truoc;
  2. so frame EKF giu dung ID sau khi CV that bai lon hon (ekf_hold_until_change, roi ekf_hold_after);
  3. goc re lon hon, roi net/diag lon hon;
  4. co che khuat.
UKF khong dung de loc; ket cuc that cua UKF duoc hien thi.

CACH DUNG
    python src/demo/turning_case_buffer90.py scan
    python src/demo/turning_case_buffer90.py build [--rank N] [--reuse-rerun]
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import single_case_buffer90 as S  # noqa: E402

OUT = Path("outputs/thesis_demos_buffer90_turning")
W = 10                  # do dai cua so heading (frame GT)
MAX_GAP = 40            # b - a toi da
MIN_VEC = 15.0          # |v| toi thieu
MIN_SPEED = 3.0
MIN_VIS = 0.5
MIN_ANGLE = 30.0
BORDER = 2              # px: box GT cach bien anh it nhat chung nay moi tin heading
AFTER_SLACK = 10        # f_fail duoc phep muon hon cuoi cua so sau toi da chung nay
CAPTION = ("Ví dụ minh họa trên tập train với track_buffer=90: trong trường hợp xe rẽ này, EKF duy trì ID lâu hơn CV. "
           "Đây là một case study được chọn, không chứng minh EKF vượt trội trên toàn bộ dữ liệu.")


def image_size(video: str) -> tuple[int, int]:
    cp = configparser.ConfigParser()
    cp.read(config.PROCESSED_DIR / S.SPLIT / video / "seqinfo.ini")
    return int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"])


def detect_turns(t: pd.DataFrame, img_w: int, img_h: int) -> list[dict]:
    """Cac doan re that su cua MOT object GT (xem docstring dau file)."""
    t = t.sort_values("frame")
    rows = {int(r.frame): r for r in t.itertuples(index=False)}

    def window(f0: int, f1: int):
        pts = []
        for f in range(f0, f1 + 1):
            r = rows.get(f)
            if r is None or r.vis < MIN_VIS:
                return None
            if r.x < BORDER or r.y < BORDER or r.x + r.w > img_w - BORDER or r.y + r.h > img_h - BORDER:
                return None
            pts.append((r.cx, r.cy))
        pts = np.array(pts)
        v = pts[-1] - pts[0]
        n = float(np.hypot(*v))
        sp = float(np.median(np.hypot(*np.diff(pts, axis=0).T)))
        if n < MIN_VEC or sp < MIN_SPEED:
            return None
        return v, n, sp

    cands = []
    frames = sorted(rows)
    for a in frames:
        r1 = window(a - W, a)
        if r1 is None:
            continue
        h1 = float(np.degrees(np.arctan2(r1[0][1], r1[0][0])))
        for b in range(a, a + MAX_GAP + 1):
            r2 = window(b, b + W)
            if r2 is None:
                continue
            h2 = float(np.degrees(np.arctan2(r2[0][1], r2[0][0])))
            ang = abs((h2 - h1 + 180) % 360 - 180)
            if ang >= MIN_ANGLE:
                cands.append(dict(turn_start=a, turn_end=b, turn_len=b - a, angle=ang, heading_before=h1,
                                  heading_after=h2, vec_before_px=r1[1], vec_after_px=r2[1],
                                  speed_before=r1[2], speed_after=r2[2]))
    # giu dinh lon nhat, loai moi ung vien giao voi no (ke ca cua so), lap lai
    cands.sort(key=lambda d: -d["angle"])
    out: list[dict] = []
    for c in cands:
        if all(c["turn_end"] + W < o["turn_start"] - W or c["turn_start"] - W > o["turn_end"] + W for o in out):
            out.append(c)
    return sorted(out, key=lambda d: d["turn_start"])


def scan(out_dir: Path) -> pd.DataFrame:
    videos = sorted(p.stem for p in (config.PROCESSED_DIR / "trackers" / S.SPLIT / S.TRACKERS["cv"][0] / "data").glob("*.txt"))
    occ = pd.read_csv(config.part_file("occlusion_segments.csv", S.PART))
    turn_rows, rows = [], []
    n_obj = n_turn_obj = n_fail_turn_obj = 0
    for vi, video in enumerate(videos, 1):
        gt = S.load_gt(video)
        img_w, img_h = image_size(video)
        tl = {k: S.timelines_for_video(gt, S.load_saved(video, S.TRACKERS[k][0])) for k in S.TRACKERS}
        for gid, t in gt.groupby("id"):
            n_obj += 1
            turns = detect_turns(t, img_w, img_h)
            for tr in turns:
                turn_rows.append(dict(video=video, gt_id=int(gid), **tr))
            if not turns:
                continue
            n_turn_obj += 1
            cv_t = tl["cv"].get(int(gid), {})
            frames = list(cv_t)
            if len(frames) < 2 * S.MIN_ESTABLISHED:
                continue
            cv_seq = [cv_t[f] for f in frames]
            ekf_seq = [tl["ekf"].get(int(gid), {}).get(f) for f in frames]
            ukf_seq = [tl["ukf"].get(int(gid), {}).get(f) for f in frames]
            for fail in S.cv_failures(cv_seq):
                i = fail["i"]
                f_fail = frames[i]
                n_fail_turn_obj += 1
                # doan re gan nhat (theo khoang cach toi loi [a, b])
                def dist_to(tr):
                    return 0 if tr["turn_start"] <= f_fail <= tr["turn_end"] else (
                        f_fail - tr["turn_end"] if f_fail > tr["turn_end"] else f_fail - tr["turn_start"])
                tr = min(turns, key=lambda d: abs(dist_to(d)))
                dist = dist_to(tr)
                in_window = (tr["turn_start"] - W) <= f_fail <= (tr["turn_end"] + W + AFTER_SLACK)
                win = S.ekf_wins_at(ekf_seq, i)
                e_id, e_est = S.run_length_before(ekf_seq, i)
                u_before, _ = S.run_length_before(ukf_seq, i)
                u_hold = S.hold_from(ukf_seq, i, u_before) if (u_before is not None and ukf_seq[i] == u_before) else 0
                d0, d1 = tr["turn_start"] - W, max(tr["turn_end"] + W, f_fail + 25)
                mm = S.motion_metrics(t, d0, d1)
                near = t[t.frame.between(f_fail - 10, f_fail + 2)]
                seg = occ[(occ.video == video) & (occ.track_id == gid) & (occ.start_frame <= d1) & (occ.end_frame >= d0)]
                rows.append(dict(
                    video=video, gt_id=int(gid), gt_first=frames[0], gt_last=frames[-1],
                    f_fail=f_fail, cv_kind=fail["kind"], cv_id_before=fail["old"], cv_id_after=fail["new"],
                    cv_gap_before_fail=fail["gap_before"], cv_lost_streak=fail.get("lost_streak", 0),
                    cv_fail_persists_10=not any(v == fail["old"] for v in cv_seq[i:i + S.MIN_EKF_HOLD]),
                    cv_id_changes_25_before=S.n_id_changes(cv_seq, i - 25, i),
                    ekf_id_changes_25_before=S.n_id_changes(ekf_seq, i - 25, i),
                    ekf_wins=win is not None, ekf_id=e_id, ekf_established_before=e_est, ekf_id_at_fail=ekf_seq[i],
                    ekf_start_offset=(win or {}).get("ekf_start_offset"),
                    ekf_hold_after=(win or {}).get("ekf_hold_after", S.hold_from(ekf_seq, i, e_id) if e_id is not None else 0),
                    ekf_hold_until_change=S.hold_until_change(ekf_seq, i, e_id),
                    ukf_id_before=u_before, ukf_id_at_fail=ukf_seq[i], ukf_hold_after=u_hold,
                    ukf_hold_until_change=S.hold_until_change(ukf_seq, i, u_before),
                    turn_start=tr["turn_start"], turn_end=tr["turn_end"], turn_len=tr["turn_len"],
                    turn_angle_deg=tr["angle"], heading_before_deg=tr["heading_before"], heading_after_deg=tr["heading_after"],
                    vec_before_px=tr["vec_before_px"], vec_after_px=tr["vec_after_px"],
                    speed_before_px=tr["speed_before"], speed_after_px=tr["speed_after"],
                    dist_fail_to_turn=dist, fail_in_turn_window=bool(in_window),
                    demo_win_start=d0, demo_win_end=d1,
                    **{k: v for k, v in mm.items() if k not in ("win_start", "win_end")},
                    occ_ratio_near_fail=float(1.0 - near.vis.min()) if len(near) else np.nan,
                    occ_segment_overlaps=bool(len(seg)),
                ))
        print(f"[{vi:2d}/{len(videos)}] {video}: object co re {sum(1 for r in turn_rows if r['video'] == video)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    turns = pd.DataFrame(turn_rows)
    turns.to_csv(out_dir / "turns_detected.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"\nObject GT: {n_obj} | object co re that su (>= {MIN_ANGLE} deg): {n_turn_obj} ({len(turns)} doan re) | "
          f"thoi diem CV that bai tren cac object do: {len(d)}")
    if not len(d):
        d.to_csv(out_dir / "candidates_turning.csv", index=False)
        return d
    d["stable_before"] = (d.cv_id_changes_25_before == 0) & (d.ekf_id_changes_25_before == 0)
    d["moving_ok"] = (d.win_frames >= S.MIN_WIN_FRAMES) & (d.path_px >= S.MIN_PATH) & (d.net_over_diag >= S.MIN_NET_OVER_DIAG)
    d["has_occlusion"] = (d.occ_ratio_near_fail > 0) | d.occ_segment_overlaps
    d["qualifies"] = d.ekf_wins & d.cv_fail_persists_10 & d.stable_before & d.fail_in_turn_window & d.moving_ok
    d["tier"] = np.where(d.dist_fail_to_turn >= 0, 0, 1)
    d["abs_dist"] = d.dist_fail_to_turn.abs()
    q = d[d.qualifies].sort_values(["tier", "abs_dist", "ekf_hold_until_change", "ekf_hold_after", "turn_angle_deg",
                                    "net_over_diag", "has_occlusion"],
                                   ascending=[True, True, False, False, False, False, False])
    d["rank"] = pd.NA
    d.loc[q.index, "rank"] = range(1, len(q) + 1)

    def reason(r):
        if not r.ekf_wins:
            if r.ekf_id is None or pd.isna(r.ekf_id) or r.ekf_established_before < S.MIN_ESTABLISHED:
                return "loai: EKF chua co ID on dinh truoc khi CV that bai"
            if r.ekf_id_at_fail is not None and not pd.isna(r.ekf_id_at_fail) and r.ekf_id_at_fail != r.ekf_id:
                return "loai: EKF cung doi ID tai f_fail"
            return f"loai: EKF khong giu dung ID >= 10 frame sau khi CV that bai (giu {int(r.ekf_hold_after)})"
        if not r.cv_fail_persists_10:
            return "loai: CV quay ve ID cu trong 10 frame (nhap nhay)"
        if not r.stable_before:
            return f"loai: doi ID trong 25 frame truoc f_fail (CV {r.cv_id_changes_25_before}, EKF {r.ekf_id_changes_25_before})"
        if not r.fail_in_turn_window:
            return f"loai: CV that bai cach doan re {int(r.dist_fail_to_turn)} frame, ngoai cua so [a-W, b+W+{AFTER_SLACK}]"
        if not r.moving_ok:
            return "loai: dich chuyen trong cua so demo khong dat (path/net)"
        return f"giu: hang {int(r['rank'])}"
    d["reason"] = [reason(r) for _, r in d.iterrows()]
    d = d.sort_values(["qualifies", "rank", "ekf_wins", "abs_dist"], ascending=[False, True, False, True], na_position="last")
    d.to_csv(out_dir / "candidates_turning.csv", index=False)
    s1 = d.ekf_wins; s2 = s1 & d.cv_fail_persists_10; s3 = s2 & d.stable_before; s4 = s3 & d.fail_in_turn_window; s5 = s4 & d.moving_ok
    print(f"  EKF giu dung ID >= 10 frame: {int(s1.sum())} | + CV that bai keo dai: {int(s2.sum())} | + on dinh truoc: {int(s3.sum())} "
          f"| + f_fail trong cua so quanh doan re: {int(s4.sum())} | + dich chuyen dat: {int(s5.sum())}")
    show = ["rank", "video", "gt_id", "f_fail", "cv_kind", "cv_id_before", "cv_id_after", "ekf_id", "ekf_hold_after",
            "ekf_hold_until_change", "ukf_id_before", "ukf_hold_until_change", "turn_start", "turn_end", "turn_angle_deg",
            "heading_before_deg", "heading_after_deg", "dist_fail_to_turn", "median_speed_px", "path_px", "net_over_diag",
            "occ_ratio_near_fail"]
    print("\nUng vien dat moi tieu chi:" if s5.sum() else "\nKHONG co ung vien dat dong thoi 're that su' + 'EKF thang CV on dinh'.")
    if s5.sum():
        print(d[d.qualifies][show].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    near = d[s3 & (d.abs_dist <= 60)]
    print("\nCac truong hop EKF thang CV on dinh tren object co re (bat ky khoang cach <= 60 frame):")
    print(near[show + ["reason"]].to_string(index=False, float_format=lambda v: f"{v:.1f}") if len(near) else "  (khong co)")
    return d


def build(out_dir: Path, rank: int = 1, reuse_rerun: bool = False) -> None:
    from rerun_with_state_log import compare_to_saved, rerun
    cands = pd.read_csv(out_dir / "candidates_turning.csv")
    q = cands[cands.qualifies & (cands["rank"] == rank)]
    if not len(q):
        raise SystemExit(f"Khong co ung vien hang {rank} trong candidates_turning.csv - khong dung video.")
    c = q.iloc[0]
    video, gid, f_fail = str(c.video), int(c.gt_id), int(c.f_fail)
    a, b = int(c.turn_start), int(c.turn_end)
    clip = (max(int(c.gt_first), a - W - 10), min(int(c.gt_last), max(b + W, f_fail + max(25, min(int(c.ekf_hold_until_change), 50)))))
    print(f"Ung vien hang {rank}: {video} GT {gid}, re {c.turn_angle_deg:.1f} deg [{a}-{b}], CV {c.cv_kind} tai {f_fail} "
          f"({c.cv_id_before} -> {c.cv_id_after}), EKF giu {int(c.ekf_id)} {int(c.ekf_hold_after)}/{int(c.ekf_hold_until_change)} frame. Clip {clip}.")
    rerun_dir = out_dir / "rerun"
    rerun_dir.mkdir(parents=True, exist_ok=True)
    gt = S.load_gt(video)
    saved, states, pools, maps, verify, runtime = {}, {}, {}, {}, {}, {}
    ver_json = rerun_dir / "verification.json"
    prior = json.load(open(ver_json, encoding="utf-8")) if (reuse_rerun and ver_json.exists()) else None
    if reuse_rerun and prior is None:
        raise SystemExit("--reuse-rerun: chua co rerun/verification.json")
    for key, (name, cfg, label, mm) in S.TRACKERS.items():
        saved[key] = S.load_saved(video, name)
        if prior is not None:
            if prior["clip"] != list(clip) or prior["video"] != video:
                raise SystemExit("--reuse-rerun: ket qua chay lai tren dia thuoc clip/video khac")
            runtime[key], verify[key] = prior["runtime"][key], prior["verify"][key]
            maps[key] = {int(k): v for k, v in verify[key]["rerun_to_saved_id_map"].items()}
            b2 = pd.read_csv(rerun_dir / f"rerun_boxes_{name}.csv")
            states[key] = pd.read_csv(rerun_dir / f"rerun_states_{name}.csv")
            pools[key] = pd.read_csv(rerun_dir / f"rerun_pred_{name}.csv")
            mapped = verify[key]
        else:
            runtime[key] = {}
            boxes, st, pool, filt = rerun(video, S.SPLIT, mm, S.CONF, clip[1], tracker_cfg=cfg, runtime_info=runtime[key])
            mapping, mapped = S.id_mapping(boxes, saved[key], clip[0], clip[1])
            mapped["basic_check_raw_ids"] = compare_to_saved(boxes, saved[key], clip[0], clip[1])
            mapped["filter_runtime"] = filt
            maps[key], verify[key] = mapping, mapped
            b2, states[key], pools[key] = S.remap_ids(boxes, mapping), S.remap_ids(st, mapping), S.remap_ids(pool, mapping)
            b2.to_csv(rerun_dir / f"rerun_boxes_{name}.csv", index=False)
            states[key].to_csv(rerun_dir / f"rerun_states_{name}.csv", index=False)
            pools[key].to_csv(rerun_dir / f"rerun_pred_{name}.csv", index=False)
            tl_s = S.target_timeline(gt, saved[key], gid, clip[0], clip[1])
            tl_r = S.target_timeline(gt, b2, gid, clip[0], clip[1])
            same = sum(int(tl_s.get(f) == tl_r.get(f)) for f in tl_s)
            verify[key].update(target_timeline_frames=len(tl_s), target_timeline_same_after_mapping=same,
                               target_timeline_same_pct=round(100 * same / max(1, len(tl_s)), 2),
                               target_id_at_fail_saved=tl_s.get(f_fail), target_id_at_fail_rerun=tl_r.get(f_fail))
        print(f"  [{label:<10}] {mapped.get('filter_runtime')} buffer={runtime[key].get('track_buffer')} | hinh hoc {mapped['geometry_match_pct']}% | "
              f"ID tho {mapped['raw_same_id_pct']}% | sau anh xa {mapped['mapped_same_id_pct']}% | timeline target trung "
              f"{mapped['target_timeline_same_after_mapping']}/{mapped['target_timeline_frames']}")
        if runtime[key].get("track_buffer") != 90:
            raise SystemExit(f"track_buffer runtime != 90 cho {name}")
        if mapped["geometry_match_pct"] < 98.0 or mapped["mapped_same_id_pct"] < 98.0:
            raise SystemExit(f"Lan chay lai KHONG tai lap prediction da luu cho {name}: {mapped}")
        if mapped["target_id_at_fail_rerun"] != mapped["target_id_at_fail_saved"]:
            raise SystemExit(f"Ket cuc target tai f_fail KHONG tai lap cho {name}")
    if prior is None:
        with open(ver_json, "w", encoding="utf-8") as f:
            json.dump(dict(video=video, gt_id=gid, clip=list(clip), runtime=runtime, verify=verify), f,
                      ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    timelines = {k: S.target_timeline(gt, saved[k], gid, clip[0], clip[1]) for k in S.TRACKERS}

    gap = int(c.cv_gap_before_fail)
    reassoc = bool(c.cv_kind == "switched" and gap > 0 and bool(c.has_occlusion))
    costs = None
    if reassoc:
        old_ids = {"cv": int(c.cv_id_before), "ekf": int(c.ekf_id), "ukf": int(c.ukf_id_before)}
        costs = S.replay_match_cost(video, gid, f_fail, gt, pools, old_ids, timelines)
        costs.to_csv(out_dir / "match_cost_at_key_frame.csv", index=False)
        print(costs.to_string(index=False))
    else:
        with open(out_dir / "match_cost_at_key_frame.csv", "w", encoding="utf-8") as f:
            f.write("# KHONG AP DUNG: khong co tai lien ket sau che khuat (cv_kind=%s, gap=%d, che khuat=%s) -> "
                    "khong phan tich association; chi chung minh bang rerun va ID truoc/sau.\n" % (c.cv_kind, gap, bool(c.has_occlusion)))
    keys = {"before_turn": max(clip[0], a - W), "during_turn": (a + b) // 2, "after_turn": min(clip[1], max(b + W, f_fail + 10))}
    turn = dict(start=a, end=b, W=W, angle=float(c.turn_angle_deg))
    cand = c.copy()
    render_info = S.render(video, gid, clip, f_fail, keys, gt, saved, states, timelines, cand, costs, out_dir,
                          mp4_name="demo_ekf_turning_buffer90.mp4", caption=CAPTION, turn=turn)
    meta = dict(
        demo="demo_ekf_turning_buffer90",
        selection=dict(rank=rank, table=str(out_dir / "candidates_turning.csv"),
                       turn_definition=dict(W=W, max_gap=MAX_GAP, min_vec_px=MIN_VEC, min_speed=MIN_SPEED, min_vis=MIN_VIS,
                                            min_angle_deg=MIN_ANGLE, border_px=BORDER, after_slack=AFTER_SLACK)),
        source=dict(split=S.SPLIT, part=S.PART, video=video, gt_id=gid, clip_frames=list(clip), cv_failure_frame=f_fail,
                    cv_failure_kind=str(c.cv_kind), turn_segment=[a, b], keyframes=keys),
        configuration=dict(condition="Stage E mo rong: conf=0.25, track_buffer=90", detector="models/yolov8n.pt",
                           weights_sha256=hashlib.sha256(open(config.MODELS_DIR / "yolov8n.pt", "rb").read()).hexdigest(),
                           conf=S.CONF, classes=config.YOLO_VEHICLE_CLASSES, tracker_runtime=runtime),
        candidate={k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in c.to_dict().items()},
        outcomes={k: dict(id_before=int(v), id_at_fail=timelines[k].get(f_fail), id_after_10=timelines[k].get(f_fail + 10))
                  for k, v in {"cv": c.cv_id_before, "ekf": c.ekf_id, "ukf": c.ukf_id_before}.items()},
        rerun_verification=verify, rerun_source=("doc lai tu rerun/" if prior is not None else "chay lai 3 tracker tu frame 1, khong cache"),
        association_replay=(None if costs is None else costs.to_dict("records")),
        render=render_info,
        interpretation_limit="Mot case study chon co chu dich tren tap train; khong phai bang chung hieu nang tong quat; khong dung test.",
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"-> {out_dir / 'demo_ekf_turning_buffer90.mp4'}  {render_info['output']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "build"])
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--reuse-rerun", action="store_true")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(Path(a.out))
    else:
        build(Path(a.out), a.rank, a.reuse_rerun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
