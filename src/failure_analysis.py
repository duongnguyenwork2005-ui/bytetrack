"""
failure_analysis.py -- (MOC 2, muc 2) Phan tich NGUYEN NHAN mat ID quanh su kien che khuat tu LOG QUYET
DINH GHEP cua tracker (assoc_log) + ground truth (GT CHI dung de phan tich, KHONG tham gia tracking).

DAU VAO
    --exp   file thi nghiem (run_experiment) da chay voi assoc_log: true va stage eval (occlusion_eval)
            -> results/phase2/<exp>/<cond>/event_outcomes_<split>.csv (su kien + outcome tung tracker)
            -> data/processed/trackers/<split>/<dir>/{data/<video>.txt, assoc_log/<video>_*.parquet}
    GT      data/processed/<split>/<video>/gt/gt.txt ; detection cache (conf > 0.05) de biet detector co
            "thay" doi tuong o muc conf thap hay khong.

DINH NGHIA (mau so = so su kien DU DIEU KIEN theo GT, occlusion_eval.add_gt_eligibility)
    preserved  (= id_match): ID tracker ghep GT o frame CUOI truoc su kien == ID o frame DAU sau su kien
    switched                : hai ID khac nhau
    lost_after              : co ID truoc, KHONG ghep duoc GT trong cua so sau (GT van con) -> that bai that
    no_match_before         : tracker chua tung ghep GT trong cua so truoc -> khong danh gia duoc phan "giu"
    id_continuous           : preserved VA moi frame co GT trong su kien deu ghep dung ID do (chat hon)

NHAN NGUYEN NHAN (nhieu nhan cho phep; cot `causes`; `primary_cause` theo thu tu uu tien duoi day;
"unknown" khi khong nhan nao co bang chung). Phan biet HIEN TUONG (cot so lieu) va SUY LUAN (cot cause).
    detector_miss        : trong cua so sau su kien khong co detection tier-high (conf >= track_high_thresh)
                           nao IoU >= 0.5 voi GT. Phu: det_low (co det 0.10-0.25) / det_cache (chi 0.05-0.10)
                           / none.
    buffer_expired       : id_before bi XOA (n_lost > track_buffer) truoc frame dau tien detector thay lai GT.
    hijacked_during      : trong su kien, id_before (dang Tracked) ghep detection thuoc GT KHAC (bam vat che).
    prediction_drift     : id_before con song tai frame detector thay lai, nhung IoU(box du doan, det) < 0.2
                           -> khong the qua match_thresh 0.8 (cost = 1 - IoU*conf >= 1 - IoU > 0.8).
    score_fusion_reject  : IoU >= 0.2 nhung 1 - IoU*conf > 0.8 (fuse_score voi conf thap lam roi nguong).
    competition          : cap (id_before, det) hop le nhung det bi track KHAC lay (Hungarian / uu tien).
    track_matched_other  : id_before ghep mot det khac (khong phai det cua GT) tai frame do.
    unknown              : con lai.
Voi `switched`: `id_after_origin` = new_track_from_det (ID moi sinh tu det) | existing_other_track.

GHEP NHAM (refind_quality.csv, dung cho C1/C2): moi su kien refind (track LOST duoc ghep lai) -> GT cua det
duoc ghep (IoU >= 0.5) so voi GT ma track theo truoc khi mat: wrong_refind = khac GT. Ti le tren cac refind
xac dinh duoc ca hai GT.

    python src/failure_analysis.py --exp configs/phase2/experiments/expA_detrac_dev.yaml --check-events
    python src/failure_analysis.py --exp ... --trackers cv ekf ukf pf500 --cond c25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
from occlusion_eval import (GT_COLS, TRK_COLS, CONTEXT_WINDOW, add_gt_eligibility, build_events,  # noqa: E402
                            iou_matrix, match_video)
from run_experiment import load_exp, plan  # noqa: E402
from assoc_log import load_log  # noqa: E402

CAUSE_ORDER = ["detector_miss", "buffer_expired", "hijacked_during", "prediction_drift", "score_fusion_reject",
               "competition", "track_matched_other", "matched_but_eval_mismatch", "unknown"]


def iou_1xn(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    return iou_matrix(box[None, :], boxes)[0]


class VideoContext:
    """Du lieu mot video cho mot tracker: GT, output tracker, ghep GT<->tracker, log, cache."""

    def __init__(self, split: str, video: str, tdir: Path, cache_id: str, classes, max_det, cfg: dict):
        self.video = video
        gt = pd.read_csv(config.PROCESSED_DIR / split / video / "gt" / "gt.txt", header=None, names=GT_COLS)
        self.gt = gt[gt["cls"] == config.MOT_CLASS_ID].reset_index(drop=True)
        self.gt_by_frame = {f: g for f, g in self.gt.groupby("frame")}
        f = tdir / "data" / f"{video}.txt"
        self.trk = (pd.read_csv(f, header=None, names=TRK_COLS) if f.exists() and f.stat().st_size > 0
                    else pd.DataFrame(columns=TRK_COLS))
        m = match_video(self.gt, self.trk)                       # (frame, gt_id, tracker_id, iou)
        self.match = m
        self.trk_to_gt = {(int(r.frame), int(r.tracker_id)): int(r.gt_id) for r in m.itertuples()}
        self.gt_to_trk = {(int(r.frame), int(r.gt_id)): int(r.tracker_id) for r in m.itertuples()}
        self.tracks, self.dets, self.events = load_log(tdir / "assoc_log", video)
        self.tracks_by_id = {k: g.sort_values("frame") for k, g in self.tracks.groupby("track_id")}
        self.dets_by_frame = {k: g for k, g in self.dets.groupby("frame")}
        self.tracks_by_frame = {k: g for k, g in self.tracks.groupby("frame")}
        self.removed_frame = {int(r.track_id): int(r.frame) for r in
                              self.events[self.events.kind == "removed"].itertuples()}
        self.new_frame = {int(r.track_id): int(r.frame) for r in self.events[self.events.kind == "new"].itertuples()}
        self.cache = det_cache.open_cache(cache_id, video, 0.05, classes, max_det)
        self.cache_index = {int(fi): i for i, fi in enumerate(self.cache.frame_idx)}
        self.cfg = cfg
        self.high = float(cfg.get("track_high_thresh", 0.25))
        self.low = float(cfg.get("track_low_thresh", 0.1))
        self.match_thresh = float(cfg.get("match_thresh", 0.8))
        self.buffer = int(cfg.get("track_buffer", 30))

    def gt_box(self, frame: int, gt_id: int):
        g = self.gt_by_frame.get(frame)
        if g is None:
            return None
        r = g[g.id == gt_id]
        return None if not len(r) else r[["x", "y", "w", "h"]].to_numpy(float)[0]

    def dets_at(self, frame: int, tier: str) -> pd.DataFrame:
        d = self.dets_by_frame.get(frame)
        if d is None:
            return pd.DataFrame(columns=self.dets.columns)
        return d[d.tier == tier]

    def cache_iou(self, frame: int, box: np.ndarray) -> tuple[float, float]:
        """(max IoU, conf cua det do) giua GT va MOI det trong cache (conf > 0.05)."""
        i = self.cache_index.get(frame)
        if i is None:
            return 0.0, 0.0
        arr = self.cache.frame(i)
        if not len(arr):
            return 0.0, 0.0
        b = np.c_[arr[:, 0], arr[:, 1], arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]]
        io = iou_1xn(box, b)
        j = int(np.argmax(io))
        return float(io[j]), float(arr[j, 4])

    def track_row(self, frame: int, tid: int):
        g = self.tracks_by_id.get(tid)
        if g is None:
            return None
        r = g[g.frame == frame]
        return None if not len(r) else r.iloc[0]


def analyze_event(vc: VideoContext, ev, context: int) -> dict:
    """Mot su kien x mot tracker -> dict so lieu + nhan nguyen nhan."""
    gt_id, s, e = int(ev.track_id), int(ev.start_frame), int(ev.end_frame)
    rec = dict(video=vc.video, track_id=gt_id, event_id=int(ev.event_id), start_frame=s, end_frame=e,
               length_frames=int(ev.length_frames), has_full=bool(ev.has_full), outcome=ev.outcome,
               id_before=ev.id_before, id_after=ev.id_after, id_continuous=bool(ev.id_continuous))
    idb = None if pd.isna(ev.id_before) else int(ev.id_before)
    ida = None if pd.isna(ev.id_after) else int(ev.id_after)
    causes: list[str] = []
    after = [f for f in range(e + 1, e + context + 1) if vc.gt_box(f, gt_id) is not None]
    rec["n_gt_after"] = len(after)
    # --- 1. detector: frame dau tien co det tier-high IoU >= 0.5 voi GT sau su kien ---
    f_det = None; best_high = 0.0; best_low = 0.0; best_cache = 0.0; best_cache_conf = 0.0
    det_row = None
    for f in after:
        g = vc.gt_box(f, gt_id)
        dh = vc.dets_at(f, "high")
        if len(dh):
            io = iou_1xn(g, dh[["x", "y", "w", "h"]].to_numpy(float))
            j = int(np.argmax(io)); best_high = max(best_high, float(io[j]))
            if io[j] >= 0.5 and f_det is None:
                f_det = f; det_row = dh.iloc[j]; det_iou_gt = float(io[j])
        dl = vc.dets_at(f, "low")
        if len(dl):
            best_low = max(best_low, float(iou_1xn(g, dl[["x", "y", "w", "h"]].to_numpy(float)).max()))
        if f_det is None:
            ci, cc = vc.cache_iou(f, g)
            if ci > best_cache:
                best_cache, best_cache_conf = ci, cc
    rec.update(first_det_frame=f_det, gap_gt_to_det=(None if f_det is None else f_det - e - 1),
               best_iou_high_after=round(best_high, 3), best_iou_low_after=round(best_low, 3),
               best_iou_cache_after=round(best_cache, 3), best_cache_conf=round(best_cache_conf, 3))
    if f_det is None:
        sub = "det_low" if best_low >= 0.5 else ("det_cache" if best_cache >= 0.5 else "none")
        causes.append("detector_miss"); rec["detector_miss_sub"] = sub
    else:
        rec["detector_miss_sub"] = ""
    # --- 2. hijack trong su kien: id_before dang Tracked ghep det cua GT khac ---
    rec["hijack_frame"] = None; rec["hijack_other_gt"] = None
    if idb is not None:
        for f in range(s, e + 1):
            other = vc.trk_to_gt.get((f, idb))
            if other is not None and other != gt_id:
                causes.append("hijacked_during"); rec["hijack_frame"] = f; rec["hijack_other_gt"] = other
                break
    # --- 3. trang thai id_before tai f_det ---
    rec.update(state_at_fdet=None, n_lost_at_fdet=None, iou_pred_det=None, center_offset_px=None,
               fused_cost=None, det_conf=None, det_matched_track=None, det_stage=None, track_matched_det=None,
               competitor_iou=None, removed_frame=vc.removed_frame.get(idb) if idb is not None else None)
    if idb is not None and f_det is not None and ev.outcome in ("switched", "lost_after", "preserved"):
        tr_row = vc.track_row(f_det, idb)
        rm = vc.removed_frame.get(idb)
        if tr_row is None:
            if rm is not None and rm <= f_det:
                causes.append("buffer_expired"); rec["state_at_fdet"] = "removed"
            else:
                rec["state_at_fdet"] = "absent"
        else:
            rec["state_at_fdet"] = {1: "Tracked", 2: "Lost", 4: "Removed", 0: "unconfirmed"}.get(int(tr_row.state), str(tr_row.state))
            rec["n_lost_at_fdet"] = int(tr_row.n_lost)
            pb = np.array([tr_row.x, tr_row.y, tr_row.w, tr_row.h], float)
            db = np.array([det_row.x, det_row.y, det_row.w, det_row.h], float)
            iou = float(iou_1xn(pb, db[None, :])[0])
            rec["iou_pred_det"] = round(iou, 4)
            rec["center_offset_px"] = round(float(np.hypot(pb[0] + pb[2] / 2 - db[0] - db[2] / 2,
                                                          pb[1] + pb[3] / 2 - db[1] - db[3] / 2)), 1)
            conf = float(det_row.conf); rec["det_conf"] = round(conf, 3)
            fused = 1.0 - iou * conf if vc.cfg.get("fuse_score", True) else 1.0 - iou
            rec["fused_cost"] = round(fused, 4)
            rec["det_matched_track"] = int(det_row.matched_track); rec["det_stage"] = int(det_row.stage)
            rec["track_matched_det"] = int(tr_row.matched_det)
            if int(det_row.matched_track) == idb:
                if ev.outcome != "preserved":
                    causes.append("matched_but_eval_mismatch")
            elif iou < 0.2:
                causes.append("prediction_drift")
            elif fused > vc.match_thresh:
                causes.append("score_fusion_reject")
            elif int(det_row.matched_track) >= 0:
                causes.append("competition")
                comp = vc.track_row(f_det, int(det_row.matched_track))
                if comp is not None:
                    cb = np.array([comp.x, comp.y, comp.w, comp.h], float)
                    rec["competitor_iou"] = round(float(iou_1xn(cb, db[None, :])[0]), 4)
            elif int(tr_row.matched_det) >= 0:
                causes.append("track_matched_other")
    # --- 4. nguon cua id_after ---
    if ida is not None and ev.outcome == "switched":
        nf = vc.new_frame.get(ida)
        rec["id_after_origin"] = "new_track_from_det" if (nf is not None and nf > s) else "existing_other_track"
        rec["id_after_new_frame"] = nf
    else:
        rec["id_after_origin"] = ""; rec["id_after_new_frame"] = None
    if ev.outcome == "no_match_before":
        causes = ["not_evaluable_no_match_before"]
    if ev.outcome == "preserved":
        causes = ["success"] + [c for c in causes if c == "hijacked_during"]
    if not causes:
        causes = ["unknown"]
    rec["causes"] = "|".join(causes)
    rec["primary_cause"] = next((c for c in CAUSE_ORDER if c in causes), causes[0])
    return rec


def refind_quality(vc: VideoContext) -> pd.DataFrame:
    """Moi refind: GT cua det duoc ghep vs GT track theo truoc khi mat."""
    rows = []
    for r in vc.events[vc.events.kind == "refind"].itertuples():
        f, tid, didx = int(r.frame), int(r.track_id), int(r.det_idx)
        d = vc.dets_by_frame.get(f)
        d = d[d.det_idx == didx] if d is not None else None
        gt_now = None
        if d is not None and len(d):
            g = vc.gt_by_frame.get(f)
            if g is not None and len(g):
                io = iou_1xn(d[["x", "y", "w", "h"]].to_numpy(float)[0], g[["x", "y", "w", "h"]].to_numpy(float))
                if io.max() >= 0.5:
                    gt_now = int(g.id.to_numpy()[int(np.argmax(io))])
        # GT truoc khi mat: frame Tracked cuoi cung cua track truoc f
        g_prev = vc.tracks_by_id.get(tid)
        gt_prev = None; f_prev = None
        if g_prev is not None:
            prev = g_prev[(g_prev.frame < f) & (g_prev.state == 1)]
            if len(prev):
                f_prev = int(prev.frame.iloc[-1]); gt_prev = vc.trk_to_gt.get((f_prev, tid))
        rows.append(dict(video=vc.video, frame=f, track_id=tid, n_lost=int(r.n_lost), gt_now=gt_now, gt_prev=gt_prev,
                         f_prev=f_prev, known=(gt_now is not None and gt_prev is not None),
                         wrong=(gt_now is not None and gt_prev is not None and gt_now != gt_prev)))
    return pd.DataFrame(rows)


def window_log(vc: VideoContext, ev, rec: dict, context: int) -> pd.DataFrame:
    """Trich log quanh su kien: hang track cua id_before/id_after + det chong len GT (IoU >= 0.3)."""
    gt_id, s, e = int(ev.track_id), int(ev.start_frame), int(ev.end_frame)
    ids = {int(x) for x in (ev.id_before, ev.id_after) if not pd.isna(x)}
    out = []
    for f in range(max(1, s - 5), e + context + 1):
        g = vc.gt_box(f, gt_id)
        tf = vc.tracks_by_frame.get(f)
        if tf is not None and ids:
            for r in tf[tf.track_id.isin(ids)].itertuples():
                iou_gt = float(iou_1xn(g, np.array([[r.x, r.y, r.w, r.h]]))[0]) if g is not None else None
                out.append(dict(video=vc.video, gt_track=gt_id, event_id=int(ev.event_id), frame=f, kind="track",
                                id=int(r.track_id), state=int(r.state), n_lost=int(r.n_lost), conf=round(float(r.score), 3),
                                x=round(r.x, 1), y=round(r.y, 1), w=round(r.w, 1), h=round(r.h, 1),
                                matched=int(r.matched_det), stage=int(r.stage),
                                iou_gt=None if iou_gt is None else round(iou_gt, 3)))
        df = vc.dets_by_frame.get(f)
        if df is not None and g is not None:
            io = iou_1xn(g, df[["x", "y", "w", "h"]].to_numpy(float))
            for r, i in zip(df.itertuples(), io):
                if i >= 0.3:
                    out.append(dict(video=vc.video, gt_track=gt_id, event_id=int(ev.event_id), frame=f, kind=f"det_{r.tier}",
                                    id=int(r.det_idx), state=-1, n_lost=-1, conf=round(float(r.conf), 3),
                                    x=round(r.x, 1), y=round(r.y, 1), w=round(r.w, 1), h=round(r.h, 1),
                                    matched=int(r.matched_track), stage=int(r.stage), iou_gt=round(float(i), 3)))
        if g is not None:
            out.append(dict(video=vc.video, gt_track=gt_id, event_id=int(ev.event_id), frame=f, kind="gt", id=gt_id,
                            state=-1, n_lost=-1, conf=None, x=round(g[0], 1), y=round(g[1], 1), w=round(g[2], 1),
                            h=round(g[3], 1), matched=-1, stage=-1, iou_gt=1.0))
    return pd.DataFrame(out)


def check_events(exp: dict, videos: list[str], context: int) -> pd.DataFrame:
    part = exp.get("occlusion", {}).get("part", exp.get("part", "train"))
    partial = pd.read_csv(config.part_file("occlusion_segments.csv", part))
    full = pd.read_csv(config.part_file("full_occlusion_segments.csv", part))
    partial = partial[partial.video.isin(videos)]; full = full[full.video.isin(videos)]
    gts = []
    for v in videos:
        d = pd.read_csv(config.PROCESSED_DIR / exp["split_name"] / v / "gt" / "gt.txt", header=None, names=GT_COLS)
        d["video"] = v; gts.append(d)
    gt = pd.concat(gts, ignore_index=True); gt = gt[gt["cls"] == config.MOT_CLASS_ID]
    ev = build_events(partial, full, None)
    ev = add_gt_eligibility(ev, gt, context)
    # khong chong lan trong cung (video, track)
    overlap = 0
    for _, g in ev.groupby(["video", "track_id"]):
        g = g.sort_values("start_frame")
        overlap += int((g.start_frame.to_numpy()[1:] <= g.end_frame.to_numpy()[:-1]).sum())
    elig = ev[ev.eligible]
    print(f"[check-events] {len(videos)} video: doan tho partial {len(partial)} + full {len(full)} = {len(partial) + len(full)}")
    print(f"               su kien sau gop (khong dem lap): {len(ev)}; chong lan cung track: {overlap}")
    print(f"               du dieu kien (GT co truoc & sau, cua so {context} frame): {len(elig)} = "
          f"full {int(elig.has_full.sum())} + partial {int((~elig.has_full).sum())}")
    for reason, n in ev.loc[~ev.eligible, "exclusion_reason"].value_counts().items():
        print(f"               loai {n}: {reason}")
    print(f"               do dai su kien du dieu kien (frame): median {elig.length_frames.median():.0f}, "
          f"max {elig.length_frames.max()}; full: median {elig[elig.has_full].length_frames.median():.0f}")
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--cond", nargs="*", default=None)
    ap.add_argument("--trackers", nargs="*", default=None, help="ten tracker trong exp (mac dinh: tat ca, PF chi seed dau)")
    ap.add_argument("--out", default=None, help="mac dinh results/phase2/failure_analysis/<exp>")
    ap.add_argument("--check-events", action="store_true")
    a = ap.parse_args()
    exp = load_exp(a.exp)
    videos = json.loads((config.PROJECT_ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"]
    occ = exp.get("occlusion", {})
    fps = float(occ.get("fps", exp.get("fps", config.FPS)))
    context = CONTEXT_WINDOW if occ.get("context_s") is None else max(1, int(round(float(occ["context_s"]) * fps)))
    out = Path(a.out) if a.out else config.RESULTS_DIR / "phase2" / "failure_analysis" / exp["name"]
    out.mkdir(parents=True, exist_ok=True)
    if a.check_events:
        ev = check_events(exp, videos, context)
        ev.to_csv(out / "events_check.csv", index=False)
    rows = plan(exp)
    conds = a.cond or list(exp["conditions"])
    split = exp["split_name"]
    summary_rows, refind_rows, all_causes, all_windows = [], [], [], []
    for cond in conds:
        eo = config.RESULTS_DIR / "phase2" / exp["name"] / cond / f"event_outcomes_{split}.csv"
        if not eo.exists():
            print(f"[{cond}] thieu {eo} (chay --stage eval truoc)"); continue
        outcomes = pd.read_csv(eo)
        seen = set()
        for r in rows:
            if r["cond"] != cond or (a.trackers and r["tracker"] not in a.trackers) or r["tracker"] in seen:
                continue
            seen.add(r["tracker"])                     # PF: chi seed dau tien
            tdir = config.PROCESSED_DIR / "trackers" / split / r["dir"]
            if not (tdir / "assoc_log").is_dir():
                print(f"  [{cond}/{r['tracker']}] khong co assoc_log -> bo qua"); continue
            meta = json.loads((tdir / "run_meta.json").read_text(encoding="utf-8"))
            cfg = meta["tracker_cfg"]
            ev_t = outcomes[(outcomes.tracker == r["dir"]) & outcomes.eligible]
            recs, wins = [], []
            for v in videos:
                vc = VideoContext(split, v, tdir, exp["cache_id"], exp["classes"], exp["max_det"], cfg)
                for ev in ev_t[ev_t.video == v].itertuples():
                    rec = analyze_event(vc, ev, context)
                    rec.update(cond=cond, tracker=r["tracker"], tracker_dir=r["dir"])
                    recs.append(rec)
                    if ev.has_full or ev.outcome in ("switched", "lost_after"):
                        w = window_log(vc, ev, rec, context)
                        if len(w):
                            w.insert(0, "tracker", r["tracker"]); w.insert(0, "cond", cond); wins.append(w)
                rq = refind_quality(vc); rq["cond"] = cond; rq["tracker"] = r["tracker"]; refind_rows.append(rq)
            df = pd.DataFrame(recs)
            df.to_csv(out / f"event_causes_{cond}_{r['tracker']}.csv", index=False)
            all_causes.append(df)
            if wins:
                all_windows.append(pd.concat(wins, ignore_index=True))
            fail = df[df.outcome.isin(["switched", "lost_after"])]
            print(f"  [{cond}/{r['tracker']:<6}] eligible {len(df)}: preserved {int((df.outcome == 'preserved').sum())}, "
                  f"switched {int((df.outcome == 'switched').sum())}, lost_after {int((df.outcome == 'lost_after').sum())}, "
                  f"no_match_before {int((df.outcome == 'no_match_before').sum())} | nguyen nhan chinh (that bai): "
                  + ", ".join(f"{k} {v}" for k, v in fail.primary_cause.value_counts().items()))
    if not all_causes:
        return 1
    causes = pd.concat(all_causes, ignore_index=True)
    causes.to_csv(out / "event_causes_all.csv", index=False)
    # --- bang tong hop: cause x tracker x tang ---
    fail = causes[causes.outcome.isin(["switched", "lost_after"])].copy()
    fail["stratum"] = np.where(fail.has_full, "full", "partial")
    summ = (fail.groupby(["cond", "tracker", "stratum", "primary_cause"]).size().reset_index(name="n"))
    tot = fail.groupby(["cond", "tracker", "stratum"]).size().reset_index(name="n_fail")
    summ = summ.merge(tot, on=["cond", "tracker", "stratum"])
    summ["frac_of_failures"] = (summ.n / summ.n_fail).round(3)
    n_elig = causes.groupby(["cond", "tracker"]).size().rename("n_eligible").reset_index()
    summ = summ.merge(n_elig, on=["cond", "tracker"])
    summ.to_csv(out / "cause_summary.csv", index=False)
    # bang outcome (mau so ro rang)
    oc = causes.groupby(["cond", "tracker"]).agg(
        n_eligible=("outcome", "size"), preserved=("outcome", lambda s: int((s == "preserved").sum())),
        switched=("outcome", lambda s: int((s == "switched").sum())), lost_after=("outcome", lambda s: int((s == "lost_after").sum())),
        no_match_before=("outcome", lambda s: int((s == "no_match_before").sum())), id_continuous=("id_continuous", "sum"),
        n_full=("has_full", "sum")).reset_index()
    oc["preserved_rate"] = (oc.preserved / oc.n_eligible).round(3)
    oc["id_continuous_rate"] = (oc.id_continuous / oc.n_eligible).round(3)
    oc.to_csv(out / "outcome_counts.csv", index=False)
    # --- ghep nham ---
    rq = pd.concat(refind_rows, ignore_index=True) if refind_rows else pd.DataFrame()
    if len(rq):
        rq.to_csv(out / "refind_events.csv", index=False)
        q = rq.groupby(["cond", "tracker"]).agg(n_refind=("wrong", "size"), n_known=("known", "sum"), n_wrong=("wrong", "sum")).reset_index()
        q["wrong_refind_rate"] = (q.n_wrong / q.n_known.clip(lower=1)).round(4)
        q.to_csv(out / "refind_quality.csv", index=False)
        print("\nGhep nham khi noi lai track (refind): ")
        print(q.to_string(index=False))
    if all_windows:
        pd.concat(all_windows, ignore_index=True).to_csv(out / "event_window_log.csv", index=False)
    # --- in bang cause ---
    print("\nNguyen nhan chinh cua that bai (switched + lost_after), theo tang:")
    piv = summ.pivot_table(index=["cond", "tracker", "stratum"], columns="primary_cause", values="n", fill_value=0)
    print(piv.to_string())
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
