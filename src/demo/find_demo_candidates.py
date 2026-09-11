"""
find_demo_candidates.py -- Quet va xep hang candidate cho 2 video demo khoa luan.

CHI DUNG DU LIEU CO SAN, KHONG chay lai tracking:
  - Ket cuc tung tracker tren tung su kien che khuat (occlusion_eval.py da cham):
        results/runA/event_outcomes_DETRAC-all.csv      (train, conf=0.25)
        results/test_eval/A/event_outcomes_DETRAC-test.csv (test,  conf=0.25)
  - Bang doan che co path_len_px / net_disp_px  -> do cong NGAY TRONG doan che
  - Bang doan che gan hoan toan (>= 0.90)         -> muc che khuat

DEMO 1 -- xe RE (quy dao cong) roi bi che, ket cuc KHAC NHAU giua CV va CTRV.
    diem = do cong trong doan che * (CTRV giu ID, CV mat) * do dai hop ly
    do cong = path_len / net_disp - 1   (0 = thang tuyet doi)
    Loc net_disp >= 40 px: xe gan nhu dung yen cho do cong rat cao nhung chi la
    nhieu annotation (vd MVI_40761 t7: net 5,9 px, curv 1,97).

DEMO 2 -- che khuat NANG (>= 0.90) keo dai 0,5-1,5 s (12-37 frame @25fps),
    uu tien ket cuc khac nhau giua cac tracker.

BUOC 2 - KIEM TRA DO ON DINH (bat buoc truoc khi chon). Ket cuc `switched`
trong event_outcomes chi so ID o frame truoc/sau su kien. Mot ID "doi" co the
chi la nhap nhay 1 frame (vd MVI_40854 t7: CV 135<->189 nhay qua lai, EKF/UKF
cung nhay nhu vay -> KHONG phai su khac biet that). Voi top candidate, doc lai
prediction da luu, ghep target tung frame trong cua so [start-25, end+25] va
dem so lan doi ID cua tung tracker. Candidate "sach" = so lan doi ID dung bang
0 (preserved) hoac 1 (switched) o CA BA tracker.

Ca hai deu doi hoi CO DU prediction cua 3 tracker tren cung video (dieu kien
`eligible` cua occlusion_eval da bao dam GT co ca truoc va sau su kien).

CACH DUNG
    python src/demo/find_demo_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FPS = 25
SOURCES = [
    # (nhan, file outcome, prefix tracker, bang partial, bang full, split, part)
    ("train", "results/runA/event_outcomes_DETRAC-all.csv", "runA",
     "occlusion_segments.csv", "full_occlusion_segments.csv", "DETRAC-all", "train"),
    ("test", "results/test_eval/A/event_outcomes_DETRAC-test.csv", "testA",
     "occlusion_segments_test.csv", "full_occlusion_segments_test.csv", "DETRAC-test", "test"),
]
MODELS = ["cv", "ekf_ctrv", "ukf_ctrv"]


def load_source(label, out_csv, prefix, part_csv, full_csv, split, part):
    ev = pd.read_csv(out_csv)
    ev = ev[ev.eligible].copy()
    key = ["video", "track_id", "event_id"]
    wide = ev[key + ["start_frame", "end_frame", "length_frames", "has_full"]].drop_duplicates(key)
    for m in MODELS:
        sub = ev[ev.tracker == f"{prefix}-{m}"][key + ["outcome", "id_before", "id_after",
                                                        "id_continuous", "recall_during"]]
        sub = sub.rename(columns={c: f"{c}__{m}" for c in
                                  ["outcome", "id_before", "id_after", "id_continuous", "recall_during"]})
        wide = wide.merge(sub, on=key, how="inner")
    wide["split"] = split
    wide["part"] = part
    wide["source"] = label

    # Do cong / muc che TRONG su kien: gop tu cac doan che thanh phan
    p = pd.read_csv(config.INTERIM_DIR / part_csv)
    f = pd.read_csv(config.INTERIM_DIR / full_csv)
    rows = []
    for r in wide.itertuples(index=False):
        sp = p[(p.video == r.video) & (p.track_id == r.track_id)
               & (p.start_frame <= r.end_frame) & (p.end_frame >= r.start_frame)]
        sf = f[(f.video == r.video) & (f.track_id == r.track_id)
               & (f.start_frame <= r.end_frame) & (f.end_frame >= r.start_frame)]
        path = float(sp.path_len_px.sum()) if len(sp) else np.nan
        net = float(sp.net_disp_px.sum()) if len(sp) else np.nan
        rows.append(dict(
            curv=(path / net - 1.0) if (net and net > 5.0) else np.nan,
            path_len=path, net_disp=net,
            max_occ=float(sp.max_occlusion_ratio.max()) if len(sp) else np.nan,
            mean_occ=float(sp.mean_occlusion_ratio.mean()) if len(sp) else np.nan,
            full_len=int(sf.length_frames.sum()) if len(sf) else 0,
            full_start=int(sf.start_frame.min()) if len(sf) else -1,
            full_end=int(sf.end_frame.max()) if len(sf) else -1,
            occluder=str(sp.main_source.mode().iloc[0]) if len(sp) and "main_source" in sp else "",
        ))
    return pd.concat([wide.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def outcome_code(df):
    """Ma ngan ket cuc: P=preserved S=switched L=lost_after N=no_match_before."""
    m = {"preserved": "P", "switched": "S", "lost_after": "L", "no_match_before": "N"}
    return df["outcome__cv"].map(m) + df["outcome__ekf_ctrv"].map(m) + df["outcome__ukf_ctrv"].map(m)


PAD = 25   # cua so kiem tra on dinh: [start - PAD, end + PAD]
_GT_CACHE: dict = {}


def _gt(split, video):
    key = (split, video)
    if key not in _GT_CACHE:
        g = pd.read_csv(config.PROCESSED_DIR / split / video / "gt" / "gt.txt", header=None,
                        names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
        g["cx"] = g.x + g.w / 2
        g["cy"] = g.y + g.h / 2
        _GT_CACHE[key] = g
    return _GT_CACHE[key]


def heading_change(split, video, gid, f0, f1, win=10, min_disp=5.0):
    """Goc doi huong (do) cua GT giua `win` frame dau va `win` frame cuoi su kien.
    Huong tinh tu dich chuyen tam box trong moi cua so; NaN neu dich chuyen
    trong cua so < min_disp px (xe dung yen -> huong khong xac dinh)."""
    t = _gt(split, video)
    t = t[t.id == gid].sort_values("frame")
    a = t[(t.frame >= f0) & (t.frame < f0 + win)]
    b = t[(t.frame > f1 - win) & (t.frame <= f1)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    da = a[["cx", "cy"]].to_numpy()[-1] - a[["cx", "cy"]].to_numpy()[0]
    db = b[["cx", "cy"]].to_numpy()[-1] - b[["cx", "cy"]].to_numpy()[0]
    if np.hypot(*da) < min_disp or np.hypot(*db) < min_disp:
        return np.nan
    h0, h1 = np.degrees(np.arctan2(da[1], da[0])), np.degrees(np.arctan2(db[1], db[0]))
    return float((h1 - h0 + 180) % 360 - 180)


def id_changes_in_window(video, split, prefix, m, gid, f0, f1):
    """Doc prediction DA LUU, ghep target tung frame (Hungarian tren toan GT cua
    frame, IoU >= 0.5) va tra ve (so lan doi ID, chuoi ID rut gon)."""
    from occlusion_eval import iou_matrix, match_frame
    gt = _gt(split, video)
    pr = pd.read_csv(config.PROCESSED_DIR / "trackers" / split / f"{prefix}-{m}" / "data" / f"{video}.txt",
                     header=None, names=["frame", "id", "x", "y", "w", "h", "conf"])
    gt = gt[(gt.frame >= f0) & (gt.frame <= f1)]
    pr = pr[(pr.frame >= f0) & (pr.frame <= f1)]
    seq = []
    for fr, g in gt.groupby("frame"):
        p = pr[pr.frame == fr]
        hit = None
        if len(p):
            iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
            gids, pids = g.id.to_numpy(), p.id.to_numpy()
            for r, c in match_frame(iou, 0.5):
                if int(gids[r]) == gid:
                    hit = int(pids[c])
        seq.append(hit)
    ids = [v for v in seq if v is not None]
    runs = [ids[0]] if ids else []
    for v in ids[1:]:
        if v != runs[-1]:
            runs.append(v)
    return len(runs) - 1 if runs else 0, "->".join(str(v) for v in runs)


def stability_check(top: pd.DataFrame, label: str) -> pd.DataFrame:
    """Buoc 2: dem so lan doi ID that trong cua so quanh su kien cho tung tracker."""
    rows = []
    for r in top.itertuples(index=False):
        prefix = "runA" if r.source == "train" else "testA"
        rec = {}
        clean = True
        for m in MODELS:
            n, chain = id_changes_in_window(r.video, r.split, prefix, m, int(r.track_id),
                                            int(r.start_frame) - PAD, int(r.end_frame) + PAD)
            expect = 0 if getattr(r, f"outcome__{m}") == "preserved" else 1
            rec[f"nchg_{m}"] = n
            rec[f"chain_{m}"] = chain
            clean &= (n == expect)
        rec["clean"] = clean
        rows.append(rec)
    out = pd.concat([top.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    print(f"\n--- {label}: kiem tra on dinh (nchg = so lan doi ID that trong cua so +-{PAD} frame; "
          f"clean = dung 0 lan neu 'preserved', dung 1 lan neu 'switched', o CA 3 tracker) ---")
    show = ["source", "video", "track_id", "event_id", "code", "nchg_cv", "nchg_ekf_ctrv", "nchg_ukf_ctrv",
            "clean", "chain_cv", "chain_ekf_ctrv", "chain_ukf_ctrv"]
    print(out[show].to_string(index=False))
    return out


def main() -> int:
    parts = [load_source(*s) for s in SOURCES]
    d = pd.concat(parts, ignore_index=True)
    d["code"] = outcome_code(d)
    d["n_preserved"] = sum((d[f"outcome__{m}"] == "preserved").astype(int) for m in MODELS)
    d["differ"] = d.code.apply(lambda c: len(set(c)) > 1)
    print(f"Tong su kien du dieu kien co du 3 tracker: {len(d)} "
          f"(train {int((d.source=='train').sum())}, test {int((d.source=='test').sum())})")
    print(f"Trong do 3 tracker cho ket cuc KHAC nhau: {int(d.differ.sum())}\n")

    out_dir = Path("outputs/thesis_demos")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DEMO 1: re + che khuat, CTRV giu ID / CV khong
    # ------------------------------------------------------------------
    d1 = d[d.curv.notna() & d.differ & (d.net_disp >= 40)].copy()
    # CTRV (EKF hoac UKF) giu duoc, CV KHONG giu duoc
    d1["ctrv_wins"] = ((d1["outcome__cv"] != "preserved") &
                       ((d1["outcome__ekf_ctrv"] == "preserved") |
                        (d1["outcome__ukf_ctrv"] == "preserved")))
    d1["ekf_wins"] = (d1["outcome__cv"] != "preserved") & (d1["outcome__ekf_ctrv"] == "preserved")
    # Goc re THAT: doi huong giua 10 frame dau va 10 frame cuoi su kien (do tren GT).
    # `curv` mot minh khong du: xe gan nhu dung yen (< 0,5 px/frame) cho curv rat
    # cao chi vi nhieu annotation (vd MVI_39311 t41: curv 0,62 nhung xe nhich
    # ~76 px trong 240 frame, huong tung doan nhay lung tung).
    d1["turn_deg"] = [heading_change(r.split, r.video, int(r.track_id), int(r.start_frame), int(r.end_frame))
                      for r in d1.itertuples(index=False)]
    d1["speed_px"] = d1.net_disp / d1.length_frames          # toc do trung binh trong su kien
    # Do dai 12-200 frame (0,5-8 s). Su kien GT co the dai hon doan mat dau that.
    d1["len_ok"] = d1.length_frames.between(12, 200)
    d1["score1"] = (d1.turn_deg.abs().fillna(0).clip(upper=45) / 10        # toi da 4,5 diem cho goc re
                    + d1.curv.fillna(0).clip(upper=0.5) * 4
                    + d1.ctrv_wins * 3 + d1.ekf_wins * 2
                    + d1.len_ok * 1
                    + d1.net_disp.clip(upper=200) / 200)      # xe phai di chuyen that
    d1 = d1.sort_values("score1", ascending=False)
    cols = ["source", "video", "track_id", "event_id", "start_frame", "end_frame",
            "length_frames", "turn_deg", "curv", "net_disp", "speed_px", "max_occ", "code", "ctrv_wins", "score1"]
    print("=" * 100)
    print("DEMO 1 -- XE RE + CHE KHUAT (turn_deg = doi huong GT dau/cuoi su kien; curv = path/net - 1; "
          "net_disp >= 40 px; code = ket cuc CV/EKF/UKF, P=giu S=doi ID L=mat N=chua ghep)")
    print("=" * 100)
    top1 = d1.head(15)
    print(top1[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    top1 = stability_check(top1, "DEMO 1")
    top1.to_csv(out_dir / "candidates_demo1.csv", index=False)
    fin = top1[top1.clean & top1.ctrv_wins & (top1.turn_deg.abs() >= 10) & (top1.speed_px >= 0.3)]
    print("\nDEMO 1 -- candidate SACH, CTRV giu ID, |goc re| >= 10 deg, toc do >= 0,3 px/frame:")
    print(fin[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(fin) else "  (khong co)")

    # ------------------------------------------------------------------
    # DEMO 2: che khuat NANG >= 0.90, dai 0,5-1,5 s
    # ------------------------------------------------------------------
    lo, hi = int(0.5 * FPS), int(1.5 * FPS)
    d2 = d[(d.full_len >= lo) & (d.full_len <= hi) & (d.max_occ >= 0.90)].copy()
    d2["ctrv_wins"] = ((d2["outcome__cv"] != "preserved") &
                       ((d2["outcome__ekf_ctrv"] == "preserved") |
                        (d2["outcome__ukf_ctrv"] == "preserved")))
    d2["veh_occ"] = d2.occluder.eq("vehicle")
    d2["score2"] = (d2.differ * 3 + d2.ctrv_wins * 3 + d2.veh_occ * 1
                    + d2.max_occ * 2 + d2.net_disp.clip(upper=150) / 150)
    d2 = d2.sort_values("score2", ascending=False)
    cols2 = ["source", "video", "track_id", "event_id", "start_frame", "end_frame",
             "length_frames", "full_len", "full_start", "full_end", "max_occ",
             "occluder", "net_disp", "code", "ctrv_wins", "score2"]
    print("\n" + "=" * 100)
    print(f"DEMO 2 -- CHE KHUAT NANG (>=0.90, doan full {lo}-{hi} frame = 0,5-1,5 s)")
    print("=" * 100)
    top2 = d2.head(12)
    print(top2[cols2].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    top2 = stability_check(top2, "DEMO 2")
    top2.to_csv(out_dir / "candidates_demo2.csv", index=False)

    d.to_csv(out_dir / "all_eligible_events_3trackers.csv", index=False)
    print(f"\nDa luu bang candidate -> {out_dir}/candidates_demo{{1,2}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
