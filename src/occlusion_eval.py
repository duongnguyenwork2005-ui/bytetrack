"""
occlusion_eval.py -- Danh gia giu ID qua doan che khuat, ban DA SUA.

Module nay THAY THE phan danh gia cua `stratified_analysis.py` nhung duoc viet
RIENG de ket qua cu van tai lap duoc. Ba nhom loi da sua, moi loi kem vi du tai
hien trong `src/test_occlusion_eval.py`.


===========================================================================
LOI A -- GHEP IoU: chay Hungarian roi moi loc nguong
===========================================================================
Ban cu chay `linear_sum_assignment` tren TOAN BO ma tran IoU roi moi loai cac
cap duoi nguong. Hungarian toi uu TONG IoU nen no co the chon cac cap duoi
nguong chi vi tong lon hon, lam mat cap hop le.

    IoU = [[0.1281, 0.3739],        nguong 0.5
           [0.3963, 0.6384]]

    Hungarian chon duong cheo phu (0,1) + (1,0) = 0.7702
    vi lon hon duong cheo chinh    (0,0) + (1,1) = 0.7665.
    Ca hai deu duoi nguong -> loai het. Cap (1,1) = 0.6384 HOP LE bi mat.

MUC TIEU TOI UU SAU KHI SUA (phat bieu ro rang):
    Uu tien 1: toi da hoa SO cap hop le (IoU >= nguong).
    Uu tien 2: trong cac loi giai cung so cap hop le, toi da hoa TONG IoU.
Cach cai: dat chi phi cua cap duoi nguong bang mot hang so phat BIG_M rat lon
(1e6) va chi phi cua cap hop le bang -IoU (trong [-1, 0]). Vi BIG_M lon hon
moi tong IoU co the co, ham muc tieu tro thanh tu dien: giam so cap khong hop
le truoc, roi moi toi tong IoU. Cac cap khong hop le con sot lai (do Hungarian
buoc phai ghep du min(n, m) cap) bi loai sau khi giai -- va viec loai chung
KHONG the day mat mot cap hop le nao, vi doi mot cap hop le lay mot cap khong
hop le luon lam tang chi phi them it nhat BIG_M - 1.

So sanh nguong: dung `>=` (dung nguong), khong phai `>`.


===========================================================================
LOI B -- DINH NGHIA SU KIEN: dem lap va coi truong hop khong the danh gia
         la tracker that bai
===========================================================================
B1. DEM LAP. Doan che >= 0.90 ("full") luon nam LONG trong doan che >= 0.10
    ("partial") cua cung mot track, vi mot bbox bi che 90% thi cung bi che 10%.
    Gop hai bang lai roi coi moi dong la mot mau doc lap la dem cung mot lan bi
    che HAI LAN. Sau khi sua: DON VI PHAN TICH la mot SU KIEN CHE KHUAT =
    hop cua cac doan chong lan/ke nhau cua cung (video, track_id). Muc do
    duoc ghi lai lam thuoc tinh (`has_full`) chu khong tao them mau.

B2. KHONG DANH GIA DUOC != TRACKER THAT BAI. Ban cu dat status = "lost" khi
    khong tim thay ket noi sau doan che, KHONG phan biet:
      - GT khong con sau su kien (het video / xe roi khoi canh)  -> khong the
        danh gia, phai LOAI khoi mau
      - GT van con nhung tracker khong ghep duoc                 -> that bai THAT
    Va dat "no_before" roi lang le bo qua khi tracker chua tung ghep duoc trude
    do -- viec bo qua nay THIEN VI tracker hong som.

B3. DIEU KIEN DU XAC DINH TU GT, khong tu tracker. Nho vay moi tracker duoc
    cham tren DUNG cung mot tap su kien, khong ai duoc loi vi tu hong.

B4. FILE TRACKING RONG khong duoc bo qua am tham: neu GT du dieu kien thi su
    kien do van duoc danh gia, va tracker nhan ket qua `no_match_before`.


===========================================================================
DINH NGHIA "GIU ID" -- hai khai niem KHAC NHAU, do ca hai
===========================================================================
    id_match      : ID ngay TRUOC su kien == ID ngay SAU su kien.
                    Khong doi hoi lien tuc trong luc bi che.
    id_continuous : ID do duoc duy tri o MOI frame trong su kien ma GT co mat
                    (va bang ID truoc su kien). Chat hon han.
Ban cu chi do `id_match` nhung goi la "giu duoc ID", de gay hieu nham. O day
bao cao ca hai va khong dung lan nhau.


CACH DUNG
    python src/occlusion_eval.py --split-name DETRAC-all \\
        --trackers yolov8n-bytetrack yolov8n-ekf-ctrv yolov8n-ukf-ctrv \\
        --out-dir results/eval_fixed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

IOU_THRESHOLD = 0.5
CONTEXT_WINDOW = 30          # so frame truoc/sau su kien dung lam moc
BIG_M = 1e6                  # phat cho cap duoi nguong; xem docstring LOI A

TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]

#: Ket qua co the danh gia duoc (su kien du dieu kien theo GT)
OUTCOMES = ["preserved", "switched", "lost_after", "no_match_before"]
#: Ly do loai khoi mau (khong the danh gia)
EXCLUSIONS = ["gt_missing_before", "gt_missing_after"]


# ---------------------------------------------------------------------------
# 1. GHEP IoU
# ---------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giua hai tap bbox dang (x, y, w, h). Tra ve ma tran (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1 = a[:, 0][:, None], a[:, 1][:, None]
    ax2, ay2 = (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx1, by1 = b[:, 0][None, :], b[:, 1][None, :]
    bx2, by2 = (b[:, 0] + b[:, 2])[None, :], (b[:, 1] + b[:, 3])[None, :]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_frame(iou: np.ndarray, threshold: float = IOU_THRESHOLD
                ) -> list[tuple[int, int]]:
    """Ghep 1-1 toi uu, CHI cho phep cap co IoU >= threshold.

    Muc tieu toi uu (tu dien): toi da hoa so cap hop le truoc, roi toi da hoa
    tong IoU trong so cac loi giai cung so cap hop le. Doi tuong khong ghep
    duoc thi de khong ghep -- khong ep ghep.

    Xem docstring dau file (LOI A) de biet vi sao khong the loc nguong SAU khi
    chay Hungarian tren toan bo ma tran.

    Returns:
        Danh sach (chi_so_hang, chi_so_cot) cua cac cap hop le.
    """
    if iou.size == 0:
        return []
    # Cap hop le: chi phi -IoU trong [-1, 0]. Cap khong hop le: chi phi BIG_M.
    cost = np.where(iou >= threshold, -iou, BIG_M)
    rows, cols = linear_sum_assignment(cost)
    # Loai cac cap khong hop le ma Hungarian buoc phai ghep (vi phai ghep du
    # min(n, m) cap). Viec loai nay khong the lam mat cap hop le nao: doi mot
    # cap hop le lay mot cap khong hop le lam chi phi tang it nhat BIG_M - 1.
    return [(int(r), int(c)) for r, c in zip(rows, cols)
            if iou[r, c] >= threshold]


def match_video(gt: pd.DataFrame, trk: pd.DataFrame) -> pd.DataFrame:
    """Ghep GT voi tracker theo tung frame cua MOT video.

    Tra ve DataFrame (frame, gt_id, tracker_id, iou). Neu `trk` rong thi tra ve
    bang rong -- nhung nguoi goi KHONG duoc coi do la "bo qua video": xem
    `evaluate_events`, su kien van duoc cham va nhan `no_match_before`.
    """
    rows = []
    trk_by_frame = {f: t for f, t in trk.groupby("frame", sort=False)} if len(trk) else {}
    for frame, g in gt.groupby("frame", sort=False):
        t = trk_by_frame.get(frame)
        if t is None or len(t) == 0:
            continue
        gb = g[["x", "y", "w", "h"]].to_numpy(dtype=float)
        tb = t[["x", "y", "w", "h"]].to_numpy(dtype=float)
        iou = iou_matrix(gb, tb)
        gid = g["id"].to_numpy()
        tid = t["id"].to_numpy()
        for r, c in match_frame(iou):
            rows.append((int(frame), int(gid[r]), int(tid[c]), float(iou[r, c])))
    return pd.DataFrame(rows, columns=["frame", "gt_id", "tracker_id", "iou"])


# ---------------------------------------------------------------------------
# 2. XAY DUNG TAP SU KIEN KHONG DEM LAP
# ---------------------------------------------------------------------------
def build_events(partial: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Gop cac doan che chong lan/ke nhau thanh SU KIEN che khuat khong dem lap.

    Mot su kien = hop cua cac doan cua cung (video, track_id) ma khoang frame
    cua chung chong lan hoac ke nhau. Doan "full" (>= 0.90) luon nam long trong
    doan "partial" (>= 0.10) cua cung lan bi che, nen neu khong gop se dem cung
    mot lan bi che hai lan.

    Cot `has_full` ghi lai su kien do co giai doan bi che gan hoan toan hay
    khong -- dung de phan tang, KHONG tao them mau.
    """
    frames = []
    for df, lvl in ((partial, "partial"), (full, "full")):
        if df is None or not len(df):
            continue
        d = df[["video", "track_id", "start_frame", "end_frame"]].copy()
        d["is_full"] = (lvl == "full")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["video", "track_id", "event_id", "start_frame",
                                     "end_frame", "length_frames", "has_full"])
    allseg = pd.concat(frames, ignore_index=True)

    out = []
    for (video, tid), g in allseg.groupby(["video", "track_id"], sort=False):
        g = g.sort_values("start_frame")
        cur_s = cur_e = None
        cur_full = False
        for r in g.itertuples(index=False):
            s, e = int(r.start_frame), int(r.end_frame)
            if cur_s is None:
                cur_s, cur_e, cur_full = s, e, bool(r.is_full)
            elif s <= cur_e + 1:                 # chong lan hoac ke nhau -> cung su kien
                cur_e = max(cur_e, e)
                cur_full = cur_full or bool(r.is_full)
            else:
                out.append((video, int(tid), cur_s, cur_e, cur_full))
                cur_s, cur_e, cur_full = s, e, bool(r.is_full)
        if cur_s is not None:
            out.append((video, int(tid), cur_s, cur_e, cur_full))

    ev = pd.DataFrame(out, columns=["video", "track_id", "start_frame",
                                    "end_frame", "has_full"])
    ev["length_frames"] = ev.end_frame - ev.start_frame + 1
    ev = ev.sort_values(["video", "track_id", "start_frame"]).reset_index(drop=True)
    ev["event_id"] = ev.groupby(["video", "track_id"]).cumcount()
    return ev[["video", "track_id", "event_id", "start_frame", "end_frame",
               "length_frames", "has_full"]]


def add_gt_eligibility(events: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Xac dinh su kien nao DU DIEU KIEN danh gia -- CHI dua vao GT.

    Du dieu kien khi GT cua chinh track do co mat o CA HAI phia:
      - trong cua so [start - CONTEXT_WINDOW, start - 1]
      - trong cua so [end + 1, end + CONTEXT_WINDOW]

    Vi tieu chi nay khong dung den ket qua tracker nao, moi tracker deu duoc
    cham tren dung cung mot tap su kien.
    """
    gt_frames: dict[tuple[str, int], np.ndarray] = {
        k: np.sort(v["frame"].to_numpy())
        for k, v in gt.groupby(["video", "id"], sort=False)
    }
    before_ok, after_ok, reasons = [], [], []
    for r in events.itertuples(index=False):
        fr = gt_frames.get((r.video, int(r.track_id)))
        if fr is None:
            b = a = False
        else:
            b = bool(((fr >= r.start_frame - CONTEXT_WINDOW) & (fr < r.start_frame)).any())
            a = bool(((fr > r.end_frame) & (fr <= r.end_frame + CONTEXT_WINDOW)).any())
        before_ok.append(b)
        after_ok.append(a)
        reasons.append("" if (b and a) else
                       ("gt_missing_before" if not b else "gt_missing_after"))
    ev = events.copy()
    ev["gt_before"] = before_ok
    ev["gt_after"] = after_ok
    ev["eligible"] = ev.gt_before & ev.gt_after
    ev["exclusion_reason"] = reasons
    return ev


# ---------------------------------------------------------------------------
# 3. CHAM TUNG SU KIEN
# ---------------------------------------------------------------------------
def evaluate_events(events: pd.DataFrame, match: pd.DataFrame,
                    gt: pd.DataFrame) -> pd.DataFrame:
    """Cham ket qua cua MOT tracker tren tap su kien da xac dinh.

    Chi cham cac su kien `eligible` (xac dinh tu GT). Ket qua:
        preserved       ID truoc == ID sau
        switched        ID truoc != ID sau
        lost_after      co ghep truoc, khong ghep duoc sau (GT van con -> that bai that)
        no_match_before tracker chua tung ghep duoc truoc su kien
    Ngoai ra do them `id_continuous`: ID duoc duy tri o MOI frame co GT trong
    su kien. Day la dinh nghia CHAT HON `preserved`, khong duoc dung lan.
    """
    by_track: dict[tuple[str, int], pd.DataFrame] = {}
    if len(match):
        for k, v in match.sort_values("frame").groupby(["video", "gt_id"], sort=False):
            by_track[k] = v

    gt_frames: dict[tuple[str, int], np.ndarray] = {
        k: np.sort(v["frame"].to_numpy())
        for k, v in gt.groupby(["video", "id"], sort=False)
    }

    rows = []
    for r in events.itertuples(index=False):
        rec = dict(video=r.video, track_id=int(r.track_id), event_id=int(r.event_id),
                   start_frame=int(r.start_frame), end_frame=int(r.end_frame),
                   length_frames=int(r.length_frames), has_full=bool(r.has_full),
                   eligible=bool(r.eligible), exclusion_reason=r.exclusion_reason,
                   outcome=None, id_before=None, id_after=None,
                   id_continuous=False, recall_during=0.0)
        if not r.eligible:
            rows.append(rec)
            continue

        m = by_track.get((r.video, int(r.track_id)))
        if m is None or not len(m):
            rec["outcome"] = "no_match_before"
            rows.append(rec)
            continue

        fr = m["frame"].to_numpy()
        ti = m["tracker_id"].to_numpy()
        before = (fr < r.start_frame) & (fr >= r.start_frame - CONTEXT_WINDOW)
        after = (fr > r.end_frame) & (fr <= r.end_frame + CONTEXT_WINDOW)
        during = (fr >= r.start_frame) & (fr <= r.end_frame)

        gfr = gt_frames.get((r.video, int(r.track_id)), np.empty(0))
        n_gt_during = int(((gfr >= r.start_frame) & (gfr <= r.end_frame)).sum())
        rec["recall_during"] = round(float(during.sum()) / max(1, n_gt_during), 4)

        if not before.any():
            rec["outcome"] = "no_match_before"
            rows.append(rec)
            continue

        idb = int(ti[before][-1])
        rec["id_before"] = idb
        if not after.any():
            # GT van con (da kiem tra o `eligible`) -> day la that bai THAT
            rec["outcome"] = "lost_after"
            rows.append(rec)
            continue

        ida = int(ti[after][0])
        rec["id_after"] = ida
        rec["outcome"] = "preserved" if idb == ida else "switched"
        # id_continuous: moi frame co GT trong su kien deu duoc ghep dung ID do
        rec["id_continuous"] = bool(
            idb == ida and n_gt_during > 0
            and int(during.sum()) == n_gt_during
            and bool(np.all(ti[during] == idb))
        )
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. THONG KE
# ---------------------------------------------------------------------------
def bootstrap_diff_by_video(wide: pd.DataFrame, col_a: str, col_b: str,
                            n_boot: int = 5000, seed: int = 0) -> dict:
    """Khoang tin cay cho chenh lech ti le giu ID, BOOTSTRAP THEO VIDEO.

    Lay mau lai o cap VIDEO (khong phai cap su kien) vi cac su kien trong cung
    mot video khong doc lap: chung dung chung canh, dung chung detector tren
    cung dieu kien anh sang/goc quay. Giu nguyen tinh GHEP CAP: moi lan lay mau
    deu tinh ca hai tracker tren DUNG cung tap su kien duoc chon.
    """
    videos = wide["video"].unique()
    rng = np.random.default_rng(seed)
    by_video = {v: g for v, g in wide.groupby("video", sort=False)}
    obs = float(wide[col_a].mean() - wide[col_b].mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(videos, size=len(videos), replace=True)
        a_sum = b_sum = n = 0.0
        for v in pick:
            g = by_video[v]
            a_sum += float(g[col_a].sum())
            b_sum += float(g[col_b].sum())
            n += len(g)
        diffs[i] = (a_sum - b_sum) / max(n, 1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(diff=obs, ci_low=float(lo), ci_high=float(hi),
                n_videos=len(videos), n_events=len(wide))


def main() -> int:
    ap = argparse.ArgumentParser(description="Danh gia giu ID qua doan che (ban da sua)")
    ap.add_argument("--split-name", default="DETRAC-all")
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/eval_fixed")
    ap.add_argument("--part", choices=["train", "test"], default="train")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Nap GT ---
    split_dir = config.PROCESSED_DIR / a.split_name
    videos = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    gts = []
    for v in videos:
        p = split_dir / v / "gt" / "gt.txt"
        if not p.exists():
            continue
        d = pd.read_csv(p, header=None, names=GT_COLS)
        d["video"] = v
        gts.append(d)
    gt = pd.concat(gts, ignore_index=True)
    print(f"[eval] {len(videos)} video, {len(gt):,} dong GT")

    # --- Xay tap su kien khong dem lap ---
    partial = pd.read_csv(config.part_file("occlusion_segments.csv", a.part))
    full = pd.read_csv(config.part_file("full_occlusion_segments.csv", a.part))
    partial = partial[partial.video.isin(videos)]
    full = full[full.video.isin(videos)]
    events = build_events(partial, full)
    events = add_gt_eligibility(events, gt)
    n_raw = len(partial) + len(full)
    print(f"[eval] doan tho: {len(partial)} partial + {len(full)} full = {n_raw}")
    print(f"[eval] sau khi gop thanh su kien khong dem lap: {len(events)} "
          f"(giam {n_raw - len(events)} do dem lap)")
    print(f"[eval] du dieu kien danh gia (xac dinh tu GT): {int(events.eligible.sum())}")
    for reason, n in events.loc[~events.eligible, "exclusion_reason"].value_counts().items():
        print(f"         loai {n:>5}  ly do: {reason}")
    events.to_csv(out / f"events_{a.split_name}.csv", index=False)

    # --- Cham tung tracker tren CUNG tap su kien ---
    per_tracker = {}
    for tr in a.trackers:
        tdir = config.PROCESSED_DIR / "trackers" / a.split_name / tr / "data"
        if not tdir.is_dir():
            print(f"  [BO QUA] khong co thu muc {tdir}")
            continue
        mats = []
        n_empty = 0
        for v in videos:
            f = tdir / f"{v}.txt"
            if f.exists() and f.stat().st_size > 0:
                t = pd.read_csv(f, header=None, names=TRK_COLS)
            else:
                # File rong: KHONG bo qua am tham. Tao bang rong de cac su kien
                # cua video nay van duoc cham (se ra `no_match_before`).
                t = pd.DataFrame(columns=TRK_COLS)
                n_empty += 1
            g = gt[gt.video == v]
            m = match_video(g, t)
            m["video"] = v
            mats.append(m)
        match = pd.concat(mats, ignore_index=True)
        res = evaluate_events(events, match, gt)
        res["tracker"] = tr
        per_tracker[tr] = res
        ev = res[res.eligible]
        print(f"\n[{tr}] {n_empty} file tracking rong (van duoc cham)")
        print(f"  {'ket qua':<18}{'so su kien':>12}{'ti le':>9}")
        print("  " + "-" * 39)
        for o in OUTCOMES:
            n = int((ev.outcome == o).sum())
            print(f"  {o:<18}{n:>12}{n / max(len(ev), 1) * 100:>8.1f}%")
        print(f"  {'id_continuous':<18}{int(ev.id_continuous.sum()):>12}"
              f"{ev.id_continuous.mean() * 100:>8.1f}%")

    if not per_tracker:
        print("[eval] khong co tracker nao de cham")
        return 1

    allres = pd.concat(per_tracker.values(), ignore_index=True)
    allres.to_csv(out / f"event_outcomes_{a.split_name}.csv", index=False)

    # --- So sanh ghep cap tren CUNG tap su kien ---
    key = ["video", "track_id", "event_id"]
    elig = events[events.eligible][key]
    wide = elig.copy()
    for tr, res in per_tracker.items():
        r = res[res.eligible][key + ["outcome", "id_continuous"]]
        r = r.rename(columns={"outcome": f"out__{tr}",
                              "id_continuous": f"cont__{tr}"})
        wide = wide.merge(r, on=key, how="left")
    for tr in per_tracker:
        wide[f"pres__{tr}"] = (wide[f"out__{tr}"] == "preserved").astype(float)
        wide[f"cont__{tr}"] = wide[f"cont__{tr}"].astype(float)
    wide["video"] = wide["video"].astype(str)
    wide.to_csv(out / f"paired_{a.split_name}.csv", index=False)

    print("\n" + "=" * 78)
    print(f"SO SANH TREN CUNG {len(wide)} SU KIEN DU DIEU KIEN "
          f"({wide.video.nunique()} video)")
    print("=" * 78)
    print(f"  {'tracker':<26}{'id_match':>12}{'id_continuous':>16}")
    print("  " + "-" * 54)
    for tr in per_tracker:
        print(f"  {tr:<26}{wide[f'pres__{tr}'].mean() * 100:>11.2f}%"
              f"{wide[f'cont__{tr}'].mean() * 100:>15.2f}%")

    base = a.trackers[0]
    print(f"\nCHENH LECH so voi {base}, bootstrap THEO VIDEO "
          f"({a.n_boot} lan, giu tinh ghep cap):")
    print(f"  {'tracker':<26}{'chenh id_match':>16}{'KTC 95%':>26}")
    print("  " + "-" * 68)
    boot_rows = []
    for tr in per_tracker:
        if tr == base:
            continue
        r = bootstrap_diff_by_video(wide, f"pres__{tr}", f"pres__{base}",
                                    n_boot=a.n_boot)
        boot_rows.append(dict(tracker=tr, baseline=base, **r))
        print(f"  {tr:<26}{r['diff'] * 100:>+15.2f}%"
              f"   [{r['ci_low'] * 100:+.2f}%, {r['ci_high'] * 100:+.2f}%]")
    pd.DataFrame(boot_rows).to_csv(out / f"bootstrap_{a.split_name}.csv", index=False)

    print("\nLUU Y KHI DOC:")
    print("  - Khoang tin cay chua 0 KHONG co nghia hai mo hinh tuong duong; no chi")
    print("    co nghia du lieu hien co chua du de phan biet (thieu luc kiem dinh).")
    print("  - Bootstrap lay mau o cap VIDEO vi cac su kien trong cung video khong")
    print("    doc lap. Neu dung McNemar theo tung su kien thi p-value se HEP hon")
    print("    thuc te vi coi cac su kien la doc lap.")
    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
