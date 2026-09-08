"""
long_compare.py -- Video minh hoa DAI: theo mot chiec xe suot ca doi track, hai
panel canh nhau (KF+CV trai | EKF+CTRV phai).

KHAC GI VOI CAC SCRIPT DEMO KHAC
    render_videos.py  : moi video chi ve MOT doan che (~3 giay), hai mo hinh ve
                        chong len cung mot khung -> box de bi de nhau, kho doc.
    side_by_side.py   : hai panel, nhung ve KET QUA TRACKER THAT -> khi tracker
                        mat track thi khong con gi de so sanh.
    long_compare.py   : hai panel + chay suot doi track + qua NHIEU lan che, nen
                        thay duoc ca luc bam sat lan luc truot, lap lai nhieu lan.

CO CHE MO PHONG (giong render_videos.py, mo rong ra ca doi track)
    Voi moi frame trong doi song cua track:
        m, P <- predict(m, P)                      # luon luon
        neu frame do KHONG bi che:  m, P <- update(m, P, z_GT)
    Tuc la trong doan bi che, bo loc CHI ngoai suy bang mo hinh chuyen dong -
    dung tinh huong mat detection hoan toan. Ra khoi doan che thi duoc nap lai
    quan sat GT, nen moi lan che la mot lan so sanh doc lap.

    Sai so hien tren man hinh:
        err(k) = || (cx_pred, cy_pred) - (cx_GT, cy_GT) ||_2      [pixel]
    Tai frame CUOI cua moi doan che, err chinh la FDE (Final Displacement Error)
    cua doan do - dung dinh nghia FDE dung trong render_videos.py.

MOT GIA DINH PHAI NOI RO KHI TRINH BAY
    Doan "bi che" lay tu annotation UA-DETRAC (occlusion_ratio >= 0.10 hoac
    >= 0.90). Bi che mot phan KHONG dong nghia detector mat hoan toan chiec xe.
    Vi o day ta cat han quan sat trong suot doan che, con so sai so la CAN TREN
    BI QUAN cho doan che mot phan, va bo loc khong bao gio bi "lac" nhu tracker
    that (vi luon duoc nap lai GT sau moi doan che). Day la minh hoa CO CHE cua
    hai mo hinh chuyen dong, khong phai do luong hieu nang tracker.

CACH DUNG
    python src/demo/long_compare.py                          # track mac dinh
    python src/demo/long_compare.py --video MVI_40992 --track 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402

FPS = 10
PANEL_GAP = 6
TRAIL = 25

COL_GT = (255, 255, 255)      # trang
COL_PRED = (0, 220, 0)        # xanh la - khi dang duoc nap quan sat
COL_DRIFT = (0, 0, 235)       # do     - khi dang ngoai suy mu (bi che)
COL_OCC = (0, 215, 255)       # vang   - vien danh dau doan che


def xyah(r) -> np.ndarray:
    """Doi bbox GT sang do do (cx, cy, ty le khung, chieu cao) ma ca 2 bo loc dung."""
    return np.array([r.cx, r.cy, r.bb_width / max(r.bb_height, 1e-6), r.bb_height], float)


def run_track(t: pd.DataFrame, occ_frames: set[int], model: str) -> np.ndarray:
    """Chay bo loc suot doi track, CHI predict o nhung frame nam trong occ_frames.

    Tra ve mang (N, 4) = (cx, cy, w, h) du doan cho tung frame cua track.

    Cong thuc mot buoc:
        predict :  x_{k|k-1} = f(x_{k-1|k-1})          (CV tuyen tinh / CTRV phi tuyen)
                   P_{k|k-1} = F P F^T + Q             (F = Jacobian voi EKF)
        update  :  K = P H^T (H P H^T + R)^{-1}
                   x_{k|k} = x_{k|k-1} + K (z - H x_{k|k-1})
    Buoc update bi BO QUA khi frame nam trong doan che.
    """
    if model == "KF":
        f = KalmanFilterXYAH()
        get = lambda m: (m[0], m[1], m[2], m[3])
        is_ctrv = False
    else:
        f = EKFTrackerCTRV()
        get = lambda m, f=f: (m[f.CX], m[f.CY], m[f.A], m[f.H])
        is_ctrv = True

    rows = list(t.itertuples())
    m, c = f.initiate(xyah(rows[0]))
    seeded = False
    traj = [(*get(m)[:2], get(m)[2] * get(m)[3], get(m)[3])]

    for r in rows[1:]:
        visible = int(r.frame) not in occ_frames
        # CTRV can 2 quan sat dau tien de suy ra van toc v va huong theta;
        # chi lam duoc khi con nhin thay xe.
        if is_ctrv and not seeded and visible:
            m, c = f.initiate_from_motion(m, c, xyah(r), n_frames=1.0)
            seeded = True
        m, c = f.predict(m, c)
        if visible:
            m, c = f.update(m, c, xyah(r))
        cx, cy, a, h = get(m)
        traj.append((cx, cy, a * h, h))
    return np.array(traj)


def draw_panel(im: np.ndarray, title: str, gt, pred, trail_gt, trail_pr,
               in_occ: bool, err: float | None, extra: str) -> np.ndarray:
    """Ve mot panel: box GT (trang) + box du doan cua dung 1 mo hinh."""
    p = im.copy()
    H, W = p.shape[:2]
    col = COL_DRIFT if in_occ else COL_PRED

    if gt is not None:
        x, y, w, h = gt.bb_left, gt.bb_top, gt.bb_width, gt.bb_height
        cv2.rectangle(p, (int(x), int(y)), (int(x + w), int(y + h)), COL_GT, 2)
    cx, cy, w, h = pred
    cv2.rectangle(p, (int(cx - w / 2), int(cy - h / 2)),
                  (int(cx + w / 2), int(cy + h / 2)), col, 3)

    for tr, tc in ((trail_gt, COL_GT), (trail_pr, col)):
        pts = tr[-TRAIL:]
        for j in range(1, len(pts)):
            cv2.line(p, pts[j - 1], pts[j], tc, 2)

    # Noi box GT voi box du doan de thay ngay sai so lech bao xa
    if gt is not None and in_occ:
        cv2.line(p, (int(gt.cx), int(gt.cy)), (int(cx), int(cy)), COL_DRIFT, 1, cv2.LINE_AA)

    cv2.rectangle(p, (0, 0), (W, 56), (0, 0, 0), -1)
    cv2.putText(p, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    txt = "dang ngoai suy mu" if in_occ else "dang duoc nap quan sat"
    cv2.putText(p, txt + (f"   sai so {err:6.1f} px" if err is not None else ""),
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if extra:
        cv2.putText(p, extra, (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if in_occ:
        cv2.rectangle(p, (0, 0), (W - 1, H - 1), COL_OCC, 5)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Video dai: KF+CV vs EKF+CTRV tren 1 track")
    ap.add_argument("--video", default="MVI_40991")
    ap.add_argument("--track", type=int, default=13)
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()

    gt = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    t = gt[(gt.video == a.video) & (gt.track_id == a.track)].sort_values("frame")
    if not len(t):
        print(f"[long_compare] khong tim thay {a.video} track {a.track}")
        return 1

    occ = pd.concat([pd.read_csv(config.INTERIM_DIR / f)
                     for f in ("full_occlusion_segments.csv", "occlusion_segments.csv")],
                    ignore_index=True)
    occ = occ[(occ.video == a.video) & (occ.track_id == a.track)].sort_values("start_frame")
    # GOP CAC DOAN CHONG NHAU. Hai bang che-mot-phan (>=0.10) va che-hoan-toan
    # (>=0.90) mo ta cung mot lan bi che o hai muc do, nen doan "hoan toan" luon
    # nam LONG trong doan "mot phan". De nguyen thi mot lan bi che bi dem thanh
    # 2-3 doan va FDE bi tinh o giua doan, sai hoan toan y nghia.
    spans: list[tuple[int, int]] = []
    for r in occ.itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        if spans and s <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    occ_frames = {f for s, e in spans for f in range(s, e + 1)}

    frames = [int(x) for x in t.frame]
    print(f"[long_compare] {a.video} track {a.track}: {len(frames)} frame "
          f"(frame {frames[0]}-{frames[-1]}), {len(spans)} doan che, "
          f"{len(occ_frames & set(frames))} frame bi che")
    for i, (s, e) in enumerate(spans, 1):
        print(f"    doan {i}: frame {s}-{e}  ({e - s + 1} frame)")

    tr = {m: run_track(t, occ_frames, m) for m in ("KF", "EKF")}

    img_root = config.find_images_root()
    first = cv2.imread(str(img_root / a.video / f"img{frames[0]:05d}.jpg"))
    H, W = first.shape[:2]
    out = config.RESULTS_DIR / "demo" / "videos"
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"long_{a.video}_t{a.track}.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (W * 2 + PANEL_GAP, H + 34))

    gt_rows = list(t.itertuples())
    tg, tk, te = [], [], []
    fde = []                       # (so doan, FDE cua KF, FDE cua EKF)
    seg_of = {}                    # frame -> chi so doan che
    for i, (s, e) in enumerate(spans):
        for f in range(s, e + 1):
            seg_of[f] = i

    for k, fr in enumerate(frames):
        im = cv2.imread(str(img_root / a.video / f"img{fr:05d}.jpg"))
        if im is None:
            continue
        r = gt_rows[k]
        in_occ = fr in occ_frames
        pk, pe = tr["KF"][k], tr["EKF"][k]
        ek = float(np.hypot(pk[0] - r.cx, pk[1] - r.cy))
        ee = float(np.hypot(pe[0] - r.cx, pe[1] - r.cy))

        tg.append((int(r.cx), int(r.cy)))
        tk.append((int(pk[0]), int(pk[1])))
        te.append((int(pe[0]), int(pe[1])))

        # Ghi lai FDE tai frame CUOI cua moi doan che
        if in_occ and fr == spans[seg_of[fr]][1]:
            fde.append((seg_of[fr] + 1, ek, ee))
        done = "   ".join(f"doan {i}: {x:.0f}px" for i, x, _ in fde)
        done_e = "   ".join(f"doan {i}: {x:.0f}px" for i, _, x in fde)

        left = draw_panel(im, "KF + CV", r, pk, tg, tk, in_occ, ek if in_occ else None, done)
        right = draw_panel(im, "EKF + CTRV", r, pe, tg, te, in_occ, ee if in_occ else None, done_e)
        gap = np.full((H, PANEL_GAP, 3), 255, np.uint8)
        combo = np.vstack([np.hstack([left, gap, right]),
                           np.full((34, W * 2 + PANEL_GAP, 3), 25, np.uint8)])

        # Thanh thoi gian: vach vang = cac doan bi che, vach trang = vi tri hien tai
        Wc = combo.shape[1]
        span = max(1, frames[-1] - frames[0])
        for s, e in spans:
            xa = int((s - frames[0]) / span * (Wc - 1))
            xb = int((e - frames[0]) / span * (Wc - 1))
            cv2.rectangle(combo, (xa, H + 8), (max(xb, xa + 2), H + 26), COL_OCC, -1)
        xc = int((fr - frames[0]) / span * (Wc - 1))
        cv2.rectangle(combo, (xc - 1, H + 4), (xc + 1, H + 30), (255, 255, 255), -1)
        cv2.putText(combo, f"frame {fr}   ({k + 1}/{len(frames)})   vang = dang bi che",
                    (8, H + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(combo)
    vw.release()

    print(f"\n[long_compare] FDE tai cuoi tung doan che (thap hon = tot hon):")
    print(f"    {'doan':<8}{'KF + CV':>12}{'EKF + CTRV':>14}{'chenh lech':>14}")
    print("    " + "-" * 46)
    for i, x, y in fde:
        print(f"    {i:<8}{x:>11.1f}px{y:>13.1f}px{x - y:>+13.1f}px")
    if fde:
        mk = float(np.mean([x for _, x, _ in fde])); me = float(np.mean([y for _, _, y in fde]))
        print(f"    {'trung binh':<8}{mk:>11.1f}px{me:>13.1f}px{mk - me:>+13.1f}px")
    print(f"\n[long_compare] {len(frames)} frame -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
