"""
phaseB_iou_diagnosis.py -- Giai doan B buoc 1: DO DE XAC NHAN NGHI VAN truoc khi sua code.

===========================================================================
NGHI VAN CAN KIEM CHUNG
===========================================================================
BAO_CAO_KHOA_LUAN.md muc 3.1: tren 1.544 doan che khuat, ca 3 motion model cho
ket cuc GIONG HET NHAU o 97.3%. Chi 42 doan (2.7%) co khac biet.

Nghi van: ByteTrack lien ket bang IoU THUAN. Ham IoU co "vach dung" - box du
doan lech 15px va lech 200px deu cho IoU = 0 y het nhau. Neu phan lon du doan
roi vao vung IoU = 0 thi chat luong ngoai suy tot hon cua CTRV KHONG DUOC ham
chi phi tieu thu -> moi motion model deu cho cung mot ket cuc, va do la ly do
that su cua con so 97.3% chu khong phai vi 3 model du doan giong nhau.

===========================================================================
DO CAI GI
===========================================================================
Voi moi doan che khuat: nap quan sat GT den truoc doan che, roi CHI predict
qua doan che (mo phong dung tinh huong khong co detection moi). Tai frame CUOI
doan che - dung luc ByteTrack se thu lien ket lai - tinh:

    IoU (box du doan, box GT)      <- cai ByteTrack thuc su dung
    DIoU(box du doan, box GT)      <- co gradient ca khi khong giao nhau
    he so gian no can thiet        <- phai noi box du doan bao nhieu lan de
                                      no giao duoc voi box GT

LUU Y VE TINH TRUNG THUC CUA PHEP DO: day la mo phong dung quy dao GT lam dau
vao, KHONG phai trang thai that cua tracker trong pipeline (tracker co the da
mat track tu truoc do vi detector truot). Vi vay day la CAN TREN lac quan cua
chat luong du doan. Neu ngay ca can tren lac quan nay ma IoU van bang 0 thi
nghi van cang duoc cung co.

===========================================================================
CACH DUNG
    python src/phaseB_iou_diagnosis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402
from diagnose_ukf import load_segments, WARMUP  # noqa: E402


def xyah(row) -> np.ndarray:
    w, h = row.bb_width, row.bb_height
    return np.array([row.cx, row.cy, w / max(h, 1e-6), h], dtype=float)


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """IoU giua 2 box dang (x_topleft, y_topleft, w, h)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def diou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """DIoU = IoU - d^2/c^2.

    d = khoang cach giua 2 tam, c = duong cheo cua box bao nho nhat chua ca hai.
    Khac biet then chot voi IoU: khi 2 box KHONG giao nhau, IoU = 0 bat ke xa
    gan, con DIoU tiep tuc giam dan theo khoang cach -> ham chi phi con "doc"
    de phan biet du doan tot va du doan te. Mien gia tri [-1, 1].
    """
    iou = iou_xywh(a, b)
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2
    ex1, ey1 = min(a[0], b[0]), min(a[1], b[1])
    ex2, ey2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    return iou - d2 / c2 if c2 > 0 else iou


def expand_factor_needed(pred: np.ndarray, gt: np.ndarray) -> float:
    """He so gian no nho nhat de box du doan giao duoc voi box GT.

    Tra ve 1.0 neu da giao san. Dung de biet "expanded gate" phai noi bao nhieu
    moi cuu duoc cac truong hop dang truot.
    """
    if iou_xywh(pred, gt) > 0:
        return 1.0
    pcx, pcy = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
    for k in np.arange(1.1, 12.01, 0.1):
        w, h = pred[2] * k, pred[3] * k
        big = np.array([pcx - w / 2, pcy - h / 2, w, h])
        if iou_xywh(big, gt) > 0:
            return float(k)
    return np.inf


def predict_box_through_occlusion(pre: pd.DataFrame, n_pred: int, model: str):
    """Nap GT truoc doan che roi CHI predict n_pred buoc. Tra ve box (x,y,w,h)."""
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
    if model == "KF + CV":
        f = KalmanFilterXYAH()
        get = lambda m: (m[0], m[1], m[2], m[3])   # x,y,a,h (x,y la TAM)
        ctrv = False
    else:
        f = EKFTrackerCTRV() if model.startswith("EKF") else UKFTrackerCTRV()
        get = lambda m, f=f: (m[f.CX], m[f.CY], m[f.A], m[f.H])
        ctrv = True

    rows = list(pre.itertuples())
    m, c = f.initiate(xyah(rows[0]))
    inited = False
    for r in rows[1:]:
        z = xyah(r)
        if ctrv and not inited:
            m, c = f.initiate_from_motion(m, c, z, n_frames=1.0)
            inited = True
        m, c = f.predict(m, c)
        m, c = f.update(m, c, z)
    for _ in range(n_pred):
        m, c = f.predict(m, c)
    cx, cy, a, h = get(m)
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, w, h]), float(np.trace(c))


def main() -> int:
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}
    MODELS = ["KF + CV", "EKF + CTRV", "UKF + CTRV"]

    rows = []
    for r in load_segments().itertuples(index=False):
        t = g.get((r.video, r.track_id))
        if t is None:
            continue
        pre = t[(t.frame < r.start_frame) &
                (t.frame >= r.start_frame - WARMUP)].sort_values("frame")
        occ = t[(t.frame >= r.start_frame) & (t.frame <= r.end_frame)].sort_values("frame")
        if len(pre) < 8 or len(occ) < 2:
            continue
        last = occ.iloc[-1]      # frame cuoi doan che = luc ByteTrack thu lien ket lai
        gt_box = np.array([last.bb_left, last.bb_top, last.bb_width, last.bb_height], float)
        for mdl in MODELS:
            try:
                pb, trP = predict_box_through_occlusion(pre, len(occ), mdl)
            except Exception:
                continue
            rows.append(dict(
                video=r.video, track_id=r.track_id, seg_id=r.seg_id, lvl=r.lvl,
                bucket=r.bucket, n_pred=len(occ), model=mdl,
                iou=iou_xywh(pb, gt_box), diou=diou_xywh(pb, gt_box),
                expand=expand_factor_needed(pb, gt_box), traceP=trP))

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "phaseB"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "iou_at_reassociation.csv", index=False)

    print("=" * 78)
    print("GIAI DOAN B buoc 1 - PHAN BO IoU TAI THOI DIEM LIEN KET LAI")
    print("=" * 78)
    print(f"\nSo doan x model: {len(df)}  ({df.seg_id.count() // len(MODELS)} doan x {len(MODELS)} model)")

    print(f"\n--- Phan bo IoU (box du doan vs box GT) ---")
    bins = [-.001, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    lab = ["= 0 (KHONG giao)", "0-0.05", "0.05-0.1", "0.1-0.2",
           "0.2-0.3", "0.3-0.5", "> 0.5"]
    for mdl in MODELS:
        d = df[df.model == mdl]
        cnt = pd.cut(d.iou, bins=bins, labels=lab).value_counts().reindex(lab)
        pct = (cnt / len(d) * 100).round(1)
        print(f"\n  {mdl}  (n={len(d)})")
        for k in lab:
            bar = "#" * int(pct[k] / 2)
            print(f"    {k:<18} {int(cnt[k]):>5} ({pct[k]:>5.1f}%) {bar}")

    print(f"\n--- Tom tat ---")
    print(f"{'model':<14} {'% IoU=0':>9} {'% IoU<0.2':>11} {'median IoU':>12} {'median DIoU':>13}")
    print("-" * 64)
    for mdl in MODELS:
        d = df[df.model == mdl]
        print(f"{mdl:<14} {(d.iou <= 0).mean() * 100:>8.1f}% "
              f"{(d.iou < 0.2).mean() * 100:>10.1f}% "
              f"{d.iou.median():>12.4f} {d.diou.median():>13.4f}")

    print(f"\n--- Theo do dai doan che (% IoU = 0) ---")
    p = df.pivot_table(index="bucket", columns="model", values="iou",
                       aggfunc=lambda s: (s <= 0).mean() * 100)
    n = df.pivot_table(index="bucket", columns="model", values="iou", aggfunc="size")
    print((p.round(1)).to_string())
    print("\n  n moi bucket:"); print(n.iloc[:, 0].astype(int).to_string())

    print(f"\n--- He so gian no can thiet de box du doan giao duoc GT ---")
    print("    (chi tinh tren cac doan dang co IoU = 0)")
    for mdl in MODELS:
        d = df[(df.model == mdl) & (df.iou <= 0)]
        if not len(d):
            continue
        fin = d[np.isfinite(d.expand)]
        print(f"  {mdl:<14} n={len(d):>4}  "
              f"gian no <=2x cuu duoc {(fin.expand <= 2).sum() / len(d) * 100:>5.1f}%  "
              f"<=3x {(fin.expand <= 3).sum() / len(d) * 100:>5.1f}%  "
              f"<=5x {(fin.expand <= 5).sum() / len(d) * 100:>5.1f}%  "
              f"khong the (>12x) {(~np.isfinite(d.expand)).sum() / len(d) * 100:>5.1f}%")

    print(f"\n  Da luu -> {out / 'iou_at_reassociation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
