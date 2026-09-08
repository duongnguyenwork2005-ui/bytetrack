"""
omega_clamp_experiment.py -- Chan |omega| co cuu duoc EKF+CTRV khong?

BOI CANH. `diagnose_ekf_circle.py` cho thay EKF+CTRV khong dat gioi han nao cho
omega: 28% doan che ngoai suy voi toc do quay vuot nguong vat ly (2,9 do/frame
= 72 do/giay), 20% doan box du doan di het it nhat mot vong tron. Cau hoi tiep
theo la khuyet diem nay co PHAI LA NGUYEN NHAN khien CTRV thua CV khong.

PHEP THU. Chay 3 cau hinh tren CUNG 48 track va CUNG cac doan che:
    (1) KF + CV                       - baseline
    (2) EKF + CTRV nguyen ban          - omega tu do
    (3) EKF + CTRV co chan omega       - |omega| <= CLAMP

Chan omega la ky thuat tieu chuan cho CTRV, khong phai meo lam dep so lieu:
CTRV ngoai suy tren duong tron ban kinh R = v/|omega|, nen omega bi uoc luong
qua lon se cho R nho hon ca chiec xe va box quay tit tai cho.

NEU chan omega dao nguoc ket qua -> ket luan cu ("CTRV khong giup") la do LOI
        CAI DAT, phai viet lai.
NEU khong dao nguoc -> khuyet diem co that nhung khong phai nguyen nhan chinh.

CACH DUNG
    python src/omega_clamp_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
# Dung DUNG lop ma tracker su dung, khong dinh nghia lai: neu nguong hay cach
# chan thay doi thi thi nghiem nay va ket qua tracking that phai khop nhau.
from ekf_ctrv_clamped import OMEGA_CLAMP as CLAMP  # noqa: E402
from ekf_ctrv_clamped import EKFTrackerCTRVClamped as EKFClamped  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402

TIE = 0.5   # px - chenh lech duoi muc nay coi nhu hoa


def xyah(r) -> np.ndarray:
    return np.array([r.cx, r.cy, r.bb_width / max(r.bb_height, 1e-6), r.bb_height], float)


def run(g: pd.DataFrame, occ_frames: set[int], kind: str) -> np.ndarray:
    """Chay bo loc suot doi track, chi `update` o frame KHONG bi che."""
    if kind == "KF":
        f = KalmanFilterXYAH(); pos = lambda m: (m[0], m[1]); ctrv = False
    else:
        f = EKFTrackerCTRV() if kind == "EKF" else EKFClamped()
        pos = lambda m, f=f: (m[f.CX], m[f.CY]); ctrv = True

    rows = [g.iloc[i] for i in range(len(g))]
    m, c = f.initiate(xyah(rows[0]))
    seeded = False
    last_obs = int(rows[0].frame)
    out = [pos(m)]
    for r in rows[1:]:
        vis = int(r.frame) not in occ_frames
        if ctrv and not seeded and vis:
            # KHOANG CACH THAT giua hai quan sat, khong phai 1. Bo loc nay chay XUYEN QUA
            # cac doan che, nen quan sat dau tien sau mot doan che co the cach quan sat
            # truoc do hang chuc frame. `initiate_from_motion` chia dich chuyen cho
            # n_frames de ra van toc; truyen 1.0 se thoi phong van toc dung bang so frame
            # da bi che (do duoc: 294 px/frame thay vi 10,9 px/frame - gap 27 lan).
            m, c = f.initiate_from_motion(
                m, c, xyah(r), n_frames=max(1, int(r.frame) - last_obs))
            seeded = True
        m, c = f.predict(m, c)
        if vis:
            m, c = f.update(m, c, xyah(r))
            last_obs = int(r.frame)
        out.append(pos(m))
    return np.array(out)


def merge_spans(o: pd.DataFrame) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for r in o.sort_values("start_frame").itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        if spans and s <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def main() -> int:
    gt = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    occ_all = pd.concat([pd.read_csv(config.INTERIM_DIR / f)
                         for f in ("full_occlusion_segments.csv", "occlusion_segments.csv")],
                        ignore_index=True)

    rows = []
    for (v, tid), g in gt.groupby(["video", "track_id"]):
        g = g.sort_values("frame").reset_index(drop=True)
        if len(g) < 150 or g.speed.median() <= 2.0:
            continue
        o = occ_all[(occ_all.video == v) & (occ_all.track_id == tid)]
        if len(o) < 2:
            continue
        spans = merge_spans(o)
        occ_frames = {f for s, e in spans for f in range(s, e + 1)}
        idx = {int(f): i for i, f in enumerate(g.frame)}
        try:
            tr = {k: run(g, occ_frames, k) for k in ("KF", "EKF", "EKFC")}
        except Exception:
            continue
        for s, e in spans:
            if e not in idx:
                continue
            i = idx[e]
            r = g.iloc[i]
            rows.append(dict(video=v, track_id=tid, start=s, end=e, T=e - s + 1,
                             **{k: float(np.hypot(tr[k][i][0] - r.cx, tr[k][i][1] - r.cy))
                                for k in ("KF", "EKF", "EKFC")}))

    d = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "demo" / "clamp_test.csv"
    d.to_csv(out, index=False)

    print("=" * 62)
    print(f"CHAN OMEGA CO CUU DUOC CTRV KHONG?   ({len(d)} doan che, 48 track)")
    print("=" * 62)
    print(f"\nFDE tai cuoi doan che (thap hon = tot hon):")
    print(f"  {'cau hinh':<28}{'trung binh':>13}{'trung vi':>12}")
    print("  " + "-" * 53)
    for k, lab in [("KF", "KF + CV"), ("EKF", "EKF + CTRV (nguyen ban)"),
                   ("EKFC", f"EKF + CTRV (chan {CLAMP})")]:
        print(f"  {lab:<28}{d[k].mean():>11.1f}px{d[k].median():>10.1f}px")

    print(f"\nSo cap tren tung doan - chan omega vs nguyen ban:")
    print(f"  chan omega TOT hon : {(d.EKFC < d.EKF - TIE).sum():>3} doan")
    print(f"  chan omega TE hon  : {(d.EKFC > d.EKF + TIE).sum():>3} doan")
    print(f"  khong doi          : {(abs(d.EKFC - d.EKF) <= TIE).sum():>3} doan")

    print(f"\nSo cap tren tung doan - EKF da chan omega vs KF + CV:")
    print(f"  EKF tot hon : {(d.EKFC < d.KF - TIE).sum():>3} doan")
    print(f"  KF  tot hon : {(d.EKFC > d.KF + TIE).sum():>3} doan")
    print(f"  hoa         : {(abs(d.EKFC - d.KF) <= TIE).sum():>3} doan")

    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
