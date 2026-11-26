"""
tune_omega_noise.py -- Thu cai thien do tin cay cua omega bang cach hieu chinh
lai nhieu qua trinh, thay vi chi dung gia tri mac dinh trong config.py.

BOI CANH (xem docstring diagnose_ukf.py phan A3)
Do tren 507 doan che khuat that: omega hoc duoc TRUOC luc bi che chi dung dau
voi khuc cua THAT xay ra TRONG luc che 55.6% so lan (EKF) - gan nhu tung dong
xu. Va |omega| uoc luong TRUNG VI gap ~3 lan gia tri GT thuc te (0.213 vs
0.071 do/frame). Day la nguyen nhan chinh khien CTRV nhieu khi ngoai suy cong
SAI HUONG va thua CV.

CTRV_STD_OMEGA (nhieu qua trinh cho toc do rieng cua omega) va
CTRV_INIT_STD_OMEGA (do bat dinh omega luc khoi tao) o config.py duoc chon TU
PHEP QUET TREN MO PHONG (compare_extrapolation.py, do sai so ngoai suy theo px).
Phep quet do KHONG dam bao toi uu cho tieu chi "dung dau voi khuc cua that"
tren du lieu video that - day chinh la khoang trong file nay lap.

CACH LAM: quet lai CTRV_STD_OMEGA / CTRV_INIT_STD_OMEGA, do lai 2 chi so:
  (a) ti le omega DUNG DAU voi khuc cua that (cang cao cang tot)
  (b) ti le |omega uoc luong| / |omega GT| (cang gan 1 cang tot, qua cao la
      dang "duoi theo nhieu")
Chi dung tinh toan filter thuan tuy tren quy dao GT da co san (khong can chay
lai YOLO/ByteTrack) -> re, nhanh, dung de SANG LOC truoc khi quyet dinh co dang
chay lai toan bo pipeline tren GPU hay khong.

CANH BAO METHODOLOGICAL: day la phep do OFFLINE, dung dung quy dao GT (khong
co detection that, khong co ByteTrack that). Ket qua tot o day la DIEU KIEN
CAN, khong phai DIEU KIEN DU - phai xac nhan lai bang chay that tren GPU truoc
khi ket luan.

CACH DUNG
    python src/tune_omega_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from diagnose_ukf import load_segments, net_turn_deg, WARMUP, MIN_SPEED  # noqa: E402

# Ung vien can quet. Gia tri hien tai trong config.py: STD_OMEGA=0.005,
# INIT_STD_OMEGA=0.10 - luon dua vao danh sach de doi chieu.
STD_OMEGA_CANDIDATES = [0.010, 0.005, 0.002, 0.001, 0.0005]
INIT_STD_OMEGA_CANDIDATES = [0.10, 0.05, 0.02, 0.01]


def xyah(row) -> np.ndarray:
    w, h = row.bb_width, row.bb_height
    return np.array([row.cx, row.cy, w / max(h, 1e-6), h], dtype=float)


def fit_omega(pre: pd.DataFrame, std_omega: float, init_std_omega: float) -> float:
    """Nap quan sat GT truoc doan che, tra ve omega EKF hoc duoc (rad/frame).

    Ghi de tham so nhieu SAU KHI tao filter (khong sua config.py) de quet
    duoc nhieu gia tri trong 1 lan chay ma khong phai reload module.
    """
    f = EKFTrackerCTRV()
    f._std_omega = std_omega
    old_init = config.CTRV_INIT_STD_OMEGA
    config.CTRV_INIT_STD_OMEGA = init_std_omega   # initiate() doc thang tu config
    try:
        rows = list(pre.itertuples())
        m, c = f.initiate(xyah(rows[0])); inited = False
        for r in rows[1:]:
            z = xyah(r)
            if not inited:
                m, c = f.initiate_from_motion(m, c, z, n_frames=1.0); inited = True
            m, c = f.predict(m, c)
            m, c = f.update(m, c, z)
        return float(m[f.OMEGA])
    finally:
        config.CTRV_INIT_STD_OMEGA = old_init


def main() -> int:
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}

    # --- Chuan bi du lieu 1 lan: (pre-warmup, omega GT) cho tung doan hop le ---
    cases = []
    for r in load_segments().itertuples(index=False):
        t = g.get((r.video, r.track_id))
        if t is None:
            continue
        pre = t[(t.frame < r.start_frame) & (t.frame >= r.start_frame - WARMUP)].sort_values("frame")
        occ = t[(t.frame >= r.start_frame) & (t.frame <= r.end_frame)].sort_values("frame")
        if len(pre) < 8 or len(occ) < 12:
            continue
        p = np.c_[occ.cx.values, occ.cy.values]
        d = np.diff(p, axis=0)
        if np.hypot(*d.T).mean() < MIN_SPEED:
            continue
        turn = net_turn_deg(p)
        if turn == 0.0:
            continue
        cases.append((pre, turn / len(d)))   # om_gt tinh rieng, khong doi qua cac lan quet

    print(f"So doan dung de quet: {len(cases)} (giong het Stage A3)\n")

    rows = []
    for so in STD_OMEGA_CANDIDATES:
        for iso in INIT_STD_OMEGA_CANDIDATES:
            om_gt = np.array([c[1] for c in cases])
            om_est = np.array([np.rad2deg(fit_omega(pre, so, iso)) for pre, _ in cases])
            same_sign = np.sign(om_est) == np.sign(om_gt)
            ratio = np.median(np.abs(om_est)) / max(np.median(np.abs(om_gt)), 1e-9)
            rows.append(dict(
                std_omega=so, init_std_omega=iso,
                sign_acc_pct=round(float(same_sign.mean()) * 100, 1),
                omega_est_median=round(float(np.median(np.abs(om_est))), 4),
                ratio_vs_GT=round(float(ratio), 2),
                is_current=(so == config.CTRV_STD_OMEGA and iso == config.CTRV_INIT_STD_OMEGA),
            ))
            mark = "  <- HIEN TAI" if rows[-1]["is_current"] else ""
            print(f"  std_omega={so:<7} init_std_omega={iso:<6} "
                  f"-> dung dau {rows[-1]['sign_acc_pct']:>5.1f}%  "
                  f"|omega| trung vi {rows[-1]['omega_est_median']:.4f} do/f "
                  f"(x{rows[-1]['ratio_vs_GT']:.2f} GT){mark}")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "stratified" / "tune_omega_noise.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*78}\nTOM TAT\n{'='*78}")
    base = df[df.is_current].iloc[0]
    print(f"  Hien tai (config.py)  : dung dau {base.sign_acc_pct}%, "
          f"|omega| gap {base.ratio_vs_GT}x GT")
    best = df.sort_values("sign_acc_pct", ascending=False).iloc[0]
    print(f"  Tot nhat trong quet    : std_omega={best.std_omega} "
          f"init_std_omega={best.init_std_omega} "
          f"-> dung dau {best.sign_acc_pct}%, |omega| gap {best.ratio_vs_GT}x GT")
    delta = best.sign_acc_pct - base.sign_acc_pct
    print(f"  Chenh lech             : {delta:+.1f} diem phan tram")
    if delta < 2.0:
        print("\n  => Xiet nhieu qua trinh KHONG cai thien dang ke ti le dung dau.")
        print("     Gia thuyet: sai dau chu yeu do xe DOI HUONG DUNG LUC vao/ra vung")
        print("     che (khong co tin hieu nao TRUOC do de bat ky muc nhieu nao bat")
        print("     duoc), khong phai do bo loc 'duoi theo nhieu do'. Can kiem tra")
        print("     bang confidence-gated CTRV thay vi tiep tuc xiet nhieu qua trinh.")
    else:
        print(f"\n  => Co cai thien. De xuat thu std_omega={best.std_omega}, "
              f"init_std_omega={best.init_std_omega} tren pipeline that (vai video "
              f"truoc khi chay toan bo 60 video).")
    print(f"\n  Da luu bang day du -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
