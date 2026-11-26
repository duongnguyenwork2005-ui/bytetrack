"""
omega_smoother.py -- Giai doan A: uoc luong omega HOI CO (retrospective) tren
cua so frame SACH ngay truoc khi track bi mat.

===========================================================================
VAN DE CAN GIAI QUYET
===========================================================================
Do o Stage A3 (xem BAO_CAO_KHOA_LUAN.md muc 3.3): omega ma bo loc dang giu
trong state chi DUNG DAU voi khuc cua that 55.6% so lan (~ tung dong xu) va
do lon |omega| bi phong dai gap ~3 lan gia tri GT.

Gia thuyet cua giai doan nay: nguyen nhan la ta dang dung omega TUC THOI trong
state, von duoc uoc luong tu tam bbox nhieu. Te hon nua, cac detection NGAY
TRUOC luc bi che thuong la box da bi cat xen (che mot phan) nen nhieu nang
nhat - dung luc ta can so lieu sach nhat thi no lai ban nhat.

=> Thu uoc luong lai omega bang cach nhin NGUOC LAI mot cua so frame SACH,
   thay vi lay gia tri tuc thoi tai frame cuoi.

===========================================================================
HAI CACH UOC LUONG
===========================================================================

--- (a) RTS SMOOTHER (Rauch-Tung-Striebel) ---
Bo loc Kalman thong thuong tai frame k chi dung thong tin tu frame 1..k
(nhan qua). Smoother chay NGUOC lai, dung ca thong tin tuong lai 1..N de uoc
luong lai trang thai tai moi frame k < N:

    C_k     = P_k|k @ F_k^T @ inv(P_k+1|k)
    x_k|N   = x_k|k + C_k @ (x_k+1|N - x_k+1|k)
    P_k|N   = P_k|k + C_k @ (P_k+1|N - P_k+1|k) @ C_k^T

LUU Y RAT QUAN TRONG (de trinh bay truoc hoi dong):
Tai frame CUOI CUNG cua cua so (k = N), smoother KHONG cai thien duoc gi:
x_N|N cua smoother trung khit voi x_N|N cua bo loc, vi khong co thong tin
tuong lai nao ngoai N. Vi vay neu ta lay omega smoothed TAI FRAME CUOI thi
se ra dung bang omega hien tai -> vo nghia.

Do do ta lay TRUNG VI cua omega smoothed TREN TOAN CUA SO. Y nghia vat ly:
"turn rate dien hinh cua xe trong doan sach ngay truoc khi bi che", chinh la
dai luong ta muon dung de ngoai suy, va no on dinh hon nhieu so voi gia tri
tuc thoi tai mot frame don le.

Thanh phan theta la GOC nen hieu (x_k+1|N - x_k+1|k) phai duoc goi ve
[-pi, pi) truoc khi nhan voi C_k - neu khong mot cap goc sat nhau vat qua bien
+-pi se tao ra so hang gia ~ 2*pi (loi kinh dien khi smoothing co bien goc).

--- (b) CIRCLE FIT ---
Khop mot cung tron di qua chuoi tam bbox trong cua so bang binh phuong toi
thieu (phuong phap dai so Kasa):

    Voi moi diem: x^2 + y^2 = 2*a*x + 2*b*y + c   voi c = R^2 - a^2 - b^2
    -> he tuyen tinh [2x, 2y, 1] @ [a, b, c]^T = x^2 + y^2
    -> giai least squares, roi R = sqrt(c + a^2 + b^2)

    omega = v / R      [rad/frame]   (v: toc do trung binh px/frame)

DAU cua omega lay tu tong tich co huong cua cac vector dich chuyen lien tiep:
    cross(d_i, d_i+1) = d_i.x * d_i+1.y - d_i.y * d_i+1.x
Tong duong = quay nguoc chieu kim dong ho theo he toa do dang dung, tuong ung
theta tang dan, tuc omega > 0. Cach nay NHAT QUAN voi quy uoc
theta = atan2(dy, dx) cua bo loc.

Khac biet ban chat so voi RTS: circle fit KHONG dung mo hinh dong hoc hay
nhieu do nao ca, no thuan tuy hinh hoc -> khong bi anh huong boi cach dat Q/R,
nhung cung khong tan dung duoc thong tin ve do tin cay cua tung phep do.

===========================================================================
CACH DUNG
===========================================================================
    from omega_smoother import clean_window, omega_rts, omega_circle_fit
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402

# --- Tham so chon cua so sach ---
WIN_MAX = 15          # so frame toi da cua cua so hoi co
WIN_MIN = 10          # duoi nguong nay coi nhu khong du du lieu -> fallback
WIN_LOOKBACK = 25     # nhin nguoc toi da bao nhieu frame de tim frame sach
MAX_OCC_RATIO = 0.10  # occlusion_ratio duoi nguong nay moi coi la "sach"
                      # (dung nguong OCCLUSION_MIN_RATIO cua Giai doan 1)


def clean_window(track: pd.DataFrame, start_frame: int) -> pd.DataFrame:
    """Lay cua so frame SACH ngay truoc `start_frame` (luc bat dau bi che).

    "Sach" = occlusion_ratio < MAX_OCC_RATIO. Ly do phai loc: cac detection
    ngay truoc doan che thuong da bi che mot phan, tam bbox bi lech manh ->
    dung chung de uoc luong omega thi chinh cho can chinh xac nhat lai nhieu
    nhat.

    Tra ve DataFrame (co the rong / ngan hon WIN_MIN -> ben goi phai fallback).
    """
    w = track[(track.frame < start_frame) &
              (track.frame >= start_frame - WIN_LOOKBACK)]
    if "occlusion_ratio" in w.columns:
        w = w[w.occlusion_ratio < MAX_OCC_RATIO]
    return w.sort_values("frame").tail(WIN_MAX)


def _xyah(row) -> np.ndarray:
    w, h = row.bb_width, row.bb_height
    return np.array([row.cx, row.cy, w / max(h, 1e-6), h], dtype=float)


def omega_rts(window: pd.DataFrame, filt: EKFTrackerCTRV | None = None) -> float | None:
    """omega hoi co bang RTS smoother. Tra ve rad/frame, hoac None neu that bai.

    Chay bo loc xuoi tren cua so (luu lai x_k|k, P_k|k, x_k+1|k, P_k+1|k, F_k),
    roi chay smoother nguoc, cuoi cung lay TRUNG VI omega tren toan cua so
    (xem giai thich o docstring dau file ve viec vi sao khong lay frame cuoi).
    """
    f = filt if filt is not None else EKFTrackerCTRV()
    rows = list(window.itertuples())
    if len(rows) < 3:
        return None

    # --- Luot xuoi: luu lai moi thu smoother can ---
    xf, Pf = [], []          # x_k|k, P_k|k        (sau update)
    xp, Pp, Fs = [], [], []  # x_k+1|k, P_k+1|k, F_k (truoc update)

    m, c = f.initiate(_xyah(rows[0]))
    inited = False
    for r in rows[1:]:
        z = _xyah(r)
        if not inited:
            m, c = f.initiate_from_motion(m, c, z, n_frames=1.0)
            inited = True
        F = f.jacobian(m)
        m_pred, c_pred = f.predict(m, c)
        Fs.append(F); xp.append(m_pred.copy()); Pp.append(c_pred.copy())
        m, c = f.update(m_pred, c_pred, z)
        xf.append(m.copy()); Pf.append(c.copy())

    n = len(xf)
    if n < 3:
        return None

    # --- Luot nguoc: RTS ---
    xs = [None] * n; Ps = [None] * n
    xs[-1], Ps[-1] = xf[-1].copy(), Pf[-1].copy()   # tai N: smoothed == filtered
    for k in range(n - 2, -1, -1):
        try:
            C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        except np.linalg.LinAlgError:
            return None
        d = xs[k + 1] - xp[k + 1]
        # theta la GOC -> phai goi hieu ve [-pi, pi), neu khong sinh so hang gia ~2pi
        d[f.THETA] = normalize_angle(d[f.THETA])
        xs[k] = xf[k] + C @ d
        xs[k][f.THETA] = normalize_angle(xs[k][f.THETA])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    om = np.array([x[f.OMEGA] for x in xs])
    return float(np.median(om))


def omega_circle_fit(window: pd.DataFrame) -> float | None:
    """omega hoi co bang khop cung tron (Kasa least squares). rad/frame."""
    p = np.c_[window.cx.values, window.cy.values]
    if len(p) < 4:
        return None
    d = np.diff(p, axis=0)
    speed = float(np.hypot(*d.T).mean())
    if speed < 1e-6:
        return 0.0

    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(p))]
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a_c, b_c, c_c = sol
    r2 = c_c + a_c ** 2 + b_c ** 2
    if not np.isfinite(r2) or r2 <= 1e-9:
        return None
    R = float(np.sqrt(r2))
    if R < 1e-6 or R > 1e7:      # gan nhu thang -> turn rate ~ 0
        return 0.0

    # Dau: tong tich co huong cua cac vector dich chuyen lien tiep.
    # Nhat quan voi quy uoc theta = atan2(dy, dx) cua bo loc.
    cross = float(np.sum(d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]))
    sign = 1.0 if cross >= 0 else -1.0
    return sign * speed / R


# ===========================================================================
# DANH GIA: so sanh 3 cach uoc luong omega tren CUNG 507 doan cua Stage A3
# ===========================================================================
def main() -> int:
    """So sanh omega state (hien tai) vs RTS vs circle fit.

    Dung Y HET tap doan va cach tinh omega GT cua diagnose_ukf.py phan A3 de
    ket qua so sanh truc tiep duoc voi so lieu da bao cao truoc do.
    """
    from diagnose_ukf import load_segments, net_turn_deg, WARMUP, MIN_SPEED

    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}

    rows = []
    for r in load_segments().itertuples(index=False):
        t = g.get((r.video, r.track_id))
        if t is None:
            continue
        pre = t[(t.frame < r.start_frame) &
                (t.frame >= r.start_frame - WARMUP)].sort_values("frame")
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
        om_gt = turn / len(d)                      # do/frame

        # --- (0) omega hien tai: gia tri trong state sau khi nap het cua so warmup ---
        f = EKFTrackerCTRV()
        rr = list(pre.itertuples())
        m, c = f.initiate(_xyah(rr[0])); inited = False
        for q in rr[1:]:
            z = _xyah(q)
            if not inited:
                m, c = f.initiate_from_motion(m, c, z, n_frames=1.0); inited = True
            m, c = f.predict(m, c)
            m, c = f.update(m, c, z)
        om_state = np.rad2deg(m[f.OMEGA])

        # --- (a)(b) hai cach hoi co tren cua so SACH ---
        win = clean_window(t, r.start_frame)
        n_clean = len(win)
        enough = n_clean >= WIN_MIN
        if enough:
            o_rts = omega_rts(win)
            o_cir = omega_circle_fit(win)
        else:
            o_rts = o_cir = None
        # Fallback ve omega hien tai khi cua so khong du sach/dai
        om_rts = np.rad2deg(o_rts) if o_rts is not None else om_state
        om_cir = np.rad2deg(o_cir) if o_cir is not None else om_state

        rows.append(dict(video=r.video, track_id=r.track_id, seg_id=r.seg_id,
                         lvl=r.lvl, nf=r.length_frames, n_clean=n_clean,
                         fallback=not enough or o_rts is None or o_cir is None,
                         om_gt=om_gt, om_state=om_state, om_rts=om_rts, om_cir=om_cir))

    df = pd.DataFrame(rows)
    out_dir = config.RESULTS_DIR / "phaseA"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "omega_methods_comparison.csv", index=False)

    print("=" * 78)
    print("GIAI DOAN A - SO SANH 3 CACH UOC LUONG OMEGA")
    print("=" * 78)
    print(f"\nSo doan phan tich : {len(df)}  (giong het Stage A3)")
    nfb = int(df.fallback.sum())
    print(f"Phai fallback     : {nfb} ({nfb / len(df) * 100:.1f}%) "
          f"- cua so sach < {WIN_MIN} frame hoac khop that bai")
    print(f"Do dai cua so sach: trung vi {df.n_clean.median():.0f} frame "
          f"(min {df.n_clean.min()}, max {df.n_clean.max()})")

    print(f"\n{'Phuong phap':<22} {'% dung dau':>11} {'median |om|':>13} {'ti le vs GT':>12}")
    print("-" * 62)
    gt_med = float(df.om_gt.abs().median())
    for col, name in [("om_state", "omega state (hien tai)"),
                      ("om_rts", "RTS smoother"),
                      ("om_cir", "Circle fit")]:
        acc = float((np.sign(df[col]) == np.sign(df.om_gt)).mean()) * 100
        med = float(df[col].abs().median())
        print(f"{name:<22} {acc:>10.1f}% {med:>12.4f} {med / gt_med:>11.2f}x")
    print(f"{'GT (that)':<22} {'-':>11} {gt_med:>12.4f} {'1.00x':>12}")

    # Chi tren cac doan KHONG phai fallback (noi phuong phap moi thuc su chay)
    sub = df[~df.fallback]
    if len(sub):
        print(f"\n[Chi tren {len(sub)} doan KHONG fallback - noi phuong phap moi thuc su chay]")
        gt2 = float(sub.om_gt.abs().median())
        for col, name in [("om_state", "omega state"), ("om_rts", "RTS"), ("om_cir", "Circle fit")]:
            acc = float((np.sign(sub[col]) == np.sign(sub.om_gt)).mean()) * 100
            med = float(sub[col].abs().median())
            print(f"  {name:<16} dung dau {acc:>5.1f}%   |om| {med:.4f} ({med / gt2:.2f}x GT)")

    print(f"\n  Da luu -> {out_dir / 'omega_methods_comparison.csv'}")

    # --- Doi chieu voi tieu chi cong ---
    best = max(float((np.sign(df[c]) == np.sign(df.om_gt)).mean()) * 100
               for c in ["om_rts", "om_cir"])
    base = float((np.sign(df.om_state) == np.sign(df.om_gt)).mean()) * 100
    print(f"\n{'=' * 78}\nTIEU CHI CONG\n{'=' * 78}")
    print(f"  Hien tai: {base:.1f}%  ->  Tot nhat cua 2 cach moi: {best:.1f}%")
    if best >= 65.0:
        print("  => NHANH 1: >= 65% - huong nay DANG dau tu, tich hop vao tracker")
        print("     va chay lai phan tang day du.")
    else:
        print("  => NHANH 2: van quanh 55-60% - DONG huong nay, ghi lai nhu negative")
        print("     result co gia tri (chung minh gioi han la CAU TRUC chu khong phai")
        print("     do nhieu uoc luong), chuyen thang sang Giai doan B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
