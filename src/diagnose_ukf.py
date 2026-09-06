"""
diagnose_ukf.py -- Stage A: chan doan vi sao UKF KHONG tot hon EKF tren video that.

BOI CANH
Ly thuyet noi UKF >= EKF: Unscented Transform chinh xac toi bac 3 khai trien
Taylor, con EKF chi tuyen tinh hoa bac 1. Nhung ket qua phan tang Giai doan 5
lai cho EKF > UKF > KF o tang che nang 0.5-1.5s (11.5% / 7.7% / 6.4%).
File nay truy nguyen nhan, chia 3 phan:

    A1  Kiem tra 3 cho de sai nhat khi dung UKF voi bien GOC (theta)
    A2  Ngoai suy thuan tuy tren track THAT co che khuat dai + xe dang re
    A3  Do do BEN VUNG cua omega qua doan che (nguyen nhan goc)

CACH DUNG
    python src/diagnose_ukf.py            # chay ca 3 phan
    python src/diagnose_ukf.py --part a1  # chi chay 1 phan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402

WARMUP = 25          # so frame quan sat GT nap vao bo loc truoc doan che
MIN_SPEED = 2.0      # px/frame - duoi nguong nay la xe dung yen / nhieu annotation


def xyah(row) -> np.ndarray:
    """Quan sat dang (cx, cy, a, h) - dung khop voi interface cua ca 3 bo loc."""
    w, h = row.bb_width, row.bb_height
    return np.array([row.cx, row.cy, w / max(h, 1e-6), h], dtype=float)


# ===========================================================================
# A1 - Kiem tra xu ly goc theta trong UKF
# ===========================================================================
def part_a1() -> None:
    print("=" * 78)
    print("A1. XU LY BIEN GOC theta TRONG UKF")
    print("=" * 78)
    f = UKFTrackerCTRV()
    T, n = f.THETA, f.NDIM

    # --- (a) Trung binh vong khi gop sigma point ---
    sig = np.zeros((2 * n + 1, n))
    sig[:, T] = np.deg2rad([179, -179] * n + [180])
    mean = f._state_mean(sig, f.Wm)
    naive = f.Wm @ sig[:, T]
    ok_a = abs(abs(np.rad2deg(mean[T])) - 180) < 1.0
    print(f"\n(a) _state_mean(): trung binh vong cho theta")
    print(f"    sigma point trai quanh bien +-180 do")
    print(f"    trung binh CONG (cach sai) : {np.rad2deg(naive):+8.2f} do")
    print(f"    code tra ve                : {np.rad2deg(mean[T]):+8.2f} do")
    print(f"    => {'DUNG' if ok_a else 'SAI'}")

    # --- (b) Goi residual truoc khi lap hiep phuong sai ---
    m = np.zeros(n); m[T] = np.deg2rad(179)
    s = np.zeros((3, n)); s[:, T] = np.deg2rad([-179, 179, -170])
    d = f._state_residual(s, m)
    ok_b = np.all(np.abs(np.rad2deg(d[:, T])) <= 180.001)
    print(f"\n(b) _state_residual(): goi hieu theta ve [-pi, pi)")
    print(f"    hieu tho (chua goi) : {np.rad2deg(s[:, T] - m[T])} do")
    print(f"    code tra ve         : {np.rad2deg(d[:, T])} do")
    # Hau qua len phuong sai neu quen goi
    sig2 = np.zeros((2 * n + 1, n)); sig2[:, T] = np.deg2rad([179.5, -179.5] * n + [180.0])
    mn = f._state_mean(sig2, f.Wm)
    cov_ok = (f._state_residual(sig2, mn) * f.Wc[:, None]).T @ f._state_residual(sig2, mn)
    draw = sig2 - mn
    cov_bad = (draw * f.Wc[:, None]).T @ draw
    print(f"    var(theta) co goi   : {cov_ok[T, T]:.3e} rad^2")
    print(f"    var(theta) neu quen : {cov_bad[T, T]:.3e} rad^2 "
          f"(phong dai {cov_bad[T, T] / max(cov_ok[T, T], 1e-30):.2e} lan)")
    print(f"    => {'DUNG' if ok_b else 'SAI'}")

    # --- (c) Measurement co goc khong? ---
    names = ["CX", "CY", "V", "THETA", "OMEGA", "A", "H", "VA", "VH"]
    z_src = [names[int(np.argmax(f._update_mat[r]))] for r in range(f._update_mat.shape[0])]
    has_angle = bool(f._update_mat[:, T].any())
    print(f"\n(c) Measurement = {z_src}")
    print(f"    theta {'CO' if has_angle else 'KHONG'} nam trong measurement "
          f"=> {'phai wrap residual' if has_angle else 'BO QUA (c), khong can wrap'}")

    # --- Trong so sigma point: nguon goc hieu ung day cung ---
    print(f"\n[Trong so sigma point]  alpha={f.alpha} beta={f.beta} kappa={f.kappa} n={n}")
    print(f"    lambda = {f.lambda_}   n+lambda = {f._n_plus_lambda}")
    print(f"    Wm[0] (diem TRUNG TAM) = {f.Wm[0]:.6f}   <-- BANG 0")
    print(f"    Wm[i>=1] = {f.Wm[1]:.6f} (x{2*n})   sum(Wm) = {f.Wm.sum():.6f}")
    print("    => Voi lambda = 0, ky vong du doan KHONG dung f(x_mean) chut nao;")
    print("       no la trung binh 18 diem LECH tam, moi diem di mot cung ban kinh")
    print("       khac nhau -> trung binh roi VAO TRONG cung (hieu ung day cung).")

    # --- Do truc tiep hieu ung day cung ---
    print(f"\n[Hieu ung day cung] quy dao that = cung tron R = v/omega")
    ekf, ukf = EKFTrackerCTRV(), UKFTrackerCTRV()
    print(f"    {'P scale':>9} | {'R_EKF':>9} | {'R_UKF':>9} | {'R that':>8}")
    for ps in [0.01, 1.0, 100.0]:
        v, om = 8.0, np.deg2rad(3.0)
        x = np.zeros(n); x[ekf.CX], x[ekf.CY] = 500., 300.
        x[ekf.V], x[ekf.THETA], x[ekf.OMEGA] = v, 0.0, om
        x[ekf.A], x[ekf.H] = 0.5, 40.
        P = np.eye(n) * 1e-6
        P[ekf.V, ekf.V] = .5 * ps; P[T, T] = .01 * ps; P[ekf.OMEGA, ekf.OMEGA] = 1e-4 * ps
        R = {}
        for nm, filt in [("EKF", ekf), ("UKF", ukf)]:
            mm, cc = x.copy(), P.copy(); tr = []
            for _ in range(50):
                mm, cc = filt.predict(mm, cc); tr.append((mm[filt.CX], mm[filt.CY]))
            tr = np.array(tr)
            p1, p2, p3 = tr[0], tr[len(tr) // 2], tr[-1]
            dd = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
            if abs(dd) < 1e-12:
                R[nm] = np.inf; continue
            ux = ((p1@p1)*(p2[1]-p3[1]) + (p2@p2)*(p3[1]-p1[1]) + (p3@p3)*(p1[1]-p2[1])) / dd
            uy = ((p1@p1)*(p3[0]-p2[0]) + (p2@p2)*(p1[0]-p3[0]) + (p3@p3)*(p2[0]-p1[0])) / dd
            R[nm] = float(np.hypot(p1[0]-ux, p1[1]-uy))
        print(f"    {ps:>9} | {R['EKF']:>9.2f} | {R['UKF']:>9.2f} | {v/om:>8.2f}")
    print("    => EKF dung CHINH XAC moi P; UKF luon HUT, hut nang hon khi P lon.")


# ===========================================================================
# A2 / A3 - dung chung ham ngoai suy tren du lieu that
# ===========================================================================
def load_segments(min_len: int = 20, max_len: int = 90) -> pd.DataFrame:
    out = []
    for lvl, f in [("full", "full_occlusion_segments.csv"),
                   ("partial", "occlusion_segments.csv")]:
        d = pd.read_csv(config.INTERIM_DIR / f); d["lvl"] = lvl; out.append(d)
    seg = pd.concat(out, ignore_index=True)
    return seg[(seg.length_frames >= min_len) & (seg.length_frames <= max_len)]


def net_turn_deg(p: np.ndarray) -> float:
    """Goc giua huong TRUNG BINH 1/3 dau va 1/3 cuoi doan.

    Khong dung tong |dtheta| tich luy: tren xe gan nhu dung yen, nhieu annotation
    vai pixel moi frame cong don thanh hang tram do gia.
    """
    d = np.diff(p, axis=0)
    if len(d) < 4:
        return 0.0
    k = max(2, len(d) // 3)
    v1, v2 = d[:k].mean(0), d[-k:].mean(0)
    if np.hypot(*v1) < 1e-6 or np.hypot(*v2) < 1e-6:
        return 0.0
    a, b = np.arctan2(v1[1], v1[0]), np.arctan2(v2[1], v2[0])
    return float(np.rad2deg((b - a + np.pi) % (2 * np.pi) - np.pi))


def fit_filters(pre: pd.DataFrame, n_pred: int):
    """Nap quan sat GT truoc doan che, roi CHI predict n_pred buoc (khong update).

    Mo phong dung tinh huong bi che hoan toan: khong co detection moi de hieu chinh.
    """
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
    res = {}
    for name in ["KF + CV", "EKF + CTRV", "UKF + CTRV"]:
        if name.startswith("KF"):
            f = KalmanFilterXYAH(); pos = lambda m: (m[0], m[1]); ctrv = False
        else:
            f = EKFTrackerCTRV() if name.startswith("EKF") else UKFTrackerCTRV()
            pos = (lambda m, f=f: (m[f.CX], m[f.CY])); ctrv = True
        rows = list(pre.itertuples())
        m, c = f.initiate(xyah(rows[0])); inited = False
        for r in rows[1:]:
            z = xyah(r)
            if ctrv and not inited:
                m, c = f.initiate_from_motion(m, c, z, n_frames=1.0); inited = True
            m, c = f.predict(m, c)
            m, c = f.update(m, c, z)
        traj = []
        state_end = (m.copy(), c.copy())
        for _ in range(n_pred):
            m, c = f.predict(m, c)
            traj.append(pos(m))
        res[name] = dict(traj=np.array(traj), filt=f, state=state_end)
    return res


def part_a2(cases: list | None = None) -> None:
    print("\n" + "=" * 78)
    print("A2. NGOAI SUY THUAN TUY TREN TRACK THAT (che dai + xe dang re)")
    print("=" * 78)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}

    if cases is None:
        cases = [("MVI_63553", 16, 65, 94, "full >=90%"),
                 ("MVI_40172", 65, 1680, 1708, "partial"),
                 ("MVI_39861", 1, 169, 241, "partial")]

    COL = {"KF + CV": "tab:orange", "EKF + CTRV": "tab:blue", "UKF + CTRV": "tab:green"}
    MK = {"KF + CV": "o", "EKF + CTRV": "s", "UKF + CTRV": "^"}
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 6))
    rows = []
    for ax, (vid, tid, s, e, lvl) in zip(np.atleast_1d(axes), cases):
        t = g.get((vid, tid))
        pre = t[(t.frame < s) & (t.frame >= s - WARMUP)].sort_values("frame")
        occ = t[(t.frame >= s) & (t.frame <= e)].sort_values("frame")
        gtocc = np.c_[occ.cx.values, occ.cy.values]
        turn = net_turn_deg(gtocc)
        out = fit_filters(pre, len(occ))

        ax.plot(pre.cx, pre.cy, "k.-", lw=2, ms=4, label="GT truoc che (co detection)")
        ax.plot(gtocc[:, 0], gtocc[:, 1], "k--", lw=3.5, alpha=.9, label="GT TRONG che (that)")
        ax.plot(gtocc[0, 0], gtocc[0, 1], "k*", ms=22, zorder=6, label="bat dau bi che")
        for nm, r in out.items():
            tr = r["traj"]
            ax.plot(tr[:, 0], tr[:, 1], marker=MK[nm], color=COL[nm], ms=4, lw=2,
                    alpha=.9, label=nm)
            err_end = float(np.hypot(*(tr[-1] - gtocc[-1])))
            err_mean = float(np.mean([np.hypot(*(tr[i] - gtocc[i])) for i in range(len(tr))]))
            ax.annotate(f"{err_end:.0f}px", tr[-1], fontsize=10, color=COL[nm], weight="bold")
            rows.append(dict(ca=f"{vid} t{tid}", muc_che=lvl, model=nm,
                             sai_so_cuoi_px=round(err_end, 1),
                             sai_so_tb_px=round(err_mean, 1)))
        ax.set_title(f"{vid} track {tid} [{lvl}]\n{e-s+1} frame ({(e-s+1)/config.FPS:.1f}s), "
                     f"re {abs(turn):.1f} do trong doan che")
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.invert_yaxis(); ax.grid(alpha=.3); ax.legend(fontsize=7)

    fig.suptitle("Stage A2: ngoai suy THUAN TUY qua doan che khuat, xe dang re thuc su\n"
                 "(so px = sai so vi tri o frame cuoi doan che)", fontsize=12)
    fig.tight_layout()
    out_dir = config.RESULTS_DIR / "stratified"; out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "A2_extrapolation_real_tracks.png"
    fig.savefig(p, dpi=150)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["ca", "muc_che"], columns="model",
                         values="sai_so_cuoi_px").round(1).to_string())
    print(f"\n  Hinh -> {p}")


def part_a3() -> None:
    """Do do BEN VUNG cua omega: omega hoc truoc doan che co du bao dung khuc cua
    XAY RA TRONG doan che khong? Day la gia dinh cot loi cua CTRV."""
    print("\n" + "=" * 78)
    print("A3. OMEGA CO BEN VUNG QUA DOAN CHE KHUAT KHONG?")
    print("=" * 78)
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}
    rows = []
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
        rec = dict(video=r.video, track_id=r.track_id, lvl=r.lvl,
                   nf=r.length_frames, om_gt=turn / len(d))
        out = fit_filters(pre, 0)
        for nm, key in [("EKF + CTRV", "EKF"), ("UKF + CTRV", "UKF")]:
            f = out[nm]["filt"]; m = out[nm]["state"][0]
            rec[f"om_{key}"] = float(np.rad2deg(m[f.OMEGA]))
        rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"\nSo doan phan tich: {len(df)} "
          f"(full={int((df.lvl == 'full').sum())}, partial={int((df.lvl == 'partial').sum())})\n")
    for k in ["EKF", "UKF"]:
        same = np.sign(df[f"om_{k}"]) == np.sign(df.om_gt)
        print(f"  {k}: omega uoc luong DUNG DAU voi khuc cua that: "
              f"{int(same.sum())}/{len(df)} = {same.mean()*100:.1f}%")
    print("  (50% = ngau nhien nhu tung dong xu)")
    print(f"\n  |omega| trung vi (do/frame): GT={df.om_gt.abs().median():.3f}  "
          f"EKF={df.om_EKF.abs().median():.3f}  UKF={df.om_UKF.abs().median():.3f}")
    print("  => bo loc uoc luong turn rate LON GAP ~3 LAN thuc te, va sai dau gan mot nua")
    out = config.RESULTS_DIR / "stratified" / "A3_omega_persistence.csv"
    df.to_csv(out, index=False)
    print(f"\n  -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage A: chan doan UKF")
    ap.add_argument("--part", choices=["a1", "a2", "a3", "all"], default="all")
    a = ap.parse_args()
    if a.part in ("a1", "all"):
        part_a1()
    if a.part in ("a2", "all"):
        part_a2()
    if a.part in ("a3", "all"):
        part_a3()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
