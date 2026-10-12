"""
mo_phong_quy_dao.py -- R48C: bo sinh quy dao GT + phep do co nhieu / mat do, seed tai lap
(giao thuc handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md muc 4.1).

Trang thai tren mat phang anh, 1 buoc = 1 frame (dt = 1, src/time_units.py), T = 90 buoc. Box GT (cx, cy, a = w/h, h).
Kich ban (omega: rad/buoc):
    thang           omega = 0                                        dung CTRV
    cua_deu         omega = 0.03 s                                   dung CTRV
    doi_omega       0 -> doc len 0.06 s (t 30..40) -> giu -> ve 0 (t 60..70)   CTRV tung doan, omega doi
    cua_gat         omega = 0.10 s voi 30 <= t < 50                  CTRV tung doan, omega nhay bac
    phoi_canh       thang deu tren mat dat, chieu pinhole (f 800, cam 6 m, xe 1.5 m), Z = Z0 (1 - 0.5 t / T)   SAI
    gia_toc         v = 4 + 0.05 t, omega = 0.03 s                    SAI (v khong hang)
    doi_lan         theta = theta0 + 0.35 sin(2 pi t / 40)            SAI (omega bien thien lien tuc)
    nhieu_duoi_day  chuyen dong nhu cua_deu; nhieu do Student-t nu = 3 + 5 % outlier +-0.5 h   SAI (likelihood)

Phep do: z_t = GT_t + kappa * sigma(h_t) * eps_t, sigma = (h/20, h/20, 0.1, h/20) = dang R cua ma (conf = None).
Mat do: L buoc lien tiep tu t_gap (30 cho doi_omega / cua_gat, 40 cho con lai); hang do bi mat = NaN.

SO NGAU NHIEN CHUNG: seed nhieu d sinh theta0, dau quay s, eps, outlier bang SeedSequence([d, ma kich ban]) - KHONG phu
thuoc L, kappa, h0 hay bo loc -> moi o va moi bo loc cung d nhan cung quy dao + cung nhieu chuan hoa (ghep cap).

    python src/kc48/mo_phong_quy_dao.py --tu-kiem       # tu kiem bo sinh -> results/phase2/kc48_pf_code/mo_phong_tu_kiem.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

KICH_BAN = ("thang", "cua_deu", "doi_omega", "cua_gat", "phoi_canh", "gia_toc", "doi_lan", "nhieu_duoi_day")
DUNG_CTRV = {"thang": True, "cua_deu": True, "doi_omega": False, "cua_gat": False,
             "phoi_canh": False, "gia_toc": False, "doi_lan": False, "nhieu_duoi_day": True}  # chuyen dong
L_GAP = (0, 5, 10, 20)
KAPPA = (0.5, 1.0, 2.0)
H0 = (24.0, 72.0)
SEED_DANH_GIA = tuple(range(30))
SEED_TINH_CHINH = tuple(range(1000, 1010))
T_MAC_DINH = 90
CX0, CY0, V0, ASPECT = 480.0, 270.0, 4.0, 1.6
NU_T, P_OUTLIER, OUTLIER_H = 3.0, 0.05, 0.5


def t_gap_cua(kich_ban: str) -> int:
    return 30 if kich_ban in ("doi_omega", "cua_gat") else 40


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _omega_cua(kich_ban: str, s: float, T: int) -> np.ndarray:
    t = np.arange(T, dtype=np.float64)
    if kich_ban in ("thang", "phoi_canh"):
        om = np.zeros(T)
    elif kich_ban in ("cua_deu", "gia_toc", "nhieu_duoi_day"):
        om = np.full(T, 0.03)
    elif kich_ban == "doi_omega":
        om = np.interp(t, [0, 30, 40, 60, 70, T], [0, 0, 0.06, 0.06, 0, 0])
    elif kich_ban == "cua_gat":
        om = np.where((t >= 30) & (t < 50), 0.10, 0.0)
    else:
        raise ValueError(kich_ban)
    return om * s


def _buoc_ctrv(cx, cy, v, th, om):
    """Mot buoc CTRV chinh xac (dt = 1), cung cong thuc EKFTrackerCTRV.f."""
    if abs(om) > 1e-4:
        return cx + v / om * (np.sin(th + om) - np.sin(th)), cy + v / om * (np.cos(th) - np.cos(th + om)), th + om
    return cx + v * np.cos(th), cy + v * np.sin(th), th


def _rng_chung(d: int, kich_ban: str, T: int) -> dict:
    """Moi so ngau nhien cua mot (d, kich ban), rut theo THU TU CO DINH cho moi kich ban."""
    rng = np.random.default_rng(np.random.SeedSequence([int(d), KICH_BAN.index(kich_ban)]))
    return {"theta0": float(rng.uniform(-np.pi, np.pi)), "s": float(rng.choice([-1.0, 1.0])),
            "eps": rng.standard_normal((T, 4)), "chi2": rng.chisquare(NU_T, size=(T, 4)),
            "u_out": rng.random(T), "dau_out": rng.choice([-1.0, 1.0], size=(T, 2))}


def sinh_quy_dao(kich_ban: str, d: int, h0: float, T: int = T_MAC_DINH) -> dict:
    """GT: box (T, 4) = (cx, cy, a, h) va trang thai CTRV (T, 9) = (cx, cy, v, theta, omega, a, h, va, vh)."""
    r = _rng_chung(d, kich_ban, T)
    th0, s = r["theta0"], r["s"]
    X = np.zeros((T, 9))
    if kich_ban == "phoi_canh":
        f, cam_h, xe_h = 800.0, 6.0, 1.5
        Z0 = f * xe_h / h0
        g = V0 * Z0 / f * s                                   # toc do ngang ban dau tren anh = V0 px/buoc
        t = np.arange(T + 1, dtype=np.float64)
        Z = Z0 * (1 - 0.5 * t / T)
        Xg = g * t
        h = f * xe_h / Z
        cx = CX0 + f * Xg / Z
        cy = CY0 + f * cam_h / Z - h / 2                      # tam box = day box - h/2
        vx, vy = np.diff(cx), np.diff(cy)
        X[:, 0], X[:, 1], X[:, 6] = cx[:T], cy[:T], h[:T]
        X[:, 2] = np.hypot(vx, vy)
        X[:, 3] = np.arctan2(vy, vx)
        X[:, 4] = _wrap(np.diff(np.append(X[:, 3], np.arctan2(vy[-1], vx[-1]))))
        X[:, 8] = np.diff(h)
    elif kich_ban == "doi_lan":
        t = np.arange(T, dtype=np.float64)
        th = th0 + 0.35 * np.sin(2 * np.pi * t / 40)
        cx, cy = CX0, CY0
        n_con = 20                                            # tich phan buoc con cho quy dao lien tuc
        for k in range(T):
            X[k, 0], X[k, 1], X[k, 3] = cx, cy, _wrap(th[k])
            tau = k + (np.arange(n_con) + 0.5) / n_con
            th_con = th0 + 0.35 * np.sin(2 * np.pi * tau / 40)
            cx += V0 * np.mean(np.cos(th_con))
            cy += V0 * np.mean(np.sin(th_con))
        X[:, 2] = V0
        X[:, 4] = np.append(_wrap(np.diff(th)), 0.35 * 2 * np.pi / 40 * np.cos(2 * np.pi * (T - 1) / 40))
        X[:, 6] = h0
    else:
        om = _omega_cua(kich_ban, s, T)
        v = V0 + 0.05 * np.arange(T) if kich_ban == "gia_toc" else np.full(T, V0)
        cx, cy, th = CX0, CY0, th0
        for k in range(T):
            X[k, 0], X[k, 1], X[k, 2], X[k, 3], X[k, 4] = cx, cy, v[k], _wrap(th), om[k]
            cx, cy, th = _buoc_ctrv(cx, cy, v[k], th, om[k])
        X[:, 6] = h0
    X[:, 5] = ASPECT
    box = X[:, [0, 1, 5, 6]].copy()
    return {"kich_ban": kich_ban, "d": int(d), "h0": float(h0), "T": T, "trang_thai": X, "box": box,
            "theta0": th0, "s": s}


def sinh_do(qd: dict, kappa: float, L: int) -> dict:
    """Phep do (T, 4) co NaN o buoc mat do, mask co_do (T,), nhieu chuan hoa eps (T, 4)."""
    kb, T = qd["kich_ban"], qd["T"]
    r = _rng_chung(qd["d"], kb, T)
    box = qd["box"]
    h = box[:, 3]
    sigma = np.stack([h / 20, h / 20, np.full(T, 0.1), h / 20], axis=1)
    if kb == "nhieu_duoi_day":
        eps = r["eps"] / np.sqrt(r["chi2"] / NU_T) / np.sqrt(NU_T / (NU_T - 2))   # Student-t, phuong sai 1
    else:
        eps = r["eps"]
    z = box + kappa * sigma * eps
    out = np.zeros(T, dtype=bool)
    if kb == "nhieu_duoi_day":
        out = r["u_out"] < P_OUTLIER
        z[out, 0] += r["dau_out"][out, 0] * OUTLIER_H * h[out]
        z[out, 1] += r["dau_out"][out, 1] * OUTLIER_H * h[out]
    co_do = np.ones(T, dtype=bool)
    tg = t_gap_cua(kb)
    co_do[tg:tg + int(L)] = False
    z[~co_do] = np.nan
    return {"z": z, "co_do": co_do, "eps": eps, "outlier": out, "t_gap": tg, "t_r": tg + int(L),
            "kappa": float(kappa), "L": int(L)}


def o_luoi(seeds=SEED_DANH_GIA):
    """Lien ke moi o (kich ban, L, kappa, h0, d) cua luoi day du."""
    for kb in KICH_BAN:
        for L in L_GAP:
            for kappa in KAPPA:
                for h0 in H0:
                    for d in seeds:
                        yield kb, L, kappa, h0, d


# ----------------------------------------------------------------------------------------------------------
def tu_kiem() -> dict:
    from ekf_ctrv import EKFTrackerCTRV
    ekf = EKFTrackerCTRV()
    kq: dict = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kiem": {}}

    def ghi(ten, ok, chi_tiet=None):
        kq["kiem"][ten] = {"dat": bool(ok), "chi_tiet": chi_tiet}
        print(f"  [{'DAT ' if ok else 'HONG'}] {ten}" + (f"  {chi_tiet}" if chi_tiet is not None else ""))

    # 1. tai lap theo seed + 2. seed khac -> khac
    a = sinh_do(sinh_quy_dao("cua_gat", 3, 24.0), 1.0, 10)
    b = sinh_do(sinh_quy_dao("cua_gat", 3, 24.0), 1.0, 10)
    c = sinh_do(sinh_quy_dao("cua_gat", 4, 24.0), 1.0, 10)
    ghi("cung (kich ban, d) -> z bit-identical", np.array_equal(a["z"], b["z"], equal_nan=True))
    ghi("d khac -> z khac", not np.array_equal(a["z"], c["z"], equal_nan=True))
    # 3. so ngau nhien chung qua L, kappa, h0
    ok_cap, ok_qd = True, True
    for kb in KICH_BAN:
        ref = sinh_do(sinh_quy_dao(kb, 7, 24.0), 1.0, 0)
        for h0 in H0:
            qd = sinh_quy_dao(kb, 7, h0)
            for kappa in KAPPA:
                for L in L_GAP:
                    m = sinh_do(qd, kappa, L)
                    ok_cap &= np.array_equal(m["eps"], ref["eps"]) and np.array_equal(m["outlier"], ref["outlier"])
                    keep = m["co_do"]
                    exp_z = qd["box"] + kappa * np.stack([qd["box"][:, 3] / 20, qd["box"][:, 3] / 20,
                                                          np.full(qd["T"], 0.1), qd["box"][:, 3] / 20], 1) * m["eps"]
                    if kb == "nhieu_duoi_day":
                        o = m["outlier"]
                        exp_z[o, :2] += ref_outlier_shift(kb, qd)[o]
                    ok_qd &= np.allclose(m["z"][keep], exp_z[keep], rtol=0, atol=1e-9)
        if kb != "phoi_canh":
            q24, q72 = sinh_quy_dao(kb, 7, 24.0), sinh_quy_dao(kb, 7, 72.0)
            ok_qd &= np.allclose(q24["trang_thai"][:, :5], q72["trang_thai"][:, :5])   # cung quy dao tam, khac h
    ghi("eps / outlier trung nhau qua moi L, kappa, h0 cung d (ghep cap)", ok_cap)
    ghi("z = GT + kappa*sigma(h)*eps (+ outlier) dung cong thuc; quy dao tam khong phu thuoc h0 (tru phoi_canh)", ok_qd)
    # 4. mask mat do
    ok_mask = True
    for kb in KICH_BAN:
        for L in L_GAP:
            m = sinh_do(sinh_quy_dao(kb, 1, 24.0), 1.0, L)
            idx = np.flatnonzero(~m["co_do"])
            ok_mask &= len(idx) == L and (L == 0 or (idx[0] == m["t_gap"] and idx[-1] == m["t_gap"] + L - 1))
            ok_mask &= bool(np.isnan(m["z"][~m["co_do"]]).all() and np.isfinite(m["z"][m["co_do"]]).all())
    ghi("mat do: dung L buoc tu t_gap, hang mat la NaN, con lai huu han", ok_mask)
    # 5. dung / sai CTRV: du doan 1 buoc bang EKFTrackerCTRV.f tu trang thai THAT.
    #    r_vt = sai so vi tri (px); r_ts = thay doi tham so CTRV giua hai buoc (|dv| + |d omega|): CTRV hang so -> 0.
    resid = {}
    for kb in KICH_BAN:
        r_vt = r_ts = 0.0
        for d in range(5):
            X = sinh_quy_dao(kb, d, 24.0)["trang_thai"]
            for k in range(X.shape[0] - 1):
                pred = ekf.f(X[k].copy())
                r_vt = max(r_vt, float(np.hypot(pred[0] - X[k + 1, 0], pred[1] - X[k + 1, 1])))
                r_ts = max(r_ts, float(abs(X[k + 1, 2] - X[k, 2]) + abs(_wrap(X[k + 1, 4] - X[k, 4]))))
        resid[kb] = {"vi_tri_px": float(f"{r_vt:.3g}"), "tham_so": float(f"{r_ts:.3g}")}
    hang = ("thang", "cua_deu", "nhieu_duoi_day")
    ok_ctrv = all(resid[k]["vi_tri_px"] < 1e-9 and resid[k]["tham_so"] < 1e-12 for k in hang)
    ok_sai = all(resid[k]["vi_tri_px"] + resid[k]["tham_so"] > 1e-3
                 for k in ("doi_omega", "cua_gat", "phoi_canh", "gia_toc", "doi_lan"))
    ghi("thang / cua_deu / nhieu_duoi_day: GT = CTRV tham so hang (vi tri < 1e-9 px, v / omega khong doi)", ok_ctrv, resid)
    ghi("doi_omega / cua_gat / phoi_canh / gia_toc / doi_lan lech CTRV hang so (vi tri hoac v / omega doi > 1e-3)", ok_sai)
    # 6. huu han, h > 0, do lech chuan nhieu chuan hoa ~ 1
    allf, epsg, epst = True, [], []
    for kb in KICH_BAN:
        for d in SEED_DANH_GIA:
            for h0 in H0:
                qd = sinh_quy_dao(kb, d, h0)
                allf &= bool(np.isfinite(qd["trang_thai"]).all() and (qd["box"][:, 3] > 0).all())
            m = sinh_do(sinh_quy_dao(kb, d, 24.0), 1.0, 0)
            (epst if kb == "nhieu_duoi_day" else epsg).append(m["eps"])
    sg, st = float(np.std(np.concatenate(epsg))), float(np.std(np.concatenate(epst)))
    ghi("moi trang thai GT huu han, h > 0 (8 kich ban x 30 seed x 2 h0)", allf)
    ghi("do lech chuan eps Gauss ~ 1 (|sd - 1| < 0.05)", abs(sg - 1) < 0.05, round(sg, 4))
    ghi("do lech chuan eps Student-t nu=3 (truoc outlier) trong [0.8, 1.2] (duoi day: hoi tu cham)", 0.8 < st < 1.2, round(st, 4))
    kq["dat_tat_ca"] = all(v["dat"] for v in kq["kiem"].values())
    kq["tham_so"] = {"KICH_BAN": KICH_BAN, "L_GAP": L_GAP, "KAPPA": KAPPA, "H0": H0, "T": T_MAC_DINH,
                     "SEED_DANH_GIA": [SEED_DANH_GIA[0], SEED_DANH_GIA[-1]],
                     "SEED_TINH_CHINH": [SEED_TINH_CHINH[0], SEED_TINH_CHINH[-1]]}
    return kq


def ref_outlier_shift(kb: str, qd: dict) -> np.ndarray:
    r = _rng_chung(qd["d"], kb, qd["T"])
    return r["dau_out"] * OUTLIER_H * qd["box"][:, 3:4]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tu-kiem", action="store_true")
    a = ap.parse_args(argv)
    if not a.tu_kiem:
        ap.print_help()
        return 0
    kq = tu_kiem()
    out = ROOT / "results/phase2/kc48_pf_code/mo_phong_tu_kiem.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT).as_posix()}  dat_tat_ca={kq['dat_tat_ca']}")
    return 0 if kq["dat_tat_ca"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
