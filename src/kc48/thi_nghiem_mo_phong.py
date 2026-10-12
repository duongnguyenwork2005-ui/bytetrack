"""
thi_nghiem_mo_phong.py -- R48D: thi nghiem kiem chung co che tren mo phong biet lien ket
(giao thuc handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md muc 4, 6, 8, 9).

Bo loc: EKF-CTRV, PF-CTRV hien tai (bootstrap), PF + K1 (proposal toi uu), PF + K2 (roughening), PF + K1 + K2.
Goi DUNG API ma that (EKFTrackerCTRV / PFCTRV + PFTrackView), cung thu tu nhu track trong ByteTrack (04 muc 4.2):
    t = 0 initiate; moi t >= 1: (buoc truoc mat do -> vh = 0) -> predict -> ghi box du doan -> neu co do:
    try_initiate_from_motion (lan dau thanh cong) -> update.  Lien ket luon dung.

Giai doan (moi lenh < 10 phut, chay lai thi bo qua don vi da xong):
    --giai-doan tinh-chinh   K2 c in {0.25, 0.5, 1.0}, EKF nhan (sigma_theta, sigma_omega) in {0.5, 1, 2} tren seed
                             tinh chinh d = 1000..1009, L in {10, 20}, kappa 1, ca hai h0 -> tinh_chinh/*.csv, chon.json
    --giai-doan danh-gia     luoi day du d = 0..29 cho bien the N = 100 (+ EKF); duong tai nguyen N = 300 / 500 o
                             kappa = 1 -> danh_gia/*.csv (moi don vi = bien the x N x kich ban)
    --giai-doan phan-tich    C1..C4 + do nhay, bootstrap ghep cap theo d (B = 5000, seed 0) -> phan_tich/
    --giai-doan hinh-hat     minh hoa hat to mau theo track so huu: xe nho canh / bi xe lon che (mo phong)
    --toi-da-giay S          khong bat dau don vi moi sau S giay (tiep tuc o lenh sau)

    python src/kc48/thi_nghiem_mo_phong.py --giai-doan tinh-chinh
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
sys.path.insert(0, str(ROOT / "src" / "kc48"))
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView  # noqa: E402
import mo_phong_quy_dao as mp  # noqa: E402

OUT = ROOT / "results/phase2/kc48_mo_phong"
CX, CY, V, TH, OM, A, H, VA, VH = range(9)
NHOM = {"thang": "dung", "cua_deu": "dung", "doi_omega": "tung_doan", "cua_gat": "tung_doan",
        "phoi_canh": "lech", "gia_toc": "lech", "doi_lan": "lech", "nhieu_duoi_day": "lech"}
NGUONG_F = 0.2 / 0.9
K2_UNG_VIEN = (0.25, 0.5, 1.0)
EKF_UNG_VIEN = (0.5, 1.0, 2.0)
SEED_PF = (0, 1, 2)
_EKF0 = EKFTrackerCTRV()
STD_THETA0, STD_OMEGA0 = float(_EKF0._std_theta), float(_EKF0._std_omega)


def _iou(b1, b2) -> float:
    """IoU hai box (cx, cy, a, h)."""
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(b2))):
        return 0.0
    w1, w2 = b1[2] * b1[3], b2[2] * b2[3]
    x1, x2 = max(b1[0] - w1 / 2, b2[0] - w2 / 2), min(b1[0] + w1 / 2, b2[0] + w2 / 2)
    y1, y2 = max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2), min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = w1 * b1[3] + w2 * b2[3] - inter
    return float(inter / u) if u > 0 else 0.0


def tao_bo_loc(bt: dict, N: int = 100):
    if bt["loai"] == "ekf":
        return EKFTrackerCTRV(std_theta=STD_THETA0 * bt["mult"], std_omega=STD_OMEGA0 * bt["mult"])
    pf = PFCTRV(num_particles=N, roughening=bt["rough"], optimal_proposal=bt["k1"])
    pf.enable_diag()
    return pf


def chay_mot(bl, qd: dict, do: dict, p_seed: int = 0, chuoi: list | None = None) -> dict:
    """Mot quy dao, mot bo loc. Tra ve chi so (04 muc 4.3)."""
    T, box, Xgt = qd["T"], qd["box"], qd["trang_thai"]
    z_all, co_do, tg, tr, L = do["z"], do["co_do"], do["t_gap"], do["t_r"], do["L"]
    la_pf = isinstance(bl, PFCTRV)
    mean, cov = bl.initiate(z_all[0])
    if la_pf:
        bl.set_seed(p_seed, video=f"{qd['kich_ban']}|{qd['d']}")
        bl.diag_rows = []
        trk = type("T", (), {})()
        trk.track_id, trk.frame_id, trk._pending_frame_id = 1, 0, 0
        trk.rng = bl.make_rng(1)
        trk.particles = bl.sample_initial(mean, cov, trk.rng)
        trk.logw = np.full(bl.num_particles, -np.log(bl.num_particles))
        trk._k1_prior = None
        view = PFTrackView(bl, trk)
    last_obs, last_t, motion_ok, lost = (float(z_all[0, 0]), float(z_all[0, 1])), 0, False, False
    pred = np.full((T, 4), np.nan)
    loc_err, ess_bam, rs_bam, tt_bam = [], [], [], []
    n_upd_bam = n_reinit_bam = 0
    out = {"ess_tr": np.nan, "tt_tr": np.nan, "rs_tr": np.nan, "deg_tr": np.nan, "reinit_tr": np.nan, "sxy_tr": np.nan,
           "sth_tr": np.nan, "sv_tr": np.nan, "nis_tr": np.nan, "err_th": np.nan, "err_om": np.nan, "co_tb": np.nan}
    x_truoc_gap = None
    t_tong = 0.0
    for t in range(1, T):
        t0 = time.perf_counter()
        m_in = mean.copy()
        if lost:
            m_in[VH] = 0.0
        if la_pf:
            trk._pending_frame_id = t
            mean, cov = view.predict(m_in, cov, lost=lost)
        else:
            mean, cov = bl.predict(m_in, cov)
        t_tong += time.perf_counter() - t0
        pred[t] = mean[[CX, CY, A, H]]
        if t == tr:
            hgt = box[t, 3]
            if la_pf:
                sxy, sth, sv = bl._diag_spread(trk.particles, trk.logw)
                out.update(sxy_tr=sxy / hgt, sth_tr=sth, sv_tr=sv)
            else:
                out.update(sxy_tr=float(np.sqrt(cov[CX, CX] + cov[CY, CY])) / hgt, sth_tr=float(np.sqrt(cov[TH, TH])),
                           sv_tr=float(np.sqrt(cov[V, V])))
            if x_truoc_gap is not None:                       # H6: co trung binh so voi f(x_hat) truoc khoang mat
                xf = x_truoc_gap[None, :].copy()
                for k in range(tg, tr + 1):
                    if k > tg:
                        xf[0, VH] = 0.0
                    xf = PFCTRV.f_batch(_EKF_F, xf)
                out["co_tb"] = float(np.hypot(*(mean[[CX, CY]] - xf[0, [CX, CY]]))) / hgt
            if co_do[t]:
                pm, S = bl.project(mean, cov)
                nu = z_all[t] - pm
                try:
                    out["nis_tr"] = float(nu @ np.linalg.solve(S, nu))
                except np.linalg.LinAlgError:
                    out["nis_tr"] = np.inf
        if co_do[t]:
            z = z_all[t]
            t0 = time.perf_counter()
            if la_pf:
                trk.frame_id = last_t
            if not motion_ok:
                tgt = view if la_pf else bl
                mean, cov, motion_ok = tgt.try_initiate_from_motion(mean, cov, z, n_frames=float(t - last_t),
                                                                    ref_pos=last_obs)
                motion_ok = bool(motion_ok)
            if la_pf:
                mean, cov = view.update(mean, cov, z)
                row = bl.diag_rows[-1]
                # Vong sua F2: nhanh tai khoi tao (row[9]) ghi ess 0 / resampled True theo QUY UOC -> khong phai ESS
                # that, khong phai systematic resampling: dem rieng, khong dua vao ESS / ty le resample.
                tai_kt = bool(row[9])
                idx = bl.last_resample_idx
                tt = len(np.unique(idx)) / bl.num_particles if idx is not None else np.nan
                if 10 <= t < tg:
                    n_upd_bam += 1
                    if tai_kt:
                        n_reinit_bam += 1
                    else:
                        ess_bam.append(row[3]); rs_bam.append(bool(row[4]))
                        if idx is not None:
                            tt_bam.append(tt)
                if t == tr:
                    out.update(ess_tr=np.nan if tai_kt else row[3], rs_tr=np.nan if tai_kt else float(row[4]),
                               tt_tr=tt, deg_tr=float(tai_kt), reinit_tr=float(tai_kt))
            else:
                mean, cov = bl.update(mean, cov, z)
            t_tong += time.perf_counter() - t0
            last_obs, last_t, lost = (float(z[0]), float(z[1])), t, False
            if t >= 10:
                loc_err.append(np.hypot(*(mean[[CX, CY]] - box[t, :2])) / box[t, 3])
            if t == tg - 1:
                x_truoc_gap = mean.copy()
                if Xgt[t, V] >= 1.0:
                    out["err_th"] = float(abs(normalize_angle(mean[TH] - Xgt[t, TH])))
                    out["err_om"] = float(mean[OM] - Xgt[t, OM])
        else:
            lost = True
        if chuoi is not None:
            r = bl.diag_rows[-1] if (la_pf and co_do[t] and bl.diag_rows) else None
            kt = bool(r[9]) if r is not None else False
            chuoi.append({"t": t, "co_do": bool(co_do[t]),
                          "e_pred_h": float(np.hypot(*(pred[t, :2] - box[t, :2])) / box[t, 3]),
                          "ess": (r[3] if (r is not None and not kt) else np.nan),
                          "resample": (bool(r[4]) if (r is not None and not kt) else None),
                          "tai_khoi_tao": (kt if r is not None else None),
                          "to_tien": (len(np.unique(bl.last_resample_idx)) / bl.num_particles
                                      if (r is not None and bl.last_resample_idx is not None) else np.nan),
                          "sxy_h": (bl._diag_spread(trk.particles, trk.logw)[0] / box[t, 3] if la_pf else
                                    float(np.sqrt(cov[CX, CX] + cov[CY, CY])) / box[t, 3])})
    hgt = box[tr, 3]
    if L == 0:
        e = [np.hypot(*(pred[k, :2] - box[k, :2])) / box[k, 3] for k in range(tg, min(tg + 20, T))]
        e_r = float(np.mean(e))
    else:
        e_r = float(np.hypot(*(pred[tr, :2] - box[tr, :2])) / hgt)
    iou = _iou(pred[tr], box[tr])
    huu_han = bool(np.all(np.isfinite(pred[1:])))
    out.update(e_r=e_r if np.isfinite(e_r) else np.inf, iou_tr=iou, F=float((iou < NGUONG_F) or not huu_han),
               F05=float((iou < 0.5) or not huu_han), rmse_h=float(np.sqrt(np.mean(np.square(loc_err)))),
               ess_bam=float(np.mean(ess_bam)) if ess_bam else np.nan,
               rs_bam=float(np.mean(rs_bam)) if rs_bam else np.nan,
               tt_bam=float(np.mean(tt_bam)) if tt_bam else np.nan,
               # n_deg: so lan tai khoi tao cua RIENG quy dao nay (diag_rows dat lai moi quy dao), khong phai bo dem
               # tich luy PFCTRV.n_degenerate; n_update: so update co do (mau so cua ty le tai khoi tao)
               n_deg=float(sum(1 for r in bl.diag_rows if r[2] == "update" and r[9])) if la_pf else np.nan,
               n_update=float(sum(1 for r in bl.diag_rows if r[2] == "update")) if la_pf else np.nan,
               n_upd_bam=float(n_upd_bam) if la_pf else np.nan, n_reinit_bam=float(n_reinit_bam) if la_pf else np.nan,
               us_buoc=1e6 * t_tong / (T - 1))
    return out


_EKF_F = PFCTRV(num_particles=2)      # chi dung f_batch (cung f voi moi bo loc)


def bien_the(chon: dict | None) -> dict:
    c = (chon or {}).get("k2_c")
    m = (chon or {}).get("ekf_mult")
    d = {"ekf": {"loai": "ekf", "mult": 1.0},
         "pf": {"loai": "pf", "k1": False, "rough": 0.0},
         "pf-k1": {"loai": "pf", "k1": True, "rough": 0.0}}
    if c is not None:
        d["pf-k2"] = {"loai": "pf", "k1": False, "rough": float(c)}
        d["pf-k1k2"] = {"loai": "pf", "k1": True, "rough": float(c)}
    if m is not None and m != 1.0:
        d["ekf-chinh"] = {"loai": "ekf", "mult": float(m)}
    return d


COT = ["bien_the", "N", "p", "kich_ban", "L", "kappa", "h0", "d", "e_r", "iou_tr", "F", "F05", "rmse_h", "err_th",
       "err_om", "nis_tr", "ess_tr", "rs_tr", "tt_tr", "deg_tr", "sxy_tr", "sth_tr", "sv_tr", "co_tb", "ess_bam", "rs_bam",
       "tt_bam", "n_deg", "us_buoc"]
# Vong sua F2: cot moi (harness da sua); COT giu nguyen de doc / doi chieu ban cu
COT_V2 = COT + ["reinit_tr", "n_update", "n_upd_bam", "n_reinit_bam"]
CHAN_DOAN_CAP_HOP_LE = ("ess_tr", "rs_bam", "ess_bam", "tt_bam")


def chay_don_vi(ten: str, bt: dict, N: int, kb: str, Ls, kappas, seeds, dich: Path) -> float:
    import pandas as pd
    t0 = time.perf_counter()
    bl = tao_bo_loc(bt, N)
    rows = []
    ps = (0,) if bt["loai"] == "ekf" else SEED_PF
    for d in seeds:
        for h0 in mp.H0:
            qd = mp.sinh_quy_dao(kb, d, h0)
            for L in Ls:
                for kappa in kappas:
                    do = mp.sinh_do(qd, kappa, L)
                    for p in ps:
                        r = chay_mot(bl, qd, do, p)
                        rows.append({"bien_the": ten, "N": N if bt["loai"] == "pf" else 0, "p": p, "kich_ban": kb,
                                     "L": L, "kappa": kappa, "h0": h0, "d": d, **r})
    df = pd.DataFrame(rows)[COT_V2]
    dich.parent.mkdir(parents=True, exist_ok=True)
    tmp = dich.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dich)
    return time.perf_counter() - t0


def _ghi_nhat_ky(muc: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "nhat_ky.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(muc, ensure_ascii=False) + "\n")


def _so_tien_trinh() -> int:
    import os
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def _chay_ds(ds, toi_da: float, giai_doan: str) -> int:
    """Chay cac don vi chua co file, song song nhieu tien trinh (chi so khong phu thuoc thu tu / tien trinh: moi
    don vi tu tao bo loc + seed). Thoi gian `us_buoc` o day do DUOI TAI song song -> chi dung cho chi so; thoi gian
    tai nguyen (C4) lay tu giai doan do-toc-do (mot tien trinh, 04 muc 4.3). Khong nop don vi moi sau `toi_da` giay."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    cho = [u for u in ds if not u[-1].exists()]
    t_bat_dau = time.perf_counter()
    nw = _so_tien_trinh()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        dang = {}
        while cho or dang:
            while cho and len(dang) < nw and time.perf_counter() - t_bat_dau <= toi_da:
                u = cho.pop(0)
                dang[ex.submit(chay_don_vi, *u)] = u
            if not dang:
                break
            xong, _ = wait(dang, return_when=FIRST_COMPLETED)
            for fu in xong:
                u = dang.pop(fu)
                s = fu.result()
                _ghi_nhat_ky({"giai_doan": giai_doan, "don_vi": u[-1].name, "giay": round(s, 2), "tien_trinh": nw,
                              "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                print(f"  {u[-1].name}: {s:.1f} s", flush=True)
    print(f"{nw} tien trinh; {time.perf_counter() - t_bat_dau:.0f} s; con {len(cho)} don vi chua chay")
    return len(cho)


def giai_doan_tinh_chinh(toi_da: float) -> int:
    import pandas as pd
    ds = []
    for c in K2_UNG_VIEN:
        for kb in mp.KICH_BAN:
            ds.append((f"pf-k2_c{c}", {"loai": "pf", "k1": False, "rough": c}, 100, kb, (10, 20), (1.0,),
                       mp.SEED_TINH_CHINH, OUT / f"tinh_chinh/pf-k2_c{c}_{kb}.csv"))
    for m in EKF_UNG_VIEN:
        for kb in mp.KICH_BAN:
            ds.append((f"ekf_m{m}", {"loai": "ekf", "mult": m}, 0, kb, (10, 20), (1.0,), mp.SEED_TINH_CHINH,
                       OUT / f"tinh_chinh/ekf_m{m}_{kb}.csv"))
    if _chay_ds(ds, toi_da, "tinh-chinh"):
        return 3
    df = pd.concat([pd.read_csv(p) for *_, p in ds], ignore_index=True)

    def chon(prefix, ung_vien, ten_fn):
        bang = []
        for u in ung_vien:
            s = df[df.bien_the == ten_fn(u)]
            bang.append({"gia_tri": u, "F_tb": float(s.F.mean()), "e_r_tb": float(s.e_r.mean()), "n_dong": len(s)})
        tot = min(bang, key=lambda r: r["F_tb"])
        hoa = [r for r in bang if r["F_tb"] - tot["F_tb"] < 0.005]
        if len(hoa) > 1:
            tot_e = min(hoa, key=lambda r: r["e_r_tb"])
            hoa = [r for r in hoa if r["e_r_tb"] == tot_e["e_r_tb"]]
        return min(r["gia_tri"] for r in hoa), bang

    c, bang_k2 = chon("pf-k2", K2_UNG_VIEN, lambda u: f"pf-k2_c{u}")
    m, bang_ekf = chon("ekf", EKF_UNG_VIEN, lambda u: f"ekf_m{u}")
    kq = {"quy_tac": "F (IoU < 0.2/0.9) trung binh nho nhat tren moi kich ban, L in {10, 20}, kappa 1, ca hai h0, "
                     "d 1000-1009, 3 seed PF; chenh < 0.005 -> e_r; van hoa -> gia tri nho hon (04 muc 8)",
          "k2_c": c, "bang_k2": bang_k2, "ekf_mult": m, "bang_ekf": bang_ekf,
          "std_theta0": STD_THETA0, "std_omega0": STD_OMEGA0, "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "chon.json").write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(kq, ensure_ascii=False, indent=2))
    return 0


def _ds_danh_gia(chon: dict):
    ds = []
    bts = bien_the(chon)
    for ten, bt in bts.items():
        for kb in mp.KICH_BAN:
            N = 100 if bt["loai"] == "pf" else 0
            ds.append((ten, bt, N, kb, mp.L_GAP, mp.KAPPA, mp.SEED_DANH_GIA, OUT / f"danh_gia/{ten}_N{N}_{kb}.csv"))
    for N in (300, 500):                                     # C4: duong tai nguyen, kappa = 1
        for ten in ("pf", "pf-k1", "pf-k2"):
            if ten in bts:
                for kb in mp.KICH_BAN:
                    ds.append((ten, bts[ten], N, kb, mp.L_GAP, (1.0,), mp.SEED_DANH_GIA,
                               OUT / f"danh_gia/{ten}_N{N}_{kb}.csv"))
    return ds


def giai_doan_danh_gia(toi_da: float) -> int:
    chon = json.loads((OUT / "chon.json").read_text(encoding="utf-8"))
    ds = _ds_danh_gia(chon)
    con = _chay_ds(ds, toi_da, "danh-gia")
    if con == 0:
        _chuoi_minh_hoa(chon)
    return 3 if con else 0


def _chuoi_minh_hoa(chon: dict) -> None:
    """Chuoi thoi gian minh hoa: d = 0, kappa = 1, L = 20, h0 = 24, p = 0, moi kich ban, moi bien the N = 100."""
    import pandas as pd
    rows = []
    for ten, bt in bien_the(chon).items():
        bl = tao_bo_loc(bt, 100)
        for kb in mp.KICH_BAN:
            qd = mp.sinh_quy_dao(kb, 0, 24.0)
            do = mp.sinh_do(qd, 1.0, 20)
            ch = []
            chay_mot(bl, qd, do, 0, chuoi=ch)
            rows += [{"bien_the": ten, "kich_ban": kb, **r} for r in ch]
    pd.DataFrame(rows).to_csv(OUT / "chuoi_thoi_gian.csv", index=False)


# ---------------------------------------------------------------------------------------------------------------
def _boot(diff: np.ndarray, idx: np.ndarray):
    m = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def giai_doan_phan_tich() -> int:
    import pandas as pd
    chon = json.loads((OUT / "chon.json").read_text(encoding="utf-8"))
    ds = _ds_danh_gia(chon)
    # Vong sua F2: don vi PF doc ban chan doan da tach tai khoi tao (chan_doan_v2/danh_gia, src/kc48/
    # sua_chan_doan_mo_phong.py); don vi EKF khong co chan doan PF -> doc ban goc.
    nguon = [(OUT / "chan_doan_v2/danh_gia" / p.name) if u[1]["loai"] == "pf" else p for *u, p in ds]
    thieu = [p.name for p in nguon if not p.exists()]
    if thieu:
        print(f"thieu {len(thieu)} don vi danh gia / chan_doan_v2: {thieu[:5]}")
        return 3
    df = pd.concat([pd.read_csv(p) for p in nguon], ignore_index=True)
    df = df.drop(columns=[c for c in ("n_update_ct", "n_upd_bam_ct", "nguon") if c in df])
    df["e_r"] = df["e_r"].replace(np.inf, np.nan)
    pa = OUT / "phan_tich"
    pa.mkdir(parents=True, exist_ok=True)
    import luu_ban_cu
    luu_ban_cu.luu([pa / n for n in ("theo_d.csv", "tom_tat_o.csv", "doi_chieu.csv", "ket_luan_doi_chieu.csv",
                                     "c4_tai_nguyen.csv", "hinh_tai_nguyen_va_ess.png", "hinh_c1_e_r.png")]
                   + [OUT / "chuoi_thoi_gian.csv"], OUT / "_ban_cu_truoc_vong_sua")
    ch_cu = pd.read_csv(OUT / "_ban_cu_truoc_vong_sua/chuoi_thoi_gian.csv")
    _chuoi_minh_hoa(chon)                                  # harness da sua: ess / resample tach tai khoi tao
    ch_moi = pd.read_csv(OUT / "chuoi_thoi_gian.csv")
    assert np.array_equal(ch_cu.e_pred_h.to_numpy(), ch_moi.e_pred_h.to_numpy()), "chuoi: du doan doi"
    print("chuoi_thoi_gian: e_pred_h trung khit ban cu; so buoc tai khoi tao:", int(ch_moi.tai_khoi_tao.fillna(False)
                                                                                   .astype(bool).sum()))
    # mau so ro rang cho chan doan PF (tung o, tren quy dao that, truoc khi trung binh seed)
    pfd = df[df.N > 0]
    mau_so = pfd.groupby(["bien_the", "N", "kich_ban", "L", "kappa", "h0"]).agg(
        n_quy_dao=("d", "size"), n_ess_tr_hop_le=("ess_tr", "count"), n_reinit_tr=("reinit_tr", "sum"),
        n_update=("n_update", "sum"), n_tai_khoi_tao=("n_deg", "sum"), n_upd_bam=("n_upd_bam", "sum"),
        n_reinit_bam=("n_reinit_bam", "sum"), n_qd_co_ess_bam=("ess_bam", "count")).reset_index()
    mau_so["ty_le_tai_khoi_tao_moi_update"] = mau_so.n_tai_khoi_tao / mau_so.n_update.clip(lower=1)
    mau_so["ty_le_reinit_tai_tr"] = mau_so.n_reinit_tr / mau_so.n_quy_dao
    mau_so.to_csv(pa / "mau_so_chan_doan.csv", index=False)
    khoa = ["bien_the", "N", "kich_ban", "L", "kappa", "h0", "d"]
    cot_so = [c for c in COT_V2 if c not in khoa + ["p"]]
    # trung binh 3 seed PF trong moi d (don vi thong ke = d), SD giua seed PF trong d
    g = df.groupby(khoa)
    theo_d = g[cot_so].mean().reset_index()
    sd_seed = g[["e_r", "F"]].std(ddof=1).reset_index().rename(columns={"e_r": "e_r_sd_seed", "F": "F_sd_seed"})
    theo_d = theo_d.merge(sd_seed, on=khoa, how="left")
    theo_d.to_csv(pa / "theo_d.csv", index=False)
    o = ["bien_the", "N", "kich_ban", "L", "kappa", "h0"]
    tom = theo_d.groupby(o).agg(n_d=("d", "nunique"), e_r_tb=("e_r", "mean"), e_r_sd_d=("e_r", "std"),
                                F_tb=("F", "mean"), F05_tb=("F05", "mean"), rmse_h_tb=("rmse_h", "mean"),
                                err_th_tb=("err_th", "mean"), nis_tr_tb=("nis_tr", "mean"),
                                ess_tr_tb=("ess_tr", "mean"), ess_bam_tb=("ess_bam", "mean"),
                                rs_bam_tb=("rs_bam", "mean"), tt_bam_tb=("tt_bam", "mean"), tt_tr_tb=("tt_tr", "mean"),
                                sxy_tr_tb=("sxy_tr", "mean"), sth_tr_tb=("sth_tr", "mean"), co_tb_tb=("co_tb", "mean"),
                                e_r_sd_seed_tb=("e_r_sd_seed", "mean"), us_buoc_tb=("us_buoc", "mean"),
                                n_d_ess_tr=("ess_tr", "count"), reinit_tr_tb=("reinit_tr", "mean"),
                                n_deg_tb=("n_deg", "mean")).reset_index()
    tom.to_csv(pa / "tom_tat_o.csv", index=False)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(mp.SEED_DANH_GIA), size=(5000, len(mp.SEED_DANH_GIA)))

    def doi_chieu(ma, a, b, Na, Nb, Ls, kappas, chi_so):
        rows = []
        for kb in mp.KICH_BAN:
            for L in Ls:
                for kappa in kappas:
                    for h0 in mp.H0:
                        sa = theo_d[(theo_d.bien_the == a) & (theo_d.N == Na) & (theo_d.kich_ban == kb) &
                                    (theo_d.L == L) & (theo_d.kappa == kappa) & (theo_d.h0 == h0)].sort_values("d")
                        sb = theo_d[(theo_d.bien_the == b) & (theo_d.N == Nb) & (theo_d.kich_ban == kb) &
                                    (theo_d.L == L) & (theo_d.kappa == kappa) & (theo_d.h0 == h0)].sort_values("d")
                        if len(sa) == 0 or len(sb) == 0:
                            continue
                        assert list(sa.d) == list(sb.d)
                        for cs in chi_so:
                            diff = sa[cs].to_numpy() - sb[cs].to_numpy()
                            ok = np.isfinite(diff)
                            if cs in CHAN_DOAN_CAP_HOP_LE and 0 < ok.sum() < len(diff):
                                # vong sua F2: chan doan PF chi xac dinh khi khong tai khoi tao -> bootstrap tren cac d
                                # ma CA HAI bien the co gia tri hop le; n_d ghi so d dung (mau so)
                                dd = diff[ok]
                                ii = np.random.default_rng(0).integers(0, len(dd), size=(5000, len(dd)))
                                m, lo, hi = _boot(dd, ii)
                                rows.append({"doi_chieu": ma, "a": a, "b": b, "kich_ban": kb, "nhom": NHOM[kb], "L": L,
                                             "kappa": kappa, "h0": h0, "chi_so": cs, "n_d": int(ok.sum()), "hieu": m,
                                             "ci_lo": lo, "ci_hi": hi, "a_tb": float(np.nanmean(sa[cs].to_numpy()[ok])),
                                             "b_tb": float(np.nanmean(sb[cs].to_numpy()[ok])),
                                             "ghi_chu": "chi d co gia tri hop le o ca hai bien the"})
                                continue
                            if ok.sum() < len(diff):
                                rows.append({"doi_chieu": ma, "a": a, "b": b, "kich_ban": kb, "nhom": NHOM[kb], "L": L,
                                             "kappa": kappa, "h0": h0, "chi_so": cs, "n_d": int(ok.sum()),
                                             "hieu": float(np.nanmean(diff)), "ci_lo": np.nan, "ci_hi": np.nan,
                                             "a_tb": float(np.nanmean(sa[cs])), "b_tb": float(np.nanmean(sb[cs])),
                                             "ghi_chu": "co d khong huu han / thieu -> khong bootstrap"})
                                continue
                            m, lo, hi = _boot(diff, idx)
                            rows.append({"doi_chieu": ma, "a": a, "b": b, "kich_ban": kb, "nhom": NHOM[kb], "L": L,
                                         "kappa": kappa, "h0": h0, "chi_so": cs, "n_d": len(diff), "hieu": m,
                                         "ci_lo": lo, "ci_hi": hi, "a_tb": float(sa[cs].mean()),
                                         "b_tb": float(sb[cs].mean()), "ghi_chu": ""})
        return rows

    rows = []
    rows += doi_chieu("C1", "pf", "ekf", 100, 0, (10, 20), (1.0,), ["e_r", "F"])
    rows += doi_chieu("C1_nhay", "pf", "ekf", 100, 0, (10, 20), (0.5, 2.0), ["e_r", "F"])
    for ma, k in (("C2", "pf-k1"), ("C3", "pf-k2")):
        if k in set(theo_d.bien_the):
            rows += doi_chieu(ma, k, "pf", 100, 100, (0, 10, 20), (1.0,), ["e_r", "F"])
            rows += doi_chieu(ma + "_chan_doan", k, "pf", 100, 100, (10, 20), (1.0,),
                              ["ess_tr", "rs_bam", "ess_bam", "err_th"])
            rows += doi_chieu(ma + "_nhay", k, "pf", 100, 100, (10, 20), (0.5, 2.0), ["e_r", "F"])
    for k in ("pf-k1", "pf-k2", "pf-k1k2", "ekf-chinh"):
        if k in set(theo_d.bien_the):
            rows += doi_chieu(f"P3_{k}_vs_ekf", k, "ekf", 100 if k.startswith("pf") else 0, 0, (10, 20), (1.0,),
                              ["e_r", "F"])
    if "pf-k1k2" in set(theo_d.bien_the):
        rows += doi_chieu("k1k2_vs_pf", "pf-k1k2", "pf", 100, 100, (10, 20), (1.0,), ["e_r", "F"])
    dc = pd.DataFrame(rows)
    dc.to_csv(pa / "doi_chieu.csv", index=False)

    # quy tac dien giai (04 muc 4.4): >= 2/3 so o CI loai 0, cung dau, khong o nao nguoc dau
    kl = []
    for (ma, cs), s in dc.groupby(["doi_chieu", "chi_so"]):
        for nhom, ss in [("tat_ca", s)] + [(n, s[s.nhom == n]) for n in ("dung", "tung_doan", "lech")]:
            ss = ss[np.isfinite(ss.ci_lo)]
            if len(ss) == 0:
                continue
            duong = int((ss.ci_lo > 0).sum()); am = int((ss.ci_hi < 0).sum())
            if duong >= 2 / 3 * len(ss) and am == 0:
                ket = "khac_biet_nhat_quan_duong"
            elif am >= 2 / 3 * len(ss) and duong == 0:
                ket = "khac_biet_nhat_quan_am"
            else:
                ket = "chua_phan_biet_phu_thuoc_kich_ban"
            kl.append({"doi_chieu": ma, "chi_so": cs, "nhom": nhom, "n_o": len(ss), "n_ci_duong": duong, "n_ci_am": am,
                       "hieu_tb_cac_o": float(ss.hieu.mean()), "ket_luan": ket})
    pd.DataFrame(kl).to_csv(pa / "ket_luan_doi_chieu.csv", index=False)

    # C4: duong tai nguyen (kappa 1, L in {10, 20})
    s = theo_d[(theo_d.kappa == 1.0) & (theo_d.L.isin([10, 20]))]
    c4 = s.groupby(["bien_the", "N"]).agg(e_r_tb=("e_r", "mean"), F_tb=("F", "mean"), us_buoc_tb=("us_buoc", "mean"),
                                          ess_tr_tb=("ess_tr", "mean"), n_o_d=("d", "size")).reset_index()
    c4 = c4.rename(columns={"us_buoc_tb": "us_buoc_song_song_tb"})
    td = OUT / "toc_do.csv"
    if td.exists():                                      # thoi gian tai nguyen: mot tien trinh (do-toc-do)
        c4 = c4.merge(pd.read_csv(td)[["bien_the", "N", "us_buoc_tb"]], on=["bien_the", "N"], how="left")
    else:
        c4["us_buoc_tb"] = np.nan
    c4.to_csv(pa / "c4_tai_nguyen.csv", index=False)
    _ve_hinh(pa, c4, dc)
    print(pd.DataFrame(kl).to_string())
    print(c4.to_string())
    return 0


def _ve_hinh(pa: Path, c4, dc) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for ten, s in c4[c4.bien_the.str.startswith("pf")].groupby("bien_the"):
        s = s.sort_values("N")
        ax[0].plot(s.us_buoc_tb, s.e_r_tb, "o-", label=ten)
        ax[1].plot(s.us_buoc_tb, s.F_tb, "o-", label=ten)
        for _, r in s.iterrows():
            ax[0].annotate(f"N={int(r.N)}", (r.us_buoc_tb, r.e_r_tb), fontsize=7)
    for ten, s in c4[~c4.bien_the.str.startswith("pf")].iterrows():
        ax[0].axhline(s.e_r_tb, ls="--", color="k" if s.bien_the == "ekf" else "gray", label=s.bien_the)
        ax[1].axhline(s.F_tb, ls="--", color="k" if s.bien_the == "ekf" else "gray", label=s.bien_the)
    ax[0].set_xlabel("us / buoc (predict + update)"); ax[0].set_ylabel("e_r tb (/h)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("us / buoc"); ax[1].set_ylabel("F tb (IoU < 0,22)")
    ch = OUT / "chuoi_thoi_gian.csv"
    if ch.exists():
        cdf = pd.read_csv(ch)
        for ten, s in cdf[(cdf.kich_ban == "cua_gat") & cdf.bien_the.str.startswith("pf")].groupby("bien_the"):
            ln = ax[2].plot(s.t, s.ess, ".-", ms=3, label=ten)[0]
            if "tai_khoi_tao" in s:
                k = s[s.tai_khoi_tao.fillna(False).astype(bool)]
                ax[2].plot(k.t, np.zeros(len(k)), "x", color=ln.get_color(), ms=6)
        ax[2].axvspan(30, 50, color="0.85", label="mat do (L = 20)")
        ax[2].set_xlabel("t (buoc)"); ax[2].set_ylabel("ESS/N truoc resample (x = tai khoi tao, khong co ESS)")
        ax[2].set_title("cua_gat, d = 0, h0 = 24")
        ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(pa / "hinh_tai_nguyen_va_ess.png", dpi=130); plt.close(fig)
    s = dc[(dc.doi_chieu == "C1") & (dc.chi_so == "e_r")]
    if len(s):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        lab = [f"{r.kich_ban}\nL{r.L} h{int(r.h0)}" for r in s.itertuples()]
        x = np.arange(len(s))
        ax.errorbar(x, s.hieu, yerr=[s.hieu - s.ci_lo, s.ci_hi - s.hieu], fmt="o", ms=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=5, rotation=90)
        ax.set_ylabel("e_r(pf100) - e_r(ekf), CI 95 %"); ax.set_title("C1, kappa = 1, bootstrap theo d (30)")
        fig.tight_layout(); fig.savefig(pa / "hinh_c1_e_r.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def giai_doan_hinh_hat() -> int:
    """Hai track PF (N = 100) biet lien ket: xe lon (h = 72, a = 2.2) va xe nho (h = 24) di cung chieu lan ben canh;
    xe nho bi che (khong co do) t = 30..41 va doi lan (re nhe) trong luc bi che. Moi track co dam may + rng rieng."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    T = 50
    t_ = np.arange(T, dtype=np.float64)
    lon = np.stack([200 + 4.0 * t_, np.full(T, 300.0), np.full(T, 2.2), np.full(T, 72.0)], axis=1)
    lech = np.clip((t_ - 30) / 12, 0, 1)
    nho = np.stack([215 + 4.2 * t_, 262 - 30 * (3 * lech ** 2 - 2 * lech ** 3), np.full(T, 1.6), np.full(T, 24.0)], 1)
    co_do_nho = ~((t_ >= 30) & (t_ < 42))
    rng_do = np.random.default_rng(48)
    pf = PFCTRV(num_particles=100)
    pf.set_seed(0, video="hinh_hat")
    tracks = {}
    for tid, gt in ((1, lon), (2, nho)):
        z0 = gt[0] + rng_do.standard_normal(4) * np.array([gt[0, 3] / 20, gt[0, 3] / 20, 0.1, gt[0, 3] / 20])
        m, P = pf.initiate(z0)
        trk = type("T", (), {})()
        trk.track_id, trk.frame_id, trk._pending_frame_id, trk._k1_prior = tid, 0, 0, None
        trk.rng = pf.make_rng(tid)
        trk.particles = pf.sample_initial(m, P, trk.rng)
        trk.logw = np.full(100, -np.log(100))
        tracks[tid] = {"trk": trk, "view": PFTrackView(pf, trk), "m": m, "P": P, "gt": gt, "lost": False, "ok": False,
                       "last": (float(z0[0]), float(z0[1])), "lt": 0}
    chup = {}
    for t in range(1, T):
        for tid, s in tracks.items():
            mi = s["m"].copy()
            if s["lost"]:
                mi[VH] = 0
            s["m"], s["P"] = s["view"].predict(mi, s["P"], lost=s["lost"])
            co = True if tid == 1 else bool(co_do_nho[t])
            if co:
                gt = s["gt"][t]
                z = gt + rng_do.standard_normal(4) * np.array([gt[3] / 20, gt[3] / 20, 0.1, gt[3] / 20])
                s["trk"].frame_id = s["lt"]
                if not s["ok"]:
                    s["m"], s["P"], s["ok"] = s["view"].try_initiate_from_motion(s["m"], s["P"], z,
                                                                                 n_frames=float(t - s["lt"]),
                                                                                 ref_pos=s["last"])
                s["m"], s["P"] = s["view"].update(s["m"], s["P"], z)
                s["last"], s["lt"], s["lost"] = (float(z[0]), float(z[1])), t, False
            else:
                s["lost"] = True
        if t in (29, 35, 41, 42):
            chup[t] = {tid: (s["trk"].particles.copy(), s["m"].copy()) for tid, s in tracks.items()}
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    mau = {1: "tab:blue", 2: "tab:orange"}
    ten_t = {29: "t = 29 (truoc khi che)", 35: "t = 35 (bi che 6 buoc)", 41: "t = 41 (bi che 12 buoc)",
             42: "t = 42 (ghep lai, sau update)"}
    for ax, (t, snap) in zip(axs, chup.items()):
        for tid, (X, m) in snap.items():
            gt = tracks[tid]["gt"][t]
            ax.scatter(X[:, CX], X[:, CY], s=4, c=mau[tid], alpha=0.5,
                       label=f"hat track {tid} ({'xe lon' if tid == 1 else 'xe nho'})")
            w = gt[2] * gt[3]
            ax.add_patch(Rectangle((gt[0] - w / 2, gt[1] - gt[3] / 2), w, gt[3], fill=False, ec=mau[tid], lw=1.2))
            wm = m[A] * m[H]
            ax.add_patch(Rectangle((m[CX] - wm / 2, m[CY] - m[H] / 2), wm, m[H], fill=False, ec=mau[tid], lw=1,
                                   ls="--"))
        ax.set_title(ten_t[t], fontsize=9); ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xlim(snap[1][1][CX] - 140, snap[1][1][CX] + 160); ax.set_ylim(360, 200)
    axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Mo phong (khong phai anh dataset): hat to mau theo track so huu; khung lien = GT, khung dut = uoc luong",
                 fontsize=9)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "hinh_hat_so_huu.png", dpi=130); plt.close(fig)
    tom = {str(t): {str(tid): {"spread_xy_px": pf._diag_spread(X, np.full(len(X), -np.log(len(X))))[0],
                               "sai_tam_px": float(np.hypot(*(m[[CX, CY]] - tracks[tid]["gt"][t][:2])))}
                    for tid, (X, m) in snap.items()} for t, snap in chup.items()}
    (OUT / "hinh_hat_so_huu.json").write_text(json.dumps(tom, indent=2), encoding="utf-8")
    print(json.dumps(tom, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--giai-doan", required=True, choices=["tinh-chinh", "danh-gia", "phan-tich", "hinh-hat", "do-toc-do",
                                                         "chup-giao-thuc"])
    ap.add_argument("--toi-da-giay", type=float, default=480.0)
    a = ap.parse_args(argv)
    if a.giai_doan == "tinh-chinh":
        return giai_doan_tinh_chinh(a.toi_da_giay)
    if a.giai_doan == "danh-gia":
        return giai_doan_danh_gia(a.toi_da_giay)
    if a.giai_doan == "phan-tich":
        return giai_doan_phan_tich()
    if a.giai_doan == "hinh-hat":
        return giai_doan_hinh_hat()
    if a.giai_doan == "chup-giao-thuc":
        return chup_giao_thuc()
    return giai_doan_do_toc_do()


def chup_giao_thuc() -> int:
    """sha256 cua giao thuc 04 + kiem 22.510 byte dau (muc 1-12 da khoa, 7063c532...) TRUOC khi chay so sanh R48D."""
    import hashlib
    p = ROOT / "handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md"
    b = p.read_bytes()
    muc = {"file": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": hashlib.sha256(b).hexdigest(), "byte": len(b),
           "tien_to_22510_sha256": hashlib.sha256(b[:22510]).hexdigest(),
           "tien_to_khop_7063c532": hashlib.sha256(b[:22510]).hexdigest().startswith("7063c532"),
           "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "giao_thuc_sha256_r48d.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(muc, ensure_ascii=False) + "\n")
    print(muc)
    return 0 if muc["tien_to_khop_7063c532"] else 1


def giai_doan_do_toc_do() -> int:
    """Thoi gian tuong moi buoc (predict + update) MOT tien trinh, tuan tu, khong tai song song (04 muc 4.3):
    moi (bien the, N) cua danh gia, d = 0..4, kappa 1, L in {10, 20}, ca hai h0, moi kich ban, p = 0.
    Ghi chu: harness bat enable_diag() cho PF (chan doan) -> thoi gian gom ca phan ghi chan doan."""
    import pandas as pd
    chon = json.loads((OUT / "chon.json").read_text(encoding="utf-8"))
    bts = bien_the(chon)
    cap = sorted({(u[0], u[2]) for u in _ds_danh_gia(chon)}, key=lambda x: (x[1], x[0]))
    rows = []
    for ten, N in cap:
        bl = tao_bo_loc(bts[ten], N)
        qd0 = mp.sinh_quy_dao("thang", 99, 24.0)
        chay_mot(bl, qd0, mp.sinh_do(qd0, 1.0, 10), 0)            # lam nong
        us = []
        for kb in mp.KICH_BAN:
            for d in range(5):
                for h0 in mp.H0:
                    qd = mp.sinh_quy_dao(kb, d, h0)
                    for L in (10, 20):
                        us.append(chay_mot(bl, qd, mp.sinh_do(qd, 1.0, L), 0)["us_buoc"])
        rows.append({"bien_the": ten, "N": N, "n_quy_dao": len(us), "us_buoc_tb": float(np.mean(us)),
                     "us_buoc_trung_vi": float(np.median(us)), "us_buoc_sd": float(np.std(us, ddof=1))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "toc_do.csv", index=False)
    _ghi_nhat_ky({"giai_doan": "do-toc-do", "so_cap": len(rows), "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
