"""
so_sanh_truoc_sau_sua.py -- R48D vong sua F1-F3: doi chieu dau ra truoc (ban sao _ban_cu_truoc_vong_sua/, sao nguyen
byte boi luu_ban_cu.py) va sau khi tinh lai. Chi doc; ghi hai JSON tom tat.

    python src/kc48/so_sanh_truoc_sau_sua.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "results/phase2/kc48_pf_dev/phan_tich"
MP = ROOT / "results/phase2/kc48_mo_phong"


def _md(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b)
    return float(np.nanmax(d)) if np.isfinite(d).any() else 0.0


def dev() -> dict:
    cu_d, out = DEV / "_ban_cu_truoc_vong_sua", {}
    bc0, bc1 = pd.read_csv(cu_d / "bang_chinh.csv"), pd.read_csv(DEV / "bang_chinh.csv")
    out["bang_chinh_lech_lon_nhat"] = {m: _md(bc0[m], bc1[m]) for m in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")}
    dc0, dc1 = pd.read_csv(cu_d / "doi_chieu_bootstrap.csv"), pd.read_csv(DEV / "doi_chieu_bootstrap.csv")
    out["bootstrap_lech_lon_nhat"] = {c: _md(dc0[c], dc1[c]) for c in ("hieu", "ci_lo", "ci_hi")}
    out["bootstrap_doi_trang_thai_ci_loai_0"] = dc1[dc0.ci_loai_0 != dc1.ci_loai_0][["doi_chieu", "a", "b", "chi_so"]] \
        .to_dict("records")
    out["bootstrap_hota_idf1_idsw_moi"] = dc1[dc1.chi_so.isin(["HOTA", "IDF1", "IDSW"])][
        ["doi_chieu", "a", "b", "chi_so", "hieu", "ci_lo", "ci_hi"]].round(4).to_dict("records")
    g0, g1 = pd.read_csv(cu_d / "giu_id_theo_track.csv"), pd.read_csv(DEV / "giu_id_theo_track.csv")
    k = ["video", "bien_the", "seed", "gt_id"]
    g = g0.merge(g1, on=k, suffixes=("_cu", "_moi"))
    doi = (g.ty_le_giu_id_cu != g.ty_le_giu_id_moi) | (g.n_doi_id_cu != g.n_doi_id_moi)
    out["giu_id_theo_track"] = {"n_dong": len(g), "n_dong_doi": int(doi.sum()),
                                "lech_ty_le_lon_nhat": _md(g.ty_le_giu_id_cu, g.ty_le_giu_id_moi),
                                "lech_n_doi_id_lon_nhat": _md(g.n_doi_id_cu, g.n_doi_id_moi),
                                "doi_theo_bien_the": g[doi].groupby("bien_the").size().to_dict()}
    p0, p1 = pd.read_csv(cu_d / "phan_tang.csv"), pd.read_csv(DEV / "phan_tang.csv")
    kk = ["tang", "gia_tri", "bien_the"]
    p = p0.merge(p1, on=kk, suffixes=("_cu", "_moi"))
    out["phan_tang_lech_lon_nhat"] = {c: _md(p[c + "_cu"], p[c + "_moi"])
                                      for c in ("ty_le_giu_id_tb", "n_doi_id_tb", "hieu_giu_id_vs_ekf", "ci_lo", "ci_hi")}

    def tt(lo, hi):
        return np.where(lo > 0, "duong", np.where(hi < 0, "am", "chua_phan_biet"))
    p["tt_cu"], p["tt_moi"] = tt(p.ci_lo_cu, p.ci_hi_cu), tt(p.ci_lo_moi, p.ci_hi_moi)
    co_ci = p.ci_lo_cu.notna() | p.ci_lo_moi.notna()
    out["phan_tang_doi_trang_thai_ci"] = p[co_ci & (p.tt_cu != p.tt_moi)][
        kk + ["hieu_giu_id_vs_ekf_cu", "ci_lo_cu", "ci_hi_cu", "hieu_giu_id_vs_ekf_moi", "ci_lo_moi", "ci_hi_moi"]] \
        .round(4).to_dict("records")
    return out


def mo_phong() -> dict:
    cu_d, pa, out = MP / "_ban_cu_truoc_vong_sua", MP / "phan_tich", {}
    k0, k1 = pd.read_csv(cu_d / "ket_luan_doi_chieu.csv"), pd.read_csv(pa / "ket_luan_doi_chieu.csv")
    kk = ["doi_chieu", "chi_so", "nhom"]
    k = k0.merge(k1, on=kk, how="outer", suffixes=("_cu", "_moi"))
    doi = (k.n_o_cu != k.n_o_moi) | (k.n_ci_duong_cu != k.n_ci_duong_moi) | (k.n_ci_am_cu != k.n_ci_am_moi) | \
        (k.ket_luan_cu != k.ket_luan_moi) | ((k.hieu_tb_cac_o_cu - k.hieu_tb_cac_o_moi).abs() > 1e-12)
    out["ket_luan_doi"] = k[doi][kk + ["n_o_cu", "n_ci_duong_cu", "n_ci_am_cu", "hieu_tb_cac_o_cu", "ket_luan_cu",
                                       "n_o_moi", "n_ci_duong_moi", "n_ci_am_moi", "hieu_tb_cac_o_moi",
                                       "ket_luan_moi"]].round(6).to_dict("records")
    out["ket_luan_doi_chi_so"] = sorted(set(k[doi].chi_so))
    t0, t1 = pd.read_csv(cu_d / "tom_tat_o.csv"), pd.read_csv(pa / "tom_tat_o.csv")
    o = ["bien_the", "N", "kich_ban", "L", "kappa", "h0"]
    t = t0.merge(t1, on=o, suffixes=("_cu", "_moi"))
    out["tom_tat_o_lech_lon_nhat"] = {c: _md(t[c + "_cu"], t[c + "_moi"]) for c in
                                      ("e_r_tb", "F_tb", "rmse_h_tb", "err_th_tb", "nis_tr_tb", "sxy_tr_tb", "co_tb_tb",
                                       "ess_tr_tb", "ess_bam_tb", "rs_bam_tb", "tt_bam_tb", "tt_tr_tb")}
    # bang 07 muc 5.3: o cua_deu, L 20, kappa 1, N 100 (EKF N 0)
    ms = pd.read_csv(pa / "mau_so_chan_doan.csv")
    s = t[(t.kich_ban == "cua_deu") & (t.L == 20) & (t.kappa == 1.0) & t.N.isin([0, 100])]
    bang = []
    for r in s.itertuples():
        m = ms[(ms.bien_the == r.bien_the) & (ms.N == r.N) & (ms.kich_ban == "cua_deu") & (ms.L == 20) &
               (ms.kappa == 1.0) & (ms.h0 == r.h0)]
        bang.append({"bien_the": r.bien_the, "h0": r.h0, "ess_tr_cu": r.ess_tr_tb_cu, "ess_tr_moi": r.ess_tr_tb_moi,
                     "ess_bam_cu": r.ess_bam_tb_cu, "ess_bam_moi": r.ess_bam_tb_moi, "rs_bam_cu": r.rs_bam_tb_cu,
                     "rs_bam_moi": r.rs_bam_tb_moi, "tt_bam_cu": r.tt_bam_tb_cu, "tt_bam_moi": r.tt_bam_tb_moi,
                     "n_d_ess_tr_moi": r.n_d_ess_tr, "reinit_tr_tb": r.reinit_tr_tb,
                     **({"n_quy_dao": int(m.n_quy_dao.iloc[0]), "n_ess_tr_hop_le": int(m.n_ess_tr_hop_le.iloc[0]),
                         "n_reinit_tr": int(m.n_reinit_tr.iloc[0]), "n_upd_bam": int(m.n_upd_bam.iloc[0]),
                         "n_reinit_bam": int(m.n_reinit_bam.iloc[0]),
                         "ty_le_tai_khoi_tao_moi_update": float(m.ty_le_tai_khoi_tao_moi_update.iloc[0])}
                        if len(m) else {})})
    out["bang_07_muc_5_3"] = pd.DataFrame(bang).round(6).to_dict("records")
    # o Codex neu (cua_deu, pf N 100, L 20, kappa 1, h0 24): trung binh gop tren quy dao khong tai khoi tao
    v2 = pd.read_csv(MP / "chan_doan_v2/danh_gia/pf_N100_cua_deu.csv")
    c = v2[(v2.L == 20) & (v2.kappa == 1.0) & (v2.h0 == 24.0)]
    out["o_codex_cua_deu_h24"] = {"n_quy_dao": len(c), "n_reinit_tr": int(c.reinit_tr.sum()),
                                  "ess_tr_tb_gop_hop_le": float(c.ess_tr.mean())}
    pf = ms[ms.N > 0]
    out["tong_tai_khoi_tao_theo_bien_the"] = pf.groupby(["bien_the", "N"]).apply(
        lambda x: pd.Series({"n_update": x.n_update.sum(), "n_tai_khoi_tao": x.n_tai_khoi_tao.sum(),
                             "ty_le": x.n_tai_khoi_tao.sum() / x.n_update.sum(), "n_upd_bam": x.n_upd_bam.sum(),
                             "n_reinit_bam": x.n_reinit_bam.sum(), "n_quy_dao": x.n_quy_dao.sum(),
                             "n_reinit_tr": x.n_reinit_tr.sum()}), include_groups=False).reset_index() \
        .round(6).to_dict("records")
    c40, c41 = pd.read_csv(cu_d / "c4_tai_nguyen.csv"), pd.read_csv(pa / "c4_tai_nguyen.csv")
    out["c4_lech_lon_nhat"] = {cc: _md(c40[cc], c41[cc]) for cc in ("e_r_tb", "F_tb", "us_buoc_tb", "ess_tr_tb")}
    return out


def main() -> int:
    a, b = dev(), mo_phong()
    (DEV / "so_sanh_truoc_sau_vong_sua.json").write_text(json.dumps(a, indent=1, ensure_ascii=False), encoding="utf-8")
    (MP / "phan_tich/so_sanh_truoc_sau_vong_sua.json").write_text(json.dumps(b, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8")
    for x in (a, b):
        for kk, v in x.items():
            print(kk, ":", json.dumps(v, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
