"""
sua_chan_doan_mo_phong.py -- R48D vong sua (Codex review_003 F2): tach TAI KHOI TAO (nhanh suy bien cua
PFCTRV.update_particles, src/pf_ctrv.py:308-322) khoi ESS hop le va systematic resampling trong chan doan mo phong.

Nhanh tai khoi tao ghi vao diag_rows ess_ratio = 0.0, resampled = True theo QUY UOC (khong phai ESS that, khong phai
systematic resampling). Harness cu (thi_nghiem_mo_phong.chay_mot truoc vong sua) lay thang hai truong do -> tron hai co
che; n_deg la bo dem TICH LUY cua PFCTRV qua moi quy dao cua mot don vi.

    --buoc kiem       chi doc danh_gia/*.csv (ban cu): n_deg theo TUNG quy dao = hieu lien tiep cua bo dem tich luy
                      (thu tu dong = thu tu chay trong chay_don_vi, mot bo loc moi don vi); dem quy dao co tai khoi tao
                      tai t_r va NGOAI t_r (ngoai t_r thi ess_bam / rs_bam cu co the bi tron) -> chan_doan_v2/kiem_ban_cu.json
    --buoc chay-lai   chay lai CUNG cau hinh / seed bang harness da sua (chay_mot moi) cho moi don vi PF cua danh-gia,
                      ghi chan_doan_v2/danh_gia/*.csv; bo qua don vi da co (tiep tuc duoc). --toi-da-giay S.
    --buoc so-sanh    doi chieu cot dau ra bo loc (e_r, iou_tr, F, F05, rmse_h, err_th, err_om, nis_tr, sxy_tr, sth_tr,
                      sv_tr, co_tb) ban moi vs ban cu: phai trung khit -> chan doan khong doi dau ra bo loc; kiem n_deg
                      moi == hieu bo dem cu; ess_tr / rs_tr ban cu tai dong khong tai khoi tao trung ban moi.

    python src/kc48/sua_chan_doan_mo_phong.py --buoc kiem
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import thi_nghiem_mo_phong as tn  # noqa: E402

OUT = tn.OUT / "chan_doan_v2"
COT_BO_LOC = ["e_r", "iou_tr", "F", "F05", "rmse_h", "err_th", "err_om", "nis_tr", "sxy_tr", "sth_tr", "sv_tr",
              "co_tb"]
KHOA = ["bien_the", "N", "p", "kich_ban", "L", "kappa", "h0", "d"]


def _don_vi_pf():
    chon = json.loads((tn.OUT / "chon.json").read_text(encoding="utf-8"))
    return [u for u in tn._ds_danh_gia(chon) if u[1]["loai"] == "pf"]


def n_deg_theo_quy_dao(df: pd.DataFrame) -> np.ndarray:
    """Bo dem tich luy (theo thu tu dong cua mot don vi) -> so lan tai khoi tao cua tung quy dao."""
    c = df["n_deg"].to_numpy(float)
    return np.diff(np.concatenate([[0.0], c]))


def buoc_kiem() -> int:
    rows = []
    for *_, p in _don_vi_pf():
        df = pd.read_csv(p)
        nd = n_deg_theo_quy_dao(df)
        deg_tr = df["deg_tr"].fillna(0).to_numpy(float)
        rows.append({"don_vi": p.name, "n_quy_dao": len(df), "bo_dem_cuoi": float(df.n_deg.iloc[-1]),
                     "tong_hieu": float(nd.sum()), "hieu_am": int((nd < 0).sum()),
                     "n_qd_tai_khoi_tao": int((nd > 0).sum()), "n_tai_khoi_tao_tai_tr": int(deg_tr.sum()),
                     "n_qd_tai_khoi_tao_ngoai_tr": int(((nd - deg_tr) > 0).sum()),
                     "ess_tr_cu_bang_0": int((df.ess_tr == 0).sum())})
    k = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tom = {"mo_ta": "ban cu danh_gia/*.csv, chi don vi PF; n_deg theo quy dao = hieu lien tiep bo dem tich luy",
           "n_don_vi": len(k), "n_quy_dao": int(k.n_quy_dao.sum()), "tong_tai_khoi_tao": float(k.tong_hieu.sum()),
           "hieu_am": int(k.hieu_am.sum()), "n_qd_tai_khoi_tao_tai_tr": int(k.n_tai_khoi_tao_tai_tr.sum()),
           "n_qd_tai_khoi_tao_ngoai_tr": int(k.n_qd_tai_khoi_tao_ngoai_tr.sum()),
           "chi_tiet": rows, "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "kiem_ban_cu.json").write_text(json.dumps(tom, indent=2, ensure_ascii=False), encoding="utf-8")
    print(k.to_string())
    print({x: tom[x] for x in tom if x != "chi_tiet"})
    return 0


BUOC_DOI_CHUNG = 50          # moi dong thu 50 (theo chi so dong) khong can chay lai -> chay lai lam doi chung


def sua_don_vi(ten: str, bt: dict, N: int, kb: str, cu_p: Path, dich: Path) -> float:
    """Mot don vi PF: dong co tai khoi tao NGOAI t_r (+ doi chung) -> chay lai chay_mot da sua, cung seed / cau hinh;
    dong con lai -> sua tu ban cu (du thong tin: khong tai khoi tao nao ngoai t_r nen ess_bam / rs_bam / tt_bam cu
    hop le; t_r tach bang deg_tr). Mau so n_update / n_upd_bam tinh tu mat na co_do cua bo sinh (kiem o doi chung)."""
    import mo_phong_quy_dao as mp
    t0 = time.perf_counter()
    cu = pd.read_csv(cu_p)
    nd = n_deg_theo_quy_dao(cu)
    deg = cu["deg_tr"].fillna(0).to_numpy(float)
    can = (nd - deg) > 0
    doi_chung = (~can) & (np.arange(len(cu)) % BUOC_DOI_CHUNG == 0)
    bl = tn.tao_bo_loc(bt, N)
    cache_qd, rows = {}, []
    for i, r in enumerate(cu.itertuples(index=False)):
        kq = (int(r.d), float(r.h0))
        if kq not in cache_qd:
            cache_qd[kq] = mp.sinh_quy_dao(kb, int(r.d), float(r.h0))
        qd = cache_qd[kq]
        do = mp.sinh_do(qd, float(r.kappa), int(r.L))
        co = np.asarray(do["co_do"], bool)
        khoa = {k: getattr(r, k) for k in KHOA}
        ct = {"n_update_ct": float(co[1:].sum()), "n_upd_bam_ct": float(co[10:do["t_gap"]].sum())}
        if can[i] or doi_chung[i]:
            moi = tn.chay_mot(bl, qd, do, int(r.p))
            rows.append({**khoa, **{c: moi[c] for c in tn.COT_V2 if c not in KHOA and c != "bien_the"},
                         **ct, "nguon": "chay_lai" if can[i] else "doi_chung"})
        else:
            x = {c: getattr(r, c) for c in tn.COT if c not in KHOA}
            kt = deg[i] > 0
            x.update(reinit_tr=r.deg_tr, ess_tr=np.nan if kt else r.ess_tr, rs_tr=np.nan if kt else r.rs_tr,
                     n_deg=nd[i], n_update=ct["n_update_ct"], n_upd_bam=ct["n_upd_bam_ct"], n_reinit_bam=0.0)
            rows.append({**khoa, **x, **ct, "nguon": "ban_cu_sua"})
    df = pd.DataFrame(rows)[tn.COT_V2 + ["n_update_ct", "n_upd_bam_ct", "nguon"]]
    dich.parent.mkdir(parents=True, exist_ok=True)
    tmp = dich.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(dich)
    return time.perf_counter() - t0


def buoc_chay_lai(toi_da: float) -> int:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    cho = [(u[0], u[1], u[2], u[3], u[-1], OUT / "danh_gia" / u[-1].name) for u in _don_vi_pf()]
    cho = [u for u in cho if not u[-1].exists()]
    t_bd, nw = time.perf_counter(), tn._so_tien_trinh()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        dang = {}
        while cho or dang:
            while cho and len(dang) < nw and time.perf_counter() - t_bd <= toi_da:
                u = cho.pop(0)
                dang[ex.submit(sua_don_vi, *u)] = u
            if not dang:
                break
            xong, _ = wait(dang, return_when=FIRST_COMPLETED)
            for fu in xong:
                u = dang.pop(fu)
                s = fu.result()
                tn._ghi_nhat_ky({"giai_doan": "chan-doan-v2", "don_vi": u[-1].name, "giay": round(s, 2),
                                 "tien_trinh": nw, "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                print(f"  {u[-1].name}: {s:.1f} s", flush=True)
    print(f"{nw} tien trinh; {time.perf_counter() - t_bd:.0f} s; con {len(cho)} don vi")
    return 3 if cho else 0


def buoc_so_sanh() -> int:
    out, ok = [], True
    for *_, p in _don_vi_pf():
        moi_p = OUT / "danh_gia" / p.name
        if not moi_p.exists():
            print("thieu", moi_p.name)
            return 3
        cu, moi = pd.read_csv(p), pd.read_csv(moi_p)
        assert len(cu) == len(moi) and (cu[KHOA].to_numpy() == moi[KHOA].to_numpy()).all(), p.name
        cl = moi.nguon.isin(["chay_lai", "doi_chung"]).to_numpy()
        dc = (moi.nguon == "doi_chung").to_numpy()
        r = {"don_vi": p.name, "n_dong": len(cu), "n_chay_lai": int((moi.nguon == "chay_lai").sum()),
             "n_doi_chung": int(dc.sum())}
        # (a) dau ra bo loc: dong chay lai bang harness da sua trung khit ban cu
        r["dau_ra_bo_loc_trung"] = all(np.array_equal(cu[c].to_numpy(float)[cl], moi[c].to_numpy(float)[cl],
                                                      equal_nan=True) for c in COT_BO_LOC)
        # (b) n_deg theo quy dao (moi dong) == hieu bo dem tich luy cu
        r["n_deg_eq_hieu_cu"] = bool(np.array_equal(moi.n_deg.to_numpy(float), n_deg_theo_quy_dao(cu)))
        # (c) doi chung (khong tai khoi tao ngoai t_r): chan doan khi bam cu == moi; mau so tinh tu co_do == harness
        r["doi_chung_bam_trung"] = all(np.array_equal(cu[c].to_numpy(float)[dc], moi[c].to_numpy(float)[dc],
                                                      equal_nan=True) for c in ("ess_bam", "rs_bam", "tt_bam"))
        r["mau_so_ct_trung"] = bool(np.array_equal(moi.n_update.to_numpy(float)[cl], moi.n_update_ct.to_numpy(float)[cl])
                                    and np.array_equal(moi.n_upd_bam.to_numpy(float)[cl],
                                                       moi.n_upd_bam_ct.to_numpy(float)[cl]))
        kt = moi.reinit_tr.fillna(0).to_numpy() > 0
        r["reinit_tr_eq_deg_tr_cu"] = bool(np.array_equal(kt, cu.deg_tr.fillna(0).to_numpy() > 0))
        r["ess_tr_khop_ngoai_tai_khoi_tao"] = bool(np.array_equal(cu.ess_tr.to_numpy(float)[~kt],
                                                                 moi.ess_tr.to_numpy(float)[~kt], equal_nan=True))
        r["ess_tr_nan_khi_tai_khoi_tao"] = bool(moi.ess_tr[kt].isna().all() and moi.rs_tr[kt].isna().all())
        r["n_reinit_tr"] = int(kt.sum())
        r["n_reinit_bam"] = int(moi.n_reinit_bam.sum())
        r["n_upd_bam"] = int(moi.n_upd_bam.sum())
        r["ess_tr_min_hop_le"] = float(np.nanmin(moi.ess_tr)) if moi.ess_tr.notna().any() else np.nan
        r["ess_bam_min_hop_le"] = float(np.nanmin(moi.ess_bam)) if moi.ess_bam.notna().any() else np.nan
        san = 1.0 / int(moi.N.iloc[0]) - 1e-12
        r["ess_hop_le_ge_1_tren_N"] = not (r["ess_tr_min_hop_le"] < san or r["ess_bam_min_hop_le"] < san)
        r["dat"] = all(r[k] for k in ("dau_ra_bo_loc_trung", "n_deg_eq_hieu_cu", "doi_chung_bam_trung",
                                      "mau_so_ct_trung", "reinit_tr_eq_deg_tr_cu", "ess_tr_khop_ngoai_tai_khoi_tao",
                                      "ess_tr_nan_khi_tai_khoi_tao", "ess_hop_le_ge_1_tren_N"))
        ok &= r["dat"]
        out.append(r)
    k = pd.DataFrame(out)
    tom = {"mo_ta": "harness da sua vs ban cu (danh_gia/*.csv): (a) cot dau ra bo loc cua dong chay lai trung khit "
                    "(np.array_equal, NaN == NaN); (b) n_deg theo quy dao == hieu bo dem tich luy cu; (c) doi chung "
                    "ess_bam / rs_bam / tt_bam trung; mau so tu co_do == harness; reinit_tr == deg_tr cu; ess_tr khop o "
                    "dong khong tai khoi tao, NaN khi tai khoi tao; ESS/N hop le >= 1/N",
           "tat_ca_dat": bool(ok), "n_don_vi": len(k), "n_dong": int(k.n_dong.sum()),
           "n_chay_lai": int(k.n_chay_lai.sum()), "n_doi_chung": int(k.n_doi_chung.sum()),
           "n_reinit_tr": int(k.n_reinit_tr.sum()), "n_reinit_bam": int(k.n_reinit_bam.sum()),
           "n_upd_bam": int(k.n_upd_bam.sum()),
           "chi_tiet": out, "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / "so_sanh_ban_cu.json").write_text(json.dumps(tom, indent=2, ensure_ascii=False), encoding="utf-8")
    print(k[["don_vi", "n_dong", "n_chay_lai", "n_doi_chung", "n_reinit_tr", "n_reinit_bam", "ess_tr_min_hop_le",
             "ess_bam_min_hop_le", "dat"]].to_string())
    print({x: tom[x] for x in tom if x != "chi_tiet"})
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", required=True, choices=["kiem", "chay-lai", "so-sanh"])
    ap.add_argument("--toi-da-giay", type=float, default=480.0)
    a = ap.parse_args(argv)
    if a.buoc == "kiem":
        return buoc_kiem()
    if a.buoc == "chay-lai":
        return buoc_chay_lai(a.toi_da_giay)
    return buoc_so_sanh()


if __name__ == "__main__":
    raise SystemExit(main())
