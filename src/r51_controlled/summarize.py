"""
summarize.py -- R51C: tong hop R51 (giao thuc handoff/logs/review51/01_giao_thuc.md muc 1, 5-7), chi DOC dau ra B.

Dau vao (da duoc approve o B): results/phase2/r51_controlled/eval/{R0,R1}/<dir>/raw_fields.json + eval_meta.json,
file tracking results/phase2/r51_controlled/runs/_processed/trackers/DETRAC-all/<dir>/data/*.txt va GT ban sao trung
byte trong cung goc. Chi ghi results/phase2/r51_controlled/summary/.

1. Bang chinh (muc 7): HOTA, AssA, DetA, IDF1, IDSW (MOTA, FP tham chieu) COMBINED tu raw_fields (cong thuc TrackEval,
   src/boot_stats.combine_stacked). PF: COMBINED rieng tung seed roi trung binh 3 seed (src/kc48/boot_seed.py).
   Khoang giua seed chi la mo ta, KHONG phai CI.
2. Bootstrap ghep cap theo video (n = 12): B = 5000, RNG default_rng(0), moi mau rut `integers(0, n, size=n)` nhu
   boot_stats.paired_bootstrap; MOT day chi so dung chung cho moi doi chung, moi bien the va moi seed PF; trong moi
   mau PF tinh COMBINED theo thu tu seed 0, 1, 2 roi trung binh. CI 95 % phan vi cho hieu so a - b. Lam rieng R0
   (chinh) va R1 (do nhay).
3. Phan tang (muc 7, chi khi >= 5 video dong gop va >= 20 don vi; khong dat -> chi so dem), dinh nghia dung lai:
   - che khuat: su kien occlusion_eval (build_events, gaps=None; add_gt_eligibility, ngu canh 30 frame) qua
     src/kc48/su_kien_dev.su_kien_che; tang con has_full; don vi = su kien.
   - xe nho canh xe lon: src/kc48/su_kien_dev.su_kien_xe_nho (tieu chi khoa canh_xe_lon, R48D: 88 su kien, 42 du
     dieu kien / 7 video); don vi = su kien.
   - cua: track GT co loai_max in {cua, cua_gat} (kc48_audit quy_dao, R48B: 32 track / 5 video); do nhay loai_p90;
     don vi = track GT; chi so ty_le_giu_id, n_doi_id (src/kc48/phan_tich_dev.giu_id).
   Ghep GT-tracker chi tren frame co GT (occlusion_eval.match_video / match_frame, IoU >= 0,5), nen quy tac R1 (bo box
   o frame khong nhan) khong doi ket qua phan tang (test). PF: trung binh 3 seed trong tung don vi (cach R48D).
   Bootstrap theo video cua tang (B = 5000, default_rng(0) moi tang, chung cho moi doi chung), hieu = trung binh
   hieu theo don vi (ty so tong theo video).

    python -B -X utf8 src/r51_controlled/summarize.py chinh       # bang chinh + bootstrap R0/R1 (~1 phut)
    python -B -X utf8 src/r51_controlled/summarize.py phan-tang   # cham su kien/track cho 20 run (vai phut)
    python -B -X utf8 src/r51_controlled/summarize.py slide       # bang slide + source map (doc 2 buoc tren)
    python -B -X utf8 src/r51_controlled/summarize.py tat-ca
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
sys.path.insert(0, str(HERE))
import boot_stats as bs  # noqa: E402
import chay_buoc_b as cb  # noqa: E402

OUT = ROOT / "results/phase2/r51_controlled/summary"
EVAL = cb.EVAL
TRK = cb.GOC_MOI / "trackers" / cb.SPLIT
QUY_TAC = ("R0", "R1")
B_BOOT = 5000
SEED_BOOT = 0
METRICS = ["HOTA", "AssA", "DetA", "IDF1", "IDSW", "MOTA", "FP"]
CHINH = [("Q1", "bt-ekf", "bt-cv"), ("Q1", "bt-ukf", "bt-cv"), ("Q1", "bt-ukf", "bt-ekf"),
         ("Q1", "bt-pf100", "bt-ekf"), ("Q1", "bt-pf100-k1", "bt-pf100"), ("Q1", "bt-pf100-k1", "bt-ekf"),
         ("Q2", "bt-cv", "sort-cv30"), ("Q2", "bt-ekf", "sort-ekf30")]
Q1 = ["bt-cv", "bt-ekf", "bt-ukf", "bt-pf100", "bt-pf100-k1"]
Q2 = ["sort-cv30", "bt-cv", "sort-ekf30", "bt-ekf"]
PHU_LUC = ["bt-pf100-k2", "bt-pf500", "bt-pf500-k1"]
NGUONG_VIDEO, NGUONG_DON_VI = 5, 20
#: chi so tong hop cua bang phan tang theo bien the (bi bo khi tang thieu mau; so dem giu lai)
COT_TONG_HOP = ["n_no_match_before_tb", "ty_le_giu_moi_su_kien", "n_su_kien_co_truoc", "ty_le_giu_khi_co_truoc",
                "ty_le_lien_tuc_moi_su_kien", "ty_le_giu_id_tb", "n_doi_id_tb"]
TRANG_THAI = {"A": "approve (review_thu_cong_R51A_vong2/answer.json)",
              "B": "approve (review_thu_cong_R51B_vong2/answer.json)", "C": "cho review"}
TAP = "UA-DETRAC dev12 (phan train, 12 video); tap test KHONG chay trong R51"


def bien_the() -> dict[str, list[tuple[str, int | None]]]:
    """nhan -> [(dir, seed)] theo thu tu seed 0, 1, 2 (20 run so sanh cua B)."""
    out: dict[str, list] = {}
    for d in cb.dir_so_sanh():
        n, s = cb.nhan_cua(d)
        out.setdefault(n, []).append((d, s))
    for n in out:
        out[n].sort(key=lambda x: -1 if x[1] is None else x[1])
    return out


def nguon_cua(nhan: str) -> str:
    return "moi" if nhan in cb.MOI else "tai_dung"


def doc_raws(quy_tac: str, bt: dict) -> dict[str, list[dict]]:
    return {n: [bs.load_raw_fields(EVAL / quy_tac / d / "raw_fields.json") for d, _ in ds] for n, ds in bt.items()}


def chi_so_bootstrap(n: int, B: int = B_BOOT, seed: int = SEED_BOOT) -> np.ndarray:
    """(B, n): cung day voi boot_stats.paired_bootstrap / boot_seed.paired_bootstrap_seed."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, size=n) for _ in range(B)])


def _tb_seed(stacks: list[dict], idx=None) -> dict:
    cs = [bs.combine_stacked(s if idx is None else {f: s[f][idx] for f in s}) for s in stacks]
    return {m: float(np.mean([c[m] for c in cs])) for m in METRICS}


def bootstrap_chung(raws: dict[str, list[dict]], videos: list[str], idxs: np.ndarray) -> tuple[dict, dict]:
    """Diem (COMBINED tung seed -> trung binh) va mau bootstrap {nhan: {metric: (B,)}} cua moi bien the tren CUNG idxs."""
    stacks = {k: [bs._stack(r, videos) for r in rl] for k, rl in raws.items()}
    diem = {k: _tb_seed(st) for k, st in stacks.items()}
    mau = {k: {m: np.empty(len(idxs)) for m in METRICS} for k in raws}
    for i, idx in enumerate(idxs):
        for k, st in stacks.items():
            c = _tb_seed(st, idx)
            for m in METRICS:
                mau[k][m][i] = c[m]
    return diem, mau


def buoc_chinh() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    bt = bien_the()
    vids = cb.videos()
    idxs = chi_so_bootstrap(len(vids))
    np.save(OUT / "chi_so_bootstrap_video.npy", idxs)
    tung_seed, ref = [], pd.read_csv(EVAL / "combined_tung_run.csv")
    for qt in QUY_TAC:
        raws = doc_raws(qt, bt)
        diem, mau = bootstrap_chung(raws, vids, idxs)
        rows = []
        for n, ds in bt.items():
            per = [bs.combine(r, vids) for r in raws[n]]
            for (d, s), c in zip(ds, per):
                tung_seed.append({"quy_tac": qt, "nhan": n, "seed": s, "dir": d, **{m: c[m] for m in METRICS}})
            row = {"quy_tac": qt, "nhan": n, "cau_hoi": "+".join(q for q, l in (("Q1", Q1), ("Q2", Q2)) if n in l)
                   or "phu_luc", "nguon": nguon_cua(n), "n_seed": len(ds), "n_video": len(vids),
                   **{m: diem[n][m] for m in METRICS}}
            if len(per) > 1:  # mo ta, KHONG phai CI (giao thuc muc 7)
                for m in ("HOTA", "IDF1", "IDSW"):
                    row[f"{m}_seed_min_mo_ta"] = min(p[m] for p in per)
                    row[f"{m}_seed_max_mo_ta"] = max(p[m] for p in per)
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"bang_chinh_{qt}.csv", index=False)
        dc = []
        for q, a, b in CHINH:
            for m in METRICS:
                d_mau = mau[a][m] - mau[b][m]
                lo, hi = np.percentile(d_mau, [2.5, 97.5])
                dc.append({"quy_tac": qt, "cau_hoi": q, "a": a, "b": b, "chi_so": m, "a_gt": diem[a][m],
                           "b_gt": diem[b][m], "hieu": diem[a][m] - diem[b][m], "ci_lo": float(lo), "ci_hi": float(hi),
                           "ci_loai_0": bool(lo > 0 or hi < 0), "nguon_a": nguon_cua(a), "nguon_b": nguon_cua(b),
                           "n_video": len(vids), "B": B_BOOT, "rng_seed": SEED_BOOT})
        pd.DataFrame(dc).to_csv(OUT / f"doi_chieu_{qt}.csv", index=False)
    ts = pd.DataFrame(tung_seed)
    ts.to_csv(OUT / "bang_tung_seed.csv", index=False)
    # doi chieu voi combined_metrics.csv cua TrackEval (bang B): tai to hop tu raw_fields phai trung
    m = ts.merge(ref, on=["quy_tac", "dir"], suffixes=("", "_te"))
    lech = {k: float((m[k] - m[f"{k}_te"]).abs().max()) for k in ("HOTA", "AssA", "DetA", "IDF1", "IDSW", "MOTA", "FP")}
    kiem = {"n_dong": len(m), "lech_lon_nhat_vs_trackeval": lech, "dat": len(m) == 40 and max(lech.values()) <= 1e-9,
            "giay": round(time.perf_counter() - t0, 1)}
    (OUT / "kiem_bang_chinh.json").write_text(json.dumps(kiem, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(kiem, ensure_ascii=False))
    d0 = pd.read_csv(OUT / "doi_chieu_R0.csv")
    print(d0[d0.chi_so.isin(["HOTA", "IDF1", "IDSW"])][["cau_hoi", "a", "b", "chi_so", "hieu", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return 0 if kiem["dat"] else 1


# ---------------------------------------------------------------------------------------------------------------
def _sk():
    import su_kien_dev as sk
    sk.PROC = cb.GOC_MOI          # GT ban sao trung byte (test_gt_sao_trung_goc cua B); khong ghi
    return sk


def tap_don_vi() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """GT, su kien che khuat, su kien xe nho, metadata cua (video, gt_id), kiem so dem."""
    sk = _sk()
    gt = sk.doc_gt()
    ev_c = sk.su_kien_che(gt)
    ev_n, kiem = sk.su_kien_xe_nho(gt)
    qd = pd.read_csv(ROOT / "results/phase2/kc48_audit/_dem/quy_dao/detrac_train.csv")
    qd = qd[qd.video.isin(cb.videos())].rename(columns={"track": "gt_id"})[["video", "gt_id", "loai_max", "loai_p90"]]
    return gt, ev_c, ev_n, qd, kiem


def doc_trk(d: str, v: str) -> pd.DataFrame:
    import occlusion_eval as oe
    f = TRK / d / "data" / f"{v}.txt"
    return pd.read_csv(f, header=None, names=oe.TRK_COLS) if f.stat().st_size > 0 else pd.DataFrame(columns=oe.TRK_COLS)


def buoc_phan_tang() -> int:
    import occlusion_eval as oe
    import phan_tich_dev as pt
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    gt, ev_c, ev_n, qd, kiem = tap_don_vi()
    vids = cb.videos()
    print("kiem xe nho:", kiem)
    if not kiem["khop"]:
        print("[DUNG] su kien xe nho khong khop 88 / 65 / 9")
        return 2
    gts_pt = {v: pt._doc_mot(cb.GOC_MOI / cb.SPLIT / v / "gt" / "gt.txt", gt=True) for v in vids}
    ev_rows, tr_rows, gio = [], [], {}
    for n, ds in bien_the().items():
        for d, s in ds:
            t1 = time.perf_counter()
            mats = []
            for v in vids:
                tr = doc_trk(d, v)
                m = oe.match_video(gt[gt.video == v], tr)
                m["video"] = v
                mats.append(m)
                g = pt.giu_id(gts_pt[v], pt._doc_mot(TRK / d / "data" / f"{v}.txt", gt=False))
                g["video"], g["nhan"], g["seed"], g["dir"] = v, n, -1 if s is None else s, d
                tr_rows.append(g)
            match = pd.concat(mats, ignore_index=True)
            for ev in (ev_c, ev_n):
                r = oe.evaluate_events(ev.drop(columns=["tang", "track_lon"], errors="ignore"), match, gt, sk_ctx())
                r["tang"], r["nhan"], r["seed"], r["dir"] = ev["tang"].iloc[0], n, -1 if s is None else s, d
                ev_rows.append(r)
            gio[d] = round(time.perf_counter() - t1, 1)
            print(f"{d}: {gio[d]} s", flush=True)
    evr = pd.concat(ev_rows, ignore_index=True)
    trr = pd.concat(tr_rows, ignore_index=True).merge(qd, on=["video", "gt_id"], how="left")
    evr.to_csv(OUT / "don_vi_su_kien.csv", index=False)
    trr.to_csv(OUT / "don_vi_track.csv", index=False)
    ev_c.to_csv(OUT / "su_kien_che_khuat.csv", index=False)
    ev_n.to_csv(OUT / "su_kien_xe_nho.csv", index=False)
    (OUT / "kiem_phan_tang.json").write_text(json.dumps({"kiem_xe_nho": kiem, "giay_tung_run": gio,
                                                         "giay": round(time.perf_counter() - t0, 1)},
                                                        indent=1, ensure_ascii=False), encoding="utf-8")
    return tong_hop_phan_tang()


def sk_ctx() -> int:
    return _sk().CONTEXT


def _boot_hieu(d: pd.DataFrame, col: str, vids: list[str], idxs: np.ndarray) -> tuple[float, float, float]:
    """Hieu trung binh theo don vi + CI 95 % bootstrap theo video (ty so tong; video khong co don vi dong gop 0)."""
    tong = d.groupby("video")[col].agg(["sum", "size"]).reindex(vids).fillna(0.0)
    sm, sz = tong["sum"].to_numpy(), tong["size"].to_numpy()
    bo = sm[idxs].sum(1) / np.maximum(sz[idxs].sum(1), 1)
    return float(sm.sum() / sz.sum()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))


def du_mau(n_video: int, n_don_vi: int) -> bool:
    return n_video >= NGUONG_VIDEO and n_don_vi >= NGUONG_DON_VI


def tong_hop_phan_tang() -> int:
    evr = pd.read_csv(OUT / "don_vi_su_kien.csv")
    trr = pd.read_csv(OUT / "don_vi_track.csv")
    r = evr[evr.eligible].copy()
    r["giu"] = (r.outcome == "preserved").astype(float)
    r["co_truoc"] = (r.outcome != "no_match_before").astype(float)
    r["lien_tuc"] = r.id_continuous.astype(float)
    key = ["tang", "video", "track_id", "event_id", "start_frame"]
    tb = r.groupby(["nhan"] + key + ["has_full"], dropna=False)[["giu", "co_truoc", "lien_tuc"]].mean().reset_index()
    tb["co_truoc_moi_seed"] = tb.co_truoc == 1.0
    tang_su_kien = {"che_khuat": tb[tb.tang == "che_khuat"],
                    "che_khuat_has_full": tb[(tb.tang == "che_khuat") & tb.has_full],
                    "xe_nho_canh_xe_lon": tb[tb.tang == "xe_nho_canh_xe_lon"]}
    nhom = {"cua": "cua", "cua_gat": "cua"}
    trr["cua_max"] = trr.loai_max.map(nhom).fillna("khac")
    trr["cua_p90"] = trr.loai_p90.map(nhom).fillna("khac")
    tt = trr.groupby(["nhan", "video", "gt_id", "cua_max", "cua_p90"])[["ty_le_giu_id", "n_doi_id"]].mean().reset_index()
    tang_track = {"cua_loai_max": tt[tt.cua_max == "cua"], "cua_loai_p90_do_nhay": tt[tt.cua_p90 == "cua"]}
    nhans = list(bien_the())
    rows, dc = [], []
    for ten, d in tang_su_kien.items():
        vids = sorted(d.video.unique())
        idxs = chi_so_bootstrap(len(vids))
        for n in nhans:
            a = d[d.nhan == n]
            ct = a[a.co_truoc_moi_seed]
            rows.append({"tang": ten, "don_vi": "su_kien", "nhan": n, "nguon": nguon_cua(n),
                         "n_video": a.video.nunique(), "n_don_vi": len(a),
                         "n_track": a[["video", "track_id"]].drop_duplicates().shape[0],
                         "n_no_match_before_tb": float((1 - a.co_truoc).sum()), "ty_le_giu_moi_su_kien": a.giu.mean(),
                         "n_su_kien_co_truoc": len(ct), "ty_le_giu_khi_co_truoc": ct.giu.mean(),
                         "ty_le_lien_tuc_moi_su_kien": a.lien_tuc.mean()})
        for q, na, nb in CHINH:
            a, b = d[d.nhan == na].set_index(key), d[d.nhan == nb].set_index(key)
            row = {"tang": ten, "don_vi": "su_kien", "cau_hoi": q, "a": na, "b": nb, "n_video": len(vids),
                   "n_don_vi": len(a)}
            if du_mau(len(vids), len(a)):
                dd = (a.giu - b.giu.reindex(a.index)).rename("d").reset_index()
                row["hieu_giu_moi_su_kien"], row["ci_lo"], row["ci_hi"] = _boot_hieu(dd, "d", vids, idxs)
                chung = a.index[a.co_truoc_moi_seed & b.co_truoc_moi_seed.reindex(a.index).fillna(False).astype(bool)]
                dcc = (a.giu.loc[chung] - b.giu.loc[chung]).rename("d").reset_index()
                row["n_su_kien_co_truoc_ca_hai"] = len(dcc)
                row["n_video_co_truoc_ca_hai"] = dcc.video.nunique()
                if du_mau(dcc.video.nunique(), len(dcc)):
                    row["hieu_giu_khi_co_truoc"], row["ci_lo_co_truoc"], row["ci_hi_co_truoc"] = \
                        _boot_hieu(dcc, "d", vids, idxs)
                else:
                    row["ghi_chu"] = "co_truoc ca hai: khong du mau, chi so dem"
            else:
                row["ghi_chu"] = f"< {NGUONG_VIDEO} video hoac < {NGUONG_DON_VI} don vi: chi so dem"
            dc.append(row)
    for ten, d in tang_track.items():
        vids = sorted(d.video.unique())
        idxs = chi_so_bootstrap(len(vids))
        for n in nhans:
            a = d[d.nhan == n]
            rows.append({"tang": ten, "don_vi": "track_gt", "nhan": n, "nguon": nguon_cua(n),
                         "n_video": a.video.nunique(), "n_don_vi": len(a), "n_track": len(a),
                         "ty_le_giu_id_tb": a.ty_le_giu_id.mean(), "n_doi_id_tb": a.n_doi_id.mean()})
        for q, na, nb in CHINH:
            a = d[d.nhan == na].set_index(["video", "gt_id"])
            b = d[d.nhan == nb].set_index(["video", "gt_id"])
            row = {"tang": ten, "don_vi": "track_gt", "cau_hoi": q, "a": na, "b": nb, "n_video": len(vids),
                   "n_don_vi": len(a)}
            if du_mau(len(vids), len(a)):
                dd = (a.ty_le_giu_id - b.ty_le_giu_id.reindex(a.index)).rename("d").reset_index()
                row["hieu_ty_le_giu_id"], row["ci_lo"], row["ci_hi"] = _boot_hieu(dd, "d", vids, idxs)
                dn = (a.n_doi_id - b.n_doi_id.reindex(a.index)).rename("d").reset_index()
                row["hieu_n_doi_id"], row["ci_lo_n_doi_id"], row["ci_hi_n_doi_id"] = _boot_hieu(dn, "d", vids, idxs)
            else:
                row["ghi_chu"] = f"< {NGUONG_VIDEO} video hoac < {NGUONG_DON_VI} don vi: chi so dem"
            dc.append(row)
    # R51C-F1: tang thieu mau (< 5 video hoac < 20 don vi) chi giu so dem / metadata, bo moi chi so tong hop
    pv = pd.DataFrame(rows)
    thieu = ~pv.apply(lambda r: du_mau(int(r.n_video), int(r.n_don_vi)), axis=1)
    pv.loc[thieu, [c for c in COT_TONG_HOP if c in pv]] = np.nan
    pv["ghi_chu"] = np.where(thieu, f"< {NGUONG_VIDEO} video hoac < {NGUONG_DON_VI} don vi: chi so dem", "")
    pv.to_csv(OUT / "phan_tang_bien_the.csv", index=False)
    pdc = pd.DataFrame(dc)
    for c in ("ci_lo", "ci_hi"):
        pdc[c] = pdc[c].astype(float) if c in pdc else np.nan
    # khong co CI (tang thieu mau) -> de trong, khong ghi False (de doc nham thanh "CI chua 0")
    pdc["ci_loai_0"] = np.where(pdc.ci_lo.notna(), ((pdc.ci_lo > 0) | (pdc.ci_hi < 0)), None)
    pdc.to_csv(OUT / "phan_tang_doi_chieu.csv", index=False)
    co = []
    for ten, p in (("che_khuat", "su_kien_che_khuat.csv"), ("xe_nho_canh_xe_lon", "su_kien_xe_nho.csv")):
        ev = pd.read_csv(OUT / p)
        co.append({"tang": ten, "don_vi": "su_kien", "n_tat_ca": len(ev), "n_du_dk": int(ev.eligible.sum()),
                   "n_video_du_dk": int(ev[ev.eligible].video.nunique()),
                   "n_track_du_dk": int(ev[ev.eligible][["video", "track_id"]].drop_duplicates().shape[0]),
                   "n_has_full_du_dk": int((ev.eligible & ev.has_full).sum()),
                   "n_video_has_full_du_dk": int(ev[ev.eligible & ev.has_full].video.nunique()),
                   **{f"loai_{k}": int(v) for k, v in ev.loc[~ev.eligible, "exclusion_reason"].value_counts().items()}})
    e0 = tt[tt.nhan == "bt-ekf"]
    for ten, col in (("cua_loai_max", "cua_max"), ("cua_loai_p90_do_nhay", "cua_p90")):
        x = e0[e0[col] == "cua"]
        co.append({"tang": ten, "don_vi": "track_gt", "n_tat_ca": len(x), "n_du_dk": len(x),
                   "n_video_du_dk": x.video.nunique(), "n_track_du_dk": len(x), "mau_so_track_gt_dev12": len(e0)})
    pd.DataFrame(co).to_csv(OUT / "co_mau_phan_tang.csv", index=False)
    print(pd.DataFrame(co).to_string(index=False))
    print(pdc.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    return 0


# ---------------------------------------------------------------------------------------------------------------
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kiem_nguon() -> dict:
    """Moi eval_meta (40) cung evaluator / GT / cache; quy_tac dung thu muc; module ma chay."""
    rows = []
    for qt in QUY_TAC:
        for d in cb.dir_so_sanh():
            em = json.loads((EVAL / qt / d / "eval_meta.json").read_text(encoding="utf-8"))
            gt, ca = em["gt"], em["cache"]
            rows.append({"quy_tac": qt, "dir": d, "evaluator": json.dumps(em["evaluator"], sort_keys=True),
                         "quy_tac_meta": em.get("quy_tac"), "gt_sha256": gt["gt_sha256"],
                         "seqinfo_sha256": gt["seqinfo_sha256"], "npz_sha256": ca["npz_sha256"]})
    df = pd.DataFrame(rows)
    return {"n_eval_meta": len(df), "evaluator": sorted(df.evaluator.unique()),
            "quy_tac_dung_thu_muc": bool((df.quy_tac == df.quy_tac_meta).all()),
            "gt_sha256": sorted(df.gt_sha256.unique()), "seqinfo_sha256": sorted(df.seqinfo_sha256.unique()),
            "npz_sha256": sorted(df.npz_sha256.unique()),
            "dat": len(df) == 40 and df.evaluator.nunique() == 1 and df.gt_sha256.nunique() == 1
            and df.npz_sha256.nunique() == 1 and bool((df.quy_tac == df.quy_tac_meta).all())}


def buoc_slide() -> int:
    kn = kiem_nguon()
    (OUT / "kiem_nguon.json").write_text(json.dumps(kn, indent=1, ensure_ascii=False), encoding="utf-8")
    bc = pd.read_csv(OUT / "bang_chinh_R0.csv").set_index("nhan")
    d0 = pd.read_csv(OUT / "doi_chieu_R0.csv")
    d1 = pd.read_csv(OUT / "doi_chieu_R1.csv")
    nhan_slide = {"bt-cv": "ByteTrack + KF-CV", "bt-ekf": "ByteTrack + EKF-CTRV", "bt-ukf": "ByteTrack + UKF-CTRV",
                  "bt-pf100": "ByteTrack + PF-CTRV N=100", "bt-pf100-k1": "ByteTrack + PF-CTRV N=100 + K1",
                  "sort-cv30": "SORT-kieu (KF-CV, max_age 30)", "sort-ekf30": "SORT-kieu (EKF-CTRV, max_age 30)",
                  "bt-pf100-k2": "ByteTrack + PF N=100 + K2 (phu luc)", "bt-pf500": "ByteTrack + PF N=500 (phu luc)",
                  "bt-pf500-k1": "ByteTrack + PF N=500 + K1 (phu luc)"}

    def bang(nhans, q):
        out = []
        for n in nhans:
            r = bc.loc[n]
            out.append({"cau_hoi": q, "bien_the": nhan_slide[n], "nhan": n, "nguon": r.nguon, "n_seed": int(r.n_seed),
                        "HOTA": round(r.HOTA, 4), "AssA": round(r.AssA, 4), "DetA": round(r.DetA, 4),
                        "IDF1": round(r.IDF1, 4), "IDSW": round(r.IDSW, 1), "tap": TAP, "quy_tac": "R0",
                        "trang_thai": "C cho review"})
        return pd.DataFrame(out)

    def bang_hieu(q):
        out = []
        for (_, a, b), g0 in d0[d0.cau_hoi == q].groupby(["cau_hoi", "a", "b"], sort=False):
            g0 = g0.set_index("chi_so")
            g1 = d1[(d1.a == a) & (d1.b == b)].set_index("chi_so")
            row = {"cau_hoi": q, "doi_chung": f"{a} - {b}", "nguon": f"{nguon_cua(a)} / {nguon_cua(b)}"}
            for m in ("HOTA", "AssA", "DetA", "IDF1", "IDSW"):
                x = g0.loc[m]
                nd = 1 if m == "IDSW" else 4
                row[f"d{m}"] = round(x.hieu, nd)
                row[f"d{m}_ci"] = f"[{x.ci_lo:+.{nd}f}; {x.ci_hi:+.{nd}f}]"
                row[f"d{m}_ci_loai_0"] = bool(x.ci_loai_0)
            row["R1_dHOTA"] = round(g1.loc["HOTA"].hieu, 4)
            row["R1_dHOTA_ci"] = f"[{g1.loc['HOTA'].ci_lo:+.4f}; {g1.loc['HOTA'].ci_hi:+.4f}]"
            row["R1_cung_ket_luan_HOTA"] = bool(g1.loc["HOTA"].ci_loai_0) == bool(g0.loc["HOTA"].ci_loai_0)
            out.append(row)
        return pd.DataFrame(out)

    s1 = pd.concat([bang(Q1, "Q1"), bang(PHU_LUC, "Q1 phu luc")], ignore_index=True)
    s1.to_csv(OUT / "slide_q1_bo_loc_bytetrack.csv", index=False)
    bang(Q2, "Q2").to_csv(OUT / "slide_q2_sort_vs_bytetrack.csv", index=False)
    hq = pd.concat([bang_hieu("Q1"), bang_hieu("Q2")], ignore_index=True)
    hq.to_csv(OUT / "slide_hieu_so_ci.csv", index=False)
    pt_path = OUT / "phan_tang_doi_chieu.csv"
    if pt_path.exists():
        p = pd.read_csv(pt_path)
        p = p[p.ci_lo.notna()]
        sp = []
        for r in p.itertuples():
            col = "hieu_giu_moi_su_kien" if r.don_vi == "su_kien" else "hieu_ty_le_giu_id"
            sp.append({"tang": r.tang, "don_vi": r.don_vi, "n_don_vi": r.n_don_vi, "n_video": r.n_video,
                       "doi_chung": f"{r.a} - {r.b}", "chi_so": "giu ID / su kien" if r.don_vi == "su_kien"
                       else "ty le frame giu ID chinh / track", "hieu": round(getattr(r, col), 4),
                       "ci": f"[{r.ci_lo:+.4f}; {r.ci_hi:+.4f}]", "ci_loai_0": bool(r.ci_lo > 0 or r.ci_hi < 0)})
        pd.DataFrame(sp).to_csv(OUT / "slide_phan_tang.csv", index=False)
    # source map / provenance
    src = {}
    for qt in QUY_TAC:
        for d in cb.dir_so_sanh():
            n, s = cb.nhan_cua(d)
            for f in ("raw_fields.json", "eval_meta.json", "combined_metrics.csv"):
                p = EVAL / qt / d / f
                src[str(p.relative_to(ROOT)).replace("\\", "/")] = {"sha256": _sha(p), "nhan": n, "seed": s,
                                                                    "nguon": nguon_cua(n), "quy_tac": qt}
    ma = {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p) for p in
          (HERE / "summarize.py", HERE / "chay_buoc_b.py", ROOT / "src/boot_stats.py", ROOT / "src/kc48/boot_seed.py",
           ROOT / "src/kc48/su_kien_dev.py", ROOT / "src/kc48/phan_tich_dev.py", ROOT / "src/occlusion_eval.py",
           ROOT / "src/kc48/audit_du_lieu.py")}
    phu = [ROOT / "results/phase2/kc48_audit/_dem/quy_dao/detrac_train.csv",
           ROOT / "results/phase2/kc48_audit/tieu_chi_cong_bo.json", ROOT / "results/phase2/manifests/detrac_dev.json"]
    dau_ra = {p.name: _sha(p) for p in sorted(OUT.glob("*.csv")) if p.name.startswith(("bang_", "doi_chieu_",
                                                                                        "phan_tang_", "slide_",
                                                                                        "co_mau_"))}
    sm = {"mo_ta": "R51C source map: moi so trong summary/ tinh tu cac file duoi day (chi doc).",
          "tap": TAP, "trang_thai": TRANG_THAI,
          "bootstrap": {"don_vi": "video", "n_video": 12, "B": B_BOOT, "rng": f"numpy default_rng({SEED_BOOT})",
                        "ci": "95 % phan vi", "chi_so_video": "chi_so_bootstrap_video.npy (dung chung moi doi chung)",
                        "pf": "COMBINED tung seed 0,1,2 roi trung binh, trong moi mau"},
          "quy_tac_cham": {"R0": "TrackEval MotChallenge2DBox hien tai (chinh)",
                           "R1": "bo box tracker o 572 frame khong <frame> (do nhay, giao thuc muc 3.4)"},
          "dau_vao_eval": src, "ma": ma, "dau_vao_phan_tang": {str(p.relative_to(ROOT)).replace("\\", "/"): _sha(p)
                                                               for p in phu},
          "dau_vao_tracking": "results/phase2/r51_controlled/runs/_processed/trackers/DETRAC-all/<dir>/data/*.txt "
                              "(hash theo eval_meta.tracking cua tung run)",
          "dau_ra": dau_ra, "kiem_nguon": kn}
    (OUT / "source_map.json").write_text(json.dumps(sm, indent=1, ensure_ascii=False), encoding="utf-8")
    print(s1.to_string(index=False))
    print(hq[["doi_chung", "dHOTA", "dHOTA_ci", "dIDF1", "dIDF1_ci", "dIDSW", "dIDSW_ci", "R1_cung_ket_luan_HOTA"]]
          .to_string(index=False))
    print("kiem nguon:", kn["dat"])
    return 0 if kn["dat"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("buoc", choices=["chinh", "phan-tang", "tong-hop-phan-tang", "slide", "tat-ca"])
    a = ap.parse_args()
    import os
    os.chdir(ROOT)
    if a.buoc == "tat-ca":
        for f in (buoc_chinh, buoc_phan_tang, buoc_slide):
            rc = f()
            if rc:
                return rc
        return 0
    return {"chinh": buoc_chinh, "phan-tang": buoc_phan_tang, "tong-hop-phan-tang": tong_hop_phan_tang,
            "slide": buoc_slide}[a.buoc]()


if __name__ == "__main__":
    raise SystemExit(main())
