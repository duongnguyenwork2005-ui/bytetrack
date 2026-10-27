"""
test_summary.py -- R51C: test truc tiep cho summarize.py tren DAU RA THAT (chi doc).

Moi test tu tinh lai tu file goc (raw_fields, combined_metrics TrackEval, file tracking, ket qua R48D), khong tin vao
JSON kiem cua summarize.py. Chay sau `summarize.py tat-ca`:
    python -B -X utf8 src/r51_controlled/test_summary.py     (exit 0 = moi test PASS)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "kc48"))

import summarize as sm  # noqa: E402
import chay_buoc_b as cb  # noqa: E402
import boot_stats as bs  # noqa: E402
import boot_seed as bsd  # noqa: E402

OUT = sm.OUT
ROOT = sm.ROOT
VIDS = cb.videos()
R48D = ROOT / "results/phase2/kc48_pf_dev"
#: nhan R51 -> ten bien the R48D (chi run tai dung kc48)
KC48 = {"bt-ekf": "ekf", "bt-pf100": "pf100", "bt-pf100-k1": "pf100-k1", "bt-pf100-k2": "pf100-k2",
        "bt-pf500": "pf500", "bt-pf500-k1": "pf500-k1"}


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


# ---------------------------------------------------------------------------
# Thiet ke: doi chung, bien the, nguon
# ---------------------------------------------------------------------------
def test_doi_chung_dung_giao_thuc():
    """CHINH == hang 'Doi chung chinh' cua giao thuc muc 1 (doc tu file da approve)."""
    dong = next(x for x in (ROOT / "handoff/logs/review51/01_giao_thuc.md").read_text(encoding="utf-8").splitlines()
                if x.startswith("| Đối chứng chính"))
    o = [c.strip() for c in dong.strip("|").split("|")[1:]]
    ten = {"ekf": "bt-ekf", "cv": "bt-cv", "ukf": "bt-ukf", "pf100": "bt-pf100", "pf100k1": "bt-pf100-k1"}
    q1 = [tuple(ten[t] for t in x.strip().split("−")) for x in o[0].split(",")]
    q2 = [tuple(x.strip().split("−")) for x in o[1].split(",")]
    assert [("Q1",) + p for p in q1] + [("Q2",) + p for p in q2] == sm.CHINH, (q1, q2)


def test_bien_the_va_nguon():
    bt = sm.bien_the()
    assert sum(len(v) for v in bt.values()) == 20
    for n, ds in bt.items():
        seeds = [s for _, s in ds]
        assert seeds == ([None] if len(ds) == 1 else [0, 1, 2]), (n, seeds)
        assert sm.nguon_cua(n) == ("moi" if n in ("bt-cv", "bt-ukf", "sort-cv30", "sort-ekf30") else "tai_dung")
        for d, _ in ds:
            assert d.startswith("p2-r51_controlled-A-" if sm.nguon_cua(n) == "moi" else "p2-kc48_pf_dev-A-"), d
    for qt in sm.QUY_TAC:
        b = _csv(f"bang_chinh_{qt}.csv").set_index("nhan")
        assert b.loc["bt-ekf", "nguon"] == "tai_dung" and b.loc["bt-cv", "nguon"] == "moi"


# ---------------------------------------------------------------------------
# Bang chinh + bootstrap
# ---------------------------------------------------------------------------
def test_diem_trung_trackeval_tung_seed_roi_trung_binh():
    """Diem bang chinh == trung binh theo seed cua combined_metrics.csv TrackEval tung run (moi quy tac)."""
    for qt in sm.QUY_TAC:
        b = _csv(f"bang_chinh_{qt}.csv").set_index("nhan")
        for n, ds in sm.bien_the().items():
            te = [pd.read_csv(cb.EVAL / qt / d / "combined_metrics.csv").iloc[0] for d, _ in ds]
            for m in ("HOTA", "AssA", "DetA", "IDF1", "IDSW", "MOTA", "FP"):
                assert abs(b.loc[n, m] - float(np.mean([float(t[m]) for t in te]))) <= 1e-9, (qt, n, m)


def test_thu_tu_seed_pf_la_combined_truoc():
    """PF: COMBINED tung seed roi trung binh (giao thuc muc 7), KHAC trung binh truong tho roi COMBINED."""
    b = _csv("bang_chinh_R0.csv").set_index("nhan")
    ds = sm.bien_the()["bt-pf100"]
    raws = [bs.load_raw_fields(cb.EVAL / "R0" / d / "raw_fields.json") for d, _ in ds]
    dung = np.mean([bs.combine(r, VIDS)["HOTA"] for r in raws])
    sai = bs.combine(bs.average_seeds(raws), VIDS)["HOTA"]
    assert abs(b.loc["bt-pf100", "HOTA"] - dung) < 1e-12
    assert abs(dung - sai) > 1e-6, "hai thu tu trung nhau -> test khong phan biet duoc"


def test_bootstrap_trung_boot_seed():
    """Moi doi chung (R0 va R1) == boot_seed.paired_bootstrap_seed(B 5000, seed 0) tinh doc lap; IDSW, HOTA, IDF1."""
    for qt in sm.QUY_TAC:
        dc = _csv(f"doi_chieu_{qt}.csv")
        bt = sm.bien_the()
        for q, a, b in sm.CHINH:
            ra = [bs.load_raw_fields(cb.EVAL / qt / d / "raw_fields.json") for d, _ in bt[a]]
            rb = [bs.load_raw_fields(cb.EVAL / qt / d / "raw_fields.json") for d, _ in bt[b]]
            r = bsd.paired_bootstrap_seed(ra, rb, VIDS, 5000, 0, ["HOTA", "IDF1", "IDSW"])
            for m in ("HOTA", "IDF1", "IDSW"):
                x = dc[(dc.a == a) & (dc.b == b) & (dc.chi_so == m)].iloc[0]
                for k, kk in (("hieu", "diff"), ("ci_lo", "ci_low"), ("ci_hi", "ci_high")):
                    assert abs(x[k] - r[m][kk]) <= 1e-12, (qt, a, b, m, k, x[k], r[m][kk])


def test_chi_so_video_dung_chung_va_dung_day():
    idx = np.load(OUT / "chi_so_bootstrap_video.npy")
    assert idx.shape == (5000, 12)
    rng = np.random.default_rng(0)
    for i in range(5000):
        assert np.array_equal(idx[i], rng.integers(0, 12, size=12))


def test_ci_khong_phai_min_max_seed():
    """doi_chieu khong co cot khoang seed; khoang seed trong bang chinh co hau to _mo_ta."""
    for qt in sm.QUY_TAC:
        assert not [c for c in _csv(f"doi_chieu_{qt}.csv").columns if "seed" in c and c != "rng_seed"]
        cot = [c for c in _csv(f"bang_chinh_{qt}.csv").columns if "seed_m" in c]
        assert cot and all(c.endswith("_mo_ta") for c in cot), cot


# ---------------------------------------------------------------------------
# Phan tang
# ---------------------------------------------------------------------------
def test_tap_su_kien_trung_r48d():
    """Bang su kien (GT) trung bang R48D; so dem 717/184/12/137 va 88/42/7/29; cua 32/5, p90 8/4."""
    for p in ("su_kien_che_khuat.csv", "su_kien_xe_nho.csv"):
        a, b = _csv(p), pd.read_csv(R48D / "su_kien" / p)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    co = _csv("co_mau_phan_tang.csv").set_index("tang")
    assert tuple(co.loc["che_khuat", ["n_tat_ca", "n_du_dk", "n_video_du_dk", "n_track_du_dk"]]) == (717, 184, 12, 137)
    assert tuple(co.loc["xe_nho_canh_xe_lon", ["n_tat_ca", "n_du_dk", "n_video_du_dk", "n_track_du_dk"]]) == (88, 42, 7, 29)
    assert int(co.loc["che_khuat", "n_has_full_du_dk"]) == 33
    assert tuple(co.loc["cua_loai_max", ["n_du_dk", "n_video_du_dk"]]) == (32, 5)
    assert tuple(co.loc["cua_loai_p90_do_nhay", ["n_du_dk", "n_video_du_dk"]]) == (8, 4)


def test_phan_tang_tai_tao_r48d_tren_run_tai_dung():
    """Run kc48 tai dung: ty le giu ID su kien va ty le giu ID track cua trung R48D (cung dinh nghia, cung file)."""
    pv = _csv("phan_tang_bien_the.csv")
    r48 = pd.read_csv(R48D / "su_kien" / "bang_su_kien.csv")
    pt = pd.read_csv(R48D / "phan_tich" / "phan_tang.csv")
    for n, cu in KC48.items():
        for tang in ("che_khuat", "che_khuat_has_full", "xe_nho_canh_xe_lon"):
            a = pv[(pv.tang == tang) & (pv.nhan == n)].iloc[0]
            b = r48[(r48.tang == tang) & (r48.bien_the == cu)].iloc[0]
            for k in ("ty_le_giu_moi_su_kien", "ty_le_giu_khi_co_truoc", "ty_le_lien_tuc_moi_su_kien"):
                assert abs(a[k] - b[k]) < 1e-12, (n, tang, k)
            assert int(a.n_su_kien_co_truoc) == int(b.n_su_kien_co_truoc) and int(a.n_don_vi) == int(b.n_su_kien_du_dk)
        a = pv[(pv.tang == "cua_loai_max") & (pv.nhan == n)].iloc[0]
        b = pt[(pt.tang == "nhom_cua") & (pt.gia_tri == "cua") & (pt.bien_the == cu)].iloc[0]
        assert abs(a.ty_le_giu_id_tb - b.ty_le_giu_id_tb) < 1e-12 and int(a.n_track) == int(b.n_track), n
        assert abs(a.n_doi_id_tb - b.n_doi_id_tb) < 1e-12, n


def test_r1_khong_doi_dau_vao_phan_tang():
    """Moi run: dong tracker tren frame co GT trung nhau giua R0 va ban R1 -> ghep GT-tracker (chi frame GT) trung."""
    import occlusion_eval as oe
    for d in cb.dir_so_sanh():
        for v in VIDS:
            g = set(pd.read_csv(cb.GOC_MOI / cb.SPLIT / v / "gt" / "gt.txt", header=None)[0])
            a = pd.read_csv(sm.TRK / d / "data" / f"{v}.txt", header=None, names=oe.TRK_COLS + ["a", "b", "c"])
            b = pd.read_csv(sm.TRK / f"{d}-R1" / "data" / f"{v}.txt", header=None, names=oe.TRK_COLS + ["a", "b", "c"])
            a, b = a[a.frame.isin(g)].reset_index(drop=True), b[b.frame.isin(g)].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)


def test_bootstrap_phan_tang_tinh_lai():
    """Mot doi chung su kien + mot doi chung track tinh lai tu don vi tho bang vong lap thuong."""
    evr = _csv("don_vi_su_kien.csv")
    pdc = _csv("phan_tang_doi_chieu.csv")
    r = evr[evr.eligible & (evr.tang == "che_khuat")].copy()
    r["giu"] = (r.outcome == "preserved").astype(float)
    key = ["video", "track_id", "event_id", "start_frame"]
    g = r.groupby(["nhan"] + key).giu.mean()
    d = (g.loc["bt-pf100"] - g.loc["bt-ekf"]).reset_index()
    vids = sorted(r.video.unique())
    idx = np.random.default_rng(0)
    bo = []
    sm_v = d.groupby("video").giu.sum().reindex(vids).fillna(0).to_numpy()
    sz_v = d.groupby("video").giu.size().reindex(vids).fillna(0).to_numpy()
    for _ in range(5000):
        i = idx.integers(0, len(vids), size=len(vids))
        bo.append(sm_v[i].sum() / max(sz_v[i].sum(), 1))
    x = pdc[(pdc.tang == "che_khuat") & (pdc.a == "bt-pf100") & (pdc.b == "bt-ekf")].iloc[0]
    assert abs(x.hieu_giu_moi_su_kien - d.giu.mean()) < 1e-12
    assert abs(x.ci_lo - np.percentile(bo, 2.5)) < 1e-12 and abs(x.ci_hi - np.percentile(bo, 97.5)) < 1e-12
    trr = _csv("don_vi_track.csv")
    t = trr[trr.loai_max.isin(["cua", "cua_gat"])].groupby(["nhan", "video", "gt_id"]).ty_le_giu_id.mean()
    dt = (t.loc["bt-cv"] - t.loc["sort-cv30"]).reset_index()
    y = pdc[(pdc.tang == "cua_loai_max") & (pdc.a == "bt-cv") & (pdc.b == "sort-cv30")].iloc[0]
    assert abs(y.hieu_ty_le_giu_id - dt.ty_le_giu_id.mean()) < 1e-12


def test_nguong_mau():
    """Tang/doi chung < 5 video hoac < 20 don vi: khong co CI (chi so dem)."""
    pdc = _csv("phan_tang_doi_chieu.csv")
    nho = pdc[(pdc.n_video < 5) | (pdc.n_don_vi < 20)]
    assert len(nho) and nho.ci_lo.isna().all() and (nho.tang == "cua_loai_p90_do_nhay").all()
    lon = pdc[(pdc.n_video >= 5) & (pdc.n_don_vi >= 20)]
    assert lon.ci_lo.notna().all() and set(lon.tang) == {"che_khuat", "che_khuat_has_full", "xe_nho_canh_xe_lon",
                                                         "cua_loai_max"}
    c = lon.dropna(subset=["n_su_kien_co_truoc_ca_hai"])
    thieu = c[(c.n_video_co_truoc_ca_hai < 5) | (c.n_su_kien_co_truoc_ca_hai < 20)]
    assert thieu.ci_lo_co_truoc.isna().all()


def test_nguong_mau_moi_bang_xuat():
    """R51C-F1: moi bang phan tang xuat ra -- tang thieu mau khong co chi so tong hop / hieu / CI, chi so dem.
    Ca cu the: cua_loai_p90_do_nhay 8 track / 4 video. Bien nguong: (4, 20), (5, 19) thieu; (5, 20) du."""
    assert not sm.du_mau(4, 20) and not sm.du_mau(5, 19) and sm.du_mau(5, 20) and sm.du_mau(12, 184)
    pv = _csv("phan_tang_bien_the.csv")
    p90 = pv[pv.tang == "cua_loai_p90_do_nhay"]
    assert len(p90) == 10 and (p90.n_don_vi == 8).all() and (p90.n_video == 4).all()
    for ten, df in (("phan_tang_bien_the", pv), ("phan_tang_doi_chieu", _csv("phan_tang_doi_chieu.csv"))):
        tong_hop = [c for c in df.columns if c.startswith(("ty_le_", "hieu_", "ci_lo", "ci_hi", "n_doi_id",
                                                           "n_no_match", "n_su_kien_co_truoc"))]
        assert tong_hop, ten
        nho = df[(df.n_video < 5) | (df.n_don_vi < 20)]
        # 10 bien the o bang theo bien the; 8 doi chung chinh o bang doi chieu
        assert len(nho) == (10 if ten == "phan_tang_bien_the" else len(sm.CHINH)), (ten, len(nho))
        assert set(nho.tang) == {"cua_loai_p90_do_nhay"}, ten
        assert nho[tong_hop].isna().all().all(), (ten, nho[tong_hop].dropna(axis=1, how="all").columns.tolist())
        assert nho.ghi_chu.str.contains("chi so dem").all(), ten
        lon = df[(df.n_video >= 5) & (df.n_don_vi >= 20)]
        assert not lon.ghi_chu.fillna("").str.contains("< 5 video").any(), ten
    sp = _csv("slide_phan_tang.csv")
    assert "cua_loai_p90_do_nhay" not in set(sp.tang) and ((sp.n_video >= 5) & (sp.n_don_vi >= 20)).all()


# ---------------------------------------------------------------------------
# Nguon, slide
# ---------------------------------------------------------------------------
def test_source_map_hash_va_evaluator():
    smap = json.loads((OUT / "source_map.json").read_text(encoding="utf-8"))
    assert len(smap["dau_vao_eval"]) == 120
    for k, v in {**{k: v["sha256"] for k, v in smap["dau_vao_eval"].items()}, **smap["ma"],
                 **smap["dau_vao_phan_tang"]}.items():
        assert hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v, k
    for k, v in smap["dau_ra"].items():
        assert hashlib.sha256((OUT / k).read_bytes()).hexdigest() == v, k
    kn = smap["kiem_nguon"]
    assert kn["dat"] and kn["evaluator"] == ['{"dataset_class": "mot", "trackeval": "1.3.0"}'], kn["evaluator"]
    assert kn["gt_sha256"] == ["07de8402ca430b84545844256f121a2fff4f16123f38a626555e769af5f3d31f"]
    assert all(v["nguon"] == sm.nguon_cua(v["nhan"]) for v in smap["dau_vao_eval"].values())


def test_bang_slide_khop_doi_chieu():
    s = _csv("slide_hieu_so_ci.csv")
    d0 = _csv("doi_chieu_R0.csv")
    assert len(s) == len(sm.CHINH)
    for r in s.itertuples():
        a, b = r.doi_chung.split(" - ")
        x = d0[(d0.a == a) & (d0.b == b) & (d0.chi_so == "HOTA")].iloc[0]
        assert abs(r.dHOTA - round(x.hieu, 4)) < 1e-12 and r.dHOTA_ci == f"[{x.ci_lo:+.4f}; {x.ci_hi:+.4f}]"
        assert bool(r.dHOTA_ci_loai_0) == bool(x.ci_loai_0)
    q1 = _csv("slide_q1_bo_loc_bytetrack.csv")
    q2 = _csv("slide_q2_sort_vs_bytetrack.csv")
    assert set(q1.cau_hoi) == {"Q1", "Q1 phu luc"} and set(q2.cau_hoi) == {"Q2"}
    assert not set(q1[q1.cau_hoi == "Q1"].nhan) & {"sort-cv30", "sort-ekf30"}
    assert (q1.trang_thai == "C cho review").all() and q1.tap.str.contains("dev12").all()


def test_chi_ghi_summary():
    """Ma summarize.py chi tao duong ghi trong OUT (quet chuoi to_csv / write_text / np.save)."""
    src = (HERE / "summarize.py").read_text(encoding="utf-8")
    for m in re.finditer(r"(to_csv|write_text|np\.save)\(", src):
        dong = src[src.rfind("\n", 0, m.start()) + 1: src.find("\n", m.start())]
        assert "OUT /" in dong or "OUT/" in dong, dong


def main() -> int:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fail = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            fail += 1; print(f"FAIL {name}"); traceback.print_exc()
    print(f"{len(tests) - fail}/{len(tests)} PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
