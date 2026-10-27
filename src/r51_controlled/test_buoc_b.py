"""
test_buoc_b.py -- R51B: test truc tiep cho chay_buoc_b.py tren DAU RA THAT (chi doc) + fixture R1.

Moi test tu tinh lai tu file goc, khong tin vao JSON ket qua cua runner (tests/*.json chi de nguoi doc).
Chay sau `chay_buoc_b.py tat-ca`:  python -X utf8 src/r51_controlled/test_buoc_b.py   (exit 0 = moi test PASS)
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import chay_buoc_b as b  # noqa: E402
import audit_protocol as ap  # noqa: E402
import run_fingerprint as rf  # noqa: E402

TRK_MOI = b.GOC_MOI / "trackers" / b.SPLIT
VIDS = b.videos()


def _rm(d: str) -> dict:
    goc = TRK_MOI if d.startswith("p2-r51_controlled-") else b.KC48_TRK
    return json.loads((goc / d / "run_meta.json").read_text(encoding="utf-8"))


def _combined(p: Path) -> dict:
    import pandas as pd
    return pd.read_csv(p).iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Ham thuan
# ---------------------------------------------------------------------------
def test_loc_r1_fixture():
    lines = ["1,7,10,10,5,5,0.9,-1,-1,-1", "3,7,10,10,5,5,0.9,-1,-1,-1", "3,8,1,1,2,2,0.8,-1,-1,-1",
             "4,7,10,10,5,5,0.9,-1,-1,-1", ""]
    assert b.loc_r1(lines, {3}) == [lines[0], lines[3]]
    assert b.loc_r1(lines, set()) == lines[:4]
    assert b.loc_r1(lines, {1, 3, 4}) == []


def test_nhan_cua():
    assert b.nhan_cua("p2-r51_controlled-A-bt-cv") == ("bt-cv", None)
    assert b.nhan_cua("p2-kc48_pf_dev-A-ekf") == ("bt-ekf", None)
    assert b.nhan_cua("p2-kc48_pf_dev-A-pf100-k1-s2") == ("bt-pf100-k1", 2)
    assert b.nhan_cua("p2-kc48_pf_dev-A-pf500-s0") == ("bt-pf500", 0)


def test_danh_sach_run_dung_giao_thuc():
    ds = b.danh_sach_b()
    assert [x["nhan"] for x in ds["run_moi_so_sanh"]] == list(b.MOI)
    assert sorted(b.NEO) == sorted(x["nhan"] + ("-s0" if x.get("seeds") else "") for x in ds["run_neo_kiem_tai_dung"])
    assert len(b.tai_dung()) == 16 and len(set(b.tai_dung())) == 16
    assert len(b.dir_so_sanh()) == 20


def _df_fx():
    import pandas as pd
    base = {k: 0.5 for k in b.METRIC_COT}
    comb = pd.DataFrame([{"video": "COMBINED_SEQ", **base}])
    pv = pd.DataFrame([{"video": v, **base} for v in ("A", "B")])
    return comb, pv


def test_so_r0_voi_cu_fixture():
    """Cong R0 (R51B-F1): ca dat cho qua; thieu file / thieu cot / NaN / lech > 1e-12 / tap video khac deu la loi."""
    import numpy as np
    import pandas as pd
    comb, pv = _df_fx()
    loi, dc, dp = b.so_r0_voi_cu(comb, comb.copy(), pv, pv.copy())
    assert loi == [] and dc == 0.0 and dp == 0.0
    c2 = comb.copy(); c2.loc[0, "HOTA"] += 5e-13                             # trong nguong
    assert b.so_r0_voi_cu(c2, comb, pv, pv)[0] == []
    hong = {
        "thieu_file": (None, comb, pv, pv),
        "thieu_cot": (comb.drop(columns=["IDF1"]), comb, pv, pv),
        "lech_combined": (comb.assign(HOTA=0.5 + 2e-12), comb, pv, pv),
        "nan_combined": (comb.assign(AssA=np.nan), comb, pv, pv),
        "lech_per_video": (comb, comb, pv.assign(IDSW=[0.5, 1.5]), pv),
        "nan_per_video": (comb, comb, pv.assign(DetA=[np.nan, 0.5]), pv),
        "thieu_video": (comb, comb, pv.iloc[:1], pv),
        "trung_video": (comb, comb, pv.assign(video=["A", "A"]), pv.assign(video=["A", "A"])),
        "hai_dong_combined": (pd.concat([comb, comb]), comb, pv, pv),
    }
    for ten, args in hong.items():
        assert b.so_r0_voi_cu(*args)[0], ten


def test_trinh_tu_cham_cong_chan_r1():
    """R1 chi chay SAU khi cham het R0 va cong R0 dat; R0 that bai hoac cong khong dat -> khong mot lan R1 nao."""
    dirs = ["d1", "d2", "d3"]

    def chay(ok_r0=True, loi_cong=(), ok_r1=True):
        goi = []

        def cham(q, g):
            goi.append((q, g))
            return ok_r0(g) if callable(ok_r0) else (ok_r0 if q == "R0" else ok_r1)

        def cong():
            goi.append(("CONG", None))
            return list(loi_cong)
        return b.trinh_tu_cham(dirs, cham, cong), goi

    rc, goi = chay()
    assert rc == 0 and [x[0] for x in goi] == ["R0"] * 3 + ["CONG"] + ["R1"] * 3, goi
    rc, goi = chay(loi_cong=["R0 d1: lech"])
    assert rc == 2 and not [x for x in goi if x[0] == "R1"] and goi[-1] == ("CONG", None), goi
    rc, goi = chay(ok_r0=lambda g: g != "d2")
    assert rc == 1 and not [x for x in goi if x[0] in ("R1", "CONG")] and len(goi) == 3, goi
    rc, goi = chay(ok_r1=False)
    assert rc == 3


def test_nhat_ky_cong_r0_truoc_r1():
    """Bang chung tren lan `cham` cuoi: cong_r0 dat (16/16) ghi TRUOC moi dong R1, sau du 20 dong R0."""
    nk = [json.loads(x) for x in b.NHAT_KY.read_text(encoding="utf-8").splitlines() if x.strip()]
    i_xong = max(i for i, x in enumerate(nk) if x.get("buoc") == "cham_xong")
    assert nk[i_xong].get("exit") == 0, nk[i_xong]          # dong cham_xong cu (truoc F1) khong co truong exit
    i_dau = max(i for i in range(i_xong) if nk[i].get("buoc_cli") or nk[i].get("buoc") == "cham_xong")
    lan = [x for x in nk[i_dau + 1:i_xong] if x.get("buoc") in ("cham", "cong_r0")]
    loai = [x["quy_tac"] if x["buoc"] == "cham" else "CONG" for x in lan]
    assert loai == ["R0"] * 20 + ["CONG"] + ["R1"] * 20, loai
    assert all(x["exit"] == 0 for x in lan if x["buoc"] == "cham")
    c = lan[20]
    assert c["dat"] is True and c["n_dat"] == c["n_run"] == 16, c


# ---------------------------------------------------------------------------
# Dau ra that
# ---------------------------------------------------------------------------
def test_gt_sao_trung_goc():
    for v in VIDS:
        for rel in b.GT_FILES:
            a, c = b.GOC_THAT / b.SPLIT / v / rel, b.GOC_MOI / b.SPLIT / v / rel
            assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(c.read_bytes()).digest(), (v, rel)


def test_ban_sao_tai_dung_trung_goc():
    for d in b.tai_dung():
        for v in VIDS:
            a, c = b.KC48_TRK / d / "data" / f"{v}.txt", TRK_MOI / d / "data" / f"{v}.txt"
            assert a.read_bytes() == c.read_bytes(), (d, v)


def test_neo_trung_12_12():
    """Tieu chi 7: run neo chay lai tai HEAD trung sha256 (quy uoc LF) 12/12 voi run kc48 cung ho."""
    for nhan, cu in b.NEO.items():
        n = sum(rf.sha256_file(TRK_MOI / b.dir_moi(nhan) / "data" / f"{v}.txt")
                == rf.sha256_file(b.KC48_TRK / cu / "data" / f"{v}.txt") != "MISSING" for v in VIDS)
        assert n == 12, (nhan, n)


def test_run_moi_dung_cau_hinh_khoa():
    """6 run moi: YAML/sha12 dung giao thuc, cung cache/dieu kien A, tracker_cfg trung run cu cung ten,
    seed PF neo = 0 (runner), cung commit HEAD; bo loc hieu luc trung cot bl_* cua buoc A."""
    import pandas as pd
    ds = b.danh_sach_b()
    khoa = {x["nhan"]: x for x in ds["run_moi_so_sanh"] + ds["run_neo_kiem_tai_dung"]}
    kk = pd.read_csv(b.PROTO / "kiem_ke_run.csv")
    cu_cung_ten = {**b.CU_KHONG_TAI_DUNG, "bt-ekf": "p2-kc48_pf_dev-A-ekf", "bt-pf100-k1-s0": "p2-kc48_pf_dev-A-pf100-k1-s0"}
    for nhan in (*b.MOI, *b.NEO):
        rm = _rm(b.dir_moi(nhan))
        k = khoa[nhan.removesuffix("-s0")]
        assert rm["status"] == "complete" and sorted(rm["videos"]) == sorted(VIDS), nhan
        assert rm["tracker_yaml"] == k["yaml"] and rm["tracker_yaml_sha256_12"] == k["yaml_sha12"], nhan
        assert rm["cache_id"] == ds["chung"]["cache_id"] and rm["conf_min"] == 0.25, nhan
        assert rm["classes"] == [2, 3, 5, 7] and rm["max_det"] == 300 and rm["fps"] == 25.0, nhan
        assert rm["elapsed_mode"] == "skip" and rm["assoc_log"] is False and rm["pf_diag"] is False, nhan
        assert rm["overrides"] == {} and rm["git_commit"] == rf._git_commit(), nhan
        assert rm["seed"] == (0 if nhan == "bt-pf100-k1-s0" else None), nhan
        cu = cu_cung_ten[nhan]
        hang = kk[kk["dir"] == cu].iloc[0]
        goc_cu = ap.ROOT / hang["goc"]
        rm_cu = json.loads((goc_cu / cu / "run_meta.json").read_text(encoding="utf-8"))
        assert rm["tracker_cfg"] == rm_cu["tracker_cfg"], (nhan, cu)
        bl = ap.bo_loc_hieu_luc(rm["tracker_cfg"]["tracker_type"], rm["tracker_cfg"])
        for col, v in bl.items():
            vk = hang[col]
            rong = v is None or v == ""                      # CSV doc lai chuoi rong / None thanh NaN
            assert (rong and pd.isna(vk)) or str(v) == str(vk) or (
                not isinstance(v, str) and not pd.isna(vk) and abs(float(v) - float(vk)) < 1e-12), (nhan, col, v, vk)


def test_run_moi_dat_cong_provenance():
    """Ap DUNG cong kiem_provenance cua buoc A (tieu chi 1-6) cho 4 run moi, voi eval_meta R0 cua buoc B."""
    code_now = rf.code_fingerprint()
    gt_now = rf.gt_fingerprint(b.SPLIT, VIDS)               # goc that data/processed (bang ban sao: test_gt_sao)
    cache_now = rf.cache_fingerprint(b.danh_sach_b()["chung"]["cache_id"], VIDS)
    cfg_head = ap.config_py_sha256("HEAD")
    for nhan in b.MOI:
        d = b.dir_moi(nhan)
        rm = _rm(d)
        em = json.loads((b.EVAL / "R0" / d / "eval_meta.json").read_text(encoding="utf-8"))
        vh = {v: rf.sha256_file(TRK_MOI / d / "data" / f"{v}.txt") for v in VIDS}
        ttype = rm["tracker_cfg"]["tracker_type"]
        rel = ap.MOD_COMMON + ap.MOD_BY_TYPE.get(ttype, ())
        cfg_khop = ap.config_py_sha256(rm["git_commit"]) == cfg_head == ap.config_py_sha256(None)
        p = ap.kiem_provenance(rm, em, VIDS, vh, rel, code_now["modules"], code_now["backend"], cache_now, gt_now,
                               ap.current_yaml_sha12(rm["tracker_yaml"]), cfg_khop)
        assert p["tai_dung_du_dieu_kien"] is True, (nhan, p["ly_do_khong_dat"])


def test_r0_tai_tao_combined_cu():
    """Kiem cham: R0 cua 16 run tai dung tai tao combined_metrics.csv cu, sai so <= 1e-12."""
    for d in b.tai_dung():
        a, c = _combined(b.EVAL / "R0" / d / "combined_metrics.csv"), _combined(b.TE_CU / d / "combined_metrics.csv")
        for k in b.METRIC_COT:
            assert abs(float(a[k]) - float(c[k])) <= 1e-12, (d, k, a[k], c[k])


def test_r1_clear_khop_so_box_bo():
    """R1: FP giam DUNG so box o 572 frame (dem lai doc lap tu file R0); TP/FN/IDSW giu nguyen."""
    unl = json.loads((b.PROTO / "frame_khong_nhan_danh_sach.json").read_text(encoding="utf-8"))
    assert sum(len(x) for x in unl.values()) == 572
    for d in b.dir_so_sanh():
        n_bo = 0
        for v in VIDS:
            bad = set(unl[v])
            n_bo += sum(1 for f in ap.tracker_frames(TRK_MOI / d / "data" / f"{v}.txt") if f in bad)
            r1 = ap.tracker_frames(TRK_MOI / f"{d}-R1" / "data" / f"{v}.txt")
            assert not (set(r1) & bad), (d, v)
        a, c = _combined(b.EVAL / "R0" / d / "combined_metrics.csv"), _combined(b.EVAL / "R1" / d / "combined_metrics.csv")
        assert int(a["FP"]) - int(c["FP"]) == n_bo, (d, a["FP"], c["FP"], n_bo)
        assert int(a["FN"]) == int(c["FN"]) and int(a["IDSW"]) == int(c["IDSW"]) and int(a["GT_dets"]) == int(c["GT_dets"])


def test_eval_meta_dung_dau_ra_va_gt():
    """eval_meta cua moi lan cham (40) tro dung file da cham: hash tung video, GT, cache, quy tac."""
    ds = b.danh_sach_b()["chung"]
    for q in ("R0", "R1"):
        for d in b.dir_so_sanh():
            em = json.loads((b.EVAL / q / d / "eval_meta.json").read_text(encoding="utf-8"))
            dd = d if q == "R0" else f"{d}-R1"
            vh = {v: rf.sha256_file(TRK_MOI / dd / "data" / f"{v}.txt") for v in VIDS}
            assert ap.hash_dau_ra_khop(em["tracking"]["per_video"], vh, VIDS) is True, (q, d)
            assert em["quy_tac"] == q and em["dir"] == dd
            assert em["gt"]["gt_sha256"] == ds["gt_sha256"] and em["gt"]["seqinfo_sha256"] == ds["seqinfo_sha256"]
            assert em["cache"]["npz_sha256"] == ds["cache_npz_sha256"]
            assert em["evaluator"].get("dataset_class") == "mot", em["evaluator"]


def test_bang_pf_trung_binh_seed():
    """PF: COMBINED rieng tung seed roi trung binh (giao thuc muc 7); bien the tat dinh co 1 run."""
    import pandas as pd
    t = pd.read_csv(b.EVAL / "combined_tung_run.csv")
    g = pd.read_csv(b.EVAL / "combined_theo_bien_the.csv")
    for (q, nhan), sub in t.groupby(["quy_tac", "nhan"]):
        row = g[(g.quy_tac == q) & (g.nhan == nhan)].iloc[0]
        assert int(row.n_seed) == len(sub) == (3 if nhan.startswith("bt-pf") else 1), (q, nhan)
        if nhan.startswith("bt-pf"):
            assert sorted(sub.seed.astype(int)) == [0, 1, 2]
        for k in ("HOTA", "AssA", "DetA", "IDF1", "IDSW"):
            assert abs(float(row[k]) - float(sub[k].mean())) < 1e-12


def test_khong_ghi_thu_muc_cam():
    """Khong co thu muc p2-r51_controlled-* trong data/processed/trackers va results/trackeval; nhat ky ghi
    thu muc cam khong doi o lan track va cham cuoi."""
    for d in (b.GOC_THAT / "trackers" / b.SPLIT, b.TE_CU):
        assert not list(d.glob("p2-r51_controlled-*")), d
    nk = [json.loads(x) for x in b.NHAT_KY.read_text(encoding="utf-8").splitlines() if x.strip()]
    for buoc in ("track_xong", "cham_xong"):
        cuoi = [x for x in nk if x.get("buoc") == buoc][-1]
        assert cuoi["thu_muc_cam_khong_doi"] is True, cuoi


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
