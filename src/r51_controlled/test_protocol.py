"""
test_protocol.py -- R51A: test cho audit_protocol.py + fixture evaluator cho frame khong co nhan.

Chi doc du lieu that; fixture TrackEval ghi vao thu muc tam cua he dieu hanh (tempfile), xoa khi xong.
Chay: python -X utf8 src/r51_controlled/test_protocol.py   (exit 0 = moi test PASS)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

import audit_protocol as ap  # noqa: E402

PROTO = ap.OUT


# ---------------------------------------------------------------------------
# Ham thuan
# ---------------------------------------------------------------------------
def test_segments():
    assert ap.segments([]) == []
    assert ap.segments([5, 3, 4, 9, 10, 12]) == [(3, 5), (9, 10), (12, 12)]
    assert ap.segments([7, 7, 8]) == [(7, 8)]


def test_classify_segment():
    labeled = {1, 2, 6, 7}
    ids = {1: {10}, 2: {10, 11}, 6: {11, 12}, 7: {12}}
    s = ap.classify_segment(3, 5, labeled, ids)
    assert s["vi_tri"] == "giua" and s["n_xuyen"] == 1 and s["loai"] == "khong_nhan_co_xe"
    ids2 = {1: {10}, 2: {10}, 6: {12}, 7: {12}}
    s2 = ap.classify_segment(3, 5, labeled, ids2)
    assert s2["n_xuyen"] == 0 and s2["loai"] == "khong_ket_luan_tu_gt"
    assert ap.classify_segment(1, 1, {2}, {2: {1}})["vi_tri"] == "dau"
    assert ap.classify_segment(3, 3, {2}, {2: {1}})["vi_tri"] == "cuoi"


def test_boundary_check():
    ids = {1: {10}, 2: {10, 11}, 6: {11, 12}, 7: {12}}
    span = ap.track_spans(ids)
    assert span == {10: (1, 2), 11: (2, 6), 12: (6, 7)}
    b = ap.boundary_check(3, 5, ids, span)
    # 10 ket thuc tai 2 (=a-1), 11 thi khong; 12 bat dau tai 6 (=b+1), 11 thi khong
    assert b == {"gt_truoc_ket_thuc_tai_a_1": 1, "gt_sau_bat_dau_tai_b_1": 1}


def test_ioa_and_mask():
    assert abs(ap.ioa([0, 0, 10, 10], [5, 0, 100, 100]) - 0.5) < 1e-12
    assert ap.ioa([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    rows = [["1", "1"], ["3", "1"], ["4.0", "2"]]
    assert ap.mask_rows(rows, {1, 2, 4}) == [["1", "1"], ["4.0", "2"]]


def test_clear_after_mask():
    raw = {"CLR_TP": 8, "CLR_FN": 2, "CLR_FP": 4, "IDSW": 1}
    e = ap.clear_after_mask(raw, 3)
    assert e["CLR_FP_moi_chinh_xac"] == 1
    assert abs(e["MOTA_cu"] - (8 - 4 - 1) / 10) < 1e-12 and abs(e["MOTA_moi_chinh_xac"] - (8 - 1 - 1) / 10) < 1e-12


def test_sum_raw_assa_trong_so_tp():
    a = {"HOTA_TP": [2.0] * 19, "HOTA_FN": [0.0] * 19, "HOTA_FP": [0.0] * 19, "AssA": [1.0] * 19,
         "CLR_TP": 2, "CLR_FN": 0, "CLR_FP": 0, "IDSW": 0}
    b = {**a, "HOTA_TP": [6.0] * 19, "AssA": [0.5] * 19, "CLR_TP": 6}
    s = ap.sum_raw({"a": a, "b": b})
    assert abs(s["AssA"][0] - (2 * 1.0 + 6 * 0.5) / 8) < 1e-12 and s["CLR_TP"] == 8


# ---------------------------------------------------------------------------
# R51A-F1: cong provenance -- fixture duong tinh + am tinh (thieu bang chung => KHONG DAT)
# ---------------------------------------------------------------------------
_V = ["v1", "v2", "v3"]
_H = {v: f"{i:x}" * 64 for i, v in enumerate(_V, start=10)}          # sha256 "day du" gia
_MODS = ("run_tracking.py", "det_cache.py")
_BE = {"backend": "ultralytics", "ultralytics": "8.4.141", "numpy": "2.4.6", "scipy": "1.17.1", "lap": "0.5.13"}
_CACHE = {"cache_id": "c", "meta_sha256": "m" * 64, "npz_sha256": "n" * 64, "weights_sha256": "w" * 64}
_GT = {"gt_sha256": "g" * 64, "seqinfo_sha256": "s" * 64}


def _fx():
    """Bo bang chung DAY DU cho mot run -> phai DAT. Moi test am tinh lam hong dung MOT truong."""
    import copy
    rm = {"status": "complete", "videos": list(_V), "cache_id": "c", "tracker_yaml_sha256_12": "y" * 12,
          "code": {"modules": {m: "a" * 16 for m in _MODS}, "backend": dict(_BE)}}
    em = {"tracking": {"n_missing": 0, "per_video": {v: _H[v][:16] for v in _V}},
          "cache": dict(_CACHE), "code": {"modules": {m: "a" * 16 for m in _MODS}},
          "gt": {**_GT, "n_missing_gt": 0, "n_missing_seqinfo": 0, "gt_per_video": {v: "x" * 16 for v in _V}}}
    kw = dict(videos=list(_V), vh=dict(_H), rel_mods=_MODS, mods_now={m: "a" * 16 for m in _MODS},
              backend_now=dict(_BE), cache_now=dict(_CACHE), gt_now=dict(_GT), yaml_sha12_now="y" * 12,
              config_khop=True)
    return copy.deepcopy(rm), copy.deepcopy(em), kw


def _kp(rm, em, kw):
    return ap.kiem_provenance(rm, em, **kw)


def test_provenance_du_bang_chung_thi_dat():
    r = _kp(*_fx())
    assert r == {"tai_dung_du_dieu_kien": True, "ly_do_khong_dat": []}, r


def test_provenance_thieu_hash_mot_video_khong_dat():
    """Dung truong hop reviewer R51A-F1 tai hien: eval_meta thieu hash 1 video -> bieu thuc cu tra True."""
    rm, em, kw = _fx()
    del em["tracking"]["per_video"]["v2"]
    old = all(kw["vh"][v].startswith(em["tracking"]["per_video"][v]) for v in _V if v in em["tracking"]["per_video"])
    assert old is True                                  # bieu thuc cu (dong 353 ban dong bang) bo sot
    assert ap.hash_dau_ra_khop(em["tracking"]["per_video"], kw["vh"], _V) is False
    r = _kp(rm, em, kw)
    assert not r["tai_dung_du_dieu_kien"] and any("per_video thieu ['v2']" in x for x in r["ly_do_khong_dat"]), r


def test_provenance_am_tinh_tung_truong():
    """Moi truong bat buoc: xoa / lam lech -> KHONG DAT voi ly do tuong ung."""
    def hong(fn):
        rm, em, kw = _fx()
        fn(rm, em, kw)
        return _kp(rm, em, kw)

    ca = {
        "file dau ra thieu": lambda rm, em, kw: kw["vh"].__setitem__("v3", "MISSING"),
        "hash rong": lambda rm, em, kw: em["tracking"]["per_video"].__setitem__("v1", ""),
        "hash lech": lambda rm, em, kw: kw["vh"].__setitem__("v1", "0" * 64),
        "video thua": lambda rm, em, kw: em["tracking"]["per_video"].__setitem__("v9", "1" * 16),
        "n_missing": lambda rm, em, kw: em["tracking"].__setitem__("n_missing", 1),
        "khong eval_meta": None,
        "status": lambda rm, em, kw: rm.__setitem__("status", "running"),
        "videos run_meta": lambda rm, em, kw: rm.__setitem__("videos", _V[:2]),
        "cache thieu weights": lambda rm, em, kw: em["cache"].pop("weights_sha256"),
        "cache lech npz": lambda rm, em, kw: em["cache"].__setitem__("npz_sha256", "z" * 64),
        "cache_id run_meta": lambda rm, em, kw: rm.__setitem__("cache_id", "khac"),
        "yaml lech": lambda rm, em, kw: kw.__setitem__("yaml_sha12_now", "q" * 12),
        "yaml thieu": lambda rm, em, kw: rm.pop("tracker_yaml_sha256_12"),
        "module thieu": lambda rm, em, kw: rm["code"]["modules"].pop("det_cache.py"),
        "module lech HEAD": lambda rm, em, kw: kw["mods_now"].__setitem__("det_cache.py", "b" * 16),
        "module luc cham": lambda rm, em, kw: em["code"]["modules"].__setitem__("run_tracking.py", "c" * 16),
        "backend thieu": lambda rm, em, kw: rm["code"].pop("backend"),
        "backend lech": lambda rm, em, kw: rm["code"]["backend"].__setitem__("numpy", "1.26.4"),
        "config.py khong xac minh": lambda rm, em, kw: kw.__setitem__("config_khop", None),
        "config.py lech": lambda rm, em, kw: kw.__setitem__("config_khop", False),
        "GT thieu dau van tay": lambda rm, em, kw: em.__setitem__("gt", {}),
        "GT lech": lambda rm, em, kw: kw["gt_now"].__setitem__("gt_sha256", "h" * 64),
        "GT thieu file": lambda rm, em, kw: em["gt"].__setitem__("n_missing_gt", 1),
        "GT thieu video": lambda rm, em, kw: em["gt"]["gt_per_video"].pop("v1"),
    }
    for ten, fn in ca.items():
        if fn is None:
            rm, em, kw = _fx()
            r = _kp(rm, None, kw)
        else:
            r = hong(fn)
        assert r["tai_dung_du_dieu_kien"] is False and r["ly_do_khong_dat"], ten


def test_bo_loc_hieu_luc_doc_tu_instance():
    """Thong so hieu luc lay tu instance bo loc: khoa vang mat -> mac dinh config.py; co khoa -> gia tri YAML.
    (Ham chi tao bo loc trong bo nho, khong sua cau hinh nao.)"""
    c = ap.config
    base = {"tracker_type": "bytetrack_pf_ctrv", "pf_num_particles": 100, "pf_seed": 0, "pf_resample_ess_ratio": 0.5,
            "pf_estimate": "mean", "pf_roughening": 0.0, "pf_degenerate_maha2": 1e4, "pf_optimal_proposal": True}
    b = ap.bo_loc_hieu_luc("bytetrack_pf_ctrv", base)
    assert (b["bl_lop"], b["bl_pf_N"], b["bl_pf_K1"], b["bl_pf_roughening"], b["bl_dt_buoc"]) == \
        ("PFCTRV", 100, True, 0.0, 1.0), b
    assert (b["bl_ctrv_std_theta"], b["bl_ctrv_std_omega"], b["bl_ctrv_init_std_theta"]) == \
        (c.CTRV_STD_THETA, c.CTRV_STD_OMEGA, c.CTRV_INIT_STD_THETA), b
    b2 = ap.bo_loc_hieu_luc("bytetrack_ekf_ctrv", {"tracker_type": "bytetrack_ekf_ctrv", "ctrv_std_omega": 0.02})
    assert b2["bl_ctrv_std_omega"] == 0.02 and b2["bl_khoa_yaml_bo_loc"] == "ctrv_std_omega", b2
    u = ap.bo_loc_hieu_luc("bytetrack_ukf_ctrv", {"tracker_type": "bytetrack_ukf_ctrv"})
    assert (u["bl_ukf_alpha"], u["bl_ukf_beta"], u["bl_ukf_kappa"]) == (c.UKF_ALPHA, c.UKF_BETA, c.UKF_KAPPA), u
    cv = ap.bo_loc_hieu_luc("bytetrack", {"tracker_type": "bytetrack"})
    assert (cv["bl_lop"], cv["bl_dt_buoc"], cv["bl_std_weight_position"], cv["bl_std_weight_velocity"]) == \
        ("KalmanFilterXYAH", 1.0, 1 / 20, 1 / 160), cv
    assert c.CTRV_DT == 1.0                              # dt noi bo bo loc = 1 buoc (config.py muc 8)


# ---------------------------------------------------------------------------
# Fixture evaluator: MotChallenge2DBox (lop DETRAC dung) voi frame khong co dong GT
# ---------------------------------------------------------------------------
def _eval_fixture(gt_lines: list[str], trk_lines: list[str], seq_len: int) -> dict:
    import trackeval
    with tempfile.TemporaryDirectory() as tmp:
        gt_seq = Path(tmp) / "gt" / "FX-t" / "s1"
        (gt_seq / "gt").mkdir(parents=True)
        (gt_seq / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        (gt_seq / "seqinfo.ini").write_text(f"[Sequence]\nname=s1\nseqLength={seq_len}\n", encoding="utf-8")
        trk = Path(tmp) / "trk" / "FX-t" / "trk" / "data"
        trk.mkdir(parents=True)
        (trk / "s1.txt").write_text("\n".join(trk_lines) + ("\n" if trk_lines else ""), encoding="utf-8")
        dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dcfg.update({"GT_FOLDER": str(Path(tmp) / "gt"), "TRACKERS_FOLDER": str(Path(tmp) / "trk"),
                     "BENCHMARK": "FX", "SPLIT_TO_EVAL": "t", "TRACKERS_TO_EVAL": ["trk"],
                     "CLASSES_TO_EVAL": ["pedestrian"], "SEQ_INFO": {"s1": seq_len},
                     "SKIP_SPLIT_FOL": False, "PRINT_CONFIG": False})
        ecfg = trackeval.Evaluator.get_default_eval_config()
        ecfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False,
                     "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                     "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
                     "LOG_ON_ERROR": str(Path(tmp) / "err.log")})
        mets = [trackeval.metrics.HOTA({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
                trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
                trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False})]
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            res, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], mets)
        r = res["MotChallenge2DBox"]["trk"]["s1"]["pedestrian"]
        return {"HOTA": float(r["HOTA"]["HOTA"].mean()), "DetA": float(r["HOTA"]["DetA"].mean()),
                "AssA": float(r["HOTA"]["AssA"].mean()), "FP": int(r["CLEAR"]["CLR_FP"]),
                "TP": int(r["CLEAR"]["CLR_TP"]), "IDSW": int(r["CLEAR"]["IDSW"]),
                "IDF1": float(r["Identity"]["IDF1"])}


GT = [f"{f},1,100,100,50,40,1,1,1.0" for f in (1, 2, 4)]                  # frame 3: KHONG co dong GT
TRK_ALL = [f"{f},7,100,100,50,40,0.9" for f in (1, 2, 3, 4)]               # tracker co box ca frame 3


def test_evaluator_dem_fp_o_frame_khong_gt():
    """Quy tac R0 (hien hanh): frame nam trong seqLength ma khong co dong GT -> moi box tracker la FP.

    Ca AssA cung giam: box cua ID 7 o frame 3 vao FPA cua cap (GT 1, ID 7) -> AssA = 3/4, khong phai 1.
    """
    r = _eval_fixture(GT, TRK_ALL, 4)
    assert r["TP"] == 3 and r["FP"] == 1 and r["IDSW"] == 0, r
    assert abs(r["DetA"] - 0.75) < 1e-12 and abs(r["AssA"] - 0.75) < 1e-12 and abs(r["HOTA"] - 0.75) < 1e-12, r
    assert abs(r["IDF1"] - 6 / 7) < 1e-12, r


def test_mask_r1_bo_frame_khong_nhan():
    """Quy tac R1 (do nhay): bo box tracker o frame khong nhan -> giong nhu frame do khong ton tai."""
    labeled = {1, 2, 4}
    masked = [",".join(x) for x in ap.mask_rows([ln.split(",") for ln in TRK_ALL], labeled)]
    r = _eval_fixture(GT, masked, 4)
    assert r["TP"] == 3 and r["FP"] == 0 and r["IDSW"] == 0, r
    assert abs(r["HOTA"] - 1.0) < 1e-12 and abs(r["IDF1"] - 1.0) < 1e-12, r


def test_frame_rong_khong_box_khong_anh_huong():
    """Frame khong GT va khong box tracker khong dong gop gi (am tinh dung)."""
    r = _eval_fixture(GT, [ln for ln in TRK_ALL if not ln.startswith("3,")], 4)
    assert r["FP"] == 0 and abs(r["HOTA"] - 1.0) < 1e-12, r


# ---------------------------------------------------------------------------
# Doi chieu doc lap tren du lieu that (chi doc) + dieu kien tien quyet cua giao thuc
# ---------------------------------------------------------------------------
def _need(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"chua chay audit_protocol.py: thieu {p}")


def test_dem_frame_doc_lap_regex():
    """Dem lai frame XML bang regex (khong dung ElementTree) va anh bang os.listdir; so voi CSV audit."""
    import csv
    _need(PROTO / "frame_khong_nhan_theo_video.csv")
    rows = {r["video"]: r for r in csv.DictReader(open(PROTO / "frame_khong_nhan_theo_video.csv", encoding="utf-8"))}
    img_root = ap.config.find_images_root()
    tot = 0
    for v in json.loads(ap.MANIFEST.read_text(encoding="utf-8"))["videos"]:
        txt = (ap.XML_DIR / f"{v}.xml").read_text(encoding="utf-8", errors="replace")
        nums = {int(m) for m in re.findall(r"<frame\b[^>]*\bnum=\"(\d+)\"", txt)}
        imgs = {int(n[3:8]) for n in os.listdir(img_root / v) if re.fullmatch(r"img\d{5}\.jpg", n)}
        miss = len(imgs - nums)
        assert int(rows[v]["n_frame_xml"]) == len(nums), v
        assert int(rows[v]["n_anh"]) == len(imgs), v
        assert int(rows[v]["n_khong_nhan"]) == miss, v
        tot += miss
    assert tot == 572, tot


def test_tien_quyet_tai_dung():
    """Dieu kien tai dung da khoa trong 01_giao_thuc.md muc 4 (doc kiem_ke_run.json)."""
    _need(PROTO / "kiem_ke_run.json")
    runs = json.loads((PROTO / "kiem_ke_run.json").read_text(encoding="utf-8"))
    kc = [r for r in runs if r["dir"].startswith("p2-kc48_pf_dev-")]
    assert len(kc) == 16, len(kc)
    for r in kc:
        assert r["module_lech_hien_tai"] == "" and r["module_khong_ghi"] == "", r["dir"]
        assert r["yaml_sha12_luc_chay"] == r["yaml_sha12_hien_tai"], r["dir"]
        assert r["n_video_co_dau_ra"] == 12 and r["hash_dau_ra_khop_eval_meta"] is True, r["dir"]
        # R51A-F1: cong day du (moi truong bat buoc co mat va khop), khong chi cac cot o tren
        assert r["tai_dung_du_dieu_kien"] is True and r["ly_do_khong_dat"] == "", (r["dir"], r["ly_do_khong_dat"])
        assert r["config_py_khop_HEAD_va_cay"] is True, r["dir"]
    # Chi 16 run kc48 dat cong; moi run cu khac deu khong dat (ly do duoc ghi)
    dat = {r["dir"] for r in runs if r["tai_dung_du_dieu_kien"]}
    assert dat == {r["dir"] for r in kc}, dat ^ {r["dir"] for r in kc}
    assert all(r["ly_do_khong_dat"] for r in runs if not r["tai_dung_du_dieu_kien"])
    npz = {r["cache_npz_sha256"] for r in runs if r["cache_npz_sha256"]}
    assert len(npz) == 1, npz
    for r in runs:
        assert r["conf_min"] == 0.25 and r["classes"] == "[2, 3, 5, 7]" and r["max_det"] == 300, r["dir"]
        assert r["fps"] == 25.0 and r["elapsed_mode"] == "skip", r["dir"]
        if r["tracker_type"].startswith("bytetrack"):
            assert (r["track_high"], r["track_low"], r["new_track"], r["match_thresh"], r["fuse_score"],
                    r["track_buffer_frames"]) == (0.25, 0.1, 0.25, 0.8, True, 30), r["dir"]
    # SORT (expB) va CV/UKF (expA) co module lech HEAD -> phai chay moi (giao thuc muc 4.3)
    for d in ("p2-expA_detrac_dev-c25-cv", "p2-expA_detrac_dev-c25-ukf",
              "p2-expB_detrac_dev_fix1-A-sort-cv30", "p2-expB_detrac_dev_fix1-A-sort-ekf30"):
        r = next(x for x in runs if x["dir"] == d)
        assert r["module_lech_hien_tai"] != "", d


def test_ma_tran_tai_dung_khop_cong():
    """ma_tran_tai_dung.csv (viet tay o buoc A) phai nhat quan voi cong provenance tinh bang ma."""
    import csv
    _need(PROTO / "kiem_ke_run.json")
    runs = json.loads((PROTO / "kiem_ke_run.json").read_text(encoding="utf-8"))
    by = {r["dir"]: r for r in runs}
    for row in csv.DictReader(open(PROTO / "ma_tran_tai_dung.csv", encoding="utf-8")):
        qd, nhan = row["quyet_dinh"], row["doi_chung"]
        if qd in ("TAI_DUNG", "TAI_DUNG_CO_NEO", "CHI_TAI_DUNG"):
            ds = [r for r in runs if r["dir"].startswith("p2-kc48_pf_dev-") and r["nhan"] == nhan]
            assert ds and all(r["tai_dung_du_dieu_kien"] for r in ds), nhan
            if row["seed"]:
                assert sorted(r["seed_runner"] for r in ds) == [int(s) for s in row["seed"].split(";")], nhan
        elif qd == "CHAY_MOI":
            d = row["nguon_ung_vien"].split(" ")[0]
            assert by[d]["tai_dung_du_dieu_kien"] is False, d


def test_bang_thong_so_hieu_luc_kc48():
    """R51A-F2: thong so bo loc hieu luc tung run tai dung khop ten cau hinh; dt danh nghia vs dt bo loc."""
    _need(PROTO / "kiem_ke_run.json")
    runs = json.loads((PROTO / "kiem_ke_run.json").read_text(encoding="utf-8"))
    c = ap.config
    for r in runs:
        if not r["dir"].startswith("p2-kc48_pf_dev-"):
            continue
        assert r["dt_danh_nghia_s"] == 0.04 and r["bl_dt_buoc"] == 1.0, r["dir"]
        assert (r["bl_ctrv_std_theta"], r["bl_ctrv_std_omega"], r["bl_ctrv_init_std_theta"],
                r["bl_ctrv_init_std_omega"], r["bl_ctrv_heading_std_after_init"],
                r["bl_ctrv_min_speed_for_heading"]) == (c.CTRV_STD_THETA, c.CTRV_STD_OMEGA, c.CTRV_INIT_STD_THETA,
                                                         c.CTRV_INIT_STD_OMEGA, c.CTRV_HEADING_STD_AFTER_INIT,
                                                         c.CTRV_MIN_SPEED_FOR_HEADING), r["dir"]
        assert not any(k.startswith("ctrv_") for k in r["bl_khoa_yaml_bo_loc"].split(";") if k), r["dir"]
        if r["tracker_type"] == "bytetrack_pf_ctrv":
            m = re.search(r"-pf(\d+)(-k(\d))?-s(\d)$", r["dir"])
            n, k, s = int(m.group(1)), m.group(3), int(m.group(4))
            assert r["bl_pf_N"] == n and r["bl_pf_seed_hieu_luc"] == s, r["dir"]
            assert r["bl_pf_K1"] is (k == "1"), r["dir"]
            assert r["bl_pf_roughening"] == (1.0 if k == "2" else 0.0), r["dir"]
            assert (r["bl_pf_ess"], r["bl_pf_estimate"]) == (0.5, "mean"), r["dir"]
        else:
            assert r["bl_lop"] == "EKFTrackerCTRV" and r["bl_pf_N"] is None, r["dir"]


def test_gt_hien_tai_trung_luc_cham_kc48():
    _need(PROTO / "audit_tong_hop.json")
    t = json.loads((PROTO / "audit_tong_hop.json").read_text(encoding="utf-8"))
    assert t["gt_hien_tai_trung_eval_meta"] is True, t["gt_eval_meta_cac_run"]
    assert t["gt_hien_tai"]["gt_sha256"].startswith("07de8402") and t["tong_khong_nhan"] == 572
    assert t["frame_xml_khong_target"] == 0 and t["doan_gt_truoc_deu_ket_thuc_va_gt_sau_deu_bat_dau"] == 8


def test_so_khop_ekf_pf_giua_cac_commit():
    _need(PROTO / "so_khop_dau_ra.csv")
    import csv
    for r in csv.DictReader(open(PROTO / "so_khop_dau_ra.csv", encoding="utf-8")):
        assert r["n_video_trung_sha256"] == "12", r


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
