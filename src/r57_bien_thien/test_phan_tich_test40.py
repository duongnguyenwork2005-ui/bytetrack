"""
test_phan_tich_test40.py -- fixture CPU cho src/r57_bien_thien/phan_tich_test40.py (R57A). Khong doc du lieu that ngoai
giao thuc configs/phase2/r57_bien_thien/giao_thuc.json (chi kiem cau truc nhom). Moi fixture tao trong
results/phase2/r57_bien_thien/tests/lan_<ts>_pid<p>/ (phien moi); ket qua: ket_qua_test.json gan SHA256 ma / giao thuc.

Phep tinh doc lap: to_hop_doc_lap() viet lai cong thuc COMBINED (HOTA 19 alpha, IDF1, IDSW) bang vong lap Python thuan,
khong goi boot_stats; dung de sinh combined_metrics.csv fixture va kiem nhanh nhom.

Exit 0 khi moi ca dat; 1 neu co ca khong dat.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phan_tich_test40 as pt  # noqa: E402

bs, bse = pt.bs, pt.bse
ROOT = pt.ROOT
GT_THAT = ROOT / "configs" / "phase2" / "r57_bien_thien" / "giao_thuc.json"
V = ["MVI_90001", "MVI_40773", "MVI_90002", "MVI_40775", "MVI_90003", "MVI_90004"]
NHOM = [["MVI_90001"], ["MVI_40773", "MVI_40775"], ["MVI_90002"], ["MVI_90003"], ["MVI_90004"]]
B_FX = 200
SES: Path | None = None
COT_CI = pt.HEADER_CI


# ------------------------------------------------------------------------------------------------ phep tinh doc lap
def to_hop_doc_lap(recs: list[dict]) -> dict:
    n_a = len(recs[0]["HOTA_TP"])
    hs = []
    for a in range(n_a):
        tp = fn = fp = num = 0.0
        for r in recs:
            tp += r["HOTA_TP"][a]
            fn += r["HOTA_FN"][a]
            fp += r["HOTA_FP"][a]
            num += r["AssA"][a] * r["HOTA_TP"][a]
        ass = max(1e-10, num) / max(1e-10, tp)
        det = tp / max(1.0, tp + fn + fp)
        hs.append(math.sqrt(det * ass))
    idtp = sum(r["IDTP"] for r in recs)
    idfn = sum(r["IDFN"] for r in recs)
    idfp = sum(r["IDFP"] for r in recs)
    return {"HOTA": sum(hs) / n_a, "IDF1": idtp / max(1.0, idtp + 0.5 * idfn + 0.5 * idfp),
            "IDSW": float(sum(r["IDSW"] for r in recs))}


def raw_ngau_nhien(rng, videos) -> dict:
    out = {}
    for v in videos:
        tp = rng.integers(20, 600, 19)
        out[v] = {"HOTA_TP": [float(x) for x in tp], "HOTA_FN": [float(x) for x in rng.integers(0, 300, 19)],
                  "HOTA_FP": [float(x) for x in rng.integers(0, 300, 19)],
                  "AssA": [float(x) for x in rng.uniform(0.2, 0.95, 19)],
                  "AssRe": [float(x) for x in rng.uniform(0.2, 0.95, 19)],
                  "AssPr": [float(x) for x in rng.uniform(0.2, 0.95, 19)],
                  "IDTP": int(rng.integers(100, 3000)), "IDFN": int(rng.integers(0, 1500)),
                  "IDFP": int(rng.integers(0, 1500)), "CLR_TP": int(rng.integers(100, 3000)),
                  "CLR_FN": int(rng.integers(0, 1500)), "CLR_FP": int(rng.integers(0, 1500)),
                  "IDSW": int(rng.integers(0, 80))}
    return out


# ------------------------------------------------------------------------------------------------------- fixture repo
def ghi(p: Path, data) -> str:
    pt.kiem_vung_ghi(ROOT, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    b = data if isinstance(data, bytes) else (data.encode("utf-8") if isinstance(data, str) else pt.json_bytes(data))
    with open(p, "xb") as f:
        f.write(b)
    return pt.sha256_bytes(b)


def tao_fixture(ten: str, bien: dict | None = None) -> tuple[Path, dict]:
    """Repo gia lap: dau vao trong in/, giao thuc tai configs/phase2/r57_bien_thien/giao_thuc.json. 'bien' tao loi
    nhat quan hash (truoc khi bam) hoac sau khi bam."""
    bien = bien or {}
    goc = SES / ten
    gt = json.loads(GT_THAT.read_text(encoding="utf-8"))
    gt["video"], gt["so_video"] = list(V), len(V)
    gt["thiet_ke"]["video"]["K"] = len(V)
    gt["thiet_ke"]["nhom"]["K"] = len(NHOM)
    gt["thiet_ke"]["nhom"]["nhom"] = [{"id": f"N{i + 1:02d}", "video": n} for i, n in enumerate(NHOM)]
    gt["bootstrap"]["B"] = B_FX
    runs = pt.danh_sach_run(gt)
    ten_runs = [r["run"] for r in runs]
    h_man = ghi(goc / "in/manifest.json", {"videos": V})
    h_gt54 = ghi(goc / "in/gt54.json", {"ma_tran": {"danh_sach": ten_runs}, "tap_du_lieu": {"so_video": len(V)}})
    h_hd = ghi(goc / "in/hop_dong.json", {"fixture": True})
    h_khoa = ghi(goc / "in/khoa.json", {"giao_thuc_sha256": h_gt54, "hop_dong_sha256": h_hd, "ma_tran": ten_runs,
                                        "file": [{"rel": "in/manifest.json", "sha256": h_man}]})
    ten_khoa = "khoa_fixture"
    tt = "in/phien_tt"
    ghi(goc / tt / "phien.json", {"khoa": ten_khoa, "khoa_sha256": h_khoa, "hop_dong_sha256": h_hd,
                                  "giao_thuc_sha256": h_gt54, "so_video": len(V), "ma_tran": ten_runs, "fixture": True})
    rng = np.random.default_rng(12345)
    raws, summary = {}, {}
    for r in runs:
        run = r["run"]
        raw = raw_ngau_nhien(rng, V)
        raws[run] = copy.deepcopy(raw)
        if bien.get("raw_thieu_video") == run:
            del raw[V[-1]]
        if bien.get("raw_nan") == run:
            raw[V[0]]["AssA"][3] = float("nan")
        lan = goc / tt / "cham" / run / "lan_1"
        tev = lan / "trackeval" / run
        comb = to_hop_doc_lap([raws[run][v] for v in V])
        per = {v: to_hop_doc_lap([raws[run][v]]) for v in V}
        bang = {"per_video_metrics.csv": (["video", "HOTA", "IDF1", "IDSW"],
                                          [[v, repr(per[v]["HOTA"]), repr(per[v]["IDF1"]), repr(per[v]["IDSW"])] for v in V]),
                "combined_metrics.csv": (["video", "HOTA", "IDF1", "IDSW"],
                                         [["COMBINED_SEQ", repr(comb["HOTA"]), repr(comb["IDF1"]), repr(comb["IDSW"])]])}
        cs = bien.get("csv_sua")
        if cs and cs[0] == run:     # lam hong TRUOC khi ghi -> hash trong marker nhat quan, loi chi con o schema
            bang[cs[1]] = cs[2](*bang[cs[1]])
        f = {"pedestrian_summary.txt": ghi(tev / "pedestrian_summary.txt", f"fixture {run}\n"),
             "raw_fields.json": ghi(tev / "raw_fields.json", json.dumps(raw).encode("utf-8")),
             **{k: ghi(tev / k, pt.csv_bytes(*bang[k])) for k in bang}}
        cache = hashlib.sha256(r["detector"].encode()).hexdigest()
        if bien.get("cache_lech") == run:
            cache = hashlib.sha256(b"khac").hexdigest()
        ghi(lan / "HOAN_TAT.json", {
            "rang_buoc": {"loai": "cham", "khoa": ten_khoa, "khoa_sha256": h_khoa, "hop_dong_sha256": h_hd,
                          "giao_thuc_sha256": h_gt54, "run": run, "seq_info": {v: 100 for v in V},
                          "manifest_cache_sha256": cache},
            "file": {f"trackeval/{run}/{k}": h for k, h in f.items()},
            "file_metric": {k: f[k] for k in ("per_video_metrics.csv", "combined_metrics.csv", "raw_fields.json")},
            "combined": comb})
        summary[run] = {"file": pt._rel(goc, tev / "pedestrian_summary.txt"), "sha256": f["pedestrian_summary.txt"]}
    sm = dict(summary)
    if bien.get("nguon_thieu_run"):
        del sm[bien["nguon_thieu_run"]]
    n56 = {"phien": "phien_tt", "so_run": len(runs), "summary": sm}
    if bien.get("nguon_trung_run"):
        s = json.dumps(n56, indent=1)
        k = bien["nguon_trung_run"]
        dong = f'  "{k}": ' + json.dumps(summary[k]) + ",\n"
        s = s.replace('"summary": {\n', '"summary": {\n' + dong, 1)
        ghi(goc / "in/nguon56.json", s)
    else:
        ghi(goc / "in/nguon56.json", n56)
    # bang R56 gia, cung quy tac voi src/r56_tong_hop/tong_hop_test40.py: summary 3 chu so (IDSW nguyen), PF tb =
    # round(trung binh 3 gia tri summary, 4). 'th_sua' / 'pf_sua' lam hong TRUOC khi ghi -> hash ghi luc chuan-bi nhat quan.
    th_hdr = ["run", "detector", "dieu_kien", "bo_loc", "seed", "HOTA", "IDF1", "IDSW"]
    th_rows, th_so = [], {}
    for r in runs:
        c = to_hop_doc_lap([raws[r["run"]][v] for v in V])
        th_so[r["run"]] = {"HOTA": round(c["HOTA"] * 100, 3), "IDF1": round(c["IDF1"] * 100, 3), "IDSW": int(c["IDSW"])}
        th_rows.append([r["run"], r["detector"], r["dieu_kien"], r["bo_loc"], "" if r["seed"] is None else r["seed"],
                        f"{c['HOTA'] * 100:.3f}", f"{c['IDF1'] * 100:.3f}", int(c["IDSW"])])
    if bien.get("th_sua"):
        th_hdr, th_rows = bien["th_sua"](th_hdr, th_rows)
    ghi(goc / "in/tong_hop.csv", pt.csv_bytes(th_hdr, th_rows))
    pf_hdr = ["detector", "dieu_kien", "so_seed", "HOTA_tb", "IDF1_tb", "IDSW_tb"]
    pf_rows = []
    for det in gt["detector"]:
        for dk in gt["dieu_kien"]:
            g = [th_so[pt.ten_run(det, dk, "pf500", s)] for s in (0, 1, 2)]
            pf_rows.append([det, dk, 3, *[round(sum(x[m] for x in g) / 3, 4) for m in ("HOTA", "IDF1", "IDSW")]])
    if bien.get("pf_sua"):
        pf_hdr, pf_rows = bien["pf_sua"](pf_hdr, pf_rows)
    ghi(goc / "in/pf500.csv", pt.csv_bytes(pf_hdr, pf_rows))
    gt["nguon"].update({
        "phien_thuc_thi": tt, "nguon_r56": {"rel": "in/nguon56.json", "sha256": None},
        "khoa": {"ten": ten_khoa, "rel": "in/khoa.json", "sha256": h_khoa},
        "giao_thuc_r54": {"rel": "in/gt54.json", "sha256": h_gt54},
        "hop_dong_r55": {"rel": "in/hop_dong.json", "sha256": h_hd},
        "manifest_video": {"rel": "in/manifest.json", "sha256": h_man},
        "bang_r56": {"tong_hop": {"rel": "in/tong_hop.csv", "sha256": None}, "pf500": {"rel": "in/pf500.csv", "sha256": None}}})
    ghi(goc / "configs/phase2/r57_bien_thien/giao_thuc.json", gt)
    if bien.get("sua_sau_hash") == "raw":
        p = goc / summary[runs[0]["run"]]["file"].replace("pedestrian_summary.txt", "raw_fields.json")
        pt.kiem_vung_ghi(ROOT, p)
        p.write_bytes(p.read_bytes() + b" ")
    return goc, {"gt": gt, "raws": raws}


GT_REL = "configs/phase2/r57_bien_thien/giao_thuc.json"


def phai_loi(fn, *a, chua: str = "", **k) -> str:
    try:
        fn(*a, **k)
    except pt.LoiDauVao as e:
        if chua and chua not in str(e):
            raise AssertionError(f"loi khac mong doi: {e}")
        return str(e)
    raise AssertionError("khong bi chan")


# -------------------------------------------------------------------------------------------------------------- ca test
def t_combined_khac_tb_video():
    rng = np.random.default_rng(1)
    raw = raw_ngau_nhien(rng, V)
    c = bs.combine(raw, V)
    dl = to_hop_doc_lap([raw[v] for v in V])
    tb = float(np.mean([to_hop_doc_lap([raw[v]])["HOTA"] for v in V]))
    assert all(abs(c[m] - dl[m]) <= 1e-12 for m in ("HOTA", "IDF1", "IDSW")), (c, dl)
    assert abs(c["HOTA"] - tb) > 1e-4, (c["HOTA"], tb)
    return {"HOTA_combined": c["HOTA"], "HOTA_tb_video": tb, "doc_lap": dl["HOTA"]}


def t_pf_thu_tu_phi_tuyen():
    rng = np.random.default_rng(2)
    gt = {"bo_loc": ["cv", "pf500"], "pf": {"bo_loc": "pf500", "seeds": [0, 1, 2]}}
    raws = {pt.ten_run("d", "c25", "pf500", s): raw_ngau_nhien(rng, V) for s in (0, 1, 2)}
    raws[pt.ten_run("d", "c25", "cv")] = raw_ngau_nhien(rng, V)
    c = {k: bs.combine(r, V) for k, r in raws.items()}
    g = pt.gia_tri_nhom(c, gt, "d", "c25", ["HOTA", "IDF1", "IDSW"])
    dung = {m: float(np.mean([to_hop_doc_lap([raws[pt.ten_run("d", "c25", "pf500", s)][v] for v in V])[m]
                              for s in (0, 1, 2)])) for m in ("HOTA", "IDF1", "IDSW")}
    sai_thu_tu = bs.combine(bs.average_seeds([raws[pt.ten_run("d", "c25", "pf500", s)] for s in (0, 1, 2)]), V)
    assert all(abs(g["pf500"][m] - dung[m]) <= 1e-12 for m in dung), (g, dung)
    assert abs(sai_thu_tu["HOTA"] - dung["HOTA"]) > 1e-9 and abs(sai_thu_tu["IDF1"] - dung["IDF1"]) > 1e-9
    assert abs(sai_thu_tu["IDSW"] - dung["IDSW"]) <= 1e-9          # tong tuyen tinh: hai thu tu trung
    return {"pf_dung": dung, "average_seeds_roi_combine": {m: sai_thu_tu[m] for m in dung}}


def t_lap_trung_bao_toan():
    rng = np.random.default_rng(3)
    raw = raw_ngau_nhien(rng, V)
    st = bs._stack(raw, V)
    idx = np.array([0, 0, 0, 3, 3, 5])
    c = bs.combine_stacked({f: st[f][idx] for f in st})
    dl = to_hop_doc_lap([raw[V[i]] for i in idx])
    duy_nhat = to_hop_doc_lap([raw[V[i]] for i in sorted(set(idx.tolist()))])
    assert all(abs(c[m] - dl[m]) <= 1e-12 for m in dl), (c, dl)
    assert c["IDSW"] == 3 * raw[V[0]]["IDSW"] + 2 * raw[V[3]]["IDSW"] + raw[V[5]]["IDSW"]
    assert abs(c["HOTA"] - duy_nhat["HOTA"]) > 1e-9
    return {"idx": idx.tolist(), "HOTA": c["HOTA"], "HOTA_khong_lap": duy_nhat["HOTA"]}


def t_cap_di_cung_va_giao_thuc_that():
    gt = pt.doc_json(GT_THAT)
    pt.kiem_giao_thuc(gt)
    tv = pt.thanh_vien_nhom(gt)
    assert len(tv) == 39 and len(gt["video"]) == 40 and gt["bootstrap"]["B"] == 5000 and gt["bootstrap"]["rng_seed"] == 0
    nhieu = [(gt["thiet_ke"]["nhom"]["nhom"][i]["id"], [gt["video"][j] for j in x]) for i, x in enumerate(tv) if len(x) > 1]
    assert nhieu == [("N22", ["MVI_40773", "MVI_40775"])], nhieu
    gi = 21
    idx_n = pt.rut_chi_so(2000, 39, 0)
    tong_cap, do_dai = 0, []
    for g in idx_n:
        vv = pt.mo_rong_nhom(g, tv)
        k = int(np.sum(g == gi))
        assert int(np.sum(vv == tv[gi][0])) == k and int(np.sum(vv == tv[gi][1])) == k
        assert len(vv) == 39 + k
        dem = np.bincount(vv, minlength=40)
        for i, x in enumerate(tv):
            assert np.all(dem[x] == int(np.sum(g == i)))
        tong_cap += k
        do_dai.append(len(vv))
    assert min(do_dai) < max(do_dai)                                 # khong cat / dem ve 40
    # mot nhom mo rong thu cong
    g = np.array([gi, 0, gi, 38])
    vv = pt.mo_rong_nhom(g, tv).tolist()
    assert vv == [tv[gi][0], tv[gi][1], 0, tv[gi][0], tv[gi][1], 39], vv
    return {"so_mau": 2000, "tong_lan_cap": tong_cap, "do_dai_min": min(do_dai), "do_dai_max": max(do_dai)}


def t_ghep_cap_moi_run_seed():
    """IDSW video j = 10^j -> moi COMBINED trong cung mau ma hoa duy nhat tap video (co so lan); moi run / seed phai
    nhan cung ma."""
    gt = {"detector": ["d1", "d2"], "dieu_kien": ["c25"], "bo_loc": ["cv", "ekf", "pf500"],
          "pf": {"bo_loc": "pf500", "seeds": [0, 1, 2]}, "hieu_so": [["ekf", "cv"], ["pf500", "cv"]]}
    rng = np.random.default_rng(4)
    stacks = {}
    for r in pt.danh_sach_run(gt):
        raw = raw_ngau_nhien(rng, V)
        for j, v in enumerate(V):
            raw[v]["IDSW"] = 10 ** j
        stacks[r["run"]] = bs._stack(raw, V)
    idx = pt.rut_chi_so(50, len(V), 0)
    ghi_lai = []
    goc_fn = bs.combine_stacked

    def gian_diep(st):
        out = goc_fn(st)
        ghi_lai.append(out["IDSW"])
        return out
    bs.combine_stacked = gian_diep
    try:
        boots = pt.bootstrap(stacks, list(idx), gt, ["HOTA", "IDSW"])
    finally:
        bs.combine_stacked = goc_fn
    n = len(stacks)
    assert len(ghi_lai) == n * len(idx)
    for i, row in enumerate(idx):
        ma = float(sum(10 ** int(j) for j in row))
        assert all(x == ma for x in ghi_lai[i * n:(i + 1) * n]), i
    assert all(np.all(boots[k] == 0) for k in boots if k[-1] == "IDSW")
    return {"so_run": n, "so_mau": len(idx), "so_lan_combine": len(ghi_lai)}


def t_dau_cuoi_fixture():
    goc, info = tao_fixture("fx_dung")
    phien = pt.chuan_bi(goc, GT_REL)
    assert sorted(x.name for x in phien.iterdir()) == sorted(pt.DAU_RA_CHUAN_BI)
    kq = pt.phan_tich(goc, phien.name)
    kc = kq["kiem_chung"]
    gt, raws = info["gt"], info["raws"]
    assert kc["so_dong_ci"] == {"video": 108, "nhom": 108, "mong": 108}
    assert kc["doi_chieu_boot_seed"]["trung"] == 108 and kc["doi_chieu_boot_seed"]["so_so_sanh"] == 108
    assert kc["tai_to_hop"]["dat"] and kc["chi_so"]["tai_tinh_trung"]
    assert not kc["dau_vao"]["hash_truoc_lech"] and not kc["dau_vao"]["hash_sau_lech"]
    co = sorted(x.name for x in phien.iterdir())
    assert co == sorted(pt.DAU_RA_CHUAN_BI + pt.DAU_RA_PHAN_TICH), co
    # SHA256SUMS dung
    for dong in (phien / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, ten = dong.split("  ")
        assert pt.sha256_file(phien / ten) == h, ten
    # nhanh video: doi chieu doc lap mot hieu so PF voi helper boot_seed
    det, dk = "yolov8n_mixed5", "c10"
    ref = bse.paired_bootstrap_seed([raws[pt.ten_run(det, dk, "pf500", s)] for s in (0, 1, 2)],
                                    [raws[pt.ten_run(det, dk, "ukf")]], V, n_boot=B_FX, seed=0, metrics=["HOTA"])
    rows = {(r["detector"], r["dieu_kien"], r["hieu_so"], r["metric"]): r
            for r in pt.doc_csv(phien / "doi_chieu_video.csv", COT_CI, "doi_chieu_video.csv")}
    r = rows[(det, dk, "pf500-ukf", "HOTA")]
    assert float(r["ci_thap"]) == ref["HOTA"]["ci_low"] and float(r["ci_cao"]) == ref["HOTA"]["ci_high"]
    assert float(r["chenh"]) == ref["HOTA"]["diff"]
    # nhanh nhom: tinh doc lap bang vong lap Python (mo rong nhom thu cong, to hop doc lap, PF trung binh seed)
    idx_n = pt.doc_chi_so_csv((phien / "chi_so_nhom.csv").read_bytes())
    assert np.array_equal(idx_n, pt.rut_chi_so(B_FX, len(NHOM), 0))
    sai_max = 0.0
    for (det, dk, a, b, m), arr in kq["boot_nhom"].items():
        if m != "HOTA" and m != "IDSW":
            continue
        for i in range(0, B_FX, 7):
            vids = [v for g in idx_n[i] for v in NHOM[int(g)]]

            def gt_bl(bl):
                if bl == "pf500":
                    return sum(to_hop_doc_lap([raws[pt.ten_run(det, dk, bl, s)][v] for v in vids])[m]
                               for s in (0, 1, 2)) / 3
                return to_hop_doc_lap([raws[pt.ten_run(det, dk, bl)][v] for v in vids])[m]
            sai_max = max(sai_max, abs((gt_bl(a) - gt_bl(b)) - arr[i]))
    assert sai_max <= 1e-9, sai_max
    # CI nhom tinh lai tu mang bootstrap
    rn = {(r["detector"], r["dieu_kien"], r["hieu_so"], r["metric"]): r
          for r in pt.doc_csv(phien / "doi_chieu_nhom.csv", COT_CI, "doi_chieu_nhom.csv")}
    for (det, dk, a, b, m), arr in kq["boot_nhom"].items():
        lo, hi = np.percentile(arr, [2.5, 97.5], method="linear")
        rr = rn[(det, dk, f"{a}-{b}", m)]
        assert float(rr["ci_thap"]) == float(lo) and float(rr["ci_cao"]) == float(hi)
    # tai tinh tu chi so da luu tren dia
    idx_v = pt.doc_chi_so_csv((phien / "chi_so_video.csv").read_bytes())
    stacks = {run: bs._stack(raws[run], V) for run in raws}
    bv = pt.bootstrap(stacks, list(idx_v), gt, ["HOTA", "IDF1", "IDSW"])
    assert all(np.array_equal(bv[k], kq["boot_video"][k]) for k in bv)
    # bang R56 gia (cung quy tac lam tron R56) dat voi dung sai khai bao, khong nang ngam
    b56 = kc["bang_r56"]
    assert b56["dat"] and b56["dung_sai_ap_dung"] == gt["dung_sai"]["bang_r56"]
    for bang, d in b56["sai_so_max"].items():
        for m, x in d.items():
            assert x <= gt["dung_sai"]["bang_r56"][bang][m] + gt["dung_sai"]["bang_r56"]["eps_dau_phay_dong"], (bang, m, x)
    # ngan ghi de: phan-tich lan hai tren cung phien
    e = phai_loi(pt.phan_tich, goc, phien.name, chua="khong ghi de")
    return {"phien": phien.name, "sai_so_nhom_doc_lap_max": sai_max, "loi_lan_hai": e,
            "khac_dien_giai": len(kc["khac_dien_giai_hai_thiet_ke"]), "bang_r56_sai_so_max": b56["sai_so_max"]}


def t_chan_sai_hash():
    goc, _ = tao_fixture("fx_sai_hash", {"sua_sau_hash": "raw"})
    return {"loi": phai_loi(pt.chuan_bi, goc, GT_REL, chua="sai hash")}


def t_chan_thieu_video():
    goc, _ = tao_fixture("fx_thieu_video", {"raw_thieu_video": "r54-yolo11n_mixed5-c10-ukf"})
    return {"loi": phai_loi(pt.chuan_bi, goc, GT_REL, chua="sai tap video")}


def t_chan_nan():
    goc, _ = tao_fixture("fx_nan", {"raw_nan": "r54-yolov8n_coco-c25-pf500-s1"})
    return {"loi": phai_loi(pt.chuan_bi, goc, GT_REL, chua="khong huu han")}


def _sua_dong(i: int, cot: int, gia):
    def fn(h, rows):
        rows = [list(r) for r in rows]
        rows[i][cot] = gia(rows[i][cot]) if callable(gia) else gia
        return h, rows
    return fn


def _them_dong_trung(i: int, cot: int):
    """chen ban sao dong i voi gia tri cot 'cot' khac -> dict comprehension cu se gop am tham."""
    def fn(h, rows):
        r = list(rows[i])
        r[cot] = "1.0"
        return h, [*rows, r]
    return fn


def _header_trung(cot: str):
    """them cot trung ten, cung gia tri -> csv.DictReader cu khong phat hien."""
    def fn(h, rows):
        j = h.index(cot)
        return [*h, cot], [[*r, r[j]] for r in rows]
    return fn


def _dong_thua_cot(i: int):
    def fn(h, rows):
        rows = [list(r) for r in rows]
        rows[i] = [*rows[i], "thua"]
        return h, rows
    return fn


def t_chan_csv_schema():
    """Bang R56 / CSV metric hong nhung hash NHAT QUAN (lam hong truoc khi ghi/bam): loi phai den tu kiem schema,
    khoa duy nhat hoac so huu han o chuan-bi, khong phai tu kiem hash."""
    run0 = "r54-yolov8n_coco-c25-cv"
    ca = {
        "th_nan": ({"th_sua": _sua_dong(0, 5, "nan")}, "khong huu han"),
        "th_dong_trung": ({"th_sua": _them_dong_trung(3, 5)}, "dong trung khoa"),
        "th_header_trung": ({"th_sua": _header_trung("HOTA")}, "header trung"),
        "th_dong_thua_cot": ({"th_sua": _dong_thua_cot(2)}, "dong 4 co 9 cot"),
        "th_seed_sai": ({"th_sua": _sua_dong(3, 4, "1")}, "khong khop ten run"),
        "pf_nan": ({"pf_sua": _sua_dong(1, 5, "NaN")}, "khong huu han"),
        "pf_dong_trung": ({"pf_sua": _them_dong_trung(0, 3)}, "dong trung khoa"),
        "pf_header_trung": ({"pf_sua": _header_trung("IDSW_tb")}, "header trung"),
        "pf_so_seed": ({"pf_sua": _sua_dong(2, 2, 2)}, "so_seed"),
        "combined_header_trung": ({"csv_sua": (run0, "combined_metrics.csv", _header_trung("HOTA"))}, "header trung"),
        "combined_dong_thua_cot": ({"csv_sua": (run0, "combined_metrics.csv", _dong_thua_cot(0))}, "dong 2 co 5 cot"),
        "per_video_dong_trung": ({"csv_sua": (run0, "per_video_metrics.csv", _sua_dong(5, 0, V[0]))}, "dong trung khoa"),
        "per_video_dong_thua_cot": ({"csv_sua": (run0, "per_video_metrics.csv", _dong_thua_cot(4))}, "dong 6 co 5 cot"),
        "per_video_nan": ({"csv_sua": (run0, "per_video_metrics.csv", _sua_dong(1, 1, "nan"))}, "khong huu han"),
    }
    out = {}
    for ten, (bien, chua) in ca.items():
        goc, _ = tao_fixture(f"fx_schema_{ten}", bien)
        out[ten] = phai_loi(pt.chuan_bi, goc, GT_REL, chua=chua)
        assert "sai hash" not in out[ten], out[ten]
        assert not (goc / pt.VUNG_REL).exists() or not any((goc / pt.VUNG_REL).glob("phien_*")), ten   # khong tao phien
    return out


def t_ranh_gioi_lam_tron_r56():
    """Dung sai doi chieu bang R56 lay dung tu giao thuc that (khong nang ngam): ranh gioi hai phia tung bang/metric;
    lech 0,0001 IDSW o bang PF (lot qua max(tb, 0,0005) cu) bi chan; quy tac lam tron R56 (summary 3 chu so ->
    round(tb, 4)) mo phong ngau nhien luon nam trong dung sai khai bao."""
    gt = json.loads(GT_THAT.read_text(encoding="utf-8"))
    pt.kiem_giao_thuc(gt)
    tb = gt["dung_sai"]["bang_r56"]
    assert tb["tong_hop_36_run"] == {**tb["tong_hop_36_run"], "HOTA": 0.0005, "IDF1": 0.0005, "IDSW": 0}
    assert tb["pf500_trung_binh"] == {**tb["pf500_trung_binh"], "HOTA": 0.00055, "IDF1": 0.00055, "IDSW": 0.00005}
    run, key, pf = "r", ("d", "c"), gt["pf"]["bo_loc"]
    c_full = {run: {"HOTA": 0.5, "IDF1": 0.6, "IDSW": 100.0}}
    diem = {key: {pf: {"HOTA": 0.5, "IDF1": 0.6, "IDSW": 9961 / 3}}}

    def lech(th_d=None, pf_d=None):
        th = {run: {"HOTA": 50.0, "IDF1": 60.0, "IDSW": 100.0, **(th_d or {})}}
        pfr = {key: {"HOTA": 50.0, "IDF1": 60.0, "IDSW": 3320.3333, **(pf_d or {})}}
        return [(x[0], x[1]) for x in pt.doi_chieu_bang_r56(c_full, diem, th, pfr, gt)[0]]

    out = {}
    assert lech() == []                                                  # 3320.3333 vs 9961/3: lech 3,3e-5 <= 5e-5
    assert lech(th_d={"HOTA": 50.0005}) == [] and lech(th_d={"HOTA": 49.9995}) == []
    assert lech(th_d={"HOTA": 50.0006}) == [(run, "HOTA")] and lech(th_d={"IDF1": 59.9994}) == [(run, "IDF1")]
    assert lech(th_d={"IDSW": 101.0}) == [(run, "IDSW")] and lech(th_d={"IDSW": 99.0}) == [(run, "IDSW")]
    pk = "d-c-" + pf + "_tb"
    assert lech(pf_d={"HOTA": 50.00055}) == [] and lech(pf_d={"IDF1": 59.99945}) == []
    assert lech(pf_d={"HOTA": 50.0006}) == [(pk, "HOTA")] and lech(pf_d={"IDF1": 59.9994}) == [(pk, "IDF1")]
    assert lech(pf_d={"IDSW": 3320.3334}) == [(pk, "IDSW")]              # lech 6,7e-5: ma cu (0,0005) cho qua
    assert lech(pf_d={"IDSW": 3320.3332}) == [(pk, "IDSW")]
    out["ranh_gioi"] = "dat"
    # mo phong quy tac R56 tren gia tri ngau nhien: HOTA/IDF1 3 seed bat ky, IDSW so nguyen
    rng = np.random.default_rng(7)
    ti_le = {"HOTA": 0.0, "IDSW": 0.0}
    for _ in range(20000):
        x = rng.uniform(0.3, 0.8, 3)
        ref = round(sum(round(float(v) * 100, 3) for v in x) / 3, 4)
        ti_le["HOTA"] = max(ti_le["HOTA"], abs(float(np.mean(x)) * 100 - ref) / tb["pf500_trung_binh"]["HOTA"])
        n = rng.integers(0, 5000, 3)
        ref = round(float(sum(int(v) for v in n)) / 3, 4)
        ti_le["IDSW"] = max(ti_le["IDSW"], abs(float(n.sum()) / 3 - ref) / tb["pf500_trung_binh"]["IDSW"])
    assert ti_le["HOTA"] <= 1.0 + 1e-6 and ti_le["IDSW"] <= 1.0 + 1e-6, ti_le
    out["mo_phong_ti_le_lech_tren_dung_sai_max"] = ti_le
    # dau-cuoi: bang PF lech 0,0001 IDSW (hash nhat quan) -> chuan-bi qua, phan-tich chan truoc khi ghi dau ra
    goc, _ = tao_fixture("fx_r56_pf_idsw", {"pf_sua": _sua_dong(0, 5, lambda x: round(x + 0.0001, 4))})
    phien = pt.chuan_bi(goc, GT_REL)
    out["dau_cuoi"] = phai_loi(pt.phan_tich, goc, phien.name, chua="lech bang R56")
    assert sorted(x.name for x in phien.iterdir()) == sorted(pt.DAU_RA_CHUAN_BI)
    return out


def t_chan_thieu_run_va_trung():
    g1, _ = tao_fixture("fx_thieu_run", {"nguon_thieu_run": "r54-yolov8n_coco-c10-pf500-s2"})
    e1 = phai_loi(pt.chuan_bi, g1, GT_REL, chua="tap run sai")
    g2, _ = tao_fixture("fx_trung_run", {"nguon_trung_run": "r54-yolov8n_coco-c25-cv"})
    e2 = phai_loi(pt.chuan_bi, g2, GT_REL, chua="trung")
    return {"thieu_run": e1, "trung_run": e2}


def t_chan_lien_ket_cache():
    goc, _ = tao_fixture("fx_cache", {"cache_lech": "r54-yolov8n_mixed5-c25-ekf"})
    return {"loi": phai_loi(pt.chuan_bi, goc, GT_REL, chua="cache")}


def t_giao_thuc_doi_sau_chuan_bi():
    goc, _ = tao_fixture("fx_doi_gt")
    phien = pt.chuan_bi(goc, GT_REL)
    p = goc / GT_REL
    pt.kiem_vung_ghi(ROOT, p)
    p.write_bytes(p.read_bytes() + b"\n")
    e = phai_loi(pt.phan_tich, goc, phien.name, chua="giao thuc trong configs da doi")
    assert sorted(x.name for x in phien.iterdir()) == sorted(pt.DAU_RA_CHUAN_BI)     # khong ghi dau ra
    return {"loi": e}


def t_duong_dan_thoat_vung():
    goc = SES / "fx_vung"
    vung = goc / pt.VUNG_REL
    ghi(vung / "giu_cho.txt", "x")
    out = {}
    out["ngoai_vung"] = phai_loi(pt.kiem_vung_ghi, goc, goc / "results/phase2/khac/a.txt", chua="ghi ngoai vung")
    out["cham_cham"] = phai_loi(pt.kiem_vung_ghi, goc, vung / ".." / "a.txt", chua="ghi ngoai vung")
    out["doc_cham_cham"] = phai_loi(pt.duong_doc, goc, "../ngoai.json", chua="khong hop le")
    out["doc_tuyet_doi"] = phai_loi(pt.duong_doc, goc, str(ROOT / "README.md"), chua="khong hop le")
    out["ten_phien"] = phai_loi(pt.phan_tich, goc, "..\\phien_x", chua="ten phien")
    out["ghi_de"] = phai_loi(pt.ghi_moi, goc, vung / "giu_cho.txt", b"y", chua="khong ghi de")
    try:
        import _winapi
        dich = goc / "dich_junction"
        dich.mkdir(parents=True)
        jn = vung / "jn"
        _winapi.CreateJunction(str(dich), str(jn))
        try:
            out["junction"] = phai_loi(pt.kiem_vung_ghi, goc, jn / "a.txt", chua="symlink/junction")
        finally:
            os.rmdir(jn)
    except (ImportError, AttributeError, OSError) as e:
        out["junction"] = f"BO QUA: {type(e).__name__}: {e}"
    return out


CA = [t_combined_khac_tb_video, t_pf_thu_tu_phi_tuyen, t_lap_trung_bao_toan, t_cap_di_cung_va_giao_thuc_that,
      t_ghep_cap_moi_run_seed, t_dau_cuoi_fixture, t_chan_sai_hash, t_chan_thieu_video, t_chan_nan,
      t_chan_csv_schema, t_ranh_gioi_lam_tron_r56, t_chan_thieu_run_va_trung, t_chan_lien_ket_cache, t_giao_thuc_doi_sau_chuan_bi, t_duong_dan_thoat_vung]


def main() -> int:
    global SES
    tests = ROOT / pt.VUNG_REL / "tests"
    pt.kiem_vung_ghi(ROOT, tests)
    tests.mkdir(parents=True, exist_ok=True)
    SES = pt.kiem_vung_ghi(ROOT, tests / ("lan_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_pid{os.getpid()}"))
    SES.mkdir(exist_ok=False)
    t0 = time.time()
    kq = []
    for fn in CA:
        t1 = time.time()
        try:
            ct = fn()
            kq.append({"ca": fn.__name__, "dat": True, "chi_tiet": ct, "giay": round(time.time() - t1, 2)})
        except Exception as e:  # noqa: BLE001
            kq.append({"ca": fn.__name__, "dat": False, "loi": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:], "giay": round(time.time() - t1, 2)})
        print(f"{'DAT ' if kq[-1]['dat'] else 'TRUOT'} {fn.__name__} ({kq[-1]['giay']} s)")
    so_dat = sum(k["dat"] for k in kq)
    out = {"phien": SES.name, "luc": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "so_ca": len(kq), "so_dat": so_dat, "tat_ca_dat": so_dat == len(kq), "giay": round(time.time() - t0, 1),
           "sha256": {**pt.sha_ma(), "test_phan_tich_test40.py": pt.sha256_file(Path(__file__).resolve()),
                      "giao_thuc.json": pt.sha256_file(GT_THAT)},
           "moi_truong": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                          "executable": sys.executable},
           "B_fixture": B_FX, "ket_qua": kq}
    pt.ghi_moi(ROOT, SES / "ket_qua_test.json", pt.json_bytes(out))
    print(f"{so_dat}/{len(kq)} dat -> {pt._rel(ROOT, SES / 'ket_qua_test.json')}")
    return 0 if so_dat == len(kq) else 1


if __name__ == "__main__":
    sys.exit(main())
