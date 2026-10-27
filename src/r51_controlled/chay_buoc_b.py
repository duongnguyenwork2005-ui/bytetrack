"""
chay_buoc_b.py -- R51B: thuc thi DUNG muc 8 cua handoff/logs/review51/01_giao_thuc.md (giao thuc da approve).

Khong sua bo loc / tracker / evaluator: buoc track goi DUNG `run_experiment.stage_track` (cung duong voi run kc48),
buoc cham goi DUNG `run_trackeval.run` + `flatten_results` + `boot_stats.save_raw_fields` + `run_fingerprint`
(cung duong voi `run_experiment.stage_eval`). Chi doi noi ghi:
  - goc `config.PROCESSED_DIR` -> results/phase2/r51_controlled/runs/_processed/  (GT sao tu data/processed, kiem sha256;
    dau ra tracker moi; ban sao dau ra kc48 tai dung, kiem sha256 trung ban goc)
  - metric -> results/phase2/r51_controlled/eval/{R0,R1}/<dir>/  (config.RESULTS_DIR doi sang eval/_ket_qua cho log loi)
Khong ghi vao data/processed/trackers, results/trackeval, results/phase2/kc48_pf_dev (kiem truoc/sau, ghi nhat ky).

Cac buoc (moi buoc tiep tuc duoc; ghi mot dong vao results/phase2/r51_controlled/runs/nhat_ky_b.jsonl):
    python -X utf8 src/r51_controlled/chay_buoc_b.py chuan-bi   # kiem YAML/cache/GT, sao GT + dau ra tai dung
    python -X utf8 src/r51_controlled/chay_buoc_b.py track      # 4 run moi + 2 run neo (CPU, tu cache)
    python -X utf8 src/r51_controlled/chay_buoc_b.py neo        # run neo trung sha256 12/12 voi kc48? lech -> exit 2
    python -X utf8 src/r51_controlled/chay_buoc_b.py r1         # tao ban R1 (bo box o 572 frame khong nhan)
    python -X utf8 src/r51_controlled/chay_buoc_b.py cham       # TrackEval R0 (20 run) -> CONG R0 -> R1 (20 run)
    python -X utf8 src/r51_controlled/chay_buoc_b.py kiem-cham  # nhac lai cong R0; R1 vs CLEAR; R1 vs du doan buoc A
CONG R0 (giao thuc muc 8, sua R51B-F1): `cham` cham R0 cho moi run, roi `cong_r0` doi chieu R0 cua 16 run tai dung voi
combined/per_video cu (<= 1e-12; thieu file, khong huu han, lech tap video deu la loi); KHONG DAT -> dung, khong cham R1.
    python -X utf8 src/r51_controlled/chay_buoc_b.py bang       # bang COMBINED R0/R1 (diem, chua CI -- bootstrap la buoc C)
    python -X utf8 src/r51_controlled/chay_buoc_b.py doi-chieu  # thong tin: sha256/HOTA moi vs run cu khong tai dung
    python -X utf8 src/r51_controlled/chay_buoc_b.py tat-ca
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import config  # noqa: E402

GOC_THAT = Path(config.PROCESSED_DIR)                      # chi DOC GT tu day
EXP_PATH = "configs/phase2/r51_controlled/r51_b.yaml"
PROTO = ROOT / "results/phase2/r51_controlled/protocol"
RUNS = ROOT / "results/phase2/r51_controlled/runs"
GOC_MOI = RUNS / "_processed"
EVAL = ROOT / "results/phase2/r51_controlled/eval"
TESTS = ROOT / "results/phase2/r51_controlled/tests"
KC48_TRK = ROOT / "results/phase2/kc48_pf_dev/_processed/trackers/DETRAC-all"
TE_CU = ROOT / "results/trackeval/DETRAC-all"
SPLIT = "DETRAC-all"
GT_FILES = ("gt/gt.txt", "seqinfo.ini")
NHAT_KY = RUNS / "nhat_ky_b.jsonl"

#: run moi so sanh (nhan trong r51_b.yaml) va run neo -> run kc48 cung ho
MOI = ("bt-cv", "bt-ukf", "sort-cv30", "sort-ekf30")
NEO = {"bt-ekf": "p2-kc48_pf_dev-A-ekf", "bt-pf100-k1-s0": "p2-kc48_pf_dev-A-pf100-k1-s0"}
METRIC_COT = ("HOTA", "DetA", "AssA", "MOTA", "MOTP", "FP", "FN", "IDSW", "GT_dets", "IDF1", "IDP", "IDR")


def sha256_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def danh_sach_b() -> dict:
    return json.loads((PROTO / "danh_sach_run_B.json").read_text(encoding="utf-8"))


def videos() -> list[str]:
    return json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"]


def tai_dung() -> list[str]:
    t = danh_sach_b()["tai_dung"]
    return list(t["chinh"]) + list(t["phu_luc"])


def dir_moi(nhan: str) -> str:
    return f"p2-r51_controlled-A-{nhan}"


def dir_so_sanh() -> list[str]:
    """20 run cua phep so sanh (R0 va R1): 4 run moi + 16 run tai dung. Run neo chi dung de kiem tai dung."""
    return [dir_moi(n) for n in MOI] + tai_dung()


def doi_goc() -> None:
    """Doi goc doc/ghi TRUOC khi nap runner; moi ham doc thuoc tinh config luc goi."""
    config.PROCESSED_DIR = GOC_MOI
    config.RESULTS_DIR = EVAL / "_ket_qua"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def anh_chup_cam() -> dict:
    """Liet ke thu muc ma buoc B KHONG duoc ghi: ten + mtime cua file cap 1 (re, khong hash du lieu lon)."""
    out = {}
    for name, d in (("data_processed_trackers", GOC_THAT / "trackers" / SPLIT), ("results_trackeval", TE_CU),
                    ("kc48_trackers", KC48_TRK)):
        out[name] = sorted(p.name for p in d.iterdir()) if d.exists() else []
    out["kc48_mtime_max"] = max((p.stat().st_mtime for d in KC48_TRK.iterdir() for p in d.rglob("*") if p.is_file()),
                                default=None)
    out["te_cu_mtime_max"] = max((p.stat().st_mtime for n in tai_dung() for p in (TE_CU / n).glob("*") if p.is_file()),
                                 default=None)
    return out


def ghi_nhat_ky(dong: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with NHAT_KY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dong, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
def buoc_chuan_bi() -> int:
    import run_fingerprint as fp
    import run_tracking
    ds = danh_sach_b()
    vids = videos()
    loi = []
    if len(vids) != ds["chung"]["n_video"]:
        loi.append(f"manifest co {len(vids)} video, giao thuc {ds['chung']['n_video']}")
    # 1) YAML hien tai trung sha12 da khoa (muc 4.3)
    khoa = {x["yaml"]: x["yaml_sha12"] for x in ds["run_moi_so_sanh"] + ds["run_neo_kiem_tai_dung"]}
    yaml_kq = {}
    for y, s12 in khoa.items():
        now = hashlib.sha256(Path(run_tracking.resolve_tracker_yaml(y)).read_bytes()).hexdigest()[:12]
        yaml_kq[y] = {"khoa": s12, "hien_tai": now, "khop": now == s12}
        if now != s12:
            loi.append(f"YAML {y}: sha12 {now} != khoa {s12}")
    # 2) cache tren dia trung giao thuc
    c = fp.cache_fingerprint(ds["chung"]["cache_id"], vids)
    for k_proto, k_fp in (("cache_npz_sha256", "npz_sha256"), ("cache_meta_sha256", "meta_sha256"),
                          ("weights_sha256", "weights_sha256")):
        if c.get(k_fp) != ds["chung"][k_proto]:
            loi.append(f"cache {k_fp}: {c.get(k_fp)} != giao thuc {ds['chung'][k_proto]}")
    # 3) backend hien tai trung giao thuc
    be = fp.code_fingerprint()["backend"]
    for k, v in ds["chung"]["backend"].items():
        if str(be.get(k)) != str(v):
            loi.append(f"backend {k}: {be.get(k)} != {v}")
    # 4) sao GT + seqinfo, kiem sha256 byte trung ban goc
    GOC_MOI.mkdir(parents=True, exist_ok=True)
    (GOC_MOI / ".gitignore").write_text("*\n", encoding="utf-8")
    gt_rows = []
    for v in vids:
        for rel in GT_FILES:
            src, dst = GOC_THAT / SPLIT / v / rel, GOC_MOI / SPLIT / v / rel
            h = sha256_bytes(src)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            ok = sha256_bytes(dst) == h
            gt_rows.append({"video": v, "file": rel, "sha256": h, "ban_sao_trung_goc": ok})
            if not ok:
                loi.append(f"ban sao GT lech: {dst}")
    doi_goc()
    g = fp.gt_fingerprint(SPLIT, vids)
    if g.get("gt_sha256") != ds["chung"]["gt_sha256"] or g.get("seqinfo_sha256") != ds["chung"]["seqinfo_sha256"]:
        loi.append(f"dau van tay GT o goc moi khac giao thuc: {g.get('gt_sha256')} / {g.get('seqinfo_sha256')}")
    # 5) sao dau ra tracker cua 16 run tai dung (chi data/<v>.txt), kiem sha256 byte trung ban goc
    td_rows = []
    for d in tai_dung():
        for v in vids:
            src = KC48_TRK / d / "data" / f"{v}.txt"
            dst = GOC_MOI / "trackers" / SPLIT / d / "data" / f"{v}.txt"
            if not src.is_file():
                loi.append(f"thieu dau ra tai dung {src}")
                continue
            h = sha256_bytes(src)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            ok = sha256_bytes(dst) == h
            td_rows.append({"dir": d, "video": v, "sha256_byte": h, "sha256_lf_16": fp.sha256_file(src)[:16],
                            "ban_sao_trung_goc": ok})
            if not ok:
                loi.append(f"ban sao dau ra lech: {dst}")
    kq = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "yaml": yaml_kq, "cache": c, "backend": be,
          "gt_fingerprint_goc_moi": {k: g.get(k) for k in ("gt_sha256", "seqinfo_sha256", "n_missing_gt",
                                                             "n_missing_seqinfo")},
          "gt_files": gt_rows, "tai_dung_files": td_rows, "loi": loi}
    TESTS.mkdir(parents=True, exist_ok=True)
    (TESTS / "chuan_bi.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[chuan-bi] YAML {sum(x['khop'] for x in yaml_kq.values())}/{len(yaml_kq)} khop; GT {len(gt_rows)} file; "
          f"tai dung {len(td_rows)} file; loi {len(loi)}")
    for x in loi:
        print("  LOI", x)
    return 2 if loi else 0


def buoc_track() -> int:
    doi_goc()
    import run_experiment as rx
    assert rx.config is config and config.PROCESSED_DIR == GOC_MOI
    exp = rx.load_exp(EXP_PATH)
    vids = videos()
    rows = rx.plan(exp)
    print(f"[track] {len(rows)} lan chay: {[r['dir'] for r in rows]}")
    truoc = anh_chup_cam()
    rc = 0
    for r in rows:                                                  # checkpoint theo tung run (va tung video)
        t0, bd = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            rx.stage_track(exp, [r], vids, skip_existing=True)
            ok = 0
        except Exception as e:  # noqa: BLE001
            print(f"[track] {r['dir']} LOI {type(e).__name__}: {e}")
            ok = 1
        rc |= ok
        ghi_nhat_ky({"buoc": "track", "dir": r["dir"], "bat_dau": bd, "giay": round(time.perf_counter() - t0, 2),
                     "exit": ok})
    sau = anh_chup_cam()
    if sau != truoc:
        print("[track] CANH BAO: thu muc cam bi thay doi", {k: (truoc[k], sau[k]) for k in truoc if truoc[k] != sau[k]})
        rc |= 4
    ghi_nhat_ky({"buoc": "track_xong", "exit": rc, "thu_muc_cam_khong_doi": sau == truoc, "git_head": _git_head()})
    return rc


def _git_head() -> str:
    import det_cache
    return det_cache.git_commit()


def buoc_neo() -> int:
    import run_fingerprint as fp
    vids = videos()
    rows, lech = [], []
    for nhan, cu in NEO.items():
        for v in vids:
            a = fp.sha256_file(GOC_MOI / "trackers" / SPLIT / dir_moi(nhan) / "data" / f"{v}.txt")
            b = fp.sha256_file(KC48_TRK / cu / "data" / f"{v}.txt")
            ok = a == b and a != "MISSING"
            rows.append({"neo": dir_moi(nhan), "so_voi": cu, "video": v, "sha256_lf_moi": a, "sha256_lf_cu": b,
                         "trung": ok})
            if not ok:
                lech.append((nhan, v))
    tong = {n: sum(r["trung"] for r in rows if r["neo"] == dir_moi(n)) for n in NEO}
    kq = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": _git_head(),
          "quy_uoc_hash": fp.HASH_CONVENTION, "trung_theo_neo": tong, "rows": rows,
          "ket_luan": "dat" if not lech else "LECH -> dung nhanh bi anh huong, cho nguoi dung quyet dinh"}
    TESTS.mkdir(parents=True, exist_ok=True)
    (TESTS / "neo_kiem.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[neo] trung sha256: {tong} (yeu cau 12/12)")
    return 2 if lech else 0


def loc_r1(lines: list[str], bad: set[int]) -> list[str]:
    """Quy tac R1 (giao thuc muc 3.4): bo moi dong tracker co frame (cot 0) thuoc tap frame khong nhan."""
    import audit_protocol as ap
    giu = [ln for ln in lines if ln.strip() and int(float(ln.split(",", 1)[0])) not in bad]
    # doi chieu voi ham R1 cua giao thuc (ap.mask_rows nhan tap frame CO nhan) tren cung du lieu
    lab = {int(float(ln.split(",", 1)[0])) for ln in lines if ln.strip()} - bad
    assert [",".join(x) for x in ap.mask_rows([ln.split(",") for ln in lines if ln.strip()], lab)] == giu
    return giu


def buoc_r1() -> int:
    import run_fingerprint as fp
    unl = json.loads((PROTO / "frame_khong_nhan_danh_sach.json").read_text(encoding="utf-8"))
    vids = videos()
    rows = []
    for d in dir_so_sanh():
        for v in vids:
            src = GOC_MOI / "trackers" / SPLIT / d / "data" / f"{v}.txt"
            dst = GOC_MOI / "trackers" / SPLIT / f"{d}-R1" / "data" / f"{v}.txt"
            bad = set(int(x) for x in unl[v])
            lines = src.read_text(encoding="utf-8").splitlines()
            giu = loc_r1(lines, bad)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("".join(ln + "\n" for ln in giu), encoding="utf-8")
            n_src = sum(1 for ln in lines if ln.strip())
            rows.append({"dir": d, "video": v, "n_dong_R0": n_src, "n_dong_R1": len(giu), "n_bo": n_src - len(giu),
                         "sha256_lf_R0_16": fp.sha256_file(src)[:16], "sha256_lf_R1_16": fp.sha256_file(dst)[:16]})
    (TESTS / "r1_tao.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    tong = {}
    for r in rows:
        tong[r["dir"]] = tong.get(r["dir"], 0) + r["n_bo"]
    print(f"[r1] {len(rows)} file; box bo theo run: {tong}")
    return 0


def _cfg_cua(d: str):
    """tracker_cfg dung cho dau van tay cham: run moi tu r51_b.yaml, run tai dung tu run_meta.json kc48."""
    import run_tracking
    goc = d[:-3] if d.endswith("-R1") else d
    if goc.startswith("p2-r51_controlled-"):
        rm = json.loads((GOC_MOI / "trackers" / SPLIT / goc / "run_meta.json").read_text(encoding="utf-8"))
    else:
        rm = json.loads((KC48_TRK / goc / "run_meta.json").read_text(encoding="utf-8"))
    cfg = run_tracking.load_tracker_cfg(rm["tracker_yaml"], rm.get("overrides") or None)
    r = {"dir": d, "set": rm.get("overrides") or {}, "conf_min": rm["conf_min"], "seed": rm["seed"]}
    return cfg, r, rm


SAI_SO_R0 = 1e-12


def so_r0_voi_cu(moi_comb, cu_comb, moi_pv, cu_pv, tol: float = SAI_SO_R0) -> tuple[list[str], float | None,
                                                                                         float | None]:
    """So R0 moi voi so cu cua MOT run tai dung (giao thuc muc 8). Tham so la DataFrame hoac None (thieu file).

    Tra (loi, chenh_lech_max_combined, chenh_lech_max_per_video). Thieu file / thieu cot / thieu dong COMBINED /
    tap video khac / gia tri khong huu han / chenh > tol deu la LOI (thieu bang chung = khong dat).
    """
    import numpy as np
    loi = []
    if any(x is None for x in (moi_comb, cu_comb, moi_pv, cu_pv)):
        return ["thieu file metric moi hoac cu"], None, None
    cols = list(METRIC_COT)
    for ten, df in (("combined moi", moi_comb), ("combined cu", cu_comb), ("per_video moi", moi_pv),
                    ("per_video cu", cu_pv)):
        thieu = [c for c in cols + ["video"] if c not in df.columns]
        if thieu:
            loi.append(f"{ten} thieu cot {thieu}")
    if loi:
        return loi, None, None
    if len(moi_comb) != 1 or len(cu_comb) != 1:
        return [f"so dong combined moi/cu {len(moi_comb)}/{len(cu_comb)} != 1"], None, None
    a, b = moi_comb[cols].to_numpy(float), cu_comb[cols].to_numpy(float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        loi.append("combined co gia tri khong huu han")
    dc = float(np.max(np.abs(a - b)))
    pa, pb = moi_pv.set_index("video").sort_index(), cu_pv.set_index("video").sort_index()
    if list(pa.index) != list(pb.index) or pa.index.has_duplicates:
        return loi + [f"tap video per_video khac: {sorted(set(pa.index) ^ set(pb.index))}"], dc, None
    va, vb = pa[cols].to_numpy(float), pb[cols].to_numpy(float)
    if not (np.isfinite(va).all() and np.isfinite(vb).all()):
        loi.append("per_video co gia tri khong huu han")
    dp = float(np.max(np.abs(va - vb)))
    if not dc <= tol:                    # `not <=` de NaN cung la loi
        loi.append(f"combined lech {dc} > {tol}")
    if not dp <= tol:
        loi.append(f"per_video lech {dp} > {tol}")
    return loi, dc, dp


def cong_r0() -> list[str]:
    """CONG bat buoc truoc R1 (giao thuc muc 8): R0 cua moi run tai dung tai tao so cu <= 1e-12.

    Ghi tests/cong_r0.json + mot dong nhat ky; tra danh sach loi (rong = dat)."""
    import pandas as pd

    def doc(p: Path):
        return pd.read_csv(p) if p.is_file() else None

    rows, loi = [], []
    for d in tai_dung():
        comb = doc(EVAL / "R0" / d / "combined_metrics.csv")
        l, dc, dp = so_r0_voi_cu(comb, doc(TE_CU / d / "combined_metrics.csv"),
                                 doc(EVAL / "R0" / d / "per_video_metrics.csv"),
                                 doc(TE_CU / d / "per_video_metrics.csv"))
        rows.append({"dir": d, "chenh_lech_max_combined": dc, "chenh_lech_max_per_video": dp, "loi": l, "dat": not l})
        loi += [f"R0 {d}: {x}" for x in l]
    kq = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sai_so": SAI_SO_R0, "n_run": len(rows),
          "n_dat": sum(r["dat"] for r in rows), "rows": rows, "loi": loi}
    TESTS.mkdir(parents=True, exist_ok=True)
    (TESTS / "cong_r0.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    ghi_nhat_ky({"buoc": "cong_r0", "thoi_diem": kq["thoi_diem"], "n_dat": kq["n_dat"], "n_run": len(rows),
                 "dat": not loi})
    print(f"[cong R0] {kq['n_dat']}/{len(rows)} run tai dung tai tao so cu (<= {SAI_SO_R0}); loi {len(loi)}")
    return loi


def trinh_tu_cham(dirs: list[str], cham_mot, cong) -> int:
    """Thu tu bat buoc: cham R0 cho moi run -> CONG R0 -> chi khi dat moi cham R1.

    cham_mot(quy_tac, goc) -> True neu cham (hoac tai dung) thanh cong; cong() -> danh sach loi.
    Exit: 0 dat; 1 co lan cham R0 that bai (khong chay cong, khong R1); 2 cong R0 khong dat (khong R1);
    3 co lan cham R1 that bai."""
    if not all([cham_mot("R0", g) for g in dirs]):       # list -> cham het R0 roi moi xet
        print("[cham] R0 co lan that bai -> DUNG truoc cong R0, khong cham R1")
        return 1
    loi = cong()
    if loi:
        print("[cham] cong R0 KHONG DAT -> DUNG, khong cham R1:\n  " + "\n  ".join(loi))
        return 2
    return 0 if all([cham_mot("R1", g) for g in dirs]) else 3


def buoc_cham(chi: list[str] | None = None) -> int:
    doi_goc()
    import run_fingerprint as fp
    import run_trackeval
    from boot_stats import save_raw_fields
    import run_experiment as rx
    exp = rx.load_exp(EXP_PATH)
    vids = videos()
    fp.check_eval_inputs(SPLIT, vids)
    truoc = anh_chup_cam()

    def cham_mot(quy_tac: str, goc: str) -> bool:
        d = goc if quy_tac == "R0" else f"{goc}-R1"
        out = EVAL / quy_tac / goc
        cfg, r, _ = _cfg_cua(d)
        new_fp = fp.run_fingerprint(exp, r, vids, cfg, "mot", False)
        new_fp["quy_tac"] = quy_tac
        have = all((out / f).exists() for f in ("combined_metrics.csv", "per_video_metrics.csv", "raw_fields.json"))
        if have and not fp.compare(fp.load(out / "eval_meta.json"), new_fp):
            print(f"  [cham {quy_tac}] {goc:<40} TAI SU DUNG (dau van tay trung)")
            ghi_nhat_ky({"buoc": "cham", "quy_tac": quy_tac, "dir": goc, "tai_su_dung": True, "exit": 0,
                         "luc": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            return True
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                results, _ = run_trackeval.run(SPLIT, d, vids, "mot")
        except Exception as e:  # noqa: BLE001
            print(f"  [cham {quy_tac}] {goc} THAT BAI {type(e).__name__}: {e}")
            ghi_nhat_ky({"buoc": "cham", "quy_tac": quy_tac, "dir": goc, "exit": 1, "loi": f"{type(e).__name__}: {e}"})
            return False
        df = run_trackeval.flatten_results(results, d)
        out.mkdir(parents=True, exist_ok=True)
        df[df.video != "COMBINED_SEQ"].sort_values("video").to_csv(out / "per_video_metrics.csv", index=False)
        df[df.video == "COMBINED_SEQ"].to_csv(out / "combined_metrics.csv", index=False)
        save_raw_fields(results, d, out / "raw_fields.json")
        fp.save(out / "eval_meta.json", new_fp, {"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        c = df[df.video == "COMBINED_SEQ"].iloc[0]
        giay = time.perf_counter() - t0
        print(f"  [cham {quy_tac}] {goc:<40} HOTA {c.HOTA:.4f} AssA {c.AssA:.4f} IDF1 {c.IDF1:.4f} "
              f"IDSW {int(c.IDSW)} ({giay:.0f}s)", flush=True)
        ghi_nhat_ky({"buoc": "cham", "quy_tac": quy_tac, "dir": goc, "giay": round(giay, 2), "exit": 0})
        return True

    dirs = [g for g in dir_so_sanh() if not chi or g in chi]
    rc = trinh_tu_cham(dirs, cham_mot, cong_r0)
    sau = anh_chup_cam()
    if sau != truoc:
        print("[cham] CANH BAO: thu muc cam bi thay doi")
        rc = rc or 4
    ghi_nhat_ky({"buoc": "cham_xong", "exit": rc, "thu_muc_cam_khong_doi": sau == truoc})
    return rc


def buoc_kiem_cham() -> int:
    rows, loi = [], []
    # (a) nhac lai cong R0 (da bat buoc trong `cham` truoc R1), ghi vao cung bao cao
    for x in cong_r0():
        loi.append(x)
    for r in json.loads((TESTS / "cong_r0.json").read_text(encoding="utf-8"))["rows"]:
        rows.append({"kiem": "R0_tai_tao_cu", **{k: r[k] for k in ("dir", "chenh_lech_max_combined",
                                                                   "chenh_lech_max_per_video", "dat")}})
    import pandas as pd
    # (b) R1 vs R0 theo CLEAR (chinh xac, audit_protocol.clear_after_mask): FP giam dung so box bo; TP/FN/IDSW giu
    r1 = json.loads((TESTS / "r1_tao.json").read_text(encoding="utf-8"))
    n_bo = {}
    for x in r1:
        n_bo[x["dir"]] = n_bo.get(x["dir"], 0) + x["n_bo"]
    for d in dir_so_sanh():
        a = pd.read_csv(EVAL / "R0" / d / "combined_metrics.csv").iloc[0]
        b = pd.read_csv(EVAL / "R1" / d / "combined_metrics.csv").iloc[0]
        ok = (int(a.FP) - int(b.FP) == n_bo[d] and int(a.FN) == int(b.FN) and int(a.IDSW) == int(b.IDSW)
              and int(a.GT_dets) == int(b.GT_dets))
        rows.append({"kiem": "R1_clear_khop_ly_thuyet", "dir": d, "n_box_bo": n_bo[d], "FP_R0": int(a.FP),
                     "FP_R1": int(b.FP), "IDSW_R0": int(a.IDSW), "IDSW_R1": int(b.IDSW), "dat": ok})
        if not ok:
            loi.append(f"R1 {d}: CLEAR khong khop ly thuyet")
    # (c) cot r1_CLR_FP_moi_chinh_xac du doan o buoc A (kiem_ke_run.csv) cho run tai dung
    kk = pd.read_csv(PROTO / "kiem_ke_run.csv")
    col = "r1_CLR_FP_moi_chinh_xac"
    if col in kk.columns and "dir" in kk.columns:
        for d in tai_dung():
            m = kk[kk["dir"] == d]
            if len(m):
                b = pd.read_csv(EVAL / "R1" / d / "combined_metrics.csv").iloc[0]
                ok = int(m[col].iloc[0]) == int(b.FP)
                rows.append({"kiem": "R1_FP_trung_du_doan_buoc_A", "dir": d, "du_doan": int(m[col].iloc[0]),
                             "thuc_te": int(b.FP), "dat": ok})
                if not ok:
                    loi.append(f"R1 {d}: FP {int(b.FP)} != du doan buoc A {int(m[col].iloc[0])}")
    kq = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows, "loi": loi}
    (TESTS / "kiem_cham.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"[kiem-cham] dat theo loai: {dict(Counter(r['kiem'] for r in rows if r['dat']))}; "
          f"tong theo loai: {dict(Counter(r['kiem'] for r in rows))}; loi {len(loi)}")
    for x in loi:
        print("  LOI", x)
    return 2 if loi else 0


def nhan_cua(d: str) -> tuple[str, int | None]:
    if d.startswith("p2-r51_controlled-A-"):
        return d[len("p2-r51_controlled-A-"):], None
    t = d[len("p2-kc48_pf_dev-A-"):]
    if t == "ekf":
        return "bt-ekf", None
    base, s = t.rsplit("-s", 1)
    return f"bt-{base}", int(s)


def buoc_bang() -> int:
    import pandas as pd
    rows = []
    for quy_tac in ("R0", "R1"):
        for d in dir_so_sanh():
            c = pd.read_csv(EVAL / quy_tac / d / "combined_metrics.csv").iloc[0]
            nhan, seed = nhan_cua(d)
            rows.append({"quy_tac": quy_tac, "nhan": nhan, "seed": seed, "dir": d,
                         "nguon": "moi" if d.startswith("p2-r51_controlled-") else "tai_dung",
                         **{k: float(c[k]) for k in METRIC_COT}})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL / "combined_tung_run.csv", index=False)
    # PF: COMBINED rieng tung seed roi trung binh 3 seed (giao thuc muc 7)
    g = (df.groupby(["quy_tac", "nhan", "nguon"], sort=False)
           .agg(n_seed=("dir", "size"), **{k: (k, "mean") for k in METRIC_COT}).reset_index())
    g.to_csv(EVAL / "combined_theo_bien_the.csv", index=False)
    # hieu so diem (chua CI; bootstrap ghep cap theo video la buoc C)
    doi = [("Q1", "bt-ekf", "bt-cv"), ("Q1", "bt-ukf", "bt-cv"), ("Q1", "bt-ukf", "bt-ekf"),
           ("Q1", "bt-pf100", "bt-ekf"), ("Q1", "bt-pf100-k1", "bt-pf100"), ("Q1", "bt-pf100-k1", "bt-ekf"),
           ("Q2", "bt-cv", "sort-cv30"), ("Q2", "bt-ekf", "sort-ekf30")]
    hs = []
    for quy_tac in ("R0", "R1"):
        gg = g[g.quy_tac == quy_tac].set_index("nhan")
        for q, a, b in doi:
            hs.append({"quy_tac": quy_tac, "cau_hoi": q, "doi_chung": f"{a} - {b}",
                       **{k: float(gg.loc[a, k] - gg.loc[b, k]) for k in ("HOTA", "AssA", "DetA", "IDF1", "IDSW")}})
    pd.DataFrame(hs).to_csv(EVAL / "hieu_so_diem.csv", index=False)
    pd.set_option("display.width", 200)
    print(g[g.quy_tac == "R0"][["nhan", "nguon", "n_seed", "HOTA", "AssA", "DetA", "IDF1", "IDSW"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(pd.DataFrame(hs).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    return 0


#: run cu KHONG tai dung cung ten (giao thuc muc 4.2, 4.3) -> chi doi chieu thong tin, khong thay so moi
CU_KHONG_TAI_DUNG = {"bt-cv": "p2-expA_detrac_dev-c25-cv", "bt-ukf": "p2-expA_detrac_dev-c25-ukf",
                     "sort-cv30": "p2-expB_detrac_dev_fix1-A-sort-cv30",
                     "sort-ekf30": "p2-expB_detrac_dev_fix1-A-sort-ekf30"}


def buoc_doi_chieu() -> int:
    """Thong tin: sha256 dau ra moi so voi run cu khong tai dung + HOTA cu/moi. Khong anh huong ket qua B."""
    import pandas as pd
    import run_fingerprint as fp
    vids = videos()
    rows = []
    for nhan, cu in CU_KHONG_TAI_DUNG.items():
        n_trung = sum(fp.sha256_file(GOC_MOI / "trackers" / SPLIT / dir_moi(nhan) / "data" / f"{v}.txt")
                      == fp.sha256_file(GOC_THAT / "trackers" / SPLIT / cu / "data" / f"{v}.txt") for v in vids)
        moi = pd.read_csv(EVAL / "R0" / dir_moi(nhan) / "combined_metrics.csv").iloc[0]
        p_cu = TE_CU / cu / "combined_metrics.csv"
        c = pd.read_csv(p_cu).iloc[0] if p_cu.exists() else None
        rows.append({"nhan": nhan, "run_cu": cu, "n_video_trung_sha256_lf": n_trung, "n_video": len(vids),
                     **{f"{k}_moi": float(moi[k]) for k in ("HOTA", "AssA", "IDF1", "IDSW")},
                     **{f"{k}_cu": (None if c is None else float(c[k])) for k in ("HOTA", "AssA", "IDF1", "IDSW")}})
    (TESTS / "doi_chieu_run_cu.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("buoc", choices=["chuan-bi", "track", "neo", "r1", "cham", "kiem-cham", "bang", "doi-chieu",
                                     "tat-ca"])
    a = ap.parse_args()
    import os
    os.chdir(ROOT)
    buoc = {"chuan-bi": buoc_chuan_bi, "track": buoc_track, "neo": buoc_neo, "r1": buoc_r1, "cham": buoc_cham,
            "kiem-cham": buoc_kiem_cham, "bang": buoc_bang, "doi-chieu": buoc_doi_chieu}
    thu_tu = list(buoc) if a.buoc == "tat-ca" else [a.buoc]
    for b in thu_tu:
        t0, bd = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rc = buoc[b]()
        ghi_nhat_ky({"buoc_cli": b, "bat_dau": bd, "giay": round(time.perf_counter() - t0, 2), "exit": rc})
        print(f"== {b}: exit {rc} ({time.perf_counter() - t0:.1f}s)", flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
