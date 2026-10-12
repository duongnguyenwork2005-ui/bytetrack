"""
kiem_ke_dau_vao_deta.py -- R50A / P487415eb: CHI KIEM KE dau vao cho thiet ke phan tich DetA PF < EKF (dev12).

Liet ke file, kich thuoc, truong / cot, so dong, va co / khong co cac truong can cho tung nhom chi so cua
handoff/logs/review48/09_thiet_ke_phan_tich_deta_pf_ekf.md. KHONG tinh chi so phan tich nao (khong doc gia tri
box / score de thong ke, khong ghep GT, khong so hash giua run), KHONG chay tracking, KHONG them instrumentation.
Chi doc; ghi duy nhat results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_ke_dau_vao_deta.py
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "results/phase2/kc48_pf_dev"
TRK = DEV / "_processed/trackers/DETRAC-all"
GT = DEV / "_processed/DETRAC-all"
TEV = ROOT / "results/trackeval/DETRAC-all"
CACHE = ROOT / "data/interim/det_cache/yolov8n-coco_conf0.05_all"
MOC2 = ROOT / "data/processed/trackers/DETRAC-all"
MOC2_RUN = ["p2-expA_detrac_dev-c25-ekf", "p2-expA_detrac_dev-c25-cv", "p2-expA_detrac_dev-c25-ukf"] + [
    f"p2-expA_detrac_dev-c25-pf{n}-s{s}" for n in (100, 300, 500) for s in (0, 1, 2)]
OUT = DEV / "thiet_ke_deta/kiem_ke.json"
META_KEYS = ["tracker_backend", "cache_id", "tracker_yaml", "tracker_yaml_sha256_12", "conf_min", "classes",
             "class_map", "max_det", "fps", "elapsed_mode", "seed", "assoc_log", "pf_diag", "git_commit",
             "track_buffer_frames", "input_fingerprint", "overrides"]


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def dem_dong(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def dong_dau(p: Path) -> str:
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.readline().rstrip("\r\n")


def mo_ta_csv(p: Path, co_header: bool) -> dict:
    if not p.exists():
        return {"ton_tai": False}
    first = dong_dau(p)
    n = dem_dong(p)
    cols = next(csv.reader([first])) if first else []
    d = {"ton_tai": True, "byte": p.stat().st_size, "so_dong": n}
    if co_header:
        d.update(cot=cols, so_dong_du_lieu=max(n - 1, 0))
    else:
        d.update(so_cot=len(cols), so_dong_du_lieu=n)
    return d


def meta_run(p: Path) -> dict:
    if not p.exists():
        return {"ton_tai": False}
    m = json.loads(p.read_text(encoding="utf-8"))
    out = {"ton_tai": True, "byte": p.stat().st_size, "khoa_cap_1": sorted(m)}
    for k in META_KEYS:
        out[k] = m.get(k, "<khong co>")
    code = m.get("code") or {}
    be = code.get("backend") if isinstance(code, dict) else None
    out["code.backend"] = be if be is not None else "<khong co>"
    cfg = m.get("tracker_cfg") or {}
    out["tracker_cfg_khoa_pf"] = {k: cfg[k] for k in sorted(cfg) if k.startswith("pf_")}
    out["tracker_cfg_bytetrack"] = {k: cfg.get(k) for k in ("track_high_thresh", "track_low_thresh",
                                                            "new_track_thresh", "match_thresh", "fuse_score",
                                                            "track_buffer")}
    return out


def mo_ta_parquet(p: Path) -> dict:
    if not p.exists():
        return {"ton_tai": False}
    d = {"ton_tai": True, "byte": p.stat().st_size}
    try:
        import fastparquet  # noqa: PLC0415
        pf = fastparquet.ParquetFile(str(p))
        d.update(cot=list(pf.columns), so_dong=int(pf.count()))
    except Exception as e:  # noqa: BLE001  (vd Smart App Control chan .pyd)
        d["loi_doc_metadata"] = f"{type(e).__name__}: {e}"
    return d


def kiem_ke_run_tracker(root: Path, run: str, videos: list[str], co_assoc_pf: bool) -> dict:
    d = root / run
    out = {"thu_muc": rel(d), "ton_tai": d.exists(), "run_meta": meta_run(d / "run_meta.json"),
           "file_cap_1": sorted(x.name for x in d.iterdir()) if d.exists() else [],
           "thu_muc_con": sorted(x.name for x in d.iterdir() if x.is_dir()) if d.exists() else []}
    vids = {}
    for v in videos:
        e = {"data_txt": mo_ta_csv(d / "data" / f"{v}.txt", co_header=False),
             "data_cls": mo_ta_csv(d / "data_cls" / f"{v}.csv", co_header=True),
             "timing": mo_ta_csv(d / "timing" / f"{v}.csv", co_header=True),
             "video_meta": {"ton_tai": (d / "video_meta" / f"{v}.json").exists()}}
        if co_assoc_pf:
            e["assoc_log"] = {n: mo_ta_parquet(d / "assoc_log" / f"{v}_{n}.parquet")
                              for n in ("tracks", "dets", "events")}
            e["pf_diag"] = mo_ta_csv(d / "pf_diag" / f"{v}.csv", co_header=True)
        vids[v] = e
    out["theo_video"] = vids
    out["tom_tat"] = {
        "n_video_co_data_txt": sum(x["data_txt"]["ton_tai"] for x in vids.values()),
        "so_cot_data_txt": sorted({x["data_txt"].get("so_cot") for x in vids.values() if x["data_txt"]["ton_tai"]}),
        "tong_dong_data_txt": sum(x["data_txt"].get("so_dong_du_lieu", 0) for x in vids.values()),
        "co_assoc_log": (d / "assoc_log").is_dir(), "co_pf_diag": (d / "pf_diag").is_dir()}
    if co_assoc_pf:
        out["tom_tat"]["n_video_assoc_log_du_3_bang"] = sum(
            all(x["assoc_log"][n]["ton_tai"] for n in ("tracks", "dets", "events")) for x in vids.values())
        out["tom_tat"]["n_video_pf_diag"] = sum(x["pf_diag"]["ton_tai"] for x in vids.values())
    return out


def kiem_ke_trackeval(run: str, videos: list[str]) -> dict:
    d = TEV / run
    out = {"thu_muc": rel(d), "ton_tai": d.exists(),
           "file": {x.name: x.stat().st_size for x in sorted(d.iterdir())} if d.exists() else {}}
    out["per_video_metrics"] = mo_ta_csv(d / "per_video_metrics.csv", co_header=True)
    out["combined_metrics"] = mo_ta_csv(d / "combined_metrics.csv", co_header=True)
    rf = d / "raw_fields.json"
    if rf.exists():
        raw = json.loads(rf.read_text(encoding="utf-8"))
        khoa_video = sorted(raw) if isinstance(raw, dict) else []
        mau = raw[khoa_video[0]] if khoa_video and isinstance(raw[khoa_video[0]], dict) else {}
        out["raw_fields"] = {
            "khoa_cap_1": khoa_video, "n_video_dev12_co_mat": sum(v in raw for v in videos) if isinstance(raw, dict) else 0,
            "truong_moi_video": sorted(mau),
            "do_dai_mang_truong": {k: (len(v) if isinstance(v, list) else "vo_huong") for k, v in mau.items()}}
    em = d / "eval_meta.json"
    if em.exists():
        out["eval_meta_khoa"] = sorted(json.loads(em.read_text(encoding="utf-8")))
    return out


def kiem_ke_cache(videos: list[str]) -> dict:
    out = {"thu_muc": rel(CACHE), "ton_tai": CACHE.exists()}
    mp = CACHE / "meta.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        out["meta_khoa"] = sorted(m)
        out["meta_chon"] = {k: m.get(k) for k in ("backend", "conf", "classes", "max_det", "dataset", "part")}
    vids = {}
    for v in videos:
        npz, js = CACHE / f"{v}.npz", CACHE / f"{v}.json"
        e = {"npz_ton_tai": npz.exists(), "json_ton_tai": js.exists()}
        if npz.exists():
            e["npz_byte"] = npz.stat().st_size
            with np.load(npz) as z:                      # chi doc ten + shape + dtype
                e["mang"] = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files}
        if js.exists():
            e["json_khoa"] = sorted(json.loads(js.read_text(encoding="utf-8")))
        vids[v] = e
    out["theo_video"] = vids
    out["ghi_chu"] = "mang data (n,6) = x1,y1,x2,y2,conf,cls theo det_cache.to_boxes; chua loc conf_min / lop"
    return out


def kiem_ke_gt(videos: list[str]) -> dict:
    vids = {}
    for v in videos:
        g = GT / v / "gt/gt.txt"
        s = GT / v / "seqinfo.ini"
        vids[v] = {"gt": mo_ta_csv(g, co_header=False), "seqinfo_ton_tai": s.exists(),
                   "seqinfo": dict(x.split("=", 1) for x in s.read_text(encoding="utf-8").splitlines()
                                   if "=" in x) if s.exists() else {}}
    return {"thu_muc": rel(GT), "theo_video": vids,
            "tong_dong_gt": sum(x["gt"].get("so_dong_du_lieu", 0) for x in vids.values())}


def truong_can(kc48: dict, moc2: dict, tev: dict) -> list[dict]:
    """Co / khong co truong can cho tung nhom chi so (chi xet su ton tai cua truong / file)."""
    r_pf = kc48.get("p2-kc48_pf_dev-A-pf100-s0", {})
    r_m2 = moc2.get("p2-expA_detrac_dev-c25-pf100-s0", {})
    t_pf = tev.get("p2-kc48_pf_dev-A-pf100-s0", {})
    v0 = next(iter(r_m2.get("theo_video", {}) or {}), None)
    al = (r_m2.get("theo_video", {}).get(v0, {}) or {}).get("assoc_log", {}) if v0 else {}
    pfd = (r_m2.get("theo_video", {}).get(v0, {}) or {}).get("pf_diag", {}) if v0 else {}
    cols_tracks = al.get("tracks", {}).get("cot", [])
    cols_dets = al.get("dets", {}).get("cot", [])
    cols_pfd = pfd.get("cot", [])
    raw_truong = t_pf.get("raw_fields", {}).get("truong_moi_video", [])
    timing_cols = (r_pf.get("theo_video", {}).get("MVI_20052", {}) or {}).get("timing", {}).get("cot", [])
    so_cot = r_pf.get("tom_tat", {}).get("so_cot_data_txt", [])
    return [
        {"truong": "box dau ra tung frame (frame, id, x, y, w, h, score)", "kc48_pf_dev": so_cot == [7],
         "nguon": "data/<video>.txt (MOT 7 cot)"},
        {"truong": "chi so detection (idx) da cap nhat track trong file dau ra", "kc48_pf_dev": False,
         "nguon": "run_tracking.py ghi 7 cot, bo idx; xem so_cot_data_txt", "so_cot_data_txt": so_cot},
        {"truong": "so detection dua vao tracker moi frame (n_dets)", "kc48_pf_dev": "n_dets" in timing_cols,
         "nguon": "timing/<video>.csv", "cot": timing_cols},
        {"truong": "HOTA_TP/FN/FP theo 19 nguong alpha moi video", "kc48_pf_dev":
            all(x in raw_truong for x in ("HOTA_TP", "HOTA_FN", "HOTA_FP")), "nguon": "raw_fields.json",
         "truong_co": raw_truong},
        {"truong": "tong do tuong dong cua cap TP (LocA)", "kc48_pf_dev": any("Loc" in x for x in raw_truong),
         "nguon": "raw_fields.json (khong co truong Loc*)"},
        {"truong": "box DU DOAN truoc update (prior) tung track", "kc48_pf_dev": r_pf.get("tom_tat", {}).get("co_assoc_log", False),
         "moc2_c25": all(x in cols_tracks for x in ("x", "y", "w", "h")), "nguon": "assoc_log/<video>_tracks.parquet"},
        {"truong": "tang ghep (stage 1/2/3) + det_idx da ghep", "kc48_pf_dev": r_pf.get("tom_tat", {}).get("co_assoc_log", False),
         "moc2_c25": "stage" in cols_tracks and "matched_det" in cols_tracks and "stage" in cols_dets,
         "nguon": "assoc_log tracks/dets"},
        {"truong": "tier detection (high/low) + ly do tu choi", "kc48_pf_dev": False,
         "moc2_c25": "tier" in cols_dets and "reject_reason" in cols_dets, "nguon": "assoc_log dets"},
        {"truong": "su kien vong doi (new/lost/refind/removed)", "kc48_pf_dev": False,
         "moc2_c25": bool(al.get("events", {}).get("ton_tai")), "nguon": "assoc_log events"},
        {"truong": "ESS / resample / suy bien / maha2_best moi update PF", "kc48_pf_dev":
            r_pf.get("tom_tat", {}).get("co_pf_diag", False),
         "moc2_c25": all(x in cols_pfd for x in ("ess_ratio", "degenerate", "maha2_best")), "nguon": "pf_diag/<video>.csv",
         "cot": cols_pfd},
        {"truong": "log-likelihood / innovation (NIS) day du moi update, ca EKF lan PF", "kc48_pf_dev": False,
         "moc2_c25": False, "nguon": "khong co log nao ghi; pf_diag chi co maha2_best cua PF, EKF khong co"},
        {"truong": "GT box tung frame (frame, id, x, y, w, h, ...)", "kc48_pf_dev": True,
         "nguon": "_processed/DETRAC-all/<video>/gt/gt.txt"},
        {"truong": "detection tho (x1,y1,x2,y2,conf,cls) tung frame", "kc48_pf_dev": True,
         "nguon": rel(CACHE) + "/<video>.npz"},
    ]


def main() -> int:
    t0 = time.time()
    videos = json.loads((DEV / "videos.json").read_text(encoding="utf-8"))
    with open(DEV / "runs.csv", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    kc48 = {r["dir"]: kiem_ke_run_tracker(TRK, r["dir"], videos, co_assoc_pf=False) for r in runs}
    tev = {r["dir"]: kiem_ke_trackeval(r["dir"], videos) for r in runs}
    moc2 = {r: kiem_ke_run_tracker(MOC2, r, videos, co_assoc_pf=True) for r in MOC2_RUN}
    moc2_tev = {r: {"thu_muc": rel(TEV / r), "ton_tai": (TEV / r).exists(),
                    "file": sorted(x.name for x in (TEV / r).iterdir()) if (TEV / r).exists() else []}
                for r in MOC2_RUN}
    phan_tich = {}
    for sub in ("phan_tich", "su_kien"):
        for p in sorted((DEV / sub).glob("*")):
            if p.is_file():
                phan_tich[rel(p)] = (mo_ta_csv(p, co_header=True) if p.suffix == ".csv"
                                     else {"ton_tai": True, "byte": p.stat().st_size})
    out = {
        "buoc": "R50A / P487415eb -- chi kiem ke dau vao, khong co ket qua phan tich",
        "thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lenh": "C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_ke_dau_vao_deta.py",
        "pham_vi": "chi liet ke file / kich thuoc / cot / so dong / co-khong co truong; khong doc gia tri de "
                   "thong ke, khong ghep GT, khong so hash giua run, khong chay tracking",
        "video_dev12": videos,
        "runs_csv": {"duong_dan": rel(DEV / "runs.csv"), "n_run": len(runs), "cot": list(runs[0]) if runs else [],
                     "dong": runs},
        "file_cau_hinh": {rel(p): p.stat().st_size for p in [DEV / "experiment.yaml",
                                                            ROOT / "configs/phase2/experiments/kc48_pf_dev.yaml"]
                          if p.exists()},
        "yaml_tracker": {y: (ROOT / y).exists() for y in sorted({r["yaml"] for r in runs})},
        "kc48_pf_dev_tracker": kc48,
        "kc48_pf_dev_trackeval": tev,
        "moc2_c25_tracker": moc2,
        "moc2_c25_trackeval": moc2_tev,
        "gt_dev12": kiem_ke_gt(videos),
        "det_cache": kiem_ke_cache(videos),
        "phan_tich_da_co": phan_tich,
        "truong_can": truong_can(kc48, moc2, tev),
        "giay": None,
    }
    out["giay"] = round(time.time() - t0, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tc = out["truong_can"]
    print(f"[kiem_ke] {len(runs)} run kc48, {len(MOC2_RUN)} run moc2 c25, {len(videos)} video -> {rel(OUT)} "
          f"({out['giay']} s)")
    for x in tc:
        print(f"  kc48={x.get('kc48_pf_dev')!s:<5} moc2={x.get('moc2_c25', '-')!s:<5} {x['truong']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
