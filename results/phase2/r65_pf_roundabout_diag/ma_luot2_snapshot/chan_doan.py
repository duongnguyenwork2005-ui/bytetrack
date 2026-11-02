"""
chan_doan.py -- (R65) chan doan PF500 thap tren RoundaboutHD prefix3000. KHONG tune, KHONG thay benchmark R64.

Quyet dinh + pham vi khoa truoc: GPT 02/11 `D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/
answer.json` Q2. imagesc001-004 frame 1-3000; CHI yolov8n_mixed5, c25; CV/EKF/UKF + PF500 seed 0/1/2 (24 don vi
camera x bo loc/seed); cache `r64_roundabout_yolov8n_mixed5_c0.05`, GT SCT da sua, cau hinh R64 (lop 1 2 3 5 7,
max 300, 15 fps, buffer 1,2 s = 18, elapsed steps).

Hai nhanh, cung bo quan sat chi doc (`quan_sat.py`): `off` (khong log) va `on` (--assoc-log --pf-diag). Cong:
MOT + data_cls trung byte off/on va trung R64 goc; metric trung; chuoi hash trang thai off == on.

    python -B src/r65_pf_roundabout/chan_doan.py giao-thuc
    python -B src/r65_pf_roundabout/chan_doan.py chay <off|on> --het-han 2026-11-02T09:30:00+07:00
    python -B src/r65_pf_roundabout/chan_doan.py cham <off|on>
    python -B src/r65_pf_roundabout/chan_doan.py parity
Ma thoat: 0 xong; 10 het ngan sach (PARTIAL, chay lai se tiep tuc tu don vi chua xong); 2 loi hop dong/parity sai;
3 loi subprocess.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(GOC / "src" / "r64_ngoai"))
import config  # noqa: E402
import danh_gia_ngoai as r64  # noqa: E402

DS, DET, COND = "roundabout", "yolov8n_mixed5", "c25"
T = r64.cau_hinh(DS, False)
SPLIT, VIDEOS, FPS = T["split"], T["videos"], T["fps"]
CACHE = r64.ten_cache(DS, DET, False)
LOC = ["cv", "ekf", "ukf", "pf500"]
NHANH = {"off": [], "on": ["--assoc-log", "--pf-diag"]}
TIEN_TO = "r65b"
LOG = GOC / "handoff" / "logs" / "review65_pf_roundabout"
KQ = config.RESULTS_DIR / "phase2" / "r65_pf_roundabout_diag"
TRACE = KQ / "quan_sat"
CAU_TRA_LOI = "D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json"
MA = ["src/r65_pf_roundabout/chan_doan.py", "src/r65_pf_roundabout/quan_sat.py", "src/run_tracking.py",
      "src/tracker_ctrv.py", "src/pf_ctrv.py", "src/ekf_ctrv.py", "src/ukf_ctrv.py", "src/assoc_log.py",
      "src/time_units.py", "src/det_cache.py", "src/ul_backend.py", "src/run_state.py", "src/run_trackeval.py",
      "src/r64_ngoai/danh_gia_ngoai.py", "third_party/ultralytics_trackers/byte_tracker.py",
      "third_party/ultralytics_trackers/basetrack.py"]
#: uoc luong (giay) moi bo loc ca 4 camera, nhanh co log + quan sat (R64: update CV 30 / EKF 38 / UKF 72 / PF 126 s/seed)
UOC_GIAY = {"cv": 120, "ekf": 130, "ukf": 200, "pf500": 3 * 400}
DU_PHONG_GIAY = 1800
DU_PHONG_DIA = 5 * 1024 ** 3
TRAN_ARTIFACT = 1024 ** 3


def ten(nhanh: str, loc: str, seed=None) -> str:
    # luot 2 (sau review GPT 02/11: them ranh gioi sau du doan) = r65b*; luot 1 r65off/r65on giu lam lich su
    return f"{TIEN_TO}{nhanh}-{DET}-{COND}-{loc}" + (f"-s{seed}" if seed is not None else "")


def ten_r64(loc: str, seed=None) -> str:
    return f"r64-{DET}-{COND}-{loc}" + (f"-s{seed}" if seed is not None else "")


def don_vi(loc: str) -> list:
    return r64.BO_LOC[loc][1] or [None]


def dung_luong(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def tien_kiem(het_han: str | None, viec: str, giay_uoc: float, byte_uoc: int) -> None:
    trong = shutil.disk_usage(config.PROCESSED_DIR).free
    con = None if het_han is None else dt.datetime.fromisoformat(het_han).timestamp() - time.time()
    da_ghi = sum(dung_luong(config.PROCESSED_DIR / "trackers" / SPLIT / ten(n, l, s))
                 for n in NHANH for l in LOC for s in don_vi(l)) + dung_luong(KQ)
    loi = []
    if het_han is None:
        loi.append("thieu --het-han")
    elif giay_uoc + DU_PHONG_GIAY > con:
        loi.append(f"uoc {giay_uoc:.0f} s + du phong {DU_PHONG_GIAY} s > con {con:.0f} s")
    if trong < byte_uoc + DU_PHONG_DIA:
        loi.append(f"o con {trong / 1e9:.1f} GB < uoc + 5 GiB")
    if da_ghi + byte_uoc > TRAN_ARTIFACT:
        loi.append(f"artifact R65 {da_ghi / 1e6:.0f} MB + uoc > tran 1 GiB")
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / "ngan_sach.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"thoi_diem": r64.bay_gio(), "viec": viec, "het_han": het_han,
                            "giay_con": None if con is None else round(con), "giay_uoc": round(giay_uoc),
                            "giay_du_phong": DU_PHONG_GIAY, "byte_trong": trong, "byte_uoc": byte_uoc,
                            "byte_artifact_r65": da_ghi, "dat": not loi, "loi": loi}, ensure_ascii=False) + "\n")
    if loi:
        raise r64.HetNganSach(f"{viec}: " + "; ".join(loi))


# ------------------------------------------------------------------------------------------------ giao thuc
def buoc_giao_thuc() -> int:
    import run_fingerprint
    import run_tracking as rt
    cong = {v: r64.kiem_don_vi_cache(DS, DET, v, False) for v in VIDEOS}
    gt = r64.kiem_gt_cham(T)
    cache_dir = Path(r64.det_cache.CACHE_ROOT) / CACHE
    yaml = {l: str(Path(rt.resolve_tracker_yaml(r64.BO_LOC[l][0])).resolve()) for l in LOC}   # CV: yaml cua ultralytics
    manifest = {
        "thoi_diem": r64.bay_gio(),
        "cache": {p.name: r64.sha256_file(p) for p in sorted(cache_dir.iterdir()) if p.is_file()},
        "gt": {v: r64.sha256_file(config.PROCESSED_DIR / SPLIT / v / "gt" / "gt.txt") for v in VIDEOS},
        "gt_log": r64.sha256_file(config.PROCESSED_DIR / SPLIT / "gt_log.json"),
        "yaml": {l: {"duong_dan": y, "sha256": r64.sha256_file(GOC / y)} for l, y in yaml.items()},
        "ma": {m: r64.sha256_file(GOC / m) for m in MA},
        "thu_vien": run_fingerprint.code_fingerprint()["backend"],
        "python": sys.version,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=GOC, capture_output=True,
                                   text=True).stdout.strip(),
    }
    giao_thuc = {
        "nguon_quyet_dinh": CAU_TRA_LOI + " (Q2)",
        "pham_vi": {"split": SPLIT, "videos": VIDEOS, "frame": [1, T["n_frame"]], "detector": DET, "dieu_kien": COND,
                    "conf_min": r64.DIEU_KIEN[COND], "bo_loc": LOC, "seed_pf": r64.BO_LOC["pf500"][1],
                    "so_don_vi_camera_x_boloc_seed": len(VIDEOS) * sum(len(don_vi(l)) for l in LOC),
                    "khong_mo_them": "detector khac, c10, frame khac, ablation nhieu/so hat/FPS/do phan giai"},
        "cau_hinh": {"cache": CACHE, "lop": r64.LOP_TRACKER, "max_det": 300, "fps": FPS, "track_buffer_sec": 1.2,
                     "track_buffer_buoc": int(round(1.2 * FPS)), "elapsed": "steps", "yaml": yaml,
                     "set_seed_video": "ten video truyen nguyen (video_salt phu thuoc ten)"},
        "nhanh": {"off": "khong log", "on": "--assoc-log --pf-diag"},
        "quan_sat": "src/r65_pf_roundabout/quan_sat.py o CA HAI nhanh (chi doc, khong rut so ngau nhien)",
        "cong": ["MOT data/<v>.txt + data_cls/<v>.csv trung byte off == on == R64 goc",
                 "per_video/combined metric trung off == on == R64",
                 "chuoi hash trang thai (frame + PF) va pf_update off == on",
                 "khong NaN / thieu frame (hau kiem R64)",
                 "Lost / refind / resample / reset duoc phu (du lieu that hoac fixture test_quan_sat.py)"],
        "dung_ngay_neu": "parity sai, provenance lech, NaN, thieu frame, log tac dong trang thai; loi thuat toan -> "
                         "chi lap finding, khong sua roi chay thay benchmark",
        "ngan_sach": "luot dau <= 2 h, 30 phut cuoi hau kiem/bao cao; artifact moi <= 1 GiB; trong >= 5 GiB",
        "dau_ra": {"log": str(LOG.relative_to(GOC)), "ket_qua": str(KQ.relative_to(GOC))},
        "cong_cache_truoc": cong, "gt_cham_sha256": gt,
    }
    LOG.mkdir(parents=True, exist_ok=True)
    for p, obj in ((LOG / "00_giao_thuc.json", giao_thuc), (LOG / "manifest_sha256.json", manifest)):
        if p.exists():
            cu = json.loads(p.read_text(encoding="utf-8"))
            so = {k: v for k, v in cu.items() if k != "thoi_diem"} == {k: v for k, v in obj.items() if k != "thoi_diem"}
            if not so and p.name == "manifest_sha256.json":
                print(f"[DUNG] {p.name} da ton tai va khac (provenance lech)"); return 2
            if so:
                continue
        r64.ghi_json_nguyen_tu(p, obj)
    print("giao thuc + manifest OK:", len(manifest["cache"]), "file cache,", len(manifest["ma"]), "file ma")
    return 0


def kiem_manifest() -> None:
    m = json.loads((LOG / "manifest_sha256.json").read_text(encoding="utf-8"))
    cache_dir = Path(r64.det_cache.CACHE_ROOT) / CACHE
    lech = [k for k, h in m["cache"].items() if r64.sha256_file(cache_dir / k) != h]
    lech += [v for v, h in m["gt"].items() if r64.sha256_file(config.PROCESSED_DIR / SPLIT / v / "gt" / "gt.txt") != h]
    lech += [x["duong_dan"] for x in m["yaml"].values() if r64.sha256_file(GOC / x["duong_dan"]) != x["sha256"]]
    lech += [k for k, h in m["ma"].items() if r64.sha256_file(GOC / k) != h]
    if lech:
        raise r64.LoiHopDong(f"provenance lech so voi manifest: {lech}")


# ------------------------------------------------------------------------------------------------ chay
def xong(nhanh: str, loc: str) -> bool:
    """Moi seed: run_meta ket thuc + du video_meta + du trace cho 4 camera."""
    import run_state
    for s in don_vi(loc):
        d = config.PROCESSED_DIR / "trackers" / SPLIT / ten(nhanh, loc, s)
        if not (d / "run_meta.json").exists():
            return False
        if not all(run_state.video_meta_path(d, v).exists() and (TRACE / ten(nhanh, loc, s) / f"{v}.json").exists()
                   for v in VIDEOS):
            return False
    return True


def buoc_chay(nhanh: str, het_han: str | None) -> int:
    kiem_manifest()
    for loc in LOC:
        if xong(nhanh, loc):
            print(f"[bo qua] {nhanh} {loc} da xong"); continue
        yaml, seeds = r64.BO_LOC[loc]
        uoc = UOC_GIAY[loc]
        tien_kiem(het_han, f"chay {nhanh} {loc}", uoc, 150 * 1024 ** 2 if nhanh == "on" else 20 * 1024 ** 2)
        cmd = [sys.executable, "-B", str(GOC / "src" / "r65_pf_roundabout" / "quan_sat.py"), "--trace-dir", str(TRACE),
               "--", "--cache-id", CACHE, "--split-name", SPLIT, "--tracker-yaml", yaml,
               "--tracker-name", ten(nhanh, loc), "--videos", *VIDEOS, "--conf-min", str(r64.DIEU_KIEN[COND]),
               "--classes", *map(str, r64.LOP_TRACKER), "--max-det", "300", "--fps", str(FPS),
               "--track-buffer-sec", "1.2", "--elapsed-mode", "steps", "--skip-existing", *NHANH[nhanh]]
        if seeds:
            cmd += ["--seeds", *map(str, seeds)]
        (LOG / "track_logs").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[{r64.bay_gio()}] chay {nhanh} {loc}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run(cmd, cwd=GOC, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        (LOG / "track_logs" / f"{ten(nhanh, loc)}_{time.strftime('%H%M%S')}.log").write_text(
            f"argv: {cmd}\nexit: {p.returncode}\ngiay: {time.time() - t0:.1f}\n--- stdout\n{p.stdout}\n--- stderr\n"
            f"{p.stderr}", encoding="utf-8")
        print(p.stdout[-1500:], flush=True)
        if p.returncode != 0:
            print(p.stderr[-3000:]); return 3
        buf = int(round(1.2 * FPS))
        mtl = r64.max_time_lost_thuc(yaml, buf)
        for s in don_vi(loc):
            r64.hau_kiem_tracking(T, ten(nhanh, loc, s), buf, mtl)
        if not xong(nhanh, loc):
            raise r64.LoiHopDong(f"{nhanh} {loc}: chay xong nhung thieu video_meta/trace")
        print(f"  xong {nhanh} {loc} {time.time() - t0:.0f} s", flush=True)
    return 0


def buoc_cham(nhanh: str) -> int:
    for loc in LOC:
        for s in don_vi(loc):
            nm = ten(nhanh, loc, s)
            out = config.RESULTS_DIR / "trackeval" / SPLIT / nm
            if (out / "combined_metrics.csv").exists():
                print("[bo qua] da cham", nm); continue
            if out.exists():
                raise r64.LoiHopDong(f"thu muc cham do dang: {out}")
            kq = r64.cham(SPLIT, nm, VIDEOS)
            print(nm, kq, flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ parity
def _sha(p: Path) -> str | None:
    return r64.sha256_file(p) if p.exists() else None


def buoc_parity() -> int:
    import pandas as pd
    kiem_manifest()
    dong, sai = [], []
    phu = {"pf_degenerate": 0, "pf_resample": 0, "pf_reseed_motion": 0, "update_tu_lost": 0, "lost_event": 0,
           "refind_event": 0}
    for loc in LOC:
        for s in don_vi(loc):
            d = {n: config.PROCESSED_DIR / "trackers" / SPLIT / ten(n, loc, s) for n in NHANH}
            d["r64"] = config.PROCESSED_DIR / "trackers" / SPLIT / ten_r64(loc, s)
            for v in VIDEOS:
                r = {"bo_loc": loc, "seed": s, "video": v}
                for f in (f"data/{v}.txt", f"data_cls/{v}.csv"):
                    h = {n: _sha(d[n] / f) for n in d}
                    r[f.split("/")[0] + "_sha256"] = h
                    if None in h.values() or len(set(h.values())) != 1:
                        sai.append(f"{loc} s{s} {v} {f}: {h}")
                tr = {n: json.loads((TRACE / ten(n, loc, s) / f"{v}.json").read_text(encoding="utf-8")) for n in NHANH}
                for k in ("chuoi_frame_sha256", "chuoi_pf_sha256", "chuoi_du_doan_sha256", "dem", "seed", "video"):
                    if tr["off"][k] != tr["on"][k]:
                        sai.append(f"{loc} s{s} {v} trace {k}: off {tr['off'][k]} != on {tr['on'][k]}")
                r["trace"] = {k: tr["off"][k] for k in ("chuoi_frame_sha256", "chuoi_pf_sha256", "chuoi_du_doan_sha256",
                                                        "dem")}
                for duoi in ("pf_update", "pf_prior"):
                    pu = {n: _sha(TRACE / ten(n, loc, s) / f"{v}_{duoi}.csv") for n in NHANH}
                    if pu["off"] != pu["on"] or (loc == "pf500" and pu["off"] is None):
                        sai.append(f"{loc} s{s} {v} {duoi}: {pu}")
                if loc == "pf500":
                    dem = tr["off"]["dem"]
                    phu["pf_degenerate"] += dem.get("pf_degenerate", 0)
                    phu["pf_resample"] += dem.get("pf_resample", 0)
                    phu["pf_reseed_motion"] += dem.get("pf_reseed_motion", 0)
                    u = pd.read_csv(TRACE / ten("off", loc, s) / f"{v}_pf_update.csv")
                    phu["update_tu_lost"] += int((u["state"] == 2).sum())
                    from assoc_log import load_log
                    ev = load_log(d["on"] / "assoc_log", v)[2]
                    phu["lost_event"] += int((ev["kind"] == "lost").sum())
                    phu["refind_event"] += int((ev["kind"] == "refind").sum())
                dong.append(r)
            # metric
            m = {}
            for n in list(NHANH) + ["r64"]:
                nm = ten(n, loc, s) if n != "r64" else ten_r64(loc, s)
                base = config.RESULTS_DIR / "trackeval" / SPLIT / nm
                m[n] = {f: pd.read_csv(base / f).drop(columns=["tracker"], errors="ignore")
                        for f in ("per_video_metrics.csv", "combined_metrics.csv")}
            for f in ("per_video_metrics.csv", "combined_metrics.csv"):
                for n in ("on", "r64"):
                    a, b = m["off"][f], m[n][f]
                    if list(a.columns) != list(b.columns) or not a.equals(b):
                        sai.append(f"{loc} s{s} metric {f}: off != {n}")
    kq = {"thoi_diem": r64.bay_gio(), "dat": not sai, "sai": sai, "phu_su_kien_pf_tong_3_seed": phu,
          "fixture_bo_sung": "src/r65_pf_roundabout/test_quan_sat.py (reset/suy bien, resample, Lost)",
          "chi_tiet": dong}
    r64.ghi_json_nguyen_tu(LOG / "parity.json", kq)
    print("PARITY", "DAT" if not sai else "SAI", len(dong), "dong; phu:", phu)
    for x in sai[:20]:
        print("  ", x)
    return 0 if not sai else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("buoc", choices=["giao-thuc", "chay", "cham", "parity"])
    ap.add_argument("nhanh", nargs="?", choices=list(NHANH))
    ap.add_argument("--het-han", default=None)
    a = ap.parse_args(argv)
    try:
        if a.buoc == "giao-thuc":
            return buoc_giao_thuc()
        if a.buoc == "chay":
            return buoc_chay(a.nhanh, a.het_han)
        if a.buoc == "cham":
            return buoc_cham(a.nhanh)
        return buoc_parity()
    except r64.HetNganSach as e:
        print("[PARTIAL]", e); return 10
    except r64.LoiHopDong as e:
        print("[DUNG]", e); return 2


if __name__ == "__main__":
    raise SystemExit(main())
