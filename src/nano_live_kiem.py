"""
nano_live_kiem.py -- Kiem chung luong truc tiep TREN NANO (chay bang Python env >= 3.8, tu goc goi):

  parity  A. GIA LAP tren 300 JPG THAT + cache smoke -> output phai trung `pc-ref` smoke (kiem kenh 3.6 <-> 3.8,
             loc, tracker, doc lap TensorRT).
          B. TensorRT THAT: live CV/EKF --save-cache -> replay `run_tracking.py` bang CHINH `replay_argv` (cung cau hinh
             hieu luc) -> `nano_parity` (bbox 1e-3, conf 1e-6, ID toan sequence; KHONG noi tolerance) + so byte.
          C. MO TA (khong phai tieu chi): detection TRT (che do doi chieu --regions) vs cache PC PyTorch
             (`nano_det_compare`); detection TRT cua lan CV vs lan EKF (tinh xac dinh cua engine).
          D. Bien: anh den (frame rong), mp4 (het nguon, so frame == khai bao), stop-file, SIGINT, kill tien trinh con,
             engine sai duong dan.
  bench   Do thoi gian TOAN LUONG theo dieu kien CONG BO TRUOC (file --dieu-kien phai ton tai): thu tu cau hinh
          `--thu-tu` (mac dinh cv ekf ekf cv cv ekf), nghi giua luot, moi luot mot thu muc; tong hop CSV/JSON.

    BYTETRACK_BACKEND=np ~/miniforge3/envs/nano/bin/python src/nano_live_kiem.py parity --engine yolov8n_fp16.engine \\
        --out results_nano/live_v6/parity
    BYTETRACK_BACKEND=np ~/miniforge3/envs/nano/bin/python src/nano_live_kiem.py bench --engine yolov8n_fp16.engine \\
        --out results_nano/live_v6/bench --dieu-kien reference/dieu_kien_do_toan_luong.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = config.PROJECT_ROOT
PY = sys.executable
SMOKE = "data/smoke/MVI_39031_f0001-0300"
SMOKE_MP4 = "data/smoke/MVI_39031_f0001-0300.mp4"
REGIONS = "data/smoke/MVI_39031/ignored_regions.txt"
CACHE_SMOKE = "nano_smoke_MVI_39031_f0001-0300"
ONNX = "models/export/yolov8n_640_op12.onnx"
ENV = {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}
KQ: list[dict] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def ghi(name: str, dat: bool | None, chi_tiet) -> None:
    KQ.append({"phep": name, "dat": dat, "chi_tiet": chi_tiet})
    nhan = "MO TA" if dat is None else ("DAT" if dat else "KHONG DAT")
    log(f"  [{nhan}] {name}\n         {json.dumps(chi_tiet, ensure_ascii=False, default=str)[:600]}")


def onnx_sha() -> str | None:
    p = ROOT / ONNX
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def live(out: Path, *args, nen: bool = False):
    cmd = [PY, "src/nano_live.py", "--out", str(out), "--progress-every", "100", *args]
    log(f"$ {' '.join(cmd)}")
    if nen:
        return subprocess.Popen(cmd, cwd=ROOT, env=ENV)
    return subprocess.run(cmd, cwd=ROOT, env=ENV).returncode


def meta(out: Path) -> dict:
    p = out / "run_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def parity(ref: Path, test: Path, js: Path) -> dict:
    r = subprocess.run([PY, "src/nano_parity.py", "--ref", str(ref), "--test", str(test), "--json", str(js)],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    a = next((ref / "data").glob("*.txt"), None)
    b = test / "data" / (a.name if a else "x")
    byte = (a is not None and b.exists()
            and a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n"))
    return {"exit": r.returncode, "dong_cuoi": (r.stdout.strip().splitlines() or [""])[-1], "byte_giong_het_LF": byte,
            "sha256_LF_test": hashlib.sha256(b.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16] if b.exists() else None}


def replay(out: Path, split: str, ten: str) -> int:
    argv = [PY if x == "python" else split if x == "<split>" else ten if x == "<ten>" else x
            for x in meta(out)["replay_argv"]]
    log(f"$ {' '.join(argv)}")
    return subprocess.run(argv, cwd=ROOT, env=ENV).returncode


def cho_pid(out: Path, timeout=120.0) -> int | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = out / "detector.pid"
        if p.exists() and p.read_text().strip():
            return int(p.read_text().strip())
        time.sleep(0.1)
    return None


def cho_hello(out: Path, timeout=180.0) -> bool:
    """Doi den khi con da gui HELLO (dong 'HELLO' trong detector_stderr.log)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = out / "detector_stderr.log"
        if p.exists() and b"HELLO" in p.read_bytes():
            return True
        time.sleep(0.2)
    return False


def ghi_bmp(path: Path, img: np.ndarray) -> None:
    import struct
    h, w = img.shape[:2]
    pad = (4 - (w * 3) % 4) % 4
    rows = b"".join(np.ascontiguousarray(img[y]).tobytes() + b"\0" * pad for y in range(h - 1, -1, -1))
    path.write_bytes(struct.pack("<2sIHHI", b"BM", 54 + len(rows), 0, 0, 54)
                     + struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(rows), 2835, 2835, 0, 0) + rows)


# --------------------------------------------------------------------------- parity + bien
def cmd_parity(a) -> int:
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        log(f"[LOI] {out} da co noi dung -> dung thu muc MOI"); return 8
    out.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d%H%M%S")
    split = f"NANO-live-{tag}"
    sha = onnx_sha()
    eng = ["--engine", a.engine] + (["--onnx-sha256", sha] if sha else [])
    base = ["--source", SMOKE, "--fps", "25", "--video-name", "MVI_39031"]

    log("== A. gia lap tren 300 JPG that + cache smoke -> phai trung pc-ref smoke")
    for t in ("cv", "ekf"):
        o = out / f"A_gialap_{t}"
        rc = live(o, *base, "--gia-lap-cache", f"data/interim/det_cache/{CACHE_SMOKE}", "--gia-lap-video", "MVI_39031",
                  "--tracker", t)
        p = parity(ROOT / f"data/processed/trackers/NANO-smoke/pc-ref-{t}", o, out / f"A_parity_{t}.json")
        ghi(f"A. gia lap {t.upper()} (300 JPG that, cache smoke) == pc-ref smoke",
            rc == 0 and meta(o).get("trang_thai") == "hoan_tat" and p["exit"] == 0 and p["byte_giong_het_LF"],
            {"exit": rc, "trang_thai": meta(o).get("trang_thai"), "parity": p})

    log("== B. TensorRT that: live -> save-cache -> replay run_tracking -> nano_parity")
    cids = {}
    for t in ("cv", "ekf"):
        o = out / f"B_trt_{t}"
        cids[t] = f"live_trt_{t}_{tag}"
        rc = live(o, *base, *eng, "--tracker", t, "--save-cache", cids[t])
        m = meta(o)
        rrc = replay(o, split, f"replay-{t}") if rc == 0 else None
        p = parity(ROOT / "data/processed/trackers" / split / f"replay-{t}", o, out / f"B_parity_{t}.json") \
            if rrc == 0 else {"exit": None}
        ghi(f"B. TensorRT {t.upper()}: live == replay tu cache cung cau hinh (tolerance cong bo, khong noi)",
            rc == 0 and m.get("trang_thai") == "hoan_tat" and m.get("dem", {}).get("n_frame_da_xu_ly") == 300 and rrc == 0
            and p["exit"] == 0 and p.get("byte_giong_het_LF"),
            {"exit_live": rc, "trang_thai": m.get("trang_thai"), "dem": m.get("dem"), "exit_replay": rrc, "parity": p,
             "fps_xu_ly": m.get("thoi_gian", {}).get("fps_xu_ly"), "engine_sha256": m.get("hello_con", {}).get("engine_sha256")})

    log("== C. MO TA: detection TRT (che do doi chieu --regions) vs cache PC; TRT lan CV vs lan EKF")
    o = out / "C_trt_regions_cv"
    cid_r = f"live_trt_regions_{tag}"
    rc = live(o, *base, *eng, "--tracker", "cv", "--regions", REGIONS, "--save-cache", cid_r)
    for ten, extra in (("c25", ["--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7", "--max-det", "300"]), ("tho", [])):
        js = out / f"C_det_compare_{ten}.json"
        r = subprocess.run([PY, "src/nano_det_compare.py", "--a", CACHE_SMOKE, "--b", cid_r, "--video", "MVI_39031",
                            *extra, "--json", str(js)], cwd=ROOT, env=ENV, capture_output=True, text=True)
        ghi(f"C. detection TRT(regions) vs cache PC PyTorch ({ten}) -- khac detector, KHONG la tieu chi",
            None, {"exit_live": rc, "exit": r.returncode, "tom_tat": r.stdout.strip().splitlines()[-4:]})
    npz = [ROOT / "data/interim/det_cache" / cids[t] / "MVI_39031.npz" for t in ("cv", "ekf")]
    if all(p.exists() for p in npz):
        za, zb = (np.load(p) for p in npz)
        giong = all(np.array_equal(za[k], zb[k]) for k in ("frame_idx", "n_per_frame", "data"))
        ghi("C. detection TRT lan CV vs lan EKF (cung engine, cung anh)", None,
            {"giong_het_bit": giong, "n_box": [int(len(za["data"])), int(len(zb["data"]))],
             "max_abs_lech": (float(np.max(np.abs(za["data"] - zb["data"]))) if za["data"].shape == zb["data"].shape
                              else "khac so box")})
    else:
        ghi("C. detection TRT lan CV vs lan EKF", False, {"loi": "thieu cache cua buoc B", "npz": [str(p) for p in npz]})

    log("== D. bien")
    den = out / "D_anh_den_src"
    den.mkdir()
    for i in range(1, 31):
        ghi_bmp(den / f"img{i:05d}.bmp", np.zeros((540, 960, 3), np.uint8))
    o = out / "D1_anh_den"
    rc = live(o, "--source", str(den), "--fps", "25", "--video-name", "den", *eng, "--tracker", "cv")
    m = meta(o)
    import pandas as pd
    n_t = len(pd.read_csv(o / "timing" / "den.csv")) if (o / "timing" / "den.csv").exists() else -1
    ghi("D1. 30 anh den (frame rong): exit 0, 0 hang MOT, timing du 30 frame (vong doi van cap nhat)",
        rc == 0 and m.get("dem", {}).get("n_hang_mot") == 0 and n_t == 30, {"exit": rc, "dem": m.get("dem"), "n_timing": n_t})

    o = out / "D2_mp4"
    rc = live(o, "--source", SMOKE_MP4, "--video-name", "MVI_39031", *eng, "--tracker", "cv")
    m = meta(o)
    ghi("D2. mp4 smoke (cv2 3.2 FFMPEG): exit 0, het nguon, so frame == khai bao 300",
        rc == 0 and m.get("dem", {}).get("n_frame_da_xu_ly") == m.get("dem", {}).get("n_frames_khai_bao") == 300,
        {"exit": rc, "trang_thai": m.get("trang_thai"), "dem": m.get("dem"), "eos": m.get("eos_con"),
         "fps_xu_ly": m.get("thoi_gian", {}).get("fps_xu_ly")})

    o = out / "D3_stop_file"
    p = live(o, *base, *eng, "--tracker", "cv", "--save-cache", f"live_trt_dung_{tag}", nen=True)
    if not cho_hello(o):
        log("[CANH BAO] khong thay HELLO trong 180 s")
    time.sleep(6)
    (o / "STOP").write_text("dung", encoding="utf-8")
    rc = p.wait(timeout=300)
    m = meta(o)
    n = m.get("dem", {}).get("n_frame_da_xu_ly", -1)
    rrc = replay(o, split, "replay-dung") if rc == 4 else None
    pp = parity(ROOT / "data/processed/trackers" / split / "replay-dung", o, out / "D3_parity.json") if rrc == 0 else {"exit": None}
    ghi("D3. stop-file giua luong: exit 4, da_dung, 0 < frame < 300, con thoat 0 (EOS dung), replay phan da xu ly khop",
        rc == 4 and m.get("trang_thai") == "da_dung" and 0 < n < 300 and m.get("con", {}).get("returncode") == 0
        and (m.get("eos_con") or {}).get("ly_do") == "dung_theo_yeu_cau" and pp["exit"] == 0,
        {"exit": rc, "n_xu_ly": n, "bay_khi_dung": m.get("dem", {}).get("n_frame_bay_khi_dung_khong_xu_ly"),
         "con": m.get("con"), "eos": m.get("eos_con"), "parity": pp})

    o = out / "D4_sigint"
    p = live(o, *base, *eng, "--tracker", "cv", nen=True)
    cho_hello(o)
    time.sleep(5)
    os.kill(p.pid, signal.SIGINT)
    rc = p.wait(timeout=300)
    m = meta(o)
    ghi("D4. SIGINT (Ctrl+C) vao tien trinh cha: exit 4, da_dung, con thoat 0",
        rc == 4 and m.get("trang_thai") == "da_dung" and m.get("con", {}).get("returncode") == 0,
        {"exit": rc, "n_xu_ly": m.get("dem", {}).get("n_frame_da_xu_ly"), "con": m.get("con")})

    o = out / "D5_kill_con"
    p = live(o, *base, *eng, "--tracker", "cv", nen=True)
    cho_hello(o)
    time.sleep(5)
    pid = cho_pid(o)
    if pid:
        os.kill(pid, signal.SIGKILL)
    rc = p.wait(timeout=300)
    m = meta(o)
    ghi("D5. kill -9 tien trinh con (PID tu detector.pid): exit 3, loi_tien_trinh_con, KHONG hoan_tat",
        rc == 3 and m.get("trang_thai") == "loi_tien_trinh_con" and not m.get("hoan_tat"),
        {"exit": rc, "pid_con": pid, "trang_thai": m.get("trang_thai"), "dem": m.get("dem"), "con": m.get("con")})

    o = out / "D6_engine_sai"
    rc = live(o, *base, "--engine", "khong_co.engine", "--tracker", "cv")
    ghi("D6. engine sai duong dan: exit 2, khong co ket qua", rc == 2 and not (o / "data").exists(), {"exit": rc})

    tong = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S"), "split_replay": split, "cache_ids": cids,
            "ket_qua": KQ, "dat_tat_ca": all(k["dat"] for k in KQ if k["dat"] is not None)}
    (out / "tong_hop_parity.json").write_text(json.dumps(tong, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    n_dat = sum(1 for k in KQ if k["dat"]); n_tc = sum(1 for k in KQ if k["dat"] is not None)
    log(f"\nKET QUA: {n_dat}/{n_tc} phep kiem DAT ({sum(1 for k in KQ if k['dat'] is None)} phep mo ta) -> {out}")
    return 0 if tong["dat_tat_ca"] else 1


# --------------------------------------------------------------------------- bench
def cmd_bench(a) -> int:
    dk = Path(a.dieu_kien)
    if not dk.is_file():
        log(f"[LOI] chua cong bo dieu kien do: {dk} khong ton tai"); return 8
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        log(f"[LOI] {out} da co noi dung -> dung thu muc MOI"); return 8
    out.mkdir(parents=True, exist_ok=True)
    (out / "dieu_kien_da_cong_bo.sha256").write_text(
        f"{hashlib.sha256(dk.read_bytes()).hexdigest()}  {dk}\n", encoding="utf-8")
    sha = onnx_sha()
    eng = ["--engine", a.engine] + (["--onnx-sha256", sha] if sha else [])
    rows = []
    dem_cfg: dict[str, int] = {}
    for i, t in enumerate(a.thu_tu, 1):
        if i > 1 and a.nghi_s > 0:
            log(f"(nghi {a.nghi_s} s)"); time.sleep(a.nghi_s)
        dem_cfg[t] = dem_cfg.get(t, 0) + 1
        o = out / f"luot{i:02d}_{t}_r{dem_cfg[t]}"
        rc = live(o, "--source", a.source, "--fps", "25", "--video-name", "MVI_39031", *eng, "--tracker", t,
                  "--warmup-frames", str(a.warmup), "--hang-doi", str(a.hang_doi), "--tegrastats-ms", "500")
        m = meta(o)
        th = m.get("thoi_gian", {})
        tt = th.get("tom_tat_sau_warmup", {})

        def tv(k):
            return (tt.get(k) or {}).get("trung_vi")
        teg = m.get("thiet_bi", {}).get("tegrastats", {})
        rows.append({"luot": i, "cau_hinh": t, "exit": rc, "trang_thai": m.get("trang_thai"),
                     "n_frame": m.get("dem", {}).get("n_frame_da_xu_ly"), "n_tong_ket": tt.get("n_frame_tong_ket"),
                     "khoi_tao_s": th.get("khoi_tao_s"), "chuan_bi_cha_s": th.get("chuan_bi_cha_s"),
                     "xu_ly_s": th.get("xu_ly_s"), "ket_thuc_s": th.get("ket_thuc_s"),
                     "fps_xu_ly": th.get("fps_xu_ly"), "fps_on_dinh": tt.get("fps_on_dinh"),
                     "moi_frame_trong_cua_so": (th.get("kiem_cua_so_do") or {}).get("moi_frame_trong_cua_so"),
                     **{f"{k}_tv": tv(k) for k in ("decode_ms", "pre_ms", "infer_ms", "h2d_ms", "exec_ms", "d2h_ms",
                                                    "post_ms", "cho_credit_ms", "truyen_ms", "cho_doc_ms", "loc_ms",
                                                    "t_update_ms", "xu_ly_ms", "tre_ms", "khoang_giua_frame_ms")},
                     "tre_ms_p90": (tt.get("tre_ms") or {}).get("p90"), "ghi_s": th.get("ghi_s"),
                     "ram_MB_max": teg.get("ram_dung_MB_max"), "swap_MB_max": teg.get("swap_dung_MB_max"),
                     "cpu_MHz": teg.get("cpu_MHz_min_max"), "gr3d_pct_tb_max": teg.get("gr3d_pct_tb_max"),
                     "nhiet_max_C": teg.get("nhiet_max_C"), "xung_truoc": m.get("thiet_bi", {}).get("truoc", {}).get("xung"),
                     "thu_muc": o.name})
        log(f"  luot {i} {t}: exit {rc} {m.get('trang_thai')} fps_xu_ly {th.get('fps_xu_ly')} "
            f"trong cua so {(th.get('kiem_cua_so_do') or {}).get('moi_frame_trong_cua_so')}")
    with (out / "bang_luot.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    def tv3(rr, k):
        v = [r[k] for r in rr if r[k] is not None]
        return {"trung_vi": statistics.median(v), "min": min(v), "max": max(v)} if v else None

    theo = {}
    for t in sorted(set(a.thu_tu)):
        rr = [r for r in rows if r["cau_hinh"] == t and r["exit"] == 0]
        theo[t] = {"n_luot_hop_le": len(rr)}
        for k in ("fps_xu_ly", "fps_on_dinh", "tre_ms_tv", "infer_ms_tv", "t_update_ms_tv", "decode_ms_tv",
                  "pre_ms_tv", "post_ms_tv", "khoi_tao_s", "chuan_bi_cha_s"):
            theo[t][k] = tv3(rr, k)
    tong = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S"), "dieu_kien": str(dk), "thu_tu": a.thu_tu, "nghi_s": a.nghi_s,
            "warmup": a.warmup, "hang_doi": a.hang_doi, "source": a.source, "luot": rows, "theo_cau_hinh": theo,
            "ghi_chu": "khong loai luot nao; FPS toan luong do truc tiep (khong cong so vong 16/20); khong render; fps_xu_ly theo cua so BAT DAU (giao thuc 2, review 9f210e2 N21-F1)"}
    (out / "tong_hop_bench.json").write_text(json.dumps(tong, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(json.dumps(theo, ensure_ascii=False, indent=1, default=str))
    return 0 if all(r["exit"] == 0 for r in rows) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="lenh", required=True)
    p1 = sub.add_parser("parity")
    p1.add_argument("--engine", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("bench")
    p2.add_argument("--engine", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--dieu-kien", required=True, help="file dieu kien do da CONG BO TRUOC")
    p2.add_argument("--source", default=SMOKE)
    p2.add_argument("--thu-tu", nargs="+", default=["cv", "ekf", "ekf", "cv", "cv", "ekf"])
    p2.add_argument("--nghi-s", type=float, default=20.0)
    p2.add_argument("--warmup", type=int, default=25)
    p2.add_argument("--hang-doi", type=int, default=4)
    a = ap.parse_args()
    return cmd_parity(a) if a.lenh == "parity" else cmd_bench(a)


if __name__ == "__main__":
    raise SystemExit(main())
