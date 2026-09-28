"""
test_nano_stream.py -- Kiem luong TRUC TIEP hai tien trinh (`nano_live.py` + `nano_detect_stream.py`).

Chay duoc tren PC (con = sys.executable) va tren Nano (`--python-detector /usr/bin/python3`: con Python 3.6 he thong,
cha Python 3.8 env `nano`). Detector GIA LAP (detection lay tu det cache theo frame, van doc anh that) de kiem
giao thuc, vong doi tien trinh, loc + tracker + ghi cache; TensorRT that duoc kiem rieng tren thiet bi.
Fixture: anh BMP tong hop ghi bang numpy (khong can cv2 o tien trinh test), det cache tong hop. Cache/ket qua replay
nam trong CACHE_ROOT / data/processed/trackers voi ten TEST_* va duoc xoa sau khi chay.

  1. Giao thuc: dong goi/doc lai FRAME (co n=0), HELLO/EOS JSON; EOF sach -> None; EOF giua thong diep, magic sai,
     payload sai kich thuoc -> LoiGiaoThuc.
  2. CV: thu muc anh (co frame 0 detection) -> exit 0, hoan_tat, du frame, timing co MOI frame doc duoc (ca frame rong);
     --save-cache -> replay `run_tracking.py` bang CHINH `replay_argv` -> nano_parity DAT va byte giong het.
  3. EKF: nhu 2.
  4. Video .mp4 (tao bang interpreter cua con): so frame == khai bao, time_s == (idx-1)/fps, hoan_tat.
  5. Dung som bang stop-file: exit 4, da_dung, 0 < so frame < N, cache == dung cac frame da xu ly, replay khop,
     con thoat 0 voi EOS 'dung_theo_yeu_cau'. 5b. SIGINT vao tien trinh cha (chi POSIX).
  6. Con loi giua chung (thoat ma 3, khong EOS) -> exit 3, loi_tien_trinh_con, KHONG hoan_tat, phan da xu ly van ghi.
  7. Con bi kill giua chung -> exit 3, loi_tien_trinh_con.
  8. Con bo mot frame -> exit 3, loi_mat_frame.
  9. Anh hong (rac + BMP bi cat -- OpenCV 3.2 tung abort ca tien trinh): mac dinh exit 3 (loi_dau_vao);
    --cho-phep-frame-hong -> exit 0, unreadable ghi nhan, con khong chet, replay khop.
 10. Loi truoc vong lap -> exit 2: nguon khong co, engine khong co, thu muc da co ket qua, con loi khoi tao.
 11. Cua so co gioi han: --hang-doi 2 + tracker cham -> du frame (khong bo), max_in_flight <= 2, con co cho credit.
 12. Cham tran det_max_det: mac dinh exit 3; --allow-truncated -> exit 0 va replay (co --allow-truncated) khop.
 13. --regions (che do doi chieu) -> metadata ghi co + sha256 file vung.
 14. (chi PC, neu co models/yolov8n.pt + anh DETRAC) detector THAT TorchPtInfer 8 frame -> replay khop.

    python src/test_nano_stream.py [--python-detector /usr/bin/python3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import nano_detector  # noqa: E402
import nano_stream_proto as proto  # noqa: E402

ROOT = config.PROJECT_ROOT
PY = sys.executable
TAG = f"TEST_nano_stream_{os.getpid()}"
SPLIT = f"TEST-nano-stream-{os.getpid()}"
ENV = {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}
_R: list[tuple[str, str]] = []
PY_DET: str | None = None
N = 40
FPS = 10.0


def check(name, ok, detail=""):
    _R.append((name, "PASS" if ok else "FAIL"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for ln in str(detail).strip().splitlines()[-6:]:
        print(f"         {ln}")


def skip(name, why):
    _R.append((name, "SKIP"))
    print(f"  [SKIP] {name}: {why}")


# --------------------------------------------------------------------------- fixture
def ghi_bmp(path: Path, img: np.ndarray) -> None:
    """BMP 24-bit BGR, khong can cv2."""
    h, w = img.shape[:2]
    pad = (4 - (w * 3) % 4) % 4
    rows = b"".join(np.ascontiguousarray(img[y]).tobytes() + b"\0" * pad for y in range(h - 1, -1, -1))
    hdr = struct.pack("<2sIHHI", b"BM", 54 + len(rows), 0, 0, 54)
    dib = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    path.write_bytes(hdr + dib + rows)


def tao_anh(d: Path, n: int = N, hong: int | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        img = np.full((96, 160, 3), 40 + (i % 50), np.uint8)
        p = d / f"img{i:05d}.bmp"
        if hong == i:
            p.write_bytes(b"khong phai anh")                  # rac
        elif hong is not None and hong + 1 == i:
            ghi_bmp(p, img)
            p.write_bytes(p.read_bytes()[:60])                 # BMP bi cat (OpenCV 3.2 tung ABORT ca tien trinh)
        else:
            ghi_bmp(p, img)
    return d


def tao_cache_gia_lap(d: Path, n: int = N, video: str = "seq", nhieu_o: int | None = None) -> Path:
    """2 xe chay ngang + 1 xe xuat hien muon; frame 11-13 KHONG co detection (frame rong)."""
    fi, ts, npf, unr, trc, data = [], [], [], [], [], []
    for i in range(1, n + 1):
        rows = []
        if not 11 <= i <= 13:
            rows.append([5 + 2 * i, 20, 35 + 2 * i, 45, 0.9, 2])
            rows.append([120 - 2 * i, 50, 150 - 2 * i, 80, 0.8, 7])
            if i > 20:
                rows.append([60, 10 + i, 80, 30 + i, 0.6, 2])
            rows.append([10, 70, 20, 80, 0.12, 2])            # conf thap -> bi loc o c25
        if nhieu_o == i:
            rows += [[1, 1, 5, 5, 0.3 + 0.01 * k, 2] for k in range(4)]
        rows.sort(key=lambda r: -r[4])
        fi.append(i); ts.append((i - 1) / FPS); npf.append(len(rows)); unr.append(False); trc.append(False)
        data += rows
    meta = {"backend": "gia_lap", "conf": 0.05, "classes": None, "max_det": 3000, "imgsz": 640,
            "weights_sha256": None, "letterbox": None, "iou": 0.7}
    nano_detector.ghi_cache(str(d), video, fi, ts, npf, unr, trc, np.asarray(data, np.float32), (96, 160), meta, {})
    return d


def chay(out: Path, src: Path, cache: Path, *extra, video="seq", nen=False):
    cmd = [PY, "src/nano_live.py", "--source", str(src), "--fps", str(FPS), "--video-name", video,
           "--gia-lap-cache", str(cache), "--gia-lap-video", video, "--out", str(out), "--warmup-frames", "5",
           "--progress-every", "0", "--timeout-con-s", "60", *extra]
    if PY_DET:
        cmd += ["--python-detector", PY_DET]
    if nen:
        return subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")


def meta(out: Path) -> dict:
    p = out / "run_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def replay_va_so(out: Path, ten: str) -> tuple[bool, str]:
    """Replay bang CHINH replay_argv trong run_meta -> nano_parity + so byte."""
    m = meta(out)
    argv = [PY if x == "python" else SPLIT if x == "<split>" else ten if x == "<ten>" else x for x in m["replay_argv"]]
    r = subprocess.run(argv, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return False, f"replay exit {r.returncode}: {(r.stdout + r.stderr)[-300:]}"
    rep = config.PROCESSED_DIR / "trackers" / SPLIT / ten
    video = next((out / "data").glob("*.txt")).stem
    js = out.parent / f"parity_{ten}.json"
    p = subprocess.run([PY, "src/nano_parity.py", "--ref", str(rep), "--test", str(out), "--json", str(js)],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    a = (rep / "data" / f"{video}.txt").read_bytes().replace(b"\r\n", b"\n")
    b = (out / "data" / f"{video}.txt").read_bytes().replace(b"\r\n", b"\n")
    return (p.returncode == 0 and a == b and len(a) > 0,
            f"parity exit {p.returncode}; byte giong het {a == b} ({len(a)} B); {p.stdout.strip().splitlines()[-1:]}")


def cho_hello(out: Path, timeout=120.0) -> bool:
    """Doi con gui HELLO (khoi tao xong) -- tren Nano Python 3.6 + cv2 khoi dong cham hon PC nhieu."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = out / "detector_stderr.log"
        if p.exists() and b"HELLO" in p.read_bytes():
            return True
        time.sleep(0.05)
    return False


def cho_pid(out: Path, timeout=30.0) -> int | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = out / "detector.pid"
        if p.exists() and p.read_text().strip():
            return int(p.read_text().strip())
        time.sleep(0.05)
    return None


# --------------------------------------------------------------------------- cac phep kiem
def t1_giao_thuc():
    import io
    det = np.array([[1, 2, 3, 4, 0.5, 2], [5, 6, 7, 8, 0.25, 7]], np.float32)
    b = (proto.dong_goi_json(proto.HELLO, {"a": 1}) + proto.dong_goi_frame(1, 0.0, det, 0, {"decode_ms": 1.5})
         + proto.dong_goi_frame(2, 0.1, np.zeros((0, 6), np.float32), proto.CO_UNREADABLE)
         + proto.dong_goi_json(proto.EOS, {"n_frame_gui": 2}))
    f = io.BytesIO(b)
    m = [proto.doc_thong_diep(f) for _ in range(5)]
    ok = (m[0]["obj"] == {"a": 1} and np.array_equal(m[1]["det"], det) and m[1]["t"]["decode_ms"] == 1.5
          and m[2]["n_det"] == 0 and m[2]["co"] == proto.CO_UNREADABLE and m[3]["obj"]["n_frame_gui"] == 2 and m[4] is None)
    loi = []
    for ten, du_lieu in (("EOF giua header", b[:10]), ("EOF giua payload", proto.dong_goi_frame(1, 0, det)[:-5]),
                         ("magic sai", b"XXXX" + b[4:proto.HDR.size])):
        try:
            proto.doc_thong_diep(io.BytesIO(du_lieu))
            loi.append(f"{ten}: KHONG bao loi")
        except proto.LoiGiaoThuc:
            pass
    check("1. giao thuc: FRAME (ke ca n=0, co unreadable), HELLO/EOS, EOF sach -> None; EOF giua chung / magic sai -> loi",
          ok and not loi, loi)


def t2_3_e2e(tmp: Path, anh: Path, cache: Path):
    for so, tr in (("2", "cv"), ("3", "ekf")):
        out = tmp / f"e2e_{tr}"
        r = chay(out, anh, cache, "--tracker", tr, "--save-cache", f"{TAG}_{tr}")
        m = meta(out)
        dem = m.get("dem", {})
        import pandas as pd
        n_timing = len(pd.read_csv(out / "timing" / "seq.csv")) if (out / "timing" / "seq.csv").exists() else -1
        n_pipe = len(pd.read_csv(out / "pipeline" / "seq.csv")) if (out / "pipeline" / "seq.csv").exists() else -1
        ok_rep, why = replay_va_so(out, f"rep-{tr}") if r.returncode == 0 else (False, "khong replay")
        check(f"{so}. {tr.upper()}: exit 0, hoan_tat, {N} frame, timing co ca frame rong (11-13), replay tu cache khop",
              r.returncode == 0 and m.get("trang_thai") == "hoan_tat" and dem.get("n_frame_da_xu_ly") == N
              and n_timing == N and n_pipe == N and m["che_do"]["gia_lap"] and ok_rep,
              f"exit {r.returncode} {m.get('trang_thai')} dem={dem} timing={n_timing} pipeline={n_pipe}\n{why}\n"
              f"{r.stdout[-300:]}")


def t4_video(tmp: Path, anh: Path, cache: Path):
    py = PY_DET or PY
    mp4 = tmp / "seq.mp4"
    code = ("import cv2, glob, sys\nfs = sorted(glob.glob(sys.argv[1] + '/*.bmp'))\n"
            "w = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*'mp4v'), %g, (160, 96))\n"
            "[w.write(cv2.imread(f)) for f in fs]\nw.release()\n" % FPS)
    r0 = subprocess.run([py, "-c", code, str(anh), str(mp4)], capture_output=True, text=True)
    if r0.returncode != 0 or not mp4.exists():
        skip("4. video mp4", f"khong tao duoc mp4 bang {py}: {r0.stderr[-200:]}")
        return
    out = tmp / "video"
    r = chay(out, mp4, cache, "--tracker", "cv", "--save-cache", f"{TAG}_vid")
    m = meta(out)
    z = np.load(det_cache.CACHE_ROOT / f"{TAG}_vid" / "seq.npz") if r.returncode == 0 else None
    t_ok = z is not None and np.allclose(z["time_s"], (z["frame_idx"] - 1) / FPS, atol=0, rtol=0)
    check("4. video .mp4: so frame == khai bao, time_s == (idx-1)/fps, hoan_tat",
          r.returncode == 0 and m.get("hello_con", {}).get("nguon", {}).get("loai") == "video_file"
          and m["dem"]["n_frame_da_xu_ly"] == m["dem"]["n_frames_khai_bao"] == N and t_ok,
          f"exit {r.returncode} {m.get('trang_thai')} {m.get('dem')} time_s ok={t_ok}\n{r.stdout[-300:]}")


def t5_dung_som(tmp: Path, anh: Path, cache: Path):
    out = tmp / "dung"
    p = chay(out, anh, cache, "--tracker", "cv", "--save-cache", f"{TAG}_dung", "--gia-lap-cham-ms", "50", nen=True)
    pid = cho_pid(out)
    cho_hello(out)
    time.sleep(0.6)
    (out / "STOP").write_text("dung", encoding="utf-8")
    so = p.communicate(timeout=120)[0].decode("utf-8", "replace")
    m = meta(out)
    n = m.get("dem", {}).get("n_frame_da_xu_ly", -1)
    z = np.load(det_cache.CACHE_ROOT / f"{TAG}_dung" / "seq.npz") if (det_cache.CACHE_ROOT / f"{TAG}_dung").exists() else None
    ok_rep, why = replay_va_so(out, "rep-dung") if m else (False, "khong co meta")
    con = m.get("con", {})
    check("5. stop-file: exit 4, da_dung, 0 < frame < N, cache == frame da xu ly, replay khop, con thoat 0 voi EOS dung",
          p.returncode == 4 and m.get("trang_thai") == "da_dung" and not m.get("hoan_tat") and 0 < n < N
          and z is not None and len(z["frame_idx"]) == n and list(z["frame_idx"]) == list(range(1, n + 1))
          and ok_rep and con.get("returncode") == 0 and (m.get("eos_con") or {}).get("ly_do") == "dung_theo_yeu_cau"
          and pid is not None,
          f"exit {p.returncode} {m.get('trang_thai')} n={n} bay={m.get('dem', {}).get('n_frame_bay_khi_dung_khong_xu_ly')} "
          f"con={con} eos={m.get('eos_con')}\n{why}\n{so[-300:]}")
    if sys.platform.startswith("win"):
        skip("5b. SIGINT vao tien trinh cha", "Windows: khong gui SIGINT rieng cho tien trinh con duoc")
        return
    out = tmp / "sigint"
    p = chay(out, anh, cache, "--tracker", "cv", "--gia-lap-cham-ms", "50", nen=True)
    cho_hello(out)
    time.sleep(0.6)
    os.kill(p.pid, signal.SIGINT)
    p.communicate(timeout=120)
    m = meta(out)
    check("5b. SIGINT vao tien trinh cha: exit 4, da_dung, con thoat 0",
          p.returncode == 4 and m.get("trang_thai") == "da_dung" and m.get("con", {}).get("returncode") == 0,
          f"exit {p.returncode} {m.get('trang_thai')} con={m.get('con')}")


def t6_7_8_loi(tmp: Path, anh: Path, cache: Path):
    out = tmp / "con_loi"
    r = chay(out, anh, cache, "--tracker", "cv", "--gia-lap-loi-o-frame", "10")
    m = meta(out)
    check("6. con loi o frame 10 (thoat 3, khong EOS): exit 3, loi_tien_trinh_con, KHONG hoan_tat, 9 frame da xu ly duoc ghi",
          r.returncode == 3 and m.get("trang_thai") == "loi_tien_trinh_con" and not m.get("hoan_tat")
          and m["dem"]["n_frame_da_xu_ly"] == 9 and (out / "data" / "seq.txt").exists()
          and m.get("con", {}).get("returncode") == 3,
          f"exit {r.returncode} {m.get('trang_thai')} {m.get('dem')} con={m.get('con')}")

    out = tmp / "con_kill"
    p = chay(out, anh, cache, "--tracker", "cv", "--gia-lap-cham-ms", "50", nen=True)
    pid = cho_pid(out)
    hello = cho_hello(out)
    time.sleep(0.6)
    if pid:
        os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    p.communicate(timeout=120)
    m = meta(out)
    check("7. con bi kill giua chung: exit 3, loi_tien_trinh_con, KHONG hoan_tat",
          pid is not None and hello and p.returncode == 3 and m.get("trang_thai") == "loi_tien_trinh_con" and not m.get("hoan_tat")
          and 0 <= m["dem"]["n_frame_da_xu_ly"] < N,
          f"exit {p.returncode} {m.get('trang_thai')} {m.get('dem')} {m.get('chi_tiet')}")

    out = tmp / "bo_frame"
    r = chay(out, anh, cache, "--tracker", "cv", "--gia-lap-bo-frame", "7")
    m = meta(out)
    check("8. con bo frame 7: exit 3, loi_mat_frame (mong 7 nhan 8), 6 frame da xu ly",
          r.returncode == 3 and m.get("trang_thai") == "loi_mat_frame" and m["dem"]["n_frame_da_xu_ly"] == 6,
          f"exit {r.returncode} {m.get('trang_thai')} {m.get('chi_tiet')}")


def t9_anh_hong(tmp: Path, cache: Path):
    anh = tao_anh(tmp / "anh_hong", hong=5)
    out = tmp / "hong_macdinh"
    r = chay(out, anh, cache, "--tracker", "cv")
    m = meta(out)
    ok1 = r.returncode == 3 and m.get("trang_thai") == "loi_dau_vao" and m["dem"]["n_frame_da_xu_ly"] == 4
    out = tmp / "hong_chophep"
    r2 = chay(out, anh, cache, "--tracker", "cv", "--cho-phep-frame-hong", "--save-cache", f"{TAG}_hong")
    m2 = meta(out)
    ok_rep, why = replay_va_so(out, "rep-hong") if r2.returncode == 0 else (False, "khong replay")
    check("9. anh hong (frame 5 rac, frame 6 BMP bi cat): mac dinh exit 3 loi_dau_vao; --cho-phep-frame-hong -> exit 0, "
          "unreadable 2, con KHONG chet, replay khop",
          ok1 and r2.returncode == 0 and m2["dem"]["n_unreadable"] == 2 and m2["dem"]["n_frame_da_xu_ly"] == N and ok_rep,
          f"mac dinh: exit {r.returncode} {m.get('trang_thai')}; cho phep: exit {r2.returncode} {m2.get('dem')}\n{why}")


def t10_loi_truoc(tmp: Path, anh: Path, cache: Path):
    kq = {}
    kq["nguon"] = chay(tmp / "e1", tmp / "khong_co", cache, "--tracker", "cv").returncode
    r = subprocess.run([PY, "src/nano_live.py", "--source", str(anh), "--fps", "10", "--engine", str(tmp / "x.engine"),
                        "--tracker", "cv", "--out", str(tmp / "e2")], cwd=ROOT, env=ENV, capture_output=True, text=True)
    kq["engine"] = r.returncode
    kq["da_co_ket_qua"] = chay(tmp / "e2e_cv", anh, cache, "--tracker", "cv").returncode   # thu muc cua phep 2
    out = tmp / "e4"
    kq["con_khoi_tao"] = chay(out, anh, tmp / "cache_khong_co", "--tracker", "cv").returncode
    m = meta(out)
    check("10. loi truoc vong lap -> exit 2: nguon/engine khong co, thu muc da co ket qua, con loi khoi tao (ERROR truoc HELLO)",
          set(kq.values()) == {2} and m.get("trang_thai") == "loi_khoi_tao" and not (tmp / "e1").exists(),
          f"{kq} meta={m.get('trang_thai')} {str(m.get('chi_tiet'))[:120]}")


def t11_cua_so(tmp: Path, anh: Path, cache: Path):
    out = tmp / "cua_so"
    r = chay(out, anh, cache, "--tracker", "cv", "--hang-doi", "2", "--gia-lap-cham-tracker-ms", "25")
    m = meta(out)
    import pandas as pd
    pl = pd.read_csv(out / "pipeline" / "seq.csv") if (out / "pipeline" / "seq.csv").exists() else None
    cho = float((pl["cho_credit_ms"] > 5).sum()) if pl is not None else 0
    check("11. cua so K=2 + tracker cham: du N frame (khong bo), max_in_flight <= 2, con phai cho credit",
          r.returncode == 0 and m["dem"]["n_frame_da_xu_ly"] == N and (m.get("eos_con") or {}).get("max_in_flight", 99) <= 2
          and cho > 0,
          f"exit {r.returncode} eos={m.get('eos_con')} so frame con cho credit > 5 ms: {cho:.0f}")


def t12_cham_tran(tmp: Path, anh: Path):
    cache = tao_cache_gia_lap(tmp / "cache_tran", nhieu_o=15)
    out = tmp / "tran_macdinh"
    r = chay(out, anh, cache, "--tracker", "cv", "--det-max-det", "6", "--max-det", "6")
    m = meta(out)
    out2 = tmp / "tran_chophep"
    r2 = chay(out2, anh, cache, "--tracker", "cv", "--det-max-det", "6", "--max-det", "6", "--allow-truncated",
              "--save-cache", f"{TAG}_tran")
    m2 = meta(out2)
    ok_rep, why = replay_va_so(out2, "rep-tran") if r2.returncode == 0 else (False, "khong replay")
    check("12. cham tran det_max_det o frame 15: mac dinh exit 3; --allow-truncated -> exit 0, replay (--allow-truncated) khop",
          r.returncode == 3 and m.get("trang_thai") == "loi_dau_vao" and r2.returncode == 0 and m2["dem"]["n_truncated"] >= 1
          and "--allow-truncated" in m2["replay_argv"] and ok_rep,
          f"mac dinh exit {r.returncode} {m.get('chi_tiet')}; cho phep exit {r2.returncode} {m2.get('dem')}\n{why}")


def t13_regions(tmp: Path, anh: Path, cache: Path):
    reg = tmp / "ignored_regions.txt"
    reg.write_text("10,10,20,20\n", encoding="utf-8")
    out = tmp / "regions"
    r = chay(out, anh, cache, "--tracker", "cv", "--regions", str(reg))
    m = meta(out)
    h = hashlib.sha256(reg.read_bytes()).hexdigest()
    check("13. --regions (che do doi chieu): metadata ghi co + sha256 file vung, van hoan_tat",
          r.returncode == 0 and m["che_do"]["che_do_doi_chieu_regions"] and m["che_do"]["regions_sha256"] == h,
          f"exit {r.returncode} {m.get('che_do')}")


def t14_torch(tmp: Path):
    w = ROOT / "models" / "yolov8n.pt"
    img = ROOT / "data" / "extracted" / "DETRAC-Images" / "MVI_39031"
    if PY_DET or not w.exists() or not img.is_dir():
        skip("14. detector that TorchPtInfer (PC)", "khong co models/yolov8n.pt / anh DETRAC, hoac dang chay tren Nano")
        return
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception as e:  # noqa: BLE001
        skip("14. detector that TorchPtInfer (PC)", f"thieu torch/ultralytics: {e}")
        return
    out = tmp / "torch"
    r = subprocess.run([PY, "src/nano_live.py", "--source", str(img), "--fps", "25", "--video-name", "MVI_39031",
                        "--torch-pt", str(w), "--max-frames", "8", "--tracker", "cv", "--out", str(out),
                        "--save-cache", f"{TAG}_torch", "--warmup-frames", "2", "--lam-nong", "1"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = meta(out)
    ok_rep, why = replay_va_so(out, "rep-torch") if r.returncode == 0 else (False, "khong replay")
    check("14. detector that TorchPtInfer 8 frame DETRAC (PC): hoan_tat (max_frames), replay tu cache khop",
          r.returncode == 0 and m["dem"]["n_frame_da_xu_ly"] == 8 and m["hello_con"]["detector"] == "torch_pt_pc" and ok_rep,
          f"exit {r.returncode} {m.get('trang_thai')} {m.get('chi_tiet')}\n{why}\n{r.stdout[-300:]}")


def main() -> int:
    global PY_DET
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-detector", default=None, help="interpreter cua tien trinh con (Nano: /usr/bin/python3)")
    a = ap.parse_args()
    PY_DET = a.python_detector
    print(f"cha: {sys.version.split()[0]} | con: {PY_DET or PY} | backend np")
    tmp = Path(tempfile.mkdtemp(prefix="nano_stream_"))
    try:
        t1_giao_thuc()
        anh = tao_anh(tmp / "anh")
        cache = tao_cache_gia_lap(tmp / "cache_gia_lap")
        t2_3_e2e(tmp, anh, cache)
        t4_video(tmp, anh, cache)
        t5_dung_som(tmp, anh, cache)
        t6_7_8_loi(tmp, anh, cache)
        t9_anh_hong(tmp, cache)
        t10_loi_truoc(tmp, anh, cache)
        t11_cua_so(tmp, anh, cache)
        t12_cham_tran(tmp, anh)
        t13_regions(tmp, anh, cache)
        t14_torch(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for p in det_cache.CACHE_ROOT.glob(f"{TAG}_*"):
            shutil.rmtree(p, ignore_errors=True)
        shutil.rmtree(config.PROCESSED_DIR / "trackers" / SPLIT, ignore_errors=True)
    n_fail = sum(1 for _, s in _R if s == "FAIL")
    n_pass = sum(1 for _, s in _R if s == "PASS")
    n_skip = sum(1 for _, s in _R if s == "SKIP")
    print(f"\nKET QUA: {n_pass}/{n_pass + n_fail} phep kiem tra PASS" + (f", {n_skip} SKIP" if n_skip else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
