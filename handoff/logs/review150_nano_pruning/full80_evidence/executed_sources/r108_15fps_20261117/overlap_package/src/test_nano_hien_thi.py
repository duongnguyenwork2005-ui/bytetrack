"""
test_nano_hien_thi.py -- Kiem HIEN THI TRUC TIEP + LAUNCHER TAI CHO tren Nano (vong 25).

Chay tren PC (con = sys.executable) va tren Nano (`--python-detector /usr/bin/python3`). Detector GIA LAP (det cache,
van doc anh that); hien thi o che do AN (moi buoc tru tao cua so: ghep theo frame_idx, ve bang render_mot_video.ve_frame,
luu mau PNG, dem, gia lap phim q). Cua so that + HDMI duoc kiem rieng tren thiet bi.

  1. Giao thuc 3: lenh R dong goi/tach dung ca khi cat tung byte, tron voi G/+/S, R rong, byte la duoc dem; float32 giu nguyen.
  2. `nano_live.hang_hien_thi(out)` == DUNG hang MOT/lop ma `run_tracking.track_video` ghi (so sanh so thuc tung hang).
  3. CV va EKF: bat hien thi (an) vs tat -> MOT + lop GIONG TUNG BYTE; so lenh R == so frame doc duoc; ghep 100 %,
     0 khong ghep; progress.json cuoi 'hoan_tat'; run_meta.hien_thi + thoi_gian.fps_hien_thi/tong_s co mat.
  4. Anh xem truoc (PNG khong mat mat) == ve_frame(anh goc, hang MOT DA GHI cua CUNG frame) tung pixel (cung OpenCV voi con).
  5. Phim q gia lap o frame 15 -> stop-file -> ma 4 da_dung; so frame xu ly >= 15 va < N; MOT == tien to cua luot du.
  6. Frame khong doc duoc (--cho-phep-frame-hong): frame hong nhan credit '+', frame con lai nhan R; 0 khong ghep.
  7. (POSIX) --hien-thi khi khong co man hinh -> ma 2 TRUOC khi khoi dong detector, thong bao ro.
  8. Launcher (controller, khong Tk): kiem trien khai (thieu engine -> loi kem cach khac phuc); argv danh sach giu duong
     dan co dau cach; cau hinh sai -> CauHinhSai; hoan tat -> HOAN_TAT + launcher.json; Dung -> DA_DUNG (ma 4); worker loi
     -> LOI; dong launcher khi dang chay -> dung co kiem soat (KHONG terminate); moi luot mot thu muc moi.
  9. (neu co Tk + man hinh) giao dien dung duoc: thieu engine -> nut Bat dau bi khoa, thong bao; dong cua so sach.
Review 20e66e9 (vong 26):
 10. N25-F1 (lop HienThi, che do THAT, HighGUI gia lap): frame 1 da hien -> dong cua so -> ket qua frame 2 toi muon:
     imshow 1 lan, n_hien_thi 1, frame cuoi 1, fps null (N < 2), 1 frame "khong hien thi vi cua so dong", khong tao lai cua so.
     10b. dong cua so GIUA hai frame (kiem truoc imshow): khong imshow/tao lai cua so. 10c. fps = (N-1)/khoang khi N >= 2.
 11. N25-F1 dau-cuoi (nano_live, hien thi an, gia lap dong cua so o frame 5): ma 4 da_dung; frame toi sau khi dong khong tinh
     la da hien thi; MOT + lop cua MOI frame tracker da xu ly == tien to luot du (khong mat du lieu tracking).
 12. N25-F2 (POSIX; tien trinh chon file THAT thay zenity, kiem theo PID): (a) mo hop chon roi dong launcher khi chua chay
     -> tien trinh chon file da thoat + thu hoi, callback cho da huy, khong cham widget sau khi dong; (b) huy hop chon roi mo
     lai va chon duoc nguon; (c) hop chon vua tu thoat dung luc dong -> chi thu hoi, bo ket qua; (d) hop chon dang mo + tracker
     dang chay -> dong launcher: tracker DA_DUNG, hop chon da thoat, tien trinh worker/detector da thoat; (e) (neu co man hinh)
     cung (a) tren cua so Tk THAT: khong loi callback sau destroy.

    python src/test_nano_hien_thi.py [--python-detector /usr/bin/python3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import nano_stream_proto as proto  # noqa: E402
import test_nano_stream as tns  # noqa: E402   (fixture + chay nano_live dung chung)

ROOT = config.PROJECT_ROOT
PY = sys.executable
_R: list[tuple[str, str]] = []
N = tns.N


def check(name, ok, detail=""):
    _R.append((name, "PASS" if ok else "FAIL"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for ln in str(detail).strip().splitlines()[-6:]:
        print(f"         {ln}")


def skip(name, why):
    _R.append((name, "SKIP"))
    print(f"  [SKIP] {name}: {why}")


def mot(out: Path, ten="seq") -> bytes:
    return (out / "data" / f"{ten}.txt").read_bytes().replace(b"\r\n", b"\n")


def lop(out: Path, ten="seq") -> bytes:
    return (out / "data_cls" / f"{ten}.csv").read_bytes().replace(b"\r\n", b"\n")


# ------------------------------------------------------------------------------------------------ 1
def t1_giao_thuc():
    hang = [(3, 10.5, 20.25, 30.125, 40.0, 0.875, 2), (17, 1e-3, 2.5, 3.75, 4.5, 0.3, 7)]
    b = (proto.LENH_BAT_DAU + proto.dong_goi_ket_qua(5, hang) + proto.LENH_CREDIT + proto.dong_goi_ket_qua(6, [])
         + b"x" + proto.LENH_DUNG)
    bo = proto.BoDocLenh()
    ev = []
    for k in range(len(b)):                        # cat TUNG byte: lenh R phai duoc ghep lai dung
        ev += bo.feed(b[k:k + 1])
    f32 = [tuple(float(np.float32(v)) if isinstance(v, float) else v for v in h) for h in hang]
    ok = (ev[0] == ("G",) and ev[1][0] == "R" and ev[1][1] == 5 and [tuple(x) for x in ev[1][2]] == f32
          and ev[2] == ("+",) and ev[3] == ("R", 6, []) and ev[4] == ("S",) and len(ev) == 5 and bo.n_byte_la == 1
          and not bo.buf and proto.PHIEN_BAN == 3)
    bo2 = proto.BoDocLenh()
    ev2 = bo2.feed(b)                              # ca khoi mot lan
    check("1. giao thuc 3: R tach dung khi cat tung byte / ca khoi, tron G + S, R rong, byte la dem, float32 giu nguyen",
          ok and ev2 == ev, f"{ev} | byte_la {bo.n_byte_la}")


# ------------------------------------------------------------------------------------------------ 2
def t2_hang_bang_mot(tmp: Path, cache: Path):
    import nano_live
    import run_tracking
    vd = det_cache.VideoDetections(cache / "seq.npz", 0.25, [1, 2, 3, 5, 7], 300)
    for ten, yaml in (("CV", "bytetrack.yaml"), ("EKF", "configs/bytetrack_ekf_ctrv.yaml")):
        cfg = run_tracking.load_tracker_cfg(yaml, {"track_buffer": 12})
        hien = []
        df, df_cls, _ = run_tracking.track_video(vd, cfg, 1 / tns.FPS, "steps", None, "seq",
                                                 on_frame=lambda i, out: hien.extend((i, *h) for h in nano_live.hang_hien_thi(out)))
        ghi = [(int(r.frame), int(r.id), float(r.bb_left), float(r.bb_top), float(r.bb_width), float(r.bb_height),
                float(r.conf)) for r in df.itertuples()]
        cls_ghi = [(int(r.frame), int(r.id), int(r.cls)) for r in df_cls.itertuples()]
        ht = [(h[0], h[1], h[2], h[3], h[4], h[5], h[6]) for h in hien]
        ht_cls = [(h[0], h[1], h[7]) for h in hien]
        check(f"2. hang_hien_thi(out) == hang MOT/lop track_video ghi ({ten}, so thuc bang nhau tung hang)",
              ht == ghi and ht_cls == cls_ghi and len(ghi) > 0, f"{len(ghi)} hang MOT, {len(ht)} hang hien thi")


# ------------------------------------------------------------------------------------------------ 3, 4
def so_mau(mau: Path, anh: Path, out: Path) -> dict:
    """So PNG xem truoc voi ve_frame(anh goc, hang MOT DA GHI cua frame do) BANG interpreter cua con (cung OpenCV)."""
    code = r"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
import cv2
import render_mot_video as rmv
mau, anh, mot, cls = sys.argv[1:5]
boxes, lop = rmv.doc_mot(mot), rmv.doc_lop(cls)
ds = rmv.liet_ke(Path(anh))
r = {"n": 0, "khop": 0, "lech": [], "frame": []}
for f in sorted(os.listdir(mau)):
    i = int(f[len("hien_thi_f"):-4])
    got = cv2.imread(os.path.join(mau, f))
    kv = cv2.imread(str(ds[i - 1]))
    rmv.ve_frame(cv2, kv, boxes.get(i, []), lop, i)
    r["n"] += 1
    r["frame"].append(i)
    if got is not None and got.shape == kv.shape and (got == kv).all():
        r["khop"] += 1
    else:
        r["lech"].append(i)
print(json.dumps(r))
"""
    py = tns.PY_DET or PY
    p = subprocess.run([py, "-c", code, str(mau), str(anh), str(out / "data" / "seq.txt"), str(out / "data_cls" / "seq.csv"),
                        str(ROOT / "src")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"n": 0, "khop": 0, "loi": (p.stdout + p.stderr)[-400:]}


def t3_4_bat_tat(tmp: Path, anh: Path, cache: Path):
    for tr in ("cv", "ekf"):
        tat, bat = tmp / f"t3_{tr}_tat", tmp / f"t3_{tr}_bat"
        r0 = tns.chay(tat, anh, cache, "--tracker", tr)
        r1 = tns.chay(bat, anh, cache, "--tracker", tr, "--hien-thi-an", "--hien-thi-luu-mau", str(tmp / f"mau_{tr}"),
                      "--hien-thi-moi", "1", "--gia-lap-cham-ms", "15")
        m1 = tns.meta(bat)
        h = m1.get("hien_thi") or {}
        d = h.get("dem") or {}
        n_doc = int((m1.get("dem") or {}).get("n_frame_vao_tracker_doc_duoc", -1))
        pr = json.loads((bat / "progress.json").read_text(encoding="utf-8")) if (bat / "progress.json").exists() else {}
        tg = m1.get("thoi_gian") or {}
        ok = (r0.returncode == 0 and r1.returncode == 0 and mot(tat) == mot(bat) and lop(tat) == lop(bat)
              and h.get("lenh_R_da_gui") == n_doc == N and h.get("lenh_credit_da_gui") == 0 and h.get("ket_qua_lech_frame") == 0
              and d.get("n_ket_qua_nhan") == N and d.get("n_ghep_duoc") == N and d.get("n_khong_ghep") == 0
              and d.get("n_hien_thi", 0) + d.get("n_bo_hien_thi", 0) == N and h.get("loi") is None
              and pr.get("trang_thai") == "hoan_tat" and pr.get("frame") == N and tg.get("tong_s") and "fps_hien_thi" in tg
              and (m1.get("che_do") or {}).get("hien_thi") is True and tns.meta(tat).get("hien_thi") == {"bat": False})
        check(f"3. {tr.upper()}: bat hien thi (an) vs tat -> MOT + lop giong tung byte; R == {N} frame, ghep 100 %, "
              "0 khong ghep; progress hoan_tat; fps_hien_thi/tong_s", ok,
              f"exit {r0.returncode}/{r1.returncode}; MOT giong {mot(tat) == mot(bat)}; R {h.get('lenh_R_da_gui')} + "
              f"{h.get('lenh_credit_da_gui')}; dem {d}; progress {pr.get('trang_thai')}")
        s = so_mau(tmp / f"mau_{tr}", anh, bat)
        check(f"4. {tr.upper()}: anh xem truoc (PNG) == ve_frame(anh goc, hang MOT da ghi cua CUNG frame) tung pixel",
              s.get("n", 0) >= 10 and s.get("khop") == s.get("n") and s.get("n") == d.get("n_mau_luu"),
              f"{s.get('khop')}/{s.get('n')} khop, lech {s.get('lech')}, frame {s.get('frame', [])[:12]}... {s.get('loi', '')}")


# ------------------------------------------------------------------------------------------------ 5
def t5_phim_q(tmp: Path, anh: Path, cache: Path):
    du, q = tmp / "t5_du", tmp / "t5_q"
    tns.chay(du, anh, cache, "--tracker", "cv")
    r = tns.chay(q, anh, cache, "--tracker", "cv", "--hien-thi-an", "--gia-lap-phim-dung-o-frame", "15",
                 "--gia-lap-cham-ms", "25")
    m = tns.meta(q)
    h = m.get("hien_thi") or {}
    n = int((m.get("dem") or {}).get("n_frame_da_xu_ly", -1))
    tien_to = [ln for ln in mot(du).split(b"\n") if ln and int(ln.split(b",")[0]) <= n]
    ok = (r.returncode == 4 and m.get("trang_thai") == "da_dung" and "stop-file" in str(m.get("chi_tiet"))
          and h.get("ly_do_dung_tu_cua_so") == "gia_lap_phim_q" and 15 <= n < N
          and mot(q) == b"\n".join(tien_to) + b"\n" and (m.get("con") or {}).get("returncode") == 0)
    check("5. phim q (gia lap) o frame 15 -> stop-file -> ma 4 da_dung; 15 <= frame xu ly < N; MOT == tien to luot du; con thoat 0",
          ok, f"exit {r.returncode} {m.get('trang_thai')} {m.get('chi_tiet')}; n={n}; ly do {h.get('ly_do_dung_tu_cua_so')}")


# ------------------------------------------------------------------------------------------------ 6
def t6_frame_hong(tmp: Path, cache: Path):
    anh = tns.tao_anh(tmp / "anh_hong", hong=5)
    out = tmp / "t6"
    r = tns.chay(out, anh, cache, "--tracker", "cv", "--hien-thi-an", "--cho-phep-frame-hong")
    m = tns.meta(out)
    h = m.get("hien_thi") or {}
    d = h.get("dem") or {}
    n_hong = int((m.get("dem") or {}).get("n_unreadable", -1))
    ok = (r.returncode == 0 and n_hong == 2 and h.get("lenh_credit_da_gui") == 2 and h.get("lenh_R_da_gui") == N - 2
          and d.get("n_khong_ghep") == 0 and d.get("n_ghep_duoc") == N - 2 and d.get("n_anh_nhan") == N - 2)
    check("6. frame khong doc duoc: 2 frame hong nhan '+', 38 frame con lai nhan R; 0 khong ghep", ok,
          f"exit {r.returncode}; hong {n_hong}; R {h.get('lenh_R_da_gui')} + {h.get('lenh_credit_da_gui')}; dem {d}")


# ------------------------------------------------------------------------------------------------ 7
def t7_khong_man_hinh(tmp: Path, anh: Path, cache: Path):
    if sys.platform.startswith("win"):
        skip("7. --hien-thi khi khong co man hinh -> ma 2 truoc khi khoi dong detector", "chi POSIX (Windows luon co man hinh)")
        return
    out = tmp / "t7"
    env = {k: v for k, v in tns.ENV.items() if k != "DISPLAY"}
    cmd = [PY, "src/nano_live.py", "--source", str(anh), "--fps", str(tns.FPS), "--video-name", "seq",
           "--gia-lap-cache", str(cache), "--gia-lap-video", "seq", "--out", str(out), "--tracker", "cv", "--hien-thi"]
    if tns.PY_DET:
        cmd += ["--python-detector", tns.PY_DET]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    check("7. --hien-thi khi khong co man hinh (DISPLAY trong) -> ma 2, KHONG khoi dong detector, thong bao DISPLAY",
          r.returncode == 2 and "DISPLAY" in r.stdout and not (out / "detector.pid").exists(),
          f"exit {r.returncode}: {r.stdout.strip()[-200:]}")


# ------------------------------------------------------------------------------------------------ 8
def t8_launcher(tmp: Path, anh: Path, cache: Path):
    import nano_launcher as nl
    # 8a. kiem trien khai
    goi = tmp / "cfg_goi.json"
    goi.write_text(json.dumps({"engine": [str(tmp / "khong_co.engine")], "python_detector": PY, "ld_preload": ""}),
                   encoding="utf-8")
    c = nl.doc_cau_hinh(goi, tmp / "khong_co_nd.json")
    tk_ = nl.kiem_trien_khai(c)
    fake = tmp / "gia.engine"
    fake.write_bytes(b"x")
    c2 = dict(c, engine=[str(tmp / "khong_co.engine"), str(fake)])
    tk2 = nl.kiem_trien_khai(c2)
    check("8a. launcher kiem trien khai: thieu engine -> loi kem cach khac phuc (01_build_engine.sh); co engine -> 0 loi",
          any("engine TensorRT" in e and "01_build_engine.sh" in e for e in tk_["loi"]) and not tk2["loi"]
          and tk2["engine"] == fake, f"{tk_['loi']} | {tk2['loi']}")

    # 8b. argv la danh sach, giu duong dan co dau cach; moi luot mot thu muc moi
    src_sp = tmp / "thu muc anh co dau cach"
    shutil.copytree(anh, src_sp)
    out_sp = tmp / "ket qua co dau cach"
    gl = ["--gia-lap-cache", str(cache), "--gia-lap-video", "seq", "--video-name", "seq", "--timeout-con-s", "60",
          "--warmup-frames", "5"]
    if tns.PY_DET:
        gl += ["--python-detector", tns.PY_DET]
    ctl = nl.NanoLauncherController(c2, tk2, timeout_dung=30, argv_them=gl)
    cfg = {"source": str(src_sp), "fps": str(tns.FPS), "tracker": "ekf", "out_base": str(out_sp), "hien_thi": True,
           "luu_cache": True}
    argv, rd = ctl.build_argv(cfg)
    _, rd2 = ctl.build_argv(cfg)
    ok_b = (str(src_sp) in argv and argv[argv.index("--source") + 1] == str(src_sp) and argv[argv.index("--out") + 1] == str(rd)
            and "--hien-thi" in argv and "--save-cache" in argv and argv[argv.index("--fps") + 1] == str(tns.FPS)
            and rd.parent == out_sp and rd.name.startswith("thu muc anh co dau cach_ekf_"))
    loi_cfg = []
    for sai in ({"source": ""}, {"source": str(tmp / "khong_co")}, {"source": str(src_sp), "fps": ""},
                {"source": str(src_sp), "fps": "0"}, {"source": str(src_sp), "fps": "10", "tracker": "ukf"},
                {"source": str(src_sp), "fps": "10", "tracker": "cv", "max_frames": "abc"}):
        try:
            ctl.build_argv({"tracker": "cv", "out_base": str(out_sp), **sai})
            loi_cfg.append(None)
        except nl.CauHinhSai as e:
            loi_cfg.append(str(e)[:60])
    check("8b. argv danh sach giu nguyen duong dan co dau cach (nguon, ket qua); cau hinh sai -> CauHinhSai tieng Viet",
          ok_b and all(loi_cfg), f"run_dir {rd.name}; loi {loi_cfg}")

    def cho_xong(ctl_, timeout=120):
        t0 = time.time()
        while ctl_.dang_chay() and time.time() - t0 < timeout:
            ctl_.poll()
            time.sleep(0.1)
        return ctl_.poll()

    # 8c. hoan tat
    ctl.entry_prefix = [PY, str(ROOT / "src" / "nano_live.py")]
    rdc = ctl.start({**cfg, "hien_thi": False})
    s = cho_xong(ctl)
    lj = json.loads((rdc / "launcher.json").read_text(encoding="utf-8")) if (rdc / "launcher.json").exists() else {}
    tt = ctl.tom_tat_ket_qua() or ""
    check("8c. launcher chay het -> HOAN_TAT (exit 0), launcher.json + launcher_nhat_ky.txt, tom tat tieng Viet, cache trong "
          "run_dir; khong con tien trinh chan khoa man hinh",
          s["trang_thai"] == "HOAN_TAT" and s["exit_code"] == 0 and lj.get("tong_s_phia_launcher") and "frame đã xử lý" in tt
          and s.get("fps_xu_ly") and any((rdc / "det_cache").rglob("*.npz")) and (rdc / "launcher_nhat_ky.txt").exists()
          and not con_chan_khoa(),
          f"{s['trang_thai']} exit {s['exit_code']} loi {s['loi']}; chan khoa con {con_chan_khoa()}; {tt[:120]}")

    # 8d. Dung (nut) -> DA_DUNG
    ctl_d = nl.NanoLauncherController(c2, tk2, timeout_dung=30, argv_them=gl + ["--gia-lap-cham-ms", "150"])
    ctl_d.entry_prefix = ctl.entry_prefix
    rdd = ctl_d.start({**cfg, "hien_thi": False, "luu_cache": False})
    t0 = time.time()
    while ctl_d.poll()["trang_thai"] != "DANG_CHAY" and time.time() - t0 < 60:
        time.sleep(0.1)
    ctl_d.request_stop()
    s = cho_xong(ctl_d)
    md = tns.meta(rdd)
    check("8d. nut Dung -> worker dung co kiem soat (ma 4) -> DA_DUNG (khong phai LOI), run_meta da_dung, khong terminate",
          s["trang_thai"] == "DA_DUNG" and s["exit_code"] == 4 and md.get("trang_thai") == "da_dung" and not ctl_d.da_terminate
          and rdd != rdc, f"{s['trang_thai']} exit {s['exit_code']} {md.get('trang_thai')} {md.get('chi_tiet')}")

    # 8e. worker loi -> LOI
    ctl_e = nl.NanoLauncherController(c2, tk2, timeout_dung=30, argv_them=gl + ["--gia-lap-loi-o-frame", "10"])
    ctl_e.entry_prefix = ctl.entry_prefix
    ctl_e.start({**cfg, "hien_thi": False, "luu_cache": False})
    s = cho_xong(ctl_e)
    check("8e. worker loi giua luong (con chet o frame 10) -> LOI, exit 3, thong bao loi", s["trang_thai"] == "LOI"
          and s["exit_code"] == 3 and s["loi"], f"{s['trang_thai']} exit {s['exit_code']} {str(s['loi'])[:120]}")

    # 8f. dong launcher khi dang chay -> dung co kiem soat
    ctl_f = nl.NanoLauncherController(c2, tk2, timeout_dung=30,
                                      argv_them=gl + ["--gia-lap-cham-ms", "150", "--hien-thi-an"])
    ctl_f.entry_prefix = ctl.entry_prefix
    rdf = ctl_f.start({**cfg, "hien_thi": True, "luu_cache": False})
    t0 = time.time()
    while ctl_f.poll()["trang_thai"] != "DANG_CHAY" and time.time() - t0 < 60:
        time.sleep(0.1)
    tt_f = ctl_f.shutdown()
    mf = tns.meta(rdf)
    check("8f. dong launcher khi dang chay (shutdown) -> DA_DUNG qua stop-file, KHONG terminate, run_meta da_dung, "
          "tra lai khoa man hinh",
          tt_f == "DA_DUNG" and not ctl_f.da_terminate and mf.get("trang_thai") == "da_dung" and not con_chan_khoa(),
          f"{tt_f} terminate={ctl_f.da_terminate} {mf.get('trang_thai')} chan khoa con {con_chan_khoa()}")


def tien_trinh_linux(ten_exe: str, argv_can: list[str]) -> list[int]:
    """PID cac tien trinh THAT co executable ten `ten_exe` va argv chua DUNG cac phan tu `argv_can` (doc /proc; KHONG dung
    `pgrep -f` vi no co the dem chinh no / shell chua mau tim -- review 20e66e9). Khong phai Linux -> []."""
    if not Path("/proc/self/exe").exists():
        return []
    ra = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            exe = os.path.basename(os.readlink(d / "exe"))
            argv = [a.decode("utf-8", "replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if exe == ten_exe and all(x in argv for x in argv_can):
            ra.append(int(d.name))
    return ra


def con_chan_khoa() -> int:
    """So tien trinh gnome-session-inhibit --app-id nano-tracking CUA launcher con song (theo executable + argv)."""
    time.sleep(0.5)
    return len(tien_trinh_linux("gnome-session-inhibit", ["--app-id", "nano-tracking"]))


def pid_con_song(pid: int) -> bool:
    """PID con ton tai (va KHONG phai zombie) -- POSIX."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    st = Path(f"/proc/{pid}/stat")
    if st.exists():
        try:
            return st.read_text().split(")")[-1].split()[0] != "Z"
        except OSError:
            return False
    return True


# ------------------------------------------------------------------------------------------------ 10, 11 (N25-F1)
class HighGuiGia:
    """Thay cac thao tac cua so HighGUI (ve / chu / imwrite dung cv2 that). `dong` = cua so da bi nguoi dung dong."""

    def __init__(self, cv2):
        self._cv2, self.imshow_n, self.named_n, self.dong = cv2, 0, 0, False

    def __getattr__(self, ten):
        return getattr(self._cv2, ten)

    def namedWindow(self, *a):
        self.named_n += 1
        self.dong = False

    def resizeWindow(self, *a):
        pass

    def imshow(self, *a):
        self.imshow_n += 1

    def waitKey(self, *a):
        return 255                      # OpenCV 3.2 tra 255 khi khong co phim

    def getWindowProperty(self, *a):
        return -1.0 if self.dong else 1.0

    def destroyAllWindows(self):
        pass


def _cho(dk, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if dk():
            return True
        time.sleep(0.01)
    return bool(dk())


def t10_f1_don_vi(tmp: Path):
    try:
        import cv2
    except ImportError as e:
        skip("10. N25-F1 lop HienThi (che do that, HighGUI gia lap)", f"khong co cv2 trong {PY} ({e})")
        return
    import nano_hien_thi as nh
    import render_mot_video as rmv
    anh = np.zeros((96, 160, 3), np.uint8)
    hang = [(1, 10.0, 10.0, 30.0, 20.0, 0.9, 2)]
    # 10: dong cua so SAU frame 1 (su kien), frame 2 toi muon
    g = HighGuiGia(cv2)
    ht = nh.HienThi(g, rmv, "fixture", an=False, stop_file=str(tmp / "STOP10"))
    ht.them_anh(1, anh); ht.nhan_ket_qua(1, hang)
    _cho(lambda: g.imshow_n == 1)
    g.dong = True                                     # nguoi dung bam x
    _cho(lambda: ht._cua_so_da_dong)
    ht.them_anh(2, anh); ht.nhan_ket_qua(2, hang)     # frame dang bay toi SAU khi dong
    tk = ht.dong("dung_theo_yeu_cau")
    d = tk["dem"]
    ok = (g.imshow_n == 1 and g.named_n == 1 and d["n_hien_thi"] == 1 and tk["frame_hien_thi_cuoi"] == 1
          and tk["fps_hien_thi"] is None and d["n_khong_hien_thi_cua_so_dong"] == 1
          and tk["frame_khong_hien_thi_cua_so_dong_dau"] == [2] and tk["ly_do_dung_tu_cua_so"] == "dong_cua_so"
          and d["n_ghep_duoc"] == d["n_hien_thi"] + d["n_bo_hien_thi"] + d["n_khong_hien_thi_cua_so_dong"]
          and (tmp / "STOP10").exists())
    check("10. N25-F1: frame 1 hien -> dong cua so -> frame 2 toi muon: imshow 1, n_hien_thi 1, frame cuoi 1, fps null, "
          "1 frame khong hien vi cua so dong, khong tao lai cua so, stop-file ghi", ok,
          f"imshow {g.imshow_n} named {g.named_n} dem {d} cuoi {tk['frame_hien_thi_cuoi']} fps {tk['fps_hien_thi']}")
    # 10b: dong GIUA hai frame -- frame 3 den khi cua so da dong nhung chua qua _su_kien: kiem truoc imshow
    g = HighGuiGia(cv2)
    ht = nh.HienThi(g, rmv, "fixture", an=False, stop_file=str(tmp / "STOP10b"))
    for i in (1, 2):
        ht.them_anh(i, anh); ht.nhan_ket_qua(i, hang)
        _cho(lambda: g.imshow_n == i)
    g.dong = True
    ht.them_anh(3, anh); ht.nhan_ket_qua(3, hang)
    tk = ht.dong("dung_theo_yeu_cau")
    d = tk["dem"]
    check("10b. N25-F1: cua so dong giua hai frame -> frame 3 KHONG imshow, KHONG tao lai cua so; n_hien_thi 2, fps co gia tri",
          g.imshow_n == 2 and g.named_n == 1 and d["n_hien_thi"] == 2 and d["n_khong_hien_thi_cua_so_dong"] == 1
          and tk["frame_hien_thi_cuoi"] == 2 and tk["fps_hien_thi"] is not None,
          f"imshow {g.imshow_n} named {g.named_n} dem {d} fps {tk['fps_hien_thi']}")
    # 10c: fps = (N - 1) / khoang
    g = HighGuiGia(cv2)
    ht = nh.HienThi(g, rmv, "fixture", an=False)
    for i in range(1, 6):
        ht.them_anh(i, anh); ht.nhan_ket_qua(i, hang)
        _cho(lambda: g.imshow_n == i)
        time.sleep(0.02)
    tk = ht.dong("het_nguon")
    n = tk["dem"]["n_hien_thi"]
    check("10c. fps_hien_thi = (N - 1) / (moc cuoi - moc dau), N = so frame da hien thi",
          n == 5 and abs(tk["fps_hien_thi"] * tk["khoang_hien_thi_s"] - (n - 1)) < 1e-6 and "(N - 1)" in tk["dinh_nghia"]["fps_hien_thi"],
          f"N {n} fps {tk['fps_hien_thi']} khoang {tk['khoang_hien_thi_s']}")


def t11_f1_dau_cuoi(tmp: Path, anh: Path, cache: Path):
    du, cs = tmp / "t11_du", tmp / "t11_dong"
    tns.chay(du, anh, cache, "--tracker", "cv")
    r = tns.chay(cs, anh, cache, "--tracker", "cv", "--hien-thi-an", "--gia-lap-dong-cua-so-o-frame", "5",
                 "--gia-lap-cham-ms", "25")
    m = tns.meta(cs)
    h = m.get("hien_thi") or {}
    d = h.get("dem") or {}
    n = int((m.get("dem") or {}).get("n_frame_da_xu_ly", -1))
    tien_to = [ln for ln in mot(du).split(b"\n") if ln and int(ln.split(b",")[0]) <= n]
    tien_to_lop = [ln for ln in lop(du).split(b"\n")[1:] if ln and int(ln.split(b",")[0]) <= n]
    khong = h.get("frame_khong_hien_thi_cua_so_dong_dau") or []
    cuoi = h.get("frame_hien_thi_cuoi") or 0
    ok = (r.returncode == 4 and m.get("trang_thai") == "da_dung" and h.get("ly_do_dung_tu_cua_so") == "gia_lap_dong_cua_so"
          and cuoi >= 5 and d.get("n_khong_hien_thi_cua_so_dong", 0) >= 1 and all(k > cuoi for k in khong)
          and n > cuoi and d.get("n_ghep_duoc") == d.get("n_hien_thi", 0) + d.get("n_bo_hien_thi", 0) + d.get("n_khong_hien_thi_cua_so_dong", 0)
          and (h.get("fps_hien_thi") is None) == (d.get("n_hien_thi", 0) < 2)
          and mot(cs) == b"\n".join(tien_to) + b"\n" and lop(cs).split(b"\n")[1:-1] == tien_to_lop)
    check("11. N25-F1 dau-cuoi: dong cua so (gia lap) sau frame 5 -> ma 4; frame toi sau khi dong KHONG tinh la da hien thi; "
          "MOT + lop moi frame tracker da xu ly == tien to luot du", ok,
          f"exit {r.returncode} {m.get('trang_thai')}; xu ly {n}; hien thi {d.get('n_hien_thi')} (cuoi {cuoi}), khong hien "
          f"{d.get('n_khong_hien_thi_cua_so_dong')} {khong[:6]}; fps {h.get('fps_hien_thi')}")


# ------------------------------------------------------------------------------------------------ 12 (N25-F2)
class _Bien:
    """StringVar gia (khong Tk); ghi lai lan `set` SAU khi launcher da dong (= cham widget da huy)."""

    def __init__(self, v, huy):
        self.v, self._huy, self.set_sau_khi_huy = v, huy, 0

    def get(self):
        return self.v

    def set(self, v):
        if self._huy:
            self.set_sau_khi_huy += 1
        self.v = v


class _Lich:
    """after/after_cancel gia lap: chay() goi cac callback dang cho (nhu vong su kien Tk)."""

    def __init__(self):
        self.cho, self.n = {}, 0

    def after(self, ms, fn):
        self.n += 1
        self.cho[self.n] = fn
        return self.n

    def after_cancel(self, ma):
        self.cho.pop(ma, None)

    def chay(self):
        for ma in sorted(self.cho):
            fn = self.cho.pop(ma, None)
            if fn is not None:
                fn()


def _tao_zenity_gia(goc: Path) -> tuple[Path, Path, Path]:
    """zenity GIA: /bin/sh exec Python (cung PID) -> ghi PID; che do doc tu file: cho | huy | chon:<duong dan>."""
    import shlex
    goc.mkdir(parents=True, exist_ok=True)
    pid_f, che_do = goc / "picker.pid", goc / "che_do.txt"
    (goc / "picker.py").write_text(
        "import os, sys, time\n"
        f"open({str(pid_f)!r}, 'a').write(str(os.getpid()) + '\\n')\n"
        f"m = open({str(che_do)!r}).read().strip()\n"
        "if m == 'huy':\n    sys.exit(1)\n"
        "if m.startswith('chon:'):\n    print(m[5:]); sys.exit(0)\n"
        "time.sleep(120)\n", encoding="utf-8")
    z = goc / "zenity"
    z.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(goc / "picker.py")) + ' "$@"\n')
    z.chmod(0o755)
    return z, pid_f, che_do


def _app_gia(nl, ctl, out_base: str):
    """NanoLauncherApp KHONG Tk (nhu harness cua Codex) + after/after_cancel gia lap chay duoc."""
    from types import SimpleNamespace
    App = nl._build_app()
    app = object.__new__(App)
    lich, huy = _Lich(), []
    app.after, app.after_cancel = lich.after, lich.after_cancel
    app.v = {"source": _Bien("", huy), "out_base": _Bien(out_base, huy)}
    app._chon_dang_mo = False
    app._dang_dong = False
    app.cau_hinh = {"thu_muc_du_lieu": out_base}
    app.ctl = ctl
    w = SimpleNamespace(configure=lambda **k: None)
    app.bt_start = app.bt_stop = app.lb_tt = w
    app.destroy = lambda: huy.append(True)
    return app, lich, huy


def t12_f2(tmp: Path, anh: Path, cache: Path):
    if os.name != "posix":
        skip("12. N25-F2 vong doi hop chon file", "can POSIX (zenity gia bang /bin/sh; tren Nano chay that)")
        return
    import nano_launcher as nl
    from types import SimpleNamespace
    goc_ra = shutil.which
    z, pid_f, che_do = _tao_zenity_gia(tmp / "zenity_gia")
    nl.shutil.which = lambda ten: str(z) if ten == "zenity" else goc_ra(ten)
    pids = lambda: [int(x) for x in pid_f.read_text().split()] if pid_f.exists() else []
    try:
        # (a) mo hop chon roi dong launcher khi CHUA chay tracker
        che_do.write_text("cho")
        ctl0 = SimpleNamespace(dang_chay=lambda: False, _ghi_log=lambda s: None)
        app, lich, huy = _app_gia(nl, ctl0, str(tmp))
        app._chon("video")
        _cho(lambda: len(pids()) == 1)
        p0 = pids()[0]
        con_truoc = pid_con_song(p0)
        app._dong()
        for _ in range(60):                       # nhu vong su kien: chay callback toi khi xong
            if huy:
                break
            lich.chay()
            time.sleep(0.05)
        lich.chay()                                # callback con sot (neu co) chay sau khi dong: khong duoc cham widget
        ok_a = (con_truoc and bool(huy) and not pid_con_song(p0) and not lich.cho and app.v["source"].set_sau_khi_huy == 0
                and app._picker is None)
        check("12a. N25-F2: mo hop chon -> dong launcher (chua chay tracker): tien trinh chon file da thoat + thu hoi (theo PID), "
              "callback cho da huy, khong cham widget sau khi dong", ok_a,
              f"pid {p0} song truoc {con_truoc} sau {pid_con_song(p0)}; destroy {bool(huy)}; callback con {len(lich.cho)}; "
              f"set sau huy {app.v['source'].set_sau_khi_huy}")
        # (b) huy hop chon roi mo lai va chon nguon thanh cong
        che_do.write_text("huy")
        app, lich, huy = _app_gia(nl, ctl0, str(tmp))
        app._chon("video")
        _cho(lambda: len(pids()) == 2)
        _cho(lambda: (lich.chay() or True) and not app._chon_dang_mo, 5)
        sau_huy = app.v["source"].get()
        che_do.write_text(f"chon:{anh}")
        app._chon("thu_muc_anh")
        _cho(lambda: len(pids()) == 3)
        _cho(lambda: (lich.chay() or True) and not app._chon_dang_mo, 5)
        ok_b = sau_huy == "" and app.v["source"].get() == str(anh) and not any(pid_con_song(p) for p in pids()[1:3])
        check("12b. N25-F2: huy hop chon (ma 1) -> nguon khong doi; mo lai + chon -> nguon = duong dan da chon; ca hai tien trinh da thu hoi",
              ok_b, f"sau huy {sau_huy!r}; sau chon {app.v['source'].get()!r}")
        app._dong(); lich.chay()
        # (c) hop chon vua tu thoat dung luc dong launcher -> chi thu hoi, bo ket qua
        che_do.write_text(f"chon:{tmp / 'KHONG_DUNG'}")
        app, lich, huy = _app_gia(nl, ctl0, str(tmp))
        app._chon("video")
        _cho(lambda: len(pids()) == 4)
        p3 = pids()[3]
        _cho(lambda: app._picker is not None and app._picker.poll() is not None)   # da thoat, CHUA ai thu hoi / doc
        app._dong(); lich.chay(); lich.chay()
        ok_c = bool(huy) and not pid_con_song(p3) and app.v["source"].get() == "" and app.v["source"].set_sau_khi_huy == 0
        check("12c. N25-F2: hop chon vua tu thoat dung luc dong -> thu hoi, BO ket qua (khong cap nhat widget), khong loi",
              ok_c, f"destroy {bool(huy)} pid {p3} song {pid_con_song(p3)} nguon {app.v['source'].get()!r}")
        # (d) hop chon dang mo + tracker dang chay -> dong launcher
        che_do.write_text("cho")
        goi = tmp / "cfg_f2.json"
        fake = tmp / "gia_f2.engine"
        fake.write_bytes(b"x")
        goi.write_text(json.dumps({"engine": [str(fake)], "python_detector": PY, "ld_preload": ""}), encoding="utf-8")
        c = nl.doc_cau_hinh(goi, tmp / "khong_co_nd.json")
        gl = ["--gia-lap-cache", str(cache), "--gia-lap-video", "seq", "--video-name", "seq", "--timeout-con-s", "60",
              "--gia-lap-cham-ms", "150"] + (["--python-detector", tns.PY_DET] if tns.PY_DET else [])
        ctl = nl.NanoLauncherController(c, nl.kiem_trien_khai(c), timeout_dung=30, argv_them=gl)
        rd = ctl.start({"source": str(anh), "fps": str(tns.FPS), "tracker": "cv", "out_base": str(tmp / "kq_f2")})
        _cho(lambda: ctl.poll()["trang_thai"] == "DANG_CHAY", 60)
        app, lich, huy = _app_gia(nl, ctl, str(tmp))
        app._chon("video")
        _cho(lambda: len(pids()) == 5)
        p4 = pids()[4]
        worker_pid = ctl.proc.pid
        det_pid = int((rd / "detector.pid").read_text()) if (rd / "detector.pid").exists() else None
        t0 = time.time()
        app._dong()
        picker_chet_luc = None
        while not huy and time.time() - t0 < 60:
            lich.chay()
            ctl.poll()
            if picker_chet_luc is None and not pid_con_song(p4):
                picker_chet_luc = time.time() - t0
            time.sleep(0.05)
        lich.chay()
        m = tns.meta(rd)
        ok_d = (bool(huy) and ctl.trang_thai == "DA_DUNG" and m.get("trang_thai") == "da_dung" and not pid_con_song(p4)
                and not pid_con_song(worker_pid) and (det_pid is None or not pid_con_song(det_pid)) and not lich.cho
                and picker_chet_luc is not None and picker_chet_luc < 1.0)
        check("12d. N25-F2: hop chon dang mo + tracker dang chay -> dong launcher: hop chon thoat ngay, tracker DA_DUNG (stop-file), "
              "worker + detector da thoat (theo PID), khong callback con lai", ok_d,
              f"destroy {bool(huy)} tracker {ctl.trang_thai} meta {m.get('trang_thai')}; picker thoat sau {picker_chet_luc}; "
              f"worker {worker_pid} song {pid_con_song(worker_pid)} detector {det_pid} song "
              f"{None if det_pid is None else pid_con_song(det_pid)}; callback con {len(lich.cho)}")
        # (e) cua so Tk THAT (neu co man hinh): mo hop chon -> dong -> khong loi callback sau destroy
        try:
            import tkinter as tk
            tk.Tk().destroy()
        except Exception as e:  # noqa: BLE001
            skip("12e. N25-F2 tren cua so Tk that", f"khong co man hinh ({type(e).__name__})")
            return
        che_do.write_text("cho")
        goi2 = tmp / "cfg_f2e.json"
        goi2.write_text(json.dumps({"engine": [str(tmp / "khong_co.engine")], "python_detector": PY, "ld_preload": "",
                                    "thu_muc_du_lieu": str(tmp)}), encoding="utf-8")
        c2 = nl.doc_cau_hinh(goi2, tmp / "khong_co_nd.json")
        app = nl._build_app()(c2, nl.kiem_trien_khai(c2), {})
        loi_cb = []
        app.report_callback_exception = lambda *e: loi_cb.append(repr(e[1]))
        app.update()
        app._chon("video")
        _cho(lambda: (app.update() or True) and len(pids()) == 6, 10)
        p5 = pids()[5]
        app._dong()
        con_cua_so = True
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                app.update()
            except tk.TclError:
                con_cua_so = False
                break
            time.sleep(0.05)
        check("12e. N25-F2 cua so Tk THAT: mo hop chon -> dong launcher -> cua so da huy, hop chon da thoat, 0 loi callback",
              not con_cua_so and not pid_con_song(p5) and not loi_cb,
              f"cua so con {con_cua_so}; pid {p5} song {pid_con_song(p5)}; loi callback {loi_cb}")
    finally:
        nl.shutil.which = goc_ra
        for p in pids():
            if pid_con_song(p):
                try:
                    os.kill(p, 9)
                except OSError:
                    pass


# ------------------------------------------------------------------------------------------------ 9
def t9_gui(tmp: Path):
    try:
        import tkinter as tk
        r = tk.Tk()
        r.destroy()
    except Exception as e:  # noqa: BLE001
        skip("9. giao dien Tk: thieu engine -> nut Bat dau khoa", f"khong mo duoc Tk ({type(e).__name__})")
        return
    import nano_launcher as nl
    goi = tmp / "cfg_gui.json"
    goi.write_text(json.dumps({"engine": [str(tmp / "khong_co.engine")], "python_detector": PY, "ld_preload": "",
                               "thu_muc_ket_qua": str(tmp / "kq_gui")}), encoding="utf-8")
    c = nl.doc_cau_hinh(goi, tmp / "khong_co_nd.json")
    tk_ = nl.kiem_trien_khai(c)
    app = nl._build_app()(c, tk_, {})
    for _ in range(6):
        app.update()
        time.sleep(0.05)
    trang_thai_nut = str(app.bt_start.cget("state"))
    chu = app.lb_mt.cget("text")
    app._dong()
    try:
        app.update()
        con_song = bool(app.winfo_exists())
    except tk.TclError:
        con_song = False
    check("9. giao dien Tk: thieu engine -> thong bao cach khac phuc, nut Bat dau bi khoa; dong cua so sach",
          trang_thai_nut == "disabled" and "engine" in chu and not con_song, f"nut {trang_thai_nut}; {chu[:100]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-detector", default=None, help="interpreter cua tien trinh con (Nano: /usr/bin/python3)")
    a = ap.parse_args()
    tns.PY_DET = a.python_detector
    print(f"cha: {sys.version.split()[0]} | con: {tns.PY_DET or PY} | backend np")
    tmp = Path(tempfile.mkdtemp(prefix="nano_hien_thi_"))
    try:
        t1_giao_thuc()
        anh = tns.tao_anh(tmp / "anh")
        cache = tns.tao_cache_gia_lap(tmp / "cache_gia_lap")
        t2_hang_bang_mot(tmp, cache)
        t3_4_bat_tat(tmp, anh, cache)
        t5_phim_q(tmp, anh, cache)
        t6_frame_hong(tmp, cache)
        t7_khong_man_hinh(tmp, anh, cache)
        t8_launcher(tmp, anh, cache)
        t9_gui(tmp)
        t10_f1_don_vi(tmp)
        t11_f1_dau_cuoi(tmp, anh, cache)
        t12_f2(tmp, anh, cache)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, s in _R if s == "FAIL")
    n_pass = sum(1 for _, s in _R if s == "PASS")
    n_skip = sum(1 for _, s in _R if s == "SKIP")
    print(f"\nKET QUA: {n_pass}/{n_pass + n_fail} phep kiem tra PASS" + (f", {n_skip} SKIP" if n_skip else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    os.environ.setdefault("BYTETRACK_BACKEND", "np")
    raise SystemExit(main())
