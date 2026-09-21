"""
test_pc_launcher.py -- Launcher PC (`src/pc_launcher.py`): dieu phoi worker `track_video_file.py`.

PHAM VI GIA LAP (ghi ro): worker chay trong SUBPROCESS THAT qua mot wrapper chi thay detector
(`det_cache.detect_frame`), bo doc anh (`cv2.imread`, co do tre) va GUI cv2. Entry point
`track_video_file.main()`, vong lap `track_video`, progress-file/stop-file, ghi ket qua/cache: MA THAT.
Khong co Tk trong test (LauncherController tach khoi widget). Thao tac nut tren cua so that: xem smoke.

Cac phep kiem:
  1. build_argv: truyen DUNG nguon/checkpoint/YAML/tham so hieu luc; duong dan co DAU CACH giu nguyen
     mot phan tu; moi luot mot thu muc rieng.
  2. Dau vao khong hop le -> CauHinhSai TRUOC khi khoi chay (khong co tien trinh nao).
  3. Bat dau -> hoan tat: trang thai HOAN_TAT, progress.json `hoan_tat`, du data/timing/pipeline/run_meta.
  4. Bat dau -> dung giua chung (stop-file): trang thai DA_DUNG, run_meta dung_som + ly_do 'stop-file',
     so dong tracking/timing == so frame da xu ly, manifest da_xu_ly < nguon, cache du -> replay tuong duong.
  5. Khong khoi chay tac vu thu hai khi tac vu truoc con chay (DangChay).
  6. Worker tra loi (cau hinh khong tuong thich -> exit 2): trang thai LOI + thong bao tu progress.json,
     KHONG bao hoan tat.
  7. shutdown() khi dang chay: yeu cau dung co kiem soat, worker ket thuc, khong con tien trinh.
  8. Dong lai khi worker KHONG phan hoi (wrapper bo qua stop-file): terminate la duong cuoi, co ghi log,
     trang thai LOI (khong bao hoan tat).
Regression review 2d1a8df (L-F1..L-F4):
  9. (L-F1) Dang TAM DUNG bang SPACE -> Dung (stop-file): worker tu ket thuc KHONG terminate, DA_DUNG,
     ly_do_dung='stop-file', KHONG doc them frame, timing frame cuoi huu han, manifest = frame da xu ly,
     cache -> replay parity. 9b: SPACE -> shutdown() cung vay. 9c: phim q trong luc pause van la 'phim q'.
 10. (L-F3) Tre 1 s SAU vong lap (quanh save_video): fps_vong_lap_tam cuoi trong progress.json == run_meta
     .thoi_gian.fps_vong_lap (cung pham vi, khong phu thuoc do tre); fps_toan_luot ghi RIENG.
 11. (L-F2, Tk THAT cua so an) chuyen indeterminate -> determinate: worker giu nguyen frame, bom event loop
     -> thanh dung yen; frame tang -> thanh cap nhat dung.
 12. (L-F4, Tk THAT) nhap 'abc' vao max_frames / seed / conf_min -> hop thoai ghi TEN O, khong worker, khong
     exception thoat callback; sua hop le -> khoi chay duoc.
 13. (L-F1, Tk THAT) dang SPACE -> dong cua so (_dong): cho bang after(), GUI van phan hoi, chi destroy
     sau khi worker ket thuc; DA_DUNG, khong terminate, khong con tien trinh.
 11-13 can display (Tk). Khong co display -> in "BO QUA" va KHONG tinh la PASS.

    python src/test_pc_launcher.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Thong bao L-F4 co tieng Viet co dau; runner co the bat stdout cp1252 -> in an toan voi moi encoding
for _luong in (sys.stdout, sys.stderr):
    try:
        _luong.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
_RESULTS: list[tuple[str, bool]] = []
ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


WRAPPER = textwrap.dedent('''
    # worker gia lap: CHI thay detector / imread / GUI cv2; entry point + dieu phoi la MA THAT
    import os, sys, time, types
    import numpy as np
    sys.path.insert(0, r"{src}")
    import cv2, det_cache, ultralytics
    ultralytics.YOLO = lambda w: types.SimpleNamespace(names={{2: "car"}})
    det_cache.detect_frame = lambda m, i, s: np.array([[10, 10, 50, 50, 0.9, 2]], np.float32)
    DELAY = float(os.environ.get("GIA_DELAY_MS", "30")) / 1000.0
    def _imread(p, *a, **k):
        time.sleep(DELAY); return np.zeros((240, 320, 3), np.uint8)
    cv2.imread = _imread
    cv2.namedWindow = lambda *a, **k: None; cv2.imshow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    # GIA_PAUSE_MARKER=<file>: gia lap bam SPACE o frame 2 (waitKey(1) lan thu 2); trong vong pause
    # (waitKey(50)) ghi marker roi tra 255 mai (khong ai bam gi) -- hoac tra 'q' neu GIA_PAUSE_Q=1
    ST = {{"n1": 0, "n50": 0, "marker": os.environ.get("GIA_PAUSE_MARKER")}}
    def _waitKey(ms=1):
        if ms == 1:
            ST["n1"] += 1
            return ord(" ") if (ST["marker"] and ST["n1"] == 2) else 255
        ST["n50"] += 1
        if ST["marker"] and not os.path.exists(ST["marker"]):
            open(ST["marker"], "w").write("in pause loop")
        if os.environ.get("GIA_PAUSE_Q") == "1" and ST["n50"] >= 3:
            return ord("q")
        return 255
    cv2.waitKey = _waitKey
    if os.environ.get("GIA_SAVE_DELAY_S"):
        # tre CHI o buoc ghi cache SAU vong lap (L-F3)
        _sv = det_cache.save_video
        def _slow(*a, **k):
            time.sleep(float(os.environ["GIA_SAVE_DELAY_S"])); return _sv(*a, **k)
        det_cache.save_video = _slow
    if os.environ.get("GIA_BO_QUA_STOP") == "1":
        # mo phong worker KHONG phan hoi yeu cau dung: xoa --stop-file khoi argv
        i = sys.argv.index("--stop-file"); del sys.argv[i:i + 2]
    import track_video_file
    sys.exit(track_video_file.main())
''')


def _cho(ctl, dk, timeout=60.0, buoc=0.05):
    """Cho den khi dk(snapshot) dung. Sau khi thoa, POLL LAI mot lan roi moi tra ve: dieu kien kieu
    `not ctl.dang_chay()` duoc danh gia SAU poll(), worker co the thoat giua hai lenh -> snapshot cu
    (DANG_DUNG) chua phan anh trang thai cuoi (race lo duoi tai full suite, lich_su_that_bai/ vong 13)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = ctl.poll()
        if dk(s):
            return ctl.poll()
        time.sleep(buoc)
    return ctl.poll()


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="pcl_"))
    try:
        import pc_launcher as L

        wrapper = tmp / "worker gia.py"                       # DAU CACH trong duong dan wrapper
        wrapper.write_text(WRAPPER.format(src=str(ROOT / "src")), encoding="utf-8")
        ck = tmp / "ck dir" / "gia.pt"
        ck.parent.mkdir(); ck.write_bytes(b"fake")
        anh = tmp / "traffic clip"; anh.mkdir()
        for i in range(1, 41):
            (anh / f"{i:03d}.jpg").write_bytes(b"x")
        yaml_sp = tmp / "cfg dir" / "ekf ctrv.yaml"
        yaml_sp.parent.mkdir(); shutil.copy(ROOT / "configs" / "bytetrack_ekf_ctrv.yaml", yaml_sp)
        out_base = tmp / "ket qua"

        def cfg(**k):
            c = {"source": str(anh), "fps": "5", "model": str(ck), "device": "cpu", "tracker_yaml": str(yaml_sp),
                 "out_base": str(out_base), "warmup_frames": "0", "classes": "2"}
            c.update(k); return c

        prefix = [sys.executable, str(wrapper)]

        # --- 1. build_argv ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        argv, rd = ctl.build_argv(cfg(conf_min="0.1", max_det="50", track_buffer_sec="0.8", seed="3",
                                     set={"match_thresh": "0.7"}, save_cache=True, show=True, save_video=True))
        def sau(flag):
            return argv[argv.index(flag) + 1]
        _check("1. build_argv truyen DUNG nguon/checkpoint/YAML (co dau cach, MOT phan tu) + tham so hieu luc "
               "(conf-min, max-det, track-buffer, seed, --set), cac co, progress/stop-file trong thu muc luot",
               sau("--source") == str(anh) and sau("--model") == str(ck) and sau("--tracker-yaml") == str(yaml_sp)
               and sau("--conf-min") == "0.1" and sau("--max-det") == "50" and sau("--track-buffer-sec") == "0.8"
               and sau("--seed") == "3" and "match_thresh=0.7" in argv and "--show" in argv and "--save-video" in argv
               and "--save-cache" in argv and sau("--fps") == "5" and argv[argv.index("--classes") + 1] == "2"
               and Path(sau("--progress-file")).parent == rd and Path(sau("--stop-file")).parent == rd
               and rd.parent == out_base and "traffic clip" in rd.name,
               f"run_dir={rd.name}")
        _, rd2 = ctl.build_argv(cfg())
        _check("   moi luot mot thu muc rieng (khong ghi de)", rd2 != rd or not rd.exists(), f"{rd.name} / {rd2.name}")

        # --- 2. dau vao khong hop le -> CauHinhSai truoc khi khoi chay ---
        loi = {}
        for ten, c in (("nguon khong ton tai", cfg(source=str(tmp / "khong_co"))),
                       ("thu muc anh thieu fps", cfg(fps="")),
                       ("checkpoint khong ton tai", cfg(model=str(tmp / "khong.pt"))),
                       ("yaml khong ton tai", cfg(tracker_yaml=str(tmp / "x" / "y.yaml"))),
                       ("fps am", cfg(fps="-1"))):
            try:
                ctl.build_argv(c); loi[ten] = None
            except L.CauHinhSai:
                loi[ten] = "CauHinhSai"
        _check("2. dau vao khong hop le -> CauHinhSai TRUOC khi khoi chay, khong co tien trinh",
               all(v == "CauHinhSai" for v in loi.values()) and ctl.proc is None, f"{loi}")

        # --- 3. bat dau -> hoan tat ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        rd = ctl.start(cfg(max_frames=8, save_cache=True))
        s = _cho(ctl, lambda s: s["trang_thai"] in ("HOAN_TAT", "DA_DUNG", "LOI"))
        pr = json.loads((rd / "progress.json").read_text(encoding="utf-8"))
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        _check("3. bat dau -> HOAN_TAT: exit 0, progress.json 'hoan_tat' frame 8/40, run_meta/data/timing/pipeline "
               "co du, launcher_argv.json ghi lai, log co dong worker",
               s["trang_thai"] == "HOAN_TAT" and s["exit_code"] == 0 and pr["trang_thai"] == "hoan_tat"
               and pr["frame"] == 8 and pr["n_frames_khai_bao"] == 40 and m["ket_qua"]["n_frame_xu_ly"] == 8
               and (rd / "data" / "traffic clip.txt").exists() and (rd / "timing" / "traffic clip.csv").exists()
               and (rd / "launcher_argv.json").exists() and any("[track_video]" in ln for ln in ctl.log)
               and not ctl.dang_chay(),
               f"tt={s['trang_thai']} exit={s['exit_code']} pr={pr.get('trang_thai')} loi={s['loi']}")
        cid3 = next(a for i, a in enumerate(ctl.argv) if ctl.argv[i - 1] == "--save-cache")

        # --- 4. bat dau -> dung giua chung ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        rd = ctl.start(cfg(save_cache=True))               # 40 frame x ~30 ms
        _cho(ctl, lambda s: s["trang_thai"] == "DANG_CHAY" and s["frame"] >= 5, timeout=30)
        ok_req = ctl.request_stop()
        s = _cho(ctl, lambda s: s["trang_thai"] in ("DA_DUNG", "HOAN_TAT", "LOI"), timeout=30)
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        fm = json.loads((rd / "frames_manifest.json").read_text(encoding="utf-8"))
        import pandas as pd
        n_xl = m["ket_qua"]["n_frame_xu_ly"]
        n_tim = len(pd.read_csv(rd / "timing" / "traffic clip.csv"))
        n_pipe = pd.read_csv(rd / "pipeline" / "traffic clip.csv")
        cid = next(a for i, a in enumerate(ctl.argv) if ctl.argv[i - 1] == "--save-cache")
        # replay tu cache bang replay_argv (subprocess that)
        argv_r = [sys.executable if x == "python" else ("PCL_STOP" if x == "<split>" else ("cv" if x == "<ten>" else x))
                  for x in m["replay_argv"]]
        argv_r[1] = str(ROOT / "src" / "run_tracking.py")
        shutil.rmtree(ROOT / "data/processed/trackers/PCL_STOP", ignore_errors=True)
        r = subprocess.run(argv_r, cwd=ROOT, capture_output=True, text=True, shell=False)
        import nano_parity as npar
        pr_ = npar.so_mot_video(rd / "data" / "traffic clip.txt", ROOT / "data/processed/trackers/PCL_STOP/cv/data/traffic clip.txt")
        _check("4. bat dau -> Dung (stop-file) -> DA_DUNG: run_meta dung_som + ly_do 'stop-file', 5 <= frame xu ly < 40, "
               "so dong timing == frame xu ly, pipeline moi dong huu han, manifest nguon 40 / da_xu_ly = frame xu ly, "
               "cache du -> replay (subprocess) parity DAT, khong con tien trinh",
               ok_req and s["trang_thai"] == "DA_DUNG" and m["ket_qua"]["dung_som"] is True
               and m["ket_qua"]["ly_do_dung"] == "stop-file" and 5 <= n_xl < 40 and n_tim == n_xl
               and n_pipe.frame_total_ms.notna().all() and fm["n_nguon"] == 40 and fm["n_da_xu_ly"] == n_xl
               and r.returncode == 0 and pr_["dat"] and pr_["n_ref"] > 0 and not ctl.dang_chay(),
               f"tt={s['trang_thai']} n_xl={n_xl} n_tim={n_tim} manifest={fm.get('n_nguon')}/{fm.get('n_da_xu_ly')} "
               f"replay={r.returncode} parity={pr_.get('ly_do','DAT')}")
        shutil.rmtree(ROOT / "data/processed/trackers/PCL_STOP", ignore_errors=True)

        # --- 5. khong khoi chay tac vu thu hai ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        rd = ctl.start(cfg())
        try:
            ctl.start(cfg()); hai = "khong bat"
        except L.DangChay:
            hai = "DangChay"
        ctl.request_stop()
        s = _cho(ctl, lambda s: not ctl.dang_chay(), timeout=30)
        _check("5. bat dau lan hai khi dang chay -> DangChay, tac vu dau van chay va dung duoc binh thuong",
               hai == "DangChay" and s["trang_thai"] == "DA_DUNG", f"{hai} tt={s['trang_thai']}")

        # --- 6. worker tra loi ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        rd = ctl.start(cfg(det_conf="0.5", conf_min="0.1"))       # check_compatible -> exit 2
        s = _cho(ctl, lambda s: not ctl.dang_chay(), timeout=30)
        _check("6. worker loi (cau hinh khong tuong thich, exit 2) -> trang thai LOI + thong bao tu progress.json; "
               "KHONG bao hoan tat",
               s["trang_thai"] == "LOI" and s["exit_code"] == 2 and s["loi"] and "khong tuong thich" in s["loi"]
               and not (rd / "run_meta.json").exists(), f"tt={s['trang_thai']} exit={s['exit_code']} loi={s['loi']}")

        # --- 7. shutdown khi dang chay ---
        ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
        rd = ctl.start(cfg())
        _cho(ctl, lambda s: s["trang_thai"] == "DANG_CHAY" and s["frame"] >= 3, timeout=30)
        tt = ctl.shutdown(cho_giay=20)
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        _check("7. shutdown() khi dang chay -> yeu cau dung co kiem soat, worker tu ket thuc (DA_DUNG, run_meta "
               "dung_som), KHONG terminate, khong con tien trinh",
               tt == "DA_DUNG" and not ctl.da_terminate and m["ket_qua"]["dung_som"] is True and not ctl.dang_chay(),
               f"tt={tt} terminate={ctl.da_terminate}")

        # --- 8. worker KHONG phan hoi -> terminate la duong cuoi, co log ---
        os.environ["GIA_BO_QUA_STOP"] = "1"; os.environ["GIA_DELAY_MS"] = "200"
        try:
            ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=2)
            rd = ctl.start(cfg())
            _cho(ctl, lambda s: s["trang_thai"] == "DANG_CHAY", timeout=30)
            t0 = time.perf_counter()
            tt = ctl.shutdown(cho_giay=2)
            mat = time.perf_counter() - t0
        finally:
            os.environ.pop("GIA_BO_QUA_STOP", None); os.environ.pop("GIA_DELAY_MS", None)
        _check("8. worker bo qua yeu cau dung -> sau timeout launcher TERMINATE (duong cuoi), ghi ro vao log, "
               "trang thai LOI (khong bao hoan tat), khong con tien trinh",
               ctl.da_terminate and tt == "LOI" and any("TERMINATE" in ln for ln in ctl.log)
               and not ctl.dang_chay() and mat < 15, f"tt={tt} terminate={ctl.da_terminate} mat={mat:.1f}s")

        # --- open_results khi chua co ---
        _check("   open_results() khong co run_dir -> False (khong mo gi)", L.LauncherController().open_results() is False)

        # ================= regression review 2d1a8df =================
        def _cho_marker(mk, ctl, timeout=40):
            t0 = time.perf_counter()
            while not mk.exists() and time.perf_counter() - t0 < timeout:
                ctl.poll(); time.sleep(0.05)
            return mk.exists()

        def _replay_parity(rd, cid_tag):
            m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
            argv_r = [sys.executable if x == "python" else (cid_tag if x == "<split>" else ("cv" if x == "<ten>" else x))
                      for x in m["replay_argv"]]
            argv_r[1] = str(ROOT / "src" / "run_tracking.py")
            shutil.rmtree(ROOT / "data/processed/trackers" / cid_tag, ignore_errors=True)
            r = subprocess.run(argv_r, cwd=ROOT, capture_output=True, text=True, shell=False)
            import nano_parity as npar
            pr_ = npar.so_mot_video(rd / "data" / "traffic clip.txt",
                                    ROOT / "data/processed/trackers" / cid_tag / "cv/data/traffic clip.txt")
            shutil.rmtree(ROOT / "data/processed/trackers" / cid_tag, ignore_errors=True)
            return r.returncode, pr_

        # --- 9. (L-F1) dang SPACE -> Dung ---
        import pandas as pd
        mk = tmp / "pause9.marker"
        os.environ["GIA_PAUSE_MARKER"] = str(mk)
        try:
            ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=3)     # timeout ngan CHI de test nhanh
            rd = ctl.start(cfg(show=True, save_cache=True))
            vao_pause = _cho_marker(mk, ctl)
            t0 = time.perf_counter(); ok_req = ctl.request_stop()
            s = _cho(ctl, lambda s: not ctl.dang_chay(), timeout=30); s = ctl.poll()
            mat = time.perf_counter() - t0
        finally:
            os.environ.pop("GIA_PAUSE_MARKER", None)
        co_meta = (rd / "run_meta.json").exists()
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8")) if co_meta else {}
        fm = json.loads((rd / "frames_manifest.json").read_text(encoding="utf-8")) if co_meta else {}
        df_t = pd.read_csv(rd / "timing" / "traffic clip.csv") if co_meta else None
        df_pp = pd.read_csv(rd / "pipeline" / "traffic clip.csv") if co_meta else None
        rc, pr_ = _replay_parity(rd, "PCL_PAUSE") if co_meta else (None, {"dat": False, "n_ref": 0})
        _check("9. (L-F1) dang TAM DUNG (SPACE o frame 2) -> Dung: worker TU ket thuc (exit 0, KHONG terminate) "
               "truoc timeout, DA_DUNG, ly_do_dung='stop-file', KHONG doc them frame (n_frame_xu_ly == 2), timing/"
               "pipeline frame cuoi huu han, manifest da_xu_ly == 2, cache -> replay parity DAT",
               vao_pause and ok_req and s["trang_thai"] == "DA_DUNG" and s["exit_code"] == 0 and not ctl.da_terminate
               and mat < 3 and co_meta and m["ket_qua"]["dung_som"] is True and m["ket_qua"]["ly_do_dung"] == "stop-file"
               and m["ket_qua"]["n_frame_xu_ly"] == 2 and len(df_t) == 2 and df_pp.frame_total_ms.notna().all()
               and fm["n_da_xu_ly"] == 2 and rc == 0 and pr_["dat"] and pr_["n_ref"] > 0 and not ctl.dang_chay(),
               f"pause={vao_pause} tt={s['trang_thai']} exit={s['exit_code']} terminate={ctl.da_terminate} mat={mat:.2f}s "
               f"meta={co_meta} n_xl={m.get('ket_qua', {}).get('n_frame_xu_ly')} replay={rc} parity={pr_.get('ly_do', 'DAT')}")

        # --- 9b. dang SPACE -> shutdown() ---
        mk = tmp / "pause9b.marker"
        os.environ["GIA_PAUSE_MARKER"] = str(mk)
        try:
            ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=3)
            rd = ctl.start(cfg(show=True))
            vao_pause = _cho_marker(mk, ctl)
            tt = ctl.shutdown(cho_giay=3)
        finally:
            os.environ.pop("GIA_PAUSE_MARKER", None)
        _check("9b. (L-F1) dang TAM DUNG -> shutdown(): DA_DUNG, KHONG terminate, run_meta co, khong con tien trinh",
               vao_pause and tt == "DA_DUNG" and not ctl.da_terminate and (rd / "run_meta.json").exists()
               and not ctl.dang_chay(), f"pause={vao_pause} tt={tt} terminate={ctl.da_terminate}")

        # --- 9c. phim q trong luc pause van la 'phim q' ---
        mk = tmp / "pause9c.marker"
        os.environ["GIA_PAUSE_MARKER"] = str(mk); os.environ["GIA_PAUSE_Q"] = "1"
        try:
            ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
            rd = ctl.start(cfg(show=True))
            s = _cho(ctl, lambda s: not ctl.dang_chay(), timeout=40); s = ctl.poll()
        finally:
            os.environ.pop("GIA_PAUSE_MARKER", None); os.environ.pop("GIA_PAUSE_Q", None)
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        _check("9c. phim q TRONG luc pause (khong co stop-file) -> DA_DUNG, ly_do_dung='phim q' (khong gan nham stop-file)",
               s["trang_thai"] == "DA_DUNG" and m["ket_qua"]["ly_do_dung"] == "phim q" and m["ket_qua"]["dung_som"] is True,
               f"tt={s['trang_thai']} ly_do={m['ket_qua']['ly_do_dung']}")

        # --- 10. (L-F3) tre sau vong lap khong doi fps vong lap cuoi ---
        os.environ["GIA_SAVE_DELAY_S"] = "1.0"
        try:
            ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
            rd = ctl.start(cfg(save_cache=True, max_frames=4))
            s = _cho(ctl, lambda s: not ctl.dang_chay(), timeout=60); s = ctl.poll()
        finally:
            os.environ.pop("GIA_SAVE_DELAY_S", None)
        m = json.loads((rd / "run_meta.json").read_text(encoding="utf-8"))
        pr = json.loads((rd / "progress.json").read_text(encoding="utf-8"))
        tg = m["thoi_gian"]
        _check("10. (L-F3) tre 1 s SAU vong lap (save_video): progress cuoi fps_vong_lap_tam == run_meta.fps_vong_lap "
               "(bang nhau CHINH XAC, cung pham vi), t_vong_lap_s == wall_vong_lap_s < 1 s, wall_ghi_s >= 1 s; "
               "fps_toan_luot ghi RIENG (< fps_vong_lap) kem pham_vi",
               s["trang_thai"] == "HOAN_TAT" and pr["fps_vong_lap_tam"] == tg["fps_vong_lap"]
               and pr["t_vong_lap_s"] == tg["wall_vong_lap_s"] and tg["wall_vong_lap_s"] < 1.0 and tg["wall_ghi_s"] >= 1.0
               and pr.get("wall_ghi_s") == tg["wall_ghi_s"] and (pr.get("fps_toan_luot") or 1e9) < pr["fps_vong_lap_tam"]
               and "pham_vi_fps_toan_luot" in pr and "KHONG gom" in pr["pham_vi_fps"],
               f"progress fps={pr['fps_vong_lap_tam']:.4f} run_meta fps={tg['fps_vong_lap']:.4f} "
               f"wall_vong_lap={tg['wall_vong_lap_s']:.3f}s wall_ghi={tg['wall_ghi_s']:.3f}s fps_toan_luot={pr.get('fps_toan_luot')}")

        # ================= 11-13: cua so Tk THAT (an) =================
        app = None
        try:
            App = L._build_app(); app = App(); app.withdraw()
        except Exception as e:  # noqa: BLE001
            print(f"  [BO QUA] 11-13 can Tk/display: {type(e).__name__}: {e} -- KHONG tinh la PASS")
        if app is not None:
            import tkinter.messagebox as mb
            hop = []
            mb.showerror = lambda t, m_: hop.append(("error", t, m_))
            mb.showwarning = lambda t, m_: hop.append(("warning", t, m_))
            mb.askyesno = lambda t, m_: True
            after_that = app.after

            # --- 11. (L-F2) thanh tien trinh ---
            class _FakeCtl:
                def __init__(self):
                    self.s = {"trang_thai": "DANG_KHOI_TAO", "frame": 0, "n_frames_khai_bao": 0, "fps_vong_lap_tam": None,
                              "tam_dung": False, "exit_code": None, "loi": None, "run_dir": None, "n_log": 0}
                    self.log = []; self.tien_trinh = {}
                def poll(self): return dict(self.s)
                def dang_chay(self): return True
            app.after = lambda *a, **k: None                 # _tick khong tu lap; goi tay
            fc = _FakeCtl(); ctl_that = app.ctl; app.ctl = fc
            app._tick(); app.update()
            mode0 = str(app.pb.cget("mode"))
            fc.s.update(trang_thai="DANG_CHAY", frame=10, n_frames_khai_bao=100)
            app._tick(); app.update()
            v0 = float(app.pb.cget("value")); mode1 = str(app.pb.cget("mode"))
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.5:            # bom event loop, worker VAN o frame 10
                app.update(); time.sleep(0.02)
            v1 = float(app.pb.cget("value"))
            fc.s.update(frame=37)
            app._tick(); app.update(); v2 = float(app.pb.cget("value")); mx = float(app.pb.cget("maximum"))
            app.ctl = ctl_that; app.after = after_that; app.after(200, app._tick)   # noi lai chuoi _tick that
            _check("11. (L-F2) khoi tao: indeterminate; -> DANG_CHAY 10/100: determinate value 10; bom event loop 0,5 s "
                   "voi worker giu nguyen frame -> thanh DUNG YEN (10); frame 37 -> value 37, maximum 100",
                   mode0 == "indeterminate" and mode1 == "determinate" and v0 == 10.0 and v1 == 10.0 and v2 == 37.0
                   and mx == 100.0, f"mode {mode0}->{mode1} value {v0}->{v1}->{v2} max={mx}")

            # --- 12. (L-F4) nhap sai -> hop thoai ghi ten o, khong worker, khong exception ---
            app.ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=10)
            app.v["source"].set(str(anh)); app.v["fps"].set("5"); app.v["model"].set(str(ck)); app.v["device"].set("cpu")
            app.v["out_base"].set(str(out_base)); app.v["warmup_frames"].set("0"); app.v["classes"].set("2")
            app.v["tracker"].set(list(L.TRACKERS)[-1]); app.v["yaml_khac"].set(str(yaml_sp))
            kq = {}
            for o, sai, ten_o in (("max_frames", "abc", "Số frame tối đa"), ("seed", "abc", "Seed"),
                                  ("conf_min", "x", "conf-min"), ("max_frames", "0", "Số frame tối đa"),
                                  ("track_buffer_sec", "-1", "track-buffer")):
                app.v["max_frames"].set(""); app.v["seed"].set(""); app.v["conf_min"].set(L.MAC_DINH["conf_min"])
                app.v["track_buffer_sec"].set(L.MAC_DINH["track_buffer_sec"])
                app.v[o].set(sai); hop.clear()
                try:
                    app._bat_dau(); exc = None
                except Exception as e:  # noqa: BLE001
                    exc = f"{type(e).__name__}: {e}"
                kq[(o, sai)] = (exc, len(hop), (hop[0][2] if hop else ""), app.ctl.proc is not None)
            sai_ok = all(exc is None and n == 1 and ten in msg and not co_proc
                         for (o, sai), (exc, n, msg, co_proc), ten in
                         zip(kq.keys(), kq.values(), ["Số frame tối đa", "Seed", "conf-min", "Số frame tối đa", "track-buffer"]))
            # sua lai hop le -> khoi chay duoc
            app.v["max_frames"].set("3"); app.v["seed"].set(""); app.v["conf_min"].set("0.25")
            app.v["track_buffer_sec"].set("1.2"); hop.clear()
            app._bat_dau()
            khoi_chay = app.ctl.proc is not None and app.ctl.trang_thai == "DANG_KHOI_TAO" and not hop
            s = _cho(app.ctl, lambda s: not app.ctl.dang_chay(), timeout=40); s = app.ctl.poll()
            _check("12. (L-F4) 'abc'/'x'/'0'/'-1' vao max_frames / seed / conf_min / track_buffer -> hop thoai ghi TEN O, "
                   "KHONG tao worker, KHONG exception thoat callback; sua hop le -> khoi chay binh thuong (HOAN_TAT 3 frame)",
                   sai_ok and khoi_chay and s["trang_thai"] == "HOAN_TAT" and s["frame"] == 3,
                   "\n".join(f"{k}: exc={v[0]} hop={v[1]} msg={v[2][:70]!r} proc={v[3]}" for k, v in kq.items())
                   + f"\nhop le: khoi_chay={khoi_chay} tt={s['trang_thai']} frame={s['frame']}")

            # --- 13. (L-F1) dang SPACE -> dong cua so (_dong) ---
            mk = tmp / "pause13.marker"
            os.environ["GIA_PAUSE_MARKER"] = str(mk)
            try:
                app.ctl = L.LauncherController(entry_prefix=prefix, timeout_dung=3)
                app.v["max_frames"].set(""); app.b["show"].set(True)
                app._bat_dau()
                ctl13 = app.ctl
                vao_pause = _cho_marker(mk, ctl13)
                t0 = time.perf_counter(); app._dong()            # = bam nut X cua cua so
                con_cua_so_ngay = True
                try:
                    con_cua_so_ngay = bool(app.winfo_exists())
                except Exception:  # noqa: BLE001
                    con_cua_so_ngay = False
                n_update, da_dong = 0, False
                while time.perf_counter() - t0 < 20:            # bom event loop nhu mainloop
                    try:
                        app.update(); n_update += 1
                        da_dong = not app.winfo_exists()        # update() KHONG raise sau destroy (Tk 8.6)
                    except Exception:  # noqa: BLE001  (TclError: application has been destroyed)
                        da_dong = True
                    if da_dong:
                        break
                    time.sleep(0.02)
                mat = time.perf_counter() - t0
                app = None
            finally:
                os.environ.pop("GIA_PAUSE_MARKER", None)
            _check("13. (L-F1) dang TAM DUNG -> dong cua so: _dong() tra ve NGAY (cua so con, cho bang after), event loop "
                   "van chay, worker tu ket thuc -> DA_DUNG, KHONG terminate, cua so dong sau khi worker ket thuc, "
                   "khong con tien trinh, run_meta co",
                   vao_pause and con_cua_so_ngay and n_update > 5 and da_dong and ctl13.trang_thai == "DA_DUNG"
                   and not ctl13.da_terminate and not ctl13.dang_chay() and (ctl13.run_dir / "run_meta.json").exists()
                   and mat < 10,
                   f"pause={vao_pause} con_ngay={con_cua_so_ngay} n_update={n_update} da_dong={da_dong} "
                   f"tt={ctl13.trang_thai} terminate={ctl13.da_terminate} mat={mat:.2f}s")
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        import traceback; traceback.print_exc()
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        # don cache launcher_* trong det_cache
        import det_cache
        for d in det_cache.CACHE_ROOT.glob("launcher_traffic clip_*"):
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
