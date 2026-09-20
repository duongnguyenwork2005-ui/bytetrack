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

    python src/test_pc_launcher.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    cv2.destroyAllWindows = lambda *a, **k: None; cv2.waitKey = lambda ms=1: 255
    if os.environ.get("GIA_BO_QUA_STOP") == "1":
        # mo phong worker KHONG phan hoi yeu cau dung: xoa --stop-file khoi argv
        i = sys.argv.index("--stop-file"); del sys.argv[i:i + 2]
    import track_video_file
    sys.exit(track_video_file.main())
''')


def _cho(ctl, dk, timeout=60.0, buoc=0.05):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        s = ctl.poll()
        if dk(s):
            return s
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
        import os
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
