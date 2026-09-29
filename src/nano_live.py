"""
nano_live.py -- LUONG TRUC TIEP tren Jetson Nano, xu ly TUNG frame:

    video / thu muc anh -> [con, Python 3.6 he thong] doc frame -> TensorRT FP16 -> hau xu ly NumPy
                        -> pipe (cua so credit K frame) ->
                        [tien trinh nay, Python >= 3.8, backend NumPy] loc detection -> tracker CV/EKF
                        -> MOT + timing + pipeline + metadata (+ det cache de REPLAY)

Hai tien trinh vi TensorRT cua JetPack 4.6 chi co binding Python 3.6 con tracker can >= 3.8. Giao thuc:
`src/nano_stream_proto.py`; tien trinh con: `src/nano_detect_stream.py`.

DUNG CHUNG MA voi duong cache (khong nhan ban):
  - Loc: `det_cache.filter_dets()` (conf > conf_min; lop; [:max_det]; class_map) -- y het `VideoDetections.frame()`.
  - Vong lap tracker: `run_tracking.track_video()` -- adapter `NanoStreamDetections` co CUNG interface voi
    `det_cache.VideoDetections` / `track_video_file.LiveVideoDetections` (`iter_frames()`, `orig_shape`).
  - Cau hinh hieu luc dung NHU `run_tracking.main`: `load_tracker_cfg(yaml, overrides)`, track_buffer tu
    `--track-buffer-sec` va fps, dt_nominal = 1/fps, `--elapsed-mode steps`.
  - Det cache (`--save-cache`) ghi bang `nano_detector.ghi_cache` (dinh dang replay duoc bang run_tracking.py),
    CHI gom cac frame DA vao tracker -> replay cung cau hinh phai cho parity.
Mac dinh = dieu kien c25 cua thi nghiem: --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --track-buffer-sec 1.2.
KHONG can GT. `--regions` (vung bo qua lay tu NHAN dataset) CHI cho che do doi chieu thi nghiem, ghi vao metadata.

MA THOAT (khong bao "hoan tat" cho ket qua thieu):
  0  hoan_tat          het nguon, so frame == khai bao (hoac --max-frames), con thoat 0
  2  loi truoc vong lap (tham so, nguon/engine khong co, con loi khoi tao) -- khong co ket qua
  3  loi giua luong    con chet / ERROR / EOF khong EOS / mat frame / frame hong / cham tran / thieu frame
                       -> VAN ghi phan da xu ly, trang_thai != hoan_tat
  4  da_dung           dung co kiem soat (`--stop-file`, mac dinh <out>/STOP; SIGINT/SIGTERM): hoan tat frame
                       dang xu ly, gui lenh dung cho con, xa frame dang bay (dem, KHONG xu ly), cho con thoat,
                       ghi DU phan da xu ly

DAU RA (<out>, thu muc MOI -- tu choi neu da co run_meta.json):
  data/<ten>.txt  data_cls/<ten>.csv  timing/<ten>.csv   (nhu run_tracking.run_video_to_dir)
  pipeline/<ten>.csv   moi frame: thoi gian tung stage o CA HAI tien trinh + moc monotonic (xem PHAM VI)
  run_meta.json        trang thai, dem frame, cau hinh hieu luc, HELLO/EOS cua con, provenance, tom tat thoi gian,
                       thiet bi truoc/sau, tegrastats tom tat, lenh replay
  detector_stderr.log  detector.pid  tegrastats.log (dong THO, neu co tegrastats)

PHAM VI DO (moi frame; tom tat bo --warmup-frames frame dau):
  con : decode_ms (cv2.imread/cap.read), mask_ms, pre_ms (letterbox+chuan hoa), infer_ms (H2D+execute_v2+D2H,
        DONG BO -> GPU da hoan tat; tach h2d_ms/exec_ms/d2h_ms), post_ms (decode+NMS+scale), cho_credit_ms
        (chan vi cua so day)
  cha : cho_doc_ms (chan doc pipe), truyen_ms (t_nhan_cha - t_gui_con, monotonic), loc_ms, t_update_ms
        (tracker.update, tu timing/), xu_ly_ms (tu khi dua frame vao tracker den khi quay lai adapter),
        tre_ms (tu truoc khi doc frame o con den xong tracker o cha)
  (review 9f210e2, N21-F1) Giao thuc 2: sau HELLO con CHO lenh BAT DAU; cha chuan bi xong, BAT timer ROI moi gui
  BAT DAU -> con chi doc frame 1 sau do => MOI frame duoc dem nam trong cua so do. `run_meta.thoi_gian`:
  khoi_tao_s      = spawn con -> nhan HELLO (Python con + import + nap engine + lam nong)
  chuan_bi_cha_s  = nhan HELLO -> ngay truoc khi gui BAT DAU (cau hinh tracker, telemetry); chua frame nao duoc doc
  xu_ly_s         = ngay truoc khi gui BAT DAU -> track_video tra ve (frame cuoi qua tracker, nhan EOS hoac dung)
  fps_xu_ly       = so frame da xu ly / xu_ly_s
  ket_thuc_s      = sau vong lap -> con da thoat (xa frame dang bay khi dung, cho/reap con, dung telemetry)
  ghi_s           = ghi MOT/CSV/cache SAU vong lap (do truoc khi ghi run_meta)
  fps_on_dinh     = (N - W) / (t_xong[N] - t_xong[W])  (bo W frame warm-up, theo moc xong tracker o cha)
  kiem_cua_so_do  = tu moc monotonic: frame dau bat dau doc >= luc gui BAT DAU va frame cuoi xong <= het cua so
  (vong 21 dung wall_vong_lap_s/fps_vong_lap: timer bat dau SAU khi con da doc frame dau -> so cu giu lam lich su)

HIEN THI TRUC TIEP (vong 25, `--hien-thi`): cua so OpenCV trong TIEN TRINH CON (noi co anh da giai ma + OpenCV GTK).
  Sau `tracker.update` cua frame i, `on_frame` dong goi CHINH hang MOT/lop cua frame i thanh lenh R(i, ...) va gui thay
  credit "+" (giao thuc 3); con ghep voi anh co CUNG frame_idx roi ve bang `render_mot_video.ve_frame`. Hien thi co MOT o
  cho: frame xem truoc chua kip ve bi thay va DEM (`hien_thi.dem.n_bo_hien_thi`); tracking khong cho hien thi, van xu ly
  du frame, khong doi timestep/nguong. Phim q/Esc hoac dong cua so -> ghi stop-file -> dung co kiem soat (ma 4).
  Thong ke: `<out>/hien_thi.json` -> `run_meta.hien_thi`; `thoi_gian.fps_hien_thi`, `thoi_gian.tong_s`. `progress.json`
  (trang thai/frame/fps tam) cho launcher `src/nano_launcher.py`. Tat `--hien-thi`: hanh vi y het vong 24 (credit "+").

    BYTETRACK_BACKEND=np python src/nano_live.py --source data/smoke/MVI_39031_f0001-0300 --fps 25 \\
        --video-name MVI_39031 --engine yolov8n_fp16.engine --tracker cv --out results_nano/live/cv-1
    # dung: touch results_nano/live/cv-1/STOP   (hoac Ctrl+C)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import nano_detector  # noqa: E402   (chi dung ghi_cache; numpy-only luc import)
import nano_stream_proto as proto  # noqa: E402
import run_tracking  # noqa: E402
import ul_backend  # noqa: E402
from time_units import track_buffer_frames  # noqa: E402

MA_HOAN_TAT, MA_LOI_DAU_VAO, MA_LOI_LUONG, MA_DA_DUNG = 0, 2, 3, 4
TRACKER_YAML = {"cv": "bytetrack.yaml", "ekf": "configs/bytetrack_ekf_ctrv.yaml"}
LIBGOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"
SRC = Path(__file__).resolve().parent
FILE_MA = ("nano_live.py", "nano_detect_stream.py", "nano_stream_proto.py", "nano_detector.py", "run_tracking.py",
           "det_cache.py")


def _la_jetson() -> bool:
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").exists()


# --------------------------------------------------------------------------- tien trinh con
class TienTrinhCon:
    """Quan ly tien trinh detector: pipe hai chieu, log stderr ra file, watchdog treo."""

    def __init__(self, cmd: list[str], env: dict, log_path: Path, pid_path: Path, timeout_s: float):
        self.log_f = open(log_path, "wb")
        # POSIX: nhom tien trinh rieng -> Ctrl+C o terminal chi toi CHA; cha dung con qua giao thuc (lenh S)
        kw = {} if sys.platform.startswith("win") else {"start_new_session": True}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_f, env=env,
                                  cwd=str(config.PROJECT_ROOT), **kw)
        pid_path.write_text(str(self.p.pid), encoding="utf-8")
        self.t_cuoi = time.monotonic()
        self.timeout_s = float(timeout_s)
        self.bi_kill_vi_treo = False
        self._dung_wd = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self) -> None:
        while not self._dung_wd.wait(0.5):
            if self.p.poll() is not None:
                return
            if self.timeout_s > 0 and time.monotonic() - self.t_cuoi > self.timeout_s:
                self.bi_kill_vi_treo = True
                self.p.kill()
                return

    def doc(self):
        msg = proto.doc_thong_diep(self.p.stdout)
        self.t_cuoi = time.monotonic()
        return msg

    def gui(self, b: bytes) -> bool:
        try:
            self.p.stdin.write(b)
            self.p.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def ket_thuc(self, timeout: float = 15.0) -> dict:
        """Dong stdin, cho con thoat; qua han -> terminate -> kill. Tra thong tin thoat."""
        cach = "tu_thoat"
        try:
            self.p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            cach = "terminate"
            self.p.terminate()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cach = "kill"
                self.p.kill()
                self.p.wait()
        self._dung_wd.set()
        try:
            self.p.stdout.close()
        except Exception:  # noqa: BLE001
            pass
        self.log_f.close()
        return {"returncode": self.p.returncode, "cach_ket_thuc": cach, "bi_kill_vi_treo": self.bi_kill_vi_treo}


# --------------------------------------------------------------------------- adapter cho track_video
class NanoStreamDetections:
    """Detection nhan TUNG frame tu tien trinh con. Interface y het `det_cache.VideoDetections`.

    Moi bat thuong (EOF khong EOS, ERROR, mat frame, frame hong, cham tran) -> ghi `self.loi` va KET THUC
    generator (khong nem ngoai le) de `track_video` van tra ve DU hang cua cac frame da xu ly.
    Credit gui cho con SAU khi tracker xu ly xong frame (cua so = so frame chua xu ly xong).
    """

    def __init__(self, con: TienTrinhCon, hello: dict, conf_min: float, classes, max_det: int, class_map,
                 det_max_det: int, allow_truncated: bool, cho_phep_frame_hong: bool, cham_tracker_ms: float = 0.0):
        self.con, self.hello = con, hello
        # (vong 25) hien thi: on_frame dat (frame_idx, lenh R) vao day; credit cua CHINH frame do se la lenh R
        self.ket_qua_cho: tuple[int, bytes] | None = None
        self.n_lenh_r = self.n_lenh_credit = self.n_ket_qua_lech = 0
        self.orig_shape = tuple(int(v) for v in hello["orig_shape"])
        self.conf_min, self.max_det, self.class_map = float(conf_min), int(max_det), class_map
        self._cls_set = None if classes is None else np.array(sorted(set(int(c) for c in classes)))
        self.det_max_det = int(det_max_det)
        self.allow_truncated, self.cho_phep_frame_hong = bool(allow_truncated), bool(cho_phep_frame_hong)
        self.cham_tracker_ms = float(cham_tracker_ms)
        self.raw = {"frame_idx": [], "time_s": [], "n_per_frame": [], "unreadable": [], "truncated": [], "data": []}
        self.pipeline: list[dict] = []
        self.next_idx, self.n_nhan, self.n_unreadable, self.n_truncated = 1, 0, 0, 0
        self.eos: dict | None = None
        self.loi: dict | None = None

    def _ghi_raw(self, idx, t_s, det, unreadable) -> None:
        r = self.raw
        r["frame_idx"].append(idx); r["time_s"].append(t_s); r["n_per_frame"].append(len(det))
        r["unreadable"].append(bool(unreadable)); r["truncated"].append(len(det) >= self.det_max_det)
        if len(det):
            r["data"].append(det)

    def iter_frames(self):
        pc = time.perf_counter
        while True:
            t0 = pc()
            try:
                msg = self.con.doc()
            except proto.LoiGiaoThuc as e:
                self.loi = {"trang_thai": "loi_giao_thuc", "chi_tiet": str(e)}
                return
            t_nhan = time.monotonic()
            t1 = pc()
            if msg is None:
                self.loi = {"trang_thai": "loi_tien_trinh_con",
                            "chi_tiet": "pipe dong (EOF) khong co EOS: tien trinh detector ket thuc bat thuong"}
                return
            if msg["loai"] == proto.ERROR:
                self.loi = {"trang_thai": "loi_tien_trinh_con", "chi_tiet": msg.get("obj")}
                return
            if msg["loai"] == proto.EOS:
                self.eos = msg.get("obj") or {}
                return
            if msg["loai"] != proto.FRAME:
                self.loi = {"trang_thai": "loi_giao_thuc", "chi_tiet": f"thong diep {msg['ten_loai']} giua luong"}
                return
            idx = int(msg["frame_idx"])
            if idx != self.next_idx:
                self.loi = {"trang_thai": "loi_mat_frame", "chi_tiet": f"mong frame {self.next_idx}, nhan frame {idx}"}
                return
            self.next_idx += 1
            self.n_nhan += 1
            det = msg["det"]
            unreadable = bool(msg["co"] & proto.CO_UNREADABLE)
            if unreadable:
                self.n_unreadable += 1
                if not self.cho_phep_frame_hong:
                    self.loi = {"trang_thai": "loi_dau_vao",
                                "chi_tiet": f"frame {idx} khong doc duoc (nguon hong); --cho-phep-frame-hong de coi "
                                            "nhu thoi gian troi khong co quan sat (nhu cache)"}
                    return
            if len(det) >= self.det_max_det:
                self.n_truncated += 1
                if not self.allow_truncated:
                    self.loi = {"trang_thai": "loi_dau_vao",
                                "chi_tiet": f"frame {idx}: {len(det)} detection cham tran det_max_det "
                                            f"{self.det_max_det} (tien to [:max_det] mat nghia); --allow-truncated"}
                    return
            self._ghi_raw(idx, float(msg["time_s"]), det, unreadable)
            arr = (np.zeros((0, 6), np.float32) if unreadable else
                   det_cache.filter_dets(det, self.conf_min, self._cls_set, self.max_det, self.class_map))
            t2 = pc()
            tm = msg["t"]
            row = {"frame": idx, "time_s": float(msg["time_s"]), "unreadable": unreadable,
                   "n_det_raw": int(len(det)), "n_det_loc": int(len(arr)),
                   **{k: float(tm[k]) for k in proto.TRUONG_MS},
                   "t_bat_dau_con": float(tm["t_bat_dau"]), "t_gui_con": float(tm["t_gui"]), "t_nhan_cha": t_nhan,
                   "cho_doc_ms": (t1 - t0) * 1e3, "truyen_ms": (t_nhan - float(tm["t_gui"])) * 1e3,
                   "loc_ms": (t2 - t1) * 1e3}
            self.pipeline.append(row)
            try:
                yield idx, float(msg["time_s"]), unreadable, arr
            finally:
                # chay ca khi generator bi close() do dung som -> timing cua frame cuoi van du
                t3 = pc()
                if self.cham_tracker_ms:
                    time.sleep(self.cham_tracker_ms / 1e3)          # CHI kiem thu: tracker cham -> cua so day
                kq, self.ket_qua_cho = self.ket_qua_cho, None
                if kq is not None and kq[0] == idx:                  # hien thi bat: credit = R(ket qua frame nay)
                    self.con.gui(kq[1])
                    self.n_lenh_r += 1
                else:                                                # tat hien thi / frame khong doc duoc: "+"
                    if kq is not None:
                        self.n_ket_qua_lech += 1
                    self.con.gui(proto.LENH_CREDIT)
                    self.n_lenh_credit += 1
                t_xong = time.monotonic()
                row.update(xu_ly_ms=(t3 - t2) * 1e3, t_xong_cha=t_xong,
                           tre_ms=(t_xong - row["t_bat_dau_con"]) * 1e3, frame_total_cha_ms=(pc() - t0) * 1e3)

    def __len__(self):
        return self.n_nhan

    def cache_arrays(self) -> dict:
        r = self.raw
        d = {k: r[k] for k in ("frame_idx", "time_s", "n_per_frame", "unreadable", "truncated")}
        d["data"] = np.concatenate(r["data"], axis=0) if r["data"] else np.zeros((0, 6), np.float32)
        return d


# --------------------------------------------------------------------------- hien thi (vong 25)
def hang_hien_thi(out) -> list[tuple]:
    """Hang ket qua cua MOT frame de hien thi: CUNG phep tinh voi hang MOT/lop trong `run_tracking._vong_lap_tracker`
    (xyxy float32 -> x, y, w = x2 - x1, h = y2 - y1 bang float32; id/cls int) -> hinh ve trung voi render hau ky tu MOT."""
    if out is None or not len(out):
        return []
    xyxy = out[:, :4].astype(np.float32)
    ids = out[:, 4].astype(int)
    confs = out[:, 5].astype(np.float32)
    clss = out[:, 6].astype(int)
    return [(int(t), float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(c), int(k))
            for (x1, y1, x2, y2), t, c, k in zip(xyxy, ids, confs, clss)]


def ghi_tien_trinh(path: Path | None, obj: dict) -> None:
    """progress.json cho launcher: ghi nguyen tu (tmp + replace) -> launcher khong doc phai file do dang."""
    if path is None:
        return
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({**obj, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _doc_hien_thi(out: Path) -> dict | None:
    """Thong ke hien thi do tien trinh con ghi khi ket thuc (`<out>/hien_thi.json`)."""
    p = out / "hien_thi.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"loi": f"khong doc duoc hien_thi.json: {e}"}


def kiem_man_hinh() -> str | None:
    """Hien thi that can man hinh X. OpenCV/GTK KHONG mo duoc man hinh se lam chet tien trinh detector -> kiem truoc.
    Tra thong bao loi hoac None."""
    if sys.platform.startswith("win"):
        return None
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        return ("--hien-thi can man hinh: bien DISPLAY trong (chay tu phien desktop cua Nano, hoac qua SSH dat "
                "DISPLAY=:0 va XAUTHORITY cua phien desktop)")
    exe = shutil.which("xprop")
    if exe:
        try:
            r = subprocess.run([exe, "-root", "_NET_SUPPORTING_WM_CHECK"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"--hien-thi: khong kiem duoc man hinh {disp}: {e}"
        if r.returncode != 0:
            return (f"--hien-thi: khong mo duoc man hinh {disp} ({(r.stderr or b'').decode(errors='replace').strip()[:160]}); "
                    "kiem DISPLAY/XAUTHORITY")
    return None


# --------------------------------------------------------------------------- thiet bi
class NhatKyTegrastats:
    """Ghi dong THO cua tegrastats ra file suot vong lap (neu co lenh). Tom tat sau."""

    def __init__(self, path: Path, ms: int):
        self.path, self.p, self.f = path, None, None
        exe = shutil.which("tegrastats")
        if ms > 0 and exe:
            self.f = open(path, "wb")
            self.p = subprocess.Popen([exe, "--interval", str(int(ms))], stdout=self.f, stderr=subprocess.STDOUT)

    def dung(self) -> dict:
        if self.p is None:
            return {"co_tegrastats": False, "ghi_chu": "khong co lenh tegrastats (khong phai Jetson?)"}
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.f.close()
        return tom_tat_tegrastats(self.path)


def tom_tat_tegrastats(path: Path) -> dict:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if "RAM" in ln]
    ram, swap, mhz, cpu_pct, gr3d, nhiet = [], [], [], [], [], {}
    for ln in lines:
        m = re.search(r"RAM (\d+)/(\d+)MB", ln)
        if m:
            ram.append(int(m.group(1)))
        m = re.search(r"SWAP (\d+)/(\d+)MB", ln)
        if m:
            swap.append(int(m.group(1)))
        m = re.search(r"CPU \[([^\]]*)\]", ln)
        if m:
            for c in m.group(1).split(","):
                mm = re.match(r"\s*(\d+)%@(\d+)", c)
                if mm:
                    cpu_pct.append(int(mm.group(1))); mhz.append(int(mm.group(2)))
        m = re.search(r"GR3D_FREQ (\d+)%(?:@(\d+))?", ln)
        if m:
            gr3d.append(int(m.group(1)))
        for ten, v in re.findall(r"(\w+)@([\d.]+)C", ln):
            nhiet.setdefault(ten, []).append(float(v))
    return {"co_tegrastats": True, "n_mau": len(lines),
            "ram_dung_MB_max": max(ram) if ram else None, "swap_dung_MB_max": max(swap) if swap else None,
            "cpu_MHz_min_max": [min(mhz), max(mhz)] if mhz else None,
            "cpu_pct_tb": round(statistics.mean(cpu_pct), 1) if cpu_pct else None,
            "gr3d_pct_tb_max": [round(statistics.mean(gr3d), 1), max(gr3d)] if gr3d else None,
            "nhiet_max_C": {k: max(v) for k, v in nhiet.items()}}


def xung_sysfs() -> dict:
    """Xung CPU/GPU doc tu sysfs neu doc duoc (khong root); thieu -> ghi thieu."""
    out = {}
    for p in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        try:
            out[p.parts[-3]] = int(p.read_text().strip())
        except Exception:  # noqa: BLE001
            out[p.parts[-3]] = "khong doc duoc"
    for pat in ("/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq", "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq"):
        p = Path(pat)
        if p.exists():
            try:
                out["gpu_cur_freq"] = int(p.read_text().strip())
            except Exception:  # noqa: BLE001
                out["gpu_cur_freq"] = "khong doc duoc"
            break
    return out or {"ghi_chu": "khong doc duoc xung tu sysfs (khong phai Jetson?)"}


def trang_thai_thiet_bi() -> dict:
    import nano_bench_tracker as nbt
    d = nbt.trang_thai_thiet_bi()
    d["xung"] = xung_sysfs()
    try:
        m = {k: v for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines()[:20])}
        d["meminfo_kB"] = {k: m[k].strip() for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree") if k in m}
    except Exception:  # noqa: BLE001
        pass
    return d


# --------------------------------------------------------------------------- tom tat
def _tk(v: list[float]) -> dict | None:
    v = [x for x in v if x == x]
    if not v:
        return None
    s = sorted(v)
    return {"n": len(s), "trung_vi": statistics.median(s), "tb": statistics.mean(s),
            "p90": s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))], "min": s[0], "max": s[-1]}


def tom_tat_thoi_gian(pipeline: list[dict], df_t, warmup: int) -> dict:
    upd = {int(f): float(u) for f, u in zip(df_t["frame"], df_t["t_update_ms"])} if len(df_t) else {}
    rows = [r for r in pipeline if not r["unreadable"] and "t_xong_cha" in r]
    sau = rows[warmup:]
    cot = list(proto.TRUONG_MS) + ["cho_doc_ms", "truyen_ms", "loc_ms", "xu_ly_ms", "frame_total_cha_ms", "tre_ms"]
    d = {k: _tk([r[k] for r in sau]) for k in cot}
    d["t_update_ms"] = _tk([upd[r["frame"]] for r in sau if r["frame"] in upd])
    xong = [r["t_xong_cha"] for r in rows]
    d["khoang_giua_frame_ms"] = _tk([(b - a) * 1e3 for a, b in zip(xong[warmup:], xong[warmup + 1:])])
    n, w = len(xong), warmup
    d["fps_on_dinh"] = ((n - w) / (xong[-1] - xong[w - 1]) if w >= 1 and n > w and xong[-1] > xong[w - 1] else None)
    d["n_frame_tong_ket"] = len(sau)
    d["warmup_bo"] = warmup
    return d


def sha256_lf(p: Path) -> str | None:
    try:
        return ul_backend.sha256_lf(p)
    except Exception:  # noqa: BLE001
        return None


def provenance() -> dict:
    d = {"file_ma_sha256_lf": {f: sha256_lf(SRC / f) for f in FILE_MA}, "git_commit": det_cache.git_commit(),
         "tracker_backend": ul_backend.backend_name()}
    pi = config.PROJECT_ROOT / "PACKAGE_INFO.json"
    if pi.exists():
        try:
            j = json.loads(pi.read_text(encoding="utf-8"))
            d["package_info"] = {k: j.get(k) for k in ("git_commit", "commit", "created", "phien_ban", "version") if k in j}
            d["package_info_sha256"] = hashlib.sha256(pi.read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001
            d["package_info"] = "khong doc duoc"
    return d


# --------------------------------------------------------------------------- kiem tra truoc khi spawn
def kiem_truoc_khi_chay(a) -> tuple[list[str], dict]:
    """(review 9f210e2, N21-F3) MOI dau vao kiem duoc thi kiem TRUOC khi khoi dong detector: tham so, nguon, engine,
    YAML tracker (ton tai + doc duoc + tracker_type hop le), override `--set` (dang key=value, khoa CO trong YAML,
    kieu gia tri), class_map, cache/regions/stop-file. Tra (loi, thong tin). Loi -> ma 2, KHONG spawn."""
    loi: list[str] = []
    out = Path(a.out)
    if (out / "run_meta.json").exists():
        loi.append(f"{out} da co run_meta.json -> dung thu muc MOI cho moi lan chay")
    if not Path(a.source).exists():
        loi.append(f"nguon khong ton tai: {a.source}")
    so = [("--fps", a.fps, lambda v: v is None or v > 0), ("--max-frames", a.max_frames, lambda v: v is None or v >= 1),
          ("--hang-doi", a.hang_doi, lambda v: v >= 1), ("--lam-nong", a.lam_nong, lambda v: v >= 0),
          ("--warmup-frames", a.warmup_frames, lambda v: v >= 0), ("--tegrastats-ms", a.tegrastats_ms, lambda v: v >= 0),
          ("--progress-every", a.progress_every, lambda v: v >= 0), ("--timeout-con-s", a.timeout_con_s, lambda v: v > 0),
          ("--imgsz", a.imgsz, lambda v: v > 0 and v % 32 == 0), ("--det-conf", a.det_conf, lambda v: 0 <= v <= 1),
          ("--det-iou", a.det_iou, lambda v: 0 < v <= 1), ("--conf-min", a.conf_min, lambda v: 0 <= v <= 1),
          ("--max-det", a.max_det, lambda v: v >= 1), ("--det-max-det", a.det_max_det, lambda v: v >= 1),
          ("--track-buffer-sec", a.track_buffer_sec, lambda v: v is None or v > 0)]
    loi += [f"{ten} = {v!r} khong hop le" for ten, v, ok in so if not ok(v)]
    if a.hien_thi_moi < 0:
        loi.append(f"--hien-thi-moi = {a.hien_thi_moi!r} khong hop le (>= 0)")
    if a.hien_thi_kich_thuoc and min(a.hien_thi_kich_thuoc) <= 0:
        loi.append(f"--hien-thi-kich-thuoc = {a.hien_thi_kich_thuoc!r} khong hop le")
    if a.hien_thi and not a.hien_thi_an:
        e = kiem_man_hinh()
        if e:
            loi.append(e)
    if a.gia_lap_cache:
        if not Path(a.gia_lap_cache).is_dir():
            loi.append(f"khong co thu muc cache gia lap: {a.gia_lap_cache}")
    elif a.torch_pt:
        if not Path(a.torch_pt).is_file():
            loi.append(f"khong co model .pt: {a.torch_pt}")
    elif not (a.engine and Path(a.engine).is_file()):
        loi.append(f"khong co engine: {a.engine}")
    if a.python_detector and not (Path(a.python_detector).is_file() or shutil.which(a.python_detector)):
        loi.append(f"khong co interpreter cho detector: {a.python_detector}")
    if a.ld_preload and not Path(a.ld_preload).is_file():
        loi.append(f"khong co file LD_PRELOAD: {a.ld_preload}")
    if a.regions and not Path(a.regions).is_file():
        loi.append(f"khong co file regions: {a.regions}")
    classes = None if a.all_classes else a.classes
    loi += det_cache.check_compatible({"conf": a.det_conf, "classes": None, "max_det": a.det_max_det},
                                      a.conf_min, classes, a.max_det)
    try:
        class_map = run_tracking.parse_class_map(a.class_map)
    except Exception as e:  # noqa: BLE001
        class_map = None
        loi.append(f"--class-map '{a.class_map}' sai: {type(e).__name__}: {e}")
    cache_root = (Path(a.cache_root) if a.cache_root else det_cache.CACHE_ROOT).resolve()
    if a.save_cache and (cache_root / a.save_cache).exists():
        loi.append(f"cache {cache_root / a.save_cache} da ton tai -> dung --save-cache khac")
    stop_path = Path(a.stop_file) if a.stop_file else out / "STOP"
    if stop_path.exists():
        loi.append(f"stop-file {stop_path} da ton tai truoc khi chay -> xoa di hoac dung --stop-file khac")
    yaml = None
    if bool(a.tracker) == bool(a.tracker_yaml):
        loi.append("can DUNG MOT trong --tracker cv|ekf hoac --tracker-yaml")
    else:
        yaml = a.tracker_yaml or TRACKER_YAML[a.tracker]
        try:
            run_tracking.resolve_tracker_yaml(yaml)
            base = run_tracking.cfg_to_dict(run_tracking.load_tracker_cfg(yaml, {}))
        except Exception as e:  # noqa: BLE001
            base = None
            loi.append(f"YAML tracker '{yaml}' khong dung duoc: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        if base is not None:
            if base.get("tracker_type") not in run_tracking.tracker_registry():
                loi.append(f"YAML '{yaml}': tracker_type {base.get('tracker_type')!r} khong co trong registry")
            for it in a.set or []:
                k, eq, v = it.partition("=")
                if not eq or not re.fullmatch(r"[A-Za-z_]\w*", k) or v == "":
                    loi.append(f"--set '{it}' sai dang key=value")
                    continue
                if k not in base:
                    loi.append(f"--set {k}: khoa khong co trong YAML '{yaml}' (co: {', '.join(sorted(base))})")
                    continue
                goc, moi = base[k], run_tracking.parse_set([it])[k]
                sai_kieu = (isinstance(goc, bool) and not isinstance(moi, bool)) or (
                    isinstance(goc, (int, float)) and not isinstance(goc, bool)
                    and (isinstance(moi, bool) or not isinstance(moi, (int, float))))
                if sai_kieu:
                    loi.append(f"--set {k}={v}: kieu {type(moi).__name__} khong khop gia tri YAML {goc!r}")
            if not loi:
                try:                       # dung thu cau hinh hieu luc voi fps tam (fps that den tu HELLO)
                    ov = run_tracking.parse_set(a.set)
                    if a.track_buffer_sec is not None:
                        ov["track_buffer"] = track_buffer_frames(a.track_buffer_sec, a.fps or 25.0)
                    run_tracking.load_tracker_cfg(yaml, ov)
                except Exception as e:  # noqa: BLE001
                    loi.append(f"cau hinh tracker + override khong dung duoc: {type(e).__name__}: {e}")
    return loi, {"yaml": yaml, "classes": classes, "class_map": class_map, "cache_root": cache_root,
                 "stop_path": stop_path}


class LoiTruocXuLy(Exception):
    """Loi sau khi spawn nhung TRUOC khi xu ly frame (con khoi tao hong, giao thuc, chuan bi o cha) -> ma 2."""

    def __init__(self, trang_thai: str, chi_tiet):
        super().__init__(str(chi_tiet))
        self.trang_thai, self.chi_tiet = trang_thai, chi_tiet


# --------------------------------------------------------------------------- CLI
def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="Luong truc tiep tren Nano: video/anh -> TensorRT -> tracker CV/EKF")
    g = ap.add_argument_group("nguon")
    g.add_argument("--source", required=True, help="file video hoac thu muc anh")
    g.add_argument("--fps", type=float, default=None, help="bat buoc voi thu muc anh; ghi de fps video")
    g.add_argument("--video-name", default=None, help="ten sequence trong MOT/cache (mac dinh: ten nguon)")
    g.add_argument("--max-frames", type=int, default=None)
    d = ap.add_argument_group("detector (tien trinh con)")
    d.add_argument("--engine", default=None, help="engine TensorRT build TREN Nano")
    d.add_argument("--onnx-sha256", default=None)
    d.add_argument("--bo-dem", choices=["cudart", "torch", "pycuda"], default="cudart")
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--det-conf", type=float, default=0.05)
    d.add_argument("--det-iou", type=float, default=0.7)
    d.add_argument("--det-max-det", type=int, default=3000)
    d.add_argument("--regions", default=None, help="CHE DO DOI CHIEU: to xam vung bo qua lay tu NHAN dataset")
    d.add_argument("--python-detector", default=None,
                   help="interpreter cua con (mac dinh: /usr/bin/python3 tren Jetson, con lai sys.executable)")
    d.add_argument("--ld-preload", default=None, help=f"mac dinh {LIBGOMP} neu co (Jetson)")
    d.add_argument("--hang-doi", type=int, default=4, help="cua so credit K: toi da K frame da gui chua xu ly xong")
    d.add_argument("--lam-nong", type=int, default=3)
    d.add_argument("--timeout-con-s", type=float, default=180.0, help="khong nhan thong diep qua han -> kill con")
    f = ap.add_argument_group("loc + tracker (y het run_tracking.py)")
    f.add_argument("--conf-min", type=float, default=0.25)
    f.add_argument("--classes", nargs="*", type=int, default=[1, 2, 3, 5, 7])
    f.add_argument("--all-classes", action="store_true")
    f.add_argument("--class-map", default=None)
    f.add_argument("--max-det", type=int, default=300)
    f.add_argument("--tracker", choices=sorted(TRACKER_YAML), default=None, help="cv | ekf (tat cho --tracker-yaml)")
    f.add_argument("--tracker-yaml", default=None)
    f.add_argument("--set", nargs="*", default=None)
    f.add_argument("--track-buffer-sec", type=float, default=1.2)
    f.add_argument("--elapsed-mode", choices=["skip", "steps"], default="steps")
    f.add_argument("--allow-truncated", action="store_true")
    f.add_argument("--cho-phep-frame-hong", action="store_true",
                   help="frame khong doc duoc -> thoi gian troi, khong quan sat (nhu cache); mac dinh: LOI")
    o = ap.add_argument_group("dau ra")
    o.add_argument("--out", required=True, help="thu muc MOI cho lan chay nay")
    o.add_argument("--save-cache", default=None, metavar="CACHE_ID", help="ghi detection DA vao tracker de replay")
    o.add_argument("--cache-root", default=None)
    o.add_argument("--warmup-frames", type=int, default=25, help="bo khoi TOM TAT thoi gian (CSV van du)")
    o.add_argument("--stop-file", default=None, help="mac dinh <out>/STOP")
    o.add_argument("--tegrastats-ms", type=int, default=500, help="0 = tat")
    o.add_argument("--progress-every", type=int, default=50)
    o.add_argument("--progress-file", default=None, help="mac dinh <out>/progress.json (trang thai cho launcher)")
    h = ap.add_argument_group("hien thi truc tiep (vong 25) -- KHONG doi detector/tracker/ket qua")
    h.add_argument("--hien-thi", action="store_true", help="cua so xem bbox/ID/lop trong luc tracking (q/Esc: dung)")
    h.add_argument("--hien-thi-an", action="store_true", help="CHI KIEM THU: moi buoc hien thi tru tao cua so")
    h.add_argument("--hien-thi-luu-mau", default=None, help="thu muc luu PNG frame da ve (kiem doi chieu)")
    h.add_argument("--hien-thi-moi", type=int, default=0, help="luu mau moi N frame (0 = khong)")
    h.add_argument("--hien-thi-kich-thuoc", type=int, nargs=2, default=None, metavar=("RONG", "CAO"))
    t = ap.add_argument_group("CHI KIEM THU (gia lap, ghi ro trong metadata)")
    t.add_argument("--gia-lap-cache", default=None, help="thu muc det cache thay suy luan (con)")
    t.add_argument("--gia-lap-video", default=None)
    t.add_argument("--gia-lap-loi-o-frame", type=int, default=None)
    t.add_argument("--gia-lap-bo-frame", type=int, default=None)
    t.add_argument("--gia-lap-cham-ms", type=float, default=0.0)
    t.add_argument("--gia-lap-cham-tracker-ms", type=float, default=0.0)
    t.add_argument("--torch-pt", default=None, help="CHI PC: suy luan .pt thay TensorRT")
    t.add_argument("--gia-lap-phim-dung-o-frame", type=int, default=None, help="hien thi gia lap phim q o frame N")
    t.add_argument("--gia-lap-dong-cua-so-o-frame", type=int, default=None,
                   help="hien thi AN gia lap nguoi dung dong cua so sau frame N (kiem N25-F1)")
    a = ap.parse_args()
    t_main = time.perf_counter()

    out = Path(a.out)
    hien_thi = bool(a.hien_thi or a.hien_thi_an)
    prog = Path(a.progress_file) if a.progress_file else out / "progress.json"
    # ---------------- (1) kiem tra TRUOC khi spawn (N21-F3): loi -> ma 2, KHONG khoi dong detector ----------------
    loi, tt0 = kiem_truoc_khi_chay(a)
    if loi:
        for e in loi:
            print(f"[LOI] {e}")
        if prog.parent.is_dir() and not (out / "run_meta.json").exists():
            ghi_tien_trinh(prog, {"trang_thai": "loi", "ma_thoat": MA_LOI_DAU_VAO, "loi": "; ".join(loi)})
        return MA_LOI_DAU_VAO
    yaml, classes, class_map = tt0["yaml"], tt0["classes"], tt0["class_map"]
    cache_root, stop_path = tt0["cache_root"], tt0["stop_path"]
    out.mkdir(parents=True, exist_ok=True)
    ghi_tien_trinh(prog, {"trang_thai": "dang_khoi_tao", "frame": 0, "n_frames_khai_bao": 0, "hien_thi": hien_thi})

    jetson = _la_jetson()
    py_det = a.python_detector or ("/usr/bin/python3" if jetson else sys.executable)
    cmd = [py_det, str(SRC / "nano_detect_stream.py"), "--source", str(Path(a.source).resolve()),
           "--imgsz", str(a.imgsz), "--conf", str(a.det_conf), "--iou", str(a.det_iou),
           "--max-det", str(a.det_max_det), "--hang-doi", str(a.hang_doi), "--lam-nong", str(a.lam_nong),
           "--bo-dem", a.bo_dem]
    for k, v in (("--fps", a.fps), ("--max-frames", a.max_frames), ("--engine", a.engine),
                 ("--onnx-sha256", a.onnx_sha256), ("--regions", a.regions), ("--gia-lap-cache", a.gia_lap_cache),
                 ("--gia-lap-video", a.gia_lap_video), ("--gia-lap-loi-o-frame", a.gia_lap_loi_o_frame),
                 ("--gia-lap-bo-frame", a.gia_lap_bo_frame), ("--torch-pt", a.torch_pt)):
        if v is not None:
            cmd += [k, str(Path(v).resolve()) if k in ("--engine", "--regions", "--gia-lap-cache", "--torch-pt") else str(v)]
    if a.gia_lap_cham_ms:
        cmd += ["--gia-lap-cham-ms", str(a.gia_lap_cham_ms)]
    if hien_thi:
        nhan = (a.tracker or Path(a.tracker_yaml).stem).upper()
        src_p = Path(a.source)
        ten_src = src_p.name if src_p.is_dir() else src_p.stem
        cmd += ["--hien-thi-an" if a.hien_thi_an else "--hien-thi",
                "--hien-thi-tieu-de", f"Xem truc tiep - {nhan} - {ten_src} (q/Esc: dung)", "--hien-thi-hud", nhan,
                "--hien-thi-json", str((out / "hien_thi.json").resolve()),
                "--yeu-cau-dung-file", str(stop_path.resolve())]
        if a.hien_thi_luu_mau and a.hien_thi_moi > 0:
            cmd += ["--hien-thi-luu-mau", str(Path(a.hien_thi_luu_mau).resolve()), "--hien-thi-moi", str(a.hien_thi_moi)]
        if a.hien_thi_kich_thuoc:
            cmd += ["--hien-thi-kich-thuoc", *[str(v) for v in a.hien_thi_kich_thuoc]]
        if a.gia_lap_phim_dung_o_frame:
            cmd += ["--gia-lap-phim-dung-o-frame", str(a.gia_lap_phim_dung_o_frame)]
        if a.gia_lap_dong_cua_so_o_frame:
            cmd += ["--gia-lap-dong-cua-so-o-frame", str(a.gia_lap_dong_cua_so_o_frame)]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    preload = a.ld_preload if a.ld_preload is not None else (LIBGOMP if jetson and Path(LIBGOMP).exists() else "")
    if preload:
        env["LD_PRELOAD"] = preload
    env.pop("BYTETRACK_BACKEND", None)

    stop_ev = threading.Event()

    def _tin_hieu(signum, _frame):
        print(f"\n[nano_live] nhan tin hieu {signum} -> dung co kiem soat sau frame hien tai")
        stop_ev.set()
    for s in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if s is not None:
            try:
                signal.signal(s, _tin_hieu)
            except (ValueError, OSError):
                pass

    # ---------------- (2) spawn: MOI duong ra deu di qua `finally` (dung/reap con, dung telemetry) ----------------
    truoc = trang_thai_thiet_bi()
    con = teg = vd = hello = cfg = None
    teg_tt: dict = {"co_tegrastats": False, "ghi_chu": "chua bat"}
    thoat: dict = {}
    loi_truoc: LoiTruocXuLy | None = None
    loi_cha: str | None = None
    df = df_cls = df_t = None
    n_bay_khi_dung = 0
    dem = {"n": 0, "dung_som": False, "ly_do_dung": None}
    ten, fps, dt_nominal, overrides, yaml_sha = None, None, None, {}, None
    tg = {"khoi_tao_s": None, "chuan_bi_cha_s": None, "xu_ly_s": None, "ket_thuc_s": None}
    moc: dict = {}
    da_bat_dau = False              # True tu luc gui BAT DAU: loi truoc do -> ma 2, sau do -> ma 3
    t_spawn = time.perf_counter()
    try:
        con = TienTrinhCon(cmd, env, out / "detector_stderr.log", out / "detector.pid", a.timeout_con_s)
        try:
            hello_msg = con.doc()
        except proto.LoiGiaoThuc as e:
            raise LoiTruocXuLy("loi_khoi_tao", f"giao thuc: {e}")
        if hello_msg is None:
            raise LoiTruocXuLy("loi_khoi_tao", "con thoat truoc HELLO (EOF)")
        if hello_msg["loai"] != proto.HELLO:
            raise LoiTruocXuLy("loi_khoi_tao", hello_msg.get("obj") or hello_msg["ten_loai"])
        hello = hello_msg["obj"]
        if hello.get("phien_ban_proto") != proto.PHIEN_BAN:
            raise LoiTruocXuLy("loi_khoi_tao", f"phien ban giao thuc con {hello.get('phien_ban_proto')} != cha {proto.PHIEN_BAN}")
        t_hello = time.perf_counter()
        tg["khoi_tao_s"] = t_hello - t_spawn

        # ---- chuan bi o cha (CHUA co frame nao duoc doc: con dang cho lenh BAT DAU) ----
        try:
            fps = float(hello["fps"])
            ten = a.video_name or hello["nguon"]["ten"]
            dt_nominal = 1.0 / fps
            overrides = run_tracking.parse_set(a.set)
            if a.track_buffer_sec is not None:
                overrides["track_buffer"] = track_buffer_frames(a.track_buffer_sec, fps)
            cfg = run_tracking.load_tracker_cfg(yaml, overrides)
            yaml_sha = hashlib.sha256(Path(run_tracking.resolve_tracker_yaml(yaml)).read_bytes()).hexdigest()[:12]
        except Exception as e:  # noqa: BLE001
            raise LoiTruocXuLy("loi_chuan_bi", f"{type(e).__name__}: {e}")
        print(f"[nano_live] {hello['detector']} | {hello['nguon']['loai']} {ten}: {hello['n_frames_khai_bao']} frame khai bao, "
              f"fps {fps:g}, khoi tao {tg['khoi_tao_s']:.1f}s | tracker {cfg.tracker_type} ({yaml}) conf>{a.conf_min} "
              f"classes={classes} max_det={a.max_det} track_buffer={cfg.track_buffer} elapsed={a.elapsed_mode} "
              f"| cua so K={a.hang_doi} | dung: touch {stop_path} hoac Ctrl+C")
        vd = NanoStreamDetections(con, hello, a.conf_min, classes, a.max_det, class_map, a.det_max_det,
                                  a.allow_truncated, a.cho_phep_frame_hong, a.gia_lap_cham_tracker_ms)

        tien_trinh = {"t_ghi": 0.0}

        def on_frame(frame_idx: int, _out) -> bool:
            dem["n"] += 1
            if hien_thi:                    # ket qua CUA FRAME NAY -> lenh R thay credit "+" (ghep anh theo frame_idx)
                vd.ket_qua_cho = (int(frame_idx), proto.dong_goi_ket_qua(frame_idx, hang_hien_thi(_out)))
            if a.progress_every and dem["n"] % a.progress_every == 0:
                print(f"  frame {frame_idx}/{hello['n_frames_khai_bao']}")
            now = time.perf_counter()
            if now - tien_trinh["t_ghi"] >= 0.3:
                tien_trinh["t_ghi"] = now
                ghi_tien_trinh(prog, {"trang_thai": "dang_chay", "frame": int(frame_idx), "n_frame_da_xu_ly": dem["n"],
                                      "n_frames_khai_bao": hello["n_frames_khai_bao"], "hien_thi": hien_thi,
                                      "fps_xu_ly_tam": dem["n"] / (now - t_bd) if now > t_bd else None})
            if stop_ev.is_set() or stop_path.exists():
                dem["dung_som"], dem["ly_do_dung"] = True, ("tin_hieu" if stop_ev.is_set() else "stop-file")
                return False
            return True

        teg = NhatKyTegrastats(out / "tegrastats.log", a.tegrastats_ms if jetson else 0)

        # ---- CUA SO XU LY (N21-F1): bat timer ROI MOI gui BAT DAU -> moi frame duoc dem nam trong cua so ----
        t_bd = time.perf_counter()
        moc["gui_bat_dau"] = time.monotonic()
        tg["chuan_bi_cha_s"] = t_bd - t_hello
        if not con.gui(proto.LENH_BAT_DAU):
            raise LoiTruocXuLy("loi_khoi_tao", "khong gui duoc lenh BAT DAU (pipe toi con da dong)")
        da_bat_dau = True
        try:
            df, df_cls, df_t = run_tracking.track_video(vd, cfg, dt_nominal, a.elapsed_mode, None, ten,
                                                        on_frame=on_frame)
        finally:
            tg["xu_ly_s"] = time.perf_counter() - t_bd
            moc["ket_thuc_xu_ly"] = time.monotonic()
    except LoiTruocXuLy as e:
        loi_truoc = e
    except Exception:  # noqa: BLE001   -- loi o CHA: van don dep; truoc BAT DAU -> ma 2, trong vong lap -> ma 3
        import traceback
        if da_bat_dau:
            loi_cha = traceback.format_exc(limit=8)
        else:
            loi_truoc = LoiTruocXuLy("loi_chuan_bi", traceback.format_exc(limit=8))
    finally:
        t_kt = time.perf_counter()
        if teg is not None:
            teg_tt = teg.dung()
        if con is not None:
            if dem["dung_som"] and loi_cha is None and vd is not None and vd.loi is None:
                con.gui(proto.LENH_DUNG)
                while True:                       # xa frame dang bay toi EOS/EOF (dem, KHONG xu ly)
                    try:
                        m = con.doc()
                    except proto.LoiGiaoThuc:
                        break
                    if m is None or m["loai"] == proto.ERROR:
                        break
                    if m["loai"] == proto.FRAME:
                        n_bay_khi_dung += 1
                    elif m["loai"] == proto.EOS:
                        vd.eos = m.get("obj") or {}
                        break
            elif loi_truoc is not None or loi_cha is not None or (vd is not None and vd.loi is not None):
                con.gui(proto.LENH_DUNG)          # con dang cho BAT DAU / dang chay -> xin dung truoc khi dong stdin
            ok_eos = vd is not None and (vd.eos is not None or dem["dung_som"])
            thoat = con.ket_thuc(15 if ok_eos else 5)
        tg["ket_thuc_s"] = time.perf_counter() - t_kt

    # ---------------- loi truoc khi xu ly frame: ma 2, khong ket qua tracking ----------------
    if loi_truoc is not None:
        print(f"[LOI] {loi_truoc.trang_thai}: {loi_truoc.chi_tiet}; con {thoat}")
        (out / "run_meta.json").write_text(json.dumps(
            {"trang_thai": loi_truoc.trang_thai, "ma_thoat": MA_LOI_DAU_VAO, "hoan_tat": False,
             "chi_tiet": loi_truoc.chi_tiet, "con": thoat, "lenh_con": cmd, "hello_con": hello,
             "thoi_gian": {k: v for k, v in tg.items() if v is not None},
             "hien_thi": _doc_hien_thi(out) if hien_thi else None,
             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        ghi_tien_trinh(prog, {"trang_thai": "loi", "ma_thoat": MA_LOI_DAU_VAO,
                              "loi": f"{loi_truoc.trang_thai}: {str(loi_truoc.chi_tiet)[:300]}"})
        return MA_LOI_DAU_VAO

    # ---------------- trang thai ----------------
    import pandas as pd
    if df is None:                      # loi o cha giua vong lap: track_video khong tra ket qua
        df = pd.DataFrame(columns=run_tracking.TRK_COLS)
        df_cls = pd.DataFrame(columns=["frame", "id", "cls"])
        df_t = pd.DataFrame(columns=["frame", "n_steps", "n_tracks", "n_dets", "t_predict_ms", "t_assoc_ms", "t_update_ms"])
    n_xu_ly = len(vd.raw["frame_idx"])
    if loi_cha is not None:
        trang_thai, ma, chi_tiet = "loi_cha", MA_LOI_LUONG, loi_cha
    elif vd.loi is not None:
        trang_thai, ma, chi_tiet = vd.loi["trang_thai"], MA_LOI_LUONG, vd.loi["chi_tiet"]
        if thoat.get("bi_kill_vi_treo"):
            trang_thai, chi_tiet = "loi_tien_trinh_con_treo", f"khong nhan thong diep > {a.timeout_con_s}s -> kill; {chi_tiet}"
    elif dem["dung_som"]:
        trang_thai, ma, chi_tiet = "da_dung", MA_DA_DUNG, f"dung theo yeu cau ({dem['ly_do_dung']})"
    elif vd.eos is None:
        trang_thai, ma, chi_tiet = "loi_tien_trinh_con", MA_LOI_LUONG, "khong nhan EOS"
    else:
        mong = hello["n_frames_khai_bao"] if not a.max_frames else min(hello["n_frames_khai_bao"] or a.max_frames,
                                                                        a.max_frames)
        if int(vd.eos.get("n_frame_gui", -1)) != vd.n_nhan:
            trang_thai, ma, chi_tiet = "loi_mat_frame", MA_LOI_LUONG, f"EOS bao {vd.eos.get('n_frame_gui')} frame, nhan {vd.n_nhan}"
        elif thoat.get("returncode") != 0:
            trang_thai, ma, chi_tiet = "loi_tien_trinh_con", MA_LOI_LUONG, f"con thoat ma {thoat.get('returncode')} sau EOS"
        elif mong and vd.n_nhan != mong:
            trang_thai, ma, chi_tiet = ("thieu_frame_so_voi_khai_bao", MA_LOI_LUONG,
                                        f"xu ly {vd.n_nhan} frame, khai bao {mong} (nguon ket thuc som / dem khai bao sai)")
        else:
            trang_thai, ma, chi_tiet = "hoan_tat", MA_HOAN_TAT, f"{vd.n_nhan} frame, ly do EOS: {vd.eos.get('ly_do')}"

    # ---------------- ghi ket qua (LUON ghi phan da xu ly) ----------------
    t_ghi = time.perf_counter()
    for sub in ("data", "data_cls", "timing", "pipeline"):
        (out / sub).mkdir(exist_ok=True)
    df.to_csv(out / "data" / f"{ten}.txt", header=False, index=False)
    df_cls.to_csv(out / "data_cls" / f"{ten}.csv", index=False)
    df_t.to_csv(out / "timing" / f"{ten}.csv", index=False)
    pd.DataFrame(vd.pipeline).to_csv(out / "pipeline" / f"{ten}.csv", index=False)
    cache_info = None
    if a.save_cache:
        arr = vd.cache_arrays()
        meta_cache = {"backend": hello["detector"],
                      "weights": hello.get("engine") or hello.get("weights") or hello.get("gia_lap_cache"),
                      "weights_sha256": hello.get("engine_sha256") or hello.get("weights_sha256"),
                      "onnx_sha256": hello.get("onnx_sha256"), "imgsz": a.imgsz, "conf": a.det_conf, "iou": a.det_iou,
                      "classes": None, "max_det": a.det_max_det, "half": hello["detector"] == "tensorrt",
                      "agnostic_nms": False, "augment": False, "batch": 1, "device": "jetson-nano-trt-live",
                      "mask_ignored_regions": bool(a.regions), "dataset": "video_file", "part": hello["nguon"]["loai"],
                      "letterbox": hello.get("letterbox"), "versions": hello.get("versions"),
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "ghi_chu": "cache do nano_live.py ghi: DUNG cac frame da vao tracker cua luong truc tiep "
                                 "(detection tho truoc loc); legacy (khong spec_sha256)"}
        cache_info = nano_detector.ghi_cache(
            str(cache_root / a.save_cache), ten, arr["frame_idx"], arr["time_s"], arr["n_per_frame"],
            arr["unreadable"], arr["truncated"], arr["data"], hello["orig_shape"], meta_cache,
            {"source": hello["nguon"]["duong_dan"], "fps": fps, "tu_luong_truc_tiep": str(out.resolve()),
             "trang_thai_luot": trang_thai})
    tg["ghi_s"] = time.perf_counter() - t_ghi

    # (N21-F2) replay phai tim DUNG noi cache duoc ghi: root mac dinh -> run_tracking.py; root khac -> wrapper
    # nano_replay.py voi DUONG DAN TUYET DOI (mot phan tu argv, giu dau cach). Placeholder <split>/<ten> la phan tu rieng.
    rt = ["--cache-id", a.save_cache or "<cache>", "--split-name", "<split>", "--tracker-yaml", yaml,
          "--tracker-name", "<ten>", "--videos", ten, "--conf-min", str(a.conf_min)]
    rt += ["--all-classes"] if classes is None else ["--classes", *[str(c) for c in classes]]
    rt += ["--max-det", str(a.max_det), "--fps", f"{fps:g}", "--track-buffer-sec", str(a.track_buffer_sec),
           "--elapsed-mode", a.elapsed_mode]
    if a.class_map:
        rt += ["--class-map", a.class_map]
    if a.set:
        rt += ["--set", *a.set]
    if a.allow_truncated:
        rt.append("--allow-truncated")
    if cache_root == det_cache.CACHE_ROOT.resolve():
        replay = ["python", "src/run_tracking.py", *rt]
    else:
        replay = ["python", "src/nano_replay.py", "--cache-root", str(cache_root), "--", *rt]

    tt = tom_tat_thoi_gian(vd.pipeline, df_t, a.warmup_frames)
    xong = [r for r in vd.pipeline if "t_xong_cha" in r]
    cua_so = None
    if xong and moc.get("gui_bat_dau") is not None and moc.get("ket_thuc_xu_ly") is not None:
        dau, cuoi = min(r["t_bat_dau_con"] for r in xong), max(r["t_xong_cha"] for r in xong)
        cua_so = {"frame_dau_bat_dau_doc_mono": dau, "frame_cuoi_xong_tracker_mono": cuoi,
                  "tre_tu_bat_dau_den_doc_frame_dau_ms": (dau - moc["gui_bat_dau"]) * 1e3,
                  "moi_frame_trong_cua_so": bool(dau >= moc["gui_bat_dau"] and cuoi <= moc["ket_thuc_xu_ly"]),
                  "khoang_bao_phu_s": cuoi - dau}
    xu_ly = tg["xu_ly_s"]
    meta = {
        "trang_thai": trang_thai, "ma_thoat": ma, "chi_tiet": chi_tiet, "hoan_tat": trang_thai == "hoan_tat",
        "dem": {"n_frames_khai_bao": hello["n_frames_khai_bao"], "max_frames": a.max_frames, "n_frame_nhan": vd.n_nhan,
                "n_frame_da_xu_ly": n_xu_ly, "n_frame_vao_tracker_doc_duoc": int(len(df_t)),
                "n_frame_bay_khi_dung_khong_xu_ly": n_bay_khi_dung, "n_unreadable": vd.n_unreadable,
                "n_truncated": vd.n_truncated, "n_hang_mot": int(len(df)),
                "n_track": int(df["id"].nunique()) if len(df) else 0},
        "che_do": {"gia_lap": hello["detector"] == "gia_lap", "detector": hello["detector"],
                   "che_do_doi_chieu_regions": bool(a.regions), "regions_sha256": hello.get("regions_sha256"),
                   "render": False, "can_gt": False, "hien_thi": hien_thi, "hien_thi_an": bool(a.hien_thi_an)},
        "hien_thi": ({**(_doc_hien_thi(out) or {"loi": "khong co hien_thi.json tu tien trinh con"}),
                      "lenh_R_da_gui": vd.n_lenh_r, "lenh_credit_da_gui": vd.n_lenh_credit,
                      "ket_qua_lech_frame": vd.n_ket_qua_lech} if hien_thi else {"bat": False}),
        "hello_con": hello, "eos_con": vd.eos, "con": thoat, "lenh_con": cmd,
        "cau_hinh": {"tracker_yaml": yaml, "tracker_yaml_sha256_12": yaml_sha, "tracker_cfg": run_tracking.cfg_to_dict(cfg),
                     "overrides": overrides, "conf_min": a.conf_min, "classes": classes, "class_map": class_map,
                     "max_det": a.max_det, "fps": fps, "dt_nominal_s": dt_nominal,
                     "track_buffer_frames": int(cfg.track_buffer), "track_buffer_sec": a.track_buffer_sec,
                     "elapsed_mode": a.elapsed_mode, "det_conf": a.det_conf, "det_iou": a.det_iou,
                     "det_max_det": a.det_max_det, "imgsz": a.imgsz, "hang_doi": a.hang_doi,
                     "allow_truncated": a.allow_truncated, "cho_phep_frame_hong": a.cho_phep_frame_hong,
                     "gia_lap_cham_tracker_ms": a.gia_lap_cham_tracker_ms},
        "thoi_gian": {**tg, "fps_xu_ly": (n_xu_ly / xu_ly if xu_ly else None), "moc_monotonic": moc,
                      "kiem_cua_so_do": cua_so, "tom_tat_sau_warmup": tt,
                      "dinh_nghia": {
                          "khoi_tao_s": "spawn con -> nhan HELLO (Python con, import, nap engine, lam nong)",
                          "chuan_bi_cha_s": "nhan HELLO -> ngay truoc khi gui BAT DAU (cau hinh tracker, telemetry); "
                                            "con CHUA doc frame nao",
                          "xu_ly_s": "ngay truoc khi gui BAT DAU -> track_video tra ve (frame cuoi da qua tracker, nhan EOS"
                                     " hoac dung); con chi doc frame 1 SAU khi nhan BAT DAU -> moi frame dem nam trong cua so",
                          "fps_xu_ly": "so frame da xu ly / xu_ly_s",
                          "ket_thuc_s": "sau vong lap -> con da thoat (xa frame dang bay khi dung, cho/reap con, dung telemetry)",
                          "ghi_s": "ghi MOT/CSV/cache sau vong lap (do truoc khi ghi run_meta)",
                          "fps_on_dinh": "(N - W) / (t_xong[N] - t_xong[W]) theo moc xong tracker o cha, bo W frame warm-up"},
                      "ghi_chu": "vong 22 thay wall_vong_lap_s/fps_vong_lap cua vong 21 (timer vong 21 bat dau SAU khi con da "
                                 "doc/infer frame dau -- review 9f210e2 N21-F1)"},
        "thiet_bi": {"truoc": truoc, "sau": trang_thai_thiet_bi(), "tegrastats": teg_tt,
                     "tegrastats_log": "tegrastats.log" if teg_tt.get("co_tegrastats") else None},
        "cache": {"cache_id": a.save_cache, "cache_root": str(cache_root),
                  "cache_dir": str(cache_root / a.save_cache) if a.save_cache else None, "info": cache_info},
        "replay_argv": replay, "provenance": provenance(), "python": sys.version.split()[0],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta["thoi_gian"]["tong_s"] = time.perf_counter() - t_main
    meta["thoi_gian"]["dinh_nghia"]["tong_s"] = ("toan luot trong nano_live: sau khi doc tham so -> ngay truoc khi ghi "
                                                 "run_meta (kiem tra, spawn/khoi tao, xu ly, ket thuc, ghi); khong gom "
                                                 "khoi dong Python cua nano_live va thoi gian launcher")
    meta["thoi_gian"]["dinh_nghia"]["tam_dung"] = "khong co chuc nang tam dung -> khong co thoi gian tam dung trong so do"
    if hien_thi:
        meta["thoi_gian"]["fps_hien_thi"] = (meta["hien_thi"] or {}).get("fps_hien_thi")
        meta["thoi_gian"]["dinh_nghia"]["fps_hien_thi"] = ("(N - 1) / (moc hien thi cuoi - moc hien thi dau), N = so anh DA "
                                                           "trinh len cua so (luong hien thi cua con); N < 2 -> null; frame "
                                                           "xem truoc bi bo / toi sau khi dong cua so khong tinh")
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ghi_tien_trinh(prog, {"trang_thai": {"hoan_tat": "hoan_tat", "da_dung": "da_dung"}.get(trang_thai, "loi"),
                          "chi_tiet_trang_thai": trang_thai, "ma_thoat": ma, "frame": n_xu_ly, "n_frame_da_xu_ly": n_xu_ly,
                          "n_frames_khai_bao": hello["n_frames_khai_bao"], "hien_thi": hien_thi,
                          "fps_xu_ly": meta["thoi_gian"]["fps_xu_ly"], "fps_hien_thi": meta["thoi_gian"].get("fps_hien_thi"),
                          "loi": None if ma in (MA_HOAN_TAT, MA_DA_DUNG) else str(chi_tiet)[:300],
                          "run_meta": str((out / "run_meta.json").resolve())})
    fps_x = meta["thoi_gian"]["fps_xu_ly"] or 0.0
    print(f"[nano_live] {trang_thai.upper()} (ma {ma}): {str(chi_tiet).strip().splitlines()[-1] if chi_tiet else ''}")
    print(f"[nano_live] {n_xu_ly} frame xu ly, {len(df)} hang MOT, xu ly {xu_ly or 0:.2f}s ({fps_x:.2f} fps), "
          f"khoi tao {tg['khoi_tao_s']:.1f}s, chuan bi cha {tg['chuan_bi_cha_s'] or 0:.2f}s, ghi {tg['ghi_s']:.2f}s -> {out}")
    return ma


if __name__ == "__main__":
    raise SystemExit(main())
