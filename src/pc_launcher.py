"""
pc_launcher.py -- Giao dien PC toi thieu (Tkinter/ttk) de chay tracking phuong tien tren mot video.

    python src/pc_launcher.py

Launcher KHONG viet lai detector/association/vong doi/vong lap: no chi DIEU PHOI `src/track_video_file.py`
(duong chay da kiem chung) trong mot WORKER PROCESS rieng (argv co cau truc, shell=False), doc tien trinh
tu `--progress-file` va yeu cau dung qua `--stop-file`.

Kien truc:
    LauncherController   -- khong phu thuoc Tk: dung argv, khoi chay/dung/theo doi worker, trang thai.
                            Kiem thu duoc bang script (test_pc_launcher.py).
    LauncherApp (tk.Tk)  -- widget mong, moi 200 ms goi controller.poll() de cap nhat.

Trang thai: SAN_SANG -> DANG_KHOI_TAO -> DANG_CHAY -> (DANG_DUNG ->) HOAN_TAT | DA_DUNG | LOI.
Dung: tao stop-file -> worker hoan tat frame hien tai roi dung qua CUNG duong voi phim q (track_video
break -> generator close -> chot timing -> ghi du tracking/cache/timing). terminate() CHI la duong cuoi
(worker khong phan hoi sau `timeout_dung` giay hoac dong cua so), va duoc ghi vao log.
Moi luot chay tao thu muc rieng <out_base>/<ten_nguon>_<tracker>_<YYYYmmdd-HHMMSS>/ -> khong ghi de.
Chi mot tac vu tai mot thoi diem.

FPS hien thi = fps VONG LAP (ke ca warm-up, ke ca ve/ghi/hien thi neu bat; KHONG gom khoi tao va ghi ket qua)
-- lay tu progress file, KHONG phai fps tracker hay detector rieng; nhan ghi ro pham vi. O trang thai cuoi,
worker ghi n_frame / wall_vong_lap da chot nen bang dung run_meta.thoi_gian.fps_vong_lap (L-F3).
Dung khi dang TAM DUNG (SPACE trong cua so tracking) van toi duoc worker (L-F1). Thanh tien trinh chi theo
frame that (timer indeterminate duoc dung truoc khi chuyen sang determinate, L-F2). Moi truong so tren form
duoc kiem TRUOC khi khoi chay, loi ghi ro ten o (L-F4). Dong cua so khi dang chay: yeu cau dung roi cho bang
after() (GUI van phan hoi), chi destroy sau khi worker ket thuc.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "src" / "track_video_file.py"

#: bo loc: nhan hien thi -> YAML (cung file voi thi nghiem da khoa)
TRACKERS = {"CV  (KF-CV, bytetrack.yaml)": "bytetrack.yaml",
            "EKF (CTRV, configs/bytetrack_ekf_ctrv.yaml)": "configs/bytetrack_ekf_ctrv.yaml",
            "UKF (CTRV, configs/bytetrack_ukf_ctrv.yaml)": "configs/bytetrack_ukf_ctrv.yaml",
            "Khác (chọn YAML ở phần nâng cao)": None}
TRANG_THAI = ("SAN_SANG", "DANG_KHOI_TAO", "DANG_CHAY", "DANG_DUNG", "HOAN_TAT", "DA_DUNG", "LOI")
VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")

#: mac dinh = mac dinh cua track_video_file.py (khong am tham doi nguong/lop/buffer/seed)
MAC_DINH = {"model": str(ROOT / "models" / "yolov8n.pt"), "device": "cpu", "conf_min": "0.25",
            "classes": "1 2 3 5 7", "max_det": "300", "track_buffer_sec": "1.2", "elapsed_mode": "steps",
            "det_conf": "0.05", "det_max_det": "3000", "warmup_frames": "25",
            "out_base": str(ROOT / "results" / "track_video" / "launcher")}


class DangChay(RuntimeError):
    """Da co tac vu dang chay -- launcher chi cho mot tac vu tai mot thoi diem."""


class CauHinhSai(ValueError):
    """Dau vao khong hop le, phat hien TRUOC khi khoi chay worker."""


#: (L-F4) ten o tren giao dien cho tung truong so -> thong bao loi chi ro o can sua
TEN_O = {"conf_min": "conf-min (ngưỡng vào tracker)", "max_det": "max-det", "track_buffer_sec": "track-buffer (giây)",
         "det_conf": "det-conf (ngưỡng detector)", "det_max_det": "det-max-det", "warmup_frames": "warm-up frames",
         "max_frames": "Số frame tối đa", "seed": "Seed", "fps": "FPS"}


def _so(cfg: dict, khoa: str, kieu, *, thap: float | None = None, cao: float | None = None,
        bat_buoc: bool = True, mac_dinh: str | None = None) -> str:
    """(L-F4) Doc mot truong so tu form: parse + kiem mien -> chuoi de dua vao argv.
    Loi -> CauHinhSai ghi TEN O va gia tri da nhap. Truong khong bat buoc de trong -> tra ''."""
    raw = cfg.get(khoa, mac_dinh)
    raw = "" if raw is None else str(raw).strip()
    if raw == "":
        if bat_buoc:
            raise CauHinhSai(f"Ô '{TEN_O.get(khoa, khoa)}' đang trống. Nhập một số.")
        return ""
    try:
        v = kieu(raw)
    except ValueError:
        loai = "số nguyên" if kieu is int else "số"
        raise CauHinhSai(f"Ô '{TEN_O.get(khoa, khoa)}' phải là {loai}"
                         + (", hoặc để trống" if not bat_buoc else "") + f"; đã nhập: {raw!r}") from None
    if (thap is not None and v < thap) or (cao is not None and v > cao):
        mien = (f"≥ {thap:g}" if cao is None else (f"≤ {cao:g}" if thap is None else f"trong [{thap:g}, {cao:g}]"))
        raise CauHinhSai(f"Ô '{TEN_O.get(khoa, khoa)}' phải {mien}; đã nhập: {raw!r}")
    return str(v) if kieu is int else raw


def thiet_bi_ho_tro() -> list[str]:
    """Danh sach --device ma pipeline THUC SU chay duoc tren may nay: 'cpu' luon co; '0' neu torch thay CUDA."""
    ds = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            ds.insert(0, "0")
    except Exception:  # noqa: BLE001
        pass
    return ds


class LauncherController:
    """Dieu phoi worker `track_video_file.py`. Khong phu thuoc Tk."""

    def __init__(self, entry_prefix: list[str] | None = None, timeout_dung: float = 30.0):
        self.entry_prefix = entry_prefix or [sys.executable, str(ENTRY)]
        self.timeout_dung = float(timeout_dung)
        self.proc: subprocess.Popen | None = None
        self.trang_thai = "SAN_SANG"
        self.run_dir: Path | None = None
        self.progress_file: Path | None = None
        self.stop_file: Path | None = None
        self.argv: list[str] = []
        self.log: list[str] = []
        self._q: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self.exit_code: int | None = None
        self.tien_trinh: dict = {}
        self.t_yeu_cau_dung: float | None = None
        self.loi: str | None = None
        self.da_terminate = False

    # ------------------------------------------------------------------ dung argv
    def build_argv(self, cfg: dict) -> tuple[list[str], Path]:
        """Kiem dau vao TRUOC khi khoi chay; tra ve (argv, thu muc ket qua rieng cua luot nay).
        Tham so nang cao duoc truyen DUNG (khong am tham doi)."""
        src = Path(str(cfg.get("source", "")).strip())
        if not str(src) or not src.exists():
            raise CauHinhSai(f"Nguồn không tồn tại: {src}")
        la_thu_muc = src.is_dir()
        if not la_thu_muc and src.suffix.lower() not in VIDEO_EXT:
            raise CauHinhSai(f"Nguồn phải là file video {VIDEO_EXT} hoặc thư mục ảnh: {src.name}")
        fps = str(cfg.get("fps", "")).strip()
        if la_thu_muc and not fps:
            raise CauHinhSai("Thư mục ảnh cần FPS (ô 'FPS').")
        if fps:
            fps = _so(cfg, "fps", float, bat_buoc=False)
            if float(fps) <= 0:
                raise CauHinhSai(f"Ô 'FPS' phải > 0; đã nhập: {fps!r}")
        model = Path(str(cfg.get("model", MAC_DINH["model"])).strip())
        if not model.is_file():
            raise CauHinhSai(f"Checkpoint không tồn tại: {model}")
        yaml = str(cfg.get("tracker_yaml", "")).strip()
        if not yaml:
            raise CauHinhSai("Chưa chọn bộ lọc / YAML tracker.")
        if "/" in yaml or "\\" in yaml:
            yp = Path(yaml)
            if not yp.is_absolute():
                yp = ROOT / yp
            if not yp.is_file():
                raise CauHinhSai(f"YAML tracker không tồn tại: {yaml}")
        manifest = str(cfg.get("frames_manifest", "")).strip()
        if manifest and not Path(manifest).is_file():
            raise CauHinhSai(f"Frames manifest không tồn tại: {manifest}")
        device = str(cfg.get("device", MAC_DINH["device"])).strip() or "cpu"
        out_base = Path(str(cfg.get("out_base", MAC_DINH["out_base"])).strip() or MAC_DINH["out_base"])

        ten = src.name if la_thu_muc else src.stem
        tracker_tag = Path(yaml).stem.replace("bytetrack_", "").replace("bytetrack", "cv")
        run_dir = out_base / f"{ten}_{tracker_tag}_{time.strftime('%Y%m%d-%H%M%S')}"
        k = 1
        while run_dir.exists():                        # khong bao gio ghi de luot truoc
            k += 1
            run_dir = out_base / f"{ten}_{tracker_tag}_{time.strftime('%Y%m%d-%H%M%S')}_{k}"

        # (L-F4) moi truong so duoc parse + kiem mien TRUOC khi khoi chay; loi chi ro o can sua.
        # Gia tri hop le duoc truyen DUNG nhu da nhap (khong lam tron / khong doi nguong).
        so = {k: _so(cfg, k, kieu, thap=th, cao=c, mac_dinh=MAC_DINH[k]) for k, kieu, th, c in (
            ("conf_min", float, 0.0, 1.0), ("max_det", int, 1, None), ("track_buffer_sec", float, 0.0, None),
            ("det_conf", float, 0.0, 1.0), ("det_max_det", int, 1, None), ("warmup_frames", int, 0, None))}
        if float(so["track_buffer_sec"]) <= 0:
            raise CauHinhSai(f"Ô '{TEN_O['track_buffer_sec']}' phải > 0; đã nhập: {so['track_buffer_sec']!r}")
        max_frames = _so(cfg, "max_frames", int, thap=1, bat_buoc=False)
        seed = _so(cfg, "seed", int, thap=0, bat_buoc=False)          # chi PF dung; de trong = mac dinh entry point

        argv = list(self.entry_prefix) + [
            "--source", str(src), "--model", str(model), "--device", device, "--tracker-yaml", yaml,
            "--conf-min", so["conf_min"], "--max-det", so["max_det"], "--track-buffer-sec", so["track_buffer_sec"],
            "--elapsed-mode", str(cfg.get("elapsed_mode", MAC_DINH["elapsed_mode"])).strip(),
            "--det-conf", so["det_conf"], "--det-max-det", so["det_max_det"], "--warmup-frames", so["warmup_frames"],
            "--out", str(run_dir),
            "--progress-file", str(run_dir / "progress.json"), "--progress-every", "10",
            "--stop-file", str(run_dir / "STOP.flag"),
        ]
        classes = str(cfg.get("classes", MAC_DINH["classes"])).split()
        argv += ["--all-classes"] if not classes else ["--classes", *classes]
        if fps:
            argv += ["--fps", fps]
        if manifest:
            argv += ["--frames-manifest", manifest]
        if max_frames:
            argv += ["--max-frames", max_frames]
        if cfg.get("show"):
            argv.append("--show")
        if cfg.get("save_video"):
            argv.append("--save-video")
        if cfg.get("save_cache"):
            argv += ["--save-cache", f"launcher_{ten}_{run_dir.name.rsplit('_', 1)[-1]}"]
        if cfg.get("allow_truncated"):
            argv.append("--allow-truncated")
        for k_, v_ in (cfg.get("set") or {}).items():
            argv += ["--set", f"{k_}={v_}"]
        if seed:
            argv += ["--seed", seed]
        return argv, run_dir

    # ------------------------------------------------------------------ vong doi worker
    def dang_chay(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: dict) -> Path:
        if self.dang_chay():
            raise DangChay("Đang có một tác vụ chạy. Dừng nó trước khi bắt đầu tác vụ mới.")
        argv, run_dir = self.build_argv(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.argv, self.run_dir = argv, run_dir
        self.progress_file, self.stop_file = run_dir / "progress.json", run_dir / "STOP.flag"
        if self.stop_file.exists():
            self.stop_file.unlink()
        self.log, self.tien_trinh, self.exit_code, self.loi = [], {}, None, None
        self.t_yeu_cau_dung, self.da_terminate = None, False
        (run_dir / "launcher_argv.json").write_text(json.dumps(argv, ensure_ascii=False, indent=1), encoding="utf-8")
        self._ghi_log("[launcher] " + subprocess.list2cmdline(argv))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                                     shell=False, creationflags=flags)
        self.trang_thai = "DANG_KHOI_TAO"
        self._reader = threading.Thread(target=self._doc_stdout, daemon=True)
        self._reader.start()
        return run_dir

    def _doc_stdout(self) -> None:
        try:
            for line in self.proc.stdout:          # type: ignore[union-attr]
                self._q.put(line.rstrip("\r\n"))
        except Exception as e:  # noqa: BLE001
            self._q.put(f"[launcher] doc stdout loi: {e}")

    def request_stop(self) -> bool:
        """Yeu cau dung co kiem soat: tao stop-file. Worker hoan tat frame hien tai roi tu ket thuc."""
        if not self.dang_chay() or self.stop_file is None:
            return False
        self.stop_file.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
        self.t_yeu_cau_dung = time.perf_counter()
        self.trang_thai = "DANG_DUNG"
        self._ghi_log("[launcher] da yeu cau dung (stop-file); cho worker hoan tat frame hien tai...")
        return True

    def force_terminate(self, ly_do: str) -> None:
        """DUONG CUOI. Chi khi worker khong phan hoi. Ghi ro vao log; ket qua co the thieu."""
        if self.dang_chay():
            self._ghi_log(f"[launcher] TERMINATE worker ({ly_do}) -- day KHONG phai duong dung thong thuong; "
                          "ket qua cua luot nay co the thieu/khong hop le.")
            self.proc.terminate()               # type: ignore[union-attr]
            try:
                self.proc.wait(timeout=5)       # type: ignore[union-attr]
            except subprocess.TimeoutExpired:
                self.proc.kill()                # type: ignore[union-attr]
            self.da_terminate = True

    def shutdown(self, cho_giay: float | None = None) -> str:
        """Dong launcher: neu dang chay -> yeu cau dung, cho toi `cho_giay`, con song thi terminate (ghi log).
        Tra ve trang thai cuoi. Khong bao gio de worker chay am tham sau khi cua so dong."""
        if not self.dang_chay():
            return self.trang_thai
        cho = self.timeout_dung if cho_giay is None else cho_giay
        self.request_stop()
        t0 = time.perf_counter()
        while self.dang_chay() and time.perf_counter() - t0 < cho:
            self.poll()
            time.sleep(0.1)
        if self.dang_chay():
            self.force_terminate(f"dong cua so, worker chua ket thuc sau {cho:.0f}s")
        self.poll()
        return self.trang_thai

    # ------------------------------------------------------------------ theo doi
    def _ghi_log(self, s: str) -> None:
        self.log.append(s)
        if len(self.log) > 2000:
            del self.log[:500]

    def _doc_tien_trinh(self) -> None:
        if self.progress_file is None or not self.progress_file.exists():
            return
        try:
            self.tien_trinh = json.loads(self.progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass                                  # ghi nguyen tu o worker; neu van loi thi giu ban cu

    def poll(self) -> dict:
        """Goi dinh ky (GUI: moi 200 ms). Khong bao gio chan luong GUI."""
        moi = []
        try:
            while True:
                moi.append(self._q.get_nowait())
        except queue.Empty:
            pass
        for ln in moi:
            self._ghi_log(ln)
        self._doc_tien_trinh()
        tt_worker = self.tien_trinh.get("trang_thai")
        if self.dang_chay():
            if self.trang_thai == "DANG_DUNG":
                if self.t_yeu_cau_dung and time.perf_counter() - self.t_yeu_cau_dung > self.timeout_dung:
                    self.force_terminate(f"khong phan hoi yeu cau dung sau {self.timeout_dung:.0f}s")
            elif tt_worker == "dang_chay":
                self.trang_thai = "DANG_CHAY"
            elif self.trang_thai == "SAN_SANG":
                self.trang_thai = "DANG_KHOI_TAO"
        elif self.proc is not None and self.exit_code is None:
            self.exit_code = self.proc.returncode
            if self._reader is not None:
                self._reader.join(timeout=2)
            try:
                while True:
                    self._ghi_log(self._q.get_nowait())
            except queue.Empty:
                pass
            self._doc_tien_trinh()
            tt_worker = self.tien_trinh.get("trang_thai")
            if self.da_terminate:
                self.trang_thai, self.loi = "LOI", "worker bi terminate (khong phai duong dung thong thuong)"
            elif self.exit_code != 0:
                self.trang_thai = "LOI"
                self.loi = self.tien_trinh.get("loi") or f"worker exit {self.exit_code} -- xem log"
            elif tt_worker == "da_dung":
                self.trang_thai = "DA_DUNG"
            elif tt_worker == "hoan_tat":
                self.trang_thai = "HOAN_TAT"
            else:
                # exit 0 nhung khong co trang thai cuoi trong progress -> KHONG bao "hoan tat" voi
                self.trang_thai, self.loi = "LOI", "worker exit 0 nhung khong ghi trang thai cuoi (progress.json)"
            self._ghi_log(f"[launcher] worker ket thuc: exit={self.exit_code} -> {self.trang_thai}"
                          + (f" ({self.loi})" if self.loi else ""))
        return self.snapshot()

    def snapshot(self) -> dict:
        tt = self.tien_trinh
        return {"trang_thai": self.trang_thai, "frame": tt.get("frame", 0),
                "n_frames_khai_bao": tt.get("n_frames_khai_bao", 0), "fps_vong_lap_tam": tt.get("fps_vong_lap_tam"),
                "tam_dung": bool(tt.get("tam_dung")) and self.trang_thai == "DANG_CHAY",
                "exit_code": self.exit_code, "loi": self.loi, "run_dir": str(self.run_dir) if self.run_dir else None,
                "n_log": len(self.log)}

    # ------------------------------------------------------------------ mo ket qua
    def open_results(self) -> bool:
        if self.run_dir is None or not self.run_dir.exists():
            return False
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self.run_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.run_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.run_dir)])
            return True
        except Exception as e:  # noqa: BLE001
            self._ghi_log(f"[launcher] khong mo duoc thu muc: {e}")
            return False


# ====================================================================== GUI
def _build_app():  # noqa: C901
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class LauncherApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Tracking phương tiện — PC launcher")
            self.minsize(760, 620)
            self.ctl = LauncherController()
            self.v = {k: tk.StringVar(value=v) for k, v in MAC_DINH.items()}
            self.v.update(source=tk.StringVar(), fps=tk.StringVar(), frames_manifest=tk.StringVar(),
                          tracker=tk.StringVar(value=list(TRACKERS)[1]), yaml_khac=tk.StringVar(),
                          seed=tk.StringVar(), max_frames=tk.StringVar())
            self.b = {k: tk.BooleanVar(value=False) for k in ("show", "save_video", "save_cache", "allow_truncated")}
            self._n_log_hien = 0
            self._dang_dong = False
            self._thiet_bi: list[str] | None = None
            self._thiet_bi_da_ap = False
            self._xay()
            self.protocol("WM_DELETE_WINDOW", self._dong)
            threading.Thread(target=self._nap_thiet_bi, daemon=True).start()
            self.after(200, self._tick)

        # ---------------------------------------------------------- widget
        def _hang(self, cha, r, nhan, var, chon=None, rong=58):
            ttk.Label(cha, text=nhan).grid(row=r, column=0, sticky="w", padx=4, pady=2)
            e = ttk.Entry(cha, textvariable=var, width=rong)
            e.grid(row=r, column=1, sticky="we", padx=4)
            if chon:
                ttk.Button(cha, text="…", width=3, command=chon).grid(row=r, column=2, padx=2)
            return e

        def _xay(self):
            f = ttk.LabelFrame(self, text="1. Nguồn")
            f.pack(fill="x", padx=8, pady=4)
            f.columnconfigure(1, weight=1)
            self._hang(f, 0, "Video / thư mục ảnh", self.v["source"], self._chon_nguon)
            self._hang(f, 1, "FPS (bắt buộc với thư mục ảnh)", self.v["fps"], rong=12)
            self._hang(f, 2, "Frames manifest (khi tên ảnh mơ hồ)", self.v["frames_manifest"], self._chon_manifest)
            self._hang(f, 3, "Chỉ xử lý tối đa N frame (trống = hết)", self.v["max_frames"], rong=12)

            f = ttk.LabelFrame(self, text="2. Detector và bộ lọc")
            f.pack(fill="x", padx=8, pady=4)
            f.columnconfigure(1, weight=1)
            self._hang(f, 0, "Checkpoint YOLO (.pt)", self.v["model"], self._chon_model)
            ttk.Label(f, text="Bộ lọc").grid(row=1, column=0, sticky="w", padx=4)
            cb = ttk.Combobox(f, textvariable=self.v["tracker"], values=list(TRACKERS), state="readonly", width=48)
            cb.grid(row=1, column=1, sticky="w", padx=4)
            ttk.Label(f, text="Thiết bị").grid(row=2, column=0, sticky="w", padx=4)
            self.cb_device = ttk.Combobox(f, textvariable=self.v["device"], values=["cpu"], state="readonly", width=10)
            self.cb_device.grid(row=2, column=1, sticky="w", padx=4)
            self.lb_device = ttk.Label(f, text="(đang dò CUDA…)")
            self.lb_device.grid(row=2, column=1, sticky="w", padx=110)

            f = ttk.LabelFrame(self, text="3. Kết quả")
            f.pack(fill="x", padx=8, pady=4)
            f.columnconfigure(1, weight=1)
            self._hang(f, 0, "Thư mục gốc (mỗi lượt tạo thư mục con riêng)", self.v["out_base"], self._chon_out)
            g = ttk.Frame(f)
            g.grid(row=1, column=0, columnspan=3, sticky="w", padx=4)
            for k, t in (("show", "Xem cửa sổ tracking (q: dừng, SPACE: tạm dừng)"),
                         ("save_video", "Lưu video có bbox/ID/lớp"), ("save_cache", "Lưu detection cache (để replay)")):
                ttk.Checkbutton(g, text=t, variable=self.b[k]).pack(side="left", padx=6)

            self.nc = ttk.LabelFrame(self, text="4. Nâng cao (mặc định = track_video_file.py; được ghi vào run_meta.json)")
            self.nc.pack(fill="x", padx=8, pady=4)
            self.nc.columnconfigure(1, weight=1)
            self._hang(self.nc, 0, "YAML tracker khác (khi chọn 'Khác')", self.v["yaml_khac"], self._chon_yaml)
            r = 1
            for k, nhan in (("conf_min", "conf-min (ngưỡng hiệu lực vào tracker)"), ("classes", "classes (COCO id)"),
                            ("max_det", "max-det"), ("track_buffer_sec", "track-buffer-sec"),
                            ("det_conf", "det-conf (thô)"), ("det_max_det", "det-max-det (thô)"),
                            ("warmup_frames", "warmup-frames (chỉ tổng kết thời gian)"), ("seed", "seed (PF)")):
                self._hang(self.nc, r, nhan, self.v[k], rong=16)
                r += 1
            ttk.Checkbutton(self.nc, text="allow-truncated (chấp nhận frame chạm trần det-max-det)",
                            variable=self.b["allow_truncated"]).grid(row=r, column=0, columnspan=2, sticky="w", padx=4)

            f = ttk.Frame(self)
            f.pack(fill="x", padx=8, pady=6)
            self.bt_start = ttk.Button(f, text="▶ Bắt đầu", command=self._bat_dau)
            self.bt_start.pack(side="left", padx=4)
            self.bt_stop = ttk.Button(f, text="■ Dừng (hoàn tất frame hiện tại)", command=self._dung_worker, state="disabled")
            self.bt_stop.pack(side="left", padx=4)
            self.bt_open = ttk.Button(f, text="📂 Mở thư mục kết quả", command=self._mo, state="disabled")
            self.bt_open.pack(side="left", padx=4)
            self.lb_tt = ttk.Label(f, text="SẴN SÀNG", font=("Segoe UI", 10, "bold"))
            self.lb_tt.pack(side="right", padx=8)

            f = ttk.Frame(self)
            f.pack(fill="x", padx=8)
            self.pb = ttk.Progressbar(f, mode="determinate", maximum=100)
            self.pb.pack(fill="x", side="top")
            self.lb_pr = ttk.Label(f, text="")
            self.lb_pr.pack(side="top", anchor="w")

            f = ttk.LabelFrame(self, text="Log worker")
            f.pack(fill="both", expand=True, padx=8, pady=4)
            self.txt = tk.Text(f, height=12, wrap="none", font=("Consolas", 9))
            sb = ttk.Scrollbar(f, command=self.txt.yview)
            self.txt.configure(yscrollcommand=sb.set, state="disabled")
            self.txt.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

        # ---------------------------------------------------------- chon file
        def _chon_nguon(self):
            if messagebox.askyesno("Nguồn", "Chọn FILE video? (Không = chọn THƯ MỤC ảnh)"):
                p = filedialog.askopenfilename(filetypes=[("Video", " ".join("*" + e for e in VIDEO_EXT)), ("Tất cả", "*.*")])
            else:
                p = filedialog.askdirectory()
            if p:
                self.v["source"].set(p)

        def _chon_model(self):
            p = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.pt"), ("Tất cả", "*.*")])
            if p:
                self.v["model"].set(p)

        def _chon_yaml(self):
            p = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
            if p:
                self.v["yaml_khac"].set(p)
                self.v["tracker"].set(list(TRACKERS)[-1])

        def _chon_manifest(self):
            p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if p:
                self.v["frames_manifest"].set(p)

        def _chon_out(self):
            p = filedialog.askdirectory()
            if p:
                self.v["out_base"].set(p)

        def _nap_thiet_bi(self):
            # chay o thread nen (import torch cham); KHONG goi Tk tu thread -- chi ghi ket qua, _tick ap vao widget
            self._thiet_bi = thiet_bi_ho_tro()

        def _ap_thiet_bi(self):
            ds = self._thiet_bi
            if ds is None or self._thiet_bi_da_ap:
                return
            self._thiet_bi_da_ap = True
            self.cb_device.configure(values=ds); self.v["device"].set(ds[0])
            self.lb_device.configure(text="(0 = GPU CUDA)" if "0" in ds else "(không thấy CUDA)")

        # ---------------------------------------------------------- hanh dong
        def _cfg(self) -> dict:
            yaml = TRACKERS[self.v["tracker"].get()] or self.v["yaml_khac"].get().strip()
            c = {k: v.get() for k, v in self.v.items()}
            c.update({k: b.get() for k, b in self.b.items()})
            c["tracker_yaml"] = yaml
            return c

        def _bat_dau(self):
            try:
                run_dir = self.ctl.start(self._cfg())
            except (CauHinhSai, DangChay) as e:
                messagebox.showerror("Không chạy được", str(e))
                return
            except OSError as e:
                messagebox.showerror("Không khởi chạy được worker", str(e))
                return
            except Exception as e:  # noqa: BLE001  (L-F4) khong de exception nao thoat khoi callback nut
                messagebox.showerror("Lỗi không mong đợi khi bắt đầu", f"{type(e).__name__}: {e}")
                return
            self._n_log_hien = 0
            self.txt.configure(state="normal"); self.txt.delete("1.0", "end"); self.txt.configure(state="disabled")
            self.bt_start.configure(state="disabled"); self.bt_stop.configure(state="normal")
            self.bt_open.configure(state="normal")
            self.title(f"Tracking phương tiện — {run_dir.name}")

        def _dung_worker(self):
            if self.ctl.request_stop():
                self.bt_stop.configure(state="disabled")

        def _mo(self):
            if not self.ctl.open_results():
                messagebox.showinfo("Kết quả", "Chưa có thư mục kết quả.")

        def _dong(self):
            """Dong cua so. Dang chay -> yeu cau dung co kiem soat roi CHO BANG after() (GUI van phan hoi,
            _tick van cap nhat trang thai); chi destroy sau khi xac nhan worker da ket thuc. Worker khong
            phan hoi -> poll() terminate sau timeout_dung (duong cuoi) + canh bao ket qua co the thieu."""
            if self._dang_dong:
                return
            if self.ctl.dang_chay():
                if not messagebox.askyesno("Đang chạy", "Worker đang chạy. Yêu cầu dừng có kiểm soát rồi đóng?"):
                    return
                self._dang_dong = True
                self.ctl.request_stop()
                self.bt_start.configure(state="disabled"); self.bt_stop.configure(state="disabled")
                self.after(100, self._cho_dong)
                return
            self._dang_dong = True
            self.destroy()

        def _cho_dong(self):
            if self.ctl.dang_chay():                       # _tick (200 ms) van poll + hien trang thai
                self.after(100, self._cho_dong)
                return
            self.ctl.poll()
            if self.ctl.da_terminate:
                messagebox.showwarning("Worker không phản hồi",
                                       "Đã phải terminate worker (đường cuối). Kết quả lượt này có thể chưa "
                                       f"đầy đủ. Thư mục: {self.ctl.run_dir}")
            self.destroy()

        # ---------------------------------------------------------- cap nhat
        def _tick(self):
            self._ap_thiet_bi()
            s = self.ctl.poll()
            tt = s["trang_thai"]
            nhan = {"SAN_SANG": "SẴN SÀNG", "DANG_KHOI_TAO": "ĐANG KHỞI TẠO (nạp mô hình)…", "DANG_CHAY": "ĐANG CHẠY",
                    "DANG_DUNG": "ĐANG DỪNG (hoàn tất frame hiện tại)…", "HOAN_TAT": "HOÀN TẤT ✔",
                    "DA_DUNG": "ĐÃ DỪNG (kết quả các frame đã xử lý được ghi đủ)", "LOI": "LỖI ✖"}[tt]
            self.lb_tt.configure(text=nhan)
            n, tot = s["frame"], s["n_frames_khai_bao"]
            if tt in ("DANG_CHAY", "DANG_DUNG") and tot > 0:
                if str(self.pb.cget("mode")) != "determinate":
                    # (L-F2) dung timer autoincrement cua che do indeterminate TRUOC (stop() co the dua value
                    # ve 0), roi moi dat maximum/value theo frame THAT tu worker
                    self.pb.stop(); self.pb.configure(mode="determinate")
                self.pb.configure(maximum=tot, value=n)
            elif tt in ("DANG_KHOI_TAO", "DANG_CHAY", "DANG_DUNG"):
                if str(self.pb.cget("mode")) != "indeterminate":
                    self.pb.configure(mode="indeterminate"); self.pb.start(80)
            else:
                self.pb.stop(); self.pb.configure(mode="determinate", maximum=max(tot, 1), value=n)
            fps = s["fps_vong_lap_tam"]
            tam_dung = bool(s.get("tam_dung"))
            self.lb_pr.configure(text=(f"frame {n}" + (f" / {tot}" if tot else " (không biết tổng)"))
                                 + ("   |   TẠM DỪNG (SPACE trong cửa sổ tracking)" if tam_dung else "")
                                 + (f"   |   {fps:.1f} fps vòng lặp (kể cả warm-up, vẽ/ghi/hiển thị nếu bật; không gồm khởi tạo/ghi kết quả; không phải fps tracker)" if fps else "")
                                 + (f"   |   lỗi: {s['loi']}" if s["loi"] else "")
                                 + (f"   |   {s['run_dir']}" if s["run_dir"] and tt in ("HOAN_TAT", "DA_DUNG", "LOI") else ""))
            if len(self.ctl.log) > self._n_log_hien:
                self.txt.configure(state="normal")
                for ln in self.ctl.log[self._n_log_hien:]:
                    self.txt.insert("end", ln + "\n")
                self.txt.see("end"); self.txt.configure(state="disabled")
                self._n_log_hien = len(self.ctl.log)
            if not self.ctl.dang_chay() and tt != "SAN_SANG":
                self.bt_start.configure(state="normal"); self.bt_stop.configure(state="disabled")
            self.after(200, self._tick)

    return LauncherApp


def main() -> int:
    app = _build_app()()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
