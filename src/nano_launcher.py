"""
nano_launcher.py -- Giao dien TAI CHO tren Jetson Nano (man hinh HDMI + chuot + ban phim), vong 25.

    bash scripts/nano/07_giao_dien.sh          # hoac bam bieu tuong "Theo vet phuong tien" tren desktop / menu
    python src/nano_launcher.py --kiem-tra     # chi kiem cau hinh trien khai, in ket qua, khong mo cua so

Chon file video hoac thu muc anh -> chon CV / EKF -> chon thu muc luu -> Bat dau. Cua so "Xem truc tiep" hien bbox, ID,
lop ngay trong luc xu ly (q / Esc hoac dong cua so do = Dung). Nut Dung, phim q/Esc, dong launcher: CUNG mot duong dung
co kiem soat (stop-file) -> ghi du ket qua cac frame da xu ly, metadata phan biet hoan_tat / da_dung / loi.

KHONG viet lai thuat toan: launcher chi DIEU PHOI `src/nano_live.py` (luong TensorRT -> tracker da nghiem thu) trong
WORKER PROCESS rieng (argv la danh sach, shell=False, duong dan co dau cach giu nguyen), tai su dung
`pc_launcher.LauncherController` (khoi chay / dung / theo doi / terminate duong cuoi, khong phu thuoc Tk).

Cau hinh trien khai (engine, Python detector, LD_PRELOAD, thu muc mac dinh): `configs/nano_ui.json` cua goi, ghi de rieng
may trong `~/.config/nano_tracking/nano_ui.json`. Thieu / sai -> thong bao cach khac phuc, nut Bat dau bi khoa.

Moi truong: chay bang Python cua env tracker (`nano`, 3.8, co Tk, KHONG can OpenCV); hien thi do tien trinh detector
(Python 3.6 he thong, OpenCV GTK) dam nhan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pc_launcher as pcl  # noqa: E402   (LauncherController, CauHinhSai, DangChay -- chi stdlib luc import)

ENTRY = ROOT / "src" / "nano_live.py"
CAU_HINH_GOI = ROOT / "configs" / "nano_ui.json"
THU_MUC_ND = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "nano_tracking"
CAU_HINH_ND = THU_MUC_ND / "nano_ui.json"
LAN_CUOI = THU_MUC_ND / "lan_cuoi.json"
VIDEO_EXT = pcl.VIDEO_EXT
TRACKER = {"cv": "CV — Kalman vận tốc không đổi (bytetrack.yaml)",
           "ekf": "EKF — CTRV (configs/bytetrack_ekf_ctrv.yaml)"}
CauHinhSai, DangChay = pcl.CauHinhSai, pcl.DangChay
MA_DA_DUNG = 4


# ====================================================================== cau hinh trien khai
def mo_rong(p) -> Path:
    q = Path(os.path.expanduser(os.path.expandvars(str(p))))
    return q if q.is_absolute() else ROOT / q


def doc_cau_hinh(goi: Path = CAU_HINH_GOI, nd: Path = CAU_HINH_ND) -> dict:
    """Cau hinh goi + ghi de rieng may (chi cac khoa co trong file nguoi dung)."""
    cfg = json.loads(goi.read_text(encoding="utf-8")) if goi.exists() else {}
    nguon = [str(goi)] if goi.exists() else []
    if nd.exists():
        try:
            cfg.update(json.loads(nd.read_text(encoding="utf-8")))
            nguon.append(str(nd))
        except (OSError, ValueError) as e:
            cfg["_loi_file_nguoi_dung"] = f"{nd}: {e}"
    cfg["_nguon"] = nguon
    return cfg


def kiem_trien_khai(cfg: dict) -> dict:
    """Kiem nhanh (khong chay tien trinh): engine, Python detector, LD_PRELOAD, ONNX, entry point. Tra
    {'loi': [...], 'canh_bao': [...], 'engine': Path|None, ...}. Moi loi kem CACH KHAC PHUC (tieng Viet)."""
    loi, cb = [], []
    sua = (f"sửa khóa tương ứng trong {CAU_HINH_ND} (tạo file nếu chưa có, chỉ ghi khóa cần đổi) "
           f"hoặc {CAU_HINH_GOI}")
    if not CAU_HINH_GOI.exists():
        loi.append(f"Thiếu file cấu hình gói {CAU_HINH_GOI}. Giải nén lại gói nano_transfer.")
    if cfg.get("_loi_file_nguoi_dung"):
        loi.append(f"File cấu hình riêng hỏng: {cfg['_loi_file_nguoi_dung']}. Sửa JSON hoặc xóa file đó.")
    ung_vien = cfg.get("engine") or []
    ung_vien = [ung_vien] if isinstance(ung_vien, str) else list(ung_vien)
    engine = next((mo_rong(p) for p in ung_vien if mo_rong(p).is_file()), None)
    if engine is None:
        loi.append("Không thấy engine TensorRT (đã tìm: " + ", ".join(str(mo_rong(p)) for p in ung_vien) + "). "
                   "Cách khắc phục: build engine trên chính Nano bằng `bash scripts/nano/01_build_engine.sh` "
                   f"(khoảng 8 phút) rồi {sua} (khóa \"engine\").")
    py_det = str(cfg.get("python_detector") or "/usr/bin/python3")
    if not (Path(py_det).is_file() or shutil.which(py_det)):
        loi.append(f"Không thấy Python cho detector: {py_det} (cần Python 3.6 hệ thống có TensorRT + OpenCV). {sua}.")
    ld = cfg.get("ld_preload") or ""
    if ld and not Path(ld).is_file():
        (loi if Path("/etc/nv_tegra_release").exists() else cb).append(
            f"Không thấy thư viện LD_PRELOAD {ld} (cần trên Jetson để import cv2 sau tensorrt). {sua}.")
    onnx = mo_rong(cfg.get("onnx") or "models/export/yolov8n_640_op12.onnx")
    if not onnx.is_file():
        cb.append(f"Không thấy ONNX {onnx}: run_meta sẽ không ghi onnx_sha256 (không ảnh hưởng chạy).")
    if not ENTRY.is_file():
        loi.append(f"Thiếu {ENTRY}: gói không đầy đủ, giải nén lại.")
    return {"loi": loi, "canh_bao": cb, "engine": engine, "python_detector": py_det, "ld_preload": ld,
            "onnx": onnx if onnx.is_file() else None, "nguon_cau_hinh": cfg.get("_nguon", [])}


def kiem_python_detector(py_det: str, ld_preload: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Kiem THAT Python detector import duoc tensorrt + cv2 (cham ~ vai giay -> chay o thread nen)."""
    env = dict(os.environ)
    if ld_preload and Path(ld_preload).is_file():
        env["LD_PRELOAD"] = ld_preload
    code = "import tensorrt, cv2, numpy; print(tensorrt.__version__, cv2.__version__, numpy.__version__)"
    try:
        r = subprocess.run([py_det, "-c", code], capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    return True, "TensorRT {} | OpenCV {} | NumPy {}".format(*r.stdout.split()[:3])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ghi_thu(d: Path) -> str | None:
    """Kiem quyen ghi thu muc ket qua (tao neu chua co). Tra loi (tieng Viet) hoac None."""
    try:
        d.mkdir(parents=True, exist_ok=True)
        t = d / f".kiem_ghi_{os.getpid()}"
        t.write_text("ok", encoding="utf-8")
        t.unlink()
        return None
    except OSError as e:
        return f"Không ghi được vào thư mục kết quả {d}: {e}. Chọn thư mục khác (trong thư mục nhà)."


# ====================================================================== controller (khong Tk)
class NanoLauncherController(pcl.LauncherController):
    """Dieu phoi worker `nano_live.py`. Ke thua vong doi cua launcher PC; them: ma 4 = dung co kiem soat,
    so do Nano trong snapshot, `launcher.json` (thoi gian toan luot phia launcher) khi worker ket thuc."""

    def __init__(self, cau_hinh: dict, trien_khai: dict | None = None, entry_prefix: list[str] | None = None,
                 timeout_dung: float | None = None, argv_them: list[str] | None = None):
        super().__init__(entry_prefix or [sys.executable, str(ENTRY)],
                         float(timeout_dung or cau_hinh.get("timeout_dung_s") or 60.0))
        self.cau_hinh = cau_hinh
        self.trien_khai = trien_khai if trien_khai is not None else kiem_trien_khai(cau_hinh)
        self.argv_them = list(argv_them or [])
        self.t_start: float | None = None
        self.t_start_iso: str | None = None
        self._da_ghi_launcher_json = False
        self._onnx_sha: str | None = None

    def build_argv(self, cfg: dict) -> tuple[list[str], Path]:
        tk = self.trien_khai
        if tk["loi"]:
            raise CauHinhSai("Cấu hình triển khai chưa đúng:\n- " + "\n- ".join(tk["loi"]))
        raw = str(cfg.get("source", "")).strip()
        if not raw:
            raise CauHinhSai("Chưa chọn nguồn: bấm 'Chọn video…' hoặc 'Chọn thư mục ảnh…'.")
        src = Path(raw)
        if not src.exists():
            raise CauHinhSai(f"Nguồn không tồn tại: {src}")
        la_thu_muc = src.is_dir()
        if not la_thu_muc and src.suffix.lower() not in VIDEO_EXT:
            raise CauHinhSai(f"Nguồn phải là file video ({' '.join(VIDEO_EXT)}) hoặc thư mục ảnh: {src.name}")
        fps = str(cfg.get("fps", "")).strip() if la_thu_muc else ""
        if la_thu_muc:
            if not fps:
                raise CauHinhSai("Thư mục ảnh cần FPS (ô 'FPS'), ví dụ 25 cho UA-DETRAC.")
            fps = pcl._so({"fps": fps}, "fps", float, bat_buoc=True)
            if float(fps) <= 0:
                raise CauHinhSai(f"Ô 'FPS' phải > 0; đã nhập: {fps!r}")
        max_frames = pcl._so(cfg, "max_frames", int, thap=1, bat_buoc=False)
        tracker = str(cfg.get("tracker", "")).strip()
        if tracker not in TRACKER:
            raise CauHinhSai("Chọn bộ lọc CV hoặc EKF.")
        out_base = Path(str(cfg.get("out_base", "")).strip() or mo_rong(self.cau_hinh.get("thu_muc_ket_qua", "~/nano_ket_qua")))
        e = ghi_thu(out_base)
        if e:
            raise CauHinhSai(e)
        ten = src.name if la_thu_muc else src.stem
        run_dir = out_base / f"{ten}_{tracker}_{time.strftime('%Y%m%d-%H%M%S')}"
        k = 1
        while run_dir.exists():                         # moi luot mot thu muc MOI, khong ghi de
            k += 1
            run_dir = out_base / f"{ten}_{tracker}_{time.strftime('%Y%m%d-%H%M%S')}_{k}"
        argv = list(self.entry_prefix) + [
            "--source", str(src), "--tracker", tracker, "--engine", str(tk["engine"]),
            "--python-detector", tk["python_detector"], "--out", str(run_dir),
            "--progress-file", str(run_dir / "progress.json"), "--stop-file", str(run_dir / "STOP.flag"),
            "--progress-every", "50"]
        if tk["ld_preload"]:
            argv += ["--ld-preload", tk["ld_preload"]]
        if tk["onnx"] is not None:
            if self._onnx_sha is None:
                self._onnx_sha = sha256_file(tk["onnx"])
            argv += ["--onnx-sha256", self._onnx_sha]
        if fps:
            argv += ["--fps", fps]
        if max_frames:
            argv += ["--max-frames", max_frames]
        if cfg.get("hien_thi"):
            argv.append("--hien-thi")
            kt = self.cau_hinh.get("hien_thi_kich_thuoc")
            if kt:
                argv += ["--hien-thi-kich-thuoc", *[str(int(v)) for v in kt]]
        if cfg.get("luu_cache"):
            argv += ["--save-cache", f"cache_{run_dir.name}", "--cache-root", str(run_dir / "det_cache")]
        argv += self.argv_them
        return argv, run_dir

    def start(self, cfg: dict) -> Path:
        run_dir = super().start(cfg)
        self.t_start, self.t_start_iso = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S")
        self._da_ghi_launcher_json = False
        return run_dir

    def _ket_luan_cuoi(self, tt_worker):
        # nano_live: ma 4 + progress 'da_dung' = dung co kiem soat (ket qua cac frame da xu ly ghi du)
        if not self.da_terminate and self.exit_code == MA_DA_DUNG and tt_worker == "da_dung":
            return "DA_DUNG", None
        return super()._ket_luan_cuoi(tt_worker)

    def poll(self) -> dict:
        s = super().poll()
        if self.exit_code is not None and not self._da_ghi_launcher_json and self.run_dir is not None:
            self._da_ghi_launcher_json = True
            tong = time.perf_counter() - self.t_start if self.t_start else None
            info = {"trang_thai": self.trang_thai, "exit_code": self.exit_code, "loi": self.loi,
                    "da_terminate": self.da_terminate, "bat_dau_luc": self.t_start_iso,
                    "ket_thuc_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "tong_s_phia_launcher": tong,
                    "dinh_nghia": "tong_s_phia_launcher: tu luc bam Bat dau (khoi chay worker) den luc launcher thay worker "
                                  "da thoat (gom khoi dong Python, nap engine, xu ly, dung/ghi ket qua; do tre poll <= 0,2 s); "
                                  "khong co tam dung",
                    "argv": self.argv}
            try:
                (self.run_dir / "launcher.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
            except OSError as e:
                self._ghi_log(f"[launcher] khong ghi duoc launcher.json: {e}")
            s = self.snapshot()
        return s

    def snapshot(self) -> dict:
        s = super().snapshot()
        tt = self.tien_trinh
        s.update(fps_xu_ly_tam=tt.get("fps_xu_ly_tam"), fps_xu_ly=tt.get("fps_xu_ly"), fps_hien_thi=tt.get("fps_hien_thi"),
                 n_frame_da_xu_ly=tt.get("n_frame_da_xu_ly"), hien_thi=tt.get("hien_thi"),
                 chi_tiet_trang_thai=tt.get("chi_tiet_trang_thai"))
        return s

    def tom_tat_ket_qua(self) -> str | None:
        """Mot doan tieng Viet tu run_meta.json + launcher.json cua luot vua xong (None neu chua co)."""
        if self.run_dir is None or not (self.run_dir / "run_meta.json").exists():
            return None
        try:
            m = json.loads((self.run_dir / "run_meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        d, tg, ht = m.get("dem") or {}, m.get("thoi_gian") or {}, m.get("hien_thi") or {}
        tong_l = None
        try:
            tong_l = json.loads((self.run_dir / "launcher.json").read_text(encoding="utf-8")).get("tong_s_phia_launcher")
        except (OSError, ValueError):
            pass
        dong = [f"{d.get('n_frame_da_xu_ly', '?')} frame đã xử lý / {d.get('n_frames_khai_bao', '?')} khai báo; "
                f"{d.get('n_hang_mot', '?')} hàng MOT, {d.get('n_track', '?')} ID."]
        if tg.get("fps_xu_ly"):
            dong.append(f"FPS xử lý {tg['fps_xu_ly']:.2f} (frame / thời gian vòng xử lý, không gồm khởi tạo và ghi).")
        if ht.get("bat"):
            dem = ht.get("dem") or {}
            f = ht.get("fps_hien_thi")
            dong.append(f"Hiển thị: {dem.get('n_hien_thi', 0)} frame đã hiện"
                        + (f", {f:.2f} FPS hiển thị" if f else "")
                        + f", {dem.get('n_bo_hien_thi', 0)} frame xem trước bị bỏ (tracking vẫn xử lý đủ).")
        if tg.get("tong_s") is not None:
            dong.append(f"Thời gian toàn lượt {tg['tong_s']:.1f} s (nano_live)"
                        + (f", {tong_l:.1f} s tính từ lúc bấm Bắt đầu" if tong_l else "") + "; không có tạm dừng.")
        dong.append(f"Kết quả: {self.run_dir}")
        return "\n".join(dong)


# ====================================================================== nho lua chon lan truoc (tien loi)
def doc_lan_cuoi() -> dict:
    try:
        return json.loads(LAN_CUOI.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ghi_lan_cuoi(d: dict) -> None:
    try:
        THU_MUC_ND.mkdir(parents=True, exist_ok=True)
        LAN_CUOI.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


# ====================================================================== GUI
def _build_app():  # noqa: C901
    import tkinter as tk
    from tkinter import filedialog, font, messagebox, ttk

    class NanoLauncherApp(tk.Tk):
        def __init__(self, cau_hinh: dict, trien_khai: dict, tu_dong: dict):
            super().__init__()
            self.title("Theo vết phương tiện — Jetson Nano")
            for ten in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    font.nametofont(ten).configure(size=12)
                except tk.TclError:
                    pass
            self.minsize(900, 700)
            self.cau_hinh, self.trien_khai, self.tu_dong = cau_hinh, trien_khai, tu_dong
            self.ctl = NanoLauncherController(cau_hinh, trien_khai)
            lc = doc_lan_cuoi()
            self.v = {"source": tk.StringVar(value=tu_dong.get("nguon") or lc.get("source", "")),
                      "fps": tk.StringVar(value=str(tu_dong.get("fps") or lc.get("fps") or cau_hinh.get("fps_thu_muc_anh", 25))),
                      "max_frames": tk.StringVar(value=str(tu_dong.get("max_frames") or "")),
                      "tracker": tk.StringVar(value=tu_dong.get("tracker") or lc.get("tracker", "cv")),
                      "out_base": tk.StringVar(value=tu_dong.get("ket_qua") or lc.get("out_base")
                                               or str(mo_rong(cau_hinh.get("thu_muc_ket_qua", "~/nano_ket_qua"))))}
            ht_md = lc.get("hien_thi", cau_hinh.get("hien_thi_mac_dinh", True))
            self.b = {"hien_thi": tk.BooleanVar(value=False if tu_dong.get("khong_hien_thi") else bool(ht_md)),
                      "luu_cache": tk.BooleanVar(value=bool(tu_dong.get("luu_cache") or lc.get("luu_cache",
                                                                   cau_hinh.get("luu_cache_mac_dinh", False))))}
            self._n_log = 0
            self._dang_dong = False
            self._chon_dang_mo = False
            self._py_det_ok: bool | None = None
            self._py_det_msg = "đang kiểm tra Python detector…"
            self._tu_dong_da = set()
            self._t_dang_chay: float | None = None
            self._xay()
            self.protocol("WM_DELETE_WINDOW", self._dong)
            if not trien_khai["loi"]:
                threading.Thread(target=self._kiem_py_det, daemon=True).start()
            self.after(200, self._tick)

        # ---------------------------------------------------------------- widget
        def _xay(self):
            pad = {"padx": 8, "pady": 4}
            self.lb_mt = ttk.Label(self, text="", justify="left", wraplength=860)
            self.lb_mt.pack(fill="x", **pad)

            f = ttk.LabelFrame(self, text="1. Dữ liệu đầu vào (video hoặc thư mục ảnh có sẵn)")
            f.pack(fill="x", **pad)
            f.columnconfigure(1, weight=1)
            ttk.Label(f, text="Nguồn").grid(row=0, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(f, textvariable=self.v["source"]).grid(row=0, column=1, sticky="we", padx=4)
            g = ttk.Frame(f)
            g.grid(row=1, column=1, sticky="w", padx=4)
            self.bt_video = ttk.Button(g, text="Chọn video…", command=lambda: self._chon("video"))
            self.bt_video.pack(side="left", padx=2)
            self.bt_anh = ttk.Button(g, text="Chọn thư mục ảnh…", command=lambda: self._chon("thu_muc_anh"))
            self.bt_anh.pack(side="left", padx=2)
            ttk.Label(f, text="FPS").grid(row=2, column=0, sticky="w", padx=4, pady=3)
            h = ttk.Frame(f)
            h.grid(row=2, column=1, sticky="w", padx=4)
            self.en_fps = ttk.Entry(h, textvariable=self.v["fps"], width=8)
            self.en_fps.pack(side="left")
            ttk.Label(h, text="  (chỉ dùng cho thư mục ảnh; video lấy FPS trong file)").pack(side="left")
            ttk.Label(f, text="Số frame tối đa").grid(row=3, column=0, sticky="w", padx=4, pady=3)
            h = ttk.Frame(f)
            h.grid(row=3, column=1, sticky="w", padx=4)
            ttk.Entry(h, textvariable=self.v["max_frames"], width=8).pack(side="left")
            ttk.Label(h, text="  (để trống = chạy hết nguồn)").pack(side="left")

            f = ttk.LabelFrame(self, text="2. Bộ lọc")
            f.pack(fill="x", **pad)
            for k, nhan in TRACKER.items():
                ttk.Radiobutton(f, text=nhan, value=k, variable=self.v["tracker"]).pack(anchor="w", padx=6, pady=2)

            f = ttk.LabelFrame(self, text="3. Lưu kết quả và hiển thị")
            f.pack(fill="x", **pad)
            f.columnconfigure(1, weight=1)
            ttk.Label(f, text="Thư mục gốc").grid(row=0, column=0, sticky="w", padx=4, pady=3)
            ttk.Entry(f, textvariable=self.v["out_base"]).grid(row=0, column=1, sticky="we", padx=4)
            self.bt_out = ttk.Button(f, text="Chọn…", command=lambda: self._chon("ket_qua"))
            self.bt_out.grid(row=0, column=2, padx=4)
            ttk.Label(f, text="(mỗi lượt chạy tạo một thư mục con mới, không ghi đè)").grid(
                row=1, column=1, sticky="w", padx=4)
            ttk.Checkbutton(f, text="Hiển thị trực tiếp: bbox, ID, lớp trong lúc xử lý (phím q hoặc Esc trong cửa sổ xem = Dừng)",
                            variable=self.b["hien_thi"]).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=2)
            ttk.Checkbutton(f, text="Lưu detection cache (để replay / kiểm lại tracking)",
                            variable=self.b["luu_cache"]).grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=2)

            f = ttk.Frame(self)
            f.pack(fill="x", **pad)
            self.bt_start = ttk.Button(f, text="▶  Bắt đầu", command=self._bat_dau)
            self.bt_start.pack(side="left", padx=4)
            if self.trien_khai["loi"]:
                self.bt_start.configure(state="disabled")      # cau hinh trien khai sai -> khong cho bat dau
            self.bt_stop = ttk.Button(f, text="■  Dừng", command=self._dung, state="disabled")
            self.bt_stop.pack(side="left", padx=4)
            self.bt_open = ttk.Button(f, text="Mở thư mục kết quả", command=self._mo, state="disabled")
            self.bt_open.pack(side="left", padx=4)
            ttk.Button(f, text="Thoát", command=self._dong).pack(side="right", padx=4)

            f = ttk.LabelFrame(self, text="Trạng thái")
            f.pack(fill="x", **pad)
            self.lb_tt = ttk.Label(f, text="SẴN SÀNG", font=("DejaVu Sans", 14, "bold"))
            self.lb_tt.pack(anchor="w", padx=6)
            self.pb = ttk.Progressbar(f, mode="determinate", maximum=100)
            self.pb.pack(fill="x", padx=6, pady=3)
            self.lb_pr = ttk.Label(f, text="", justify="left", wraplength=860)
            self.lb_pr.pack(anchor="w", padx=6)
            self.lb_kq = ttk.Label(f, text="", justify="left", wraplength=860)
            self.lb_kq.pack(anchor="w", padx=6, pady=3)

            f = ttk.LabelFrame(self, text="Nhật ký")
            f.pack(fill="both", expand=True, **pad)
            self.txt = tk.Text(f, height=8, wrap="none", font=("DejaVu Sans Mono", 9))
            sb = ttk.Scrollbar(f, command=self.txt.yview)
            self.txt.configure(yscrollcommand=sb.set, state="disabled")
            self.txt.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            self._cap_nhat_mt()

        def _cap_nhat_mt(self):
            tk_ = self.trien_khai
            if tk_["loi"]:
                s = "✖ Chưa chạy được — cấu hình triển khai:\n- " + "\n- ".join(tk_["loi"])
            else:
                s = (f"Engine: {tk_['engine']}   |   Python detector: {tk_['python_detector']} — "
                     + ("✔ " if self._py_det_ok else ("✖ " if self._py_det_ok is False else "")) + self._py_det_msg)
            if tk_["canh_bao"]:
                s += "\nLưu ý: " + " ".join(tk_["canh_bao"])
            self.lb_mt.configure(text=s, foreground="#b00000" if (tk_["loi"] or self._py_det_ok is False) else "")

        def _kiem_py_det(self):
            # thread nen: KHONG goi Tk tu day, chi ghi ket qua; _tick ap vao widget
            ok, msg = kiem_python_detector(self.trien_khai["python_detector"], self.trien_khai["ld_preload"])
            self._py_det_ok, self._py_det_msg = ok, (msg if ok else f"lỗi import: {msg}. Kiểm tra TensorRT/OpenCV hệ thống.")

        # ---------------------------------------------------------------- chon file (zenity neu co, nguoc lai Tk)
        def _chon(self, loai: str):
            if self._chon_dang_mo:
                return
            dau = self.v["source"].get() if loai != "ket_qua" else self.v["out_base"].get()
            dau = dau or str(mo_rong(self.cau_hinh.get("thu_muc_du_lieu", "~")))
            if Path(dau).is_file():
                dau = str(Path(dau).parent)
            z = shutil.which("zenity")
            if z:
                cmd = [z, "--file-selection", "--filename", str(Path(dau)) + "/"]
                if loai == "video":
                    cmd += ["--title", "Chọn file video", "--file-filter",
                            "Video | " + " ".join("*" + e for e in VIDEO_EXT), "--file-filter", "Tất cả | *"]
                else:
                    cmd += ["--directory", "--title",
                            "Chọn thư mục ảnh" if loai == "thu_muc_anh" else "Chọn thư mục gốc lưu kết quả"]
                self._chon_dang_mo = True
                kq: dict = {}

                def chay():
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        kq["p"] = r.stdout.strip() if r.returncode == 0 else ""
                    except OSError:
                        kq["p"] = None
                threading.Thread(target=chay, daemon=True).start()

                def cho():
                    if "p" not in kq:
                        self.after(100, cho)
                        return
                    self._chon_dang_mo = False
                    if kq["p"] is None:
                        self._chon_tk(loai, dau)
                    elif kq["p"]:
                        self._dat_chon(loai, kq["p"])
                self.after(100, cho)
                return
            self._chon_tk(loai, dau)

        def _chon_tk(self, loai, dau):
            if loai == "video":
                p = filedialog.askopenfilename(initialdir=dau, title="Chọn file video",
                                               filetypes=[("Video", " ".join("*" + e for e in VIDEO_EXT)), ("Tất cả", "*")])
            else:
                p = filedialog.askdirectory(initialdir=dau, title="Chọn thư mục")
            if p:
                self._dat_chon(loai, p)

        def _dat_chon(self, loai, p):
            self.v["out_base" if loai == "ket_qua" else "source"].set(p)

        # ---------------------------------------------------------------- hanh dong
        def _cfg(self) -> dict:
            c = {k: v.get() for k, v in self.v.items()}
            c.update({k: b.get() for k, b in self.b.items()})
            return c

        def _bat_dau(self):
            if self._py_det_ok is False:
                messagebox.showerror("Chưa chạy được", "Python detector lỗi: " + self._py_det_msg)
                return
            try:
                run_dir = self.ctl.start(self._cfg())
            except (CauHinhSai, DangChay) as e:
                messagebox.showerror("Không chạy được", str(e))
                return
            except Exception as e:  # noqa: BLE001  -- khong de exception thoat khoi callback nut
                messagebox.showerror("Lỗi không mong đợi khi bắt đầu", f"{type(e).__name__}: {e}")
                return
            c = self._cfg()
            ghi_lan_cuoi({k: c[k] for k in ("source", "fps", "tracker", "out_base", "hien_thi", "luu_cache")})
            self._n_log = 0
            self.txt.configure(state="normal"); self.txt.delete("1.0", "end"); self.txt.configure(state="disabled")
            self.lb_kq.configure(text="")
            self._t_dang_chay = None
            self.bt_start.configure(state="disabled"); self.bt_stop.configure(state="normal")
            self.bt_open.configure(state="normal")
            self.title(f"Theo vết phương tiện — {run_dir.name}")

        def _dung(self):
            if self.ctl.request_stop():
                self.bt_stop.configure(state="disabled")

        def _mo(self):
            if not self.ctl.open_results():
                messagebox.showinfo("Kết quả", "Chưa có thư mục kết quả.")

        def _dong(self):
            """Dong launcher. Dang chay -> DUNG CO KIEM SOAT (stop-file) roi cho bang after() (giao dien van phan hoi);
            chi dong khi worker da ket thuc. Worker khong phan hoi sau timeout -> terminate (duong cuoi, ghi log)."""
            if self._dang_dong:
                return
            self._dang_dong = True
            if self.ctl.dang_chay():
                self.ctl.request_stop()
                self.bt_start.configure(state="disabled"); self.bt_stop.configure(state="disabled")
                self.lb_tt.configure(text="ĐANG DỪNG ĐỂ ĐÓNG (ghi kết quả các frame đã xử lý)…")
                self.after(100, self._cho_dong)
                return
            self.destroy()

        def _cho_dong(self):
            if self.ctl.dang_chay():
                self.after(100, self._cho_dong)
                return
            self.ctl.poll()
            if self.ctl.da_terminate:
                messagebox.showwarning("Worker không phản hồi", "Đã phải terminate worker (đường cuối). Kết quả lượt "
                                       f"này có thể chưa đầy đủ. Thư mục: {self.ctl.run_dir}")
            self.destroy()

        # ---------------------------------------------------------------- tu dong (CHI de kiem thu tren thiet bi)
        def _tu_dong_buoc(self, tt: str):
            td = self.tu_dong
            if not td:
                return
            if td.get("bat_dau") and "bd" not in self._tu_dong_da and self._py_det_ok is not None and tt == "SAN_SANG":
                self._tu_dong_da.add("bd")
                self._ghi_log_gui(f"[tu dong] bam Bat dau ({time.strftime('%H:%M:%S')})")
                self._bat_dau()
                return
            if tt == "DANG_CHAY" and self._t_dang_chay is None:
                self._t_dang_chay = time.perf_counter()
            dang = self._t_dang_chay is not None and tt == "DANG_CHAY"
            if td.get("dung_sau") is not None and dang and "dung" not in self._tu_dong_da \
                    and time.perf_counter() - self._t_dang_chay >= float(td["dung_sau"]):
                self._tu_dong_da.add("dung")
                self._ghi_log_gui("[tu dong] bam Dung")
                self._dung()
            if td.get("dong_sau") is not None and dang and "dong" not in self._tu_dong_da \
                    and time.perf_counter() - self._t_dang_chay >= float(td["dong_sau"]):
                self._tu_dong_da.add("dong")
                self._ghi_log_gui("[tu dong] dong cua so launcher (WM_DELETE_WINDOW)")
                self._dong()
            if td.get("dong_khi_xong") and tt in ("HOAN_TAT", "DA_DUNG", "LOI") and "xong" not in self._tu_dong_da:
                self._tu_dong_da.add("xong")
                self.after(3000, self._dong)

        def _ghi_log_gui(self, s: str):
            self.ctl._ghi_log(s)

        # ---------------------------------------------------------------- cap nhat dinh ky
        def _tick(self):
            s = self.ctl.poll()
            tt = s["trang_thai"]
            self._cap_nhat_mt()
            if not self._dang_dong:
                nhan = {"SAN_SANG": "SẴN SÀNG", "DANG_KHOI_TAO": "ĐANG KHỞI TẠO (nạp engine TensorRT, ~10 giây)…",
                        "DANG_CHAY": "ĐANG CHẠY", "DANG_DUNG": "ĐANG DỪNG (hoàn tất frame hiện tại, ghi kết quả)…",
                        "HOAN_TAT": "HOÀN TẤT ✔", "DA_DUNG": "ĐÃ DỪNG theo yêu cầu (kết quả các frame đã xử lý được ghi đủ)",
                        "LOI": "LỖI ✖"}[tt]
                self.lb_tt.configure(text=nhan, foreground={"LOI": "#b00000", "HOAN_TAT": "#006000"}.get(tt, ""))
            n, tot = s["frame"], s["n_frames_khai_bao"]
            if tt in ("DANG_CHAY", "DANG_DUNG") and tot:
                if str(self.pb.cget("mode")) != "determinate":
                    self.pb.stop(); self.pb.configure(mode="determinate")
                self.pb.configure(maximum=tot, value=n)
            elif tt in ("DANG_KHOI_TAO", "DANG_CHAY", "DANG_DUNG"):
                if str(self.pb.cget("mode")) != "indeterminate":
                    self.pb.configure(mode="indeterminate"); self.pb.start(80)
            else:
                self.pb.stop(); self.pb.configure(mode="determinate", maximum=max(tot or 1, 1), value=n or 0)
            chu = ""
            if tt in ("DANG_CHAY", "DANG_DUNG", "HOAN_TAT", "DA_DUNG") or n:
                chu = f"frame {n}" + (f" / {tot}" if tot else "")
                f = s.get("fps_xu_ly") or s.get("fps_xu_ly_tam")
                if f:
                    chu += f"   |   {f:.1f} FPS xử lý (TensorRT + tracker; không phải FPS danh nghĩa của video)"
            if s["loi"]:
                chu += f"\nLỗi: {s['loi']}"
            self.lb_pr.configure(text=chu)
            if tt in ("HOAN_TAT", "DA_DUNG", "LOI") and not self.lb_kq.cget("text"):
                kq = self.ctl.tom_tat_ket_qua()
                if kq:
                    self.lb_kq.configure(text=kq)
            if len(self.ctl.log) > self._n_log:
                self.txt.configure(state="normal")
                for ln in self.ctl.log[self._n_log:]:
                    self.txt.insert("end", ln + "\n")
                self.txt.see("end"); self.txt.configure(state="disabled")
                self._n_log = len(self.ctl.log)
            if not self.ctl.dang_chay() and tt != "SAN_SANG" and not self._dang_dong:
                ok = not self.trien_khai["loi"]
                self.bt_start.configure(state="normal" if ok else "disabled"); self.bt_stop.configure(state="disabled")
            if self.trien_khai["loi"]:
                self.bt_start.configure(state="disabled")
            self._tu_dong_buoc(tt)
            self.after(200, self._tick)

    return NanoLauncherApp


def main() -> int:
    ap = argparse.ArgumentParser(description="Giao dien tai cho tren Jetson Nano: chon nguon, chay CV/EKF, xem truc tiep")
    ap.add_argument("--kiem-tra", action="store_true", help="chi kiem cau hinh trien khai + Python detector, in ra, thoat")
    t = ap.add_argument_group("dien san / TU DONG (chi de kiem thu tren thiet bi; nguoi dung khong can)")
    t.add_argument("--nguon", default=None)
    t.add_argument("--tracker", choices=sorted(TRACKER), default=None)
    t.add_argument("--ket-qua", default=None, help="thu muc goc luu ket qua")
    t.add_argument("--fps", default=None)
    t.add_argument("--max-frames", default=None)
    t.add_argument("--khong-hien-thi", action="store_true")
    t.add_argument("--luu-cache", action="store_true")
    t.add_argument("--tu-dong-bat-dau", action="store_true", help="bam Bat dau ngay khi kiem moi truong xong")
    t.add_argument("--tu-dong-dung-sau", type=float, default=None, help="bam Dung sau S giay DANG CHAY")
    t.add_argument("--tu-dong-dong-sau", type=float, default=None, help="dong cua so launcher sau S giay DANG CHAY")
    t.add_argument("--tu-dong-dong-khi-xong", action="store_true", help="dong launcher 3 s sau khi luot ket thuc")
    a = ap.parse_args()
    os.environ["BYTETRACK_BACKEND"] = "np"              # worker nano_live: backend tracker NumPy (nhu script 04)
    cau_hinh = doc_cau_hinh()
    trien_khai = kiem_trien_khai(cau_hinh)
    if a.kiem_tra:
        print(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in trien_khai.items()},
                         ensure_ascii=False, indent=1))
        if not trien_khai["loi"]:
            ok, msg = kiem_python_detector(trien_khai["python_detector"], trien_khai["ld_preload"])
            print(f"python detector: {'OK' if ok else 'LOI'} -- {msg}")
            return 0 if ok else 1
        return 1
    tu_dong = {"nguon": a.nguon, "tracker": a.tracker, "ket_qua": a.ket_qua, "fps": a.fps, "max_frames": a.max_frames,
               "khong_hien_thi": a.khong_hien_thi, "luu_cache": a.luu_cache, "bat_dau": a.tu_dong_bat_dau,
               "dung_sau": a.tu_dong_dung_sau, "dong_sau": a.tu_dong_dong_sau, "dong_khi_xong": a.tu_dong_dong_khi_xong}
    tu_dong = {k: v for k, v in tu_dong.items() if v not in (None, False)}
    app = _build_app()(cau_hinh, trien_khai, tu_dong)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
