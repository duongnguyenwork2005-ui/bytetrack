# -*- coding: utf-8 -*-
"""
nano_detect_stream.py -- TIEN TRINH CON cua luong truc tiep tren Nano: doc TUNG frame (video / thu muc anh) ->
TensorRT FP16 -> hau xu ly NumPy -> gui detection cua frame do sang tien trinh tracker qua pipe
(`nano_stream_proto`). Khong doi detector chay het video roi moi tracking.

Chay boi `src/nano_live.py` (tien trinh cha, Python >= 3.8). Tren Nano: Python 3.6 HE THONG (co `tensorrt`),
`LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` (loi static TLS khi import cv2 sau tensorrt, do o vong 16).
File nay CO Y viet bang cu phap 3.6 va TAI SU DUNG `nano_detector` (letterbox, TRTEngine, decode_nms, scale_boxes,
liet_ke_anh, mask_regions) -- khong nhan ban tien/hau xu ly.

KENH: stdout GOC cua tien trinh duoc nhan ban (os.dup) lam kenh giao thuc, sau do fd 1 bi tro sang stderr -> moi
chuoi in ra stdout cua thu vien C (TensorRT, OpenCV) KHONG the lam hong giao thuc. Log -> stderr.

BAT DAU (giao thuc 2, review 9f210e2 N21-F1): sau HELLO con CHO byte `G` tu cha roi moi doc frame dau tien -> cha chuan
bi xong (cau hinh, telemetry) va bat timer TRUOC khi bat ky frame nao duoc doc. Byte `S` truoc `G` -> EOS 0 frame, thoat 0.

CUA SO CREDIT (`--hang-doi K`): con chi gui frame khi so frame DA GUI ma cha CHUA XU LY XONG < K; nguoc lai CHO
byte credit tu stdin. -> bo dem giua hai tien trinh co gioi han K frame; che do video offline KHONG BO frame.
Byte `S` tu cha = dung co kiem soat: gui EOS (dung_theo_yeu_cau) roi thoat 0. stdin dong (cha chet) -> thoat.

QUY UOC (giong `track_video_file.NguonFrame` / `nano_detector.chay_nguon`): frame_idx 1-based theo thu tu,
time_s = (idx - 1) / fps (KHONG dung CAP_PROP_POS_MSEC); frame anh khong doc duoc -> co unreadable, 0 detection.

GIA LAP (CHI de kiem thu, ghi ro `detector: gia_lap` trong HELLO): `--gia-lap-cache` thay suy luan bang
detection doc tu mot det cache theo frame_idx (van doc anh that); `--gia-lap-loi-o-frame N` (con thoat ma 3 khong
EOS o frame N), `--gia-lap-bo-frame N` (khong gui frame N), `--gia-lap-cham-ms` (cham moi frame).
`--torch-pt` (CHI tren PC): suy luan bang model .pt qua `nano_detector.TorchPtInfer` thay TensorRT.
"""
import argparse

import json
import os
import select
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_detector as nd  # noqa: E402
import nano_stream_proto as proto  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")


def log(msg):
    sys.stderr.write("[nano_detect_stream] %s\n" % msg)
    sys.stderr.flush()


def sha256_file(p):
    return nd.sha256_file(p)


def tep_anh_hop_le(p):
    """Kiem chu ky + kich thuoc TRUOC khi giao cho cv2. Ly do (do tren Nano, vong 21): bo giai ma BMP cua OpenCV 3.2
    (Python 3.6 he thong) ABORT CA TIEN TRINH (std::out_of_range) voi file BMP cut cut/rac, thay vi tra None.
    Tep khong qua kiem -> coi la frame KHONG DOC DUOC (cung duong voi imread tra None)."""
    try:
        n = os.path.getsize(p)
        with open(p, "rb") as f:
            h = f.read(26)
    except OSError:
        return False
    duoi = os.path.splitext(p)[1].lower()
    if duoi in (".jpg", ".jpeg"):
        return n >= 4 and h[:2] == b"\xff\xd8"
    if duoi == ".png":
        return n >= 33 and h[:8] == b"\x89PNG\r\n\x1a\n"
    if duoi == ".bmp":
        if n < 54 or h[:2] != b"BM":
            return False
        khai_bao = int.from_bytes(h[2:6], "little")
        return khai_bao <= n
    return n > 0


class Nguon(object):
    """Lap (idx, time_s, img|None, decode_ms). Thu muc anh: thu tu theo SO trong ten; video: thu tu container."""

    def __init__(self, source, fps, max_frames):
        import cv2
        self.cv2, self.max_frames, self.cap, self.paths = cv2, max_frames, None, None
        if os.path.isdir(source):
            self.paths = nd.liet_ke_anh(source)
            if not self.paths:
                raise ValueError("thu muc anh rong: %s" % source)
            if not fps:
                raise ValueError("thu muc anh khong co fps: can --fps")
            self.fps = float(fps)
            img0 = cv2.imread(self.paths[0]) if tep_anh_hop_le(self.paths[0]) else None
            if img0 is None:
                raise ValueError("khong doc duoc anh dau tien %s" % self.paths[0])
            self.orig_shape = [int(img0.shape[0]), int(img0.shape[1])]
            self.loai, self.n_frames_khai_bao = "image_dir", len(self.paths)
            self.ten = os.path.basename(os.path.normpath(source))
            self.sha256 = None
            self.quy_uoc = "sap theo GIA TRI SO o cuoi ten file; time_s = (idx-1)/fps"
        elif os.path.isfile(source) and os.path.splitext(source)[1].lower() in VIDEO_EXT:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError("cv2 khong mo duoc video %s" % source)
            fps_meta = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not fps and fps_meta <= 0:
                raise ValueError("video khong co fps trong metadata: can --fps")
            self.cap, self.fps, self.fps_metadata = cap, float(fps or fps_meta), fps_meta
            self.orig_shape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
            self.loai, self.n_frames_khai_bao = "video_file", int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.ten = os.path.splitext(os.path.basename(source))[0]
            self.sha256 = sha256_file(source)
            self.quy_uoc = "thu tu frame cua container; time_s = (idx-1)/fps (KHONG dung CAP_PROP_POS_MSEC)"
        else:
            raise ValueError("nguon khong ton tai / khong nhan dang duoc: %s" % source)

    def __iter__(self):
        pc = time.perf_counter
        n = 0
        if self.paths is not None:
            for i, p in enumerate(self.paths, 1):
                if self.max_frames and n >= self.max_frames:
                    return
                n += 1
                t_moc = time.monotonic()
                t0 = pc()
                img = self.cv2.imread(p) if tep_anh_hop_le(p) else None
                if img is None:
                    log("frame %d: khong doc duoc %s" % (i, os.path.basename(p)))
                yield i, (i - 1) / self.fps, img, (pc() - t0) * 1e3, t_moc
            return
        idx = 0
        while True:
            if self.max_frames and n >= self.max_frames:
                return
            t_moc = time.monotonic()
            t0 = pc()
            ok, img = self.cap.read()
            dec = (pc() - t0) * 1e3
            if not ok:
                return
            idx += 1
            n += 1
            yield idx, (idx - 1) / self.fps, img, dec, t_moc

    def dong(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class GiaLapCache(object):
    """Detection lay tu det cache theo frame_idx (CHI de kiem thu luong/giao thuc)."""

    def __init__(self, cache_dir, video):
        z = np.load(os.path.join(cache_dir, video + ".npz"))
        off = np.concatenate([[0], np.cumsum(z["n_per_frame"])]).astype(np.int64)
        self.theo_frame = {}
        for k, fi in enumerate(z["frame_idx"]):
            self.theo_frame[int(fi)] = z["data"][off[k]:off[k + 1]].astype(np.float32)

    def detect(self, idx):
        return self.theo_frame.get(int(idx), np.zeros((0, 6), np.float32))


class KenhDieuKhien(object):
    """Doc byte lenh tu stdin cua con (credit / dung). Linux: select de kiem khong chan; Windows (chi kiem thu
    tren PC): khong co select cho pipe -> chi doc chan khi cua so day."""

    def __init__(self, hang_doi):
        self.fd = sys.stdin.fileno()
        self.hang_doi = int(hang_doi)
        self.dang_bay = 0
        self.max_bay = 0
        self.dung = False
        self.cha_dong = False
        self.bat_dau = False
        self.co_select = not sys.platform.startswith("win")

    def _xu_ly(self, b):
        if not b:
            self.cha_dong = True
            return
        for ch in b:
            ch = bytes([ch])
            if ch == proto.LENH_CREDIT:
                self.dang_bay = max(0, self.dang_bay - 1)
            elif ch == proto.LENH_DUNG:
                self.dung = True
            elif ch == proto.LENH_BAT_DAU:
                self.bat_dau = True

    def kiem_khong_chan(self):
        if not self.co_select or self.cha_dong:
            return
        while True:
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                return
            self._xu_ly(os.read(self.fd, 64))
            if self.cha_dong:
                return

    def cho_bat_dau(self):
        """Chan cho lenh BAT DAU (hoac dung / cha dong stdin). Tra ms da cho."""
        t0 = time.perf_counter()
        while not self.bat_dau and not self.dung and not self.cha_dong:
            self._xu_ly(os.read(self.fd, 64))
        return (time.perf_counter() - t0) * 1e3

    def cho_cho_trong(self):
        """Chan cho den khi cua so con cho (hoac co lenh dung / cha dong stdin). Tra ms da cho."""
        t0 = time.perf_counter()
        while self.dang_bay >= self.hang_doi and not self.dung and not self.cha_dong:
            self._xu_ly(os.read(self.fd, 64))
        return (time.perf_counter() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser(description="Tien trinh detector cua luong truc tiep tren Nano (Python 3.6)")
    ap.add_argument("--source", required=True, help="file video hoac thu muc anh")
    ap.add_argument("--fps", type=float, default=None, help="bat buoc voi thu muc anh; ghi de fps cua video")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--engine", default=None, help="engine TensorRT build TREN Nano")
    ap.add_argument("--onnx-sha256", default=None)
    ap.add_argument("--bo-dem", choices=["cudart", "torch", "pycuda"], default="cudart")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--regions", default=None, help="vung bo qua tu NHAN (chi che do doi chieu)")
    ap.add_argument("--hang-doi", type=int, default=4, help="cua so credit K (frame da gui chua xu ly xong)")
    ap.add_argument("--lam-nong", type=int, default=3, help="so lan suy luan lam nong tren frame dau (khong gui)")
    ap.add_argument("--torch-pt", default=None, help="CHI PC: suy luan bang model .pt (TorchPtInfer)")
    ap.add_argument("--gia-lap-cache", default=None, help="CHI KIEM THU: thu muc det cache thay suy luan")
    ap.add_argument("--gia-lap-video", default=None, help="ten video trong cache gia lap (mac dinh: ten nguon)")
    ap.add_argument("--gia-lap-loi-o-frame", type=int, default=None)
    ap.add_argument("--gia-lap-bo-frame", type=int, default=None)
    ap.add_argument("--gia-lap-cham-ms", type=float, default=0.0)
    a = ap.parse_args()
    # Ctrl+C la viec cua tien trinh CHA (dung co kiem soat qua lenh S); con bo qua SIGINT de khong chet giua chung
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass

    # kenh giao thuc = ban sao stdout goc; fd 1 -> stderr (chan rac tu thu vien C)
    fd_proto = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out = os.fdopen(fd_proto, "wb")          # BufferedWriter: write() ghi DU byte, flush() ngay moi thong diep

    def gui(b):
        out.write(b)
        out.flush()

    t0 = time.perf_counter()
    nguon = None
    try:
        nguon = Nguon(a.source, a.fps, a.max_frames)
        regions = nd.doc_regions(a.regions) if a.regions else []
        import cv2
        ver = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__}
        info = {"detector": None}
        gia_lap = None
        det = None
        eng = None
        if a.gia_lap_cache:
            gia_lap = GiaLapCache(a.gia_lap_cache, a.gia_lap_video or nguon.ten)
            info = {"detector": "gia_lap", "gia_lap_cache": os.path.abspath(a.gia_lap_cache),
                    "canh_bao": "GIA LAP DE KIEM THU -- khong phai suy luan that"}
        elif a.torch_pt:
            infer = nd.TorchPtInfer(a.torch_pt, device="cpu", half=False)
            det = nd.NanoDetector(infer, a.imgsz, a.conf, a.iou, a.max_det, None, False)
            info = {"detector": "torch_pt_pc", "weights": os.path.abspath(a.torch_pt),
                    "weights_sha256": sha256_file(a.torch_pt), "canh_bao": "CHI kiem tren PC, khong phai TensorRT"}
        else:
            if not a.engine or not os.path.isfile(a.engine):
                raise ValueError("khong co engine: %s" % a.engine)
            eng = nd.TRTEngine(a.engine, bo_dem=a.bo_dem)
            det = nd.NanoDetector(eng, a.imgsz, a.conf, a.iou, a.max_det, None, False)
            e_info = eng.info()
            ver.update({k: v for k, v in e_info.items() if k != "bindings"})
            info = {"detector": "tensorrt", "engine": os.path.abspath(a.engine), "engine_sha256": sha256_file(a.engine),
                    "onnx_sha256": a.onnx_sha256, "bo_dem": a.bo_dem, "bindings": e_info.get("bindings"),
                    "letterbox": "auto=False %dx%d (ONNX static)" % (a.imgsz, a.imgsz)}
        if det is not None and a.lam_nong > 0:
            img0 = cv2.imread(nguon.paths[0]) if nguon.paths is not None else None
            if img0 is None:
                img0 = np.full((nguon.orig_shape[0], nguon.orig_shape[1], 3), 114, np.uint8)
            for _ in range(a.lam_nong):
                det.detect(img0)
        t_khoi_tao = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001  -- moi loi khoi tao -> ERROR truoc HELLO
        log("LOI khoi tao: %s: %s" % (type(e).__name__, e))
        try:
            gui(proto.dong_goi_json(proto.ERROR, {"giai_doan": "khoi_tao", "loi": "%s: %s" % (type(e).__name__, e)}))
        except Exception:  # noqa: BLE001
            pass
        if nguon is not None:
            nguon.dong()
        return 2

    hello = {"phien_ban_proto": proto.PHIEN_BAN, "pid": os.getpid(), "nguon": {
                 "duong_dan": os.path.abspath(a.source), "loai": nguon.loai, "ten": nguon.ten, "sha256": nguon.sha256,
                 "fps": nguon.fps, "fps_metadata": getattr(nguon, "fps_metadata", None),
                 "n_frames_khai_bao": nguon.n_frames_khai_bao, "quy_uoc": nguon.quy_uoc},
             "orig_shape": nguon.orig_shape, "fps": nguon.fps, "n_frames_khai_bao": nguon.n_frames_khai_bao,
             "max_frames": a.max_frames, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "max_det": a.max_det,
             "classes": None, "regions": os.path.abspath(a.regions) if a.regions else None, "n_regions": len(regions),
             "regions_sha256": sha256_file(a.regions) if a.regions else None, "hang_doi": a.hang_doi,
             "lam_nong": a.lam_nong, "t_khoi_tao_s": t_khoi_tao, "versions": ver}
    hello.update(info)
    gui(proto.dong_goi_json(proto.HELLO, hello))
    log("HELLO: %s %s, %s frame khai bao, fps %g, khoi tao %.2f s" %
        (nguon.loai, nguon.ten, nguon.n_frames_khai_bao, nguon.fps, t_khoi_tao))

    kenh = KenhDieuKhien(a.hang_doi)
    n_gui, ly_do = 0, "het_nguon"
    cho_bat_dau_ms = kenh.cho_bat_dau()
    t_nhan_bat_dau = time.monotonic()
    log("nhan BAT DAU sau %.1f ms (dung=%s, cha_dong=%s)" % (cho_bat_dau_ms, kenh.dung, kenh.cha_dong))
    if kenh.cha_dong and not kenh.dung:
        nguon.dong()
        return 4
    try:
        for idx, t_s, img, dec_ms, t_moc in (nguon if not kenh.dung else []):
            kenh.kiem_khong_chan()
            if kenh.dung or kenh.cha_dong:
                ly_do = "dung_theo_yeu_cau" if kenh.dung else "cha_dong_stdin"
                break
            if a.gia_lap_loi_o_frame and idx == a.gia_lap_loi_o_frame:
                log("GIA LAP: loi o frame %d -> thoat 3 khong EOS" % idx)
                os._exit(3)
            tm = {"decode_ms": dec_ms, "t_bat_dau": t_moc}
            co = 0
            if img is None:
                d = np.zeros((0, 6), np.float32)
                co |= proto.CO_UNREADABLE
            else:
                t1 = time.perf_counter()
                if regions:
                    img = nd.mask_regions(img, regions)
                tm["mask_ms"] = (time.perf_counter() - t1) * 1e3
                if gia_lap is not None:
                    d = gia_lap.detect(idx)
                else:
                    d, t_det = det.detect(img)
                    tm.update(t_det)
                    if eng is not None:
                        tm.update(eng.t_last)
                if len(d) >= a.max_det:
                    co |= proto.CO_TRUNCATED
            if a.gia_lap_cham_ms:
                time.sleep(a.gia_lap_cham_ms / 1e3)
            if a.gia_lap_bo_frame and idx == a.gia_lap_bo_frame:
                log("GIA LAP: bo frame %d (khong gui)" % idx)
                continue
            tm["cho_credit_ms"] = kenh.cho_cho_trong()
            if kenh.dung or kenh.cha_dong:
                ly_do = "dung_theo_yeu_cau" if kenh.dung else "cha_dong_stdin"
                break
            tm["t_gui"] = time.monotonic()
            gui(proto.dong_goi_frame(idx, t_s, d, co, tm))
            n_gui += 1
            kenh.dang_bay += 1
            kenh.max_bay = max(kenh.max_bay, kenh.dang_bay)
        else:
            if kenh.dung:
                ly_do = "dung_theo_yeu_cau"
            elif a.max_frames and n_gui >= a.max_frames:
                ly_do = "max_frames"
    except (BrokenPipeError, OSError) as e:
        log("kenh toi cha bi dong (%s) -> thoat" % e)
        nguon.dong()
        return 4
    except Exception as e:  # noqa: BLE001
        log("LOI giua luong: %s: %s" % (type(e).__name__, e))
        try:
            gui(proto.dong_goi_json(proto.ERROR, {"giai_doan": "vong_lap", "frame_gui": n_gui,
                                                  "loi": "%s: %s" % (type(e).__name__, e)}))
        except Exception:  # noqa: BLE001
            pass
        nguon.dong()
        return 3
    nguon.dong()
    eos = {"n_frame_gui": n_gui, "ly_do": ly_do, "n_frames_khai_bao": nguon.n_frames_khai_bao,
           "max_in_flight": kenh.max_bay, "hang_doi": a.hang_doi, "cho_bat_dau_ms": cho_bat_dau_ms,
           "t_nhan_bat_dau_monotonic": t_nhan_bat_dau}
    try:
        gui(proto.dong_goi_json(proto.EOS, eos))
        out.close()
    except (BrokenPipeError, OSError):
        return 4
    log("EOS: %s" % json.dumps(eos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
