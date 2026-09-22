# -*- coding: utf-8 -*-
"""
nano_detector.py -- Detector YOLOv8 (ONNX/TensorRT) cho Jetson Nano: tien xu ly + suy luan + hau xu ly NumPy,
ghi DETECTION CACHE dung dinh dang cua `det_cache.py` de tracker (Python >= 3.8, backend NumPy) replay.

    # tren Nano, Python HE THONG 3.6 (noi co `tensorrt` cua JetPack; torch 1.10 CUDA dung lam bo dem GPU):
    python3 src/nano_detector.py --engine yolov8n_fp16.engine --images data/smoke/MVI_39031_f0001-0300 --fps 25 \
        --cache-id nano_trt_MVI_39031_f0001-0300 [--regions data/smoke/MVI_39031/ignored_regions.txt] [--max-frames N]
    # sau do (venv >= 3.8):
    BYTETRACK_BACKEND=np python src/run_tracking.py --cache-id nano_trt_MVI_39031_f0001-0300 --split-name NANO-e2e ...

VI SAO TACH HAI TIEN TRINH: binding Python cua TensorRT tren JetPack 4.6 chi co cho Python 3.6, con ma tracker can
Python >= 3.8 -> detector chay o Python 3.6 va giao tiep voi tracker bang det cache (cung co che voi PC). File nay
vi the CO Y viet bang cu phap 3.6 (khong f-string phuc tap, khong dataclass, khong `from __future__ import annotations`)
va chi phu thuoc numpy + cv2 (+ tensorrt, va torch HOAC pycuda cho bo dem GPU khi chay engine).

QUY UOC (giong ultralytics 8.4 de ket qua so duoc voi cache PC):
  - Letterbox: LetterBox(new_shape, auto=False, scaleup=True, center=True, padding 114) -- input TINH 640x640 cua ONNX
    export. LUU Y: cache PC dung model .pt voi rect=True (auto=True) -> input 640x384 cho anh 960x540. Day la khac biet
    CO CHU Y giua hai duong (ngoai FP16), duoc ghi vao meta.json (`letterbox`).
  - BGR -> RGB, HWC -> CHW, /255, float32.
  - Dau ra `1 x 84 x 8400` (xywh + 80 diem lop da sigmoid, KHONG NMS trong graph) -> lay lop max, loc conf > conf_thres,
    NMS theo lop (offset lop * 7680, IoU > iou_thres bi loai, sap xep theo diem giam dan), cat max_det; scale ve anh goc
    theo dung `ops.scale_boxes` (pad = round((in - round(orig*gain))/2 - 0.1)), clip.
  - Ket qua (n, 6) float32 [x1, y1, x2, y2, conf, cls] -- cung hop dong voi `det_cache.detect_frame`.
  - Cache: <cache_root>/<cache_id>/{meta.json, <video>.npz, <video>.json}; npz co frame_idx, n_per_frame, unreadable,
    truncated, time_s, data, orig_shape (frame khong detection van co n_per_frame=0).
Kiem tren PC: `src/test_nano_detector.py` (letterbox pixel-identical voi ultralytics; decode+NMS giong het
`nms.non_max_suppression` + `ops.scale_boxes` tren CUNG dau ra tho; cache ghi ra replay duoc). Phan TensorRT chi
chay duoc tren thiet bi.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

MAX_WH = 7680       # offset lop trong NMS (ultralytics max_wh)
MAX_NMS = 30000     # so box toi da dua vao NMS (ultralytics max_nms)
PAD_VALUE = 114
IGNORE_MASK_COLOR_BGR = (114, 114, 114)   # = config.IGNORE_MASK_COLOR_BGR (khong import config: giu cu phap 3.6)


# ------------------------------------------------------------------------------------- tien xu ly
def letterbox(img, new_shape=640, pad_value=PAD_VALUE):
    """Giong ultralytics LetterBox(new_shape, auto=False, scale_fill=False, scaleup=True, center=True).
    Tra ve (img_lb, ratio, (left, top))."""
    import cv2
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    shape = img.shape[:2]                                   # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))   # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return img, r, (left, top)


def preprocess(img_lb):
    """BGR uint8 HWC -> float32 (1, 3, H, W) RGB trong [0, 1] (giong BasePredictor.preprocess)."""
    x = img_lb[:, :, ::-1].transpose(2, 0, 1)               # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None]


def mask_regions(img, regions, color=IGNORE_MASK_COLOR_BGR):
    """To xam vung bo qua (giong baseline_track.mask_ignored_regions): regions = [(left, top, w, h), ...]."""
    if not regions:
        return img
    h, w = img.shape[:2]
    for left, top, rw, rh in regions:
        x1, y1 = max(0, int(round(left))), max(0, int(round(top)))
        x2, y2 = min(w, int(round(left + rw))), min(h, int(round(top + rh)))
        if x2 > x1 and y2 > y1:
            img[y1:y2, x1:x2] = color
    return img


def doc_regions(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            v = [float(t) for t in ln.replace(";", ",").split(",")[:4]]
            if len(v) == 4:
                out.append(tuple(v))
    return out


# ------------------------------------------------------------------------------------- hau xu ly
def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_numpy(boxes, scores, iou_thres):
    """NMS tham lam giong torchvision.ops.nms: xet theo diem giam dan, loai box co IoU > iou_thres voi box da giu.
    Tra ve chi so theo thu tu diem giam dan."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def decode_nms(raw, conf_thres=0.05, iou_thres=0.7, max_det=3000, classes=None, agnostic=False):
    """raw: (1, 4+nc, N) hoac (4+nc, N) -> (n, 6) float32 [x1, y1, x2, y2, conf, cls] trong toa do letterbox,
    giong `ultralytics.utils.nms.non_max_suppression(multi_label=False)`."""
    p = np.asarray(raw, dtype=np.float32)
    if p.ndim == 3:
        p = p[0]
    nc = p.shape[0] - 4
    p = p.T                                                   # (N, 4+nc)
    cls_scores = p[:, 4:4 + nc]
    conf = cls_scores.max(axis=1)
    keep0 = conf > conf_thres                                 # xc: candidates
    p, conf = p[keep0], conf[keep0]
    if p.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    j = cls_scores[keep0].argmax(axis=1).astype(np.float32)
    box = xywh2xyxy(p[:, :4])
    x = np.concatenate([box, conf[:, None], j[:, None]], axis=1)
    if classes is not None:
        x = x[np.isin(x[:, 5], np.asarray(classes, dtype=np.float32))]
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if n > MAX_NMS:
        x = x[np.argsort(-x[:, 4], kind="stable")[:MAX_NMS]]
    c = x[:, 5:6] * (0 if agnostic else MAX_WH)
    boxes = x[:, :4] + c
    i = nms_numpy(boxes, x[:, 4], iou_thres)[:max_det]
    return np.ascontiguousarray(x[i], dtype=np.float32)


def scale_boxes(boxes, img1_shape, img0_shape):
    """Giong ultralytics ops.scale_boxes(ratio_pad=None, padding=True) + clip_boxes. boxes (n, >=4) xyxy, sua tai cho."""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - round(img0_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((img1_shape[0] - round(img0_shape[0] * gain)) / 2 - 0.1)
    boxes[:, 0] -= pad_x; boxes[:, 2] -= pad_x
    boxes[:, 1] -= pad_y; boxes[:, 3] -= pad_y
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    return boxes


# ------------------------------------------------------------------------------------- detector
class NanoDetector(object):
    """Ghep: letterbox -> preprocess -> infer(x) -> decode_nms -> scale_boxes. `infer` la ham (1,3,H,W) float32 ->
    (1, 4+nc, N) float32 (TensorRT tren Nano; tren PC test dung torch FP32 de kiem hop dong hau xu ly)."""

    def __init__(self, infer, imgsz=640, conf=0.05, iou=0.7, max_det=3000, classes=None, agnostic=False):
        self.infer, self.imgsz = infer, imgsz
        self.conf, self.iou, self.max_det, self.classes, self.agnostic = conf, iou, max_det, classes, agnostic
        self.t = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "n": 0}

    def detect(self, img_bgr):
        t0 = time.perf_counter()
        lb, r, pad = letterbox(img_bgr, self.imgsz)
        x = preprocess(lb)
        t1 = time.perf_counter()
        raw = self.infer(x)
        t2 = time.perf_counter()
        det = decode_nms(raw, self.conf, self.iou, self.max_det, self.classes, self.agnostic)
        det = scale_boxes(det, lb.shape[:2], img_bgr.shape[:2])
        t3 = time.perf_counter()
        self.t["pre_ms"] += (t1 - t0) * 1e3; self.t["infer_ms"] += (t2 - t1) * 1e3; self.t["post_ms"] += (t3 - t2) * 1e3
        self.t["n"] += 1
        return det, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}


class TRTEngine(object):
    """Chay engine TensorRT (build TREN Nano bang trtexec). Bo dem GPU: torch (neu co) hoac pycuda.
    Input binding 'images' (1,3,640,640) float32; output 'output0' (1,84,8400) float32 (engine FP16 van I/O FP32)."""

    def __init__(self, engine_path, dung_torch=None):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("khong deserialize duoc engine: %s" % engine_path)
        self.ctx = self.engine.create_execution_context()
        self.bindings_info = []
        for i in range(self.engine.num_bindings):
            self.bindings_info.append({"name": self.engine.get_binding_name(i), "input": self.engine.binding_is_input(i),
                                       "shape": tuple(self.engine.get_binding_shape(i)),
                                       "dtype": trt.nptype(self.engine.get_binding_dtype(i)).__name__})
        self.in_idx = [i for i, b in enumerate(self.bindings_info) if b["input"]][0]
        self.out_idx = [i for i, b in enumerate(self.bindings_info) if not b["input"]][0]
        self.in_shape = self.bindings_info[self.in_idx]["shape"]
        self.out_shape = self.bindings_info[self.out_idx]["shape"]
        self.in_dtype = np.dtype(self.bindings_info[self.in_idx]["dtype"])
        self.out_dtype = np.dtype(self.bindings_info[self.out_idx]["dtype"])
        if dung_torch is None:
            try:
                import torch  # noqa: F401
                dung_torch = True
            except Exception:  # noqa: BLE001
                dung_torch = False
        self.dung_torch = dung_torch
        if dung_torch:
            import torch
            self.torch = torch
            self.d_in = torch.empty(int(np.prod(self.in_shape)), dtype=getattr(torch, self.in_dtype.name), device="cuda")
            self.d_out = torch.empty(int(np.prod(self.out_shape)), dtype=getattr(torch, self.out_dtype.name), device="cuda")
            self.stream = torch.cuda.Stream()
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.in_idx] = int(self.d_in.data_ptr()); self.bindings[self.out_idx] = int(self.d_out.data_ptr())
        else:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            self.cuda = cuda
            self.h_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)), self.out_dtype)
            self.h_in = cuda.pagelocked_empty(int(np.prod(self.in_shape)), self.in_dtype)
            self.d_in = cuda.mem_alloc(self.h_in.nbytes); self.d_out = cuda.mem_alloc(self.h_out.nbytes)
            self.stream = cuda.Stream()
            self.bindings = [0] * self.engine.num_bindings
            self.bindings[self.in_idx] = int(self.d_in); self.bindings[self.out_idx] = int(self.d_out)

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=self.in_dtype).reshape(-1)
        if self.dung_torch:
            torch = self.torch
            with torch.cuda.stream(self.stream):
                self.d_in.copy_(torch.from_numpy(x))
                ok = self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.cuda_stream)
                out = self.d_out.cpu().numpy()
            self.stream.synchronize()
        else:
            np.copyto(self.h_in, x)
            self.cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
            ok = self.ctx.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            self.cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
            self.stream.synchronize()
            out = np.array(self.h_out)
        if not ok:
            raise RuntimeError("execute_async_v2 tra ve False")
        return out.reshape(self.out_shape).astype(np.float32)

    def info(self):
        import tensorrt as trt
        return {"tensorrt": trt.__version__, "bindings": self.bindings_info, "bo_dem": "torch" if self.dung_torch else "pycuda"}


class TorchPtInfer(object):
    """CHI DE KIEM TREN PC (co torch + ultralytics): suy luan FP32 bang model .pt -> dau ra tho (1, 84, 8400).
    Khong dung tren Nano."""

    def __init__(self, weights, device="cpu", half=False):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        y = YOLO(weights)
        self.net = y.model.to(device).eval()
        self.half = half
        if half:
            self.net = self.net.half()
        self.device = device
        self.names = dict(getattr(y, "names", {}) or {})

    def __call__(self, x):
        t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        if self.half:
            t = t.half()
        with self.torch.no_grad():
            out = self.net(t)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.float().cpu().numpy()


# ------------------------------------------------------------------------------------- nguon anh
_SO_CUOI = re.compile(r"^(.*?)(\d+)$")


def liet_ke_anh(thu_muc):
    """Sap theo SO trong ten (giong track_video_file.sap_xep_anh, khong co manifest)."""
    ds = [f for f in os.listdir(thu_muc) if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    khoa = []
    for f in ds:
        m = _SO_CUOI.match(os.path.splitext(f)[0])
        if not m:
            raise ValueError("khong suy ra duoc thu tu tu ten '%s'" % f)
        khoa.append((m.group(1), int(m.group(2)), f))
    if len(set(k[0] for k in khoa)) > 1:
        raise ValueError("ten anh co nhieu tien to khac nhau -> thu tu mo ho")
    khoa.sort(key=lambda k: k[1])
    return [os.path.join(thu_muc, k[2]) for k in khoa]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------- ghi cache
def ghi_cache(cache_dir, video, frame_idx, time_s, n_per, unreadable, truncated, data, orig_shape, meta, extra):
    """Ghi <video>.npz + <video>.json + meta.json theo dinh dang det_cache.py (khong import det_cache: 3.6)."""
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    np.savez_compressed(os.path.join(cache_dir, video + ".npz"),
                        frame_idx=np.asarray(frame_idx, dtype=np.int32), n_per_frame=np.asarray(n_per, dtype=np.int32),
                        unreadable=np.asarray(unreadable, dtype=bool), truncated=np.asarray(truncated, dtype=bool),
                        time_s=np.asarray(time_s, dtype=np.float64),
                        data=(np.asarray(data, dtype=np.float32) if len(data) else np.zeros((0, 6), np.float32)),
                        orig_shape=np.asarray(orig_shape, dtype=np.int32))
    info = {"video": video, "n_frames": int(len(frame_idx)), "n_boxes": int(len(data)),
            "n_unreadable": int(np.sum(unreadable)), "n_truncated": int(np.sum(truncated)),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    info.update(extra)
    with open(os.path.join(cache_dir, video + ".json"), "w") as f:
        json.dump(info, f, indent=2)
    mp = os.path.join(cache_dir, "meta.json")
    if os.path.isfile(mp):
        with open(mp) as f:
            cu = json.load(f)
        for k in ("conf", "iou", "classes", "max_det", "imgsz", "weights_sha256", "letterbox"):
            if cu.get(k) != meta.get(k):
                raise RuntimeError("cache %s da ton tai voi %s khac (%r != %r) -> dung --cache-id khac"
                                   % (cache_dir, k, cu.get(k), meta.get(k)))
    else:
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)
    return info


def chay_nguon(det, duong_dan_anh, fps, regions, max_frames, warmup, log_every=50):
    """Chay detector tren danh sach anh -> cac mang cache + timing tung frame."""
    import cv2
    frame_idx, time_s, n_per, unreadable, truncated, chunks, timing = [], [], [], [], [], [], []
    orig_shape = None
    N = len(duong_dan_anh) if not max_frames else min(len(duong_dan_anh), max_frames)
    for i in range(N):
        t0 = time.perf_counter()
        img = cv2.imread(duong_dan_anh[i])
        t_dec = (time.perf_counter() - t0) * 1e3
        frame_idx.append(i + 1); time_s.append(i / float(fps))
        if img is None:
            unreadable.append(True); truncated.append(False); n_per.append(0)
            timing.append({"frame": i + 1, "decode_ms": t_dec, "unreadable": True})
            continue
        if orig_shape is None:
            orig_shape = img.shape[:2]
        if regions:
            img = mask_regions(img, regions)
        d, tm = det.detect(img)
        unreadable.append(False); truncated.append(len(d) >= det.max_det); n_per.append(len(d)); chunks.append(d)
        tm.update({"frame": i + 1, "decode_ms": t_dec, "n_det": int(len(d))}); timing.append(tm)
        if log_every and (i + 1) % log_every == 0:
            print("  frame %d/%d  det=%d  infer %.1f ms" % (i + 1, N, len(d), tm["infer_ms"]))
    data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), np.float32)
    tong = {}
    ok = [t for t in timing[warmup:] if not t.get("unreadable")]
    for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms"):
        v = [t[k] for t in ok if k in t]
        tong[k] = float(np.mean(v)) if v else None
    tong["n_frame_tong_ket"] = len(ok); tong["warmup_bo"] = warmup
    tong["tong_ms_moi_frame"] = (sum(tong[k] for k in ("decode_ms", "pre_ms", "infer_ms", "post_ms") if tong[k] is not None)
                                 if ok else None)
    return dict(frame_idx=frame_idx, time_s=time_s, n_per=n_per, unreadable=unreadable, truncated=truncated,
                data=data, orig_shape=orig_shape or (0, 0)), timing, tong


def main():
    ap = argparse.ArgumentParser(description="Detector YOLOv8 TensorRT tren Nano -> det cache (Python 3.6)")
    ap.add_argument("--engine", required=True, help="engine TensorRT build TREN Nano (trtexec)")
    ap.add_argument("--onnx-sha256", default=None, help="sha256 cua ONNX da dung de build engine (ghi vao meta)")
    ap.add_argument("--images", required=True, help="thu muc anh (thu tu theo so trong ten)")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--video-name", default=None, help="ten video trong cache (mac dinh: ten thu muc)")
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--cache-root", default=None, help="mac dinh <goc goi>/data/interim/det_cache")
    ap.add_argument("--regions", default=None, help="ignored_regions.txt -> to xam truoc detector (nhu cache PC)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5, help="so frame dau bo khi tong ket thoi gian")
    ap.add_argument("--timing-json", default=None)
    ap.add_argument("--no-torch", action="store_true", help="ep dung pycuda thay vi torch cho bo dem GPU")
    a = ap.parse_args()

    goc = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    cache_root = a.cache_root or os.path.join(goc, "data", "interim", "det_cache")
    video = a.video_name or os.path.basename(os.path.normpath(a.images))
    if not os.path.isfile(a.engine):
        print("[LOI] khong co engine: %s" % a.engine); return 2
    if not os.path.isdir(a.images):
        print("[LOI] khong co thu muc anh: %s" % a.images); return 2
    anh = liet_ke_anh(a.images)
    if not anh:
        print("[LOI] thu muc anh rong: %s" % a.images); return 2
    regions = doc_regions(a.regions) if a.regions else []

    t0 = time.perf_counter()
    eng = TRTEngine(a.engine, dung_torch=(False if a.no_torch else None))
    det = NanoDetector(eng, a.imgsz, a.conf, a.iou, a.max_det, None, False)
    # lam nong
    import cv2
    img0 = cv2.imread(anh[0])
    for _ in range(3):
        det.detect(img0)
    det.t = {"pre_ms": 0.0, "infer_ms": 0.0, "post_ms": 0.0, "n": 0}
    t_khoi_tao = time.perf_counter() - t0
    print("[nano_detector] engine=%s %s" % (a.engine, json.dumps(eng.info())))
    print("[nano_detector] %d anh, fps=%g, regions=%d, conf=%g iou=%g max_det=%d imgsz=%d" %
          (len(anh), a.fps, len(regions), a.conf, a.iou, a.max_det, a.imgsz))

    t1 = time.perf_counter()
    arr, timing, tong = chay_nguon(det, anh, a.fps, regions, a.max_frames, a.warmup)
    wall = time.perf_counter() - t1
    ver = {"numpy": np.__version__, "opencv": cv2.__version__, "python": sys.version.split()[0]}
    ver.update({k: v for k, v in eng.info().items() if k != "bindings"})
    meta = {"backend": "tensorrt", "weights": os.path.abspath(a.engine), "weights_sha256": sha256_file(a.engine),
            "onnx_sha256": a.onnx_sha256, "imgsz": a.imgsz, "conf": a.conf, "iou": a.iou, "classes": None,
            "max_det": a.max_det, "half": True, "agnostic_nms": False, "augment": False, "batch": 1,
            "device": "jetson-nano-trt", "mask_ignored_regions": bool(regions), "dataset": "video_file", "part": "images",
            "letterbox": "auto=False %dx%d (ONNX static); cache PC .pt dung rect auto=True" % (a.imgsz, a.imgsz),
            "gt_jitter": 0.0, "gt_dropout": 0.0, "gt_seed": 0, "gt_conf_range": [0.3, 0.95],
            "versions": ver, "git_commit": None, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ghi_chu": "cache do nano_detector.py (Python 3.6, TensorRT) tao; khong co spec_sha256 (legacy) -> "
                       "open_cache chap nhan; KHONG ky vong bit-identical voi cache PC (FP16 + letterbox 640x640)"}
    cdir = os.path.join(cache_root, a.cache_id)
    info = ghi_cache(cdir, video, arr["frame_idx"], arr["time_s"], arr["n_per"], arr["unreadable"], arr["truncated"],
                     arr["data"], arr["orig_shape"], meta,
                     {"source": os.path.abspath(a.images), "fps": a.fps, "regions": a.regions, "n_regions": len(regions),
                      "wall_vong_lap_s": wall, "wall_khoi_tao_s": t_khoi_tao, "fps_vong_lap": (len(arr["frame_idx"]) / wall if wall > 0 else None),
                      "tong_ket_ms": tong, "pham_vi": "fps_vong_lap = n_frame / (decode + pre + infer + post + ghi mang) sau khoi tao; "
                                                     "tong_ket_ms bo %d frame warm-up" % a.warmup})
    print("[nano_detector] %d frame, %d box, %d frame rong, wall %.1fs (%.1f fps vong lap) | trung binh sau warm-up: %s" %
          (info["n_frames"], info["n_boxes"], int(sum(1 for n in arr["n_per"] if n == 0)), wall,
           info["fps_vong_lap"] or 0.0, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tong.items()})))
    print("[nano_detector] -> %s" % cdir)
    if a.timing_json:
        with open(a.timing_json, "w") as f:
            json.dump({"engine": a.engine, "engine_info": eng.info(), "tong_ket": tong, "wall_vong_lap_s": wall,
                       "wall_khoi_tao_s": t_khoi_tao, "timing": timing}, f, indent=1)
        print("[nano_detector] timing -> %s" % a.timing_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
