#!/usr/bin/env python3
"""R63 - do tung cong doan detector tren Jetson Nano voi CHINH ma cua mot goi (vd ban sao v16), khong sua goi.

Chay bang Python detector cua Nano (Python 3.6 he thong, TensorRT 8.2.1.9, CUDA 10.2, OpenCV 3.2):
    python3 bench_engine_pipeline.py --goi ~/nano_run/r63/baseline/nano_transfer \
        --engine ~/nano_run/engine/yolov8n_fp16.engine --anh <thu_muc_jpg> --ra <thu_muc_ra> \
        [--so-anh 300] [--pass 3] [--warmup 25]

Hai che do, chay xen ke theo pass (A, B, A, B...), cung anh, cung engine:
  dong_bo : goi dung NanoDetector.detect cua goi (letterbox -> preprocess -> TRTEngine.__call__ ->
            decode_nms -> scale_boxes). Ghi pre/infer/post cua goi + h2d/exec/d2h (host wall quanh
            cudaMemcpy/execute_v2 dong bo) tu TRTEngine.t_last. Day la cach do cua pipeline R62.
  su_kien : cung buffer GPU cua engine, nhung H2D/execute_async_v2/D2H tren MOT stream rieng va dat
            CUDA event truoc/sau tung lenh -> thoi gian GPU thuc thi engine (event) tach khoi thoi gian
            host goi enqueue va cho dong bo. Copy `astype(np.float32)` va decode_nms/scale_boxes duoc do
            rieng (ung vien A cua khao sat 32).
Cong doan doc anh (cv2.imread) do rieng, khong phai giai ma MP4.

Dau ra: <ra>/frame.csv (moi frame x pass), <ra>/tom_tat.json (median/p90 theo che do, bo `warmup` frame dau
moi pass), <ra>/moi_truong.json, <ra>/det_pass0.npz (detection cua pass dong_bo dau tien, de doi chieu).
Khong ghi ngoai <ra>. Wrapper chi o cap do: doc truc tiep thuoc tinh cua TRTEngine (ctx, bindings, d_in,
d_out, h_out, cudart) - neu goi khac cau truc thi dung lai va bao loi, khong doan.
"""
import argparse
import ctypes
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time

import numpy as np


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def phan_vi(xs, q):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q * 100))


def liet_ke_anh(thu_muc, n):
    ds = sorted(x for x in os.listdir(thu_muc) if x.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(thu_muc, x) for x in ds[:n]]


class SuKien(object):
    """CUDA event/stream qua ctypes libcudart cua chinh TRTEngine (khong torch/pycuda)."""

    def __init__(self, cudart):
        self.c = cudart
        c_vp = ctypes.c_void_p
        for ten, args in (("cudaEventCreate", [ctypes.POINTER(c_vp)]), ("cudaEventRecord", [c_vp, c_vp]),
                          ("cudaEventSynchronize", [c_vp]),
                          ("cudaEventElapsedTime", [ctypes.POINTER(ctypes.c_float), c_vp, c_vp]),
                          ("cudaStreamCreate", [ctypes.POINTER(c_vp)]), ("cudaStreamSynchronize", [c_vp]),
                          ("cudaMemcpyAsync", [c_vp, c_vp, ctypes.c_size_t, ctypes.c_int, c_vp])):
            getattr(self.c, ten).argtypes = args
            getattr(self.c, ten).restype = ctypes.c_int
        self.stream = c_vp()
        self._chk(self.c.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.ev = []
        for _ in range(4):
            e = c_vp()
            self._chk(self.c.cudaEventCreate(ctypes.byref(e)), "cudaEventCreate")
            self.ev.append(e)

    def _chk(self, rc, ten):
        if rc != 0:
            raise RuntimeError("%s: cudaError %d" % (ten, rc))

    def ms(self, a, b):
        f = ctypes.c_float()
        self._chk(self.c.cudaEventElapsedTime(ctypes.byref(f), self.ev[a], self.ev[b]), "cudaEventElapsedTime")
        return float(f.value)

    def chay(self, eng, x):
        """H2D -> execute_async_v2 -> D2H tren stream rieng. Tra ve dict thoi gian (ms)."""
        c, s, ev = self.c, self.stream, self.ev
        x = np.ascontiguousarray(x, dtype=eng.in_dtype).reshape(-1)
        t0 = time.perf_counter()
        self._chk(c.cudaEventRecord(ev[0], s), "rec0")
        self._chk(c.cudaMemcpyAsync(eng.d_in, x.ctypes.data_as(ctypes.c_void_p), x.nbytes, 1, s), "H2D")
        self._chk(c.cudaEventRecord(ev[1], s), "rec1")
        t1 = time.perf_counter()
        ok = eng.ctx.execute_async_v2(bindings=eng.bindings, stream_handle=int(s.value or 0))
        t2 = time.perf_counter()
        self._chk(c.cudaEventRecord(ev[2], s), "rec2")
        self._chk(c.cudaMemcpyAsync(eng.h_out.ctypes.data_as(ctypes.c_void_p), eng.d_out, eng.h_out.nbytes, 2, s), "D2H")
        self._chk(c.cudaEventRecord(ev[3], s), "rec3")
        t3 = time.perf_counter()
        self._chk(c.cudaStreamSynchronize(s), "sync")
        t4 = time.perf_counter()
        if not ok:
            raise RuntimeError("execute_async_v2 tra ve False")
        return {"gpu_h2d_ms": self.ms(0, 1), "gpu_exec_ms": self.ms(1, 2), "gpu_d2h_ms": self.ms(2, 3),
                "gpu_tong_ms": self.ms(0, 3), "host_h2d_goi_ms": (t1 - t0) * 1e3,
                "host_enqueue_ms": (t2 - t1) * 1e3, "host_d2h_goi_ms": (t3 - t2) * 1e3,
                "host_cho_sync_ms": (t4 - t3) * 1e3, "host_infer_ms": (t4 - t0) * 1e3}


def main(argv=None):
    ap = argparse.ArgumentParser(description="R63 do cong doan detector tren Nano")
    ap.add_argument("--goi", required=True, help="thu muc goi (chua src/nano_detector.py), vd ban sao v16")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--anh", required=True, help="thu muc JPG (vd MVI_39031 f0001-0300)")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--so-anh", type=int, default=300)
    ap.add_argument("--pass", dest="n_pass", type=int, default=4, help="so pass, xen ke dong_bo/su_kien")
    ap.add_argument("--warmup", type=int, default=25, help="so frame dau moi pass bo khoi thong ke")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=3000)
    a = ap.parse_args(argv)

    src = os.path.join(os.path.abspath(os.path.expanduser(a.goi)), "src")
    if not os.path.isfile(os.path.join(src, "nano_detector.py")):
        print("[LOI] khong thay %s/nano_detector.py" % src, file=sys.stderr)
        return 2
    sys.path.insert(0, src)
    import cv2
    import nano_detector as nd

    os.makedirs(a.ra, exist_ok=True)
    if os.listdir(a.ra):
        print("[LOI] thu muc ra khong rong: %s (moi luot mot thu muc moi)" % a.ra, file=sys.stderr)
        return 2
    anh = liet_ke_anh(a.anh, a.so_anh)
    if len(anh) != a.so_anh:
        print("[LOI] can %d anh, thay %d trong %s" % (a.so_anh, len(anh), a.anh), file=sys.stderr)
        return 2
    h_vao = hashlib.sha256()
    for duong in anh:                                        # thu tu sap xep + noi dung tung anh
        h_vao.update(os.path.basename(duong).encode() + b"\0" + sha256_file(duong).encode() + b"\n")

    t0 = time.perf_counter()
    eng = nd.TRTEngine(a.engine, bo_dem="cudart")
    for thuoc_tinh in ("ctx", "bindings", "d_in", "d_out", "h_out", "cudart", "in_shape", "out_shape"):
        if not hasattr(eng, thuoc_tinh):
            print("[LOI] TRTEngine cua goi khong co '%s' - cau truc khac, khong doan" % thuoc_tinh, file=sys.stderr)
            return 2
    ins, outs = tuple(int(x) for x in eng.in_shape), tuple(int(x) for x in eng.out_shape)
    if len(ins) != 4 or ins[0] != 1 or ins[1] != 3 or min(ins) <= 0:
        print("[LOI] input engine %s khong phai tinh 1x3xHxW" % (ins,), file=sys.stderr)
        return 2
    hw = (ins[2], ins[3])                                   # (H, W) tu engine; letterbox nhan tuple
    n_anchor = sum((hw[0] // s) * (hw[1] // s) for s in (8, 16, 32))
    if len(outs) != 3 or outs[0] != 1 or outs[2] != n_anchor or eng.h_out.size != int(np.prod(outs)):
        print("[LOI] output engine %s / buffer %d khong khop 1x(4+nc)x%d" % (outs, eng.h_out.size, n_anchor),
              file=sys.stderr)
        return 2
    det = nd.NanoDetector(eng, hw, a.conf, a.iou, a.max_det, None, False)
    sk = SuKien(eng.cudart)
    t_khoi_tao = time.perf_counter() - t0

    moi_truong = {
        "host": socket.gethostname(), "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "opencv": cv2.__version__, "tensorrt": eng.info().get("tensorrt"),
        "goi": os.path.abspath(os.path.expanduser(a.goi)), "nano_detector_sha256": sha256_file(os.path.join(src, "nano_detector.py")),
        "engine": os.path.abspath(a.engine), "engine_sha256": sha256_file(a.engine), "bindings": eng.info()["bindings"],
        "letterbox_hw": hw, "in_shape": ins, "out_shape": outs, "anh_sha256_tong": h_vao.hexdigest(),
        "anh": os.path.abspath(a.anh), "so_anh": len(anh), "anh_dau": os.path.basename(anh[0]),
        "anh_cuoi": os.path.basename(anh[-1]), "conf": a.conf, "iou": a.iou, "max_det": a.max_det,
        "n_pass": a.n_pass, "warmup": a.warmup, "khoi_tao_s": t_khoi_tao,
        "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip() if os.path.exists("/proc/sys/kernel/random/boot_id") else None,
        "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
    }
    with open(os.path.join(a.ra, "moi_truong.json"), "w") as f:
        json.dump(moi_truong, f, indent=2)

    cot = ["pass", "che_do", "frame", "dec_ms", "lb_pre_ms", "infer_ms", "post_ms", "h2d_ms", "exec_ms", "d2h_ms",
           "gpu_h2d_ms", "gpu_exec_ms", "gpu_d2h_ms", "gpu_tong_ms", "host_h2d_goi_ms", "host_enqueue_ms",
           "host_d2h_goi_ms", "host_cho_sync_ms", "host_infer_ms", "copy_f32_ms", "decode_nms_ms", "scale_ms", "n_det"]
    hang = []
    det_pass0 = []        # detection pass dong_bo dau tien
    det_pass1 = []        # detection pass su_kien dau tien (cung frame) -> kiem parity hai che do
    for p in range(a.n_pass):
        che_do = "dong_bo" if p % 2 == 0 else "su_kien"
        for i, duong in enumerate(anh):
            t_a = time.perf_counter()
            img = cv2.imread(duong)
            t_b = time.perf_counter()
            r = {"pass": p, "che_do": che_do, "frame": i + 1, "dec_ms": (t_b - t_a) * 1e3}
            if img is None:
                raise RuntimeError("khong doc duoc %s" % duong)
            if che_do == "dong_bo":
                d, t = det.detect(img)
                r.update({"lb_pre_ms": t["pre_ms"], "infer_ms": t["infer_ms"], "post_ms": t["post_ms"]})
                r.update(eng.t_last)
                if p == 0:
                    det_pass0.append(d.copy())
            else:
                t1 = time.perf_counter()
                lb, _, _ = nd.letterbox(img, hw)
                x = nd.preprocess(lb)
                t2 = time.perf_counter()
                r["lb_pre_ms"] = (t2 - t1) * 1e3
                r.update(sk.chay(eng, x))
                t3 = time.perf_counter()
                raw = eng.h_out.reshape(eng.out_shape).astype(np.float32)   # giong TRTEngine.__call__ dong cuoi
                t4 = time.perf_counter()
                d = nd.decode_nms(raw, a.conf, a.iou, a.max_det, None, False)
                t5 = time.perf_counter()
                d = nd.scale_boxes(d, lb.shape[:2], img.shape[:2])
                t6 = time.perf_counter()
                r.update({"copy_f32_ms": (t4 - t3) * 1e3, "decode_nms_ms": (t5 - t4) * 1e3, "scale_ms": (t6 - t5) * 1e3})
                if p == 1:
                    det_pass1.append(d.copy())
            r["n_det"] = int(len(d))
            hang.append(r)
        print("[bench] pass %d/%d (%s) xong" % (p + 1, a.n_pass, che_do))
        sys.stdout.flush()

    with open(os.path.join(a.ra, "frame.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        for r in hang:
            w.writerow(dict((k, r.get(k, "")) for k in cot))

    tom = {"moi_truong": moi_truong, "theo_che_do": {}}
    for che_do in ("dong_bo", "su_kien"):
        rs = [r for r in hang if r["che_do"] == che_do and r["frame"] > a.warmup]
        tk = {"n_frame_thong_ke": len(rs)}
        for k in cot[3:]:
            xs = [float(r[k]) for r in rs if k in r and r[k] != ""]
            if xs:
                tk[k] = {"median": phan_vi(xs, 0.5), "p90": phan_vi(xs, 0.9), "mean": float(np.mean(xs))}
        tom["theo_che_do"][che_do] = tk
    # detection pass dong_bo dau: luu + hash de doi chieu voi pipeline / ung vien
    if det_pass0:
        n = np.asarray([len(x) for x in det_pass0], dtype=np.int32)
        data = np.concatenate([x for x in det_pass0 if len(x)] or [np.zeros((0, 6), np.float32)]).astype(np.float32)
        np.savez(os.path.join(a.ra, "det_pass0.npz"), n_per_frame=n, data=data)
        tom["det_pass0_sha256"] = hashlib.sha256(n.tobytes() + data.tobytes()).hexdigest()
    if det_pass0 and det_pass1:
        trung = [bool(np.array_equal(x, y)) for x, y in zip(det_pass0, det_pass1)]
        tom["parity_dong_bo_su_kien"] = {"n_frame": len(trung), "n_trung_byte": int(sum(trung)),
                                         "frame_lech_dau": [i + 1 for i, ok in enumerate(trung) if not ok][:20]}
    tom["ghi_chu_dien_giai"] = (
        "su_kien la DUONG DO RIENG (execute_async_v2 tren stream rieng + CUDA event), khong phai production "
        "execute_v2. gpu_exec_ms (event) cho biet thoi gian GPU cua engine; KHONG lay hieu hai trung vi "
        "exec_ms(dong_bo) - gpu_exec_ms(su_kien) lam phi goi chinh xac cua execute_v2. Neu parity hai che do "
        "khong trung hoac so lieu khong nhat quan voi trtexec thi ket luan 'chua phan ra du'.")
    tom["ket_thuc"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(a.ra, "tom_tat.json"), "w") as f:
        json.dump(tom, f, indent=2)
    db = tom["theo_che_do"].get("dong_bo", {}); sk_ = tom["theo_che_do"].get("su_kien", {})
    print("[bench] exec host (dong_bo) median=%s | gpu_exec event median=%s | enqueue host median=%s" % (
        db.get("exec_ms", {}).get("median"), sk_.get("gpu_exec_ms", {}).get("median"),
        sk_.get("host_enqueue_ms", {}).get("median")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
