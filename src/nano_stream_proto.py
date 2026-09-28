# -*- coding: utf-8 -*-
"""
nano_stream_proto.py -- Giao thuc truyen TUNG FRAME tu tien trinh detector (Python 3.6 he thong, TensorRT)
sang tien trinh tracker (Python >= 3.8, backend NumPy) tren Jetson Nano.

VI SAO: binding TensorRT cua JetPack 4.6 chi co cho Python 3.6, tracker can Python >= 3.8 -> hai tien trinh.
File nay CO Y viet bang cu phap 3.6 va chi dung `struct`/`json`/numpy de CA HAI phia import duoc.

KENH:
  con -> cha (stdout rieng cua con): chuoi THONG DIEP nhi phan little-endian = HEADER co dinh + PAYLOAD.
      HELLO  (1)  payload JSON: nguon, detector, engine, fps, orig_shape, n_frames_khai_bao, phien ban...
      FRAME  (2)  payload n_det x 6 float32 [x1, y1, x2, y2, conf, cls] (dung hop dong det_cache), co the rong
      EOS    (3)  payload JSON: so frame da gui, ly do ket thuc, max_in_flight...
      ERROR  (4)  payload JSON: thong bao loi (con gui truoc khi thoat voi ma khac 0 neu kip)
  cha -> con (stdin cua con): moi byte la mot lenh
      b"+"   CREDIT: cha da XU LY XONG mot frame -> con duoc gui them mot frame (cua so co gioi han)
      b"S"   yeu cau DUNG: con gui EOS (ly_do dung_theo_yeu_cau) roi thoat 0

HEADER (HDR): magic, loai, co (bit0 unreadable, bit1 truncated), frame_idx (1-based), n_det, time_s,
  thoi gian (ms) cua con: decode, mask, pre, infer, h2d, exec, d2h, post, cho_credit;
  moc `time.monotonic()` cua con: t_bat_dau (TRUOC khi doc frame), t_gui (ngay truoc khi ghi thong diep);
  do dai payload. `time.monotonic()` = CLOCK_MONOTONIC tren Linux -> so duoc GIUA hai tien trinh cung may.
"""
import json
import struct

import numpy as np

MAGIC = b"NDS1"
PHIEN_BAN = 1
HELLO, FRAME, EOS, ERROR = 1, 2, 3, 4
TEN_LOAI = {HELLO: "HELLO", FRAME: "FRAME", EOS: "EOS", ERROR: "ERROR"}
CO_UNREADABLE, CO_TRUNCATED = 1, 2
LENH_CREDIT, LENH_DUNG = b"+", b"S"

TRUONG_MS = ("decode_ms", "mask_ms", "pre_ms", "infer_ms", "h2d_ms", "exec_ms", "d2h_ms", "post_ms", "cho_credit_ms")
TRUONG_MOC = ("t_bat_dau", "t_gui")
HDR = struct.Struct("<4sBBHiid" + "d" * (len(TRUONG_MS) + len(TRUONG_MOC)) + "I")
MAX_PAYLOAD = 64 * 1024 * 1024          # chan doc rac: 64 MB >> 3000 det x 24 B


class LoiGiaoThuc(Exception):
    """Luong du lieu khong dung giao thuc (magic sai, EOF giua thong diep, payload sai kich thuoc...)."""


def _dong_goi(loai, co, frame_idx, time_s, payload, t=None):
    t = t or {}
    vals = [float(t.get(k, 0.0)) for k in TRUONG_MS] + [float(t.get(k, 0.0)) for k in TRUONG_MOC]
    n_det = len(payload) // 24 if loai == FRAME else 0
    return HDR.pack(MAGIC, loai, co, 0, int(frame_idx), int(n_det), float(time_s), *(vals + [len(payload)])) + payload


def dong_goi_frame(frame_idx, time_s, det, co=0, t=None):
    """FRAME: det (n, 6) -> float32 little-endian. n = 0 van la mot frame (tracker van cap nhat vong doi)."""
    arr = np.ascontiguousarray(np.asarray(det, dtype="<f4").reshape(-1, 6))
    return _dong_goi(FRAME, co, frame_idx, time_s, arr.tobytes(), t)


def dong_goi_json(loai, obj):
    return _dong_goi(loai, 0, 0, 0.0, json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def doc_dung(f, n):
    """Doc DUNG n byte. Tra None neu EOF ngay tu byte dau (ranh gioi thong diep); EOF giua chung -> LoiGiaoThuc."""
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise LoiGiaoThuc("EOF giua thong diep: can %d byte, moi co %d" % (n, len(buf)))
        buf += chunk
    return bytes(buf)


def doc_thong_diep(f):
    """Doc MOT thong diep tu file nhi phan. None = EOF sach (khong con thong diep)."""
    h = doc_dung(f, HDR.size)
    if h is None:
        return None
    v = HDR.unpack(h)
    magic, loai, co, _pad, frame_idx, n_det, time_s = v[:7]
    n_ms, n_moc = len(TRUONG_MS), len(TRUONG_MOC)
    ms = v[7:7 + n_ms]
    moc = v[7 + n_ms:7 + n_ms + n_moc]
    n_payload = v[-1]
    if magic != MAGIC:
        raise LoiGiaoThuc("magic sai %r (mong %r)" % (magic, MAGIC))
    if loai not in TEN_LOAI:
        raise LoiGiaoThuc("loai thong diep la: %d" % loai)
    if n_payload > MAX_PAYLOAD:
        raise LoiGiaoThuc("payload qua lon: %d byte" % n_payload)
    payload = doc_dung(f, n_payload) if n_payload else b""
    if payload is None:
        raise LoiGiaoThuc("EOF truoc payload (%d byte)" % n_payload)
    tm = dict(zip(TRUONG_MS, ms))
    tm.update(zip(TRUONG_MOC, moc))
    msg = {"loai": loai, "ten_loai": TEN_LOAI[loai], "co": co, "frame_idx": frame_idx, "n_det": n_det,
           "time_s": time_s, "t": tm}
    if loai == FRAME:
        if n_payload != n_det * 24:
            raise LoiGiaoThuc("frame %d: payload %d byte != n_det %d x 24" % (frame_idx, n_payload, n_det))
        msg["det"] = np.frombuffer(payload, dtype="<f4").reshape(n_det, 6).astype(np.float32)
    else:
        try:
            msg["obj"] = json.loads(payload.decode("utf-8")) if payload else {}
        except ValueError as e:
            raise LoiGiaoThuc("payload JSON hong o thong diep %s: %s" % (TEN_LOAI[loai], e))
    return msg
