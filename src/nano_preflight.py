# -*- coding: utf-8 -*-
"""
nano_preflight.py -- MOT hop dong kiem DAU VAO trien khai (goi nano_transfer) dung chung cho `nano_env_check.py`
(chay duoc bang Python 3.6 he thong de chan doan) va `nano_run_cache.py` (venv >= 3.8). Review 20207ec (N15-P1/P3):
"co thu muc / co file" KHONG du -- phai doc/parse duoc va dung schema, thieu bat ky thu gi -> loi, khong chay tracker.

CO Y viet bang cu phap Python 3.6 (khong f-string, khong annotation kieu list[str], khong dataclass, khong
`from __future__ import annotations`). Chi phu thuoc numpy.

Hop dong (kiem_dau_vao(goc) -> dict):
  - DUNG HAI cache: CACHES = {smoke: nano_smoke_MVI_39031_f0001-0300, full: yolov8n-coco_conf0.05_all}; video MVI_39031.
    Moi cache: meta.json parse duoc + co conf/max_det/classes; MVI_39031.npz ton tai, np.load duoc, du 7 mang, cac mang
    cung do dai >= 1, data (N, 6), sum(n_per_frame) == N, frame_idx tang ngat, time_s tang ngat, orig_shape (2,) > 0;
    MVI_39031.json parse duoc va khop schema: video == MVI_39031, n_frames == so frame, n_boxes == N.
  - DU BON tham chieu PC doc duoc: NANO-{smoke,full}/pc-ref-{cv,ekf}/data/MVI_39031.txt: file text, >= 1 dong, MOI dong
    >= 6 truong so (MOTChallenge), frame >= 1.
  - YAML tracker: third_party/ultralytics_trackers/bytetrack.yaml (CV) va configs/bytetrack_ekf_ctrv.yaml (EKF).
  - Ma: src/run_tracking.py, src/nano_parity.py, third_party/ultralytics_trackers/byte_tracker.py.
Tra ve {"loi": [...], "cache": {...}, "pc_ref": {...}}; `loi` rong <=> DAT.
"""
import json
import os

import numpy as np

VIDEO = "MVI_39031"
CACHES = {"smoke": "nano_smoke_MVI_39031_f0001-0300", "full": "yolov8n-coco_conf0.05_all"}
SPLITS = {"smoke": "NANO-smoke", "full": "NANO-full"}
TRACKERS = ("cv", "ekf")
YAMLS = ("third_party/ultralytics_trackers/bytetrack.yaml", "configs/bytetrack_ekf_ctrv.yaml")
MA = ("src/run_tracking.py", "src/nano_parity.py", "third_party/ultralytics_trackers/byte_tracker.py")
MANG_CAN = ("frame_idx", "n_per_frame", "unreadable", "truncated", "time_s", "data", "orig_shape")


def _doc_json(path):
    with open(path, "r") as f:
        return json.load(f)


def kiem_cache(cdir, ten):
    """Kiem MOT cache. Tra ve (loi: list, thong_tin: dict)."""
    loi = []
    tt = {}
    if not os.path.isdir(cdir):
        return ["cache %s: khong co thu muc %s" % (ten, cdir)], tt
    # meta.json
    mp = os.path.join(cdir, "meta.json")
    if not os.path.isfile(mp):
        loi.append("cache %s: thieu meta.json (%s)" % (ten, mp))
    else:
        try:
            meta = _doc_json(mp)
            for k in ("conf", "max_det", "classes"):
                if k not in meta:
                    loi.append("cache %s: meta.json thieu khoa '%s'" % (ten, k))
            tt["meta_conf"] = meta.get("conf")
        except Exception as e:  # noqa: BLE001
            loi.append("cache %s: meta.json khong parse duoc (%s: %s)" % (ten, type(e).__name__, e))
    # npz
    npz = os.path.join(cdir, VIDEO + ".npz")
    n_frame = n_box = None
    if not os.path.isfile(npz):
        loi.append("cache %s: thieu %s" % (ten, npz))
    else:
        try:
            z = np.load(npz)
            thieu = [k for k in MANG_CAN if k not in z.files]
            if thieu:
                loi.append("cache %s: npz thieu mang %s" % (ten, thieu))
            else:
                fi, npf, un, tr, ts = z["frame_idx"], z["n_per_frame"], z["unreadable"], z["truncated"], z["time_s"]
                data, osh = z["data"], z["orig_shape"]
                n_frame = int(len(fi))
                if n_frame < 1:
                    loi.append("cache %s: npz khong co frame nao" % ten)
                if not (len(npf) == len(un) == len(tr) == len(ts) == n_frame):
                    loi.append("cache %s: cac mang theo frame khong cung do dai" % ten)
                if data.ndim != 2 or data.shape[1] != 6:
                    loi.append("cache %s: data phai co dang (N, 6), dang la %s" % (ten, tuple(data.shape)))
                n_box = int(len(data))
                if int(npf.sum()) != n_box:
                    loi.append("cache %s: sum(n_per_frame)=%d != len(data)=%d" % (ten, int(npf.sum()), n_box))
                if (npf < 0).any():
                    loi.append("cache %s: n_per_frame am" % ten)
                if n_frame > 1 and not bool(np.all(np.diff(fi.astype(np.int64)) > 0)):
                    loi.append("cache %s: frame_idx khong tang ngat" % ten)
                if n_frame > 1 and not bool(np.all(np.diff(ts.astype(np.float64)) > 0)):
                    loi.append("cache %s: time_s khong tang ngat (timeline khong hop le)" % ten)
                if osh.shape != (2,) or not bool(np.all(osh > 0)):
                    loi.append("cache %s: orig_shape phai la (h, w) > 0" % ten)
                tt.update({"n_frame": n_frame, "n_box": n_box, "n_frame_rong": int((npf == 0).sum()),
                           "time_s_dau_cuoi": [float(ts[0]), float(ts[-1])] if n_frame else None})
        except Exception as e:  # noqa: BLE001
            loi.append("cache %s: npz khong doc duoc (%s: %s)" % (ten, type(e).__name__, e))
    # sidecar
    sp = os.path.join(cdir, VIDEO + ".json")
    if not os.path.isfile(sp):
        loi.append("cache %s: thieu sidecar %s" % (ten, sp))
    else:
        try:
            side = _doc_json(sp)
            if not isinstance(side, dict):
                loi.append("cache %s: sidecar khong phai JSON object" % ten)
            else:
                if side.get("video") != VIDEO:
                    loi.append("cache %s: sidecar video=%r != %r" % (ten, side.get("video"), VIDEO))
                if n_frame is not None and side.get("n_frames") != n_frame:
                    loi.append("cache %s: sidecar n_frames=%r != so frame trong npz %d" % (ten, side.get("n_frames"), n_frame))
                if n_box is not None and side.get("n_boxes") != n_box:
                    loi.append("cache %s: sidecar n_boxes=%r != len(data) %d" % (ten, side.get("n_boxes"), n_box))
        except Exception as e:  # noqa: BLE001
            loi.append("cache %s: sidecar %s khong parse duoc (%s: %s)" % (ten, os.path.basename(sp), type(e).__name__, e))
    return loi, tt


def kiem_tham_chieu(path):
    """Kiem MOT file tracking tham chieu (MOTChallenge). Tra ve (loi: list, n_hang)."""
    if not os.path.isfile(path):
        return ["tham chieu PC thieu: %s" % path], 0
    n = 0
    try:
        with open(path, "r") as f:
            for i, ln in enumerate(f):
                ln = ln.strip()
                if not ln:
                    continue
                v = ln.split(",")
                if len(v) < 6:
                    return ["tham chieu PC %s: dong %d co %d truong (< 6)" % (path, i + 1, len(v))], n
                try:
                    fr = float(v[0]); float(v[1]); [float(t) for t in v[2:6]]
                except ValueError:
                    return ["tham chieu PC %s: dong %d khong phai so: %r" % (path, i + 1, ln[:60])], n
                if fr < 1:
                    return ["tham chieu PC %s: dong %d frame < 1" % (path, i + 1)], n
                n += 1
    except Exception as e:  # noqa: BLE001
        return ["tham chieu PC %s khong doc duoc (%s: %s)" % (path, type(e).__name__, e)], n
    if n == 0:
        return ["tham chieu PC %s rong (0 dong)" % path], 0
    return [], n


def kiem_dau_vao(goc):
    """Toan bo hop dong. goc = thu muc goi (chua src/, data/, ...)."""
    goc = os.path.abspath(goc)
    loi = []
    cache = {}
    for ten, cid in CACHES.items():
        l, tt = kiem_cache(os.path.join(goc, "data", "interim", "det_cache", cid), ten)
        loi.extend(l)
        cache[ten] = dict(tt, cache_id=cid)
    pc_ref = {}
    for ten, split in SPLITS.items():
        for t in TRACKERS:
            p = os.path.join(goc, "data", "processed", "trackers", split, "pc-ref-" + t, "data", VIDEO + ".txt")
            l, n = kiem_tham_chieu(p)
            loi.extend(l)
            pc_ref["%s/pc-ref-%s" % (split, t)] = {"n_hang": n, "ok": not l}
    for y in YAMLS:
        if not os.path.isfile(os.path.join(goc, y)):
            loi.append("YAML tracker thieu: %s" % y)
    for m in MA:
        if not os.path.isfile(os.path.join(goc, m)):
            loi.append("thieu ma: %s" % m)
    return {"loi": loi, "cache": cache, "pc_ref": pc_ref, "dat": not loi}


def mo_ta(kq):
    """Cac dong mo ta ngan (DAT) de in."""
    out = []
    for ten, tt in kq["cache"].items():
        if "n_frame" in tt:
            out.append("cache %s (%s): %d frame (%d frame RONG giu nguyen), %d box, time_s %.2f..%.2f" %
                       (ten, tt["cache_id"], tt["n_frame"], tt["n_frame_rong"], tt["n_box"], tt["time_s_dau_cuoi"][0], tt["time_s_dau_cuoi"][1]))
    out.append("tham chieu PC: " + ", ".join("%s (%d hang)" % (k, v["n_hang"]) for k, v in sorted(kq["pc_ref"].items())))
    return out
