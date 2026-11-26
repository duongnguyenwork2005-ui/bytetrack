"""
nano_det_compare.py -- So DETECTION giua hai det cache (vd cache PC .pt FP32 rect vs cache Nano TensorRT FP16 640x640)
de tach "sai khac detector" khoi "loi tracker". MO TA, khong phai phep kiem dat/khong dat (khong ky vong bit-identical).

    python src/nano_det_compare.py --a yolov8n-coco_conf0.05_all --b nano_trt_MVI_39031_f0001-0300 --video MVI_39031 \
        [--max-frames 300] [--conf-min 0.25 --classes 1 2 3 5 7 --max-det 300] [--iou 0.5] --json out.json

--a/--b: cache_id (trong data/interim/det_cache) hoac duong dan thu muc cache. Chi so frame chung (giao cua frame_idx).
Tuy chon loc (--conf-min/--classes/--max-det) dung `det_cache.filter_dets` = dung dieu kien vao tracker.
Ghep tung frame: tham lam theo IoU giam dan (class-agnostic), cap IoU >= --iou. Bao cao: so box moi ben, so cap,
khong ghep moi ben, IoU trung binh/nho nhat, |d conf| trung binh/lon nhat, ti le trung lop, phan bo chenh so box/frame.
Exit 2 khi thieu dau vao; con lai exit 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache  # noqa: E402


def _dir(x: str) -> Path:
    p = Path(x)
    return p if p.is_dir() else det_cache.CACHE_ROOT / x


def nap(cdir: Path, video: str, conf_min, classes, max_det):
    npz = cdir / f"{video}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    z = np.load(npz)
    off = np.concatenate([[0], np.cumsum(z["n_per_frame"])])
    cls_set = None if classes is None else np.array(sorted(set(int(c) for c in classes)))
    out = {}
    for i, f in enumerate(z["frame_idx"]):
        arr = z["data"][off[i]:off[i + 1]].astype(np.float32)
        if conf_min is not None:
            arr = det_cache.filter_dets(arr, float(conf_min), cls_set, int(max_det), None)
        out[int(f)] = arr
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8")) if (cdir / "meta.json").is_file() else {}
    return out, meta


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = [a[:, i][:, None] for i in range(4)]
    bx1, by1, bx2, by2 = [b[:, i][None, :] for i in range(4)]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def ghep(a: np.ndarray, b: np.ndarray, iou_min: float):
    """Tham lam theo IoU giam dan. Tra ve list (ia, ib, iou)."""
    m = iou_matrix(a, b)
    pairs = []
    if m.size == 0:
        return pairs
    used_a, used_b = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    for ia, ib in order:
        if m[ia, ib] < iou_min:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia); used_b.add(ib); pairs.append((int(ia), int(ib), float(m[ia, ib])))
    return pairs


def so_sanh(A: dict, B: dict, iou_min: float) -> dict:
    frames = sorted(set(A) & set(B))
    n_a = n_b = n_pair = 0
    ious, dconf, cls_ok, dn = [], [], [], []
    chi_a_frames = sorted(set(A) - set(B)); chi_b_frames = sorted(set(B) - set(A))
    for f in frames:
        a, b = A[f], B[f]
        n_a += len(a); n_b += len(b); dn.append(len(b) - len(a))
        for ia, ib, v in ghep(a, b, iou_min):
            n_pair += 1; ious.append(v); dconf.append(abs(float(a[ia, 4]) - float(b[ib, 4]))); cls_ok.append(int(a[ia, 5]) == int(b[ib, 5]))
    dn = np.asarray(dn, dtype=int)
    return {"iou_ghep_toi_thieu": iou_min, "n_frame_chung": len(frames), "frame_chi_a": len(chi_a_frames), "frame_chi_b": len(chi_b_frames),
            "n_box_a": n_a, "n_box_b": n_b, "n_cap": n_pair, "khong_ghep_a": n_a - n_pair, "khong_ghep_b": n_b - n_pair,
            "ti_le_ghep_a": (n_pair / n_a if n_a else None), "ti_le_ghep_b": (n_pair / n_b if n_b else None),
            "iou_trung_binh": (float(np.mean(ious)) if ious else None), "iou_nho_nhat": (float(np.min(ious)) if ious else None),
            "d_conf_trung_binh": (float(np.mean(dconf)) if dconf else None), "d_conf_lon_nhat": (float(np.max(dconf)) if dconf else None),
            "ti_le_trung_lop": (float(np.mean(cls_ok)) if cls_ok else None),
            "chenh_so_box_moi_frame": {"trung_binh": (float(dn.mean()) if len(dn) else None), "abs_trung_binh": (float(np.abs(dn).mean()) if len(dn) else None),
                                       "max": (int(dn.max()) if len(dn) else None), "min": (int(dn.min()) if len(dn) else None),
                                       "frame_bang_nhau": int((dn == 0).sum())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--conf-min", type=float, default=None); ap.add_argument("--classes", nargs="*", type=int, default=None)
    ap.add_argument("--max-det", type=int, default=300); ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    try:
        A, ma = nap(_dir(a.a), a.video, a.conf_min, a.classes, a.max_det)
        B, mb = nap(_dir(a.b), a.video, a.conf_min, a.classes, a.max_det)
    except FileNotFoundError as e:
        print(f"[LOI] thieu dau vao: {e}"); return 2
    if a.max_frames:
        A = {f: v for f, v in A.items() if f <= a.max_frames}; B = {f: v for f, v in B.items() if f <= a.max_frames}
    kq = so_sanh(A, B, a.iou)
    kq.update({"a": str(_dir(a.a)), "b": str(_dir(a.b)), "video": a.video, "loc": {"conf_min": a.conf_min, "classes": a.classes, "max_det": a.max_det},
               "meta_a": {k: ma.get(k) for k in ("backend", "imgsz", "conf", "iou", "half", "letterbox", "device")},
               "meta_b": {k: mb.get(k) for k in ("backend", "imgsz", "conf", "iou", "half", "letterbox", "device")},
               "ghi_chu": "MO TA sai khac detector (khong phai dat/khong dat). Khac FP16/letterbox -> khong ky vong bit-identical."})
    print(f"[det_compare] {a.video}: frame chung {kq['n_frame_chung']} | box a={kq['n_box_a']} b={kq['n_box_b']} cap={kq['n_cap']} "
          f"(ghep a {kq['ti_le_ghep_a'] if kq['ti_le_ghep_a'] is None else round(kq['ti_le_ghep_a'], 4)}, b {kq['ti_le_ghep_b'] if kq['ti_le_ghep_b'] is None else round(kq['ti_le_ghep_b'], 4)}) "
          f"| IoU tb {kq['iou_trung_binh']} | |dconf| tb {kq['d_conf_trung_binh']} max {kq['d_conf_lon_nhat']} | trung lop {kq['ti_le_trung_lop']} "
          f"| chenh box/frame tb {kq['chenh_so_box_moi_frame']['trung_binh']} (bang nhau {kq['chenh_so_box_moi_frame']['frame_bang_nhau']} frame)")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
