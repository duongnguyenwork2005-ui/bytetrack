"""R63 G4: tuong duong Nano <-> PC cho mot ung vien. So detection TensorRT FP16 cua luot Nano (cache live, khong to xam,
conf 0,05) voi PC ultralytics FP32 (det_cache.detect_frame, cung imgsz/rect, khong to xam) tren CUNG cac frame.
Ghep 1-1 occlusion_eval.match_frame IoU >= 0,5, khong phan biet lop. Tieu chi CHOT TRUOC (ke hoach muc G4):
ti le ghep >= 99 % HAI CHIEU va IoU trung vi >= 0,98 o conf > 0,25 (dieu kien tracking c25).

    python handoff/logs/review63/cong_cu/g4_nano_pc.py --nano-npz <MVI_39031.npz> --anh <thu muc img%05d.jpg> \
        [--no-rect] --ra <json>
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano-npz", required=True)
    ap.add_argument("--anh", required=True)
    ap.add_argument("--no-rect", action="store_true")
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    if Path(a.ra).exists():
        print("[LOI] da ton tai", a.ra); return 2
    import cv2
    import det_cache as dc
    from occlusion_eval import iou_matrix, match_frame
    from ultralytics import YOLO
    z = np.load(a.nano_npz)
    fidx, nper, data = z["frame_idx"], z["n_per_frame"], z["data"]
    off = np.concatenate([[0], np.cumsum(nper)])
    model = YOLO(str(GOC / "models" / "yolov8n.pt"))
    spec = dc.DetectorSpec(weights_sha256="-" * 64, conf=0.05, iou=0.7, max_det=3000, rect=not a.no_rect, device="0")
    h = hashlib.sha256()
    kq = {}
    nano_f, pc_f = [], []
    for k, f in enumerate(fidx):
        p = Path(a.anh) / ("img%05d.jpg" % int(f))
        h.update(p.read_bytes())
        img = cv2.imread(str(p))
        nano_f.append(data[off[k]:off[k + 1]])
        pc_f.append(dc.detect_frame(model, img, spec))
    for nguong in (0.05, 0.25):
        nn = npc = ghep = cung_lop = 0
        ious = []
        for dn, dp in zip(nano_f, pc_f):
            dn = dn[dn[:, 4] > nguong]; dp = dp[dp[:, 4] > nguong]
            nn += len(dn); npc += len(dp)
            xa = np.c_[dn[:, :2], dn[:, 2:4] - dn[:, :2]]; xb = np.c_[dp[:, :2], dp[:, 2:4] - dp[:, :2]]
            M = iou_matrix(xa, xb); cap = match_frame(M, 0.5)
            ghep += len(cap)
            for r, c in cap:
                ious.append(M[r, c]); cung_lop += int(dn[r, 5] == dp[c, 5])
        kq[f"conf>{nguong}"] = {"n_nano": nn, "n_pc": npc, "n_ghep": ghep, "ty_le_ghep_nano": ghep / max(nn, 1),
                                "ty_le_ghep_pc": ghep / max(npc, 1), "iou_trung_vi": float(np.median(ious)) if ious else None,
                                "iou_p5": float(np.percentile(ious, 5)) if ious else None, "cung_lop": cung_lop / max(ghep, 1)}
    c25 = kq["conf>0.25"]
    dat = c25["ty_le_ghep_nano"] >= 0.99 and c25["ty_le_ghep_pc"] >= 0.99 and (c25["iou_trung_vi"] or 0) >= 0.98
    out = {"nano_npz": a.nano_npz, "anh": a.anh, "n_frame": len(fidx), "anh_sha256_tong": h.hexdigest(),
           "pc": {"rect": spec.rect, "imgsz": spec.imgsz, "fp32": True, "to_xam": False}, "ket_qua": kq,
           "tieu_chi": ">=99% hai chieu va IoU trung vi >=0.98 o conf>0.25", "dat": bool(dat)}
    Path(a.ra).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(kq, indent=1)); print("DAT" if dat else "KHONG DAT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
