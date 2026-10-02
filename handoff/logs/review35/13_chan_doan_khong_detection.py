#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- CHAN DOAN (chi doc, khong train, khong doi tham so): vi sao best.pt YOLO11n sau smoke 1 epoch cho 0 detection
tren dev smoke (evaluator conf 0,001: n_det = 0 moi lop; ultralytics val P/R/mAP = 0), trong khi YOLOv8n cung du lieu,
cung runner va cung evaluator van co detection.

Chay tu goc repo:  python handoff/logs/review35/13_chan_doan_khong_detection.py --ra handoff/logs/review35/13_chan_doan_khong_detection.json
So 4 checkpoint: khoi tao COCO (hang lop COCO cung ten: car 2, truck 7, bus 5, motorcycle 3, bicycle 1 -- dung cac hang
ma cls_remap da chep sang dau 5 lop) va best.pt smoke lan 2 cua tung model:
- bias cua lop Conv2d cuoi nhanh phan lop (Detect.cv3[i][-1]) theo tung lop va tung tang dac trung;
- tren 84 anh dev smoke: diem lop lon nhat moi anh (predict conf 1e-6, iou 0,7, max_det 300, imgsz 640, FP32), so box
  >= 0,001 (nguong evaluator) va >= 0,25 (nguong van hanh).
Ket qua chi mo ta checkpoint smoke; KHONG dung de so sanh model hay du doan training dai.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
LOP5 = ["car", "truck", "bus", "motorcycle", "bicycle"]
COCO_ID = {"car": 2, "truck": 7, "bus": 5, "motorcycle": 3, "bicycle": 1}
CKPT = {
    "yolov8n_khoi_tao_coco": ("models/yolov8n.pt", True),
    "yolov8n_smoke_lan2_best": ("models/finetune_phase2/train_bdd5_v1_yolov8n_smoke_lan2/weights/best.pt", False),
    "yolo11n_khoi_tao_coco": ("models/candidates/yolo11n.pt", True),
    "yolo11n_smoke_lan2_best": ("models/finetune_phase2/train_bdd5_v1_yolo11n_smoke_lan2/weights/best.pt", False),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def trung_vi(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO
    import ultralytics
    import torch
    anh = sorted(str(p) for p in (GOC / "data/yolo_bdd5_v1_smoke/images/dev").glob("*.jpg"))
    kq = {"muc_dich": __doc__.split("\n")[1], "ultralytics": ultralytics.__version__, "torch": torch.__version__,
          "so_anh_dev": len(anh), "checkpoint": {}}
    for ten, (w, coco) in CKPT.items():
        p = GOC / w
        m = YOLO(str(p))
        det = m.model.model[-1]
        hang = [COCO_ID[c] for c in LOP5] if coco else list(range(5))
        names = m.model.names
        bias = {c: [round(float(det.cv3[i][-1].bias[r]), 3) for i in range(det.nl)] for c, r in zip(LOP5, hang)}
        max_anh, n_001, n_025 = [], 0, 0
        for r in m.predict(anh, conf=1e-6, iou=0.7, max_det=300, classes=hang if coco else None, imgsz=640,
                           device=a.device, half=False, verbose=False, stream=True):
            c = r.boxes.conf
            max_anh.append(float(c.max()) if len(c) else 0.0)
            n_001 += int((c >= 0.001).sum())
            n_025 += int((c >= 0.25).sum())
        kq["checkpoint"][ten] = {
            "duong_dan": w, "sha256": sha256(p), "bytes": p.stat().st_size, "so_lop_dau": int(det.nc),
            "hang_lop_dung": {c: int(r) for c, r in zip(LOP5, hang)}, "ten_hang": {c: names[r] for c, r in zip(LOP5, hang)},
            "bias_phan_lop_theo_tang_P3_P4_P5": bias,
            "diem_lop_max_moi_anh": {"trung_vi": trung_vi(max_anh), "max": max(max_anh) if max_anh else None,
                                     "min": min(max_anh) if max_anh else None},
            "so_box_conf_ge_0_001": n_001, "so_box_conf_ge_0_25": n_025}
        k = kq["checkpoint"][ten]
        print(f"{ten:26s} nc={k['so_lop_dau']:2d} | bias car P3/P4/P5 {bias['car']} | max diem/anh: trung vi "
              f"{k['diem_lop_max_moi_anh']['trung_vi']:.6f} max {k['diem_lop_max_moi_anh']['max']:.6f} | "
              f"box>=0,001: {n_001} | box>=0,25: {n_025}", flush=True)
        del m
        torch.cuda.empty_cache()
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"[ra] {a.ra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
