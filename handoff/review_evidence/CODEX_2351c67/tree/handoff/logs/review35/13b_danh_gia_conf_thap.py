#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- CHAN DOAN bo sung cho 13_*: chay CHINH evaluator eval_detector_5lop (cung dataset dev smoke da xac minh,
cung anh xa lop, cung NMS) nhung ha nguong conf du doan 0,001 -> 1e-6, cho best.pt smoke lan 2 cua hai model.

Muc dich: phan biet "diem lop bi day xuong duoi 0,001 nhung xep hang/anh xa lop van dung" voi "loi duong ong" (sai lop,
sai toa do). KHONG thay ket qua danh gia smoke chinh thuc (conf 0,001 trong results/phase2/detector_smoke/*_lan2.json),
KHONG dung de so sanh model hay chon tham so.
Chay tu goc repo:  python handoff/logs/review35/13b_danh_gia_conf_thap.py --ra handoff/logs/review35/13b_danh_gia_conf_thap.json
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    import eval_detector_5lop as ev
    quy_tac = copy.deepcopy(ev.QUY_TAC["du_doan"])
    quy_tac["conf"] = 1e-6
    dt_dev, anh, gt = ev.chuan_bi_danh_gia(str(GOC / "data/yolo_bdd5_v1_smoke/data.yaml"), "val")
    kq = {"ghi_chu": "CHAN DOAN: evaluator voi conf du doan 1e-6 (chinh thuc 0,001); khong so sanh model",
          "quy_tac_du_doan": quy_tac, "quy_tac_van_hanh": ev.QUY_TAC["van_hanh"], "so_anh": len(anh), "ket_qua": {}}
    for m in ("yolov8n", "yolo11n"):
        w = GOC / f"models/finetune_phase2/train_bdd5_v1_{m}_smoke_lan2/weights/best.pt"
        dd, mt = ev.du_doan_ultralytics(str(w), anh, quy_tac, device=a.device)
        d = ev.danh_gia(gt, dd)
        kq["ket_qua"][m] = {"weights": str(w.relative_to(GOC)).replace("\\", "/"), "anh_xa_lop": mt, **d}
        print(f"{m}: " + " | ".join(
            f"{c} n_gt {v['n_gt']} n_det {v['n_det_conf_thap']} AP50 "
            f"{'-' if v['AP50'] is None else round(v['AP50'], 4)}" for c, v in d["theo_lop"].items())
            + f" | mAP50 {round(d['macro']['mAP50'], 4)} mAP50_95 {round(d['macro']['mAP50_95'], 4)}", flush=True)
    def sach(x):                                    # NaN (AP lop khong co GT) -> null: JSON chuan
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, dict):
            return {k: sach(v) for k, v in x.items()}
        if isinstance(x, list):
            return [sach(v) for v in x]
        return x
    Path(a.ra).write_text(json.dumps(sach(kq), ensure_ascii=False, indent=1, default=str, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")
    print(f"[ra] {a.ra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
