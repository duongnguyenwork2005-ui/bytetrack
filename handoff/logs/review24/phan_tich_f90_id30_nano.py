# -*- coding: utf-8 -*-
"""Vong 24: phan tich box DUY NHAT khong nhan dang dung ten lop khi kiem MOI box (frame 90, id 30, ky vong 'car', bo kiem chon
'bus'). Chay bang CUNG OpenCV da render (Nano, /usr/bin/python3, OpenCV 3.2). Cu phap Python 3.6.

    python phan_tich_f90_id30_nano.py --src SRC --video V --images IMG --mot MOT --cls CLS --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    for k in ("--src", "--video", "--images", "--mot", "--cls", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--frame", type=int, default=90); ap.add_argument("--id", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import render_mot_video as rmv
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    i, tid = a.frame, a.id
    boxes, lop = rmv.doc_mot(a.mot), rmv.doc_lop(a.cls)
    anh = rmv.liet_ke(Path(a.images))
    cap = cv2.VideoCapture(a.video)
    for _ in range(i):
        ok, fr = cap.read()
    cap.release()
    goc = cv2.imread(str(anh[i - 1]))
    bf = boxes[i]
    ve = {}
    for c in [None, 1, 2, 3, 5, 7]:
        l2 = dict(lop)
        if c is None:
            l2.pop((i, tid), None)
        else:
            l2[(i, tid)] = c
        im = goc.copy()
        rmv.ve_frame(cv2, im, bf, l2, i)
        ve[c] = im
    mong = lop[(i, tid)]
    kv = ve[mong]
    f32 = lambda x: x.astype(np.float32)
    mae = lambda x, y, m: float(np.abs(f32(x) - f32(y))[m].mean())
    vung = np.zeros(goc.shape[:2], bool)
    for im in ve.values():
        vung |= np.any(im != kv, axis=2)
    kq = {"opencv": cv2.__version__, "frame": i, "id": tid, "lop_ky_vong": rmv.COCO.get(mong, mong),
          "box": [b for b in bf if b[0] == tid][0][1:5], "n_diem_vung_hop": int(vung.sum()),
          "mae_vung_hop": {str(rmv.COCO.get(c, c)): round(mae(fr, im, vung), 3) for c, im in ve.items()}}
    pb = np.any(ve[2] != ve[5], axis=2)
    kq["car_vs_bus"] = {"n_diem_khac": int(pb.sum()), "mae_giai_ma_vs_car": round(mae(fr, ve[2], pb), 3),
                        "mae_giai_ma_vs_bus": round(mae(fr, ve[5], pb), 3),
                        "mae_car_vs_bus_truoc_ma_hoa": round(mae(ve[2], ve[5], pb), 3)}
    # chat luong giai ma vung nhan cua box nay so voi cac nhan khac trong 300 frame dau (cung phien ban ve)
    x, y = int(kq["box"][0]), int(kq["box"][1])
    y0 = max(12, y - 4)
    y1, y2, x1, x2 = y0 - 14, y0 + 4, x, min(goc.shape[1], x + 60)
    m_nhan = np.any(kv[y1:y2, x1:x2] != goc[y1:y2, x1:x2], axis=2)
    kq["nhan_nay"] = {"mae_giai_ma_vs_ky_vong_tren_diem_chu": round(mae(fr[y1:y2, x1:x2], kv[y1:y2, x1:x2], m_nhan), 3),
                      "tuong_phan_chu_vs_nen": round(float(np.abs(f32(kv[y1:y2, x1:x2]) - f32(goc[y1:y2, x1:x2]))[m_nhan].mean()), 3)}
    hang = np.concatenate([cv2.resize(im[y1 - 4:y2 + 4, x1 - 10:x2], None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                           for im in (fr, ve[2], ve[5], goc)], axis=0)
    cv2.imwrite(str(out / ("f%d_id%d_giaima_car_bus_goc.png" % (i, tid))), hang)
    (out / ("f%d_id%d.json" % (i, tid))).write_text(json.dumps(kq, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(kq, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
