# -*- coding: utf-8 -*-
"""Vong 24: tach nguon lech khi chay bo kiem render tren PC thay vi Nano. So PNG xuat tren Nano (OpenCV 3.2, giai_ma_png_nano.py)
voi ket qua cung buoc tren PC (OpenCV cua interpreter nay):
  (a) frame GIAI MA tu cung file mp4 (cung sha256);
  (b) anh goc sau imread (giai ma JPEG, cung sha256 file);
  (c) net VE: ve_frame cua PC len CHINH anh goc cua Nano, so voi ban ve cua Nano (loai anh huong JPEG).
    python so_nguon_lech_pc_nano.py PNG_NANO_DIR VIDEO IMAGES MOT CLS
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import render_mot_video as rmv  # noqa: E402


def lech(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max": int(d.max()), "mae": round(float(d.mean()), 4), "ti_le_diem_khac": round(float(np.any(d > 0, axis=2).mean()), 5)}


def main():
    nd, vid, img, mot, cls = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
    info_nano = json.loads((nd / "xuat_png.json").read_text(encoding="utf-8"))
    frames = sorted(int(k) for k in info_nano["giai_ma"])
    boxes, lop = rmv.doc_mot(mot), rmv.doc_lop(cls)
    anh = rmv.liet_ke(img)
    cap = cv2.VideoCapture(vid)
    gm = {}
    i = 0
    while i < frames[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in frames:
            gm[i] = fr
    cap.release()
    kq = {"opencv_pc": cv2.__version__, "opencv_nano": info_nano["opencv"], "frame": {}}
    print("OpenCV PC %s vs Nano %s" % (cv2.__version__, info_nano["opencv"]))
    print("%5s | %-38s | %-38s | %-38s" % ("frame", "(a) giai ma mp4: max/MAE/ti le diem", "(b) imread JPEG", "(c) net ve (cung anh goc Nano)"))
    for i in frames:
        g_n = cv2.imread(str(nd / ("giai_ma_f%04d.png" % i)))
        goc_n = cv2.imread(str(nd / ("goc_f%04d.png" % i)))
        kv_n = cv2.imread(str(nd / ("ky_vong_f%04d.png" % i)))
        goc_p = cv2.imread(str(anh[i - 1]))
        kv_p = goc_n.copy()
        rmv.ve_frame(cv2, kv_p, boxes.get(i, []), lop, i)
        r = {"a_giai_ma": lech(gm[i], g_n), "b_imread": lech(goc_p, goc_n), "c_net_ve": lech(kv_p, kv_n),
             "giai_ma_nano_vs_ky_vong_nano_mae": round(float(np.abs(g_n.astype(np.int16) - kv_n.astype(np.int16)).mean()), 4),
             "giai_ma_pc_vs_ky_vong_nano_mae": round(float(np.abs(gm[i].astype(np.int16) - kv_n.astype(np.int16)).mean()), 4)}
        kq["frame"][i] = r
        f = lambda x: "%3d / %7.4f / %.5f" % (x["max"], x["mae"], x["ti_le_diem_khac"])
        print("%5d | %-38s | %-38s | %-38s" % (i, f(r["a_giai_ma"]), f(r["b_imread"]), f(r["c_net_ve"])))
    print("MAE so voi ban ve ky vong (Nano): giai ma Nano %s | giai ma PC %s" % (
        [kq["frame"][i]["giai_ma_nano_vs_ky_vong_nano_mae"] for i in frames],
        [kq["frame"][i]["giai_ma_pc_vs_ky_vong_nano_mae"] for i in frames]))
    if len(sys.argv) > 6:
        Path(sys.argv[6]).write_text(json.dumps(kq, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
