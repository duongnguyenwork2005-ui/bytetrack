# -*- coding: utf-8 -*-
"""Vong 25: so anh XEM TRUOC da luu trong luc tracking (PNG khong mat mat, `--hien-thi-luu-mau`) voi
render_mot_video.ve_frame(anh goc cua CUNG frame, hang MOT/lop DA GHI cua frame do). Khop tung pixel = nen, bbox, ID, lop
tren anh hien thi thuoc dung frame va dung ket qua da luu. Chay bang CUNG interpreter/OpenCV voi tien trinh detector
(Nano: /usr/bin/python3, OpenCV 3.2). Cu phap Python 3.6.

    python3 so_mau_hien_thi.py <thu_muc_mau> <thu_muc_anh_nguon> <mot.txt> <cls.csv> <goc_goi/src> [ket_qua.json]
"""
import json
import os
import sys
from pathlib import Path


def main():
    mau, anh, mot, cls, src = sys.argv[1:6]
    sys.path.insert(0, src)
    import cv2
    import render_mot_video as rmv
    boxes, lop = rmv.doc_mot(mot), rmv.doc_lop(cls)
    ds = rmv.liet_ke(Path(anh))
    r = {"opencv": cv2.__version__, "n": 0, "khop": 0, "lech": [], "frame": []}
    for f in sorted(os.listdir(mau)):
        if not (f.startswith("hien_thi_f") and f.endswith(".png")):
            continue
        i = int(f[len("hien_thi_f"):-4])
        got = cv2.imread(os.path.join(mau, f))
        kv = cv2.imread(str(ds[i - 1]))
        rmv.ve_frame(cv2, kv, boxes.get(i, []), lop, i)
        r["n"] += 1
        r["frame"].append(i)
        if got is not None and got.shape == kv.shape and (got == kv).all():
            r["khop"] += 1
        else:
            r["lech"].append(i)
    r["dat"] = r["n"] > 0 and r["khop"] == r["n"]
    s = json.dumps(r)
    print(s)
    if len(sys.argv) > 6:
        with open(sys.argv[6], "w") as fo:
            fo.write(s)
    return 0 if r["dat"] else 1


if __name__ == "__main__":
    sys.exit(main())
