# -*- coding: utf-8 -*-
"""Vong 24: xuat PNG (khong mat mat) cua mot so frame bang OpenCV cua interpreter dang chay, de tach nguon lech giua
Nano (OpenCV 3.2) va PC: (a) frame GIAI MA tu video, (b) anh goc sau imread (giai ma JPEG), (c) ban VE ky vong truoc ma
hoa (anh goc + render_mot_video.ve_frame). Cu phap Python 3.6.

    python giai_ma_png_nano.py --video V --out DIR [--src SRC --images IMG --mot MOT --cls CLS] 1 12 90 ...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="thu muc src/ chua render_mot_video.py")
    ap.add_argument("--images", default=None); ap.add_argument("--mot", default=None); ap.add_argument("--cls", default=None)
    ap.add_argument("frames", nargs="+", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    muon = sorted(set(a.frames))
    h = lambda im: hashlib.sha256(im.tobytes()).hexdigest()
    info = {"video": a.video, "opencv": cv2.__version__, "python": sys.version.split()[0], "giai_ma": {}, "goc": {}, "ky_vong": {}}
    cap = cv2.VideoCapture(a.video)
    i = 0
    while i < muon[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in muon:
            cv2.imwrite(str(out / ("giai_ma_f%04d.png" % i)), fr)
            info["giai_ma"][i] = h(fr)
    cap.release()
    if a.images:
        sys.path.insert(0, a.src)
        import render_mot_video as rmv
        anh = rmv.liet_ke(Path(a.images))
        boxes, lop = rmv.doc_mot(a.mot), rmv.doc_lop(a.cls)
        for i in muon:
            goc = cv2.imread(str(anh[i - 1]))
            kv = goc.copy()
            rmv.ve_frame(cv2, kv, boxes.get(i, []), lop, i)
            cv2.imwrite(str(out / ("goc_f%04d.png" % i)), goc)
            cv2.imwrite(str(out / ("ky_vong_f%04d.png" % i)), kv)
            info["goc"][i], info["ky_vong"][i] = h(goc), h(kv)
    (out / "xuat_png.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
