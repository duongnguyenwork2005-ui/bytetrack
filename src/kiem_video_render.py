"""
kiem_video_render.py -- Kiem video do `render_mot_video.py` tao ra: DUNG so frame, DUNG FPS/kich thuoc, bbox/ID ve DUNG CHO
tren DUNG frame, ten lop DUNG. Cu phap Python 3.6 (chay bang Python he thong cua Jetson, co cv2).

Codec mp4v co mat mat -> khong so bang nhau tung byte. Voi moi frame i (ve lai bang CHINH `render_mot_video.ve_frame`):
  (1) KHUNG + SO FRAME (moi frame): tap diem = noi (anh + ve day du) khac (anh + chi so frame); tren tap do, sai so trung binh
      cua frame GIAI MA so voi ban ky vong phai <= TI_LE x sai so so voi ban khong ve, va <= TRAN. Box lech cho, sai frame,
      sai mau (mau theo ID) deu day ti le len. Frame khong box: kiem phan so frame.
  (2) TEN LOP (mau moi `--buoc-lop` frame): voi TUNG box, ve lai frame voi moi ung vien ten lop (5 lop COCO + khong ten) va
      chon ung vien gan frame giai ma nhat trong vung chu cua box -> phai trung ten lop ky vong.

    python src/kiem_video_render.py --video out.mp4 --images DIR --mot MOT.txt [--cls CLS.csv] --fps 25 [--max-frames N]
        [--buoc-lop 5] [--json ket_qua.json]
exit 0 = DAT moi phep; 1 = co phep KHONG DAT; 2 = khong mo duoc video / thieu dau vao.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_mot_video as rmv  # noqa: E402

TI_LE = 0.5        # sai so tren diem ve so voi ban ky vong phai <= 50 % so voi ban khong ve
TRAN = 40.0        # va <= 40 (thang 0-255, trung binh tri tuyet doi 3 kenh) -- du rong cho mat mat mp4v
UNG_VIEN_LOP = [None, 1, 2, 3, 5, 7]


def _mae(a, b, mask):
    return float(np.abs(a[mask].astype(np.int16) - b[mask].astype(np.int16)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--mot", required=True); ap.add_argument("--cls", default=None)
    ap.add_argument("--fps", type=float, required=True); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--buoc-lop", type=int, default=5, help="kiem ten lop tren moi N frame")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import cv2
    anh = rmv.liet_ke(Path(a.images))
    if a.max_frames:
        anh = anh[:a.max_frames]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened() or not anh:
        print("[LOI] khong mo duoc video hoac khong co anh"); return 2
    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    wh_video = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    n_khai_bao = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    boxes = rmv.doc_mot(a.mot)
    lop = rmv.doc_lop(a.cls)
    loi_khung, loi_lop = [], []
    n_box = n_frame_co_box = n_box_kiem_lop = 0
    dem_lop = {}
    ti_le_max = loi_kv_max = 0.0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i > len(anh):
            loi_khung.append("video co frame thua %d" % i)
            continue
        goc = cv2.imread(str(anh[i - 1]))
        if goc is None or goc.shape != fr.shape:
            loi_khung.append("frame %d: kich thuoc/anh goc khong khop" % i)
            continue
        bf = boxes.get(i, [])
        ky_vong = goc.copy()
        rmv.ve_frame(cv2, ky_vong, bf, lop, i)
        nen = goc.copy()
        rmv.ve_frame(cv2, nen, [], {}, i)                        # chi so frame
        n_box += len(bf)
        n_frame_co_box += 1 if bf else 0
        # (1) khung + so frame
        mask = np.any(ky_vong != nen, axis=2)
        if mask.any():
            loi_kv, loi_ref = _mae(fr, ky_vong, mask), _mae(fr, nen, mask)
        else:
            mask = np.any(nen != goc, axis=2)
            loi_kv, loi_ref = _mae(fr, nen, mask), _mae(fr, goc, mask)
        ti_le = loi_kv / max(loi_ref, 1e-6)
        ti_le_max, loi_kv_max = max(ti_le_max, ti_le), max(loi_kv_max, loi_kv)
        if ti_le > TI_LE or loi_kv > TRAN:
            loi_khung.append("frame %d: loi diem ve %.1f vs khong ve %.1f (ti le %.2f)" % (i, loi_kv, loi_ref, ti_le))
        # (2) ten lop: chon ung vien gan nhat trong vung chu cua tung box
        if bf and a.buoc_lop and (i - 1) % a.buoc_lop == 0:
            for tid, *_r in bf:
                mong = lop.get((i, tid))
                ten_mong = rmv.COCO.get(mong, str(mong)) if mong is not None else ""
                dem_lop[ten_mong] = dem_lop.get(ten_mong, 0) + 1
                ve = {}
                for c in UNG_VIEN_LOP + ([mong] if mong not in UNG_VIEN_LOP else []):
                    lop2 = dict(lop)
                    if c is None:
                        lop2.pop((i, tid), None)
                    else:
                        lop2[(i, tid)] = c
                    im = goc.copy()
                    rmv.ve_frame(cv2, im, bf, lop2, i)
                    ve[c] = im
                vung = np.zeros(goc.shape[:2], bool)
                for c, im in ve.items():
                    vung |= np.any(im != ky_vong, axis=2)
                if not vung.any():
                    continue
                n_box_kiem_lop += 1
                sai = {c: _mae(fr, im, vung) for c, im in ve.items()}
                chon = min(sai, key=sai.get)
                if chon != mong:
                    loi_lop.append("frame %d id %d: ky vong lop %r, gan nhat %r (sai so %.1f vs %.1f)"
                                   % (i, tid, mong, chon, sai[mong], sai[chon]))
    cap.release()
    w0, h0 = int(cv2.imread(str(anh[0])).shape[1]), int(cv2.imread(str(anh[0])).shape[0])
    phep = {
        "so frame giai ma == so anh": (i == len(anh), "%d vs %d" % (i, len(anh))),
        "CAP_PROP_FRAME_COUNT == so anh": (n_khai_bao == len(anh), "%d vs %d" % (n_khai_bao, len(anh))),
        "fps video == --fps": (abs(fps_video - a.fps) < 1e-3, "%g vs %g" % (fps_video, a.fps)),
        "kich thuoc == anh goc": (wh_video == [w0, h0], str(wh_video)),
        "moi frame: khung bbox/mau ID/so frame dung cho": (not loi_khung, "%d frame loi" % len(loi_khung)),
        "ten lop dung (mau moi %d frame)" % a.buoc_lop: (not loi_lop and n_box_kiem_lop > 0 or not boxes,
                                                         "%d box kiem, %d sai" % (n_box_kiem_lop, len(loi_lop))),
    }
    for ten, (ok, ct) in phep.items():
        print("  [%s] %s (%s)" % ("DAT" if ok else "KHONG DAT", ten, ct))
    for e in (loi_khung + loi_lop)[:10]:
        print("     " + e)
    kq = {"video": a.video, "n_frame_giai_ma": i, "n_anh": len(anh), "fps_video": fps_video, "kich_thuoc_wh": wh_video,
          "n_box": n_box, "n_frame_co_box": n_frame_co_box, "n_box_kiem_lop": n_box_kiem_lop, "dem_nhan_lop_mau": dem_lop,
          "ti_le_loi_max": ti_le_max, "loi_diem_ve_max": loi_kv_max, "nguong": {"ti_le": TI_LE, "tran": TRAN},
          "loi_khung": loi_khung[:50], "loi_lop": loi_lop[:50], "phep": {k: v[0] for k, v in phep.items()},
          "dat": all(v[0] for v in phep.values()), "opencv": cv2.__version__}
    print("KET QUA: %s | %d frame, %d box, ti le loi max %.3f, %d box kiem lop, nhan lop mau %s" %
          ("DAT" if kq["dat"] else "KHONG DAT", i, n_box, ti_le_max, n_box_kiem_lop, json.dumps(dem_lop, ensure_ascii=False)))
    if a.json:
        Path(a.json).write_text(json.dumps(kq, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if kq["dat"] else 1


if __name__ == "__main__":
    sys.exit(main())
