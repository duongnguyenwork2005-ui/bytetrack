"""
render_mot_video.py -- Ve ket qua tracking (MOTChallenge txt) len anh -> mp4 minh hoa (RENDER HAU KY, khong phai hien thi
trong luc tracking). Chi can numpy + opencv; cu phap Python 3.6 (chay duoc bang Python he thong cua Jetson, noi co cv2).

    python src/render_mot_video.py --images data/smoke/MVI_39031_f0001-0300 --mot data/processed/trackers/NANO-e2e/nano-trt-cv/data/MVI_39031.txt \
        [--cls data/processed/trackers/NANO-e2e/nano-trt-cv/data_cls/MVI_39031.csv] --fps 25 --out results_nano/nano-trt-cv.mp4 [--max-frames N]

MOT txt: frame,id,x,y,w,h,conf(,...) -- frame 1-based = thu tu anh (theo SO trong ten). Mau theo ID. Khong doi du lieu.
Ghi kem `<out>.json`: so frame, fps, kich thuoc, so box, sha256 dau vao. Kiem video: `src/kiem_video_render.py` (ve lai
TUNG frame bang CHINH `ve_frame` va so voi frame giai ma).
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

COCO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_SO = re.compile(r"^(.*?)(\d+)$")


def liet_ke(d: Path):
    ds = [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return [p for p in sorted(ds, key=lambda p: int(_SO.match(p.stem).group(2)))]


def doc_mot(path) -> dict:
    """{frame: [(id, x, y, w, h, conf), ...]} tu MOT txt."""
    boxes = defaultdict(list)
    with open(str(path), encoding="utf-8") as f:
        for ln in f:
            v = ln.strip().split(",")
            if len(v) < 6:
                continue
            boxes[int(float(v[0]))].append((int(float(v[1])), float(v[2]), float(v[3]), float(v[4]), float(v[5]),
                                            float(v[6]) if len(v) > 6 else 1.0))
    return boxes


def doc_lop(path) -> dict:
    """{(frame, id): cls} tu data_cls csv (rong neu khong co file)."""
    lop = {}
    if path and Path(path).is_file():
        with open(str(path), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    lop[(int(float(r["frame"])), int(float(r["id"])))] = int(float(r.get("cls", r.get("class", -1))))
                except Exception:  # noqa: BLE001
                    pass
    return lop


def mau_id(tid: int):
    return ((tid * 37) % 256, (tid * 91) % 256, (tid * 53) % 256)


SO_FRAME_GOC = (8, 20)                                   # vi tri chu "f<so frame>" (goc trai tren)


def ve_so_frame(cv2, img, chu: str):
    """Ve chu so frame (vd "f12") DUNG kieu cua video render. Dung CHUNG cho render va bo kiem."""
    cv2.putText(img, chu, SO_FRAME_GOC, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def ve_frame(cv2, img, box_frame, lop: dict, i: int, ve_so: bool = True):
    """Ve bbox + "<id> <lop>" (+ so frame, VE SAU CUNG) len `img` (tai cho). Tra so box da ve. Dung CHUNG cho render
    va kiem; `ve_so=False` chi dung trong bo kiem (ban khong co so frame)."""
    for tid, x, y, bw, bh, _conf in box_frame:
        c = mau_id(tid)
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), c, 2)
        ten = lop.get((i, tid))
        ten = COCO.get(ten, str(ten)) if ten is not None else ""
        cv2.putText(img, "%d%s" % (tid, (" " + ten) if ten else ""), (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    if ve_so:
        ve_so_frame(cv2, img, "f%d" % i)
    return len(box_frame)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(str(p), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--mot", required=True); ap.add_argument("--cls", default=None)
    ap.add_argument("--fps", type=float, default=25.0); ap.add_argument("--out", required=True); ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    import cv2
    anh = liet_ke(Path(a.images))
    if a.max_frames:
        anh = anh[:a.max_frames]
    if not anh:
        print("[LOI] khong co anh"); return 2
    boxes = doc_mot(a.mot)
    lop = doc_lop(a.cls)
    h, w = cv2.imread(str(anh[0])).shape[:2]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    if not wr.isOpened():
        print("[LOI] khong mo duoc VideoWriter"); return 2
    n_box = 0
    for i, p in enumerate(anh, start=1):
        img = cv2.imread(str(p))
        if img is None:
            print("[LOI] khong doc duoc anh %s" % p); wr.release(); return 3
        n_box += ve_frame(cv2, img, boxes.get(i, []), lop, i)
        wr.write(img)
    wr.release()
    info = {"video": str(a.out), "n_frame": len(anh), "fps": a.fps, "kich_thuoc_wh": [w, h], "n_box": n_box,
            "codec": "mp4v", "images": str(Path(a.images).resolve()), "mot": str(Path(a.mot).resolve()),
            "mot_sha256": sha256_file(a.mot), "cls": str(Path(a.cls).resolve()) if a.cls else None,
            "cls_sha256": sha256_file(a.cls) if a.cls and Path(a.cls).is_file() else None,
            "anh_dau": anh[0].name, "anh_cuoi": anh[-1].name, "opencv": cv2.__version__,
            "video_sha256": sha256_file(a.out), "loai": "render HAU KY tu anh + MOT (khong phai hien thi trong luc tracking)"}
    Path(str(a.out) + ".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[render] %d frame, %d box -> %s" % (len(anh), n_box, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
