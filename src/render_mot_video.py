"""
render_mot_video.py -- Ve ket qua tracking (MOTChallenge txt) len anh -> mp4 minh hoa. Chi can numpy + opencv.

    python src/render_mot_video.py --images data/smoke/MVI_39031_f0001-0300 --mot data/processed/trackers/NANO-e2e/nano-trt-cv/data/MVI_39031.txt \
        [--cls data/processed/trackers/NANO-e2e/nano-trt-cv/data_cls/MVI_39031.csv] --fps 25 --out results_nano/nano-trt-cv.mp4 [--max-frames N]

MOT txt: frame,id,x,y,w,h,conf(,...) -- frame 1-based = thu tu anh (theo SO trong ten). Mau theo ID. Khong doi du lieu.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

COCO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_SO = re.compile(r"^(.*?)(\d+)$")


def liet_ke(d: Path):
    ds = [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return [p for p in sorted(ds, key=lambda p: int(_SO.match(p.stem).group(2)))]


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
    boxes = defaultdict(list)
    with open(a.mot, encoding="utf-8") as f:
        for ln in f:
            v = ln.strip().split(",")
            if len(v) < 6:
                continue
            boxes[int(float(v[0]))].append((int(float(v[1])), float(v[2]), float(v[3]), float(v[4]), float(v[5]), float(v[6]) if len(v) > 6 else 1.0))
    lop = {}
    if a.cls and Path(a.cls).is_file():
        with open(a.cls, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    lop[(int(float(r["frame"])), int(float(r["id"])))] = int(float(r.get("cls", r.get("class", -1))))
                except Exception:  # noqa: BLE001
                    pass
    h, w = cv2.imread(str(anh[0])).shape[:2]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    if not wr.isOpened():
        print("[LOI] khong mo duoc VideoWriter"); return 2
    n_box = 0
    for i, p in enumerate(anh, start=1):
        img = cv2.imread(str(p))
        for tid, x, y, bw, bh, conf in boxes.get(i, []):
            c = ((tid * 37) % 256, (tid * 91) % 256, (tid * 53) % 256)
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), c, 2)
            ten = lop.get((i, tid)); ten = COCO.get(ten, str(ten)) if ten is not None else ""
            cv2.putText(img, f"{tid}{(' ' + ten) if ten else ''}", (int(x), max(12, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            n_box += 1
        cv2.putText(img, f"f{i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        wr.write(img)
    wr.release()
    print(f"[render] {len(anh)} frame, {n_box} box -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
