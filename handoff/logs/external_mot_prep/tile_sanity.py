"""One-frame detector diagnostic on fixed 4x4 UTUAV tiles; no tracking/cache writes."""

from __future__ import annotations

import argparse
import collections

import cv2
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("weights")
    args = ap.parse_args()
    image = cv2.imread(args.image)
    if image is None or image.shape[:2] != (2160, 3840):
        raise ValueError("expected a readable 3840x2160 UTUAV frame")
    model = YOLO(args.weights)
    counts = collections.Counter()
    counts_high = collections.Counter()
    for row in range(4):
        for col in range(4):
            tile = image[row * 540:(row + 1) * 540, col * 960:(col + 1) * 960]
            result = model.predict(tile, imgsz=640, conf=0.05, classes=[0, 1, 2, 3],
                                   max_det=3000, device=0, verbose=False)[0]
            for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                counts[model.names[int(cls)]] += 1
                if conf > 0.25:
                    counts_high[model.names[int(cls)]] += 1
    print("raw conf>=0.05:", dict(counts))
    print("tracker conf>0.25:", dict(counts_high))


if __name__ == "__main__":
    main()
