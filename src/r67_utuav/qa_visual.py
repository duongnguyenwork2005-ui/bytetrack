"""Create fixed first/middle/last GT contact sheet for B/C converter review."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare as prep


def main():
    sheet = Image.new("RGB", (960*3, 570*2), "#242424")
    for row, (name, (_, first, last, _)) in enumerate(prep.RANGES.items()):
        n = last-first+1
        gt = np.loadtxt(prep.config.PROCESSED_DIR/prep.SPLIT/name/"gt/gt.txt", delimiter=",", ndmin=2)
        for col, idx in enumerate((1, (n+1)//2, n)):
            path = prep.IMAGES/name/f"{idx:04d}.jpg"
            with Image.open(path) as original:
                picture = original.convert("RGB").resize((960, 540))
            draw = ImageDraw.Draw(picture)
            boxes = gt[gt[:, 0] == idx]
            for x in boxes:
                px, py, w, h = x[2:6]/4
                draw.rectangle((px, py, px+w, py+h), outline="#00ff66" if x[7] == 1 else "#ff4444", width=2)
            sheet.paste(picture, (col*960, row*570+30))
            label = ImageDraw.Draw(sheet)
            label.text((col*960+5, row*570+5),
                       f"{name}: frame {idx} (source {first+idx-1}), dynamic=green, static=red",
                       fill="white")
    target = prep.LOG/"gt_visual_first_mid_last.jpg"
    sheet.save(target, quality=90)
    print(target)


if __name__ == "__main__":
    main()
