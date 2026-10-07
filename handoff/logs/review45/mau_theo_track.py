# -*- coding: utf-8 -*-
"""Vong 45: mau xem tay THEO TRACK (moi track mot crop o frame giua track) cho vehicle_type cua UA-DETRAC test -- CHI DOC.
    python handoff/logs/review45/mau_theo_track.py --loai others --toi-da 0 --csv <csv> --png <png ngoai repo>
--toi-da 0 = moi track; > 0 = chon ngau nhien seed 0. CSV (commit duoc) ghi video / track / frame / box de tai lap;
PNG (crop tu anh dataset) KHONG commit."""
from __future__ import annotations

import argparse
import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loai", required=True)
    ap.add_argument("--toi-da", type=int, default=0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--png")
    a = ap.parse_args()
    videos = sorted(json.loads((REPO / "results/phase2/manifests/detrac_test.json").read_text(encoding="utf-8"))["videos"])
    tracks = defaultdict(list)
    for v in videos:
        root = ET.parse(REPO / f"data/extracted/DETRAC-Test-Annotations-XML/{v}.xml").getroot()
        for fr in root.findall("frame"):
            for t in fr.findall("target_list/target"):
                if t.find("attribute").get("vehicle_type") == a.loai:
                    b = t.find("box")
                    tracks[(v, int(t.get("id")))].append((int(fr.get("num")),
                                                          tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))))
    keys = sorted(tracks)
    if a.toi_da and len(keys) > a.toi_da:
        keys = sorted(random.Random(0).sample(keys, a.toi_da))
    chon = []
    for k in keys:
        ds = sorted(tracks[k])
        num, box = ds[len(ds) // 2]
        chon.append((k[0], k[1], num, len(ds), box))
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["loai", "stt", "video", "track_id", "frame_giua", "so_box_track", "left", "top", "width", "height"])
        for i, (v, tid, num, n, b) in enumerate(chon):
            w.writerow([a.loai, i, v, tid, num, n, *b])
    print(f"{a.loai}: {len(tracks)} track, chon {len(chon)}")
    if a.png:
        from PIL import Image, ImageDraw
        cot = 8
        cs = Image.new("RGB", (cot * 160, ((len(chon) + cot - 1) // cot) * 130), (40, 40, 40))
        dr = ImageDraw.Draw(cs)
        for i, (v, tid, num, n, (left, top, bw, bh)) in enumerate(chon):
            with Image.open(REPO / f"data/extracted/DETRAC-Images/{v}/img{num:05d}.jpg") as im:
                cr = im.crop((max(0, left - 0.15 * bw), max(0, top - 0.15 * bh), left + 1.15 * bw,
                              top + 1.15 * bh)).convert("RGB")
            cr.thumbnail((156, 112))
            x0, y0 = (i % cot) * 160, (i // cot) * 130
            cs.paste(cr, (x0 + 2, y0 + 2))
            dr.text((x0 + 3, y0 + 115), f"{i} {v[4:]} id{tid} n{n}", fill=(255, 255, 0))
        cs.save(a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
