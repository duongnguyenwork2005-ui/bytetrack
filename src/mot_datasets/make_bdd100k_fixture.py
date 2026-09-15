"""
make_bdd100k_fixture.py -- Sinh fixture BDD100K-MOT (box_track_20, Scalabel) NHO de
kiem tra pipeline khi chua co du lieu that. KHONG co anh.

Schema theo tai lieu chinh thuc BDD100K (doc.bdd100k.com/format.html, muc MOT):
    [ { "name": "<video>-<frameIndex+1:07d>.jpg", "videoName": "<video>", "frameIndex": int,
        ("timestamp": int ms, tuy chon),
        "labels": [ { "id": "<str>", "category": "<str>",
                      "attributes": {"occluded": bool, "truncated": bool, "crowd": bool},
                      "box2d": {"x1": f, "y1": f, "x2": f, "y2": f} }, ... ] }, ... ]

Ba video:
    b1  12 frame: car A di thang, truck B bi che (occluded) frame 5-7, motorcycle C +
        rider D (nguoi ngoi tren C - box RIENG, KHONG gop), pedestrian E, mot box CROWD
        (car, crowd=true) rong bao 3 xe nho F1..F3 (co ca nhan rieng cho F1, con F2/F3
        khong co nhan rieng - chinh la tinh huong crowd cua BDD).
    b2  12 frame: car G co GAP frame 5-7 KHONG bang chung (frame 4, 8 khong occluded -> gap_evidence
        'unknown'); car M co GAP frame 7-8 voi frame 6 va 9 occluded (-> 'occluded_adjacent');
        bicycle H truncated; 'other vehicle' I (distractor); truck occluded moi 4 frame.
    b3   8 frame CO timestamp (ms), mot khoang bi bo frame (timestamp nhay 400 ms);
        bus J, motorcycle K re trai (theta doi dan).

    python src/mot_datasets/make_bdd100k_fixture.py [--out data/fixtures/bdd100k_mini]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

W, H = 1280, 720
FPS = 5


def box(x, y, w, h):
    return {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h)}


def label(id_, cat, b, occluded=False, truncated=False, crowd=False):
    return {"id": str(id_), "category": cat,
            "attributes": {"occluded": bool(occluded), "truncated": bool(truncated), "crowd": bool(crowd)},
            "box2d": b}


def frame(video, i, labels, timestamp=None):
    f = {"name": f"{video}-{i + 1:07d}.jpg", "videoName": video, "frameIndex": i, "labels": labels}
    if timestamp is not None:
        f["timestamp"] = int(timestamp)
    return f


def video_b1():
    v = "b1c66a42-6f7d68ca"
    frames = []
    for i in range(12):
        L = [label(1, "car", box(100 + 30 * i, 300, 120, 80)),                      # A
             label(3, "motorcycle", box(700 + 12 * i, 420, 50, 70)),                # C
             label(4, "rider", box(710 + 12 * i, 380, 30, 60)),                     # D (nguoi tren C)
             label(5, "pedestrian", box(1150 - 5 * i, 350, 35, 90)),                # E
             label(7, "car", box(200, 120, 400, 150), crowd=True),                  # crowd bao F1..F3
             label(8, "car", box(220, 140, 80, 60))]                                # F1 (co nhan rieng)
        occ = 5 <= i <= 7
        L.insert(1, label(2, "truck", box(400 + 20 * i, 250, 160, 110), occluded=occ))   # B
        frames.append(frame(v, i, L))
    return v, frames


def video_b2():
    v = "b2d0a5e2-3c1f9a7b"
    frames = []
    for i in range(12):
        L = []
        if not (4 <= i <= 6):                                                       # GAP frame 5-7 (1-based)
            L.append(label(11, "car", box(300 + 25 * i, 330, 130, 85)))            # G
        L.append(label(12, "bicycle", box(1220 - 10 * i, 400, 40, 60), truncated=(i < 3)))   # H
        L.append(label(13, "other vehicle", box(50, 500, 90, 60)))                  # I distractor
        L.append(label(14, "truck", box(800, 200 + 6 * i, 170, 120), occluded=(i % 4 == 0)))
        # car M (id 15): GAP frame 7-8 (1-based) voi bang chung che khuat: frame 6 va 9 deu occluded
        if not (6 <= i <= 7):
            L.append(label(15, "car", box(500 + 20 * i, 450, 110, 70), occluded=(i in (5, 8))))
        frames.append(frame(v, i, L))
    return v, frames


def video_b3():
    v = "b3e7f1c0-9a2b4d6e"
    frames = []
    ts = [0, 200, 400, 600, 1000, 1200, 1400, 1600]     # bo 1 frame giua 600 -> 1000 ms
    import math
    for i, t in enumerate(ts):
        ang = 0.25 * i
        cx, cy = 640 + 150 * math.sin(ang), 450 - 150 * math.cos(ang) + 150
        L = [label(21, "bus", box(150 + 15 * i, 220, 220, 150)),
             label(22, "motorcycle", box(cx - 25, cy - 35, 50, 70), occluded=(i == 3))]
        frames.append(frame(v, i, L, timestamp=1_600_000_000_000 + t))
    return v, frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fixtures/bdd100k_mini")
    ap.add_argument("--part", default="val")
    a = ap.parse_args()
    root = Path(a.out) / "labels" / "box_track_20" / a.part
    root.mkdir(parents=True, exist_ok=True)
    for v, frames in (video_b1(), video_b2(), video_b3()):
        (root / f"{v}.json").write_text(json.dumps(frames, indent=1), encoding="utf-8")
        print(f"  {root / (v + '.json')}  {len(frames)} frame")
    (Path(a.out) / "README.md").write_text(
        "Fixture BDD100K-MOT tong hop (khong anh) sinh boi src/mot_datasets/make_bdd100k_fixture.py.\n"
        "Chi de kiem tra pipeline; KHONG dung de bao cao ket qua.\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
