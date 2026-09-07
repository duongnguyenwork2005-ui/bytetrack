"""
side_by_side.py -- Ghep video song song: KF+CV (trai) vs EKF+CTRV (phai), tren
KET QUA TRACKER THAT (khong phai mo phong tu GT).

Khac voi render_videos.py (Phan 3 cua demo, nap GT roi CHI predict qua 1 doan
che de minh hoa co che), file nay ve THANG box+ID ma ByteTrack thuc su xuat ra
khi chay tren video, frame-by-frame, ca doan khong che lan doan bi che.

CACH DUNG
    python src/demo/side_by_side.py --video MVI_40131 --start 100 --end 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
FPS_OUT = 12
PANEL_GAP = 6


def color_for_id(tid: int) -> tuple[int, int, int]:
    """Mau on dinh theo track id (HSV -> BGR) de cung 1 ID luon cung 1 mau."""
    hue = (tid * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_panel(frame: np.ndarray, frame_num: int, boxes: pd.DataFrame,
               title: str, in_occ: bool) -> np.ndarray:
    im = frame.copy()
    for r in boxes.itertuples():
        col = color_for_id(int(r.id))
        x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, f"ID{int(r.id)}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, f"{title}  |  frame {frame_num}  |  {len(boxes)} track",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if in_occ:
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), (0, 215, 255), 5)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="Ghep video song song CV vs EKF+CTRV")
    ap.add_argument("--video", default="MVI_40131")
    ap.add_argument("--split-name", default="DETRAC-all")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--cv-tracker", default="yolov8n-bytetrack")
    ap.add_argument("--ekf-tracker", default="yolov8n-ekf-ctrv")
    ap.add_argument("--track-id", type=int, default=None,
                    help="Chi danh dau doan che cua DUNG track nay (khong thi lay ca video, "
                         "co the nham voi track khac dang bi che cung luc).")
    a = ap.parse_args()

    img_root = config.find_images_root()
    vdir = img_root / a.video
    n_img = len(list(vdir.glob("img*.jpg")))
    end = a.end or n_img
    print(f"[side_by_side] {a.video}: dung frame {a.start}-{end} (video co {n_img} anh)")

    def load(tracker):
        p = (config.PROCESSED_DIR / "trackers" / a.split_name / tracker / "data" /
             f"{a.video}.txt")
        return pd.read_csv(p, header=None, names=TRK_COLS)

    cv_df, ekf_df = load(a.cv_tracker), load(a.ekf_tracker)

    # Danh dau doan bi che (>=0.10, gom ca mot phan lan gan hoan toan) de ve vien
    # vang, doi chieu voi annotation that. LOC THEO TRACK khi biet track_id: video
    # co the co NHIEU xe bi che cung luc (vd xe do bi vat can che gan het video),
    # khong loc se ve vien vang sai cho toan bo doan chi vi mot xe KHAC dang bi che.
    occ_frames = set()
    for f in ["full_occlusion_segments.csv", "occlusion_segments.csv"]:
        p = config.INTERIM_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.video == a.video]
        if a.track_id is not None:
            d = d[d.track_id == a.track_id]
        for r in d.itertuples():
            occ_frames.update(range(int(r.start_frame), int(r.end_frame) + 1))

    first = cv2.imread(str(vdir / f"img{a.start:05d}.jpg"))
    H, W = first.shape[:2]
    out_w = W * 2 + PANEL_GAP
    out_path = config.RESULTS_DIR / "demo" / f"sbs_{a.video}_{a.start}_{end}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (out_w, H))

    cv_by_f = {k: v for k, v in cv_df.groupby("frame")}
    ekf_by_f = {k: v for k, v in ekf_df.groupby("frame")}

    n_written = 0
    for fr in range(a.start, end + 1):
        fp = vdir / f"img{fr:05d}.jpg"
        im = cv2.imread(str(fp))
        if im is None:
            continue
        in_occ = fr in occ_frames
        left = draw_panel(im, fr, cv_by_f.get(fr, cv_df.iloc[:0]), "KF + CV", in_occ)
        right = draw_panel(im, fr, ekf_by_f.get(fr, ekf_df.iloc[:0]), "EKF + CTRV", in_occ)
        gap = np.full((H, PANEL_GAP, 3), 255, dtype=np.uint8)
        combo = np.hstack([left, gap, right])
        vw.write(combo)
        n_written += 1
    vw.release()
    print(f"[side_by_side] da ghi {n_written} frame -> {out_path}")
    print(f"[side_by_side] {len(occ_frames & set(range(a.start, end + 1)))} frame "
          f"nam trong doan che khuat >=90% (vien vang)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
