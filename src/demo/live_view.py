"""
live_view.py -- Xem TRUC TIEP (cua so OpenCV) YOLOv8n + ByteTrack tren mot video bat ky,
                ba motion model chay song song tren CUNG detection: KF+CV | EKF+CTRV | UKF+CTRV.

Cach hoat dong: moi frame chay YOLO dung MOT lan (cung weights, conf, classes, cach to xam
ignored_region nhu baseline_track.py), roi dua cung tap detection vao ba tracker ByteTrack
doc lap (moi tracker mot motion model). Day chinh la cach ultralytics goi tracker ben trong
model.track() (tracker.update(boxes, frame)), nen ket qua giong chay tung model rieng.

Nguon video:
    --video MVI_40991            sequence UA-DETRAC (train hoac test), doc anh trong data/extracted
    --file  duong/dan/video.mp4  file video bat ky (khong to xam ignored_region)

Phim: SPACE tam dung/tiep tuc | n (khi tam dung) tien 1 frame | s chup man hinh | +/- nhanh/cham | q / ESC thoat

Vi du:
    python src/demo/live_view.py --video MVI_40991 --start 1650 --end 1850
    python src/demo/live_view.py --video MVI_40991 --track-buffer 90
    python src/demo/live_view.py --file C:/videos/traffic.mp4 --models cv ekf_ctrv
    python src/demo/live_view.py --video MVI_39031 --save outputs/live/MVI_39031_live.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import tracker_ctrv  # noqa: E402
from baseline_track import load_ignored_regions, mask_ignored_regions  # noqa: E402

LABELS = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
BUFFER_CFG = {  # track_buffer -> yaml cho tung motion model
    30: {"cv": None, "ekf_ctrv": None, "ukf_ctrv": None},                     # None = yaml mac dinh trong config.MOTION_MODELS
    90: {"cv": "configs/buffer_sweep/cv_buf90.yaml",
         "ekf_ctrv": "configs/buffer_sweep/ekf_ctrv_buf90.yaml",
         "ukf_ctrv": "configs/buffer_sweep/ukf_ctrv_buf90.yaml"},
}


def color_for_id(tid: int) -> tuple[int, int, int]:
    hue = (int(hashlib.md5(str(tid).encode()).hexdigest(), 16) % 160) + 10
    b, g, r = cv2.cvtColor(np.uint8([[[hue % 180, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def dashed_rect(im, x1, y1, x2, y2, col, thick=1, dash=6):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    for xa in range(x1, x2, dash * 2):
        cv2.line(im, (xa, y1), (min(xa + dash, x2), y1), col, thick)
        cv2.line(im, (xa, y2), (min(xa + dash, x2), y2), col, thick)
    for ya in range(y1, y2, dash * 2):
        cv2.line(im, (x1, ya), (x1, min(ya + dash, y2)), col, thick)
        cv2.line(im, (x2, ya), (x2, min(ya + dash, y2)), col, thick)


def make_tracker(motion_model: str, yaml_path: str | None):
    """Tao tracker ByteTrack dung nhu ultralytics lam trong model.track() (on_predict_start)."""
    from ultralytics.trackers import track as track_mod
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml

    tracker_ctrv.register()
    yaml_path = yaml_path or config.MOTION_MODELS[motion_model]["cfg"]
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(yaml_path)))
    return track_mod.TRACKER_MAP[cfg.tracker_type](args=cfg), cfg


def frame_source(args):
    """Tra ve (iterator (frame_idx, anh BGR), fps nguon, ten, regions)."""
    if args.video:
        root = config.find_images_root()
        if root is None or not (root / args.video).is_dir():
            raise SystemExit(f"Khong tim thay thu muc anh cua {args.video} duoi {root}")
        imgs = sorted((root / args.video).glob("img*.jpg"))
        split = "DETRAC-all" if args.video in config.list_videos("train") else "DETRAC-test"
        regions = load_ignored_regions(args.video, split)
        f0 = max(1, args.start)
        f1 = min(len(imgs), args.end) if args.end else len(imgs)

        def it():
            for i in range(f0, f1 + 1):
                yield i, cv2.imread(str(imgs[i - 1]))
        return it(), 25.0, f"{args.video} ({split}, frame {f0}-{f1})", regions
    cap = cv2.VideoCapture(args.file)
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc {args.file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if args.start > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)

    def it():
        i = max(1, args.start)
        while True:
            ok, fr = cap.read()
            if not ok or (args.end and i > args.end):
                break
            yield i, fr
            i += 1
        cap.release()
    return it(), fps, Path(args.file).name, []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Ten sequence UA-DETRAC, vd MVI_40991")
    src.add_argument("--file", help="Duong dan file video bat ky")
    ap.add_argument("--models", nargs="+", default=["cv", "ekf_ctrv", "ukf_ctrv"], choices=list(LABELS))
    ap.add_argument("--track-buffer", type=int, default=30, choices=sorted(BUFFER_CFG),
                    help="30 = ByteTrack mac dinh (baseline da khoa); 90 = cau hinh Stage E mo rong")
    ap.add_argument("--conf", type=float, default=config.YOLO_DEFAULT_CONF)
    ap.add_argument("--model", default=str(config.MODELS_DIR / "yolov8n.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0, help="0 = den het")
    ap.add_argument("--scale", type=float, default=0.0, help="Ti le phong moi panel (0 = tu chon de vua 1920 px ngang)")
    ap.add_argument("--speed", type=float, default=1.0, help="Toc do phat so voi thoi gian thuc (1.0 = thoi gian thuc)")
    ap.add_argument("--trail", type=int, default=30, help="So frame ve duoi quy dao moi ID")
    ap.add_argument("--no-lost", action="store_true", help="Khong ve box du doan cua track dang LOST (net dut)")
    ap.add_argument("--no-mask", action="store_true", help="Khong to xam ignored_region (mac dinh co, giong baseline)")
    ap.add_argument("--save", default=None, help="Ghi them ra file .mp4 (mp4v)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    trackers, cfgs = {}, {}
    for m in args.models:
        trackers[m], cfgs[m] = make_tracker(m, BUFFER_CFG[args.track_buffer][m])
        print(f"[{LABELS[m]:<10}] {type(trackers[m]).__name__}  bo loc={type(trackers[m].kalman_filter).__name__}  "
              f"track_buffer={cfgs[m].track_buffer} match_thresh={cfgs[m].match_thresh} fuse_score={cfgs[m].fuse_score}")
    frames, src_fps, title, regions = frame_source(args)
    print(f"Nguon: {title} | YOLO {Path(args.model).name} conf={args.conf} classes={config.YOLO_VEHICLE_CLASSES} | "
          f"SPACE tam dung, n tien 1 frame, s chup, +/- toc do, q thoat")

    trails = {m: defaultdict(lambda: deque(maxlen=args.trail)) for m in args.models}
    seen_ids = {m: set() for m in args.models}
    writer = None
    paused, step, speed = False, False, args.speed
    win = "live: " + " | ".join(LABELS[m] for m in args.models)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_prev = time.time()
    fps_disp = 0.0

    for fi, frame in frames:
        if frame is None:
            continue
        if regions and not args.no_mask:
            frame = mask_ignored_regions(frame, regions)
        t0 = time.time()
        r = model.predict(frame, conf=args.conf, classes=config.YOLO_VEHICLE_CLASSES, device=args.device, verbose=False)[0]
        det = r.boxes.cpu().numpy()
        h0, w0 = frame.shape[:2]
        scale = args.scale or min(1.0, 1920 / (w0 * len(args.models)))
        pw, ph = int(w0 * scale), int(h0 * scale)
        panels = []
        for m in args.models:
            tr = trackers[m]
            tracks = tr.update(det, frame)            # (N, 8): x1,y1,x2,y2,id,score,cls,idx
            im = frame.copy()
            ids_now = set()
            for row in tracks:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                ids_now.add(tid)
                col = color_for_id(tid)
                seen_ids[m].add(tid)
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(im, f"{tid}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
                trails[m][tid].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            for tid, q in trails[m].items():
                for i in range(1, len(q)):
                    cv2.line(im, q[i - 1], q[i], color_for_id(tid), 2, cv2.LINE_AA)
            # track dang LOST: box du doan (net dut) - day la vi tri motion model dang doan trong luc mat dau
            n_lost = 0
            if not args.no_lost:
                for st in tr.lost_stracks:
                    if st.mean is None:
                        continue
                    n_lost += 1
                    x, y, w, h = st.tlwh
                    col = color_for_id(int(st.track_id))
                    dashed_rect(im, x, y, x + w, y + h, col, 1)
                    cv2.putText(im, f"{int(st.track_id)}?", (int(x), max(12, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                    trails[m][int(st.track_id)].append((int(x + w / 2), int(y + h / 2)))
            panel = cv2.resize(im, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (pw, 24), (0, 0, 0), -1)
            cv2.putText(panel, f"{LABELS[m]}  buffer={cfgs[m].track_buffer}  track: {len(tracks)}  lost: {n_lost}  ID da tao: {len(seen_ids[m])}",
                        (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        canvas = np.hstack(panels)
        dt = time.time() - t_prev
        t_prev = time.time()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(dt, 1e-6)) if fps_disp else 1.0 / max(dt, 1e-6)
        bar = np.zeros((26, canvas.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{title}   frame {fi}   t={fi / src_fps:6.2f}s   det: {len(det)}   {fps_disp:4.1f} fps   x{speed:.2g}"
                         + ("   [TAM DUNG]" if paused else ""),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        canvas = np.vstack([bar, canvas])
        if args.save:
            if writer is None:
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        cv2.imshow(win, canvas)

        # dieu khien toc do: cho du de dat thoi gian thuc * speed
        proc = time.time() - t0
        wait_ms = max(1, int((1.0 / src_fps / speed - proc) * 1000))
        while True:
            k = cv2.waitKey(50 if paused else wait_ms) & 0xFF
            if k in (ord("q"), 27):
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                print("Thoat.")
                return 0
            if k == ord(" "):
                paused = not paused
            elif k == ord("n") and paused:
                step = True
            elif k == ord("s"):
                out = Path("outputs/live") / f"{Path(title.split()[0]).stem}_frame{fi:05d}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), canvas)
                print(f"Da luu {out}")
            elif k in (ord("+"), ord("=")):
                speed = min(8.0, speed * 1.5)
            elif k == ord("-"):
                speed = max(0.1, speed / 1.5)
            if not paused or step:
                step = False
                break
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Het video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
