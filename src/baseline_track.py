"""
baseline_track.py -- Baseline pipeline: YOLOv8 (pretrained) + ByteTrack.

Chay detector + tracker tren tung video, xuat ket qua sang format MOTChallenge
de TrackEval doc va tinh MOTA/IDF1/HOTA (xem src/run_trackeval.py).

KIEN TRUC PIPELINE
-------------------
    anh (img00001.jpg, ...)
        -> che vung ignored_region (mask_ignored_regions)
        -> YOLOv8 detect (loc rieng cac class xe trong COCO)
        -> ByteTrack gan track id (motion model: Kalman Filter voi mo hinh CV
           - day chinh la "motion model chuan" se duoc thay bang EKF/UKF+CTRV
           o Giai doan 3-4)
        -> ghi ra data/processed/trackers/<split>/<ten_tracker>/data/<video>.txt

VI SAO DUNG model.track(frame, persist=True) THEO TUNG ANH (khong ghep video)?
-------------------------------------------------------------------------------
UA-DETRAC luu san moi frame la 1 file .jpg rieng (khong phai file video). De
tranh mat chat luong do nen video lai (vi du .mp4), script doc TUNG ANH GOC
bang cv2.imread() va goi model.track(frame, persist=True, ...) cho tung frame
theo dung thu tu. Da doi chieu voi source code cua ultralytics
(ultralytics/trackers/track.py, ham on_predict_postprocess_end): voi
persist=True, tracker KHONG bi reset giua cac lan goi, dung la co che chinh
thuc de "streaming" tung frame rieach le ma van giu trang thai theo doi lien tuc.

Giua 2 VIDEO KHAC NHAU, script tao MODEL MOI (YOLO(...) lai tu dau) thay vi chi
goi tracker.reset(), de dam bao khong con sot trang thai nao (id counter, track
list) tu video truoc - uu tien do an toan / de debug hon la toi uu toc do (chi
phi tai lai model yolov8n la ~0.3-2s, khong dang ke so voi thoi gian chay tracking).

XU LY ignored_region (VUNG BO QUA CUA UA-DETRAC)
--------------------------------------------------
Thay vi loc detection SAU khi da chay xong (co the sot truong hop ByteTrack da
lo gan track id cho mot detection sai trong vung nay), script TO XAM cac vung
ignored_region truc tiep tren anh dau vao TRUOC KHI dua vao YOLO. Mau xam
(114,114,114) trung voi mau padding mac dinh cua YOLO nen khong tao canh gia
lam detector chu y nham. Cach nay dam bao detector KHONG BAO GIO thay duoc vat
the trong vung bi che, giong het cach dataset goc dinh nghia "vung khong
annotate".

FORMAT FILE KET QUA (tracker output cho TrackEval)
----------------------------------------------------
    frame, id, bb_left, bb_top, bb_width, bb_height, conf

Chi 7 cot (khong co conf/class/visibility nhu gt.txt). Da doi chieu voi source
code trackeval/datasets/mot_challenge_2d_box.py: neu file co < 8 cot, TrackEval
tu dong gan class = 1 (~ 'pedestrian' trong bang class chuan, trung voi class
ma ta dung cho GT xe o Giai doan 1) cho MOI dong - day la quy uoc pho bien nhat
cho tracker output va tranh nham lan voi cot "x/y/z" (toa do 3D, thuong la -1)
neu dung du 10 cot.

CACH DUNG
---------
    python src/baseline_track.py                        # 5 video mau (config.SAMPLE_VIDEOS)
    python src/baseline_track.py --videos MVI_20011      # chi 1 video
    python src/baseline_track.py --model yolov8s.pt --conf 0.3
    python src/baseline_track.py --tracker-name YOLOv8n-ByteTrack-CV
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Mask vung bo qua
# ---------------------------------------------------------------------------
def load_ignored_regions(video: str, split_name: str) -> list[tuple[float, float, float, float]]:
    """Doc gt/ignored_regions.txt (da tao o Giai doan 1) cho 1 video.

    Tra ve list (left, top, width, height). Rong neu video khong co vung nao.
    """
    p = config.PROCESSED_DIR / split_name / video / "gt" / "ignored_regions.txt"
    if not p.exists() or p.stat().st_size == 0:
        return []
    arr = np.loadtxt(p, delimiter=",", ndmin=2)
    return [tuple(row) for row in arr]


def mask_ignored_regions(frame: np.ndarray, regions: list[tuple[float, float, float, float]]) -> np.ndarray:
    """To mau xam trung tinh len cac vung ignored_region truoc khi dua vao detector."""
    if not regions:
        return frame
    h, w = frame.shape[:2]
    for left, top, rw, rh in regions:
        x1, y1 = max(0, int(round(left))), max(0, int(round(top)))
        x2, y2 = min(w, int(round(left + rw))), min(h, int(round(top + rh)))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = config.IGNORE_MASK_COLOR_BGR
    return frame


# ---------------------------------------------------------------------------
# Chay 1 video
# ---------------------------------------------------------------------------
def track_one_video(video: str, img_root: Path, regions: list, model_name: str,
                     tracker_cfg: str, conf: float, device) -> pd.DataFrame:
    """Chay YOLOv8 + ByteTrack tren tat ca frame cua 1 video, tra ve DataFrame ket qua."""
    from ultralytics import YOLO

    frame_paths = sorted((img_root / video).glob("img*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"Khong tim thay anh nao trong {img_root / video}")

    # Model moi cho moi video -> tracker (ByteTrack) bat dau tu trang thai sach,
    # khong the nham track id / lich su giua cac video khac nhau.
    model = YOLO(model_name)

    rows = []
    for frame_num, fp in enumerate(tqdm(frame_paths, desc=f"    {video}", unit="frame",
                                        leave=False, ncols=90), start=1):
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"    [WARN] Khong doc duoc anh: {fp}")
            continue
        frame = mask_ignored_regions(frame, regions)

        results = model.track(frame, persist=True, tracker=tracker_cfg,
                              classes=config.YOLO_VEHICLE_CLASSES, conf=conf,
                              device=device, verbose=False)
        r = results[0]
        if r.boxes.id is None:
            continue  # frame nay khong track nao duoc xac nhan (chua qua nguong new_track_thresh)

        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
            rows.append((frame_num, int(tid), float(x1), float(y1),
                        float(x2 - x1), float(y2 - y1), float(cf)))

    return pd.DataFrame(rows, columns=["frame", "id", "bb_left", "bb_top",
                                       "bb_width", "bb_height", "conf"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline YOLOv8 + ByteTrack -> MOTChallenge format")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Danh sach video (mac dinh: config.SAMPLE_VIDEOS)")
    ap.add_argument("--split-name", default="DETRAC-sample",
                    help="Ten split GT tuong ung (phai da chay to_motchallenge.py cho split nay)")
    ap.add_argument("--model", default=config.YOLO_DEFAULT_MODEL, help="Trong so YOLOv8 (.pt)")
    ap.add_argument("--motion-model", choices=list(config.MOTION_MODELS), default="cv",
                    help="Motion model: 'cv' = baseline KF/Constant Velocity, "
                         "'ekf_ctrv' = EKF voi mo hinh CTRV")
    ap.add_argument("--tracker-cfg", default=None,
                    help="Ghi de truc tiep file cau hinh tracker (.yaml), bo qua --motion-model")
    ap.add_argument("--conf", type=float, default=config.YOLO_DEFAULT_CONF, help="Nguong tin cay YOLO")
    ap.add_argument("--tracker-name", default=None,
                    help="Ten thu muc tracker trong data/processed/trackers/ (mac dinh tu suy tu model+tracker)")
    ap.add_argument("--device", default="0", help="'0' = GPU dau tien, 'cpu' = chay tren CPU")
    args = ap.parse_args()

    videos = args.videos or config.SAMPLE_VIDEOS
    img_root = config.find_images_root()
    if img_root is None:
        print("[ERROR] Khong tim thay thu muc anh. Chay src/extract_verify.py truoc.")
        return 1

    # --- Chon motion model ---
    spec = config.MOTION_MODELS[args.motion_model]
    tracker_cfg = args.tracker_cfg or spec["cfg"]
    if args.motion_model != "cv":
        # Cac tracker CTRV la lop tu viet, phai dang ky vao TRACKER_MAP cua
        # ultralytics TRUOC khi goi model.track(), neu khong se bao loi
        # "Only [...] are supported for now".
        import tracker_ctrv
        tracker_ctrv.register()

    tracker_name = args.tracker_name or f"{Path(args.model).stem}-{spec['suffix']}"
    out_dir = config.PROCESSED_DIR / "trackers" / args.split_name / tracker_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[baseline] model={args.model}  motion_model={args.motion_model} ({spec['desc']})")
    print(f"[baseline] tracker_cfg={tracker_cfg}  conf={args.conf}  device={args.device}")
    print(f"[baseline] {len(videos)} video -> {out_dir}")

    device = args.device
    if device not in ("cpu",):
        import torch
        if not torch.cuda.is_available():
            print("[WARN] Khong tim thay GPU, chuyen sang CPU.")
            device = "cpu"

    summary = []
    for video in videos:
        n_img = len(list((img_root / video).glob("img*.jpg")))
        if n_img == 0:
            print(f"  [MISS] {video}: khong co anh trong {img_root / video}, bo qua.")
            continue

        regions = load_ignored_regions(video, args.split_name)
        t0 = time.time()
        df = track_one_video(video, img_root, regions, args.model, tracker_cfg, args.conf, device)
        dt = time.time() - t0

        out_file = out_dir / f"{video}.txt"
        df.to_csv(out_file, header=False, index=False)

        n_tracks = df["id"].nunique() if len(df) else 0
        fps = n_img / dt if dt > 0 else 0
        print(f"  {video:<12} {n_img:>5} anh  {len(regions)} vung bo qua  "
              f"-> {len(df):>6,} bbox, {n_tracks:>4} track  ({dt:5.1f}s, {fps:5.1f} fps)")
        summary.append({"video": video, "n_images": n_img, "n_ignored_regions": len(regions),
                        "n_det_boxes": len(df), "n_tracks": n_tracks,
                        "seconds": round(dt, 1), "fps": round(fps, 1)})

    out_csv = config.INTERIM_DIR / f"baseline_track_summary_{tracker_name}.csv"
    pd.DataFrame(summary).to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(f"TONG KET: {tracker_name}")
    print("=" * 78)
    s = pd.DataFrame(summary)
    if len(s):
        print(f"  Tong anh xu ly    : {s['n_images'].sum():,}")
        print(f"  Tong bbox xuat ra : {s['n_det_boxes'].sum():,}")
        print(f"  Tong thoi gian    : {s['seconds'].sum():.1f}s  "
              f"(trung binh {s['n_images'].sum() / s['seconds'].sum():.1f} fps)")
    print(f"\n  Ket qua tracker  -> {out_dir}")
    print(f"  Bang tom tat     -> {out_csv}")
    print("\n  Buoc tiep theo: python src/run_trackeval.py "
          f"--split-name {args.split_name} --tracker {tracker_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
