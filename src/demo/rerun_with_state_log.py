"""
rerun_with_state_log.py -- Chay lai tracker tren MOT video, ghi them VI TRI DU DOAN
                           cua cac track dang bi mat (LOST), roi doi chieu voi
                           prediction da luu.

VI SAO CAN. File prediction da luu (data/processed/trackers/.../<video>.txt) chi
chua box o trang thai Tracked. Khi track bi che va chuyen sang Lost, ByteTrack van
predict vi tri moi frame nhung KHONG xuat ra file. Muon ve "trajectory du doan"
trong luc che khuat thi phai lay tu trang thai Kalman/EKF/UKF that trong runtime.

CACH LAM. Chay lai model.track() tu frame 1 den frame cuoi cua clip, DUNG cung
weights, cung config tracker, cung conf. Sau moi frame, doc thang
tracker.tracked_stracks va tracker.lost_stracks, ghi (frame, id, cx, cy, w, h,
state). Vi tri lay tu STrack.tlwh -> chinh la mean da predict.

DOI CHIEU BAT BUOC. ByteTrack tren GPU khong bit-reproducible (da ghi nhan o
Giai doan 1). Nen sau khi chay lai, so box xuat ra (state Tracked) voi prediction
da luu trong cua so clip: cung frame, cung ID, IoU >= 0.9. Bao cao ti le khop.
Neu khop thap thi trajectory du doan KHONG duoc coi la cua prediction da luu.

KHONG ghi de prediction goc. Ket qua ghi vao outputs/thesis_demos/<demo>/rerun/.

CACH DUNG (goi tu render_thesis_demo.py, hoac doc lap):
    python src/demo/rerun_with_state_log.py --video MVI_40854 --split DETRAC-test \
        --tracker-dir testA-ekf_ctrv --motion-model ekf_ctrv --conf 0.25 \
        --end-frame 160 --out outputs/thesis_demos/_probe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import tracker_ctrv  # noqa: E402
from baseline_track import load_ignored_regions, mask_ignored_regions  # noqa: E402
from occlusion_eval import iou_matrix  # noqa: E402

TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]


def rerun(video: str, split: str, motion_model: str, conf: float,
          end_frame: int, device: str = "0") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Chay lai tracker tu frame 1 den end_frame.

    Tra ve (boxes_tracked, states_all, pred_boxes, ten_lop_bo_loc):
      boxes_tracked : box xuat ra (giong file prediction da luu)
      states_all    : trang thai SAU frame (tracked + lost)
      pred_boxes    : box DU DOAN cua moi track trong pool ghep cap, lay NGAY SAU
                      multi_predict va TRUOC khi ghep voi detection. Day chinh la
                      box ma ByteTrack dung de tinh IoU voi detection o frame do.
    """
    import cv2
    from ultralytics import YOLO
    from ultralytics.trackers.basetrack import TrackState

    tracker_ctrv.register()
    spec = config.MOTION_MODELS[motion_model]
    model = YOLO(str(config.MODELS_DIR / "yolov8n.pt"))
    img_root = config.find_images_root()
    imgs = sorted((img_root / video).glob("img*.jpg"))[:end_frame]
    # GIONG HET baseline_track.py: doc bang cv2, to xam ignored_region TRUOC khi
    # dua vao detector, persist=True tu frame dau, frame_num = thu tu file.
    regions = load_ignored_regions(video, split)

    out_rows, state_rows, pred_rows = [], [], []
    filt_name = None
    cur = {"frame": 0}

    def _hook_multi_predict(tr):
        # Boc multi_predict cua CHINH instance tracker: goi ham that roi ghi lai
        # box vua du doan. Khong doi thuat toan, chi quan sat.
        orig = tr.multi_predict

        def logged(tracks):
            orig(tracks)
            for st in tracks:
                if st.mean is None:
                    continue
                x, y, w, h = st.tlwh
                pred_rows.append(dict(frame=cur["frame"], id=int(st.track_id), cx=float(x + w / 2),
                                      cy=float(y + h / 2), w=float(w), h=float(h),
                                      state=("tracked" if st.state == TrackState.Tracked else "lost")))
        tr.multi_predict = logged

    for fr, p in enumerate(imgs, start=1):
        cur["frame"] = fr
        frame = cv2.imread(str(p))
        frame = mask_ignored_regions(frame, regions)
        r = model.track(frame, persist=True, tracker=spec["cfg"], conf=conf,
                        classes=config.YOLO_VEHICLE_CLASSES, device=device, verbose=False)[0]
        tr = model.predictor.trackers[0]
        if filt_name is None:
            filt_name = type(tr.kalman_filter).__name__
            _hook_multi_predict(tr)        # frame 1 chua co track nao de predict
        # Box xuat ra (giong file prediction): chi track Tracked va da activated
        if r.boxes is not None and r.boxes.id is not None:
            xywh = r.boxes.xywh.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), tid, c in zip(xywh, ids, cf):
                out_rows.append((fr, int(tid), cx - w / 2, cy - h / 2, w, h, float(c)))
        # Trang thai THAT cua moi track sau frame nay (ke ca LOST)
        for st_list, name in ((tr.tracked_stracks, "tracked"), (tr.lost_stracks, "lost")):
            for st in st_list:
                if st.mean is None:
                    continue
                x, y, w, h = st.tlwh
                state_rows.append(dict(frame=fr, id=int(st.track_id), cx=float(x + w / 2),
                                       cy=float(y + h / 2), w=float(w), h=float(h),
                                       state=name, is_activated=bool(st.is_activated)))
    model.predictor.trackers = []
    boxes = pd.DataFrame(out_rows, columns=TRK_COLS)
    states = pd.DataFrame(state_rows)
    preds = pd.DataFrame(pred_rows, columns=["frame", "id", "cx", "cy", "w", "h", "state"])
    return boxes, states, preds, filt_name or "?"


def compare_to_saved(rerun_boxes: pd.DataFrame, saved: pd.DataFrame,
                     f0: int, f1: int) -> dict:
    """So box chay lai voi prediction da luu trong cua so [f0, f1]."""
    a = rerun_boxes[(rerun_boxes.frame >= f0) & (rerun_boxes.frame <= f1)]
    b = saved[(saved.frame >= f0) & (saved.frame <= f1)]
    n_a, n_b = len(a), len(b)
    same_id_iou = 0
    any_iou = 0
    per_frame_ids_equal = 0
    frames = sorted(set(a.frame) | set(b.frame))
    for fr in frames:
        ra = a[a.frame == fr]
        rb = b[b.frame == fr]
        if set(ra.id) == set(rb.id):
            per_frame_ids_equal += 1
        if not len(ra) or not len(rb):
            continue
        iou = iou_matrix(ra[["x", "y", "w", "h"]].to_numpy(float),
                         rb[["x", "y", "w", "h"]].to_numpy(float))
        ida, idb = ra.id.to_numpy(), rb.id.to_numpy()
        for i in range(len(ra)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= 0.9:
                any_iou += 1
                if ida[i] == idb[j]:
                    same_id_iou += 1
    return dict(frames=len(frames), n_rerun=n_a, n_saved=n_b,
                frames_same_id_set=per_frame_ids_equal,
                pct_frames_same_id_set=round(per_frame_ids_equal / max(1, len(frames)) * 100, 1),
                boxes_matched_iou90=any_iou,
                boxes_matched_iou90_same_id=same_id_iou,
                pct_same_id=round(same_id_iou / max(1, n_a) * 100, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tracker-dir", required=True, help="vd testA-ekf_ctrv")
    ap.add_argument("--motion-model", required=True, choices=list(config.MOTION_MODELS))
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--end-frame", type=int, required=True)
    ap.add_argument("--cmp-start", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    boxes, states, preds, filt = rerun(a.video, a.split, a.motion_model, a.conf, a.end_frame)
    saved = pd.read_csv(config.PROCESSED_DIR / "trackers" / a.split / a.tracker_dir / "data" / f"{a.video}.txt",
                        header=None, names=TRK_COLS)
    cmp = compare_to_saved(boxes, saved, a.cmp_start, a.end_frame)
    boxes.to_csv(out / f"rerun_boxes_{a.tracker_dir}.csv", index=False)
    states.to_csv(out / f"rerun_states_{a.tracker_dir}.csv", index=False)
    preds.to_csv(out / f"rerun_pred_{a.tracker_dir}.csv", index=False)
    print(f"[{a.tracker_dir}] bo loc runtime = {filt}")
    print(f"  cua so [{a.cmp_start}, {a.end_frame}]: {cmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
