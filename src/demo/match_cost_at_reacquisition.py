"""
match_cost_at_reacquisition.py -- Giai thich VI SAO mot tracker noi lai duoc ID
                                  con tracker khac thi khong, tai cac frame tai ghep.

ByteTrack (ultralytics, fuse_score=True, match_thresh=0.8) ghep track voi
detection khi   IoU(box du doan, detection) x score(detection) > 0.2.
Script nay do dung dai luong do cho target cua demo, o tung frame quanh luc xe
xuat hien lai:

  1. Chay YOLO (cung weights, cung conf, cung cach to xam ignored_region nhu
     baseline_track.py) tren frame do -> lay detection co IoU cao nhat voi GT.
  2. Box du doan cua track goc lay tu rerun_pred_<tracker>.csv (ghi NGAY SAU
     multi_predict, TRUOC khi ghep -> dung box ByteTrack da dung).
  3. Tinh IoU x score cho track goc, va cho MOI track khac trong pool (de biet
     co track nao "tranh" detection do voi diem cao hon khong).

Ket qua ghi vao outputs/thesis_demos/<demo>/match_cost_at_reacquisition.csv va
duoc render_thesis_demo.py nhung vao metadata.json (khoa reacquisition_match_cost).

HAN CHE: detection duoc tinh lai bang model.predict() tren cung frame; GPU khong
bit-reproducible nen toa do co the lech duoi 1 px so voi lan chay tracking goc.
Cac box du doan thi lay tu lan chay lai da doi chieu khop 100% voi prediction da luu.

CACH DUNG
    python src/demo/match_cost_at_reacquisition.py --demo 1 --frames 300-312
    python src/demo/match_cost_at_reacquisition.py --demo 2 --frames 1615-1620
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from baseline_track import load_ignored_regions, mask_ignored_regions  # noqa: E402
from occlusion_eval import iou_matrix  # noqa: E402
from render_thesis_demo import DEMOS, load_gt  # noqa: E402

FUSED_THRESH = 1.0 - 0.8      # match_thresh 0.8 tren chi phi (1 - IoU*score)


def tlwh(df: pd.DataFrame) -> np.ndarray:
    return np.c_[df.cx - df.w / 2, df.cy - df.h / 2, df.w, df.h].astype(float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True, choices=list(DEMOS))
    ap.add_argument("--frames", required=True, help="vd 300-312")
    ap.add_argument("--out", default="outputs/thesis_demos")
    ap.add_argument("--ids", default=None,
                    help="ID track goc cua CV,EKF,UKF (mac dinh: ID ghep target o frame truoc su kien)")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO

    D = DEMOS[a.demo]
    video, split = D["video"], D["split"]
    f0, f1 = (int(v) for v in a.frames.split("-"))
    out_dir = Path(a.out) / D["name"]
    gt = load_gt(split, video)
    gt_t = gt[gt.id == D["gt_id"]]
    keys = list(D["trackers"])
    preds = {k: pd.read_csv(out_dir / "rerun" / f"rerun_pred_{D['trackers'][k]}.csv") for k in keys}

    if a.ids:
        ids = dict(zip(keys, (int(v) for v in a.ids.split(","))))
    else:
        # ID track goc = ID trong metadata (id_before) neu co, khong thi doc tu prediction
        import json
        meta = json.load(open(out_dir / "metadata.json", encoding="utf-8"))
        ids = {k: int(meta["trackers"][k]["id_before"]) for k in keys}

    model = YOLO(str(config.MODELS_DIR / "yolov8n.pt"))
    regions = load_ignored_regions(video, split)
    root = config.find_images_root() / video
    rows = []
    print(f"{D['name']}  {video}  GT {D['gt_id']}  track goc {ids}")
    print("ByteTrack ghep khi IoU(du doan, detection) x score > 0.2. "
          "'doi thu' = track khac trong pool co diem cao nhat voi cung detection.")
    for fr in range(f0, f1 + 1):
        img = mask_ignored_regions(cv2.imread(str(root / f"img{fr:05d}.jpg")), regions)
        r = model.predict(img, conf=D["conf"], classes=config.YOLO_VEHICLE_CLASSES, device="0", verbose=False)[0]
        g = gt_t[gt_t.frame == fr]
        if not len(g) or r.boxes is None or not len(r.boxes):
            print(f"  frame {fr}: khong co GT hoac khong co detection")
            continue
        gb = g[["x", "y", "w", "h"]].to_numpy(float)
        xywh = r.boxes.xywh.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        dets = np.c_[xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 2], xywh[:, 3]]
        ig = iou_matrix(gb, dets)[0]
        j = int(np.argmax(ig))
        row = dict(frame=fr, gt_vis=round(float(g.vis.iloc[0]), 2),
                   det_iou_with_gt=round(float(ig[j]), 3), det_score=round(float(cf[j]), 3))
        line = f"  frame {fr}: GT nhin thay {row['gt_vis']:.2f} | det: IoU(GT)={ig[j]:.3f} score={cf[j]:.3f}"
        for k in keys:
            pool = preds[k][preds[k].frame == fr]
            me = pool[pool.id == ids[k]]
            if not len(me):
                row[f"{k}_iou"] = None; row[f"{k}_fused"] = None; row[f"{k}_matchable"] = None
                row[f"{k}_state"] = "xoa"; row[f"{k}_rival_id"] = None; row[f"{k}_rival_fused"] = None
                line += f" | {k}: (track da bi xoa)"
                continue
            iou_me = float(iou_matrix(tlwh(me), dets[j:j + 1])[0, 0])
            fused_me = iou_me * float(cf[j])
            others = pool[pool.id != ids[k]]
            if len(others):
                iou_o = iou_matrix(tlwh(others), dets[j:j + 1])[:, 0] * float(cf[j])
                jo = int(np.argmax(iou_o))
                rival_id, rival_fused = int(others.id.iloc[jo]), float(iou_o[jo])
            else:
                rival_id, rival_fused = None, 0.0
            row[f"{k}_iou"] = round(iou_me, 3); row[f"{k}_fused"] = round(fused_me, 3)
            row[f"{k}_matchable"] = bool(fused_me > FUSED_THRESH)
            row[f"{k}_state"] = str(me.state.iloc[0])
            row[f"{k}_rival_id"] = rival_id; row[f"{k}_rival_fused"] = round(rival_fused, 3)
            tag = "GHEP" if fused_me > FUSED_THRESH else " -- "
            line += (f" | {k}: IoU={iou_me:.3f} x s={fused_me:.3f} {tag}"
                     + (f" (doi thu {rival_id}: {rival_fused:.3f})" if rival_fused > FUSED_THRESH else ""))
        print(line)
        rows.append(row)
    del model
    out = out_dir / "match_cost_at_reacquisition.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
