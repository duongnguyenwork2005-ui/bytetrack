"""
visdrone_smoke_det.py -- R66 Q3: smoke DETECTION (khong tracking) VisDrone2019-MOT val de kiem kha nang chuyen mien.

Quyet dinh GPT 02/11 (`hoi_codex_R66_dataset_moi_pf_20261102/answer.json` Q3), co dinh TRUOC khi xem ket qua:
    - Mau: moi chuoi val lay 10 frame dau, 10 frame giua, 10 frame cuoi (7 chuoi -> 210 frame).
    - Ca 3 detector R54, cung spec cache R64 (imgsz 640 rect, conf 0,05, iou 0,7, max_det 3000, FP32); mixed5 -> ID COCO.
    - GT hop le: score == 1 va category trong vehicle4 {4 car, 5 van, 6 truck, 9 bus}; du doan: COCO {2 car, 5 bus,
      7 truck}.
    - Vung bo qua: hop cac box category 0 (ignored regions) va 11 (others) TUNG FRAME; box (GT va du doan) co ty le dien
      tich nam trong vung >= 0,5 bi loai (quy tac dropObjects cua toolkit).
    - Recall ghep mot-mot IoU >= 0,5 tai conf > 0,05 / > 0,10 / > 0,25; bao so GT, kich thuoc box, ty le frame co TP.
    - Cong kha thi: >= 100 GT hop le toan smoke; neu CA BA detector recall < 1 % tai conf > 0,05 -> dung tracking dai.
Khong tang resolution, doi checkpoint hay nguong sau khi xem.

    python -B src/r66_ngoai/visdrone_smoke_det.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src")); sys.path.insert(0, str(GOC / "src" / "r64_ngoai"))
import danh_gia_ngoai as r64  # noqa: E402
import det_cache  # noqa: E402
from visdrone_ignore import ignored_fraction  # noqa: E402

VD = Path("F:/LABOTARY_data/visdrone_mot/VisDrone2019-MOT-val")
OUT = GOC / "handoff" / "logs" / "external_eval" / "r66" / "visdrone" / "smoke_det"
GT_LOP = {4, 5, 6, 9}
DU_DOAN_LOP = {2, 5, 7}
NGUONG = [0.05, 0.10, 0.25]


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    it = iw * ih
    return it / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - it)


def main() -> int:
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in (VD / "sequences").iterdir())
    mau = {}
    for s in seqs:
        anh = sorted((VD / "sequences" / s).glob("*.jpg"))
        n = len(anh); g = n // 2 - 5
        idx = list(range(1, 11)) + list(range(g + 1, g + 11)) + list(range(n - 9, n + 1))
        mau[s] = [(k, anh[k - 1]) for k in idx]
    gt = {s: np.loadtxt(VD / "annotations" / f"{s}.txt", delimiter=",", ndmin=2) for s in seqs}
    kq = {"mau": {s: [k for k, _ in v] for s, v in mau.items()},
          "ignore_method": "VisDrone2018-MOT-toolkit/eval/dropObjects.m geometry",
          "detector": {}}
    for det in ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"):
        c = r64.detector_r54(det)
        ck = GOC / c["checkpoint"]
        if r64.sha256_file(ck) != c["sha256"]:
            raise SystemExit(f"sha256 {det} lech")
        spec = det_cache.DetectorSpec(backend="ultralytics", weights=str(ck), weights_sha256=c["sha256"], imgsz=640,
                                      conf=0.05, iou=0.7, classes=None, max_det=3000, half=False, agnostic_nms=False,
                                      augment=False, batch=1, device="0", mask_ignored_regions=False,
                                      dataset="visdrone", part="val")
        model = YOLO(str(ck))
        bang = r64.bang_anh_xa_coco(model.names, c["loai_checkpoint"])
        tp = {t: 0 for t in NGUONG}; n_gt = 0; frame_co_tp = {t: 0 for t in NGUONG}; n_frame = 0
        h_gt, h_tp = [], {t: [] for t in NGUONG}; cham_tran = 0
        for s in seqs:
            g = gt[s]
            for k, p in mau[s]:
                img = cv2.imread(str(p)); H, W = img.shape[:2]
                d = r64.doi_id_sang_coco(det_cache.detect_frame(model, img, spec), bang)
                cham_tran += int(len(d) >= 3000)
                gk = g[g[:, 0] == k]
                vung = gk[np.isin(gk[:, 7], [0, 11])][:, 2:6]
                hop = gk[(gk[:, 6] == 1) & np.isin(gk[:, 7], list(GT_LOP))][:, 2:6]
                hop = hop[ignored_fraction(hop, vung, H, W) < 0.5]
                n_gt += len(hop); n_frame += 1; h_gt += hop[:, 3].tolist()
                for t in NGUONG:
                    dd = d[(d[:, 4] > t) & np.isin(d[:, 5], list(DU_DOAN_LOP))]
                    db = np.c_[dd[:, 0], dd[:, 1], dd[:, 2] - dd[:, 0], dd[:, 3] - dd[:, 1]]
                    db = db[ignored_fraction(db, vung, H, W) < 0.5] if len(db) else db
                    m = iou(hop, db)
                    if m.size:
                        r_, c_ = linear_sum_assignment(np.where(m >= 0.5, -m, 1e6))
                        ok = m[r_, c_] >= 0.5
                        tp[t] += int(ok.sum()); frame_co_tp[t] += int(ok.any())
                        h_tp[t] += hop[r_[ok], 3].tolist()
        q = lambda x, p: float(np.quantile(x, p)) if len(x) else None  # noqa: E731
        kq["detector"][det] = {
            "n_frame": n_frame, "n_gt_hop_le": n_gt, "frame_cham_tran_3000": cham_tran,
            "recall": {f">{t}": tp[t] / n_gt if n_gt else None for t in NGUONG},
            "tp": {f">{t}": tp[t] for t in NGUONG},
            "ty_le_frame_co_tp": {f">{t}": frame_co_tp[t] / n_frame for t in NGUONG},
            "h_gt_px": {"p10": q(h_gt, .1), "trung_vi": q(h_gt, .5), "p90": q(h_gt, .9)},
            "h_tp_px_trung_vi": {f">{t}": q(h_tp[t], .5) for t in NGUONG}}
        print(det, kq["detector"][det]["recall"], "n_gt", n_gt, flush=True)
    rec = [kq["detector"][d]["recall"][">0.05"] for d in kq["detector"]]
    n_gt = kq["detector"]["yolov8n_coco"]["n_gt_hop_le"]
    kq["cong"] = ("CHUA_DU_BANG_CHUNG (< 100 GT)" if n_gt < 100 else
                  "DUNG_TRACKING_DAI (ca 3 recall < 1 %)" if all(r < 0.01 for r in rec) else
                  "QUA_CONG_KHA_THI (con phai xac minh FPS truoc full)")
    # Keep the R66 result for audit; this file is the corrected rerun.
    (OUT / "ket_qua_v2.json").write_text(json.dumps(kq, indent=1), encoding="utf-8")
    print("cong:", kq["cong"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
