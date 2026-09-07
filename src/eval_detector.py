"""
eval_detector.py -- Giai doan C: do chat luong detector TRUOC va SAU fine-tune
tren cung tap val, cung thang do.

===========================================================================
VI SAO PHAI TU VIET, KHONG DUNG THANG model.val() CUA ULTRALYTICS?
===========================================================================
Model goc (COCO pretrained) xuat 80 lop, dataset cua ta co 1 lop ("vehicle").
`model.val()` khong so sanh truc tiep duoc. Phai anh xa cac lop xe cua COCO
(2=car, 3=motorcycle, 5=bus, 7=truck - dung config.YOLO_VEHICLE_CLASSES) ve
lop 0 roi moi tinh.

De hai ben THUC SU so sanh duoc, ca hai deu duoc cham bang CUNG mot ham AP tu
viet o day, tren CUNG danh sach anh val.

===========================================================================
HAI THANG DO, DO HAI THU KHAC NHAU
===========================================================================
1. AP@0.5 va AP@0.5:0.95 - thang do chuan cua bai toan phat hien, doc lap
   nguong tin cay.

2. Precision / Recall tai conf = 0.25 - thang do QUAN TRONG HON voi de tai nay,
   vi 0.25 chinh la nguong ma tracker dang dung (config.YOLO_DEFAULT_CONF).
   Ti le `lost` cua tracker bi chi phoi truc tiep boi RECALL tai nguong nay:
   detector khong ra detection thi khong motion model nao cuu duoc.

===========================================================================
CACH DUNG
    python src/eval_detector.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CONF_EVAL = 0.001    # nguong thap khi tinh AP (can ca duoi cong tin cay)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giua 2 tap box dang xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """AP theo phuong phap tich phan toan bo duong cong (all-point interpolation).

    Sap xep detection theo do tin cay giam dan, tinh precision/recall tich luy,
    lam "don dieu hoa" precision tu phai sang trai roi lay dien tich duoi duong.
    """
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):      # lam don dieu tu phai sang trai
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def load_gt(img_path: Path) -> np.ndarray:
    """Doc nhan YOLO cua mot anh, tra ve box xyxy theo pixel."""
    lbl = Path(str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
    lbl = lbl.with_suffix(".txt")
    if not lbl.exists():
        return np.zeros((0, 4))
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        out.append([(cx - w / 2) * config.IMG_W, (cy - h / 2) * config.IMG_H,
                    (cx + w / 2) * config.IMG_W, (cy + h / 2) * config.IMG_H])
    return np.array(out) if out else np.zeros((0, 4))


def evaluate(model_path: str, img_list: list[Path], coco_classes: bool,
             label: str) -> dict:
    """Cham mot model tren mot danh sach anh. Tra ve AP + P/R tai conf 0.25."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    cls_filter = config.YOLO_VEHICLE_CLASSES if coco_classes else None

    thr_list = np.arange(0.5, 1.0, 0.05)
    tp_at = {t: [] for t in thr_list}
    confs = []
    n_gt = 0
    # P/R tai nguong van hanh cua tracker
    tp25 = fp25 = fn25 = 0

    for p in tqdm(img_list, desc=f"  {label}", unit="img"):
        gt = load_gt(p)
        n_gt += len(gt)
        r = model.predict(str(p), conf=CONF_EVAL, classes=cls_filter,
                          device="0", verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            fn25 += len(gt)
            continue
        det = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        order = np.argsort(-cf)
        det, cf = det[order], cf[order]
        ious = iou_matrix(det, gt)
        confs.append(cf)
        for t in thr_list:
            used = np.zeros(len(gt), bool)
            tp = np.zeros(len(det))
            for i in range(len(det)):
                if len(gt) == 0:
                    break
                j = int(np.argmax(np.where(used, -1, ious[i])))
                if ious[i, j] >= t and not used[j]:
                    tp[i] = 1
                    used[j] = True
            tp_at[t].append(tp)
        # --- P/R tai conf 0.25, IoU 0.5 ---
        keep = cf >= config.YOLO_DEFAULT_CONF
        d2, i2 = det[keep], ious[keep]
        used = np.zeros(len(gt), bool)
        m = 0
        for i in range(len(d2)):
            if len(gt) == 0:
                break
            j = int(np.argmax(np.where(used, -1, i2[i])))
            if i2[i, j] >= 0.5 and not used[j]:
                m += 1
                used[j] = True
        tp25 += m
        fp25 += len(d2) - m
        fn25 += len(gt) - m

    conf_all = np.concatenate(confs) if confs else np.zeros(0)
    aps = {}
    for t in thr_list:
        tp_all = np.concatenate(tp_at[t]) if tp_at[t] else np.zeros(0)
        aps[round(float(t), 2)] = average_precision(tp_all, conf_all, n_gt)

    prec = tp25 / max(tp25 + fp25, 1)
    rec = tp25 / max(tp25 + fn25, 1)
    return {
        "n_images": len(img_list), "n_gt": n_gt,
        "mAP50": round(aps[0.5], 4),
        "mAP50-95": round(float(np.mean(list(aps.values()))), 4),
        "precision@0.25": round(prec, 4),
        "recall@0.25": round(rec, 4),
        "f1@0.25": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Do detector truoc/sau fine-tune")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=400,
                    help="Lay mau bao nhieu anh val moi fold (0 = tat ca)")
    a = ap.parse_args()

    root = config.DATA_DIR / "yolo_detrac"
    out = {}
    for k in range(a.n_folds):
        val_txt = root / f"fold{k}_val.txt"
        if not val_txt.exists():
            print(f"  [MISS] {val_txt}")
            continue
        imgs = [Path(x) for x in val_txt.read_text().splitlines() if x.strip()]
        if a.max_images and len(imgs) > a.max_images:
            step = len(imgs) // a.max_images
            imgs = imgs[::step][:a.max_images]
        ft = config.PROJECT_ROOT / "models" / "finetune" / f"fold{k}" / "weights" / "best.pt"
        print(f"\n=== FOLD {k} ({len(imgs)} anh val) ===")
        out[f"fold{k}"] = {
            "truoc (COCO pretrained)": evaluate(str(config.MODELS_DIR / "yolov8n.pt"),
                                                imgs, True, "truoc"),
            "sau (fine-tune)": evaluate(str(ft), imgs, False, "sau"),
        }

    res_dir = config.RESULTS_DIR / "phaseC"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "detector_before_after.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 86)
    print("CHAT LUONG DETECTOR TRUOC / SAU FINE-TUNE (tren video val cua tung fold)")
    print("=" * 86)
    print(f"{'fold':<7} {'model':<24} {'mAP50':>8} {'mAP50-95':>10} "
          f"{'P@0.25':>8} {'R@0.25':>8} {'F1@0.25':>9}")
    print("-" * 86)
    for f, d in out.items():
        for name, m in d.items():
            print(f"{f:<7} {name:<24} {m['mAP50']:>8.4f} {m['mAP50-95']:>10.4f} "
                  f"{m['precision@0.25']:>8.4f} {m['recall@0.25']:>8.4f} {m['f1@0.25']:>9.4f}")
        print()
    if out:
        for name in ("truoc (COCO pretrained)", "sau (fine-tune)"):
            v = [d[name] for d in out.values() if name in d]
            print(f"  TRUNG BINH {name:<24} mAP50={np.mean([x['mAP50'] for x in v]):.4f}  "
                  f"R@0.25={np.mean([x['recall@0.25'] for x in v]):.4f}")
    print(f"\n  Da luu -> {res_dir / 'detector_before_after.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
