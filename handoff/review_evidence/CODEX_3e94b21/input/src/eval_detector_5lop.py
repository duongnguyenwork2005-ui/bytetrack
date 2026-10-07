"""
eval_detector_5lop.py -- Danh gia detector 5 lop (car, truck, bus, motorcycle, bicycle) theo CUNG mot quy tac cho moi
checkpoint: YOLOv8n COCO (baseline, 80 lop), YOLOv8n fine-tune, YOLO11n fine-tune (5 lop, ID 0..4).

Thay the `eval_detector.py` (ban cu: UA-DETRAC mot lop "vehicle", bo class ID cua GT, kich thuoc anh co dinh 960x540)
cho moi bao cao chat luong detector 5 lop. KHONG gop lop: moi so lieu bao cao theo tung lop; trung binh (macro) chi
lay tren cac lop co GT trong tap.

QUY TAC (ghi nguyen vao ket qua):
  Du doan : ultralytics predict, imgsz 640, conf 0.001, NMS IoU 0.7 theo lop (khong agnostic), max_det 300/anh,
            FP32; chi giu lop cua checkpoint anh xa duoc sang 5 lop THEO TEN (`detector_classes`), loc TRUOC NMS.
  Dataset : TRUOC moi du doan, `bdd_yolo_export.kiem_hop_dong_dataset` (cung hop dong voi exporter / runner) xac minh
            noi dung anh, nhan, ignore cua split duoc danh gia; danh tinh da xac minh ghi vao ket qua. GT doc theo khai
            bao cua manifest (file ignore da khai bao ma thieu -> LOI, khong doc thanh 0 vung).
  GT      : nhan YOLO cua dataset (class ID 0..4), doi ra pixel bang KICH THUOC ANH THUC (header file).
  Ghep    : theo COCO (`COCOeval.evaluateImg`) -- trong tung (anh, lop), detection xep conf giam dan (mergesort), ghep
            voi GT chua ghep co IoU cao nhat >= nguong; HOA IoU -> GT SAU trong thu tu duyet (nhu pycocotools, R27-F3);
            IoU tinh nhu `bbIou` cua pycocotools. Detection khong ghep duoc ma IoA (giao / dien tich detection) voi mot
            vung bo qua (to xam, file `ignore/<split>/<anh>.txt`) >= nguong -> BO QUA (khong TP, khong FP).
  AP      : 10 nguong IoU 0.50:0.05:0.95, precision bao dong tu phai, noi suy 101 diem recall (COCO).
            AP50 = nguong 0.50; AP50-95 = trung binh 10 nguong. Lop khong co GT -> AP = null (khong tinh vao mAP).
  Van hanh: precision / recall / F1 tai conf >= 0.25 (nguong tracker c25) va IoU 0.5, theo lop.

    python src/eval_detector_5lop.py --data data/yolo_bdd5_v1/data.yaml --split val \
        --model yolov8n_coco=models/yolov8n.pt --model yolov8n_ft=<best.pt> --out results/phase2/detector_eval/<ten>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector_classes import LOP_5, anh_xa_model_sang_lop5, mo_ta_anh_xa  # noqa: E402

#: Nguong da chon (giu nguyen tu vong 27). Khac mang `Params.iouThrs` cua pycocotools DUY NHAT o 0.9 (pycocotools:
#: 0.8999999999999999, lech 1 ulp) -> chi khac ket qua neu co IoU nam trong 1 ulp duoi 0.9 (ghi o gioi han).
NGUONG_IOU = np.round(np.arange(0.5, 0.951, 0.05), 2)
DIEM_RECALL = np.linspace(0.0, 1.0, 101)
QUY_TAC = {
    "du_doan": {"imgsz": 640, "conf": 0.001, "iou_nms": 0.7, "max_det": 300, "agnostic_nms": False, "half": False,
                "loc_lop": "chi lop anh xa duoc sang 5 lop theo ten, loc truoc NMS"},
    "ghep": "COCO (pycocotools evaluateImg): conf giam dan (mergesort) trong (anh, lop); GT chua ghep co IoU cao nhat "
            ">= nguong, HOA IoU -> GT SAU trong thu tu duyet; IoU tinh nhu bbIou cua pycocotools (tu xywh)",
    "bo_qua": "detection khong ghep ma IoA voi vung to xam >= nguong -> bo qua (nhu crowd cua COCO)",
    "ap": "10 nguong IoU 0.50:0.05:0.95, noi suy 101 diem recall (COCO); lop khong co GT -> null",
    "van_hanh": {"conf": 0.25, "iou": 0.5},
    "gop_lop": "KHONG -- moi so lieu theo lop; macro = trung binh cac lop co GT",
}


# =========================================================================== doc GT
def kich_thuoc_anh(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return int(im.size[0]), int(im.size[1])


def duong_dan_nhan(img: Path, thu_muc: str = "labels") -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = thu_muc
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"{img}: khong nam duoi thu muc 'images'")


def doc_gt(img: Path, ignore_khai_bao: bool | None = None) -> dict:
    """GT cua mot anh: cls (n,), xyxy (n,4) pixel; vung bo qua (m,4). Thieu file nhan -> LOI (khong coi la anh rong).

    `ignore_khai_bao` (tu manifest): True -> file ignore PHAI ton tai (thieu = LOI, khong doc thanh 0 vung);
    False -> anh KHONG co vung ignore, file ignore khong duoc ton tai; None -> khong co manifest (chi dung cho test)."""
    W, H = kich_thuoc_anh(img)
    lbl = duong_dan_nhan(img)
    if not lbl.is_file():
        raise FileNotFoundError(f"thieu file nhan {lbl} (khong coi anh thieu nhan la anh am tinh)")
    cls, box = [], []
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            raise ValueError(f"{lbl}: dong sai dinh dang {ln!r}")
        c = int(p[0])
        if not 0 <= c < len(LOP_5):
            raise ValueError(f"{lbl}: lop {c} ngoai 0..{len(LOP_5) - 1}")
        cx, cy, w, h = (float(x) for x in p[1:])
        cls.append(c)
        box.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    ign = []
    ip = duong_dan_nhan(img, "ignore")
    if ignore_khai_bao is True and not ip.is_file():
        raise FileNotFoundError(f"thieu file ignore DA KHAI BAO {ip} (khong doc thanh 0 vung)")
    if ignore_khai_bao is False and ip.exists():
        raise ValueError(f"{ip}: manifest khai bao anh KHONG co vung ignore nhung file ton tai")
    if ip.is_file():
        for ln in ip.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                ign.append([float(x) for x in ln.split()[:4]])
    return {"anh": img.name, "w": W, "h": H, "cls": np.array(cls, int), "xyxy": np.array(box, float).reshape(-1, 4),
            "ign": np.array(ign, float).reshape(-1, 4)}


def ds_anh_tu_data_yaml(data_yaml: Path, split: str = "val") -> list[Path]:
    import yaml
    y = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    goc = Path(y.get("path") or Path(data_yaml).parent)
    d = goc / y[split]
    if not d.is_dir():
        raise FileNotFoundError(f"{d} (split {split} cua {data_yaml})")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def chuan_bi_danh_gia(data_yaml: Path, split: str = "val") -> tuple[dict, list[Path], list[dict]]:
    """Kiem HOP DONG dataset (cung ham voi exporter / runner; xac minh NOI DUNG anh, nhan, ignore cua split duoc danh
    gia) roi doc GT theo khai bao ignore cua manifest. Tra ve (danh tinh dataset da xac minh, anh, GT)."""
    import bdd_yolo_export as bx
    ten_split = {"val": "dev", "train": "train"}.get(split)
    if ten_split is None:
        raise ValueError(f"split {split!r} khong thuoc hop dong (val = dev noi bo, train)")
    danh_tinh = bx.kiem_hop_dong_dataset(data_yaml, split_noi_dung=(ten_split,))
    goc = Path(data_yaml).resolve().parent
    anh = [goc / "images" / ten_split / p.name for p in bx.liet_ke_anh_nhu_ultralytics(goc / "images" / ten_split)]
    khai = bx.gt_theo_manifest(data_yaml, ten_split)
    gt = [doc_gt(p, ignore_khai_bao=khai[p.name]["ignore"]) for p in anh]
    return danh_tinh, anh, gt


# =========================================================================== ghep + AP
def _giao_xywh(a: np.ndarray, b: np.ndarray):
    """Phan giao nhu `bbIou` (maskApi.c) cua pycocotools: box doi sang xywh, canh phai/duoi = x + w / y + h."""
    ax, ay, aw, ah = a[:, 0], a[:, 1], a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    bx, by, bw, bh = b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    w = np.minimum(ax[:, None] + aw[:, None], bx[None, :] + bw[None, :]) - np.maximum(ax[:, None], bx[None, :])
    h = np.minimum(ay[:, None] + ah[:, None], by[None, :] + bh[None, :]) - np.maximum(ay[:, None], by[None, :])
    inter = np.where((w > 0) & (h > 0), w * h, 0.0)
    return inter, aw * ah, bw * bh


def iou_ma_tran(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (a: detection, b: GT) dung cong thuc bbIou cua pycocotools; khong giao -> 0."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter, da, ga = _giao_xywh(a, b)
    u = da[:, None] + ga[None, :] - inter
    return np.divide(inter, u, out=np.zeros_like(inter), where=inter > 0)


def ioa_ma_tran(det: np.ndarray, vung: np.ndarray) -> np.ndarray:
    """Giao / dien tich DETECTION -- nhu IoU voi GT iscrowd cua pycocotools (u = da)."""
    if len(det) == 0 or len(vung) == 0:
        return np.zeros((len(det), len(vung)))
    inter, da, _ = _giao_xywh(det, vung)
    return np.divide(inter, np.broadcast_to(da[:, None], inter.shape), out=np.zeros_like(inter), where=inter > 0)


def ghep_anh_lop(det: np.ndarray, conf: np.ndarray, gt: np.ndarray, ign: np.ndarray, nguong=NGUONG_IOU):
    """Tra ve (thu tu conf giam dan, tp[T, n], bo_qua[T, n]) cho mot (anh, lop).

    Theo dung vong lap `COCOeval.evaluateImg`: moi detection (conf giam dan) duyet GT theo thu tu, bo GT da ghep, giu GT
    co IoU >= muc hien tai (`if iou < best: continue; best = iou; m = g`) -> khi HOA IoU chon GT SAU (R27-F3). Khong ghep
    duoc GT thuong ma IoA voi mot vung bo qua >= nguong -> bo qua (nhu GT iscrowd, cho phep ghep nhieu lan).
    """
    order = np.argsort(-conf, kind="mergesort")
    det = det[order]
    n, T = len(det), len(nguong)
    tp = np.zeros((T, n), bool)
    bq = np.zeros((T, n), bool)
    if n == 0:
        return order, tp, bq
    iou = iou_ma_tran(det, gt)
    ioa = ioa_ma_tran(det, ign)
    max_iou = iou.max(1) if gt.size else np.zeros(n)
    max_ioa = ioa.max(1) if ign.size else np.zeros(n)
    for ti, t in enumerate(nguong):
        t_eff = min(float(t), 1 - 1e-10)
        da_ghep = np.zeros(len(gt), bool)
        for d in range(n):
            if max_iou[d] >= t_eff:
                cand = np.where(da_ghep, -np.inf, iou[d])
                mx = cand.max()
                if mx >= t_eff:
                    m = int(np.flatnonzero(cand == mx)[-1])        # HOA -> GT SAU (nhu pycocotools)
                    tp[ti, d] = True
                    da_ghep[m] = True
                    continue
            if max_ioa[d] >= t_eff:
                bq[ti, d] = True
    return order, tp, bq


def ap_coco(conf: np.ndarray, tp: np.ndarray, bo_qua: np.ndarray, n_gt: int) -> float:
    """AP 101 diem cho MOT nguong IoU; conf/tp/bo_qua da noi theo thu tu anh (moi anh da xep conf giam dan)."""
    if n_gt == 0:
        return float("nan")
    giu = ~bo_qua
    conf, tp = conf[giu], tp[giu]
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf, kind="mergesort")
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rc = ctp / n_gt
    pr = ctp / np.maximum(ctp + cfp, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, DIEM_RECALL, side="left")
    q = np.zeros(len(DIEM_RECALL))
    ok = idx < len(pr)
    q[ok] = pr[idx[ok]]
    return float(q.mean())


def danh_gia(gt_ds: list[dict], du_doan: list[dict], conf_van_hanh: float = 0.25, iou_van_hanh: float = 0.5) -> dict:
    """gt_ds[i] khop du_doan[i] (cung anh): du_doan = {'xyxy','conf','cls'} voi cls da o he 5 lop."""
    if len(gt_ds) != len(du_doan):
        raise ValueError("so anh GT va du doan khac nhau")
    ti_vh = int(np.argmin(np.abs(NGUONG_IOU - iou_van_hanh)))
    if abs(NGUONG_IOU[ti_vh] - iou_van_hanh) > 1e-9:
        raise ValueError("iou van hanh phai la mot trong cac nguong 0.50:0.05:0.95")
    theo_lop = {}
    for c, ten in enumerate(LOP_5):
        confs, tps, bqs = [], [], []
        n_gt = n_anh = 0
        for g, p in zip(gt_ds, du_doan):
            mg = g["cls"] == c
            md = p["cls"] == c
            n_gt += int(mg.sum())
            n_anh += int(mg.any())
            o, tp, bq = ghep_anh_lop(p["xyxy"][md], p["conf"][md], g["xyxy"][mg], g["ign"])
            confs.append(p["conf"][md][o])
            tps.append(tp)
            bqs.append(bq)
        conf = np.concatenate(confs) if confs else np.zeros(0)
        tp = np.concatenate(tps, axis=1) if tps else np.zeros((len(NGUONG_IOU), 0), bool)
        bq = np.concatenate(bqs, axis=1) if bqs else np.zeros((len(NGUONG_IOU), 0), bool)
        aps = [ap_coco(conf, tp[t], bq[t], n_gt) for t in range(len(NGUONG_IOU))]
        vh = (conf >= conf_van_hanh) & ~bq[ti_vh]
        TP = int(tp[ti_vh][vh].sum())
        FP = int(vh.sum()) - TP
        P = TP / (TP + FP) if TP + FP else float("nan")
        R = TP / n_gt if n_gt else float("nan")
        theo_lop[ten] = {"n_gt": n_gt, "n_anh_co_lop": n_anh, "n_det_conf_thap": int(len(conf)),
                         "AP50": aps[0], "AP75": aps[int(np.argmin(np.abs(NGUONG_IOU - 0.75)))],
                         "AP50_95": float(np.mean(aps)) if n_gt else float("nan"),
                         "van_hanh": {"TP": TP, "FP": FP, "FN": n_gt - TP, "precision": P, "recall": R,
                                      "f1": 2 * P * R / (P + R) if n_gt and TP else (0.0 if n_gt else float("nan"))}}
    co_gt = [t for t in LOP_5 if theo_lop[t]["n_gt"] > 0]

    def tb(f):
        v = [f(theo_lop[t]) for t in co_gt]
        v = [x for x in v if x == x]
        return float(np.mean(v)) if v else float("nan")

    macro = {"so_lop_co_gt": len(co_gt), "lop_khong_co_gt": [t for t in LOP_5 if t not in co_gt],
             "mAP50": tb(lambda r: r["AP50"]), "mAP50_95": tb(lambda r: r["AP50_95"]),
             "precision_vh": tb(lambda r: r["van_hanh"]["precision"]),
             "recall_vh": tb(lambda r: r["van_hanh"]["recall"])}
    return {"theo_lop": theo_lop, "macro": macro, "n_anh": len(gt_ds),
            "van_hanh": {"conf": conf_van_hanh, "iou": iou_van_hanh}}


# =========================================================================== du doan
def du_doan_ultralytics(weights: str, anh: list[Path], qt: dict, batch: int = 16, device="0"):
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    m = anh_xa_model_sang_lop5(names)
    keep = sorted(m)
    ra = []
    for i in range(0, len(anh), batch):
        res = model.predict([str(p) for p in anh[i:i + batch]], imgsz=qt["imgsz"], conf=qt["conf"], iou=qt["iou_nms"],
                            max_det=qt["max_det"], classes=keep, agnostic_nms=qt["agnostic_nms"], half=qt["half"],
                            device=device, verbose=False)
        for r in res:
            b = r.boxes
            cls_m = b.cls.cpu().numpy().astype(int) if b is not None and len(b) else np.zeros(0, int)
            ra.append({"xyxy": b.xyxy.cpu().numpy().astype(float) if len(cls_m) else np.zeros((0, 4)),
                       "conf": b.conf.cpu().numpy().astype(float) if len(cls_m) else np.zeros(0),
                       "cls": np.array([m[int(k)] for k in cls_m], int)})
    return ra, mo_ta_anh_xa(names)


def luu_du_doan(p: Path, anh: list[Path], dd: list[dict]) -> None:
    idx = np.concatenate([np.full(len(d["conf"]), i) for i, d in enumerate(dd)]) if dd else np.zeros(0)
    np.savez_compressed(p, anh=np.array([a.name for a in anh]), idx=idx.astype(np.int32),
                        xyxy=np.concatenate([d["xyxy"] for d in dd]).astype(np.float32) if dd else np.zeros((0, 4)),
                        conf=np.concatenate([d["conf"] for d in dd]).astype(np.float32) if dd else np.zeros(0),
                        cls=np.concatenate([d["cls"] for d in dd]).astype(np.int8) if dd else np.zeros(0))


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json_safe(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description="Danh gia detector 5 lop, cung quy tac cho moi checkpoint")
    ap.add_argument("--data", required=True, help="data.yaml cua dataset (vd data/yolo_bdd5_v1/data.yaml)")
    ap.add_argument("--split", default="val", help="khoa trong data.yaml (val = DEV NOI BO)")
    ap.add_argument("--model", action="append", required=True, metavar="TEN=DUONG_DAN")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-van-hanh", type=float, default=QUY_TAC["van_hanh"]["conf"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--gioi-han-anh", type=int, default=0, help="chi N anh dau (0 = tat ca) -- CHI cho smoke")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"[LOI] {out} da ton tai -- khong ghi de ket qua cu", file=sys.stderr)
        return 1
    import bdd_yolo_export as bx
    try:
        danh_tinh, anh, gt = chuan_bi_danh_gia(Path(a.data), a.split)      # HOP DONG: truoc moi du doan
    except (bx.LoiDuLieu, FileNotFoundError, ValueError) as e:
        print(f"[LOI DATASET] {e}", file=sys.stderr)
        return 2
    if a.gioi_han_anh:
        anh, gt = anh[:a.gioi_han_anh], gt[:a.gioi_han_anh]
    ds_info = {"data": str(a.data), "split": a.split, "n_anh": len(anh), "gioi_han_anh": a.gioi_han_anh,
               "dataset_da_xac_minh": danh_tinh}
    out.mkdir(parents=True)
    kq = {"tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "quy_tac": QUY_TAC, "dataset": ds_info,
          "mo_hinh": {}}
    rows = []
    for spec in a.model:
        ten, _, w = spec.partition("=")
        dd, mt = du_doan_ultralytics(w, anh, QUY_TAC["du_doan"], a.batch, a.device)
        luu_du_doan(out / f"du_doan_{ten}.npz", anh, dd)
        r = danh_gia(gt, dd, a.conf_van_hanh)
        kq["mo_hinh"][ten] = {"checkpoint": w, "sha256": sha256_file(Path(w)), "anh_xa_lop": mt, **r}
        for lop, v in r["theo_lop"].items():
            rows.append({"mo_hinh": ten, "lop": lop, "n_gt": v["n_gt"], "AP50": v["AP50"], "AP50_95": v["AP50_95"],
                         "precision_vh": v["van_hanh"]["precision"], "recall_vh": v["van_hanh"]["recall"]})
        print(f"[{ten}] mAP50 {r['macro']['mAP50']:.4f}  mAP50-95 {r['macro']['mAP50_95']:.4f}  "
              + "  ".join(f"{k}:AP50={v['AP50']:.3f}" for k, v in r["theo_lop"].items()))
    (out / "ket_qua.json").write_text(json.dumps(_json_safe(kq), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(out / "theo_lop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
