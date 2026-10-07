"""
danh_gia_detector_bdd_val.py -- Danh gia detector 5 lop tren BDD100K MOT val (ban COCO-video da dinh danh cua du an).

Vi sao co file nay (vong 43): `eval_detector_5lop.py` (evaluator 5 lop da nghiem thu) CHI doc dataset YOLO theo hop dong
`bdd_yolo_export` voi split dev / train (`chuan_bi_danh_gia`: {'val': 'dev', 'train': 'train'}); BDD val chi co anh goc
+ mot file nhan COCO-video, nen evaluator KHONG co duong vao (tai hien: logs/review43/02_tai_hien_chan_bdd_val.log).
File nay CHI them buoc DOC GT; du doan, ghep, AP va van hanh dung NGUYEN ham cua evaluator:
    du_doan_ultralytics (imgsz 640, conf 0.001, NMS IoU 0.7 theo lop, max_det 300, FP32, loc lop theo TEN truoc NMS),
    danh_gia (ghep COCO, vung bo qua theo IoA, AP 101 diem x 10 nguong IoU, van hanh conf 0.25 / IoU 0.5), luu_du_doan.

GT tu COCO-video -- dung NGUYEN quy tac cua exporter 5 lop (`bdd_yolo_export`, da nghiem thu cho train / dev):
  - doc nhan bang `NguonCocoVid` (iscrowd, ignore giu nguyen cua file; KHONG sua 183 lech lop / 50 ngoai le ignore so voi
    native, KHONG dung native val);
  - `vai_tro`: car/truck/bus/motorcycle/bicycle khong crowd, khong ignore -> GT lop 0..4; crowd/iscrowd cua 5 lop va
    distractor nhom xe (ignore=1 voi car/truck trong COCO-video) -> VUNG BO QUA; pedestrian, rider, train, other person
    -> khong GT, khong vung bo qua;
  - `_hop_le_box`: cat box vao khung anh; box < 1 px sau cat -> bo (dem rieng);
  - toa do pixel theo KICH THUOC ANH THUC (header); khac w/h khai bao trong nhan -> LOI.
  KHONG to xam anh (anh val giu nguyen byte); vung bo qua chi dung khi cham diem (nhu crowd cua COCO).
Tap anh: MOI anh ma file nhan liet ke cho 200 video cua manifest da khoa (39.973 anh), KE CA anh khong co annotation
nao (khong bo anh am tinh / anh khong nhan); danh sach phai TRUNG KHOP tap .jpg tren dia (thieu / thua -> LOI).

Kiem truoc moi du doan (sai -> exit 2, khong du doan):
  sha256 file nhan == manifest; tap video trong nhan == manifest; tap anh == dia; w/h; sha256 NOI DUNG tung anh (doc
  lai toan bo, khong dung loi tat mtime) == bang hash da luu (`--hash-anh`); sha256 checkpoint == `--sha256` khai bao.

    python src/danh_gia_detector_bdd_val.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=<hex> \
        --out results/phase2/detector_eval/bdd_val_cocovid/<ten luot>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from bdd_yolo_export import LoiDuLieu, NguonCocoVid, _hop_le_box, vai_tro  # noqa: E402
from detector_classes import ID_LOP_5, LOP_5  # noqa: E402
from eval_detector_5lop import (QUY_TAC, _json_safe, danh_gia, du_doan_ultralytics, kich_thuoc_anh,  # noqa: E402
                                luu_du_doan, sha256_file)

MAC_DINH = {
    "nhan": "data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json",
    "anh": "data/raw/bdd100k/images/track/val",
    "manifest": "results/phase2/manifests/bdd100k_transfer_cocovid_val.json",
    "hash_anh": "data/interim/hash_tap_danh_gia/bdd100k_mot_val_sha256.csv",
}
QUY_TAC_GT = {
    "nguon": "COCO-video val cua du an (khong native); iscrowd / ignore giu nguyen",
    "gan_nhan": "car/truck/bus/motorcycle/bicycle, khong crowd, khong ignore -> GT lop 0..4 (bdd_yolo_export.vai_tro)",
    "vung_bo_qua": "crowd/iscrowd cua 5 lop + distractor nhom xe (ignore=1 voi car/truck) -> vung bo qua (vai_tro)",
    "lop_khac": "pedestrian, rider, train, other person -> khong GT, khong vung bo qua",
    "box": "cat vao khung anh; < 1 px sau cat -> bo (bdd_yolo_export._hop_le_box); pixel theo kich thuoc anh thuc",
    "anh": "moi anh trong file nhan cua 200 video manifest, ke ca anh khong annotation; khong to xam anh",
}


def p_abs(s: str | Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def sha256_ds(rows) -> str:
    """Dau van tay tong hop: sha256 cua cac dong '<rel>\\t<sha256>\\n' da sap xep."""
    h = hashlib.sha256()
    for rel, s in sorted(rows):
        h.update(f"{rel}\t{s}\n".encode())
    return h.hexdigest()


def kiem_va_doc_gt(nhan: Path, anh_goc: Path, manifest: Path, hash_anh: Path | None,
                   kiem_noi_dung: bool = True) -> tuple[dict, list[Path], list[dict]]:
    """Kiem dau vao roi doc GT theo quy tac exporter. Tra ve (danh tinh, anh, gt). Loi -> LoiDuLieu."""
    mf = json.loads(manifest.read_text(encoding="utf-8"))
    sha_nhan = sha256_file(nhan)
    if sha_nhan != mf.get("nguon_nhan_sha256"):
        raise LoiDuLieu(f"{nhan}: sha256 {sha_nhan[:16]} khac manifest {str(mf.get('nguon_nhan_sha256'))[:16]}")
    nguon = NguonCocoVid(nhan)
    vids = sorted(mf["videos"])
    co = nguon.danh_sach_video()
    if co != vids:
        raise LoiDuLieu(f"tap video trong nhan ({len(co)}) khac manifest ({len(vids)}): "
                        f"thieu {sorted(set(vids) - set(co))[:3]}, thua {sorted(set(co) - set(vids))[:3]}")
    frames = [f for v in vids for f in nguon.doc_video(v)]
    rel_nhan = [f.rel for f in frames]
    if len(set(rel_nhan)) != len(rel_nhan):
        raise LoiDuLieu("file nhan liet ke trung anh")
    tren_dia = sorted(p.relative_to(anh_goc).as_posix() for v in vids for p in (anh_goc / v).glob("*.jpg"))
    if sorted(rel_nhan) != tren_dia:
        a, b = set(rel_nhan), set(tren_dia)
        raise LoiDuLieu(f"tap anh trong nhan ({len(a)}) khac tren dia ({len(b)}): thieu tren dia {sorted(a - b)[:3]}, "
                        f"anh khong co trong nhan {sorted(b - a)[:3]}")
    dem = {"anh": 0, "anh_khong_annotation": 0, "anh_khong_gt_5_lop": 0, "anh_co_vung_bo_qua": 0,
           "gt_theo_lop": {c: 0 for c in LOP_5}, "vung_bo_qua": 0, "annotation_lop_khac": 0,
           "box_nho_hon_1px_sau_cat": 0, "vung_bo_qua_rong": 0, "box_bi_cat_vao_khung": 0}
    anh, gt, hash_moi = [], [], []
    for f in frames:
        p = anh_goc / f.rel
        W, H = kich_thuoc_anh(p)
        if f.w is not None and (f.w, f.h) != (W, H):
            raise LoiDuLieu(f"{p}: nhan khai bao {f.w}x{f.h} khac anh thuc {W}x{H}")
        cls, box, ign = [], [], []
        for cat, x1, y1, x2, y2, crowd, ig in f.nhan:
            r = vai_tro(cat, crowd, ig)
            if r == "lop_khac":
                dem["annotation_lop_khac"] += 1
                continue
            b = _hop_le_box(x1, y1, x2, y2, W, H)
            if b is None:
                dem["box_nho_hon_1px_sau_cat" if r == "gan_nhan" else "vung_bo_qua_rong"] += 1
                continue
            if r == "gan_nhan":
                if (x1, y1, x2, y2) != b:
                    dem["box_bi_cat_vao_khung"] += 1
                cls.append(ID_LOP_5[cat])
                box.append(b)
                dem["gt_theo_lop"][cat] += 1
            else:
                ign.append(b)
        dem["anh"] += 1
        dem["anh_khong_annotation"] += int(not f.nhan)
        dem["anh_khong_gt_5_lop"] += int(not cls)
        dem["anh_co_vung_bo_qua"] += int(bool(ign))
        dem["vung_bo_qua"] += len(ign)
        anh.append(p)
        gt.append({"anh": p.name, "w": W, "h": H, "cls": np.array(cls, int),
                   "xyxy": np.array(box, float).reshape(-1, 4), "ign": np.array(ign, float).reshape(-1, 4)})
        if kiem_noi_dung:
            hash_moi.append((f.rel, sha256_file(p)))
    danh_tinh = {"nhan": str(nhan), "nhan_sha256": sha_nhan, "manifest": str(manifest),
                 "manifest_sha256": sha256_file(manifest), "anh_goc": str(anh_goc), "so_video": len(vids),
                 "danh_sach_anh_sha256": sha256_ds((r, "") for r in rel_nhan), "dem": dem,
                 "sequence_da_xem_truoc": mf.get("sequence_da_xem_truoc"), "ghi_chu_manifest": mf.get("ghi_chu")}
    if kiem_noi_dung:
        danh_tinh["noi_dung_anh_sha256_tong_hop"] = sha256_ds(hash_moi)
        if hash_anh is not None:
            with open(hash_anh, newline="", encoding="utf-8") as fh:
                cu = {r["rel"]: r["sha256"] for r in csv.DictReader(fh)}
            lech = [r for r, s in hash_moi if cu.get(r) != s]
            danh_tinh["doi_chieu_hash_anh"] = {"file": str(hash_anh), "file_sha256": sha256_file(hash_anh),
                                              "so_anh_bang_hash": len(cu), "khop": len(hash_moi) - len(lech),
                                              "lech": lech[:5]}
            if lech or len(cu) != len(hash_moi):
                raise LoiDuLieu(f"noi dung anh khac bang hash: {len(lech)} anh lech, bang co {len(cu)} / doc {len(hash_moi)}")
    return danh_tinh, anh, gt


def chan_doan(gt: list[dict], dd: list[dict], conf: float) -> dict:
    """So detection conf >= conf tren anh khong co GT 5 lop (chi de mo ta, khong doi metric)."""
    n_khong_gt = sum(1 for g in gt if len(g["cls"]) == 0)
    n_det = sum(int((d["conf"] >= conf).sum()) for g, d in zip(gt, dd) if len(g["cls"]) == 0)
    return {"so_anh_khong_gt_5_lop": n_khong_gt, f"so_det_conf_ge_{conf}_tren_anh_khong_gt": n_det}


def main() -> int:
    ap = argparse.ArgumentParser(description="Danh gia detector 5 lop tren BDD100K MOT val (COCO-video) -- ham "
                                             "du doan / ghep / AP cua eval_detector_5lop")
    ap.add_argument("--nhan-cocovid", default=MAC_DINH["nhan"])
    ap.add_argument("--anh-goc", default=MAC_DINH["anh"])
    ap.add_argument("--manifest-video", default=MAC_DINH["manifest"])
    ap.add_argument("--hash-anh", default=MAC_DINH["hash_anh"], help="bang sha256 anh da luu ('' = bo doi chieu)")
    ap.add_argument("--model", action="append", required=True, metavar="TEN=DUONG_DAN")
    ap.add_argument("--sha256", action="append", required=True, metavar="TEN=SHA256",
                    help="sha256 bat buoc cua checkpoint (lech -> exit 2, khong du doan)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-van-hanh", type=float, default=QUY_TAC["van_hanh"]["conf"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--gioi-han-anh", type=int, default=0, help="chi N anh dau (0 = tat ca) -- CHI cho smoke ky thuat")
    a = ap.parse_args()
    out = p_abs(a.out)
    if out.exists():
        print(f"[LOI] {out} da ton tai -- khong ghi de ket qua cu", file=sys.stderr)
        return 1
    mo_hinh = dict(s.partition("=")[::2] for s in a.model)
    sha_kb = dict(s.partition("=")[::2] for s in a.sha256)
    if set(mo_hinh) != set(sha_kb):
        print(f"[LOI] --model {sorted(mo_hinh)} va --sha256 {sorted(sha_kb)} khong cung tap ten", file=sys.stderr)
        return 2
    for ten, w in mo_hinh.items():
        s = sha256_file(p_abs(w))
        if s != sha_kb[ten]:
            print(f"[LOI CHECKPOINT] {ten}: {w} sha256 {s[:16]} khac khai bao {sha_kb[ten][:16]}", file=sys.stderr)
            return 2
    t0 = dt.datetime.now()
    try:
        danh_tinh, anh, gt = kiem_va_doc_gt(p_abs(a.nhan_cocovid), p_abs(a.anh_goc), p_abs(a.manifest_video),
                                            p_abs(a.hash_anh) if a.hash_anh else None)
    except (LoiDuLieu, FileNotFoundError, ValueError, KeyError) as e:
        print(f"[LOI DATASET] {e}", file=sys.stderr)
        return 2
    print(f"[du lieu] {len(anh)} anh, {danh_tinh['so_video']} video, GT {danh_tinh['dem']['gt_theo_lop']}, "
          f"vung bo qua {danh_tinh['dem']['vung_bo_qua']}, kiem {(dt.datetime.now() - t0).total_seconds():.0f} s", flush=True)
    if a.gioi_han_anh:
        anh, gt = anh[:a.gioi_han_anh], gt[:a.gioi_han_anh]
    out.mkdir(parents=True)
    kq = {"tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "quy_tac": QUY_TAC, "quy_tac_gt": QUY_TAC_GT,
          "dataset": {"ten": "bdd100k_mot_val_cocovid", "n_anh": len(anh), "gioi_han_anh": a.gioi_han_anh,
                      "dataset_da_xac_minh": danh_tinh},
          "lenh": [sys.executable, *sys.argv], "mo_hinh": {}}
    rows = []
    for ten, w in mo_hinh.items():
        t1 = dt.datetime.now()
        dd, mt = du_doan_ultralytics(str(p_abs(w)), anh, QUY_TAC["du_doan"], a.batch, a.device)
        if len(dd) != len(anh):
            print(f"[LOI] {ten}: {len(dd)} ket qua du doan cho {len(anh)} anh", file=sys.stderr)
            return 3
        luu_du_doan(out / f"du_doan_{ten}.npz", anh, dd)
        r = danh_gia(gt, dd, a.conf_van_hanh)
        kq["mo_hinh"][ten] = {"checkpoint": w, "sha256": sha_kb[ten], "anh_xa_lop": mt,
                              "so_anh_da_du_doan": len(dd), "thoi_gian_s": round((dt.datetime.now() - t1).total_seconds(), 1),
                              "chan_doan": chan_doan(gt, dd, a.conf_van_hanh), **r}
        for lop, v in r["theo_lop"].items():
            rows.append({"mo_hinh": ten, "lop": lop, "n_gt": v["n_gt"], "AP50": v["AP50"], "AP50_95": v["AP50_95"],
                         "precision_vh": v["van_hanh"]["precision"], "recall_vh": v["van_hanh"]["recall"]})
        print(f"[{ten}] mAP50 {r['macro']['mAP50']:.4f}  mAP50-95 {r['macro']['mAP50_95']:.4f}  "
              + "  ".join(f"{k}:AP50={v['AP50']:.3f}" for k, v in r["theo_lop"].items()), flush=True)
    (out / "ket_qua.json").write_text(json.dumps(_json_safe(kq), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(out / "theo_lop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
