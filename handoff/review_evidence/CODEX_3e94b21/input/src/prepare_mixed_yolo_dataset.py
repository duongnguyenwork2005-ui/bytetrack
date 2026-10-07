"""
prepare_mixed_yolo_dataset.py -- Xuat dataset YOLO 5 lop {car, truck, bus, motorcycle, bicycle} cho fine-tune
detector (Giai doan 2).

CHE DO CHINH (plan da chot, vong 27): CHI BDD100K MOT train, dieu khien bang file cau hinh:

    python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon
    python src/prepare_mixed_yolo_dataset.py --config ... --buoc chia        # KHOA split theo video (file commit)
    python src/prepare_mixed_yolo_dataset.py --config ... --buoc xuat [--tap-con smoke]
    python src/prepare_mixed_yolo_dataset.py --config ... --buoc kiem [--tap-con smoke]

Logic + quy tac: `src/bdd_yolo_export.py`. Ma thoat: 0 dat; 1 sai cach dung; 2 THIEU du lieu (in dung goi/duong dan con thieu, KHONG
tao thu muc dich); 3 du lieu sai / trung tap danh gia / dataset xuat ra khong hop le.

CHE DO CU (khong dung trong plan hien tai): UA-DETRAC train -> YOLO 5 lop, chi chay khi goi TUONG MINH
`--nguon detrac`. Ban cu mac dinh tron UA-DETRAC voi BDD va lang le bo qua BDD khi thieu -- ca hai da bi chan:
    - khong con nguon mac dinh: thieu --config va --nguon -> loi;
    - BDD chi xuat qua --config (tron DETRAC + BDD nam ngoai plan -> loi, khong bo qua).

    python src/prepare_mixed_yolo_dataset.py --nguon detrac --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from detector_classes import ID_LOP_5, LOP_5  # noqa: E402

CLASSES = list(LOP_5)
CID = dict(ID_LOP_5)
DETRAC_MAP = {"car": "car", "van": "car", "others": "truck", "bus": "bus"}       # GIA DINH (xem yaml)
DETRAC_MAP_4W = {"car": "car", "van": "car", "bus": "bus"}                        # four-wheel-only: bo 'others'


def write_sample(img_path: Path, boxes, masks, out_img: Path, out_lbl: Path, dry: bool) -> int:
    """boxes: (cid, left, top, w, h) px; masks: (x, y, w, h) px to xam. Tra ve so nhan."""
    import cv2
    if dry:
        return len(boxes)
    im = cv2.imread(str(img_path))
    if im is None:
        return 0
    H, W = im.shape[:2]
    for x, y, w, h in masks:
        x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(W, int(x + w)), min(H, int(y + h))
        if x2 > x1 and y2 > y1:
            im[y1:y2, x1:x2] = config.IGNORE_MASK_COLOR_BGR
    lines = [f"{c} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
             for c, x, y, w, h in boxes if w > 0 and h > 0]
    cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_lbl.write_text("\n".join(lines))
    return len(lines)


def export_detrac(a, out: Path, stats: Counter, lists: dict, dry: bool) -> dict:
    from export_yolo_dataset import pick_val_videos
    from mot_datasets.detrac import DetracDataset
    ds = DetracDataset()
    df = ds.gt_dataframe("train")
    videos = sorted(df.video.unique())
    val = a.detrac_val_videos or pick_val_videos(a.detrac_n_val, a.seed)
    train = [v for v in videos if v not in val]
    cmap = DETRAC_MAP if a.detrac_classes == "mapped" else DETRAC_MAP_4W
    n_img = 0
    for split, vids in (("train", train), ("val", val)):
        for v in vids:
            g = df[df.video == v]
            regions = ds.ignored_regions(v, "train")
            frames = sorted(g.frame.unique())[::a.detrac_stride]
            for fr in frames:
                rows = g[g.frame == fr]
                boxes = []
                for r in rows.itertuples():
                    cat = cmap.get(str(r.vehicle_type))
                    if cat is None:
                        continue
                    boxes.append((CID[cat], float(r.bb_left), float(r.bb_top), float(r.bb_width), float(r.bb_height)))
                    stats[f"detrac_{split}_{cat}"] += 1
                if not boxes:
                    continue
                src = ds.img_root / v / f"img{fr:05d}.jpg"
                stem = f"detrac_{v}_{fr:05d}"
                n = write_sample(src, boxes, regions, out / "images" / split / f"{stem}.jpg",
                                 out / "labels" / split / f"{stem}.txt", dry)
                if n:
                    lists[split].append(f"images/{split}/{stem}.jpg")
                    n_img += 1
    return {"detrac_train_videos": train, "detrac_val_videos": val, "n_images": n_img, "class_map": cmap}


def chay_theo_cau_hinh(a) -> int:
    import bdd_yolo_export as bx
    try:
        cfg = bx.doc_cau_hinh(Path(a.config))
        file_split = bx.p_abs(cfg, cfg["file_split"])
        if a.buoc == "kiem-nguon":
            bc = bx.kiem_nguon(cfg)
            bc = {k: v for k, v in bc.items() if k != "video_dung_duoc"} | {"so_video_dung_duoc": len(bc["video_dung_duoc"])}
            print(json.dumps(bc, ensure_ascii=False, indent=1)[:4000])
        elif a.buoc == "chia":
            rec = bx.tao_split(cfg, file_split)
            print(f"[chia] da khoa {file_split}: train_fit {len(rec['train_fit'])} video, dev {len(rec['dev'])} video, "
                  f"sha256_danh_sach {rec['sha256_danh_sach'][:16]}")
            print(json.dumps(rec["thong_ke_nguon_theo_split"], ensure_ascii=False, indent=1))
        elif a.buoc == "xuat":
            tom = bx.xuat_dataset(cfg, file_split, a.tap_con, kiem_hash_noi_dung=not a.bo_kiem_hash_noi_dung)
            print(json.dumps({k: tom[k] for k in ("ten", "video", "dem_frame", "thong_ke", "manifest_anh",
                                                  "kiem_sau_xuat")}, ensure_ascii=False, indent=1))
        elif a.buoc == "kiem":
            out = bx.p_abs(cfg, cfg["dau_ra"] + (f"_{a.tap_con}" if a.tap_con else ""))
            print(json.dumps(bx.kiem_dataset(out), ensure_ascii=False, indent=1))
        return 0
    except bx.ThieuDuLieu as e:
        print(f"[THIEU DU LIEU] {e}", file=sys.stderr)
        for m in e.thieu:
            print(f"  - {m}", file=sys.stderr)
        print("  -> xem handoff/BDD100K_INVENTORY.md muc 'Vong 27' (goi can tai + duong dan dat). "
              "KHONG thay bang UA-DETRAC hay BDD val.", file=sys.stderr)
        return 2
    except bx.LoiDuLieu as e:
        print(f"[LOI DU LIEU] {e}", file=sys.stderr)
        return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="cau hinh BDD-only (vd configs/phase2/detector/bdd5_train.yaml)")
    ap.add_argument("--buoc", choices=["kiem-nguon", "chia", "xuat", "kiem"], help="di kem --config")
    ap.add_argument("--tap-con", default=None, help="ten tap con trong cau hinh (vd smoke)")
    ap.add_argument("--bo-kiem-hash-noi-dung", action="store_true",
                    help="bo kiem sha256 anh voi tap danh gia (CHI dung trong test fixture)")
    ap.add_argument("--nguon", nargs="+", choices=["detrac", "bdd"], default=None,
                    help="CHE DO CU: nguon tuong minh (khong con mac dinh)")
    ap.add_argument("--out", default=str(config.DATA_DIR / "yolo_mixed5"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detrac-stride", type=int, default=5)
    ap.add_argument("--detrac-n-val", type=int, default=12)
    ap.add_argument("--detrac-val-videos", nargs="*", default=None)
    ap.add_argument("--detrac-classes", choices=["mapped", "four-wheel-only"], default="mapped")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    def sai_cach_dung(msg: str) -> int:              # exit 1 (exit 2 danh cho THIEU du lieu)
        print(f"[SAI CACH DUNG] {msg}", file=sys.stderr)
        return 1

    if a.config:
        if not a.buoc:
            return sai_cach_dung("--config can --buoc {kiem-nguon,chia,xuat,kiem}")
        if a.nguon:
            return sai_cach_dung("--config (BDD-only) khong di cung --nguon")
        return chay_theo_cau_hinh(a)
    if not a.nguon:
        return sai_cach_dung("khong con nguon mac dinh: dung --config <bdd5_train.yaml> (plan hien tai) "
                             "hoac --nguon detrac (che do cu)")
    if "bdd" in a.nguon:
        return sai_cach_dung("BDD chi xuat qua --config (chia theo video, am tinh, crowd/ignore, kiem trung tap "
                             "danh gia); tron UA-DETRAC + BDD nam ngoai plan da chot")

    out = Path(a.out)
    if not a.dry_run:
        for s in ("train", "val"):
            (out / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    lists = {"train": [], "val": []}
    manifest = {"classes": CLASSES, "args": vars(a), "detrac": export_detrac(a, out, stats, lists, a.dry_run)}
    manifest["label_counts"] = dict(sorted(stats.items()))
    manifest["n_train_images"], manifest["n_val_images"] = len(lists["train"]), len(lists["val"])
    if not a.dry_run:
        (out / "train.txt").write_text("\n".join(lists["train"]))
        (out / "val.txt").write_text("\n".join(lists["val"]))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "args"}, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
