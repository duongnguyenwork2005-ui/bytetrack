# -*- coding: utf-8 -*-
"""Vong 43: kiem TRUC TIEP ba checkpoint truoc khi danh gia -- CHI DOC (khong sua file, khong train).

    python handoff/logs/review43/kiem_checkpoint.py --out handoff/logs/review43/01_kiem_checkpoint.json

Moi checkpoint:
  1. sha256 tinh tu FILE THUC, so voi gia tri da co dinh (nguoi dung / review 4d54744). Lech -> KHONG nap, exit 1.
  2. Nap bang ham san co `tiep_tuc_huan_luyen.doc_checkpoint` (torch.load, map_location cpu) -> cau truc: cac khoa,
     kieu `model`, so tensor / tham so, dtype, epoch, phien ban ultralytics, train_args chinh.
  3. Tensor HUU HAN: moi tensor so thuc trong `model.state_dict()` phai isfinite toan bo (buoc moi -- du an chua co
     cong cu nay; dung torch.isfinite).
  4. Ten lop: `ultralytics.YOLO(path).names` + `detector_classes.loai_checkpoint` / `mo_ta_anh_xa` (san co). Fine-tune
     phai la 'lop5' dung 5 lop car, truck, bus, motorcycle, bicycle (ID 0..4); baseline phai la 'coco80', anh xa sang
     5 lop THEO TEN (khong dung chung chi so lop voi model fine-tune).
Thoat 0 khi ca ba dat; 1 khi co checkpoint khong dat."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))

CHECKPOINT = [
    ("yolov8n_coco", "models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36", "coco80"),
    ("yolov8n_ft", "models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt",
     "5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a", "lop5"),
    ("yolo11n_ft", "models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt",
     "d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923", "lop5"),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kiem(ten: str, rel: str, sha_mong: str, loai_mong: str) -> dict:
    import torch
    from detector_classes import LOP_5, loai_checkpoint, mo_ta_anh_xa
    from tiep_tuc_huan_luyen import doc_checkpoint
    p = GOC / rel
    kq = {"ten": ten, "duong_dan": rel, "bytes": p.stat().st_size, "sha256_mong_doi": sha_mong}
    kq["sha256_file"] = sha(p)
    kq["sha256_khop"] = kq["sha256_file"] == sha_mong
    if not kq["sha256_khop"]:
        kq["dat"] = False
        kq["ly_do"] = "sha256 lech -- KHONG nap"
        return kq
    ck = doc_checkpoint(p)                                   # ham san co (torch.load, cpu)
    m = ck.get("model")
    sd = m.state_dict() if m is not None else {}
    tensor_so_thuc = {k: v for k, v in sd.items() if hasattr(v, "is_floating_point") and v.is_floating_point()}
    khong_huu_han = [k for k, v in tensor_so_thuc.items() if not bool(torch.isfinite(v).all())]
    ta = ck.get("train_args") or {}
    kq["cau_truc"] = {
        "khoa": sorted(ck.keys()), "kieu_model": type(m).__name__ if m is not None else None,
        "ema": None if ck.get("ema") is None else type(ck.get("ema")).__name__,
        "optimizer": None if ck.get("optimizer") is None else "co", "epoch": ck.get("epoch"),
        "version": ck.get("version"), "date": ck.get("date"),
        "so_tensor_state_dict": len(sd), "so_tensor_so_thuc": len(tensor_so_thuc),
        "so_tham_so": int(sum(p_.numel() for p_ in m.parameters())) if m is not None else None,
        "dtype_tham_so": sorted({str(p_.dtype) for p_ in m.parameters()}) if m is not None else None,
        "train_args_chinh": {k: ta.get(k) for k in ("model", "data", "epochs", "imgsz", "batch", "optimizer", "seed",
                                                    "name") if k in ta}}
    kq["tensor_huu_han"] = {"so_tensor_kiem": len(tensor_so_thuc), "khong_huu_han": khong_huu_han,
                            "dat": not khong_huu_han}
    from ultralytics import YOLO
    names = YOLO(str(p)).names                              # cach train_detector / evaluator doc ten lop
    loai = loai_checkpoint(names)
    kq["lop"] = {"so_lop": len(names), "loai_checkpoint": loai, "loai_mong_doi": loai_mong,
                 "ten_0_4": [names.get(i) for i in range(min(5, len(names)))],
                 "anh_xa_sang_lop5": mo_ta_anh_xa(names)}
    kq["lop"]["dat"] = (loai == loai_mong) and (loai != "lop5" or [names[i] for i in range(5)] == list(LOP_5))
    kq["dat"] = kq["sha256_khop"] and kq["tensor_huu_han"]["dat"] and kq["lop"]["dat"] and m is not None
    return kq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import ultralytics
    ds = [kiem(*c) for c in CHECKPOINT]
    out = {"tao_luc": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "moi_truong": {"python": sys.version.split()[0], "torch": torch.__version__,
                          "ultralytics": ultralytics.__version__},
           "checkpoint": ds, "dat_ca_ba": all(d["dat"] for d in ds)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    for d in ds:
        print(f"[{'DAT' if d['dat'] else 'KHONG DAT'}] {d['ten']}: sha256 {d['sha256_file'][:12]}... khop={d['sha256_khop']}"
              + (f" | {d['cau_truc']['kieu_model']} {d['cau_truc']['so_tham_so']} tham so {d['cau_truc']['dtype_tham_so']}"
                 f" | tensor huu han {d['tensor_huu_han']['so_tensor_kiem'] - len(d['tensor_huu_han']['khong_huu_han'])}/"
                 f"{d['tensor_huu_han']['so_tensor_kiem']} | lop {d['lop']['loai_checkpoint']} ({d['lop']['so_lop']} lop) "
                 f"{d['lop']['anh_xa_sang_lop5']['ten']}" if d["sha256_khop"] else ""))
    print("KET LUAN:", "DAT ca ba" if out["dat_ca_ba"] else "KHONG DAT")
    return 0 if out["dat_ca_ba"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
