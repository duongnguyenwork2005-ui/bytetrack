#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 32 -- chay kiem_nguon CUA DU AN tren ban nhan UNG VIEN dang cach ly, KHONG nhap vao du lieu chinh.

Giong het `prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon`, chi khac:
bdd.nhan.duong_dan bi doi TRONG BO NHO sang thu muc cach ly (doi so --nhan). Khong sua config da khoa, khong ghi file.
Ma thoat nhu CLI: 0 dat, 2 ThieuDuLieu, 3 LoiDuLieu.
"""
import argparse
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))

import bdd_yolo_export as bx  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2/detector/bdd5_train.yaml")
    ap.add_argument("--nhan", required=True, help="thu muc box_track_20/train dang cach ly")
    a = ap.parse_args()
    cfg = bx.doc_cau_hinh(GOC / a.config)
    cu = cfg["bdd"]["nhan"]["duong_dan"]
    cfg["bdd"]["nhan"]["duong_dan"] = str(Path(a.nhan).resolve())
    print(f"[cach ly] bdd.nhan.duong_dan: {cu} -> {cfg['bdd']['nhan']['duong_dan']} (chi trong bo nho)")
    try:
        bc = bx.kiem_nguon(cfg)
        bc = {k: v for k, v in bc.items() if k != "video_dung_duoc"} | {"so_video_dung_duoc": len(bc["video_dung_duoc"])}
        print(json.dumps(bc, ensure_ascii=False, indent=1)[:4000])
        return 0
    except bx.ThieuDuLieu as e:
        print(f"[THIEU DU LIEU] {e}", file=sys.stderr)
        for m in e.thieu:
            print(f"  - {m}", file=sys.stderr)
        return 2
    except bx.LoiDuLieu as e:
        print(f"[LOI DU LIEU] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
