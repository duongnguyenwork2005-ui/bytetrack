#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 33 -- bam lai nhan cach ly (theo manifest vong 32) va COCO val, so voi gia tri da ghi. Chi doc.

python handoff/logs/review33/kiem_hash_nhan.py --ra <json>
Ma thoat 0 neu moi file khop manifest va COCO val khop sha256 da khoa; 1 neu co file lech/thieu.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

Q = Path("data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot")
MANIFEST = Path("handoff/logs/review32/09_manifest_nhan_ung_vien.tsv")
COCO = Path("data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
COCO_SHA256 = "074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    dong = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t"))
    lech, thieu = [], []
    for r in dong:
        p = Q / r["duong_dan"]
        if not p.is_file():
            thieu.append(r["duong_dan"])
        elif sha256(p) != r["sha256"]:
            lech.append(r["duong_dan"])
    coco = sha256(COCO)
    kq = {"manifest": MANIFEST.as_posix(), "sha256_manifest": sha256(MANIFEST), "so_file_manifest": len(dong),
          "so_khop": len(dong) - len(lech) - len(thieu), "so_lech": len(lech), "vi_du_lech": lech[:10],
          "so_thieu": len(thieu), "vi_du_thieu": thieu[:10],
          "coco_val": COCO.as_posix(), "coco_val_sha256": coco, "coco_val_khop_gia_tri_khoa": coco == COCO_SHA256}
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(kq, ensure_ascii=True))
    return 0 if not lech and not thieu and coco == COCO_SHA256 else 1


if __name__ == "__main__":
    sys.exit(main())
