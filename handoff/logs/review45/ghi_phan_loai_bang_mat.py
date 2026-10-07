# -*- coding: utf-8 -*-
"""Vong 45: ghi PHAN LOAI BANG MAT cua tac gia (CHU QUAN, mot crop o frame giua moi track, chua kiem cheo) cho mau
track `others` (du 31 track) va `van` (48 track ngau nhien seed 0), roi tinh ti le track / box theo nhom.
KHONG doi nhan dataset; chi la bang chung mo ta de nguoi dung quyet D1 / D2.
    python handoff/logs/review45/ghi_phan_loai_bang_mat.py --thu-muc handoff/logs/review45/phan_tich"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# stt trong mau_track_<loai>.csv -> nhom quan sat duoc
OTHERS = {**{i: "xe tai cho hang (ben / thung / san / buu chinh)" for i in
             (0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 26, 28, 30)},
          **{i: "xe cuu ho / keo xe" for i in (5, 20, 21, 22, 25, 27, 29)},
          9: "trong giong xe buyt",
          1: "khong ro (nhin tu tren xuong, nhu nap thung xe tai)",
          6: "khong ro (xe nho, co the la xe cuu ho nhin truc dien)"}
VAN = {**{i: "van cho nguoi / minivan / MPV (kieu xe con)" for i in range(48)},
       29: "minibus / xe khach nho (kieu xe buyt)", 42: "minibus / xe khach nho (kieu xe buyt)",
       44: "van cho hang (panel van)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thu-muc", required=True)
    a = ap.parse_args()
    d = Path(a.thu_muc)
    for loai, bang in (("others", OTHERS), ("van", VAN)):
        hang = list(csv.DictReader(open(d / f"mau_track_{loai}.csv", encoding="utf-8")))
        assert len(hang) == (31 if loai == "others" else 48)
        tr, bx = Counter(), Counter()
        for r in hang:
            r["nhom_bang_mat"] = bang[int(r["stt"])]
            tr[r["nhom_bang_mat"]] += 1
            bx[r["nhom_bang_mat"]] += int(r["so_box_track"])
        with open(d / f"phan_loai_bang_mat_{loai}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hang[0]))
            w.writeheader()
            w.writerows(hang)
        tong_b = sum(bx.values())
        print(f"== {loai}: {len(hang)} track mau, {tong_b} box cua cac track mau")
        for k in tr:
            print(f"   {k}: {tr[k]} track, {bx[k]} box ({bx[k] / tong_b:.1%} box cua mau)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
