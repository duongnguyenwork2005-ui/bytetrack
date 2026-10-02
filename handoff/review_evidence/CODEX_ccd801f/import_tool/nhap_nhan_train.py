#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 34 -- nhap NGUYEN BYTE 1.400 JSON native train (phuong an C) tu ban cach ly vao duong chinh.

Nguon : data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot/bdd100k/labels/box_track_20/train/   (giu nguyen)
Dich  : data/raw/bdd100k/labels/box_track_20/train/<video>.json
Chuan : handoff/logs/review32/09_manifest_nhan_ung_vien.tsv -- cac hang bdd100k/labels/box_track_20/train/*.json

Pha 1 (KIEM TRUOC, chua ghi gi; sai -> exit 2):
  - nguon co DUNG tap file cua manifest (khong thieu, khong thua), tung file dung so byte + sha256;
  - dich: file trung sha256 -> "da co" (giu nguyen); file khac sha256 hoac file NGOAI danh sach -> XUNG DOT, dung,
    khong ghi de, khong xoa.
Pha 2 (NHAP): chep tung file con thieu qua file tam roi doi ten; kiem sha256 ngay sau khi ghi.
Pha 3 (KIEM SAU): dich dung 1.400 file .json, khong file la; sha256 khop manifest va khop nguon; nguon khong doi;
  COCO val khong doi.
Provenance + manifest nhap ghi vao --ra (handoff), KHONG dat vao thu muc nhan.
--chi-kiem: chi chay pha 1. Ma thoat: 0 dat; 1 kiem sau sai; 2 kiem truoc sai / xung dot.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

TIEN_TO = "bdd100k/labels/box_track_20/train/"
REVISION = "694197099692fe1254e93215e73e68ece8b1f41f"
HF_REPO = "jaffe03195/bdd100k_sot"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bay_gio():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goc-cach-ly", default="data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot")
    ap.add_argument("--dich", default="data/raw/bdd100k/labels/box_track_20/train")
    ap.add_argument("--manifest", default="handoff/logs/review32/09_manifest_nhan_ung_vien.tsv")
    ap.add_argument("--coco-val", default="data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
    ap.add_argument("--coco-val-sha256", default="074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d")
    ap.add_argument("--so-file", type=int, default=1400)
    ap.add_argument("--ra", required=True, help="JSON provenance/ket qua (dat trong handoff)")
    ap.add_argument("--ra-tsv", help="manifest nhap tung file (dat trong handoff)")
    ap.add_argument("--chi-kiem", action="store_true")
    a = ap.parse_args()

    goc, dich = Path(a.goc_cach_ly), Path(a.dich)
    kq = {"t_bat_dau": bay_gio(), "phuong_an": "C", "nguon_hf": {"repo": HF_REPO, "revision": REVISION, "file": "bdd100k.zip"},
          "manifest": {"duong_dan": a.manifest, "sha256": sha256(a.manifest)}, "nguon_cach_ly": (goc / TIEN_TO).as_posix(),
          "dich": dich.as_posix(), "che_do": "chi_kiem" if a.chi_kiem else "nhap"}

    def ket_thuc(ma, **them):
        kq.update(them)
        kq["t_xong"], kq["exit"] = bay_gio(), ma
        Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: v for k, v in kq.items() if k not in ("vi_du",)}, ensure_ascii=True))
        return ma

    # --- pha 1: kiem truoc
    hang = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"), delimiter="\t")
            if r["duong_dan"].startswith(TIEN_TO) and r["duong_dan"].endswith(".json")]
    chuan = {r["duong_dan"][len(TIEN_TO):]: (int(r["byte"]), r["sha256"]) for r in hang}
    if len(chuan) != a.so_file or len(hang) != a.so_file:
        return ket_thuc(2, loi=f"manifest co {len(hang)} hang train ({len(chuan)} ten khac nhau), can {a.so_file}")
    thu_muc_nguon = goc / TIEN_TO
    co_nguon = {p.name for p in thu_muc_nguon.iterdir() if p.is_file()} if thu_muc_nguon.is_dir() else set()
    thieu_nguon, thua_nguon = sorted(set(chuan) - co_nguon), sorted(co_nguon - set(chuan))
    lech_nguon = []
    for ten in sorted(set(chuan) & co_nguon):
        p = thu_muc_nguon / ten
        if p.stat().st_size != chuan[ten][0] or sha256(p) != chuan[ten][1]:
            lech_nguon.append(ten)
    coco_truoc = sha256(a.coco_val)
    kq["kiem_truoc"] = {"so_file_manifest_train": len(chuan), "so_file_nguon": len(co_nguon), "thieu_o_nguon": thieu_nguon[:20],
                        "so_thieu_o_nguon": len(thieu_nguon), "thua_o_nguon": thua_nguon[:20], "so_thua_o_nguon": len(thua_nguon),
                        "so_lech_byte_hoac_sha256": len(lech_nguon), "vi_du_lech": lech_nguon[:20],
                        "coco_val_sha256_truoc": coco_truoc, "coco_val_khop_gia_tri_nghiem_thu": coco_truoc == a.coco_val_sha256}
    da_co, xung_dot, la = [], [], []
    if dich.exists():
        for p in sorted(dich.iterdir()):
            if p.name not in chuan:
                la.append(p.name)
            elif sha256(p) == chuan[p.name][1]:
                da_co.append(p.name)
            else:
                xung_dot.append(p.name)
    kq["kiem_dich_truoc"] = {"dich_ton_tai": dich.exists(), "so_da_co_trung_hash": len(da_co),
                             "so_khac_hash": len(xung_dot), "vi_du_khac_hash": xung_dot[:20],
                             "so_file_ngoai_danh_sach": len(la), "vi_du_ngoai_danh_sach": la[:20]}
    if thieu_nguon or thua_nguon or lech_nguon or coco_truoc != a.coco_val_sha256:
        return ket_thuc(2, loi="nguon cach ly / COCO val khong khop manifest hoac gia tri da nghiem thu -- DUNG truoc khi nhap")
    if xung_dot or la:
        return ket_thuc(2, loi="dich da co file khac hash hoac file ngoai danh sach -- XUNG DOT, dung, khong ghi de/xoa")
    if a.chi_kiem:
        return ket_thuc(0, ghi_chu="chi kiem truoc; chua ghi gi")

    # --- pha 2: nhap nguyen byte
    dich.mkdir(parents=True, exist_ok=True)
    moi = 0
    for ten in sorted(chuan):
        if ten in da_co:
            continue
        ra, tam = dich / ten, dich / (ten + ".tmp_nhap")
        b = (thu_muc_nguon / ten).read_bytes()
        tam.write_bytes(b)
        if hashlib.sha256(tam.read_bytes()).hexdigest() != chuan[ten][1]:
            tam.unlink()
            return ket_thuc(1, loi=f"sha256 sai ngay sau khi ghi {ten}; da xoa file tam, dung")
        os.replace(tam, ra)
        moi += 1
    kq["nhap"] = {"so_ghi_moi": moi, "so_da_co_giu_nguyen": len(da_co)}

    # --- pha 3: kiem sau
    co_dich = sorted(p.name for p in dich.iterdir())
    lech_dich = [t for t in co_dich if t in chuan and sha256(dich / t) != chuan[t][1]]
    la_dich = [t for t in co_dich if t not in chuan]
    lech_nguon_sau = [t for t in sorted(chuan) if sha256(thu_muc_nguon / t) != chuan[t][1]]
    coco_sau = sha256(a.coco_val)
    kq["kiem_sau"] = {"so_file_dich": len(co_dich), "so_khop_manifest": len(co_dich) - len(lech_dich) - len(la_dich),
                      "so_lech": len(lech_dich), "file_la_trong_dich": la_dich[:20], "nguon_cach_ly_van_khop": not lech_nguon_sau,
                      "so_file_nguon_con_lai": sum(1 for _ in thu_muc_nguon.iterdir()),
                      "coco_val_sha256_sau": coco_sau, "coco_val_khong_doi": coco_sau == coco_truoc}
    if a.ra_tsv:
        with open(a.ra_tsv, "w", encoding="utf-8", newline="\n") as f:
            f.write("duong_dan_dich\tbyte\tsha256\tnguon_cach_ly\n")
            for ten in sorted(chuan):
                f.write(f"{(dich / ten).as_posix()}\t{chuan[ten][0]}\t{chuan[ten][1]}\t{(thu_muc_nguon / ten).as_posix()}\n")
        kq["manifest_nhap"] = {"duong_dan": a.ra_tsv, "sha256": sha256(a.ra_tsv)}
    dat = (len(co_dich) == a.so_file and not lech_dich and not la_dich and not lech_nguon_sau and coco_sau == coco_truoc)
    return ket_thuc(0 if dat else 1)


if __name__ == "__main__":
    sys.exit(main())
