#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- DEM (khong xuat) so anh dataset DAY DU theo split da khoa, de cap nhat du toan training local.

Chay tu goc repo:  python handoff/logs/review35/dem_anh_day_du.py --ra handoff/logs/review35/<ten>.json
Dung DUNG ham cua bo xuat (bdd_yolo_export.doc_split -> chon_tap_con(None) -> chon_frame) nen so anh la so bo xuat se
chon voi cung split/nguon nhan. KHONG ghi anh, nhan hay thu muc dataset nao; chi doc nhan JSON va stat file anh.
- so_anh: chinh xac theo quy tac chon frame (buoc_frame trong cau hinh = 5, va them buoc 10 de so sanh phuong an).
- box/anh co lop/anh to xam/am tinh: dem tren nhan nguon theo vai_tro() TRUOC khi cat box vao khung anh -> XAP XI
  (bo xuat con bo box < 1 px sau khi cat; anh co box chi nam ngoai khung co the thanh am tinh).
- byte_anh_nguon: tong kich thuoc JPEG nguon cua frame duoc chon (anh khong to xam duoc chep nguyen byte; anh to xam ghi
  lai JPEG q95 -> kich thuoc khac nguon) -> la XAP XI dung luong anh cua dataset xuat.
"""
import argparse
import json
import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))
import bdd_yolo_export as bx  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(GOC / "configs/phase2/detector/bdd5_train.yaml"))
    ap.add_argument("--split", default=str(GOC / "configs/phase2/detector/splits/bdd5_train_v1.json"))
    ap.add_argument("--buoc", type=int, nargs="+", default=None, help="mac dinh: buoc_frame cau hinh va 10")
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cfg = bx.doc_cau_hinh(Path(a.config))
    goc = bx.goc_du_an(cfg)
    nguon = bx.mo_nguon(cfg["bdd"]["nhan"], goc)
    file_split = Path(a.split)
    sp = bx.doc_split(file_split, nguon)            # kiem sha256_danh_sach + van tay nguon nhan nhu luc xuat
    ds = bx.chon_tap_con(sp, cfg, None)             # dataset day du = toan bo video cua split da khoa
    anh_dir = bx.p_abs(cfg, cfg["bdd"]["anh"])
    buoc_cfg = int(cfg["lay_mau"]["buoc_frame"])
    ds_buoc = a.buoc or sorted({buoc_cfg, 10})
    kq = {"muc_dich": "dem so anh dataset day du (KHONG xuat) de cap nhat du toan training local",
          "config": {"duong_dan": str(Path(a.config).relative_to(GOC)).replace("\\", "/"),
                     "sha256": bx.sha256_file(Path(a.config))},
          "split": {"duong_dan": str(file_split.relative_to(GOC)).replace("\\", "/"), "sha256": bx.sha256_file(file_split),
                    "sha256_danh_sach": sp["sha256_danh_sach"]},
          "nguon_nhan_sha256": sp["nguon_nhan"]["sha256"], "buoc_frame_cau_hinh": buoc_cfg, "theo_buoc": {}}
    frames_cache = {}
    for s in ("train", "dev"):
        for v in ds[s]:
            frames_cache[(s, v)] = nguon.doc_video(v)
    t_doc = time.time() - t0
    for buoc in ds_buoc:
        r = {}
        for s in ("train", "dev"):
            tk = {"so_video": len(ds[s]), "frame_nguon": 0, "so_anh": 0, "khong_theo_buoc": 0, "ngoai_khoang_co_nhan": 0,
                  "am_tinh_xap_xi": 0, "anh_to_xam_xap_xi": 0, "byte_anh_nguon": 0,
                  "box_xap_xi": {c: 0 for c in bx.LOP_5}, "anh_co_lop_xap_xi": {c: 0 for c in bx.LOP_5}}
            for v in ds[s]:
                frames = frames_cache[(s, v)]
                tk["frame_nguon"] += len(frames)
                chon, d = bx.chon_frame(frames, buoc)
                tk["khong_theo_buoc"] += d["khong_theo_buoc"]
                tk["ngoai_khoang_co_nhan"] += d["ngoai_khoang_co_nhan"]
                tk["so_anh"] += len(chon)
                for f in chon:
                    tk["byte_anh_nguon"] += (anh_dir / f.rel).stat().st_size
                    co, xam = set(), False
                    for cat, *_b, crowd, ign in f.nhan:
                        vt = bx.vai_tro(cat, crowd, ign)
                        if vt == "gan_nhan":
                            tk["box_xap_xi"][cat] += 1
                            co.add(cat)
                        elif vt == "to_xam":
                            xam = True
                    for c in co:
                        tk["anh_co_lop_xap_xi"][c] += 1
                    tk["am_tinh_xap_xi"] += not co
                    tk["anh_to_xam_xap_xi"] += xam
            assert tk["frame_nguon"] == tk["so_anh"] + tk["khong_theo_buoc"] + tk["ngoai_khoang_co_nhan"]
            tk["byte_trung_binh_moi_anh"] = round(tk["byte_anh_nguon"] / max(tk["so_anh"], 1))
            r[s] = tk
        kq["theo_buoc"][str(buoc)] = r
    kq["thoi_gian_s"] = {"doc_nhan": round(t_doc, 1), "tong": round(time.time() - t0, 1)}
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for b, r in kq["theo_buoc"].items():
        print(f"buoc {b}: train {r['train']['so_anh']} anh ({r['train']['byte_anh_nguon'] / 1e9:.2f} GB nguon), "
              f"dev {r['dev']['so_anh']} anh ({r['dev']['byte_anh_nguon'] / 1e9:.2f} GB nguon)")
    print(f"[ra] {a.ra}  ({kq['thoi_gian_s']['tong']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
