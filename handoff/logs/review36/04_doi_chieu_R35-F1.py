#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 36 -- doi chieu R35-F1 tu bang chung CO SAN cua vong 35 (CHI DOC file log/JSON; khong nap model, khong
inference, khong train). Muc dich: tach "da quan sat" voi "chua do" truoc khi dinh chinh dien giai.

Chay tu goc repo:  python handoff/logs/review36/04_doi_chieu_R35-F1.py
"""
import json
import re
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
L35 = GOC / "handoff/logs/review35"


def dong_log(ten, mau):
    out = []
    for s in (L35 / ten).read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\[\+\s*[0-9.]+s\] (.*)$", s)
        if m and any(x in m.group(1) for x in mau):
            out.append(m.group(1).strip())
    return out


def main():
    print("# A. Phep so cua 13_chan_doan_khong_detection.py (CKPT): checkpoint nao, bao nhieu lop dau")
    k13 = json.loads((L35 / "13_chan_doan_khong_detection.json").read_text(encoding="utf-8"))
    for ten, v in k13["checkpoint"].items():
        d = v["diem_lop_max_moi_anh"]
        print(f"  {ten:26s} {v['duong_dan']:70s} so_lop_dau={v['so_lop_dau']:2d} | diem lop max/anh trung vi "
              f"{d['trung_vi']:.6f} | box>=0,001: {v['so_box_conf_ge_0_001']}")
    print("  -> so sanh COCO 80 lop (truoc chuyen) voi 5 lop SAU smoke. KHONG co checkpoint/so do model 5 lop sau khoi tao/"
          "nap trong so va TRUOC buoc toi uu dau tien.")
    print("\n# B. Log khoi tao model 5 lop trong smoke lan 2 (chuyen 80 -> 5 lop)")
    for m in ("yolov8n", "yolo11n"):
        for s in dong_log(f"12_smoke_{m}_lan2.log", ("Overriding model.yaml", "Remapped", "Transferred")):
            print(f"  {m}: {s}")
    print("  -> 5 hang phan lop chep tu lop COCO cung ten, nhung KHONG phai moi tensor duoc chuyen; anh huong cua buoc "
          "chuyen 80 -> 5 lop len diem lop CHUA duoc do rieng.")
    print("\n# C. Warmup / LR / buoc toi uu: bang chung thuc chay co trong ket qua runner")
    for m in ("yolov8n", "yolo11n"):
        r = json.loads((L35 / f"12_ket_qua_runner_{m}_lan2.json").read_text(encoding="utf-8"))
        ts, hl, e = r["tham_so_khai_bao"], r["hieu_luc"], r["epoch"][0]
        print(f"  {m}: khai bao epochs={ts['epochs']} warmup_epochs={ts['warmup_epochs']} warmup_bias_lr="
              f"{ts['warmup_bias_lr']} | luc on_train_start: accumulate={hl['accumulate']}, lr nhom = "
              f"{[g['lr'] for g in hl['param_groups']]} | cuoi epoch: lr={e['lr']} | so_batch={e['so_batch']}")
    print("  -> runner chi chup LR luc bat dau va cuoi epoch; KHONG co log tung buoc (so warmup iteration, LR, accumulate, "
          "so buoc optimizer thuc chay). Nhan dinh 'khong co warmup khi epochs=1' chi dua tren DOC MA NGUON ultralytics "
          "(xem logs/review36/00_trang_thai_dau.log), khong phai log thuc chay.")
    print("\n# D. Danh gia chinh (conf 0,001, giu nguyen) va chan doan conf 1e-6 (13b, rieng)")
    k13b = json.loads((L35 / "13b_danh_gia_conf_thap.json").read_text(encoding="utf-8"))
    for m in ("yolov8n", "yolo11n"):
        r = json.loads((L35 / f"12_ket_qua_runner_{m}_lan2.json").read_text(encoding="utf-8"))
        tl = r["danh_gia_dev_smoke"]["theo_lop"]
        tl6 = k13b["ket_qua"][m]["theo_lop"]
        for c in tl:
            print(f"  {m} {c:10s} n_gt {tl[c]['n_gt']:3d} | conf 0,001: n_det {tl[c]['n_det_conf_thap']:5d} AP50 "
                  f"{tl[c]['AP50']} | conf 1e-6: n_det {tl6[c]['n_det_conf_thap']:5d} AP50 {tl6[c]['AP50']}")
    print("  -> o conf 1e-6, YOLO11n co du doan car khop GT (AP50 ~0,233) tren 84 anh dev smoke; truck/bus AP50 0 (11n: "
          "1 / 0 du doan); motorcycle/bicycle KHONG co GT. Khong suy rong cho moi lop, moi toa do hay toan duong ong.")
    print("\n# KET LUAN DOI CHIEU: da QUAN SAT diem lop cua checkpoint smoke 5 lop thap hon checkpoint COCO 80 lop; "
          "NGUYEN NHAN CHUA XAC DINH (chuyen 80 -> 5 lop/khoi tao va cac buoc hoc -- gom warmup -- chua duoc tach).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
