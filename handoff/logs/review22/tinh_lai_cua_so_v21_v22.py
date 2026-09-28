# -*- coding: utf-8 -*-
"""Dinh chinh N21-F1 cho so vong 21 (KHONG sua timestamp goc) + doi chieu vong 22.

Vong 21: `wall_vong_lap_s` bat dau SAU khi con da doc/infer frame dau -> FPS hoi cao. Tinh lai TU TIMESTAMP DA CO trong
pipeline CSV mot pham vi xac dinh ro: tu `t_bat_dau_con` cua frame 1 (truoc khi doc) den `t_xong_cha` cua frame cuoi
(xong tracker), monotonic chung cua may. Pham vi nay KHONG gom nhan EOS / ghi ket qua. Vong 22 in them `xu_ly_s` (cua so
BAT DAU -> track_video tra ve) de doi chieu.

    python handoff/logs/review22/tinh_lai_cua_so_v21_v22.py
"""
import csv
import json
from pathlib import Path

G = Path(__file__).resolve().parent
NGUON = {"vong21": G.parent / "review21/nano_live_v7/results_nano/live_v7/bench",
         "vong22": G / "nano_live_v8/results_nano/live_v8/bench"}


def doc(p):
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


print("# vong | luot | cau hinh | n | FPS bao cao (dinh nghia cua vong) | FPS frame1-doc -> frameN-xong | thieu (ms) | cua so v22")
for vong, d in NGUON.items():
    for luot in sorted(p for p in d.iterdir() if p.is_dir()):
        m = json.loads((luot / "run_meta.json").read_text(encoding="utf-8"))
        rows = doc(luot / "pipeline" / "MVI_39031.csv")
        dau = min(float(r["t_bat_dau_con"]) for r in rows)
        cuoi = max(float(r["t_xong_cha"]) for r in rows)
        bp = cuoi - dau
        th = m["thoi_gian"]
        if vong == "vong21":
            w, fps, ten = th["wall_vong_lap_s"], th["fps_vong_lap"], "fps_vong_lap"
            thieu = (bp - w) * 1e3
            cs = "-"
        else:
            w, fps, ten = th["xu_ly_s"], th["fps_xu_ly"], "fps_xu_ly"
            thieu = (bp - w) * 1e3
            cs = f"moi_frame_trong_cua_so={th['kiem_cua_so_do']['moi_frame_trong_cua_so']}"
        print(f"{vong} | {luot.name} | {m['cau_hinh']['tracker_yaml']} | {len(rows)} | {ten} {fps:.4f} | "
              f"{len(rows) / bp:.4f} | {thieu:+.3f} | {cs}")
print("# 'thieu' > 0: khoang bao phu dai hon cua so da dung de tinh FPS (vong 21 = N21-F1);"
      " vong 22 < 0: cua so con gom nhan EOS sau frame cuoi")
