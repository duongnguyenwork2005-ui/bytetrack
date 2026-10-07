# -*- coding: utf-8 -*-
"""Vong 45, ban SUA vong 46 (R45-F3): dem PHIA GT (chua co du doan, CHI DOC) theo tung phuong an D1 (van) x D2 (others)
x D4 (173 anh khong <frame>).
    python handoff/logs/review46/dem_theo_phuong_an.py --out <json> [--out-csv <csv bang chung theo video>]

Moi phuong an: so GT theo lop danh gia (car / bus / truck), so anh co lop, so anh KHONG con GT hop le (chi con vung bo
qua hoac khong co gi), va `vung_bo_qua_cong_moi_anh`.

`vung_bo_qua_cong_moi_anh` la SO LUOT VUNG AP DUNG TREN ANH (dem lap qua anh), KHONG phai so vung hinh hoc duy nhat:
  moi anh duoc cham dong gop (so `ignored_region` cap sequence cua video) + (so box van / others bi chuyen thanh vung bo
  qua tren anh do, neu D1 / D2 = bo qua).
Sua R45-F3: ban vong 45 KHONG cong `ignored_region` cap sequence cho cac anh khong co <frame> duoc GIU (D4 = giu) ->
thieu 265 luot. Ban nay cong du; `--out-csv` ghi dong gop theo video (anh giu them x vung tinh moi anh) va danh sach
frame. Box chi doi vai tro, KHONG doi toa do; khong cat khung / loai box nho (so GT o day la box XML goc, CHUA phai so
GT sau buoc cat khung cua bo doc GT du kien)."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PA_VAN = {"bo_qua": None, "car": "car"}
PA_OTHERS = {"bo_qua": None, "truck": "truck"}
DON_VI = ("so LUOT vung ap dung tren anh (dem lap qua anh): ignored_region cap sequence x so anh cham + box van / others "
          "chuyen thanh vung bo qua; KHONG phai so vung hinh hoc duy nhat")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-csv")
    a = ap.parse_args()
    videos = sorted(json.loads((REPO / "results/phase2/manifests/detrac_test.json").read_text(encoding="utf-8"))["videos"])
    anh = []              # (so ignored_region, Counter loai) cho moi anh CO <frame>
    khong_frame = []      # (video, frame, so ignored_region) cho moi anh KHONG co <frame>
    so_vung_hinh_hoc = 0
    for v in videos:
        root = ET.parse(REPO / f"data/extracted/DETRAC-Test-Annotations-XML/{v}.xml").getroot()
        n_ign = len(root.findall("ignored_region/box"))
        so_vung_hinh_hoc += n_ign
        frames = {int(fr.get("num")): Counter(t.find("attribute").get("vehicle_type")
                                              for t in fr.findall("target_list/target")) for fr in root.findall("frame")}
        so_anh = sorted(int(p.stem[3:]) for p in (REPO / f"data/extracted/DETRAC-Images/{v}").glob("img*.jpg"))
        khong_frame += [(v, f, n_ign) for f in so_anh if f not in frames]
        anh += [(n_ign, c) for c in frames.values()]
    vung_khong_frame = sum(n for _, _, n in khong_frame)
    ra = {"_don_vi_vung_bo_qua_cong_moi_anh": DON_VI, "_so_ignored_region_hinh_hoc": so_vung_hinh_hoc,
          "_anh_khong_frame": len(khong_frame), "_luot_ignored_region_tren_anh_khong_frame": vung_khong_frame}
    for (pv, mv), (po, mo), giu in itertools.product(PA_VAN.items(), PA_OTHERS.items(), (False, True)):
        gt, co_lop, khong_gt, vung = Counter(), Counter(), 0, 0
        for n_ign, c in anh:
            cc = Counter({"car": c["car"], "bus": c["bus"]})
            if mv:
                cc[mv] += c["van"]
            if mo:
                cc[mo] += c["others"]
            gt.update(cc)
            co_lop.update(k for k, x in cc.items() if x)
            khong_gt += sum(cc.values()) == 0
            vung += n_ign + (0 if mv else c["van"]) + (0 if mo else c["others"])
        if giu:
            vung += vung_khong_frame          # R45-F3: anh giu them van chiu ignored_region cap sequence
        ten = f"D1_van={pv}|D2_others={po}|D4={'giu_173_am_tinh' if giu else 'loai_173'}"
        ra[ten] = {"so_anh_cham": len(anh) + (len(khong_frame) if giu else 0), "gt_theo_lop": dict(gt),
                   "so_anh_co_lop": dict(co_lop), "anh_khong_gt_hop_le": khong_gt + (len(khong_frame) if giu else 0),
                   "vung_bo_qua_cong_moi_anh": vung,
                   "lop_bao_AP": sorted(k for k in gt if gt[k]), "lop_NA": sorted({"car", "truck", "bus", "motorcycle",
                                                                                      "bicycle"} - {k for k in gt if gt[k]})}
    Path(a.out).write_text(json.dumps(ra, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.out_csv:
        theo_video = Counter(v for v, _, _ in khong_frame)
        vung_tinh = {v: n for v, _, n in khong_frame}
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "anh_khong_frame", "frame", "ignored_region_moi_anh", "luot_vung"])
            for v in sorted(theo_video):
                fr = [str(x) for vv, x, _ in khong_frame if vv == v]
                w.writerow([v, theo_video[v], " ".join(fr), vung_tinh[v], theo_video[v] * vung_tinh[v]])
            w.writerow(["TONG", len(khong_frame), "", "", vung_khong_frame])
    for k, v in ra.items():
        print(k, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
