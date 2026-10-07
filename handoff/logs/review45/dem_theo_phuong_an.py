# -*- coding: utf-8 -*-
"""Vong 45: dem PHIA GT (chua co du doan, CHI DOC) theo tung phuong an D1 (van) x D2 (others) x D4 (173 anh khong <frame>).
    python handoff/logs/review45/dem_theo_phuong_an.py --out <json>
Moi phuong an: so GT theo lop danh gia (car / bus / truck), so anh co lop, so anh KHONG con GT hop le (chi con vung bo
qua hoac khong co gi), so vung bo qua (ignored_region + box loai bi chuyen thanh vung bo qua). Box chi doi vai tro,
KHONG doi toa do; cat vao khung 960x540 khong ap dung o day (chi dem)."""
from __future__ import annotations

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PA_VAN = {"bo_qua": None, "car": "car"}
PA_OTHERS = {"bo_qua": None, "truck": "truck"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    videos = sorted(json.loads((REPO / "results/phase2/manifests/detrac_test.json").read_text(encoding="utf-8"))["videos"])
    anh = []          # (so ignored_region, Counter loai) cho moi anh co <frame>
    n_khong_frame = 0
    for v in videos:
        root = ET.parse(REPO / f"data/extracted/DETRAC-Test-Annotations-XML/{v}.xml").getroot()
        n_ign = len(root.findall("ignored_region/box"))
        frames = {int(fr.get("num")): Counter(t.find("attribute").get("vehicle_type")
                                              for t in fr.findall("target_list/target")) for fr in root.findall("frame")}
        so_anh = len(list((REPO / f"data/extracted/DETRAC-Images/{v}").glob("img*.jpg")))
        n_khong_frame += so_anh - len(frames)
        anh += [(n_ign, c) for c in frames.values()]
    ra = {}
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
        ten = f"D1_van={pv}|D2_others={po}|D4={'giu_173_am_tinh' if giu else 'loai_173'}"
        n_anh = len(anh) + (n_khong_frame if giu else 0)
        ra[ten] = {"so_anh_cham": n_anh, "gt_theo_lop": dict(gt), "so_anh_co_lop": dict(co_lop),
                   "anh_khong_gt_hop_le": khong_gt + (n_khong_frame if giu else 0),
                   "vung_bo_qua_cong_moi_anh": vung,
                   "lop_bao_AP": sorted(k for k in gt if gt[k]), "lop_NA": sorted({"car", "truck", "bus", "motorcycle",
                                                                                      "bicycle"} - {k for k in gt if gt[k]})}
    Path(a.out).write_text(json.dumps(ra, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in ra.items():
        print(k, json.dumps(v, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
