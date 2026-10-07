# -*- coding: utf-8 -*-
"""Vong 45 (D5): do CHI DOC quan he giua GT UA-DETRAC test va `ignored_region`, de so quy tac BDD hien dung voi toolkit.
    python handoff/logs/review45/phan_tich_vung_bo_qua.py --out <json>

Hai cach tinh IoA cua mot box voi vung bo qua:
  max_don  : max_k  dien_tich(box giao vung_k) / dien_tich(box), hinh hoc so thuc (`eval_detector_5lop.ioa_ma_tran`,
             quy tac da khoa cho BDD: detection KHONG ghep duoc GT ma max_don >= nguong IoU dang xet -> bo qua).
  hop_pixel: ti le pixel cua box nam trong HOP cac vung (raster nhu `DETRAC-MOT-toolkit` dropDetections / parseGT,
             nguong co dinh 0.5).
Dung GT lam DAI DIEN vi tri xe (chua co du doan): dem box theo loai voi max_don / hop_pixel >= 0.5 va >= 0.95, so box
lech giua hai cach o 0.5; so cap vung bo qua chong nhau."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phan_tich_detrac_test import ioa as ioa_hop_pixel, mat_na_ign  # noqa: E402


def ioa_max_don(box, vung: np.ndarray) -> float:
    if not len(vung):
        return 0.0
    l, t, w, h = box
    x1, y1, x2, y2 = l, t, l + w, t + h
    iw = np.clip(np.minimum(x2, vung[:, 2]) - np.maximum(x1, vung[:, 0]), 0, None)
    ih = np.clip(np.minimum(y2, vung[:, 3]) - np.maximum(y1, vung[:, 1]), 0, None)
    return float((iw * ih).max() / max(w * h, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    videos = sorted(json.loads((REPO / "results/phase2/manifests/detrac_test.json").read_text(encoding="utf-8"))["videos"])
    dem = defaultdict(Counter)
    cap_chong = 0
    for v in videos:
        root = ET.parse(REPO / f"data/extracted/DETRAC-Test-Annotations-XML/{v}.xml").getroot()
        ign = [tuple(float(b.get(k)) for k in ("left", "top", "width", "height")) for b in root.findall("ignored_region/box")]
        vung = np.array([[l, t, l + w, t + h] for l, t, w, h in ign]) if ign else np.zeros((0, 4))
        for i in range(len(vung)):
            for j in range(i + 1, len(vung)):
                if min(vung[i, 2], vung[j, 2]) > max(vung[i, 0], vung[j, 0]) and \
                        min(vung[i, 3], vung[j, 3]) > max(vung[i, 1], vung[j, 1]):
                    cap_chong += 1
        m = mat_na_ign(ign, 960, 540)
        for fr in root.findall("frame"):
            for tg in fr.findall("target_list/target"):
                b = tg.find("box")
                box = tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
                loai = tg.find("attribute").get("vehicle_type")
                c = dem[loai]
                c["tong"] += 1
                if not ign:
                    continue
                md, hp = ioa_max_don(box, vung), ioa_hop_pixel(box, m)
                c["max_don_ge_0.5"] += md >= 0.5
                c["max_don_ge_0.95"] += md >= 0.95
                c["max_don_eq_1"] += md >= 1 - 1e-9
                c["hop_pixel_ge_0.5"] += hp >= 0.5
                c["hop_ge_0.5_nhung_max_don_lt_0.5"] += (hp >= 0.5) and (md < 0.5)
                c["max_don_ge_0.5_nhung_hop_lt_0.5"] += (md >= 0.5) and (hp < 0.5)
    ra = {"cap_vung_bo_qua_chong_nhau": cap_chong, "theo_loai": {k: dict(v) for k, v in dem.items()}}
    Path(a.out).write_text(json.dumps(ra, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(ra, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
