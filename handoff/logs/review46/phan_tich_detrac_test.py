# -*- coding: utf-8 -*-
"""Vong 45: phan tich CHI DOC UA-DETRAC test (40 video manifest khoa) de chuan bi quyet dinh D1-D5. Khong suy luan,
khong GPU, khong sua dataset; chi ghi vao --out (JSON/CSV nho) va --anh-out (anh xem tay, KHONG commit).
Ban vong 46 (sua R45-F4): `doc_xml()` doc dung `sequence_attribute` (truoc do bieu thuc `node or Element` lam mat thuoc
tinh cua phan tu tu dong); cot camera_state / weather ghi ro "khong co phan tu" / "khong co thuoc tinh" thay vi de trong.
Ban vong 45 giu nguyen o logs/review45/ (bang chung lich su).

    python handoff/logs/review46/phan_tich_detrac_test.py --out handoff/logs/review46/phan_tich \
        [--anh-out <thu muc ngoai repo>]

Do:
  1. Moi video: sha256 XML, so <frame>, dai `num`, `density` == so <target>?, frame co target_list rong, so anh tren
     dia, anh KHONG co <frame> (vi tri: dau / cuoi / khe giua video) va <frame> khong co anh.
  2. vehicle_type: dem box, so video, so track; kich thuoc box (trung vi rong / cao / dien tich), truncation > 0.
  3. ignored_region: so vung / video, ti le dien tich anh bi phu (HOP cac vung, raster pixel nhu toolkit DETRAC),
     so GT co IoA(GT, hop vung) >= 0.5 (nguong toolkit `parseGT.m` / `dropDetections.m`) theo loai; doi chieu vung trong
     XML voi file `DETRAC-MOT-toolkit/evaluation/igrs/<video>_IgR.txt`.
  4. Kich thuoc anh (header JPEG dau / cuoi moi video).
  5. Anh khong co <frame>: so target o frame CO nhan gan nhat truoc / sau (bang chung gian tiep anh co xe hay khong).
  6. --anh-out: bang ghep anh (contact sheet) cho (a) mau ngau nhien seed 0 cua box van / others / car / bus,
     (b) moi anh khong co <frame> (thu nho). Danh sach mau ghi vao CSV de tai lap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
XML_DIR = REPO / "data/extracted/DETRAC-Test-Annotations-XML"
IMG_DIR = REPO / "data/extracted/DETRAC-Images"
IGR_DIR = REPO / "data/extracted/DETRAC-MOT-toolkit/evaluation/igrs"
MANIFEST = REPO / "results/phase2/manifests/detrac_test.json"
NGUONG_IOA = 0.5


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kich_thuoc_jpeg(p: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def doc_xml(p: Path) -> dict:
    root = ET.parse(p).getroot()
    ign = [tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
           for b in root.findall("ignored_region/box")]
    frames = {}
    for fr in root.findall("frame"):
        num = int(fr.get("num"))
        targets = []
        for t in fr.findall("target_list/target"):
            b, at = t.find("box"), t.find("attribute")
            targets.append({"id": int(t.get("id")), "box": tuple(float(b.get(k)) for k in ("left", "top", "width", "height")),
                            "type": at.get("vehicle_type"), "trunc": float(at.get("truncation_ratio") or 0)})
        frames[num] = {"density": int(fr.get("density")) if fr.get("density") is not None else None, "targets": targets}
    # R45-F4 (vong 46): KHONG dung `node or ...` -- Element khong co phan tu con co truthiness False du CO thuoc tinh.
    node = root.find("sequence_attribute")
    return {"ign": ign, "frames": frames, "attr": dict(node.attrib) if node is not None else {},
            "co_sequence_attribute": node is not None}


THIEU_PHAN_TU = "(XML khong co phan tu sequence_attribute)"
THIEU_THUOC_TINH = "(phan tu sequence_attribute khong co thuoc tinh nay)"


def thuoc_tinh_xml(d: dict, ten: str) -> str:
    """Gia tri thuoc tinh dung nhu XML; thieu thi ghi RO loai thieu, KHONG dien gia tri suy doan."""
    if not d["co_sequence_attribute"]:
        return THIEU_PHAN_TU
    return d["attr"].get(ten, THIEU_THUOC_TINH)


def mat_na_ign(ign, w: int, h: int) -> np.ndarray:
    """Raster vung bo qua nhu toolkit: igrMap(top:min(H,top+h), left:min(W,left+w)) = 1 (MATLAB 1-based, ca hai dau)."""
    m = np.zeros((h, w), bool)
    for left, top, rw, rh in ign:
        x1, y1 = max(0, int(round(left)) - 1), max(0, int(round(top)) - 1)
        x2, y2 = min(w, int(round(left + rw))), min(h, int(round(top + rh)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
    return m


def ioa(box, m: np.ndarray) -> float:
    left, top, bw, bh = box
    h, w = m.shape
    x1, y1 = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
    x2, y2 = min(w, int(np.ceil(left + bw))), min(h, int(np.ceil(top + bh)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(m[y1:y2, x1:x2].mean())


def doan(ds: list[int]) -> list[tuple[int, int]]:
    if not ds:
        return []
    ra, a = [], ds[0]
    for x, y in zip(ds, ds[1:] + [None]):
        if y != x + 1:
            ra.append((a, x))
            a = y
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anh-out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
    videos = sorted(mf["videos"])
    tong = {"manifest": str(MANIFEST.relative_to(REPO)), "manifest_sha256": sha_file(MANIFEST), "so_video": len(videos)}
    dem_loai, dem_video_loai, track_loai = Counter(), defaultdict(set), defaultdict(set)
    kt = defaultdict(lambda: {"w": [], "h": [], "a": [], "trunc": 0})
    ign_gt = Counter()
    ign_gt_video = defaultdict(Counter)
    hang_video, anh_thieu, mau = [], [], defaultdict(list)
    n_frame = n_anh = n_box = n_ign = 0
    density_lech = rong = 0
    kich_thuoc = Counter()
    igr_khop, igr_lech = 0, []
    for v in videos:
        px = XML_DIR / f"{v}.xml"
        d = doc_xml(px)
        anh = sorted(IMG_DIR.joinpath(v).glob("img*.jpg"))
        so_anh = {int(p.stem[3:]): p for p in anh}
        w, h = kich_thuoc_jpeg(anh[0])
        w2, h2 = kich_thuoc_jpeg(anh[-1])
        kich_thuoc[(w, h)] += 1
        kich_thuoc[(w2, h2)] += 0
        m = mat_na_ign(d["ign"], w, h)
        frames = d["frames"]
        nums = sorted(frames)
        thieu = sorted(set(so_anh) - set(frames))
        thua = sorted(set(frames) - set(so_anh))
        for num, fr in frames.items():
            if fr["density"] is not None and fr["density"] != len(fr["targets"]):
                density_lech += 1
            if not fr["targets"]:
                rong += 1
            for t in fr["targets"]:
                dem_loai[t["type"]] += 1
                dem_video_loai[t["type"]].add(v)
                track_loai[t["type"]].add((v, t["id"]))
                _, _, bw, bh = t["box"]
                k = kt[t["type"]]
                k["w"].append(bw)
                k["h"].append(bh)
                k["a"].append(bw * bh)
                k["trunc"] += t["trunc"] > 0
                if d["ign"] and ioa(t["box"], m) >= NGUONG_IOA:
                    ign_gt[t["type"]] += 1
                    ign_gt_video[v][t["type"]] += 1
                mau[t["type"]].append((v, num, t["id"], t["box"]))
        # file igrs cua toolkit
        pf = IGR_DIR / f"{v}_IgR.txt"
        if pf.is_file():
            txt = [tuple(float(x) for x in ln.split(",")) for ln in pf.read_text().split() if ln.strip()]
            if sorted(txt) == sorted(d["ign"]):
                igr_khop += 1
            else:
                igr_lech.append(v)
        else:
            igr_lech.append(f"{v} (khong co file)")
        # anh thieu <frame>: vi tri + so target o frame co nhan lan can
        for s, e in doan(thieu):
            vi_tri = ("dau" if s == min(so_anh) else "cuoi" if e == max(so_anh) else "giua")
            truoc = max([n for n in nums if n < s], default=None)
            sau = min([n for n in nums if n > e], default=None)
            for f in range(s, e + 1):
                anh_thieu.append({"video": v, "frame": f, "doan": f"{s}-{e}", "do_dai_doan": e - s + 1, "vi_tri": vi_tri,
                                  "frame_co_nhan_truoc": truoc, "so_target_truoc": len(frames[truoc]["targets"]) if truoc else "",
                                  "frame_co_nhan_sau": sau, "so_target_sau": len(frames[sau]["targets"]) if sau else "",
                                  "anh": str(so_anh[f].relative_to(REPO)).replace("\\", "/")})
        hang_video.append({"video": v, "xml_sha256": sha_file(px), "so_frame_xml": len(frames), "num_min": nums[0],
                           "num_max": nums[-1], "so_anh": len(anh), "anh_min": min(so_anh), "anh_max": max(so_anh),
                           "anh_khong_frame": len(thieu), "doan_anh_khong_frame": ";".join(f"{s}-{e}" for s, e in doan(thieu)),
                           "frame_khong_anh": len(thua), "so_box": sum(len(f["targets"]) for f in frames.values()),
                           "so_vung_bo_qua": len(d["ign"]), "ti_le_anh_bi_phu_bo_qua": round(float(m.mean()), 4),
                           "kich_thuoc": f"{w}x{h}", "kich_thuoc_anh_cuoi": f"{w2}x{h2}",
                           "camera_state": thuoc_tinh_xml(d, "camera_state"),
                           "weather": thuoc_tinh_xml(d, "sence_weather")})
        n_frame += len(frames)
        n_anh += len(anh)
        n_box += sum(len(f["targets"]) for f in frames.values())
        n_ign += len(d["ign"])
    loai = {}
    for t, k in kt.items():
        loai[t] = {"so_box": dem_loai[t], "so_track": len(track_loai[t]), "so_video": len(dem_video_loai[t]),
                   "rong_trung_vi": round(statistics.median(k["w"]), 1), "cao_trung_vi": round(statistics.median(k["h"]), 1),
                   "dien_tich_trung_vi": round(statistics.median(k["a"]), 0), "ti_le_bi_cat_khung": round(k["trunc"] / dem_loai[t], 4),
                   "so_box_ioa_ge_0.5_voi_vung_bo_qua": ign_gt[t]}
    tong.update({"so_frame_xml": n_frame, "so_anh": n_anh, "so_anh_khong_frame": len(anh_thieu), "so_box": n_box,
                 "so_vung_bo_qua": n_ign, "frame_density_khac_so_target": density_lech, "frame_target_list_rong": rong,
                 "kich_thuoc_anh_dau": {f"{w}x{h}": c for (w, h), c in kich_thuoc.items() if c},
                 "loai": dict(sorted(loai.items(), key=lambda x: -x[1]["so_box"])),
                 "video_co_gt_ioa_ge_0.5": {v: dict(c) for v, c in ign_gt_video.items()},
                 "igrs_toolkit_khop_xml": igr_khop, "igrs_toolkit_lech": igr_lech,
                 "anh_khong_frame_theo_vi_tri": dict(Counter(x["vi_tri"] for x in anh_thieu)),
                 "anh_khong_frame_theo_video": dict(Counter(x["video"] for x in anh_thieu))})
    with open(out / "theo_video.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(hang_video[0]))
        wr.writeheader()
        wr.writerows(hang_video)
    with open(out / "anh_khong_frame.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(anh_thieu[0]))
        wr.writeheader()
        wr.writerows(anh_thieu)
    rng = random.Random(0)
    chon = {t: rng.sample(mau[t], min(48, len(mau[t]))) for t in ("van", "others", "car", "bus") if t in mau}
    with open(out / "mau_box_xem_tay.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["loai", "stt", "video", "frame", "target_id", "left", "top", "width", "height"])
        for t, ds in chon.items():
            for i, (v, num, tid, b) in enumerate(ds):
                wr.writerow([t, i, v, num, tid, *b])
    (out / "tong_hop.json").write_text(json.dumps(tong, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.anh_out:
        from PIL import Image, ImageDraw
        ao = Path(a.anh_out)
        ao.mkdir(parents=True, exist_ok=True)
        for t, ds in chon.items():
            cs = Image.new("RGB", (8 * 160, 6 * 130), (40, 40, 40))
            dr = ImageDraw.Draw(cs)
            for i, (v, num, tid, (left, top, bw, bh)) in enumerate(ds):
                with Image.open(IMG_DIR / v / f"img{num:05d}.jpg") as im:
                    pad = 0.15
                    cr = im.crop((max(0, left - pad * bw), max(0, top - pad * bh), left + (1 + pad) * bw,
                                  top + (1 + pad) * bh)).convert("RGB")
                cr.thumbnail((156, 112))
                x0, y0 = (i % 8) * 160, (i // 8) * 130
                cs.paste(cr, (x0 + 2, y0 + 2))
                dr.text((x0 + 3, y0 + 115), f"{i} {v[4:]} f{num} {int(bw)}x{int(bh)}", fill=(255, 255, 0))
            cs.save(ao / f"mau_{t}.png")
        n = len(anh_thieu)
        cot = 12
        cs = Image.new("RGB", (cot * 162, ((n + cot - 1) // cot) * 108), (40, 40, 40))
        dr = ImageDraw.Draw(cs)
        for i, x in enumerate(anh_thieu):
            with Image.open(REPO / x["anh"]) as im:
                th = im.convert("RGB").resize((160, 90))
            x0, y0 = (i % cot) * 162, (i // cot) * 108
            cs.paste(th, (x0 + 1, y0 + 1))
            dr.text((x0 + 2, y0 + 92), f"{x['video'][4:]} f{x['frame']}", fill=(255, 255, 0))
        cs.save(ao / "anh_khong_frame.png")
    print(json.dumps({k: v for k, v in tong.items() if k not in ("video_co_gt_ioa_ge_0.5",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
