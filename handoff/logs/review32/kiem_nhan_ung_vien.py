#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 32 -- kiem nhan MOT 2020 UNG VIEN (Scalabel box_track_20) dang CACH LY. Chi doc, khong sua/chep gi.

1. JSON + schema Scalabel tung file (train, val): moi file = list frame {name, videoName, frameIndex, labels}; moi nhan
   {id, category, box2d{x1,y1,x2,y2}, attributes{occluded,truncated,crowd}}; videoName == ten file; name ==
   <video>-<7 so>.jpg va frameIndex == so - 1; box hop le (x1<=x2, y1<=y2, trong anh 1280x720 co dung sai).
2. Doi chieu voi anh DA XAC MINH: video co anh ma khong co nhan (va nguoc lai); frame trong nhan khong co anh; anh khong
   co frame trong nhan. Video thieu nhan KHONG duoc coi la anh am tinh -- chi bao cao.
3. Val: doi chieu tung nhan voi ban COCO-video cua du an, ghep theo (file anh, scalabel_id): category (ghi bang anh xa),
   crowd/iscrowd, bbox (thu ca hai quy uoc w = x2-x1 va w = x2-x1+1). Nhan chi co o mot ben duoc dem rieng.
Ghi JSON tong hop vao --ra. Ma thoat: 0 khi khong co loi schema/JSON va doi chieu anh day du; 1 neu co bat thuong.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

W, H, DUNG_SAI = 1280, 720, 2.0
MAU_TEN = re.compile(r"^(?P<v>.+)-(?P<so>\d{7})\.jpg$")


def kiem_file(p, dem, giu_nhan):
    """Doc 1 file JSON nhan; tra (video, {ten_frame: [nhan gon] hoac so nhan}) va cong loi vao dem.

    giu_nhan=False (train): chi giu SO nhan moi frame de doi chieu anh -- khong giu ca 1 GB JSON trong RAM (may 7,8 GB).
    giu_nhan=True (val): giu bo gon (id, category, crowd, x1, y1, x2, y2) cho doi chieu COCO.
    """
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        dem["json_hong"].append(f"{p.name}: {e!r}"[:200])
        return None, {}
    if not isinstance(d, list):
        dem["khong_phai_list"].append(p.name)
        return None, {}
    video = p.stem
    frames = {}
    ids = set()
    for fr in d:
        if not isinstance(fr, dict) or not isinstance(fr.get("name"), str):
            dem["frame_sai_dang"] += 1
            continue
        if fr.get("videoName") != video:
            dem["videoName_khac_ten_file"] += 1
        m = MAU_TEN.match(fr["name"])
        if not m or m.group("v") != video:
            dem["ten_frame_sai_mau"] += 1
        elif fr.get("frameIndex") != int(m.group("so")) - 1:
            dem["frameIndex_lech_ten"] += 1
        if fr["name"] in frames:
            dem["frame_trung_ten"] += 1
        nhan = fr.get("labels") or []
        if not isinstance(nhan, list):
            dem["labels_khong_phai_list"] += 1
            nhan = []
        tot = []
        for n in nhan:
            b = n.get("box2d") if isinstance(n, dict) else None
            at = n.get("attributes") if isinstance(n, dict) else None
            if not (isinstance(n, dict) and isinstance(n.get("id"), str) and isinstance(n.get("category"), str)
                    and isinstance(b, dict) and all(isinstance(b.get(k), (int, float)) for k in ("x1", "y1", "x2", "y2"))):
                dem["nhan_sai_dang"] += 1
                continue
            if not isinstance(at, dict) or not all(isinstance(at.get(k), bool) for k in ("occluded", "truncated", "crowd")):
                dem["attributes_thieu_hoac_sai_kieu"] += 1
                at = at if isinstance(at, dict) else {}
            if b["x1"] > b["x2"] or b["y1"] > b["y2"]:
                dem["box_dao"] += 1
            if b["x1"] < -DUNG_SAI or b["y1"] < -DUNG_SAI or b["x2"] > W + DUNG_SAI or b["y2"] > H + DUNG_SAI:
                dem["box_ra_ngoai_anh"] += 1
            dem["so_nhan"] += 1
            dem["theo_lop"][n["category"]] += 1
            if at.get("crowd"):
                dem["crowd_theo_lop"][n["category"]] += 1
            tot.append((n["id"], n["category"], bool(at.get("crowd")), b["x1"], b["y1"], b["x2"], b["y2"]))
        frames[fr["name"]] = tot if giu_nhan else len(tot)
        dem["so_frame"] += 1
        if not tot:
            dem["so_frame_khong_nhan"] += 1
        ids.update(t[0] for t in tot)
    dem["so_track_id"] += len(ids)
    return video, frames


def kiem_split(thu_muc_nhan, thu_muc_anh, giu_nhan=False):
    dem = collections.defaultdict(int)
    for k in ("json_hong", "khong_phai_list"):
        dem[k] = []
    dem["theo_lop"] = collections.Counter()
    dem["crowd_theo_lop"] = collections.Counter()
    files = sorted(Path(thu_muc_nhan).glob("*.json"))
    tat_ca = {}
    for p in files:
        v, fr = kiem_file(p, dem, giu_nhan)
        if v is not None:
            tat_ca[v] = fr
    anh = {d.name: {x.name for x in d.iterdir() if x.is_file()} for d in Path(thu_muc_anh).iterdir() if d.is_dir()}
    co_nhan, co_anh = set(tat_ca), set(anh)
    frame_khong_anh, anh_khong_frame = [], []
    for v in sorted(co_nhan & co_anh):
        frame_khong_anh += [f"{v}/{x}" for x in sorted(set(tat_ca[v]) - anh[v])]
        anh_khong_frame += [f"{v}/{x}" for x in sorted(anh[v] - set(tat_ca[v]))]
    kq = {
        "thu_muc_nhan": Path(thu_muc_nhan).as_posix(), "thu_muc_anh": Path(thu_muc_anh).as_posix(),
        "so_file_json": len(files), "so_file_doc_duoc": len(tat_ca),
        "so_video_anh": len(co_anh), "video_co_anh_khong_co_nhan": sorted(co_anh - co_nhan),
        "video_co_nhan_khong_co_anh": sorted(co_nhan - co_anh),
        "so_frame_nhan_khong_co_anh": len(frame_khong_anh), "vi_du_frame_nhan_khong_co_anh": frame_khong_anh[:10],
        "so_anh_khong_co_frame_nhan": len(anh_khong_frame), "vi_du_anh_khong_co_frame_nhan": anh_khong_frame[:10],
        "so_anh": sum(len(x) for x in anh.values()),
        "so_frame": dem["so_frame"], "so_frame_khong_nhan": dem["so_frame_khong_nhan"], "so_nhan": dem["so_nhan"],
        "so_track_id_tong_cac_video": dem["so_track_id"],
        "theo_lop": dict(dem["theo_lop"].most_common()), "crowd_theo_lop": dict(dem["crowd_theo_lop"].most_common()),
        "loi_schema": {k: dem[k] for k in ("frame_sai_dang", "videoName_khac_ten_file", "ten_frame_sai_mau", "frameIndex_lech_ten",
                                           "frame_trung_ten", "labels_khong_phai_list", "nhan_sai_dang",
                                           "attributes_thieu_hoac_sai_kieu", "box_dao", "box_ra_ngoai_anh")},
        "json_hong": dem["json_hong"][:10], "khong_phai_list": dem["khong_phai_list"][:10],
    }
    return kq, tat_ca


def doi_chieu_coco(tat_ca_val, coco_path):
    c = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    ten_lop = {x["id"]: x["name"] for x in c["categories"]}
    ten_anh = {x["id"]: x["file_name"] for x in c["images"]}
    coco = {}
    for a in c["annotations"]:
        coco[(ten_anh[a["image_id"]], a["scalabel_id"])] = a
    anh_xa_lop = collections.Counter()
    crowd_khac = 0
    lech = {"w=x2-x1": 0.0, "w=x2-x1+1": 0.0}
    khop_box = {"w=x2-x1": 0, "w=x2-x1+1": 0}
    da_ghep = set()
    chi_scalabel = collections.Counter()
    id_map = collections.defaultdict(dict)
    id_bat_nhat_quan = 0
    for v, frames in tat_ca_val.items():
        for ten, nhan in frames.items():
            for nid, cat, crowd, x1, y1, x2, y2 in nhan:
                k = (f"{v}/{ten}", nid)
                a = coco.get(k)
                if a is None:
                    chi_scalabel[cat] += 1
                    continue
                da_ghep.add(k)
                anh_xa_lop[(cat, ten_lop[a["category_id"]], int(a.get("ignore", 0)))] += 1
                if crowd != bool(a.get("iscrowd")):
                    crowd_khac += 1
                x, y, w, h = a["bbox"]
                for qu, cong in (("w=x2-x1", 0.0), ("w=x2-x1+1", 1.0)):
                    d = max(abs(x - x1), abs(y - y1), abs(w - (x2 - x1 + cong)), abs(h - (y2 - y1 + cong)))
                    lech[qu] = max(lech[qu], d)
                    khop_box[qu] += d < 1e-6
                cu = id_map[v].setdefault(nid, a["instance_id"])
                if cu != a["instance_id"]:
                    id_bat_nhat_quan += 1
    chi_coco = collections.Counter(ten_lop[a["category_id"]] for k, a in coco.items() if k not in da_ghep)
    return {
        "coco": Path(coco_path).as_posix(), "so_annotation_coco": len(c["annotations"]), "so_anh_coco": len(c["images"]),
        "so_video_coco": len(c["videos"]), "so_cap_ghep_duoc": len(da_ghep),
        "nhan_chi_co_o_scalabel_theo_lop": dict(chi_scalabel.most_common()),
        "annotation_chi_co_o_coco_theo_lop": dict(chi_coco.most_common()),
        "anh_xa_lop_scalabel_coco_ignore": {f"{s} -> {t} (ignore={i})": n for (s, t, i), n in anh_xa_lop.most_common()},
        "so_crowd_khac_iscrowd": crowd_khac, "so_box_khop_tuyet_doi": khop_box, "lech_box_lon_nhat": lech,
        "so_track_id_anh_xa_bat_nhat_quan": id_bat_nhat_quan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nhan", required=True, help="thu muc chua train/ va val/ (box_track_20) dang cach ly")
    ap.add_argument("--anh", default="data/raw/bdd100k/images/track")
    ap.add_argument("--coco-val", default="data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    kq = {}
    tat_ca_val = {}
    for split in ("train", "val"):
        if (Path(a.nhan) / split).is_dir():
            kq[split], tat_ca = kiem_split(Path(a.nhan) / split, Path(a.anh) / split, giu_nhan=(split == "val"))
            if split == "val":
                tat_ca_val = tat_ca
    if tat_ca_val and Path(a.coco_val).exists():
        kq["doi_chieu_val_coco"] = doi_chieu_coco(tat_ca_val, a.coco_val)
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    bat_thuong = False
    for split in ("train", "val"):
        s = kq.get(split)
        if s is None:
            continue
        tom = {k: v for k, v in s.items() if not k.startswith("vi_du") and k not in ("theo_lop", "crowd_theo_lop")}
        tom["video_co_anh_khong_co_nhan"] = len(s["video_co_anh_khong_co_nhan"])
        tom["video_co_nhan_khong_co_anh"] = len(s["video_co_nhan_khong_co_anh"])
        print(split, json.dumps(tom, ensure_ascii=True))
        bat_thuong |= bool(s["json_hong"] or s["khong_phai_list"] or any(s["loi_schema"].values())
                           or s["video_co_anh_khong_co_nhan"] or s["video_co_nhan_khong_co_anh"]
                           or s["so_frame_nhan_khong_co_anh"] or s["so_anh_khong_co_frame_nhan"])
    if "doi_chieu_val_coco" in kq:
        print("doi_chieu_val_coco", json.dumps(kq["doi_chieu_val_coco"], ensure_ascii=True))
    return 1 if bat_thuong else 0


if __name__ == "__main__":
    sys.exit(main())
