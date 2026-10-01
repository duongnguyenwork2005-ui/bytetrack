#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kiem nhan MOT 2020 UNG VIEN (Scalabel box_track_20) dang CACH LY. Chi doc, khong sua/chep gi.

Vong 32 tao; vong 33 sua theo R32-F2 (review c4d64a1): khong bao dat khi thieu dau vao, kiem kieu/so huu han/ID trung,
song anh ID hai chieu, trang thai doi chieu COCO tach rieng.

PHAM VI (chon ro):
  --split train val     split BAT BUOC kiem (mac dinh ca hai). Split khong chon -> NOT_RUN trong bao cao.
  --coco-val F          file COCO-video val de doi chieu (mac dinh file cua du an). Doi chieu chay khi val nam trong
                        --split; file duoc yeu cau ma thieu/hong -> loi dau vao. --khong-doi-chieu-coco -> NOT_RUN.

1. Schema Scalabel tung file: list frame {name, videoName, frameIndex, labels}. labels: list (ke ca []) la hop le; null
   hoac thieu khoa duoc chap nhan theo dac ta Scalabel nhung DEM RIENG; kieu khac (dict, chuoi, so) la LOI. Moi nhan:
   id (chuoi khong rong, KHONG trung trong frame), category (chuoi khong rong), box2d{x1,y1,x2,y2} la so int/float HUU HAN
   (khong NaN/Inf, khong boolean), x1<=x2, y1<=y2, trong anh 1280x720 co dung sai; attributes{occluded,truncated,crowd}
   kieu bool. videoName == ten file; name == <video>-<7 so>.jpg, frameIndex == so - 1; frame trung ten bi bat TRUOC khi
   dua vao dictionary.
2. Do day du so voi anh DA XAC MINH: video co anh ma khong co nhan (va nguoc lai); frame nhan khong co anh; anh khong co
   frame nhan. Video thieu nhan KHONG duoc coi la anh am tinh -- chi bao cao.
3. Doi chieu val voi COCO-video, ghep theo (file anh, scalabel_id); khoa COCO trung bi bat TRUOC khi dua vao dictionary.
   Box theo quy uoc Scalabel w = x2-x1+1, h = y2-y1+1. Song anh ID HAI CHIEU trong tung video (khong gop ID giua cac
   video). Lop: lech neu COCO khac anh xa {other vehicle->car, other person->pedestrian, trailer->truck, con lai giu
   nguyen}. Ignore: ngoai le neu (native thuoc 3 lop tren) != (COCO ignore). iscrowd: ngoai le neu iscrowd != (crowd hoac
   ignore). Hai phep lop/ignore la HAI phep kiem doc lap -- khong cong so hang thanh so annotation khac nhau.
   Ghi tung hang lech lop / ngoai le ignore ra CSV (cung cot voi bang chung reviewer).

TRANG THAI (trong JSON --ra, khoa "trang_thai"):
  dau_vao: DAT | LOI; schema_va_do_day_du: train/val/tong = DAT | LOI | NOT_RUN;
  doi_chieu_coco: MATCH | DIFFERENCES | NOT_RUN (kem ly do) | LOI_DAU_VAO.
MA THOAT:
  0 = moi split duoc chon DAT va doi chieu COCO MATCH (hoac NOT_RUN do chon ro / val khong trong pham vi);
  1 = co loi schema hoac thieu/thua anh-nhan;
  2 = thieu hoac sai dau vao (root, split bat buoc, bo nhan rong, thu muc anh, file COCO duoc yeu cau) -- chua kiem duoc;
  3 = schema DAT nhung doi chieu COCO co KHAC BIET (chi bao cao). Exit 0 KHONG bao gio co nghia "co khac biet".
"""
import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

W, H, DUNG_SAI = 1280, 720, 2.0
MAU_TEN = re.compile(r"^(?P<v>.+)-(?P<so>\d{7})\.jpg$")
ANH_XA = {"other vehicle": "car", "other person": "pedestrian", "trailer": "truck"}
COT_HANG = ["video", "frame", "scalabel_id", "instance_id", "native_class", "coco_class", "native_crowd", "coco_iscrowd",
            "native_ignore", "coco_ignore"]
KHOA_LOI = ("file_khong_co_frame", "frame_sai_dang", "videoName_khac_ten_file", "ten_frame_sai_mau", "frameIndex_lech_ten",
            "frame_trung_ten", "labels_khong_phai_list", "nhan_sai_dang", "id_sai_kieu", "category_sai_kieu",
            "box_thieu_hoac_sai_kieu", "box_toa_do_bool", "box_khong_huu_han", "id_trung_trong_frame",
            "attributes_thieu_hoac_sai_kieu", "box_dao", "box_ra_ngoai_anh")


class LoiDauVao(Exception):
    pass


def so_huu_han(v):
    """(hop_le, ly_do): int/float that, khong bool, huu han."""
    if isinstance(v, bool):
        return False, "box_toa_do_bool"
    if type(v) not in (int, float):
        return False, "box_thieu_hoac_sai_kieu"
    if not math.isfinite(v):
        return False, "box_khong_huu_han"
    return True, None


def kiem_file(p, dem, giu_nhan):
    """Doc 1 file JSON nhan; tra (video, {ten_frame: [nhan gon] hoac so nhan}) va cong loi vao dem.

    giu_nhan=False (train): chi giu SO nhan moi frame -- khong giu ca 1 GB JSON trong RAM (may 7,8 GB).
    giu_nhan=True (val): giu bo gon (id, category, crowd, ignored, x1, y1, x2, y2) cho doi chieu COCO.
    Nhan sai schema KHONG duoc dua vao thong ke/doi chieu, chi dem loi.
    """
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        dem["json_hong"].append(f"{p.name}: {e!r}"[:200])
        return None, {}
    if not isinstance(d, list):
        dem["khong_phai_list"].append(p.name)
        return None, {}
    if not d:
        dem["file_khong_co_frame"] += 1
    video = p.stem
    frames = {}
    ids_video = collections.defaultdict(set)
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
        if fr["name"] in frames:  # bat TRUOC khi dua vao dictionary; giu ban dau, khong ghi de
            dem["frame_trung_ten"] += 1
            continue
        if "labels" not in fr:
            dem["frame_thieu_khoa_labels"] += 1
            nhan = []
        elif fr["labels"] is None:
            dem["frame_labels_null"] += 1
            nhan = []
        elif isinstance(fr["labels"], list):
            nhan = fr["labels"]
        else:
            dem["labels_khong_phai_list"] += 1  # vd {} -- KHONG bien thanh frame rong
            frames[fr["name"]] = [] if giu_nhan else 0
            dem["so_frame"] += 1
            continue
        tot = []
        id_frame = set()
        for n in nhan:
            if not isinstance(n, dict):
                dem["nhan_sai_dang"] += 1
                continue
            loi = []
            if not isinstance(n.get("id"), str) or not n["id"]:
                loi.append("id_sai_kieu")
            if not isinstance(n.get("category"), str) or not n["category"]:
                loi.append("category_sai_kieu")
            b = n.get("box2d")
            if not isinstance(b, dict):
                loi.append("box_thieu_hoac_sai_kieu")
            else:
                for k in ("x1", "y1", "x2", "y2"):
                    ok, ly_do = so_huu_han(b.get(k))
                    if not ok:
                        loi.append(ly_do)
                        break
            at = n.get("attributes")
            if not isinstance(at, dict) or not all(type(at.get(k)) is bool for k in ("occluded", "truncated", "crowd")):
                loi.append("attributes_thieu_hoac_sai_kieu")
            if isinstance(n.get("id"), str) and n["id"]:
                if n["id"] in id_frame:  # ID trung trong cung frame
                    loi.append("id_trung_trong_frame")
                id_frame.add(n["id"])
            if loi:
                for x in set(loi):
                    dem[x] += 1
                continue
            if b["x1"] > b["x2"] or b["y1"] > b["y2"]:
                dem["box_dao"] += 1
            if b["x1"] < -DUNG_SAI or b["y1"] < -DUNG_SAI or b["x2"] > W + DUNG_SAI or b["y2"] > H + DUNG_SAI:
                dem["box_ra_ngoai_anh"] += 1
            dem["so_nhan"] += 1
            dem["theo_lop"][n["category"]] += 1
            if at["crowd"]:
                dem["crowd_theo_lop"][n["category"]] += 1
            ids_video[n["id"]].add(n["category"])
            tot.append((n["id"], n["category"], at["crowd"], at.get("ignored"), b["x1"], b["y1"], b["x2"], b["y2"]))
        frames[fr["name"]] = tot if giu_nhan else len(tot)
        dem["so_frame"] += 1
        if not tot:
            dem["so_frame_khong_nhan"] += 1
    dem["so_track_id"] += len(ids_video)
    dem["so_track_doi_lop"] += sum(1 for c in ids_video.values() if len(c) > 1)
    return video, frames


def kiem_split(thu_muc_nhan, thu_muc_anh, giu_nhan=False):
    thu_muc_nhan, thu_muc_anh = Path(thu_muc_nhan), Path(thu_muc_anh)
    if not thu_muc_nhan.is_dir():
        raise LoiDauVao(f"thieu thu muc nhan cua split bat buoc: {thu_muc_nhan.as_posix()}")
    files = sorted(thu_muc_nhan.glob("*.json"))
    if not files:
        raise LoiDauVao(f"bo nhan rong (0 file JSON): {thu_muc_nhan.as_posix()}")
    if not thu_muc_anh.is_dir():
        raise LoiDauVao(f"thieu thu muc anh de doi chieu: {thu_muc_anh.as_posix()}")
    dem = collections.defaultdict(int)
    for k in ("json_hong", "khong_phai_list"):
        dem[k] = []
    dem["theo_lop"] = collections.Counter()
    dem["crowd_theo_lop"] = collections.Counter()
    tat_ca = {}
    for p in files:
        v, fr = kiem_file(p, dem, giu_nhan)
        if v is not None:
            tat_ca[v] = fr
    anh = {d.name: {x.name for x in d.iterdir() if x.is_file()} for d in thu_muc_anh.iterdir() if d.is_dir()}
    co_nhan, co_anh = set(tat_ca), set(anh)
    frame_khong_anh, anh_khong_frame = [], []
    for v in sorted(co_nhan & co_anh):
        frame_khong_anh += [f"{v}/{x}" for x in sorted(set(tat_ca[v]) - anh[v])]
        anh_khong_frame += [f"{v}/{x}" for x in sorted(anh[v] - set(tat_ca[v]))]
    loi_schema = {k: dem[k] for k in KHOA_LOI}
    kq = {
        "thu_muc_nhan": thu_muc_nhan.as_posix(), "thu_muc_anh": thu_muc_anh.as_posix(),
        "so_file_json": len(files), "so_file_doc_duoc": len(tat_ca),
        "so_video_anh": len(co_anh), "video_co_anh_khong_co_nhan": sorted(co_anh - co_nhan),
        "video_co_nhan_khong_co_anh": sorted(co_nhan - co_anh),
        "so_frame_nhan_khong_co_anh": len(frame_khong_anh), "vi_du_frame_nhan_khong_co_anh": frame_khong_anh[:10],
        "so_anh_khong_co_frame_nhan": len(anh_khong_frame), "vi_du_anh_khong_co_frame_nhan": anh_khong_frame[:10],
        "so_anh": sum(len(x) for x in anh.values()),
        "so_frame": dem["so_frame"], "so_frame_khong_nhan": dem["so_frame_khong_nhan"],
        "so_frame_labels_null": dem["frame_labels_null"], "so_frame_thieu_khoa_labels": dem["frame_thieu_khoa_labels"],
        "so_nhan": dem["so_nhan"], "so_track_id_tong_cac_video": dem["so_track_id"],
        "so_track_doi_lop_trong_video": dem["so_track_doi_lop"],
        "theo_lop": dict(dem["theo_lop"].most_common()), "crowd_theo_lop": dict(dem["crowd_theo_lop"].most_common()),
        "loi_schema": loi_schema,
        "json_hong": dem["json_hong"][:10], "so_json_hong": len(dem["json_hong"]),
        "khong_phai_list": dem["khong_phai_list"][:10], "so_khong_phai_list": len(dem["khong_phai_list"]),
    }
    loi = (kq["so_json_hong"] or kq["so_khong_phai_list"] or any(loi_schema.values()) or kq["video_co_anh_khong_co_nhan"]
           or kq["video_co_nhan_khong_co_anh"] or kq["so_frame_nhan_khong_co_anh"] or kq["so_anh_khong_co_frame_nhan"])
    kq["trang_thai"] = "LOI" if loi else "DAT"
    return kq, tat_ca


def doc_coco(coco_path):
    p = Path(coco_path)
    if not p.is_file():
        raise LoiDauVao(f"file COCO duoc yeu cau nhung khong ton tai: {p.as_posix()}")
    try:
        c = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LoiDauVao(f"file COCO khong doc duoc: {p.as_posix()}: {e!r}"[:300])
    if not isinstance(c, dict) or not all(isinstance(c.get(k), list) for k in ("categories", "images", "annotations")):
        raise LoiDauVao(f"file COCO thieu categories/images/annotations dang list: {p.as_posix()}")
    return c


def doi_chieu_coco(tat_ca_val, c, coco_path, ra_hang_tien_to):
    ten_lop = {x["id"]: x["name"] for x in c["categories"]}
    ten_anh, anh_trung = {}, 0
    for x in c["images"]:
        if x["id"] in ten_anh:
            anh_trung += 1
            continue
        ten_anh[x["id"]] = x["file_name"]
    coco, khoa_trung, anh_la, thieu_truong = {}, 0, 0, 0
    for a in c["annotations"]:
        if not all(k in a for k in ("image_id", "scalabel_id", "instance_id", "category_id", "bbox")):
            thieu_truong += 1
            continue
        if a["image_id"] not in ten_anh:
            anh_la += 1
            continue
        k = (ten_anh[a["image_id"]], a["scalabel_id"])
        if k in coco:  # khoa trung bi bat TRUOC khi dua vao dictionary; giu ban dau
            khoa_trung += 1
            continue
        coco[k] = a
    if thieu_truong:
        raise LoiDauVao(f"{thieu_truong} annotation COCO thieu truong bat buoc (image_id/scalabel_id/instance_id/category_id/bbox)")
    anh_xa_lop = collections.Counter()
    lech = {"w=x2-x1": 0.0, "w=x2-x1+1": 0.0}
    khop_box = {"w=x2-x1": 0, "w=x2-x1+1": 0}
    da_ghep = set()
    chi_native = collections.Counter()
    xuoi = collections.defaultdict(set)   # (video, native id) -> {instance_id}
    nguoc = collections.defaultdict(set)  # (video, instance_id) -> {native id}
    hang_lop, hang_ignore = [], []
    crowd_khac, iscrowd_ngoai_quy_tac = 0, 0
    for v, frames in tat_ca_val.items():
        for ten, nhan in frames.items():
            for nid, cat, crowd, ignored, x1, y1, x2, y2 in nhan:
                k = (f"{v}/{ten}", nid)
                a = coco.get(k)
                if a is None:
                    chi_native[cat] += 1
                    continue
                da_ghep.add(k)
                c_cat = ten_lop.get(a["category_id"], f"<category_id {a['category_id']}>")
                c_ign, c_crowd = a.get("ignore"), a.get("iscrowd")
                anh_xa_lop[(cat, c_cat, int(bool(c_ign)))] += 1
                hang = dict(zip(COT_HANG, (v, ten, nid, a["instance_id"], cat, c_cat, crowd, c_crowd, ignored, c_ign)))
                if ANH_XA.get(cat, cat) != c_cat:
                    hang_lop.append(hang)
                if (cat in ANH_XA) != bool(c_ign):
                    hang_ignore.append(hang)
                if crowd != bool(c_crowd):
                    crowd_khac += 1
                if bool(c_crowd) != (crowd or bool(c_ign)):
                    iscrowd_ngoai_quy_tac += 1
                x, y, w, h = a["bbox"]
                for qu, cong in (("w=x2-x1", 0.0), ("w=x2-x1+1", 1.0)):
                    d = max(abs(x - x1), abs(y - y1), abs(w - (x2 - x1 + cong)), abs(h - (y2 - y1 + cong)))
                    lech[qu] = max(lech[qu], d)
                    khop_box[qu] += d < 1e-6
                xuoi[(v, nid)].add(a["instance_id"])
                nguoc[(v, a["instance_id"])].add(nid)
    chi_coco = collections.Counter(ten_lop.get(a["category_id"]) for k, a in coco.items() if k not in da_ghep)
    mot_native_nhieu_coco = sum(1 for s in xuoi.values() if len(s) > 1)
    mot_coco_nhieu_native = sum(1 for s in nguoc.values() if len(s) > 1)
    so_box_lech = len(da_ghep) - khop_box["w=x2-x1+1"]
    for duoi, hang in (("lop_khac.csv", hang_lop), ("ngoai_le_ignore.csv", hang_ignore)):
        with open(f"{ra_hang_tien_to}_{duoi}", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=COT_HANG)
            wr.writeheader()
            wr.writerows(hang)
    khac = {
        "nhan_chi_co_o_native": sum(chi_native.values()), "annotation_chi_co_o_coco": sum(chi_coco.values()),
        "box_lech": so_box_lech, "id_mot_native_nhieu_coco": mot_native_nhieu_coco,
        "id_mot_coco_nhieu_native": mot_coco_nhieu_native, "lop_khac_anh_xa": len(hang_lop),
        "ngoai_le_ignore": len(hang_ignore), "iscrowd_ngoai_quy_tac": iscrowd_ngoai_quy_tac,
        "coco_khoa_trung": khoa_trung, "coco_image_id_trung": anh_trung, "coco_tham_chieu_anh_khong_ton_tai": anh_la,
    }
    return {
        "coco": Path(coco_path).as_posix(), "so_annotation_coco": len(c["annotations"]), "so_anh_coco": len(c["images"]),
        "so_video_coco": len(c.get("videos", [])), "so_cap_ghep_duoc": len(da_ghep),
        "nhan_chi_co_o_scalabel_theo_lop": dict(chi_native.most_common()),
        "annotation_chi_co_o_coco_theo_lop": dict(chi_coco.most_common()),
        "anh_xa_lop_scalabel_coco_ignore": {f"{s} -> {t} (ignore={i})": n for (s, t, i), n in anh_xa_lop.most_common()},
        "so_crowd_khac_iscrowd": crowd_khac, "so_box_khop_tuyet_doi": khop_box, "lech_box_lon_nhat": lech,
        "so_track_id_anh_xa_bat_nhat_quan": mot_native_nhieu_coco + mot_coco_nhieu_native,
        "lop_khac": {"so_annotation": len(hang_lop), "so_track": len({(h["video"], h["scalabel_id"]) for h in hang_lop}),
                     "so_video": len({h["video"] for h in hang_lop}),
                     "theo_cap": dict(collections.Counter(f"{h['native_class']}->{h['coco_class']}" for h in hang_lop).most_common())},
        "ngoai_le_ignore": {"so_annotation": len(hang_ignore),
                            "theo_cap": dict(collections.Counter(f"{h['native_class']}->{h['coco_class']} (coco ignore={h['coco_ignore']})"
                                                                 for h in hang_ignore).most_common())},
        "khac_biet": khac,
        "file_hang": [f"{ra_hang_tien_to}_lop_khac.csv", f"{ra_hang_tien_to}_ngoai_le_ignore.csv"],
        "ghi_chu": "lop_khac va ngoai_le_ignore la HAI phep kiem; khong cong so hang thanh so annotation khac nhau",
    }, ("DIFFERENCES" if any(khac.values()) else "MATCH")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nhan", required=True, help="thu muc box_track_20 chua cac split (train/, val/) dang cach ly")
    ap.add_argument("--anh", default="data/raw/bdd100k/images/track")
    ap.add_argument("--split", nargs="+", choices=["train", "val"], default=["train", "val"],
                    help="split BAT BUOC kiem; split khong chon ghi NOT_RUN")
    ap.add_argument("--coco-val", default="data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
    ap.add_argument("--khong-doi-chieu-coco", action="store_true", help="bo qua doi chieu COCO (ghi NOT_RUN)")
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    ra = Path(a.ra)
    kq = {"phien_ban_bo_kiem": "vong33",
          "pham_vi": {"split_bat_buoc": a.split, "coco_val": None if a.khong_doi_chieu_coco else a.coco_val}}
    tt = {"dau_vao": "DAT", "loi_dau_vao": [],
          "schema_va_do_day_du": {s: ("NOT_RUN" if s not in a.split else None) for s in ("train", "val")},
          "doi_chieu_coco": None, "ly_do_doi_chieu_coco": None}
    tat_ca_val = {}
    try:
        if not Path(a.nhan).is_dir():
            raise LoiDauVao(f"thieu thu muc goc nhan: {Path(a.nhan).as_posix()}")
        for split in ("train", "val"):
            if split in a.split:
                kq[split], tat_ca = kiem_split(Path(a.nhan) / split, Path(a.anh) / split, giu_nhan=(split == "val"))
                tt["schema_va_do_day_du"][split] = kq[split]["trang_thai"]
                if split == "val":
                    tat_ca_val = tat_ca
        if "val" not in a.split:
            tt["doi_chieu_coco"], tt["ly_do_doi_chieu_coco"] = "NOT_RUN", "val khong nam trong --split"
        elif a.khong_doi_chieu_coco:
            tt["doi_chieu_coco"], tt["ly_do_doi_chieu_coco"] = "NOT_RUN", "nguoi chay chon --khong-doi-chieu-coco"
        else:
            c = doc_coco(a.coco_val)
            kq["doi_chieu_val_coco"], tt["doi_chieu_coco"] = doi_chieu_coco(tat_ca_val, c, a.coco_val, str(ra.with_suffix("")))
    except LoiDauVao as e:
        tt["dau_vao"] = "LOI"
        tt["loi_dau_vao"].append(str(e))
        if tt["doi_chieu_coco"] is None:
            tt["doi_chieu_coco"] = "LOI_DAU_VAO" if "COCO" in str(e) else "NOT_RUN"
        for s in ("train", "val"):
            if tt["schema_va_do_day_du"][s] is None:
                tt["schema_va_do_day_du"][s] = "NOT_RUN"
    cac = [tt["schema_va_do_day_du"][s] for s in a.split]
    tt["schema_va_do_day_du"]["tong"] = "LOI" if "LOI" in cac else ("NOT_RUN" if "NOT_RUN" in cac else "DAT")
    if tt["dau_vao"] == "LOI":
        ma, dg = 2, "thieu/sai dau vao -- CHUA kiem du pham vi da chon"
    elif tt["schema_va_do_day_du"]["tong"] == "LOI":
        ma, dg = 1, "co loi schema hoac thieu/thua anh-nhan"
    elif tt["doi_chieu_coco"] == "DIFFERENCES":
        ma, dg = 3, "schema va do day du DAT; doi chieu COCO co KHAC BIET (chi bao cao) -- KHONG phai moi nhan khop"
    else:
        ma, dg = 0, ("schema va do day du DAT; doi chieu COCO " + tt["doi_chieu_coco"]
                     + (f" ({tt['ly_do_doi_chieu_coco']})" if tt["ly_do_doi_chieu_coco"] else ""))
    tt["exit"], tt["dien_giai"] = ma, dg
    kq["trang_thai"] = tt
    ra.write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for split in ("train", "val"):
        s = kq.get(split)
        if s is None:
            continue
        tom = {k: v for k, v in s.items() if not k.startswith("vi_du") and k not in ("theo_lop", "crowd_theo_lop", "json_hong", "khong_phai_list")}
        tom["video_co_anh_khong_co_nhan"] = len(s["video_co_anh_khong_co_nhan"])
        tom["video_co_nhan_khong_co_anh"] = len(s["video_co_nhan_khong_co_anh"])
        print(split, json.dumps(tom, ensure_ascii=True))
    if "doi_chieu_val_coco" in kq:
        d = kq["doi_chieu_val_coco"]
        print("doi_chieu_val_coco", json.dumps({k: d[k] for k in ("so_cap_ghep_duoc", "khac_biet", "lop_khac", "ngoai_le_ignore")}, ensure_ascii=True))
    print("trang_thai", json.dumps(tt, ensure_ascii=True))
    return ma


if __name__ == "__main__":
    sys.exit(main())
