#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 33 -- regression cho kiem_nhan_ung_vien.py (R32-F2, review c4d64a1). Fixture tam, khong cham du lieu that.

Chay:  python handoff/logs/review33/test_kiem_nhan_ung_vien.py [--script <duong dan bo kiem>]
Mac dinh kiem handoff/logs/review32/kiem_nhan_ung_vien.py (ban sua). Truyen ban c4d64a1 de chung minh test BAT loi cu.
Ma thoat: 0 moi ca PASS, 1 co ca FAIL.
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
kq = []


def nhan(i, cat="car", x1=10.0, crowd=False):
    return {"id": i, "category": cat, "attributes": {"occluded": False, "truncated": False, "crowd": crowd},
            "box2d": {"x1": x1, "y1": 20.0, "x2": x1 + 30.0, "y2": 60.0}}


def du_lieu_sach():
    """train: ta (3 frame, frame 3 rong); val: va (2 frame, id 1 car + id 2 other vehicle), vb (2 frame, id 1 pedestrian)."""
    d = {"train": {"ta": [[nhan("1")], [nhan("1"), nhan("2", "pedestrian", 100.0)], []]},
         "val": {"va": [[nhan("1"), nhan("2", "other vehicle", 200.0)], [nhan("1"), nhan("2", "other vehicle", 200.0)]],
                 "vb": [[nhan("1", "pedestrian", 50.0)], [nhan("1", "pedestrian", 50.0)]]}}
    inst = {("va", "1"): 1, ("va", "2"): 2, ("vb", "1"): 3}  # cung id "1" o hai video -> instance khac nhau, KHONG duoc gop
    return d, inst


def ghi_fixture(goc, d, inst, coco_sua=None, bo_split=(), bo_root=False, bo_anh=(), them_frame=None):
    lop = {"pedestrian": 1, "car": 3, "truck": 4}
    coco = {"categories": [{"id": v, "name": k} for k, v in lop.items()], "videos": [], "images": [], "annotations": []}
    for split, videos in d.items():
        for v, frames in videos.items():
            fr_json = []
            for k, labs in enumerate(frames, 1):
                ten = f"{v}-{k:07d}.jpg"
                if split not in bo_anh:
                    img = goc / "images" / split / v / ten
                    img.parent.mkdir(parents=True, exist_ok=True)
                    img.write_bytes(b"x")
                fr_json.append({"name": ten, "videoName": v, "frameIndex": k - 1, "labels": labs})
                if split == "val":
                    iid = len(coco["images"]) + 1
                    coco["images"].append({"id": iid, "video_id": 1, "file_name": f"{v}/{ten}"})
                    for n in labs if isinstance(labs, list) else []:
                        try:
                            b = n["box2d"]
                            phu = n["category"] in ("other vehicle", "other person", "trailer")
                            ca = {"other vehicle": "car", "other person": "pedestrian", "trailer": "truck"}.get(n["category"], n["category"])
                            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid, "scalabel_id": n["id"],
                                                        "instance_id": inst[(v, n["id"])], "category_id": lop[ca],
                                                        "ignore": int(phu), "iscrowd": int(n["attributes"]["crowd"] or phu),
                                                        "bbox": [b["x1"], b["y1"], b["x2"] - b["x1"] + 1, b["y2"] - b["y1"] + 1]})
                        except (KeyError, TypeError):
                            pass
            if them_frame and them_frame[0] == (split, v):
                fr_json.append(them_frame[1])
            if bo_root or split in bo_split:
                continue
            f = goc / "labels" / split / f"{v}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(fr_json, allow_nan=True), encoding="utf-8")
    if coco_sua:
        coco_sua(coco)
    (goc / "coco.json").write_text(json.dumps(coco), encoding="utf-8")


def chay(script, goc, *them, coco=None):
    ra = goc / "ra.json"
    lenh = [sys.executable, str(script), "--nhan", str(goc / "labels"), "--anh", str(goc / "images"),
            "--coco-val", str(coco or goc / "coco.json"), "--ra", str(ra), *them]
    r = subprocess.run(lenh, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        bc = json.loads(ra.read_text(encoding="utf-8")) if ra.exists() else {}
    except ValueError:
        bc = {}
    return r.returncode, bc, (r.stdout + r.stderr)[-400:]


def g(d, *duong, mac_dinh=None):
    for k in duong:
        if not isinstance(d, dict) or k not in d:
            return mac_dinh
        d = d[k]
    return d


def ca(ten, script, chuan_bi, ky_vong, *them, coco=None):
    with tempfile.TemporaryDirectory(prefix="v33_kiem_nhan_") as td:
        goc = Path(td)
        chuan_bi(goc)
        rc, bc, log = chay(script, goc, *them, coco=coco)
        try:
            ok, ct = ky_vong(rc, bc, goc)
        except Exception as e:  # noqa: BLE001 -- ban cu thieu khoa -> FAIL, khong lam sap bo test
            ok, ct = False, f"loi khi doc bao cao: {e!r}"
    kq.append((ten, bool(ok)))
    print(("PASS" if ok else "FAIL"), ten, f"| exit={rc}", "|", ct if not ok else "", "|" if not ok else "", log[-200:] if not ok else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(REPO / "handoff/logs/review32/kiem_nhan_ung_vien.py"))
    s = Path(ap.parse_args().script).resolve()
    print("SCRIPT", s)

    def bt(**kw):
        def f(goc):
            d, inst = du_lieu_sach()
            sua = kw.pop("sua_du_lieu", None)
            if sua:
                sua(d)
            ghi_fixture(goc, d, inst, **kw)
        return f

    tt = lambda bc, *k: g(bc, "trang_thai", *k)  # noqa: E731
    loi = lambda bc, split, k: g(bc, split, "loi_schema", k, mac_dinh=-1)  # noqa: E731
    kb = lambda bc, k: g(bc, "doi_chieu_val_coco", "khac_biet", k, mac_dinh=-1)  # noqa: E731

    # --- doi chung
    ca("01 sach -> exit 0, schema DAT, COCO MATCH (id '1' o 2 video khong bi gop)", s, bt(),
       lambda rc, bc, goc: (rc == 0 and tt(bc, "doi_chieu_coco") == "MATCH" and tt(bc, "schema_va_do_day_du", "tong") == "DAT"
                            and g(bc, "doi_chieu_val_coco", "so_cap_ghep_duoc") == 6, tt(bc)))
    # --- thieu dau vao
    ca("02 thieu root -> exit 2, dau_vao LOI", s, bt(bo_root=True),
       lambda rc, bc, goc: (rc == 2 and tt(bc, "dau_vao") == "LOI", tt(bc)))
    ca("03 thieu split train (pham vi mac dinh) -> exit 2", s, bt(bo_split=("train",)),
       lambda rc, bc, goc: (rc == 2 and tt(bc, "dau_vao") == "LOI" and "train" in " ".join(tt(bc, "loi_dau_vao") or []), tt(bc)))
    ca("04 thieu train nhung chon ro --split val -> exit 0, train NOT_RUN", s, bt(bo_split=("train",)),
       lambda rc, bc, goc: (rc == 0 and tt(bc, "schema_va_do_day_du", "train") == "NOT_RUN"
                            and tt(bc, "schema_va_do_day_du", "val") == "DAT" and tt(bc, "doi_chieu_coco") == "MATCH", tt(bc)),
       "--split", "val")

    def nhan_rong(goc):
        bt()(goc)
        for p in (goc / "labels" / "train").glob("*.json"):
            p.unlink()
    ca("05 bo nhan train rong (0 JSON) -> exit 2", s, nhan_rong,
       lambda rc, bc, goc: (rc == 2 and tt(bc, "dau_vao") == "LOI", tt(bc)))
    ca("06 thieu thu muc anh val -> exit 2", s, bt(bo_anh=("val",)),
       lambda rc, bc, goc: (rc == 2 and tt(bc, "dau_vao") == "LOI", tt(bc)))
    ca("07 --split train -> val NOT_RUN, COCO NOT_RUN, exit 0", s, bt(),
       lambda rc, bc, goc: (rc == 0 and tt(bc, "schema_va_do_day_du", "val") == "NOT_RUN" and tt(bc, "doi_chieu_coco") == "NOT_RUN", tt(bc)),
       "--split", "train")

    # --- schema / kieu
    def sua_box(gia_tri):
        def f(d):
            d["train"]["ta"][0][0]["box2d"]["x1"] = gia_tri
        return f
    ca("08 x1 = NaN -> exit 1, box_khong_huu_han", s, bt(sua_du_lieu=sua_box(float("nan"))),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "box_khong_huu_han") == 1, g(bc, "train", "loi_schema")))
    ca("09 x1 = Inf -> exit 1, box_khong_huu_han", s, bt(sua_du_lieu=sua_box(float("inf"))),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "box_khong_huu_han") == 1, g(bc, "train", "loi_schema")))
    ca("10 x1 = true (boolean) -> exit 1, box_toa_do_bool", s, bt(sua_du_lieu=sua_box(True)),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "box_toa_do_bool") == 1, g(bc, "train", "loi_schema")))
    ca("11 x1 = '10' (chuoi) -> exit 1, box_thieu_hoac_sai_kieu", s, bt(sua_du_lieu=sua_box("10")),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "box_thieu_hoac_sai_kieu") == 1, g(bc, "train", "loi_schema")))

    def sua_labels(gia_tri, xoa=False):
        def f(goc):
            d, inst = du_lieu_sach()
            ghi_fixture(goc, d, inst)
            p = goc / "labels/train/ta.json"
            fr = json.loads(p.read_text(encoding="utf-8"))
            if xoa:
                del fr[2]["labels"]
            else:
                fr[2]["labels"] = gia_tri
            p.write_text(json.dumps(fr), encoding="utf-8")
        return f
    ca("12 labels = {} -> exit 1, labels_khong_phai_list (khong thanh frame rong)", s, sua_labels({}),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "labels_khong_phai_list") == 1, g(bc, "train", "loi_schema")))
    ca("13 labels = 'x' -> exit 1, labels_khong_phai_list", s, sua_labels("x"),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "labels_khong_phai_list") == 1, g(bc, "train", "loi_schema")))
    ca("14 labels = null -> hop le theo Scalabel, dem rieng, exit 0", s, sua_labels(None),
       lambda rc, bc, goc: (rc == 0 and g(bc, "train", "so_frame_labels_null") == 1, g(bc, "train")))
    ca("15 thieu khoa labels -> hop le theo Scalabel, dem rieng, exit 0", s, sua_labels(None, xoa=True),
       lambda rc, bc, goc: (rc == 0 and g(bc, "train", "so_frame_thieu_khoa_labels") == 1, g(bc, "train")))

    def id_trung(d):
        d["train"]["ta"][0].append(nhan("1", x1=300.0))
    ca("16 hai nhan cung ID trong 1 frame -> exit 1, id_trung_trong_frame", s, bt(sua_du_lieu=id_trung),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "id_trung_trong_frame") == 1, g(bc, "train", "loi_schema")))
    ca("17 frame trung ten -> exit 1, frame_trung_ten", s,
       bt(them_frame=(("train", "ta"), {"name": "ta-0000001.jpg", "videoName": "ta", "frameIndex": 0, "labels": []})),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "frame_trung_ten") == 1, g(bc, "train", "loi_schema")))

    def crowd_sai(d):
        d["train"]["ta"][0][0]["attributes"]["crowd"] = "no"
    ca("18 attributes.crowd = 'no' -> exit 1", s, bt(sua_du_lieu=crowd_sai),
       lambda rc, bc, goc: (rc == 1 and loi(bc, "train", "attributes_thieu_hoac_sai_kieu") == 1, g(bc, "train", "loi_schema")))

    # --- doi chieu COCO
    ca("19 file COCO duoc yeu cau nhung khong ton tai -> exit 2, LOI_DAU_VAO", s, bt(),
       lambda rc, bc, goc: (rc == 2 and tt(bc, "doi_chieu_coco") == "LOI_DAU_VAO", tt(bc)), coco=Path("Z:/khong/ton/tai.json"))
    ca("20 --khong-doi-chieu-coco -> COCO NOT_RUN co ly do, exit 0", s, bt(),
       lambda rc, bc, goc: (rc == 0 and tt(bc, "doi_chieu_coco") == "NOT_RUN" and tt(bc, "ly_do_doi_chieu_coco"), tt(bc)),
       "--khong-doi-chieu-coco")

    def coco_sua(f):
        return bt(coco_sua=f)
    ca("21 khoa COCO trung (cung frame + scalabel_id) -> exit 3, coco_khoa_trung=1", s,
       coco_sua(lambda c: c["annotations"].append(dict(copy.deepcopy(c["annotations"][0]), id=999))),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "coco_khoa_trung") == 1 and tt(bc, "doi_chieu_coco") == "DIFFERENCES", g(bc, "doi_chieu_val_coco", "khac_biet")))

    def gop(c):  # id native "2" o va gop vao instance cua id "1" (moi frame) -> 1 COCO <- 2 native
        for a in c["annotations"]:
            if a["scalabel_id"] == "2":
                a["instance_id"] = 1
    ca("22 hai ID native gop vao 1 instance COCO -> exit 3, id_mot_coco_nhieu_native=1", s, coco_sua(gop),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "id_mot_coco_nhieu_native") == 1 and g(bc, "doi_chieu_val_coco", "so_track_id_anh_xa_bat_nhat_quan", mac_dinh=0) >= 1,
                            g(bc, "doi_chieu_val_coco", "khac_biet")))

    def tach(c):  # id "1" o va: frame 2 sang instance moi -> 1 native -> 2 COCO
        for a in c["annotations"]:
            if a["scalabel_id"] == "1" and a["image_id"] == 2:
                a["instance_id"] = 77
    ca("23 mot ID native tach thanh 2 instance COCO -> exit 3, id_mot_native_nhieu_coco=1", s, coco_sua(tach),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "id_mot_native_nhieu_coco") == 1, g(bc, "doi_chieu_val_coco", "khac_biet")))

    def box5(c):
        c["annotations"][0]["bbox"][0] += 5.0
    ca("24 bbox lech 5 px -> exit 3, box_lech=1", s, coco_sua(box5),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "box_lech") == 1, g(bc, "doi_chieu_val_coco", "khac_biet")))

    def doi_lop(c):
        c["annotations"][0]["category_id"] = 4  # car -> truck
    def ky_vong_lop(rc, bc, goc):
        hang = list(csv.DictReader(open(goc / "ra_lop_khac.csv", encoding="utf-8")))
        return (rc == 3 and kb(bc, "lop_khac_anh_xa") == 1 and len(hang) == 1 and hang[0]["native_class"] == "car"
                and hang[0]["coco_class"] == "truck", g(bc, "doi_chieu_val_coco", "khac_biet"))
    ca("25 doi lop car->truck -> exit 3, lop_khac=1, CSV 1 hang", s, coco_sua(doi_lop), ky_vong_lop)

    def bo_ignore(c):
        for a in c["annotations"]:
            if a["scalabel_id"] == "2" and a["image_id"] == 1:
                a["ignore"], a["iscrowd"] = 0, 0
    ca("26 other vehicle co ignore=0 -> exit 3, ngoai_le_ignore=1", s, coco_sua(bo_ignore),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "ngoai_le_ignore") == 1 and kb(bc, "lop_khac_anh_xa") == 0, g(bc, "doi_chieu_val_coco", "khac_biet")))

    def iscrowd_sai(c):
        c["annotations"][0]["iscrowd"] = 1  # car, crowd False, ignore 0 -> iscrowd phai la 0
    ca("27 iscrowd ngoai quy tac (crowd hoac ignore) -> exit 3", s, coco_sua(iscrowd_sai),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "iscrowd_ngoai_quy_tac") == 1, g(bc, "doi_chieu_val_coco", "khac_biet")))
    ca("28 nhan chi co o native -> exit 3", s, coco_sua(lambda c: c["annotations"].pop()),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "nhan_chi_co_o_native") == 1, g(bc, "doi_chieu_val_coco", "khac_biet")))
    ca("29 annotation chi co o COCO -> exit 3", s,
       coco_sua(lambda c: c["annotations"].append(dict(copy.deepcopy(c["annotations"][0]), id=998, scalabel_id="999", instance_id=50))),
       lambda rc, bc, goc: (rc == 3 and kb(bc, "annotation_chi_co_o_coco") == 1, g(bc, "doi_chieu_val_coco", "khac_biet")))

    def ca30(goc):
        d, inst = du_lieu_sach()
        sua_box(float("nan"))(d)
        ghi_fixture(goc, d, inst, coco_sua=box5)
    ca("30 loi schema + COCO khac -> exit 1 (schema uu tien), COCO van DIFFERENCES", s, ca30,
       lambda rc, bc, goc: (rc == 1 and tt(bc, "doi_chieu_coco") == "DIFFERENCES", tt(bc)))

    so_pass = sum(v for _, v in kq)
    print("TONG", so_pass, "/", len(kq), "PASS")
    return 0 if so_pass == len(kq) else 1


if __name__ == "__main__":
    sys.exit(main())
