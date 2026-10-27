"""
test_mixed5.py -- test R52A cho dataset tron mixed5_v1: anh xa lop, quy tac frame/buoc, to xam, ro ri, split, hop dong.

    python -X utf8 src/mixed5/test_mixed5.py

Fixture tao trong results/phase2/mixed5/_tmp_test (xoa khi xong). Test cuoi doc split that (chi doc).
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed5_lib as m  # noqa: E402
from mixed5_lib import GOC, LOP_5, bx  # noqa: E402

TMP = GOC / "results/phase2/mixed5/_tmp_test"


def anh(p: Path, w=100, h=80, mau=(0, 0, 200)) -> Path:
    import cv2
    p.parent.mkdir(parents=True, exist_ok=True)
    im = np.zeros((h, w, 3), np.uint8)
    im[:] = mau
    cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return p


def xml_fixture(p: Path, video: str, ignored: list, frames: dict) -> Path:
    s = [f'<?xml version="1.0" encoding="utf-8"?>\n<sequence name="{video}">']
    if ignored:
        s.append("<ignored_region>" + "".join(f'<box left="{l}" top="{t}" width="{w}" height="{h}"/>'
                                               for l, t, w, h in ignored) + "</ignored_region>")
    for n, tg in frames.items():
        s.append(f'<frame density="{len(tg)}" num="{n}"><target_list>')
        for i, (vt, l, t, w, h) in enumerate(tg, 1):
            s.append(f'<target id="{i}"><box left="{l}" top="{t}" width="{w}" height="{h}"/>'
                     f'<attribute orientation="0" speed="0" trajectory_length="1" truncation_ratio="0" '
                     f'vehicle_type="{vt}"/></target>')
        s.append("</target_list></frame>")
    s.append("</sequence>\n")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(s), encoding="utf-8")
    return p


def dem_moi() -> dict:
    return {k: 0 for k in m.KHOA_DEM} | {"box_xml_khong_hop_le": 0}


class CoTmp(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)


class TestAnhXaLop(unittest.TestCase):
    def test_anh_xa_dung_quyet_dinh(self):
        self.assertEqual(m.ANH_XA_LOP, {"car": "car", "van": "car", "bus": "bus", "others": "truck"})
        dem = dem_moi()
        tg = [("car", 1, 1, 5, 5), ("van", 1, 1, 5, 5), ("others", 1, 1, 5, 5), ("bus", 1, 1, 5, 5)]
        fr = m.frame_cho_bx("V", 10, tg, [], m.ANH_XA_LOP, dem)
        self.assertEqual([n[0] for n in fr.nhan], ["car", "car", "truck", "bus"])
        self.assertTrue(all(bx.vai_tro(*n[:1], n[5], n[6]) == "gan_nhan" for n in fr.nhan))
        self.assertEqual(fr.rel, "V/img00010.jpg")

    def test_lop_la_bao_loi(self):
        with self.assertRaises(m.LoiDuLieu):
            m.frame_cho_bx("V", 10, [("motorbike", 1, 1, 5, 5)], [], m.ANH_XA_LOP, dem_moi())
        with self.assertRaises(m.LoiDuLieu):
            m.frame_cho_bx("V", 10, [(None, 1, 1, 5, 5)], [], m.ANH_XA_LOP, dem_moi())

    def test_box_xml_hong_va_ignore(self):
        dem = dem_moi()
        fr = m.frame_cho_bx("V", 10, [("car", 1, 1, 0, 5), ("car", float("nan"), 1, 3, 3)], [(0, 0, 10, 10)],
                            m.ANH_XA_LOP, dem)
        self.assertEqual(dem["box_xml_khong_hop_le"], 2)
        self.assertEqual(len(fr.nhan), 1)
        self.assertEqual(bx.vai_tro(fr.nhan[0][0], fr.nhan[0][5], fr.nhan[0][6]), "to_xam")


class TestChonFrame(unittest.TestCase):
    def test_buoc_10_va_chi_frame_co_trong_xml(self):
        xml = set(range(1, 41)) - {20, 21}
        anh_ = set(range(1, 46))
        chon, dem = m.chon_frame_detrac(xml, anh_, 10)
        self.assertEqual(chon, [10, 30, 40])                     # 20 khong co <frame> -> loai
        self.assertEqual(dem["anh_khong_co_frame"], 7)           # 20, 21, 41..45
        self.assertEqual(dem["anh_theo_buoc_khong_co_frame"], 1)  # 20
        self.assertEqual(dem["frame_xml_khong_theo_buoc"], len(xml) - 3)

    def test_frame_xml_thieu_anh_la_loi(self):
        with self.assertRaises(m.LoiDuLieu):
            m.chon_frame_detrac({10, 20}, {10}, 10)

    def test_ten_anh_khong_trung_giua_video(self):
        self.assertEqual(m.ten_anh_detrac("MVI_20011", 10), "MVI_20011_img00010.jpg")
        self.assertNotEqual(m.ten_anh_detrac("MVI_1", 10), m.ten_anh_detrac("MVI_2", 10))


class TestXuatDetrac(CoTmp):
    def _xuat(self, video, ignored, frames):
        x = m.doc_xml_detrac(xml_fixture(TMP / "xml" / f"{video}.xml", video, ignored, frames))
        for n in range(1, 31):
            anh(TMP / "img" / video / f"img{n:05d}.jpg")
        chon, _ = m.chon_frame_detrac(x["frames"], m.liet_ke_anh_detrac(TMP / "img" / video), 10)
        dem, rows = dem_moi(), {}
        for n in chon:
            fr = m.frame_cho_bx(video, n, x["frames"][n], x["ignored"], m.ANH_XA_LOP, dem)
            rows[n] = m.xuat_frame_detrac(TMP / "img" / video / f"img{n:05d}.jpg", fr, TMP / "ds", "train", video,
                                          m.ten_anh_detrac(video, n), dem)
        return rows, dem

    def test_to_xam_nhan_am_tinh_cat_khung(self):
        import cv2
        frames = {10: [("car", 10, 10, 20, 20), ("van", 60, 50, 60, 60)],    # van vuot khung -> cat
                  15: [("car", 1, 1, 5, 5)],                                  # khong theo buoc
                  20: [("others", 30, 30, 10, 10), ("bus", 99.5, 70, 5, 5)],  # bus < 1 px sau cat -> bo
                  30: [("car", 5, 5, 0, 10)]}                                 # box hong -> frame am tinh
        rows, dem = self._xuat("MVI_1", [(0, 0, 40, 40)], frames)
        self.assertEqual(sorted(rows), [10, 20, 30])
        r10 = rows[10]
        self.assertEqual((r10["n_car"], r10["n_truck"], r10["xu_ly"], r10["n_vung_to_xam"]), (2, 0, "to_xam", 1))
        dong = (TMP / "ds" / r10["nhan_ra"]).read_text(encoding="utf-8").split("\n")
        self.assertEqual(dong[1], "0 0.800000 0.812500 0.400000 0.375000")   # van cat vao [60,100]x[50,80]
        self.assertEqual(dem["box_bi_cat_vao_khung"], 1)                     # van (frame 10); bus bi bo, khong tinh
        self.assertEqual((rows[20]["n_truck"], rows[20]["n_bus"]), (1, 0))
        self.assertEqual(dem["box_nho_hon_1px_sau_cat"], 1)
        self.assertEqual((rows[30]["am_tinh"], (TMP / "ds" / rows[30]["nhan_ra"]).read_bytes()), (1, b""))
        self.assertEqual(dem["box_xml_khong_hop_le"], 1)
        im = cv2.imread(str(TMP / "ds" / r10["anh_ra"])).astype(int)
        self.assertLess(np.abs(im[2:7, 2:32] - 114).max(), 12)               # vung ignore ngoai box -> xam
        self.assertLess(np.abs(im[2:7, 2:32].mean() - 114), 2)               # (JPEG chroma 2x2: mau box do lan ~16 px)
        self.assertGreater(im[12:28, 12:28, 2].mean(), 180)                  # box co nhan giu nguyen (do)
        self.assertGreater(im[50:78, 50:98, 2].mean(), 180)                  # ngoai vung ignore giu nguyen
        self.assertEqual((TMP / "ds" / r10["ignore_ra"]).read_text(encoding="utf-8"), "0.00 0.00 40.00 40.00\n")
        self.assertEqual(r10["frame_idx"], 10)

    def test_khong_ignore_sao_chep_nguyen_byte(self):
        rows, _ = self._xuat("MVI_2", [], {10: [("bus", 10, 10, 20, 20)]})
        r = rows[10]
        self.assertEqual((r["xu_ly"], r["ignore_ra"], r["sha256_ra"]), ("sao_chep", "", r["sha256_nguon"]))
        self.assertFalse((TMP / "ds" / "ignore" / "train" / "MVI_2_img00010.txt").exists())


def row_gia(split, video, sha, anh_ra="images/train/x.jpg", nguon="detrac", anh_nguon="V/img00010.jpg"):
    return {"split": split, "video": video, "sha256_nguon": sha, "sha256_ra": sha, "anh_ra": anh_ra, "nguon": nguon,
            "anh_nguon": anh_nguon}


class TestRoRi(unittest.TestCase):
    TAP = {"ua_detrac_test": {"sha256": {"t1"}, "rel": {"MVI_9/img00001.jpg"}, "video": {"MVI_9"}, "so_dong": 1,
                              "nguon_anh": "detrac"},
           "bdd100k_mot_val": {"sha256": {"v1"}, "rel": {"b9/b9-0000001.jpg"}, "video": {"b9"}, "so_dong": 1,
                               "nguon_anh": "bdd"}}
    VS = {"train": {"A", "B"}, "dev": {"C"}}

    def test_sach(self):
        rows = [row_gia("train", "A", "a", "images/train/a.jpg"), row_gia("dev", "C", "c", "images/dev/c.jpg")]
        bb = m.kiem_ro_ri(rows, self.TAP, self.VS)
        self.assertEqual(bb["ua_detrac_test"]["trung_sha256_nguon"], 0)

    def test_cac_kieu_ro_ri(self):
        ca = {
            "sha test": [row_gia("train", "A", "t1")],
            "sha val": [row_gia("train", "A", "v1")],
            "video test": [row_gia("train", "MVI_9", "a")],
            "anh nguon test": [row_gia("train", "A", "a", anh_nguon="MVI_9/img00001.jpg")],
            "train/dev cung sha": [row_gia("train", "A", "s", "images/train/a.jpg"), row_gia("dev", "C", "s", "images/dev/c.jpg")],
            "train/dev cung ten": [row_gia("train", "A", "a", "images/train/k.jpg"), row_gia("dev", "C", "c", "images/dev/k.jpg")],
            "anh dev thuoc video train": [row_gia("dev", "A", "a", "images/dev/a.jpg")],
        }
        for ten, rows in ca.items():
            with self.subTest(ten), self.assertRaises(m.LoiDuLieu):
                m.kiem_ro_ri(rows, self.TAP, self.VS)
        with self.assertRaises(m.LoiDuLieu):                      # dev12 lot vao train
            m.kiem_ro_ri([], self.TAP, {"train": {"A", "C"}, "dev": {"C"}})


class TestSplit(CoTmp):
    def _fixture(self, dev=("MVI_2", "MVI_4")):
        ds = {"train_fit": ["b1", "b2"], "dev": ["b3"]}
        bsp = TMP / "bdd.json"
        bsp.write_text(json.dumps({**ds, "sha256_danh_sach": m.sha256_json(ds)}), encoding="utf-8")
        for i in range(1, 6):
            xml_fixture(TMP / "xml" / f"MVI_{i}.xml", f"MVI_{i}", [], {})
        dv = TMP / "dev.json"
        dv.write_text(json.dumps({"videos": list(dev)}), encoding="utf-8")
        return bsp, dv

    def test_quy_tac_chia(self):
        bsp, dv = self._fixture()
        rec = m.tao_ban_ghi_split("t", bsp, TMP / "xml", dv, 5, 2)
        self.assertEqual(rec["detrac"]["train_fit"], ["MVI_1", "MVI_3", "MVI_5"])
        self.assertEqual(rec["detrac"]["dev"], ["MVI_2", "MVI_4"])
        self.assertEqual(rec["train_fit"], sorted(["b1", "b2", "MVI_1", "MVI_3", "MVI_5"]))
        self.assertEqual(rec["dev"], sorted(["b3", "MVI_2", "MVI_4"]))
        p = TMP / "sp.json"
        m.ghi_json(p, rec)
        self.assertEqual(m.doc_split_mixed5(p, bsp)["sha256_danh_sach"], rec["sha256_danh_sach"])
        rec2 = dict(rec, train_fit=rec["train_fit"] + ["MVI_2"])      # sua tay -> loi
        m.ghi_json(p, rec2)
        with self.assertRaises(m.LoiDuLieu):
            m.doc_split_mixed5(p, bsp)

    def test_dev_ngoai_xml_hoac_sai_so_luong(self):
        bsp, dv = self._fixture(dev=("MVI_2", "MVI_77"))
        with self.assertRaises(m.LoiDuLieu):
            m.tao_ban_ghi_split("t", bsp, TMP / "xml", dv, 5, 2)
        with self.assertRaises(m.LoiDuLieu):
            m.tao_ban_ghi_split("t", bsp, TMP / "xml", dv, 6, 2)


class TestHopDong(CoTmp):
    def test_dataset_mini_dat_hop_dong_va_bat_sua(self):
        # nguon "BDD" mini (dinh dang cua data/yolo_bdd5_v1)
        src = TMP / "bdd"
        bdd_rows = []
        for s, v, i, mau in (("train", "b1", 0, (10, 200, 10)), ("dev", "b3", 0, (200, 10, 10))):
            ten = f"{v}-{i + 1:07d}.jpg"
            img = anh(src / "images" / s / ten, mau=mau)
            lbl = src / "labels" / s / f"{Path(ten).stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_bytes(b"0 0.500000 0.500000 0.200000 0.200000\n")
            bdd_rows.append({"split": s, "video": v, "frame_idx": i, "anh_nguon": f"{v}/{ten}",
                             "sha256_nguon": bx.sha256_file(img), "anh_ra": f"images/{s}/{ten}",
                             "sha256_ra": bx.sha256_file(img), "w": 100, "h": 80,
                             **{f"n_{c}": int(c == "car") for c in LOP_5}, "n_vung_to_xam": 0, "xu_ly": "sao_chep",
                             "am_tinh": 0, "nhan_ra": f"labels/{s}/{Path(ten).stem}.txt",
                             "sha256_nhan": bx.sha256_file(lbl), "ignore_ra": "", "sha256_ignore": ""})
        out = TMP / "ds"
        rows = []
        for r in bdd_rows:
            c = {m.dat_file_bdd(src / r[k], out / r[k], True) for k in ("anh_ra", "nhan_ra") if r[k]}
            self.assertEqual(c, {"hardlink"})
            self.assertTrue((out / r["anh_ra"]).samefile(src / r["anh_ra"]))
            rows.append({**r, "nguon": "bdd", "dat_file": "hardlink"})
        for s, v, mau in (("train", "MVI_1", (0, 0, 200)), ("dev", "MVI_2", (0, 200, 200))):
            anh(TMP / "img" / v / "img00010.jpg", mau=mau)
            fr = m.frame_cho_bx(v, 10, [("van", 10, 10, 20, 20)], [(50, 0, 30, 30)], m.ANH_XA_LOP, dem_moi())
            rows.append(m.xuat_frame_detrac(TMP / "img" / v / "img00010.jpg", fr, out, s, v, m.ten_anh_detrac(v, 10),
                                            dem_moi()))
        sp = {"ten": "mini", "train_fit": ["MVI_1", "b1"], "dev": ["MVI_2", "b3"]}
        sp["sha256_danh_sach"] = m.sha256_json({"train_fit": sp["train_fit"], "dev": sp["dev"]})
        fsp = TMP / "split.json"
        m.ghi_json(fsp, sp)
        m.ghi_manifest(out, rows, fsp, sp, {}, "test")
        hd = bx.kiem_hop_dong_dataset(out / "data.yaml")
        self.assertEqual((hd["loi"], hd["so_anh"], hd["so_anh_co_file_ignore"]),
                         (0, {"train": 2, "dev": 2}, {"train": 1, "dev": 1}))
        tom = json.loads((out / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
        self.assertEqual(tom["thong_ke"]["detrac"]["train"]["box"]["car"], 1)
        # sua 1 file nhan -> hop dong phai bao loi
        (out / rows[2]["nhan_ra"]).write_bytes(b"1 0.5 0.5 0.1 0.1\n")
        with self.assertRaises(bx.LoiDuLieu):
            bx.kiem_hop_dong_dataset(out / "data.yaml")


class TestSplitThat(unittest.TestCase):
    """Doc (chi doc) split da khoa that + XML + dev12."""

    def test_split_mixed5_v1(self):
        p = GOC / "configs/phase2/detector/splits/mixed5_v1.json"
        bsp_p = GOC / "configs/phase2/detector/splits/bdd5_train_v1.json"
        if not p.is_file():
            self.skipTest("chua khoa split")
        sp = m.doc_split_mixed5(p, bsp_p)
        xml = {q.stem for q in (GOC / "data/extracted/DETRAC-Train-Annotations-XML").glob("*.xml")}
        dev12 = set(json.loads((GOC / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"])
        self.assertEqual((len(xml), len(dev12)), (60, 12))
        self.assertEqual(set(sp["detrac"]["dev"]), dev12)
        self.assertEqual(set(sp["detrac"]["train_fit"]), xml - dev12)
        self.assertEqual(len(sp["detrac"]["train_fit"]), 48)
        bsp = json.loads(bsp_p.read_text(encoding="utf-8"))
        self.assertEqual((len(bsp["train_fit"]), len(bsp["dev"])), (1261, 139))
        self.assertEqual(set(sp["train_fit"]) - xml, set(bsp["train_fit"]))
        self.assertEqual(set(sp["dev"]) - xml, set(bsp["dev"]))
        self.assertEqual(sp["bdd"]["sha256"], bx.sha256_file(bsp_p))
        test_v = {q.stem for q in (GOC / "data/extracted/DETRAC-Test-Annotations-XML").glob("*.xml")}
        self.assertEqual(len(test_v), 40)
        self.assertFalse(test_v & (set(sp["train_fit"]) | set(sp["dev"])))


class TestR52BCauHinh(unittest.TestCase):
    """R52B: bo so khoa-gia-tri train_mixed5 vs train_bdd5."""

    def setUp(self):
        import so_cau_hinh_train as sc
        self.sc = sc
        self.goc = {"ten": "train_bdd5_v1", "model": {"yolov8n": {"weights": "a.pt", "sha256": "x"}},
                    "dataset": {"day_du": "data/yolo_bdd5_v1/data.yaml", "smoke": "data/yolo_bdd5_v1_smoke/data.yaml"},
                    "huan_luyen": {"epochs": 15, "batch": 8, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
                                   "momentum": 0.937, "weight_decay": 0.0005, "patience": 5, "seed": 0,
                                   "deterministic": True, "amp": False, "cache": False, "imgsz": 640, "pretrained": True,
                                   "warmup_epochs": 1.0},
                    "smoke": {"epochs": 1, "patience": 1}}

    def moi(self):
        d = json.loads(json.dumps(self.goc))
        d["ten"] = "train_mixed5_v1"
        d["dataset"] = {"day_du": "data/yolo_mixed5_v1/data.yaml", "smoke": "data/yolo_mixed5_v1_smoke/data.yaml"}
        return d

    def test_chi_khac_khoa_duoc_phep(self):
        kq = self.sc.so(self.goc, self.moi())
        self.assertEqual(kq["loi"], [])
        self.assertEqual(set(kq["khoa_khac"]), {"ten", "dataset.day_du", "dataset.smoke"})

    def test_bat_khac_ngoai_khoa_va_kieu(self):
        for duong, gia in ((("huan_luyen", "lr0"), 0.02), (("huan_luyen", "warmup_epochs"), 1),
                           (("smoke", "patience"), 2), (("model", "yolov8n", "sha256"), "y")):
            d = self.moi()
            x = d
            for k in duong[:-1]:
                x = x[k]
            x[duong[-1]] = gia
            self.assertTrue(self.sc.so(self.goc, d)["loi"], duong)      # 1.0 -> 1 cung bi bat (so ca kieu)
        d = self.moi()
        d["huan_luyen"]["mosaic"] = 1.0                                   # khoa them
        self.assertTrue(self.sc.so(self.goc, d)["loi"])
        d = self.moi()
        d["dataset"]["smoke"] = "data/yolo_bdd5_v1_smoke/data.yaml"       # khoa duoc phep nhung gia tri sai ke hoach
        self.assertTrue(self.sc.so(self.goc, d)["loi"])

    def test_cau_hinh_that(self):
        import yaml
        g = yaml.safe_load((GOC / "configs/phase2/detector/train_bdd5.yaml").read_text(encoding="utf-8"))
        n = yaml.safe_load((GOC / "configs/phase2/detector/train_mixed5.yaml").read_text(encoding="utf-8"))
        self.assertEqual(self.sc.so(g, n)["loi"], [])


class TestR52BSmoke(unittest.TestCase):
    """R52B: quy tac chon tap con smoke (tat dinh, dung split, giu nguyen dong nguon)."""

    def test_chon_video_detrac(self):
        import xuat_smoke_mixed5 as xs
        sp = {"detrac": {"train_fit": [f"T{i:02d}" for i in range(48)][::-1], "dev": [f"D{i:02d}" for i in range(12)]}}
        a, b = xs.chon_video_detrac(sp), xs.chon_video_detrac(sp)
        self.assertEqual(a, b)                                            # tat dinh, khong phu thuoc thu tu dau vao
        self.assertEqual((len(a["train"]), len(a["dev"])), (3, 1))
        self.assertTrue(set(a["train"]) <= set(sp["detrac"]["train_fit"]) and set(a["dev"]) <= set(sp["detrac"]["dev"]))
        sp2 = {"detrac": {k: sorted(v) for k, v in sp["detrac"].items()}}
        self.assertEqual(xs.chon_video_detrac(sp2), a)

    def test_chon_dong(self):
        import xuat_smoke_mixed5 as xs
        cot = {k: "0" for k in bx.COT_MANIFEST}
        bdd = [{**cot, "split": "train", "video": "b1", "anh_ra": "images/train/b1.jpg"}]
        full = [{**cot, "split": "train", "video": "V1", "anh_ra": "images/train/V1.jpg", "nguon": "detrac"},
                {**cot, "split": "dev", "video": "V1", "anh_ra": "images/dev/V1x.jpg", "nguon": "detrac"},
                {**cot, "split": "train", "video": "V2", "anh_ra": "images/train/V2.jpg", "nguon": "detrac"},
                {**cot, "split": "train", "video": "b1", "anh_ra": "images/train/b1.jpg", "nguon": "bdd"}]
        rows = xs.chon_dong(bdd, full, {"train": ["V1"], "dev": []})
        self.assertEqual([(r["nguon"], r["anh_ra"]) for r in rows], [("bdd", "images/train/b1.jpg"),
                                                                      ("detrac", "images/train/V1.jpg")])
        self.assertTrue(all(set(bx.COT_MANIFEST) <= set(r) for r in rows))

    def test_smoke_that(self):
        p = GOC / "data/yolo_mixed5_v1_smoke/data.yaml"
        if not p.is_file():
            self.skipTest("chua dung tap smoke")
        hd = bx.kiem_hop_dong_dataset(p)
        self.assertEqual((hd["loi"], hd["tap_con"]), (0, "smoke"))
        tom = json.loads((p.parent / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
        vd = tom["quy_tac_tap_con"]["detrac"]["video"]
        dev12 = set(json.loads((GOC / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"])
        self.assertTrue(set(vd["dev"]) <= dev12 and not set(vd["train"]) & dev12)


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=2).result
    sys.exit(0 if r.wasSuccessful() else 1)
