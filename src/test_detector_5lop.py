"""
test_detector_5lop.py -- Duong chuan bi dataset / danh gia / huan luyen detector 5 lop (vong 27).

Chay tren FIXTURE TONG HOP trong thu muc tam (anh JPEG + nhan Scalabel va COCO-video cung noi dung) -- khong can
BDD100K train that, khong dung du lieu nghien cuu. Nhom kiem:
  A. Anh xa lop checkpoint <-> 5 lop theo TEN (COCO 80 lop, 5 lop, thu tu la, thieu/trung lop).
  B. Xuat Scalabel: kiem nguon, chia THEO VIDEO (roi nhau, tat dinh, phan tang), khoa split, anh am tinh hop le vs
     duoi video khong nhan, cat box vao khung, bo box < 1 px, to xam crowd/ignore TRU pixel box co nhan, file vung bo
     qua, toa do theo kich thuoc anh thuc, sao chep nguyen byte, data.yaml val = dev.
  C. COCO-video cho CUNG split + CUNG nhan/anh xuat nhu Scalabel.
  D. Du lieu thieu / chay nham: thieu nguon (exit 2, khong tao thu muc), thieu anh le, trung ten video / noi dung anh
     voi tap danh gia, nguon nhan doi sau khi khoa split, split bi sua tay, thu muc dich da co, thieu file nhan, CLI
     cu khong con mac dinh tron UA-DETRAC / bo qua BDD.
  E. Evaluator: AP khop pycocotools (co vung bo qua kieu crowd), P/R van hanh tinh tay, kich thuoc anh thuc, thieu
     nhan -> loi, lop khong co GT -> null, du doan that YOLOv8n COCO tren CPU chi ra lop 0..4.
  F. Runner huan luyen: --dry-run khong train/khong tao thu muc, tu choi thu muc luot da co, tu choi 'auto',
     day-du can xac nhan, sai sha256 checkpoint.
Vong 28 (review 4937c85):
  E7-E9 (R27-F3) hoa IoU (fixture review), IoU dung bang nguong, GT hop le chong vung ignore -- doi chieu pycocotools
     THAT (thieu thu vien -> SKIP "CHUA KIEM", khong tinh PASS).
  G  (R27-F1) runner qua ENTRY POINT `train_detector.main()`; `ultralytics.YOLO` duoc thay bang SPY (GIA LAP ranh gioi
     nap model / training): train tro vao dev, thieu nhan (anh co box va anh am tinh) -> exit 2, spy khong bi goi;
     dataset hop le -> toi dung ranh gioi training.
  H  (R27-F2) anh doi noi dung cung kich thuoc, toa do nhan doi cung lop/so box, ignore bi xoa/sua, file ignore thua,
     manifest phien ban 1, anh thua, `path` tuong doi, trung noi dung train/dev luc xuat.
  I  (R27-F2) du lieu ultralytics THUC SU nap: trainer GIA LAP (I1-I3) va YOLODataset THAT voi cache nhan cu cung
     kich thuoc (I4, cach cache ultralytics chi kiem kich thuoc + duong dan) / cache moi (I5).
Vong 29 (review 07615ba):
  J  (R28-F1) cache ANH `<anh>.npy` (khac cache NHAN `.cache` o nhom I): dataset sach -> YOLODataset THAT doc pixel trung
     JPEG; .npy anh den canh anh train / dev -> ultralytics that nap anh den (nguy co) va hop dong + chot runner tu choi;
     entry point runner + SPY: exit 2, khong nap model / train / output; `cache: disk` bi tu choi; callback voi
     YOLODataset THAT chan .npy xuat hien sau hop dong.
  Dem: thieu weights / thu vien -> SKIP "CHUA KIEM" (A7, E6, F, G, J3/J4/J6), KHONG tinh la PASS.
Chay duoc tren ban sao ma cu (DET5_MODELS_DIR=<thu muc models>) de chung minh regression bat ma cu.

    python src/test_detector_5lop.py
"""
from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import config  # noqa: E402

PY = sys.executable
_RESULTS: list[tuple[str, bool]] = []
_SKIP: list[tuple[str, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail[:450]})" if detail else ""))
    return bool(ok)


def _skip(name: str, ly_do: str) -> None:
    """Phep kiem KHONG chay duoc (vd thieu pycocotools): ghi ro CHUA KIEM, KHONG tinh la PASS."""
    _SKIP.append((name, ly_do))
    print(f"  [SKIP] {name}  ({ly_do}) -- CHUA KIEM, khong tinh PASS")


def _thu(name: str, fn) -> None:
    """Moi phep kiem doc lap: loi khong mong doi -> FAIL rieng phep do, cac phep sau van chay (chay duoc tren ma cu)."""
    try:
        ok, detail = fn()
        _check(name, ok, detail)
    except Exception as e:  # noqa: BLE001
        _check(name, False, f"loi {type(e).__name__}: {str(e)[:300]}")


def _models_dir() -> Path:
    """Thu muc weights (bien moi truong DET5_MODELS_DIR cho phep chay test nay tren ban sao ma cu khong co models/)."""
    return Path(os.environ.get("DET5_MODELS_DIR") or config.MODELS_DIR)


def _chay(nhom: str, fn) -> None:
    print(f"\n== {nhom}")
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        _check(f"{nhom}: loi khong mong doi {type(e).__name__}: {e}", False)
        traceback.print_exc()


def _yaml(p: Path, d: dict) -> Path:
    import yaml
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


# =========================================================================== A
def nhom_a():
    import detector_classes as dc
    import yaml
    coco = yaml.safe_load((Path(__import__("ultralytics").__file__).parent / "cfg" / "datasets" / "coco.yaml")
                          .read_text(encoding="utf-8"))["names"]
    m = dc.anh_xa_model_sang_lop5(coco)
    _check("A1 COCO 80 lop -> {2:car,7:truck,5:bus,3:motorcycle,1:bicycle}",
           m == {2: 0, 7: 1, 5: 2, 3: 3, 1: 4} and dc.loai_checkpoint(coco) == "coco80", str(m))
    l5 = {i: c for i, c in enumerate(dc.LOP_5)}
    _check("A2 checkpoint 5 lop -> dong nhat", dc.anh_xa_model_sang_lop5(l5) == {i: i for i in range(5)}
           and dc.loai_checkpoint(l5) == "lop5")
    la = {0: "bus", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "truck"}
    _check("A3 thu tu la -> anh xa THEO TEN, loai 'khac'",
           dc.anh_xa_model_sang_lop5(la) == {2: 0, 4: 1, 0: 2, 3: 3, 1: 4} and dc.loai_checkpoint(la) == "khac")
    for ten, names in (("thieu bicycle", {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}),
                       ("trung car", {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle", 5: "car"})):
        try:
            dc.anh_xa_model_sang_lop5(names)
            _check(f"A4 {ten} -> LoiAnhXaLop", False)
        except dc.LoiAnhXaLop:
            _check(f"A4 {ten} -> LoiAnhXaLop", True)
    _check("A5 sang_id_coco giu quy uoc tracker (--classes 1 2 3 5 7)",
           [dc.sang_id_coco(i) for i in range(5)] == [2, 7, 5, 3, 1])
    try:
        dc.kiem_names_dataset(la)
        _check("A6 names dataset sai thu tu -> loi", False)
    except dc.LoiAnhXaLop:
        _check("A6 names dataset sai thu tu -> loi", True)
    for w, ten in ((_models_dir() / "yolov8n.pt", "yolov8n"), (_models_dir() / "candidates" / "yolo11n.pt",
                                                                     "yolo11n")):
        if w.is_file():
            from ultralytics import YOLO
            n = YOLO(str(w)).names
            _check(f"A7 doc names tu checkpoint that {ten}: coco80, anh xa dung",
                   dc.loai_checkpoint(n) == "coco80" and dc.anh_xa_model_sang_lop5(n) == {2: 0, 7: 1, 5: 2, 3: 3, 1: 4})
        else:
            _skip(f"A7 doc names tu checkpoint that {ten}", f"khong co {w}")


# =========================================================================== B/C/D fixture
def tao_moi_truong(tmp: Path, n_video=8, n_frame=16) -> dict:
    from mot_datasets.make_bdd5_train_fixture import tao_fixture
    info = tao_fixture(tmp / "bdd", n_video=n_video, n_frame=n_frame, W=160, H=90, seed=0)
    val = tao_fixture(tmp / "bddval_tam", n_video=2, n_frame=6, W=160, H=90, seed=7)
    vdir = tmp / "bdd" / "images" / "track" / "val"
    shutil.move(str(tmp / "bddval_tam" / "images" / "track" / "train"), str(vdir))
    (tmp / "bdd" / "labels_cocovid" / "val_cocofmt.json").write_text(
        (tmp / "bddval_tam" / "labels_cocovid" / "fx_box_track_train_cocofmt.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    (tmp / "detrac_xml").mkdir()
    (tmp / "detrac_xml" / "MVI_00001.xml").write_text("<x/>", encoding="utf-8")
    return {"info": info, "val": val}


def cfg_mau(tmp: Path, ten="t", dinh_dang="scalabel", buoc=1, **them) -> dict:
    nhan = ({"dinh_dang": "scalabel", "duong_dan": "bdd/labels/box_track_20/train"} if dinh_dang == "scalabel" else
            {"dinh_dang": "cocovid", "duong_dan": "bdd/labels_cocovid/fx_box_track_train_cocofmt.json"})
    c = {"ten": ten, "goc_du_an": str(tmp), "nguon": ["bdd100k_mot_train"],
         "lop": ["car", "truck", "bus", "motorcycle", "bicycle"],
         "bdd": {"split": "train", "anh": "bdd/images/track/train", "nhan": nhan, "chi_video_du_anh": False},
         "chia": {"cach": "theo_video", "ti_le_dev": 0.34, "seed": 0, "phan_tang_theo": ["motorcycle", "bicycle"]},
         "lay_mau": {"buoc_frame": buoc}, "file_split": f"splits/{ten}.json", "dau_ra": f"out/{ten}",
         "thu_muc_hash_tap_danh_gia": "hash_cache",
         "tap_con": {"smoke": {"so_video_train": 2, "so_video_dev": 1, "buoc_frame": 4, "seed": 0}},
         "tap_danh_gia_cam": [{"ten": "bdd_val_gia", "loai": "bdd_mot", "nhan_cocovid": "bdd/labels_cocovid/val_cocofmt.json",
                               "anh": "bdd/images/track/val"},
                              {"ten": "detrac_test_gia", "loai": "detrac_xml", "xml": "detrac_xml"}]}
    for k, v in them.items():
        c[k] = v
    return c


def nap_cfg(tmp: Path, c: dict):
    import bdd_yolo_export as bx
    return bx.doc_cau_hinh(_yaml(tmp / f"cfg_{c['ten']}.yaml", c))


def nhom_bcd(tmp: Path):

    import bdd_yolo_export as bx
    import cv2
    env = tao_moi_truong(tmp)
    videos, NF = env["info"]["videos"], env["info"]["n_frame"]
    v0 = videos[0]

    # ---------------- B: Scalabel
    cfg = nap_cfg(tmp, cfg_mau(tmp, "sc"))
    kn = bx.kiem_nguon(cfg)
    _check("B1 kiem nguon: 8 video, 128 frame nhan, khong thieu anh",
           kn["so_video_nhan"] == 8 and kn["so_frame_nhan"] == 8 * NF and kn["so_anh_thieu"] == 0)
    fs = bx.p_abs(cfg, cfg["file_split"])
    sp = bx.tao_split(cfg, fs)
    tr, dv = set(sp["train_fit"]), set(sp["dev"])
    _check("B2 chia theo video: roi nhau, hop = toan bo video, dev > 0",
           not (tr & dv) and tr | dv == set(videos) and len(dv) > 0, f"train {len(tr)} dev {len(dv)}")
    tk = {v: bx.thong_ke_video(bx.mo_nguon(cfg["bdd"]["nhan"], tmp).doc_video(v)) for v in videos}
    lai = bx.chia_theo_video(tk, 0.34, 0)
    khac = any(bx.chia_theo_video(tk, 0.34, s)["dev"] != lai["dev"] for s in range(1, 8))
    _check("B3 chia tat dinh voi seed, doi seed doi ket qua", lai["dev"] == sp["dev"] and khac)
    ok_tang = all(t["so_dev"] == int(np.floor(0.34 * t["so_video"] + 0.5)) for t in sp["chia"]["tang"].values())
    moto_dev = sum(1 for v in dv if tk[v]["motorcycle"] > 0)
    _check("B4 phan tang theo motorcycle/bicycle: so dev moi tang = floor(ti_le * n + 0.5), dev co motorcycle",
           ok_tang and moto_dev >= 1, f"tang {sp['chia']['tang']}")
    try:
        bx.tao_split(cfg, fs)
        _check("B5 khoa split: tao lai -> tu choi ghi de", False)
    except bx.LoiDuLieu:
        _check("B5 khoa split: tao lai -> tu choi ghi de", True)

    tom = bx.xuat_dataset(cfg, fs)
    out = bx.p_abs(cfg, cfg["dau_ra"])
    rows = list(csv.DictReader(open(out / "manifest" / "anh.csv", encoding="utf-8")))
    split_v0 = "train" if v0 in tr else "dev"
    r0 = {int(r["frame_idx"]): r for r in rows if r["video"] == v0}
    _check("B6 kiem sau xuat dat, moi anh co file nhan, so anh = frame hop le",
           tom["kiem_sau_xuat"]["loi"] == 0 and len(rows) == 8 * NF - 3
           and all((out / "labels" / r["split"] / (Path(r["anh_ra"]).stem + ".txt")).is_file() for r in rows),
           f"{len(rows)} anh")
    lbl = lambda i: (out / "labels" / split_v0 / f"{v0}-{i + 1:07d}.txt").read_text(encoding="utf-8")  # noqa: E731
    _check("B7 am tinh HOP LE: frame chi co pedestrian + frame rong trong khoang co nhan -> giu, nhan rong",
           3 in r0 and 4 in r0 and r0[3]["am_tinh"] == "1" and r0[4]["am_tinh"] == "1" and lbl(3) == "" and lbl(4) == "")
    _check("B8 duoi video KHONG nhan (3 frame cuoi) -> loai, dem ngoai_khoang_co_nhan = 3",
           all(i not in r0 for i in range(NF - 3, NF)) and tom["dem_frame"]["ngoai_khoang_co_nhan"] == 3)
    dong = [list(map(float, ln.split())) for ln in lbl(0).splitlines()]
    W, H = 160, 90
    tran = [d for d in dong if d[0] == 0 and abs(d[1] + d[3] / 2 - 1.0) < 1e-6]
    _check("B9 box tran khung duoc CAT vao khung (x2 = W)", len(tran) == 1 and tom["dem_nhan"]["box_bi_cat_vao_khung"] > 0)
    _check("B10 box < 1 px sau cat -> bo, dem rieng", tom["dem_nhan"]["box_nho_hon_1px_sau_cat"] >= NF - 5
           and not any(d[0] == 4 and d[3] * W < 1 for d in dong))
    c803 = [d for d in dong if d[0] == 0 and abs(d[1] - 0.6) < 1e-6 and abs(d[2] - 0.67) < 1e-6]
    _check("B11 toa do theo kich thuoc anh thuc (car 0.55W..0.65W -> cx 0.6, w 0.1)",
           len(c803) == 1 and abs(c803[0][3] - 0.1) < 1e-6 and abs(c803[0][4] - 0.1) < 1e-6)
    src = cv2.imread(str(tmp / "bdd" / "images" / "track" / "train" / v0 / f"{v0}-0000001.jpg"))
    ra = cv2.imread(str(out / "images" / split_v0 / f"{v0}-0000001.jpg"))
    co_nhan = np.zeros((H, W), np.uint8)                                   # pixel thuoc box DA gan nhan
    for _c, cx, cy, bw, bh in dong:
        co_nhan[int((cy - bh / 2) * H):int(np.ceil((cy + bh / 2) * H)), int((cx - bw / 2) * W):int(np.ceil((cx + bw / 2) * W))] = 1

    def diem_xa_nhat(fx1, fy1, fx2, fy2):
        """Pixel trong vung [f*W, f*H] va NGOAI moi box co nhan, xa bien nhat (tranh vien JPEG)."""
        m = np.zeros((H, W), np.uint8)
        m[int(fy1 * H):int(fy2 * H), int(fx1 * W):int(fx2 * W)] = 1
        m[co_nhan > 0] = 0
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        return int(y), int(x), float(d[y, x])

    xam = lambda y, x: bool(np.all(np.abs(ra[y, x].astype(int) - 114) <= 10))  # noqa: E731
    giu = lambda y, x: bool(np.all(np.abs(ra[y, x].astype(int) - src[y, x].astype(int)) <= 12))  # noqa: E731
    vung_xam = [diem_xa_nhat(0.5, 0.6, 0.8, 0.8), diem_xa_nhat(0.02, 0.8, 0.1, 0.9), diem_xa_nhat(0.85, 0.2, 0.95, 0.3)]
    tam803 = (int(0.67 * H), int(0.6 * W))
    _check("B12 to xam: crowd car (ngoai box co nhan), 'other vehicle', 'trailer' -> xam; car co nhan nam trong "
           "crowd, crowd pedestrian -> GIU nguyen",
           all(dd >= 2.5 and xam(y, x) for y, x, dd in vung_xam)
           and giu(*tam803) and not xam(*tam803) and giu(int(0.35 * H), int(0.41 * W)),
           f"diem kiem {vung_xam}")
    ign = (out / "ignore" / split_v0 / f"{v0}-0000001.txt").read_text(encoding="utf-8").splitlines()
    _check("B13 file vung bo qua: 3 vung (crowd car, other vehicle, trailer)", len(ign) == 3, str(len(ign)))
    sc = [r for r in rows if r["xu_ly"] == "sao_chep"]
    _check("B14 anh khong to xam -> sao chep nguyen byte", len(sc) > 0 and all(r["sha256_ra"] == r["sha256_nguon"] for r in sc))
    import yaml
    y = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    _check("B15 data.yaml: val -> images/dev (dev noi bo), names = 5 lop da khoa",
           y["val"] == "images/dev" and y["train"] == "images/train" and list(y["names"].values()) == list(bx.LOP_5)
           and Path(y["path"]) == out.resolve())
    _check("B16 thong ke theo lop co motorcycle (box, anh, video) tren train va dev",
           all(tom["thong_ke"][s]["box"]["motorcycle"] >= 0 for s in ("train", "dev"))
           and tom["thong_ke"]["train"]["video_co_lop"]["motorcycle"] + tom["thong_ke"]["dev"]["video_co_lop"]["motorcycle"]
           == sum(1 for v in videos if tk[v]["motorcycle"] > 0))
    _check("B17 kiem trung tap danh gia co chay (ten video + ten anh + noi dung, 12 anh val da bam)",
           tom["kiem_trung"]["bdd_val_gia"]["so_anh_da_bam"] == 12 and tom["kiem_trung"]["detrac_test_gia"]["so_video_tap"] == 1)
    smk = bx.xuat_dataset(cfg, fs, "smoke")
    _check("B18 tap con smoke: dung so video, buoc frame rieng, ghi loai SMOKE",
           smk["video"] == {"train": 2, "dev": 1} and smk["quy_tac"]["buoc_frame"] == 4 and "SMOKE" in smk["loai"])

    # ---------------- C: COCO-video
    cfg_c = nap_cfg(tmp, cfg_mau(tmp, "co", dinh_dang="cocovid"))
    fs_c = bx.p_abs(cfg_c, cfg_c["file_split"])
    sp_c = bx.tao_split(cfg_c, fs_c)
    _check("C1 COCO-video: cung split train/dev nhu Scalabel", sp_c["train_fit"] == sp["train_fit"] and sp_c["dev"] == sp["dev"])
    tom_c = bx.xuat_dataset(cfg_c, fs_c)
    out_c = bx.p_abs(cfg_c, cfg_c["dau_ra"])
    giong = all((out / sub / s / p.name).read_bytes() == p.read_bytes()
                for sub in ("labels", "ignore", "images") for s in ("train", "dev") for p in (out_c / sub / s).iterdir())
    so = all(len(list((out / sub / s).iterdir())) == len(list((out_c / sub / s).iterdir()))
             for sub in ("labels", "ignore", "images") for s in ("train", "dev"))
    _check("C2 COCO-video: nhan, vung bo qua, anh xuat GIONG TUNG BYTE ban Scalabel", giong and so)
    _check("C3 COCO-video: cung thong ke + dem", tom_c["thong_ke"] == tom["thong_ke"] and tom_c["dem_nhan"] == tom["dem_nhan"])

    # ---------------- D: thieu / chay nham
    c_thieu = cfg_mau(tmp, "thieu")
    c_thieu["bdd"]["nhan"]["duong_dan"] = "khong_co/labels/box_track_20/train"
    c_thieu["bdd"]["anh"] = "khong_co/images/track/train"
    p_thieu = _yaml(tmp / "cfg_thieu.yaml", c_thieu)
    r = subprocess.run([PY, str(SRC / "prepare_mixed_yolo_dataset.py"), "--config", str(p_thieu), "--buoc", "xuat"],
                       capture_output=True, text=True)
    _check("D1 thieu nhan + anh train -> exit 2, liet ke duong dan, KHONG tao thu muc dich",
           r.returncode == 2 and "khong_co" in r.stderr and not (tmp / "out" / "thieu").exists()
           and not (tmp / "out" / "thieu.dang_tao").exists(), f"exit {r.returncode}")
    anh_mat = tmp / "bdd" / "images" / "track" / "train" / videos[5] / f"{videos[5]}-0000003.jpg"
    tam = anh_mat.with_suffix(".bak")
    anh_mat.rename(tam)
    try:
        try:
            bx.kiem_nguon(nap_cfg(tmp, cfg_mau(tmp, "mat")))
            _check("D2 thieu 1 anh le -> ThieuDuLieu (chi_video_du_anh=false)", False)
        except bx.ThieuDuLieu as e:
            _check("D2 thieu 1 anh le -> ThieuDuLieu (chi_video_du_anh=false)", any(videos[5] in m for m in e.thieu))
        c2 = cfg_mau(tmp, "mat2")
        c2["bdd"]["chi_video_du_anh"] = True
        cf2 = nap_cfg(tmp, c2)
        sp2 = bx.tao_split(cf2, bx.p_abs(cf2, cf2["file_split"]))
        _check("D3 chi_video_du_anh=true -> loai video thieu anh, GHI vao file split",
               sp2["video_bi_loai_vi_thieu_anh"] == [videos[5]] and videos[5] not in sp2["train_fit"] + sp2["dev"])
    finally:
        tam.rename(anh_mat)
    trung_v = tmp / "bdd" / "images" / "track" / "val" / videos[2]
    trung_v.mkdir()
    try:
        cf3 = nap_cfg(tmp, cfg_mau(tmp, "trungvideo"))
        bx.tao_split(cf3, bx.p_abs(cf3, cf3["file_split"]))
        _check("D4 video train trung ten video tap danh gia -> LoiDuLieu", False)
    except bx.LoiDuLieu as e:
        _check("D4 video train trung ten video tap danh gia -> LoiDuLieu", videos[2] in str(e))
    finally:
        trung_v.rmdir()
    vval = env["val"]["videos"][0]
    ban_sao = tmp / "bdd" / "images" / "track" / "val" / vval / "ban_sao_noi_dung.jpg"
    shutil.copyfile(tmp / "bdd" / "images" / "track" / "train" / v0 / f"{v0}-0000002.jpg", ban_sao)
    try:
        cf4 = nap_cfg(tmp, cfg_mau(tmp, "trungnd"))
        shutil.copyfile(fs, bx.p_abs(cf4, cf4["file_split"]))
        bx.xuat_dataset(cf4, bx.p_abs(cf4, cf4["file_split"]))
        _check("D5 anh train trung NOI DUNG anh tap danh gia (khac ten) -> LoiDuLieu", False)
    except bx.LoiDuLieu as e:
        _check("D5 anh train trung NOI DUNG anh tap danh gia (khac ten) -> LoiDuLieu", "noi dung" in str(e))
    finally:
        ban_sao.unlink()
    cf5 = nap_cfg(tmp, cfg_mau(tmp, "doinhan"))
    fs5 = bx.p_abs(cf5, cf5["file_split"])
    shutil.copyfile(fs, fs5)
    jp = tmp / "bdd" / "labels" / "box_track_20" / "train" / f"{videos[1]}.json"
    goc_json = jp.read_bytes()
    jp.write_bytes(goc_json.replace(b'"x1": ', b'"x1":  ', 1))
    try:
        bx.xuat_dataset(cf5, fs5)
        _check("D6 nguon nhan doi sau khi khoa split -> tu choi xuat", False)
    except bx.LoiDuLieu as e:
        _check("D6 nguon nhan doi sau khi khoa split -> tu choi xuat", "da doi" in str(e))
    finally:
        jp.write_bytes(goc_json)
    sp_sua = json.loads(fs.read_text(encoding="utf-8"))
    sp_sua["dev"] = sp_sua["dev"][:-1]
    sp_sua["train_fit"] = sorted(sp_sua["train_fit"] + [sp["dev"][-1]])
    fs6 = bx.p_abs(cf5, "splits/sua_tay.json")
    fs6.write_text(json.dumps(sp_sua), encoding="utf-8")
    try:
        bx.xuat_dataset(cf5, fs6)
        _check("D7 file split bi sua tay (doi video train/dev) -> tu choi", False)
    except bx.LoiDuLieu as e:
        _check("D7 file split bi sua tay (doi video train/dev) -> tu choi", "sha256_danh_sach" in str(e))
    try:
        bx.xuat_dataset(cfg, fs)
        _check("D8 thu muc dataset dich da ton tai -> tu choi ghi de", False)
    except bx.LoiDuLieu as e:
        _check("D8 thu muc dataset dich da ton tai -> tu choi ghi de", "da ton tai" in str(e))
    xoa = next((out / "labels" / "train").glob("*.txt"))
    noi_dung = xoa.read_bytes()
    xoa.unlink()
    try:
        bx.kiem_dataset(out)
        _check("D9 anh THIEU file nhan -> kiem dataset bao loi (khong coi la am tinh)", False)
    except bx.LoiDuLieu as e:
        _check("D9 anh THIEU file nhan -> kiem dataset bao loi (khong coi la am tinh)",
               "THIEU file nhan" in str(e) or "KHONG co file nhan" in str(e))     # cau chu vong 28 / vong 27
    finally:
        xoa.write_bytes(noi_dung)
    r1 = subprocess.run([PY, str(SRC / "prepare_mixed_yolo_dataset.py")], capture_output=True, text=True)
    r2 = subprocess.run([PY, str(SRC / "prepare_mixed_yolo_dataset.py"), "--nguon", "detrac", "bdd"],
                        capture_output=True, text=True)
    _check("D10 CLI cu: khong con nguon mac dinh (tron UA-DETRAC), --nguon bdd bi chan -> exit 1",
           r1.returncode == 1 and r2.returncode == 1 and "mac dinh" in r1.stderr and "--config" in r2.stderr)
    c_tron = cfg_mau(tmp, "tron")
    c_tron["nguon"] = ["bdd100k_mot_train", "ua_detrac_train"]
    try:
        nap_cfg(tmp, c_tron)
        _check("D11 cau hinh tron nguon (BDD + UA-DETRAC) -> tu choi", False)
    except bx.LoiDuLieu:
        _check("D11 cau hinh tron nguon (BDD + UA-DETRAC) -> tu choi", True)
    return out


# =========================================================================== E
def nhom_e(tmp: Path, out_sc: Path | None):
    import eval_detector_5lop as ev
    rng = np.random.default_rng(3)
    n_img, K = 30, 5
    gts, dds, cg, cd = [], [], {"images": [], "annotations": [], "categories": []}, []
    cg["categories"] = [{"id": k + 1, "name": str(k)} for k in range(K)]
    aid = 0
    for i in range(n_img):
        W, H = 640, 360
        cg["images"].append({"id": i + 1, "width": W, "height": H})
        n = int(rng.integers(0, 7))
        cls = rng.choice([0, 1, 3], size=n)                       # lop 2 (bus) va 4 (bicycle) KHONG co GT
        xy = rng.uniform(0, [W - 60, H - 60], size=(n, 2))
        wh = rng.uniform(15, 60, size=(n, 2))
        g = np.concatenate([xy, xy + wh], 1) if n else np.zeros((0, 4))
        ign = np.array([[10.0, 10.0, 90.0, 70.0]]) if i % 4 == 0 else np.zeros((0, 4))
        gts.append({"cls": cls.astype(int), "xyxy": g, "ign": ign})
        for c, b in zip(cls, g):
            aid += 1
            cg["annotations"].append({"id": aid, "image_id": i + 1, "category_id": int(c) + 1, "iscrowd": 0,
                                      "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        for b in ign:
            for k in range(K):
                aid += 1
                cg["annotations"].append({"id": aid, "image_id": i + 1, "category_id": k + 1, "iscrowd": 1,
                                          "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        dx, dcl = [], []
        for c, b in zip(cls, g):
            if rng.random() < 0.85:
                dx.append(b + rng.normal(0, 4, 4))
                dcl.append(int(c) if rng.random() < 0.9 else 2)
        for _ in range(int(rng.integers(0, 5))):
            p = rng.uniform(0, [W - 50, H - 50])
            dx.append(np.concatenate([p, p + rng.uniform(10, 50, 2)]))
            dcl.append(int(rng.integers(0, K)))
        if i % 4 == 0:
            dx.append(np.array([20.0, 15.0, 60.0, 55.0]))         # nam trong vung bo qua
            dcl.append(1)
        dx = np.array(dx).reshape(-1, 4)
        dx[:, 2:] = np.maximum(dx[:, 2:], dx[:, :2] + 1)
        conf = rng.uniform(0.01, 0.99, len(dx))
        dds.append({"xyxy": dx, "conf": conf, "cls": np.array(dcl, int)})
        for b, s, c in zip(dx, conf, dcl):
            cd.append({"image_id": i + 1, "category_id": int(c) + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                       "score": float(s)})
    kq = ev.danh_gia(gts, dds)
    try:
        import contextlib
        import io

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = cg
            gt.createIndex()
            dt_ = gt.loadRes(cd)
            E = COCOeval(gt, dt_, "bbox")
            E.params.maxDets = [1, 10, 1000]
            E.evaluate()
            E.accumulate()
        pr = E.eval["precision"]                                  # [T, R, K, A, M]
        lech = 0.0
        for k, ten in enumerate(ev.LOP_5):
            p = pr[:, :, k, 0, 2]
            if (p > -1).any():
                ap_t = p.mean(1)
                lech = max(lech, abs(ap_t[0] - kq["theo_lop"][ten]["AP50"]), abs(ap_t.mean() - kq["theo_lop"][ten]["AP50_95"]))
        _check("E1 AP50 / AP50-95 theo lop KHOP pycocotools (co vung bo qua kieu crowd)", lech < 1e-9, f"lech max {lech:.2e}")
    except ImportError as e:
        _skip("E1 AP50 / AP50-95 theo lop khop pycocotools", f"khong import duoc pycocotools: {e}")
    _check("E2 lop khong co GT (bus, bicycle) -> AP null, khong vao mAP",
           np.isnan(kq["theo_lop"]["bus"]["AP50"]) and kq["macro"]["so_lop_co_gt"] == 3
           and kq["macro"]["lop_khong_co_gt"] == ["bus", "bicycle"])
    g = [{"cls": np.array([0, 0, 1]), "xyxy": np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 70, 70]], float),
          "ign": np.array([[100, 100, 200, 200]], float)}]
    d = [{"xyxy": np.array([[0, 0, 10, 10], [21, 21, 31, 31], [0, 0, 9, 9], [120, 120, 140, 140], [50, 50, 70, 70],
                            [300, 300, 310, 310]], float),
          "conf": np.array([0.9, 0.5, 0.3, 0.8, 0.2, 0.6]), "cls": np.array([0, 0, 0, 0, 1, 0])}]
    k2 = ev.danh_gia(g, d, conf_van_hanh=0.25)
    car = k2["theo_lop"]["car"]["van_hanh"]
    truck = k2["theo_lop"]["truck"]["van_hanh"]
    _check("E3 P/R van hanh tinh tay: car TP2 FP2 (det trung + det xa; det trong vung bo qua KHONG tinh), "
           "truck conf 0.2 < 0.25 -> FN",
           (car["TP"], car["FP"], car["FN"]) == (2, 2, 0) and abs(car["precision"] - 0.5) < 1e-12
           and (truck["TP"], truck["FP"], truck["FN"]) == (0, 0, 1) and truck["recall"] == 0.0, f"{car} {truck}")
    import cv2
    dd = tmp / "eval_ds" / "images" / "dev"
    dd.mkdir(parents=True)
    (tmp / "eval_ds" / "labels" / "dev").mkdir(parents=True)
    cv2.imwrite(str(dd / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp / "eval_ds" / "labels" / "dev" / "a.txt").write_text("3 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    ga = ev.doc_gt(dd / "a.jpg")
    _check("E4 GT doi ra pixel theo kich thuoc ANH THUC (200x100), khong theo 960x540",
           ga["w"] == 200 and ga["h"] == 100 and np.allclose(ga["xyxy"][0], [80, 30, 120, 70]) and ga["cls"][0] == 3)
    cv2.imwrite(str(dd / "b.jpg"), np.zeros((100, 200, 3), np.uint8))
    try:
        ev.doc_gt(dd / "b.jpg")
        _check("E5 anh thieu file nhan -> LOI (khong coi la anh am tinh)", False)
    except FileNotFoundError:
        _check("E5 anh thieu file nhan -> LOI (khong coi la anh am tinh)", True)
    w = _models_dir() / "yolov8n.pt"
    if not (w.is_file() and out_sc is not None):
        _skip("E6 du doan that YOLOv8n COCO (CPU)", f"khong co {w} hoac dataset fixture")
    else:
        anh = ev.ds_anh_tu_data_yaml(out_sc / "data.yaml", "val")[:3]
        dd_, mt = ev.du_doan_ultralytics(str(w), anh, ev.QUY_TAC["du_doan"], batch=3, device="cpu")
        lop = set(np.concatenate([x["cls"] for x in dd_]).tolist()) if dd_ else set()
        _check("E6 du doan that YOLOv8n COCO (CPU): anh xa theo ten ghi lai, lop dau ra chi trong 0..4",
               mt["loai_checkpoint"] == "coco80" and mt["model_id_sang_lop5"] == {"1": 4, "2": 0, "3": 3, "5": 2, "7": 1}
               and lop <= set(range(5)) and len(dd_) == 3, f"lop {sorted(lop)}")


# =========================================================================== F
def cfg_train_tuyet_doi() -> dict:
    """train_bdd5.yaml voi duong dan weights TUYET DOI (tu _models_dir) -- dung duoc ca tren ban sao ma cu."""
    import yaml
    c = yaml.safe_load((config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5.yaml").read_text(encoding="utf-8"))
    c["model"]["yolov8n"]["weights"] = str(_models_dir() / "yolov8n.pt")
    c["model"]["yolo11n"]["weights"] = str(_models_dir() / "candidates" / "yolo11n.pt")
    return c


def nhom_f(tmp: Path, out_sc: Path | None):
    if out_sc is None:
        _check("F0 can dataset fixture tu nhom B", False)
        return
    goc = cfg_train_tuyet_doi()
    pcfg = _yaml(tmp / "train_cfg.yaml", goc)
    base = [PY, str(SRC / "train_detector.py"), "--config", str(pcfg), "--model", "yolov8n", "--data",
            str(out_sc / "data.yaml"), "--project", str(tmp / "runs"), "--ket-qua-dir", str(tmp / "kq")]
    w_ok = (_models_dir() / "yolov8n.pt").is_file()
    if not w_ok:
        _skip("F1..F5 runner qua subprocess", f"khong co {_models_dir() / 'yolov8n.pt'}")   # SKIP, KHONG tinh PASS
        return
    r = subprocess.run(base + ["--dry-run"], capture_output=True, text=True)
    _check("F1 --dry-run = kiem cau hinh: exit 0, KHONG train, KHONG tao thu muc luot / file ket qua",
           r.returncode == 0 and "KHONG train" in r.stdout and not (tmp / "runs").exists() and not (tmp / "kq").exists()
           and '"optimizer": "SGD"' in r.stdout, f"exit {r.returncode} {r.stderr[-300:]}")
    (tmp / "runs" / "da_co").mkdir(parents=True)
    r = subprocess.run(base + ["--che-do", "smoke", "--ten", "da_co"], capture_output=True, text=True)
    _check("F2 thu muc luot da ton tai -> exit 2, khong ghi de", r.returncode == 2 and "da ton tai" in r.stderr)
    g2 = copy.deepcopy(goc)
    g2["huan_luyen"]["optimizer"] = "auto"
    p2 = _yaml(tmp / "train_auto.yaml", g2)
    r = subprocess.run([*base[:3], str(p2), *base[4:], "--dry-run"], capture_output=True, text=True)
    _check("F3 optimizer 'auto' -> tu choi (phai khai bao gia tri tuong minh)", r.returncode == 2 and "auto" in r.stderr)
    r = subprocess.run(base + ["--che-do", "day-du"], capture_output=True, text=True)
    _check("F4 day-du thieu --xac-nhan-train-dai -> exit 1", r.returncode == 1 and "xac-nhan" in r.stderr)
    g3 = copy.deepcopy(goc)
    g3["model"]["yolov8n"]["sha256"] = "0" * 64
    p3 = _yaml(tmp / "train_sha.yaml", g3)
    r = subprocess.run([*base[:3], str(p3), *base[4:], "--dry-run"], capture_output=True, text=True)
    _check("F5 sha256 checkpoint khoi tao sai -> exit 2", r.returncode == 2 and "sha256" in r.stderr)


# =========================================================================== E7-E9: ghep COCO (R27-F3)
def _coco_theo_lop(gts: list[dict], dds: list[dict], K: int = 5):
    """Chay pycocotools THAT: {lop: {AP50, AP50_95, TP50}}; vung bo qua -> GT iscrowd cho moi lop. None neu thieu thu vien."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    import contextlib
    import io
    cg = {"images": [], "annotations": [], "categories": [{"id": k + 1, "name": str(k)} for k in range(K)]}
    cd, aid = [], 0
    for i, (g, p) in enumerate(zip(gts, dds)):
        cg["images"].append({"id": i + 1, "width": 1000, "height": 1000})
        for c, b in zip(g["cls"], g["xyxy"]):
            aid += 1
            cg["annotations"].append({"id": aid, "image_id": i + 1, "category_id": int(c) + 1, "iscrowd": 0,
                                      "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                                      "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        for b in g["ign"]:
            for k in range(K):
                aid += 1
                cg["annotations"].append({"id": aid, "image_id": i + 1, "category_id": k + 1, "iscrowd": 1,
                                          "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                                          "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        for b, s, c in zip(p["xyxy"], p["conf"], p["cls"]):
            cd.append({"image_id": i + 1, "category_id": int(c) + 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                       "score": float(s)})
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO()
        G.dataset = cg
        G.createIndex()
        E = COCOeval(G, G.loadRes(cd), "bbox")
        E.params.maxDets = [1, 10, 1000]
        E.evaluate()
        E.accumulate()
    pr = E.eval["precision"]
    out = {}
    for k in range(K):
        p = pr[:, :, k, 0, 2]
        if (p > -1).any():
            tp50 = sum(int(((e["dtMatches"][0] > 0) & ~np.asarray(e["dtIgnore"][0], bool)).sum()) for e in E.evalImgs
                       if e is not None and e["category_id"] == k + 1 and list(e["aRng"]) == list(E.params.areaRng[0])
                       and e["maxDet"] == 1000)
            out[k] = {"AP50": float(p[0].mean()), "AP50_95": float(p.mean(1).mean()), "TP50": tp50}
    return out


def nhom_e2():
    import eval_detector_5lop as ev

    def so_voi_coco(ten, gts, dds, lop, mong_doi_tp50):
        kq = ev.danh_gia(gts, dds)
        r = kq["theo_lop"][ev.LOP_5[lop]]
        tp50_ta = int(sum(ev.ghep_anh_lop(p["xyxy"][p["cls"] == lop], p["conf"][p["cls"] == lop],
                                          g["xyxy"][g["cls"] == lop], g["ign"])[1][0].sum() for g, p in zip(gts, dds)))
        _check(f"{ten}: TP tai IoU 0,50 cua evaluator = {mong_doi_tp50}", tp50_ta == mong_doi_tp50,
               f"evaluator {tp50_ta}; AP50 {r['AP50']:.10f}; AP50-95 {r['AP50_95']:.10f}")
        c = _coco_theo_lop(gts, dds)
        if c is None:
            _skip(f"{ten}: doi chieu pycocotools", "khong import duoc pycocotools")
            return
        rc = c[lop]
        ok = (abs(rc["AP50"] - r["AP50"]) < 1e-12 and abs(rc["AP50_95"] - r["AP50_95"]) < 1e-12
              and rc["TP50"] == tp50_ta)
        _check(f"{ten}: AP50, AP50-95, TP@0,50 KHOP pycocotools (chay that)", ok,
               f"pycocotools AP50 {rc['AP50']:.10f} AP50-95 {rc['AP50_95']:.10f} TP {rc['TP50']}")

    z = np.zeros((0, 4))
    # E7 -- fixture cua review: det 0,9 co IoU 2/3 voi CA HAI GT; pycocotools ghep GT SAU -> det 0,8 ghep GT dau
    so_voi_coco("E7 hoa IoU (fixture review 4937c85)",
                [{"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float), "ign": z}],
                [{"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 15, 10], [0, 0, 10, 10]], float),
                  "conf": np.array([0.9, 0.8])}], 0, 2)
    kq7 = ev.danh_gia([{"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float), "ign": z}],
                      [{"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 15, 10], [0, 0, 10, 10]], float),
                        "conf": np.array([0.9, 0.8])}])["theo_lop"]["car"]
    _check("E7 hoa IoU: van hanh TP 2 / FP 0 / FN 0, AP50 = 1,0 (truoc sua: TP 1 / FP 1, AP50 0,50495)",
           (kq7["van_hanh"]["TP"], kq7["van_hanh"]["FP"], kq7["van_hanh"]["FN"]) == (2, 0, 0) and kq7["AP50"] == 1.0,
           f"{kq7['van_hanh']} AP50 {kq7['AP50']}")
    # E8 -- IoU dung bang nguong (0,6 = 60/100 va 0,7 = 70/100): nguong float giong pycocotools
    so_voi_coco("E8 IoU dung bang nguong 0,6 / 0,7",
                [{"cls": np.array([1, 1]), "xyxy": np.array([[0, 0, 10, 10], [50, 0, 60, 10]], float), "ign": z}],
                [{"cls": np.array([1, 1]), "xyxy": np.array([[0, 0, 10, 6], [50, 0, 60, 7]], float),
                  "conf": np.array([0.9, 0.8])}], 1, 2)
    # E9 -- GT HOP LE nam trong vung bo qua: det khop GT la TP (khong bi bo qua); det khac trong vung -> bo qua
    g9 = [{"cls": np.array([0]), "xyxy": np.array([[100, 100, 150, 150]], float),
           "ign": np.array([[90, 90, 200, 200]], float)}]
    d9 = [{"cls": np.array([0, 0, 0]), "xyxy": np.array([[100, 100, 150, 150], [160, 160, 190, 190], [300, 300, 340, 340]],
                                                         float), "conf": np.array([0.9, 0.8, 0.7])}]
    so_voi_coco("E9 GT hop le chong vung ignore", g9, d9, 0, 1)
    v9 = ev.danh_gia(g9, d9)["theo_lop"]["car"]["van_hanh"]
    _check("E9 van hanh: det khop GT trong vung ignore = TP, det chi trong vung = bo qua, det xa = FP",
           (v9["TP"], v9["FP"], v9["FN"]) == (1, 1, 0), str(v9))


# =========================================================================== G: runner qua entry point (R27-F1)
class _SpyDungTrain(RuntimeError):
    """Spy chan training that tai ranh gioi model.train (GIA LAP)."""


class _SpyYOLO:
    """GIA LAP `ultralytics.YOLO` tai ranh gioi nap model / training: ghi lai moi lan khoi tao, gan callback, goi train;
    `train()` nem _SpyDungTrain thay vi huan luyen. Moi thu truoc ranh gioi (runner, hop dong dataset) chay THAT."""
    goi: list = []
    names_coco: dict = {}

    def __init__(self, w, *a, **k):
        _SpyYOLO.goi.append(("khoi_tao", str(w)))
        self.names = dict(_SpyYOLO.names_coco)

    def add_callback(self, su_kien, *a, **k):
        _SpyYOLO.goi.append(("callback", su_kien))

    def train(self, **kw):
        _SpyYOLO.goi.append(("train", str(kw.get("data"))))
        raise _SpyDungTrain("spy: chan training that (ranh gioi gia lap)")


def _chay_runner_spy(argv: list[str]) -> tuple[int, str, str, list]:
    import contextlib
    import io

    import ultralytics
    import train_detector
    import yaml
    if not _SpyYOLO.names_coco:
        _SpyYOLO.names_coco = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml")
                                             .read_text(encoding="utf-8"))["names"]
    goc = ultralytics.YOLO
    _SpyYOLO.goi = []
    ultralytics.YOLO = _SpyYOLO
    o, e, argv_cu = io.StringIO(), io.StringIO(), sys.argv
    sys.argv = ["train_detector.py", *argv]
    try:
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            code = train_detector.main()
    finally:
        sys.argv = argv_cu
        ultralytics.YOLO = goc
    return code, o.getvalue(), e.getvalue(), list(_SpyYOLO.goi)


def nhom_g(tmp: Path, out: Path):
    if not (_models_dir() / "yolov8n.pt").is_file():
        _skip("G runner qua entry point", "khong co yolov8n.pt (can cho kiem sha256 checkpoint)")
        return
    pcfg = _yaml(tmp / "train_cfg_g.yaml", cfg_train_tuyet_doi())
    data = out / "data.yaml"

    def chay(ten):
        return _chay_runner_spy(["--config", str(pcfg), "--model", "yolov8n", "--che-do", "smoke", "--data", str(data),
                                 "--project", str(tmp / "runs_g"), "--ket-qua-dir", str(tmp / "kq_g"), "--ten", ten])

    def khong_cham(ten, goi):
        return (not any(g[0] in ("khoi_tao", "train") for g in goi) and not (tmp / "runs_g" / ten).exists()
                and not (tmp / "kq_g" / f"{ten}.json").exists())

    rows = list(csv.DictReader(open(out / "manifest" / "anh.csv", encoding="utf-8")))
    n_that = {s: len([p for p in (out / "images" / s).iterdir() if p.suffix == ".jpg"]) for s in ("train", "dev")}

    def g0():
        code, so, se, goi = chay("g0_hop_le")
        tr = [g for g in goi if g[0] == "train"]
        return (len(tr) == 1 and Path(tr[0][1]) == data and code == 3 and "spy: chan training" in so,
                f"exit {code} (spy chan train -> 3), goi train {len(tr)} lan")
    _thu("G0 dataset HOP LE -> runner di toi training (spy ghi 1 lan goi train voi dung data.yaml)", g0)

    def g0b():
        k = json.loads((tmp / "kq_g" / "g0_hop_le.json").read_text(encoding="utf-8"))["dataset"]
        return (k.get("hop_dong") == "bdd5-dataset-v2" and k.get("so_anh") == n_that
                and sum(k.get("so_nhan_rong_hop_le", {}).values()) > 0
                and k.get("noi_dung_da_xac_minh") == {"train": True, "dev": True},
                f"so_anh {k.get('so_anh')} / thuc te {n_that}, nhan rong hop le {k.get('so_nhan_rong_hop_le')}")
    _thu("G0b so anh bao cao = so anh ultralytics se dung; nhan RONG hop le duoc chap nhan; danh tinh da xac minh ghi vao "
         "ket qua", g0b)

    y_goc = data.read_bytes()

    def g1():
        data.write_bytes(y_goc.replace(b"train: images/train", b"train: images/dev"))
        try:
            code, so, se, goi = chay("g1_train_la_dev")
        finally:
            data.write_bytes(y_goc)
        return (code == 2 and khong_cham("g1_train_la_dev", goi) and "train/val" in se,
                f"exit {code}; spy {goi[:3]}; {se.strip()[-160:]}")
    _thu("G1 data.yaml `train: images/dev` -> runner exit 2, KHONG nap model, KHONG goi train, KHONG tao thu muc", g1)

    co_nhan = next(r for r in rows if r["split"] == "train" and int(r["am_tinh"]) == 0)
    rong = next(r for r in rows if int(r["am_tinh"]) == 1)
    for ma, r, mo_ta in (("g2_thieu_nhan", co_nhan, "anh CO box"), ("g3_thieu_nhan_rong", rong, "anh am tinh (nhan rong)")):
        def g2(ma=ma, r=r):
            f = out / r["nhan_ra"] if "nhan_ra" in r else out / "labels" / r["split"] / (Path(r["anh_ra"]).stem + ".txt")
            b = f.read_bytes()
            f.unlink()
            try:
                code, so, se, goi = chay(ma)
            finally:
                f.write_bytes(b)
            return (code == 2 and khong_cham(ma, goi) and "THIEU file nhan" in se, f"exit {code}; spy {goi[:3]}")
        _thu(f"G{ma[1]} THIEU file nhan cua {mo_ta} -> runner exit 2, khong nap model / train (thieu != rong)", g2)


# =========================================================================== H: toan ven noi dung (R27-F2)
def nhom_h(tmp: Path, out: Path, cfg_goc: dict):
    import bdd_yolo_export as bx
    import eval_detector_5lop as ev
    import train_detector as tr
    from PIL import Image
    data = out / "data.yaml"
    rows = list(csv.DictReader(open(out / "manifest" / "anh.csv", encoding="utf-8")))

    def tu_choi(fn) -> tuple[bool, str]:
        """(bi tu choi?, TOAN VAN thong diep) -- khop chuoi tren toan van, khong tren ban cat ngan."""
        try:
            fn()
            return False, "CHAP NHAN"
        except Exception as e:  # noqa: BLE001
            return True, f"{type(e).__name__}: {e}"

    def ca_hai(ten, bien_doi, khoi_phuc, can_co=""):
        def f():
            bien_doi()
            try:
                a, ma = tu_choi(lambda: bx.kiem_dataset(out))
                b, mb = tu_choi(lambda: tr.kiem_dataset(data))
            finally:
                khoi_phuc()
            return a and b and can_co in ma, f"exporter: {ma} | runner: {mb}"
        _thu(ten, f)

    tr_row = next(r for r in rows if r["split"] == "train" and int(r["am_tinh"]) == 0)
    img = out / tr_row["anh_ra"]
    b_img = img.read_bytes()

    def lat_anh(p=img):
        with Image.open(p) as im:
            im2 = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im2.save(p, format="JPEG")
    ca_hai("H1 anh DOI NOI DUNG, GIU kich thuoc (lat ngang) -> exporter + runner tu choi", lat_anh,
           lambda: img.write_bytes(b_img), "noi dung anh")

    dv_row = next(r for r in rows if r["split"] == "dev")
    img_dv = out / dv_row["anh_ra"]
    b_dv = img_dv.read_bytes()

    def h1b():
        lat_anh(img_dv)
        try:
            ok, m = tu_choi(lambda: ev.chuan_bi_danh_gia(data, "val"))
        finally:
            img_dv.write_bytes(b_dv)
        return ok and "noi dung anh" in m, m
    _thu("H1b anh DEV doi noi dung -> evaluator tu choi truoc khi danh gia", h1b)

    lbl = out / "labels" / "train" / (img.stem + ".txt")
    b_lbl = lbl.read_bytes()

    def sua_toa_do():
        dong = b_lbl.decode().splitlines()
        p = dong[0].split()
        dong[0] = f"{p[0]} 0.500000 0.500000 0.050000 0.050000"
        lbl.write_bytes(("\n".join(dong) + "\n").encode())
    ca_hai("H2 toa do nhan DOI, GIU lop + so box -> exporter + runner tu choi", sua_toa_do,
           lambda: lbl.write_bytes(b_lbl), "noi dung nhan")

    ig_row = next(r for r in rows if int(r["n_vung_to_xam"]) > 0)
    ig = out / "ignore" / ig_row["split"] / (Path(ig_row["anh_ra"]).stem + ".txt")    # suy tu anh_ra (chay ca manifest v1)
    b_ig = ig.read_bytes()
    ca_hai("H3 file ignore DA KHAI BAO bi XOA -> tu choi (khong doc thanh 0 vung)", ig.unlink,
           lambda: ig.write_bytes(b_ig), "THIEU file ignore")
    ca_hai("H4 file ignore DA KHAI BAO bi SUA -> tu choi",
           lambda: ig.write_bytes(b_ig + b"0.00 0.00 1.00 1.00\n"), lambda: ig.write_bytes(b_ig), "ignore")
    img_ig = out / ig_row["anh_ra"]

    def h3b():
        ig.unlink()
        try:
            ok, m = tu_choi(lambda: ev.doc_gt(img_ig, ignore_khai_bao=True))
        finally:
            ig.write_bytes(b_ig)
        return ok and "FileNotFoundError" in m, m
    _thu("H3b evaluator: ignore da khai bao ma thieu -> LOI (truoc sua: doc thanh 0 vung)", h3b)

    khong_ig = next(r for r in rows if int(r["n_vung_to_xam"]) == 0 and r["split"] == ig_row["split"])
    thua = out / "ignore" / khong_ig["split"] / (Path(khong_ig["anh_ra"]).stem + ".txt")

    def tao_thua():
        thua.write_bytes(b"1.00 1.00 5.00 5.00\n")
    ca_hai("H5 anh KHONG co vung ignore ma xuat hien file ignore -> tu choi (khac voi 'khong co vung')", tao_thua,
           lambda: thua.unlink(), "KHONG khai bao")

    def h5b():
        tao_thua()
        try:
            ok, m = tu_choi(lambda: ev.doc_gt(out / khong_ig["anh_ra"], ignore_khai_bao=False))
        finally:
            thua.unlink()
        return ok and "ValueError" in m, m
    _thu("H5b evaluator: anh khai bao 0 vung ma co file ignore -> LOI", h5b)

    def h6():
        k = bx.kiem_dataset(out)
        n_ig = sum(1 for r in rows if int(r["n_vung_to_xam"]) > 0)
        return (k["so_file_da_bam"] == {"anh": len(rows), "nhan": len(rows), "ignore": n_ig}
                and sum(k["so_nhan_rong_hop_le"].values()) > 0 and sum(k["so_anh_co_file_ignore"].values()) == n_ig,
                f"da bam {k['so_file_da_bam']}, nhan rong {k['so_nhan_rong_hop_le']}, co ignore {k['so_anh_co_file_ignore']}")
    _thu("H6 dataset hop le: bam du anh + nhan + ignore; nhan RONG hop le va anh KHONG ignore duoc chap nhan", h6)

    tom = out / "manifest" / "tom_tat.json"
    b_tom = tom.read_bytes()

    def h7():
        t = json.loads(b_tom)
        t.pop("phien_ban_manifest", None)
        moi = (json.dumps(t, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        tom.write_bytes(moi)
        try:
            ok, m = tu_choi(lambda: bx.kiem_dataset(out))
            khong_bu = tom.read_bytes() == moi
        finally:
            tom.write_bytes(b_tom)
        return ok and "phien ban 1" in m and khong_bu, f"{m} | tom_tat khong bi bo sung: {khong_bu}"
    _thu("H7 manifest PHIEN BAN 1 (khong khoa nhan/ignore) -> tu choi, KHONG tu bo sung hash", h7)

    them = out / "images" / "train" / "anh_them_ngoai_manifest.jpg"

    def h8():
        them.write_bytes(b_img)
        try:
            ok, m = tu_choi(lambda: tr.kiem_dataset(data))
        finally:
            them.unlink()
        return ok and "KHONG co trong manifest" in m, m
    _thu("H8 anh THUA trong images/train (ultralytics van se dung) -> runner tu choi", h8)
    y_goc = data.read_bytes()

    def h9():
        data.write_bytes(y_goc.replace(f"path: {out.resolve().as_posix()}".encode(), b"path: yolo_tuong_doi"))
        try:
            ok, m = tu_choi(lambda: tr.kiem_dataset(data))
        finally:
            data.write_bytes(y_goc)
        return ok and "path" in m, m
    _thu("H9 `path` tuong doi trong data.yaml (ultralytics resolve theo DATASETS_DIR) -> tu choi", h9)

    def h10():
        cfg = nap_cfg(tmp, cfg_mau(tmp, "trung_noi_bo"))
        fs = bx.p_abs(cfg, cfg["file_split"])
        shutil.copyfile(bx.p_abs(cfg_goc, cfg_goc["file_split"]), fs)
        sp = json.loads(fs.read_text(encoding="utf-8"))
        goc_anh = tmp / "bdd" / "images" / "track" / "train"
        a = goc_anh / sp["train_fit"][0] / f"{sp['train_fit'][0]}-0000002.jpg"
        b = goc_anh / sp["dev"][0] / f"{sp['dev'][0]}-0000002.jpg"
        b_cu = b.read_bytes()
        shutil.copyfile(a, b)
        try:
            ok, m = tu_choi(lambda: bx.xuat_dataset(cfg, fs))
        finally:
            b.write_bytes(b_cu)
        dich = bx.p_abs(cfg, cfg["dau_ra"])
        return (ok and "TRUNG NOI DUNG" in m and not dich.exists() and not dich.with_name(dich.name + ".dang_tao").exists(),
                m)
    _thu("H10 anh dev TRUNG NOI DUNG anh train-fit (khac video) -> xuat tu choi TRUOC khi ghi, khong tao thu muc", h10)


# =========================================================================== I: du lieu ultralytics THUC SU nap
def nhom_i(tmp: Path, out: Path):
    from types import SimpleNamespace

    import bdd_yolo_export as bx
    import train_detector as tr
    data = out / "data.yaml"

    def nap_gia(md):
        """Trainer GIA LAP: labels dung dinh dang YOLODataset.labels (im_file, cls (n,1), bboxes (n,4) xywh chuan hoa)."""
        def ds(s):
            return SimpleNamespace(labels=[{"im_file": k, "cls": v[:, :1].astype(np.float32),
                                            "bboxes": v[:, 1:].astype(np.float32)} for k, v in md[s].items()])
        return SimpleNamespace(train_loader=SimpleNamespace(dataset=ds("train")),
                               test_loader=SimpleNamespace(dataset=ds("dev")))

    def i1():
        md = tr.nhan_mong_doi(data)
        k = tr.kiem_dau_vao_da_nap(nap_gia(md), md)
        return k["train"]["nhan_khac"] == 0 and k["dev"]["thieu"] == 0, str(k)
    _thu("I1 (gia lap trainer) nhan da nap KHOP file nhan da xac minh -> cho qua", i1)

    def i2():
        md = tr.nhan_mong_doi(data)
        sai = {s: dict(v) for s, v in md.items()}
        k0 = next(k for k, v in sai["train"].items() if len(v))
        sai["train"][k0] = sai["train"][k0].copy()
        sai["train"][k0][0, 1] += 0.01
        try:
            tr.kiem_dau_vao_da_nap(nap_gia(sai), md)
            return False, "khong phat hien"
        except tr.LoiDauVaoDaNap as e:
            return "nhan khac 1" in str(e), str(e)[:200]
    _thu("I2 (gia lap) nhan da nap lech 1 box (vd cache cu) -> LoiDauVaoDaNap, dung truoc vong train", i2)

    def i3():
        md = tr.nhan_mong_doi(data)
        thieu = {s: dict(v) for s, v in md.items()}
        thieu["dev"].pop(next(iter(thieu["dev"])))
        try:
            tr.kiem_dau_vao_da_nap(nap_gia(thieu), md)
            return False, "khong phat hien"
        except tr.LoiDauVaoDaNap as e:
            return "thieu 1" in str(e), str(e)[:200]
    _thu("I3 (gia lap) ultralytics nap THIEU 1 anh (vd anh hong bi loai) -> LoiDauVaoDaNap", i3)
    try:
        from ultralytics.data.dataset import YOLODataset
    except ImportError as e:
        _skip("I4/I5 YOLODataset that", f"khong import duoc ultralytics: {e}")
        return
    names = {i: c for i, c in enumerate(bx.LOP_5 if hasattr(bx, "LOP_5") else ["car", "truck", "bus", "motorcycle",
                                                                             "bicycle"])}

    def yds(s):
        return YOLODataset(img_path=str(out / "images" / s), data={"names": names, "nc": 5, "channels": 3}, task="detect",
                           augment=False, imgsz=640)

    def xoa_cache():
        for s in ("train", "dev"):
            (out / "labels" / f"{s}.cache").unlink(missing_ok=True)

    def trainer_that():
        return SimpleNamespace(train_loader=SimpleNamespace(dataset=yds("train")),
                               test_loader=SimpleNamespace(dataset=yds("dev")))

    def i4():
        xoa_cache()
        rows = list(csv.DictReader(open(out / "manifest" / "anh.csv", encoding="utf-8")))
        r = next(x for x in rows if x["split"] == "train" and int(x["am_tinh"]) == 0)
        f = out / "labels" / "train" / (Path(r["anh_ra"]).stem + ".txt")
        goc_b = f.read_bytes()
        s = goc_b.decode()
        i = s.index(" ") + 3                      # chu so thu 2 sau dau thap phan cua cx -> doi, GIU so byte
        sua = (s[:i] + ("1" if s[i] != "1" else "2") + s[i + 1:]).encode()
        assert len(sua) == len(goc_b) and sua != goc_b
        try:
            f.write_bytes(sua)
            yds("train")                          # ultralytics tao labels/train.cache tu nhan DA SUA
        finally:
            f.write_bytes(goc_b)                  # khoi phuc: file dung manifest, cung kich thuoc -> cache van "hop le"
        k_hd = bx.kiem_dataset(out)               # hop dong file: DAT (file nhan da khop manifest)
        md = tr.nhan_mong_doi(data)
        try:
            tr.kiem_dau_vao_da_nap(trainer_that(), md)
            ket = (False, "ultralytics nap nhan tu cache cu nhung khong phat hien")
        except tr.LoiDauVaoDaNap as e:
            ket = ("nhan khac 1" in str(e) and k_hd["loi"] == 0, f"hop dong file DAT; nap: {str(e)[:160]}")
        finally:
            xoa_cache()
        return ket
    _thu("I4 (ultralytics THAT) cache nhan cu cung kich thuoc -> hop dong file dat nhung du lieu nap khac -> dung", i4)

    def i5():
        xoa_cache()
        try:
            k = tr.kiem_dau_vao_da_nap(trainer_that(), tr.nhan_mong_doi(data))
        finally:
            xoa_cache()
        return all(v["thieu"] == v["thua"] == v["nhan_khac"] == 0 for v in k.values()), str(k)
    _thu("I5 (ultralytics THAT) cache moi tu file da xac minh -> du lieu nap khop, cho qua", i5)


# =========================================================================== J: cache ANH .npy (R28-F1)
def nhom_j(tmp: Path, out: Path):
    """Cache ANH `<anh>.npy` -- khac cache NHAN `labels/<split>.cache` (nhom I). Nguon pixel kiem bang YOLODataset THAT;
    thu tu chan o runner kiem qua entry point voi SPY (gia lap ranh gioi nap model / training)."""
    from types import SimpleNamespace

    import bdd_yolo_export as bx
    import cv2
    import train_detector as tr
    try:
        from ultralytics.data.dataset import YOLODataset
    except ImportError as e:
        _skip("J cache anh .npy (YOLODataset that)", f"khong import duoc ultralytics: {e}")
        return
    data = out / "data.yaml"
    names = {i: c for i, c in enumerate(["car", "truck", "bus", "motorcycle", "bicycle"])}
    anh = {s: sorted((out / "images" / s).glob("*.jpg"))[0] for s in ("train", "dev")}
    imgsz = max(cv2.imread(str(anh["train"])).shape[:2])        # canh dai = imgsz -> load_image khong resize

    def yds(s):
        return YOLODataset(img_path=str(out / "images" / s), data={"names": names, "nc": 5, "channels": 3},
                           task="detect", augment=False, imgsz=imgsz, cache=False)

    def xoa_cache_nhan():
        for s in ("train", "dev"):
            (out / "labels" / f"{s}.cache").unlink(missing_ok=True)

    def dat_npy_den(s):
        jpg = cv2.imread(str(anh[s]))
        np.save(anh[s].with_suffix(".npy"), np.zeros_like(jpg))
        return jpg

    def go_npy(s):
        anh[s].with_suffix(".npy").unlink(missing_ok=True)

    def tu_choi(fn) -> tuple[bool, str]:
        try:
            fn()
            return False, "CHAP NHAN"
        except Exception as e:  # noqa: BLE001
            return True, f"{type(e).__name__}: {e}"

    def j0():
        xoa_cache_nhan()
        try:
            a, ma = tu_choi(lambda: bx.kiem_dataset(out))
            dts = {s: yds(s) for s in ("train", "dev")}
            sai = [f for ds in dts.values() for i, f in enumerate(ds.im_files)
                   if not np.array_equal(ds.get_image_and_label(i)["img"], cv2.imread(f))]
            n = sum(len(ds.im_files) for ds in dts.values())
        finally:
            xoa_cache_nhan()
        return (not a and not sai and n > 0, f"hop dong: {ma}; {n} anh doc qua YOLODataset that, lech pixel {len(sai)}")
    _thu("J0 dataset SACH (doi chung): hop dong chap nhan; YOLODataset THAT (cache=False, truoc augmentation) doc pixel "
         "TRUNG JPEG o moi anh train + dev", j0)

    def j0b():
        k = bx.kiem_dataset(out)
        return k["cache_anh"]["so_npy_canh_anh"] == {"train": 0, "dev": 0}, str(k["cache_anh"])
    _thu("J0b danh tinh dataset ghi chinh sach cache anh + so .npy canh anh = 0", j0b)

    for s, ma_ca in (("train", "J1"), ("dev", "J2")):
        def j12(s=s):
            jpg = dat_npy_den(s)
            try:
                xoa_cache_nhan()
                ds = yds(s)
                idx = [bx._nc(f) for f in ds.im_files].index(bx._nc(anh[s]))
                nap = ds.get_image_and_label(idx)["img"]
                a, ma = tu_choi(lambda: bx.kiem_dataset(out))
                b, mb = tu_choi(lambda: tr.kiem_dataset(data))
            finally:
                go_npy(s)
                xoa_cache_nhan()
            nguy_co = float(nap.mean()) == 0.0 and float(jpg.mean()) > 0      # ultralytics 8.4.141 THAT nap anh den
            ok = nguy_co and a and b and f"images/{s}" in ma and ".npy" in ma and "cache=False" in ma and ".npy" in mb
            return ok, (f"YOLODataset that nap TB pixel {nap.mean():.2f} (JPEG {jpg.mean():.2f}); exporter: {ma[-230:]} | "
                        f"runner: {mb[-120:]}")
        _thu(f"{ma_ca} .npy anh DEN canh anh {s} (JPEG/nhan/hash giu nguyen): ultralytics that nap anh den -> hop dong "
             f"(exporter + chot runner) TU CHOI, neu split / duong dan / ly do", j12)

    if (_models_dir() / "yolov8n.pt").is_file():
        pcfg = _yaml(tmp / "train_cfg_j.yaml", cfg_train_tuyet_doi())
        for s, ma_ca in (("train", "J3"), ("dev", "J4")):
            def j34(s=s):
                dat_npy_den(s)
                ten = f"j_npy_{s}"
                try:
                    code, _so, se, goi = _chay_runner_spy(
                        ["--config", str(pcfg), "--model", "yolov8n", "--che-do", "smoke", "--data", str(data),
                         "--project", str(tmp / "runs_j"), "--ket-qua-dir", str(tmp / "kq_j"), "--ten", ten])
                finally:
                    go_npy(s)
                khong_cham = (not any(g[0] in ("khoi_tao", "train") for g in goi) and not (tmp / "runs_j" / ten).exists()
                              and not (tmp / "kq_j" / f"{ten}.json").exists())
                return (code == 2 and khong_cham and ".npy" in se and f"images/{s}" in se,
                        f"exit {code}; spy {goi[:2]}; {se.strip()[-200:]}")
            _thu(f"{ma_ca} entry point runner (spy GIA LAP ranh gioi model/training): .npy canh anh {s} -> exit 2, KHONG nap "
                 f"model, KHONG goi train, KHONG tao thu muc / file ket qua", j34)
        g6 = cfg_train_tuyet_doi()
        g6["huan_luyen"]["cache"] = "disk"
        pcfg6 = _yaml(tmp / "train_cfg_j_disk.yaml", g6)

        def j6():
            code, _so, se, goi = _chay_runner_spy(
                ["--config", str(pcfg6), "--model", "yolov8n", "--che-do", "smoke", "--data", str(data),
                 "--project", str(tmp / "runs_j"), "--ket-qua-dir", str(tmp / "kq_j"), "--ten", "j_cache_disk"])
            return code == 2 and not goi and "cache" in se, f"exit {code}; spy {goi[:2]}; {se.strip()[-160:]}"
        _thu("J6 cau hinh `cache: disk` (khac chinh sach cache: false) -> runner exit 2 truoc khi nap model", j6)
    else:
        _skip("J3/J4/J6 entry point runner", f"khong co {_models_dir() / 'yolov8n.pt'} (can cho kiem sha256 checkpoint)")

    md = tr.nhan_mong_doi(data)
    for s, ma_ca in (("train", "J5a"), ("dev", "J5b")):
        def j5(s=s):
            xoa_cache_nhan()
            try:
                dts = {x: yds(x) for x in ("train", "dev")}       # dung khi dataset con SACH (sau hop dong)
                trainer = SimpleNamespace(train_loader=SimpleNamespace(dataset=dts["train"]),
                                          test_loader=SimpleNamespace(dataset=dts["dev"]))
                sach = tr.kiem_dau_vao_da_nap(trainer, md)
                dat_npy_den(s)                                     # .npy xuat hien SAU hop dong, truoc vong train
                try:
                    tr.kiem_dau_vao_da_nap(trainer, md)
                    ket = (False, "callback KHONG phat hien .npy")
                except tr.LoiDauVaoDaNap as e:
                    ket = (str(e).startswith(f"{s}:") and ".npy" in str(e) and sach[s]["npy_canh_anh"] == 0, str(e)[:220])
            finally:
                go_npy(s)
                xoa_cache_nhan()
            return ket
        _thu(f"{ma_ca} callback truoc vong train voi YOLODataset THAT: sach -> qua; .npy xuat hien canh anh {s} sau hop dong "
             f"-> LoiDauVaoDaNap", j5)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test_det5_"))
    out_sc = None
    try:
        _chay("A. anh xa lop", nhom_a)

        def bcd():
            nonlocal out_sc
            out_sc = nhom_bcd(tmp)
        _chay("B/C/D. xuat dataset BDD -> YOLO 5 lop", bcd)
        _chay("E. evaluator 5 lop", lambda: nhom_e(tmp, out_sc))
        _chay("E7-E9. ghep COCO: hoa IoU, IoU dung nguong, GT chong vung ignore (R27-F3)", nhom_e2)
        _chay("F. runner huan luyen", lambda: nhom_f(tmp, out_sc))
        _chay("G. runner qua entry point, spy tai ranh gioi model/training (R27-F1)", lambda: nhom_g(tmp, out_sc))
        _chay("H. toan ven noi dung dataset: anh / nhan / ignore / manifest (R27-F2)",
              lambda: nhom_h(tmp, out_sc, nap_cfg(tmp, cfg_mau(tmp, "sc"))))
        _chay("I. du lieu ultralytics THUC SU nap (cache nhan) truoc vong train (R27-F2)", lambda: nhom_i(tmp, out_sc))
        _chay("J. cache ANH .npy: nguon pixel qua YOLODataset that + chot runner (R28-F1)", lambda: nhom_j(tmp, out_sc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS"
          + (f", {len(_SKIP)} SKIP (CHUA KIEM, khong tinh PASS): {[s for s, _ in _SKIP]}" if _SKIP else ""))
    return 0 if _RESULTS and n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
