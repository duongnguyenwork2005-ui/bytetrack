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

    python src/test_detector_5lop.py
"""
from __future__ import annotations

import copy
import json
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


def _check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


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
    for w, ten in ((config.MODELS_DIR / "yolov8n.pt", "yolov8n"), (config.MODELS_DIR / "candidates" / "yolo11n.pt",
                                                                     "yolo11n")):
        if w.is_file():
            from ultralytics import YOLO
            n = YOLO(str(w)).names
            _check(f"A7 doc names tu checkpoint that {ten}: coco80, anh xa dung",
                   dc.loai_checkpoint(n) == "coco80" and dc.anh_xa_model_sang_lop5(n) == {2: 0, 7: 1, 5: 2, 3: 3, 1: 4})


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
    import csv

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
        _check("D9 anh THIEU file nhan -> kiem dataset bao loi (khong coi la am tinh)", "KHONG co file nhan" in str(e))
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
    except ImportError:
        _check("E1 pycocotools khong co -> bo qua doi chieu", True)
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
    w = config.MODELS_DIR / "yolov8n.pt"
    if w.is_file() and out_sc is not None:
        anh = ev.ds_anh_tu_data_yaml(out_sc / "data.yaml", "val")[:3]
        dd_, mt = ev.du_doan_ultralytics(str(w), anh, ev.QUY_TAC["du_doan"], batch=3, device="cpu")
        lop = set(np.concatenate([x["cls"] for x in dd_]).tolist()) if dd_ else set()
        _check("E6 du doan that YOLOv8n COCO (CPU): anh xa theo ten ghi lai, lop dau ra chi trong 0..4",
               mt["loai_checkpoint"] == "coco80" and mt["model_id_sang_lop5"] == {"1": 4, "2": 0, "3": 3, "5": 2, "7": 1}
               and lop <= set(range(5)) and len(dd_) == 3, f"lop {sorted(lop)}")


# =========================================================================== F
def nhom_f(tmp: Path, out_sc: Path | None):
    import yaml
    if out_sc is None:
        _check("F0 can dataset fixture tu nhom B", False)
        return
    goc = yaml.safe_load((config.CONFIGS_DIR / "phase2" / "detector" / "train_bdd5.yaml").read_text(encoding="utf-8"))
    pcfg = _yaml(tmp / "train_cfg.yaml", goc)
    base = [PY, str(SRC / "train_detector.py"), "--config", str(pcfg), "--model", "yolov8n", "--data",
            str(out_sc / "data.yaml"), "--project", str(tmp / "runs"), "--ket-qua-dir", str(tmp / "kq")]
    w_ok = (config.MODELS_DIR / "yolov8n.pt").is_file()
    if not w_ok:
        _check("F1..F5 bo qua: khong co models/yolov8n.pt", True)
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
        _chay("F. runner huan luyen", lambda: nhom_f(tmp, out_sc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 0 if _RESULTS and n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
