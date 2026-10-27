"""
mixed5_lib.py -- logic dataset tron 5 lop BDD100K + UA-DETRAC (R52A, quyet dinh nguoi dung 16/10/2026).

Diem vao CLI: `src/mixed5/xuat_mixed5.py`. Module nay chi chua ham thuan + ham ghi theo duong dan tuong minh, de test
goi truc tiep tren fixture. KHONG sua ma cu: dung lai `bdd_yolo_export` (xuat_mot_anh, sha256, hop dong dataset).

QUY TAC (ghi vao manifest/tom_tat.json)
1. Split: BDD giu NGUYEN bdd5_train_v1 (train-fit 1.261 / dev 139 video, tham chieu sha256, khong chia lai).
   UA-DETRAC: dev = 12 video cua results/phase2/manifests/detrac_dev.json; train = 60 video co XML train tru dev12.
   40 video UA-DETRAC test va BDD100K val khong bao gio vao dataset.
2. BDD: anh / nhan / ignore lay NGUYEN BYTE tu data/yolo_bdd5_v1 (hardlink cung o, khong duoc thi sao chep); dong
   manifest BDD giu nguyen gia tri cac cot cua manifest bdd5_v1.
3. UA-DETRAC frame: `num` cua <frame> (= so trong imgNNNNN.jpg, bat dau 1); giu num % buoc == 0 VA co <frame> trong
   XML. Anh khong co <frame> -> loai (dem rieng). Cot frame_idx cua dong DETRAC = num (1-based).
4. Lop: car->car, van->car, bus->bus, others->truck (gia dinh nguoi dung); vehicle_type khac -> loi (khong lang le bo).
5. ignored_region cua video -> vung to xam moi frame, theo DUNG ham BDD `xuat_mot_anh` (tru pixel trong box co nhan,
   mau (114,114,114), JPEG q95, toa do vao ignore/<split>/<anh>.txt). Box cat vao khung, < 1 px sau cat -> bo.
6. Frame co <frame> ma 0 box hop le -> giu, file nhan rong (anh am tinh), nhu quy tac BDD (frame da gan nhan).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import bdd_yolo_export as bx  # noqa: E402
from detector_classes import LOP_5  # noqa: E402

GOC = SRC.parent
LoiDuLieu = bx.LoiDuLieu
ThieuDuLieu = bx.ThieuDuLieu
PHIEN_BAN_QUY_TAC = "mixed5-v1"
ANH_XA_LOP = {"car": "car", "van": "car", "bus": "bus", "others": "truck"}
LOP_TO_XAM_DETRAC = "other vehicle"           # nhan vai tro 'to_xam' trong bx.vai_tro (ign=True)
COT = bx.COT_MANIFEST + ["nguon", "dat_file"]  # 2 cot them; hop dong chi doc cac cot cua bx.COT_MANIFEST
KHOA_DEM = ("nhan_lop_khac", "box_nho_hon_1px_sau_cat", "vung_to_xam_rong", "box_bi_cat_vao_khung", "so_pixel_to_xam")


# =========================================================================== tien ich
def sha256_json(obj) -> str:
    return bx.sha256_bytes(json.dumps(obj, sort_keys=True).encode())


def ghi_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def co_eoi_jpeg(p: Path) -> bool:
    """Dieu kien ultralytics `verify_image_label` KHONG ghi lai JPEG: 2 byte cuoi la FF D9."""
    with open(p, "rb") as f:
        f.seek(-2, 2)
        return f.read() == b"\xff\xd9"


def ten_anh_detrac(video: str, num: int) -> str:
    return f"{video}_img{num:05d}.jpg"


# =========================================================================== UA-DETRAC XML
def doc_xml_detrac(path: Path) -> dict:
    """XML 1 video -> {'video', 'frames': {num: [(vehicle_type, l, t, w, h)]}, 'ignored': [(l, t, w, h)]}."""
    root = ET.parse(str(path)).getroot()
    video = root.get("name") or path.stem
    if video != path.stem:
        raise LoiDuLieu(f"{path}: sequence name {video!r} khac ten file")
    ignored = []
    ir = root.find("ignored_region")
    if ir is not None:
        for b in ir.findall("box"):
            ignored.append(tuple(float(b.get(k)) for k in ("left", "top", "width", "height")))
    frames: dict[int, list] = {}
    for fr in root.findall("frame"):
        num = int(fr.get("num"))
        if num in frames:
            raise LoiDuLieu(f"{path}: frame num={num} lap lai")
        ds = []
        tl = fr.find("target_list")
        for tg in (tl.findall("target") if tl is not None else []):
            b, at = tg.find("box"), tg.find("attribute")
            vt = at.get("vehicle_type") if at is not None else None
            if b is None:
                ds.append((vt, math.nan, math.nan, math.nan, math.nan))
                continue
            ds.append((vt, *(float(b.get(k, "nan")) for k in ("left", "top", "width", "height"))))
        frames[num] = ds
    return {"video": video, "frames": frames, "ignored": ignored}


def liet_ke_anh_detrac(thu_muc: Path) -> dict[int, str]:
    """{num: ten file} cho moi imgNNNNN.jpg trong thu muc video."""
    out = {}
    for p in thu_muc.iterdir():
        n = p.name
        if p.is_file() and n.startswith("img") and n.lower().endswith(".jpg") and n[3:-4].isdigit():
            out[int(n[3:-4])] = n
    return out


def chon_frame_detrac(nums_xml, nums_anh, buoc: int) -> tuple[list[int], dict]:
    """Quy tac 3: giu num % buoc == 0 VA co <frame> trong XML. Dem anh theo buoc khong co <frame> (bi loai)."""
    xml, anh = set(nums_xml), set(nums_anh)
    thieu_anh = sorted(xml - anh)
    if thieu_anh:
        raise LoiDuLieu(f"{len(thieu_anh)} frame XML khong co anh (vd {thieu_anh[:3]})")
    chon = sorted(n for n in xml if n % buoc == 0)
    dem = {"frame_xml": len(xml), "anh_nguon": len(anh), "anh_khong_co_frame": len(anh - xml),
           "anh_theo_buoc_khong_co_frame": sum(1 for n in anh - xml if n % buoc == 0),
           "frame_xml_khong_theo_buoc": len(xml) - len(chon), "chon": len(chon)}
    return chon, dem


def frame_cho_bx(video: str, num: int, targets: list, ignored: list, anh_xa: dict, dem: dict) -> bx.Frame:
    """Dung bx.Frame de xuat bang CHINH ham BDD: box co nhan -> (lop 5, ..., crowd=False, ign=False); ignored_region ->
    ('other vehicle', ..., ign=True) => vai tro 'to_xam'. Box XML khong huu han / w,h <= 0 -> bo, dem rieng."""
    nhan = []
    for vt, l, t, w, h in targets:
        if vt not in anh_xa:
            raise LoiDuLieu(f"{video} frame {num}: vehicle_type {vt!r} ngoai anh xa {sorted(anh_xa)}")
        if not all(math.isfinite(x) for x in (l, t, w, h)) or w <= 0 or h <= 0:
            dem["box_xml_khong_hop_le"] = dem.get("box_xml_khong_hop_le", 0) + 1
            continue
        nhan.append((anh_xa[vt], l, t, l + w, t + h, False, False))
    for l, t, w, h in ignored:
        nhan.append((LOP_TO_XAM_DETRAC, l, t, l + w, t + h, False, True))
    return bx.Frame(idx=num, rel=f"{video}/img{num:05d}.jpg", nhan=nhan)


# =========================================================================== split
def tao_ban_ghi_split(ten: str, bdd_split_path: Path, xml_dir: Path, dev12_path: Path, so_xml: int, so_dev: int,
                      cau_hinh: dict | None = None) -> dict:
    """Split mixed5: BDD nguyen bdd5_train_v1 + UA-DETRAC (XML train - dev12 / dev12). Kiem so luong + roi nhau."""
    bsp = json.loads(bdd_split_path.read_text(encoding="utf-8"))
    bds = {"train_fit": bsp["train_fit"], "dev": bsp["dev"]}
    if sha256_json(bds) != bsp["sha256_danh_sach"]:
        raise LoiDuLieu(f"{bdd_split_path}: danh sach khong khop sha256_danh_sach")
    xml = sorted(p.stem for p in xml_dir.glob("*.xml"))
    if len(xml) != so_xml:
        raise LoiDuLieu(f"{xml_dir}: {len(xml)} XML, can {so_xml}")
    dev12 = sorted(json.loads(dev12_path.read_text(encoding="utf-8"))["videos"])
    if len(set(dev12)) != so_dev or not set(dev12) <= set(xml):
        raise LoiDuLieu(f"{dev12_path}: can {so_dev} video khac nhau thuoc tap XML train")
    det_tr = [v for v in xml if v not in set(dev12)]
    trung = (set(bds["train_fit"]) | set(bds["dev"])) & set(xml)
    if trung:
        raise LoiDuLieu(f"ten video BDD trung UA-DETRAC: {sorted(trung)[:3]}")
    ds = {"train_fit": sorted(bds["train_fit"] + det_tr), "dev": sorted(bds["dev"] + dev12)}
    if set(ds["train_fit"]) & set(ds["dev"]):
        raise LoiDuLieu("split mixed5: train_fit va dev trung video")
    vt_xml = [(p.name, bx.sha256_file(p)) for p in sorted(xml_dir.glob("*.xml"))]
    return {
        "ten": ten, "phien_ban_quy_tac": PHIEN_BAN_QUY_TAC, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "git_head": bx.git_head(), "cau_hinh": cau_hinh or {},
        "quyet_dinh": "nguoi dung 16/10/2026: train = BDD train-fit + 48 video UA-DETRAC (60 XML train tru dev12); "
                      "dev = BDD dev + dev12; 40 video UA-DETRAC test khong vao huan luyen",
        "bdd": {"file_split": bx.rel(bdd_split_path, GOC), "sha256": bx.sha256_file(bdd_split_path),
                "sha256_danh_sach": bsp["sha256_danh_sach"],
                "so_video": {"train_fit": len(bds["train_fit"]), "dev": len(bds["dev"])},
                "ghi_chu": "danh sach video BDD lay NGUYEN tu file tren, khong chia lai"},
        "detrac": {"thu_muc_xml": bx.rel(xml_dir, GOC), "so_xml": len(xml),
                   "dau_van_tay_xml": bx.sha256_bytes("".join(f"{n}\t{h}\n" for n, h in vt_xml).encode()),
                   "cach_bam_xml": "sha256 cua cac dong '<ten file>\\t<sha256 file>\\n' da sap xep",
                   "dev12": {"duong_dan": bx.rel(dev12_path, GOC), "sha256": bx.sha256_file(dev12_path)},
                   "quy_tac": "train = video co XML train tru dev12; dev = dev12",
                   "train_fit": det_tr, "dev": dev12},
        **ds, "sha256_danh_sach": sha256_json(ds),
    }


def doc_split_mixed5(path: Path, bdd_split_path: Path) -> dict:
    """Doc split da khoa + kiem: sha256_danh_sach, phan BDD trung NGUYEN bdd5_train_v1 (file + danh sach)."""
    sp = json.loads(path.read_text(encoding="utf-8"))
    ds = {"train_fit": sp["train_fit"], "dev": sp["dev"]}
    if sha256_json(ds) != sp["sha256_danh_sach"]:
        raise LoiDuLieu(f"{path}: danh sach khong khop sha256_danh_sach (file bi sua tay?)")
    if bx.sha256_file(bdd_split_path) != sp["bdd"]["sha256"]:
        raise LoiDuLieu(f"{bdd_split_path} da doi sha256 so voi luc khoa split mixed5")
    bsp = json.loads(bdd_split_path.read_text(encoding="utf-8"))
    det = sp["detrac"]
    for k in ("train_fit", "dev"):
        if sorted(bsp[k] + det[k]) != sp[k]:
            raise LoiDuLieu(f"split mixed5 {k} != bdd5 {k} + detrac {k}")
    if set(det["train_fit"]) & set(det["dev"]):
        raise LoiDuLieu("UA-DETRAC: dev12 nam trong train")
    return sp


# =========================================================================== ro ri
def doc_bang_sha(p: Path) -> dict:
    """Bang sha256 tap danh gia (cot rel, sha256) -> {'sha256', 'rel', 'video', 'ten_anh'}."""
    sha, rels = set(), set()
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sha.add(r["sha256"])
            rels.add(r["rel"])
    return {"sha256": sha, "rel": rels, "video": {r.split("/", 1)[0] for r in rels},
            "ten_anh": {r.rsplit("/", 1)[-1] for r in rels}, "so_dong": len(rels)}


def kiem_ro_ri(rows: list[dict], tap_cam: dict[str, dict], video_split: dict[str, set]) -> dict:
    """rows: dong manifest (split, video, anh_nguon, sha256_nguon, sha256_ra, anh_ra, nguon). tap_cam: ten -> bang sha.
    Loi neu: trung sha256 (nguon hoac xuat) / ten video / anh nguon voi tap danh gia; train/dev trung video, sha256 anh
    nguon, sha256 anh xuat hoac ten anh xuat. Tra ve bien ban dem."""
    bb, loi = {}, []
    sha_ng = {r["sha256_nguon"] for r in rows}
    sha_ra = {r["sha256_ra"] for r in rows}
    videos = {r["video"] for r in rows} | set().union(*video_split.values())
    for ten, t in tap_cam.items():
        tr_ng, tr_ra = sorted(sha_ng & t["sha256"]), sorted(sha_ra & t["sha256"])
        tr_v = sorted(videos & t["video"])
        tr_rel = sorted({r["anh_nguon"] for r in rows if r["nguon"] == t.get("nguon_anh")} & t["rel"])
        bb[ten] = {"so_anh_bang": t["so_dong"], "so_video_bang": len(t["video"]), "trung_sha256_nguon": len(tr_ng),
                   "trung_sha256_xuat": len(tr_ra), "trung_video": len(tr_v), "trung_duong_dan_anh_nguon": len(tr_rel)}
        if tr_ng or tr_ra or tr_v or tr_rel:
            loi.append(f"{ten}: sha nguon {tr_ng[:2]}, sha xuat {tr_ra[:2]}, video {tr_v[:3]}, anh {tr_rel[:2]}")
    tr, dv = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]
    v_tr, v_dv = video_split["train"], video_split["dev"]
    noi_bo = {"trung_video": len(v_tr & v_dv)}
    if v_tr & v_dv:
        loi.append(f"train/dev trung video {sorted(v_tr & v_dv)[:3]}")
    for cot in ("sha256_nguon", "sha256_ra", "anh_ra"):
        a = {Path(r[cot]).name if cot == "anh_ra" else r[cot] for r in tr}
        b = {Path(r[cot]).name if cot == "anh_ra" else r[cot] for r in dv}
        noi_bo[f"trung_{cot}"] = len(a & b)
        if a & b:
            loi.append(f"train/dev trung {cot}: {len(a & b)} (vd {sorted(a & b)[:2]})")
    for s, rr in (("train", tr), ("dev", dv)):
        bad = sorted({r["video"] for r in rr} - video_split[s])
        if bad:
            loi.append(f"anh {s} thuoc video ngoai split {s}: {bad[:3]}")
    noi_bo["so_anh"] = {"train": len(tr), "dev": len(dv)}
    noi_bo["sha256_nguon_lap_trong_cung_split"] = {
        s: len(rr) - len({r["sha256_nguon"] for r in rr}) for s, rr in (("train", tr), ("dev", dv))}
    bb["train_dev_noi_bo"] = noi_bo
    if loi:
        raise LoiDuLieu("RO RI / TRUNG: " + "; ".join(loi[:10]))
    return bb


# =========================================================================== xuat
def dat_file_bdd(src: Path, dst: Path, cho_hardlink: bool) -> str:
    """hardlink (uu tien) hoac sao chep nguyen byte. Da ton tai -> giu (tiep tuc), tra ve cach dat hien tai."""
    if dst.exists():
        return "hardlink" if os.path.samefile(src, dst) else "sao_chep"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cho_hardlink:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copyfile(src, dst)
    return "sao_chep"


def xuat_frame_detrac(src: Path, frame: bx.Frame, out: Path, split: str, video: str, ten: str, dem: dict) -> dict:
    """Xuat 1 frame UA-DETRAC bang bx.xuat_mot_anh -> dong manifest (cot COT)."""
    stem = Path(ten).stem
    for sub in ("images", "labels", "ignore"):
        (out / sub / split).mkdir(parents=True, exist_ok=True)
    img, lbl, ign = out / "images" / split / ten, out / "labels" / split / f"{stem}.txt", out / "ignore" / split / f"{stem}.txt"
    r = bx.xuat_mot_anh(src, frame, img, lbl, ign, dem)
    if not r["co_ignore"] and ign.exists():
        ign.unlink()                                  # sot tu lan chay bi ngat (khong the xay ra voi cung XML)
    ign_rel = f"ignore/{split}/{stem}.txt" if r["co_ignore"] else ""
    return {"split": split, "video": video, "frame_idx": frame.idx, "anh_nguon": frame.rel,
            "sha256_nguon": bx.sha256_file(src), "anh_ra": f"images/{split}/{ten}", "sha256_ra": bx.sha256_file(img),
            "w": r["w"], "h": r["h"], **{f"n_{c}": r["n"][c] for c in LOP_5}, "n_vung_to_xam": r["n_vung"],
            "xu_ly": r["xu_ly"], "am_tinh": int(sum(r["n"].values()) == 0), "nhan_ra": f"labels/{split}/{stem}.txt",
            "sha256_nhan": bx.sha256_file(lbl), "ignore_ra": ign_rel,
            "sha256_ignore": bx.sha256_file(out / ign_rel) if ign_rel else "", "nguon": "detrac", "dat_file": "ghi_moi"}


def ghi_data_yaml(out: Path, ghi_chu: str) -> Path:
    p = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(LOP_5))
    p.write_bytes((f"# SINH TU DONG boi src/mixed5/xuat_mixed5.py -- KHONG sua tay. {ghi_chu}\n"
                   f"# 'val' = DEV NOI BO: BDD dev (tach tu BDD100K MOT train) + UA-DETRAC dev12 (tu 60 video train).\n"
                   f"# KHONG phai BDD100K val chinh thuc, KHONG phai UA-DETRAC test.\n"
                   f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/dev\n"
                   f"nc: {len(LOP_5)}\nnames:\n{names}\n").encode("utf-8"))
    return p


def thong_ke_rows(rows: list[dict]) -> dict:
    """Thong ke theo nguon x split: so video/anh/am tinh/anh to xam, box + anh co lop theo lop."""
    tk: dict = {}
    for r in rows:
        for ng in (r["nguon"], "tong"):
            t = tk.setdefault(ng, {}).setdefault(r["split"], {
                "so_video": set(), "so_anh": 0, "am_tinh": 0, "anh_to_xam": 0,
                "box": {c: 0 for c in LOP_5}, "anh_co_lop": {c: 0 for c in LOP_5}})
            t["so_video"].add(r["video"])
            t["so_anh"] += 1
            t["am_tinh"] += int(r["am_tinh"])
            t["anh_to_xam"] += int(r["xu_ly"] == "to_xam")
            for c in LOP_5:
                n = int(r[f"n_{c}"])
                t["box"][c] += n
                t["anh_co_lop"][c] += int(n > 0)
    for ng in tk.values():
        for t in ng.values():
            t["so_video"] = len(t["so_video"])
            t["tong_box"] = sum(t["box"].values())
    return tk


def ghi_manifest(out: Path, rows: list[dict], file_split: Path, sp: dict, tom_them: dict, ghi_chu_yaml: str) -> dict:
    """Ghi manifest v2 theo hop dong bx.kiem_hop_dong_dataset: anh.csv, video.json, ban sao split, data.yaml, tom_tat."""
    man_dir = out / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    man = man_dir / "anh.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COT)
        w.writeheader()
        w.writerows(rows)
    vj = {"train": list(sp["train_fit"]), "dev": list(sp["dev"])}
    (man_dir / "video.json").write_bytes((json.dumps(vj, indent=1) + "\n").encode("utf-8"))
    shutil.copyfile(file_split, man_dir / "split_da_khoa.json")
    ghi_data_yaml(out, ghi_chu_yaml)
    tom = {
        "ten": sp["ten"], "tap_con": None, "loai": "day_du", "phien_ban_manifest": bx.PHIEN_BAN_MANIFEST,
        "phien_ban_quy_tac": PHIEN_BAN_QUY_TAC, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "git_head": bx.git_head(), "lop": list(LOP_5),
        "file_split": {"duong_dan": bx.rel(file_split, GOC), "duong_dan_tuyet_doi": file_split.resolve().as_posix(),
                       "sha256": bx.sha256_file(file_split), "sha256_danh_sach": sp["sha256_danh_sach"],
                       "ban_sao": "manifest/split_da_khoa.json"},
        "sha256_data_yaml": bx.sha256_file(out / "data.yaml"),
        "sha256_video_json": bx.sha256_file(man_dir / "video.json"),
        "video": {"train": len(vj["train"]), "dev": len(vj["dev"])},
        "thong_ke": thong_ke_rows(rows),
        **tom_them,
        "manifest_anh": {"duong_dan": "manifest/anh.csv", "so_dong": len(rows), "sha256": bx.sha256_file(man),
                         "cot_them": ["nguon", "dat_file"]},
    }
    ghi_json(man_dir / "tom_tat.json", tom)
    return tom
