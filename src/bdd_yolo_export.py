"""
bdd_yolo_export.py -- BDD100K MOT (CHI split `train` chinh thuc) -> dataset YOLO 5 lop cho fine-tune detector.

Diem vao CLI: `src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc ...`.
Module nay chua logic + ham kiem, de test goi truc tiep.

===========================================================================
NGUON DU LIEU
===========================================================================
Chi BDD100K **MOT 2020** split `train` (anh `images/track/train/<video>/<video>-<7 so>.jpg` + nhan theo video).
Hai dinh dang nhan doc duoc (cung noi dung, cung ket qua xuat -- test kiem):
    scalabel : `labels/box_track_20/train/<video>.json` (dinh dang goc, moi video mot file, it RAM)
    cocovid  : mot file `*_cocofmt.json` (ban chuyen doi COCO-video, can nhieu RAM voi split train)
KHONG dung: BDD100K val (nguon danh gia da dinh danh), UA-DETRAC, goi detection 100K (`images/100k`, `det_20`
-- anh tinh, khac nguon; muon dung phai hoi truoc). Thieu nguon -> ThieuDuLieu (exit 2), khong tao dataset rong.

===========================================================================
QUY TAC (ghi vao manifest)
===========================================================================
1. Chia train-fit / dev THEO VIDEO tu split train chinh thuc (khong chia theo frame). Phan tang theo SU CO MAT
   cua lop hiem (mac dinh motorcycle, bicycle) trong video; trong moi tang: sap xep ten, hoan vi bang
   `np.random.default_rng(seed)`, lay floor(ti_le_dev * n + 0.5) video cho dev. Danh sach duoc KHOA vao file split
   (commit) TRUOC khi xuat/huan luyen; buoc xuat tu choi neu nguon nhan doi hash.
2. Lay thua frame: giu frame co chi so (0-based, theo video) chia het cho `buoc_frame` -- quy tac co dinh, khong
   phu thuoc nhan, dung chung cho moi model.
3. Anh am tinh: frame nam TRONG khoang [frame co nhan dau, frame co nhan cuoi] cua video (nhan bat ky lop, ke
   ca pedestrian/crowd) duoc coi la DA GAN NHAN -> giu, file nhan rong. Frame NGOAI khoang do (vd video co anh toi
   frame 202 nhung nhan dung o frame 94) KHONG chac co nhan -> loai, dem rieng. Moi anh xuat ra deu co file .txt
   (ultralytics coi anh THIEU .txt la nen -> ta khong de truong hop do xay ra; buoc kiem bat loi nay).
4. Lop: car/truck/bus/motorcycle/bicycle (khong crowd, khong ignore) -> nhan 0..4. pedestrian, rider, train,
   other person: KHONG gan nhan, KHONG to xam (khong thuoc 5 lop; rider la box rieng, motorcycle chi bao xe).
5. crowd / ignore: box `crowd` (hoac `iscrowd`) thuoc 5 lop, va distractor nhom xe (`other vehicle`, `trailer`;
   trong COCO-video la `ignore=1` voi lop car/truck) -> TO XAM (114,114,114) vung do, TRU pixel nam trong box da
   gan nhan (khong xoa doi tuong co nhan). Toa do vung to ghi vao `ignore/<split>/<anh>.txt` (px, x1 y1 x2 y2)
   de danh gia bo qua detection roi vao vung nay.
6. Toa do: theo KICH THUOC ANH THUC doc tu file anh; nhan co w/h (COCO-video) khac anh thuc -> loi. Box cat vao
   khung anh; box con < 1 px sau khi cat -> bo, dem rieng.
7. Anh khong to xam -> SAO CHEP nguyen byte (khong ma hoa lai, khong hard link: ultralytics co the ghi de JPEG
   hong); anh co to xam -> ghi JPEG chat luong 95.
8. Khong trung video/anh voi tap danh gia: BDD100K val chinh thuc -> ten video, ten anh va sha256 noi dung anh nguon
   (so voi toan bo anh BDD val); UA-DETRAC test -> CHI ten video (khong so noi dung). Train-fit va dev: khac video
   VA khac sha256 anh nguon (vong 28), kiem truoc khi ghi.
9. Manifest phien ban 2 (vong 28, R27-F2): moi anh co sha256 anh xuat, file nhan (ke ca file RONG) va file ignore
   (neu co vung); sha256 data.yaml / video.json; ban sao split da khoa. `kiem_hop_dong_dataset()` la HOP DONG DUNG
   CHUNG cho exporter, runner huan luyen va evaluator: kiem dung duong dan ultralytics resolve tu data.yaml va noi
   dung thuc te; manifest phien ban 1 bi tu choi (khong bo sung hash ve sau).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from detector_classes import ID_LOP_5, LOP_5  # noqa: E402

DISTRACTOR_XE = ("other vehicle", "trailer")
CHAT_LUONG_JPEG = 95
MAU_TO_XAM = tuple(int(c) for c in config.IGNORE_MASK_COLOR_BGR)
PHIEN_BAN_QUY_TAC = "bdd5-v1"
#: v1 (vong 27): chi sha256 anh. v2 (vong 28, R27-F2): them duong dan + sha256 file NHAN va file IGNORE cua tung anh,
#: sha256 data.yaml / video.json, ban sao split da khoa. Ban v1 KHONG duoc bo sung hash ve sau: phai xuat lai.
PHIEN_BAN_MANIFEST = 2
SHA256_RONG = hashlib.sha256(b"").hexdigest()
COT_MANIFEST = ["split", "video", "frame_idx", "anh_nguon", "sha256_nguon", "anh_ra", "sha256_ra", "w", "h",
                *[f"n_{c}" for c in LOP_5], "n_vung_to_xam", "xu_ly", "am_tinh",
                "nhan_ra", "sha256_nhan", "ignore_ra", "sha256_ignore"]


class ThieuDuLieu(RuntimeError):
    """Dau vao chua co tren may (exit 2). `thieu` = danh sach muc cu the."""

    def __init__(self, msg: str, thieu: list[str] | None = None):
        super().__init__(msg)
        self.thieu = thieu or []


class LoiDuLieu(RuntimeError):
    """Dau vao co nhung sai / trung tap danh gia / xuat ra khong hop le (exit 3)."""


# =========================================================================== tien ich
def sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def kich_thuoc_anh(p: Path) -> tuple[int, int]:
    """(w, h) doc tu header file anh (khong giai ma toan bo)."""
    from PIL import Image
    with Image.open(p) as im:
        return int(im.size[0]), int(im.size[1])


def rel(p: Path, goc: Path) -> str:
    try:
        return p.resolve().relative_to(goc.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, capture_output=True, text=True)
        return r.stdout.strip() or "?"
    except OSError:
        return "?"


# =========================================================================== doc nguon
@dataclass
class Frame:
    idx: int                 # chi so frame 0-based trong video (frameIndex / frame_id)
    rel: str                 # "<video>/<ten anh>.jpg" tuong doi voi thu muc anh cua split
    w: int | None = None     # kich thuoc khai bao trong nhan (COCO-video co; Scalabel khong)
    h: int | None = None
    nhan: list = field(default_factory=list)   # (category, x1, y1, x2, y2, crowd, ignore)


def vai_tro(cat: str, crowd: bool, ign: bool) -> str:
    """'gan_nhan' | 'to_xam' | 'lop_khac' (quy tac 4-5 o docstring)."""
    if cat in ID_LOP_5 and not crowd and not ign:
        return "gan_nhan"
    if (crowd or ign) and (cat in ID_LOP_5 or cat in DISTRACTOR_XE):
        return "to_xam"
    return "lop_khac"


class NguonScalabel:
    """`labels/box_track_20/<split>/<video>.json` (dinh dang goc BDD100K MOT)."""

    dinh_dang = "scalabel"

    def __init__(self, duong_dan: Path):
        self.dir = Path(duong_dan)

    def ton_tai(self) -> bool:
        return self.dir.is_dir() and any(self.dir.glob("*.json"))

    def danh_sach_video(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def doc_video(self, video: str) -> list[Frame]:
        from mot_datasets.bdd100k_mot import DISTRACTOR_CATEGORIES, check_schema
        p = self.dir / f"{video}.json"
        frames = json.loads(p.read_text(encoding="utf-8"))
        try:
            check_schema(frames, video)
        except ValueError as e:
            raise LoiDuLieu(f"nhan Scalabel sai schema: {e}") from e
        out = []
        for f in frames:
            vn = f.get("videoName") or video
            if vn != video:
                raise LoiDuLieu(f"{p.name}: videoName {vn!r} khac ten file")
            nh = []
            for lb in f["labels"]:
                b = lb["box2d"]
                cat = lb["category"]
                nh.append((cat, float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]),
                           bool(lb["attributes"]["crowd"]), cat in DISTRACTOR_CATEGORIES))
            out.append(Frame(idx=int(f["frameIndex"]), rel=f"{video}/{f['name']}", nhan=nh))
        return out

    def dau_van_tay(self) -> dict:
        files = sorted(self.dir.glob("*.json"))
        h = hashlib.sha256()
        for p in files:
            h.update(f"{p.name}\t{sha256_file(p)}\n".encode())
        return {"dinh_dang": self.dinh_dang, "duong_dan": rel(self.dir, config.PROJECT_ROOT), "so_file": len(files),
                "sha256": h.hexdigest(), "cach_bam": "sha256 cua cac dong '<ten file>\\t<sha256 file>\\n' da sap xep"}


class NguonCocoVid:
    """Mot file `*_cocofmt.json` (COCO-video). Doc ca file vao RAM (split train can nhieu RAM)."""

    dinh_dang = "cocovid"

    def __init__(self, duong_dan: Path):
        self.file = Path(duong_dan)
        self._theo_video: dict[str, list[Frame]] | None = None

    def ton_tai(self) -> bool:
        return self.file.is_file()

    def _nap(self) -> dict[str, list[Frame]]:
        if self._theo_video is None:
            from mot_datasets.bdd100k_coco import check_schema
            d = json.loads(self.file.read_text(encoding="utf-8"))
            try:
                check_schema(d)
            except ValueError as e:
                raise LoiDuLieu(f"nhan COCO-video sai schema: {e}") from e
            vid = {v["id"]: v["name"] for v in d["videos"]}
            cat = {c["id"]: c["name"] for c in d["categories"]}
            fr: dict[int, Frame] = {}
            theo: dict[str, list[Frame]] = {}
            for im in d["images"]:
                ten = vid[im["video_id"]]
                f = Frame(idx=int(im["frame_id"]), rel=str(im["file_name"]),
                          w=int(im["width"]) if im.get("width") else None,
                          h=int(im["height"]) if im.get("height") else None)
                if not f.rel.startswith(f"{ten}/"):
                    raise LoiDuLieu(f"file_name {f.rel!r} khong nam trong thu muc video {ten!r}")
                fr[im["id"]] = f
                theo.setdefault(ten, []).append(f)
            for a in d["annotations"]:
                x, y, w, h = (float(v) for v in a["bbox"])
                fr[a["image_id"]].nhan.append((cat[int(a["category_id"])], x, y, x + w, y + h,
                                               bool(a.get("iscrowd", 0)), bool(a.get("ignore", 0))))
            del d
            for v in theo.values():
                v.sort(key=lambda f: f.idx)
            self._theo_video = theo
        return self._theo_video

    def danh_sach_video(self) -> list[str]:
        return sorted(self._nap())

    def doc_video(self, video: str) -> list[Frame]:
        return self._nap()[video]

    def dau_van_tay(self) -> dict:
        return {"dinh_dang": self.dinh_dang, "duong_dan": rel(self.file, config.PROJECT_ROOT), "so_file": 1,
                "sha256": sha256_file(self.file), "cach_bam": "sha256 cua file"}


def mo_nguon(cfg_nhan: dict, goc: Path):
    dd = cfg_nhan.get("dinh_dang")
    p = Path(cfg_nhan["duong_dan"])
    p = p if p.is_absolute() else goc / p
    if dd == "scalabel":
        return NguonScalabel(p)
    if dd == "cocovid":
        return NguonCocoVid(p)
    raise LoiDuLieu(f"dinh_dang nhan {dd!r} khong ho tro (scalabel | cocovid)")


# =========================================================================== cau hinh
def doc_cau_hinh(path: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in ("ten", "nguon", "lop", "bdd", "chia", "lay_mau", "dau_ra", "tap_danh_gia_cam"):
        if k not in cfg:
            raise LoiDuLieu(f"cau hinh {path} thieu khoa {k!r}")
    if cfg["nguon"] != ["bdd100k_mot_train"]:
        raise LoiDuLieu(f"cau hinh nay chi cho phep nguon ['bdd100k_mot_train'] (gap {cfg['nguon']}); tron "
                        "UA-DETRAC hoac BDD val nam ngoai plan da chot")
    if list(cfg["lop"]) != list(LOP_5):
        raise LoiDuLieu(f"thu tu lop {cfg['lop']} khac thu tu da khoa {list(LOP_5)}")
    if cfg["bdd"].get("split") != "train":
        raise LoiDuLieu("bdd.split phai la 'train' (split train chinh thuc)")
    b = int(cfg["lay_mau"]["buoc_frame"])
    if b < 1:
        raise LoiDuLieu("lay_mau.buoc_frame phai >= 1")
    r = float(cfg["chia"]["ti_le_dev"])
    if not 0 < r < 1:
        raise LoiDuLieu("chia.ti_le_dev phai trong (0, 1)")
    cfg["_duong_dan"] = str(path)
    cfg["_sha256"] = sha256_file(Path(path))
    return cfg


def goc_du_an(cfg: dict) -> Path:
    return Path(cfg.get("goc_du_an") or config.PROJECT_ROOT)


def p_abs(cfg: dict, s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else goc_du_an(cfg) / p


# =========================================================================== kiem nguon
def kiem_nguon(cfg: dict) -> dict:
    """Kiem dau vao split train: nhan + anh theo tung video. Thieu -> ThieuDuLieu (liet ke cu the)."""
    goc = goc_du_an(cfg)
    nguon = mo_nguon(cfg["bdd"]["nhan"], goc)
    anh_dir = p_abs(cfg, cfg["bdd"]["anh"])
    thieu = []
    if not nguon.ton_tai():
        thieu.append(f"nhan train ({nguon.dinh_dang}): {getattr(nguon, 'dir', getattr(nguon, 'file', '?'))}")
    if not anh_dir.is_dir():
        thieu.append(f"anh train: {anh_dir}")
    if thieu:
        raise ThieuDuLieu("chua co du lieu BDD100K MOT train tren may", thieu)
    videos = nguon.danh_sach_video()
    co_anh = {p.name for p in anh_dir.iterdir() if p.is_dir()}
    khong_anh = [v for v in videos if v not in co_anh]
    thua_anh = sorted(co_anh - set(videos))
    bao_cao = {"dinh_dang": nguon.dinh_dang, "so_video_nhan": len(videos), "so_video_anh": len(co_anh),
               "video_co_nhan_khong_co_anh": khong_anh, "video_co_anh_khong_co_nhan": thua_anh,
               "video_thieu_anh_le": {}, "so_frame_nhan": 0, "so_anh_thieu": 0}
    for v in videos:
        if v in khong_anh:
            continue
        frames = nguon.doc_video(v)
        bao_cao["so_frame_nhan"] += len(frames)
        mat = [f.rel for f in frames if not (anh_dir / f.rel).is_file()]
        if mat:
            bao_cao["video_thieu_anh_le"][v] = {"so_thieu": len(mat), "vi_du": mat[:3]}
            bao_cao["so_anh_thieu"] += len(mat)
    chi_du = bool(cfg["bdd"].get("chi_video_du_anh", False))
    if (khong_anh or bao_cao["video_thieu_anh_le"]) and not chi_du:
        raise ThieuDuLieu(
            f"{len(khong_anh)} video co nhan nhung khong co thu muc anh, {len(bao_cao['video_thieu_anh_le'])} video "
            f"thieu {bao_cao['so_anh_thieu']} anh (bdd.chi_video_du_anh=false)",
            [f"anh video {v}: {anh_dir / v}" for v in khong_anh[:20]]
            + [f"anh le trong {v}: {x['vi_du']}" for v, x in list(bao_cao['video_thieu_anh_le'].items())[:20]])
    bao_cao["video_dung_duoc"] = [v for v in videos if v not in khong_anh and v not in bao_cao["video_thieu_anh_le"]]
    return bao_cao


# =========================================================================== thong ke + chia
def thong_ke_video(frames: list[Frame]) -> dict:
    """So box gan nhan theo lop tren TOAN BO frame cua video (truoc lay mau) + so vung to xam, lop khac."""
    tk = {c: 0 for c in LOP_5}
    tk["_to_xam"] = tk["_lop_khac"] = 0
    tk["_frame"] = len(frames)
    tk["_frame_co_nhan"] = sum(1 for f in frames if f.nhan)
    for f in frames:
        for cat, *_b, crowd, ign in f.nhan:
            r = vai_tro(cat, crowd, ign)
            if r == "gan_nhan":
                tk[cat] += 1
            elif r == "to_xam":
                tk["_to_xam"] += 1
            else:
                tk["_lop_khac"] += 1
    return tk


def chia_theo_video(tk_video: dict[str, dict], ti_le_dev: float, seed: int,
                    phan_tang: tuple[str, ...] = ("motorcycle", "bicycle")) -> dict:
    """Chia train-fit / dev theo video, phan tang theo su co mat cua lop hiem. Tat dinh voi seed."""
    tang: dict[tuple, list[str]] = {}
    for v, tk in tk_video.items():
        key = tuple(int(tk.get(c, 0) > 0) for c in phan_tang)
        tang.setdefault(key, []).append(v)
    rng = np.random.default_rng(seed)
    dev: list[str] = []
    mo_ta = {}
    for key in sorted(tang):
        vids = sorted(tang[key])
        perm = rng.permutation(len(vids))
        n_dev = int(np.floor(ti_le_dev * len(vids) + 0.5))          # lam tron nua len (khong kieu banker)
        dev += [vids[i] for i in perm[:n_dev]]
        mo_ta["|".join(f"{c}={k}" for c, k in zip(phan_tang, key))] = {"so_video": len(vids), "so_dev": n_dev}
    dev = sorted(dev)
    train = sorted(set(tk_video) - set(dev))
    return {"train_fit": train, "dev": dev, "phan_tang_theo": list(phan_tang), "tang": mo_ta}


def chon_frame(frames: list[Frame], buoc: int) -> tuple[list[Frame], dict]:
    """Quy tac 2 + 3: frame co idx % buoc == 0 va nam trong khoang co nhan cua video."""
    dem = {"khong_theo_buoc": 0, "ngoai_khoang_co_nhan": 0}
    co_nhan = [f.idx for f in frames if f.nhan]
    lo, hi = (min(co_nhan), max(co_nhan)) if co_nhan else (1, 0)
    chon = []
    for f in sorted(frames, key=lambda f: f.idx):
        if f.idx % buoc:
            dem["khong_theo_buoc"] += 1
        elif not lo <= f.idx <= hi:
            dem["ngoai_khoang_co_nhan"] += 1
        else:
            chon.append(f)
    return chon, dem


def ban_ghi_split(cfg: dict, nguon, chia: dict, tk_video: dict) -> dict:
    ds = {"train_fit": chia["train_fit"], "dev": chia["dev"]}
    return {
        "ten": cfg["ten"], "phien_ban_quy_tac": PHIEN_BAN_QUY_TAC,
        "tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
        "cau_hinh": {"duong_dan": rel(Path(cfg["_duong_dan"]), goc_du_an(cfg)), "sha256": cfg["_sha256"]},
        "nguon_nhan": nguon.dau_van_tay(), "anh": cfg["bdd"]["anh"],
        "chia": {"cach": "theo_video", "ti_le_dev": float(cfg["chia"]["ti_le_dev"]), "seed": int(cfg["chia"]["seed"]),
                 "phan_tang_theo": chia["phan_tang_theo"], "tang": chia["tang"]},
        "video_bi_loai_vi_thieu_anh": sorted(set(nguon.danh_sach_video()) - set(tk_video)),
        "so_video": {k: len(v) for k, v in ds.items()},
        "sha256_danh_sach": sha256_bytes(json.dumps(ds, sort_keys=True).encode()),
        "thong_ke_nguon_theo_split": {k: tong_hop_tk([tk_video[v] for v in vs]) for k, vs in ds.items()},
        **ds,
    }


def tong_hop_tk(ds_tk: list[dict]) -> dict:
    out = {"so_video": len(ds_tk), "so_frame": sum(t["_frame"] for t in ds_tk),
           "so_frame_co_nhan": sum(t["_frame_co_nhan"] for t in ds_tk), "box": {}, "video_co_lop": {}}
    for c in LOP_5:
        out["box"][c] = sum(t[c] for t in ds_tk)
        out["video_co_lop"][c] = sum(1 for t in ds_tk if t[c] > 0)
    out["vung_to_xam"] = sum(t["_to_xam"] for t in ds_tk)
    out["nhan_lop_khac"] = sum(t["_lop_khac"] for t in ds_tk)
    return out


def tao_split(cfg: dict, file_split: Path) -> dict:
    """Buoc 'chia': kiem nguon -> thong ke theo video -> chia -> ghi file split (tu choi ghi de)."""
    if file_split.exists():
        raise LoiDuLieu(f"{file_split} da ton tai -- split da khoa, khong ghi de (doi ten cau hinh neu can split moi)")
    kiem = kiem_nguon(cfg)
    nguon = mo_nguon(cfg["bdd"]["nhan"], goc_du_an(cfg))
    tk_video = {v: thong_ke_video(nguon.doc_video(v)) for v in kiem["video_dung_duoc"]}
    kiem_trung_ten_video(set(tk_video), cfg)
    chia = chia_theo_video(tk_video, float(cfg["chia"]["ti_le_dev"]), int(cfg["chia"]["seed"]),
                           tuple(cfg["chia"].get("phan_tang_theo", ("motorcycle", "bicycle"))))
    rec = ban_ghi_split(cfg, nguon, chia, tk_video)
    file_split.parent.mkdir(parents=True, exist_ok=True)
    file_split.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rec


# =========================================================================== tap danh gia cam
def _dir_hash_cache(cfg: dict) -> Path:
    d = p_abs(cfg, cfg.get("thu_muc_hash_tap_danh_gia", "data/interim/hash_tap_danh_gia"))
    d.mkdir(parents=True, exist_ok=True)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")
    return d


_TAP_CAM_CACHE: dict[tuple, list[dict]] = {}


def tap_cam(cfg: dict) -> list[dict]:
    """Thong tin nhan dien cua tung tap danh gia: ten video, ten anh (sha256 anh: `sha256_tap_bdd`).

    Nho theo (cau hinh, goc, trang thai thu muc) de khong doc lai JSON val 117 MB nhieu lan trong mot lan chay."""
    khoa = (cfg.get("_sha256"), str(goc_du_an(cfg)), json.dumps(cfg["tap_danh_gia_cam"], sort_keys=True),
            tuple(sorted(_dau_thu_muc(p_abs(cfg, t[k])) for t in cfg["tap_danh_gia_cam"]
                         for k in ("anh", "xml", "nhan_cocovid") if t.get(k))))
    if khoa not in _TAP_CAM_CACHE:
        _TAP_CAM_CACHE[khoa] = _tap_cam(cfg)
    return [dict(t) for t in _TAP_CAM_CACHE[khoa]]


def _dau_thu_muc(p: Path) -> tuple:
    """Dau hieu thay doi re tien: mtime + so muc con (thu muc) hoac size (file)."""
    if not p.exists():
        return (str(p), None)
    st = p.stat()
    return (str(p), st.st_mtime_ns, sum(1 for _ in p.iterdir()) if p.is_dir() else st.st_size)


def _tap_cam(cfg: dict) -> list[dict]:
    out = []
    for t in cfg["tap_danh_gia_cam"]:
        rec = {"ten": t["ten"], "loai": t["loai"], "video": set(), "ten_anh": set(), "nguon": []}
        if t["loai"] == "bdd_mot":
            if t.get("nhan_cocovid"):
                p = p_abs(cfg, t["nhan_cocovid"])
                if p.is_file():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    rec["video"] |= {v["name"] for v in d["videos"]}
                    rec["ten_anh"] |= {Path(i["file_name"]).name for i in d["images"]}
                    rec["nguon"].append(rel(p, goc_du_an(cfg)))
                    del d
            if t.get("anh"):
                a = p_abs(cfg, t["anh"])
                if a.is_dir():
                    for vd in a.iterdir():
                        if vd.is_dir():
                            rec["video"].add(vd.name)
                            rec["ten_anh"] |= {p.name for p in vd.glob("*.jpg")}
                    rec["nguon"].append(rel(a, goc_du_an(cfg)))
        elif t["loai"] == "detrac_xml":
            a = p_abs(cfg, t["xml"])
            if a.is_dir():
                rec["video"] |= {p.stem for p in a.glob("*.xml")}
                rec["nguon"].append(rel(a, goc_du_an(cfg)))
        else:
            raise LoiDuLieu(f"loai tap danh gia {t['loai']!r} khong ho tro")
        if not rec["video"] and t.get("bat_buoc", True):
            raise ThieuDuLieu(f"khong doc duoc tap danh gia {t['ten']} de kiem trung", [str(t)])
        out.append(rec)
    return out


def kiem_trung_ten_video(videos: set[str], cfg: dict) -> dict:
    bc = {}
    for t in tap_cam(cfg):
        trung = sorted(videos & t["video"])
        bc[t["ten"]] = {"so_video_tap": len(t["video"]), "trung": trung}
        if trung:
            raise LoiDuLieu(f"{len(trung)} video train trung tap danh gia {t['ten']}: {trung[:5]}")
    return bc


def sha256_tap_bdd(cfg: dict, t_cfg: dict) -> dict[str, str]:
    """sha256 moi anh cua tap danh gia BDD (cache theo (duong dan, size, mtime))."""
    a = p_abs(cfg, t_cfg["anh"])
    cache = _dir_hash_cache(cfg) / f"{t_cfg['ten']}_sha256.csv"
    cu = {}
    if cache.exists():
        with open(cache, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cu[r["rel"]] = r
    moi = {}
    for p in sorted(a.rglob("*.jpg")):
        r = p.relative_to(a).as_posix()
        st = p.stat()
        c = cu.get(r)
        if c and int(c["size"]) == st.st_size and c["mtime_ns"] == str(st.st_mtime_ns):
            moi[r] = c
        else:
            moi[r] = {"rel": r, "size": st.st_size, "mtime_ns": str(st.st_mtime_ns), "sha256": sha256_file(p)}
    with open(cache, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rel", "size", "mtime_ns", "sha256"])
        w.writeheader()
        w.writerows(moi.values())
    return {r: v["sha256"] for r, v in moi.items()}


# =========================================================================== xuat
def _hop_le_box(x1, y1, x2, y2, W, H):
    x1, x2 = max(0.0, min(float(W), x1)), max(0.0, min(float(W), x2))
    y1, y2 = max(0.0, min(float(H), y1)), max(0.0, min(float(H), y2))
    return (x1, y1, x2, y2) if (x2 - x1) >= 1.0 and (y2 - y1) >= 1.0 else None


def xuat_mot_anh(src: Path, frame: Frame, dst_img: Path, dst_lbl: Path, dst_ign: Path, dem: dict) -> dict:
    W, H = kich_thuoc_anh(src)
    if frame.w is not None and (frame.w, frame.h) != (W, H):
        raise LoiDuLieu(f"{src}: nhan khai bao {frame.w}x{frame.h} khac anh thuc {W}x{H}")
    boxes, vung = [], []
    for cat, x1, y1, x2, y2, crowd, ign in frame.nhan:
        r = vai_tro(cat, crowd, ign)
        if r == "lop_khac":
            dem["nhan_lop_khac"] += 1
            continue
        b = _hop_le_box(x1, y1, x2, y2, W, H)
        if b is None:
            dem["box_nho_hon_1px_sau_cat" if r == "gan_nhan" else "vung_to_xam_rong"] += 1
            continue
        if r == "gan_nhan":
            if (x1, y1, x2, y2) != b:
                dem["box_bi_cat_vao_khung"] += 1
            boxes.append((ID_LOP_5[cat], *b))
        else:
            vung.append(b)
    if vung:
        import cv2
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if im is None or im.shape[:2] != (H, W):
            raise LoiDuLieu(f"khong giai ma duoc {src} hoac kich thuoc khac header")
        m = np.zeros((H, W), bool)
        for x1, y1, x2, y2 in vung:
            m[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = True
        for _c, x1, y1, x2, y2 in boxes:
            m[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = False
        im[m] = MAU_TO_XAM
        ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, CHAT_LUONG_JPEG])
        if not ok:
            raise LoiDuLieu(f"khong ma hoa duoc {dst_img}")
        dst_img.write_bytes(enc.tobytes())
        # ghi BYTE (dong ket thuc '\n' co dinh, khong doi theo he dieu hanh) -> sha256 trong manifest on dinh
        dst_ign.write_bytes("".join(f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n" for x1, y1, x2, y2 in vung)
                            .encode("utf-8"))
        xu_ly = "to_xam"
        dem["so_pixel_to_xam"] += int(m.sum())
    else:
        shutil.copyfile(src, dst_img)
        xu_ly = "sao_chep"
    dst_lbl.write_bytes("".join(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} "
                                f"{(y2 - y1) / H:.6f}\n" for c, x1, y1, x2, y2 in boxes).encode("utf-8"))
    n = {c: 0 for c in LOP_5}
    for c, *_ in boxes:
        n[LOP_5[c]] += 1
    return {"w": W, "h": H, "n": n, "n_vung": len(vung), "xu_ly": xu_ly, "co_ignore": bool(vung)}


def doc_split(file_split: Path, nguon) -> dict:
    if not file_split.exists():
        raise ThieuDuLieu(f"chua co file split {file_split} -- chay --buoc chia truoc", [str(file_split)])
    sp = json.loads(file_split.read_text(encoding="utf-8"))
    ds = {"train_fit": sp["train_fit"], "dev": sp["dev"]}
    if sha256_bytes(json.dumps(ds, sort_keys=True).encode()) != sp["sha256_danh_sach"]:
        raise LoiDuLieu(f"{file_split}: danh sach video khong khop sha256_danh_sach (file bi sua tay?)")
    if set(sp["train_fit"]) & set(sp["dev"]):
        raise LoiDuLieu("split: train_fit va dev trung video")
    vt = nguon.dau_van_tay()
    if vt["sha256"] != sp["nguon_nhan"]["sha256"] or vt["dinh_dang"] != sp["nguon_nhan"]["dinh_dang"]:
        raise LoiDuLieu(f"nguon nhan da doi so voi luc khoa split ({vt['sha256'][:12]} != "
                        f"{sp['nguon_nhan']['sha256'][:12]}) -- khong xuat tren split cu")
    return sp


def chon_tap_con(sp: dict, cfg: dict, tap_con: str | None) -> dict[str, list[str]]:
    ds = {"train": list(sp["train_fit"]), "dev": list(sp["dev"])}
    if not tap_con:
        return ds
    tc = cfg.get("tap_con", {}).get(tap_con)
    if not tc:
        raise LoiDuLieu(f"cau hinh khong co tap_con.{tap_con}")
    rng = np.random.default_rng(int(tc["seed"]))
    out = {}
    for k, n in (("train", int(tc["so_video_train"])), ("dev", int(tc["so_video_dev"]))):
        v = sorted(ds[k])
        out[k] = sorted(v[i] for i in rng.permutation(len(v))[:n])
    return out


def ghi_data_yaml(out: Path, ghi_chu: str, path_cuoi: Path | None = None) -> Path:
    """`path` trong YAML = thu muc CUOI (sau khi doi ten tu thu muc tam)."""
    p = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(LOP_5))
    p.write_bytes((f"# SINH TU DONG boi src/bdd_yolo_export.py -- KHONG sua tay. {ghi_chu}\n"
                   f"# 'val' = DEV NOI BO tach tu BDD100K MOT train chinh thuc, KHONG phai BDD100K val chinh thuc.\n"
                   f"path: {(path_cuoi or out).resolve().as_posix()}\ntrain: images/train\nval: images/dev\n"
                   f"nc: {len(LOP_5)}\nnames:\n{names}\n").encode("utf-8"))
    return p


def xuat_dataset(cfg: dict, file_split: Path, tap_con: str | None = None, kiem_hash_noi_dung: bool = True) -> dict:
    """Buoc 'xuat': doc split da khoa -> xuat anh/nhan -> manifest -> kiem. Thu muc dich phai CHUA ton tai."""
    goc = goc_du_an(cfg)
    nguon = mo_nguon(cfg["bdd"]["nhan"], goc)
    if not nguon.ton_tai():
        kiem_nguon(cfg)                            # nem ThieuDuLieu voi danh sach cu the
    sp = doc_split(file_split, nguon)
    anh_dir = p_abs(cfg, cfg["bdd"]["anh"])
    tc = cfg.get("tap_con", {}).get(tap_con, {}) if tap_con else {}
    buoc = int(tc.get("buoc_frame", cfg["lay_mau"]["buoc_frame"]))
    out = p_abs(cfg, cfg["dau_ra"] + (f"_{tap_con}" if tap_con else ""))
    if out.exists():
        raise LoiDuLieu(f"{out} da ton tai -- khong ghi de dataset cu (xoa tay hoac doi ten cau hinh)")
    tam = out.with_name(out.name + ".dang_tao")
    if tam.exists():
        raise LoiDuLieu(f"{tam} con sot tu lan chay truoc bi ngat -- kiem tra roi xoa tay")
    chon = chon_tap_con(sp, cfg, tap_con)
    kiem_trung_ten_video(set(chon["train"]) | set(chon["dev"]), cfg)

    dem ={k: 0 for k in ("nhan_lop_khac", "box_nho_hon_1px_sau_cat", "vung_to_xam_rong", "box_bi_cat_vao_khung",
                          "so_pixel_to_xam")}
    dem_frame = {"khong_theo_buoc": 0, "ngoai_khoang_co_nhan": 0, "frame_nguon": 0}
    thong_ke = {s: {"so_video": len(chon[s]), "so_anh": 0, "am_tinh": 0, "anh_to_xam": 0,
                    "box": {c: 0 for c in LOP_5}, "anh_co_lop": {c: 0 for c in LOP_5},
                    "video_co_lop": {c: 0 for c in LOP_5}} for s in ("train", "dev")}
    # --- luot 1: chon frame + bam anh nguon, KIEM TRUNG tap danh gia TRUOC khi ghi bat ky anh nao
    ke_hoach = []
    sha_nguon: dict[str, str] = {}
    for s in ("train", "dev"):
        for v in chon[s]:
            frames = nguon.doc_video(v)
            dem_frame["frame_nguon"] += len(frames)
            ds_chon, d = chon_frame(frames, buoc)
            for k in ("khong_theo_buoc", "ngoai_khoang_co_nhan"):
                dem_frame[k] += d[k]
            for f in ds_chon:
                src = anh_dir / f.rel
                if not src.is_file():
                    raise ThieuDuLieu(f"thieu anh {src}", [str(src)])
                ten = Path(f.rel).name
                sha_nguon[f"{s}/{ten}"] = sha256_file(src)
                ke_hoach.append((s, v, f, src, ten))
    if set(chon["train"]) & set(chon["dev"]):
        raise LoiDuLieu("train va dev trung video")
    # train-fit / dev tach biet ca ve NOI DUNG (review 4937c85: khac ten video nhung cung byte anh van lot qua)
    sha_train = {h: k for k, h in sha_nguon.items() if k.startswith("train/")}
    trung_noi_bo = sorted((sha_train[h], k) for k, h in sha_nguon.items() if k.startswith("dev/") and h in sha_train)
    if trung_noi_bo:
        raise LoiDuLieu(f"{len(trung_noi_bo)} anh dev TRUNG NOI DUNG (sha256 anh nguon) voi anh train-fit du khac video, "
                        f"vd {trung_noi_bo[:3]} -- khong tu loai anh; can nguoi dung quyet dinh cach xu ly")
    trung = {"train_dev_noi_bo": {"cach": "sha256 anh nguon, truoc khi ghi", "so_cap_trung": 0,
                                  "so_anh_train": len(sha_train), "so_anh_dev": len(sha_nguon) - len(sha_train)}}
    ten_xuat = {k.split("/", 1)[1] for k in sha_nguon}
    for t, t_cfg in zip(tap_cam(cfg), cfg["tap_danh_gia_cam"]):
        tr_ten = sorted(ten_xuat & t["ten_anh"])
        tr_sha = []
        if kiem_hash_noi_dung and t["loai"] == "bdd_mot" and t_cfg.get("anh") and p_abs(cfg, t_cfg["anh"]).is_dir():
            bam = sha256_tap_bdd(cfg, t_cfg)
            sha_tap = set(bam.values())
            tr_sha = sorted(k for k, h in sha_nguon.items() if h in sha_tap)
            t["so_anh_da_bam"], t["so_hash_khac_nhau"] = len(bam), len(sha_tap)
        trung[t["ten"]] = {"so_video_tap": len(t["video"]), "so_ten_anh_tap": len(t["ten_anh"]),
                           "so_anh_da_bam": t.get("so_anh_da_bam", 0),
                           "so_hash_khac_nhau": t.get("so_hash_khac_nhau", 0), "trung_ten_anh": tr_ten[:20],
                           "trung_noi_dung": tr_sha[:20], "nguon": t["nguon"]}
        if tr_ten or tr_sha:
            raise LoiDuLieu(f"anh xuat trung tap danh gia {t['ten']}: ten {tr_ten[:3]}, noi dung {tr_sha[:3]}")
    if not kiem_hash_noi_dung:
        trung["_ghi_chu"] = "BO kiem sha256 noi dung (--bo-kiem-hash-noi-dung) -- chi hop le cho fixture"
    kt_trong = {s: {"so_video": len(chon[s])} for s in chon}

    # --- luot 2: ghi anh / nhan / vung bo qua vao thu muc tam, doi ten khi xong
    for s in ("train", "dev"):
        for sub in ("images", "labels", "ignore"):
            (tam / sub / s).mkdir(parents=True, exist_ok=True)
    (tam / "manifest").mkdir()
    (tam / ".gitignore").write_text("*\n", encoding="utf-8")
    rows = []
    co_lop_video: dict[tuple[str, str], set] = {}
    for s, v, f, src, ten in ke_hoach:
        stem = Path(ten).stem
        r = xuat_mot_anh(src, f, tam / "images" / s / ten, tam / "labels" / s / f"{stem}.txt",
                         tam / "ignore" / s / f"{stem}.txt", dem)
        ign_rel = f"ignore/{s}/{stem}.txt" if r["co_ignore"] else ""
        row = {"split": s, "video": v, "frame_idx": f.idx, "anh_nguon": f.rel, "sha256_nguon": sha_nguon[f"{s}/{ten}"],
               "anh_ra": f"images/{s}/{ten}", "sha256_ra": sha256_file(tam / "images" / s / ten),
               "w": r["w"], "h": r["h"], **{f"n_{c}": r["n"][c] for c in LOP_5},
               "n_vung_to_xam": r["n_vung"], "xu_ly": r["xu_ly"], "am_tinh": int(sum(r["n"].values()) == 0),
               "nhan_ra": f"labels/{s}/{stem}.txt", "sha256_nhan": sha256_file(tam / "labels" / s / f"{stem}.txt"),
               "ignore_ra": ign_rel, "sha256_ignore": sha256_file(tam / ign_rel) if ign_rel else ""}
        rows.append(row)
        t = thong_ke[s]
        t["so_anh"] += 1
        t["am_tinh"] += row["am_tinh"]
        t["anh_to_xam"] += int(r["xu_ly"] == "to_xam")
        cl = co_lop_video.setdefault((s, v), set())
        for c in LOP_5:
            t["box"][c] += r["n"][c]
            t["anh_co_lop"][c] += int(r["n"][c] > 0)
            if r["n"][c]:
                cl.add(c)
    for (s, _v), cl in co_lop_video.items():
        for c in cl:
            thong_ke[s]["video_co_lop"][c] += 1

    man = tam / "manifest" / "anh.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COT_MANIFEST)
        w.writeheader()
        w.writerows(rows)
    (tam / "manifest" / "video.json").write_bytes((json.dumps(chon, indent=1) + "\n").encode("utf-8"))
    shutil.copyfile(file_split, tam / "manifest" / "split_da_khoa.json")       # ban sao NGUYEN BYTE split da khoa
    ghi_data_yaml(tam, f"cau hinh {cfg['ten']}" + (f", tap con {tap_con}" if tap_con else ""), path_cuoi=out)
    tom = {
        "ten": cfg["ten"] + (f"_{tap_con}" if tap_con else ""), "tap_con": tap_con,
        "loai": "SMOKE (tap con nho, KHONG dung cho ket qua nghien cuu)" if tap_con else "day_du",
        "phien_ban_manifest": PHIEN_BAN_MANIFEST,
        "phien_ban_quy_tac": PHIEN_BAN_QUY_TAC, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"),
        "git_head": git_head(), "lop": list(LOP_5),
        "cau_hinh": {"duong_dan": rel(Path(cfg["_duong_dan"]), goc), "sha256": cfg["_sha256"]},
        "file_split": {"duong_dan": rel(file_split, goc), "duong_dan_tuyet_doi": file_split.resolve().as_posix(),
                       "sha256": sha256_file(file_split), "sha256_danh_sach": sp["sha256_danh_sach"],
                       "ban_sao": "manifest/split_da_khoa.json"},
        "sha256_data_yaml": sha256_file(tam / "data.yaml"),
        "sha256_video_json": sha256_file(tam / "manifest" / "video.json"),
        "nguon_nhan": sp["nguon_nhan"], "anh_goc": cfg["bdd"]["anh"],
        "quy_tac": {"buoc_frame": buoc, "am_tinh": "frame trong [frame co nhan dau, cuoi] cua video",
                    "to_xam": f"crowd/ignore thuoc 5 lop + {list(DISTRACTOR_XE)}; tru pixel trong box co nhan; "
                              f"mau {list(MAU_TO_XAM)}; JPEG q{CHAT_LUONG_JPEG}",
                    "lop_khong_gan_nhan": ["pedestrian", "rider", "train", "other person"],
                    "box_toi_thieu_px": 1.0, "toa_do": "theo kich thuoc anh thuc (header file)"},
        "video": {s: len(v) for s, v in chon.items()},
        "dem_frame": dem_frame, "dem_nhan": dem, "thong_ke": thong_ke, "kiem_trung": trung,
        "kiem_trong_noi_bo": kt_trong,
        "manifest_anh": {"duong_dan": "manifest/anh.csv", "so_dong": len(rows), "sha256": sha256_file(man)},
    }
    (tam / "manifest" / "tom_tat.json").write_text(json.dumps(tom, ensure_ascii=False, indent=1) + "\n",
                                                   encoding="utf-8")
    os.replace(tam, out)
    tom["kiem_sau_xuat"] = kiem_dataset(out)
    (out / "manifest" / "kiem_sau_xuat.json").write_text(
        json.dumps(tom["kiem_sau_xuat"], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return tom


# =========================================================================== hop dong kiem dataset (DUNG CHUNG)
#: Dinh dang anh ma ultralytics 8.4.141 nhan (`ultralytics.data.utils.IMG_FORMATS`). Runner so bang nay voi ban
#: ultralytics dang chay truoc khi train (khac -> dung), de cach liet ke o day DUNG la cach trainer liet ke.
IMG_FORMATS_ULTRALYTICS = frozenset({"avif", "bmp", "dng", "heic", "heif", "jp2", "jpeg", "jpg", "mpo", "png", "tif",
                                     "tiff", "webp"})
HOP_DONG = "bdd5-dataset-v2"
SPLIT_YAML = {"train": "images/train", "dev": "images/dev"}      # khoa trong data.yaml: train -> train, val -> dev


def _nc(p) -> str:
    """So sanh duong dan khong phan biet hoa/thuong tren Windows."""
    return os.path.normcase(os.path.abspath(str(p)))


def liet_ke_anh_nhu_ultralytics(d: Path) -> list[Path]:
    """Cach `BaseDataset.get_img_files` tim anh trong mot thu muc: glob DE QUY '**/*.*', loc theo duoi file."""
    import glob
    f = glob.glob(str(Path(glob.escape(str(d))) / "**" / "*.*"), recursive=True)
    return sorted(Path(x) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS_ULTRALYTICS)


def nhan_nhu_ultralytics(img: Path) -> Path:
    """`ultralytics.data.utils.img2label_paths`: thay '/images/' CUOI CUNG bang '/labels/', doi duoi thanh .txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    x = str(img)
    return Path(sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def doc_dong_nhan(b: bytes, ten: str) -> tuple[list[tuple], list[str]]:
    """Nhan YOLO (bytes) -> [(lop, cx, cy, w, h)] + danh sach loi dinh dang. File rong = 0 box (am tinh hop le)."""
    out, loi = [], []
    for ln in b.decode("utf-8", "replace").splitlines():
        p = ln.split()
        if not p:
            continue
        if len(p) != 5:
            loi.append(f"{ten}: dong sai dinh dang {ln!r}")
            continue
        try:
            c = int(p[0])
            cx, cy, w, h = (float(x) for x in p[1:])
        except ValueError:
            loi.append(f"{ten}: khong doc duoc so {ln!r}")
            continue
        if not 0 <= c < len(LOP_5):
            loi.append(f"{ten}: lop {c} ngoai 0..{len(LOP_5) - 1}")
            continue
        if not (w > 0 and h > 0 and cx - w / 2 >= -1e-6 and cy - h / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6
                and cy + h / 2 <= 1 + 1e-6):
            loi.append(f"{ten}: toa do ngoai [0,1] {ln!r}")
        out.append((c, cx, cy, w, h))
    return out, loi


def kiem_hop_dong_dataset(data_yaml, split_noi_dung=("train", "dev"), gioi_han_loi: int = 20) -> dict:
    """HOP DONG kiem dataset DUNG CHUNG cho exporter (`--buoc xuat` / `--buoc kiem`), runner huan luyen va evaluator.

    Kiem DUNG cac duong dan ultralytics se dung khi nhan `data_yaml` NGUYEN VAN:
      1. YAML: `path` tuyet doi = thu muc chua data.yaml (ultralytics resolve `path` tuong doi theo DATASETS_DIR, khong
         theo file YAML); `train` == images/train, `val` == images/dev, khong `test`; names = 5 lop da khoa.
      2. Manifest phien ban 2 (ban cu -> LOI, khong tu bo sung hash); sha256 anh.csv, data.yaml, video.json; ban sao split
         da khoa (va ban goc neu con tren may) khop sha256; video train/dev nam trong split da khoa, roi nhau.
      3. Danh sach anh THUC TE (liet ke nhu ultralytics: glob de quy + IMG_FORMATS) == manifest tung split; file nhan
         (duong dan nhu ultralytics suy ra) PHAI ton tai -- file RONG hop le, file THIEU la loi; file ignore ton tai
         DUNG khi manifest khai bao vung ignore (anh khong co vung ignore -> KHONG duoc co file); khong co file thua.
      4. NOI DUNG (cac split trong `split_noi_dung`): sha256 tung anh, tung file nhan, tung file ignore khop manifest;
         kich thuoc anh khop; so box theo lop khop; khong trung noi dung anh giua train va dev.
      5. CACHE ANH (vong 29, R28-F1): khong co `<anh>.npy` canh anh train/dev nao -- ultralytics `load_image` nap .npy
         THAY JPEG ke ca khi cache=False; chinh sach hien tai (cache: false) = pixel nap chinh la JPEG da khoa sha256.
         (Khac cache NHAN `labels/<split>.cache`: cai do runner doi chieu o callback truoc vong train.)
    Loi -> LoiDuLieu (liet ke toi da `gioi_han_loi` muc). Dat -> DANH TINH dataset da xac minh (ghi vao ket qua).
    """
    import yaml
    from detector_classes import LoiAnhXaLop, kiem_names_dataset
    t0 = time.time()
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise LoiDuLieu(f"khong co {data_yaml}")
    goc = data_yaml.parent.resolve()
    try:
        y = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise LoiDuLieu(f"{data_yaml}: YAML hong ({type(e).__name__}: {e})") from e
    if not isinstance(y, dict):
        raise LoiDuLieu(f"{data_yaml}: khong phai mapping YAML")
    try:
        kiem_names_dataset(y.get("names") or {})
    except LoiAnhXaLop as e:
        raise LoiDuLieu(f"{data_yaml}: {e}") from e
    if y.get("nc", len(LOP_5)) != len(LOP_5):
        raise LoiDuLieu(f"{data_yaml}: nc={y.get('nc')!r} khac {len(LOP_5)}")
    if y.get("train") != SPLIT_YAML["train"] or y.get("val") != SPLIT_YAML["dev"]:
        raise LoiDuLieu(f"{data_yaml}: train/val phai DUNG la '{SPLIT_YAML['train']}' / '{SPLIT_YAML['dev']}' (dev noi bo); "
                        f"gap train={y.get('train')!r}, val={y.get('val')!r} -> ultralytics se train/val tren duong dan khac "
                        f"hop dong da khoa")
    if y.get("test") is not None:
        raise LoiDuLieu(f"{data_yaml}: khong duoc co khoa 'test' (gap {y.get('test')!r})")
    p = y.get("path")
    if not p or not Path(str(p)).is_absolute() or _nc(Path(str(p)).resolve()) != _nc(goc):
        raise LoiDuLieu(f"{data_yaml}: 'path' phai la duong dan TUYET DOI toi chinh thu muc chua data.yaml ({goc}); gap "
                        f"{p!r} (ultralytics resolve 'path' tuong doi theo DATASETS_DIR, khong theo file YAML)")
    tom_p, man = goc / "manifest" / "tom_tat.json", goc / "manifest" / "anh.csv"
    if not tom_p.is_file() or not man.is_file():
        raise LoiDuLieu(f"{goc}: thieu manifest/tom_tat.json hoac manifest/anh.csv (dataset phai do bdd_yolo_export sinh)")
    tom = json.loads(tom_p.read_text(encoding="utf-8"))
    pb = tom.get("phien_ban_manifest", 1)
    if pb != PHIEN_BAN_MANIFEST:
        raise LoiDuLieu(
            f"{goc}: manifest phien ban {pb} (can {PHIEN_BAN_MANIFEST}). Ban 1 (vong 27) chi co sha256 anh, CHUA khoa noi dung "
            f"file nhan va file ignore -> KHONG du bang chung de xac minh dataset (R27-F2). Xuat lai dataset bang ma hien tai; "
            f"KHONG bo sung hash vao manifest cu roi coi nhu da xac minh tu luc xuat.")

    loi: list[str] = []
    for ten, f, ky in (("manifest/anh.csv", man, tom["manifest_anh"]["sha256"]),
                       ("data.yaml", data_yaml, tom.get("sha256_data_yaml")),
                       ("manifest/video.json", goc / "manifest" / "video.json", tom.get("sha256_video_json"))):
        if not f.is_file():
            loi.append(f"thieu {ten}")
        elif sha256_file(f) != ky:
            loi.append(f"{ten} khac sha256 trong tom_tat.json (bi sua sau khi xuat)")
    fsp = tom["file_split"]
    ban_sao = goc / fsp.get("ban_sao", "manifest/split_da_khoa.json")
    if not ban_sao.is_file() or sha256_file(ban_sao) != fsp["sha256"]:
        loi.append(f"ban sao split da khoa {ban_sao.name} thieu hoac khac sha256 luc xuat")
    goc_split = Path(fsp.get("duong_dan_tuyet_doi", ""))
    if fsp.get("duong_dan_tuyet_doi") and goc_split.is_file():
        trang_thai_split_goc = "khop" if sha256_file(goc_split) == fsp["sha256"] else "DA DOI"
        if trang_thai_split_goc != "khop":
            loi.append(f"file split goc {goc_split} da doi sau khi xuat dataset")
    else:
        trang_thai_split_goc = "khong co tren may nay (chi kiem ban sao trong dataset)"
    if loi:
        raise LoiDuLieu(f"{len(loi)} loi hop dong o {goc}: " + "; ".join(loi[:gioi_han_loi]))

    sp = json.loads(ban_sao.read_text(encoding="utf-8"))
    ds = {"train_fit": sp["train_fit"], "dev": sp["dev"]}
    bam_ds = sha256_bytes(json.dumps(ds, sort_keys=True).encode())
    if bam_ds != sp["sha256_danh_sach"] or bam_ds != fsp["sha256_danh_sach"]:
        loi.append("danh sach video trong split da khoa khong khop sha256_danh_sach")
    vj = json.loads((goc / "manifest" / "video.json").read_text(encoding="utf-8"))
    v_tr, v_dv = set(vj.get("train", [])), set(vj.get("dev", []))
    if not v_tr <= set(sp["train_fit"]):
        loi.append(f"{len(v_tr - set(sp['train_fit']))} video train KHONG thuoc train_fit cua split da khoa")
    if not v_dv <= set(sp["dev"]):
        loi.append(f"{len(v_dv - set(sp['dev']))} video dev KHONG thuoc dev cua split da khoa")
    if tom.get("tap_con") is None and (v_tr != set(sp["train_fit"]) or v_dv != set(sp["dev"])):
        loi.append("dataset day du phai dung TOAN BO video cua split da khoa")
    if v_tr & v_dv:
        loi.append(f"train/dev trung video {sorted(v_tr & v_dv)[:3]}")

    with open(man, newline="", encoding="utf-8") as f:
        doc = csv.DictReader(f)
        thieu_cot = [c for c in COT_MANIFEST if c not in (doc.fieldnames or [])]
        rows = list(doc)
    if thieu_cot:
        raise LoiDuLieu(f"manifest/anh.csv thieu cot {thieu_cot} (phien ban khai bao {pb})")
    theo_split: dict[str, list[dict]] = {"train": [], "dev": []}
    for r in rows:
        if r["split"] not in theo_split:
            loi.append(f"dong manifest co split la {r['split']!r}")
            continue
        theo_split[r["split"]].append(r)
        if r["video"] not in (v_tr if r["split"] == "train" else v_dv):
            loi.append(f"{r['anh_ra']}: video {r['video']} khong thuoc {r['split']} trong video.json")
    if len({r["anh_ra"] for r in rows}) != len(rows):
        loi.append("manifest co anh_ra trung lap")
    for cot, mo_ta in (("anh_ra", "ten anh"), ("sha256_ra", "noi dung anh xuat"), ("sha256_nguon", "noi dung anh nguon")):
        a = {Path(r[cot]).name if cot == "anh_ra" else r[cot] for r in theo_split["train"]}
        b = {Path(r[cot]).name if cot == "anh_ra" else r[cot] for r in theo_split["dev"]}
        if a & b:
            loi.append(f"train/dev trung {mo_ta}: {len(a & b)} muc (vd {sorted(a & b)[:2]})")

    dem_bam = {"anh": 0, "nhan": 0, "ignore": 0}
    so_anh, so_nhan_rong, so_ignore, so_npy = {}, {}, {}, {}
    for s, thu_muc in (("train", goc / "images" / "train"), ("dev", goc / "images" / "dev")):
        noi_dung = s in split_noi_dung
        if not thu_muc.is_dir():
            loi.append(f"thieu thu muc {thu_muc}")
            continue
        tren_dia = {p.relative_to(goc).as_posix(): p for p in liet_ke_anh_nhu_ultralytics(thu_muc)}
        khai = {r["anh_ra"] for r in theo_split[s]}
        thua, thieu = sorted(set(tren_dia) - khai), sorted(khai - set(tren_dia))
        if thua:
            loi.append(f"images/{s}: {len(thua)} anh KHONG co trong manifest nhung ultralytics van se dung (vd {thua[:2]})")
        if thieu:
            loi.append(f"images/{s}: {len(thieu)} anh trong manifest KHONG ton tai (vd {thieu[:2]})")
        # Cache ANH (khac cache NHAN .cache): ultralytics `BaseDataset.load_image` doc `<anh>.npy` (cung thu muc, cung ten)
        # THAY anh JPEG neu file ton tai, KE CA khi cache=False (R28-F1). Chinh sach hien tai (cache: false): khong dung
        # cache anh -> .npy canh anh train/dev la loi. Chi kiem .npy canh CHINH cac anh ultralytics se nap; khong tu xoa.
        npy_thay = sorted(p.with_suffix(".npy").relative_to(goc).as_posix() for p in tren_dia.values()
                          if p.with_suffix(".npy").exists())
        if npy_thay:
            loi.append(f"images/{s}: {len(npy_thay)} file cache anh .npy canh anh da khoa (vd {npy_thay[:3]}) -- ultralytics "
                       f"load_image() nap .npy THAY anh JPEG ke ca khi cache=False; chinh sach cache: false khong dung "
                       f"cache anh -> di chuyen/xoa TAY cac file .npy nay (hop dong khong tu xoa)")
        so_npy[s] = len(npy_thay)
        so_anh[s] = len(tren_dia)
        so_nhan_rong[s] = so_ignore[s] = 0
        nhan_khai, ign_khai = set(), set()
        for r in theo_split[s]:
            img = goc / r["anh_ra"]
            lbl = goc / r["nhan_ra"]
            if _nc(nhan_nhu_ultralytics(img)) != _nc(lbl):
                loi.append(f"{r['anh_ra']}: nhan_ra {r['nhan_ra']} khong phai file ultralytics se doc")
            nhan_khai.add(_nc(lbl))
            if not lbl.is_file():
                loi.append(f"THIEU file nhan {r['nhan_ra']} (ultralytics se coi anh la NEN; khac voi file nhan RONG hop le)")
            else:
                b = lbl.read_bytes()
                if noi_dung:
                    dem_bam["nhan"] += 1
                    if sha256_bytes(b) != r["sha256_nhan"]:
                        loi.append(f"{r['nhan_ra']}: noi dung nhan khac manifest (sha256)")
                dong, ld = doc_dong_nhan(b, r["nhan_ra"])
                loi += ld
                n = {c: 0 for c in LOP_5}
                for c, *_ in dong:
                    n[LOP_5[c]] += 1
                if any(n[c] != int(r[f"n_{c}"]) for c in LOP_5):
                    loi.append(f"{r['nhan_ra']}: so box {n} khac manifest")
                so_nhan_rong[s] += int(not dong)
            n_v = int(r["n_vung_to_xam"])
            ign_dung = f"ignore/{s}/{Path(r['anh_ra']).stem}.txt"
            if n_v > 0:
                so_ignore[s] += 1
                ign_khai.add(_nc(goc / ign_dung))
                if r["ignore_ra"] != ign_dung or not r["sha256_ignore"]:
                    loi.append(f"{r['anh_ra']}: manifest khai bao {n_v} vung ignore nhung ignore_ra/sha256 sai")
                elif not (goc / ign_dung).is_file():
                    loi.append(f"THIEU file ignore da khai bao {ign_dung} ({n_v} vung) -- danh gia se coi la 0 vung")
                else:
                    bi = (goc / ign_dung).read_bytes()
                    if noi_dung:
                        dem_bam["ignore"] += 1
                        if sha256_bytes(bi) != r["sha256_ignore"]:
                            loi.append(f"{ign_dung}: noi dung vung ignore khac manifest (sha256)")
                    if sum(1 for ln in bi.decode("utf-8", "replace").splitlines() if ln.strip()) != n_v:
                        loi.append(f"{ign_dung}: so vung khac manifest ({n_v})")
            elif r["ignore_ra"] or r["sha256_ignore"]:
                loi.append(f"{r['anh_ra']}: manifest mau thuan (0 vung nhung co ignore_ra/sha256_ignore)")
            if not img.is_file():
                continue
            W, H = kich_thuoc_anh(img)
            if (W, H) != (int(r["w"]), int(r["h"])):
                loi.append(f"{r['anh_ra']}: kich thuoc {W}x{H} khac manifest {r['w']}x{r['h']}")
            if noi_dung:
                dem_bam["anh"] += 1
                if sha256_file(img) != r["sha256_ra"]:
                    loi.append(f"{r['anh_ra']}: noi dung anh khac manifest (sha256) -- kich thuoc {W}x{H}")
        for thu, khai_bao, mo_ta in ((goc / "labels" / s, nhan_khai, "file nhan"), (goc / "ignore" / s, ign_khai, "file ignore")):
            if thu.is_dir():
                thua_f = sorted(p.relative_to(goc).as_posix() for p in thu.rglob("*") if p.is_file() and _nc(p) not in khai_bao)
                if thua_f:
                    loi.append(f"{thu.relative_to(goc).as_posix()}: {len(thua_f)} {mo_ta} KHONG khai bao trong manifest "
                               f"(vd {thua_f[:2]})")
    if loi:
        raise LoiDuLieu(f"{len(loi)} loi hop dong o {goc}: " + "; ".join(loi[:gioi_han_loi]))
    return {
        "hop_dong": HOP_DONG, "phien_ban_manifest": pb, "loi": 0, "ten": tom.get("ten"), "loai": tom.get("loai"),
        "tap_con": tom.get("tap_con"), "data_yaml": str(data_yaml.resolve()), "sha256_data_yaml": sha256_file(data_yaml),
        "sha256_tom_tat": sha256_file(tom_p), "manifest_sha256": tom["manifest_anh"]["sha256"],
        "sha256_video_json": tom.get("sha256_video_json"),
        "split_da_khoa": {"sha256": fsp["sha256"], "sha256_danh_sach": fsp["sha256_danh_sach"],
                          "ban_goc": trang_thai_split_goc},
        "duong_dan_resolve": {"train": str(goc / "images" / "train"), "val": str(goc / "images" / "dev")},
        "so_anh": so_anh, "so_video": {"train": len({r['video'] for r in theo_split['train']}),
                                       "dev": len({r['video'] for r in theo_split['dev']})},
        "so_nhan_rong_hop_le": so_nhan_rong, "so_anh_co_file_ignore": so_ignore,
        "cache_anh": {"chinh_sach": "khong dung cache anh (.npy); pixel nap = JPEG da khoa sha256",
                      "so_npy_canh_anh": so_npy},
        "noi_dung_da_xac_minh": {s: s in split_noi_dung for s in ("train", "dev")}, "so_file_da_bam": dem_bam,
        "xac_minh_luc": dt.datetime.now().isoformat(timespec="seconds"), "thoi_gian_s": round(time.time() - t0, 2),
    }


def kiem_dataset(out: Path) -> dict:
    """Kiem day du (noi dung ca train + dev) dataset tai thu muc `out` -- cung hop dong voi runner / evaluator."""
    return kiem_hop_dong_dataset(Path(out) / "data.yaml")


def gt_theo_manifest(data_yaml, split: str = "dev") -> dict[str, dict]:
    """Khai bao cua manifest cho tung anh cua split: {ten anh: {'ignore': bool, 'n_vung': int}} (evaluator dung de phan
    biet 'anh khong co vung ignore' voi 'file ignore da khai bao nhung bi thieu')."""
    goc = Path(data_yaml).resolve().parent
    with open(goc / "manifest" / "anh.csv", newline="", encoding="utf-8") as f:
        return {Path(r["anh_ra"]).name: {"ignore": int(r["n_vung_to_xam"]) > 0, "n_vung": int(r["n_vung_to_xam"])}
                for r in csv.DictReader(f) if r["split"] == split}
