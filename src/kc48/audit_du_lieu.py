"""
audit_du_lieu.py -- R48B / M2 (+ phan kiem cau hinh augmentation cua M5): audit CHI DOC du lieu UA-DETRAC va
BDD100K MOT: nguon, split, phan bo lop, chat luong anh/nhan, ban chuyen nhan YOLO, trung / gan trung xuyen split,
quy dao doi huong, nhom xe nho canh xe lon, augmentation hieu luc cua checkpoint.

KHONG sua / xoa / tao file trong data/, models/. Moi file ghi nam trong results/phase2/kc48_audit/ (kiem bang
`_duong_ghi`): anh dataset va overlay tren anh dataset CHI o results/phase2/kc48_audit/overlay/ (co .gitignore '*');
bo nho dem lon o results/phase2/kc48_audit/_dem/ (tu tao .gitignore '*').

Cac phan (chay rieng; moi lan < 10 phut; phan `anh` tiep tuc theo video):
  cong_bo     ghi tieu chi (quy dao, mau truc quan, gan trung, xe nho canh xe lon, mau YOLO) TRUOC khi tinh; khong ghi de
  nhan        --tap bdd_train|bdd_val|detrac_train|detrac_test : phan bo, chat luong nhan, quy dao, canh xe lon, mau
  anh         --tap bdd_train|bdd_val|detrac : doc anh (sha256 | header), SOI/EOI, giai ma day du + dHash tren mau
  trung       trung chinh xac (sha256) + gan trung (dHash-256) xuyen video / split
  yolo        kiem ban chuyen nhan YOLO data/yolo_bdd5_v1 tren mau phan tang (tai suy tu nhan goc, doc lap voi exporter)
  nguon       hash tung file nhan BDD train so voi manifest nhap vong 34 + hash tong so voi split / tom_tat xuat
  checkpoint  log huan luyen -> augmentation hieu luc, epoch, so buoc optimizer; sha256 trong so
  truc_quan   overlay mau phan tang (chi trong overlay/)
  tong_hop    CSV tong hop + moc doi chieu slide lan 2 + bieu do

Lenh: C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/audit_du_lieu.py --phan <phan> [--tap <tap>]

Sua loi hau audit (review_003 F1, 2026-10-11): track nghi_nhieu theo MAX dpsi bi loai khoi tap hop le cua ca chi so
chinh (max) lan do nhay (p90); truoc do ma ap nguong nghi_nhieu len p90 doc lap va tinh nghi_nhieu vao n_hop_le.
Tieu chi va nguong (tieu_chi_cong_bo.json) KHONG doi. Kiem: src/kc48/kiem_loc_quy_dao.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
OUT = GOC / "results" / "phase2" / "kc48_audit"
DEM = OUT / "_dem"
OVERLAY = OUT / "overlay"
FILE_TIEU_CHI = OUT / "tieu_chi_cong_bo.json"

LOP5 = ("car", "truck", "bus", "motorcycle", "bicycle")
DISTRACTOR_XE = ("other vehicle", "trailer")                    # src/bdd_yolo_export.py:68
DISTRACTOR_BDD = ("other person", "trailer", "other vehicle")   # src/mot_datasets/bdd100k_mot.py:82
LOAI_DETRAC = ("car", "bus", "van", "others")
DETRAC_SANG_5LOP = {"car": "car", "bus": "bus", "van": "vung_bo_qua (D1)", "others": "vung_bo_qua (D2)"}
FPS = {"bdd": 5, "detrac": 25}          # BDD MOT nhan 5 fps (configs/phase2/detector/bdd5_train.yaml:25); src/config.py:100
KT_DANH_NGHIA = {"bdd": (1280, 720), "detrac": (960, 540)}
BUOC_FRAME = 5                            # configs/phase2/detector/bdd5_train.yaml:25

P = {
    "bdd_nhan_train": GOC / "data/raw/bdd100k/labels/box_track_20/train",
    "bdd_anh_train": GOC / "data/raw/bdd100k/images/track/train",
    "bdd_anh_val": GOC / "data/raw/bdd100k/images/track/val",
    "bdd_val_coco": GOC / "data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json",
    "split": GOC / "configs/phase2/detector/splits/bdd5_train_v1.json",
    "yolo": GOC / "data/yolo_bdd5_v1",
    "xml_train": GOC / "data/extracted/DETRAC-Train-Annotations-XML",
    "xml_test": GOC / "data/extracted/DETRAC-Test-Annotations-XML",
    "detrac_anh": GOC / "data/extracted/DETRAC-Images",
    "man_dev": GOC / "results/phase2/manifests/detrac_dev.json",
    "man_test": GOC / "results/phase2/manifests/detrac_test.json",
    "train_log": GOC / "results/phase2/detector_train",
    "nhap_r34": GOC / "handoff/logs/review34/03_manifest_nhap_nhan_train.tsv",
    "cach_ly_hf": GOC / "data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot",
    "train_cfg": GOC / "configs/phase2/detector/train_bdd5.yaml",
}

TIEU_CHI = {
    "phien_ban": "kc48-audit-v1",
    "quy_dao": {
        "tam": "tam hop (cx, cy) [px] tren anh, nhan goc (khong lam tron)",
        "fps": FPS,
        "nua_cua_so_huong_s": 0.4,
        "khoang_doi_huong_s": 1.0,
        "do_dai_toi_thieu_s": 2.0,
        "dich_chuyen_toi_thieu_px": 8.0,
        "dich_chuyen_toi_thieu_theo_h": 0.25,
        "ti_le_huong_hop_le_toi_thieu": 0.5,
        "nguong_deg": {"thang_duoi": 15, "cong_nhe_duoi": 30, "cua_duoi": 60, "nghi_nhieu_tren": 150},
        "dinh_nghia": ("huong psi(t) = atan2 cua c(t+k) - c(t-k), k = nua_cua_so*fps frame, chi khi ca hai frame co mat va "
                       "|dich chuyen| >= max(dich_chuyen_toi_thieu_px, dich_chuyen_toi_thieu_theo_h * h(t)); "
                       "dpsi(t) = |wrap(psi(t+T) - psi(t))|, T = khoang_doi_huong*fps frame, ca hai huong hop le. "
                       "Loai: do dai < do_dai_toi_thieu -> loai_ngan; ti le huong hop le / huong tinh duoc < nguong hoac "
                       "khong co dpsi -> loai_cham; max dpsi > nghi_nhieu_tren -> nghi_nhieu (dao chieu: nhieu / dung / "
                       "doi ID); < thang_duoi -> thang; < cong_nhe_duoi -> cong_nhe; < cua_duoi -> cua; con lai -> cua_gat."),
        "chi_so_chinh": "max dpsi tren track",
        "chi_so_do_nhay": "p90 dpsi tren track (giam anh huong mot diem ngoai lai)",
        "lop": {"bdd": "5 lop, box gan nhan (khong crowd / distractor)", "detrac": "car, bus, van, others (loai goc)"},
        "ghi_chu_bdd": "camera gan tren xe dang chay (ego-motion): ket qua la DOI HUONG TREN ANH, khong phai xe cua that",
        "ghi_chu_detrac": "camera co dinh, phoi canh: goc tren anh khac goc tren mat duong",
    },
    "mau_truc_quan": {
        "seed": 48,
        "tap_con": ["bdd_trainfit_b5", "bdd_val", "detrac_train60", "detrac_test40"],
        "lop": {"bdd": list(LOP5), "detrac": list(LOAI_DETRAC)},
        "kich_thuoc": "dien tich px goc: nho < 32^2, vua < 96^2, lon >= 96^2 (quy uoc COCO)",
        "dieu_kien": {"bdd_train": "kho = occluded hoac truncated (Scalabel); thuong = con lai",
                      "bdd_val": "unknown (COCO-video khong co occluded/truncated) -> mot nhom",
                      "detrac": "kho = co region_overlap hoac truncation_ratio > 0 hoac sence_weather in {night, rainy}"},
        "cach_chon": ("moi o (lop x kich thuoc x dieu kien): 1 box, lay mau hoc chua (reservoir) deu tren moi box cua o, "
                      "duyet video theo ten, frame theo thu tu; numpy default_rng(seed) rieng tung tap con"),
        "anh_toan_khung": "2 frame / tap con (reservoir tren frame co >= 1 box gan nhan), ve moi box",
        "gioi_han": "mau nho, chi phat hien loai loi; KHONG ket luan toan tap sach",
    },
    "gan_trung": {
        "mau_frame": "moi video 4 frame: vi tri floor(0.1n), floor(0.5n), floor(0.9n) + 1 chi so ngau nhien "
                     "(default_rng(crc32(video) ^ 48)) trong danh sach anh da sap xep",
        "hash": "dHash 256 bit: anh xam resize 17x16 (BILINEAR), so sanh cot lien ke",
        "nguong_ung_vien": 20,
        "pham_vi": "chi so cap thuoc HAI VIDEO KHAC NHAU, trong cung mien (BDD voi BDD, DETRAC voi DETRAC)",
        "ghi_chu": "ung vien phai xac nhan bang mat; frame lien nhau trong mot video khong tinh la boi canh doc lap",
    },
    "canh_xe_lon": {
        "nho": {"bdd": ["car", "motorcycle", "bicycle"], "detrac": ["car"]},
        "lon": {"bdd": ["bus", "truck"], "detrac": ["bus", "others"]},
        "dieu_kien": "dien tich box lon >= 2 x box nho VA giao / dien tich box nho (IoA) >= 0.10, cung frame",
        "su_kien": "chuoi frame lien tiep (khoang cach <= 1 frame nhan) cua cung cap (track nho, track lon)",
        "detrac_theo_nhan": "region_overlap co occlusion_id tro toi target lop lon trong cung frame (dem rieng)",
        "ghi_chu": "others cua UA-DETRAC la lop hon hop, thanh phan khong xac minh",
    },
    "mau_yolo": {"seed": 48, "tang": "split (train/dev) x xu_ly (sao_chep/to_xam) x co lop hiem (motorcycle|bicycle)",
                 "so_moi_tang": 40, "dung_sai_toa_do_chuan_hoa": 1e-6, "dung_sai_vung_px": 0.006},
}


# =========================================================================== tien ich
def _duong_ghi(p: Path) -> Path:
    p = Path(p).resolve()
    if OUT.resolve() not in p.parents and p != OUT.resolve():
        raise RuntimeError(f"tu choi ghi ngoai {OUT}: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dam_bao_dem():
    gi = _duong_ghi(DEM / ".gitignore")
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")


def ghi_json(p: Path, obj) -> None:
    p = _duong_ghi(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def ghi_csv(p: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    p = _duong_ghi(p)
    if cols is None:
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)


def doc_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def doc_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        g = GOC / ".git"
        h = (g / "HEAD").read_text(encoding="utf-8").strip()
        if not h.startswith("ref:"):
            return h
        ref = h.split(None, 1)[1]
        if (g / ref).exists():
            return (g / ref).read_text(encoding="utf-8").strip()
        for line in (g / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "?"


def nap_tieu_chi() -> tuple[dict, str]:
    if not FILE_TIEU_CHI.exists():
        raise SystemExit("chua co tieu_chi_cong_bo.json: chay --phan cong_bo truoc")
    d = doc_json(FILE_TIEU_CHI)
    return d, sha256_file(FILE_TIEU_CHI)


def pv(a, qs=(5, 25, 50, 75, 95)) -> dict:
    x = np.frombuffer(a, dtype=np.float32) if isinstance(a, array) else np.asarray(a, dtype=float)
    if len(x) == 0:
        return {f"p{q}": "" for q in qs}
    return {f"p{q}": round(float(v), 4) for q, v in zip(qs, np.percentile(x, qs))}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def vai_tro(cat: str, crowd: bool, ign: bool) -> str:
    """Quy tac 4-5 cua src/bdd_yolo_export.py (viet lai doc lap): gan_nhan | to_xam | lop_khac."""
    if cat in LOP5 and not crowd and not ign:
        return "gan_nhan"
    if (crowd or ign) and (cat in LOP5 or cat in DISTRACTOR_XE):
        return "to_xam"
    return "lop_khac"


def _list_jpg(d: Path):
    if not d.is_dir():
        return None
    return {e.name for e in os.scandir(d) if e.is_file() and e.name.lower().endswith(".jpg")}


def _ioa(a, b) -> float:
    """giao(a, b) / dien tich a; a, b = (x1, y1, x2, y2)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    return iw * ih / ar if ar > 0 else 0.0


# =========================================================================== bo dem
class ChatLuong:
    def __init__(self):
        self.d: dict[tuple, list] = {}

    def dem(self, tap: str, kiem: str, loi: bool, vi_du=None, n: int = 1):
        r = self.d.setdefault((tap, kiem), [0, 0, []])
        r[0] += n
        if loi:
            r[1] += 1 if n == 1 else n
            if vi_du is not None and len(r[2]) < 5:
                r[2].append(str(vi_du))

    def dem_n(self, tap: str, kiem: str, n_kiem: int, n_loi: int, vi_du=None):
        r = self.d.setdefault((tap, kiem), [0, 0, []])
        r[0] += n_kiem
        r[1] += n_loi
        for v in (vi_du or []):
            if len(r[2]) < 5:
                r[2].append(str(v))

    def rows(self) -> list[dict]:
        return [{"tap": t, "kiem": k, "n_kiem": a, "n_loi": b, "ti_le_loi": round(b / a, 6) if a else "",
                 "vi_du": " ; ".join(c)} for (t, k), (a, b, c) in sorted(self.d.items())]


class TichLuy:
    """Thong ke theo lop cho mot tap con."""

    def __init__(self, ten: str, lop, fps: float):
        self.ten, self.lop, self.fps = ten, tuple(lop), fps
        self.video: set[str] = set()
        self.n_frame = 0
        self.box = Counter()
        self.n_track = Counter()
        self.video_lop = {c: set() for c in self.lop}
        self.rel = {c: array("f") for c in self.lop}
        self.h640 = {c: array("f") for c in self.lop}
        self.bin = {c: Counter() for c in self.lop}
        self.occ = {c: Counter() for c in self.lop}
        self.trunc = {c: Counter() for c in self.lop}
        self.occ_r = {c: array("f") for c in self.lop}
        self.trunc_r = {c: array("f") for c in self.lop}
        self.mat_do = {c: array("H") for c in self.lop}
        self.mat_do_tong = array("H")
        self.track_len = {c: array("I") for c in self.lop}
        self.dk_video = Counter()
        self.dk_box = Counter()

    def them_box(self, video, c, x1, y1, x2, y2, W, H, occ, trunc, occ_r, trunc_r):
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        self.box[c] += 1
        self.video_lop[c].add(video)
        self.rel[c].append(math.sqrt(w * h / (W * H)))
        self.h640[c].append(h * 640.0 / max(W, H))
        a = w * h
        self.bin[c]["nho" if a < 32 * 32 else "vua" if a < 96 * 96 else "lon"] += 1
        self.occ[c]["unknown" if occ is None else "co" if occ else "khong"] += 1
        self.trunc[c]["unknown" if trunc is None else "co" if trunc else "khong"] += 1
        if occ_r is not None:
            self.occ_r[c].append(occ_r)
        if trunc_r is not None:
            self.trunc_r[c].append(trunc_r)

    def them_frame(self, dem: Counter):
        self.n_frame += 1
        for c in self.lop:
            self.mat_do[c].append(min(dem.get(c, 0), 65535))
        self.mat_do_tong.append(min(sum(dem.get(c, 0) for c in self.lop), 65535))

    def ket_thuc_video(self, video: str, cnt: Counter, meta: dict):
        self.video.add(video)
        nb = 0
        for (c, _tid), n in cnt.items():
            self.n_track[c] += 1
            self.track_len[c].append(n)
            nb += n
        for k, v in meta.items():
            self.dk_video[(k, v)] += 1
            self.dk_box[(k, v)] += nb

    def tom_tat(self) -> list[dict]:
        rows, nf = [], self.n_frame
        for c in self.lop:
            nb = self.box[c]
            md = np.frombuffer(self.mat_do[c], dtype=np.uint16)
            r = {"tap_con": self.ten, "lop": c, "n_video_tap": len(self.video), "n_frame_tap": nf, "n_bbox": nb,
                 "n_track": self.n_track[c], "n_video_co_lop": len(self.video_lop[c]),
                 "ti_le_video_co_lop": round(len(self.video_lop[c]) / max(1, len(self.video)), 4)}
            r.update({f"kt_rel_{k}": v for k, v in pv(self.rel[c]).items()})
            r.update({f"h640px_{k}": v for k, v in pv(self.h640[c], (5, 50)).items()})
            for b in ("nho", "vua", "lon"):
                r[f"ti_le_{b}"] = round(self.bin[c][b] / nb, 4) if nb else ""
            co = md[md > 0]
            r["mat_do_tb_moi_frame"] = round(float(md.mean()), 4) if nf else ""
            r["ti_le_frame_co_lop"] = round(len(co) / nf, 4) if nf else ""
            r["mat_do_tb_khi_co"] = round(float(co.mean()), 3) if len(co) else ""
            r["mat_do_p95"] = float(np.percentile(md, 95)) if nf else ""
            o, t = self.occ[c], self.trunc[c]
            r["ti_le_che"] = round(o["co"] / (o["co"] + o["khong"]), 4) if o["co"] + o["khong"] else "unknown"
            r["ti_le_cat_khung"] = round(t["co"] / (t["co"] + t["khong"]), 4) if t["co"] + t["khong"] else "unknown"
            oc = np.frombuffer(self.occ_r[c], dtype=np.float32)
            tr = np.frombuffer(self.trunc_r[c], dtype=np.float32)
            r["ti_le_che_ge_0_5"] = round(float((oc >= 0.5).mean()), 4) if len(oc) else "unknown"
            r["che_ratio_p50_khi_che"] = round(float(np.median(oc[oc > 0])), 4) if (oc > 0).any() else (
                "unknown" if not len(oc) else "")
            r["trunc_ratio_p95"] = round(float(np.percentile(tr, 95)), 4) if len(tr) else "unknown"
            tl = np.frombuffer(self.track_len[c], dtype=np.uint32)
            r["track_len_p50_frame"] = float(np.median(tl)) if len(tl) else ""
            r["track_len_p50_s"] = round(float(np.median(tl)) / self.fps, 2) if len(tl) else ""
            rows.append(r)
        return rows

    def tom_tat_dk(self) -> list[dict]:
        return [{"tap_con": self.ten, "khoa": k, "gia_tri": v, "n_video": n, "n_bbox": self.dk_box[(k, v)]}
                for (k, v), n in sorted(self.dk_video.items())]

    def mat_do_tong_tt(self) -> dict:
        md = np.frombuffer(self.mat_do_tong, dtype=np.uint16)
        if not len(md):
            return {}
        return {"tap_con": self.ten, "n_frame": len(md), "tb": round(float(md.mean()), 3),
                "p50": float(np.median(md)), "p95": float(np.percentile(md, 95)), "max": int(md.max()),
                "ti_le_frame_khong_box": round(float((md == 0).mean()), 4)}


class MauTrucQuan:
    def __init__(self, ten: str, seed: int, k_frame: int = 2):
        self.ten, self.rng, self.k = ten, np.random.default_rng(seed), k_frame
        self.o: dict[str, list] = {}
        self.frames: list = []
        self.n_frame = 0

    def box(self, cell: str, item: dict):
        r = self.o.setdefault(cell, [0, None])
        r[0] += 1
        if r[0] == 1 or int(self.rng.integers(r[0])) == 0:
            r[1] = item

    def frame(self, item_fn):
        i = self.n_frame
        self.n_frame += 1
        if i < self.k:
            self.frames.append(item_fn())
        else:
            j = int(self.rng.integers(i + 1))
            if j < self.k:
                self.frames[j] = item_fn()

    def to_json(self) -> dict:
        return {"tap_con": self.ten, "o": {k: {"n_trong_o": v[0], "chon": v[1]} for k, v in sorted(self.o.items())},
                "frame": self.frames, "n_frame_ung_vien": self.n_frame}


# =========================================================================== quy dao
def _loai(x: float, g: dict) -> str:
    if x > g["nghi_nhieu_tren"]:
        return "nghi_nhieu"
    if x < g["thang_duoi"]:
        return "thang"
    if x < g["cong_nhe_duoi"]:
        return "cong_nhe"
    if x < g["cua_duoi"]:
        return "cua"
    return "cua_gat"


def phan_loai_track(dm_deg: float, d90_deg: float, g: dict) -> dict:
    """nghi_nhieu xet tren MAX dpsi (tieu chi cong bo: 'max dpsi > nghi_nhieu_tren -> nghi_nhieu') va loai track khoi
    tap hop le cua CA chi so chinh (max) LAN do nhay (p90). loai_p90_tho = phan loai p90 khi chua loc nghi_nhieu, chi
    dung de bao cao truoc / sau loc (sua loi hau audit F1)."""
    lm, lp = _loai(dm_deg, g), _loai(d90_deg, g)
    return {"loai_max": lm, "loai_p90": "nghi_nhieu" if lm == "nghi_nhieu" else lp, "loai_p90_tho": lp}


def quy_dao(fr, cx, cy, hh, fps: float, tc: dict) -> dict:
    order = sorted(range(len(fr)), key=lambda i: fr[i])
    pos: dict[int, int] = {}
    for i in order:
        pos.setdefault(fr[i], i)
    fs = sorted(pos)
    dur = (fs[-1] - fs[0] + 1) / fps
    out = {"n_frame": len(fs), "thoi_gian_s": round(dur, 3), "n_huong_tinh": 0, "n_huong_hop_le": 0, "n_dpsi": 0,
           "max_dpsi_deg": "", "p90_dpsi_deg": ""}
    if dur < tc["do_dai_toi_thieu_s"]:
        out["loai_max"] = out["loai_p90"] = out["loai_p90_tho"] = "loai_ngan"
        return out
    k = max(1, round(tc["nua_cua_so_huong_s"] * fps))
    T = max(1, round(tc["khoang_doi_huong_s"] * fps))
    psi, n_tinh = {}, 0
    for f in fs:
        a, b = pos.get(f - k), pos.get(f + k)
        if a is None or b is None:
            continue
        n_tinh += 1
        dx, dy = cx[b] - cx[a], cy[b] - cy[a]
        if math.hypot(dx, dy) >= max(tc["dich_chuyen_toi_thieu_px"], tc["dich_chuyen_toi_thieu_theo_h"] * hh[pos[f]]):
            psi[f] = math.atan2(dy, dx)
    d = [abs(_wrap(psi[f + T] - p)) for f, p in psi.items() if (f + T) in psi]
    out.update(n_huong_tinh=n_tinh, n_huong_hop_le=len(psi), n_dpsi=len(d))
    if n_tinh == 0 or len(psi) / n_tinh < tc["ti_le_huong_hop_le_toi_thieu"] or not d:
        out["loai_max"] = out["loai_p90"] = out["loai_p90_tho"] = "loai_cham"
        return out
    dm, d90 = math.degrees(max(d)), math.degrees(float(np.percentile(d, 90)))
    out.update(max_dpsi_deg=round(dm, 2), p90_dpsi_deg=round(d90, 2), **phan_loai_track(dm, d90, tc["nguong_deg"]))
    return out


LOAI_QD = ("loai_ngan", "loai_cham", "nghi_nhieu", "thang", "cong_nhe", "cua", "cua_gat")
NHOM_CD = ("thang", "cong_nhe", "cua", "cua_gat")
COT_QD_TH = (["tap_con", "lop", "chi_so", "n_track", "loai_ngan", "loai_cham", "n_truoc_loc_nhieu", "nghi_nhieu",
              "n_hop_le", *NHOM_CD, "ti_le_cua_hoac_gat_tren_hop_le", "ti_le_cua_gat_tren_hop_le",
              "ti_le_nghi_nhieu_tren_truoc_loc"] + [f"truoc_loc_{x}" for x in NHOM_CD + ("nghi_nhieu",)] +
             ["ti_le_cua_hoac_gat_truoc_loc", "nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc"])


def tong_hop_quy_dao(qrows: list[dict]) -> list[dict]:
    """Dem theo (tap_con, lop, chi_so). Tap hop le = bo loai_ngan, loai_cham va nghi_nhieu (theo max, chung cho hai chi
    so) -> n_hop_le = thang + cong_nhe + cua + cua_gat. Cot truoc_loc_*: phan loai cua chinh chi so khi CHUA loai
    nghi_nhieu, mau so n_truoc_loc_nhieu = n_track - loai_ngan - loai_cham (= so da bao cao truoc khi sua F1)."""
    agg = Counter()
    for r in qrows:
        tho = {"loai_max": r["loai_max"], "loai_p90": r["loai_p90_tho"]}
        for s in r["tap_con"].split("|"):
            for lop in (r["lop"], "_tat_ca"):
                for chi_so in ("loai_max", "loai_p90"):
                    k = (s, lop, chi_so)
                    agg[k + ("loc", r[chi_so])] += 1
                    agg[k + ("tho", tho[chi_so])] += 1
                    agg[k + ("nghi_tho_cua",)] += r["loai_max"] == "nghi_nhieu" and tho[chi_so] in ("cua", "cua_gat")
    out = []
    for k in sorted({k[:3] for k in agg}):
        r = dict(zip(("tap_con", "lop", "chi_so"), k))
        r.update({x: agg[k + ("loc", x)] for x in LOAI_QD})
        r["n_track"] = sum(r[x] for x in LOAI_QD)
        nt = r["n_track"] - r["loai_ngan"] - r["loai_cham"]
        hl = nt - r["nghi_nhieu"]
        if hl != sum(r[x] for x in NHOM_CD):
            raise RuntimeError(f"bat bien n_hop_le = tong 4 nhom hong o {k}")
        r.update(n_truoc_loc_nhieu=nt, n_hop_le=hl,
                 ti_le_cua_hoac_gat_tren_hop_le=round((r["cua"] + r["cua_gat"]) / hl, 4) if hl else "",
                 ti_le_cua_gat_tren_hop_le=round(r["cua_gat"] / hl, 4) if hl else "",
                 ti_le_nghi_nhieu_tren_truoc_loc=round(r["nghi_nhieu"] / nt, 4) if nt else "")
        r.update({f"truoc_loc_{x}": agg[k + ("tho", x)] for x in NHOM_CD + ("nghi_nhieu",)})
        r["ti_le_cua_hoac_gat_truoc_loc"] = round((r["truoc_loc_cua"] + r["truoc_loc_cua_gat"]) / nt, 4) if nt else ""
        r["nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc"] = agg[k + ("nghi_tho_cua",)]
        out.append(r)
    return out


def dem_su_kien(pairs: list[tuple]) -> tuple[int, int]:
    by = defaultdict(set)
    for s, l, f in pairs:
        by[(s, l)].add(f)
    n_ev = 0
    for fs in by.values():
        fs = sorted(fs)
        n_ev += 1 + sum(1 for a, b in zip(fs, fs[1:]) if b - a > 1)
    return n_ev, len({s for s, _l, _f in pairs})


# =========================================================================== doc nhan theo video
class VideoNhan:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _bool_hoac_none(at: dict, k: str):
    return bool(at[k]) if k in at else None


def it_bdd_train(q: ChatLuong, raw: dict):
    sp = doc_json(P["split"])
    tf, dv = set(sp["train_fit"]), set(sp["dev"])
    for p in sorted(P["bdd_nhan_train"].glob("*.json")):
        v = p.stem
        s = "trainfit" if v in tf else "dev" if v in dv else None
        if s is None:
            q.dem("bdd_train", "video_nhan_ngoai_split", True, v)
            continue
        tq = f"bdd_{s}"
        frames = doc_json(p)
        dl = _list_jpg(P["bdd_anh_train"] / v)
        q.dem(tq, "video_khong_co_thu_muc_anh", dl is None, v)
        dl = dl or set()
        idx = [int(f["frameIndex"]) for f in frames if f.get("labels")]
        lo, hi = (min(idx), max(idx)) if idx else (1, 0)
        out, ten_nhan = [], set()
        for f in frames:
            fi, ten = int(f["frameIndex"]), f["name"]
            ten_nhan.add(ten)
            q.dem(tq, "frame_nhan_thieu_anh", ten not in dl, f"{v}/{ten}")
            q.dem(tq, "videoName_khac_ten_file", (f.get("videoName") or v) != v, f"{v}/{ten}")
            b5 = fi % BUOC_FRAME == 0 and lo <= fi <= hi
            boxes = []
            for lb in f.get("labels") or []:
                cat, at, b = lb.get("category"), lb.get("attributes") or {}, lb.get("box2d")
                crowd = bool(at.get("crowd", False))
                raw[tq][f"{cat}|crowd={int(crowd)}"] += 1
                if not b:
                    q.dem(tq, "nhan_thieu_box2d", True, f"{v}/{ten}")
                    continue
                boxes.append((vai_tro(cat, crowd, cat in DISTRACTOR_BDD), cat, float(b["x1"]), float(b["y1"]),
                              float(b["x2"]), float(b["y2"]), str(lb.get("id")), _bool_hoac_none(at, "occluded"),
                              _bool_hoac_none(at, "truncated"), None, None, ()))
            out.append((fi, ten, b5, boxes))
        thua = sorted(dl - ten_nhan)
        q.dem_n(tq, "anh_khong_co_frame_nhan", len(dl), len(thua), [f"{v}/{x}" for x in thua[:1]])
        yield VideoNhan(video=v, ds="bdd", tap_q=tq, sub_all=[f"bdd_{s}_all"], sub_b5=[f"bdd_{s}_b5"],
                        sub_qd=f"bdd_{s}", W=1280, H=720, meta={}, frames=out, ign=[],
                        anh_rel=f"data/raw/bdd100k/images/track/train/{v}")


def it_bdd_val(q: ChatLuong, raw: dict):
    d = doc_json(P["bdd_val_coco"])
    cats = {c["id"]: c["name"] for c in d["categories"]}
    vids = {x["id"]: x["name"] for x in d["videos"]}
    ann = defaultdict(list)
    for a in d["annotations"]:
        ann[a["image_id"]].append(a)
    theo_v = defaultdict(list)
    for im in d["images"]:
        theo_v[vids[im["video_id"]]].append(im)
    del d
    tq = "bdd_val"
    for v in sorted(theo_v):
        ims = sorted(theo_v[v], key=lambda im: im["frame_id"])
        dl = _list_jpg(P["bdd_anh_val"] / v)
        q.dem(tq, "video_khong_co_thu_muc_anh", dl is None, v)
        dl = dl or set()
        out, ten_nhan, WH = [], set(), None
        for im in ims:
            fn = str(im["file_name"])
            ten = fn.split("/", 1)[1] if "/" in fn else fn
            q.dem(tq, "file_name_ngoai_thu_muc_video", not fn.startswith(f"{v}/"), fn)
            ten_nhan.add(ten)
            q.dem(tq, "frame_nhan_thieu_anh", ten not in dl, fn)
            W, H = int(im.get("width") or 1280), int(im.get("height") or 720)
            q.dem(tq, "kich_thuoc_khai_bao_khac_1280x720", (W, H) != (1280, 720), fn)
            WH = WH or (W, H)
            boxes = []
            for a in ann.get(im["id"], []):
                cat = cats[int(a["category_id"])]
                crowd, ign = bool(a.get("iscrowd", 0)), bool(a.get("ignore", 0))
                raw[tq][f"{cat}|iscrowd={int(crowd)}|ignore={int(ign)}"] += 1
                x, y, w, h = (float(t) for t in a["bbox"])
                boxes.append((vai_tro(cat, crowd, ign), cat, x, y, x + w, y + h, str(a.get("instance_id")),
                              None, None, None, None, ()))
            out.append((int(im["frame_id"]), ten, False, boxes))
        thua = sorted(dl - ten_nhan)
        q.dem_n(tq, "anh_khong_co_frame_nhan", len(dl), len(thua), [f"{v}/{x}" for x in thua[:1]])
        W, H = WH or (1280, 720)
        yield VideoNhan(video=v, ds="bdd", tap_q=tq, sub_all=["bdd_val"], sub_b5=[], sub_qd="bdd_val", W=W, H=H,
                        meta={}, frames=out, ign=[], anh_rel=f"data/raw/bdd100k/images/track/val/{v}")


def doc_xml_detrac(p: Path):
    root = ET.parse(p).getroot()
    sa = root.find("sequence_attribute")
    meta = {k: sa.get(k) for k in ("sence_weather", "camera_state")} if sa is not None else {}
    ign = []
    ir = root.find("ignored_region")
    if ir is not None:
        for b in ir.findall("box"):
            x, y, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            ign.append((x, y, x + w, y + h))
    frames = []
    for fr in root.findall("frame"):
        tl = fr.find("target_list")
        ts = []
        for t in (tl.findall("target") if tl is not None else []):
            b, at = t.find("box"), t.find("attribute")
            x, y, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
            occl = []
            oc = t.find("occlusion")
            if oc is not None:
                for ro in oc.findall("region_overlap"):
                    rx, ry, rw, rh = (float(ro.get(k)) for k in ("left", "top", "width", "height"))
                    occl.append((rx, ry, rx + rw, ry + rh, ro.get("occlusion_id"), ro.get("occlusion_status")))
            ts.append({"id": t.get("id"), "box": (x, y, x + w, y + h),
                       "loai": at.get("vehicle_type") if at is not None else None,
                       "trunc": float(at.get("truncation_ratio")) if at is not None and at.get("truncation_ratio")
                       is not None else None, "occl": occl})
        frames.append((int(fr.get("num")), fr.get("density"), ts))
    return root.get("name"), meta, ign, frames


def it_detrac(part: str, q: ChatLuong, raw: dict):
    dev = set(doc_json(P["man_dev"])["videos"])
    xdir = P["xml_train" if part == "train" else "xml_test"]
    tq = "detrac_train60" if part == "train" else "detrac_test40"
    for p in sorted(xdir.glob("*.xml")):
        v = p.stem
        ten_seq, meta, ign, frames = doc_xml_detrac(p)
        q.dem(tq, "ten_sequence_khac_ten_file", ten_seq != v, v)
        dl = _list_jpg(P["detrac_anh"] / v)
        q.dem(tq, "video_khong_co_thu_muc_anh", dl is None, v)
        dl = dl or set()
        q.dem_n(tq, "vung_bo_qua_ignored_region", 1, len(ign))
        out, ten_nhan = [], set()
        for num, _dens, ts in frames:
            ten = f"img{num:05d}.jpg"
            ten_nhan.add(ten)
            q.dem(tq, "frame_nhan_thieu_anh", ten not in dl, f"{v}/{ten}")
            boxes = []
            for t in ts:
                x1, y1, x2, y2 = t["box"]
                raw[tq][f"{t['loai']}"] += 1
                ar = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                s = 0.0
                for rx1, ry1, rx2, ry2, _oid, ost in t["occl"]:
                    raw[tq][f"occlusion_status={ost}"] += 1
                    s += _ioa((rx1, ry1, rx2, ry2), t["box"]) * (rx2 - rx1) * (ry2 - ry1)
                    q.dem(tq, "region_overlap_ngoai_box(>1px)", rx1 < x1 - 1 or ry1 < y1 - 1 or rx2 > x2 + 1
                          or ry2 > y2 + 1, f"{v}/{ten}#{t['id']}")
                occ_r = min(1.0, s / ar) if ar > 0 else 0.0
                q.dem(tq, "loai_xe_ngoai_4_loai", t["loai"] not in LOAI_DETRAC, f"{v}/{ten}#{t['id']}:{t['loai']}")
                q.dem(tq, "nhan_trong_vung_bo_qua(IoA>=0.5)", any(_ioa(t["box"], g) >= 0.5 for g in ign),
                      f"{v}/{ten}#{t['id']}")
                boxes.append(("gan_nhan" if t["loai"] in LOAI_DETRAC else "lop_khac", t["loai"], x1, y1, x2, y2,
                              str(t["id"]), bool(t["occl"]), (t["trunc"] or 0.0) > 0, occ_r, t["trunc"],
                              tuple(str(o[4]) for o in t["occl"])))
            out.append((num, ten, False, boxes))
        thua = sorted(dl - ten_nhan)
        q.dem_n(tq, "anh_khong_co_frame_nhan", len(dl), len(thua), [f"{v}/{x}" for x in thua[:1]])
        sub = [tq] + (["detrac_dev12"] if v in dev else [])
        yield VideoNhan(video=v, ds="detrac", tap_q=tq, sub_all=sub, sub_b5=[], sub_qd="|".join(sub), W=960, H=540,
                        meta=meta, frames=out, ign=ign, anh_rel=f"data/extracted/DETRAC-Images/{v}")


# =========================================================================== xu ly chung
def kiem_box(q, tq, x1, y1, x2, y2, W, H, ex):
    if not all(math.isfinite(t) for t in (x1, y1, x2, y2)):
        q.dem(tq, "box_khong_huu_han", True, ex)
        return
    q.dem(tq, "box_suy_bien(w<=0|h<=0)", x2 <= x1 or y2 <= y1, ex)
    q.dem(tq, "box_canh<1px", (x2 - x1) < 1 or (y2 - y1) < 1, ex)
    q.dem(tq, "box_ngoai_anh(bat_ky)", x1 < 0 or y1 < 0 or x2 > W or y2 > H, ex)
    q.dem(tq, "box_ngoai_anh(>1px)", x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1, ex)


def _dk(ds, occ, trunc, occ_r, trunc_r, meta) -> str:
    if ds == "detrac":
        return "kho" if (occ_r or 0) > 0 or (trunc_r or 0) > 0 or meta.get("sence_weather") in ("night", "rainy") \
            else "thuong"
    if occ is None and trunc is None:
        return "unknown"
    return "kho" if occ or trunc else "thuong"


def xu_ly_video(vn, acc, q, qd_rows, canh_rows, mau, tc):
    fps = FPS[vn.ds]
    lop_nho, lop_lon = set(tc["canh_xe_lon"]["nho"][vn.ds]), set(tc["canh_xe_lon"]["lon"][vn.ds])
    tracks = {}
    cnt = defaultdict(Counter)
    pairs, pairs_nhan = [], []
    for fi, ten, b5, boxes in vn.frames:
        subs = vn.sub_all + (vn.sub_b5 if b5 else [])
        dem, seen = Counter(), Counter()
        nho, lon, by_id = [], [], {}
        for role, cat, x1, y1, x2, y2, tid, occ, trunc, occ_r, trunc_r, occl in boxes:
            kiem_box(q, vn.tap_q, x1, y1, x2, y2, vn.W, vn.H, f"{vn.video}/{ten}#{tid}")
            if role != "gan_nhan":
                continue
            seen[tid] += 1
            dem[cat] += 1
            for s in subs:
                acc[s].them_box(vn.video, cat, x1, y1, x2, y2, vn.W, vn.H, occ, trunc, occ_r, trunc_r)
                cnt[s][(cat, tid)] += 1
            t = tracks.setdefault(tid, ([], [], [], [], Counter()))
            t[0].append(fi)
            t[1].append((x1 + x2) / 2)
            t[2].append((y1 + y2) / 2)
            t[3].append(y2 - y1)
            t[4][cat] += 1
            bx = (x1, y1, x2, y2)
            ar = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            by_id[tid] = (cat, ar)
            if cat in lop_nho:
                nho.append((tid, bx, ar, occl))
            if cat in lop_lon:
                lon.append((tid, bx, ar))
            for s in subs:
                if s in mau:
                    kt = "nho" if ar < 32 * 32 else "vua" if ar < 96 * 96 else "lon"
                    dk = _dk(vn.ds, occ, trunc, occ_r, trunc_r, vn.meta)
                    mau[s].box(f"{cat}|{kt}|{dk}", {"video": vn.video, "anh": f"{vn.anh_rel}/{ten}", "frame": fi,
                                                    "lop": cat, "kt": kt, "dk": dk, "box": [x1, y1, x2, y2],
                                                    "track": tid, "occ": occ, "trunc": trunc, "occ_r": occ_r,
                                                    "trunc_r": trunc_r, "thoi_tiet": vn.meta.get("sence_weather")})
        q.dem(vn.tap_q, "trung_id_trong_frame", any(n > 1 for n in seen.values()), f"{vn.video}/{ten}")
        for s in subs:
            acc[s].them_frame(dem)
        for tid_s, bs, as_, occl in nho:
            for tid_l, bl, al in lon:
                if tid_l != tid_s and al >= 2 * as_ and _ioa(bs, bl) >= 0.10:
                    pairs.append((tid_s, tid_l, fi))
            for oid in occl:
                if oid in by_id and by_id[oid][0] in lop_lon and oid != tid_s:
                    pairs_nhan.append((tid_s, oid, fi))
        if dem:
            for s in subs:
                if s in mau:
                    mau[s].frame(lambda: {"video": vn.video, "anh": f"{vn.anh_rel}/{ten}", "frame": fi,
                                          "boxes": [[b[0], b[1], b[2], b[3], b[4], b[5]] for b in boxes],
                                          "vung_bo_qua": [list(g) for g in vn.ign]})
    for s in set(vn.sub_all):
        acc[s].ket_thuc_video(vn.video, cnt[s], vn.meta)
    for s in set(vn.sub_b5):
        acc[s].ket_thuc_video(vn.video, cnt[s], vn.meta)
    for tid, (fr, cx, cy, hh, cats) in tracks.items():
        q.dem(vn.tap_q, "track_doi_lop", len(cats) > 1, f"{vn.video}#{tid}:{dict(cats)}")
        r = quy_dao(fr, cx, cy, hh, fps, tc["quy_dao"])
        r.update({"tap_con": vn.sub_qd, "video": vn.video, "track": tid, "lop": cats.most_common(1)[0][0]})
        qd_rows.append(r)
    n_ev, n_nho = dem_su_kien(pairs)
    n_ev_n, n_nho_n = dem_su_kien(pairs_nhan)
    canh_rows.append({"tap_con": vn.sub_qd, "video": vn.video, "thoi_tiet": vn.meta.get("sence_weather", "unknown"),
                      "n_track": len(tracks), "n_cap_frame": len({(a, b, f) for a, b, f in pairs}),
                      "n_su_kien": n_ev, "n_track_nho_lien_quan": n_nho,
                      "n_cap_frame_theo_nhan_che": len(set(pairs_nhan)) if vn.ds == "detrac" else "unknown",
                      "n_su_kien_theo_nhan_che": n_ev_n if vn.ds == "detrac" else "unknown",
                      "n_track_nho_theo_nhan_che": n_nho_n if vn.ds == "detrac" else "unknown"})


COT_QD = ["tap_con", "video", "track", "lop", "n_frame", "thoi_gian_s", "n_huong_tinh", "n_huong_hop_le", "n_dpsi",
          "max_dpsi_deg", "p90_dpsi_deg", "loai_max", "loai_p90", "loai_p90_tho"]


def phan_nhan(tap: str):
    tc, sha_tc = nap_tieu_chi()
    t0 = time.time()
    q, raw = ChatLuong(), defaultdict(Counter)
    if tap == "bdd_train":
        it = it_bdd_train(q, raw)
        acc = {k: TichLuy(k, LOP5, 5.0 if k.endswith("_all") else 1.0)
               for k in ("bdd_trainfit_all", "bdd_trainfit_b5", "bdd_dev_all", "bdd_dev_b5")}
    elif tap == "bdd_val":
        it = it_bdd_val(q, raw)
        acc = {"bdd_val": TichLuy("bdd_val", LOP5, 5.0)}
    elif tap in ("detrac_train", "detrac_test"):
        it = it_detrac(tap.split("_")[1], q, raw)
        ks = ("detrac_train60", "detrac_dev12") if tap == "detrac_train" else ("detrac_test40",)
        acc = {k: TichLuy(k, LOAI_DETRAC, 25.0) for k in ks}
    else:
        raise SystemExit(f"tap {tap!r} khong ho tro")
    mts = tc["mau_truc_quan"]
    mau = {k: MauTrucQuan(k, int(mts["seed"])) for k in mts["tap_con"] if k in acc}
    qd_rows, canh_rows, n_video = [], [], 0
    for vn in it:
        xu_ly_video(vn, acc, q, qd_rows, canh_rows, mau, tc)
        n_video += 1
        if n_video % 200 == 0:
            print(f"  {tap}: {n_video} video, {time.time() - t0:.0f} s", flush=True)
    dam_bao_dem()
    ghi_csv(DEM / "quy_dao" / f"{tap}.csv", qd_rows, COT_QD)
    ghi_csv(DEM / "canh_xe_lon" / f"{tap}.csv", canh_rows)
    for k, m in mau.items():
        ghi_json(DEM / "mau" / f"{k}.json", m.to_json())
    ghi_json(DEM / "nhan" / f"{tap}.json", {
        "tap": tap, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
        "tieu_chi_sha256": sha_tc, "so_video": n_video, "giay": round(time.time() - t0, 1),
        "phan_bo": [r for a in acc.values() for r in a.tom_tat()],
        "dieu_kien": [r for a in acc.values() for r in a.tom_tat_dk()],
        "mat_do_tong": [a.mat_do_tong_tt() for a in acc.values()],
        "nhan_goc": {k: dict(sorted(v.items())) for k, v in raw.items()},
        "chat_luong": q.rows()})
    print(f"[nhan] {tap}: {n_video} video, {len(qd_rows)} track, {time.time() - t0:.1f} s")


# =========================================================================== anh
def dhash256(im) -> str:
    from PIL import Image
    g = np.asarray(im.convert("L").resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).tobytes().hex()


def kiem_anh(p: Path, day_du: bool, giai_ma: bool) -> dict:
    from PIL import Image
    r = {"ten": p.name, "byte": "", "sha256": "", "w": "", "h": "", "mode": "", "soi": "", "eoi": "", "loi": "",
         "giai_ma": "", "dhash": "", "che_do": "day_du" if day_du else "header"}
    try:
        if day_du:
            b = p.read_bytes()
            r["byte"], r["sha256"] = len(b), hashlib.sha256(b).hexdigest()
            head, tail = b, b[-64:]
        else:
            with open(p, "rb") as f:
                head = f.read(1 << 16)
                f.seek(0, 2)
                r["byte"] = f.tell()
                f.seek(max(0, r["byte"] - 64))
                tail = f.read()
            b = None
        r["soi"] = int(head[:2] == b"\xff\xd8")
        r["eoi"] = int(tail.rstrip(b"\x00")[-2:] == b"\xff\xd9")
        with Image.open(io.BytesIO(head)) as im:
            r["w"], r["h"], r["mode"] = im.size[0], im.size[1], im.mode
        if giai_ma:
            r["giai_ma"] = "loi"
            with Image.open(io.BytesIO(b if b is not None else p.read_bytes())) as im:
                im.load()
                r["giai_ma"] = "ok"
                r["dhash"] = dhash256(im)
    except Exception as e:  # noqa: BLE001 -- ghi loi tung anh, khong dung ca lan chay
        r["loi"] = f"{type(e).__name__}: {e}"[:200]
    return r


COT_ANH = ["ten", "byte", "sha256", "w", "h", "mode", "soi", "eoi", "loi", "giai_ma", "dhash", "che_do"]


def anh_dung_huan_luyen() -> set[str]:
    man = P["yolo"] / "manifest" / "anh.csv"
    out = set()
    for r in doc_csv(man):
        a = r["anh_nguon"].replace("\\", "/")
        out.add("/".join(a.split("/")[-2:]))
    return out


def phan_anh(tap: str, han_giay: float, che_do: str):
    root = {"bdd_train": P["bdd_anh_train"], "bdd_val": P["bdd_anh_val"], "detrac": P["detrac_anh"]}[tap]
    videos = sorted(e.name for e in os.scandir(root) if e.is_dir())
    dung = anh_dung_huan_luyen() if (tap == "bdd_train" and che_do == "hon_hop") else None
    dam_bao_dem()
    od = DEM / "anh" / tap
    t0, n_file, n_byte, xong, bo_qua = time.time(), 0, 0, 0, 0
    for v in videos:
        o = od / f"{v}.csv"
        if o.exists():
            bo_qua += 1
            continue
        if time.time() - t0 > han_giay:
            break
        tat_ca = sorted(e.name for e in os.scandir(root / v) if e.is_file())
        names = [x for x in tat_ca if x.lower().endswith(".jpg")]
        n = len(names)
        mau = set()
        if n:
            rng = np.random.default_rng(zlib.crc32(v.encode()) ^ 48)
            mau = {int(0.1 * n), int(0.5 * n), min(n - 1, int(0.9 * n)), int(rng.integers(n))}
        rows = []
        for i, nm in enumerate(names):
            full = dung is None or f"{v}/{nm}" in dung or i in mau
            r = kiem_anh(root / v / nm, full, i in mau)
            rows.append(r)
            n_file += 1
            n_byte += int(r["byte"] or 0) if full else 0
        for x in tat_ca:
            if not x.lower().endswith(".jpg"):
                rows.append({"ten": x, "loi": "file_khong_phai_jpg"})
        ghi_csv(o, rows, COT_ANH)
        xong += 1
    con = sum(1 for v in videos if not (od / f"{v}.csv").exists())
    s = time.time() - t0
    print(f"[anh] {tap} ({che_do}): +{xong} video, da co {bo_qua}, con lai {con}/{len(videos)}; {n_file} anh, "
          f"{n_byte / 1e9:.2f} GB doc day du, {s:.0f} s ({n_byte / 1e6 / max(s, 1e-9):.0f} MB/s)")


# =========================================================================== trung / gan trung
def ban_do_split() -> dict:
    sp = doc_json(P["split"])
    m = {}
    for v in sp["train_fit"]:
        m[("bdd_train", v)] = "bdd_trainfit"
    for v in sp["dev"]:
        m[("bdd_train", v)] = "bdd_dev"
    if P["bdd_anh_val"].is_dir():
        for e in os.scandir(P["bdd_anh_val"]):
            if e.is_dir():
                m[("bdd_val", e.name)] = "bdd_val"
    dev = set(doc_json(P["man_dev"])["videos"])
    for p in P["xml_train"].glob("*.xml"):
        m[("detrac", p.stem)] = "detrac_dev12" if p.stem in dev else "detrac_train_khac"
    for p in P["xml_test"].glob("*.xml"):
        m[("detrac", p.stem)] = "detrac_test40"
    return m


def phan_trung():
    tc, sha_tc = nap_tieu_chi()
    T = int(tc["gan_trung"]["nguong_ung_vien"])
    sp = ban_do_split()
    sha = defaultdict(list)
    dh = {"bdd": [], "detrac": []}
    n_anh = Counter()
    for tap in ("bdd_train", "bdd_val", "detrac"):
        d = DEM / "anh" / tap
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.csv")):
            v = f.stem
            s = sp.get((tap, v), f"{tap}_ngoai_split")
            for r in doc_csv(f):
                if r.get("sha256"):
                    sha[r["sha256"]].append((s, v, r["ten"]))
                    n_anh[(s, "sha256")] += 1
                if r.get("dhash"):
                    dh["bdd" if tap.startswith("bdd") else "detrac"].append((s, v, r["ten"], r["dhash"]))
                    n_anh[(s, "dhash")] += 1
    nhom = Counter()
    vi_du = defaultdict(list)
    for h, ds in sha.items():
        if len(ds) < 2:
            continue
        vids = {(s, v) for s, v, _ in ds}
        spl = sorted({s for s, _, _ in ds})
        k = "trong_cung_video" if len(vids) == 1 else ("cung_split_khac_video" if len(spl) == 1
                                                         else "xuyen_split:" + "+".join(spl))
        nhom[k] += 1
        if len(vi_du[k]) < 5:
            vi_du[k].append([f"{s}/{v}/{t}" for s, v, t in ds[:4]])
    rows_ung, tom_gan = [], []
    for mien, L in dh.items():
        if len(L) < 2:
            continue
        H = np.stack([np.frombuffer(bytes.fromhex(x[3]), dtype=np.uint64) for x in L])
        vid = np.array([hash((x[0], x[1])) for x in L])
        n = len(L)
        gan_nhat = np.full(n, 999, dtype=np.int32)
        for i0 in range(0, n, 256):
            x = np.bitwise_count(H[i0:i0 + 256, None, :] ^ H[None, :, :]).sum(axis=2).astype(np.int32)
            x[vid[i0:i0 + 256, None] == vid[None, :]] = 999
            gan_nhat[i0:i0 + 256] = x.min(axis=1)
            ii, jj = np.nonzero(x <= T)
            for a, b in zip(ii + i0, jj):
                if b > a:
                    rows_ung.append({"mien": mien, "khoang_cach": int(x[a - i0, b]), "split_a": L[a][0],
                                     "anh_a": f"{L[a][1]}/{L[a][2]}", "split_b": L[b][0], "anh_b": f"{L[b][1]}/{L[b][2]}",
                                     "xuyen_split": int(L[a][0] != L[b][0])})
        bins = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 998)]
        for s in sorted({x[0] for x in L}):
            m = np.array([x[0] == s for x in L])
            g = gan_nhat[m]
            r = {"mien": mien, "split": s, "n_frame_mau": int(m.sum())}
            for a, b in bins:
                r[f"gan_nhat_{a}_{b}"] = int(((g >= a) & (g <= b)).sum())
            r["gan_nhat_p50"] = float(np.median(g)) if len(g) else ""
            tom_gan.append(r)
    rows_ung.sort(key=lambda r: (-r["xuyen_split"], r["khoang_cach"]))
    cap_video = Counter()
    for r in rows_ung:
        cap_video[(r["mien"], "+".join(sorted({r["split_a"], r["split_b"]})),
                   tuple(sorted((r["anh_a"].split("/")[0], r["anh_b"].split("/")[0]))))] += 1
    tom_cap = Counter()
    for (mien, spl, _vp), _n in cap_video.items():
        tom_cap[(mien, spl)] += 1
    ghi_csv(OUT / "gan_trung_ung_vien.csv", rows_ung,
            ["mien", "khoang_cach", "xuyen_split", "split_a", "anh_a", "split_b", "anh_b"])
    ghi_csv(OUT / "gan_trung_khoang_cach_gan_nhat.csv", tom_gan)
    dam_bao_dem()
    ghi_json(DEM / "trung.json", {
        "tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "tieu_chi_sha256": sha_tc, "nguong": T,
        "so_anh_theo_split": {f"{s}|{k}": n for (s, k), n in sorted(n_anh.items())},
        "trung_chinh_xac_nhom": dict(nhom), "trung_chinh_xac_vi_du": vi_du,
        "gan_trung_so_cap_frame": len(rows_ung),
        "gan_trung_so_cap_video_theo_split": {f"{m}|{s}": n for (m, s), n in sorted(tom_cap.items())}})
    print(f"[trung] sha256: {sum(n_anh[k] for k in n_anh if k[1] == 'sha256')} anh, nhom trung {dict(nhom)}; "
          f"dHash: {len(rows_ung)} cap ung vien <= {T}; cap video {dict(tom_cap)}")


# =========================================================================== YOLO
def phan_yolo():
    from PIL import Image
    tc, sha_tc = nap_tieu_chi()
    my = tc["mau_yolo"]
    tol, tol_v = float(my["dung_sai_toa_do_chuan_hoa"]), float(my["dung_sai_vung_px"])
    goc = P["yolo"]
    rows = doc_csv(goc / "manifest" / "anh.csv")
    tang = defaultdict(list)
    for r in rows:
        hiem = int(r.get("n_motorcycle") or 0) + int(r.get("n_bicycle") or 0) > 0
        tang[(r["split"], r["xu_ly"], int(hiem))].append(r)
    rng = np.random.default_rng(int(my["seed"]))
    mau = []
    for k in sorted(tang):
        L = tang[k]
        idx = rng.choice(len(L), size=min(len(L), int(my["so_moi_tang"])), replace=False)
        mau += [L[i] for i in sorted(idx)]
    cache: dict[str, dict] = {}
    out = []
    id5 = {c: i for i, c in enumerate(LOP5)}
    for r in mau:
        v = r["video"]
        a = r["anh_nguon"].replace("\\", "/")
        ten = a.split("/")[-1]
        src = P["bdd_anh_train"] / v / ten
        if v not in cache:
            if len(cache) > 50:
                cache.clear()
            cache[v] = {f["name"]: f for f in doc_json(P["bdd_nhan_train"] / f"{v}.json")}
        fr = cache[v].get(ten)
        kq = {"split": r["split"], "video": v, "anh": ten, "xu_ly": r["xu_ly"]}
        if fr is None:
            kq["loi"] = "khong tim thay frame trong nhan goc"
            out.append(kq)
            continue
        with Image.open(src) as im:
            W, H = im.size
        exp, vung = [], []
        for lb in fr.get("labels") or []:
            b = lb.get("box2d")
            if not b:
                continue
            at = lb.get("attributes") or {}
            vt = vai_tro(lb["category"], bool(at.get("crowd", False)), lb["category"] in DISTRACTOR_BDD)
            if vt == "lop_khac":
                continue
            x1, x2 = (max(0.0, min(float(W), float(b[k]))) for k in ("x1", "x2"))
            y1, y2 = (max(0.0, min(float(H), float(b[k]))) for k in ("y1", "y2"))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            if vt == "gan_nhan":
                exp.append((id5[lb["category"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            else:
                vung.append((x1, y1, x2, y2))
        p_lbl = goc / r["nhan_ra"]
        txt = p_lbl.read_bytes()
        got = [tuple([int(t.split()[0])] + [float(u) for u in t.split()[1:]]) for t in txt.decode().splitlines() if t]
        e_s, g_s = sorted(exp), sorted(got)
        kq["n_box_ky_vong"], kq["n_box_file"] = len(e_s), len(g_s)
        kq["lop_khop"] = int(Counter(x[0] for x in e_s) == Counter(x[0] for x in g_s))
        kq["toa_do_khop"] = int(len(e_s) == len(g_s) and all(a[0] == b[0] and max(abs(u - w) for u, w in zip(a[1:], b[1:]))
                                                              <= tol for a, b in zip(e_s, g_s)))
        kq["n_cot_manifest_khop"] = int(all(int(r[f"n_{c}"]) == sum(1 for x in e_s if x[0] == i)
                                            for i, c in enumerate(LOP5)))
        kq["sha_nhan_khop_manifest"] = int(hashlib.sha256(txt).hexdigest() == r["sha256_nhan"])
        if vung:
            pi = goc / r["ignore_ra"] if r.get("ignore_ra") else None
            gv = sorted(tuple(float(u) for u in t.split()) for t in pi.read_text(encoding="utf-8").splitlines() if t) \
                if pi is not None and pi.exists() else []
            ev = sorted(vung)
            kq["vung_khop"] = int(len(ev) == len(gv) and all(max(abs(u - w) for u, w in zip(a, b)) <= tol_v
                                                             for a, b in zip(ev, gv)))
        else:
            kq["vung_khop"] = int(not r.get("ignore_ra"))
        kq["xu_ly_khop"] = int((r["xu_ly"] == "to_xam") == bool(vung))
        sha_ra = sha256_file(goc / r["anh_ra"])
        kq["sha_anh_ra_khop_manifest"] = int(sha_ra == r["sha256_ra"])
        if r["xu_ly"] == "sao_chep":
            kq["sao_chep_nguyen_byte"] = int(sha256_file(src) == r["sha256_nguon"] == sha_ra)
        else:
            with Image.open(goc / r["anh_ra"]) as im2:
                kq["sao_chep_nguyen_byte"] = "khong_ap_dung"
                kq["anh_to_xam_cung_kich_thuoc"] = int(im2.size == (W, H))
        kq["wh_khop"] = int((int(r["w"]), int(r["h"])) == (W, H))
        out.append(kq)
    ghi_csv(OUT / "yolo_kiem_mau.csv", out)
    cot = ["lop_khop", "toa_do_khop", "n_cot_manifest_khop", "sha_nhan_khop_manifest", "vung_khop", "xu_ly_khop",
           "sha_anh_ra_khop_manifest", "wh_khop"]
    tom = {"n_mau": len(out), "tieu_chi_sha256": sha_tc,
           "tang": {"|".join(map(str, k)): len(v) for k, v in sorted(tang.items())},
           "loi_tim_frame": sum(1 for x in out if "loi" in x)}
    for c in cot + ["sao_chep_nguyen_byte", "anh_to_xam_cung_kich_thuoc"]:
        vals = [x.get(c) for x in out if x.get(c) not in (None, "khong_ap_dung")]
        tom[c] = {"n": len(vals), "dat": sum(1 for x in vals if x == 1)}
    ds = {"train": P["yolo"] / "images" / "train", "dev": P["yolo"] / "images" / "dev"}
    for s, d in ds.items():
        tom[f"so_file_anh_{s}"] = sum(1 for e in os.scandir(d) if e.is_file())
        tom[f"so_file_nhan_{s}"] = sum(1 for e in os.scandir(P["yolo"] / "labels" / s) if e.is_file())
        tom[f"so_video_tu_ten_file_{s}"] = len({e.name.rsplit("-", 1)[0] for e in os.scandir(d) if e.is_file()})
        tom[f"so_dong_manifest_{s}"] = sum(1 for r in rows if r["split"] == s)
    dam_bao_dem()
    ghi_json(DEM / "yolo.json", tom)
    print("[yolo]", json.dumps(tom, ensure_ascii=False))


# =========================================================================== nguon nhan
def phan_nguon():
    tsv = {}
    with open(P["nhap_r34"], encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tsv[Path(r["duong_dan_dich"]).name] = (int(r["byte"]), r["sha256"])
    files = sorted(P["bdd_nhan_train"].glob("*.json"))
    h = hashlib.sha256()
    khop = lech = thieu_tsv = 0
    vd = []
    for p in files:
        s = sha256_file(p)
        h.update(f"{p.name}\t{s}\n".encode())
        t = tsv.get(p.name)
        if t is None:
            thieu_tsv += 1
        elif t == (p.stat().st_size, s):
            khop += 1
        else:
            lech += 1
            vd.append(p.name)
    sp = doc_json(P["split"])
    tom = doc_json(P["yolo"] / "manifest" / "tom_tat.json")
    sha_val = sha256_file(P["bdd_val_coco"])
    kq = {"so_file_nhan_train": len(files), "so_dong_manifest_r34": len(tsv), "khop_byte_sha256": khop,
          "lech": lech, "lech_vi_du": vd[:5], "file_khong_co_trong_manifest": thieu_tsv,
          "file_trong_manifest_khong_co_tren_dia": len(set(tsv) - {p.name for p in files}),
          "sha256_tong_hien_tai": h.hexdigest(), "sha256_tong_split": sp["nguon_nhan"]["sha256"],
          "sha256_tong_tom_tat_xuat": tom["nguon_nhan"]["sha256"],
          "tong_khop_split": h.hexdigest() == sp["nguon_nhan"]["sha256"],
          "tong_khop_xuat": h.hexdigest() == tom["nguon_nhan"]["sha256"],
          "coco_val_sha256": sha_val, "coco_val_khop_tien_to_r34_074ff795": sha_val.startswith("074ff795"),
          "thu_muc_cach_ly_hf_ton_tai": P["cach_ly_hf"].is_dir(),
          "nguon_ghi_nhan": "HF jaffe03195/bdd100k_sot revision 694197099692fe1254e93215e73e68ece8b1f41f "
                            "(handoff/REVIEW34_CHI_TIET.md:40)"}
    dam_bao_dem()
    ghi_json(DEM / "nguon.json", kq)
    print("[nguon]", json.dumps(kq, ensure_ascii=False))


# =========================================================================== checkpoint
AUG = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "bgr",
       "mosaic", "mixup", "cutmix", "copy_paste", "copy_paste_mode", "auto_augment", "erasing", "close_mosaic",
       "imgsz", "batch", "nbs", "epochs", "patience", "rect", "multi_scale", "fraction", "cache", "seed",
       "deterministic", "optimizer", "lr0", "lrf", "warmup_epochs", "data", "model", "resume")


def _duyet(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _duyet(v)
    elif isinstance(o, list):
        for v in o:
            yield from _duyet(v)


def phan_checkpoint():
    import importlib.metadata as md
    import importlib.util
    files = sorted(p for p in P["train_log"].glob("train_bdd5_v1_*_day_du*.json"))
    theo_model = defaultdict(lambda: {"file": [], "args": defaultdict(set), "epoch_train": {}, "epoch_log": {},
                                      "mosaic": {}, "accumulate": set()})
    for f in files:
        d = doc_json(f)
        m = d.get("model") if d.get("model") in ("yolov8n", "yolo11n") else None
        m = m or ("yolo11n" if "yolo11n" in f.name else "yolov8n")
        g = theo_model[m]
        g["file"].append(f.name)
        for o in _duyet(d):
            if "hsv_h" in o and "mosaic" in o:
                for k in AUG:
                    if k in o:
                        g["args"][k].add(json.dumps(o[k]))
            if "epoch" in o and "so_batch" in o and isinstance(o["epoch"], int):
                g["epoch_train"][o["epoch"]] = o["so_batch"]
            if "epoch" in o and "ema_updates" in o and isinstance(o["epoch"], int):
                g["epoch_log"][o["epoch"]] = {k: o.get(k) for k in ("fitness", "best_fitness", "ema_updates", "dung")}
            if "mosaic_thuc_te" in o and isinstance(o.get("epoch"), int):
                g["mosaic"][o["epoch"]] = o["mosaic_thuc_te"]
            if "accumulate" in o and isinstance(o["accumulate"], (int, float)):
                g["accumulate"].add(o["accumulate"])
    rows_ck, rows_aug = [], []
    import yaml
    cfg = yaml.safe_load(P["train_cfg"].read_text(encoding="utf-8"))["huan_luyen"]
    for m, g in sorted(theo_model.items()):
        el = g["epoch_log"]
        e_max = max(el) if el else None
        best = max(el.items(), key=lambda kv: (kv[1]["fitness"] or -1))[0] if el else None
        rows_ck.append({
            "checkpoint": f"{m}_ft", "file_log": " ; ".join(g["file"]), "epoch_xong": e_max,
            "epoch_best_theo_fitness_dev": best, "fitness_best": el[best]["fitness"] if best else "",
            "dung_som_epoch_cuoi": el[e_max]["dung"] if e_max else "",
            "so_buoc_optimizer_ema_updates_cuoi": el[e_max]["ema_updates"] if e_max else "",
            "so_batch_moi_epoch": " ".join(sorted({str(v) for v in g["epoch_train"].values()})),
            "accumulate_ghi_nhan": " ".join(str(x) for x in sorted(g["accumulate"])),
            "epoch_mosaic_tat": " ".join(str(e) for e, v in sorted(g["mosaic"].items()) if v is False),
            "epoch_co_ban_ghi_mosaic": " ".join(str(e) for e in sorted(g["mosaic"])),
            "data": " | ".join(sorted(g["args"].get("data", []))),
            "imgsz": " ".join(sorted(g["args"].get("imgsz", []))), "batch": " ".join(sorted(g["args"].get("batch", []))),
            "nbs": " ".join(sorted(g["args"].get("nbs", [])))})
        for k in AUG:
            if k in ("data", "model", "resume"):
                continue
            rows_aug.append({"model": m, "tham_so": k, "cau_hinh_train_bdd5_yaml": json.dumps(cfg.get(k)) if k in cfg
                             else "(khong khai bao -> mac dinh ultralytics)",
                             "gia_tri_hieu_luc_trong_log": " | ".join(sorted(g["args"].get(k, []))) or "(khong ghi)"})
    wt = {"yolov8n_coco": (GOC / "models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
          "yolov8n_ft": (GOC / "models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt",
                         "5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a"),
          "yolo11n_ft": (GOC / "models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt",
                         "d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923")}
    sha_wt = {k: {"duong_dan": str(p.relative_to(GOC)).replace("\\", "/"), "sha256": sha256_file(p) if p.exists() else
                  "THIEU", "sha256_r47": s} for k, (p, s) in wt.items()}
    for v in sha_wt.values():
        v["khop_r47"] = v["sha256"] == v["sha256_r47"]
    try:
        ul = md.version("ultralytics")
    except md.PackageNotFoundError:
        ul = "khong cai"
    moi_truong = {"ultralytics_hien_tai": ul,
                  "albumentations_cai": importlib.util.find_spec("albumentations") is not None,
                  "python": sys.version.split()[0]}
    ghi_csv(OUT / "checkpoint_huan_luyen.csv", rows_ck)
    ghi_csv(OUT / "augmentation_hieu_luc.csv", rows_aug)
    dam_bao_dem()
    ghi_json(DEM / "checkpoint.json", {"trong_so": sha_wt, "moi_truong": moi_truong, "file_log": [f.name for f in files]})
    print("[checkpoint]", json.dumps({"trong_so": {k: v["khop_r47"] for k, v in sha_wt.items()}, **moi_truong,
                                      "checkpoint": rows_ck}, ensure_ascii=False, default=str))


# =========================================================================== truc quan
MAU_LOP = {"car": (0, 200, 0), "truck": (255, 140, 0), "bus": (230, 0, 0), "motorcycle": (220, 0, 220),
           "bicycle": (0, 210, 230), "van": (255, 220, 0), "others": (140, 90, 255)}


def _ve_khung(dr, b, mau, w=2):
    x1, y1, x2, y2 = b
    for i in range(w):
        dr.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=mau)


def phan_truc_quan():
    from PIL import Image, ImageDraw
    tc, sha_tc = nap_tieu_chi()
    lst = []
    for k in tc["mau_truc_quan"]["tap_con"]:
        p = DEM / "mau" / f"{k}.json"
        if not p.exists():
            print(f"[truc_quan] thieu {p.name} (chay --phan nhan truoc)")
            continue
        m = doc_json(p)
        tiles, so = [], 0
        for cell, o in m["o"].items():
            it = o["chon"]
            so += 1
            im = Image.open(GOC / it["anh"]).convert("RGB")
            x1, y1, x2, y2 = it["box"]
            pad = 0.5 * max(x2 - x1, y2 - y1) + 16
            cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            cx2, cy2 = min(im.size[0], int(x2 + pad) + 1), min(im.size[1], int(y2 + pad) + 1)
            cr = im.crop((cx1, cy1, cx2, cy2))
            s = min(230 / max(1, cr.size[0]), 150 / max(1, cr.size[1]))
            cr = cr.resize((max(1, int(cr.size[0] * s)), max(1, int(cr.size[1] * s))))
            t = Image.new("RGB", (240, 180), (0, 0, 0))
            t.paste(cr, (5, 25))
            dr = ImageDraw.Draw(t)
            _ve_khung(dr, ((x1 - cx1) * s + 5, (y1 - cy1) * s + 25, (x2 - cx1) * s + 5, (y2 - cy1) * s + 25),
                      MAU_LOP.get(it["lop"], (255, 255, 255)), 2)
            idx = len(lst) + 1
            dr.text((3, 2), f"#{idx} {cell}", fill=(255, 255, 255))
            dr.text((3, 12), f"{it['video'][:17]} f{it['frame']}", fill=(180, 180, 180))
            tiles.append(t)
            lst.append({"so": idx, "tap_con": k, "o": cell, "n_trong_o": o["n_trong_o"], **{x: it.get(x) for x in
                        ("video", "anh", "frame", "lop", "kt", "dk", "track", "occ", "trunc", "occ_r", "trunc_r",
                         "thoi_tiet")}, "box": " ".join(f"{u:.1f}" for u in it["box"])})
        cols = 6
        rows_n = max(1, math.ceil(len(tiles) / cols))
        mt = Image.new("RGB", (cols * 240, rows_n * 180), (40, 40, 40))
        for i, t in enumerate(tiles):
            mt.paste(t, ((i % cols) * 240, (i // cols) * 180))
        mt.save(_duong_ghi(OVERLAY / f"mau_box_{k}.jpg"), quality=90)
        fr_tiles = []
        for it in m["frame"]:
            im = Image.open(GOC / it["anh"]).convert("RGB")
            dr = ImageDraw.Draw(im)
            for g in it.get("vung_bo_qua") or []:
                _ve_khung(dr, g, (255, 255, 0), 1)
            for role, cat, x1, y1, x2, y2 in it["boxes"]:
                c = MAU_LOP.get(cat, (255, 255, 255)) if role == "gan_nhan" else (
                    (255, 255, 255) if role == "to_xam" else (128, 128, 128))
                _ve_khung(dr, (x1, y1, x2, y2), c, 2 if role == "gan_nhan" else 1)
            s = 640 / im.size[0]
            im = im.resize((640, int(im.size[1] * s)))
            ImageDraw.Draw(im).text((4, 4), f"{k} {it['video']} f{it['frame']}", fill=(255, 255, 0))
            fr_tiles.append(im)
            lst.append({"so": f"F{len(fr_tiles)}", "tap_con": k, "o": "toan_khung", "video": it["video"],
                        "anh": it["anh"], "frame": it["frame"], "n_box": len(it["boxes"])})
        if fr_tiles:
            hh = max(t.size[1] for t in fr_tiles)
            mt = Image.new("RGB", (640 * len(fr_tiles), hh), (40, 40, 40))
            for i, t in enumerate(fr_tiles):
                mt.paste(t, (640 * i, 0))
            mt.save(_duong_ghi(OVERLAY / f"mau_khung_{k}.jpg"), quality=90)
        print(f"[truc_quan] {k}: {so} o co mau / {len(m['o'])} o co box; {len(fr_tiles)} frame")
    ung = OUT / "gan_trung_ung_vien.csv"
    if ung.exists():
        rows = doc_csv(ung)[:8]
        tiles = []
        for r in rows:
            pair = []
            for sk, ak in (("split_a", "anh_a"), ("split_b", "anh_b")):
                s = r[sk]
                base = P["detrac_anh"] if s.startswith("detrac") else (
                    P["bdd_anh_val"] if s == "bdd_val" else P["bdd_anh_train"])
                im = Image.open(base / r[ak]).convert("RGB")
                im = im.resize((320, int(im.size[1] * 320 / im.size[0])))
                ImageDraw.Draw(im).text((4, 4), f"{s} {r[ak][:30]} d={r['khoang_cach']}", fill=(255, 255, 0))
                pair.append(im)
            t = Image.new("RGB", (640, max(x.size[1] for x in pair)), (0, 0, 0))
            t.paste(pair[0], (0, 0))
            t.paste(pair[1], (320, 0))
            tiles.append(t)
        if tiles:
            mt = Image.new("RGB", (1280, sum(t.size[1] for t in tiles[::2]) + 10), (40, 40, 40))
            y = 0
            for i in range(0, len(tiles), 2):
                mt.paste(tiles[i], (0, y))
                if i + 1 < len(tiles):
                    mt.paste(tiles[i + 1], (640, y))
                y += tiles[i].size[1]
            mt.save(_duong_ghi(OVERLAY / "gan_trung_cap_ung_vien.jpg"), quality=90)
    ghi_csv(OUT / "mau_truc_quan.csv", lst)
    print(f"[truc_quan] {len(lst)} muc -> mau_truc_quan.csv; anh trong overlay/")


# =========================================================================== tong hop
TAP_CON = ["bdd_trainfit_all", "bdd_trainfit_b5", "bdd_dev_all", "bdd_dev_b5", "bdd_val", "detrac_train60",
           "detrac_dev12", "detrac_test40"]


def phan_tong_hop():
    nh = {}
    for tap in ("bdd_train", "bdd_val", "detrac_train", "detrac_test"):
        p = DEM / "nhan" / f"{tap}.json"
        if p.exists():
            nh[tap] = doc_json(p)
    pb = {(r["tap_con"], r["lop"]): r for d in nh.values() for r in d["phan_bo"]}
    # 1. phan bo 5 lop (DETRAC: car / bus theo loai goc; truck / motorcycle / bicycle khong co trong nhan)
    rows = []
    for s in TAP_CON:
        for c in LOP5:
            r = pb.get((s, c))
            if r is None and s.startswith("detrac"):
                r = {"tap_con": s, "lop": c, "n_bbox": "khong_co_lop_trong_nhan_DETRAC", "n_track": "n/a",
                     "n_video_co_lop": "n/a"}
            if r is not None:
                rows.append(r)
    ghi_csv(OUT / "phan_bo_lop.csv", rows)
    ghi_csv(OUT / "phan_bo_detrac_loai_goc.csv",
            [r for (s, c), r in pb.items() if s.startswith("detrac")] +
            [{"tap_con": "anh_xa_5_lop", "lop": k, "n_bbox": v} for k, v in DETRAC_SANG_5LOP.items()])
    ghi_csv(OUT / "dieu_kien_canh.csv", [r for d in nh.values() for r in d["dieu_kien"]] +
            [{"tap_con": s, "khoa": "dieu_kien_canh", "gia_tri": "unknown (nhan BDD MOT khong co thuoc tinh canh)"}
             for s in TAP_CON if s.startswith("bdd")])
    ghi_csv(OUT / "mat_do_frame.csv", [r for d in nh.values() for r in d["mat_do_tong"] if r])
    ghi_csv(OUT / "nhan_goc_theo_loai.csv", [{"tap": t, "loai": k, "n": n} for d in nh.values()
                                            for t, kv in d["nhan_goc"].items() for k, n in kv.items()])
    # 2. chat luong
    cl = [r for d in nh.values() for r in d["chat_luong"]]
    for tap in ("bdd_train", "bdd_val", "detrac"):
        d = DEM / "anh" / tap
        if not d.is_dir():
            continue
        sp = ban_do_split()
        agg = defaultdict(Counter)
        vd = defaultdict(list)
        kt = defaultdict(Counter)
        for f in sorted(d.glob("*.csv")):
            s = sp.get((tap, f.stem), f"{tap}_ngoai_split")
            for r in doc_csv(f):
                a = agg[s]
                if r.get("loi") == "file_khong_phai_jpg":
                    a["file_khong_phai_jpg"] += 1
                    continue
                a["anh"] += 1
                a["anh_doc_day_du_sha256"] += r["che_do"] == "day_du"
                a["anh_chi_header"] += r["che_do"] == "header"
                for k, bad in (("loi_doc_hoac_header", bool(r["loi"])), ("thieu_SOI", r["soi"] == "0"),
                               ("thieu_EOI(cat_cut?)", r["eoi"] == "0")):
                    if bad:
                        a[k] += 1
                        if len(vd[(s, k)]) < 3:
                            vd[(s, k)].append(f"{f.stem}/{r['ten']}")
                if r["giai_ma"]:
                    a["giai_ma_day_du_mau"] += 1
                    if r["giai_ma"] != "ok":
                        a["giai_ma_loi"] += 1
                        if len(vd[(s, "giai_ma_loi")]) < 3:
                            vd[(s, "giai_ma_loi")].append(f"{f.stem}/{r['ten']}")
                kt[s][f"{r['w']}x{r['h']}"] += 1
        for s, a in sorted(agg.items()):
            n = a["anh"]
            for k in ("loi_doc_hoac_header", "thieu_SOI", "thieu_EOI(cat_cut?)"):
                cl.append({"tap": s, "kiem": f"anh:{k}", "n_kiem": n, "n_loi": a[k],
                           "ti_le_loi": round(a[k] / n, 6) if n else "", "vi_du": " ; ".join(vd[(s, k)])})
            cl.append({"tap": s, "kiem": "anh:giai_ma_day_du_tren_mau", "n_kiem": a["giai_ma_day_du_mau"],
                       "n_loi": a["giai_ma_loi"], "ti_le_loi": round(a["giai_ma_loi"] / a["giai_ma_day_du_mau"], 6)
                       if a["giai_ma_day_du_mau"] else "", "vi_du": " ; ".join(vd[(s, "giai_ma_loi")])})
            cl.append({"tap": s, "kiem": "anh:file_khong_phai_jpg_trong_thu_muc", "n_kiem": n + a["file_khong_phai_jpg"],
                       "n_loi": a["file_khong_phai_jpg"]})
            cl.append({"tap": s, "kiem": "anh:pham_vi_doc (day_du_sha256 / chi_header)", "n_kiem": n,
                       "n_loi": "", "vi_du": f"day_du={a['anh_doc_day_du_sha256']} header={a['anh_chi_header']}"})
            cl.append({"tap": s, "kiem": "anh:kich_thuoc_thuc (phan bo)", "n_kiem": n, "n_loi": "",
                       "vi_du": " ; ".join(f"{k}:{v}" for k, v in kt[s].most_common(4))})
    for name in ("trung", "yolo", "nguon"):
        p = DEM / f"{name}.json"
        if not p.exists():
            cl.append({"tap": "-", "kiem": f"{name}: CHUA CHAY"})
            continue
        d = doc_json(p)
        if name == "trung":
            for k, n in d["trung_chinh_xac_nhom"].items():
                cl.append({"tap": "sha256_toan_bo_anh_da_bam", "kiem": f"trung_chinh_xac:{k}", "n_loi": n,
                           "vi_du": json.dumps(d["trung_chinh_xac_vi_du"].get(k, [])[:5], ensure_ascii=False)})
            if not d["trung_chinh_xac_nhom"]:
                cl.append({"tap": "sha256_toan_bo_anh_da_bam", "kiem": "trung_chinh_xac (nhom sha256 > 1 anh)",
                           "n_kiem": sum(v for k, v in d["so_anh_theo_split"].items() if k.endswith("sha256")),
                           "n_loi": 0})
            for k, n in d["gan_trung_so_cap_video_theo_split"].items():
                cl.append({"tap": "dhash_mau_4_frame_moi_video", "kiem": f"gan_trung_ung_vien(cap video):{k}",
                           "n_loi": n, "vi_du": f"nguong <= {d['nguong']} bit / 256"})
        elif name == "yolo":
            for k, v in d.items():
                if isinstance(v, dict) and "n" in v:
                    cl.append({"tap": "yolo_bdd5_v1_mau", "kiem": f"yolo:{k}", "n_kiem": v["n"],
                               "n_loi": v["n"] - v["dat"]})
        else:
            cl.append({"tap": "bdd_train_nhan", "kiem": "nguon:khop_byte_sha256_manifest_r34",
                       "n_kiem": d["so_file_nhan_train"], "n_loi": d["so_file_nhan_train"] - d["khop_byte_sha256"]})
            cl.append({"tap": "bdd_train_nhan", "kiem": "nguon:hash_tong_khop_split_va_xuat", "n_kiem": 2,
                       "n_loi": 2 - int(d["tong_khop_split"]) - int(d["tong_khop_xuat"])})
    ghi_csv(OUT / "chat_luong.csv", cl, ["tap", "kiem", "n_kiem", "n_loi", "ti_le_loi", "vi_du"])
    # 3. quy dao
    qrows = []
    for tap in ("bdd_train", "bdd_val", "detrac_train", "detrac_test"):
        p = DEM / "quy_dao" / f"{tap}.csv"
        if p.exists():
            qrows += doc_csv(p)
    out = tong_hop_quy_dao(qrows)
    ghi_csv(OUT / "quy_dao_cua.csv", out, COT_QD_TH)
    # 4. canh xe lon
    crows = []
    for tap in ("bdd_train", "bdd_val", "detrac_train", "detrac_test"):
        p = DEM / "canh_xe_lon" / f"{tap}.csv"
        if p.exists():
            crows += doc_csv(p)
    ag = defaultdict(Counter)
    for r in crows:
        for s in r["tap_con"].split("|"):
            a = ag[s]
            a["n_video"] += 1
            for k in ("n_cap_frame", "n_su_kien", "n_track_nho_lien_quan"):
                a[k] += int(r[k])
            a["n_video_co_su_kien"] += int(r["n_su_kien"]) > 0
            if r["n_su_kien_theo_nhan_che"] not in ("unknown", ""):
                for k in ("n_cap_frame_theo_nhan_che", "n_su_kien_theo_nhan_che", "n_track_nho_theo_nhan_che"):
                    a[k] += int(r[k])
                a["n_video_co_su_kien_theo_nhan_che"] += int(r["n_su_kien_theo_nhan_che"]) > 0
    ghi_csv(OUT / "canh_xe_lon_tom_tat.csv", [{"tap_con": s, **dict(a)} for s, a in sorted(ag.items())])
    ghi_csv(OUT / "canh_xe_lon_video_detrac.csv", [r for r in crows if "detrac" in r["tap_con"]])
    # 5. moc doi chieu slide lan 2 + video tung tap
    sp = doc_json(P["split"])
    tom = doc_json(P["yolo"] / "manifest" / "tom_tat.json")
    yj = doc_json(DEM / "yolo.json") if (DEM / "yolo.json").exists() else {}
    fr = {(r["tap_con"]): r["n_frame_tap"] for r in pb.values()}
    vv = {(r["tap_con"]): r["n_video_tap"] for r in pb.values()}
    val_dirs = [e for e in os.scandir(P["bdd_anh_val"]) if e.is_dir()]
    n_val_files = sum(len(_list_jpg(Path(e.path)) or ()) for e in val_dirs)
    moc = [
        ("BDD train (train-fit) so video", 1261, [("split.train_fit", len(sp["train_fit"])),
                                                   ("dem lai tu nhan (bdd_trainfit_all)", vv.get("bdd_trainfit_all")),
                                                   ("ten file anh xuat images/train", yj.get("so_video_tu_ten_file_train"))]),
        ("BDD train (train-fit) so anh", 50487, [("quy tac buoc 5 tai suy (bdd_trainfit_b5)", fr.get("bdd_trainfit_b5")),
                                                  ("manifest/anh.csv split=train", yj.get("so_dong_manifest_train")),
                                                  ("file trong data/yolo_bdd5_v1/images/train", yj.get("so_file_anh_train"))]),
        ("BDD dev noi bo so video", 139, [("split.dev", len(sp["dev"])), ("dem lai tu nhan (bdd_dev_all)",
                                                                           vv.get("bdd_dev_all")),
                                           ("ten file anh xuat images/dev", yj.get("so_video_tu_ten_file_dev"))]),
        ("BDD dev noi bo so anh", 5559, [("quy tac buoc 5 tai suy (bdd_dev_b5)", fr.get("bdd_dev_b5")),
                                          ("manifest/anh.csv split=dev", yj.get("so_dong_manifest_dev")),
                                          ("file trong data/yolo_bdd5_v1/images/dev", yj.get("so_file_anh_dev"))]),
        ("BDD val so video", 200, [("COCO-video videos (bdd_val)", vv.get("bdd_val")),
                                    ("thu muc images/track/val", len(val_dirs))]),
        ("BDD val so anh", 39973, [("COCO-video images (bdd_val)", fr.get("bdd_val")),
                                    ("file jpg trong images/track/val", n_val_files)]),
    ]
    mrows = []
    for ten, slide, ds in moc:
        for nguon, dem in ds:
            mrows.append({"moc": ten, "gia_tri_slide_lan_2": slide, "nguon_dem": nguon, "dem_thuc": dem,
                          "ket_luan": "" if dem is None else ("khop" if int(dem) == slide else f"lech {int(dem) - slide:+d}")})
    for s in TAP_CON:
        if s in vv:
            mrows.append({"moc": f"so video / frame co nhan: {s}", "gia_tri_slide_lan_2": "", "nguon_dem": "dem tu nhan",
                          "dem_thuc": f"{vv[s]} video / {fr[s]} frame", "ket_luan": "vai chuc video" if int(vv[s]) < 100
                          else ""})
    mrows.append({"moc": "BDD train tong frame nhan (1.400 video)", "nguon_dem": "tom_tat xuat dem_frame.frame_nguon",
                  "dem_thuc": tom["dem_frame"]["frame_nguon"]})
    ghi_csv(OUT / "moc_doi_chieu.csv", mrows)
    # 5b. tien to ten video BDD (doan truoc '-'): cac video chung tien to co the cung mot lan ghi -> chia theo video
    #     khong tach duoc boi canh; chi la DAU HIEU, y nghia cua tien to chua xac minh tu tai lieu BDD100K.
    vset = {"bdd_trainfit": set(sp["train_fit"]), "bdd_dev": set(sp["dev"]), "bdd_val": {e.name for e in val_dirs}}
    tt = {k: defaultdict(set) for k in vset}
    for k, vs in vset.items():
        for v in vs:
            tt[k][v.split("-", 1)[0]].add(v)
    trows = [{"tap": k, "n_video": len(vset[k]), "n_tien_to": len(tt[k]),
              "n_tien_to_co_>=2_video": sum(1 for x in tt[k].values() if len(x) > 1)} for k in vset]
    for a, b in (("bdd_trainfit", "bdd_dev"), ("bdd_trainfit", "bdd_val"), ("bdd_dev", "bdd_val")):
        chung = set(tt[a]) & set(tt[b])
        trows.append({"tap": f"{a} & {b}", "n_tien_to_chung": len(chung),
                      f"n_video_{a}_lien_quan": sum(len(tt[a][x]) for x in chung),
                      f"n_video_{b}_lien_quan": sum(len(tt[b][x]) for x in chung),
                      "vi_du": " ; ".join(f"{x}: {sorted(tt[a][x])[0]} | {sorted(tt[b][x])[0]}" for x in sorted(chung)[:3])})
    ghi_csv(OUT / "tien_to_video_bdd.csv", trows)
    # 6. bieu do
    try:
        ve_bieu_do(rows, out)
    except Exception as e:  # noqa: BLE001
        print(f"[tong_hop] bieu do loi: {type(e).__name__}: {e}")
    print(f"[tong_hop] phan_bo_lop {len(rows)} dong, chat_luong {len(cl)} dong, quy_dao_cua {len(out)} dong, "
          f"moc {len(mrows)} dong")


def ve_bieu_do(rows_pb, rows_qd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    taps = ["bdd_trainfit_b5", "bdd_dev_b5", "bdd_val", "detrac_dev12", "detrac_test40"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.16
    for i, s in enumerate(taps):
        vals = []
        for c in LOP5:
            r = next((x for x in rows_pb if x["tap_con"] == s and x["lop"] == c), None)
            v = r["n_bbox"] if r else 0
            vals.append(v if isinstance(v, (int, float)) and v > 0 else np.nan)
        ax[0].bar(np.arange(5) + (i - 2) * w, vals, w, label=s)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(5), LOP5)
    ax[0].set_ylabel("so bbox (log)")
    ax[0].set_title("Phan bo lop (DETRAC: chi car/bus co nhan)")
    ax[0].legend(fontsize=7)
    for i, s in enumerate(taps):
        for j, c in enumerate(LOP5):
            r = next((x for x in rows_pb if x["tap_con"] == s and x["lop"] == c), None)
            if not r or r.get("kt_rel_p50") in ("", None):
                continue
            x0 = j + (i - 2) * w
            ax[1].plot([x0, x0], [r["kt_rel_p25"], r["kt_rel_p75"]], color=f"C{i}", lw=3)
            ax[1].plot([x0], [r["kt_rel_p50"]], "o", color=f"C{i}", ms=4, label=s if j == 0 else None)
    ax[1].set_xticks(range(5), LOP5)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("sqrt(dien tich box / dien tich anh)")
    ax[1].set_title("Kich thuoc tuong doi: p25-p75 (vach), p50 (cham)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_duong_ghi(OUT / "bieu_do_phan_bo_lop_kich_thuoc.png"), dpi=110)
    plt.close(fig)
    taps2 = ["bdd_trainfit", "bdd_dev", "bdd_val", "detrac_train60", "detrac_dev12", "detrac_test40"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, chi_so, ten in ((axs[0], "loai_max", "chinh: max |dpsi|"), (axs[1], "loai_p90", "do nhay: p90 |dpsi|")):
        rs = [next((x for x in rows_qd if x["tap_con"] == s and x["lop"] == "_tat_ca" and x["chi_so"] == chi_so), None)
              for s in taps2]
        bottom = np.zeros(len(taps2))
        for k in NHOM_CD:   # tap hop le da loai nghi_nhieu (theo max) -> 4 nhom cong lai = 1
            v = np.array([r[k] / r["n_hop_le"] if r and r["n_hop_le"] else 0 for r in rs])
            ax.bar(taps2, v, bottom=bottom, label=k)
            bottom += v
        for i, r in enumerate(rs):
            if r:
                ax.text(i, 1.01, f"hop le {r['n_hop_le']}\nloai nghi_nhieu {r['nghi_nhieu']}\n"
                                 f"({100 * r['nghi_nhieu'] / r['n_truoc_loc_nhieu']:.1f} %)" if r["n_truoc_loc_nhieu"]
                        else "", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, 1.3)
        ax.set_title(f"{ten} / 1 s, sau loai nghi_nhieu (max > 150 do)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    axs[0].set_ylabel("ti le tren track hop le (sau loc)")
    axs[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Doi huong tren anh theo tieu chi cong bo truoc (BDD: ego-motion, khong phai xe cua)", fontsize=10)
    fig.tight_layout()
    fig.savefig(_duong_ghi(OUT / "bieu_do_quy_dao_cua.png"), dpi=110)
    plt.close(fig)


# =========================================================================== main
def phan_cong_bo():
    if FILE_TIEU_CHI.exists():
        cu = doc_json(FILE_TIEU_CHI)
        cu.pop("cong_bo_luc", None)
        cu.pop("git_head", None)
        if cu != json.loads(json.dumps(TIEU_CHI)):
            raise SystemExit("tieu_chi_cong_bo.json da co va KHAC hang so trong ma: khong ghi de (doi phien_ban neu can)")
        print(f"[cong_bo] da co, khop ma; sha256 {sha256_file(FILE_TIEU_CHI)}")
        return
    ghi_json(FILE_TIEU_CHI, {**TIEU_CHI, "cong_bo_luc": dt.datetime.now().isoformat(timespec="seconds"),
                             "git_head": git_head()})
    print(f"[cong_bo] ghi {FILE_TIEU_CHI.relative_to(GOC)} sha256 {sha256_file(FILE_TIEU_CHI)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--phan", required=True, choices=["cong_bo", "nhan", "anh", "trung", "yolo", "nguon", "checkpoint",
                                                      "truc_quan", "tong_hop"])
    ap.add_argument("--tap", default=None)
    ap.add_argument("--gioi_han_giay", type=float, default=480.0)
    ap.add_argument("--che_do", choices=["day_du", "hon_hop"], default="day_du",
                    help="anh: day_du = doc toan bo byte moi anh (sha256); hon_hop (chi bdd_train) = day_du cho anh "
                         "dung huan luyen/dev + mau giai ma, chi header + EOI cho anh con lai")
    a = ap.parse_args(argv)
    t0 = time.time()
    bat_dau = dt.datetime.now().isoformat(timespec="seconds")
    if a.phan == "cong_bo":
        phan_cong_bo()
    elif a.phan == "nhan":
        phan_nhan(a.tap)
    elif a.phan == "anh":
        phan_anh(a.tap, a.gioi_han_giay, a.che_do)
    elif a.phan == "trung":
        phan_trung()
    elif a.phan == "yolo":
        phan_yolo()
    elif a.phan == "nguon":
        phan_nguon()
    elif a.phan == "checkpoint":
        phan_checkpoint()
    elif a.phan == "truc_quan":
        phan_truc_quan()
    elif a.phan == "tong_hop":
        phan_tong_hop()
    p = _duong_ghi(OUT / "nhat_ky_chay.jsonl")
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"phan": a.phan, "tap": a.tap, "che_do": a.che_do if a.phan == "anh" else None,
                            "bat_dau": bat_dau, "giay": round(time.time() - t0, 1), "git_head": git_head(),
                            "argv": sys.argv[1:]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
