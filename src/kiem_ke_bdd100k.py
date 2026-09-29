"""
kiem_ke_bdd100k.py -- Kiem ke THUC TE du lieu BDD100K tren may (archive, thu muc da giai nen, nguon nhan) truoc khi
chuan bi dataset huan luyen detector (vong 27). Chi DOC, khong sua / giai nen / xoa gi.

Phan biet hai ho du lieu KHAC NGUON (khong thay the truc tiep cho nhau):
  - MOT 2020 (tracking): anh `images/track/{train,val,test}/<video>/...`, nhan `labels/box_track_20/<split>/<video>.json`
    (Scalabel) hoac ban chuyen doi COCO-video `*_cocofmt.json`. Nhan 5 fps, theo video, co ID doi tuong.
  - Detection 2020 (100K): anh tinh `images/100k/{train,val,test}/<ten>.jpg`, nhan `labels/det_20/det_{train,val}.json`.
    Moi anh la mot khung hinh tu mot video 40 s; khong co ID theo thoi gian.

    python src/kiem_ke_bdd100k.py --out handoff/logs/review27/01_kiem_ke_bdd100k.json [--bam-archive]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from detector_classes import LOP_5  # noqa: E402

MAU = [(re.compile(r"(^|/)images/track/(train|val|test)/"), "mot_anh"),
       (re.compile(r"(^|/)labels/box_track_20/(train|val)/"), "mot_nhan_scalabel"),
       (re.compile(r"(^|/)images/100k/(train|val|test)/"), "det_anh_100k"),
       (re.compile(r"(^|/)images/10k/(train|val|test)/"), "anh_10k"),
       (re.compile(r"(^|/)labels/det_20/"), "det_nhan"),
       (re.compile(r"cocofmt\.json$"), "mot_nhan_cocovid")]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def phan_loai(ten: str) -> tuple[str, str | None]:
    for rx, loai in MAU:
        m = rx.search(ten)
        if m:
            return loai, (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
    return "khac", None


def kiem_zip(p: Path, bam: bool) -> dict:
    out = {"duong_dan": str(p), "bytes": p.stat().st_size}
    if bam:
        out["sha256"] = sha256_file(p)
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
    out["so_muc"] = len(names)
    dem = collections.Counter()
    video = collections.defaultdict(set)
    for n in names:
        if n.endswith("/"):
            continue
        loai, split = phan_loai(n)
        dem[f"{loai}:{split}"] += 1
        if loai == "mot_anh":
            video[split].add(n.split("/images/track/")[-1].split("/")[1] if "/images/track/" in n
                             else n.split("images/track/")[-1].split("/")[1])
    out["noi_dung"] = dict(dem)
    out["so_video_mot"] = {k: len(v) for k, v in video.items()}
    out["tien_to"] = sorted({n.split("/")[0] for n in names})[:5]
    return out


def kiem_json(p: Path, bam: bool) -> dict:
    out = {"duong_dan": str(p), "bytes": p.stat().st_size}
    if bam:
        out["sha256"] = sha256_file(p)
    with open(p, "rb") as f:
        dau = f.read(4096).decode("utf-8", "replace")
    out["dinh_dang_doan"] = ("cocovid" if '"videos"' in dau or ('"categories"' in dau and '"images"' in dau)
                             else "scalabel/det" if dau.lstrip().startswith("[") else "khong ro")
    return out


def thong_ke_cocovid(p: Path, anh_dir: Path | None, kiem_kich_thuoc: bool) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    vid = {v["id"]: v["name"] for v in d["videos"]}
    cat = {c["id"]: c["name"] for c in d["categories"]}
    img = {i["id"]: i for i in d["images"]}
    theo_lop = {c: {"box": 0, "crowd": 0, "ignore": 0, "anh": set(), "video": set()} for c in cat.values()}
    anh_co_nhan = set()
    for a in d["annotations"]:
        c = cat[a["category_id"]]
        t = theo_lop[c]
        im = img[a["image_id"]]
        anh_co_nhan.add(a["image_id"])
        if a.get("iscrowd"):
            t["crowd"] += 1
        if a.get("ignore"):
            t["ignore"] += 1
        if not a.get("iscrowd") and not a.get("ignore"):
            t["box"] += 1
            t["anh"].add(a["image_id"])
            t["video"].add(im["video_id"])
    frame_video = collections.defaultdict(list)
    for i in d["images"]:
        frame_video[i["video_id"]].append((i["frame_id"], i["id"]))
    ngoai_khoang = khong_nhan_trong_khoang = 0
    for v, fr in frame_video.items():
        co = [f for f, iid in fr if iid in anh_co_nhan]
        lo, hi = (min(co), max(co)) if co else (1, 0)
        for f, iid in fr:
            if iid not in anh_co_nhan:
                if lo <= f <= hi:
                    khong_nhan_trong_khoang += 1
                else:
                    ngoai_khoang += 1
    out = {"duong_dan": str(p), "so_video": len(vid), "so_anh": len(img), "so_annotation": len(d["annotations"]),
           "frame_moi_video": {"min": min(len(v) for v in frame_video.values()),
                               "max": max(len(v) for v in frame_video.values()),
                               "trung_binh": round(len(img) / max(1, len(vid)), 1)},
           "anh_khong_annotation": {"trong_khoang_co_nhan": khong_nhan_trong_khoang, "ngoai_khoang": ngoai_khoang},
           "kich_thuoc_khai_bao": dict(collections.Counter(f"{i.get('width')}x{i.get('height')}" for i in d["images"])),
           "theo_lop": {c: {"box": t["box"], "crowd": t["crowd"], "ignore": t["ignore"], "anh_co_lop": len(t["anh"]),
                            "video_co_lop": len(t["video"]), "thuoc_5_lop": c in LOP_5} for c, t in theo_lop.items()}}
    if anh_dir is not None and anh_dir.is_dir():
        tren_dia = {pp.name for pp in anh_dir.iterdir() if pp.is_dir()}
        thieu_video = sorted(set(vid.values()) - tren_dia)
        thua_video = sorted(tren_dia - set(vid.values()))
        thieu_anh = [i["file_name"] for i in d["images"] if not (anh_dir / i["file_name"]).is_file()]
        so_file = sum(1 for v in tren_dia for _ in (anh_dir / v).glob("*.jpg"))
        out["doi_chieu_anh"] = {"thu_muc": str(anh_dir), "so_video_tren_dia": len(tren_dia),
                                "so_anh_tren_dia": so_file, "video_trong_nhan_khong_co_anh": thieu_video[:10],
                                "so_video_thieu": len(thieu_video), "video_tren_dia_khong_co_nhan": thua_video[:10],
                                "so_anh_thieu": len(thieu_anh), "vi_du_anh_thieu": thieu_anh[:5]}
        if kiem_kich_thuoc:
            from PIL import Image
            kt = collections.Counter()
            lech = 0
            for i in d["images"]:
                f = anh_dir / i["file_name"]
                if f.is_file():
                    with Image.open(f) as im:
                        s = im.size
                    kt[f"{s[0]}x{s[1]}"] += 1
                    lech += int(s != (i.get("width"), i.get("height")))
            out["doi_chieu_anh"]["kich_thuoc_thuc_header"] = dict(kt)
            out["doi_chieu_anh"]["so_anh_lech_kich_thuoc_khai_bao"] = lech
    return out


def cay_bdd(goc: Path) -> dict:
    out = {"goc": str(goc), "ton_tai": goc.is_dir()}
    if not goc.is_dir():
        return out
    for rel in ("images/track/train", "images/track/val", "images/track/test", "images/100k/train", "images/100k/val",
                "images/100k/test", "labels/box_track_20/train", "labels/box_track_20/val", "labels/det_20",
                "labels_cocovid"):
        p = goc / rel
        if not p.is_dir():
            out[rel] = None
            continue
        if rel.startswith("images/track"):
            vids = [x for x in p.iterdir() if x.is_dir()]
            out[rel] = {"so_video": len(vids), "so_anh": sum(1 for v in vids for _ in v.glob("*.jpg"))}
        else:
            out[rel] = {"so_file": sum(1 for _ in p.iterdir()), "vi_du": sorted(x.name for x in p.iterdir())[:3]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--noi-tim", nargs="*", default=[str(config.RAW_DIR), str(config.PROJECT_ROOT.parent)],
                    help="thu muc tim archive/JSON BDD (khong de quy qua 2 cap)")
    ap.add_argument("--bam-archive", action="store_true", help="tinh sha256 archive (5 GB ~ 1 phut)")
    ap.add_argument("--khong-kiem-kich-thuoc", action="store_true")
    a = ap.parse_args()
    goc = config.RAW_DIR / "bdd100k"
    ket = {"tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "archive": [], "json_roi": []}
    thay = set()
    for d in a.noi_tim:
        d = Path(d)
        for p in list(d.glob("*")) + list(d.glob("*/*")):
            if p.is_file() and (re.search(r"bdd|images20|track|det_20|100k", p.name, re.I)) and p not in thay:
                thay.add(p)
                if p.suffix.lower() == ".zip":
                    ket["archive"].append(kiem_zip(p, a.bam_archive))
                elif p.suffix.lower() == ".json":
                    ket["json_roi"].append(kiem_json(p, a.bam_archive))
    ket["thu_muc_giai_nen"] = cay_bdd(goc)
    ket["nguon_nhan_cocovid"] = []
    for p in sorted((goc / "labels_cocovid").glob("*.json")) if (goc / "labels_cocovid").is_dir() else []:
        split = "val" if "val" in p.name else "train" if "train" in p.name else None
        ket["nguon_nhan_cocovid"].append(
            {"split_theo_ten": split, **thong_ke_cocovid(p, goc / "images" / "track" / split if split else None,
                                                         not a.khong_kiem_kich_thuoc)})
    t = ket["thu_muc_giai_nen"]
    ket["ket_luan"] = {
        "mot_train_anh": bool(t.get("images/track/train")),
        "mot_train_nhan_scalabel": bool(t.get("labels/box_track_20/train")),
        "mot_train_nhan_cocovid": any(x["split_theo_ten"] == "train" for x in ket["nguon_nhan_cocovid"]),
        "mot_val_anh": bool(t.get("images/track/val")),
        "mot_val_nhan": bool(t.get("labels/box_track_20/val")) or any(x["split_theo_ten"] == "val"
                                                                       for x in ket["nguon_nhan_cocovid"]),
        "det_100k_anh": any(t.get(k) for k in ("images/100k/train", "images/100k/val")),
        "det_20_nhan": bool(t.get("labels/det_20")),
    }
    ket["ket_luan"]["du_de_chuan_bi_train"] = ket["ket_luan"]["mot_train_anh"] and (
        ket["ket_luan"]["mot_train_nhan_scalabel"] or ket["ket_luan"]["mot_train_nhan_cocovid"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(ket, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(ket, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
