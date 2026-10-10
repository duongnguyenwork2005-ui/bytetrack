"""
fingerprint_detrac_test.py -- Dau van tay du lieu UA-DETRAC test truoc khi danh gia detector (ke hoach R47, moc A).

Vi sao co file nay: du thao giao thuc vong 45 (`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`, muc 7) yeu cau,
TRUOC khi viet adapter (moc B) va khoa ke hoach, ghim NOI DUNG tap test: sha256 noi dung MOI anh (bam lai toan bo, KHONG
dung loi tat mtime), w/h MOI anh doc tu header (vong 45 moi doc anh dau / cuoi moi video), danh sach anh `video/frame` co
thu tu, danh sach anh bi loai theo D4, sha256 manifest va tung XML.

CHI DOC du lieu: anh, XML, manifest mo che do 'rb'; khong sua / xoa / tao file trong data/extracted, data/raw, results
(`--out` nam trong cac cay do -> tu choi). Chi thu vien chuan + PIL (doc header, cung thu vien voi
`eval_detector_5lop.kich_thuoc_anh`); khong pandas / pyarrow. Moi anh doc MOT lan: size, sha256, w/h cung tu mot chuoi byte.

Dau ra (`--out`, mac dinh data/interim/hash_tap_danh_gia/):
  detrac_test_sha256.csv       rel,size,mtime_ns,sha256,w,h -- dinh dang bang BDD (`bdd_yolo_export.sha256_tap_bdd`) them
                               w,h. rel = <video>/img%05d.jpg; thu tu = video theo manifest roi frame tang dan: thu tu dong
                               CHINH LA danh sach anh co thu tu. mtime_ns chi ghi lai, khong dung de bo qua bam.
  detrac_test_loai_D4.csv      rel,video,frame -- anh tren dia KHONG co <frame> trong XML (D4).
  detrac_test_fingerprint.json tom tat may doc duoc; khoi `van_tay` chi gom gia tri suy ra tu NOI DUNG (tat dinh).

Kiem cung (sai -> exit 2, KHONG ghi gi): manifest hop le; tap XML trong `--xml` == tap video manifest; moi video co thu
muc anh; moi muc trong thu muc video la file img%05d.jpg, danh so lien tuc tu 1; moi <frame num> co anh; PIL doc duoc
header moi anh; tong so anh == `--so-anh-ky-vong` (neu khai bao).
Lech ky vong (van ghi dau ra, exit 3, moi cho lech liet ke trong JSON): w/h khac 960x540, dinh dang khac JPEG,
<sequence name> khac ten file, danh sach D4 khac `anh_khong_frame.csv` vong 45, so dem / sha256 XML khac `theo_video.csv`
vong 45. `--kiem-lai`: tinh lai roi so voi 3 file da ghi, khac -> exit 4.

    python src/fingerprint_detrac_test.py --so-anh-ky-vong 56340              # dung bang, ghi 3 file
    python src/fingerprint_detrac_test.py --video MVI_39031 MVI_39051         # chay thu mot phan, KHONG ghi
    python src/fingerprint_detrac_test.py --so-anh-ky-vong 56340 --kiem-lai   # tai kiem: tinh lai, so voi file, KHONG ghi
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAC_DINH = {
    "manifest": "results/phase2/manifests/detrac_test.json",
    "anh": "data/extracted/DETRAC-Images",
    "xml": "data/extracted/DETRAC-Test-Annotations-XML",
    "out": "data/interim/hash_tap_danh_gia",
    "doi_chieu_d4": "handoff/logs/review45/phan_tich/anh_khong_frame.csv",
    "doi_chieu_theo_video": "handoff/logs/review45/phan_tich/theo_video.csv",
}
TEN_RA = {"bang": "detrac_test_sha256.csv", "d4": "detrac_test_loai_D4.csv", "tom_tat": "detrac_test_fingerprint.json"}
COT_BANG = ["rel", "size", "mtime_ns", "sha256", "w", "h"]
COT_NOI_DUNG = ["rel", "size", "sha256", "w", "h"]  # --kiem-lai so cac cot nay + thu tu; mtime_ns chi dem
COT_D4 = ["rel", "video", "frame"]
KICH_THUOC = (960, 540)
TEN_ANH = re.compile(r"img(\d{5})\.jpg")
CAY_CHI_DOC = ("data/extracted", "data/raw", "results")
#: cot cua theo_video.csv vong 45 duoc doi chieu (gia tri so khop theo chuoi)
TRUONG_THEO_VIDEO = ("xml_sha256", "so_frame_xml", "num_min", "num_max", "so_anh", "anh_min", "anh_max", "anh_khong_frame",
                     "doan_anh_khong_frame", "frame_khong_anh", "so_box", "so_vung_bo_qua")
QUY_TAC_D4 = ("anh tren dia khong co <frame num> trong XML -> loai khoi tap danh gia va luu danh sach; khong coi la anh "
              "am tinh da xac nhan (D4, nguoi dung chot 2026-10-09; handoff/REVIEW46_CHI_TIET.md muc 11.4)")
CACH_TINH = {
    "noi_dung_anh_sha256_tong_hop": "sha256_ds((rel, sha256 anh)): sha256 cua cac dong '<rel>\\t<sha256>\\n' da sap xep "
                                    "(nhu danh_gia_detector_bdd_val.sha256_ds)",
    "danh_sach_anh_sha256": "sha256_ds((rel, '')): KHONG phu thuoc thu tu (nhu truong cung ten cua adapter BDD)",
    "danh_sach_anh_co_thu_tu_sha256": "sha256 cua cac dong '<rel>\\n' theo DUNG thu tu dong CSV (khong sap xep lai)",
}


class LoiDuLieu(Exception):
    """Tap video / so anh khong khop manifest / dia, hoac dau vao hong -> exit 2, khong ghi dau ra."""


def p_abs(s: str | Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else REPO / p


def rel_repo(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def doc_byte(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_ds(rows) -> str:
    """Nhu `danh_gia_detector_bdd_val.sha256_ds`: sha256 cua cac dong '<rel>\\t<sha256>\\n' da sap xep."""
    h = hashlib.sha256()
    for rel, s in sorted(rows):
        h.update(f"{rel}\t{s}\n".encode())
    return h.hexdigest()


def sha256_thu_tu(rels) -> str:
    """sha256 cua cac dong '<rel>\\n' theo dung thu tu dua vao (danh sach anh CO THU TU)."""
    h = hashlib.sha256()
    for r in rels:
        h.update(f"{r}\n".encode())
    return h.hexdigest()


def doc_bang_hash(p: Path) -> list[dict]:
    """Doc bang hash anh DETRAC test (moi gia tri la chuoi, giu thu tu dong). Cot khac `COT_BANG` -> LoiDuLieu."""
    with open(p, newline="", encoding="utf-8") as f:
        doc = csv.DictReader(f)
        rows = list(doc)
    if doc.fieldnames != COT_BANG:
        raise LoiDuLieu(f"{p}: cot {doc.fieldnames} khac {COT_BANG}")
    return rows


def csv_bytes(rows: list[dict], cot: list[str]) -> bytes:
    """CSV nhu bang BDD (csv.DictWriter mac dinh, dong ket thuc '\\r\\n'); tra ve byte de bam dung noi dung se ghi."""
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=cot)
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode("utf-8")


def doan(ds: list[int]) -> str:
    """[992, 993, ..., 1060] -> '992-1060' (nhieu doan noi bang ';', nhu theo_video.csv vong 45)."""
    ra, i = [], 0
    while i < len(ds):
        j = i
        while j + 1 < len(ds) and ds[j + 1] == ds[j] + 1:
            j += 1
        ra.append(f"{ds[i]}-{ds[j]}")
        i = j + 1
    return ";".join(ra)


def doc_manifest(p: Path) -> tuple[dict, list[str]]:
    b = doc_byte(p)
    mf = json.loads(b.decode("utf-8"))
    vids = mf.get("videos")
    if not isinstance(vids, list) or not vids or not all(isinstance(v, str) and v for v in vids):
        raise LoiDuLieu(f"{p}: 'videos' khong phai danh sach ten video")
    if len(set(vids)) != len(vids):
        raise LoiDuLieu(f"{p}: manifest liet ke trung video")
    if "n_videos" in mf and mf["n_videos"] != len(vids):
        raise LoiDuLieu(f"{p}: n_videos={mf['n_videos']} khac {len(vids)} video liet ke")
    return {"rel": rel_repo(p), "sha256": sha256_bytes(b), "size": len(b), "so_video": len(vids), "videos": vids}, vids


def doc_xml(p: Path, video: str) -> tuple[dict, set[int]]:
    """sha256 + so dem THO cua mot XML (chua phai GT; bo doc GT theo D1-D5 la viec cua adapter, moc B)."""
    b = doc_byte(p)
    try:
        root = ET.fromstring(b)
    except ET.ParseError as e:
        raise LoiDuLieu(f"{p.name}: XML hong ({e})") from None
    nums = []
    for fr in root.findall("frame"):
        s = fr.get("num") or ""
        if not s.isdigit():
            raise LoiDuLieu(f"{p.name}: <frame> co num khong hop le {s!r}")
        nums.append(int(s))
    if len(set(nums)) != len(nums):
        raise LoiDuLieu(f"{p.name}: trung <frame num> ({len(nums) - len(set(nums))} lan)")
    loai = Counter()
    for t in root.iterfind("frame/target_list/target"):
        at = t.find("attribute")
        loai[(at.get("vehicle_type") if at is not None else None) or "(khong co vehicle_type)"] += 1
    return {"video": video, "rel": rel_repo(p), "sha256": sha256_bytes(b), "size": len(b),
            "ten_sequence": root.get("name"), "so_frame": len(nums), "num_min": min(nums, default=None),
            "num_max": max(nums, default=None), "so_box": sum(loai.values()), "so_box_theo_loai": dict(sorted(loai.items())),
            "so_vung_bo_qua": len(root.findall("ignored_region/box"))}, set(nums)


def liet_ke_anh(d: Path) -> list[tuple[int, Path]]:
    """Moi muc trong thu muc video phai la file img%05d.jpg, danh so lien tuc tu 1 (khac -> LoiDuLieu)."""
    if not d.is_dir():
        raise LoiDuLieu(f"thieu thu muc anh {rel_repo(d)}")
    ds, la = [], []
    with os.scandir(d) as it:
        for e in it:
            m = TEN_ANH.fullmatch(e.name)
            if m and e.is_file():
                ds.append((int(m.group(1)), Path(e.path)))
            else:
                la.append(e.name)
    if la:
        raise LoiDuLieu(f"{d.name}: {len(la)} muc khong phai file img%05d.jpg: {sorted(la)[:5]}")
    if not ds:
        raise LoiDuLieu(f"{d.name}: khong co anh")
    ds.sort()
    if [n for n, _ in ds] != list(range(1, len(ds) + 1)):
        thieu = sorted(set(range(1, ds[-1][0] + 1)) - {n for n, _ in ds})
        raise LoiDuLieu(f"{d.name}: {len(ds)} anh, so thu tu khong lien tuc tu 1 (thieu {thieu[:5]})")
    return ds


def bam_anh(p: Path, Image) -> dict:
    """Doc anh MOT lan: size, sha256 toan bo noi dung, w/h tu header (PIL, khong giai ma pixel), dinh dang."""
    st = p.stat()
    b = doc_byte(p)
    if len(b) != st.st_size:
        raise LoiDuLieu(f"{p}: kich thuoc doi trong luc doc ({st.st_size} -> {len(b)} byte)")
    try:
        with Image.open(io.BytesIO(b)) as im:
            w, h, fmt = int(im.size[0]), int(im.size[1]), im.format
    except Exception as e:  # noqa: BLE001 -- moi loi doc header deu la du lieu hong
        raise LoiDuLieu(f"{p}: PIL khong doc duoc header ({type(e).__name__}: {e})") from None
    return {"size": len(b), "mtime_ns": str(st.st_mtime_ns), "sha256": sha256_bytes(b), "w": w, "h": h, "dinh_dang": fmt}


def doc_csv_doi_chieu(p: Path, cot: set[str]) -> tuple[bytes, list[dict]]:
    """Doc file doi chieu (CSV vong 45); thieu cot -> LoiDuLieu (khong de KeyError giua chung)."""
    b = doc_byte(p)
    doc = csv.DictReader(io.StringIO(b.decode("utf-8"), newline=""))
    rows = list(doc)
    thieu = cot - set(doc.fieldnames or [])
    if thieu:
        raise LoiDuLieu(f"{p}: file doi chieu thieu cot {sorted(thieu)}")
    return b, rows


def doi_chieu_d4(p: Path, d4: list[dict], chay: list[str], vids: list[str], goc_anh_rel: str) -> dict:
    """So danh sach loai D4 voi anh_khong_frame.csv vong 45 (chi cac video dang chay); liet ke TUNG cho lech."""
    b, tat_ca = doc_csv_doi_chieu(p, {"video", "frame", "anh"})
    chon = set(chay)
    ref = [r for r in tat_ca if r["video"] in chon]
    try:
        k_ref = [(r["video"], int(r["frame"])) for r in ref]
    except ValueError as e:
        raise LoiDuLieu(f"{p}: cot frame khong phai so ({e})") from None
    k_moi = [(r["video"], r["frame"]) for r in d4]
    s_ref, s_moi = set(k_ref), set(k_moi)
    kq = {"file": rel_repo(p), "file_sha256": sha256_bytes(b), "so_dong_doi_chieu": len(ref), "so_anh_loai_D4": len(d4),
          "chung": len(s_ref & s_moi), "cung_thu_tu": k_ref == k_moi, "dong_trung_trong_file": len(k_ref) - len(s_ref),
          "chi_co_trong_file": [f"{v}/img{n:05d}.jpg" for v, n in sorted(s_ref - s_moi)],
          "chi_co_trong_D4_moi": [f"{v}/img{n:05d}.jpg" for v, n in sorted(s_moi - s_ref)],
          "cot_anh_khac_duong_dan": [r["anh"] for r in ref
                                     if r["anh"] != f"{goc_anh_rel}/{r['video']}/img{int(r['frame']):05d}.jpg"],
          "dong_video_ngoai_manifest": sorted({r["video"] for r in tat_ca} - set(vids))}
    kq["khop"] = not (kq["chi_co_trong_file"] or kq["chi_co_trong_D4_moi"] or kq["dong_trung_trong_file"]
                      or kq["cot_anh_khac_duong_dan"] or kq["dong_video_ngoai_manifest"])
    return kq


def doi_chieu_theo_video(p: Path, xmls: list[dict], theo_video: list[dict], day_du: bool) -> dict:
    """So sha256 XML va so dem tung video voi theo_video.csv vong 45; liet ke TUNG truong lech."""
    b, rows = doc_csv_doi_chieu(p, {"video", *TRUONG_THEO_VIDEO})
    ref = {r["video"]: r for r in rows}
    lech = []
    if len(ref) != len(rows):
        lech.append({"video": "(file)", "truong": "dong trung video", "vong45": len(rows), "moi": len(ref)})
    for x, t in zip(xmls, theo_video):
        moi = {"xml_sha256": x["sha256"], "so_frame_xml": x["so_frame"], "num_min": x["num_min"], "num_max": x["num_max"],
               "so_anh": t["so_anh"], "anh_min": t["frame_dau"], "anh_max": t["frame_cuoi"],
               "anh_khong_frame": t["so_anh_loai_D4"], "doan_anh_khong_frame": t["doan_loai_D4"], "frame_khong_anh": 0,
               "so_box": x["so_box"], "so_vung_bo_qua": x["so_vung_bo_qua"]}
        r = ref.get(x["video"])
        if r is None:
            lech.append({"video": x["video"], "truong": "(khong co dong trong file)", "vong45": None, "moi": None})
            continue
        lech += [{"video": x["video"], "truong": k, "vong45": r.get(k), "moi": moi[k]}
                 for k in TRUONG_THEO_VIDEO if r.get(k) != str(moi[k])]
    thua = sorted(set(ref) - {x["video"] for x in xmls}) if day_du else []
    return {"file": rel_repo(p), "file_sha256": sha256_bytes(b), "so_video_doi_chieu": len(xmls),
            "truong": list(TRUONG_THEO_VIDEO), "lech": lech, "video_thua_trong_file": thua, "khop": not (lech or thua)}


def tinh(a: argparse.Namespace) -> dict:
    """Doc + kiem + bam. Tra ve dong bang, dong D4, khoi van_tay, doi chieu, lech ky vong. Loi cung -> LoiDuLieu."""
    import PIL
    from PIL import Image
    man, goc_anh, goc_xml = p_abs(a.manifest), p_abs(a.anh), p_abs(a.xml)
    mf_info, vids = doc_manifest(man)
    co_xml = sorted(p.stem for p in goc_xml.glob("*.xml"))
    if set(co_xml) != set(vids):
        raise LoiDuLieu(f"tap XML trong {rel_repo(goc_xml)} ({len(co_xml)}) khac manifest ({len(vids)}): thieu "
                        f"{sorted(set(vids) - set(co_xml))[:5]}, thua {sorted(set(co_xml) - set(vids))[:5]}")
    thieu_dir = [v for v in vids if not (goc_anh / v).is_dir()]
    if thieu_dir:
        raise LoiDuLieu(f"{len(thieu_dir)} video manifest khong co thu muc anh: {thieu_dir[:5]}")
    chay = vids
    if a.video:
        ngoai = [v for v in a.video if v not in vids]
        if ngoai:
            raise LoiDuLieu(f"--video ngoai manifest: {ngoai}")
        chay = [v for v in vids if v in set(a.video)]  # giu thu tu manifest
    rows, d4, xmls, theo_video, lech_kt = [], [], [], [], []
    kt, dinh_dang = Counter(), Counter()
    for i, v in enumerate(chay, 1):
        t0 = time.perf_counter()
        xi, nums = doc_xml(goc_xml / f"{v}.xml", v)
        ds = liet_ke_anh(goc_anh / v)
        khong_anh = sorted(nums - {n for n, _ in ds})
        if khong_anh:
            raise LoiDuLieu(f"{v}: {len(khong_anh)} <frame> khong co anh tren dia: {khong_anh[:5]}")
        n_bytes, d4_v = 0, []
        for n, p in ds:
            rel = f"{v}/{p.name}"
            r = bam_anh(p, Image)
            rows.append({"rel": rel, **{k: r[k] for k in COT_BANG[1:]}})
            n_bytes += r["size"]
            kt[f"{r['w']}x{r['h']}"] += 1
            dinh_dang[str(r["dinh_dang"])] += 1
            if (r["w"], r["h"]) != KICH_THUOC:
                lech_kt.append({"rel": rel, "w": r["w"], "h": r["h"]})
            if n not in nums:
                d4_v.append(n)
        d4 += [{"rel": f"{v}/img{n:05d}.jpg", "video": v, "frame": n} for n in d4_v]
        xmls.append(xi)
        theo_video.append({"video": v, "so_anh": len(ds), "frame_dau": ds[0][0], "frame_cuoi": ds[-1][0],
                           "bytes": n_bytes, "so_anh_loai_D4": len(d4_v), "doan_loai_D4": doan(d4_v)})
        print(f"[{i}/{len(chay)}] {v}: {len(ds)} anh, {len(nums)} frame XML, {len(d4_v)} anh loai D4, "
              f"{time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    kiem_cung = [f"manifest hop le, {len(vids)} video khong trung",
                 f"tap XML trong {rel_repo(goc_xml)} == tap video manifest ({len(co_xml)})",
                 f"moi video manifest co thu muc anh ({len(vids)})",
                 "moi muc trong thu muc video la file img%05d.jpg, danh so lien tuc tu 1",
                 "moi <frame num> trong XML co anh tren dia", "PIL doc duoc header moi anh"]
    if a.so_anh_ky_vong is not None and not a.video:
        if len(rows) != a.so_anh_ky_vong:
            raise LoiDuLieu(f"tong so anh tren dia {len(rows)} khac --so-anh-ky-vong {a.so_anh_ky_vong}")
        kiem_cung.append(f"tong so anh == --so-anh-ky-vong ({a.so_anh_ky_vong})")
    rels = [r["rel"] for r in rows]
    bo = {r["rel"] for r in d4}
    con = [r for r in rows if r["rel"] not in bo]
    loai_tong = Counter()
    for x in xmls:
        loai_tong.update(x["so_box_theo_loai"])
    van_tay = {
        "manifest": mf_info,
        "xml": xmls,
        "xml_tong": {"so_xml": len(xmls), "so_frame": sum(x["so_frame"] for x in xmls),
                     "so_box": sum(x["so_box"] for x in xmls), "so_box_theo_loai": dict(sorted(loai_tong.items())),
                     "so_vung_bo_qua": sum(x["so_vung_bo_qua"] for x in xmls),
                     "ghi_chu": "so dem tren XML goc (chua cat khung, chua ap D1-D5); khong phai metric"},
        "anh": {"thu_muc": rel_repo(goc_anh), "mau_rel": "<video>/img%05d.jpg", "so_video": len(chay),
                "so_anh": len(rows), "tong_bytes": sum(t["bytes"] for t in theo_video),
                "kich_thuoc_ky_vong": f"{KICH_THUOC[0]}x{KICH_THUOC[1]}", "kich_thuoc": dict(sorted(kt.items())),
                "so_anh_lech_kich_thuoc": len(lech_kt), "anh_lech_kich_thuoc_100_dau": lech_kt[:100],
                "dinh_dang": dict(sorted(dinh_dang.items())), "theo_video": theo_video},
        "tong_hop": {"noi_dung_anh_sha256_tong_hop": sha256_ds((r["rel"], r["sha256"]) for r in rows),
                     "danh_sach_anh_sha256": sha256_ds((r, "") for r in rels),
                     "danh_sach_anh_co_thu_tu_sha256": sha256_thu_tu(rels), "cach_tinh": CACH_TINH},
        "loai_D4": {"quy_tac": QUY_TAC_D4, "so_anh": len(d4),
                    "theo_video": {t["video"]: {"so_anh": t["so_anh_loai_D4"], "doan": t["doan_loai_D4"]}
                                   for t in theo_video if t["so_anh_loai_D4"]},
                    "danh_sach_co_thu_tu_sha256": sha256_thu_tu(r["rel"] for r in d4)},
        "tap_danh_gia_sau_D4": {"so_anh": len(con), "danh_sach_anh_co_thu_tu_sha256": sha256_thu_tu(r["rel"] for r in con),
                                "noi_dung_anh_sha256_tong_hop": sha256_ds((r["rel"], r["sha256"]) for r in con),
                                "cach_lay": "dong cua bang hash, bo cac rel co trong file loai D4, giu nguyen thu tu"},
    }
    lech = []
    if lech_kt:
        lech.append(f"{len(lech_kt)} anh co w/h khac {KICH_THUOC[0]}x{KICH_THUOC[1]}")
    if set(dinh_dang) != {"JPEG"}:
        lech.append(f"dinh dang anh khong chi JPEG: {dict(dinh_dang)}")
    sai_ten = [x["video"] for x in xmls if x["ten_sequence"] != x["video"]]
    if sai_ten:
        lech.append(f"{len(sai_ten)} XML co <sequence name> khac ten file: {sai_ten}")
    doi_chieu = {}
    if a.doi_chieu_d4:
        dc = doi_chieu_d4(p_abs(a.doi_chieu_d4), d4, chay, vids, rel_repo(goc_anh))
        doi_chieu["d4_vong45"] = dc
        if not dc["khop"]:
            lech.append(f"danh sach D4 khac {dc['file']} (xem doi_chieu.d4_vong45)")
    if a.doi_chieu_theo_video:
        dc = doi_chieu_theo_video(p_abs(a.doi_chieu_theo_video), xmls, theo_video, day_du=not a.video)
        doi_chieu["theo_video_vong45"] = dc
        if not dc["khop"]:
            lech.append(f"{len(dc['lech'])} truong khac {dc['file']} (xem doi_chieu.theo_video_vong45)")
    return {"rows": rows, "d4": d4, "van_tay": van_tay, "doi_chieu": doi_chieu, "lech": lech, "kiem_cung": kiem_cung,
            "PIL": PIL.__version__}


def kiem_lai(out: Path, rows: list[dict], b_d4: bytes, van_tay: dict) -> dict:
    """So ket qua tinh lai voi 3 file da ghi: noi dung + thu tu dong bang hash, file D4, khoi van_tay; mtime_ns chi dem."""
    p = {k: out / v for k, v in TEN_RA.items()}
    thieu = [x.name for x in p.values() if not x.is_file()]
    if thieu:
        return {"khop": False, "loi": [f"thieu file {thieu}"]}
    loi = []
    b_tom_cu, b_bang_cu, b_d4_cu = doc_byte(p["tom_tat"]), doc_byte(p["bang"]), doc_byte(p["d4"])
    cu = json.loads(b_tom_cu.decode("utf-8"))
    dr = cu.get("dau_ra") or {}
    if sha256_bytes(b_bang_cu) != (dr.get("bang") or {}).get("sha256"):
        loi.append("sha256 file bang hash khac gia tri ghi trong JSON")
    if sha256_bytes(b_d4_cu) != (dr.get("d4") or {}).get("sha256"):
        loi.append("sha256 file loai D4 khac gia tri ghi trong JSON")
    if b_d4_cu != b_d4:
        loi.append("danh sach loai D4 tinh lai khac file")
    doc = csv.DictReader(io.StringIO(b_bang_cu.decode("utf-8"), newline=""))
    cu_rows = list(doc)
    if doc.fieldnames != COT_BANG:
        loi.append(f"cot bang hash {doc.fieldnames} khac {COT_BANG}")
    if len(cu_rows) != len(rows):
        loi.append(f"bang hash co {len(cu_rows)} dong, tinh lai {len(rows)}")
    khac = [m["rel"] for c, m in zip(cu_rows, rows) if any(c.get(k) != str(m[k]) for k in COT_NOI_DUNG)]
    if khac:
        loi.append(f"{len(khac)} dong khac noi dung / thu tu, vd {khac[:5]}")
    vt, vt_cu = json.loads(json.dumps(van_tay)), cu.get("van_tay") or {}
    if vt != vt_cu:
        loi.append(f"khoi van_tay khac o: {sorted(k for k in set(vt) | set(vt_cu) if vt.get(k) != vt_cu.get(k))}")
    return {"khop": not loi, "loi": loi, "so_dong_da_so": min(len(cu_rows), len(rows)),
            "so_dong_chi_khac_mtime_ns": sum(c.get("mtime_ns") != str(m["mtime_ns"]) for c, m in zip(cu_rows, rows)),
            "file": {k: rel_repo(x) for k, x in p.items()},
            "sha256_file_tren_dia": {"bang": sha256_bytes(b_bang_cu), "d4": sha256_bytes(b_d4_cu),
                                     "tom_tat": sha256_bytes(b_tom_cu)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=MAC_DINH["manifest"])
    ap.add_argument("--anh", default=MAC_DINH["anh"], help="thu muc goc anh, moi video mot thu muc con")
    ap.add_argument("--xml", default=MAC_DINH["xml"], help="thu muc XML nhan test, <video>.xml")
    ap.add_argument("--out", default=MAC_DINH["out"], help="thu muc ghi 3 file dau ra")
    ap.add_argument("--doi-chieu-d4", default=MAC_DINH["doi_chieu_d4"], help="anh_khong_frame.csv vong 45 ('' = bo)")
    ap.add_argument("--doi-chieu-theo-video", default=MAC_DINH["doi_chieu_theo_video"],
                    help="theo_video.csv vong 45 ('' = bo)")
    ap.add_argument("--so-anh-ky-vong", type=int, help="tong so anh ky vong tren dia (bo qua khi --video); khac -> exit 2")
    ap.add_argument("--video", nargs="+", help="chay thu tren mot phan video cua manifest; KHONG ghi dau ra")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--khong-ghi", action="store_true", help="tinh va kiem day du nhung KHONG ghi dau ra")
    g.add_argument("--kiem-lai", action="store_true", help="tinh lai va so voi 3 file da ghi trong --out; KHONG ghi")
    a = ap.parse_args()
    if a.video and a.kiem_lai:
        ap.error("--kiem-lai can tinh tren DU manifest, khong dung cung --video")
    che_do = "kiem_lai" if a.kiem_lai else "thu_mot_phan" if a.video else "khong_ghi" if a.khong_ghi else "ghi"
    out = p_abs(a.out).resolve()
    for cay in CAY_CHI_DOC:
        goc = (REPO / cay).resolve()
        if out == goc or goc in out.parents:
            print(f"[LOI] --out {out} nam trong cay chi doc {cay}", file=sys.stderr)
            return 2
    bat_dau, t0 = dt.datetime.now().astimezone().isoformat(timespec="seconds"), time.perf_counter()
    try:
        kq = tinh(a)
    except LoiDuLieu as e:
        print(f"[LOI DU LIEU] {e}\nKhong ghi dau ra. exit 2", file=sys.stderr)
        return 2
    b_bang, b_d4 = csv_bytes(kq["rows"], COT_BANG), csv_bytes(kq["d4"], COT_D4)
    dau_ra = {"bang": {"rel": rel_repo(out / TEN_RA["bang"]), "sha256": sha256_bytes(b_bang), "so_dong": len(kq["rows"]),
                       "cot": COT_BANG, "thu_tu": "video theo manifest, roi frame tang dan (= danh sach anh co thu tu)",
                       "mtime_ns": "chi ghi lai; khong dung de bo qua bam; --kiem-lai khong tinh la lech"},
              "d4": {"rel": rel_repo(out / TEN_RA["d4"]), "sha256": sha256_bytes(b_d4), "so_dong": len(kq["d4"]),
                     "cot": COT_D4}}
    ma = 3 if kq["lech"] else 0
    kl = None
    if a.kiem_lai:
        kl = kiem_lai(out, kq["rows"], b_d4, kq["van_tay"])
        if not kl["khop"]:
            ma = 4
    tom = {"phien_ban": 1,
           "mo_ta": "Dau van tay du lieu UA-DETRAC test (ke hoach R47 moc A; du thao vong 45 muc 7). Chi doc du lieu.",
           "lan_chay": {"che_do": che_do, "lenh": [sys.executable, *sys.argv], "bat_dau": bat_dau,
                        "ket_thuc": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "thoi_gian_s": round(time.perf_counter() - t0, 1), "python": sys.version.split()[0],
                        "PIL": kq["PIL"], "script": rel_repo(Path(__file__)),
                        "script_sha256": sha256_bytes(doc_byte(Path(__file__))),
                        "so_anh_ky_vong": a.so_anh_ky_vong, "kiem_cung_da_qua": kq["kiem_cung"]},
           "van_tay": kq["van_tay"], "dau_ra": dau_ra, "doi_chieu": kq["doi_chieu"], "lech_ky_vong": kq["lech"],
           "exit": ma}
    b_tom = (json.dumps(tom, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if che_do == "ghi":
        out.mkdir(parents=True, exist_ok=True)
        for ten, b in (("d4", b_d4), ("bang", b_bang), ("tom_tat", b_tom)):
            with open(out / TEN_RA[ten], "wb") as f:
                f.write(b)
    vt = kq["van_tay"]
    in_ra = {"che_do": che_do, "so_video": vt["anh"]["so_video"], "so_anh": vt["anh"]["so_anh"],
             "so_frame_xml": vt["xml_tong"]["so_frame"], "so_anh_loai_D4": vt["loai_D4"]["so_anh"],
             "so_anh_sau_D4": vt["tap_danh_gia_sau_D4"]["so_anh"], "kich_thuoc": vt["anh"]["kich_thuoc"],
             "so_anh_lech_kich_thuoc": vt["anh"]["so_anh_lech_kich_thuoc"], "dinh_dang": vt["anh"]["dinh_dang"],
             "manifest_sha256": vt["manifest"]["sha256"],
             "tong_hop": {k: v for k, v in vt["tong_hop"].items() if k != "cach_tinh"},
             "sau_D4": {k: v for k, v in vt["tap_danh_gia_sau_D4"].items() if k != "cach_lay"},
             "dau_ra_sha256": {k: v["sha256"] for k, v in dau_ra.items()},
             "doi_chieu_khop": {k: v["khop"] for k, v in kq["doi_chieu"].items()},
             "doi_chieu_lech_chi_tiet": {k: v for k, v in kq["doi_chieu"].items() if not v["khop"]},
             "lech_ky_vong": kq["lech"],
             "kiem_lai": kl, "da_ghi": che_do == "ghi",
             "tom_tat_da_ghi_sha256": sha256_bytes(b_tom) if che_do == "ghi" else None,
             "thoi_gian_s": tom["lan_chay"]["thoi_gian_s"], "exit": ma}
    print(json.dumps(in_ra, ensure_ascii=True, indent=1))
    return ma


if __name__ == "__main__":
    sys.exit(main())
