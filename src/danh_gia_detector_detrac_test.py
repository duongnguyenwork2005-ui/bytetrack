"""
danh_gia_detector_detrac_test.py -- Danh gia detector 5 lop tren UA-DETRAC test (ke hoach R47 moc B; du thao
`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md` muc 8; quyet dinh D1-D5 cua nguoi dung ngay 2026-10-09,
`handoff/REVIEW46_CHI_TIET.md` muc 11.4).

File nay CHI them: bo doc GT theo D1-D5, kiem du lieu theo fingerprint cua moc A, chuan hoa bao cao theo D3, danh sach anh
'video/frame' co thu tu. Du doan, ghep, AP va van hanh dung NGUYEN ham cua `eval_detector_5lop` (du_doan_ultralytics,
danh_gia, luu_du_doan) va `danh_gia_detector_bdd_val` (quy_tac_hieu_luc, chan_doan) -- khong doi phep ghep / AP.

Quy tac GT (nguyen van quyet dinh: REVIEW46_CHI_TIET.md muc 11.4):
  - car, bus trong XML -> GT lop car (0), bus (2);
  - D1: box van -> VUNG BO QUA cho moi lop; khong gop van vao GT car;
  - D2: box others -> VUNG BO QUA cho moi lop; khong gan others thanh GT truck;
  - ignored_region (cap sequence) -> vung bo qua cho moi anh cua video;
  - D4: anh tren dia khong co <frame num> trong XML -> LOAI khoi tap danh gia, luu danh sach; khong coi la anh am tinh;
  - D5: quy tac bo qua cua evaluator (nhu BDD): ghep GT hop le truoc; CHI detection chua ghep moi xet bo qua; IoA = giao /
    dien tich detection, max theo tung vung; nguong IoA = nguong IoU dang xet; GT hop le nam trong vung bo qua VAN duoc
    cham; KHONG to xam anh;
  - box (left, top, width, height) -> xyxy, cat vao khung anh thuc (`bdd_yolo_export._hop_le_box`); < 1 px sau cat -> bo
    va dem rieng; khong sua XML.
D3 (bao cao): lop danh gia car, bus; 'mAP (car, bus)' = trung binh 2 lop; truck / motorcycle / bicycle = 'N/A' trong ca JSON
va CSV, khong vao mAP. Bo dem tho cua scorer cho 3 lop do nam o truong / file rieng 'so_lieu_tho_khong_phai_metric'.

Kiem truoc moi du doan (sai -> exit 2, KHONG du doan): sha256 checkpoint == --sha256; manifest va tung XML == fingerprint;
bang hash va danh sach D4 == fingerprint; tap + thu tu anh tren dia == bang hash; w/h MOI anh (header) == kich thuoc ky vong
va == bang hash; sha256 NOI DUNG tung anh == bang hash (doc lai toan bo, khong dung mtime); danh sach loai D4 tinh lai ==
file D4; danh sach anh danh gia co thu tu va tong hop noi dung == fingerprint (tap_danh_gia_sau_D4).
Sau moi NPZ (TRUOC khi cham) ghi `nguon_du_doan_<ten>.json`: checkpoint, cau hinh du doan, danh tinh du lieu, sha256 NPZ
(ban sua sau review Codex 2026-10-10). `--cham-lai` doc file nay (luot cua ban dau: lay tu ket_qua.json), chi cham khi danh
tinh du lieu hien tai, cau hinh du doan va NPZ trung dung luc du doan; buoc cham loi van cham lai duoc ma khong du doan lai.

    python -u src/danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=<hex> \
        --out results/phase2/detector_eval/detrac_test/<luot>
    python -u src/danh_gia_detector_detrac_test.py --chi-kiem                      # chi kiem du lieu + doc GT, khong nap model
    python -u src/danh_gia_detector_detrac_test.py --cham-lai <thu muc luot> --out <thu muc moi>   # cham lai tu NPZ da luu
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import fingerprint_detrac_test as fpt  # noqa: E402  (bam / doc bang / danh sach co thu tu cua moc A)
from bdd_yolo_export import _hop_le_box  # noqa: E402
from danh_gia_detector_bdd_val import chan_doan, quy_tac_hieu_luc  # noqa: E402
from detector_classes import ID_LOP_5, LOP_5  # noqa: E402
from eval_detector_5lop import QUY_TAC, _json_safe, danh_gia, du_doan_ultralytics, luu_du_doan, sha256_file  # noqa: E402

LoiDuLieu = fpt.LoiDuLieu
MAC_DINH = {
    "xml": "data/extracted/DETRAC-Test-Annotations-XML",
    "anh": "data/extracted/DETRAC-Images",
    "manifest": "results/phase2/manifests/detrac_test.json",
    "fingerprint": "data/interim/hash_tap_danh_gia/detrac_test_fingerprint.json",
    "hash_anh": "data/interim/hash_tap_danh_gia/detrac_test_sha256.csv",
    "loai_d4": "data/interim/hash_tap_danh_gia/detrac_test_loai_D4.csv",
    "kich_thuoc": "960x540",
}
LOAI_GT = {"car": "car", "bus": "bus"}           # vehicle_type XML -> lop GT
LOAI_BO_QUA = ("van", "others")                   # D1, D2: thanh vung bo qua cho moi lop
LOP_DANH_GIA = ("car", "bus")                     # D3
NHAN_MAP = "mAP (car, bus)"
NA = "N/A"
GHI_CHU_THO = ("tho, khong phai metric: bo dem cua scorer cho lop khong thuoc tap danh gia D3 (UA-DETRAC test khong co GT "
               "lop nay); khong trinh bay nhu chat luong detector")
QUY_TAC_GT = {
    "nguon_quyet_dinh": "D1-D5 nguoi dung chot 2026-10-09 (handoff/REVIEW46_CHI_TIET.md muc 11.4)",
    "gan_nhan": "vehicle_type car, bus -> GT lop car (0), bus (2)",
    "D1_van": "box van -> vung bo qua cho moi lop; khong gop van vao GT car",
    "D2_others": "box others -> vung bo qua cho moi lop; khong gan others thanh GT truck",
    "ignored_region": "vung bo qua cap sequence, ap cho moi anh cua video",
    "D3": "chi bao AP cho car, bus; 'mAP (car, bus)' = trung binh 2 lop; truck, motorcycle, bicycle = N/A, khong vao mAP",
    "D4": "anh tren dia khong co <frame num> trong XML -> loai khoi tap danh gia, luu danh sach; khong coi la anh am tinh",
    "D5": ("quy tac bo qua cua evaluator nhu BDD: ghep GT hop le truoc; chi detection chua ghep moi xet bo qua; IoA = giao / "
           "dien tich detection, max theo tung vung; nguong IoA = nguong IoU dang xet; GT hop le trong vung bo qua van duoc "
           "cham; khong to xam anh"),
    "box": "(left, top, width, height) -> xyxy; cat vao khung anh thuc; < 1 px sau cat -> bo (dem rieng); khong sua XML",
}


def p_abs(s: str | Path) -> Path:
    p = Path(s)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def _box(b) -> tuple[float, float, float, float]:
    x, y, w, h = (float(b.get(k)) for k in ("left", "top", "width", "height"))
    return x, y, x + w, y + h


def doc_gt_xml(p: Path) -> tuple[list, dict]:
    """(vung bo qua cap sequence, {num: [(vehicle_type, xyxy), ...]}). vehicle_type ngoai car/bus/van/others -> LoiDuLieu."""
    try:
        root = ET.fromstring(p.read_bytes())
    except ET.ParseError as e:
        raise LoiDuLieu(f"{p.name}: XML hong ({e})") from None
    ign = [_box(b) for b in root.findall("ignored_region/box")]
    frames = {}
    for fr in root.findall("frame"):
        num = int(fr.get("num"))
        if num in frames:
            raise LoiDuLieu(f"{p.name}: trung <frame num={num}>")
        tg = []
        for t in fr.findall("target_list/target"):
            b, at = t.find("box"), t.find("attribute")
            loai = at.get("vehicle_type") if at is not None else None
            if loai not in LOAI_GT and loai not in LOAI_BO_QUA:
                raise LoiDuLieu(f"{p.name} frame {num}: vehicle_type {loai!r} ngoai car/bus/van/others")
            if b is None:
                raise LoiDuLieu(f"{p.name} frame {num}: target thieu <box>")
            tg.append((loai, _box(b)))
        frames[num] = tg
    return ign, frames


def kiem_va_doc_gt(xml_dir: Path, anh_goc: Path, manifest: Path, fingerprint: Path, hash_anh: Path, loai_d4: Path,
                   kich_thuoc: tuple[int, int], kiem_noi_dung: bool = True) -> tuple[dict, list[Path], list[str], list[dict]]:
    """Kiem du lieu theo fingerprint cua moc A roi doc GT theo D1-D5. Tra ve (danh tinh, anh, rel co thu tu, gt)."""
    from PIL import Image
    fp = json.loads(fingerprint.read_text(encoding="utf-8"))
    vt, ra = fp["van_tay"], fp["dau_ra"]
    mf, vids = fpt.doc_manifest(manifest)
    if mf["sha256"] != vt["manifest"]["sha256"] or vids != vt["manifest"]["videos"]:
        raise LoiDuLieu(f"manifest {mf['sha256'][:16]} / {len(vids)} video khac fingerprint {vt['manifest']['sha256'][:16]}")
    for ten, p in (("bang", hash_anh), ("d4", loai_d4)):
        s = sha256_file(p)
        if s != ra[ten]["sha256"]:
            raise LoiDuLieu(f"{p.name}: sha256 {s[:16]} khac fingerprint {ra[ten]['sha256'][:16]}")
    bang = fpt.doc_bang_hash(hash_anh)
    theo_rel = {r["rel"]: r for r in bang}
    with open(loai_d4, newline="", encoding="utf-8") as f:
        d4_file = [r["rel"] for r in csv.DictReader(f)]
    xml_fp = {x["video"]: x["sha256"] for x in vt["xml"]}
    W0, H0 = kich_thuoc
    dem = {"anh_tren_dia": 0, "anh_danh_gia": 0, "anh_loai_D4": 0, "anh_khong_gt": 0, "gt_theo_lop": {c: 0 for c in LOP_5},
           "vung_bo_qua_ignored_region": 0, "vung_bo_qua_tu_van": 0, "vung_bo_qua_tu_others": 0, "vung_bo_qua_rong": 0,
           "box_gt_nho_hon_1px_sau_cat": 0, "box_gt_bi_cat_vao_khung": 0}
    anh, rel_ds, gt, d4_moi, rel_dia, hash_eval = [], [], [], [], [], []
    for v in vids:
        px = xml_dir / f"{v}.xml"
        s = sha256_file(px)
        if s != xml_fp.get(v):
            raise LoiDuLieu(f"{px.name}: sha256 {s[:16]} khac fingerprint {str(xml_fp.get(v))[:16]}")
        ign_seq, frames = doc_gt_xml(px)
        ds = fpt.liet_ke_anh(anh_goc / v)
        thieu_anh = sorted(set(frames) - {n for n, _ in ds})
        if thieu_anh:
            raise LoiDuLieu(f"{v}: {len(thieu_anh)} <frame> khong co anh (vd {thieu_anh[:3]})")
        for num, p in ds:
            rel = f"{v}/{p.name}"
            rel_dia.append(rel)
            r = theo_rel.get(rel)
            if r is None:
                raise LoiDuLieu(f"{rel}: co tren dia nhung khong co trong bang hash")
            if kiem_noi_dung:
                info = fpt.bam_anh(p, Image)
                W, H, sha = info["w"], info["h"], info["sha256"]
                if sha != r["sha256"]:
                    raise LoiDuLieu(f"{rel}: noi dung anh {sha[:16]} khac bang hash {r['sha256'][:16]}")
            else:
                with Image.open(p) as im:
                    W, H = int(im.size[0]), int(im.size[1])
                sha = r["sha256"]
            if (W, H) != (W0, H0) or (str(W), str(H)) != (r["w"], r["h"]):
                raise LoiDuLieu(f"{rel}: kich thuoc {W}x{H} khac ky vong {W0}x{H0} / bang hash {r['w']}x{r['h']}")
            dem["anh_tren_dia"] += 1
            if num not in frames:            # D4
                d4_moi.append(rel)
                dem["anh_loai_D4"] += 1
                continue
            cls, box, ign = [], [], []
            for b0 in ign_seq:
                b = _hop_le_box(*b0, W, H)
                if b is None:
                    dem["vung_bo_qua_rong"] += 1
                else:
                    ign.append(b)
                    dem["vung_bo_qua_ignored_region"] += 1
            for loai, b0 in frames[num]:
                b = _hop_le_box(*b0, W, H)
                if loai in LOAI_GT:
                    if b is None:
                        dem["box_gt_nho_hon_1px_sau_cat"] += 1
                        continue
                    if b != b0:
                        dem["box_gt_bi_cat_vao_khung"] += 1
                    cls.append(ID_LOP_5[LOAI_GT[loai]])
                    box.append(b)
                    dem["gt_theo_lop"][LOAI_GT[loai]] += 1
                elif b is None:              # D1 / D2: van, others -> vung bo qua
                    dem["vung_bo_qua_rong"] += 1
                else:
                    ign.append(b)
                    dem[f"vung_bo_qua_tu_{loai}"] += 1
            dem["anh_danh_gia"] += 1
            dem["anh_khong_gt"] += int(not cls)
            anh.append(p)
            rel_ds.append(rel)
            hash_eval.append((rel, sha))
            gt.append({"anh": p.name, "rel": rel, "w": W, "h": H, "cls": np.array(cls, int),
                       "xyxy": np.array(box, float).reshape(-1, 4), "ign": np.array(ign, float).reshape(-1, 4)})
    if rel_dia != [r["rel"] for r in bang]:
        raise LoiDuLieu(f"tap / thu tu anh tren dia ({len(rel_dia)}) khac bang hash ({len(bang)})")
    if d4_moi != d4_file:
        raise LoiDuLieu(f"danh sach loai D4 tinh lai ({len(d4_moi)}) khac file D4 ({len(d4_file)})")
    sau = vt["tap_danh_gia_sau_D4"]
    thu_tu = fpt.sha256_thu_tu(rel_ds)
    if thu_tu != sau["danh_sach_anh_co_thu_tu_sha256"] or len(rel_ds) != sau["so_anh"]:
        raise LoiDuLieu(f"danh sach anh danh gia ({len(rel_ds)}, {thu_tu[:16]}) khac fingerprint ({sau['so_anh']}, "
                        f"{sau['danh_sach_anh_co_thu_tu_sha256'][:16]})")
    danh_tinh = {"manifest": str(manifest), "manifest_sha256": mf["sha256"], "so_video": len(vids),
                 "fingerprint": str(fingerprint), "fingerprint_sha256": sha256_file(fingerprint),
                 "bang_hash_sha256": ra["bang"]["sha256"], "loai_D4_sha256": ra["d4"]["sha256"],
                 "danh_sach_anh_co_thu_tu_sha256": thu_tu, "loai_D4": d4_moi, "dem": dem,
                 "kiem_noi_dung_anh": kiem_noi_dung}
    if kiem_noi_dung:
        tong = fpt.sha256_ds(hash_eval)
        if tong != sau["noi_dung_anh_sha256_tong_hop"]:
            raise LoiDuLieu(f"tong hop noi dung anh danh gia {tong[:16]} khac fingerprint {sau['noi_dung_anh_sha256_tong_hop'][:16]}")
        danh_tinh["noi_dung_anh_sha256_tong_hop"] = tong
    return danh_tinh, anh, rel_ds, gt


def chuan_hoa_d3(r: dict) -> dict:
    """Bao cao theo D3 tu ket qua `danh_gia` (5 lop). Lop danh gia khong co GT, hoac lop ngoai tap co GT -> LoiDuLieu."""
    tl = r["theo_lop"]
    for lop in LOP_5:
        if lop not in LOP_DANH_GIA and tl[lop]["n_gt"]:
            raise LoiDuLieu(f"lop {lop} ngoai tap D3 lai co GT ({tl[lop]['n_gt']})")
    co = [lop for lop in LOP_DANH_GIA if tl[lop]["n_gt"]]
    if list(co) != list(LOP_DANH_GIA):
        raise LoiDuLieu(f"lop danh gia khong co GT: {sorted(set(LOP_DANH_GIA) - set(co))}")

    def tb(f):
        return float(np.mean([f(tl[lop]) for lop in LOP_DANH_GIA]))

    m = {"mAP50": tb(lambda x: x["AP50"]), "mAP50_95": tb(lambda x: x["AP50_95"]), "mAP75": tb(lambda x: x["AP75"]),
         "precision_vh": tb(lambda x: x["van_hanh"]["precision"]), "recall_vh": tb(lambda x: x["van_hanh"]["recall"])}
    mac = r["macro"]          # evaluator: trung binh cac lop co GT -> phai trung dung car, bus
    if mac["lop_khong_co_gt"] != [lop for lop in LOP_5 if lop not in LOP_DANH_GIA] or \
            abs(mac["mAP50"] - m["mAP50"]) > 1e-12 or abs(mac["mAP50_95"] - m["mAP50_95"]) > 1e-12:
        raise LoiDuLieu(f"macro cua evaluator khac {NHAN_MAP} tinh lai")
    return {"lop_danh_gia": list(LOP_DANH_GIA), "lop_ngoai_tap": [lop for lop in LOP_5 if lop not in LOP_DANH_GIA],
            NHAN_MAP: m, "theo_lop": {lop: (tl[lop] if lop in LOP_DANH_GIA else NA) for lop in LOP_5},
            "so_lieu_tho_khong_phai_metric": {"ghi_chu": GHI_CHU_THO,
                                               "theo_lop": {lop: tl[lop] for lop in LOP_5 if lop not in LOP_DANH_GIA}},
            "n_anh": r["n_anh"], "van_hanh": r["van_hanh"]}


def doc_du_doan(npz: Path, n: int) -> tuple[np.ndarray, list[dict]]:
    """Du doan da luu boi `luu_du_doan` -> (ten anh, danh sach du doan theo anh).

    Doc moi mang MOT lan: z[...] cua NpzFile giai nen lai ca mang tu file zip o moi lan truy cap (truoc day goi trong vong
    lap theo anh -> ~24 h tren 56.167 anh that)."""
    with np.load(npz) as z:
        anh, idx, xyxy, conf, cls = z["anh"], z["idx"].astype(np.int64), z["xyxy"], z["conf"], z["cls"]
    if len(idx) and (np.any(np.diff(idx) < 0) or idx.min() < 0 or idx.max() >= n):
        raise LoiDuLieu(f"{npz.name}: chi so anh khong hop le")
    cat = np.searchsorted(idx, np.arange(n + 1))
    dd = [{"xyxy": xyxy[cat[i]:cat[i + 1]].astype(float), "conf": conf[cat[i]:cat[i + 1]].astype(float),
           "cls": cls[cat[i]:cat[i + 1]].astype(int)} for i in range(n)]
    return anh, dd


DANH_TINH_NGUON = ("manifest_sha256", "fingerprint_sha256", "bang_hash_sha256", "loai_D4_sha256",
                   "danh_sach_anh_co_thu_tu_sha256", "noi_dung_anh_sha256_tong_hop")


def danh_tinh_nguon(danh_tinh: dict) -> dict:
    """Danh tinh du lieu ma mot bo du doan duoc tinh tren do (de cham lai chi tren DUNG du lieu do)."""
    return {k: danh_tinh.get(k) for k in DANH_TINH_NGUON}


def ghi_nguon(out: Path, ten: str, w: str, sha: str, mt, qt: dict, danh_tinh: dict, n: int) -> None:
    """nguon_du_doan_<ten>.json: ghi NGAY sau NPZ, TRUOC khi cham -- checkpoint, cau hinh du doan, danh tinh du lieu, sha256
    NPZ. --cham-lai doc file nay (khong can ket_qua.json), nen buoc cham loi van phuc hoi duoc ma khong du doan lai."""
    npz = out / f"du_doan_{ten}.npz"
    meta = {"phien_ban": 1, "mo_hinh": ten, "checkpoint": w, "sha256": sha, "anh_xa_lop": mt, "quy_tac_du_doan": qt["du_doan"],
            "npz": npz.name, "npz_sha256": sha256_file(npz), "so_anh": n, "du_lieu": danh_tinh_nguon(danh_tinh),
            "adapter_sha256": sha256_file(Path(__file__)), "tao_luc": dt.datetime.now().isoformat(timespec="seconds")}
    (out / f"nguon_du_doan_{ten}.json").write_text(json.dumps(_json_safe(meta), ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")


def doc_nguon(nguon: Path) -> dict:
    """{ten: metadata} cua cac bo du doan trong mot thu muc luot. Luot cua ban dau (chua co nguon_du_doan_*.json): lay tu
    ket_qua.json (danh tinh du lieu lay tu dataset_da_xac_minh; sha256 NPZ khi do chua duoc ghi)."""
    tep = sorted(nguon.glob("nguon_du_doan_*.json"))
    if tep:
        return {m["mo_hinh"]: m for m in (json.loads(p.read_text(encoding="utf-8")) for p in tep)}
    cu = json.loads((nguon / "ket_qua.json").read_text(encoding="utf-8"))
    dl = danh_tinh_nguon(cu["dataset"]["dataset_da_xac_minh"])
    return {ten: {"mo_hinh": ten, "checkpoint": v["checkpoint"], "sha256": v["sha256"], "anh_xa_lop": v.get("anh_xa_lop"),
                  "quy_tac_du_doan": cu["quy_tac"]["du_doan"], "npz": f"du_doan_{ten}.npz", "npz_sha256": None, "du_lieu": dl,
                  "tu_ket_qua_json": True} for ten, v in cu["mo_hinh"].items()}


def ghi_ket_qua(out: Path, kq: dict, ket: dict) -> None:
    """ket_qua.json + 3 CSV: theo_lop (D3, lop ngoai tap = N/A), tong_hop (mAP (car, bus)), so lieu tho rieng."""
    (out / "ket_qua.json").write_text(json.dumps(_json_safe(kq), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    lop_rows, tong_rows, tho_rows = [], [], []
    for ten, d3 in ket.items():
        for lop in LOP_5:
            v = d3["theo_lop"][lop]
            if v == NA:
                lop_rows.append({"mo_hinh": ten, "lop": lop, "n_gt": NA, "AP50": NA, "AP75": NA, "AP50_95": NA,
                                 "precision_vh": NA, "recall_vh": NA})
                t = d3["so_lieu_tho_khong_phai_metric"]["theo_lop"][lop]
                tho_rows.append({"mo_hinh": ten, "lop": lop, "ghi_chu": "tho, khong phai metric", "n_det_conf_thap": t["n_det_conf_thap"],
                                 "TP_vh": t["van_hanh"]["TP"], "FP_vh": t["van_hanh"]["FP"], "precision_vh": t["van_hanh"]["precision"]})
            else:
                lop_rows.append({"mo_hinh": ten, "lop": lop, "n_gt": v["n_gt"], "AP50": v["AP50"], "AP75": v["AP75"],
                                 "AP50_95": v["AP50_95"], "precision_vh": v["van_hanh"]["precision"], "recall_vh": v["van_hanh"]["recall"]})
        m = d3[NHAN_MAP]
        tong_rows.append({"mo_hinh": ten, "chi_so": NHAN_MAP, "mAP50": m["mAP50"], "mAP75": m["mAP75"], "mAP50_95": m["mAP50_95"],
                          "precision_vh": m["precision_vh"], "recall_vh": m["recall_vh"], "n_anh": d3["n_anh"]})
    for name, rows in (("theo_lop.csv", lop_rows), ("tong_hop.csv", tong_rows), ("so_lieu_tho_khong_phai_metric.csv", tho_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(_json_safe(rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Danh gia detector 5 lop tren UA-DETRAC test theo D1-D5 (ham du doan / ghep / "
                                             "AP cua eval_detector_5lop)")
    for k in ("xml", "anh", "manifest", "fingerprint", "hash_anh", "loai_d4", "kich_thuoc"):
        ap.add_argument("--" + k.replace("_", "-"), default=MAC_DINH[k])
    ap.add_argument("--model", action="append", default=[], metavar="TEN=DUONG_DAN")
    ap.add_argument("--sha256", action="append", default=[], metavar="TEN=SHA256",
                    help="sha256 bat buoc cua checkpoint (lech -> exit 2, khong du doan)")
    ap.add_argument("--out")
    ap.add_argument("--conf-van-hanh", type=float, default=QUY_TAC["van_hanh"]["conf"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--chi-kiem", action="store_true", help="chi kiem du lieu + doc GT roi dung; khong nap model, khong ghi")
    ap.add_argument("--cham-lai", help="thu muc luot da chay: cham lai tu NPZ da luu (khong du doan); ghi vao --out moi")
    ap.add_argument("--gioi-han-anh", type=int, default=0, help="chi N anh dau (0 = tat ca) -- CHI cho smoke ky thuat")
    a = ap.parse_args(argv)
    try:
        W, H = (int(x) for x in a.kich_thuoc.lower().split("x"))
    except ValueError:
        print(f"[LOI] --kich-thuoc {a.kich_thuoc!r} phai dang WxH", file=sys.stderr)
        return 2
    out = p_abs(a.out) if a.out else None
    if not a.chi_kiem:
        if out is None:
            print("[LOI] can --out (thu muc moi)", file=sys.stderr)
            return 2
        if out.exists():
            print(f"[LOI] {out} da ton tai -- khong ghi de ket qua cu", file=sys.stderr)
            return 1
    mo_hinh = dict(s.partition("=")[::2] for s in a.model)
    sha_kb = dict(s.partition("=")[::2] for s in a.sha256)
    if not a.chi_kiem and not a.cham_lai:
        if not mo_hinh or set(mo_hinh) != set(sha_kb):
            print(f"[LOI] --model {sorted(mo_hinh)} va --sha256 {sorted(sha_kb)} phai co va cung tap ten", file=sys.stderr)
            return 2
        for ten, w in mo_hinh.items():
            s = sha256_file(p_abs(w))
            if s != sha_kb[ten]:
                print(f"[LOI CHECKPOINT] {ten}: {w} sha256 {s[:16]} khac khai bao {sha_kb[ten][:16]}", file=sys.stderr)
                return 2
    t0 = dt.datetime.now()
    try:
        danh_tinh, anh, rel_ds, gt = kiem_va_doc_gt(p_abs(a.xml), p_abs(a.anh), p_abs(a.manifest), p_abs(a.fingerprint),
                                                    p_abs(a.hash_anh), p_abs(a.loai_d4), (W, H))
    except (LoiDuLieu, FileNotFoundError, ValueError, KeyError) as e:
        print(f"[LOI DATASET] {e}", file=sys.stderr)
        return 2
    d = danh_tinh["dem"]
    print(f"[du lieu] {d['anh_danh_gia']} anh danh gia ({d['anh_loai_D4']} loai D4), {danh_tinh['so_video']} video, "
          f"GT car {d['gt_theo_lop']['car']} bus {d['gt_theo_lop']['bus']}, vung bo qua: ignored_region "
          f"{d['vung_bo_qua_ignored_region']}, van {d['vung_bo_qua_tu_van']}, others {d['vung_bo_qua_tu_others']}; "
          f"kiem {(dt.datetime.now() - t0).total_seconds():.0f} s", flush=True)
    if a.chi_kiem:
        return 0
    if a.gioi_han_anh:
        anh, rel_ds, gt = anh[:a.gioi_han_anh], rel_ds[:a.gioi_han_anh], gt[:a.gioi_han_anh]
    qt = quy_tac_hieu_luc(a.conf_van_hanh)
    conf_vh = qt["van_hanh"]["conf"]
    nguon, meta = None, {}
    if a.cham_lai:
        nguon = p_abs(a.cham_lai)
        try:
            luu = json.loads((nguon / "danh_sach_anh.json").read_text(encoding="utf-8"))
            meta = doc_nguon(nguon)
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"[LOI CHAM LAI] {e}", file=sys.stderr)
            return 2
        if luu["rel"] != rel_ds or luu["sha256"] != fpt.sha256_thu_tu(rel_ds):
            print("[LOI CHAM LAI] danh sach anh co thu tu da luu khac tap danh gia hien tai", file=sys.stderr)
            return 2
        hien_tai = danh_tinh_nguon(danh_tinh)
        for ten, m in meta.items():
            khac = sorted(k for k in DANH_TINH_NGUON if m["du_lieu"].get(k) != hien_tai.get(k))
            if khac:
                print(f"[LOI CHAM LAI] {ten}: du lieu hien tai khac du lieu luc du doan ({', '.join(khac)})", file=sys.stderr)
                return 2
            if m["quy_tac_du_doan"] != qt["du_doan"]:
                print(f"[LOI CHAM LAI] {ten}: cau hinh du doan da luu khac cau hinh hien tai", file=sys.stderr)
                return 2
            if m.get("npz_sha256") and sha256_file(nguon / m["npz"]) != m["npz_sha256"]:
                print(f"[LOI CHAM LAI] {ten}: {m['npz']} da doi sau khi luu", file=sys.stderr)
                return 2
        mo_hinh = {ten: m["checkpoint"] for ten, m in meta.items()}
        sha_kb = {ten: m["sha256"] for ten, m in meta.items()}
    out.mkdir(parents=True)
    (out / "danh_sach_anh.json").write_text(json.dumps({"rel": rel_ds, "sha256": fpt.sha256_thu_tu(rel_ds),
                                                        "cach_tinh": fpt.CACH_TINH["danh_sach_anh_co_thu_tu_sha256"]},
                                                       ensure_ascii=False) + "\n", encoding="utf-8")
    kq = {"tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "quy_tac": qt, "quy_tac_gt": QUY_TAC_GT,
          "dataset": {"ten": "ua_detrac_test", "n_anh": len(anh), "gioi_han_anh": a.gioi_han_anh, "dataset_da_xac_minh": danh_tinh},
          "lenh": [sys.executable, *(argv if argv is not None else sys.argv[1:])], "cham_lai_tu": str(nguon) if nguon else None,
          "mo_hinh": {}}
    ket = {}
    for ten, w in mo_hinh.items():
        t1 = dt.datetime.now()
        if nguon is not None:
            ten_anh, dd = doc_du_doan(nguon / meta[ten]["npz"], len(anh))
            if [str(x) for x in ten_anh] != [p.name for p in anh]:
                print(f"[LOI CHAM LAI] {ten}: ten anh trong NPZ khac tap danh gia", file=sys.stderr)
                return 2
            mt = meta[ten].get("anh_xa_lop")
        else:
            dd, mt = du_doan_ultralytics(str(p_abs(w)), anh, qt["du_doan"], a.batch, a.device)
            if len(dd) != len(anh):
                print(f"[LOI] {ten}: {len(dd)} ket qua du doan cho {len(anh)} anh", file=sys.stderr)
                return 3
            luu_du_doan(out / f"du_doan_{ten}.npz", anh, dd)
            ghi_nguon(out, ten, w, sha_kb[ten], mt, qt, danh_tinh, len(anh))   # TRUOC khi cham: cham loi van cham lai duoc
        r = danh_gia(gt, dd, conf_vh)
        if r.get("van_hanh") != qt["van_hanh"]:
            print(f"[LOI] {ten}: nguong van hanh cua scorer {r.get('van_hanh')} khac cau hinh hieu luc {qt['van_hanh']}",
                  file=sys.stderr)
            return 3
        try:
            d3 = chuan_hoa_d3(r)
        except LoiDuLieu as e:
            print(f"[LOI D3] {ten}: {e}", file=sys.stderr)
            return 3
        ket[ten] = d3
        kq["mo_hinh"][ten] = {"checkpoint": w, "sha256": sha_kb.get(ten), "anh_xa_lop": mt, "so_anh_da_du_doan": len(dd),
                              "thoi_gian_s": round((dt.datetime.now() - t1).total_seconds(), 1),
                              "chan_doan_tho": chan_doan(gt, dd, conf_vh), "ket_qua_D3": d3,
                              **({"nguon_du_doan": meta[ten]} if nguon is not None else {})}
        m = d3[NHAN_MAP]
        print(f"[{ten}] {NHAN_MAP}: mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}  car AP50 "
              f"{d3['theo_lop']['car']['AP50']:.3f}  bus AP50 {d3['theo_lop']['bus']['AP50']:.3f}", flush=True)
    ghi_ket_qua(out, kq, ket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
