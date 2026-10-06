# -*- coding: utf-8 -*-
"""Vong 43: KIEM tinh hop le + TONG HOP ket qua danh gia detector tren BDD val (CHI DOC ket qua; chi ghi vao --out).

    python handoff/logs/review43/tong_hop_danh_gia.py --out handoff/logs/review43/ket_qua [--tai-tinh]

Voi moi luot r43_<ten> (thu tu ke hoach khoa: yolov8n_coco -> yolov8n_ft -> yolo11n_ft):
  1. Tieu chi hop le muc 9 cua ke hoach (04_ke_hoach_danh_gia.md): exit 0 (tom_tat.json cua bo giam sat), 39.973 anh da du
     doan, kiem du lieu (sha256 nhan / manifest / bang hash, 200 video, noi dung 39.973/39.973 khop), sha256 checkpoint +
     loai anh xa lop dung muc 2. Them: lenh dung mau khoa (khong tham so phu), quy tac du doan / van hanh dung muc 5,
     `quy_tac` / `quy_tac_gt` / `dataset_da_xac_minh` TRUNG NHAU giua cac luot, theo_lop.csv == ket_qua.json (so thuc
     doc lai bang float, so sanh bang ==), sha256 ma danh gia hien tai == gia tri ghi truoc khi chay (05_*.log).
  2. --tai-tinh: doc lai GT (kiem_va_doc_gt, KHONG bam lai noi dung anh -- cac luot da bam) + du_doan_<ten>.npz, goi lai
     `danh_gia` / `chan_doan` -> phai TRUNG KHOP (==) moi so trong ket_qua.json. Chung minh NPZ du de tai tinh metric.
Ghi vao --out:
  bang_theo_lop.csv   moi (model, lop): n_gt, n_anh_co_lop, n_det (moi detection sau NMS, conf >= 0,001), AP50, AP75,
                      AP50_95, chenh lech so voi baseline (DIEM AP = x100), TP/FP/FN/P/R/F1 tai conf >= 0,25 + IoU 0,5
  bang_macro.csv      moi model: mAP50 / mAP50_95 tren cac lop co GT (+ danh sach lop), chenh lech, P/R macro, chan doan
  bang_ket_qua.md     bang Markdown (dau phay thap phan) cho phu luc
  kiem_hop_le.json    ket qua tung phep kiem
  <ten>/              ket_qua.json, theo_lop.csv (chep nguyen byte), giam_sat/{tom_tat.json, trang_thai.json, log_loc.txt}
  tep_lon_khong_ban_giao.csv   du_doan_<ten>.npz, log.txt day du, ram_mau_tho.csv: duong dan, bytes, sha256
log_loc.txt = log.txt cua bo giam sat BO cac dong canh bao lap "'half' is deprecated" cua ultralytics (dem so dong bo).
Thoat 0 khi moi phep kiem dat; 1 khi co phep kiem khong dat."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

GOC = Path(os.environ.get("GOC_REPO") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(GOC / "src"))

KQ = GOC / "results/phase2/detector_eval/bdd_val_cocovid"
GS = GOC / "results/phase2/detector_eval/giam_sat"
BASELINE = "yolov8n_coco"
# muc 2 / 3 / 5 cua ke hoach khoa
MO_HINH = {
    "yolov8n_coco": ("models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36", "coco80"),
    "yolov8n_ft": ("models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt",
                   "5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a", "lop5"),
    "yolo11n_ft": ("models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt",
                   "d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923", "lop5"),
}
DU_LIEU = {"n_anh": 39973, "so_video": 200,
           "nhan_sha256": "074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d",
           "manifest_sha256": "2071c15bd300e5dfff156fd708393b7018b074777ec4ef83da236409eb34c3ba",
           "bang_hash_sha256": "60fea2a9ce623450c1526953b7e427016ed917cfd702a7c6275525fca3582d20"}
QUY_TAC_KHOA = {"du_doan": {"imgsz": 640, "conf": 0.001, "iou_nms": 0.7, "max_det": 300, "agnostic_nms": False,
                            "half": False},
                "van_hanh": {"conf": 0.25, "iou": 0.5}}
# sha256 ma danh gia ghi NGAY TRUOC luot dau (05_truoc_khi_chay_yolov8n_coco.log)
MA_KHOA = {"src/danh_gia_detector_bdd_val.py": "d1e2fd22a01a5fa117e64d909b13434a1f5226b27b5c0ed0f768ddfb6cc84ba1",
           "src/eval_detector_5lop.py": "30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f",
           "src/bdd_yolo_export.py": "1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493",
           "src/detector_classes.py": "7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e",
           "src/mot_datasets/bdd100k_coco.py": "5028ffb98ddff6ba423bef14ff9331fd02f8e40b1055438c88224182914f8363",
           "src/chay_giam_sat.py": "2fd153bf6016f7e217ebfea407378549e54b9d62806d70f99f550555b4b8f842"}
CANH_BAO_LAP = "'half' is deprecated"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(GOC)).replace("\\", "/")
    except ValueError:
        return str(p)


def doc_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


class Kiem:
    def __init__(self):
        self.ds = []

    def __call__(self, ten: str, dat: bool, chi_tiet=None):
        self.ds.append({"phep_kiem": ten, "dat": bool(dat), "chi_tiet": chi_tiet})
        print(f"  [{'DAT' if dat else 'KHONG DAT'}] {ten}" + (f" -- {chi_tiet}" if chi_tiet is not None else ""))
        return dat


def lenh_mong_doi(ten: str) -> list[str]:
    w, s, _ = MO_HINH[ten]
    return ["-u", "src\\danh_gia_detector_bdd_val.py", "--model", f"{ten}={w}", "--sha256", f"{ten}={s}",
            "--out", f"results\\phase2\\detector_eval\\bdd_val_cocovid\\r43_{ten}"]


def kiem_luot(ten: str, k: Kiem) -> dict:
    d, g = KQ / f"r43_{ten}", GS / f"r43_{ten}"
    print(f"== {ten}: {rel(d)}")
    kq, tt = doc_json(d / "ket_qua.json"), doc_json(g / "tom_tat.json")
    m = kq["mo_hinh"][ten]
    w, s, loai = MO_HINH[ten]
    k(f"{ten}: exit code that = 0 (tom_tat.json)", tt["exit_code"] == 0 and not tt["ctrl_c"],
      {"exit_code": tt["exit_code"], "ctrl_c": tt["ctrl_c"], "thoi_gian_s": tt["thoi_gian_s"]})
    k(f"{ten}: lenh dung mau khoa (khong tham so phu)", tt["lenh"][1:] == lenh_mong_doi(ten) and
      kq["lenh"][1:] == [x for x in lenh_mong_doi(ten) if x != "-u"], tt["lenh"][1:])
    k(f"{ten}: chi mot model trong ket_qua.json", list(kq["mo_hinh"]) == [ten], list(kq["mo_hinh"]))
    k(f"{ten}: so anh du kien = da du doan = {DU_LIEU['n_anh']}",
      kq["dataset"]["n_anh"] == m["so_anh_da_du_doan"] == m["n_anh"] == DU_LIEU["n_anh"]
      and kq["dataset"]["gioi_han_anh"] == 0,
      {"n_anh": kq["dataset"]["n_anh"], "so_anh_da_du_doan": m["so_anh_da_du_doan"],
       "gioi_han_anh": kq["dataset"]["gioi_han_anh"]})
    ds = kq["dataset"]["dataset_da_xac_minh"]
    dc = ds["doi_chieu_hash_anh"]
    k(f"{ten}: kiem du lieu (sha256 nhan / manifest / bang hash, {DU_LIEU['so_video']} video, noi dung anh khop het)",
      ds["nhan_sha256"] == DU_LIEU["nhan_sha256"] and ds["manifest_sha256"] == DU_LIEU["manifest_sha256"]
      and dc["file_sha256"] == DU_LIEU["bang_hash_sha256"] and ds["so_video"] == DU_LIEU["so_video"]
      and dc["so_anh_bang_hash"] == dc["khop"] == DU_LIEU["n_anh"] and not dc["lech"]
      and ds["dem"]["anh"] == DU_LIEU["n_anh"],
      {"nhan": ds["nhan_sha256"][:12], "video": ds["so_video"], "khop_noi_dung": f"{dc['khop']}/{dc['so_anh_bang_hash']}",
       "lech": len(dc["lech"])})
    k(f"{ten}: sha256 checkpoint + duong dan + anh xa lop dung muc 2",
      m["sha256"] == s and m["checkpoint"] == w and m["anh_xa_lop"]["loai_checkpoint"] == loai
      and m["anh_xa_lop"]["lop5"] == ["car", "truck", "bus", "motorcycle", "bicycle"],
      {"sha256": m["sha256"][:12], "loai": m["anh_xa_lop"]["loai_checkpoint"],
       "model_id_sang_lop5": m["anh_xa_lop"]["model_id_sang_lop5"]})
    dd_kq = kq["quy_tac"]["du_doan"]                         # + khoa mo ta 'loc_lop' (chuoi, khong phai tham so so)
    k(f"{ten}: quy tac du doan / van hanh dung muc 5",
      {x: dd_kq.get(x) for x in QUY_TAC_KHOA["du_doan"]} == QUY_TAC_KHOA["du_doan"]
      and set(dd_kq) - set(QUY_TAC_KHOA["du_doan"]) == {"loc_lop"}
      and kq["quy_tac"]["van_hanh"] == QUY_TAC_KHOA["van_hanh"] and m["van_hanh"] == QUY_TAC_KHOA["van_hanh"],
      {x: v for x, v in dd_kq.items() if x != "loc_lop"})
    with open(d / "theo_lop.csv", newline="", encoding="utf-8") as f:
        hang = list(csv.DictReader(f))
    khop_csv = len(hang) == 5 and all(
        r["mo_hinh"] == ten and int(r["n_gt"]) == m["theo_lop"][r["lop"]]["n_gt"]
        and float(r["AP50"]) == m["theo_lop"][r["lop"]]["AP50"]
        and float(r["AP50_95"]) == m["theo_lop"][r["lop"]]["AP50_95"]
        and float(r["precision_vh"]) == m["theo_lop"][r["lop"]]["van_hanh"]["precision"]
        and float(r["recall_vh"]) == m["theo_lop"][r["lop"]]["van_hanh"]["recall"] for r in hang)
    k(f"{ten}: theo_lop.csv == ket_qua.json (5 lop, so thuc ==)", khop_csv, len(hang))
    return {"kq": kq, "tt": tt, "trang_thai": doc_json(g / "trang_thai.json")}


def tai_tinh(luot: dict, k: Kiem) -> None:
    import numpy as np
    from danh_gia_detector_bdd_val import MAC_DINH, chan_doan, kiem_va_doc_gt, p_abs
    from eval_detector_5lop import danh_gia
    t0 = dt.datetime.now()
    danh_tinh, anh, gt = kiem_va_doc_gt(p_abs(MAC_DINH["nhan"]), p_abs(MAC_DINH["anh"]), p_abs(MAC_DINH["manifest"]),
                                        None, kiem_noi_dung=False)
    print(f"  doc lai GT: {len(anh)} anh ({(dt.datetime.now() - t0).total_seconds():.0f} s, khong bam lai noi dung)")
    ten_anh = [a.name for a in anh]
    for ten, v in luot.items():
        t1 = dt.datetime.now()
        with np.load(KQ / f"r43_{ten}" / f"du_doan_{ten}.npz") as z:
            # nap MOI mang MOT lan: moi lan truy cap z[...] cua NpzFile giai nen lai ca mang (lan 1 cham vi loi nay)
            ten_npz, idx, xyxy, conf, cls = z["anh"], z["idx"], z["xyxy"], z["conf"], z["cls"]
        if list(ten_npz) != ten_anh or len(set(ten_anh)) != len(ten_anh):
            k(f"{ten}: tai tinh tu NPZ", False, "danh sach anh trong NPZ khac thu tu GT / ten trung")
            continue
        bien = np.searchsorted(idx, np.arange(len(anh) + 1))
        dd = [{"xyxy": xyxy[bien[i]:bien[i + 1]].astype(float), "conf": conf[bien[i]:bien[i + 1]].astype(float),
               "cls": cls[bien[i]:bien[i + 1]].astype(int)} for i in range(len(anh))]
        m = v["kq"]["mo_hinh"][ten]
        so_det = np.diff(bien)
        v["bo_sung"] = {"so_detection_npz": int(len(idx)), "so_det_tb_moi_anh": round(float(so_det.mean()), 2),
                        "so_det_max_moi_anh": int(so_det.max()),
                        f"so_anh_cham_max_det_{QUY_TAC_KHOA['du_doan']['max_det']}":
                            int((so_det >= QUY_TAC_KHOA["du_doan"]["max_det"]).sum())}
        r = danh_gia(gt, dd, m["van_hanh"]["conf"], m["van_hanh"]["iou"])
        cd = chan_doan(gt, dd, m["van_hanh"]["conf"])
        bang = json.loads(json.dumps(r))                     # cung duong JSON nhu ket_qua.json (NaN -> khong co o day)
        khop = all(bang["theo_lop"][c] == m["theo_lop"][c] for c in m["theo_lop"]) and bang["macro"] == m["macro"] \
            and cd == m["chan_doan"] and bool(np.all(np.diff(idx) >= 0))
        k(f"{ten}: tai tinh tu du_doan_{ten}.npz == ket_qua.json (moi so theo lop, macro, chan doan)", khop,
          {"so_detection": int(len(idx)), "thoi_gian_s": round((dt.datetime.now() - t1).total_seconds(), 1)})


def loc_log(src: Path, dst: Path) -> dict:
    giu = bo = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for dong in fi:
            if CANH_BAO_LAP in dong.decode("utf-8", "replace"):
                bo += 1
                continue
            fo.write(dong)
            giu += 1
    with open(dst, "ab") as fo:
        fo.write(f"# [log_loc] giu {giu} dong; bo {bo} dong chua \"{CANH_BAO_LAP}\" (canh bao lap cua ultralytics, moi "
                 f"lan goi predict mot batch). Ban day du: {rel(src)} (sha256 trong tep_lon_khong_ban_giao.csv)\n"
                 .encode("utf-8"))
    return {"giu": giu, "bo": bo}


def diem(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def chenh(d) -> str:
    """d = chenh lech DA tinh bang diem AP (x100); '' -> khong ap dung"""
    return "—" if d == "" else f"{d:+.1f}".replace(".", ",")


def so(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mo-hinh", nargs="+", default=list(MO_HINH))
    ap.add_argument("--tai-tinh", action="store_true", help="tai tinh metric tu NPZ (doc GT ~2 phut, ~1-2 GiB RAM)")
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = GOC / out
    out.mkdir(parents=True, exist_ok=True)
    k = Kiem()
    print("== ma danh gia: sha256 hien tai so voi luc truoc khi chay")
    for f, s in MA_KHOA.items():
        k(f"sha256 {f} == gia tri 05_*.log", sha(GOC / f) == s, sha(GOC / f)[:12])
    luot = {ten: kiem_luot(ten, k) for ten in a.mo_hinh}
    if len(luot) > 1:
        dau = luot[a.mo_hinh[0]]["kq"]
        for ten, v in list(luot.items())[1:]:
            for khoa in ("quy_tac", "quy_tac_gt"):
                k(f"{ten}: {khoa} trung luot {a.mo_hinh[0]}", v["kq"][khoa] == dau[khoa])
            k(f"{ten}: dataset_da_xac_minh trung luot {a.mo_hinh[0]} (moi dem, moi sha256)",
              v["kq"]["dataset"]["dataset_da_xac_minh"] == dau["dataset"]["dataset_da_xac_minh"])
    if a.tai_tinh:
        print("== tai tinh tu NPZ")
        tai_tinh(luot, k)

    # ---------------------------------------------------------------- bang
    goc = luot[BASELINE]["kq"]["mo_hinh"][BASELINE] if BASELINE in luot else None
    hang_lop, hang_macro = [], []
    for ten, v in luot.items():
        m = v["kq"]["mo_hinh"][ten]
        so_goc = goc is not None and ten != BASELINE             # baseline: o chenh lech de trong, khong ghi 0
        for lop, r in m["theo_lop"].items():
            vh = r["van_hanh"]
            b = goc["theo_lop"][lop] if so_goc else None
            hang_lop.append({"mo_hinh": ten, "lop": lop, "n_gt": r["n_gt"], "n_anh_co_lop": r["n_anh_co_lop"],
                             "n_det_conf_ge_0.001": r["n_det_conf_thap"], "AP50": r["AP50"], "AP75": r["AP75"],
                             "AP50_95": r["AP50_95"],
                             "chenh_AP50_diem_vs_baseline": (r["AP50"] - b["AP50"]) * 100 if b else "",
                             "chenh_AP50_95_diem_vs_baseline": (r["AP50_95"] - b["AP50_95"]) * 100 if b else "",
                             "TP_vh": vh["TP"], "FP_vh": vh["FP"], "FN_vh": vh["FN"], "precision_vh": vh["precision"],
                             "recall_vh": vh["recall"], "f1_vh": vh["f1"]})
        mc = m["macro"]
        hang_macro.append({"mo_hinh": ten, "checkpoint": m["checkpoint"], "sha256": m["sha256"],
                           "so_lop_tinh_mAP": mc["so_lop_co_gt"],
                           "lop_tinh_mAP": " ".join(c for c in m["theo_lop"] if c not in mc["lop_khong_co_gt"]),
                           "mAP50": mc["mAP50"], "mAP50_95": mc["mAP50_95"],
                           "chenh_mAP50_diem_vs_baseline": (mc["mAP50"] - goc["macro"]["mAP50"]) * 100 if so_goc else "",
                           "chenh_mAP50_95_diem_vs_baseline":
                               (mc["mAP50_95"] - goc["macro"]["mAP50_95"]) * 100 if so_goc else "",
                           "precision_vh_macro": mc["precision_vh"], "recall_vh_macro": mc["recall_vh"],
                           "so_anh": m["n_anh"], "so_anh_da_du_doan": m["so_anh_da_du_doan"],
                           "so_anh_khong_gt_5_lop": m["chan_doan"]["so_anh_khong_gt_5_lop"],
                           "so_det_conf_ge_0.25_tren_anh_khong_gt": m["chan_doan"]["so_det_conf_ge_0.25_tren_anh_khong_gt"],
                           "exit_code": v["tt"]["exit_code"], "thoi_gian_luot_s": v["tt"]["thoi_gian_s"],
                           "bat_dau": v["tt"]["bat_dau"], "ket_thuc": v["tt"]["ket_thuc"],
                           "rss_tien_trinh_max_mib": v["tt"]["ram"]["rss_tong_mib"]["max"],
                           "ram_may_trong_min_mib_bo_giam_sat": v["tt"]["ram"]["ram_he_thong_con_trong_mib"]["min"]})
    for ten_f, hang in (("bang_theo_lop.csv", hang_lop), ("bang_macro.csv", hang_macro)):
        with open(out / ten_f, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hang[0]))
            w.writeheader()
            w.writerows(hang)

    L = ["<!-- sinh tu dong boi tong_hop_danh_gia.py tu ket_qua.json; khong sua tay -->", "",
         "**AP theo lớp** — BDD100K MOT val (COCO-video), 39.973 ảnh, 200 video. AP nội suy 101 điểm; AP50–95 = trung bình "
         "10 ngưỡng IoU 0,50:0,05:0,95. Chênh lệch Δ = model − baseline `yolov8n_coco`, tính bằng **điểm AP** (×100).", "",
         "| Lớp | Số GT | Số ảnh có lớp | Model | AP50 | AP50–95 | ΔAP50 | ΔAP50–95 |", "|:--|--:|--:|:--|--:|--:|--:|--:|"]
    for lop in ("car", "truck", "bus", "motorcycle", "bicycle"):
        for h in (x for x in hang_lop if x["lop"] == lop):
            la_goc = h["mo_hinh"] == BASELINE
            L.append(f"| {lop} | {so(h['n_gt'])} | {so(h['n_anh_co_lop'])} | `{h['mo_hinh']}` | {diem(h['AP50'])} | "
                     f"{diem(h['AP50_95'])} | {chenh('' if la_goc else h['chenh_AP50_diem_vs_baseline'])} | "
                     f"{chenh('' if la_goc else h['chenh_AP50_95_diem_vs_baseline'])} |")
    L += ["", "**mAP (macro, 5 lớp: car, truck, bus, motorcycle, bicycle — mọi lớp đều có GT)**", "",
          "| Model | mAP50 | mAP50–95 | ΔmAP50 | ΔmAP50–95 | P macro @0,25 | R macro @0,25 |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for h in hang_macro:
        la_goc = h["mo_hinh"] == BASELINE
        L.append(f"| `{h['mo_hinh']}` | {diem(h['mAP50'])} | {diem(h['mAP50_95'])} | "
                 f"{chenh('' if la_goc else h['chenh_mAP50_diem_vs_baseline'])} | "
                 f"{chenh('' if la_goc else h['chenh_mAP50_95_diem_vs_baseline'])} | "
                 f"{diem(h['precision_vh_macro'])} | {diem(h['recall_vh_macro'])} |")
    L += ["", "**Điểm vận hành** conf ≥ 0,25, IoU 0,5 (ngưỡng tracker c25). Detection không ghép mà IoA với vùng bỏ qua ≥ "
          "ngưỡng thì không tính FP.", "",
          "| Lớp | Model | TP | FP | FN | Precision | Recall | F1 |", "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for lop in ("car", "truck", "bus", "motorcycle", "bicycle"):
        for h in (x for x in hang_lop if x["lop"] == lop):
            L.append(f"| {lop} | `{h['mo_hinh']}` | {so(h['TP_vh'])} | {so(h['FP_vh'])} | {so(h['FN_vh'])} | "
                     f"{diem(h['precision_vh'])} | {diem(h['recall_vh'])} | {diem(h['f1_vh'])} |")
    L += ["", "**Chẩn đoán mô tả** (không đổi metric)", "",
          "| Model | Ảnh đã dự đoán | Ảnh không có GT 5 lớp | Detection conf ≥ 0,25 trên các ảnh đó |", "|:--|--:|--:|--:|"]
    for h in hang_macro:
        L.append(f"| `{h['mo_hinh']}` | {so(h['so_anh_da_du_doan'])}/{so(h['so_anh'])} | {so(h['so_anh_khong_gt_5_lop'])} | "
                 f"{so(h['so_det_conf_ge_0.25_tren_anh_khong_gt'])} |")
    if all("bo_sung" in v for v in luot.values()):
        md_max = QUY_TAC_KHOA["du_doan"]["max_det"]
        L += ["", f"**Chẩn đoán bổ sung** (đếm từ `du_doan_<ten>.npz` lúc tái tính; KHÔNG có trong kế hoạch khóa, chỉ mô "
              f"tả, không đổi metric): số detection sau NMS mỗi ảnh (conf ≥ 0,001, trần `max_det` {md_max}).", "",
              f"| Model | Tổng detection | Trung bình / ảnh | Tối đa / ảnh | Số ảnh chạm trần {md_max} |", "|:--|--:|--:|--:|--:|"]
        for ten, v in luot.items():
            bs = v["bo_sung"]
            L.append(f"| `{ten}` | {so(bs['so_detection_npz'])} | {str(bs['so_det_tb_moi_anh']).replace('.', ',')} | "
                     f"{bs['so_det_max_moi_anh']} | {so(bs[f'so_anh_cham_max_det_{md_max}'])} |")
    L += ["", "**Lượt chạy** (bộ giám sát `chay_giam_sat.py`; RSS = working set của tiến trình đánh giá; RAM máy trống = "
          "`psutil.virtual_memory().available` theo bộ giám sát, mẫu 1 s)", "",
          "| Model | Bắt đầu | Kết thúc | Thời gian (s) | Exit | RSS tối đa (MiB) | RAM máy trống thấp nhất (MiB) |",
          "|:--|:--|:--|--:|--:|--:|--:|"]
    for h in hang_macro:
        L.append(f"| `{h['mo_hinh']}` | {h['bat_dau'][:19].replace('T', ' ')} | {h['ket_thuc'][:19].replace('T', ' ')} | "
                 f"{str(h['thoi_gian_luot_s']).replace('.', ',')} | {h['exit_code']} | "
                 f"{str(h['rss_tien_trinh_max_mib']).replace('.', ',')} | "
                 f"{str(h['ram_may_trong_min_mib_bo_giam_sat']).replace('.', ',')} |")
    md = "\n".join(L) + "\n"
    (out / "bang_ket_qua.md").write_text(md, encoding="utf-8")

    # ---------------------------------------------------------------- chep bang chung nho + khai bao file lon
    lon, log_info = [], {}
    for ten in luot:
        d, g = KQ / f"r43_{ten}", GS / f"r43_{ten}"
        (out / ten / "giam_sat").mkdir(parents=True, exist_ok=True)
        for n in ("ket_qua.json", "theo_lop.csv"):
            shutil.copy2(d / n, out / ten / n)
        for n in ("tom_tat.json", "trang_thai.json"):
            shutil.copy2(g / n, out / ten / "giam_sat" / n)
        log_info[ten] = loc_log(g / "log.txt", out / ten / "giam_sat" / "log_loc.txt")
        lon += [(d / f"du_doan_{ten}.npz", "du doan day du (conf >= 0,001 sau NMS) -- du de tai tinh metric"),
                (g / "log.txt", "log day du cua bo giam sat (ban loc log_loc.txt nam trong goi)"),
                (g / "ram_mau_tho.csv", "mau RAM 1 s cua bo giam sat (tom tat trong tom_tat.json)")]
    with open(out / "tep_lon_khong_ban_giao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["duong_dan", "bytes", "sha256", "ghi_chu"])
        for p, gc in lon:
            w.writerow([rel(p), p.stat().st_size, sha(p), gc])
    tong = {"tao_luc": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "mo_hinh": list(luot),
            "tai_tinh_tu_npz": a.tai_tinh, "log_loc": log_info,
            "chan_doan_bo_sung_tu_npz": {ten: v.get("bo_sung") for ten, v in luot.items()}, "phep_kiem": k.ds,
            "dat_tat_ca": all(x["dat"] for x in k.ds)}
    (out / "kiem_hop_le.json").write_text(json.dumps(tong, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_dat = sum(x["dat"] for x in k.ds)
    print(f"KET LUAN: {n_dat}/{len(k.ds)} phep kiem dat -> {'DAT' if tong['dat_tat_ca'] else 'KHONG DAT'}")
    return 0 if tong["dat_tat_ca"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
