"""R63 G4: AP 5 lop theo TUNG MIEN dev (BDD dev / UA-DETRAC dev12) tu du doan da luu cua eval_detector_5lop
(du_doan_<ten>.npz trong hai thu muc ket qua: doi chung 640x640 va ung vien 384x640, cung tap anh dev mixed5).
Cham bang DUNG eval_detector_5lop (ghep_anh_lop + ap_coco); bootstrap ghep cap theo VIDEO cho hieu mAP50 / mAP50-95
(ung vien - doi chung), so lan va seed KHOA truoc khi chay (5.000, seed 0), CI 95 % percentile.

LOI BAN 1 (01/11, GPT phat hien; bang chung giu o g4/ap_dev_theo_mien.json + g4/06_ap_theo_mien.log): ten video lay
bang ten.rsplit('-', 1) -> anh UA-DETRAC 'MVI_20052_img00010.jpg' khong co '-' nen moi anh thanh mot "video"
(1.631 cum thay vi 12) -> CI dev12 SAI. Ban 2: anh -> video lay tu MANIFEST cua dataset
(data/yolo_mixed5_v1/manifest/anh.csv: split, video, anh_ra, nguon), kiem DUNG danh sach video moi mien (UA-DETRAC =
results/phase2/manifests/detrac_dev.json; BDD = dev cua split_da_khoa.json tru MVI_*).

Tang toc (khong doi ket qua): ghep (anh, lop) MOT LAN, roi noi san theo (video, lop) theo thu tu anh trong video; moi
mau bootstrap noi cac khoi video theo thu tu mau -> CUNG mang nhu noi theo anh (kiem bang test). Diem toan tap kiem
bang ev.danh_gia (lech > 1e-12 -> loi). Lop khong co GT trong mau -> khong vao macro (nhu ev.danh_gia).

    python handoff/logs/review63/cong_cu/g4_ap_theo_mien.py --doi-chung <dir 640x640> --ung-vien <dir 384x640> \
        --ten yolov8n_coco --ra handoff/logs/review63/g4/ap_dev_theo_mien_v2.json --n-boot 5000
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))
import eval_detector_5lop as ev  # noqa: E402

MANIFEST_ANH = GOC / "data/yolo_mixed5_v1/manifest/anh.csv"
SPLIT_KHOA = GOC / "data/yolo_mixed5_v1/manifest/split_da_khoa.json"
DEV12 = GOC / "results/phase2/manifests/detrac_dev.json"


def doc_du_doan(p: Path, n: int) -> tuple[list[str], list[dict]]:
    z = np.load(p)
    anh = [str(x) for x in z["anh"]]
    if len(anh) != n:
        raise ValueError(f"{p}: {len(anh)} anh != {n}")
    idx = z["idx"]
    dd = []
    for i in range(n):
        m = idx == i
        dd.append({"xyxy": z["xyxy"][m].astype(float), "conf": z["conf"][m].astype(float), "cls": z["cls"][m].astype(int)})
    return anh, dd


def anh_sang_video(manifest: Path = MANIFEST_ANH, split: str = "dev") -> dict[str, tuple[str, str]]:
    """ten anh (basename cua anh_ra) -> (video, nguon) theo manifest; ten trung -> loi."""
    out = {}
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            ten = Path(r["anh_ra"]).name
            if ten in out:
                raise ValueError(f"manifest: anh trung ten {ten}")
            out[ten] = (r["video"], r["nguon"])
    return out


def ghep_truoc(gt: list[dict], dd: list[dict]) -> list[list[tuple]]:
    """[anh][lop] = (conf da xep, tp[T,n], bo_qua[T,n], n_gt) bang dung ev.ghep_anh_lop (doc lap giua cac anh)."""
    out = []
    for g, p in zip(gt, dd):
        hang = []
        for c in range(len(ev.LOP_5)):
            mg, md = g["cls"] == c, p["cls"] == c
            o, tp, bq = ev.ghep_anh_lop(p["xyxy"][md], p["conf"][md], g["xyxy"][mg], g["ign"])
            hang.append((p["conf"][md][o], tp, bq, int(mg.sum())))
        out.append(hang)
    return out


def khoi_video(gh: list[list[tuple]], theo_v: dict[str, list[int]]) -> dict[str, list[tuple]]:
    """Noi san theo (video, lop), giu thu tu anh trong video."""
    T = len(ev.NGUONG_IOU)
    out = {}
    for v, ids in theo_v.items():
        hang = []
        for c in range(len(ev.LOP_5)):
            hang.append((np.concatenate([gh[i][c][0] for i in ids]) if ids else np.zeros(0),
                         np.concatenate([gh[i][c][1] for i in ids], axis=1) if ids else np.zeros((T, 0), bool),
                         np.concatenate([gh[i][c][2] for i in ids], axis=1) if ids else np.zeros((T, 0), bool),
                         sum(gh[i][c][3] for i in ids)))
        out[v] = hang
    return out


def map_tu_khoi(kh: dict[str, list[tuple]], mau: list[str]) -> tuple[float, float]:
    """mAP50, mAP50-95 macro tren cac lop co GT trong mau video `mau` (thu tu, lap lai giu nguyen)."""
    a50, a95 = [], []
    for c in range(len(ev.LOP_5)):
        n_gt = sum(kh[v][c][3] for v in mau)
        if n_gt == 0:
            continue
        conf = np.concatenate([kh[v][c][0] for v in mau])
        tp = np.concatenate([kh[v][c][1] for v in mau], axis=1)
        bq = np.concatenate([kh[v][c][2] for v in mau], axis=1)
        aps = [ev.ap_coco(conf, tp[t], bq[t], n_gt) for t in range(len(ev.NGUONG_IOU))]
        a50.append(aps[0]); a95.append(float(np.mean(aps)))
    return float(np.mean(a50)), float(np.mean(a95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(GOC / "data/yolo_mixed5_v1/data.yaml"))
    ap.add_argument("--doi-chung", required=True)
    ap.add_argument("--ung-vien", required=True)
    ap.add_argument("--ten", default="yolov8n_coco")
    ap.add_argument("--ra", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if Path(a.ra).exists():
        print("[LOI] da ton tai", a.ra); return 2
    _, anh, gt = ev.chuan_bi_danh_gia(Path(a.data), "val")
    ten_anh = [p.name for p in anh]
    a_anh, dd_a = doc_du_doan(Path(a.doi_chung) / f"du_doan_{a.ten}.npz", len(anh))
    b_anh, dd_b = doc_du_doan(Path(a.ung_vien) / f"du_doan_{a.ten}.npz", len(anh))
    if a_anh != ten_anh or b_anh != ten_anh:
        print("[LOI] thu tu anh du doan khac tap dev"); return 2
    a2v = anh_sang_video()
    thieu = [t for t in ten_anh if t not in a2v]
    if thieu or len(a2v) != len(ten_anh):
        print("[LOI] manifest khong khop tap anh dev:", len(thieu), "thieu,", len(a2v), "dong manifest"); return 2
    ky_vong = {"detrac": sorted(json.loads(DEV12.read_text(encoding="utf-8"))["videos"]),
               "bdd": sorted(v for v in json.loads(SPLIT_KHOA.read_text(encoding="utf-8"))["dev"] if not v.startswith("MVI_"))}
    mien = {"detrac_dev12": "detrac", "bdd_dev": "bdd"}
    gh_a, gh_b = ghep_truoc(gt, dd_a), ghep_truoc(gt, dd_b)
    rng = np.random.default_rng(a.seed)
    out = {"phien_ban": 2, "doi_chung": a.doi_chung, "ung_vien": a.ung_vien, "n_boot": a.n_boot, "seed": a.seed,
           "anh_sang_video": f"{MANIFEST_ANH.relative_to(GOC)} (split dev: anh_ra -> video, nguon)", "mien": {},
           "phuong_phap": "ghep (anh, lop) mot lan bang ev.ghep_anh_lop; noi san theo (video, lop); moi mau bootstrap "
                          "theo video noi khoi va goi ev.ap_coco; diem toan tap kiem bang ev.danh_gia (<= 1e-12); lop "
                          "khong co GT trong mau khong vao macro"}
    for ten_m, nguon in mien.items():
        ids = [i for i, t in enumerate(ten_anh) if a2v[t][1] == nguon]
        vids = sorted({a2v[ten_anh[i]][0] for i in ids})
        if vids != ky_vong[nguon]:
            print("[LOI] danh sach video", ten_m, "khac ky vong:", len(vids), "vs", len(ky_vong[nguon])); return 2
        theo_v = {v: [i for i in ids if a2v[ten_anh[i]][0] == v] for v in vids}
        ra_ = ev.danh_gia([gt[i] for i in ids], [dd_a[i] for i in ids])
        rb_ = ev.danh_gia([gt[i] for i in ids], [dd_b[i] for i in ids])
        a50, a5095 = ra_["macro"]["mAP50"], ra_["macro"]["mAP50_95"]
        b50, b5095 = rb_["macro"]["mAP50"], rb_["macro"]["mAP50_95"]
        kh_a, kh_b = khoi_video(gh_a, theo_v), khoi_video(gh_b, theo_v)
        kiem = map_tu_khoi(kh_a, vids) + map_tu_khoi(kh_b, vids)
        if max(abs(x - y) for x, y in zip(kiem, (a50, a5095, b50, b5095))) > 1e-12:
            print("[LOI] tinh nhanh lech danh_gia", ten_m, kiem, (a50, a5095, b50, b5095)); return 3
        t0 = time.time()
        boots = np.empty((a.n_boot, 2))
        for k in range(a.n_boot):
            if k in (1, 10) or k % 500 == 0:
                print("[%s] boot %d/%d, %.0f s" % (ten_m, k, a.n_boot, time.time() - t0), flush=True)
            mau = [vids[j] for j in rng.integers(0, len(vids), size=len(vids))]
            x50, x95 = map_tu_khoi(kh_a, mau); y50, y95 = map_tu_khoi(kh_b, mau)
            boots[k] = (y50 - x50, y95 - x95)
        hop_le = np.isfinite(boots).all(1)
        bo = boots[hop_le]
        out["mien"][ten_m] = {
            "n_anh": len(ids), "n_video": len(vids), "n_boot_hop_le": int(hop_le.sum()),
            "doi_chung": {"mAP50": a50, "mAP50_95": a5095, "theo_lop": {k: {"AP50": v["AP50"], "AP50_95": v["AP50_95"],
                          "n_gt": v["n_gt"]} for k, v in ra_["theo_lop"].items()}},
            "ung_vien": {"mAP50": b50, "mAP50_95": b5095, "theo_lop": {k: {"AP50": v["AP50"], "AP50_95": v["AP50_95"],
                         "n_gt": v["n_gt"]} for k, v in rb_["theo_lop"].items()}},
            "hieu_mAP50": b50 - a50, "ci95_mAP50": [float(x) for x in np.percentile(bo[:, 0], [2.5, 97.5])],
            "hieu_mAP50_95": b5095 - a5095, "ci95_mAP50_95": [float(x) for x in np.percentile(bo[:, 1], [2.5, 97.5])]}
        m = out["mien"][ten_m]
        print("[%s] %d video, %d anh: mAP50 %.4f -> %.4f (%+.2f diem, CI %+.2f..%+.2f) | mAP50-95 %.4f -> %.4f (%+.2f, CI %+.2f..%+.2f)" % (
            ten_m, len(vids), len(ids), a50, b50, 100 * m["hieu_mAP50"], 100 * m["ci95_mAP50"][0], 100 * m["ci95_mAP50"][1],
            a5095, b5095, 100 * m["hieu_mAP50_95"], 100 * m["ci95_mAP50_95"][0], 100 * m["ci95_mAP50_95"][1]), flush=True)
    Path(a.ra).write_text(json.dumps(ev._json_safe(out), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
