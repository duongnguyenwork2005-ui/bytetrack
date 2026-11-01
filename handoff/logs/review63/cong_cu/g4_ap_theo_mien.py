"""R63 G4: AP 5 lop theo TUNG MIEN dev (BDD dev / UA-DETRAC dev12) tu du doan da luu cua eval_detector_5lop
(du_doan_<ten>.npz trong hai thu muc ket qua: doi chung 640x640 va ung vien 384x640, cung tap anh dev mixed5).
Cham bang DUNG eval_detector_5lop.danh_gia; bootstrap ghep cap theo VIDEO (ten anh '<video>-<7 so>.jpg') cho
hieu mAP50 / mAP50-95 (ung vien - doi chung), seed 0, CI 95 % percentile.

    python handoff/logs/review63/cong_cu/g4_ap_theo_mien.py --doi-chung <dir 640x640> --ung-vien <dir 384x640> \
        --ten yolov8n_coco --ra handoff/logs/review63/g4/ap_dev_theo_mien.json [--n-boot 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))
import eval_detector_5lop as ev  # noqa: E402


def doc_du_doan(p: Path, n: int) -> tuple[list[str], list[dict]]:
    z = np.load(p)
    anh = [str(x) for x in z["anh"]]
    if len(anh) != n:
        raise ValueError(f"{p}: {len(anh)} anh != {n}")
    dd = [{"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, int)} for _ in anh]
    idx = z["idx"]
    for i in range(n):
        m = idx == i
        dd[i] = {"xyxy": z["xyxy"][m].astype(float), "conf": z["conf"][m].astype(float), "cls": z["cls"][m].astype(int)}
    return anh, dd


def video_cua(ten: str) -> str:
    return ten.rsplit("-", 1)[0]


def ghep_truoc(gt: list[dict], dd: list[dict]) -> list[list[tuple]]:
    """Ghep (anh, lop) MOT LAN bang dung ev.ghep_anh_lop (khong phu thuoc anh khac) -> [anh][lop] =
    (conf da xep, tp[T,n], bo_qua[T,n], n_gt). Bootstrap chi noi lai cac phan nay roi goi ev.ap_coco."""
    out = []
    for g, p in zip(gt, dd):
        hang = []
        for c in range(len(ev.LOP_5)):
            mg, md = g["cls"] == c, p["cls"] == c
            o, tp, bq = ev.ghep_anh_lop(p["xyxy"][md], p["conf"][md], g["xyxy"][mg], g["ign"])
            hang.append((p["conf"][md][o], tp, bq, int(mg.sum())))
        out.append(hang)
    return out


def map_tu_ghep(gh: list[list[tuple]], sel: list[int]) -> tuple[float, float]:
    """mAP50, mAP50-95 macro tren cac lop co GT trong mau `sel` (thu tu va lap lai giu nguyen) -- cung cong thuc
    noi + ap_coco nhu ev.danh_gia."""
    a50, a95 = [], []
    for c in range(len(ev.LOP_5)):
        n_gt = sum(gh[i][c][3] for i in sel)
        if n_gt == 0:
            continue
        conf = np.concatenate([gh[i][c][0] for i in sel])
        tp = np.concatenate([gh[i][c][1] for i in sel], axis=1)
        bq = np.concatenate([gh[i][c][2] for i in sel], axis=1)
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
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    if Path(a.ra).exists():
        print("[LOI] da ton tai", a.ra); return 2
    _, anh, gt = ev.chuan_bi_danh_gia(Path(a.data), "val")
    ten_anh = [p.name for p in anh]
    a_anh, dd_a = doc_du_doan(Path(a.doi_chung) / f"du_doan_{a.ten}.npz", len(anh))
    b_anh, dd_b = doc_du_doan(Path(a.ung_vien) / f"du_doan_{a.ten}.npz", len(anh))
    if a_anh != ten_anh or b_anh != ten_anh:
        print("[LOI] thu tu anh du doan khac tap dev"); return 2
    mien = {"detrac_dev12": [i for i, t in enumerate(ten_anh) if t.startswith("MVI_")],
            "bdd_dev": [i for i, t in enumerate(ten_anh) if not t.startswith("MVI_")]}
    rng = np.random.default_rng(0)
    gh_a, gh_b = ghep_truoc(gt, dd_a), ghep_truoc(gt, dd_b)
    out = {"doi_chung": a.doi_chung, "ung_vien": a.ung_vien, "n_boot": a.n_boot, "mien": {},
           "phuong_phap": "ghep (anh, lop) mot lan bang ev.ghep_anh_lop; moi mau bootstrap theo video noi lai va goi "
                          "ev.ap_coco; diem toan tap kiem bang ev.danh_gia (lech > 1e-12 -> loi)"}
    for ten_m, ids in mien.items():
        vids = sorted({video_cua(ten_anh[i]) for i in ids})
        theo_v = {v: [i for i in ids if video_cua(ten_anh[i]) == v] for v in vids}
        ra_ = ev.danh_gia([gt[i] for i in ids], [dd_a[i] for i in ids])
        rb_ = ev.danh_gia([gt[i] for i in ids], [dd_b[i] for i in ids])
        a50, a5095 = ra_["macro"]["mAP50"], ra_["macro"]["mAP50_95"]
        b50, b5095 = rb_["macro"]["mAP50"], rb_["macro"]["mAP50_95"]
        kiem = map_tu_ghep(gh_a, ids) + map_tu_ghep(gh_b, ids)
        if max(abs(x - y) for x, y in zip(kiem, (a50, a5095, b50, b5095))) > 1e-12:
            print("[LOI] tinh nhanh lech danh_gia", ten_m, kiem, (a50, a5095, b50, b5095)); return 3
        import time
        t0 = time.time()
        boots = []
        for k_boot in range(a.n_boot):
            if k_boot in (1, 10) or k_boot % 100 == 0:
                print("[%s] boot %d/%d, %.0f s" % (ten_m, k_boot, a.n_boot, time.time() - t0), flush=True)
            vv = rng.choice(len(vids), size=len(vids), replace=True)
            sel = [i for k in vv for i in theo_v[vids[k]]]
            x50, x95 = map_tu_ghep(gh_a, sel); y50, y95 = map_tu_ghep(gh_b, sel)
            boots.append((y50 - x50, y95 - x95))
        bo = np.array(boots)
        out["mien"][ten_m] = {
            "n_anh": len(ids), "n_video": len(vids),
            "doi_chung": {"mAP50": a50, "mAP50_95": a5095, "theo_lop": {k: {"AP50": v["AP50"], "AP50_95": v["AP50_95"],
                          "n_gt": v["n_gt"]} for k, v in ra_["theo_lop"].items()}},
            "ung_vien": {"mAP50": b50, "mAP50_95": b5095, "theo_lop": {k: {"AP50": v["AP50"], "AP50_95": v["AP50_95"],
                         "n_gt": v["n_gt"]} for k, v in rb_["theo_lop"].items()}},
            "hieu_mAP50": b50 - a50, "ci95_mAP50": [float(x) for x in np.percentile(bo[:, 0], [2.5, 97.5])],
            "hieu_mAP50_95": b5095 - a5095, "ci95_mAP50_95": [float(x) for x in np.percentile(bo[:, 1], [2.5, 97.5])]}
        m = out["mien"][ten_m]
        print("[%s] mAP50 %.4f -> %.4f (%+.2f diem, CI %+.2f..%+.2f) | mAP50-95 %.4f -> %.4f (%+.2f, CI %+.2f..%+.2f)" % (
            ten_m, a50, b50, 100 * m["hieu_mAP50"], 100 * m["ci95_mAP50"][0], 100 * m["ci95_mAP50"][1],
            a5095, b5095, 100 * m["hieu_mAP50_95"], 100 * m["ci95_mAP50_95"][0], 100 * m["ci95_mAP50_95"][1]))
    Path(a.ra).write_text(json.dumps(ev._json_safe(out), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
