#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 38 -- KIEM DOC LAP dataset day du data/yolo_bdd5_v1 sau khi xuat (CHI DOC; khong sua dataset, khong train).

    python handoff/logs/review38/05_kiem_doc_lap.py --ra handoff/logs/review38/05_kiem_doc_lap.json
Bo sung cho hop dong dung chung (kiem_hop_dong_dataset, chay o buoc xuat va --buoc kiem):
  1. Tap (split, video, frame_idx) trong manifest == tap tinh lai bang CHINH bx.doc_split / chon_tap_con(None) /
     chon_frame (buoc_frame cau hinh) tren nguon nhan -- khop TUNG PHAN TU, khong chi khop so luong.
  2. Moi video train_fit / dev cua split da khoa co mat dung split; khong video nao o ca hai.
  3. Moi file nhan YOLO: dong co 5 truong, lop nguyen trong 0..4, cx/cy trong [0, 1], w/h trong (0, 1], box nam trong
     khung anh (dung sai 1e-6); dem nhan rong; khop so box theo lop voi manifest.
  4. Thong ke theo split: anh, video, box tung lop, anh co lop, video co lop, anh am tinh, anh co vung ignore.
  5. Khong co .npy canh anh; data.yaml tro dung images/train va images/dev.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))
import bdd_yolo_export as bx  # noqa: E402

LOP = ["car", "truck", "bus", "motorcycle", "bicycle"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(GOC / "configs/phase2/detector/bdd5_train.yaml"))
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cfg = bx.doc_cau_hinh(Path(a.config))
    out = bx.p_abs(cfg, cfg["dau_ra"])
    nguon = bx.mo_nguon(cfg["bdd"]["nhan"], bx.goc_du_an(cfg))
    sp = bx.doc_split(bx.p_abs(cfg, cfg["file_split"]), nguon)
    ds = bx.chon_tap_con(sp, cfg, None)
    buoc = int(cfg["lay_mau"]["buoc_frame"])
    kq = {"dataset": str(out), "buoc_frame": buoc, "loi": []}

    # 1. tap mong doi tu nguon nhan (cung ham bo xuat)
    mong = set()
    for s in ("train", "dev"):
        for v in ds[s]:
            chon, _ = bx.chon_frame(nguon.doc_video(v), buoc)
            mong |= {(s, v, int(f.idx)) for f in chon}
    rows = list(csv.DictReader(open(out / "manifest" / "anh.csv", newline="", encoding="utf-8")))
    thuc = {(r["split"], r["video"], int(r["frame_idx"])) for r in rows}
    kq["khop_tap_frame"] = {"mong_doi": len(mong), "manifest": len(thuc), "dong_manifest": len(rows),
                            "thieu_trong_manifest": len(mong - thuc), "thua_trong_manifest": len(thuc - mong),
                            "vd_thieu": sorted(mong - thuc)[:5], "vd_thua": sorted(thuc - mong)[:5]}
    if mong != thuc or len(rows) != len(thuc):
        kq["loi"].append("tap (split, video, frame_idx) cua manifest khac tap tinh lai tu nguon nhan")

    # 2. video theo split
    vid = defaultdict(set)
    for r in rows:
        vid[r["split"]].add(r["video"])
    kq["video"] = {"train_trong_manifest": len(vid["train"]), "dev_trong_manifest": len(vid["dev"]),
                   "train_split": len(sp["train_fit"]), "dev_split": len(sp["dev"]),
                   "train_ngoai_split": len(vid["train"] - set(sp["train_fit"])),
                   "dev_ngoai_split": len(vid["dev"] - set(sp["dev"])), "chung_train_dev": len(vid["train"] & vid["dev"]),
                   "video_split_khong_co_anh": {"train": sorted(set(sp["train_fit"]) - vid["train"])[:10],
                                                "dev": sorted(set(sp["dev"]) - vid["dev"])[:10]}}
    if kq["video"]["train_ngoai_split"] or kq["video"]["dev_ngoai_split"] or kq["video"]["chung_train_dev"]:
        kq["loi"].append("video sai split hoac trung giua train va dev")

    # 3. nhan YOLO + 4. thong ke
    tk = {s: {"so_anh": 0, "am_tinh": 0, "anh_co_ignore": 0, "box": Counter(), "anh_co_lop": Counter(),
              "video_co_lop": defaultdict(set)} for s in ("train", "dev")}
    loi_nhan, lech_manifest, so_dong = [], 0, 0
    for r in rows:
        s = r["split"]
        W, H = int(r["w"]), int(r["h"])
        dem = Counter()
        for i, ln in enumerate((out / r["nhan_ra"]).read_text(encoding="utf-8").splitlines()):
            if not ln.strip():
                continue
            so_dong += 1
            p = ln.split()
            try:
                c = int(p[0])
                cx, cy, w, h = (float(x) for x in p[1:])
            except (ValueError, IndexError):
                loi_nhan.append(f"{r['nhan_ra']}:{i + 1} khong doc duoc: {ln[:60]}")
                continue
            ok = (len(p) == 5 and 0 <= c < 5 and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
                  and cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6 and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6)
            if not ok:
                loi_nhan.append(f"{r['nhan_ra']}:{i + 1} ngoai mien: {ln[:60]} (anh {W}x{H})")
                continue
            dem[LOP[c]] += 1
        if any(dem[c] != int(r[f"n_{c}"]) for c in LOP):
            lech_manifest += 1
        t = tk[s]
        t["so_anh"] += 1
        t["am_tinh"] += not sum(dem.values())
        t["anh_co_ignore"] += bool(r["ignore_ra"])
        for c, n in dem.items():
            t["box"][c] += n
            t["anh_co_lop"][c] += 1
            t["video_co_lop"][c].add(r["video"])
    kq["nhan"] = {"so_dong_box": so_dong, "dong_loi": len(loi_nhan), "vd_loi": loi_nhan[:10],
                  "anh_so_box_lech_manifest": lech_manifest}
    if loi_nhan or lech_manifest:
        kq["loi"].append("nhan YOLO sai dinh dang / ngoai mien hoac lech so box manifest")
    kq["thong_ke"] = {s: {"so_anh": t["so_anh"], "so_video": len(vid[s]), "am_tinh": t["am_tinh"],
                          "anh_co_vung_ignore": t["anh_co_ignore"], "box": {c: t["box"][c] for c in LOP},
                          "anh_co_lop": {c: t["anh_co_lop"][c] for c in LOP},
                          "video_co_lop": {c: len(t["video_co_lop"][c]) for c in LOP}} for s, t in tk.items()}

    # 5. .npy + data.yaml
    import yaml
    y = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    npy = [str(p) for s in ("train", "dev") for p in (out / "images" / s).glob("*.npy")]
    kq["data_yaml"] = {k: y.get(k) for k in ("path", "train", "val", "nc", "names")}
    kq["npy_canh_anh"] = len(npy)
    if npy or y.get("train") != "images/train" or y.get("val") != "images/dev" or y.get("nc") != 5:
        kq["loi"].append("data.yaml sai duong dan / so lop hoac co .npy canh anh")
    kq["thoi_gian_s"] = round(time.time() - t0, 1)
    kq["ket_luan"] = "DAT" if not kq["loi"] else "KHONG DAT"
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1, default=list) + "\n", encoding="utf-8",
                          newline="\n")
    print(json.dumps({k: kq[k] for k in ("khop_tap_frame", "video", "nhan", "npy_canh_anh", "data_yaml", "ket_luan",
                                         "loi", "thoi_gian_s")}, ensure_ascii=False, indent=1, default=list))
    for s, t in kq["thong_ke"].items():
        print(s, json.dumps(t, ensure_ascii=False))
    return 0 if not kq["loi"] else 1


if __name__ == "__main__":
    sys.exit(main())
