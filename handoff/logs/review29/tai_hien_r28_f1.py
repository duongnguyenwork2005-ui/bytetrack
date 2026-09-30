# -*- coding: utf-8 -*-
"""
tai_hien_r28_f1.py -- TAI DUNG (Claude, vong 29) finding R28-F1 cua review 07615ba theo MO TA trong CODEX_REVIEW_07615ba.md,
vi luc bat dau vong KHONG co REPRO_REVIEW_07615ba.py / goi CODEX_REVIEW_07615ba_EVIDENCE.zip tren may. KHONG phai script
Codex; khi co script Codex se chay no tren cung ma.

Cach lam (dung ma THAT cua repo o <repo>/src, fixture trong thu muc tam, tu don):
  - xuat fixture bang exporter cua du an (helper cua test_detector_5lop, giong script Codex vong 27);
  - DOI CHUNG: dataset sach -> chot runner `train_detector.kiem_dataset` chap nhan; `YOLODataset(cache=False)` THAT doc
    pixel trung khop tung byte voi JPEG (cv2.imread); callback `kiem_dau_vao_da_nap` cho qua;
  - moi split (train, dev): dat file `<anh>.npy` ANH DEN cung kich thuoc canh MOT anh da khoa (JPEG/nhan/hash giu nguyen),
    goi chot runner + dung YOLODataset THAT cho ca hai split + goi callback THAT, doc pixel qua get_image_and_label.
    LOI TAI HIEN neu: chot chap nhan VA callback chap nhan VA pixel nap la anh den (khac JPEG).
Exit 1 neu bug_count > 0; 0 neu khong con loi va doi chung dat; 2 neu doi chung hong / loi ha tang (KHONG phai bang chung).

    python tai_hien_r28_f1.py <repo> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def outcome(fn, *a):
    try:
        return {"accepted": True, "result_keys": sorted((fn(*a) or {}).keys())[:6]}
    except Exception as e:  # noqa: BLE001
        return {"accepted": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / "src"))
    import cv2
    import bdd_yolo_export as bx
    import train_detector as tr
    import test_detector_5lop as fx
    from ultralytics.data.dataset import YOLODataset
    kq = {"ghi_chu": "script TAI DUNG boi Claude tu mo ta review 07615ba, KHONG phai REPRO_REVIEW_07615ba.py",
          "repo": str(a.repo.resolve())}
    names = {i: c for i, c in enumerate(["car", "truck", "bus", "motorcycle", "bicycle"])}
    try:
        with tempfile.TemporaryDirectory(prefix="tai_hien_r28f1_") as t:
            root = Path(t)
            fx.tao_moi_truong(root)
            cfg = fx.nap_cfg(root, fx.cfg_mau(root, "review"))
            fs = bx.p_abs(cfg, cfg["file_split"])
            bx.tao_split(cfg, fs)
            bx.xuat_dataset(cfg, fs)
            out = bx.p_abs(cfg, cfg["dau_ra"])
            data = out / "data.yaml"
            md = tr.nhan_mong_doi(data)

            def yds(s, imgsz):
                return YOLODataset(img_path=str(out / "images" / s), data={"names": names, "nc": 5, "channels": 3},
                                   task="detect", augment=False, imgsz=imgsz, cache=False)

            def xoa_cache_nhan():
                for s in ("train", "dev"):
                    (out / "labels" / f"{s}.cache").unlink(missing_ok=True)

            anh0 = sorted((out / "images" / "train").glob("*.jpg"))[0]
            h, w = cv2.imread(str(anh0)).shape[:2]
            imgsz = max(h, w)                                     # khong resize trong load_image -> so pixel tung byte
            # ---------------- doi chung: dataset sach
            xoa_cache_nhan()
            dts = {s: yds(s, imgsz) for s in ("train", "dev")}
            khop = all(np.array_equal(ds.get_image_and_label(i)["img"], cv2.imread(f))
                       for ds in dts.values() for i, f in enumerate(ds.im_files))
            cb = outcome(tr.kiem_dau_vao_da_nap, SimpleNamespace(train_loader=SimpleNamespace(dataset=dts["train"]),
                                                                  test_loader=SimpleNamespace(dataset=dts["dev"])), md)
            kq["doi_chung_sach"] = {"chot_runner": outcome(tr.kiem_dataset, data), "callback": cb,
                                    "pixel_nap_trung_jpeg_moi_anh": khop,
                                    "so_anh": {s: len(d.im_files) for s, d in dts.items()}}
            if not (kq["doi_chung_sach"]["chot_runner"]["accepted"] and cb["accepted"] and khop):
                kq["loi_ha_tang"] = "doi chung dataset sach KHONG dat -> khong ket luan duoc"
                a.out.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
                print(json.dumps(kq, ensure_ascii=False, indent=1))
                return 2
            # ---------------- moi split: .npy anh den canh mot anh da khoa
            for s in ("train", "dev"):
                anh = sorted((out / "images" / s).glob("*.jpg"))[0]
                jpg = cv2.imread(str(anh))
                npy = anh.with_suffix(".npy")
                np.save(npy, np.zeros_like(jpg))
                try:
                    chot = outcome(tr.kiem_dataset, data)
                    xoa_cache_nhan()
                    dts = {x: yds(x, imgsz) for x in ("train", "dev")}
                    cbk = outcome(tr.kiem_dau_vao_da_nap,
                                  SimpleNamespace(train_loader=SimpleNamespace(dataset=dts["train"]),
                                                  test_loader=SimpleNamespace(dataset=dts["dev"])), md)
                    ds = dts[s]
                    idx = [bx._nc(f) for f in ds.im_files].index(bx._nc(anh))
                    nap = ds.get_image_and_label(idx)["img"]
                    kq[f"R28F1_{s}"] = {
                        "anh": anh.name, "npy": npy.name, "jpeg_giu_nguyen_hash": bx.sha256_file(anh) == next(
                            r["sha256_ra"] for r in __import__("csv").DictReader(open(out / "manifest" / "anh.csv",
                                                                                     encoding="utf-8"))
                            if Path(r["anh_ra"]).name == anh.name),
                        "trung_binh_pixel_jpeg": float(jpg.mean()), "trung_binh_pixel_nap": float(nap.mean()),
                        "chot_runner": chot, "callback_du_lieu_nap": cbk,
                        "bug_reproduced": bool(chot["accepted"] and cbk["accepted"] and not np.array_equal(nap, jpg))}
                finally:
                    npy.unlink(missing_ok=True)
                    xoa_cache_nhan()
    except Exception:  # noqa: BLE001
        kq["loi_ha_tang"] = traceback.format_exc()
        a.out.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
        print(kq["loi_ha_tang"])
        return 2
    kq["bug_count"] = sum(int(v["bug_reproduced"]) for k, v in kq.items() if k.startswith("R28F1_"))
    a.out.write_text(json.dumps(kq, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["bug_reproduced"] if isinstance(v, dict) and "bug_reproduced" in v else v)
                      for k, v in kq.items() if k.startswith("R28F1_") or k == "bug_count"}, indent=1))
    return 1 if kq["bug_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
