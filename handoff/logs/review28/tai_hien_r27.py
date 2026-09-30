# -*- coding: utf-8 -*-
"""
tai_hien_r27.py -- TAI DUNG (Claude, vong 28) ba finding R27-F1/F2/F3 cua review 4937c85 theo MO TA trong yeu cau cua nguoi
dung, vi luc bat dau vong KHONG co REPRO_REVIEW_4937c85.py / CODEX_REVIEW_4937c85.md tren may. Day KHONG phai script
Codex; khi co script Codex se chay them no tren cung ma cu / ma moi.

Chay tren ma nguon o --src (ban sao ma cu 4937c85 hoac ma da sua), fixture tong hop trong thu muc tam:
  F1a  data.yaml sua `train: images/dev`            -> runner --dry-run phai TU CHOI (exit != 0)
  F1b  dataset thieu 1 file nhan                    -> runner --dry-run phai TU CHOI
  F2a  anh doi noi dung, giu kich thuoc             -> bo kiem dataset phai bao loi
  F2b  toa do nhan doi, giu lop + so box            -> bo kiem dataset phai bao loi
  F2c  file ignore da khai bao bi XOA               -> bo kiem dataset phai bao loi
  F2d  file ignore da khai bao bi SUA               -> bo kiem dataset phai bao loi
  F3   hai GT IoU bang nhau voi det1; det2 chi khop GT dau -> AP50 phai = pycocotools (1.0)
Exit 1 neu CON it nhat mot loi tai hien (ma cu), 0 neu khong con (ma da sua). Exit 2 = loi ha tang cua chinh script.

    python tai_hien_r27.py --src <thu muc src> --weights <yolov8n.pt> [--json ket_qua.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def cfg_fixture(tmp: Path) -> dict:
    return {"ten": "tr", "goc_du_an": str(tmp), "nguon": ["bdd100k_mot_train"],
            "lop": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "bdd": {"split": "train", "anh": "bdd/images/track/train",
                    "nhan": {"dinh_dang": "scalabel", "duong_dan": "bdd/labels/box_track_20/train"},
                    "chi_video_du_anh": False},
            "chia": {"cach": "theo_video", "ti_le_dev": 0.34, "seed": 0, "phan_tang_theo": ["motorcycle", "bicycle"]},
            "lay_mau": {"buoc_frame": 1}, "file_split": "splits/tr.json", "dau_ra": "out/tr",
            "thu_muc_hash_tap_danh_gia": "hash_cache",
            "tap_danh_gia_cam": [{"ten": "detrac_gia", "loai": "detrac_xml", "xml": "detrac_xml"}]}


def dung_dataset(tmp: Path) -> Path:
    import yaml
    import bdd_yolo_export as bx
    from mot_datasets.make_bdd5_train_fixture import tao_fixture
    tao_fixture(tmp / "bdd", n_video=8, n_frame=12, W=160, H=90, seed=0)
    (tmp / "detrac_xml").mkdir()
    (tmp / "detrac_xml" / "MVI_00001.xml").write_text("<x/>", encoding="utf-8")
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg_fixture(tmp), sort_keys=False), encoding="utf-8")
    cfg = bx.doc_cau_hinh(p)
    fs = bx.p_abs(cfg, cfg["file_split"])
    bx.tao_split(cfg, fs)
    bx.xuat_dataset(cfg, fs)
    return bx.p_abs(cfg, cfg["dau_ra"])


def ban_sao(goc: Path, dich: Path) -> Path:
    shutil.copytree(goc, dich)
    y = (dich / "data.yaml").read_text(encoding="utf-8")
    (dich / "data.yaml").write_text(y.replace(goc.resolve().as_posix(), dich.resolve().as_posix()), encoding="utf-8")
    return dich


def chay_runner(src: Path, data_yaml: Path, cfg_train: Path, tmp: Path, ten: str) -> dict:
    r = subprocess.run([sys.executable, str(src / "train_detector.py"), "--config", str(cfg_train), "--model", "yolov8n",
                        "--dry-run", "--data", str(data_yaml), "--project", str(tmp / "runs"),
                        "--ket-qua-dir", str(tmp / "kq"), "--ten", ten], capture_output=True, text=True)
    so_anh = None
    try:
        j = json.loads(r.stdout[:r.stdout.rindex("}") + 1])
        so_anh = j.get("dataset", {}).get("so_anh")
    except (ValueError, json.JSONDecodeError):
        pass
    return {"exit": r.returncode, "chap_nhan": r.returncode == 0, "so_anh_bao_cao": so_anh,
            "stderr_cuoi": r.stderr.strip().splitlines()[-1][:300] if r.stderr.strip() else ""}


def kiem_bo_kiem(out: Path) -> dict:
    import bdd_yolo_export as bx
    try:
        bx.kiem_dataset(out)
        return {"bao_loi": False, "thong_diep": ""}
    except Exception as e:  # noqa: BLE001
        return {"bao_loi": True, "thong_diep": f"{type(e).__name__}: {str(e)[:300]}"}


def f3() -> dict:
    import eval_detector_5lop as ev
    gt = [{"cls": np.array([0, 0]), "xyxy": np.array([[5, 0, 15, 10], [7, 0, 17, 10]], float), "ign": np.zeros((0, 4))}]
    dd = [{"xyxy": np.array([[6, 0, 16, 10], [2, 0, 12, 10]], float), "conf": np.array([0.9, 0.8]),
           "cls": np.array([0, 0])}]
    kq = ev.danh_gia(gt, dd)
    car = kq["theo_lop"]["car"]
    out = {"AP50_ta": car["AP50"], "TP_FP_van_hanh": (car["van_hanh"]["TP"], car["van_hanh"]["FP"])}
    try:
        import contextlib
        import io

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        g = {"images": [{"id": 1, "width": 40, "height": 20}], "categories": [{"id": k + 1, "name": str(k)} for k in range(5)],
             "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                              "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float((b[2] - b[0]) * (b[3] - b[1]))}
                             for i, b in enumerate(gt[0]["xyxy"])]}
        d = [{"image_id": 1, "category_id": 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)}
             for b, s in zip(dd[0]["xyxy"], dd[0]["conf"])]
        with contextlib.redirect_stdout(io.StringIO()):
            G = COCO()
            G.dataset = g
            G.createIndex()
            E = COCOeval(G, G.loadRes(d), "bbox")
            E.params.maxDets = [1, 10, 1000]
            E.evaluate()
            E.accumulate()
        out["AP50_pycocotools"] = float(E.eval["precision"][0, :, 0, 0, 2].mean())
        out["pycocotools"] = "da chay"
        out["tai_hien"] = abs(out["AP50_ta"] - out["AP50_pycocotools"]) > 1e-12
    except ImportError as e:
        out["pycocotools"] = f"KHONG co ({e}) -> CHUA KIEM"
        out["tai_hien"] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cfg-train", required=True, help="train_bdd5.yaml (weights se duoc doi sang duong dan tuyet doi)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    src = Path(a.src).resolve()
    sys.path.insert(0, str(src))
    import yaml
    kq = {"src": str(src), "ghi_chu": "script TAI DUNG boi Claude tu mo ta review, KHONG phai REPRO_REVIEW_4937c85.py"}
    tmp = Path(tempfile.mkdtemp(prefix="tai_hien_r27_"))
    try:
        out = dung_dataset(tmp)
        c = yaml.safe_load(Path(a.cfg_train).read_text(encoding="utf-8"))
        c["model"]["yolov8n"]["weights"] = str(Path(a.weights).resolve())     # ban sao ma cu khong co models/
        cfg_train = tmp / "train.yaml"
        cfg_train.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        hop_le = chay_runner(src, out / "data.yaml", cfg_train, tmp, "hop_le")
        kq["F1_doi_chung_dataset_hop_le"] = hop_le
        # F1a: train tro vao dev
        d1 = ban_sao(out, tmp / "f1a")
        y = (d1 / "data.yaml").read_text(encoding="utf-8").replace("train: images/train", "train: images/dev")
        (d1 / "data.yaml").write_text(y, encoding="utf-8")
        r = chay_runner(src, d1 / "data.yaml", cfg_train, tmp, "f1a")
        kq["F1a_train_tro_vao_dev"] = r | {"tai_hien": r["chap_nhan"]}
        # F1b: thieu 1 file nhan
        d2 = ban_sao(out, tmp / "f1b")
        next((d2 / "labels" / "train").glob("*.txt")).unlink()
        r = chay_runner(src, d2 / "data.yaml", cfg_train, tmp, "f1b")
        kq["F1b_thieu_file_nhan"] = r | {"tai_hien": r["chap_nhan"]}
        # F2a: anh doi noi dung, cung kich thuoc
        import cv2
        d3 = ban_sao(out, tmp / "f2a")
        img = sorted((d3 / "images" / "train").glob("*.jpg"))[0]
        im = cv2.imread(str(img))
        im[10:20, 10:20] = 255 - im[10:20, 10:20]
        cv2.imwrite(str(img), im)
        r = kiem_bo_kiem(d3)
        kq["F2a_anh_doi_noi_dung"] = r | {"anh": img.name, "tai_hien": not r["bao_loi"]}
        # F2b: toa do nhan doi, giu lop + so box
        d4 = ban_sao(out, tmp / "f2b")
        lbl = next(p for p in sorted((d4 / "labels" / "train").glob("*.txt")) if p.read_text(encoding="utf-8").strip())
        dong = lbl.read_text(encoding="utf-8").splitlines()
        p0 = dong[0].split()
        p0[1] = f"{min(0.95, float(p0[1]) + 0.01):.6f}"
        dong[0] = " ".join(p0)
        lbl.write_text("\n".join(dong) + "\n", encoding="utf-8")
        r = kiem_bo_kiem(d4)
        kq["F2b_toa_do_nhan_doi"] = r | {"nhan": lbl.name, "tai_hien": not r["bao_loi"]}
        # F2c / F2d: file ignore bi xoa / bi sua
        for k, hanh_dong in (("F2c_ignore_bi_xoa", "xoa"), ("F2d_ignore_bi_sua", "sua")):
            dk = ban_sao(out, tmp / k)
            ig = next(p for s in ("train", "dev") for p in sorted((dk / "ignore" / s).glob("*.txt")))
            if hanh_dong == "xoa":
                ig.unlink()
            else:
                ig.write_text(ig.read_text(encoding="utf-8") + "0.00 0.00 1.00 1.00\n", encoding="utf-8")
            r = kiem_bo_kiem(dk)
            kq[k] = r | {"file": ig.name, "tai_hien": not r["bao_loi"]}
        kq["F3_hoa_iou"] = f3()
    except Exception:  # noqa: BLE001
        kq["loi_script"] = traceback.format_exc()
        print(json.dumps(kq, ensure_ascii=False, indent=1, default=str))
        return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    con = {k: v["tai_hien"] for k, v in kq.items() if isinstance(v, dict) and "tai_hien" in v}
    kq["tong_ket"] = {"con_tai_hien": [k for k, v in con.items() if v], "khong_tai_hien": [k for k, v in con.items() if v is False],
                      "chua_kiem": [k for k, v in con.items() if v is None]}
    s = json.dumps(kq, ensure_ascii=False, indent=1, default=str)
    print(s)
    if a.json:
        Path(a.json).write_text(s + "\n", encoding="utf-8")
    return 1 if kq["tong_ket"]["con_tai_hien"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
