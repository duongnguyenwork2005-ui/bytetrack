"""
test_danh_gia_bdd_val.py -- Kiem `danh_gia_detector_bdd_val.py` (vong 43) tren FIXTURE TONG HOP (anh JPEG nho + nhan
COCO-video tu tao) trong thu muc tam. Khong dung BDD val that, khong nap model that.

  B1  GT theo quy tac exporter: car thuong -> GT 0; truck crowd -> vung bo qua; car ignore=1 (distractor) -> vung bo
      qua; pedestrian thuong / crowd, rider, train -> khong GT, khong vung; motorcycle vuot khung -> cat + dem; box
      < 1 px sau cat -> bo + dem; anh khong annotation -> VAN co mat (anh am tinh).
  B2  Doi chieu DUONG DA NGHIEM THU: cung frame xuat bang `bdd_yolo_export.xuat_mot_anh` roi doc bang
      `eval_detector_5lop.doc_gt` -> cung lop, box lech <= 1e-3 px (nhan YOLO 6 chu so), vung lech <= 0.01 px.
  B3  Tu choi (LoiDuLieu): sha256 nhan khac manifest; tap video khac; anh thua tren dia; anh thieu tren dia; w/h khai
      bao khac anh thuc; noi dung anh khac bang hash.
  B4  Entry point `main()` voi du doan GIA LAP (thay `du_doan_ultralytics` -- ranh gioi nap model): exit 0; ket_qua.json
      / theo_lop.csv / npz; metric = `eval_detector_5lop.danh_gia` goi truc tiep tren cung GT + du doan; so anh du doan
      = so anh; thu muc ra da ton tai -> exit 1; sha256 checkpoint sai -> exit 2 va KHONG goi du doan; ten
      --model / --sha256 khong khop -> exit 2.
  B5  (vong 44, R43-N1) nguong van hanh HIEU LUC qua entry point `main()`; nguong scorer bat bang ham boc quanh
      `danh_gia` THAT: mac dinh -> metadata `quy_tac.van_hanh` = nguong scorer = 0.25; `--conf-van-hanh 0.5` -> ca hai
      0.5, P/R + chan doan = goi truc tiep voi 0.5; hai lan goi lien tiep (0.4 roi mac dinh) trong cung tien trinh ->
      moi lan dung nguong cua no, `QUY_TAC` dung chung va mac dinh CLI van 0.25 (khong ro ri cau hinh).

    python src/test_danh_gia_bdd_val.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

_RESULTS: list[tuple[str, bool]] = []


def _check(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CATS = [{"id": 1, "name": "pedestrian"}, {"id": 2, "name": "rider"}, {"id": 3, "name": "car"},
        {"id": 4, "name": "truck"}, {"id": 5, "name": "bus"}, {"id": 6, "name": "train"},
        {"id": 7, "name": "motorcycle"}, {"id": 8, "name": "bicycle"}]
W, H = 64, 48


def tao_fixture(tmp: Path) -> dict:
    """2 video x 3 frame, anh 64x48. Tra ve duong dan nhan / anh / manifest / hash."""
    from PIL import Image
    rng = np.random.default_rng(0)
    anh_goc = tmp / "anh"
    videos = [{"id": 1, "name": "v1"}, {"id": 2, "name": "v2"}]
    images, ann = [], []
    iid = 0
    for v in videos:
        (anh_goc / v["name"]).mkdir(parents=True)
        for k in range(3):
            iid += 1
            rel = f"{v['name']}/{v['name']}-{k + 1:07d}.jpg"
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(anh_goc / rel, quality=90)
            images.append({"id": iid, "video_id": v["id"], "frame_id": k, "file_name": rel, "width": W, "height": H})
    aid = 0

    def a(img, cat, x, y, w, h, crowd=0, ign=0):
        nonlocal aid
        aid += 1
        ann.append({"id": aid, "image_id": img, "category_id": cat, "instance_id": aid, "bbox": [x, y, w, h],
                    "iscrowd": crowd, "ignore": ign})
    a(1, 3, 2.5, 3.25, 20, 10)                 # car thuong -> GT 0
    a(1, 4, 30, 5, 20, 15, crowd=1)            # truck crowd -> vung bo qua
    a(1, 3, 40, 25, 10, 10, crowd=1, ign=1)    # car ignore (distractor) -> vung bo qua
    a(1, 1, 5, 20, 6, 12)                      # pedestrian -> bo
    a(1, 1, 10, 20, 8, 12, crowd=1, ign=1)     # pedestrian crowd/ignore -> bo (khong phai 5 lop)
    a(2, 2, 3, 3, 5, 9)                        # rider -> bo
    a(2, 6, 10, 10, 30, 20)                    # train -> bo
    a(2, 7, 50, 30, 30, 30)                    # motorcycle vuot khung -> cat
    a(2, 8, 63.6, 10, 5, 5)                    # bicycle < 1 px sau cat -> bo
    a(2, 5, 1, 1, 12, 12)                      # bus -> GT 2
    a(4, 8, 4, 4, 9, 9)                        # bicycle -> GT 4 (video 2)
    # anh 3, 5, 6: khong annotation (anh am tinh)
    d = {"videos": videos, "images": images, "annotations": ann, "categories": CATS}
    nhan = tmp / "nhan_cocofmt.json"
    nhan.write_text(json.dumps(d), encoding="utf-8")
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"videos": ["v1", "v2"], "nguon_nhan_sha256": sha(nhan),
                                    "sequence_da_xem_truoc": [], "ghi_chu": "fixture"}), encoding="utf-8")
    bang = tmp / "hash.csv"
    with open(bang, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel", "size", "mtime_ns", "sha256"])
        for im in images:
            p = anh_goc / im["file_name"]
            w.writerow([im["file_name"], p.stat().st_size, p.stat().st_mtime_ns, sha(p)])
    return {"nhan": nhan, "anh": anh_goc, "manifest": manifest, "hash": bang, "coco": d}


def nhom_b(tmp: Path) -> None:
    import bdd_yolo_export as bx
    import danh_gia_detector_bdd_val as m
    import eval_detector_5lop as ev
    fx = tao_fixture(tmp / "fx")
    dt_, anh, gt = m.kiem_va_doc_gt(fx["nhan"], fx["anh"], fx["manifest"], fx["hash"])
    g = {x["anh"]: x for x in gt}
    g1, g2 = g["v1-0000001.jpg"], g["v1-0000002.jpg"]
    _check("B1a 6 anh (ca 3 anh khong annotation) -- khong bo anh am tinh", len(anh) == 6 and
           dt_["dem"]["anh_khong_annotation"] == 3, f"{len(anh)} anh, khong annotation {dt_['dem']['anh_khong_annotation']}")
    _check("B1b car thuong -> GT lop 0, toa do pixel", list(g1["cls"]) == [0] and
           np.allclose(g1["xyxy"], [[2.5, 3.25, 22.5, 13.25]]), str(g1["xyxy"].tolist()))
    _check("B1c truck crowd + car ignore=1 -> 2 vung bo qua; pedestrian (thuong / crowd) -> khong GT, khong vung",
           len(g1["ign"]) == 2 and np.allclose(sorted(g1["ign"].tolist()), [[30, 5, 50, 20], [40, 25, 50, 35]]),
           str(g1["ign"].tolist()))
    _check("B1d rider / train -> bo; motorcycle vuot khung -> cat [50,30,64,48]; bicycle < 1 px -> bo; bus -> lop 2",
           sorted(g2["cls"].tolist()) == [2, 3] and np.allclose(g2["xyxy"][list(g2["cls"]).index(3)], [50, 30, 64, 48])
           and len(g2["ign"]) == 0, f"cls {g2['cls'].tolist()} xyxy {g2['xyxy'].tolist()}")
    dm = dt_["dem"]
    _check("B1e dem: GT theo lop, vung, lop khac, box cat, box < 1 px",
           dm["gt_theo_lop"] == {"car": 1, "truck": 0, "bus": 1, "motorcycle": 1, "bicycle": 1} and dm["vung_bo_qua"] == 2
           and dm["annotation_lop_khac"] == 4 and dm["box_bi_cat_vao_khung"] == 1 and dm["box_nho_hon_1px_sau_cat"] == 1,
           json.dumps(dm))
    _check("B1f doi chieu bang hash: 6/6 khop", dt_["doi_chieu_hash_anh"]["khop"] == 6)

    # B2: duong exporter -> evaluator (da nghiem thu) cho CUNG frame
    nguon = bx.NguonCocoVid(fx["nhan"])
    out = tmp / "xuat"
    ok_b2, chi_tiet = True, []
    for v in ("v1", "v2"):
        for fr in nguon.doc_video(v):
            src = fx["anh"] / fr.rel
            ten = Path(fr.rel).name
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "labels").mkdir(parents=True, exist_ok=True)
            (out / "ignore").mkdir(parents=True, exist_ok=True)
            dem = {k: 0 for k in ("nhan_lop_khac", "box_nho_hon_1px_sau_cat", "vung_to_xam_rong",
                                  "box_bi_cat_vao_khung", "so_pixel_to_xam")}
            r = bx.xuat_mot_anh(src, fr, out / "images" / ten, out / "labels" / (Path(ten).stem + ".txt"),
                                out / "ignore" / (Path(ten).stem + ".txt"), dem)
            ge = ev.doc_gt(out / "images" / ten, ignore_khai_bao=r["co_ignore"])
            gm = g[ten]
            o1, o2 = np.argsort(ge["cls"], kind="stable"), np.argsort(gm["cls"], kind="stable")
            dong = (list(ge["cls"][o1]) == list(gm["cls"][o2]) and np.allclose(ge["xyxy"][o1], gm["xyxy"][o2], atol=1e-3)
                    and len(ge["ign"]) == len(gm["ign"])
                    and np.allclose(sorted(ge["ign"].tolist()), sorted(gm["ign"].tolist()), atol=0.01))
            ok_b2 &= bool(dong)
            chi_tiet.append(f"{ten}:{'ok' if dong else 'LECH'}")
    _check("B2 GT truc tiep == duong exporter -> evaluator da nghiem thu (6 anh)", ok_b2, " ".join(chi_tiet))

    # B3: tu choi
    def tu_choi(fn) -> tuple[bool, str]:
        try:
            fn()
        except bx.LoiDuLieu as e:
            return True, str(e)[:90]
        return False, "KHONG tu choi"
    fx2 = tao_fixture(tmp / "fx_sha")
    mf = json.loads(fx2["manifest"].read_text(encoding="utf-8"))
    mf["nguon_nhan_sha256"] = "0" * 64
    fx2["manifest"].write_text(json.dumps(mf), encoding="utf-8")
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx2["nhan"], fx2["anh"], fx2["manifest"], fx2["hash"]))
    _check("B3a sha256 nhan khac manifest -> LoiDuLieu", ok, msg)
    fx3 = tao_fixture(tmp / "fx_video")
    mf = json.loads(fx3["manifest"].read_text(encoding="utf-8"))
    mf["videos"] = ["v1"]
    fx3["manifest"].write_text(json.dumps(mf), encoding="utf-8")
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx3["nhan"], fx3["anh"], fx3["manifest"], fx3["hash"]))
    _check("B3b tap video khac manifest -> LoiDuLieu", ok, msg)
    fx4 = tao_fixture(tmp / "fx_thua")
    (fx4["anh"] / "v1" / "v1-0000099.jpg").write_bytes((fx4["anh"] / "v1" / "v1-0000001.jpg").read_bytes())
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx4["nhan"], fx4["anh"], fx4["manifest"], fx4["hash"]))
    _check("B3c anh thua tren dia (khong co trong nhan) -> LoiDuLieu", ok, msg)
    fx5 = tao_fixture(tmp / "fx_thieu")
    (fx5["anh"] / "v2" / "v2-0000003.jpg").unlink()
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx5["nhan"], fx5["anh"], fx5["manifest"], fx5["hash"]))
    _check("B3d anh thieu tren dia -> LoiDuLieu", ok, msg)
    fx6 = tao_fixture(tmp / "fx_wh")
    d6 = fx6["coco"]
    d6["images"][0]["width"] = W + 1
    fx6["nhan"].write_text(json.dumps(d6), encoding="utf-8")
    mf = json.loads(fx6["manifest"].read_text(encoding="utf-8"))
    mf["nguon_nhan_sha256"] = sha(fx6["nhan"])
    fx6["manifest"].write_text(json.dumps(mf), encoding="utf-8")
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx6["nhan"], fx6["anh"], fx6["manifest"], fx6["hash"]))
    _check("B3e w/h khai bao khac anh thuc -> LoiDuLieu", ok, msg)
    fx7 = tao_fixture(tmp / "fx_noidung")
    from PIL import Image
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(fx7["anh"] / "v1" / "v1-0000002.jpg", quality=90)
    ok, msg = tu_choi(lambda: m.kiem_va_doc_gt(fx7["nhan"], fx7["anh"], fx7["manifest"], fx7["hash"]))
    _check("B3f noi dung anh khac bang hash (cung ten, cung kich thuoc) -> LoiDuLieu", ok, msg)

    # B4: entry point voi du doan GIA LAP
    ck = tmp / "ck.pt"
    ck.write_bytes(b"checkpoint gia")
    goi = []

    def gia_lap(weights, anh_ds, qt, batch=16, device="0"):
        """Box GT + nhieu nho (conf cao) va box ngau nhien (conf thap) -> metric KHAC 0, kiem duoc that."""
        goi.append(len(anh_ds))
        rng = np.random.default_rng(1)
        ra = []
        for p in anh_ds:
            gg = g[Path(p).name]
            n = 3
            x1 = rng.uniform(0, 40, n)
            y1 = rng.uniform(0, 30, n)
            xyxy = np.concatenate([gg["xyxy"] + rng.normal(0, 0.8, gg["xyxy"].shape),
                                   np.stack([x1, y1, x1 + 15, y1 + 12], 1)])
            ra.append({"xyxy": xyxy, "conf": np.concatenate([rng.uniform(0.5, 1, len(gg["cls"])),
                                                              rng.uniform(0.001, 0.6, n)]),
                       "cls": np.concatenate([gg["cls"], rng.integers(0, 5, n)]).astype(int)})
        return ra, {"loai_checkpoint": "gia_lap"}
    goc = m.du_doan_ultralytics
    m.du_doan_ultralytics = gia_lap
    try:
        def chay(out, sha_ck, ten_sha="gia", them=()):
            argv = ["x", "--nhan-cocovid", str(fx["nhan"]), "--anh-goc", str(fx["anh"]), "--manifest-video",
                    str(fx["manifest"]), "--hash-anh", str(fx["hash"]), "--model", f"gia={ck}",
                    "--sha256", f"{ten_sha}={sha_ck}", "--out", str(out), "--device", "cpu", *them]
            cu = sys.argv
            sys.argv = argv
            try:
                return m.main()
            finally:
                sys.argv = cu
        out = tmp / "ra"
        rc = chay(out, sha(ck))
        kq = json.loads((out / "ket_qua.json").read_text(encoding="utf-8")) if (out / "ket_qua.json").exists() else {}
        dd, _ = gia_lap(str(ck), anh, {})
        goi.pop()
        tt = ev.danh_gia(gt, dd, 0.25)
        mh = kq.get("mo_hinh", {}).get("gia", {})
        _check("B4a main() exit 0; ket_qua.json + theo_lop.csv + du_doan_gia.npz; so anh du doan = 6",
               rc == 0 and (out / "theo_lop.csv").is_file() and (out / "du_doan_gia.npz").is_file()
               and mh.get("so_anh_da_du_doan") == 6, f"rc {rc}")
        _check("B4b metric trong ket_qua = eval_detector_5lop.danh_gia goi truc tiep (cung GT + du doan), metric > 0",
               mh and tt["macro"]["mAP50"] > 0.1 and abs(mh["macro"]["mAP50_95"] - tt["macro"]["mAP50_95"]) < 1e-12
               and all(abs((mh["theo_lop"][c]["AP50"] or 0) - (np.nan_to_num(tt["theo_lop"][c]["AP50"]))) < 1e-12
                       for c in tt["theo_lop"]), f"{mh.get('macro')} vs {tt['macro']}")
        _check("B4c ket qua ghi quy tac GT + danh tinh du lieu (sha256 nhan, doi chieu hash)",
               kq.get("quy_tac_gt", {}).get("vung_bo_qua", "").startswith("crowd")
               and kq["dataset"]["dataset_da_xac_minh"]["nhan_sha256"] == sha(fx["nhan"]))
        rc2 = chay(out, sha(ck))
        _check("B4d thu muc ra da ton tai -> exit 1 (khong ghi de)", rc2 == 1, f"rc {rc2}")
        n_goi = len(goi)
        rc3 = chay(tmp / "ra_sai_sha", "f" * 64)
        _check("B4e sha256 checkpoint sai -> exit 2, KHONG goi du doan, khong tao thu muc",
               rc3 == 2 and len(goi) == n_goi and not (tmp / "ra_sai_sha").exists(), f"rc {rc3}")
        rc4 = chay(tmp / "ra_ten", sha(ck), ten_sha="khac")
        _check("B4f ten --model / --sha256 khong khop -> exit 2", rc4 == 2 and not (tmp / "ra_ten").exists(), f"rc {rc4}")

        # B5 (vong 44, R43-N1): nguong van hanh HIEU LUC. Nguong scorer THUC dung = gia tri `danh_gia` that tra ve
        # (`van_hanh`), bat bang ham boc; metadata, scorer va chan doan phai cung mot gia tri; QUY_TAC khong doi.
        goc_dg = m.danh_gia
        nguong_scorer = []

        def dg_theo_doi(*a, **kw):
            r_ = goc_dg(*a, **kw)
            nguong_scorer.append(dict(r_["van_hanh"]))
            return r_
        anh_chup = json.dumps(ev.QUY_TAC, sort_keys=True)
        m.danh_gia = dg_theo_doi
        try:
            def chay_nguong(ten_out, nguong):
                o = tmp / ten_out
                rc_ = chay(o, sha(ck), them=[] if nguong is None else ["--conf-van-hanh", str(nguong)])
                k_ = json.loads((o / "ket_qua.json").read_text(encoding="utf-8")) if (o / "ket_qua.json").exists() else {}
                return rc_, k_, (nguong_scorer[-1] if nguong_scorer else None)

            def khop(k_, ns, mong):
                qt_ = k_.get("quy_tac", {}).get("van_hanh", {})
                mh_ = k_.get("mo_hinh", {}).get("gia", {})
                return (qt_.get("conf") == mong and ns is not None and ns.get("conf") == mong and qt_ == ns
                        and mh_.get("van_hanh", {}).get("conf") == mong
                        and f"so_det_conf_ge_{mong}_tren_anh_khong_gt" in mh_.get("chan_doan", {}))

            def mo_ta(rc_, k_, ns):
                return f"rc {rc_}, metadata {k_.get('quy_tac', {}).get('van_hanh')}, scorer {ns}"

            nguong_scorer.clear()
            rc5a, k5a, ns5a = chay_nguong("ra_b5a", None)
            _check("B5a mac dinh (khong --conf-van-hanh): metadata quy_tac.van_hanh = nguong scorer = 0.25",
                   rc5a == 0 and khop(k5a, ns5a, 0.25), mo_ta(rc5a, k5a, ns5a))
            nguong_scorer.clear()
            rc5b, k5b, ns5b = chay_nguong("ra_b5b", 0.5)
            tt5 = ev.danh_gia(gt, dd, 0.5)
            mh5, mh5a = k5b.get("mo_hinh", {}).get("gia", {}), k5a.get("mo_hinh", {}).get("gia", {})
            vh_khop = bool(mh5) and all(mh5["theo_lop"][c]["van_hanh"] == ev._json_safe(tt5["theo_lop"][c]["van_hanh"])
                                        for c in tt5["theo_lop"])
            khac_025 = bool(mh5) and bool(mh5a) and any(mh5["theo_lop"][c]["van_hanh"] != mh5a["theo_lop"][c]["van_hanh"]
                                                        for c in tt5["theo_lop"])
            _check("B5b --conf-van-hanh 0.5: metadata = nguong scorer = 0.5; P/R + chan doan = goi truc tiep voi 0.5 "
                   "(khac ket qua 0.25)", rc5b == 0 and khop(k5b, ns5b, 0.5) and vh_khop and khac_025
                   and mh5.get("chan_doan") == m.chan_doan(gt, dd, 0.5), mo_ta(rc5b, k5b, ns5b))
            nguong_scorer.clear()
            rc5c1, k5c1, ns5c1 = chay_nguong("ra_b5c_1", 0.4)
            rc5c2, k5c2, ns5c2 = chay_nguong("ra_b5c_2", None)
            _check("B5c 2 lan goi lien tiep trong cung tien trinh (0.4 roi mac dinh): moi lan dung nguong cua no; QUY_TAC "
                   "dung chung + mac dinh CLI van 0.25",
                   rc5c1 == 0 and rc5c2 == 0 and khop(k5c1, ns5c1, 0.4) and khop(k5c2, ns5c2, 0.25)
                   and json.dumps(ev.QUY_TAC, sort_keys=True) == anh_chup and ev.QUY_TAC["van_hanh"]["conf"] == 0.25
                   and m.QUY_TAC is ev.QUY_TAC,
                   f"lan 1: {mo_ta(rc5c1, k5c1, ns5c1)} | lan 2: {mo_ta(rc5c2, k5c2, ns5c2)} | "
                   f"QUY_TAC.van_hanh {ev.QUY_TAC['van_hanh']}")
        finally:
            m.danh_gia = goc_dg
    finally:
        m.du_doan_ultralytics = goc


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="test_bdd_val_") as td:
        try:
            nhom_b(Path(td))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            _check(f"loi khong mong doi {type(e).__name__}: {e}", False)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} PASS")
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
