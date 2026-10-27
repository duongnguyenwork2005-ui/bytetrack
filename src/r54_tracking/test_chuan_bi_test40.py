"""
test_chuan_bi_test40.py -- Fixture cho wrapper R54 (khong suy luan that, khong GPU).

Kiem: anh xa COCO/mixed5 (thieu/trung ten), loc strict >, max_det, pixel to xam, frame khong GT van qua
tracker THAT va evaluator THAT, PF seed + K1/K2 tat, lap PF trung byte, cham lai trung byte; ca loi: sai hash,
thieu GT, sai timeline, D4 co GT, duong dan thoat vung ghi, tu choi ghi de; CLI THAT cua wrapper (buoc la,
ten phien xau, thieu khoa, khoa bi sua, ghi de) va chuoi tien-kiem -> khoa -> kiem-khoa tren du lieu that
trong vung tests/. Detector la mo hinh GIA (tim hinh chu nhat trang), chay qua det_cache.build_video_yolo.
Hoi quy review R54A: F1 (H1: time_s du 2 x 128 frame), F2 (I2/I3: ma Ultralytics/TrackEval trong khoa, kiem
truoc suy luan chan backend/nguon/cau hinh/thu vien lech), F3 (I2: kiem-khoa mac dinh chi doc, --luu), F4 (I4:
dau vao doi sau tien-kiem -> khoa tu choi), F5 (D1: hau_kiem bat sua cache/metric/MOT lap/cham lai/GT smoke);
vong 2 F2 (I6: khoi tao DetectionPredictor that nap utils.events -> trong khoa; khoa thieu/sai hash -> chan).

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r54_tracking/test_chuan_bi_test40.py --buoc fixture
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import chuan_bi_test40 as W  # noqa: E402

GOC = W.GOC
VUNG_TESTS = W.VUNG_R54 / "tests"
PY = sys.executable
KET_QUA: list[dict] = []


def ca(ten: str):
    def deco(fn):
        def chay(*a, **k):
            t0 = time.time()
            try:
                ct = fn(*a, **k)
                KET_QUA.append({"ten": ten, "dat": True, "chi_tiet": ct, "giay": round(time.time() - t0, 2)})
                print(f"  [DAT ] {ten}")
            except Exception as e:  # noqa: BLE001
                KET_QUA.append({"ten": ten, "dat": False, "loi": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-2500:], "giay": round(time.time() - t0, 2)})
                print(f"  [LOI ] {ten}: {type(e).__name__}: {e}")
        return chay
    return deco


def phai_loi(fn, kieu, chua: str = ""):
    try:
        fn()
    except kieu as e:
        if chua and chua not in str(e):
            raise AssertionError(f"loi dung kieu nhung thieu '{chua}': {e}")
        return str(e)
    raise AssertionError(f"khong nem {kieu.__name__}")


# ---------------------------------------------------------------------------
# Snapshot de kiem khong ghi ngoai vung
# ---------------------------------------------------------------------------
def snapshot() -> dict:
    out = {}

    def them(p: Path, de_quy: bool):
        if not p.exists():
            return
        it = p.rglob("*") if de_quy else p.iterdir()
        for q in it:
            try:
                st = q.stat()
                out[str(q)] = (st.st_size, st.st_mtime_ns if q.is_file() else 0)
            except OSError:
                pass
    them(GOC, False)
    for d in ("src", "src/r54_tracking", "src/mot_datasets", "configs/phase2/r54_tracking", "data/interim/det_cache",
              "data/processed/trackers", "results", "results/trackeval", "results/phase2", "results/phase2/r54_tracking",
              "data/processed", "handoff/logs/review54"):
        them(GOC / d, False)
    for v in W.MAU_SMOKE["videos"]:
        them(GOC / "data/processed/DETRAC-test" / v, True)
    for d in (Path.home() / "AppData/Roaming/Ultralytics", Path.home() / ".config/Ultralytics"):
        them(d, True)
    return out


# ---------------------------------------------------------------------------
# Fixture du lieu + mo hinh gia
# ---------------------------------------------------------------------------
TEN_COCO = {i: f"lop{i}" for i in range(80)} | {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TEN_LOP5 = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle"}
VUNG_IGN = (800.0, 20.0, 100.0, 80.0)
N_FX = 20


def hop(i: int, video_k: int) -> dict:
    """Hinh chu nhat cua frame i: A (car), B (truck), C (bicycle), D (trong vung bo qua)."""
    return {"A": (100 + 10 * i + 30 * video_k, 200, 80, 50), "B": (600 - 8 * i, 350, 90, 60),
            "C": (300, 100, 40, 40), "D": (820, 30, 40, 40)}


class _T:
    def __init__(self, a): self.a = a
    def cpu(self): return self
    def numpy(self): return self.a


class _R:
    def __init__(self, a): self.boxes = type("B", (), {"data": _T(a)})()


class MoHinhGia:
    """Tim hinh chu nhat trang; lop theo vi tri y; ghi nhan pixel vung bo qua khi nhan anh."""

    def __init__(self, loai: str):
        self.loai = loai
        self.names = dict(TEN_COCO if loai == "coco80" else TEN_LOP5)
        self.id = ({"car": 2, "truck": 7, "bicycle": 1} if loai == "coco80" else {"car": 0, "truck": 1, "bicycle": 4})
        self.so_anh = 0
        self.vung_xam_dung = 0

    def predict(self, img, conf, iou, imgsz, classes, max_det, agnostic_nms, augment, device, verbose, **kw):
        import cv2
        import numpy as np
        self.so_anh += 1
        x, y, w, h = (int(v) for v in VUNG_IGN)
        if (img[y:y + h, x:x + w] == 114).all():
            self.vung_xam_dung += 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnt, _ = cv2.findContours((g > 200).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = []
        for c in cnt:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw * bh < 200:
                continue
            ten, cf = ("bicycle", 0.8) if by < 150 else (("car", 0.9) if by < 300 else ("truck", 0.6))
            rows.append([bx, by, bx + bw, by + bh, cf, self.id[ten]])
        a = np.array(sorted(rows, key=lambda r: -r[4]), dtype=np.float32).reshape(-1, 6)
        a = a[a[:, 4] > conf][:max_det]
        return [_R(a)]


def tao_fixture(root: Path, bien_the: str = "") -> tuple[dict, W.Nguon, dict]:
    """Du lieu gia cung bo cuc repo (2 video x 20 anh). Tra ve (giao thuc fixture, Nguon, ten->loai)."""
    import cv2
    import numpy as np
    gt = copy.deepcopy(W.doc_giao_thuc())
    videos = list(W.MAU_SMOKE["videos"])
    n = gt["nguon"]
    img_root, gt_root = root / n["thu_muc_anh"], root / n["thu_muc_gt"]
    rows_hash, d4_rows, xml, anh_fp = [], [], [], []
    d4 = {videos[0]: set(), videos[1]: {18, 19, 20}}
    khong_box = {videos[0]: {8, 9, 10}, videos[1]: set()}
    for k, v in enumerate(videos):
        (img_root / v).mkdir(parents=True, exist_ok=True)
        (gt_root / v / "gt").mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(1, N_FX + 1):
            im = np.full((540, 960, 3), 60, np.uint8)
            for nm, (x, y, w, h) in hop(i, k).items():
                im[y:y + h, x:x + w] = 255
            p = img_root / v / f"img{i:05d}.jpg"
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows_hash.append({"rel": f"{v}/{p.name}", "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                              "sha256": W.sha256_file(p), "w": 960, "h": 540})
            if i in d4[v]:
                d4_rows.append({"rel": f"{v}/{p.name}", "video": v, "frame": i})
                continue
            if i in khong_box[v]:
                continue
            for tid, nm in ((1, "A"), (2, "B")):
                x, y, w, h = hop(i, k)[nm]
                lines.append(f"{i},{tid},{x},{y},{w},{h},1,1,1.0\n")
        if bien_the == "d4_co_gt" and k == 1:
            lines.append("19,1,10,10,20,20,1,1,1.0\n")
        if bien_the != "thieu_gt" or k != 1:
            (gt_root / v / "gt" / "gt.txt").write_text("".join(lines), encoding="utf-8")
        sl = N_FX + (1 if bien_the == "sai_seqlength" and k == 0 else 0)
        (gt_root / v / "seqinfo.ini").write_text(
            f"[Sequence]\nname={v}\nimDir=img1\nframeRate=25\nseqLength={sl}\nimWidth=960\nimHeight=540\nimExt=.jpg\n",
            encoding="utf-8")
        (gt_root / v / "gt" / "ignored_regions.txt").write_text(",".join(str(x) for x in VUNG_IGN) + "\n", encoding="utf-8")
        xml.append({"video": v, "so_frame": N_FX - len(d4[v]), "so_box": len(lines), "so_vung_bo_qua": 1})
        anh_fp.append({"video": v, "so_anh": N_FX, "frame_dau": 1, "frame_cuoi": N_FX, "so_anh_loai_D4": len(d4[v])})
    if bien_the == "sai_timeline":          # bo mot dong giua bang hash -> anh khong lien tuc
        rows_hash = [r for r in rows_hash if r["rel"] != f"{videos[0]}/img00005.jpg"]
    if bien_the == "sai_hash_anh":          # sua byte mot anh mau sau khi bang hash da ghi
        p = img_root / videos[0] / "img00003.jpg"
        b = bytearray(p.read_bytes()); b[-10] ^= 0xFF; p.write_bytes(bytes(b))

    def ghi(rel: str, data: str) -> str:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8", newline="\n")
        return W.sha256_file(p)

    man = json.dumps({"videos": videos})
    s_man = ghi(n["manifest"]["rel"], man)
    hdr = "rel,size,mtime_ns,sha256,w,h\n"
    s_hash = ghi(n["bang_hash"]["rel"], hdr + "".join(f"{r['rel']},{r['size']},{r['mtime_ns']},{r['sha256']},960,540\n"
                                                      for r in rows_hash))
    s_d4 = ghi(n["loai_d4"]["rel"], "rel,video,frame\n" + "".join(f"{r['rel']},{r['video']},{r['frame']}\n" for r in d4_rows))
    fp = {"van_tay": {"manifest": {"videos": videos}, "xml": xml, "anh": {"theo_video": anh_fp},
                      "tong_hop": {"danh_sach_anh_co_thu_tu_sha256": W.sha256_bytes("".join(f"{r['rel']}\n" for r in rows_hash).encode()),
                                   "noi_dung_anh_sha256_tong_hop": W.sha256_ds((r["rel"], r["sha256"]) for r in rows_hash)}}}
    s_fp = ghi(n["fingerprint"]["rel"], json.dumps(fp))
    loai = {}
    for det, spec in gt["detector"].items():
        p = root / spec["checkpoint"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"checkpoint gia {det}".encode())
        spec["sha256"] = W.sha256_file(p)
        loai[det] = spec["loai_checkpoint"]
    khoa53 = {"mo_hinh": {d: {"checkpoint": s["checkpoint"], "sha256": s["sha256"], "loai_checkpoint_ky_vong": s["loai_checkpoint"]}
                          for d, s in gt["detector"].items()},
              "du_lieu": {"detrac_test": {"file": {k53: {"duong_dan": n[k]["rel"], "sha256": s} for k53, k, s in (
                  ("manifest", "manifest", s_man), ("fingerprint", "fingerprint", s_fp), ("hash_anh", "bang_hash", s_hash),
                  ("loai_d4", "loai_d4", s_d4))}}}}
    s_k = ghi(n["khoa_r53"]["rel"], json.dumps(khoa53))
    ghi(n["sha256sums_r53"]["rel"], f"{s_k}  {n['khoa_r53']['rel']}\n")
    for k, s in (("manifest", s_man), ("fingerprint", s_fp), ("bang_hash", s_hash), ("loai_d4", s_d4), ("khoa_r53", s_k)):
        n[k]["sha256"] = s
    if bien_the == "sai_hash_gt_khoa":      # giao thuc ky vong hash bang khac file tren dia
        n["bang_hash"]["sha256"] = "0" * 64
    gt["tap_du_lieu"].update({"so_video": 2, "so_anh": 2 * N_FX, "so_anh_D4": 3, "so_anh_co_frame_xml": 2 * N_FX - 3})
    return gt, W.Nguon(gt, root), loai


def names_gia(loai: dict):
    def f(p: Path):
        det = next(d for d in loai if p.as_posix().endswith(W.doc_giao_thuc()["detector"][d]["checkpoint"]))
        return dict(TEN_COCO if loai[det] == "coco80" else TEN_LOP5)
    return f


MAU_FX = {"videos": W.MAU_SMOKE["videos"], "frame_dau": 1, "frame_cuoi": N_FX}


# ---------------------------------------------------------------------------
# Cac ca
# ---------------------------------------------------------------------------
@ca("A1 anh xa coco80 giu ID, lop5 -> COCO theo ten, khong doi box/conf/thu tu")
def ca_anh_xa():
    import numpy as np
    assert W.bang_anh_xa_coco(TEN_COCO, "coco80") is None
    b = W.bang_anh_xa_coco(TEN_LOP5, "lop5")
    assert b == {0: 2, 1: 7, 2: 5, 3: 3, 4: 1}, b
    a = np.array([[1, 2, 3, 4, 0.9, 0], [5, 6, 7, 8, 0.5, 4], [9, 9, 9, 9, 0.3, 1], [1, 1, 2, 2, 0.2, 2]], np.float32)
    m = W.doi_id_sang_coco(a, b)
    assert np.array_equal(m[:, :5], a[:, :5]) and m[:, 5].tolist() == [2, 1, 7, 5]
    assert np.array_equal(W.doi_id_sang_coco(a, None), a)
    phai_loi(lambda: W.doi_id_sang_coco(np.array([[0, 0, 1, 1, 0.5, 9]], np.float32), b), W.LoiDauVao, "khong co trong bang")
    return {"bang_lop5": b}


@ca("A2 thieu / trung ten lop, sai loai checkpoint -> loi")
def ca_anh_xa_loi():
    import detector_classes as dc
    thieu = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"}
    trung = {0: "car", 1: "truck", 2: "bus", 3: "motorcycle", 4: "bicycle", 5: "car"}
    hoan_vi = {0: "bus", 1: "car", 2: "truck", 3: "motorcycle", 4: "bicycle"}
    e1 = phai_loi(lambda: W.bang_anh_xa_coco(thieu, "lop5"), dc.LoiAnhXaLop, "thieu")
    e2 = phai_loi(lambda: W.bang_anh_xa_coco(trung, "lop5"), dc.LoiAnhXaLop, "trung")
    e3 = phai_loi(lambda: W.bang_anh_xa_coco(hoan_vi, "lop5"), W.LoiDauVao, "loai checkpoint")
    e4 = phai_loi(lambda: W.bang_anh_xa_coco(TEN_LOP5, "coco80"), W.LoiDauVao, "loai checkpoint")
    return [e1, e2, e3, e4]


@ca("B1 loc luc load: conf strict >, lop [2,3,5,7], max_det 300 (VideoDetections + mo_cache that)")
def ca_loc(lan: Path):
    import numpy as np
    import det_cache
    d = lan / "cache_loc"
    spec = det_cache.DetectorSpec(backend="gt-synthetic", conf=0.05, max_det=3000, classes=None, dataset="detrac")
    det_cache.write_meta(d, spec, mode="create")
    f1 = np.array([[0, 0, 10, 10, 0.9, 2], [0, 0, 10, 10, 0.2500001, 3], [0, 0, 10, 10, 0.25, 5],
                   [0, 0, 10, 10, 0.2, 7], [0, 0, 10, 10, 0.1000001, 1], [0, 0, 10, 10, 0.1, 2],
                   [0, 0, 10, 10, 0.06, 7]], np.float32)
    f2 = np.array([[i, i, i + 5, i + 5, 0.99 - i * 1e-3, 2] for i in range(305)], np.float32)
    arrays = {"frame_idx": np.array([1, 2], np.int32), "n_per_frame": np.array([len(f1), len(f2)], np.int32),
              "unreadable": np.zeros(2, bool), "truncated": np.zeros(2, bool), "time_s": np.array([0, 0.04]),
              "data": np.concatenate([f1, f2]), "orig_shape": np.array([540, 960], np.int32)}
    det_cache.save_video(d, "V", arrays, spec=spec)
    c25 = W.mo_cache(d, "V", 0.25, [2, 3, 5, 7], 300)
    c10 = W.mo_cache(d, "V", 0.10, [2, 3, 5, 7], 300)
    a25, a10 = c25.frame(0), c10.frame(0)
    assert a25[:, 4].tolist() == [np.float32(0.9), np.float32(0.2500001)], a25
    # c10: 0.25 (lop 5) va 0.2 (lop 7) qua; 0.1000001 lop 1 (bicycle) bi loai theo lop; 0.1 bi loai vi strict >
    assert a10[:, 5].tolist() == [2, 3, 5, 7], a10
    b = c25.frame(1)
    assert len(b) == 300 and np.array_equal(b, f2[:300])
    phai_loi(lambda: W.mo_cache(d, "V", 0.04, [2, 3, 5, 7], 300), W.LoiDauVao, "khong tai lap")
    phai_loi(lambda: W.mo_cache(d, "V", 0.25, [2, 3, 5, 7], 3001), W.LoiDauVao, "khong tai lap")
    phai_loi(lambda: det_cache.write_meta(d, spec, mode="create"), RuntimeError, "da ton tai")
    return {"c25": a25.tolist(), "c10": a10[:, 4:].tolist()}


@ca("C1 pixel to xam = 114 trong vung, ngoai vung giu nguyen (ham hien hanh)")
def ca_to_xam():
    import numpy as np
    import det_cache
    import config
    img = (np.arange(540 * 960 * 3) % 251).astype(np.uint8).reshape(540, 960, 3)
    goc = img.copy()
    out = det_cache._mask_regions(img, [VUNG_IGN, (950.4, 530.6, 50, 50)])
    x, y, w, h = (int(v) for v in VUNG_IGN)
    assert tuple(config.IGNORE_MASK_COLOR_BGR) == (114, 114, 114)
    assert (out[y:y + h, x:x + w] == 114).all() and (out[531:, 950:] == 114).all()
    m = np.ones(out.shape[:2], bool); m[y:y + h, x:x + w] = False; m[531:, 950:] = False
    assert np.array_equal(out[m], goc[m])
    return {"so_pixel_xam": int((~m).sum())}


@ca("D1 smoke fixture: 3 detector gia -> cache -> 36 to hop x 2 video -> cham THAT; lap PF; cham lai; hau_kiem")
def ca_smoke_fixture(lan: Path):
    import numpy as np
    import pandas as pd
    root = lan / "fx_smoke"
    gt, ng, loai = tao_fixture(root)
    phien = lan / "smoke_fx"
    phien.mkdir()
    models = {}

    def mo(det, ck):
        models[det] = MoHinhGia(loai[det])
        return models[det]

    def spec_fx(s):
        s.versions = {"fixture": "1"}
        s.created = "fixture"

    kq = W.chay_smoke(gt, ng, phien, mo, spec_fx, mau=MAU_FX, so_khop_dataset=False, log=lambda *a: None)
    assert kq["dat"], kq["loi"]
    assert kq["so_run"] == 36 and kq["so_luot_video"] == 72
    assert all(m.so_anh == 2 * N_FX and m.vung_xam_dung == 2 * N_FX for m in models.values())
    for det, c in kq["cache"].items():
        for v, cv in c["video"].items():
            assert 1 in cv["cls_coco"] and (4 in cv["cls_goc"]) == (loai[det] == "lop5"), (det, cv)
            z = np.load(phien / "cache" / det / f"{v}.npz")
            assert not ((z["data"][:, 0] >= 800) & (z["data"][:, 1] < 100)).any()   # D trong vung xam khong bi phat hien
    # coco gia va mixed5 gia (sau anh xa) -> MOT trung byte
    for r in ("c25-cv", "c10-ekf", "c25-pf500-s1"):
        a = (phien / "trackers" / W.SPLIT_SMOKE / f"r54-yolov8n_coco-{r}" / "data" / f"{MAU_FX['videos'][0]}.txt").read_bytes()
        b = (phien / "trackers" / W.SPLIT_SMOKE / f"r54-yolo11n_mixed5-{r}" / "data" / f"{MAU_FX['videos'][0]}.txt").read_bytes()
        assert a == b and len(a) > 0, r
    # frame khong GT (8-10 video 1, D4 18-20 video 2) van qua tracker + evaluator
    run = kq["runs"]["r54-yolov8n_coco-c25-cv"]
    for i in run["video"]:
        assert i["n_buoc_timing"] == N_FX and i["n_steps"] == [1]
    mot = pd.read_csv(phien / "trackers" / W.SPLIT_SMOKE / "r54-yolov8n_coco-c25-cv" / "data" / f"{MAU_FX['videos'][1]}.txt", header=None)
    assert set(mot[0]) >= {18, 19, 20}
    pv = pd.read_csv(phien / "trackeval" / "r54-yolov8n_coco-c25-cv" / "per_video_metrics.csv")
    assert (pv["FP"] > 0).all() and pv["HOTA"].between(0, 1).all(), pv
    # PF: seed, N, K1/K2
    for s in W.SEEDS_PF:
        for i in kq["runs"][f"r54-yolov8n_mixed5-c10-pf500-s{s}"]["video"]:
            assert (i["seed_thuc_dung"], i["num_particles"], i["k1_optimal_proposal"], i["k2_roughening"]) == (s, 500, False, 0.0), i
    assert len(kq["lap_pf"]) == 9 and all(x["trung_byte"] for x in kq["lap_pf"].values())
    assert len(kq["cham_lai"]) == 12 and all(x["file_trung_byte"] for x in kq["cham_lai"].values())
    kq["exit"] = 0
    W.ghi_json(phien / "ket_qua_smoke.json", kq, [phien])
    assert W.hau_kiem(phien, gt, MAU_FX) == []
    # hau_kiem phat hien tung kieu sua (moi ca: sua -> phai bi tu choi -> khoi phuc nguyen byte)
    v0 = MAU_FX["videos"][0]
    pf = kq["lap_pf"]["r54-yolov8n_mixed5-c25-pf500-s1"]
    ca_sua = {
        "npz cache": (phien / "cache" / "yolov8n_coco" / f"{v0}.npz", "sua", "sha256 lech"),
        "meta.json cache": (phien / "cache" / "yolov8n_mixed5" / "meta.json", "sua", "meta.json"),
        "json video cache": (phien / "cache" / "yolo11n_mixed5" / f"{v0}.json", "sua", f"{v0}.json"),
        "xoa raw_fields": (phien / "trackeval" / "r54-yolov8n_coco-c25-cv" / "raw_fields.json", "xoa", "raw_fields.json"),
        "sua per_video": (phien / "trackeval" / "r54-yolo11n_mixed5-c10-ukf" / "per_video_metrics.csv", "sua",
                          "per_video_metrics.csv"),
        "sua MOT lap PF": (phien / "lap_pf" / "yolov8n_mixed5" / "r54-yolov8n_mixed5-c25-pf500-s1" / "data" / f"{pf['video']}.txt",
                           "sua", "lap r54-yolov8n_mixed5-c25-pf500-s1"),
        "xoa cham lai": (phien / "cham_lai" / "r54-yolov8n_coco-c25-ekf" / "combined_metrics.csv", "xoa", "cham lai"),
        "sua GT smoke": (phien / "gt" / W.SPLIT_SMOKE / v0 / "gt" / "gt.txt", "sua", "gt smoke"),
    }
    bat = {}
    for ten, (p, kieu, chua) in ca_sua.items():
        b = p.read_bytes()
        if kieu == "xoa":
            p.unlink()
        else:
            p.write_bytes(b + b"x")
        loi = W.hau_kiem(phien, gt, MAU_FX)
        p.write_bytes(b)
        assert any(chua in x for x in loi), (ten, loi)
        bat[ten] = [x for x in loi if chua in x][:2]
    assert W.hau_kiem(phien, gt, MAU_FX) == []
    return {"so_run": kq["so_run"], "thoi_gian": kq["thoi_gian"], "hota_cv_c25": run["cham"]["combined"]["HOTA"],
            "hau_kiem_bat_duoc": bat}


@ca("D2 chay_tracker ghi Y HET run_tracking.run_video_to_dir (cv, pf500 s0)")
def ca_ghi_giong(lan: Path):
    import run_tracking as rt
    d = lan / "cache_loc"
    reg = rt.tracker_registry()
    out = {}
    for ten, yaml, seed in (("cv", "bytetrack.yaml", None), ("pf", "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml", 0)):
        cfg = rt.load_tracker_cfg(W.yaml_arg(yaml), {"track_buffer": 30})
        a, b = lan / "ghi_giong" / f"{ten}_a", lan / "ghi_giong" / f"{ten}_b"
        W.chay_tracker(a, "V", W.mo_cache(d, "V", 0.25, [2, 3, 5, 7], 300), cfg, seed, reg, True, True)
        rt.run_video_to_dir(b, "V", W.mo_cache(d, "V", 0.25, [2, 3, 5, 7], 300), cfg, 0.04, "skip", seed, reg,
                            assoc_log=True, pf_diag=True)
        for sub in ("data/V.txt", "data_cls/V.csv"):
            assert (a / sub).read_bytes() == (b / sub).read_bytes(), (ten, sub)
        ta, tb = (a / "timing/V.csv").read_text().splitlines(), (b / "timing/V.csv").read_text().splitlines()
        assert [x.split(",")[:4] for x in ta] == [x.split(",")[:4] for x in tb]
        out[ten] = sorted(str(p.relative_to(a)) for p in a.rglob("*") if p.is_file())
    return out


@ca("E1 tien_kiem fixture hop le -> dat; bang 2 dong; frame khong GT ngoai D4 duoc bao")
def ca_tien_kiem_dat(lan: Path):
    gt, ng, loai = tao_fixture(lan / "fx_tk")
    kq = W.tien_kiem(gt, ng, doc_names=names_gia(loai), mau=MAU_FX)
    assert kq["ket_luan"] == "dat", kq["lech"]
    assert len(kq["bang"]) == 2 and kq["bang"][0]["so_frame_khong_gt_ngoai_D4"] == 3 and kq["bang"][1]["so_anh_D4"] == 3
    assert kq["mau_smoke"]["so_anh"] == 2 * N_FX and all(a["khop"] for a in kq["anh_tu_bam_lai"])
    assert kq["cau_hinh_hieu_luc"]["bo_loc"]["pf500"]["cfg_hieu_luc"]["track_buffer"] == 30
    return {"tong": kq["tong"], "yaml": {k: v["yaml_resolve"] for k, v in kq["cau_hinh_hieu_luc"]["bo_loc"].items()}}


def _tk_loi(lan: Path, bien_the: str, chua: str):
    gt, ng, loai = tao_fixture(lan / f"fx_{bien_the}", bien_the)
    try:
        kq = W.tien_kiem(gt, ng, doc_names=names_gia(loai), mau=MAU_FX, kiem_cau_hinh=False)
    except W.LoiTienKiem as e:
        assert chua in str(e), e
        return {"exception": str(e)}
    assert kq["ket_luan"] == "khong_dat" and any(chua in x for x in kq["lech"]), kq["lech"]
    return {"lech": kq["lech"][:6]}


@ca("E2 sai hash bang (so voi khoa) -> khong dat")
def ca_sai_hash(lan): return _tk_loi(lan, "sai_hash_gt_khoa", "sha256 bang_hash")


@ca("E3 sai hash anh mau (tu bam lai) -> khong dat")
def ca_sai_hash_anh(lan): return _tk_loi(lan, "sai_hash_anh", "sha256 tu bam")


@ca("E4 thieu GT -> khong dat")
def ca_thieu_gt(lan): return _tk_loi(lan, "thieu_gt", "thieu gt")


@ca("E5 sai timeline (bang hash mat 1 anh) -> khong dat")
def ca_sai_timeline(lan): return _tk_loi(lan, "sai_timeline", "timeline")


@ca("E6 seqLength != so anh -> khong dat")
def ca_sai_seqlength(lan): return _tk_loi(lan, "sai_seqlength", "seqLength")


@ca("E7 anh D4 co dong GT -> khong dat")
def ca_d4_co_gt(lan): return _tk_loi(lan, "d4_co_gt", "anh D4 lai co dong GT")


@ca("F1 duong dan thoat vung ghi + tu choi ghi de (ham)")
def ca_vung_ghi(lan: Path):
    v = lan / "vung"
    v.mkdir()
    phai_loi(lambda: W.kiem_trong_vung(GOC / "src" / "x.txt", [v]), W.LoiDauVao, "ngoai vung")
    phai_loi(lambda: W.kiem_trong_vung(v / ".." / ".." / "x", [v]), W.LoiDauVao, "ngoai vung")
    phai_loi(lambda: W.tao_phien(v, "..", "x"), W.LoiDauVao, "khong hop le")
    phai_loi(lambda: W.tao_phien(v, "a/../../b", "x"), W.LoiDauVao, "khong hop le")
    p = W.tao_phien(v, "p1", "x")
    phai_loi(lambda: W.tao_phien(v, "p1", "x"), W.LoiGhiDe, "ghi de")
    W.ghi_json(p / "a.json", {"a": 1}, [p])
    phai_loi(lambda: W.ghi_json(p / "a.json", {"a": 2}, [p]), W.LoiGhiDe)
    phai_loi(lambda: W.ghi_json(GOC / "results" / "x.json", {}, [p]), W.LoiDauVao, "ngoai vung")
    phai_loi(lambda: W.cham(v, v, "DETRAC-x", "r", {}, GOC / "results" / "trackeval", [v]), W.LoiDauVao, "ngoai vung")
    assert json.loads((p / "a.json").read_text(encoding="utf-8")) == {"a": 1}
    return "ok"


@ca("G1 giao thuc that: 36 run, 3 detector, 40 video; ban sua them nhanh bi tu choi")
def ca_giao_thuc():
    gt = W.doc_giao_thuc()
    assert W.kiem_giao_thuc(gt) == []
    m = W.ma_tran(gt)
    assert len(m) == 36 and len({r["detector"] for r in m}) == 3
    g2 = copy.deepcopy(gt); g2["tracker"]["bo_loc"][3]["seeds"] = [0, 1, 2, 3]
    g3 = copy.deepcopy(gt); g3["tracker"]["bo_loc"].append({"ten": "pf100", "yaml": "x", "seeds": [0]})
    g4 = copy.deepcopy(gt); g4["smoke"]["frame_cuoi"] = 256
    assert all(W.kiem_giao_thuc(g) for g in (g2, g3, g4))
    return {"run_dau": m[0]["run"], "run_cuoi": m[-1]["run"]}


@ca("H1 duong that: khung mau DU 2 x 128 frame khop DetracDataset.frames (index, path, time_s bang ==); "
    "vung bo qua khop ham hien hanh; khong suy luan")
def ca_duong_that():
    from mot_datasets.detrac import DetracDataset
    import baseline_track
    gt = W.doc_giao_thuc()
    ng = W.Nguon(gt)
    bang = W.doc_bang_hash(ng.bang_hash)
    out = {}
    for v in W.MAU_SMOKE["videos"]:
        fr, hs = W.khung_mau(ng, v, {"videos": [v], "frame_dau": W.MAU_SMOKE["frame_dau"],
                                     "frame_cuoi": W.MAU_SMOKE["frame_cuoi"]}, bang[v], True)
        assert [f.index for f in fr] == list(range(1, 129)) and len(hs) == 128
        ref = DetracDataset(img_root=ng.thu_muc_anh).frames(v, "test")[:128]
        assert [f.time_s for f in fr] == [f.time_s for f in ref]
        # hoi quy F1: (i-1)/fps khac (i-1)*(1/fps) o mot so frame (vd 36) -> cach cu se bi tu choi
        so_khac_cach_cu = sum(1 for f in ref if (f.index - 1) / W.FPS != f.time_s)
        r = W.doc_vung(ng.vung_bo_qua(v))
        assert r == DetracDataset(img_root=ng.thu_muc_anh).ignored_regions(v, "test")
        assert [tuple(map(float, x)) for x in baseline_track.load_ignored_regions(v, "DETRAC-test")] == r
        out[v] = {"so_frame": len(fr), "frame_cuoi": fr[-1].index, "time_s_36": fr[35].time_s,
                  "so_frame_cach_cu_lech": so_khac_cach_cu, "vung": len(r)}
    assert all(x["so_frame_cach_cu_lech"] > 0 for x in out.values()), out
    return out


def cli(args: list[str], env_goc: Path | None, timeout=900) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != W.ENV_TEST}
    if env_goc is not None:
        env[W.ENV_TEST] = str(env_goc)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([PY, str(DIR / "chuan_bi_test40.py"), *args], cwd=GOC, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)


@ca("I1 CLI that: buoc la / ten phien xau / env ngoai tests / thieu khoa / thieu tien-kiem -> nonzero, khong ghi")
def ca_cli_loi(lan: Path):
    g = lan / "cli_loi"
    out = {}
    for ten, args, env, ma in (("full", ["--buoc", "full"], g, 2),
                               ("lan_xau", ["--buoc", "tien-kiem", "--lan", "../thoat"], g, 2),
                               ("khoa_xau", ["--buoc", "kiem-khoa", "--khoa", "..\\x"], g, 2),
                               ("env_ngoai", ["--buoc", "kiem-khoa"], W.VUNG_R54 / "chuan_bi", 2),
                               ("smoke_khong_khoa", ["--buoc", "smoke"], g, 2),
                               ("kiem_khoa_khong_khoa", ["--buoc", "kiem-khoa"], g, 2),
                               ("khoa_khong_tien_kiem", ["--buoc", "khoa"], g, 2),
                               ("hau_kiem_khong_smoke", ["--buoc", "hau-kiem"], g, 2)):
        r = cli(args, env)
        out[ten] = {"exit": r.returncode, "stderr": r.stderr.strip()[-300:]}
        assert r.returncode == ma, (ten, r.returncode, r.stdout[-500:], r.stderr[-800:])
    assert not g.exists(), "CLI loi da ghi"
    assert not (W.VUNG_R54 / "chuan_bi" / "kiem_khoa").exists()
    return out


def liet_ke(d: Path) -> dict:
    return {str(q.relative_to(d)): (q.stat().st_size, q.stat().st_mtime_ns) for q in d.rglob("*")} if d.exists() else {}


def sao_khoa(g: Path, goc: str, moi: str, sua) -> str:
    """Ban sao phien khoa TRONG vung tests, sua khoa.json bang `sua(dict)`, ghi lai SHA256SUMS khop (de chi con
    lech do noi dung bi sua). Tra ve ten phien moi."""
    k = json.loads((g / "chuan_bi" / goc / "khoa.json").read_text(encoding="utf-8"))
    sua(k)
    d = g / "chuan_bi" / moi
    s = W.ghi_json(d / "khoa.json", k, [g])
    (d / "SHA256SUMS").write_text(f"{s}  khoa.json\n", encoding="utf-8")
    return moi


def sao_tien_kiem(g: Path, goc: str, moi: str, sua_json=None, sua_csv: bool = False) -> str:
    src, d = g / "chuan_bi" / goc, g / "chuan_bi" / moi
    d.mkdir()
    for f in ("bang_tien_kiem.csv", "mau_smoke.json"):
        b = (src / f).read_bytes()
        (d / f).write_bytes(b + (b"x" if sua_csv and f.endswith(".csv") else b""))
    tk = json.loads((src / "tien_kiem.json").read_text(encoding="utf-8"))
    if sua_json:
        sua_json(tk)
    W.ghi_json(d / "tien_kiem.json", tk, [g])
    return moi


def _ra(r) -> str:
    return (r.stdout[-600:] + " | " + r.stderr[-900:]).strip()


@ca("I2 CLI that tren du lieu that (vung tests): tien-kiem -> khoa -> kiem-khoa 0; ghi de 1; kiem-khoa mac dinh CHI DOC; "
    "--luu chuan_bi / smoke chi ghi vao vung chon (F3); khoa gom ma Ultralytics/TrackEval, backend, cau hinh (F2)")
def ca_cli_chuoi(lan: Path):
    g = lan / "cli_chuoi"
    r = cli(["--buoc", "tien-kiem", "--lan", "tien_kiem_cli"], g)
    assert r.returncode == 0, (r.returncode, _ra(r))
    r2 = cli(["--buoc", "tien-kiem", "--lan", "tien_kiem_cli"], g)
    assert r2.returncode == 1 and "ghi de" in r2.stderr, r2.stderr
    # ket qua fixture GIA trong vung tests cua env (ham khoa doi sha wrapper/test hien tai)
    W.ghi_json(g / "tests" / "lan_gia" / "ket_qua_test.json",
               {"tat_ca_dat": True, "so_test": 0, "wrapper_sha256": W.sha256_file(DIR / "chuan_bi_test40.py"),
                "test_sha256": W.sha256_file(Path(__file__)), "ghi_chu": "gia, chi de kiem duong khoa trong fixture"}, [g])
    r3 = cli(["--buoc", "khoa", "--lan", "khoa_cli"], g)
    assert r3.returncode == 0, _ra(r3)
    # F3: kiem-khoa mac dinh khong ghi gi
    truoc = liet_ke(g)
    r4 = cli(["--buoc", "kiem-khoa", "--khoa", "khoa_cli"], g)
    assert r4.returncode == 0, _ra(r4)
    assert liet_ke(g) == truoc, "kiem-khoa mac dinh da ghi"
    r4b = cli(["--buoc", "kiem-khoa", "--khoa", "khoa_cli", "--lan", "x"], g)
    assert r4b.returncode == 2 and liet_ke(g) == truoc, _ra(r4b)
    r4c = cli(["--buoc", "kiem-khoa", "--khoa", "khoa_cli", "--luu", "chuan_bi", "--lan", "kiem_khoa_cli_cb"], g)
    assert r4c.returncode == 0 and (g / "chuan_bi" / "kiem_khoa_cli_cb" / "ket_qua.json").is_file(), _ra(r4c)
    truoc_cb = liet_ke(g / "chuan_bi")
    r4d = cli(["--buoc", "kiem-khoa", "--khoa", "khoa_cli", "--luu", "smoke", "--lan", "kiem_khoa_cli_sm"], g)
    assert r4d.returncode == 0 and (g / "smoke" / "kiem_khoa_cli_sm" / "ket_qua.json").is_file(), _ra(r4d)
    assert liet_ke(g / "chuan_bi") == truoc_cb, "kiem-khoa --luu smoke da ghi vao chuan_bi"
    assert sorted(p.name for p in (g / "smoke").iterdir()) == ["kiem_khoa_cli_sm"]
    r4e = cli(["--buoc", "kiem-khoa", "--luu", "khac"], g)
    r4f = cli(["--buoc", "tien-kiem", "--luu", "smoke"], g)
    assert r4e.returncode == 2 and r4f.returncode == 2, (_ra(r4e), _ra(r4f))
    khoa = json.loads((g / "chuan_bi" / "khoa_cli" / "khoa.json").read_text(encoding="utf-8"))
    vai = {}
    for f in khoa["file"]:
        k = f["vai_tro"].split(":")[0]
        vai[k] = vai.get(k, 0) + 1
    assert vai.get("gt") == 40 and vai.get("seqinfo") == 40 and vai.get("ignored_regions") == 40 and vai.get("checkpoint") == 3
    assert vai.get("yaml_tracker") == 4 and vai.get("ma_phu_thuoc", 0) >= 10
    ul = {Path(f["rel"]).as_posix().split("site-packages/")[-1] for f in khoa["file"] if f["vai_tro"] == "ma_ultralytics"}
    can = {"ultralytics/trackers/byte_tracker.py", "ultralytics/trackers/basetrack.py",
           "ultralytics/trackers/utils/matching.py", "ultralytics/trackers/utils/kalman_filter.py",
           "ultralytics/engine/predictor.py", "ultralytics/models/yolo/detect/predict.py", "ultralytics/utils/nms.py",
           "ultralytics/nn/autobackend.py", "ultralytics/nn/backends/pytorch.py", "ultralytics/nn/tasks.py",
           "ultralytics/data/augment.py", "ultralytics/utils/events.py", "ultralytics/utils/files.py"}
    assert can <= ul, sorted(can - ul)
    te = {Path(f["rel"]).name for f in khoa["file"] if f["vai_tro"] == "ma_trackeval"}
    assert {"mot_challenge_2d_box.py", "hota.py", "clear.py", "identity.py", "eval.py"} <= te, te
    assert khoa["backend_tracker"]["backend"] == "ultralytics" and khoa["cau_hinh_hieu_luc"]["bo_loc"]["pf500"]
    assert all(W.tuyet_doi(r) in {W.tuyet_doi(f["rel"]) for f in khoa["file"]} for r in khoa["backend_tracker"]["file"].values())
    tk = json.loads((g / "chuan_bi" / "tien_kiem_cli" / "tien_kiem.json").read_text(encoding="utf-8"))
    return {"tien_kiem": {"ket_luan": tk["ket_luan"], "tong": tk["tong"], "thoi_gian_s": tk["thoi_gian_s"]},
            "khoa_so_file": khoa["so_file"], "vai_tro": vai, "so_ma_ultralytics": len(ul),
            "thu_vien_record": sorted(f["vai_tro"] for f in khoa["file"] if f["vai_tro"].startswith("thu_vien_record"))}


@ca("I3 F2 truoc suy luan (CLI that, ban sao khoa trong tests): khoa hop le -> dung exit 5 (fixture khong suy luan); "
    "BYTETRACK_BACKEND=np / sha nguon Ultralytics doi / module nap ngoai khoa / cau hinh resolve / thu vien lech -> chan")
def ca_cli_truoc_suy_luan(lan: Path):
    g = lan / "cli_chuoi"
    out = {}

    def smoke(khoa: str, ten: str, env_them: dict | None = None):
        if env_them:
            os.environ.update(env_them)
        try:
            r = cli(["--buoc", "smoke", "--khoa", khoa, "--lan", ten], g)
        finally:
            for k in (env_them or {}):
                os.environ.pop(k, None)
        kq = g / "smoke" / ten / "ket_qua_smoke.json"
        d = json.loads(kq.read_text(encoding="utf-8")) if kq.is_file() else None
        assert not (g / "smoke" / ten / "cache").exists(), f"{ten}: da tao cache"
        return r, d

    r, d = smoke("khoa_cli", "lan_hop_le")
    assert r.returncode == 5 and d["kiem_truoc_suy_luan"]["dat"] and d["dung_truoc_suy_luan"], _ra(r)
    out["hop_le"] = {"exit": r.returncode, "so_module_nap": d["kiem_truoc_suy_luan"]["so_module_nap"]}
    r, d = smoke("khoa_cli", "lan_np", {"BYTETRACK_BACKEND": "np"})
    assert r.returncode == 2 and any("BYTETRACK_BACKEND" in x for x in d["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    out["backend_np"] = d["kiem_truoc_suy_luan"]["lech"][:1]

    def doi_sha(ten_file):
        def f(k):
            for x in k["file"]:
                if Path(x["rel"]).as_posix().endswith(ten_file):
                    x["sha256"] = "0" * 64
                    return
            raise AssertionError(f"khong co {ten_file} trong khoa")
        return f
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_sha_bytetracker", doi_sha("ultralytics/trackers/byte_tracker.py"))
    rk = cli(["--buoc", "kiem-khoa", "--khoa", kb], g)
    assert rk.returncode == 2 and "byte_tracker.py" in rk.stdout, _ra(rk)
    rs = cli(["--buoc", "smoke", "--khoa", kb, "--lan", "lan_sha"], g)
    assert rs.returncode == 2 and not (g / "smoke" / "lan_sha").exists(), _ra(rs)
    out["sha_byte_tracker"] = {"kiem_khoa": rk.returncode, "smoke": rs.returncode}

    def bo(ten_file):
        def f(k):
            n = len(k["file"])
            k["file"] = [x for x in k["file"] if not Path(x["rel"]).as_posix().endswith(ten_file)]
            assert len(k["file"]) == n - 1
        return f
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_bo_kalman", bo("ultralytics/trackers/utils/kalman_filter.py"))
    rk = cli(["--buoc", "kiem-khoa", "--khoa", kb], g)
    assert rk.returncode == 0, _ra(rk)
    r, d = smoke(kb, "lan_bo_kalman")
    assert r.returncode == 2 and any("module nap ngoai khoa" in x and "kalman_filter.py" in x
                                     for x in d["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    out["bo_kalman"] = d["kiem_truoc_suy_luan"]["lech"][:1]

    def doi_cfg(k):
        k["cau_hinh_hieu_luc"]["bo_loc"]["ekf"]["cfg_hieu_luc"]["track_buffer"] = 31
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_cfg", doi_cfg)
    r, d = smoke(kb, "lan_cfg")
    assert r.returncode == 2 and any("cau hinh tracker" in x for x in d["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_tv", lambda k: k["thu_vien"].update({"ultralytics": "0.0.0"}))
    r, d = smoke(kb, "lan_tv")
    assert r.returncode == 2 and any("thu vien" in x for x in d["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_bk", lambda k: k["backend_tracker"].update({"backend": "np"}))
    r, d = smoke(kb, "lan_bk")
    assert r.returncode == 2 and any("backend tracker" in x for x in d["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    # hau-kiem khong nhan phien dung truoc suy luan
    rh = cli(["--buoc", "hau-kiem", "--khoa", "khoa_cli", "--smoke", "lan_hop_le", "--lan", "hau_kiem_cli"], g)
    assert rh.returncode == 4, _ra(rh)
    out["hau_kiem_phien_dung"] = rh.returncode
    return out


@ca("I6 F2 vong 2: khoi tao DetectionPredictor THAT (3 checkpoint, CPU; khong setup_model/predict/warmup/GPU) nap "
    "utils.events -> nam trong khoa, khong module Ultralytics ngoai khoa; ban sao khoa thieu / sai hash events.py "
    "-> chan truoc suy luan; thu muc predictor ngoai vung -> loi, trong tests -> khong tao")
def ca_predictor_that(lan: Path):
    g = lan / "cli_chuoi"
    ev = "ultralytics/utils/events.py"
    khoa = json.loads((g / "chuan_bi" / "khoa_cli" / "khoa.json").read_text(encoding="utf-8"))
    gt = W.doc_giao_thuc()
    d0 = next(iter(gt["detector"]))                       # detector dau tien: predictor dau tien nap module luoi
    out = {}
    # (a) khoa that (CLI, tien trinh moi): events.py co trong khoa; predictor cua ca 3 checkpoint khoi tao dung
    f_ev = [f for f in khoa["file"] if Path(f["rel"]).as_posix().endswith(ev)]
    assert len(f_ev) == 1 and f_ev[0]["vai_tro"] == "ma_ultralytics", f_ev
    pk = khoa["predictor_khoi_tao"]
    assert list(pk) == list(gt["detector"]), list(pk)
    for d, x in pk.items():
        assert x["lop"].endswith(".DetectionPredictor") and not x["setup_model"] and not x["tao_thu_muc"] \
            and not x["save"] and x["save_dir_trong_vung"] and x["events_callback"] and not x["events_bat"] \
            and not x["cuda_da_khoi_tao"], (d, x)
    moi = [Path(m).as_posix() for m in pk[d0]["module_moi"]]
    assert any(m.endswith(ev) for m in moi), moi          # chinh predictor.__init__ nap events.py
    assert not (g / "chuan_bi" / "khoa_cli" / W.TEN_PREDICTOR_KHONG_GHI).exists()
    out["khoa_cli"] = {"module_moi_predictor_dau": moi, "events_sha256": f_ev[0]["sha256"]}
    # (b) smoke hop le cua I3 (CLI, exit 5): predictor da khoi tao, khong module ngoai khoa, khong tao thu muc
    d = json.loads((g / "smoke" / "lan_hop_le" / "ket_qua_smoke.json").read_text(encoding="utf-8"))
    tr = d["kiem_truoc_suy_luan"]
    assert tr["dat"] and not tr["lech"] and list(tr["predictor"]) == list(gt["detector"]), tr
    assert not (g / "smoke" / "lan_hop_le" / W.TEN_PREDICTOR_KHONG_GHI).exists()
    out["smoke_hop_le_module_moi"] = [Path(m).name for m in tr["predictor"][d0]["module_moi"]]

    # (c) ban sao khoa THIEU events.py: kiem-khoa 0 (khoa tu nhat quan) nhung smoke chan exit 2 truoc suy luan
    def bo(k):
        n = len(k["file"])
        k["file"] = [x for x in k["file"] if not Path(x["rel"]).as_posix().endswith(ev)]
        assert len(k["file"]) == n - 1
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_bo_events", bo)
    rk = cli(["--buoc", "kiem-khoa", "--khoa", kb], g)
    assert rk.returncode == 0, _ra(rk)
    r = cli(["--buoc", "smoke", "--khoa", kb, "--lan", "lan_bo_events"], g)
    dd = json.loads((g / "smoke" / "lan_bo_events" / "ket_qua_smoke.json").read_text(encoding="utf-8"))
    assert r.returncode == 2 and dd["dung_truoc_suy_luan"] and any(
        "module nap ngoai khoa" in x and "events.py" in x for x in dd["kiem_truoc_suy_luan"]["lech"]), _ra(r)
    assert not (g / "smoke" / "lan_bo_events" / "cache").exists()
    out["thieu_events"] = {"kiem_khoa": rk.returncode, "smoke": r.returncode,
                           "lech": dd["kiem_truoc_suy_luan"]["lech"][:1]}

    # (d) ban sao khoa SAI HASH events.py: kiem-khoa 2, smoke 2 khong tao phien smoke
    def doi(k):
        for x in k["file"]:
            if Path(x["rel"]).as_posix().endswith(ev):
                x["sha256"] = "0" * 64
                return
        raise AssertionError("khong co events.py trong khoa")
    kb = sao_khoa(g, "khoa_cli", "khoa_cli_sha_events", doi)
    rk = cli(["--buoc", "kiem-khoa", "--khoa", kb], g)
    assert rk.returncode == 2 and "events.py" in rk.stdout, _ra(rk)
    r = cli(["--buoc", "smoke", "--khoa", kb, "--lan", "lan_sha_events"], g)
    assert r.returncode == 2 and not (g / "smoke" / "lan_sha_events").exists(), _ra(r)
    out["sai_hash_events"] = {"kiem_khoa": rk.returncode, "smoke": r.returncode}

    # (e) ham THAT trong tien trinh fixture: thu muc ngoai VUNG_R54 -> loi; trong tests -> khong tao, khong setup
    from ultralytics import YOLO
    ng = W.Nguon(gt)
    m = YOLO(str(ng.checkpoint[d0]))
    out["ngoai_vung"] = phai_loi(lambda: W.khoi_tao_predictor(gt, d0, m, ng.checkpoint[d0], GOC / "runs"),
                                 W.LoiDauVao, "ngoai vung")
    tm = lan / "i6_predictor"
    info = W.khoi_tao_predictor(gt, d0, m, ng.checkpoint[d0], tm)
    assert not tm.exists() and not info["setup_model"] and m.predictor is None, info
    da_khoa = {W.tuyet_doi(f["rel"]) for f in khoa["file"]}
    goc_ul = W.goc_ma_ngoai()["ultralytics"]
    ngoai = [W.rel_goc(p) for p in W.file_module_da_nap() if W.trong(p, goc_ul) and p not in da_khoa]
    assert not ngoai, ngoai
    out["ham_that"] = {k: info[k] for k in ("lop", "save_dir", "tao_thu_muc", "events_bat", "cuda_da_khoi_tao")}
    return out


@ca("I4 F4 dau vao doi sau tien-kiem -> khoa tu choi: ham (GT fixture bi sua) + CLI (ban sao bang chung tien-kiem "
    "lech GT / bang csv / ma da nap)")
def ca_khoa_doi_chieu(lan: Path):
    # (a) ham, du lieu fixture: sua GT THAT cua fixture sau tien-kiem
    gt, ng, loai = tao_fixture(lan / "fx_f4")
    kq = W.tien_kiem(gt, ng, doc_names=names_gia(loai), mau=MAU_FX, kiem_cau_hinh=False)
    assert kq["ket_luan"] == "dat", kq["lech"]
    assert W.doi_chieu_tien_kiem(kq | {"exit": 0}, W.dau_vao_sha(gt, ng)) == []
    v = MAU_FX["videos"][1]
    p = ng.gt_txt(v)
    p.write_bytes(p.read_bytes() + b"5,9,1,1,10,10,1,1,1.0\n")
    lech = W.doi_chieu_tien_kiem(kq | {"exit": 0}, W.dau_vao_sha(gt, ng))
    assert len(lech) == 1 and f"{v}/gt/gt.txt" in lech[0] and "doi sau tien-kiem" in lech[0], lech
    ck = ng.checkpoint["yolo11n_mixed5"]
    ck.write_bytes(ck.read_bytes() + b"x")
    assert len(W.doi_chieu_tien_kiem(kq | {"exit": 0}, W.dau_vao_sha(gt, ng))) == 2
    out = {"ham": lech}
    # (b) CLI that: ban sao bang chung tien-kiem trong tests, mo phong dau vao da doi
    g = lan / "cli_chuoi"
    gt_rel = "data/processed/DETRAC-test/MVI_39051/gt/gt.txt"
    t1 = sao_tien_kiem(g, "tien_kiem_cli", "tien_kiem_cli_gt",
                       lambda t: t["dau_vao_sha256"].__setitem__(gt_rel, "1" * 64))
    t2 = sao_tien_kiem(g, "tien_kiem_cli", "tien_kiem_cli_csv", sua_csv=True)
    for t, chua in ((t1, gt_rel), (t2, "bang_tien_kiem.csv")):
        r = cli(["--buoc", "khoa", "--tien-kiem", t, "--lan", f"khoa_tu_{t}"], g)
        assert r.returncode == 2 and chua in r.stderr and not (g / "chuan_bi" / f"khoa_tu_{t}").exists(), _ra(r)
        out[t] = r.returncode
    ma_rel = "src/run_tracking.py"
    t3 = sao_tien_kiem(g, "tien_kiem_cli", "tien_kiem_cli_ma",
                       lambda t: t["ma_da_nap_sha256"].__setitem__(ma_rel, "2" * 64))
    r = cli(["--buoc", "khoa", "--tien-kiem", t3, "--lan", "khoa_tu_ma"], g)
    d = g / "chuan_bi" / "khoa_tu_ma"
    assert r.returncode == 2 and ma_rel in r.stderr and not (d / "khoa.json").is_file() \
        and (d / "khoa_that_bai.json").is_file(), _ra(r)
    # phien khoa that bai khong duoc chon lam khoa mac dinh
    assert W.phien_moi_nhat(g / "chuan_bi", "khoa_", "khoa.json").name != "khoa_tu_ma"
    out[t3] = r.returncode
    return out


@ca("I5 khoa.json bi sua (ban sao) -> kiem-khoa 2, smoke 2 khong tao thu muc smoke")
def ca_cli_khoa_sua(lan: Path):
    g = lan / "cli_chuoi"
    d = g / "chuan_bi" / "khoa_cli_sua_noi_dung"
    d.mkdir()
    (d / "SHA256SUMS").write_bytes((g / "chuan_bi" / "khoa_cli" / "SHA256SUMS").read_bytes())
    (d / "khoa.json").write_bytes((g / "chuan_bi" / "khoa_cli" / "khoa.json").read_bytes()
                                  .replace(b'"so_test": 0', b'"so_test": 1'))
    r5 = cli(["--buoc", "kiem-khoa", "--khoa", d.name], g)
    assert r5.returncode == 2, _ra(r5)
    r6 = cli(["--buoc", "smoke", "--khoa", d.name, "--lan", "lan_khoa_sua"], g)
    assert r6.returncode == 2 and not (g / "smoke" / "lan_khoa_sua").exists(), _ra(r6)
    return {"kiem_khoa": r5.stdout.strip()[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", required=True, choices=["fixture"])
    ap.parse_args()
    lan = VUNG_TESTS / f"lan_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}_pid{os.getpid()}"
    lan.mkdir(parents=True, exist_ok=False)
    W.dat_moi_truong(lan)
    truoc = snapshot()
    t0 = time.time()
    ca_giao_thuc()
    ca_anh_xa()
    ca_anh_xa_loi()
    ca_loc(lan)
    ca_to_xam()
    ca_ghi_giong(lan)
    ca_smoke_fixture(lan)
    ca_tien_kiem_dat(lan)
    for f in (ca_sai_hash, ca_sai_hash_anh, ca_thieu_gt, ca_sai_timeline, ca_sai_seqlength, ca_d4_co_gt, ca_vung_ghi):
        f(lan)
    ca_duong_that()
    ca_cli_loi(lan)
    ca_cli_chuoi(lan)
    ca_cli_truoc_suy_luan(lan)
    ca_predictor_that(lan)
    ca_khoa_doi_chieu(lan)
    ca_cli_khoa_sua(lan)
    sau = snapshot()
    khac = sorted(k for k in set(truoc) | set(sau) if truoc.get(k) != sau.get(k) and not W.trong(Path(k), lan))
    KET_QUA.append({"ten": "Z1 khong ghi ngoai vung (snapshot thu muc nhay cam)", "dat": not khac, "chi_tiet": khac[:50]})
    print(f"  [{'DAT ' if not khac else 'LOI '}] Z1 khong ghi ngoai vung ({len(khac)} khac)")
    so_dat = sum(1 for k in KET_QUA if k["dat"])
    tat_ca = so_dat == len(KET_QUA)
    kq = {"buoc": "fixture", "lan": W.rel_goc(lan), "luc": W.bay_gio(), "tong_giay": round(time.time() - t0, 1),
          "so_test": len(KET_QUA), "so_dat": so_dat, "tat_ca_dat": tat_ca,
          "wrapper_sha256": W.sha256_file(DIR / "chuan_bi_test40.py"), "test_sha256": W.sha256_file(Path(__file__)),
          "giao_thuc_sha256": W.sha256_file(W.GIAO_THUC), "thu_vien": W.phien_ban_thu_vien(),
          "gioi_han": ["detector la mo hinh gia; duong spec.fill_env()/YOLO that va GPU khong chay o fixture",
                       "fixture smoke dung so_khop_dataset=False; doi chieu DetracDataset/vung bo qua that o ca H1",
                       "CLI smoke o che do fixture (R54_GOC_KET_QUA_TEST) luon dung sau kiem truoc suy luan (exit 5); "
                       "duong suy luan that chua chay o R54A",
                       "F2: ca 'nguon Ultralytics doi' mo phong bang ban sao khoa co sha sai (khong sua file thu vien that)",
                       "F2 vong 2: predictor that chi KHOI TAO (khong setup_model/AutoBackend/select_device/predict); "
                       "module nap luoi trong setup_model/stream_inference chi ra soat tinh, con lai bat o kiem sau smoke",
                       "F4: ca CLI mo phong dau vao doi bang ban sao bang chung tien-kiem; ca ham sua GT/checkpoint fixture that",
                       "snapshot chi phu cac thu muc liet ke trong snapshot(), khong phai toan bo dia"],
          "ket_qua": KET_QUA, "exit": 0 if tat_ca else 1}
    W.ghi_json(lan / "ket_qua_test.json", kq, [lan])
    print(f"[fixture] {so_dat}/{len(KET_QUA)} dat -> {W.rel_goc(lan)}  exit {kq['exit']}")
    return kq["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
