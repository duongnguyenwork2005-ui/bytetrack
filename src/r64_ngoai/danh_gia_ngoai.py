"""R64 — danh gia HE THONG (3 detector R54 x CV/EKF/UKF/PF500, c25/c10) tren tap ngoai: SVMOT test, RoundaboutHD.

Giao thuc do GPT chot 01/11/2026 (D:/LABOTARY/automation/runs_local/hoi_codex_R64_dataset_moi_20261101/answer.json),
nguoi dung uy quyen ("Du nhu R54/R61"); sua theo review smoke 02/11 (.../hoi_codex_R64_review_smoke_20261102/answer.json:
B1 bo dong GT trung, B_sua cong hoan tat/noi dung/provenance/hau kiem/ngan sach/PARTIAL). Namespace moi; KHONG sua helper
da khoa (det_cache, run_tracking, run_trackeval, boot_stats) — chi goi chung.
  svmot      : 9 chuoi test chinh thuc (scene_05/06), anh JPG 1920x1080, 30 fps; GT goc 7 lop gop class 1, giu visibility.
               Anh toan khung, KHONG to xam roi_mask.png (giao thuc full-image du an, khong phai giao thuc chinh thuc SVMOT).
  roundabout : imagesc001..004, frame 1..3000 lien tuc moi camera (doc truc tiep video.mp4 3840x2160, 15 fps, khong
               ghi lai anh); GT = SCT/SCT_GT.txt (xyxy -> xywh), gop class 1; xe do ngoai SCT_GT chiu FP. Loi nguon cung
               (frame, id) hai box -> BO ca cac dong cua cap trung (B1), ghi danh sach dong bo.
Detector/cache: checkpoint + sha256 theo configs/phase2/r54_tracking/giao_thuc_test40.json; imgsz 640 rect mac dinh,
conf 0,05, iou 0,7, max_det 3000, FP32, batch 1; mixed5 -> doi ID sang COCO (bang R61), luu cls_goc.
Tracker: lop 1 2 3 5 7, max_det 300, conf > 0,25 / > 0,10, buffer 1,2 s (36 / 18 buoc), elapsed steps, evaluator mot.

    python -B src/r64_ngoai/danh_gia_ngoai.py gt <ds> [--smoke]
    python -B src/r64_ngoai/danh_gia_ngoai.py cache <ds> <detector> [--smoke] [--het-han ISO]
    python -B src/r64_ngoai/danh_gia_ngoai.py xac-minh <ds> [--smoke]      # bien ban xac minh bo sung cache da sinh
    python -B src/r64_ngoai/danh_gia_ngoai.py track <ds> <detector> <c25|c10> <cv|ekf|ukf|pf500> [--smoke] [--het-han ISO]
    python -B src/r64_ngoai/danh_gia_ngoai.py tong-hop <ds> [--smoke]
--smoke: 128 frame dau cua chuoi dinh truoc (seq_22_morning / imagesc001), split/cache/ten rieng.
Ma thoat: 0 xong; 10 dung do ngan sach (PARTIAL, chay lai tiep tuc); 2 loi hop dong; 3 loi subprocess.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))
import config  # noqa: E402
import det_cache  # noqa: E402
from mot_datasets.base import FrameRef  # noqa: E402

GIAO_THUC_R54 = GOC / "configs" / "phase2" / "r54_tracking" / "giao_thuc_test40.json"
QUYET_DINH = ["D:/LABOTARY/automation/runs_local/hoi_codex_R64_dataset_moi_20261101/answer.json",
              "D:/LABOTARY/automation/runs_local/hoi_codex_R64_review_smoke_20261102/answer.json"]
DETECTOR = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")
BO_LOC = {"cv": ("bytetrack.yaml", None), "ekf": ("configs/bytetrack_ekf_ctrv.yaml", None),
          "ukf": ("configs/bytetrack_ukf_ctrv.yaml", None),
          "pf500": ("configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml", [0, 1, 2])}
DIEU_KIEN = {"c25": 0.25, "c10": 0.10}
LOP_TRACKER = [1, 2, 3, 5, 7]
SMOKE_N = 128
MA_THUC_DUNG = ["src/r64_ngoai/danh_gia_ngoai.py", "src/det_cache.py", "src/detector_classes.py", "src/run_tracking.py",
                "src/run_trackeval.py", "src/boot_stats.py"]
DU_PHONG_DIA = 5 * 1024 ** 3
#: uoc luong de canh ngan sach thoi gian (s / frame), do o smoke + 4 chuoi SVMOT / 2 camera RoundaboutHD dau
UOC_CACHE_S_FRAME = {"svmot": 1 / 8.0, "roundabout": 1 / 18.0}
UOC_TRACK_S_FRAME = {"cv": 0.004, "ekf": 0.004, "ukf": 0.005, "pf500": 0.03}
TAP = {
    "svmot": {"split": "SVMOTSYS-test", "fps": 30.0, "hw": (1080, 1920), "nguon": Path("F:/LABOTARY_data/svmot/data"),
              "videos": ["seq_22_morning", "seq_23_morning", "seq_24_morning", "seq_25_morning", "seq_26_evening",
                         "seq_27_evening", "seq_28_evening", "seq_29_evening", "seq_30_evening"],
              "n_goc": {"seq_22_morning": 1200, "seq_23_morning": 900, "seq_24_morning": 900, "seq_25_morning": 900,
                        "seq_26_evening": 870, "seq_27_evening": 870, "seq_28_evening": 870, "seq_29_evening": 870,
                        "seq_30_evening": 870},
              "smoke": "seq_22_morning", "n_frame": None},
    "roundabout": {"split": "RoundaboutHD-prefix3000", "fps": 15.0, "hw": (2160, 3840),
                   "nguon": Path("C:/LABOTARY_data/RoundaboutHD"),
                   "videos": ["imagesc001", "imagesc002", "imagesc003", "imagesc004"], "smoke": "imagesc001",
                   "n_goc": {"imagesc001": 9000, "imagesc002": 9000, "imagesc003": 9000, "imagesc004": 9000},
                   "n_frame": 3000},
}


class LoiHopDong(RuntimeError):
    pass


class HetNganSach(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ghi_json_nguyen_tu(p: Path, obj) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def bay_gio() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def hash_ma() -> dict:
    return {m: sha256_file(GOC / m) for m in MA_THUC_DUNG}


def cau_hinh(ds: str, smoke: bool) -> dict:
    t = dict(TAP[ds])
    if smoke:
        t["split"] += "-smoke128"
        t["videos"] = [t["smoke"]]
        t["n_frame"] = SMOKE_N
    return t


def so_frame(t: dict, v: str) -> int:
    return t["n_frame"] or t["n_goc"][v]


def ten_cache(ds: str, det: str, smoke: bool) -> str:
    return f"r64{'smoke' if smoke else ''}_{ds}_{det}_c0.05"


DU_PHONG_GIAY = 1800          # review cuoi GPT 02/11: tu danh 30 phut hau kiem/bao cao truoc han


def tien_kiem_job(ds: str, smoke: bool, het_han: str | None, viec: str, giay_uoc: float, thu_muc: Path,
                  byte_uoc: int) -> None:
    """Tien kiem ngan sach TRUOC moi job (review cuoi GPT 02/11): bat buoc --het-han (tru smoke), con lai >= du toan +
    30 phut, o dich trong >= du toan ghi + 5 GiB; MOI lan kiem (dat hay khong) ghi mot dong JSON vao
    handoff/logs/external_eval/[smoke/]<ds>/ngan_sach.jsonl."""
    trong = shutil.disk_usage(thu_muc).free
    con = None if het_han is None else dt.datetime.fromisoformat(het_han).timestamp() - time.time()
    loi = []
    if het_han is None and not smoke:
        loi.append("thieu --het-han")
    if con is not None and giay_uoc + DU_PHONG_GIAY > con:
        loi.append(f"uoc {giay_uoc:.0f} s + du phong {DU_PHONG_GIAY} s > con {con:.0f} s")
    if trong < byte_uoc + DU_PHONG_DIA:
        loi.append(f"o {thu_muc} con {trong / 1e9:.1f} GB < uoc {byte_uoc / 1e9:.3f} GB + 5 GiB")
    nk = GOC / "handoff" / "logs" / "external_eval" / ("smoke" if smoke else "") / ds / "ngan_sach.jsonl"
    nk.parent.mkdir(parents=True, exist_ok=True)
    with open(nk, "a", encoding="utf-8") as f:
        f.write(json.dumps({"thoi_diem": bay_gio(), "viec": viec, "het_han": het_han,
                            "giay_con": None if con is None else round(con), "giay_uoc": round(giay_uoc),
                            "giay_du_phong": DU_PHONG_GIAY, "o_dich": str(thu_muc), "byte_trong": trong,
                            "byte_uoc": byte_uoc, "byte_du_phong": DU_PHONG_DIA, "dat": not loi, "loi": loi},
                           ensure_ascii=False) + "\n")
    if loi:
        raise HetNganSach(f"{viec}: " + "; ".join(loi))


# --------------------------------------------------------------------------------------------- GT
def doc_gt_svmot(p: Path) -> np.ndarray:
    a = np.loadtxt(p, delimiter=",", ndmin=2)
    if a.shape[1] != 9:
        raise LoiHopDong(f"{p}: {a.shape[1]} cot, can 9 (frame,id,x,y,w,h,conf,lop,visibility)")
    return a


def doc_gt_roundabout(p: Path) -> np.ndarray:
    """SCT_GT: `frame id x1 y1 x2 y2 cls` -> mang MOT 9 cot (x,y,w,h; conf 1; lop goc; visibility 1)."""
    a = np.loadtxt(p, ndmin=2)
    if a.shape[1] != 7:
        raise LoiHopDong(f"{p}: {a.shape[1]} cot, can 7")
    if not (a[:, 4] > a[:, 2]).all() or not (a[:, 5] > a[:, 3]).all():
        raise LoiHopDong(f"{p}: co box x2<=x1 hoac y2<=y1 (khong phai xyxy?)")
    out = np.zeros((len(a), 9))
    out[:, 0:2] = a[:, 0:2]
    out[:, 2:4] = a[:, 2:4]
    out[:, 4] = a[:, 4] - a[:, 2]
    out[:, 5] = a[:, 5] - a[:, 3]
    out[:, 6] = 1
    out[:, 7] = a[:, 6]          # lop goc (gop o buoc ghi)
    out[:, 8] = 1.0
    return out


def dong_trung(a: np.ndarray) -> np.ndarray:
    """Mask cac dong thuoc cap (frame, id) xuat hien > 1 lan."""
    khoa = a[:, 0].astype(np.int64) * 10_000_000 + a[:, 1].astype(np.int64)
    _, inv, dem = np.unique(khoa, return_inverse=True, return_counts=True)
    return dem[inv] > 1


def kiem_gt(a: np.ndarray, n_frame: int, ten: str, cho_phep_trung: bool = False) -> dict:
    if not np.isfinite(a).all():
        raise LoiHopDong(f"{ten}: GT co NaN/inf")
    if (a[:, 4] <= 0).any() or (a[:, 5] <= 0).any():
        raise LoiHopDong(f"{ten}: GT co w/h <= 0")
    if (a[:, 0] != np.round(a[:, 0])).any() or a[:, 0].min() < 1 or a[:, 0].max() > n_frame:
        raise LoiHopDong(f"{ten}: frame GT ngoai 1..{n_frame}")
    if dong_trung(a).any() and not cho_phep_trung:
        raise LoiHopDong(f"{ten}: trung (frame, id)")
    return {"n_box": int(len(a)), "n_id": int(len(np.unique(a[:, 1]))), "frame_min": int(a[:, 0].min()),
            "frame_max": int(a[:, 0].max()), "lop_goc": {str(int(k)): int(v) for k, v in
                                                        zip(*np.unique(a[:, 7], return_counts=True))}}


def buoc_gt(ds: str, smoke: bool) -> int:
    t = cau_hinh(ds, smoke)
    root = config.PROCESSED_DIR / t["split"]
    seqmap = config.PROCESSED_DIR / "seqmaps" / f"{t['split']}.txt"
    if root.exists() or seqmap.exists():
        print("[LOI] da ton tai", root, "/", seqmap); return 2
    log = {"ds": ds, "split": t["split"], "smoke": smoke, "tao_luc": bay_gio(), "ma": hash_ma(), "videos": {}}
    H, W = t["hw"]
    for v in t["videos"]:
        if ds == "svmot":
            src = t["nguon"] / v / "gt" / "gt.txt"
            n_anh = len(list((t["nguon"] / v / "img").glob("*.jpg")))
            if n_anh != t["n_goc"][v]:
                raise LoiHopDong(f"{v}: {n_anh} anh != {t['n_goc'][v]}")
            a = doc_gt_svmot(src)
        else:
            src = t["nguon"] / v / "SCT" / "SCT_GT.txt"
            a = doc_gt_roundabout(src)
        info = kiem_gt(a, t["n_goc"][v], v, cho_phep_trung=(ds == "roundabout"))
        n = so_frame(t, v)
        dong_nguon = np.arange(1, len(a) + 1)            # so dong 1-based trong file nguon
        giu = a[:, 0] <= n
        a, dong_nguon = a[giu], dong_nguon[giu]
        # B1 (GPT 02/11): loi nguon RoundaboutHD — cung (frame, id) cho HAI box khac nhau. Toan nguon cam1/cam3/cam4 co
        # 2/10/14 CAP (4/20/28 dong); trong frame <= 3000 chi cam4 8 cap = 16 dong (id 105, frame 2536–2543). BO ca cac
        # dong cua moi cap (khong bo ca track, khong giu dong dau, khong doi ID); khong tao vung ignore.
        trung = dong_trung(a)
        info["dong_bo_vi_trung_frame_id"] = [{"dong_nguon": int(d), "frame": int(r[0]), "id": int(r[1]),
                                              "xywh": [round(float(x), 2) for x in r[2:6]]}
                                             for d, r in zip(dong_nguon[trung], a[trung])]
        a = a[~trung]
        kiem_gt(a, n, v)                                   # sau khi bo: KHONG con trung
        out = a.copy()
        out[:, 7] = 1                                      # gop moi lop xe -> lop danh gia 1
        info.update({"nguon": str(src), "nguon_sha256": sha256_file(src), "n_frame_dung": n,
                     "n_box_dung": int(len(out)), "n_id_dung": int(len(np.unique(out[:, 1]))) if len(out) else 0})
        d = root / v
        (d / "gt").mkdir(parents=True)
        order = np.lexsort((out[:, 1], out[:, 0]))
        with open(d / "gt" / "gt.txt", "w", newline="\n") as f:
            for r in out[order]:
                f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.3f\n" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
        (d / "seqinfo.ini").write_text(f"[Sequence]\nname={v}\nimDir=img\nframeRate={t['fps']:g}\nseqLength={n}\n"
                                       f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n", encoding="utf-8")
        info["gt_sha256"] = sha256_file(d / "gt" / "gt.txt")
        log["videos"][v] = info
        print(f"[gt] {v}: {info['n_box_dung']} box, {info['n_id_dung']} id, {n} frame, bo {int(trung.sum())} dong trung, "
              f"lop goc {info['lop_goc']}")
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(t["videos"]) + "\n", encoding="utf-8")
    (root / "gt_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def kiem_gt_cham(t: dict) -> dict:
    """Truoc cham: GT thuc dung khong trung (frame, id) tren MOI dong (TrackEval kiem truoc khi loc distractor), huu han,
    seqLength = so frame; tra hash GT thuc dung."""
    out = {}
    for v in t["videos"]:
        p = config.PROCESSED_DIR / t["split"] / v / "gt" / "gt.txt"
        a = np.loadtxt(p, delimiter=",", ndmin=2)
        kiem_gt(a, so_frame(t, v), f"{t['split']}/{v}")
        out[v] = sha256_file(p)
    return out


# --------------------------------------------------------------------------------------------- cache
def detector_r54(det: str) -> dict:
    return json.loads(GIAO_THUC_R54.read_text(encoding="utf-8"))["detector"][det]


def bang_anh_xa_coco(names, loai_ky_vong: str) -> dict | None:
    """Nhu r61_bdd.dieu_phoi_bdd.bang_anh_xa_coco (khong import de khong keo hop dong R61)."""
    import detector_classes as dc
    loai = dc.loai_checkpoint(names)
    if loai != loai_ky_vong:
        raise LoiHopDong(f"loai checkpoint {loai!r} != {loai_ky_vong!r}")
    if loai == "coco80":
        return None
    return {int(k): dc.sang_id_coco(v) for k, v in dc.anh_xa_model_sang_lop5(names).items()}


def doi_id_sang_coco(data: np.ndarray, bang: dict | None) -> np.ndarray:
    a = np.array(data, dtype=np.float32, copy=True).reshape(-1, 6)
    if bang is None or not len(a):
        return a
    goc = a[:, 5].astype(np.int64)
    thieu = sorted(set(goc.tolist()) - set(bang))
    if thieu or not np.all(a[:, 5] == goc):
        raise LoiHopDong(f"cls {thieu} khong co trong bang anh xa / khong nguyen")
    a[:, 5] = np.array([bang[int(c)] for c in goc], dtype=np.float32)
    return a


def anh_svmot(t: dict, v: str) -> list[Path]:
    n = so_frame(t, v)
    img_dir = t["nguon"] / v / "img"
    ky_vong = [f"{i:04d}.jpg" for i in range(1, n + 1)]
    co = sorted(p.name for p in img_dir.glob("*.jpg"))
    if co[:n] != ky_vong:
        raise LoiHopDong(f"{v}: ten anh khong lien tuc 0001..{n:04d}")
    return [img_dir / k for k in ky_vong]


def nguon_don_vi(ds: str, t: dict, v: str) -> dict:
    """Manifest nguon: anh (ten + sha256 tung file, gop) hoac video (sha256 file, pham vi frame)."""
    n = so_frame(t, v)
    if ds == "svmot":
        anh = anh_svmot(t, v)
        tung = [sha256_file(p) for p in anh]
        return {"loai": "anh_jpg", "thu_muc": str(anh[0].parent), "frame": [1, n], "ten_dau": anh[0].name,
                "ten_cuoi": anh[-1].name, "sha256_gop_tung_anh": hashlib.sha256("\n".join(tung).encode()).hexdigest(),
                "fps": t["fps"]}
    mp4 = t["nguon"] / v / "video.mp4"
    return {"loai": "mp4", "file": str(mp4), "sha256": sha256_file(mp4), "byte": mp4.stat().st_size, "frame": [1, n],
            "fps": t["fps"]}


def build_tu_video(model, mp4: Path, n: int, fps: float, spec, orig_shape) -> dict:
    """Nhu det_cache.build_video_yolo nhung doc frame tu video (khong ghi lai JPG, khong to xam): frame 1..n."""
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise LoiHopDong(f"khong mo duoc {mp4}")
    n_per = np.zeros(n, dtype=np.int32)
    truncated = np.zeros(n, dtype=bool)
    chunks = []
    try:
        for i in range(n):
            ok, img = cap.read()
            if not ok or img is None:
                raise LoiHopDong(f"{mp4}: decode hong o frame {i + 1}/{n} -- chan, khong bo frame")
            if img.shape[:2] != tuple(orig_shape):
                raise LoiHopDong(f"{mp4}: frame {i + 1} co kich thuoc {img.shape[:2]} != {tuple(orig_shape)}")
            d = det_cache.detect_frame(model, img, spec)
            n_per[i] = len(d)
            truncated[i] = len(d) >= spec.max_det
            chunks.append(d)
    finally:
        cap.release()
    return dict(frame_idx=np.arange(1, n + 1, dtype=np.int32), n_per_frame=n_per, unreadable=np.zeros(n, dtype=bool),
                truncated=truncated, time_s=np.arange(n, dtype=np.float64) / fps,
                data=np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), dtype=np.float32),
                orig_shape=np.array(orig_shape, dtype=np.int32))


def kiem_noi_dung_cache(ds: str, det: str, v: str, smoke: bool) -> dict:
    """Cong noi dung (Q5.5, B_sua): meta/spec, json don vi, mapping, timeline, huu han, lop, n_per_frame, orig_shape."""
    t = cau_hinh(ds, smoke)
    c = detector_r54(det)
    cdir = det_cache.CACHE_ROOT / ten_cache(ds, det, smoke)
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
    ky_vong_meta = {"backend": "ultralytics", "weights_sha256": c["sha256"], "imgsz": 640, "conf": 0.05, "iou": 0.7,
                    "classes": None, "max_det": 3000, "half": False, "agnostic_nms": False, "augment": False,
                    "batch": 1, "mask_ignored_regions": False, "dataset": ds}
    lech = {k: (meta.get(k), x) for k, x in ky_vong_meta.items() if meta.get(k) != x}
    if meta.get("rect", True) is not True:
        lech["rect"] = (meta.get("rect"), True)
    if lech:
        raise LoiHopDong(f"{cdir.name}/meta.json lech: {lech}")
    fp = hashlib.sha256(json.dumps(det_cache.meta_gen_spec(meta), sort_keys=True, default=str).encode()).hexdigest()
    if meta.get("spec_sha256") != fp:
        raise LoiHopDong(f"{cdir.name}/meta.json: spec_sha256 khong khop fingerprint tinh lai tu GEN_FIELDS")
    info = json.loads((cdir / f"{v}.json").read_text(encoding="utf-8"))
    if info.get("spec_sha256") != meta.get("spec_sha256"):
        raise LoiHopDong(f"{cdir.name}/{v}: spec_sha256 json != meta")
    if info.get("detector") != det or info.get("dataset") != ds or info.get("checkpoint_sha256") != c["sha256"]:
        raise LoiHopDong(f"{cdir.name}/{v}: json detector/dataset/checkpoint lech")
    names = {int(k): n_ for k, n_ in info["names"].items()}
    bang = bang_anh_xa_coco(names, c["loai_checkpoint"])
    bang_json = None if info["anh_xa_coco"] is None else {int(k): int(x) for k, x in info["anh_xa_coco"].items()}
    if bang != bang_json:
        raise LoiHopDong(f"{cdir.name}/{v}: anh_xa_coco trong json != bang tinh lai")
    z = np.load(cdir / f"{v}.npz")
    n = so_frame(t, v)
    loi = []
    if not np.array_equal(z["frame_idx"], np.arange(1, n + 1)):
        loi.append(f"frame_idx khong phai 1..{n}")
    if len(z["time_s"]) != n or not np.allclose(z["time_s"], np.arange(n) / t["fps"], atol=1e-9, rtol=0):
        loi.append("time_s != (i-1)/fps")
    if int(z["n_per_frame"].sum()) != len(z["data"]) or len(z["n_per_frame"]) != n:
        loi.append("tong n_per_frame != so box")
    if z["unreadable"].any() or z["truncated"].any():
        loi.append("co unreadable/truncated")
    if tuple(int(x) for x in z["orig_shape"]) != tuple(t["hw"]):
        loi.append(f"orig_shape {z['orig_shape']} != {t['hw']}")
    d = z["data"]
    if d.ndim != 2 or d.shape[1] != 6 or not np.isfinite(d).all():
        loi.append("data khong (n,6) huu han")
    elif len(d):
        if d[:, 4].min() <= 0.05 - 1e-6 or d[:, 4].max() > 1 + 1e-6:
            loi.append("conf ngoai (0,05; 1]")
        if ((d[:, 2] <= d[:, 0]) | (d[:, 3] <= d[:, 1])).any():
            loi.append("box x2<=x1/y2<=y1")
        goc = d.copy(); goc[:, 5] = z["cls_goc"]
        if z["cls_goc"].shape != (len(d),) or not np.array_equal(doi_id_sang_coco(goc, bang)[:, 5], d[:, 5]):
            loi.append("cls != anh xa(cls_goc)")
    if loi:
        raise LoiHopDong(f"{cdir.name}/{v}: " + "; ".join(loi))
    return {"n_frames": n, "n_boxes": int(len(d)), "max_n_per_frame": int(z["n_per_frame"].max()) if n else 0}


def kiem_don_vi_cache(ds: str, det: str, v: str, smoke: bool, cho_legacy_chua_xac_minh: bool = False) -> dict:
    """Cong hoan tat: marker doc duoc, dung detector/video, du artifact, sha256 khop, roi cong noi dung. Marker MOI phai
    co dataset/split dung va json don vi co provenance bat buoc (nguon, ma_sha256, split); marker LEGACY (truoc review,
    khong co dataset) chi duoc nhan khi co bien ban <video>.xac_minh_bo_sung.json gan dung hash marker (tru khi chinh
    buoc xac-minh goi). Thieu/hong -> LoiHopDong (khong skip, khong ghi de)."""
    t = cau_hinh(ds, smoke)
    cdir = det_cache.CACHE_ROOT / ten_cache(ds, det, smoke)
    mk = cdir / f"{v}.xong.json"
    try:
        m = json.loads(mk.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LoiHopDong(f"{cdir.name}/{v}: marker khong doc duoc: {e}") from e
    if m.get("detector") != det or m.get("video") != v:
        raise LoiHopDong(f"{cdir.name}/{v}: marker detector/video lech")
    for f in (f"{v}.npz", f"{v}.json", "meta.json"):
        if not (cdir / f).is_file():
            raise LoiHopDong(f"{cdir.name}/{v}: thieu {f}")
        if sha256_file(cdir / f) != m.get("sha256", {}).get(f):
            raise LoiHopDong(f"{cdir.name}/{v}: sha256 {f} != marker")
    if "dataset" in m:
        if m["dataset"] != ds or m.get("split") != t["split"]:
            raise LoiHopDong(f"{cdir.name}/{v}: marker dataset/split lech")
        info = json.loads((cdir / f"{v}.json").read_text(encoding="utf-8"))
        thieu = [k for k in ("nguon", "ma_sha256", "split", "gt_luc_sinh_sha256") if not info.get(k)]
        if thieu or info["split"] != t["split"]:
            raise LoiHopDong(f"{cdir.name}/{v}: provenance bat buoc thieu/lech {thieu or 'split'}")
    elif not cho_legacy_chua_xac_minh:
        bb = cdir / f"{v}.xac_minh_bo_sung.json"
        try:
            x = json.loads(bb.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise LoiHopDong(f"{cdir.name}/{v}: marker legacy KHONG co bien ban xac minh bo sung hop le: {e}") from e
        if x.get("marker_sha256") != sha256_file(mk) or "DAT" not in str(x.get("cong", "")):
            raise LoiHopDong(f"{cdir.name}/{v}: bien ban xac minh bo sung khong gan dung marker / khong DAT")
    return kiem_noi_dung_cache(ds, det, v, smoke)


def buoc_cache(ds: str, det: str, smoke: bool, videos: list[str] | None, het_han: str | None) -> int:
    from ultralytics import YOLO
    t = cau_hinh(ds, smoke)
    c = detector_r54(det)
    ck = GOC / c["checkpoint"]
    sha = sha256_file(ck)
    if sha != c["sha256"]:
        raise LoiHopDong(f"sha256 checkpoint {det} {sha} != {c['sha256']}")
    cid = ten_cache(ds, det, smoke)
    cdir = det_cache.CACHE_ROOT / cid
    ds_videos = videos or t["videos"]
    if cdir.exists():
        co_artifact = [p.name for p in cdir.iterdir() if p.suffix in (".npz", ".json") and p.name != "meta.json"]
        if co_artifact and not (cdir / "meta.json").exists():
            raise LoiHopDong(f"{cid}: co artifact nhung thieu meta.json -- khong tu khoi tao lai")
    for v in ds_videos:                                     # don vi do: co npz/json ma khong co marker -> dung
        if v not in t["videos"]:
            raise LoiHopDong(f"{v} khong thuoc danh sach {ds}")
        if not (cdir / f"{v}.xong.json").exists() and any((cdir / f"{v}{s}").exists() for s in (".npz", ".json")):
            raise LoiHopDong(f"{cid}/{v}: don vi do (artifact khong marker) -- khong ghi de, can xu ly tay")
    lam = []
    for v in ds_videos:
        if (cdir / f"{v}.xong.json").exists():
            kiem_don_vi_cache(ds, det, v, smoke)
            print(f"[cache] {cid}/{v}: da xong + qua cong hoan tat/noi dung -- bo qua", flush=True)
        else:
            lam.append(v)
    if not lam:
        return 0
    spec = det_cache.DetectorSpec(backend="ultralytics", weights=str(ck), weights_sha256=sha, imgsz=640, conf=0.05,
                                  iou=0.7, classes=None, max_det=3000, half=False, agnostic_nms=False, augment=False,
                                  batch=1, device="0", mask_ignored_regions=False, dataset=ds, part="test")
    spec.fill_env()
    det_cache.write_meta(cdir, spec, mode="append" if (cdir / "meta.json").exists() else "create")
    model = YOLO(str(ck))
    bang = bang_anh_xa_coco(model.names, c["loai_checkpoint"])
    gt_root = config.PROCESSED_DIR / t["split"]
    ma = hash_ma()
    for v in lam:
        n = so_frame(t, v)
        # du toan ghi: ~0,5 KB/frame quan sat (npz 3000 frame 4K ~1,4 MB) -> lay 2 KB/frame + 1 MB json/meta
        tien_kiem_job(ds, smoke, het_han, f"cache {cid}/{v}", n * UOC_CACHE_S_FRAME[ds] + 60, det_cache.CACHE_ROOT,
                      n * 2048 + 1024 ** 2)
        nguon = nguon_don_vi(ds, t, v)
        t0 = time.time()
        if ds == "svmot":
            frames = [FrameRef(index=i, path=p, time_s=(i - 1) / t["fps"]) for i, p in enumerate(anh_svmot(t, v), 1)]
            arrays = det_cache.build_video_yolo(model, frames, [], spec, t["hw"])
        else:
            arrays = build_tu_video(model, t["nguon"] / v / "video.mp4", n, t["fps"], spec, t["hw"])
        giay = time.time() - t0
        if arrays["unreadable"].any() or arrays["truncated"].any():
            raise LoiHopDong(f"{v}: unreadable {int(arrays['unreadable'].sum())} truncated "
                             f"{int(arrays['truncated'].sum())} -- chan")
        goc = arrays["data"]
        moi = doi_id_sang_coco(goc, bang)
        if moi.shape != goc.shape or not np.array_equal(moi[:, :5], goc[:, :5]):
            raise LoiHopDong("doi ID lam doi box/conf/thu tu")
        arrays["data"] = moi
        arrays["cls_goc"] = goc[:, 5].astype(np.int16)
        extra = {"r64": True, "detector": det, "checkpoint_sha256": sha, "dataset": ds, "split": t["split"],
                 "names": {str(k): str(n_) for k, n_ in sorted(dict(model.names).items())},
                 "anh_xa_coco": None if bang is None else {str(k): v_ for k, v_ in sorted(bang.items())},
                 "timeline": {"n": n, "fps": t["fps"]}, "nguon": nguon, "ma_sha256": ma,
                 "gt_luc_sinh_sha256": sha256_file(gt_root / v / "gt" / "gt.txt"),
                 "ghi_chu_gt": "GT luc sinh cache chi de doi chieu; GT cham co the sua sau (vd B1) ma khong doi detection"}
        det_cache.save_video(cdir, v, arrays, extra=extra, spec=spec)
        tom = kiem_noi_dung_cache(ds, det, v, smoke)               # hau kiem TRUOC khi ghi marker
        ghi_json_nguyen_tu(cdir / f"{v}.xong.json", {
            "detector": det, "video": v, "dataset": ds, "split": t["split"], "giay_suy_luan": round(giay, 2),
            **tom, "tao_luc": bay_gio(),
            "sha256": {f: sha256_file(cdir / f) for f in (f"{v}.npz", f"{v}.json", "meta.json")}})
        print(f"[cache] {cid}/{v}: {n} frame, {tom['n_boxes']} box, toi da {tom['max_n_per_frame']}/frame, "
              f"{giay:.0f} s ({n / giay:.1f} fps)", flush=True)
    return 0


def buoc_xac_minh(ds: str, smoke: bool) -> int:
    """Bien ban xac minh BO SUNG cho don vi cache da sinh (marker cu): KHONG sua metadata cu; ghi rieng
    <video>.xac_minh_bo_sung.json voi thoi diem, ket qua cong, manifest nguon va hash ma TAI THOI DIEM XAC MINH."""
    t = cau_hinh(ds, smoke)
    ma = hash_ma()
    for det in DETECTOR:
        cdir = det_cache.CACHE_ROOT / ten_cache(ds, det, smoke)
        for v in t["videos"]:
            mk = cdir / f"{v}.xong.json"
            if not mk.exists():
                continue
            if "dataset" in json.loads(mk.read_text(encoding="utf-8")):
                continue                                         # marker moi: provenance luc sinh da du
            if (cdir / f"{v}.xac_minh_bo_sung.json").exists():
                continue                                         # khong ghi de bien ban da co
            tom = kiem_don_vi_cache(ds, det, v, smoke, cho_legacy_chua_xac_minh=True)
            ghi_json_nguyen_tu(cdir / f"{v}.xac_minh_bo_sung.json", {
                "loai": "xac minh bo sung SAU khi sinh (khong phai provenance luc sinh)", "thoi_diem": bay_gio(),
                "cong": "kiem_don_vi_cache (marker + sha256 + noi dung) DAT", **tom,
                "nguon_tai_thoi_diem_xac_minh": nguon_don_vi(ds, t, v), "ma_sha256_tai_thoi_diem_xac_minh": ma,
                "marker_sha256": sha256_file(mk)})
            print(f"[xac-minh] {cdir.name}/{v}: DAT ({tom['n_boxes']} box)", flush=True)
    return 0


# --------------------------------------------------------------------------------------------- track + cham
def ten_run(det: str, cond: str, loc: str, smoke: bool) -> str:
    return f"r64{'smoke' if smoke else ''}-{det}-{cond}-{loc}"


def cham(split: str, tracker: str, videos: list[str]) -> dict:
    import boot_stats
    import run_trackeval as rte
    out = config.RESULTS_DIR / "trackeval" / split / tracker
    if out.exists():
        raise LoiHopDong(f"da ton tai {out}")
    results, _ = rte.run(split, tracker, videos, "mot")
    df = rte.flatten_results(results, tracker)
    out.mkdir(parents=True)
    df[df["video"] != "COMBINED_SEQ"].sort_values("video").to_csv(out / "per_video_metrics.csv", index=False)
    comb = df[df["video"] == "COMBINED_SEQ"]
    comb.to_csv(out / "combined_metrics.csv", index=False)
    boot_stats.save_raw_fields(results, tracker, out / "raw_fields.json")
    r = comb.iloc[0]
    return {"HOTA": float(r["HOTA"]), "IDF1": float(r["IDF1"]), "IDSW": int(r["IDSW"]), "MOTA": float(r["MOTA"])}


def max_time_lost_thuc(yaml: str, buf: int) -> int:
    import run_tracking as rt
    tr = rt.make_tracker(rt.load_tracker_cfg(yaml, {"track_buffer": buf}), rt.tracker_registry())
    # tracker vendored (third_party/ultralytics_trackers/byte_tracker.py) dat max_frames_lost = track_buffer va so voi
    # frame_id - end_frame; ban ultralytics goc goi la max_time_lost
    for ten in ("max_frames_lost", "max_time_lost"):
        if hasattr(tr, ten):
            return int(getattr(tr, ten))
    raise LoiHopDong(f"{type(tr).__name__}: khong co max_frames_lost/max_time_lost")


def hau_kiem_tracking(t: dict, nm: str, buf: int, mtl: int) -> None:
    import pandas as pd
    d = config.PROCESSED_DIR / "trackers" / t["split"] / nm
    rm = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    if rm["track_buffer_frames"] != buf or rm["elapsed_mode"] != "steps" or abs(rm["fps"] - t["fps"]) > 1e-9:
        raise LoiHopDong(f"{nm}: run_meta lech (buffer {rm['track_buffer_frames']} != {buf} / elapsed / fps)")
    if mtl != buf:
        raise LoiHopDong(f"{nm}: max_time_lost thuc {mtl} != {buf}")
    for v in t["videos"]:
        n = so_frame(t, v)
        tm = pd.read_csv(d / "timing" / f"{v}.csv")
        if not np.array_equal(tm["frame"].to_numpy(), np.arange(1, n + 1)) or (tm["n_steps"] != 1).any():
            raise LoiHopDong(f"{nm}/{v}: timing khong du 1..{n} hoac n_steps != 1")
        p = d / "data" / f"{v}.txt"
        a = np.loadtxt(p, delimiter=",", ndmin=2) if p.stat().st_size else np.zeros((0, 10))
        if len(a) and (not np.isfinite(a[:, :7]).all() or a[:, 0].min() < 1 or a[:, 0].max() > n
                       or (a[:, 4] <= 0).any() or (a[:, 5] <= 0).any()):
            raise LoiHopDong(f"{nm}/{v}: output khong huu han / frame ngoai / w,h <= 0")
        if len(a) and dong_trung(a).any():
            raise LoiHopDong(f"{nm}/{v}: tracker trung (frame, id)")


def buoc_track(ds: str, det: str, cond: str, loc: str, smoke: bool, het_han: str | None) -> int:
    t = cau_hinh(ds, smoke)
    yaml, seeds = BO_LOC[loc]
    ten = ten_run(det, cond, loc, smoke)
    names = [ten + (f"-s{s}" if s is not None else "") for s in (seeds or [None])]
    for nm in names:
        if (config.PROCESSED_DIR / "trackers" / t["split"] / nm).exists():
            print("[DUNG] da ton tai", nm); return 2
    n_tong = sum(so_frame(t, v) for v in t["videos"])
    tien_kiem_job(ds, smoke, het_han, f"track {ten}",
                  len(names) * (n_tong * UOC_TRACK_S_FRAME[loc] + 30 * len(t["videos"])) + 60,
                  config.PROCESSED_DIR, len(names) * n_tong * 2000)
    cong = {v: kiem_don_vi_cache(ds, det, v, smoke) for v in t["videos"]}   # cong truoc: cache hoan tat + noi dung
    gt_hash = kiem_gt_cham(t)
    buf = int(round(1.2 * t["fps"]))
    mtl = max_time_lost_thuc(yaml, buf)
    cmd = [sys.executable, "-B", str(GOC / "src" / "run_tracking.py"), "--cache-id", ten_cache(ds, det, smoke),
           "--split-name", t["split"], "--tracker-yaml", yaml, "--tracker-name", ten, "--videos", *t["videos"],
           "--conf-min", str(DIEU_KIEN[cond]), "--classes", *map(str, LOP_TRACKER), "--max-det", "300",
           "--fps", str(t["fps"]), "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
    if seeds:
        cmd += ["--seeds", *map(str, seeds)]
    log_dir = GOC / "handoff" / "logs" / "external_eval" / ("smoke" if smoke else "") / ds / "track_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=GOC, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (log_dir / f"{ten}.log").write_text(f"argv: {cmd}\nexit: {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}",
                                        encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:]); return 3
    for nm in names:
        hau_kiem_tracking(t, nm, buf, mtl)
        kq = cham(t["split"], nm, t["videos"])
        ghi_json_nguyen_tu(config.RESULTS_DIR / "trackeval" / t["split"] / nm / "cong_r64.json",
                           {"cong_truoc_cache": cong, "gt_cham_sha256": gt_hash, "max_time_lost_thuc": mtl,
                            "hau_kiem": "DAT", "ket_qua": kq, "thoi_diem": bay_gio(), "ma_sha256": hash_ma()})
        print(f"[track] {nm}: HOTA {100 * kq['HOTA']:.2f} IDF1 {100 * kq['IDF1']:.2f} IDSW {kq['IDSW']} "
              f"MOTA {100 * kq['MOTA']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    return 0


def trang_thai_run(ds: str, det: str, cond: str, loc: str, smoke: bool) -> tuple[str, list[str]]:
    """Trang thai mot cau hinh (PF: du 3 seed) de chay.sh quyet dinh bo qua / chay / dung (review cuoi GPT 02/11):
    'xong'  : moi seed co output tracker + ket qua cham + cong_r64.json hop le (hau_kiem DAT, dung tap video, hash GT
              cham = GT hien tai, raw_fields dung tap video);
    'chua'  : chua co thu muc nao cua moi seed;
    'do'    : con lai (thieu seed, cong hong, lech) -> KHONG ghi de, bao ro ly do."""
    t = cau_hinh(ds, smoke)
    _, seeds = BO_LOC[loc]
    names = [ten_run(det, cond, loc, smoke) + (f"-s{s}" if s is not None else "") for s in (seeds or [None])]
    co = [n for n in names if (config.PROCESSED_DIR / "trackers" / t["split"] / n).exists()
          or (config.RESULTS_DIR / "trackeval" / t["split"] / n).exists()]
    if not co:
        return "chua", []
    ly_do = [f"{n}: chua bat dau" for n in names if n not in co]
    gt_hien = {v: sha256_file(config.PROCESSED_DIR / t["split"] / v / "gt" / "gt.txt") for v in t["videos"]}
    for n in co:
        kq = config.RESULTS_DIR / "trackeval" / t["split"] / n
        try:
            c = json.loads((kq / "cong_r64.json").read_text(encoding="utf-8"))
            raw = json.loads((kq / "raw_fields.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            ly_do.append(f"{n}: thieu/hong cong_r64.json hoac raw_fields.json ({type(e).__name__})"); continue
        if c.get("hau_kiem") != "DAT":
            ly_do.append(f"{n}: hau_kiem != DAT")
        if sorted(c.get("cong_truoc_cache", {})) != sorted(t["videos"]) or sorted(raw) != sorted(t["videos"]):
            ly_do.append(f"{n}: tap video lech")
        if c.get("gt_cham_sha256") != gt_hien:
            ly_do.append(f"{n}: GT cham khac GT hien tai")
        if not (kq / "combined_metrics.csv").exists() or not all(
                (config.PROCESSED_DIR / "trackers" / t["split"] / n / "data" / f"{v}.txt").exists() for v in t["videos"]):
            ly_do.append(f"{n}: thieu combined_metrics hoac output tracker")
    return ("xong" if not ly_do else "do"), ly_do


# --------------------------------------------------------------------------------------------- tong hop
SCENE_SVMOT = {"scene_05": ["seq_22_morning", "seq_23_morning", "seq_24_morning", "seq_25_morning"],
               "scene_06": ["seq_26_evening", "seq_27_evening", "seq_28_evening", "seq_29_evening", "seq_30_evening"]}


def buoc_tong_hop(ds: str, smoke: bool) -> int:
    """COMBINED tung run (PF: tung seed + trung binh metric COMBINED 3 seed), tung chuoi/camera, SVMOT theo scene — moi
    so gop deu TAI TO HOP tu raw_fields (boot_stats.combine), khong trung binh metric giua chuoi; kiem tai to hop toan tap
    == COMBINED. Thieu run -> trang thai PARTIAL."""
    import csv
    import boot_stats
    import pandas as pd
    t = cau_hinh(ds, smoke)
    goc = config.RESULTS_DIR / "trackeval" / t["split"]
    hang, chuoi, nhom, thieu, lech = [], [], [], [], []
    M = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")
    for det in DETECTOR:
        for cond in DIEU_KIEN:
            for loc, (_, seeds) in BO_LOC.items():
                pf = []
                for s in (seeds or [None]):
                    nm = ten_run(det, cond, loc, smoke) + (f"-s{s}" if s is not None else "")
                    d = goc / nm
                    if not (d / "combined_metrics.csv").exists() or not (d / "cong_r64.json").exists() and not smoke:
                        thieu.append(nm); continue
                    c = pd.read_csv(d / "combined_metrics.csv").iloc[0]
                    raw = boot_stats.load_raw_fields(d / "raw_fields.json")
                    vids = sorted(raw)
                    if vids != sorted(t["videos"]):
                        lech.append(f"{nm}: tap video {vids}")
                    re = boot_stats.combine(raw, vids)
                    for m in ("HOTA", "IDF1", "MOTA", "IDSW"):
                        if abs(re[m] - float(c[m])) > 1e-9:
                            lech.append(f"{nm}: tai to hop {m} {re[m]} != {c[m]}")
                    goc_hang = {"detector": det, "dieu_kien": cond, "bo_loc": loc, "seed": "" if s is None else s}
                    hang.append({**goc_hang, **{m: re[m] for m in M}})
                    pf.append(re)
                    for v in vids:
                        chuoi.append({**goc_hang, "chuoi": v, **{m: x for m, x in boot_stats.combine(raw, [v]).items()
                                                                  if m in M}})
                    if ds == "svmot" and not smoke:
                        for sc, vs in SCENE_SVMOT.items():
                            nhom.append({**goc_hang, "scene": sc, **{m: x for m, x in boot_stats.combine(raw, vs).items()
                                                                      if m in M}})
                if seeds and len(pf) == len(seeds):
                    hang.append({"detector": det, "dieu_kien": cond, "bo_loc": loc, "seed": "tb3",
                                 **{m: float(np.mean([r[m] for r in pf])) for m in M}})
    ra = GOC / "handoff" / "logs" / "external_eval" / ("smoke" if smoke else "") / ds
    ra.mkdir(parents=True, exist_ok=True)
    for ten_f, rows in (("combined_tung_run.csv", hang), ("tung_chuoi.csv", chuoi), ("theo_scene.csv", nhom)):
        if rows:
            with open(ra / ten_f, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_ky_vong = len(DETECTOR) * len(DIEU_KIEN) * sum(len(s or [None]) for _, s in BO_LOC.values())
    trang_thai = "XONG" if not thieu and not lech else ("LECH" if lech else "PARTIAL")
    ghi_json_nguyen_tu(ra / "tong_hop_trang_thai.json", {"trang_thai": trang_thai, "n_run_ky_vong": n_ky_vong,
                                                          "n_run_co": len([h for h in hang if h["seed"] != "tb3"]),
                                                          "thieu": thieu, "lech": lech, "thoi_diem": bay_gio()})
    print(f"[tong-hop] {ds}: {trang_thai} — {n_ky_vong - len(thieu)}/{n_ky_vong} run, lech {len(lech)}")
    for x in lech:
        print("  [LECH]", x)
    return 0 if trang_thai == "XONG" else (10 if trang_thai == "PARTIAL" else 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("buoc", choices=["gt", "cache", "xac-minh", "track", "kiem-run", "tong-hop"])
    ap.add_argument("ds", choices=sorted(TAP))
    ap.add_argument("con_lai", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--het-han", default=None, help="ISO thoi diem phai xong (vd 2026-11-02T07:30:00+07:00)")
    a = ap.parse_args(argv)
    try:
        if a.buoc == "gt":
            return buoc_gt(a.ds, a.smoke)
        if a.buoc == "cache":
            (det,) = a.con_lai
            if det not in DETECTOR:
                raise LoiHopDong(f"detector {det}")
            return buoc_cache(a.ds, det, a.smoke, a.videos, a.het_han)
        if a.buoc == "xac-minh":
            return buoc_xac_minh(a.ds, a.smoke)
        if a.buoc == "track":
            det, cond, loc = a.con_lai
            if det not in DETECTOR or cond not in DIEU_KIEN or loc not in BO_LOC:
                raise LoiHopDong(f"tham so {a.con_lai}")
            return buoc_track(a.ds, det, cond, loc, a.smoke, a.het_han)
        if a.buoc == "kiem-run":                     # ma thoat: 0 xong (bo qua), 1 chua (chay), 2 do (dung)
            det, cond, loc = a.con_lai
            tt, ly_do = trang_thai_run(a.ds, det, cond, loc, a.smoke)
            print(f"[kiem-run] {ten_run(det, cond, loc, a.smoke)}: {tt}", *ly_do, sep="\n  ")
            return {"xong": 0, "chua": 1}.get(tt, 2)
        return buoc_tong_hop(a.ds, a.smoke)
    except HetNganSach as e:
        print("[PARTIAL - het ngan sach]", e)
        return 10
    except LoiHopDong as e:
        print("[LOI HOP DONG]", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
