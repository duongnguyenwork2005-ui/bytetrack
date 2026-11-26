"""
danh_gia_r66.py -- R66: danh gia he thong (3 detector R54 x CV/EKF/UKF/PF500 s0/1/2 x c25/c10) tren dataset moi.

Giao thuc: GPT 02/11 `D:/LABOTARY/automation/runs_local/hoi_codex_R66_dataset_moi_pf_20261102/answer.json` (Q1 Sherbrooke,
Q4 HELMET test36, Q5 van hanh). Wrapper RIENG, ke thua co che R64 (cong cache/provenance/hau kiem/ngan sach) bang cach GOI
helper da khoa trong `src/r64_ngoai/danh_gia_ngoai.py`; KHONG sua ma R64.

    python -B src/r66_ngoai/danh_gia_r66.py gt       <ds> [--smoke]
    python -B src/r66_ngoai/danh_gia_r66.py cache    <ds> <det> [--smoke] [--het-han ISO]
    python -B src/r66_ngoai/danh_gia_r66.py track    <ds> <det> <c25|c10> <loc> [--smoke] [--het-han ISO]
    python -B src/r66_ngoai/danh_gia_r66.py kiem-run <ds> <det> <cond> <loc> [--smoke]   (0 xong, 1 chua, 2 do/hong)
    python -B src/r66_ngoai/danh_gia_r66.py tong-hop <ds> [--smoke]
Ma thoat: 0 xong; 10 het ngan sach (PARTIAL); 2 loi hop dong; 3 loi subprocess.

Sherbrooke (Q1): split `Sherbrooke-cars-f2754-3754-roi-v1`, frame nguon 2754-3754 -> MOT 1..1001, 30 fps, GT DUY NHAT
`sherbrooke_gt_cars.sqlite` (15 ID ke ca 'unknown'), xyxy -> xywh (w = x2 - x1, KHONG +1: x2 lon nhat = 799 = W - 1),
tracker chi lop COCO 2 (car). Detector/tracker chay tren anh NGUYEN; truoc khi cham ap CUNG quy tac "tam box trong
annotated-zone mask (gia tri 1)" cho GT va du doan, giu ID + timeline; output tho giu o `data_tho/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(GOC / "src" / "r64_ngoai"))
import config  # noqa: E402
import danh_gia_ngoai as r64  # noqa: E402
import det_cache  # noqa: E402
from mot_datasets.base import FrameRef  # noqa: E402

LoiHopDong, HetNganSach = r64.LoiHopDong, r64.HetNganSach
NGUON = GOC / "data" / "processed" / "_r66_nguon"
QUYET_DINH = "D:/LABOTARY/automation/runs_local/hoi_codex_R66_dataset_moi_pf_20261102/answer.json"
MA_THUC_DUNG = ["src/r66_ngoai/danh_gia_r66.py", "src/r64_ngoai/danh_gia_ngoai.py", "src/det_cache.py",
                "src/detector_classes.py", "src/run_tracking.py", "src/run_trackeval.py", "src/boot_stats.py"]
SMOKE_N = 128
SHER_LOCK = GOC / "handoff/logs/external_eval/r66/sherbrooke/v2/source_lock.json"
DU_PHONG_GIAY, DU_PHONG_DIA = 1800, 5 * 1024 ** 3
#: uoc luong de canh ngan sach (s/frame); do lai o smoke
UOC_CACHE_S_FRAME = {"sherbrooke": 1 / 8.0, "helmet": 1 / 8.0}
UOC_TRACK_S_FRAME = {"cv": 0.004, "ekf": 0.004, "ukf": 0.006, "pf500": 0.03}

TAP = {
    "sherbrooke": {"split": "Sherbrooke-cars-f2754-3754-roi-v1", "fps": 30.0, "hw": (600, 800),
                   "videos": ["sherbrooke"], "n_frame": 1001, "n_goc": {"sherbrooke": 1001},
                   "anh": "F:/LABOTARY_data/sherbrooke/sherbrooke_frames/{:08d}.jpg", "frame_dau_nguon": 2754,
                   "gt_sqlite": "F:/LABOTARY_data/sherbrooke/sherbrooke_annotations/sherbrooke_gt_cars.sqlite",
                   "mask": NGUON / "sherbrooke" / "sherbrooke_mask.png", "lop_tracker": [2]},
}


def sha256_file(p) -> str:
    return r64.sha256_file(Path(p))


def ghi_json(p: Path, obj) -> None:
    r64.ghi_json_nguyen_tu(p, obj)


def cau_hinh(ds: str, smoke: bool) -> dict:
    t = dict(TAP[ds])
    if smoke:
        t["split"] += "-smoke128"
        t["n_frame"] = SMOKE_N
        t["n_goc"] = {v: SMOKE_N for v in t["videos"]}
    return t


def so_frame(t: dict, v: str) -> int:
    return t["n_goc"][v]


def giao_thuc(ds: str) -> dict:
    """Moi thu xac dinh noi dung danh gia -> hash tham gia ten cache/fingerprint (Q5)."""
    t = TAP[ds]
    g = {"ds": ds, "split": t["split"], "fps": t["fps"], "hw": t["hw"], "videos": t["videos"], "n_goc": t["n_goc"],
         "lop_tracker": t["lop_tracker"], "frame_dau_nguon": t.get("frame_dau_nguon")}
    if t.get("gt_sqlite"):
        g["gt_sqlite_sha256"] = sha256_file(t["gt_sqlite"])
    if t.get("mask"):
        g["mask_sha256"] = sha256_file(t["mask"])
    return g


def hash_giao_thuc(ds: str) -> str:
    return sha256_file(SHER_LOCK)[:16]


def ten_cache(ds: str, det: str, smoke: bool) -> str:
    return f"r66v2{'smoke' if smoke else ''}_{ds}_{hash_giao_thuc(ds)}_{det}_c0.05"


def ten_run(det: str, cond: str, loc: str, smoke: bool) -> str:
    return f"r66{'smoke' if smoke else ''}-{det}-{cond}-{loc}"


def log_dir(ds: str, smoke: bool) -> Path:
    return GOC / "handoff" / "logs" / "external_eval" / "r66" / ds / "v2" / ("smoke" if smoke else "full")


def tien_kiem(ds: str, smoke: bool, het_han: str | None, viec: str, giay_uoc: float, thu_muc: Path,
              byte_uoc: int) -> None:
    """Nhu R64: bat buoc --het-han (tru smoke), con lai >= du toan + 30 phut, o ghi >= du toan + 5 GiB, ghi jsonl."""
    import datetime as dt
    trong = shutil.disk_usage(thu_muc).free
    con = None if het_han is None else dt.datetime.fromisoformat(het_han).timestamp() - time.time()
    loi = []
    if het_han is None:
        loi.append("thieu --het-han")
    if con is not None and giay_uoc + DU_PHONG_GIAY > con:
        loi.append(f"uoc {giay_uoc:.0f} s + du phong {DU_PHONG_GIAY} s > con {con:.0f} s")
    if trong < byte_uoc + DU_PHONG_DIA:
        loi.append(f"o {thu_muc} con {trong / 1e9:.1f} GB < uoc + 5 GiB")
    d = log_dir(ds, smoke); d.mkdir(parents=True, exist_ok=True)
    with open(d / "ngan_sach.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"thoi_diem": r64.bay_gio(), "viec": viec, "het_han": het_han,
                            "giay_con": None if con is None else round(con), "giay_uoc": round(giay_uoc),
                            "o_dich": str(thu_muc), "byte_trong": trong, "byte_uoc": byte_uoc, "dat": not loi,
                            "loi": loi}, ensure_ascii=False) + "\n")
    if loi:
        raise HetNganSach(f"{viec}: " + "; ".join(loi))


# ------------------------------------------------------------------------------------------------ ROI
def mask_roi(t: dict) -> np.ndarray | None:
    if not t.get("mask"):
        return None
    from PIL import Image
    m = np.array(Image.open(t["mask"]))
    if m.shape != tuple(t["hw"]) or set(np.unique(m).tolist()) - {0, 1}:
        raise LoiHopDong(f"mask {m.shape} / gia tri {np.unique(m)} khong hop le")
    return m


def trong_roi(m: np.ndarray, x, y, w, h) -> np.ndarray:
    from sher_metric import roi_contains
    return roi_contains(m, x, y, w, h)


# ------------------------------------------------------------------------------------------------ GT
def doc_gt_sherbrooke(t: dict) -> tuple[np.ndarray, dict]:
    c = sqlite3.connect(f"file:{t['gt_sqlite']}?mode=ro", uri=True)
    rows = c.execute("select object_id, frame_number, x_top_left, y_top_left, x_bottom_right, y_bottom_right "
                     "from bounding_boxes order by object_id, frame_number").fetchall()
    obj = {int(o): (int(r), d) for o, r, d in c.execute("select object_id, road_user_type, description from objects")}
    ids = sorted({int(r[0]) for r in rows})
    if sorted(obj) != ids:
        raise LoiHopDong(f"bang objects {sorted(obj)} != id trong bounding_boxes {ids}")
    map_id = {o: i + 1 for i, o in enumerate(ids)}               # id nguon (co 0) -> id MOT duong, co dinh
    f0, n = t["frame_dau_nguon"], TAP["sherbrooke"]["n_frame"]
    a = []
    for o, f, x1, y1, x2, y2 in rows:
        if not (f0 <= f <= f0 + n - 1):
            raise LoiHopDong(f"frame nguon {f} ngoai {f0}..{f0 + n - 1}")
        if not (x2 > x1 and y2 > y1):
            raise LoiHopDong(f"box suy bien id {o} frame {f}")
        a.append([f - f0 + 1, map_id[int(o)], x1, y1, x2 - x1, y2 - y1, 1, 1, 1])
    meta = {"so_box_nguon": len(rows), "so_id": len(ids),
            "map_id": {str(o): map_id[o] for o in ids},
            "objects": {str(o): {"road_user_type": obj[o][0], "description": obj[o][1]} for o in ids},
            "quy_uoc_canh": "w = x2 - x1, h = y2 - y1 (KHONG +1); x2 lon nhat 799 = W-1"}
    return np.array(a, dtype=np.float64), meta


def buoc_gt(ds: str, smoke: bool) -> int:
    t = cau_hinh(ds, smoke)
    a, meta = doc_gt_sherbrooke(t)
    m = mask_roi(t)
    vao = trong_roi(m, a[:, 2], a[:, 3], a[:, 4], a[:, 5])
    bo = a[~vao]
    a = a[vao]
    if smoke:
        a = a[a[:, 0] <= SMOKE_N]
        bo = bo[bo[:, 0] <= SMOKE_N]
    a = a[np.lexsort((a[:, 1], a[:, 0]))]
    kiem = r64.kiem_gt(a, t["n_frame"], "sherbrooke")
    root = config.PROCESSED_DIR / t["split"]
    v = t["videos"][0]
    (root / v / "gt").mkdir(parents=True, exist_ok=True)
    np.savetxt(root / v / "gt" / "gt.txt", a, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f",
                                                                   "%d", "%d", "%.3f"])
    anh = Path(t["anh"].format(t["frame_dau_nguon"])).parent
    (root / v / "seqinfo.ini").write_text(
        f"[Sequence]\nname={v}\nimDir={anh}\nframeRate={int(t['fps'])}\nseqLength={t['n_frame']}\n"
        f"imWidth={t['hw'][1]}\nimHeight={t['hw'][0]}\nimExt=.jpg\n", encoding="utf-8")
    seqmap = config.PROCESSED_DIR / "seqmaps" / f"{t['split']}.txt"
    seqmap.write_text("name\n" + "\n".join(t["videos"]) + "\n", encoding="utf-8")
    ghi_json(root / "gt_log.json", {
        "quyet_dinh": QUYET_DINH + " Q1", "thoi_diem": r64.bay_gio(), "giao_thuc": giao_thuc(ds), **meta,
        "roi": {"quy_tac": "tam box trong mask==1 (floor, kep)", "gt_bi_loai_ngoai_roi": int(len(bo)),
                "dong_bi_loai": bo[:, :6].tolist()},
        "so_dong_gt_cham": int(len(a)), "kiem": kiem, "gt_sha256": sha256_file(root / v / "gt" / "gt.txt")})
    print(f"[gt] {t['split']}: {len(a)} dong, {len(set(a[:, 1].astype(int)))} ID, loai ngoai ROI {len(bo)}")
    return 0


def kiem_gt_cham(t: dict) -> dict:
    out = {}
    for v in t["videos"]:
        p = config.PROCESSED_DIR / t["split"] / v / "gt" / "gt.txt"
        a = np.loadtxt(p, delimiter=",", ndmin=2)
        r64.kiem_gt(a, so_frame(t, v), v)
        out[v] = sha256_file(p)
    return out


# ------------------------------------------------------------------------------------------------ cache
def anh_don_vi(t: dict, v: str) -> list[Path]:
    n = so_frame(t, v)
    ds = [k for k, x in TAP.items() if x["split"] == t["split"].removesuffix("-smoke128")][0]
    if ds == "sherbrooke":
        anh = [Path(t["anh"].format(t["frame_dau_nguon"] + i)) for i in range(n)]
    else:
        raise LoiHopDong(f"chua co adapter anh cho {ds}")
    thieu = [p for p in anh if not p.exists()]
    if thieu:
        raise LoiHopDong(f"{v}: thieu {len(thieu)} anh, vd {thieu[0]}")
    return anh


def nguon_don_vi(t: dict, v: str) -> dict:
    anh = anh_don_vi(t, v)
    tung = [sha256_file(p) for p in anh]
    return {"loai": "anh_jpg", "thu_muc": str(anh[0].parent), "frame": [1, len(anh)], "ten_dau": anh[0].name,
            "ten_cuoi": anh[-1].name, "sha256_gop_tung_anh": hashlib.sha256("\n".join(tung).encode()).hexdigest(),
            "fps": t["fps"]}


def kiem_noi_dung_cache(ds: str, det: str, v: str, smoke: bool) -> dict:
    """Ban R66 cua r64.kiem_noi_dung_cache: cung cac kiem (meta/spec/fingerprint, json don vi, mapping tinh lai,
    timeline 1..N + time_s, n_per_frame, orig_shape, huu han, conf, box, cls = anh xa(cls_goc))."""
    t = cau_hinh(ds, smoke)
    c = r64.detector_r54(det)
    cdir = det_cache.CACHE_ROOT / ten_cache(ds, det, smoke)
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
    ky_vong = {"backend": "ultralytics", "weights_sha256": c["sha256"], "imgsz": 640, "conf": 0.05, "iou": 0.7,
               "classes": None, "max_det": 3000, "half": False, "agnostic_nms": False, "augment": False, "batch": 1,
               "mask_ignored_regions": False, "dataset": ds}
    lech = {k: (meta.get(k), x) for k, x in ky_vong.items() if meta.get(k) != x}
    if meta.get("rect", True) is not True:
        lech["rect"] = (meta.get("rect"), True)
    if lech:
        raise LoiHopDong(f"{cdir.name}/meta.json lech: {lech}")
    fp = hashlib.sha256(json.dumps(det_cache.meta_gen_spec(meta), sort_keys=True, default=str).encode()).hexdigest()
    if meta.get("spec_sha256") != fp:
        raise LoiHopDong(f"{cdir.name}: spec_sha256 khong khop fingerprint tinh lai")
    info = json.loads((cdir / f"{v}.json").read_text(encoding="utf-8"))
    lock = json.loads(SHER_LOCK.read_text(encoding="utf-8"))
    locked = lock["splits"][t["split"]]
    for field, expected in {"nguon": locked["images"], "ma_sha256": lock["cache_code"],
                            "split": t["split"], "gt_luc_sinh_sha256": locked["gt"],
                            "names": lock["detectors"][det]["names"],
                            "giao_thuc": lock["inputs"]}.items():
        if info.get(field) != expected:
            raise LoiHopDong(f"{cdir.name}/{v}: provenance mismatch for {field}")
    if locked["gt"] != sha256_file(config.PROCESSED_DIR / t["split"] / v / "gt" / "gt.txt"):
        raise LoiHopDong(f"{v}: GT changed")
    if info.get("spec_sha256") != meta.get("spec_sha256"):
        raise LoiHopDong(f"{cdir.name}/{v}: spec_sha256 json != meta")
    if (info.get("detector") != det or info.get("dataset") != ds or info.get("checkpoint_sha256") != c["sha256"]
            or info.get("giao_thuc_hash") != hash_giao_thuc(ds)):
        raise LoiHopDong(f"{cdir.name}/{v}: json detector/dataset/checkpoint/giao thuc lech")
    for k in ("nguon", "ma_sha256", "split", "gt_luc_sinh_sha256"):
        if not info.get(k):
            raise LoiHopDong(f"{cdir.name}/{v}: thieu provenance {k}")
    names = {int(k): n_ for k, n_ in info["names"].items()}
    bang = r64.bang_anh_xa_coco(names, c["loai_checkpoint"])
    bang_json = None if info["anh_xa_coco"] is None else {int(k): int(x) for k, x in info["anh_xa_coco"].items()}
    if bang != bang_json:
        raise LoiHopDong(f"{cdir.name}/{v}: anh_xa_coco != bang tinh lai")
    z = np.load(cdir / f"{v}.npz")
    n = so_frame(t, v)
    loi = []
    if not np.array_equal(z["frame_idx"], np.arange(1, n + 1)):
        loi.append(f"frame_idx khong phai 1..{n}")
    if len(z["time_s"]) != n or not np.allclose(z["time_s"], np.arange(n) / t["fps"], atol=1e-9, rtol=0):
        loi.append("time_s != (i-1)/fps")
    if (int(z["n_per_frame"].sum()) != len(z["data"]) or len(z["n_per_frame"]) != n
            or np.any(z["n_per_frame"] < 0) or np.any(z["n_per_frame"] >= 3000)):
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
        if z["cls_goc"].shape != (len(d),) or not np.array_equal(r64.doi_id_sang_coco(goc, bang)[:, 5], d[:, 5]):
            loi.append("cls != anh xa(cls_goc)")
    if loi:
        raise LoiHopDong(f"{cdir.name}/{v}: " + "; ".join(loi))
    return {"n_frames": n, "n_boxes": int(len(d)), "max_n_per_frame": int(z["n_per_frame"].max()) if n else 0}


def kiem_don_vi_cache(ds: str, det: str, v: str, smoke: bool) -> dict:
    """Marker hoan tat + sha256 artifact + noi dung (khong co duong legacy: cache R66 deu sinh bang ma moi)."""
    cdir = det_cache.CACHE_ROOT / ten_cache(ds, det, smoke)
    mk = cdir / f"{v}.xong.json"
    if not mk.exists():
        raise LoiHopDong(f"{cdir.name}/{v}: thieu marker")
    m = json.loads(mk.read_text(encoding="utf-8"))
    if set(m.get("sha256", {})) != {f"{v}.npz", f"{v}.json", "meta.json"}:
        raise LoiHopDong(f"{cdir.name}/{v}: incomplete cache hash list")
    if (m.get("detector"), m.get("video"), m.get("dataset"), m.get("split")) != (det, v, ds, cau_hinh(ds, smoke)["split"]):
        raise LoiHopDong(f"{cdir.name}/{v}: marker identity mismatch")
    for f, h in m["sha256"].items():
        if sha256_file(cdir / f) != h:
            raise LoiHopDong(f"{cdir.name}/{f}: sha256 lech marker")
    return {"marker_sha256": sha256_file(mk), **kiem_noi_dung_cache(ds, det, v, smoke)}


def buoc_cache(ds: str, det: str, smoke: bool, het_han: str | None) -> int:
    from ultralytics import YOLO
    t = cau_hinh(ds, smoke)
    c = r64.detector_r54(det)
    ck = GOC / c["checkpoint"]
    sha = sha256_file(ck)
    if sha != c["sha256"]:
        raise LoiHopDong(f"sha256 checkpoint {det} lech")
    cid = ten_cache(ds, det, smoke)
    cdir = det_cache.CACHE_ROOT / cid
    for v in t["videos"]:
        if not (cdir / f"{v}.xong.json").exists() and any((cdir / f"{v}{s}").exists() for s in (".npz", ".json")):
            raise LoiHopDong(f"{cid}/{v}: don vi do (artifact khong marker) -- khong ghi de")
    lam = []
    for v in t["videos"]:
        if (cdir / f"{v}.xong.json").exists():
            kiem_don_vi_cache(ds, det, v, smoke)
            print(f"[cache] {cid}/{v}: da xong + qua cong -- bo qua", flush=True)
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
    bang = r64.bang_anh_xa_coco(model.names, c["loai_checkpoint"])
    ma = {m: sha256_file(GOC / m) for m in MA_THUC_DUNG}
    for v in lam:
        n = so_frame(t, v)
        tien_kiem(ds, smoke, het_han, f"cache {cid}/{v}", n * UOC_CACHE_S_FRAME[ds] + 60, det_cache.CACHE_ROOT,
                  n * 2048 + 1024 ** 2)
        nguon = nguon_don_vi(t, v)
        frames = [FrameRef(index=i, path=p, time_s=(i - 1) / t["fps"]) for i, p in enumerate(anh_don_vi(t, v), 1)]
        t0 = time.time()
        arrays = det_cache.build_video_yolo(model, frames, [], spec, t["hw"])
        giay = time.time() - t0
        if arrays["unreadable"].any() or arrays["truncated"].any():
            raise LoiHopDong(f"{v}: unreadable/truncated -- chan")
        goc = arrays["data"]
        moi = r64.doi_id_sang_coco(goc, bang)
        if moi.shape != goc.shape or not np.array_equal(moi[:, :5], goc[:, :5]):
            raise LoiHopDong("doi ID lam doi box/conf/thu tu")
        arrays["data"] = moi
        arrays["cls_goc"] = goc[:, 5].astype(np.int16)
        extra = {"r66": True, "detector": det, "checkpoint_sha256": sha, "dataset": ds, "split": t["split"],
                 "giao_thuc_hash": hash_giao_thuc(ds), "giao_thuc": giao_thuc(ds),
                 "names": {str(k): str(n_) for k, n_ in sorted(dict(model.names).items())},
                 "anh_xa_coco": None if bang is None else {str(k): v_ for k, v_ in sorted(bang.items())},
                 "timeline": {"n": n, "fps": t["fps"]}, "nguon": nguon, "ma_sha256": ma,
                 "gt_luc_sinh_sha256": sha256_file(config.PROCESSED_DIR / t["split"] / v / "gt" / "gt.txt")}
        det_cache.save_video(cdir, v, arrays, extra=extra, spec=spec)
        tom = kiem_noi_dung_cache(ds, det, v, smoke)
        ghi_json(cdir / f"{v}.xong.json", {"detector": det, "video": v, "dataset": ds, "split": t["split"],
                                           "giay_suy_luan": round(giay, 2), **tom, "tao_luc": r64.bay_gio(),
                                           "sha256": {f: sha256_file(cdir / f)
                                                      for f in (f"{v}.npz", f"{v}.json", "meta.json")}})
        print(f"[cache] {cid}/{v}: {n} frame, {tom['n_boxes']} box, {giay:.0f} s ({n / giay:.1f} fps)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ track
def ap_roi_output(t: dict, run_dir: Path) -> dict:
    """Ap CUNG quy tac ROI cua GT len output tracker truoc khi cham; giu ID + timeline; ban tho -> data_tho/."""
    m = mask_roi(t)
    if m is None:
        return {}
    tho = run_dir / "data_tho"
    if tho.exists():
        raise LoiHopDong(f"{run_dir.name}: data_tho da ton tai")
    shutil.copytree(run_dir / "data", tho)
    out = {}
    for v in t["videos"]:
        p = run_dir / "data" / f"{v}.txt"
        dong = p.read_bytes().splitlines(keepends=True)          # GIU NGUYEN byte tung dong, chi loc dong
        a = np.array([[float(x) for x in d.split(b",")[:7]] for d in dong]).reshape(-1, 7)
        giu = trong_roi(m, a[:, 2], a[:, 3], a[:, 4], a[:, 5]) if len(a) else np.zeros(0, bool)
        p.write_bytes(b"".join(d for d, g in zip(dong, giu) if g))
        out[v] = {"dong_tho": int(len(a)), "dong_giu": int(giu.sum()),
                  "sha256_tho": sha256_file(tho / f"{v}.txt"), "sha256_roi": sha256_file(p)}
    return out


def buoc_track(ds: str, det: str, cond: str, loc: str, smoke: bool, het_han: str | None) -> int:
    t = cau_hinh(ds, smoke)
    yaml, seeds = r64.BO_LOC[loc]
    ten = ten_run(det, cond, loc, smoke)
    names = [ten + (f"-s{s}" if s is not None else "") for s in (seeds or [None])]
    for nm in names:
        if (config.PROCESSED_DIR / "trackers" / t["split"] / nm).exists():
            print("[DUNG] da ton tai", nm); return 2
    n_tong = sum(so_frame(t, v) for v in t["videos"])
    tien_kiem(ds, smoke, het_han, f"track {ten}", len(names) * (n_tong * UOC_TRACK_S_FRAME[loc] + 30) + 60,
              config.PROCESSED_DIR, len(names) * n_tong * 2000)
    cong = {v: kiem_don_vi_cache(ds, det, v, smoke) for v in t["videos"]}
    gt_hash = kiem_gt_cham(t)
    buf = int(round(1.2 * t["fps"]))
    mtl = r64.max_time_lost_thuc(yaml, buf)
    cmd = [sys.executable, "-B", str(GOC / "src" / "run_tracking.py"), "--cache-id", ten_cache(ds, det, smoke),
           "--split-name", t["split"], "--tracker-yaml", yaml, "--tracker-name", ten, "--videos", *t["videos"],
           "--conf-min", str(r64.DIEU_KIEN[cond]), "--classes", *map(str, t["lop_tracker"]), "--max-det", "300",
           "--fps", str(t["fps"]), "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
    if seeds:
        cmd += ["--seeds", *map(str, seeds)]
    ld = log_dir(ds, smoke) / "track_logs"; ld.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=GOC, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (ld / f"{ten}.log").write_text(f"argv: {cmd}\nexit: {p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}",
                                   encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:]); return 3
    for nm in names:
        r64.hau_kiem_tracking(t, nm, buf, mtl)                     # tren output THO (truoc ROI)
        roi = ap_roi_output(t, config.PROCESSED_DIR / "trackers" / t["split"] / nm)
        kq = r64.cham(t["split"], nm, t["videos"])
        ghi_json(config.RESULTS_DIR / "trackeval" / t["split"] / nm / "cong_r66.json",
                 {"cong_truoc_cache": cong, "gt_cham_sha256": gt_hash, "max_time_lost_thuc": mtl, "roi": roi,
                  "giao_thuc_hash": hash_giao_thuc(ds), "ma_sha256": {m: sha256_file(GOC / m) for m in MA_THUC_DUNG},
                  "thoi_diem": r64.bay_gio(), "combined": kq})
        print(f"[track] {nm}: HOTA {kq['HOTA'] * 100:.2f} IDF1 {kq['IDF1'] * 100:.2f} IDSW {kq['IDSW']}", flush=True)
    return 0


def trang_thai_run(ds: str, det: str, cond: str, loc: str, smoke: bool) -> tuple[str, list[str]]:
    t = cau_hinh(ds, smoke)
    seeds = r64.BO_LOC[loc][1]
    ten = ten_run(det, cond, loc, smoke)
    names = [ten + (f"-s{s}" if s is not None else "") for s in (seeds or [None])]
    co = [(config.PROCESSED_DIR / "trackers" / t["split"] / nm).exists() for nm in names]
    if not any(co):
        return "chua", []
    loi = []
    for nm in names:
        c = config.RESULTS_DIR / "trackeval" / t["split"] / nm / "cong_r66.json"
        if not c.exists() or not (c.parent / "combined_metrics.csv").exists():
            loi.append(f"{nm}: thieu cong_r66/combined"); continue
        g = json.loads(c.read_text(encoding="utf-8"))
        if g.get("gt_cham_sha256") != kiem_gt_cham(t) or g.get("giao_thuc_hash") != hash_giao_thuc(ds):
            loi.append(f"{nm}: GT/giao thuc doi sau khi cham")
    return ("xong" if not loi else "do"), loi


# ------------------------------------------------------------------------------------------------ tong hop
def buoc_tong_hop(ds: str, smoke: bool) -> int:
    import pandas as pd
    import boot_stats
    t = cau_hinh(ds, smoke)
    dong, thieu = [], []
    for det in ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"):
        for cond in ("c25", "c10"):
            for loc, (_, seeds) in r64.BO_LOC.items():
                for s in (seeds or [None]):
                    nm = ten_run(det, cond, loc, smoke) + (f"-s{s}" if s is not None else "")
                    out = config.RESULTS_DIR / "trackeval" / t["split"] / nm
                    if not (out / "combined_metrics.csv").exists():
                        thieu.append(nm); continue
                    c = pd.read_csv(out / "combined_metrics.csv").iloc[0]
                    raw = json.loads((out / "raw_fields.json").read_text(encoding="utf-8"))
                    comb = boot_stats.combine(raw, list(raw))
                    lech = max(abs(float(comb[k]) - float(c[k])) for k in ("HOTA", "IDF1", "MOTA"))
                    dong.append({"detector": det, "dieu_kien": cond, "bo_loc": loc, "seed": s,
                                 **{k: float(c[k]) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")},
                                 "lech_tai_to_hop": lech})
    df = pd.DataFrame(dong)
    d = log_dir(ds, smoke); d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "combined_tung_run.csv", index=False)
    trang_thai = "XONG" if not thieu else "PARTIAL"
    if len(df) and df["lech_tai_to_hop"].max() > 1e-9:
        trang_thai = "LECH"
    ghi_json(d / "tong_hop_trang_thai.json", {"trang_thai": trang_thai, "thieu": thieu, "so_run": len(df),
                                              "thoi_diem": r64.bay_gio()})
    print(trang_thai, len(df), "run; thieu", len(thieu))
    return 0 if trang_thai == "XONG" else 10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("buoc", choices=["gt", "cache", "track", "kiem-run", "tong-hop"])
    ap.add_argument("ds", choices=list(TAP))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--het-han", default=None)
    a = ap.parse_args(argv)
    try:
        if a.buoc == "gt":
            return buoc_gt(a.ds, a.smoke)
        if a.buoc == "cache":
            return buoc_cache(a.ds, a.rest[0], a.smoke, a.het_han)
        if a.buoc == "track":
            return buoc_track(a.ds, a.rest[0], a.rest[1], a.rest[2], a.smoke, a.het_han)
        if a.buoc == "kiem-run":
            tt, loi = trang_thai_run(a.ds, a.rest[0], a.rest[1], a.rest[2], a.smoke)
            print(tt, loi)
            return {"xong": 0, "chua": 1}.get(tt, 2)
        return buoc_tong_hop(a.ds, a.smoke)
    except HetNganSach as e:
        print("[PARTIAL]", e); return 10
    except LoiHopDong as e:
        print("[DUNG]", e); return 2


if __name__ == "__main__":
    raise SystemExit("Sherbrooke: use python -B src/r66_ngoai/sher_matrix.py --help (review B1-B7).")
