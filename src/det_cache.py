"""
det_cache.py -- Cache detection cua YOLO de moi tracker so sanh tren CUNG MOT
tap detection (Giai doan 2).

===========================================================================
VI SAO CAN CACHE
===========================================================================
Giai doan 1 chay YOLO lai cho moi motion model (baseline_track.py goi
model.track(frame) tung frame). Ba model "dung chung detection" chi nho cung
weights/conf/mask; GPU khong bit-reproducible (xem src/demo/rerun_with_state_log.py)
nen phep ghep cap giua cac tracker chi la xap xi. Voi cache, moi tracker doc
DUNG cung mot mang detection -> so sanh bo loc la ghep cap chinh xac, va chay
tracking nhanh hon hang chuc lan (khong con detector).

===========================================================================
TINH TUONG THICH: LOC LUC LOAD THAY VI CHAY LAI DETECTOR
===========================================================================
Cache duoc xay MOT LAN o nguong THAP (conf = 0.05), TAT CA 80 lop COCO,
max_det = 3000. Cac dieu kien thi nghiem (conf 0.25 / 0.10, tap lop
[2,3,5,7] hay [1,2,3,5,7], max_det 300) duoc ap dung LUC LOAD. Vi sao cho
cung ket qua nhu chay detector voi dung dieu kien do (da doi chieu ma nguon
ultralytics/utils/nms.py 8.4.141):

  1. Loc conf la `conf > conf_thres` (NGAT, dong 81 va 127) truoc NMS.
  2. NMS theo lop (offset box theo class, dong 149) - box khac lop KHONG BAO GIO
     triet tieu nhau, nen loc lop truoc hay sau NMS deu cho cung tap box.
  3. NMS tham lam (torchvision.ops.nms) xet box theo diem GIAM DAN; box i bi
     loai chi vi mot box GIU LAI co diem CAO HON chong lan > iou. Tap box co
     conf > c_cao la mot TIEN TO (theo thu tu diem) cua tap box co conf > c_thap,
     nen quyet dinh giu/loai cua moi box trong tien to do la Y HET.
  4. `max_det` cat tien to `i[:max_det]` sau khi sap xep giam dan (dong 163).
     Cat tien to cua tien to = cat tien to => loc conf roi cat [:300] trung
     voi chay max_det = 300, MIEN LA cache khong tu bi cat (max_det cache = 3000
     va co danh dau `truncated` neu frame nao cham tran).
  => load(cache(0.05, all, 3000), conf > 0.25, classes, [:300]) == run(0.25, classes, 300)
     (test_det_cache.py kiem tra tinh chat nay tren tensor du doan tho).

Nhung thu PHAI GIONG giua luc xay cache va luc chay baseline (ghi trong
meta.json, doi chieu khi load): weights (sha256), imgsz, iou NMS, half,
agnostic_nms, augment, batch, device, mask vung bo qua (sha cua danh sach
vung), phien ban ultralytics/torch/cuda. `conf`, `classes`, `max_det` cua cache
chi can KHONG NGAT hon dieu kien can tai lap.

===========================================================================
DINH DANG
===========================================================================
data/interim/det_cache/<cache_id>/
    meta.json                  thong so detector + moi truong (mot lan cho ca cache)
    <video>.npz
        frame_idx   (F,)  int32    chi so frame 1-based, LIEN TUC, ke ca frame khong co detection
        n_per_frame (F,)  int32    so detection cua tung frame
        unreadable  (F,)  bool     anh khong doc duoc (khong chay detector)
        truncated   (F,)  bool     frame cham tran max_det cua cache (hiem)
        time_s      (F,)  float64  thoi diem frame [s] (danh nghia hoac tu timestamp)
        data        (sum n, 6) float32   [x1, y1, x2, y2, conf, cls] THEO THU TU NMS (giam dan diem)
        orig_shape  (2,)  int32    (H, W)
    <video>.json               thong ke nho cua video (so frame, so box, thoi gian xay)

Nguon `gt`: detection tong hop tu ground truth (them nhieu + bo sot ngau nhien)
- dung cho fixture / smoke test khong can anh, va lam "oracle detector" de tach
rieng tac dong cua bo loc. meta.backend = "gt-synthetic".

CACH DUNG
    python src/det_cache.py --dataset detrac --part test --videos MVI_39031 --cache-id yolov8n-coco_conf0.05_all
    python src/det_cache.py --dataset detrac --part train --manifest results/phase2/manifests/detrac_dev.json ...
    python src/det_cache.py --dataset bdd100k --part val --source gt --cache-id gt-synth_j0.03_d0.15 --gt-seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mot_datasets import FrameRef, get_dataset  # noqa: E402

CACHE_ROOT = config.INTERIM_DIR / "det_cache"
DET_COLS = ["x1", "y1", "x2", "y2", "conf", "cls"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Thong so detector
# ---------------------------------------------------------------------------
@dataclass
class DetectorSpec:
    """Moi thong so anh huong den tap detection. Ghi vao meta.json."""
    backend: str = "ultralytics"           # ultralytics | gt-synthetic
    weights: str = config.YOLO_DEFAULT_MODEL
    weights_sha256: str | None = None
    imgsz: int = 640
    conf: float = 0.05
    iou: float = 0.7                       # nguong NMS (mac dinh ultralytics)
    classes: list[int] | None = None       # None = tat ca lop cua model
    max_det: int = 3000
    half: bool = False
    agnostic_nms: bool = False
    augment: bool = False
    batch: int = 1
    device: str = "0"
    mask_ignored_regions: bool = True
    dataset: str = "detrac"
    part: str = "train"
    # nguon gt-synthetic
    gt_jitter: float = 0.0
    gt_dropout: float = 0.0
    gt_seed: int = 0
    gt_conf_range: tuple[float, float] = (0.3, 0.95)
    # moi truong
    versions: dict = field(default_factory=dict)
    git_commit: str | None = None
    created: str = ""

    def fill_env(self):
        if self.backend == "ultralytics":
            import torch
            import ultralytics
            self.versions = {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                             "cuda": torch.version.cuda, "cudnn": (torch.backends.cudnn.version()
                                                                   if torch.cuda.is_available() else None),
                             "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}
            w = Path(self.weights)
            if w.exists() and self.weights_sha256 is None:
                self.weights_sha256 = sha256_file(w)
        self.git_commit = git_commit()
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
        return self

    #: cac truong PHAI trung khi dung lai cache (thuoc tinh cua DETECTOR + dataset;
    #: `part` KHONG nam trong day: mot cache co the chua ca video train lan test cua cung
    #: dataset - part cua tung video ghi trong <video>.json)
    STRICT_FIELDS = ("backend", "weights_sha256", "imgsz", "iou", "half", "agnostic_nms",
                     "augment", "batch", "mask_ignored_regions", "dataset",
                     "gt_jitter", "gt_dropout", "gt_seed")


def load_meta(cache_dir: Path) -> dict:
    p = cache_dir / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"Khong co {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def check_compatible(meta: dict, conf_min: float, classes: list[int] | None, max_det: int) -> list[str]:
    """Kiem tra dieu kien can tai lap co NAM TRONG cache khong. Tra ve danh sach loi (rong = ok).

    - conf_min phai >= conf cua cache (loc len duoc, khong loc xuong duoc)
    - classes yeu cau phai la tap con cua classes cache (None = tat ca)
    - max_det yeu cau phai <= max_det cache
    """
    errs = []
    if conf_min < float(meta["conf"]) - 1e-12:
        errs.append(f"conf_min {conf_min} < conf cache {meta['conf']}: cache khong chua detection thap hon")
    cc = meta.get("classes")
    if cc is not None and classes is not None and not set(classes) <= set(cc):
        errs.append(f"classes {classes} khong nam trong classes cache {cc}")
    if cc is not None and classes is None:
        errs.append(f"yeu cau tat ca lop nhung cache chi co {cc}")
    if max_det > int(meta["max_det"]):
        errs.append(f"max_det {max_det} > max_det cache {meta['max_det']}")
    return errs


# ---------------------------------------------------------------------------
# Xay cache
# ---------------------------------------------------------------------------
def _mask_regions(frame, regions):
    from baseline_track import mask_ignored_regions   # tai dung y het Giai doan 1
    return mask_ignored_regions(frame, regions)


def _regions_hash(regions) -> str:
    return hashlib.sha1(json.dumps([list(map(float, r)) for r in regions]).encode()).hexdigest()[:12]


def build_video_yolo(model, frames: list[FrameRef], regions, spec: DetectorSpec, orig_shape):
    """Chay YOLO tung frame, tra ve (arrays cho npz). Frame khong doc duoc -> unreadable=True."""
    import cv2
    F = len(frames)
    frame_idx = np.array([f.index for f in frames], dtype=np.int32)
    time_s = np.array([f.time_s for f in frames], dtype=np.float64)
    n_per = np.zeros(F, dtype=np.int32)
    unreadable = np.zeros(F, dtype=bool)
    truncated = np.zeros(F, dtype=bool)
    chunks = []
    for i, fr in enumerate(frames):
        img = cv2.imread(str(fr.path)) if fr.path is not None else None
        if img is None:
            unreadable[i] = True
            continue
        if spec.mask_ignored_regions:
            img = _mask_regions(img, regions)
        kw = {"half": True} if spec.half else {}      # chi truyen khi bat (8.4: 'half' -> canh bao deprecated)
        r = model.predict(img, conf=spec.conf, iou=spec.iou, imgsz=spec.imgsz, classes=spec.classes,
                          max_det=spec.max_det, agnostic_nms=spec.agnostic_nms,
                          augment=spec.augment, device=spec.device, verbose=False, **kw)[0]
        det = r.boxes.data.cpu().numpy().astype(np.float32)      # (n, 6) theo thu tu NMS
        if det.shape[1] != 6:
            raise RuntimeError(f"Boxes.data co {det.shape[1]} cot, mong 6 (khong phai ket qua track?)")
        n_per[i] = len(det)
        truncated[i] = len(det) >= spec.max_det
        chunks.append(det)
    data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), dtype=np.float32)
    return dict(frame_idx=frame_idx, n_per_frame=n_per, unreadable=unreadable, truncated=truncated,
                time_s=time_s, data=data, orig_shape=np.array(orig_shape, dtype=np.int32))


def build_video_gt(gt: "pd.DataFrame", frames: list[FrameRef], spec: DetectorSpec, orig_shape,
                   category_to_coco: dict[str, int]):
    """Detection tong hop tu GT: box + nhieu Gaussian ti le kich thuoc, bo sot ngau nhien.

    conf ~ U(gt_conf_range); cls = COCO id theo category (khong co trong bang -> bo qua).
    rng co dinh theo (gt_seed, ten video) -> tai lap duoc, doc lap thu tu video.
    """
    import pandas as pd  # noqa: F401
    video = str(gt["video"].iloc[0]) if len(gt) else ""
    seed = np.random.SeedSequence([spec.gt_seed, int(hashlib.sha1(video.encode()).hexdigest(), 16) % (2**31)])
    rng = np.random.default_rng(seed)
    F = len(frames)
    frame_idx = np.array([f.index for f in frames], dtype=np.int32)
    time_s = np.array([f.time_s for f in frames], dtype=np.float64)
    n_per = np.zeros(F, dtype=np.int32)
    chunks = []
    by_frame = {k: g for k, g in gt.groupby("frame")}
    H, W = orig_shape
    for i, fr in enumerate(frames):
        g = by_frame.get(fr.index)
        if g is None or len(g) == 0:
            continue
        rows = []
        has_cat = "category" in g.columns
        for _, b in g.iterrows():
            cat = str(b["category"]) if has_cat else str(b.get("vehicle_type", "vehicle"))
            coco = category_to_coco.get(cat)
            if coco is None or rng.random() < spec.gt_dropout:
                continue
            w, h = float(b["bb_width"]), float(b["bb_height"])
            jx, jy = rng.normal(0, spec.gt_jitter * w), rng.normal(0, spec.gt_jitter * h)
            jw, jh = rng.normal(0, spec.gt_jitter * w), rng.normal(0, spec.gt_jitter * h)
            x1 = float(b["bb_left"]) + jx; y1 = float(b["bb_top"]) + jy
            x2 = x1 + max(2.0, w + jw); y2 = y1 + max(2.0, h + jh)
            x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            conf = rng.uniform(*spec.gt_conf_range)
            rows.append([x1, y1, x2, y2, conf, float(coco)])
        if rows:
            arr = np.array(rows, dtype=np.float32)
            arr = arr[np.argsort(-arr[:, 4], kind="stable")]     # giam dan theo conf nhu NMS
            arr = arr[arr[:, 4] > spec.conf][:spec.max_det]
            n_per[i] = len(arr)
            chunks.append(arr)
    data = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), dtype=np.float32)
    return dict(frame_idx=frame_idx, n_per_frame=n_per, unreadable=np.zeros(F, bool),
                truncated=np.zeros(F, bool), time_s=time_s, data=data,
                orig_shape=np.array(orig_shape, dtype=np.int32))


def save_video(cache_dir: Path, video: str, arrays: dict, extra: dict | None = None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{video}.npz", **arrays)
    info = {"video": video, "n_frames": int(len(arrays["frame_idx"])),
            "n_boxes": int(len(arrays["data"])), "n_unreadable": int(arrays["unreadable"].sum()),
            "n_truncated": int(arrays["truncated"].sum()), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        info.update(extra)
    (cache_dir / f"{video}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def write_meta(cache_dir: Path, spec: DetectorSpec):
    """Ghi meta.json; neu da co thi cac truong STRICT phai trung, neu khong -> loi."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / "meta.json"
    new = asdict(spec)
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), new.get(k)) for k in DetectorSpec.STRICT_FIELDS if old.get(k) != new.get(k)}
        if diff:
            raise RuntimeError(f"Cache {cache_dir} da ton tai voi thong so khac: {diff}. "
                               "Dung --cache-id khac.")
        return old
    p.write_text(json.dumps(new, indent=2), encoding="utf-8")
    return new


# ---------------------------------------------------------------------------
# Doc cache
# ---------------------------------------------------------------------------
class VideoDetections:
    """Detection cua mot video, da loc theo dieu kien thi nghiem.

    Lap qua `iter_frames()` -> (frame_idx, time_s, unreadable, arr (n,6) float32).
    """

    def __init__(self, npz_path: Path, conf_min: float, classes: list[int] | None,
                 max_det: int = 300, class_map: dict[int, int] | None = None,
                 allow_truncated: bool = False):
        z = np.load(npz_path)
        self.frame_idx = z["frame_idx"]
        self.n_per_frame = z["n_per_frame"]
        self.unreadable = z["unreadable"]
        self.truncated = z["truncated"]
        if self.truncated.any() and not allow_truncated:
            # Frame cham tran max_det cua cache: tinh chat tien to KHONG con dung
            # (test_det_cache.py chung minh bang so) -> tu choi thay vi am tham sai.
            bad = self.frame_idx[self.truncated][:10].tolist()
            raise ValueError(f"{npz_path.name}: {int(self.truncated.sum())} frame cham tran max_det cua cache "
                             f"(vd frame {bad}); xay lai cache voi --max-det lon hon hoac allow_truncated=True")
        self.time_s = z["time_s"]
        self.orig_shape = tuple(int(v) for v in z["orig_shape"])
        data = z["data"].astype(np.float32, copy=False)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_per_frame)])
        self.conf_min, self.classes, self.max_det = float(conf_min), classes, int(max_det)
        self._data = data
        self._cls_set = None if classes is None else np.array(sorted(set(int(c) for c in classes)))
        self.class_map = class_map    # COCO id -> id khac (vd gop ve 1 lop); None = giu nguyen

    def frame(self, i: int) -> np.ndarray:
        arr = self._data[self.offsets[i]:self.offsets[i + 1]]
        if len(arr):
            keep = arr[:, 4] > self.conf_min                       # NGAT, giong nms.py
            if self._cls_set is not None:
                keep &= np.isin(arr[:, 5].astype(np.int64), self._cls_set)
            arr = arr[keep][:self.max_det]                          # cat tien to
            if self.class_map is not None and len(arr):
                arr = arr.copy()
                arr[:, 5] = [self.class_map.get(int(c), c) for c in arr[:, 5]]
        return np.ascontiguousarray(arr, dtype=np.float32)

    def iter_frames(self):
        for i in range(len(self.frame_idx)):
            yield int(self.frame_idx[i]), float(self.time_s[i]), bool(self.unreadable[i]), self.frame(i)

    def __len__(self):
        return len(self.frame_idx)


def to_boxes(arr: np.ndarray, orig_shape: tuple[int, int]):
    """Boi lai `Boxes` y het thu tracker nhan trong callback cua ultralytics
    (on_predict_postprocess_end: `result.boxes.cpu().numpy()`), dtype float32."""
    from ultralytics.engine.results import Boxes
    a = np.asarray(arr, dtype=np.float32).reshape(-1, 6)
    return Boxes(a, tuple(orig_shape))


def open_cache(cache_id: str, video: str, conf_min: float, classes: list[int] | None,
               max_det: int = 300, class_map: dict[int, int] | None = None,
               strict: bool = True, allow_truncated: bool = False) -> VideoDetections:
    cache_dir = CACHE_ROOT / cache_id
    meta = load_meta(cache_dir)
    errs = check_compatible(meta, conf_min, classes, max_det)
    if errs and strict:
        raise ValueError(f"Cache {cache_id} khong tai lap duoc dieu kien yeu cau:\n  - " + "\n  - ".join(errs))
    p = cache_dir / f"{video}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Chua co cache cho video {video}: {p}")
    return VideoDetections(p, conf_min, classes, max_det, class_map, allow_truncated)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Xay detection cache (YOLO hoac GT tong hop)")
    ap.add_argument("--dataset", default="detrac", help="detrac | bdd100k")
    ap.add_argument("--part", default="train")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--manifest", default=None, help="JSON co khoa 'videos' (danh sach video)")
    ap.add_argument("--all", action="store_true", help="tat ca video cua part")
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--source", choices=["yolo", "gt"], default="yolo")
    ap.add_argument("--model", default=config.YOLO_DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", nargs="*", type=int, default=None, help="mac dinh: TAT CA lop")
    ap.add_argument("--max-det", type=int, default=3000)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-mask", action="store_true", help="khong to xam vung bo qua (KHONG dung cho baseline)")
    ap.add_argument("--gt-jitter", type=float, default=0.03)
    ap.add_argument("--gt-dropout", type=float, default=0.15)
    ap.add_argument("--gt-seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--data-root", default=None, help="thu muc goc dataset (bdd100k)")
    args = ap.parse_args()

    ds_kwargs = {"root": args.data_root} if (args.dataset.startswith("bdd") and args.data_root) else {}
    ds = get_dataset(args.dataset, **ds_kwargs)
    if args.videos:
        videos = args.videos
    elif args.manifest:
        videos = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["videos"]
    elif args.all:
        videos = ds.list_videos(args.part)
    else:
        print("[ERROR] Can --videos, --manifest hoac --all"); return 1

    spec = DetectorSpec(backend="ultralytics" if args.source == "yolo" else "gt-synthetic",
                        weights=str(args.model), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                        classes=args.classes, max_det=args.max_det, half=args.half, device=args.device,
                        mask_ignored_regions=not args.no_mask, dataset=ds.name, part=args.part,
                        gt_jitter=args.gt_jitter if args.source == "gt" else 0.0,
                        gt_dropout=args.gt_dropout if args.source == "gt" else 0.0,
                        gt_seed=args.gt_seed if args.source == "gt" else 0).fill_env()
    cache_dir = CACHE_ROOT / args.cache_id
    write_meta(cache_dir, spec)
    print(f"[det_cache] {args.cache_id}: backend={spec.backend} conf={spec.conf} iou={spec.iou} "
          f"imgsz={spec.imgsz} classes={spec.classes} max_det={spec.max_det} -> {cache_dir}")

    model = None
    if args.source == "yolo":
        from ultralytics import YOLO
        model = YOLO(spec.weights)
        gt_all = None
    else:
        gt_all = ds.gt_dataframe(args.part)
        cat2coco = getattr(ds, "category_to_coco", {"vehicle": 2})

    for v in videos:
        out = cache_dir / f"{v}.npz"
        if args.skip_existing and out.exists():
            print(f"  {v:<20} [SKIP] da co"); continue
        t0 = time.time()
        frames = ds.frames(v, args.part)
        regions = ds.ignored_regions(v, args.part) if spec.mask_ignored_regions else []
        if args.source == "yolo":
            arrays = build_video_yolo(model, frames, regions, spec, ds.orig_shape)
        else:
            gt_v = gt_all[gt_all["video"] == v]
            arrays = build_video_gt(gt_v, frames, spec, ds.orig_shape, cat2coco)
        info = save_video(cache_dir, v, arrays, {"part": args.part, "regions_hash": _regions_hash(regions),
                                                 "n_regions": len(regions), "seconds": round(time.time() - t0, 1)})
        print(f"  {v:<20} {info['n_frames']:>5} frame  {info['n_boxes']:>7} box  "
              f"unreadable={info['n_unreadable']} truncated={info['n_truncated']}  ({info['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
