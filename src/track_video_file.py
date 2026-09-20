"""
track_video_file.py -- CHAY MOT VIDEO qua detector + tracker (moc he thong PC).

Dau vao : file video (.mp4/.avi/...), thu muc anh (chuoi .jpg), hoac sequence dataset
          (--dataset detrac --part test --video MVI_39031).
Dau ra  : <out>/data/<ten>.txt         tracking MOTChallenge  (frame,id,x,y,w,h,conf)
          <out>/data_cls/<ten>.csv     lop cua tung track theo frame
          <out>/timing/<ten>.csv       thoi gian TRACKER theo frame (predict / assoc / update)
          <out>/pipeline/<ten>.csv     thoi gian TOAN PIPELINE theo frame (decode/detect/track/render/write)
          <out>/run_meta.json          moi thu can de TAI HIEN (nguon, checkpoint sha256, DetectorSpec,
                                       cau hinh tracker HIEU LUC, override, fps/dt, seed, backend, commit)
          <out>/<ten>_tracked.mp4      video co bbox + ID + ten lop (neu --save-video)
          det cache <cache_id>/        detection tho de REPLAY bang run_tracking.py (neu --save-cache)

THONG NHAT VOI DUONG CHAY CACHE -- ba cho, deu dung CHUNG ma, khong nhan ban:
  1. Detector: `det_cache.detect_frame()` voi CUNG `DetectorSpec` cua cache (conf 0,05, MOI lop,
     max_det 3000, imgsz 640). ByteTrack can detection confidence THAP cho vong ghep thu hai
     (`track_low_thresh`), nen detector KHONG duoc chay o nguong 0,25 -- nguong do ap o buoc loc.
  2. Loc: `det_cache.filter_dets()` (conf_min > ; classes ; [:max_det] ; class_map) -- y het
     `VideoDetections.frame()`.
  3. Vong lap tracker: `run_tracking.track_video()` -- cung `make_tracker` (YAML override di vao
     tracker that), cung `n_steps` theo quy uoc thoi gian, cung cap nhat vong doi ke ca frame rong.
     Lop `LiveVideoDetections` chi la ADAPTER co cung interface voi `VideoDetections`.

Vi vay: cung anh + cung cau hinh => ket qua tracking PHAI y het replay tu cache. Kiem bang
`src/nano_parity.py` (xem `src/test_track_video_file.py`).

GT KHONG tham gia: khong doc gt.txt, khong doc annotation. GT chi dung o buoc danh gia rieng.

    python src/track_video_file.py --source video.mp4 --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \\
        --classes 1 2 3 5 7 --conf-min 0.25 --device 0 --out results/track_video/xe --save-video --show
    python src/track_video_file.py --dataset detrac --part test --video MVI_39031 --tracker-yaml bytetrack.yaml
    python src/track_video_file.py --source anh/  --fps 25 ...           # thu muc .jpg

Phim khi --show: q thoat (ket qua da xu ly VAN duoc ghi day du), SPACE tam dung.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import ul_backend  # noqa: E402
from time_units import track_buffer_frames  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- nguon frame
class NguonFrame:
    """Lap qua frame cua mot nguon: (frame_idx 1-based, time_s, img|None). Dong tai nguyen khi xong."""

    def __init__(self, source: str | None, dataset: str | None, part: str, video: str | None,
                 fps: float | None, max_frames: int | None):
        import cv2
        self.cv2, self.max_frames = cv2, max_frames
        self.cap, self.paths, self.regions = None, None, []
        if dataset:
            from mot_datasets import get_dataset
            ds = get_dataset(dataset)
            frames = ds.frames(video, part)
            self.paths = [(f.index, f.time_s, f.path) for f in frames]
            self.regions = ds.ignored_regions(video, part)
            self.fps = float(fps or ds.fps)
            self.orig_shape = tuple(ds.orig_shape)
            self.ten, self.loai, self.duong_dan = video, "dataset", f"{dataset}/{part}/{video}"
            self.n_frames_khai_bao = len(frames)
            return
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"nguon khong ton tai: {p}")
        if p.is_dir():
            paths = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
            if not paths:
                raise FileNotFoundError(f"thu muc {p} khong co anh {sorted(IMG_EXT)}")
            if fps is None:
                raise ValueError("thu muc anh khong co fps: can --fps")
            self.fps = float(fps)
            self.paths = [(i, (i - 1) / self.fps, q) for i, q in enumerate(paths, start=1)]
            img0 = cv2.imread(str(paths[0]))
            if img0 is None:
                raise RuntimeError(f"khong doc duoc anh dau tien {paths[0]}")
            self.orig_shape = tuple(img0.shape[:2])
            self.ten, self.loai, self.duong_dan = p.name, "image_dir", str(p)
            self.n_frames_khai_bao = len(paths)
            return
        if p.suffix.lower() not in VIDEO_EXT:
            raise ValueError(f"khong nhan dang duoc nguon {p} (video {sorted(VIDEO_EXT)} / thu muc anh)")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 khong mo duoc video {p} (file hong hoac codec khong ho tro)")
        self.cap = cap
        fps_meta = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps is None and fps_meta <= 0:
            cap.release()
            raise ValueError(f"video khong co fps trong metadata: can --fps")
        self.fps = float(fps or fps_meta)
        self.fps_metadata = float(fps_meta)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_shape = (h, w)
        self.ten, self.loai, self.duong_dan = p.stem, "video_file", str(p)
        self.n_frames_khai_bao = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.sha256 = det_cache.sha256_file(p)

    def __iter__(self):
        n = 0
        if self.paths is not None:
            for idx, t_s, q in self.paths:
                if self.max_frames and n >= self.max_frames:
                    return
                n += 1
                yield idx, float(t_s), (self.cv2.imread(str(q)) if q is not None else None)
            return
        idx = 0
        while True:
            if self.max_frames and n >= self.max_frames:
                return
            ok, img = self.cap.read()
            if not ok:
                return
            idx += 1
            n += 1
            # Thoi diem theo CHI SO frame va fps danh nghia (nhu FrameRef cua dataset). Khong dung
            # CAP_PROP_POS_MSEC: nhieu codec tra ve 0 hoac lech, se lam n_steps sai.
            yield idx, (idx - 1) / self.fps, img

    def dong(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# --------------------------------------------------------------------------- adapter cho track_video
class LiveVideoDetections:
    """Detector chay TRUC TIEP tren frame, interface y het `det_cache.VideoDetections`
    (`iter_frames()` -> (frame_idx, t_s, unreadable, arr) va `orig_shape`).

    Sau khi lap xong, `raw` chua detection THO (chua loc) de ghi det cache -> replay duoc.
    `pipeline` chua thoi gian decode/detect theo frame. `last_img` = anh cua frame vua yield.
    """

    def __init__(self, nguon: NguonFrame, model, spec: det_cache.DetectorSpec, conf_min: float,
                 classes: list[int] | None, max_det: int, class_map: dict[int, int] | None):
        self.nguon, self.model, self.spec = nguon, model, spec
        self.conf_min, self.max_det, self.class_map = float(conf_min), int(max_det), class_map
        self._cls_set = None if classes is None else np.array(sorted(set(int(c) for c in classes)))
        self.orig_shape = tuple(int(v) for v in nguon.orig_shape)
        self.last_img = None
        self.raw = {"frame_idx": [], "time_s": [], "n_per_frame": [], "unreadable": [], "truncated": [],
                    "data": []}
        self.pipeline: list[dict] = []
        self.n_frames = 0

    def iter_frames(self):
        for idx, t_s, img in self.nguon:
            self.n_frames += 1
            t0 = time.perf_counter()
            if img is None:
                self.raw["frame_idx"].append(idx); self.raw["time_s"].append(t_s)
                self.raw["n_per_frame"].append(0); self.raw["unreadable"].append(True)
                self.raw["truncated"].append(False)
                self.pipeline.append({"frame": idx, "unreadable": True, "decode_ms": 0.0,
                                      "detect_ms": 0.0, "n_det_raw": 0, "n_det_loc": 0})
                self.last_img = None
                yield idx, t_s, True, np.zeros((0, 6), dtype=np.float32)
                continue
            if self.spec.mask_ignored_regions and self.nguon.regions:
                img = det_cache._mask_regions(img, self.nguon.regions)
            t1 = time.perf_counter()
            det = det_cache.detect_frame(self.model, img, self.spec)      # CUNG ma voi build cache
            t2 = time.perf_counter()
            self.raw["frame_idx"].append(idx); self.raw["time_s"].append(t_s)
            self.raw["n_per_frame"].append(len(det)); self.raw["unreadable"].append(False)
            self.raw["truncated"].append(len(det) >= self.spec.max_det); self.raw["data"].append(det)
            arr = det_cache.filter_dets(det, self.conf_min, self._cls_set, self.max_det, self.class_map)
            self.pipeline.append({"frame": idx, "unreadable": False, "decode_ms": (t1 - t0) * 1e3,
                                  "detect_ms": (t2 - t1) * 1e3, "n_det_raw": int(len(det)),
                                  "n_det_loc": int(len(arr))})
            self.last_img = img
            yield idx, t_s, False, arr

    def __len__(self):
        return self.n_frames

    def cache_arrays(self) -> dict:
        r = self.raw
        data = np.concatenate(r["data"], axis=0) if r["data"] else np.zeros((0, 6), dtype=np.float32)
        return dict(frame_idx=np.array(r["frame_idx"], dtype=np.int32),
                    n_per_frame=np.array(r["n_per_frame"], dtype=np.int32),
                    unreadable=np.array(r["unreadable"], dtype=bool),
                    truncated=np.array(r["truncated"], dtype=bool),
                    time_s=np.array(r["time_s"], dtype=np.float64), data=data.astype(np.float32),
                    orig_shape=np.array(self.orig_shape, dtype=np.int32))


# --------------------------------------------------------------------------- ve
def ve_frame(img, out: np.ndarray, ten_lop: dict) -> None:
    """Ve bbox + ID + ten lop len anh (tai cho)."""
    import cv2
    for row in out:
        x1, y1, x2, y2, tid = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
        cls = int(row[6]) if len(row) > 6 else -1
        mau = ((tid * 37) % 256, (tid * 91) % 256, (tid * 53) % 256)
        cv2.rectangle(img, (x1, y1), (x2, y2), mau, 2)
        nhan = f"#{tid} {ten_lop.get(cls, cls)}"
        (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), mau, -1)
        cv2.putText(img, nhan, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- CLI
def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="Chay mot video qua detector + tracker")
    g = ap.add_argument_group("nguon")
    g.add_argument("--source", default=None, help="file video hoac thu muc anh")
    g.add_argument("--dataset", default=None, help="hoac: sequence dataset (detrac | bdd100k_coco ...)")
    g.add_argument("--part", default="test")
    g.add_argument("--video", default=None, help="ten sequence khi dung --dataset")
    g.add_argument("--fps", type=float, default=None, help="bat buoc voi thu muc anh; ghi de metadata video")
    g.add_argument("--max-frames", type=int, default=None)
    d = ap.add_argument_group("detector (cung DetectorSpec voi det cache)")
    d.add_argument("--model", default=config.YOLO_DEFAULT_MODEL, help="checkpoint YOLO (.pt)")
    d.add_argument("--device", default="0", help="0 | cpu")
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--half", action="store_true")
    d.add_argument("--det-conf", type=float, default=0.05,
                   help="nguong detector THO (mac dinh 0,05 = cache). GIU THAP cho vong ghep 2 cua ByteTrack")
    d.add_argument("--det-max-det", type=int, default=3000)
    f = ap.add_argument_group("loc detection truoc khi vao tracker (y het run_tracking.py)")
    f.add_argument("--conf-min", type=float, default=0.25)
    f.add_argument("--classes", nargs="*", type=int, default=[1, 2, 3, 5, 7],
                   help="COCO id: 1 bicycle 2 car 3 motorcycle 5 bus 7 truck")
    f.add_argument("--all-classes", action="store_true")
    f.add_argument("--class-map", default=None, help="vd '7:2,5:2' gop lop truoc khi vao tracker")
    f.add_argument("--max-det", type=int, default=300)
    t = ap.add_argument_group("tracker")
    t.add_argument("--tracker-yaml", default="bytetrack.yaml",
                   help="bytetrack.yaml (KF-CV) | configs/bytetrack_ekf_ctrv.yaml | ..._ukf_ctrv.yaml | ...pf...")
    t.add_argument("--set", nargs="*", default=None, help="ghi de khoa yaml: key=value")
    t.add_argument("--track-buffer-sec", type=float, default=1.2)
    t.add_argument("--elapsed-mode", choices=["skip", "steps"], default="steps")
    t.add_argument("--seed", type=int, default=None, help="PF")
    o = ap.add_argument_group("dau ra")
    o.add_argument("--out", default=None, help="mac dinh results/track_video/<ten>")
    o.add_argument("--save-video", action="store_true")
    o.add_argument("--show", action="store_true")
    o.add_argument("--save-cache", default=None, metavar="CACHE_ID",
                   help="ghi detection tho vao det cache de replay bang run_tracking.py")
    o.add_argument("--warmup-frames", type=int, default=25,
                   help="so frame dau BO khoi phan TONG KET thoi gian (CSV theo frame van giu du)")
    a = ap.parse_args()

    if not a.source and not a.dataset:
        ap.error("can --source hoac --dataset/--video")
    if a.dataset and not a.video:
        ap.error("--dataset can --video")

    # ---- nguon ----
    nguon = NguonFrame(a.source, a.dataset, a.part, a.video, a.fps, a.max_frames)
    ten = nguon.ten
    out = Path(a.out) if a.out else (config.RESULTS_DIR / "track_video" / ten)
    out.mkdir(parents=True, exist_ok=True)

    # ---- detector: CUNG DetectorSpec voi cache ----
    w = Path(a.model)
    if not w.is_file():
        nguon.dong()
        print(f"[LOI] checkpoint khong ton tai: {w}")
        return 2
    spec = det_cache.DetectorSpec(weights=str(w), imgsz=a.imgsz, conf=a.det_conf, classes=None,
                                  max_det=a.det_max_det, half=a.half, device=a.device,
                                  mask_ignored_regions=True, dataset=(a.dataset or "video_file"),
                                  part=(a.part if a.dataset else nguon.loai))
    try:
        spec.fill_env()
        from ultralytics import YOLO
        model = YOLO(spec.weights)
    except Exception as e:  # noqa: BLE001
        nguon.dong()
        print(f"[LOI] khong nap duoc detector {w}: {type(e).__name__}: {e}")
        return 2
    ten_lop = dict(getattr(model, "names", {}) or {})

    # ---- tracker: cung cach nap cau hinh voi runner (YAML override di vao tracker THAT) ----
    overrides = run_tracking.parse_set(a.set)
    overrides["track_buffer"] = track_buffer_frames(a.track_buffer_sec, nguon.fps)
    cfg = run_tracking.load_tracker_cfg(a.tracker_yaml, overrides)
    classes = None if a.all_classes else a.classes
    class_map = run_tracking.parse_class_map(a.class_map)
    seed = a.seed
    if cfg.tracker_type in run_tracking.STOCHASTIC_TRACKERS and seed is None:
        seed = int(getattr(cfg, "pf_seed", 0))
    dt_nominal = 1.0 / nguon.fps
    yaml_path = Path(run_tracking.resolve_tracker_yaml(a.tracker_yaml))
    yaml_sha = ul_backend.sha256_lf(yaml_path)

    print(f"[track_video] nguon={nguon.loai}:{nguon.duong_dan}  {nguon.orig_shape[1]}x{nguon.orig_shape[0]} "
          f"@ {nguon.fps:g} fps  ({nguon.n_frames_khai_bao} frame khai bao)")
    print(f"[track_video] detector={w.name} conf_tho={spec.conf} max_det_tho={spec.max_det} device={a.device}")
    print(f"[track_video] loc: conf>{a.conf_min} classes={classes} max_det={a.max_det} class_map={class_map}")
    print(f"[track_video] tracker={cfg.tracker_type} yaml={a.tracker_yaml} track_buffer={cfg.track_buffer} "
          f"buoc ({a.track_buffer_sec}s) elapsed={a.elapsed_mode} seed={seed} backend={ul_backend.backend_name()}")

    # ---- hien thi / ghi video ----
    import cv2
    writer, cua_so = None, None
    video_out = out / f"{ten}_tracked.mp4"
    if a.save_video:
        h, wd = nguon.orig_shape
        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), nguon.fps, (wd, h))
        if not writer.isOpened():
            nguon.dong()
            print(f"[LOI] khong mo duoc VideoWriter {video_out}")
            return 2
    if a.show:
        cua_so = f"track_video: {ten}"
        cv2.namedWindow(cua_so, cv2.WINDOW_NORMAL)

    vd = LiveVideoDetections(nguon, model, spec, a.conf_min, classes, a.max_det, class_map)
    trang_thai = {"dung_som": False, "render_ms": [], "write_ms": [], "n_ve": 0}

    def on_frame(frame_idx: int, out_arr: np.ndarray):
        if vd.last_img is None or (writer is None and cua_so is None):
            return True
        t0 = time.perf_counter()
        img = vd.last_img.copy()
        ve_frame(img, out_arr, ten_lop)
        cv2.putText(img, f"f{frame_idx} {cfg.tracker_type} n={len(out_arr)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        t1 = time.perf_counter()
        if writer is not None:
            writer.write(img)
        t2 = time.perf_counter()
        trang_thai["render_ms"].append((t1 - t0) * 1e3)
        trang_thai["write_ms"].append((t2 - t1) * 1e3)
        trang_thai["n_ve"] += 1
        if cua_so is not None:
            cv2.imshow(cua_so, img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 == ord(" "):
                        break
                    if k2 == ord("q"):
                        trang_thai["dung_som"] = True
                        return False
            if k == ord("q"):
                trang_thai["dung_som"] = True
                return False
        return True

    # ---- chay: CUNG vong lap voi runner ----
    t_bat_dau = time.perf_counter()
    try:
        df, df_cls, df_t = run_tracking.track_video(vd, cfg, dt_nominal, a.elapsed_mode, seed, ten,
                                                     on_frame=on_frame)
    finally:
        # GIAI PHONG tai nguyen du thanh cong, dung som hay loi
        nguon.dong()
        if writer is not None:
            writer.release()
        if cua_so is not None:
            cv2.destroyAllWindows()
    wall = time.perf_counter() - t_bat_dau

    # ---- ghi ket qua ----
    (out / "data").mkdir(exist_ok=True); (out / "data_cls").mkdir(exist_ok=True)
    (out / "timing").mkdir(exist_ok=True); (out / "pipeline").mkdir(exist_ok=True)
    df.to_csv(out / "data" / f"{ten}.txt", header=False, index=False)
    df_cls.to_csv(out / "data_cls" / f"{ten}.csv", index=False)
    df_t.to_csv(out / "timing" / f"{ten}.csv", index=False)
    import pandas as pd
    df_p = pd.DataFrame(vd.pipeline)
    if len(df_p):
        df_p = df_p.merge(df_t[["frame", "t_update_ms", "t_predict_ms"]], on="frame", how="left")
    df_p.to_csv(out / "pipeline" / f"{ten}.csv", index=False)

    cache_info = None
    if a.save_cache:
        cdir = det_cache.CACHE_ROOT / a.save_cache
        det_cache.write_meta(cdir, spec, mode="append")
        cache_info = det_cache.save_video(cdir, ten, vd.cache_arrays(),
                                          extra={"part": spec.part, "n_regions": len(nguon.regions),
                                                 "source": nguon.duong_dan, "fps": nguon.fps},
                                          spec=spec)

    # ---- thong ke thoi gian: tracker RIENG, pipeline RIENG, ve/ghi RIENG ----
    def tb(x):
        return float(np.mean(x)) if len(x) else float("nan")
    # TONG KET bo `--warmup-frames` frame dau (CUDA/cuDNN khoi dong, cache lanh). CSV theo frame giu DU.
    wu = a.warmup_frames
    ok = df_p[(~df_p.unreadable) & (df_p.frame > wu)] if len(df_p) else df_p
    t_ok = df_t[df_t.frame > wu] if len(df_t) else df_t
    ve_ok = trang_thai["render_ms"][wu:]
    ghi_ok = trang_thai["write_ms"][wu:]
    thoi_gian = {
        "n_frame_xu_ly": int(len(df_p)), "n_frame_unreadable": int(df_p.unreadable.sum()) if len(df_p) else 0,
        "warmup_frames_bo": wu, "n_frame_tong_ket": int(len(ok)),
        "tracker_update_ms": tb(t_ok.t_update_ms), "tracker_predict_ms": tb(t_ok.t_predict_ms),
        "decode_ms": tb(ok.decode_ms) if len(ok) else float("nan"),
        "detect_ms": tb(ok.detect_ms) if len(ok) else float("nan"),
        "pipeline_khong_ve_ms": tb(ok.decode_ms + ok.detect_ms + ok.t_update_ms) if len(ok) else float("nan"),
        "render_ms": tb(ve_ok), "write_ms": tb(ghi_ok),
        "pipeline_co_ve_ghi_ms": (tb(ok.decode_ms + ok.detect_ms + ok.t_update_ms) + tb(ve_ok) + tb(ghi_ok)
                                  if len(ok) and ve_ok else None),
        "wall_s": wall, "fps_wall": (len(df_p) / wall if wall > 0 else float("nan")),
        "ghi_chu": ("Cac so *_ms la TRUNG BINH sau khi bo warmup_frames_bo frame dau. "
                    "pipeline_khong_ve_ms = decode + detect + tracker.update, KHONG gom ve/ghi/hien thi. "
                    "pipeline_co_ve_ghi_ms = cong them render + write (chi khi --save-video/--show). "
                    "wall_s gom TAT CA ke ca warm-up. Do MOT luot, khong lap co kiem soat -> so mo ta, "
                    "khong dung de xep hang."),
    }

    code_fp = __import__("run_fingerprint").code_fingerprint()
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": det_cache.git_commit(),
        "lenh": " ".join(sys.argv),
        "nguon": {"loai": nguon.loai, "duong_dan": nguon.duong_dan, "ten": ten,
                  "sha256": getattr(nguon, "sha256", None), "fps": nguon.fps,
                  "fps_metadata": getattr(nguon, "fps_metadata", None),
                  "n_frames_khai_bao": nguon.n_frames_khai_bao, "orig_shape_hw": list(nguon.orig_shape),
                  "n_regions_bo_qua": len(nguon.regions), "max_frames": a.max_frames},
        "detector_spec": asdict(spec),
        "loc": {"conf_min": a.conf_min, "classes": classes, "class_map": class_map, "max_det": a.max_det},
        "tracker": {"yaml": a.tracker_yaml, "yaml_resolved": str(yaml_path), "yaml_sha256": yaml_sha,
                    "overrides": overrides, "cfg_hieu_luc": run_tracking.cfg_to_dict(cfg),
                    "tracker_type": cfg.tracker_type, "track_buffer_frames": int(cfg.track_buffer),
                    "track_buffer_sec": a.track_buffer_sec, "elapsed_mode": a.elapsed_mode,
                    "dt_nominal_s": dt_nominal, "seed": seed, "backend": code_fp["backend"]},
        "code": code_fp, "ten_lop": {int(k): v for k, v in ten_lop.items()},
        "ket_qua": {"n_frame_xu_ly": int(len(df_p)), "n_hang_tracking": int(len(df)),
                    "n_track": int(df["id"].nunique()) if len(df) else 0,
                    "dung_som": trang_thai["dung_som"],
                    "n_frame_ve": trang_thai["n_ve"],
                    "video_tracked": str(video_out) if a.save_video else None,
                    "det_cache": ({"cache_id": a.save_cache, "dir": str(det_cache.CACHE_ROOT / a.save_cache),
                                   **cache_info} if cache_info else None)},
        "thoi_gian": thoi_gian,
        "replay": (None if not a.save_cache else
                   f"python src/run_tracking.py --cache-id {a.save_cache} --split-name <split> "
                   f"--tracker-yaml {a.tracker_yaml} --tracker-name <ten> --videos {ten} "
                   f"--conf-min {a.conf_min} {'--all-classes' if classes is None else '--classes ' + ' '.join(map(str, classes))} "
                   f"--max-det {a.max_det} --fps {nguon.fps:g} --track-buffer-sec {a.track_buffer_sec} "
                   f"--elapsed-mode {a.elapsed_mode}"
                   + (f" --class-map {a.class_map}" if a.class_map else "")
                   + (f" --seeds {seed}" if seed is not None else "")
                   + (" --set " + " ".join(a.set) if a.set else "")),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False),
                                       encoding="utf-8")

    print(f"[track_video] {len(df_p)} frame ({thoi_gian['n_frame_unreadable']} unreadable), "
          f"{len(df)} hang tracking, {meta['ket_qua']['n_track']} track"
          + (" -- DUNG SOM theo yeu cau, ket qua da xu ly van ghi day du" if trang_thai["dung_som"] else ""))
    print(f"[track_video] tracker {thoi_gian['tracker_update_ms']:.2f} ms/frame | detector "
          f"{thoi_gian['detect_ms']:.1f} | decode {thoi_gian['decode_ms']:.1f} | "
          f"pipeline (khong ve) {thoi_gian['pipeline_khong_ve_ms']:.1f} ms/frame"
          + (f" | ve {thoi_gian['render_ms']:.1f} + ghi {thoi_gian['write_ms']:.1f}" if trang_thai["n_ve"] else "")
          + f" | wall {wall:.1f}s = {thoi_gian['fps_wall']:.1f} fps")
    print(f"[track_video] -> {out}")
    if cache_info:
        print(f"[track_video] det cache -> {det_cache.CACHE_ROOT / a.save_cache}  (replay: xem run_meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
