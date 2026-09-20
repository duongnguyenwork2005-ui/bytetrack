"""
track_video_file.py -- CHAY MOT VIDEO qua detector + tracker (moc he thong PC).

Dau vao : file video (.mp4/.avi/...), thu muc anh (chuoi .jpg), hoac sequence dataset
          (--dataset detrac --part test --video MVI_39031).
Dau ra  : <out>/data/<ten>.txt         tracking MOTChallenge  (frame,id,x,y,w,h,conf)
          <out>/data_cls/<ten>.csv     lop cua tung track theo frame
          <out>/timing/<ten>.csv       thoi gian TRACKER theo frame (predict / assoc / update)
          <out>/pipeline/<ten>.csv     thoi gian TOAN PIPELINE theo frame, TUNG STAGE (xem PHAM VI DO)
          <out>/frames_manifest.json   thu tu file -> frame -> time_s DA DUNG (thu muc anh / dataset)
          <out>/run_meta.json          moi thu can de TAI HIEN (nguon, checkpoint sha256, DetectorSpec,
                                       cau hinh tracker HIEU LUC, override, fps/dt, seed, backend, commit,
                                       sha256 cua CHINH file nay, trang thai working tree)
          <out>/<ten>_tracked.mp4      video co bbox + ID + ten lop (neu --save-video)
          det cache <cache_id>/        detection tho de REPLAY bang run_tracking.py (neu --save-cache)

THONG NHAT VOI DUONG CHAY CACHE -- ba cho, deu dung CHUNG ma, khong nhan ban:
  1. Detector: `det_cache.detect_frame()` voi CUNG `DetectorSpec` cua cache (mac dinh conf 0,05,
     MOI lop, max_det 3000, imgsz 640).
  2. Loc: `det_cache.filter_dets()` (conf_min > ; classes ; [:max_det] ; class_map) -- y het
     `VideoDetections.frame()`.
  3. Vong lap tracker: `run_tracking.track_video()` -- cung `make_tracker` (YAML override di vao
     tracker that), cung `n_steps` theo quy uoc thoi gian, cung cap nhat vong doi ke ca frame rong.
     Lop `LiveVideoDetections` chi la ADAPTER co cung interface voi `VideoDetections`.
  + (review 8abb368, PC-F2) CUNG CHINH SACH TUONG THICH producer/consumer voi duong doc cache:
     `det_cache.check_compatible()` chay TRUOC khi xu ly (conf_min >= det_conf, max_det <= det_max_det,
     classes la tap con); frame CHAM TRAN det_max_det bi TU CHOI mac dinh y het `VideoDetections`
     (`--allow-truncated` de chap nhan, va co ghi vao metadata + lenh replay).

NGUONG HIEU LUC vao tracker = `--conf-min` (mac dinh 0,25 = dieu kien c25 cua thi nghiem). Detector
chay o `--det-conf` 0,05 chi de cache DUNG LAI DUOC cho ca c25 lan c10; no KHONG tu bao dam vong ghep
confidence thap cua ByteTrack co detection: voi `--conf-min 0.25`, moi detection < 0,25 bi loc TRUOC
khi vao tracker. Muon vong ghep thu hai thay detection 0,1-0,25 thi dat `--conf-min 0.1` (c10).

GT: che do file/thu muc KHONG doc GT hay annotation. Che do --dataset co the doc VUNG BO QUA
(`gt/ignored_regions.txt` cua UA-DETRAC) de to xam anh theo quy uoc benchmark -- KHONG dua ID/box GT
vao association.

PHAM VI DO (review 8abb368, PC-F1) -- theo TUNG frame, cot trong pipeline/<ten>.csv:
    decode_ms      quanh dung `cv2.imread()` / `cap.read()` (trong NguonFrame)
    mask_ms        to xam vung bo qua (chi che do dataset co vung)
    detect_ms      `det_cache.detect_frame()` (preprocess + suy luan + NMS)
    t_update_ms    `tracker.update()` (tu timing/<ten>.csv; t_predict_ms la phan `multi_predict`)
    render_ms / write_ms / show_ms   ve bbox / `VideoWriter.write` / `imshow+waitKey` (0 neu tat)
    frame_total_ms tu TRUOC khi doc frame den SAU khi tracker + ve/ghi/hien thi xong (do trong adapter)
    overhead_ms    = frame_total - (decode+mask+detect+t_update+render+write+show): loc, chuyen doi,
                    generator, ghi raw... Ghi RIENG, khong gop vao ten stage nao.
Tong ket trong run_meta.json bo `--warmup-frames` frame dau va dung CUNG tap frame cho MOI cot.
    wall_vong_lap_s  = quanh `track_video()` (moi frame, KE CA warm-up, ke ca ve/ghi/hien thi)
    wall_khoi_tao_s  = mo nguon + nap detector + dung tracker
    wall_ghi_s       = ghi MOT/CSV/cache/meta SAU vong lap
    fps_vong_lap     = n_frame / wall_vong_lap_s          (co warm-up)
    fps_theo_stage   = 1000 / mean(frame_total_ms sau warm-up)

    python src/track_video_file.py --source video.mp4 --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \\
        --classes 1 2 3 5 7 --conf-min 0.25 --device 0 --out results/track_video/xe --save-video --show
    python src/track_video_file.py --dataset detrac --part test --video MVI_39031 --tracker-yaml bytetrack.yaml
    python src/track_video_file.py --source anh/ --fps 25 ...          # thu muc .jpg, thu tu theo SO trong ten
    python src/track_video_file.py --source anh/ --fps 25 --frames-manifest thu_tu.json   # thu tu tuong minh

Phim khi --show: q thoat (ket qua da xu ly VAN duoc ghi day du), SPACE tam dung.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
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
_SO_CUOI = re.compile(r"^(.*?)(\d+)$")


class TruncatedFrame(RuntimeError):
    """Frame co so detection cham tran `det_max_det` cua detector -- mac dinh TU CHOI (nhu VideoDetections)."""


# --------------------------------------------------------------------------- thu tu anh (PC-F4)
def sap_xep_anh(paths: list[Path], manifest: "Path | None" = None) -> list[Path]:
    """Chot QUY UOC thu tu anh trong thu muc.

    - Co `manifest` (JSON: list ten file, hoac list {"file": ...}): dung DUNG thu tu do; moi file phai ton tai.
    - Khong co manifest: ten (khong duoi) phai co dang <tien to>{so}; MOI file cung tien to -> sap theo
      GIA TRI SO (1, 2, 10 chu khong phai 1, 10, 2). Zero-padding hay khong deu duoc.
    - Ten khong co so cuoi, hoac tien to khac nhau (a1, b2), hoac trung so -> LOI co huong dan dung
      --frames-manifest. Khong am tham coi thu tu chu la thu tu thoi gian.
    """
    if manifest is not None:
        m = json.loads(Path(manifest).read_text(encoding="utf-8"))
        ten = [x["file"] if isinstance(x, dict) else x for x in (m["frames"] if isinstance(m, dict) else m)]
        d = {p.name: p for p in paths}
        thieu = [t for t in ten if t not in d]
        if thieu:
            raise FileNotFoundError(f"frames-manifest liet ke {len(thieu)} file khong co trong thu muc: {thieu[:5]}")
        return [d[t] for t in ten]
    khoa = []
    for p in paths:
        m = _SO_CUOI.match(p.stem)
        if not m:
            raise ValueError(f"khong suy ra duoc thu tu tu ten '{p.name}' (khong ket thuc bang so). "
                             "Dung --frames-manifest de chi dinh thu tu tuong minh.")
        khoa.append((m.group(1), int(m.group(2)), p))
    tien_to = {k[0] for k in khoa}
    if len(tien_to) > 1:
        raise ValueError(f"ten anh co {len(tien_to)} tien to khac nhau {sorted(tien_to)[:4]} -> thu tu mo ho. "
                         "Dung --frames-manifest.")
    so = [k[1] for k in khoa]
    if len(set(so)) != len(so):
        raise ValueError("co ten anh trung so thu tu -> mo ho. Dung --frames-manifest.")
    return [k[2] for k in sorted(khoa, key=lambda k: k[1])]


# --------------------------------------------------------------------------- nguon frame
class NguonFrame:
    """Lap qua frame: yield (frame_idx 1-based, time_s, img|None, decode_ms).

    decode_ms do QUANH DUNG `cv2.imread()` / `cap.read()` (PC-F1). Dong tai nguyen bang `dong()`.
    """

    def __init__(self, source: str | None, dataset: str | None, part: str, video: str | None,
                 fps: float | None, max_frames: int | None, frames_manifest: str | None = None):
        import cv2
        self.cv2, self.max_frames = cv2, max_frames
        self.cap, self.paths, self.regions = None, None, []
        self.sha256, self.fps_metadata = None, None
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
            self.quy_uoc_thu_tu = f"dataset {dataset}: FrameRef.index/time_s cua adapter"
            return
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"nguon khong ton tai: {p}")
        if p.is_dir():
            paths = [q for q in p.iterdir() if q.suffix.lower() in IMG_EXT]
            if not paths:
                raise FileNotFoundError(f"thu muc {p} khong co anh {sorted(IMG_EXT)}")
            if fps is None:
                raise ValueError("thu muc anh khong co fps: can --fps")
            paths = sap_xep_anh(paths, Path(frames_manifest) if frames_manifest else None)
            self.fps = float(fps)
            self.paths = [(i, (i - 1) / self.fps, q) for i, q in enumerate(paths, start=1)]
            img0 = cv2.imread(str(paths[0]))
            if img0 is None:
                raise RuntimeError(f"khong doc duoc anh dau tien {paths[0]}")
            self.orig_shape = tuple(img0.shape[:2])
            self.ten, self.loai, self.duong_dan = p.name, "image_dir", str(p)
            self.n_frames_khai_bao = len(paths)
            self.quy_uoc_thu_tu = ("frames-manifest" if frames_manifest else
                                   "sap theo GIA TRI SO o cuoi ten file (cung tien to)")
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
            raise ValueError("video khong co fps trong metadata: can --fps")
        self.fps = float(fps or fps_meta)
        self.fps_metadata = float(fps_meta)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_shape = (h, w)
        self.ten, self.loai, self.duong_dan = p.stem, "video_file", str(p)
        self.n_frames_khai_bao = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.sha256 = det_cache.sha256_file(p)
        self.quy_uoc_thu_tu = "thu tu frame cua container; time_s = (idx-1)/fps (KHONG dung CAP_PROP_POS_MSEC)"

    def manifest_frames(self) -> list[dict]:
        """Thu tu file -> frame -> time_s DA DUNG (de tai hien). Video file: khong co ten file."""
        if self.paths is None:
            return []
        return [{"frame": int(i), "time_s": float(t), "file": (str(q.name) if q is not None else None)}
                for i, t, q in self.paths]

    def __iter__(self):
        pc = time.perf_counter
        n = 0
        if self.paths is not None:
            for idx, t_s, q in self.paths:
                if self.max_frames and n >= self.max_frames:
                    return
                n += 1
                t0 = pc()
                img = self.cv2.imread(str(q)) if q is not None else None
                yield idx, float(t_s), img, (pc() - t0) * 1e3
            return
        idx = 0
        while True:
            if self.max_frames and n >= self.max_frames:
                return
            t0 = pc()
            ok, img = self.cap.read()
            dec = (pc() - t0) * 1e3
            if not ok:
                return
            idx += 1
            n += 1
            # time_s theo CHI SO frame va fps danh nghia (nhu FrameRef). Khong dung CAP_PROP_POS_MSEC:
            # nhieu codec tra 0 hoac lech -> n_steps sai. Video fps BIEN THIEN bi coi la deu (gioi han).
            yield idx, (idx - 1) / self.fps, img, dec

    def dong(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# --------------------------------------------------------------------------- adapter cho track_video
class LiveVideoDetections:
    """Detector chay TRUC TIEP tren frame, interface y het `det_cache.VideoDetections`
    (`iter_frames()` -> (frame_idx, t_s, unreadable, arr) va `orig_shape`).

    - `raw`: detection THO (chua loc) de ghi det cache -> replay duoc.
    - `pipeline`: thoi gian TUNG STAGE theo frame (PC-F1). `frame_total_ms` do tu TRUOC khi doc frame
      den khi generator duoc resume (tuc SAU tracker.update + on_frame cua frame do).
    - `last_img`: anh cua frame vua yield.
    - `allow_truncated=False` (mac dinh, y het VideoDetections): frame cham tran `spec.max_det` ->
      nem `TruncatedFrame` NGAY, khong tiep tuc voi detection da bi cat.
    """

    def __init__(self, nguon: NguonFrame, model, spec: det_cache.DetectorSpec, conf_min: float,
                 classes: list[int] | None, max_det: int, class_map: dict[int, int] | None,
                 allow_truncated: bool = False, ghi_ve: "dict | None" = None):
        self.nguon, self.model, self.spec = nguon, model, spec
        self.conf_min, self.max_det, self.class_map = float(conf_min), int(max_det), class_map
        self.allow_truncated = bool(allow_truncated)
        self._cls_set = None if classes is None else np.array(sorted(set(int(c) for c in classes)))
        self.orig_shape = tuple(int(v) for v in nguon.orig_shape)
        self.last_img = None
        self.raw = {"frame_idx": [], "time_s": [], "n_per_frame": [], "unreadable": [], "truncated": [],
                    "data": []}
        self.pipeline: list[dict] = []
        self.n_frames = 0
        self.n_truncated = 0
        # ghi_ve: dict do callback dien render/write/show cua frame hien tai (ms) -> gop vao pipeline
        self.ghi_ve = ghi_ve if ghi_ve is not None else {}

    def _ghi_raw(self, idx, t_s, det, unreadable: bool) -> None:
        r = self.raw
        r["frame_idx"].append(idx); r["time_s"].append(t_s)
        r["n_per_frame"].append(0 if det is None else len(det)); r["unreadable"].append(unreadable)
        r["truncated"].append(False if det is None else len(det) >= self.spec.max_det)
        if det is not None:
            r["data"].append(det)

    def iter_frames(self):
        pc = time.perf_counter
        it = iter(self.nguon)
        while True:
            t_bat_dau = pc()                       # TRUOC khi doc frame (PC-F1)
            try:
                idx, t_s, img, dec = next(it)     # gom cv2.imread / cap.read
            except StopIteration:
                return
            self.n_frames += 1
            row = {"frame": idx, "unreadable": img is None, "decode_ms": dec, "mask_ms": 0.0,
                   "detect_ms": 0.0, "n_det_raw": 0, "n_det_loc": 0,
                   "render_ms": 0.0, "write_ms": 0.0, "show_ms": 0.0, "frame_total_ms": float("nan")}
            if img is None:
                self._ghi_raw(idx, t_s, None, True)
                self.pipeline.append(row)
                self.last_img = None
                yield idx, t_s, True, np.zeros((0, 6), dtype=np.float32)
                row["frame_total_ms"] = (pc() - t_bat_dau) * 1e3
                continue
            t1 = pc()
            if self.spec.mask_ignored_regions and self.nguon.regions:
                img = det_cache._mask_regions(img, self.nguon.regions)
            t2 = pc()
            det = det_cache.detect_frame(self.model, img, self.spec)      # CUNG ma voi build cache
            t3 = pc()
            self._ghi_raw(idx, t_s, det, False)
            if len(det) >= self.spec.max_det:
                self.n_truncated += 1
                if not self.allow_truncated:
                    raise TruncatedFrame(
                        f"frame {idx}: {len(det)} detection cham tran --det-max-det {self.spec.max_det}; "
                        "tinh chat tien to [:max_det] khong con dung (y het VideoDetections tu choi khi "
                        "replay). Tang --det-max-det, hoac dung --allow-truncated (se ghi vao metadata va "
                        "lenh replay).")
            arr = det_cache.filter_dets(det, self.conf_min, self._cls_set, self.max_det, self.class_map)
            row.update(mask_ms=(t2 - t1) * 1e3, detect_ms=(t3 - t2) * 1e3,
                       n_det_raw=int(len(det)), n_det_loc=int(len(arr)))
            self.pipeline.append(row)
            self.last_img = img
            self.ghi_ve.clear()
            yield idx, t_s, False, arr
            # resume: tracker.update + on_frame (ve/ghi/hien thi) cua frame nay DA xong
            row["frame_total_ms"] = (pc() - t_bat_dau) * 1e3
            for k in ("render_ms", "write_ms", "show_ms"):
                row[k] = float(self.ghi_ve.get(k, 0.0))

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


# --------------------------------------------------------------------------- provenance
def _entry_point_provenance() -> dict:
    """sha256 (LF) cua CHINH file nay + trang thai working tree cua src/ luc chay (PC provenance).
    Khong mo rong CODE_FILES chung (se lam thi nghiem cu bi danh gia lai vo co)."""
    me = Path(__file__).resolve()
    d = {"file": me.name, "sha256_lf": ul_backend.sha256_lf(me)}
    try:
        st = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=config.PROJECT_ROOT,
                                     text=True, stderr=subprocess.DEVNULL)
        sua = [ln[3:] for ln in st.splitlines() if ln.strip()]
        d["working_tree_src"] = {"sach": not sua, "file_khac_HEAD": sua}
    except Exception:  # noqa: BLE001
        d["working_tree_src"] = {"sach": None, "ghi_chu": "khong doc duoc git status"}
    return d


def lenh_replay(a, ten: str, classes, seed, fps: float, allow_truncated: bool) -> str:
    """Lenh run_tracking.py voi DUNG cau hinh hieu luc + chinh sach cua lan live."""
    c = ["python src/run_tracking.py", f"--cache-id {a.save_cache}", "--split-name <split>",
         f"--tracker-yaml {a.tracker_yaml}", "--tracker-name <ten>", f"--videos {ten}",
         f"--conf-min {a.conf_min}",
         ("--all-classes" if classes is None else "--classes " + " ".join(map(str, classes))),
         f"--max-det {a.max_det}", f"--fps {fps:g}", f"--track-buffer-sec {a.track_buffer_sec}",
         f"--elapsed-mode {a.elapsed_mode}"]
    if a.class_map:
        c.append(f"--class-map {a.class_map}")
    if seed is not None:
        c.append(f"--seeds {seed}")
    if a.set:
        c.append("--set " + " ".join(a.set))
    if allow_truncated:
        c.append("--allow-truncated")
    return " ".join(c)


# --------------------------------------------------------------------------- CLI
def main() -> int:  # noqa: C901
    t_main = time.perf_counter()
    ap = argparse.ArgumentParser(description="Chay mot video qua detector + tracker")
    g = ap.add_argument_group("nguon")
    g.add_argument("--source", default=None, help="file video hoac thu muc anh")
    g.add_argument("--dataset", default=None, help="hoac: sequence dataset (detrac | bdd100k_coco ...)")
    g.add_argument("--part", default="test")
    g.add_argument("--video", default=None, help="ten sequence khi dung --dataset")
    g.add_argument("--fps", type=float, default=None, help="bat buoc voi thu muc anh; ghi de metadata video")
    g.add_argument("--max-frames", type=int, default=None)
    g.add_argument("--frames-manifest", default=None,
                   help="thu muc anh: JSON liet ke thu tu file tuong minh (khi ten khong suy ra duoc thu tu)")
    d = ap.add_argument_group("detector (cung DetectorSpec voi det cache)")
    d.add_argument("--model", default=config.YOLO_DEFAULT_MODEL, help="checkpoint YOLO (.pt)")
    d.add_argument("--device", default="0", help="0 | cpu")
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--half", action="store_true")
    d.add_argument("--det-conf", type=float, default=0.05,
                   help="nguong detector THO (mac dinh 0,05 = cache). Phai <= --conf-min")
    d.add_argument("--det-max-det", type=int, default=3000, help="tran detection cua detector. Phai >= --max-det")
    d.add_argument("--allow-truncated", action="store_true",
                   help="chap nhan frame cham tran --det-max-det (mac dinh TU CHOI, y het replay cache)")
    f = ap.add_argument_group("loc detection truoc khi vao tracker (y het run_tracking.py)")
    f.add_argument("--conf-min", type=float, default=0.25,
                   help="NGUONG HIEU LUC vao tracker (0.25 = c25, 0.10 = c10)")
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
    classes = None if a.all_classes else a.classes
    class_map = run_tracking.parse_class_map(a.class_map)

    # ---- PC-F2: CUNG chinh sach tuong thich producer/consumer voi duong doc cache, TRUOC khi lam gi ----
    spec_meta = {"conf": a.det_conf, "classes": None, "max_det": a.det_max_det}
    loi_tt = det_cache.check_compatible(spec_meta, a.conf_min, classes, a.max_det)
    if loi_tt:
        print("[LOI] cau hinh live KHONG the replay tu cache (cung guard voi det_cache.check_compatible):")
        for e in loi_tt:
            print(f"      - {e}")
        return 2

    # ---- nguon ----
    try:
        nguon = NguonFrame(a.source, a.dataset, a.part, a.video, a.fps, a.max_frames, a.frames_manifest)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[LOI] nguon: {e}")
        return 2
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

    # ---- tracker: cung cach nap cau hinh voi runner ----
    overrides = run_tracking.parse_set(a.set)
    overrides["track_buffer"] = track_buffer_frames(a.track_buffer_sec, nguon.fps)
    cfg = run_tracking.load_tracker_cfg(a.tracker_yaml, overrides)
    seed = a.seed
    if cfg.tracker_type in run_tracking.STOCHASTIC_TRACKERS and seed is None:
        seed = int(getattr(cfg, "pf_seed", 0))
    dt_nominal = 1.0 / nguon.fps
    yaml_path = Path(run_tracking.resolve_tracker_yaml(a.tracker_yaml))
    yaml_sha = ul_backend.sha256_lf(yaml_path)
    # Lam nong: lan `make_tracker` DAU TIEN mat ~0,4 s (import luoi cac module tracker). Goi mot lan o day
    # de chi phi do thuoc `wall_khoi_tao`, khong lot vao `wall_vong_lap` ma khong thuoc frame nao.
    # `track_video` van tu dung tracker rieng (reset id) -> ket qua khong doi (test 6b kiem bit-identical).
    run_tracking.make_tracker(cfg)

    print(f"[track_video] nguon={nguon.loai}:{nguon.duong_dan}  {nguon.orig_shape[1]}x{nguon.orig_shape[0]} "
          f"@ {nguon.fps:g} fps  ({nguon.n_frames_khai_bao} frame khai bao); thu tu: {nguon.quy_uoc_thu_tu}")
    print(f"[track_video] detector={w.name} det_conf={spec.conf} det_max_det={spec.max_det} device={a.device} "
          f"allow_truncated={a.allow_truncated}")
    print(f"[track_video] loc: conf>{a.conf_min} (NGUONG HIEU LUC vao tracker) classes={classes} "
          f"max_det={a.max_det} class_map={class_map}")
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

    ghi_ve: dict = {}
    vd = LiveVideoDetections(nguon, model, spec, a.conf_min, classes, a.max_det, class_map,
                             allow_truncated=a.allow_truncated, ghi_ve=ghi_ve)
    trang_thai = {"dung_som": False, "n_ve": 0}
    wall_khoi_tao = time.perf_counter() - t_main

    def on_frame(frame_idx: int, out_arr: np.ndarray):
        if vd.last_img is None or (writer is None and cua_so is None):
            return True
        pc = time.perf_counter
        t0 = pc()
        img = vd.last_img.copy()
        ve_frame(img, out_arr, ten_lop)
        cv2.putText(img, f"f{frame_idx} {cfg.tracker_type} n={len(out_arr)}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        t1 = pc()
        if writer is not None:
            writer.write(img)
        t2 = pc()
        ghi_ve["render_ms"], ghi_ve["write_ms"] = (t1 - t0) * 1e3, (t2 - t1) * 1e3
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
                        ghi_ve["show_ms"] = (pc() - t2) * 1e3
                        return False
            ghi_ve["show_ms"] = (pc() - t2) * 1e3
            if k == ord("q"):
                trang_thai["dung_som"] = True
                return False
        return True

    # ---- chay: CUNG vong lap voi runner ----
    t_vong_lap = time.perf_counter()
    loi_tran = None
    try:
        df, df_cls, df_t = run_tracking.track_video(vd, cfg, dt_nominal, a.elapsed_mode, seed, ten,
                                                     on_frame=on_frame)
    except TruncatedFrame as e:
        loi_tran = str(e)
    finally:
        # GIAI PHONG tai nguyen du thanh cong, dung som hay loi
        nguon.dong()
        if writer is not None:
            writer.release()
        if cua_so is not None:
            cv2.destroyAllWindows()
    wall_vong_lap = time.perf_counter() - t_vong_lap
    if loi_tran is not None:
        print(f"[LOI] {loi_tran}")
        print("[LOI] KHONG ghi ket qua: tracking da nhan detection bi cat (khong hop le theo chinh sach strict).")
        return 3

    # ---- ghi ket qua ----
    t_ghi = time.perf_counter()
    (out / "data").mkdir(exist_ok=True); (out / "data_cls").mkdir(exist_ok=True)
    (out / "timing").mkdir(exist_ok=True); (out / "pipeline").mkdir(exist_ok=True)
    df.to_csv(out / "data" / f"{ten}.txt", header=False, index=False)
    df_cls.to_csv(out / "data_cls" / f"{ten}.csv", index=False)
    df_t.to_csv(out / "timing" / f"{ten}.csv", index=False)
    import pandas as pd
    df_p = pd.DataFrame(vd.pipeline)
    if len(df_p):
        df_p = df_p.merge(df_t[["frame", "t_update_ms", "t_predict_ms"]], on="frame", how="left")
        stage = df_p[["decode_ms", "mask_ms", "detect_ms", "t_update_ms", "render_ms", "write_ms", "show_ms"]]
        df_p["overhead_ms"] = df_p["frame_total_ms"] - stage.fillna(0.0).sum(axis=1)
    df_p.to_csv(out / "pipeline" / f"{ten}.csv", index=False)
    (out / "frames_manifest.json").write_text(json.dumps(
        {"quy_uoc": nguon.quy_uoc_thu_tu, "fps": nguon.fps, "frames": nguon.manifest_frames()},
        ensure_ascii=False, indent=1), encoding="utf-8")

    cache_info = None
    if a.save_cache:
        cdir = det_cache.CACHE_ROOT / a.save_cache
        det_cache.write_meta(cdir, spec, mode="append")
        cache_info = det_cache.save_video(cdir, ten, vd.cache_arrays(),
                                          extra={"part": spec.part, "n_regions": len(nguon.regions),
                                                 "source": nguon.duong_dan, "fps": nguon.fps,
                                                 "allow_truncated": a.allow_truncated},
                                          spec=spec)
    wall_ghi = time.perf_counter() - t_ghi

    # ---- tong ket thoi gian: CUNG tap frame (sau warm-up, doc duoc) cho MOI cot ----
    def tb(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return float(x.mean()) if len(x) else float("nan")
    wu = a.warmup_frames
    ok = df_p[(~df_p.unreadable) & (df_p.frame > wu)] if len(df_p) else df_p
    cols = ["decode_ms", "mask_ms", "detect_ms", "t_update_ms", "t_predict_ms", "render_ms", "write_ms",
            "show_ms", "overhead_ms", "frame_total_ms"]
    tong_ket = {c: (tb(ok[c]) if len(ok) else float("nan")) for c in cols}
    thoi_gian = {
        "pham_vi": {
            "decode_ms": "quanh cv2.imread/cap.read", "mask_ms": "to xam vung bo qua (dataset)",
            "detect_ms": "det_cache.detect_frame (preprocess+suy luan+NMS)",
            "t_update_ms": "tracker.update (t_predict_ms = phan multi_predict, nam trong t_update)",
            "render_ms/write_ms/show_ms": "ve bbox / VideoWriter.write / imshow+waitKey (0 neu tat)",
            "frame_total_ms": "TRUOC khi doc frame -> SAU tracker + ve/ghi/hien thi (do trong adapter)",
            "overhead_ms": "frame_total - (decode+mask+detect+t_update+render+write+show): loc, generator, ghi raw",
            "tap_frame_tong_ket": f"frame > {wu} va doc duoc; CUNG tap cho moi cot",
        },
        "n_frame_xu_ly": int(len(df_p)), "n_frame_unreadable": int(df_p.unreadable.sum()) if len(df_p) else 0,
        "n_frame_truncated": int(vd.n_truncated), "warmup_frames_bo": wu, "n_frame_tong_ket": int(len(ok)),
        "trung_binh_ms": tong_ket,
        "tong_stage_ms": (tong_ket["decode_ms"] + tong_ket["mask_ms"] + tong_ket["detect_ms"]
                          + tong_ket["t_update_ms"] + tong_ket["render_ms"] + tong_ket["write_ms"]
                          + tong_ket["show_ms"]),
        "fps_theo_stage": (1000.0 / tong_ket["frame_total_ms"] if tong_ket["frame_total_ms"] > 0 else float("nan")),
        "wall_khoi_tao_s": wall_khoi_tao, "wall_vong_lap_s": wall_vong_lap, "wall_ghi_s": wall_ghi,
        "wall_tong_s": time.perf_counter() - t_main,
        # phan cua wall_vong_lap KHONG thuoc frame nao (dung tracker, StopIteration, dung DataFrame)
        "wall_ngoai_frame_s": (wall_vong_lap - float(np.nansum(df_p.frame_total_ms)) / 1000.0) if len(df_p) else None,
        "fps_vong_lap": (len(df_p) / wall_vong_lap if wall_vong_lap > 0 else float("nan")),
        "ghi_chu": ("trung_binh_ms tinh tren CUNG tap frame sau warm-up. fps_theo_stage = 1000/frame_total_ms "
                    "(sau warm-up). fps_vong_lap = n_frame/wall_vong_lap (KE CA warm-up; gom ve/ghi/hien thi "
                    "neu bat; KHONG gom khoi tao va ghi ket qua). wall_vong_lap = sum(frame_total moi frame) "
                    "+ wall_ngoai_frame. Do MOT luot, khong lap co kiem soat -> so mo ta, khong dung de xep hang."),
    }

    code_fp = __import__("run_fingerprint").code_fingerprint()
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": det_cache.git_commit(),
        "entry_point": _entry_point_provenance(), "lenh": " ".join(sys.argv),
        "nguon": {"loai": nguon.loai, "duong_dan": nguon.duong_dan, "ten": ten, "sha256": nguon.sha256,
                  "fps": nguon.fps, "fps_metadata": nguon.fps_metadata,
                  "n_frames_khai_bao": nguon.n_frames_khai_bao, "orig_shape_hw": list(nguon.orig_shape),
                  "n_regions_bo_qua": len(nguon.regions), "max_frames": a.max_frames,
                  "quy_uoc_thu_tu": nguon.quy_uoc_thu_tu, "frames_manifest": a.frames_manifest,
                  "doc_gt": ("khong" if nguon.loai != "dataset" else
                             "chi vung bo qua (ignored_regions) de mask anh; KHONG dua GT vao association")},
        "detector_spec": asdict(spec),
        "loc": {"conf_min": a.conf_min, "classes": classes, "class_map": class_map, "max_det": a.max_det,
                "nguong_hieu_luc_vao_tracker": a.conf_min,
                "allow_truncated": a.allow_truncated,
                "tuong_thich_cache": "da kiem det_cache.check_compatible truoc khi chay: OK"},
        "tracker": {"yaml": a.tracker_yaml, "yaml_resolved": str(yaml_path), "yaml_sha256": yaml_sha,
                    "overrides": overrides, "cfg_hieu_luc": run_tracking.cfg_to_dict(cfg),
                    "tracker_type": cfg.tracker_type, "track_buffer_frames": int(cfg.track_buffer),
                    "track_buffer_sec": a.track_buffer_sec, "elapsed_mode": a.elapsed_mode,
                    "dt_nominal_s": dt_nominal, "seed": seed, "backend": code_fp["backend"]},
        "code": code_fp, "ten_lop": {int(k): v for k, v in ten_lop.items()},
        "ket_qua": {"n_frame_xu_ly": int(len(df_p)), "n_hang_tracking": int(len(df)),
                    "n_track": int(df["id"].nunique()) if len(df) else 0,
                    "dung_som": trang_thai["dung_som"], "n_frame_ve": trang_thai["n_ve"],
                    "n_frame_truncated": int(vd.n_truncated),
                    "video_tracked": str(video_out) if a.save_video else None,
                    "det_cache": ({"cache_id": a.save_cache, "dir": str(det_cache.CACHE_ROOT / a.save_cache),
                                   **cache_info} if cache_info else None)},
        "thoi_gian": thoi_gian,
        "replay": (lenh_replay(a, ten, classes, seed, nguon.fps, a.allow_truncated) if a.save_cache else None),
        "replay_trang_thai": (None if not a.save_cache else
                              ("san dung: cau hinh da qua check_compatible; "
                               + ("co frame cham tran, replay CAN --allow-truncated (da them)" if vd.n_truncated
                                  else "khong frame nao cham tran"))),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False),
                                       encoding="utf-8")

    tk = tong_ket
    print(f"[track_video] {len(df_p)} frame ({thoi_gian['n_frame_unreadable']} unreadable, "
          f"{vd.n_truncated} cham tran), {len(df)} hang tracking, {meta['ket_qua']['n_track']} track"
          + (" -- DUNG SOM theo yeu cau, ket qua da xu ly van ghi day du" if trang_thai["dung_som"] else ""))
    print(f"[track_video] sau warm-up ({len(ok)} frame): decode {tk['decode_ms']:.2f} | mask {tk['mask_ms']:.2f} | "
          f"detect {tk['detect_ms']:.2f} | tracker {tk['t_update_ms']:.2f} | ve {tk['render_ms']:.2f} | "
          f"ghi {tk['write_ms']:.2f} | show {tk['show_ms']:.2f} | overhead {tk['overhead_ms']:.2f} "
          f"= frame_total {tk['frame_total_ms']:.2f} ms  -> {thoi_gian['fps_theo_stage']:.1f} fps theo stage")
    print(f"[track_video] wall: khoi tao {wall_khoi_tao:.1f}s | vong lap {wall_vong_lap:.1f}s "
          f"({thoi_gian['fps_vong_lap']:.1f} fps, ke ca warm-up) | ghi {wall_ghi:.1f}s")
    print(f"[track_video] -> {out}")
    if cache_info:
        print(f"[track_video] det cache -> {det_cache.CACHE_ROOT / a.save_cache}  ({meta['replay_trang_thai']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
