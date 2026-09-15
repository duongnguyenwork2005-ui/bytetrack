"""
run_tracking.py -- Chay tracker tren DETECTION CACHE (khong chay detector).

Duong chay Y HET callback cua ultralytics (trackers/track.py,
on_predict_postprocess_end): `det = result.boxes.cpu().numpy()` roi
`tracker.update(det, orig_img)`. O day `det` duoc boi lai tu cache bang
det_cache.to_boxes() (float32, thu tu NMS) va `orig_img = None` (ByteTrack
khong dung anh). Frame khong co detection VAN goi update voi Boxes rong (nhu
model.track van goi callback) de frame_id tang deu. Ket qua tracker (mang
[x1,y1,x2,y2,id,score,cls,idx]) ghi ra MOTChallenge 7 cot y het
baseline_track.py (xem test_run_tracking_replay.py: bit-identical khi cung
detection).

QUY TAC DE TAI LAP
    - Moi (tracker, seed, video) MOT tracker moi: bo dem track_id cua ultralytics
      la toan cuc, xen ke hai tracker se cap ID chen nhau.
    - Tracker xac dinh (cv / ekf / ukf / sort) chay 1 lan du co nhieu seed; tracker
      ngau nhien (pf) chay moi seed mot lan, thu muc them hau to -s<seed>.
    - `--elapsed-mode skip` (mac dinh, tai lap baseline): frame khong doc duoc bi
      bo qua hoan toan nhu baseline_track.py. `--elapsed-mode steps`: so buoc da
      troi qua (tu time_s cua cache) duoc dua vao update(n_steps=k) - xem time_units.py.

THU MUC KET QUA
    data/processed/trackers/<split>/<tracker_name>/
        data/<video>.txt        frame,id,x,y,w,h,conf   (TrackEval + occlusion_eval doc)
        data_cls/<video>.csv    frame,id,cls            (tach theo lop cho BDD100K)
        timing/<video>.csv      frame,n_steps,n_tracks,n_dets,t_predict_ms,t_assoc_ms,t_update_ms
        run_meta.json           cache_id, yaml (sha), override, seed, elapsed_mode, commit, phien ban

CACH DUNG
    python src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name DETRAC-test \
        --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --tracker-name p2-ekf --videos MVI_39031 --conf-min 0.25
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import tracker_ctrv  # noqa: E402
from time_units import steps_between, track_buffer_frames, with_elapsed_steps  # noqa: E402

from ultralytics.utils import IterableSimpleNamespace, YAML  # noqa: E402

#: tracker_type ma output phu thuoc seed (chay moi seed mot lan)
STOCHASTIC_TRACKERS = {"bytetrack_pf_ctrv"}
TRK_COLS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]


def tracker_registry() -> dict:
    """tracker_type -> lop tracker. Gom ultralytics TRACKER_MAP + CTRV + SORT (neu co)."""
    tracker_ctrv.register()
    from ultralytics.trackers.track import TRACKER_MAP
    reg = dict(TRACKER_MAP)
    try:
        import sort_tracker
        reg.update(sort_tracker.SORT_TRACKERS)
    except ImportError:
        pass
    return reg


def resolve_tracker_yaml(yaml_path: str | Path) -> str:
    """Ten tran nhu 'bytetrack.yaml' -> file trong package ultralytics (giong model.track)."""
    from ultralytics.utils.checks import check_yaml
    return str(check_yaml(str(yaml_path)))


def load_tracker_cfg(yaml_path: str | Path, overrides: dict | None = None) -> IterableSimpleNamespace:
    d = YAML.load(resolve_tracker_yaml(yaml_path))
    for k, v in (overrides or {}).items():
        d[k] = v
    return IterableSimpleNamespace(**d)


def parse_set(items: list[str] | None) -> dict:
    """--set key=value (value: so / bool / chuoi)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        vl = v.lower()
        if vl in ("true", "false"):
            out[k] = vl == "true"
        else:
            try:
                out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


def parse_class_map(s: str | None) -> dict[int, int] | None:
    """'7:2,5:2' -> {7: 2, 5: 2}"""
    if not s:
        return None
    return {int(a): int(b) for a, b in (p.split(":") for p in s.split(","))}


def make_tracker(cfg: IterableSimpleNamespace, registry: dict | None = None, assoc_log: bool = False):
    registry = registry or tracker_registry()
    if cfg.tracker_type not in registry:
        raise ValueError(f"tracker_type {cfg.tracker_type!r} khong co trong registry {sorted(registry)}")
    cls = registry[cfg.tracker_type]
    if getattr(cfg, "lost_no_fuse", False) or str(getattr(cfg, "lost_recover", "none")).lower() not in ("none", "false", ""):
        from lost_recovery import with_lost_recovery    # Moc 2 C2: chi khi .yaml bat (mac dinh: khong boc)
        cls = with_lost_recovery(cls)
    cls = with_elapsed_steps(cls)
    if assoc_log:
        from assoc_log import with_assoc_log       # Moc 2: log quyet dinh ghep (khong doi hanh vi)
        cls = with_assoc_log(cls)
    tr = cls(args=cfg)
    tr.reset()
    return tr


class _PredictTimer:
    """Boc tracker.multi_predict de do thoi gian bo loc, khong doi hanh vi."""

    def __init__(self, tracker):
        self.tracker = tracker
        self.t = 0.0
        self._orig = tracker.multi_predict

        def timed(tracks):
            t0 = time.perf_counter()
            self._orig(tracks)
            self.t += time.perf_counter() - t0
        tracker.multi_predict = timed

    def pop(self) -> float:
        t, self.t = self.t, 0.0
        return t


def track_video(vd: det_cache.VideoDetections, cfg: IterableSimpleNamespace, dt_nominal: float,
                elapsed_mode: str = "skip", seed: int | None = None, video: str = "",
                registry: dict | None = None, log_sink: dict | None = None):
    """Chay mot tracker moi tren toan bo frame cua cache. Tra ve (rows, cls_rows, timing).

    log_sink (Moc 2, tuy chon): dict; neu co khoa "assoc_log": True thi tracker duoc boc
    AssociationLogMixin va sau khi chay `log_sink["assoc_log"]` = AssocLog; neu "pf_diag": True
    (tracker PF) thi `log_sink["pf_diag"]` = DataFrame chan doan particle; luon dat
    `log_sink["tracker"]` = tracker (de test doc trang thai cuoi).
    """
    log_sink = log_sink if log_sink is not None else {}
    tr = make_tracker(cfg, registry, assoc_log=bool(log_sink.get("assoc_log")))
    if seed is not None and hasattr(tr, "set_seed"):
        tr.set_seed(seed, video)          # PF: rng theo (seed, video, track_id)
    if log_sink.get("pf_diag") and hasattr(tr, "kalman_filter") and hasattr(tr.kalman_filter, "enable_diag"):
        tr.kalman_filter.enable_diag()
    timer = _PredictTimer(tr)
    rows, cls_rows, timing = [], [], []
    last_t = None
    pending_steps = 0
    for frame_idx, t_s, unreadable, arr in vd.iter_frames():
        if unreadable:
            if elapsed_mode == "steps":
                pending_steps += 1        # thoi gian van troi, khong co quan sat
            continue                      # 'skip': bo qua hoan toan nhu baseline_track.py
        if elapsed_mode == "steps" and last_t is not None:
            n_steps, _regular = steps_between(t_s - last_t, dt_nominal)
        else:
            n_steps = 1
        pending_steps = 0
        last_t = t_s
        boxes = det_cache.to_boxes(arr, vd.orig_shape)
        tr.log_frame = int(frame_idx)      # chi so frame cua video (cho assoc_log / pf_diag)
        t0 = time.perf_counter()
        out = tr.update(boxes.cpu().numpy(), None, n_steps=n_steps)
        t_update = time.perf_counter() - t0
        t_pred = timer.pop()
        n_tracks = len(getattr(tr, "tracked_stracks", None) or getattr(tr, "tracks", []))
        timing.append((frame_idx, n_steps, n_tracks, len(arr),
                       t_pred * 1e3, (t_update - t_pred) * 1e3, t_update * 1e3))
        if len(out) == 0:
            continue
        xyxy = out[:, :4].astype(np.float32)      # giong r.boxes.xyxy.cpu().numpy() cua baseline
        ids = out[:, 4].astype(int)
        confs = out[:, 5].astype(np.float32)
        clss = out[:, 6].astype(int)
        for (x1, y1, x2, y2), tid, cf, c in zip(xyxy, ids, confs, clss):
            rows.append((frame_idx, int(tid), float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1), float(cf)))
            cls_rows.append((frame_idx, int(tid), int(c)))
    df = pd.DataFrame(rows, columns=TRK_COLS)
    df_cls = pd.DataFrame(cls_rows, columns=["frame", "id", "cls"])
    df_t = pd.DataFrame(timing, columns=["frame", "n_steps", "n_tracks", "n_dets",
                                         "t_predict_ms", "t_assoc_ms", "t_update_ms"])
    log_sink["tracker"] = tr
    if log_sink.get("assoc_log"):
        log_sink["assoc_log"] = tr.assoc_log
    if log_sink.get("pf_diag") and hasattr(getattr(tr, "kalman_filter", None), "diag_frame"):
        log_sink["pf_diag"] = tr.kalman_filter.diag_frame()
    return df, df_cls, df_t


def run_video_to_dir(out_root: Path, video: str, vd, cfg, dt_nominal, elapsed_mode, seed=None,
                     registry=None, assoc_log: bool = False, pf_diag: bool = False):
    sink = {"assoc_log": assoc_log, "pf_diag": pf_diag}
    df, df_cls, df_t = track_video(vd, cfg, dt_nominal, elapsed_mode, seed, video, registry, log_sink=sink)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "data_cls").mkdir(exist_ok=True)
    (out_root / "timing").mkdir(exist_ok=True)
    df.to_csv(out_root / "data" / f"{video}.txt", header=False, index=False)
    df_cls.to_csv(out_root / "data_cls" / f"{video}.csv", index=False)
    df_t.to_csv(out_root / "timing" / f"{video}.csv", index=False)
    if assoc_log:
        from assoc_log import save_log
        save_log(sink["assoc_log"], out_root / "assoc_log", video)
    if pf_diag and isinstance(sink.get("pf_diag"), pd.DataFrame):
        (out_root / "pf_diag").mkdir(exist_ok=True)
        sink["pf_diag"].to_csv(out_root / "pf_diag" / f"{video}.csv", index=False)
    return df, df_t


def main() -> int:
    ap = argparse.ArgumentParser(description="Chay tracker tren detection cache")
    ap.add_argument("--cache-id", required=True)
    ap.add_argument("--split-name", required=True, help="split GT (thu muc data/processed/<split>)")
    ap.add_argument("--tracker-yaml", required=True)
    ap.add_argument("--tracker-name", required=True)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--all-cached", action="store_true", help="moi video co trong cache")
    ap.add_argument("--conf-min", type=float, default=0.25, help="loc detection > conf (dieu kien A=0.25, B=0.10)")
    ap.add_argument("--classes", nargs="*", type=int, default=[2, 3, 5, 7])
    ap.add_argument("--all-classes", action="store_true")
    ap.add_argument("--class-map", default=None, help="vd '7:2,5:2' gop lop truoc khi vao tracker")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--fps", type=float, default=None, help="mac dinh: theo dataset trong meta cache")
    ap.add_argument("--track-buffer-sec", type=float, default=None, help="doi sang buoc theo fps, ghi de yaml")
    ap.add_argument("--set", nargs="*", default=None, help="ghi de khoa yaml: key=value")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--elapsed-mode", choices=["skip", "steps"], default="skip")
    ap.add_argument("--allow-truncated", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--assoc-log", action="store_true", help="ghi assoc_log/<video>_*.parquet (Moc 2)")
    ap.add_argument("--pf-diag", action="store_true", help="ghi pf_diag/<video>.csv (PF)")
    args = ap.parse_args()

    cache_dir = det_cache.CACHE_ROOT / args.cache_id
    meta = det_cache.load_meta(cache_dir)
    if args.videos:
        videos = args.videos
    elif args.manifest:
        videos = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["videos"]
    elif args.all_cached:
        videos = sorted(p.stem for p in cache_dir.glob("*.npz"))
    else:
        print("[ERROR] Can --videos, --manifest hoac --all-cached"); return 1

    fps = args.fps
    if fps is None:
        from mot_datasets import get_dataset
        fps = get_dataset(meta["dataset"]).fps
    dt_nominal = 1.0 / fps

    overrides = parse_set(args.set)
    if args.track_buffer_sec is not None:
        overrides["track_buffer"] = track_buffer_frames(args.track_buffer_sec, fps)
    cfg = load_tracker_cfg(args.tracker_yaml, overrides)
    registry = tracker_registry()
    classes = None if args.all_classes else args.classes
    class_map = parse_class_map(args.class_map)
    stochastic = cfg.tracker_type in STOCHASTIC_TRACKERS
    seeds = args.seeds if (stochastic and args.seeds) else [None]
    if stochastic and not args.seeds:
        seeds = [int(getattr(cfg, "pf_seed", 0))]

    yaml_sha = hashlib.sha256(Path(resolve_tracker_yaml(args.tracker_yaml)).read_bytes()).hexdigest()[:12]
    print(f"[run_tracking] cache={args.cache_id} tracker={cfg.tracker_type} yaml={args.tracker_yaml} "
          f"conf>{args.conf_min} classes={classes} max_det={args.max_det} fps={fps} "
          f"track_buffer={cfg.track_buffer} elapsed={args.elapsed_mode} seeds={seeds}")
    for seed in seeds:
        name = args.tracker_name + (f"-s{seed}" if seed is not None else "")
        out_root = config.PROCESSED_DIR / "trackers" / args.split_name / name
        out_root.mkdir(parents=True, exist_ok=True)
        run_meta = {"cache_id": args.cache_id, "cache_meta": meta, "tracker_yaml": str(args.tracker_yaml),
                    "tracker_yaml_sha256_12": yaml_sha, "tracker_cfg": vars(cfg), "overrides": overrides,
                    "conf_min": args.conf_min, "classes": classes, "class_map": class_map,
                    "max_det": args.max_det, "fps": fps, "dt_nominal_s": dt_nominal,
                    "track_buffer_frames": int(cfg.track_buffer),
                    "track_buffer_sec": round(int(cfg.track_buffer) / fps, 4),
                    "elapsed_mode": args.elapsed_mode, "seed": seed,
                    "git_commit": det_cache.git_commit(), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
        summary = []
        for v in videos:
            out_file = out_root / "data" / f"{v}.txt"
            if args.skip_existing and out_file.exists():
                print(f"  {name:<28} {v:<14} [SKIP]"); continue
            vd = det_cache.open_cache(args.cache_id, v, args.conf_min, classes, args.max_det, class_map,
                                      allow_truncated=args.allow_truncated)
            t0 = time.time()
            df, df_t = run_video_to_dir(out_root, v, vd, cfg, dt_nominal, args.elapsed_mode, seed, registry,
                                        assoc_log=args.assoc_log, pf_diag=args.pf_diag)
            sec = time.time() - t0
            n_tr = int(df["id"].nunique()) if len(df) else 0
            print(f"  {name:<28} {v:<14} {len(vd):>5} frame -> {len(df):>7,} bbox {n_tr:>4} track  "
                  f"predict {df_t.t_predict_ms.mean():.2f} ms/frame  assoc {df_t.t_assoc_ms.mean():.2f}  "
                  f"({sec:.1f}s, {len(vd) / max(sec, 1e-9):.0f} fps)")
            summary.append({"video": v, "n_frames": len(vd), "n_boxes": len(df), "n_tracks": n_tr,
                            "predict_ms_per_frame": round(float(df_t.t_predict_ms.mean()), 4),
                            "assoc_ms_per_frame": round(float(df_t.t_assoc_ms.mean()), 4),
                            "update_ms_per_frame": round(float(df_t.t_update_ms.mean()), 4),
                            "seconds": round(sec, 2)})
            pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
