"""
profile_pipeline.py -- Do thoi gian TUNG STAGE va bo nhem cua pipeline detector + tracker (Giai doan 2).

Stage (ms/frame, trung binh + p50/p95 sau warm-up):
    decode        cv2.imread
    mask          to xam vung bo qua (UA-DETRAC)
    preprocess    letterbox + to tensor  (Results.speed['preprocess'] cua ultralytics, da sync CUDA)
    inference     forward detector        (Results.speed['inference'])
    postprocess   NMS + scale box         (Results.speed['postprocess'])
    predict       bo loc: tracker.multi_predict (perf_counter)
    assoc         association + vong doi = update - predict
    render        ve bbox (tuy chon --render)
    total         decode -> tracker (khong gom render)
Bo nhem: RSS (psutil) truoc/sau, torch.cuda.max_memory_allocated / reserved.
Metadata: do phan giai anh, imgsz, batch = 1, half, so frame warm-up / xu ly / bo, phien ban
ultralytics/torch/cuda/cudnn, GPU, driver, host, cau hinh tracker, git commit.

Detector chay qua model.predict (khong qua model.track) va tracker chay rieng tren ket qua
-> do duoc rieng bo loc va association; duong chay tracker y het runner (det_cache.to_boxes).

Jetson Nano: CHUA DO TREN THIET BI. Script nay chay tren PC (GTX 1650) lam tham chieu; tren
Nano goc (JetPack 4.6, Python 3.6) ultralytics 8.4 khong chay duoc -> xem scripts/nano/README.md.

    python src/profile_pipeline.py --video MVI_39031 --part test --n-frames 500 --warmup 50 \
        --tracker-yaml configs/bytetrack_ekf_ctrv.yaml [--half] [--imgsz 640] [--device 0|cpu] [--render]
    -> results/phase2/profile/<host>_<detector>_<tracker>_<device>.json (+ .csv per-frame)
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
from baseline_track import mask_ignored_regions  # noqa: E402
from mot_datasets import get_dataset  # noqa: E402


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="detrac")
    ap.add_argument("--part", default="test")
    ap.add_argument("--video", default="MVI_39031")
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--model", default=config.YOLO_DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="*", type=int, default=[2, 3, 5, 7])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--tracker-yaml", default="bytetrack.yaml")
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out-dir", default=str(config.RESULTS_DIR / "phase2" / "profile"))
    a = ap.parse_args()

    import psutil
    import torch
    import ultralytics
    from ultralytics import YOLO

    ds = get_dataset(a.dataset)
    frames = ds.frames(a.video, a.part)[: a.n_frames]
    regions = ds.ignored_regions(a.video, a.part)
    model = YOLO(a.model)
    cfg = run_tracking.load_tracker_cfg(a.tracker_yaml, run_tracking.parse_set(a.set))
    tr = run_tracking.make_tracker(cfg)
    if hasattr(tr, "set_seed"):
        tr.set_seed(0, a.video)
    timer = run_tracking._PredictTimer(tr)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    if torch.cuda.is_available() and a.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    kw = {"half": True} if a.half else {}

    rows = []
    n_skipped = 0
    for i, fr in enumerate(frames):
        t0 = time.perf_counter()
        img = cv2.imread(str(fr.path))
        t1 = time.perf_counter()
        if img is None:
            n_skipped += 1
            continue
        img = mask_ignored_regions(img, regions)
        t2 = time.perf_counter()
        r = model.predict(img, conf=a.conf, imgsz=a.imgsz, classes=a.classes, device=a.device,
                          verbose=False, **kw)[0]
        cuda_sync()
        t3 = time.perf_counter()
        det = r.boxes.data.cpu().numpy().astype(np.float32)
        boxes = det_cache.to_boxes(det, img.shape[:2])
        t4 = time.perf_counter()
        out = tr.update(boxes.cpu().numpy(), None)
        t5 = time.perf_counter()
        t_pred = timer.pop()
        t_render = 0.0
        if a.render:
            t6 = time.perf_counter()
            for x1, y1, x2, y2, tid in out[:, :5]:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, str(int(tid)), (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            t_render = (time.perf_counter() - t6) * 1e3
        sp = r.speed
        rows.append({"frame": fr.index, "warmup": i < a.warmup, "n_dets": len(det), "n_tracks": len(out),
                     "decode_ms": (t1 - t0) * 1e3, "mask_ms": (t2 - t1) * 1e3,
                     "preprocess_ms": sp["preprocess"], "inference_ms": sp["inference"],
                     "postprocess_ms": sp["postprocess"], "detector_wall_ms": (t3 - t2) * 1e3,
                     "to_boxes_ms": (t4 - t3) * 1e3, "predict_ms": t_pred * 1e3,
                     "assoc_ms": ((t5 - t4) - t_pred) * 1e3, "tracker_ms": (t5 - t4) * 1e3,
                     "render_ms": t_render, "total_ms": (t5 - t0) * 1e3})
    df = pd.DataFrame(rows)
    m = df[~df.warmup]
    stages = ["decode_ms", "mask_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "detector_wall_ms",
              "predict_ms", "assoc_ms", "tracker_ms", "render_ms", "total_ms"]
    summary = {s: {"mean": float(m[s].mean()), "p50": float(m[s].median()), "p95": float(m[s].quantile(0.95))}
               for s in stages}
    rss1 = proc.memory_info().rss
    gpu = {}
    if torch.cuda.is_available() and a.device != "cpu":
        gpu = {"name": torch.cuda.get_device_name(0), "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
               "max_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
    meta = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None, "gpu": gpu,
            "device": a.device, "detector": str(a.model), "weights_sha256": det_cache.sha256_file(Path(a.model)) if Path(a.model).exists() else None,
            "imgsz": a.imgsz, "conf": a.conf, "classes": a.classes, "half": a.half, "batch": 1,
            "image_shape": list(ds.orig_shape), "tracker_yaml": a.tracker_yaml, "tracker_type": cfg.tracker_type,
            "tracker_cfg": vars(cfg), "video": a.video, "n_frames_requested": a.n_frames,
            "n_frames_processed": int(len(df)), "n_warmup": int(df.warmup.sum()), "n_measured": int(len(m)),
            "n_skipped_unreadable": n_skipped, "rss_mb_before": rss0 / 2**20, "rss_mb_after": rss1 / 2**20,
            "fps_end_to_end": 1000.0 / summary["total_ms"]["mean"], "git_commit": det_cache.git_commit(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "Do tren PC. Jetson Nano: CHUA DO TREN THIET BI (xem scripts/nano/README.md).",
            "stages_ms": summary}
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{socket.gethostname()}_{Path(a.model).stem}_{cfg.tracker_type}_{'cpu' if a.device == 'cpu' else 'gpu'}{'_half' if a.half else ''}"
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    print(f"[profile] {stem}: {len(m)} frame do (warm-up {a.warmup}), {ds.orig_shape[1]}x{ds.orig_shape[0]} -> imgsz {a.imgsz}")
    for s in stages:
        v = summary[s]
        print(f"  {s:<18} mean {v['mean']:7.2f}  p50 {v['p50']:7.2f}  p95 {v['p95']:7.2f} ms")
    print(f"  end-to-end {meta['fps_end_to_end']:.1f} fps; RSS {rss0 / 2**20:.0f} -> {rss1 / 2**20:.0f} MB; GPU {gpu}")
    print(f"  -> {out_dir / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
