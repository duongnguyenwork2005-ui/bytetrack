"""Build one shared vehicle detector cache for all R90 filters."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
from r90_uavdt.audit import EVIDENCE, SOURCE

CACHE = "r90_uavdt_yolov8n_coco_c005"
WEIGHTS = ROOT / "models/yolov8n.pt"


def main() -> None:
    selected = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
    labels = json.loads((EVIDENCE / "label_audit.json").read_text(encoding="utf-8"))
    cdir = det_cache.CACHE_ROOT / CACHE
    # This script invokes detect_frame on raw decoded images, not build_video_yolo;
    # no region mask is applied even though DetectorSpec's legacy default flag is true.
    spec = det_cache.DetectorSpec(weights=str(WEIGHTS), imgsz=640, conf=.05, iou=.7,
        max_det=3000, device="0" if torch.cuda.is_available() else "cpu", dataset="uavdt-r90",
        part="train8")
    spec.fill_env()
    det_cache.write_meta(cdir, spec, mode="append")
    model = YOLO(str(WEIGHTS))
    with zipfile.ZipFile(SOURCE / "UAV-benchmark-M.zip") as z:
        for seq in selected:
            if (cdir / f"{seq}.npz").exists():
                det_cache.open_cache(CACHE, seq, .05, [2, 5, 7], 300)
                print("REUSE", seq, flush=True)
                continue
            n = labels[seq]["frames"]
            counts = np.zeros(n, dtype=np.int32)
            unreadable = np.zeros(n, dtype=bool)
            truncated = np.zeros(n, dtype=bool)
            chunks = []
            for frame in range(1, n + 1):
                member = f"UAV-benchmark-M/{seq}/img{frame:06d}.jpg"
                image = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (540, 1024):
                    raise ValueError((seq, frame, "image decode/shape"))
                det = det_cache.detect_frame(model, image, spec)
                counts[frame - 1] = len(det)
                truncated[frame - 1] = len(det) >= 3000
                chunks.append(det)
                if frame % 100 == 0 or frame == n:
                    print("CACHE", seq, frame, "/", n, flush=True)
            assert not truncated.any()
            idx = np.arange(1, n + 1, dtype=np.int32)
            arrays = dict(frame_idx=idx, n_per_frame=counts, unreadable=unreadable,
                truncated=truncated, time_s=(idx - 1).astype(np.float64) / 30,
                data=np.concatenate(chunks).astype(np.float32),
                orig_shape=np.array([540, 1024], dtype=np.int32))
            result = det_cache.save_video(cdir, seq, arrays,
                extra={"source_archive": str(SOURCE / "UAV-benchmark-M.zip"),
                       "source_member_prefix": f"UAV-benchmark-M/{seq}/img",
                       "gt_sha256": labels[seq]["sha256"]["gt"]}, spec=spec)
            print("SAVED", seq, result, flush=True)


if __name__ == "__main__":
    main()
