"""Check v22 NMS parity on raw YOLO outputs from 300 real smoke frames (PC FP32)."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


root = Path(__file__).resolve().parents[3]
base = load("r85_real_base", root / "handoff/logs/review83_nano_opt/v21/src/nano_detector.py")
candidate = load("r85_real_candidate", Path(__file__).resolve().parent / "v22/src/nano_detector.py")
infer = base.TorchPtInfer(str(root / "models/yolov8n.pt"), device="cuda", half=False)
image_dir = root / "data/extracted/DETRAC-Images/MVI_39031"
counts = []
boxes = []
all_nms_boxes = []
all_nms_scores = []
offsets = [0]
for frame in range(1, 301):
    image = cv2.imread(str(image_dir / ("img%05d.jpg" % frame)))
    if image is None:
        raise RuntimeError("Missing frame {}".format(frame))
    lb, _, _ = base.letterbox(image, (384, 640))
    raw = infer(base.preprocess(lb))
    n = int((np.max(raw[0, 4:], axis=0) > 0.05).sum())
    p = raw[0].T
    score = p[:, 4:].max(axis=1)
    keep = score > 0.05
    p, score = p[keep], score[keep]
    klass = p[:, 4:].argmax(axis=1).astype(np.float32)
    nms_boxes = base.xywh2xyxy(p[:, :4]) + klass[:, None] * base.MAX_WH
    all_nms_boxes.append(nms_boxes.astype(np.float32))
    all_nms_scores.append(score.astype(np.float32))
    offsets.append(offsets[-1] + len(score))
    a = base.decode_nms(raw)
    b = candidate.decode_nms(raw)
    if not np.array_equal(a, b):
        raise RuntimeError("Mismatch frame {}, candidates {}, outputs {}/{}".format(frame, n, len(a), len(b)))
    counts.append(n)
    boxes.append(len(a))
np.savez_compressed(Path(__file__).resolve().parent / "nms_real_cases_pc_fp32.npz",
                    boxes=np.concatenate(all_nms_boxes), scores=np.concatenate(all_nms_scores),
                    offsets=np.asarray(offsets, dtype=np.int32))
out = {"frames": len(counts), "parity": True, "candidates_p50": float(np.median(counts)),
       "candidates_p90": float(np.percentile(counts, 90)), "candidates_max": int(max(counts)),
       "nms_outputs_p50": float(np.median(boxes)), "nms_outputs_max": int(max(boxes)),
       "fallback_frames": int(sum(n > 256 for n in counts))}
print(json.dumps(out, indent=2))
