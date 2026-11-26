"""
export_detector.py -- Export detector sang ONNX (opset thap, static shape) va kiem tra tap op so voi
TensorRT 8.2 (Jetson Nano goc). Dung de CHON ung vien detector truoc khi dau tu huan luyen.

    python src/export_detector.py --weights models/yolov8n.pt --opset 12 --imgsz 640
    python src/export_detector.py --weights yolo11n.pt --opset 12 --imgsz 640     # ung vien (tu tai neu chua co)
    -> models/export/<stem>_<imgsz>_op<opset>.onnx, results/phase2/export/<stem>_op<opset>.json (op list + ket luan)

Bang op TensorRT 8.2 (scripts/nano/trt82_ops.txt) la danh sach op ONNX ma TRT 8.2 ho tro theo tai lieu
onnx-tensorrt (release 8.2). Op ngoai bang -> canh bao "can plugin / khong ho tro": ung vien bi loai
hoac phai doi cach export (vd end2end=False, dynamic=False). Ket luan cuoi cung PHAI kiem tra bang
trtexec tren Nano (scripts/nano/01_build_engine.sh) - o day chi la kiem tra tinh tren PC.
Can `onnx` (pip install onnx); ultralytics tu cai neu thieu.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TRT82_OPS_FILE = config.PROJECT_ROOT / "scripts" / "nano" / "trt82_ops.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=config.YOLO_DEFAULT_MODEL)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 ONNX (thuong KHONG can: de trtexec --fp16 lo)")
    ap.add_argument("--simplify", action="store_true", help="can onnxslim")
    ap.add_argument("--out-dir", default=str(config.MODELS_DIR / "export"))
    a = ap.parse_args()
    from ultralytics import YOLO
    import onnx

    model = YOLO(a.weights)
    stem = Path(a.weights).stem
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = model.export(format="onnx", opset=a.opset, imgsz=a.imgsz, dynamic=False, simplify=a.simplify,
                        half=a.half, nms=False, batch=1)
    dst = out_dir / f"{stem}_{a.imgsz}_op{a.opset}.onnx"
    Path(path).replace(dst)
    m = onnx.load(str(dst))
    onnx.checker.check_model(m)
    ops = Counter(n.op_type for n in m.graph.node)
    supported = set()
    if TRT82_OPS_FILE.exists():
        supported = {l.strip() for l in TRT82_OPS_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")}
    unsupported = sorted(op for op in ops if supported and op not in supported)
    inp = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outp = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    info = {"weights": str(a.weights), "onnx": str(dst), "size_mb": round(dst.stat().st_size / 2**20, 2),
            "opset": a.opset, "imgsz": a.imgsz, "ir_version": m.ir_version, "inputs": inp, "outputs": outp,
            "n_nodes": len(m.graph.node), "ops": dict(sorted(ops.items())),
            "trt82_table": str(TRT82_OPS_FILE) if supported else None, "ops_not_in_trt82_table": unsupported,
            "end2end": bool(getattr(model.model, "end2end", False)),
            "verdict": ("can kiem tra bang trtexec tren Nano" if not unsupported else
                        f"CO OP NGOAI BANG TRT 8.2: {unsupported} -> can plugin / doi cach export"),
            "note": "Kiem tra TINH tren PC; ket luan cuoi cung phai build engine tren Nano."}
    res_dir = config.RESULTS_DIR / "phase2" / "export"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{stem}_op{a.opset}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
