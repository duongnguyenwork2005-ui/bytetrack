"""R63 - xuat ONNX tinh (batch 1) tu .pt voi kich thuoc dau vao (H, W), CUNG thiet lap ONNX cua goi v16
(src/export_detector.py: opset 12, dynamic=False, simplify=False, half=False, nms=False, batch=1), va ghi manifest.

Chay tren PC (khoaluan-mot):
    python xuat_onnx.py --weights models/yolov8n.pt --hw 384x640 --ra <thu_muc_moi>
    python xuat_onnx.py --weights models/yolov8n.pt --hw 640x640 --ra <thu_muc_moi> \
        --so-voi models/export/yolov8n_640_op12.onnx          # kiem tai lap ONNX v16

Khong ghi vao models/: .pt duoc sao vao <ra>/ truoc khi export (ultralytics ghi ONNX canh .pt).
`--so-voi`: so sha256 file; neu khac, so DO THI bo qua metadata (ten/op/thu tu node, attribute, input/output
shape, initializer tung byte). Ket luan 'tuong_duong_do_thi' chi khi moi muc trung.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tom_tat_do_thi(m) -> dict:
    import onnx
    from onnx import numpy_helper
    nodes = [(n.op_type, list(n.input), list(n.output),
              sorted((a.name, str(onnx.helper.get_attribute_value(a))) for a in n.attribute)) for n in m.graph.node]
    init = {i.name: hashlib.sha256(numpy_helper.to_array(i).tobytes()).hexdigest() for i in m.graph.initializer}
    io = lambda vs: [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in vs]  # noqa: E731
    return {"nodes": nodes, "init": init, "inputs": io(m.graph.input), "outputs": io(m.graph.output),
            "opset": [(o.domain, o.version) for o in m.opset_import], "ir_version": m.ir_version,
            "metadata": {p.key: p.value for p in m.metadata_props}}


def so_sanh(a: Path, b: Path) -> dict:
    import onnx
    ta, tb = tom_tat_do_thi(onnx.load(str(a))), tom_tat_do_thi(onnx.load(str(b)))
    kq = {"sha256_trung": sha256(a) == sha256(b)}
    for k in ("nodes", "init", "inputs", "outputs", "opset", "ir_version"):
        kq[f"{k}_trung"] = ta[k] == tb[k]
    kq["so_node"] = [len(ta["nodes"]), len(tb["nodes"])]
    kq["so_initializer"] = [len(ta["init"]), len(tb["init"])]
    kq["metadata_khac"] = {k: [ta["metadata"].get(k), tb["metadata"].get(k)]
                           for k in sorted(set(ta["metadata"]) | set(tb["metadata"]))
                           if ta["metadata"].get(k) != tb["metadata"].get(k)}
    kq["tuong_duong_do_thi"] = all(kq[f"{k}_trung"] for k in ("nodes", "init", "inputs", "outputs", "opset", "ir_version"))
    return kq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hw", required=True, help="HxW, moi canh boi so 32, vd 384x640")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--ra", required=True, help="thu muc MOI (khong ton tai hoac rong)")
    ap.add_argument("--so-voi", default=None, help="ONNX tham chieu de kiem tai lap")
    a = ap.parse_args(argv)
    h, w = (int(x) for x in a.hw.lower().split("x"))
    if h % 32 or w % 32:
        print(f"[LOI] {a.hw}: moi canh phai la boi so 32 (stride YOLOv8)", file=sys.stderr)
        return 2
    ra = Path(a.ra)
    if ra.exists() and any(ra.iterdir()):
        print(f"[LOI] {ra} khong rong", file=sys.stderr)
        return 2
    ra.mkdir(parents=True, exist_ok=True)
    src = Path(a.weights)
    pt = ra / src.name
    shutil.copy2(src, pt)

    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO
    t0 = time.time()
    out = Path(YOLO(str(pt)).export(format="onnx", opset=a.opset, imgsz=(h, w), dynamic=False, simplify=False,
                                    half=False, nms=False, batch=1, device="cpu"))
    dst = ra / f"{src.stem}_{h}x{w}_op{a.opset}.onnx"
    out.replace(dst)
    pt.unlink()                                             # chi giu ONNX + manifest
    m = onnx.load(str(dst))
    onnx.checker.check_model(m)
    tt = tom_tat_do_thi(m)
    man = {"weights": str(src), "weights_sha256": sha256(src), "onnx": str(dst), "onnx_sha256": sha256(dst),
           "onnx_bytes": dst.stat().st_size, "hw": [h, w], "opset": a.opset, "inputs": tt["inputs"],
           "outputs": tt["outputs"], "n_nodes": len(tt["nodes"]), "metadata": tt["metadata"],
           "thiet_lap": {"dynamic": False, "simplify": False, "half": False, "nms": False, "batch": 1, "device": "cpu"},
           "ultralytics": ultralytics.__version__, "torch": torch.__version__, "onnx_version": onnx.__version__,
           "python": platform.python_version(), "thoi_gian_s": round(time.time() - t0, 2),
           "luu_y": "Dung luong ONNX nho hon KHONG phai bang chung tang FPS; phai build engine + do tren Nano."}
    if a.so_voi:
        man["so_voi"] = {"tham_chieu": a.so_voi, "tham_chieu_sha256": sha256(Path(a.so_voi)),
                         **so_sanh(dst, Path(a.so_voi))}
    (ra / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("onnx", "onnx_sha256", "inputs", "outputs", "n_nodes")}, indent=1))
    if a.so_voi:
        print(json.dumps(man["so_voi"], indent=1, ensure_ascii=False))
        return 0 if man["so_voi"]["tuong_duong_do_thi"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
