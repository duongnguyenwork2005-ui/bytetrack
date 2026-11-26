"""Fine-tune a structurally reduced, weight-initialized YOLO11n student."""
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[3]
ANCHOR = ROOT / "models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt"
MIXED = ROOT / "data/yolo_mixed5_v1"
SEED = 103


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(paths, count):
    return sorted(paths, key=lambda path: hashlib.sha256(f"{SEED}|{path.name}".encode()).hexdigest())[:count]


def make_student(path):
    torch.set_num_threads(4)
    teacher = YOLO(str(ANCHOR), task="detect").model.float().eval()
    student = deepcopy(teacher)
    c3 = student.model[8]
    attention = student.model[10]
    assert type(c3).__name__ == "C3k2" and type(attention).__name__ == "C2PSA"
    assert len(c3.m) == len(attention.m) == 1
    c3.m = torch.nn.ModuleList()
    attention.m = torch.nn.Sequential()
    cv2 = c3.cv2.conv
    cv2.in_channels = 2 * c3.c
    cv2.weight = torch.nn.Parameter(cv2.weight[:, :cv2.in_channels].contiguous())
    for index in (8, 10):
        student.yaml["backbone"][index][1] = 0
    with torch.inference_mode():
        y = student(torch.zeros(1, 3, 384, 640))[0]
        assert y.shape == (1, 9, 5040) and torch.isfinite(y).all()
    torch.save({"model": student, "train_args": {"task": "detect", "imgsz": 640}}, path)
    return {"teacher_sha256": digest(ANCHOR), "teacher_params": sum(p.numel() for p in teacher.parameters()),
            "student_init_sha256": digest(path), "student_params": sum(p.numel() for p in student.parameters())}


def prepare_data(output):
    manifest = ROOT / "handoff/logs/review91_pf_assoc/balanced_pilot/train.txt"
    old = select([Path(line) for line in manifest.read_text().splitlines() if line], 1200)
    assert len(old) == len(set(old)) == 1200 and all(p.parent == MIXED / "images/train" for p in old)
    supplement = ROOT / "tmp/r101_detector_fit/ridesafe_train.txt"
    extra = [Path(line) for line in supplement.read_text().splitlines() if "r101_detector_fit/ridesafe/images/" in line]
    assert len(extra) == 200 and {path.name[:8] for path in extra} == {"20220824", "20221119"}
    with (MIXED / "manifest/anh.csv").open(encoding="utf-8-sig", newline="") as file:
        dev = select([MIXED / row["anh_ra"] for row in csv.DictReader(file) if row["split"] == "dev"], 300)
    assert all(path.is_file() for path in old + extra + dev)
    train, monitor = output / "train.txt", output / "monitor_dev300.txt"
    train.write_text("\n".join(path.as_posix() for path in old + extra) + "\n")
    monitor.write_text("\n".join(path.as_posix() for path in dev) + "\n")
    config = output / "data.yaml"
    config.write_text(yaml.safe_dump({"path": output.as_posix(), "train": train.as_posix(), "val": monitor.as_posix(),
                                      "names": dict(enumerate(("car", "truck", "bus", "motorcycle", "bicycle")))}))
    return config, {"n_mixed5": len(old), "n_ridesafe": len(extra), "n_dev_monitor": len(dev), "train_sha256": digest(train)}


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config, sources = prepare_data(out)
    checkpoint = out / "student_init.pt"
    architecture = make_student(checkpoint)
    (out / "inputs.json").write_text(json.dumps(dict(architecture, **sources, seed=SEED), indent=2))
    model = YOLO(str(checkpoint), task="detect")
    model.train(data=str(config), epochs=3, imgsz=640, batch=8, workers=0, optimizer="SGD", lr0=.002,
                warmup_epochs=.5, mosaic=0, mixup=0, amp=False, device=0, seed=SEED, deterministic=True,
                patience=0, plots=False, project=str(out / "runs"), name="student", exist_ok=False, verbose=False)
    final = out / "runs/student/weights/best.pt"
    trained = YOLO(str(final), task="detect").model
    assert sum(p.numel() for p in trained.parameters()) == architecture["student_params"]
    print(json.dumps({"checkpoint": str(final), "sha256": digest(final), **architecture}, indent=2), flush=True)


if __name__ == "__main__":
    main()
