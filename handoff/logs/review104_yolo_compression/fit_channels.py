"""Recover detection accuracy after physically removing hidden channels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from prepare import ANCHOR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review103_nano_optimization"))
from fit_student import prepare_data


class PrunedTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is None:
            return super().get_model(cfg, weights, verbose)
        # YAML describes the original channel widths. Load the exact sliced modules.
        return deepcopy(weights)


def train_with_distillation(model, train_args):
    teacher = YOLO(str(ANCHOR), task="detect").model.eval().to("cuda:0").float()
    detection_loss = DetectionModel.loss

    def distillation_loss(student, batch, preds=None):
        if not student.training:
            return detection_loss(student, batch, preds)
        if preds is None:
            preds = student.predict(batch["img"])
        supervised, items = detection_loss(student, batch, preds)
        with torch.no_grad():
            _, reference = teacher(batch["img"].float())
            if not (torch.isfinite(reference["scores"]).all() and torch.isfinite(reference["boxes"]).all()):
                raise FloatingPointError("Non-finite predictions from the teacher")
            probabilities = reference["scores"].sigmoid()
            confidence = probabilities.amax(dim=1, keepdim=True)
            weights = confidence * (confidence > .10)
        normalizer = weights.sum().clamp_min(1.)
        classification = F.binary_cross_entropy_with_logits(preds["scores"].float(), probabilities, reduction="none")
        classification = (classification.mean(dim=1, keepdim=True) * weights).sum() / normalizer
        shape = (batch["img"].shape[0], 4, 16, -1)
        teacher_boxes = reference["boxes"].float().reshape(shape).softmax(dim=2)
        student_boxes = preds["boxes"].float().reshape(shape).log_softmax(dim=2)
        boxes = F.kl_div(student_boxes, teacher_boxes, reduction="none").sum(dim=2)
        location = (boxes.mean(dim=1, keepdim=True) * weights).sum() / normalizer
        auxiliary = torch.stack((.25 * location, .5 * classification, location.new_zeros(())))
        return supervised + auxiliary * batch["img"].shape[0], items

    DetectionModel.loss = distillation_loss
    try:
        model.train(**train_args)
    finally:
        DetectionModel.loss = detection_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("fraction", type=int, choices=(25, 50))
    parser.add_argument("--distill", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    name = f"channels{args.fraction}"
    run_name = name + "_distill_fp32" if args.distill else name
    work.joinpath("train_data").mkdir(exist_ok=True)
    config, sources = prepare_data(work / "train_data")
    model = YOLO(str(work / f"{name}_init.pt"), task="detect")
    original = sum(p.numel() for p in model.model.parameters())
    train_args = dict(trainer=PrunedTrainer, data=str(config), epochs=1 if args.distill else 3,
                      imgsz=512 if args.distill else 640, batch=8, workers=0, optimizer="SGD", lr0=.002,
                      warmup_epochs=.5, mosaic=0, mixup=0, amp=False, device=0, seed=104, deterministic=True,
                      patience=0, plots=False, project=str(work / "runs"), name=run_name,
                      exist_ok=False, verbose=False)
    if args.distill:
        train_with_distillation(model, train_args)
    else:
        model.train(**train_args)
    best = work / "runs" / run_name / "weights/best.pt"
    trained = YOLO(str(best), task="detect")
    assert sum(p.numel() for p in trained.model.parameters()) == original
    payload = {"checkpoint": str(best), "parameters": original, "sources": sources,
               "epochs": train_args["epochs"], "imgsz": train_args["imgsz"], "distillation": args.distill}
    (work / f"{run_name}_fit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
