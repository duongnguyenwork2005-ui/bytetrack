"""Independent round-27 probes. No weights, GPU, training, or real dataset required.

Usage: python REPRO_REVIEW_4937c85.py /path/to/repo --out results.json
Dependencies: numpy, pillow, opencv-python, pyyaml, pandas, pycocotools.
Uses the supplied fixture builder and actual exporter, validator, and evaluator.
No production source is changed. All dataset mutations are in a temporary folder.
Exit 1 means at least one reviewed bug still reproduces; exit 0 means none does.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def outcome(fn, *args):
    try:
        return {"accepted": True, "result": fn(*args)}
    except Exception as exc:
        return {"accepted": False, "error": type(exc).__name__ + ": " + str(exc)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    import bdd_yolo_export as bx
    import eval_detector_5lop as ev
    import train_detector as tr
    import test_detector_5lop as fixtures
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = {"target": "4937c85", "scope": "synthetic CPU probes; no YOLO training"}
    with tempfile.TemporaryDirectory(prefix="codex_r27_") as temp:
        root = Path(temp)
        fixtures.tao_moi_truong(root)
        cfg = fixtures.nap_cfg(root, fixtures.cfg_mau(root, "review"))
        split_path = bx.p_abs(cfg, cfg["file_split"])
        split = bx.tao_split(cfg, split_path)
        bx.xuat_dataset(cfg, split_path)
        out = bx.p_abs(cfg, cfg["dau_ra"])
        data = out / "data.yaml"
        baseline = tr.kiem_dataset(data)
        results["baseline"] = {"export_check": bx.kiem_dataset(out), "train_guard": baseline}
        original_yaml = data.read_bytes()
        y = yaml.safe_load(original_yaml)
        y["train"] = y["val"]
        data.write_text(yaml.safe_dump(y), encoding="utf-8")
        redirected = outcome(tr.kiem_dataset, data)
        results["F1_train_is_dev"] = {
            "bug_reproduced": redirected["accepted"],
            "effective_train": y["train"], "effective_val": y["val"],
            "runner_guard": redirected, "export_checker": outcome(bx.kiem_dataset, out),
        }
        data.write_bytes(original_yaml)

        label = next(p for p in sorted((out / "labels" / "train").glob("*.txt")) if p.read_text().strip())
        original_label = label.read_bytes()
        label.unlink()
        results["F1_missing_label"] = {
            "bug_reproduced": outcome(tr.kiem_dataset, data)["accepted"],
            "runner_guard": outcome(tr.kiem_dataset, data), "export_checker": outcome(bx.kiem_dataset, out),
        }
        label.write_bytes(original_label)

        rows = list(csv.DictReader((out / "manifest" / "anh.csv").open(encoding="utf-8")))
        row = next(r for r in rows if Path(r["anh_ra"]).stem == label.stem and r["split"] == "train")
        image = out / row["anh_ra"]
        original_image = image.read_bytes()
        with Image.open(image) as im:
            im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(image, format="JPEG")
        changed_image_check = outcome(bx.kiem_dataset, out)
        changed_image_train = outcome(tr.kiem_dataset, data)
        results["F2_image_changed"] = {
            "bug_reproduced": changed_image_check["accepted"] and changed_image_train["accepted"],
            "manifest_image_sha": row["sha256_ra"], "actual_image_sha": digest(image),
            "export_checker": changed_image_check,
            "same_manifest_identity": changed_image_train["accepted"] and changed_image_train["result"]["manifest_sha256"] == baseline["manifest_sha256"],
        }
        image.write_bytes(original_image)

        lines = original_label.decode().splitlines()
        lines[0] = lines[0].split()[0] + " 0.500000 0.500000 0.050000 0.050000"
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        changed_label_check = outcome(bx.kiem_dataset, out)
        changed_label_train = outcome(tr.kiem_dataset, data)
        results["F2_label_coordinates_changed"] = {
            "bug_reproduced": label.read_bytes() != original_label and changed_label_check["accepted"] and changed_label_train["accepted"],
            "export_checker": changed_label_check,
            "same_manifest_identity": changed_label_train["accepted"] and changed_label_train["result"]["manifest_sha256"] == baseline["manifest_sha256"],
        }
        label.write_bytes(original_label)

        ignore = next((out / "ignore" / "train").glob("*.txt"))
        image_for_ignore = out / "images" / "train" / (ignore.stem + ".jpg")
        ignore_before = len(ev.doc_gt(image_for_ignore)["ign"])
        ignore.unlink()
        ignore_after = len(ev.doc_gt(image_for_ignore)["ign"])
        ignore_check = outcome(bx.kiem_dataset, out)
        results["F2_ignore_removed"] = {
            "bug_reproduced": ignore_before > 0 and ignore_after == 0 and ignore_check["accepted"],
            "ignore_regions_before": ignore_before, "ignore_regions_after": ignore_after,
            "export_checker": ignore_check,
        }

        # Same content under different train/dev video names is not compared internally.
        train_video, dev_video = split["train_fit"][0], split["dev"][0]
        imgroot = root / "bdd" / "images" / "track" / "train"
        train_source = imgroot / train_video / (train_video + "-0000002.jpg")
        dev_source = imgroot / dev_video / (dev_video + "-0000002.jpg")
        shutil.copyfile(train_source, dev_source)
        cfg_duplicate = dict(cfg, dau_ra="out/duplicate_probe")
        duplicate_export = outcome(bx.xuat_dataset, cfg_duplicate, split_path)
        results["internal_duplicate_probe"] = {
            "accepted": duplicate_export["accepted"],
            "distinct_video_names": train_video != dev_video,
            "identical_source_content": digest(train_source) == digest(dev_source),
        }

    # Ambiguous equal-IoU association: official COCO prefers the later GT at a tie.
    g = {"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 10, 10], [5, 0, 15, 10]], float),
         "ign": np.zeros((0, 4))}
    d = {"cls": np.array([0, 0]), "xyxy": np.array([[0, 0, 15, 10], [0, 0, 10, 10]], float),
         "conf": np.array([0.9, 0.8])}
    custom = ev.danh_gia([g], [d])["theo_lop"]["car"]
    cg = {"images": [{"id": 1, "width": 20, "height": 20}],
          "categories": [{"id": 1, "name": "car"}], "annotations": []}
    for idx, box in enumerate(g["xyxy"], 1):
        x, y, x2, y2 = box
        cg["annotations"].append({"id": idx, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                  "bbox": [x, y, x2-x, y2-y], "area": (x2-x)*(y2-y)})
    predictions = []
    for box, score in zip(d["xyxy"], d["conf"]):
        x, y, x2, y2 = box
        predictions.append({"image_id": 1, "category_id": 1, "bbox": [x, y, x2-x, y2-y], "score": float(score)})
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = cg
        coco.createIndex()
        ce = COCOeval(coco, coco.loadRes(predictions), "bbox")
        ce.params.maxDets = [1, 10, 300]
        ce.evaluate()
        ce.accumulate()
    precision = ce.eval["precision"][:, :, 0, 0, 2]
    coco_ap50 = float(precision[0].mean())
    coco_ap = float(precision.mean())
    results["F3_equal_iou_COCO"] = {
        "bug_reproduced": abs(custom["AP50"] - coco_ap50) > 1e-9,
        "custom_AP50": custom["AP50"], "pycocotools_AP50": coco_ap50,
        "custom_AP50_95": custom["AP50_95"], "pycocotools_AP50_95": coco_ap,
        "custom_operating": custom["van_hanh"],
        "pycocotools_TP_at_050": int((ce.evalImgs[0]["dtMatches"][0] > 0).sum()),
    }
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    flags = {k: v["bug_reproduced"] for k, v in results.items() if isinstance(v, dict) and "bug_reproduced" in v}
    print(json.dumps(flags, indent=2))
    print("Evidence:", args.out)
    return 1 if any(flags.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
