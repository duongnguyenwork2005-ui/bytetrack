"""Independent CPU checks for R28-F1, no model spy, checkpoint, GPU or training.

The clean dry-run control must reach the missing-checkpoint boundary. Negative
cases must stop earlier with a dataset/configuration error and create no output.
All inputs are disposable synthetic fixtures. Production source is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    import bdd_yolo_export as bx
    import eval_detector_5lop as ev
    import test_detector_5lop as fx
    import train_detector as tr
    checks = []
    with tempfile.TemporaryDirectory(prefix="codex_r29_cli_") as td:
        root = Path(td)
        fx.tao_moi_truong(root)
        c = fx.nap_cfg(root, fx.cfg_mau(root, "r29_cli"))
        sp = bx.p_abs(c, c["file_split"])
        bx.tao_split(c, sp)
        bx.xuat_dataset(c, sp)
        out = bx.p_abs(c, c["dau_ra"])
        data = out / "data.yaml"
        train_cfg = yaml.safe_load((repo / "configs/phase2/detector/train_bdd5.yaml").read_text())
        train_cfg["model"]["yolov8n"] = {"weights": str(root / "intentionally_absent.pt"), "sha256": "0" * 64}
        conf = root / "train.yaml"

        def run(name, cache=False):
            train_cfg["huan_luyen"]["cache"] = cache
            conf.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")
            project, result = root / (name + "_runs"), root / (name + "_results")
            cmd = [sys.executable, str(repo / "src/train_detector.py"), "--config", str(conf),
                   "--model", "yolov8n", "--data", str(data), "--dry-run", "--device", "cpu",
                   "--project", str(project), "--ket-qua-dir", str(result)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            return {"exit": p.returncode, "stderr": p.stderr, "stdout": p.stdout,
                    "output_absent": not project.exists() and not result.exists()}

        clean = tr.kiem_dataset(data)
        r = run("clean")
        checks.append({"name": "clean reaches checkpoint boundary", **r,
                       "pass": clean["loi"] == 0 and r["exit"] == 2 and
                               "thieu checkpoint khoi tao" in r["stderr"] and r["output_absent"]})

        np.save(out / "images/train/unrelated_without_image.npy", np.zeros((2, 2, 3), dtype=np.uint8))
        np.save(root / "unrelated_elsewhere.npy", np.zeros((2, 2, 3), dtype=np.uint8))
        checks.append({"name": "unrelated npy allowed", "pass": tr.kiem_dataset(data)["loi"] == 0})

        for split in ("train", "dev"):
            image = next((out / "images" / split).glob("*.jpg"))
            npy = image.with_suffix(".npy")
            np.save(npy, np.zeros_like(cv2.imread(str(image))))
            before = hashlib.sha256(npy.read_bytes()).hexdigest()
            r = run("stale_" + split)
            try:
                ev.chuan_bi_danh_gia(data, "val")
                eval_blocked = False
            except bx.LoiDuLieu as exc:
                eval_blocked = ".npy" in str(exc)
            preserved = npy.exists() and hashlib.sha256(npy.read_bytes()).hexdigest() == before
            checks.append({"name": "stale npy " + split, **r, "evaluator_blocked": eval_blocked,
                           "npy_preserved": preserved,
                           "pass": r["exit"] == 2 and r["output_absent"] and eval_blocked and preserved
                                   and ("images/" + split) in r["stderr"] and ".npy" in r["stderr"]
                                   and "thieu checkpoint khoi tao" not in r["stderr"]})
            npy.unlink()
        for mode in (True, "ram", "disk"):
            r = run("cache_" + str(mode), mode)
            checks.append({"name": "cache configuration " + str(mode), **r,
                           "pass": r["exit"] == 2 and r["output_absent"] and "huan_luyen.cache" in r["stderr"]
                                   and "thieu checkpoint khoi tao" not in r["stderr"]})
    result = {"target": "bcb92f8", "scope": "real CLI, synthetic CPU, no model spy, no training",
              "checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks)}
    result["exit_code"] = int(result["passed"] != result["total"])
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("target", "passed", "total", "exit_code")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
