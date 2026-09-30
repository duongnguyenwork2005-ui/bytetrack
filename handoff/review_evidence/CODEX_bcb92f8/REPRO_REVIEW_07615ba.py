"""Independent review of the round-28 dataset guard.

Usage: python REPRO_REVIEW_07615ba.py /path/to/repo --out report.json
Dependencies: numpy, pillow, pyyaml, opencv-python, ultralytics (tested 8.4.141).

All input mutations use a temporary synthetic exported dataset. No research data,
weights, GPU, Nano, or training is used. Export/validation/YOLODataset/load_image
are REAL. SimpleNamespace only holds the two real datasets for the callback.
The separate entry-point checks explicitly MOCK the YOLO model boundary using
the repository's spy; their dummy checkpoint is not loaded as a model.

Exit 1: stale-image-cache bug reproduced. Exit 0: bug not reproduced and controls
pass. Other exceptions: environment/test setup failed; not a passing result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    import bdd_yolo_export as bx
    import train_detector as tr
    import test_detector_5lop as fx
    import ultralytics
    from ultralytics.data.dataset import YOLODataset

    rec = {
        "target": "07615ba",
        "scope": "synthetic CPU; real exporter, guards and Ultralytics datasets; no training",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
                        "numpy": np.__version__, "torch": importlib.metadata.version("torch")},
        "source_sha256": {p: sha(repo / "src" / p) for p in
                          ("bdd_yolo_export.py", "train_detector.py", "eval_detector_5lop.py")},
    }
    with tempfile.TemporaryDirectory(prefix="codex_r28_") as tmp:
        root = Path(tmp)
        fx.tao_moi_truong(root)
        cfg = fx.nap_cfg(root, fx.cfg_mau(root, "independent_review"))
        split_file = bx.p_abs(cfg, cfg["file_split"])
        bx.tao_split(cfg, split_file)
        bx.xuat_dataset(cfg, split_file)
        out = bx.p_abs(cfg, cfg["dau_ra"])
        data_yaml = out / "data.yaml"
        with (out / "manifest" / "anh.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        expected_labels = tr.nhan_mong_doi(data_yaml)
        names = dict(enumerate(["car", "truck", "bus", "motorcycle", "bicycle"]))

        def datasets():
            return {s: YOLODataset(img_path=str(out / "images" / s),
                                   data={"names": names, "nc": 5, "channels": 3},
                                   task="detect", augment=False, imgsz=160, cache=False)
                    for s in ("train", "dev")}

        def check_loaded(ds):
            holder = SimpleNamespace(train_loader=SimpleNamespace(dataset=ds["train"]),
                                     test_loader=SimpleNamespace(dataset=ds["dev"]))
            return tr.kiem_dau_vao_da_nap(holder, expected_labels)

        def image_loaded(ds, row):
            d = ds[row["split"]]
            p = out / row["anh_ra"]
            idx = next(i for i, name in enumerate(d.im_files) if Path(name).resolve() == p.resolve())
            # This method is used by YOLODataset.__getitem__ before augmentation.
            im = d.get_image_and_label(idx)["img"]
            return im, {"cache_setting": d.cache, "image_count": d.ni,
                        "loaded_shape": list(im.shape), "npy_path": str(d.npy_files[idx].relative_to(out))}

        rec["clean_contract"] = tr.kiem_dataset(data_yaml)
        clean_ds = datasets()
        rec["clean_callback"] = check_loaded(clean_ds)
        rec["stale_npy_cases"] = []
        for split in ("train", "dev"):
            row = next(r for r in rows if r["split"] == split and int(r["am_tinh"]) == 0)
            p = out / row["anh_ra"]
            original = cv2.imread(str(p))
            assert original is not None and original.mean() > 0
            before, _ = image_loaded(clean_ds, row)
            assert np.array_equal(before, original), "Clean real loader must match source JPEG."
            before_hash = sha(p)
            npy = p.with_suffix(".npy")
            np.save(npy, np.zeros_like(original))
            contract_error, callback_error = None, None
            try:
                contract = tr.kiem_dataset(data_yaml)
            except (tr.LoiCauHinh, bx.LoiDuLieu) as e:
                contract, contract_error = None, str(e)
            loaded_check, loaded, detail = None, None, {}
            if contract is not None:
                ds = datasets()
                try:
                    loaded_check = check_loaded(ds)
                except tr.LoiDauVaoDaNap as e:
                    callback_error = str(e)
                if callback_error is None:
                    loaded, detail = image_loaded(ds, row)
            wrong_pixels = loaded is not None and not np.array_equal(loaded, original)
            only_cache_pixels = loaded is not None and np.array_equal(loaded, np.zeros_like(original))
            contract_accepted = contract is not None and contract["loi"] == 0
            callback_accepted = loaded_check is not None and all(
                v["nhan_khac"] == v["thieu"] == v["thua"] == 0 for v in loaded_check.values())
            item = {"split": split, "image": row["anh_ra"], **detail,
                    "contract_accepted": contract_accepted, "contract_error": contract_error,
                    "callback_accepted": callback_accepted, "callback_error": callback_error,
                    "callback_summary": loaded_check,
                    "source_sha256": before_hash, "source_hash_unchanged": sha(p) == before_hash,
                    "declared_source_sha256": row["sha256_ra"],
                    "source_mean_pixel": float(original.mean()),
                    "loaded_mean_pixel": float(loaded.mean()) if loaded is not None else None,
                    "wrong_pixels_loaded": wrong_pixels, "loaded_exactly_stale_npy": only_cache_pixels,
                    "bug_reproduced": contract_accepted and callback_accepted and wrong_pixels and only_cache_pixels}
            npy.unlink(missing_ok=True)
            restored_ds = datasets()
            restored, _ = image_loaded(restored_ds, row)
            item["delete_npy_restores_exact_source_pixels"] = bool(np.array_equal(restored, original))
            assert item["delete_npy_restores_exact_source_pixels"]
            assert item["source_hash_unchanged"] and before_hash == row["sha256_ra"]
            rec["stale_npy_cases"].append(item)

        # Runner entry-point controls: reuse the project's explicit model spy,
        # with a dummy file and matching hash. No real checkpoint is needed.
        dummy = root / "dummy_model_boundary_only.pt"
        dummy.write_bytes(b"Codex model-boundary spy; NOT a real checkpoint.\n")
        tc = yaml.safe_load((repo / "configs/phase2/detector/train_bdd5.yaml").read_text(encoding="utf-8"))
        tc["model"]["yolov8n"] = {"weights": str(dummy), "sha256": sha(dummy)}
        tc_path = root / "spy_train_cfg.yaml"
        tc_path.write_text(yaml.safe_dump(tc), encoding="utf-8")
        common = ["--config", str(tc_path), "--model", "yolov8n", "--che-do", "smoke",
                  "--data", str(data_yaml), "--device", "cpu", "--project", str(root / "spy_runs"),
                  "--ket-qua-dir", str(root / "spy_results")]

        def runner(name):
            rc, stdout, stderr, calls = fx._chay_runner_spy(common + ["--ten", name])
            return {"exit": rc, "calls": calls,
                    "model_called": any(c[0] == "khoi_tao" for c in calls),
                    "train_called": any(c[0] == "train" for c in calls),
                    "run_dir_created": (root / "spy_runs" / name).exists(),
                    "result_created": (root / "spy_results" / (name + ".json")).exists(),
                    "stderr": stderr[-1200:], "stdout_tail": stdout[-900:]}

        entry = {"valid": runner("valid")}
        original_yaml = data_yaml.read_bytes()
        data_yaml.write_bytes(original_yaml.replace(b"train: images/train", b"train: images/dev"))
        try:
            entry["train_redirected_to_dev"] = runner("train_redirected_to_dev")
        finally:
            data_yaml.write_bytes(original_yaml)
        for empty in (False, True):
            r = next(r for r in rows if r["split"] == "train" and bool(int(r["am_tinh"])) == empty)
            label = out / r["nhan_ra"]
            b = label.read_bytes()
            label.unlink()
            name = "missing_empty_label" if empty else "missing_positive_label"
            try:
                entry[name] = runner(name)
            finally:
                label.write_bytes(b)
        assert entry["valid"]["exit"] == 3 and entry["valid"]["train_called"]
        for k, v in entry.items():
            if k != "valid":
                assert v["exit"] == 2 and not v["model_called"] and not v["train_called"]
                assert not v["run_dir_created"] and not v["result_created"]
        rec["entry_point_controls_model_boundary_mocked"] = entry

    rec["bug_count"] = sum(c["bug_reproduced"] for c in rec["stale_npy_cases"])
    rec["exit_code"] = 1 if rec["bug_count"] else 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": rec["target"], "stale_npy_bug_count": rec["bug_count"],
                      "entry_point_controls": "PASS (model boundary mocked)", "out": str(args.out)},
                     ensure_ascii=False))
    return rec["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
