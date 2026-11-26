"""Replay R61 BDD detections with the two R72 PF motion scales.

Only BDDCOCOVID-val is run here. UA-DETRAC R72 results are reused for the
two-dataset comparison. This script does not modify a tracker or evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import det_cache
import run_tracking

CACHE = ROOT / "results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/cache/yolov8n_coco"
MANIFEST = ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json"
SPLIT = "BDDCOCOVID-val"
SMOKE = ["b1e1a7b8-a7426a97", "b22f385b-5d7e5202", "b23b2649-8349d2a1"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(videos: list[str]) -> dict:
    protocol = json.loads((ROOT / "configs/phase2/r61_bdd/giao_thuc.json").read_text(encoding="utf-8"))
    expected = protocol["tap_du_lieu"]["manifest"]["sha256"]
    if digest(MANIFEST) != expected:
        raise ValueError("BDD manifest SHA256 differs from R61")
    if len(videos) != 200 or len(set(videos)) != 200:
        raise ValueError("BDD val must contain 200 distinct videos")
    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    if meta["weights_sha256"] != protocol["detector"]["yolov8n_coco"]["sha256"]:
        raise ValueError("detector checkpoint mismatch")
    if meta["dataset"] != "bdd100k_coco" or meta["part"] != "val" or meta["conf"] != .05:
        raise ValueError("cache metadata mismatch")
    meta_digest = digest(CACHE / "meta.json")
    frames = 0
    for video in videos:
        marker = json.loads((CACHE / f"{video}.xong.json").read_text(encoding="utf-8"))
        if marker["video"] != video or marker["detector"] != "yolov8n_coco":
            raise ValueError(f"cache marker mismatch: {video}")
        for name, expected_digest in marker["artifact"].items():
            path = CACHE / name
            if digest(path) != expected_digest:
                raise ValueError(f"cache hash mismatch: {name}")
        if marker["artifact"]["meta.json"] != meta_digest:
            raise ValueError(f"meta hash mismatch: {video}")
        gt = ROOT / "data/processed" / SPLIT / video
        if digest(gt / "gt/gt.txt") != marker["gt_sha256"]:
            raise ValueError(f"GT hash mismatch: {video}")
        if digest(gt / "seqinfo.ini") != marker["seqinfo_sha256"]:
            raise ValueError(f"seqinfo hash mismatch: {video}")
        frames += int(marker["n_frames"])
    if frames != 39973:
        raise ValueError(f"BDD frame count {frames} != 39973")
    return {"videos": len(videos), "frames": frames, "cache_meta_sha256": meta_digest,
            "manifest_sha256": expected, "checkpoint_sha256": meta["weights_sha256"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["smoke", "full", "preflight"], required=True)
    args = ap.parse_args()
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    report = preflight(videos)
    out = ROOT / "results/phase2/r74_pf_bdd"
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("preflight", report, flush=True)
    if args.scope == "preflight":
        return 0
    det_cache.CACHE_ROOT = CACHE.parent
    selected = SMOKE if args.scope == "smoke" else videos
    for scale in (100, 200):
        name = f"r74-bdd-{args.scope}-motion{scale}"
        sys.argv = [str(__file__), "--cache-id", "yolov8n_coco", "--split-name", SPLIT,
                    "--tracker-yaml", f"configs/phase2/trackers/r72/motion{scale}.yaml",
                    "--tracker-name", name, "--videos", *selected,
                    "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
                    "--max-det", "300", "--fps", "5", "--track-buffer-sec", "1.2",
                    "--elapsed-mode", "steps", "--seeds", "0", "1", "2", "--skip-existing"]
        rc = run_tracking.main()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
