"""Prepare official SVMOT test GT and two common detection caches without images."""

from __future__ import annotations

import configparser
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import det_cache  # noqa: E402


BASE = Path(__file__).resolve().parent
PROTOCOL = json.loads((BASE / "svmot_protocol.json").read_text(encoding="utf-8"))
ARCHIVE = Path(PROTOCOL["archive"])
EXPECTED_SHA = "aae153a21a46f9fe37068ee006a65abd58894efaabc4390404eadcc08be8bd7a"
CACHE_IDS = {"release": "svmot_release_v1", "visibility_drop": "svmot_occ05_v1"}
SPLIT = "SVMOT-test"


def arrays_from_rows(frame_count: int, fps: float, width: int, height: int,
                     det: pd.DataFrame, visibility: dict[tuple, float], drop_low_visibility: bool):
    n_per = np.zeros(frame_count, dtype=np.int32)
    chunks = []
    dropped = 0
    for frame, group in det.groupby(0, sort=True):
        if not 1 <= frame <= frame_count:
            raise ValueError(f"invalid frame {frame}")
        rows = []
        for row in group.itertuples(index=False, name=None):
            key = (row[0], row[2], row[3], row[4], row[5])
            vis = visibility[key]
            if drop_low_visibility and vis < 0.5:
                dropped += 1
                continue
            x, y, w, h, conf = float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
            rows.append([x, y, x + w, y + h, conf, 0.0])
        if rows:
            arr = np.array(rows, dtype=np.float32)
            arr = arr[np.argsort(-arr[:, 4], kind="stable")]
            n_per[int(frame) - 1] = len(arr)
            chunks.append(arr)
    data = np.concatenate(chunks) if chunks else np.zeros((0, 6), dtype=np.float32)
    return {"frame_idx": np.arange(1, frame_count + 1, dtype=np.int32),
            "n_per_frame": n_per, "unreadable": np.zeros(frame_count, bool),
            "truncated": np.zeros(frame_count, bool),
            "time_s": np.arange(frame_count, dtype=np.float64) / fps,
            "data": data, "orig_shape": np.array([height, width], dtype=np.int32)}, dropped


def main() -> None:
    recorded_hash = (BASE / "svmot_sha256.txt").read_text(encoding="ascii").strip().lower()
    if recorded_hash != EXPECTED_SHA:
        raise RuntimeError("SVMOT SHA256 not verified against the Zenodo release")
    specs = {}
    for condition in CACHE_IDS:
        specs[condition] = det_cache.DetectorSpec(
            backend="svmot-released-boxes" if condition == "release" else "svmot-visibility-drop",
            weights="published SVMOT det.txt; not a YOLO checkpoint", dataset="svmot", part="test",
            conf=0.0, classes=[0], max_det=3000, device="cpu",
            versions={"archive_sha256": EXPECTED_SHA, "condition": condition,
                      "visibility_policy": "retain all" if condition == "release" else "drop GT visibility <0.5"},
        ).fill_env()
        det_cache.write_meta(det_cache.CACHE_ROOT / CACHE_IDS[condition], specs[condition], mode="create")

    summary = {}
    processed = ROOT / "data" / "processed" / SPLIT
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in PROTOCOL["official_test_sequences"]:
            prefix = f"data/{name}/"
            ini_bytes = archive.read(prefix + "seqinfo.ini")
            ini = configparser.ConfigParser()
            ini.read_string(ini_bytes.decode("utf-8"))
            frame_count = int(ini["Sequence"]["seqlength"])
            fps = float(ini["Sequence"]["framerate"])
            width = int(ini["Sequence"]["imwidth"])
            height = int(ini["Sequence"]["imheight"])
            gt = pd.read_csv(io.BytesIO(archive.read(prefix + "gt/gt.txt")), header=None)
            det = pd.read_csv(io.BytesIO(archive.read(prefix + "det/det.txt")), header=None)
            if gt.shape[1] != 9 or det.shape[1] != 10:
                raise ValueError(f"unexpected GT/det columns: {name}")
            key_cols = [0, 2, 3, 4, 5]
            if gt.duplicated(key_cols).any() or det.duplicated(key_cols).any():
                raise ValueError(f"duplicate exact box key: {name}")
            gt_keys = set(map(tuple, gt[key_cols].itertuples(index=False, name=None)))
            det_keys = set(map(tuple, det[key_cols].itertuples(index=False, name=None)))
            if gt_keys != det_keys:
                raise ValueError(f"released detections differ from GT box geometry: {name}")
            visibility = {key: float(v) for key, v in zip(
                gt[key_cols].itertuples(index=False, name=None), gt[8])}

            seq_dir = processed / name
            (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
            pooled = gt.copy()
            pooled[7] = 1  # TrackEval's class 1 is the project's pooled vehicle convention.
            pooled.to_csv(seq_dir / "gt" / "gt.txt", index=False, header=False)
            (seq_dir / "seqinfo.ini").write_bytes(ini_bytes)

            item = {"frames": frame_count, "fps": fps, "gt_boxes": len(gt),
                    "visibility_lt_05": int((gt[8] < 0.5).sum()), "release_det": len(det)}
            for condition, cache_id in CACHE_IDS.items():
                arrays, dropped = arrays_from_rows(frame_count, fps, width, height, det, visibility,
                                                  drop_low_visibility=(condition == "visibility_drop"))
                det_cache.save_video(det_cache.CACHE_ROOT / cache_id, name, arrays,
                                     extra={"source": prefix + "det/det.txt", "policy": condition,
                                            "archive_sha256": EXPECTED_SHA, "dropped_visibility_lt_05": dropped},
                                     spec=specs[condition])
                item[condition + "_cached"] = int(len(arrays["data"]))
                item[condition + "_dropped"] = dropped
            if item["visibility_drop_dropped"] != item["visibility_lt_05"]:
                raise AssertionError(f"drop count mismatch: {name}")
            summary[name] = item
    seqmap = ROOT / "data" / "processed" / "seqmaps" / f"{SPLIT}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(PROTOCOL["official_test_sequences"]) + "\n", encoding="utf-8")
    print(json.dumps({"archive_sha256": EXPECTED_SHA, "split": SPLIT,
                      "cache_ids": CACHE_IDS, "sequences": summary}, indent=2))


if __name__ == "__main__":
    main()
