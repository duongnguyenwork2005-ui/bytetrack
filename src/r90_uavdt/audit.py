"""Audit publisher UAVDT MOT labels and select vehicle sequences before scoring."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"F:\LABOTARY_data\goc_tai_ve\UAVDT-R90")
EVIDENCE = ROOT / "handoff/logs/review90_uavdt"
GT_PREFIX = "UAV-benchmark-MOTD_v1.0/GT/"


def read_csv(z: zipfile.ZipFile, name: str) -> tuple[np.ndarray, str]:
    raw = z.read(name)
    arr = np.loadtxt(io.BytesIO(raw), delimiter=",", ndmin=2)
    assert arr.shape[1] == 9 and np.isfinite(arr).all(), name
    return arr, hashlib.sha256(raw).hexdigest()


def turn_rate(gt: np.ndarray) -> tuple[int, int, float]:
    """Fraction of 10-frame track segments changing direction by >45 degrees."""
    yes = count = 0
    for tid in np.unique(gt[:, 1]):
        track = gt[gt[:, 1] == tid]
        p = {int(r[0]): r[2:4] + r[4:6] / 2 for r in track}
        for t in p:
            if t - 10 not in p or t + 10 not in p:
                continue
            a, b = p[t] - p[t - 10], p[t + 10] - p[t]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 5 or nb < 5:
                continue
            count += 1
            yes += np.dot(a, b) / (na * nb) < np.cos(np.pi / 4)
    return int(yes), int(count), float(yes / count) if count else 0.0


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    with zipfile.ZipFile(SOURCE / "M_attr.zip") as az, zipfile.ZipFile(
        SOURCE / "UAV-benchmark-MOTD_v1.0.zip"
    ) as gz:
        train_names = sorted(
            Path(n).name.removesuffix("_attr.txt")
            for n in az.namelist()
            if n.startswith("M_attr/train/") and n.endswith("_attr.txt")
        )
        assert len(train_names) == 30 and len(set(train_names)) == 30
        for seq in train_names:
            gt, gt_sha = read_csv(gz, f"{GT_PREFIX}{seq}_gt.txt")
            whole, whole_sha = read_csv(gz, f"{GT_PREFIX}{seq}_gt_whole.txt")
            ignore, ignore_sha = read_csv(gz, f"{GT_PREFIX}{seq}_gt_ignore.txt") if gz.getinfo(
                f"{GT_PREFIX}{seq}_gt_ignore.txt"
            ).file_size else (np.empty((0, 9)), hashlib.sha256(b"").hexdigest())
            assert (gt[:, 0:2] == np.rint(gt[:, 0:2])).all()
            assert (whole[:, 0:2] == np.rint(whole[:, 0:2])).all()
            pairs = gt[:, 0:2].astype(np.int64)
            assert len(np.unique(pairs, axis=0)) == len(gt), seq
            assert (gt[:, 4:6] > 0).all() and (whole[:, 4:6] > 0).all(), seq
            assert set(np.unique(gt[:, 6])).issubset({0, 1}), seq
            assert set(np.unique(whole[:, 7])).issubset({1, 2, 3, 4}), seq
            assert set(np.unique(whole[:, 8])).issubset({1, 2, 3}), seq
            # DET rows retain the vehicle category and occlusion; MOT rows decide evaluation.
            whole_by_pair = {(int(r[0]), int(r[1])): r for r in whole}
            matched = sum(tuple(map(int, pair)) in whole_by_pair for pair in pairs)
            assert matched == len(gt), (seq, matched, len(gt))
            # Verify same vehicle box in both publisher variants.
            assert all(np.array_equal(r[2:6], whole_by_pair[int(r[0]), int(r[1])][2:6]) for r in gt), seq
            eligible = gt[gt[:, 6] == 1]
            yes, count, rate = turn_rate(eligible)
            attr = [int(x) for x in az.read(f"M_attr/train/{seq}_attr.txt").decode().strip().split(",")]
            assert len(attr) == 10 and all(x in (0, 1) for x in attr)
            out[seq] = {
                "frames": int(max(gt[:, 0].max(), whole[:, 0].max())),
                "mot_rows": int(len(gt)), "eligible_vehicle_rows": int(len(eligible)),
                "vehicle_ids": int(len(np.unique(eligible[:, 1]))),
                "ignore_rows": int(len(ignore)), "whole_rows": int(len(whole)),
                "occluded_rows": int(np.sum(whole[:, 7] != 1)),
                "occlusion_fraction": float(np.mean(whole[:, 7] != 1)),
                "vehicle_classes": {str(c): int(np.sum(whole[:, 8] == c)) for c in (1, 2, 3)},
                "turn_events": yes, "turn_opportunities": count, "turn_fraction": rate,
                "sequence_attributes": attr,
                "sha256": {"gt": gt_sha, "whole": whole_sha, "ignore": ignore_sha},
            }
    eligible = [s for s in train_names if out[s]["frames"] >= 300 and out[s]["vehicle_ids"] >= 10 and out[s]["eligible_vehicle_rows"] >= 1000]
    high = sorted(eligible, key=lambda s: (-out[s]["occlusion_fraction"], s))[:4]
    rest = [s for s in eligible if s not in high]
    turns = sorted(rest, key=lambda s: (-out[s]["turn_fraction"], s))[:2]
    rest = [s for s in rest if s not in turns]
    ordinary = sorted(rest, key=lambda s: (out[s]["occlusion_fraction"], s))[:2]
    selection = {"high_occlusion": high, "direction_change": turns, "ordinary": ordinary,
                 "all": high + turns + ordinary,
                 "rule": "Among train clips with >=300 frames, >=10 IDs and >=1000 eligible GT boxes: top 4 by occlusion fraction, then top 2 remaining by 10-frame turn fraction, then bottom 2 remaining by occlusion fraction; ties by sequence name. Selected before detector/tracker scores."}
    (EVIDENCE / "label_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (EVIDENCE / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"train_sequences": len(train_names), "selection": selection,
                      "selected_rows": {s: out[s] for s in selection["all"]}}, indent=2))


if __name__ == "__main__":
    main()
