"""Read-only SVMOT ZIP inventory: split candidates, labels, occlusion, detections."""

from __future__ import annotations

import collections
import configparser
import io
import json
import math
import zipfile
from pathlib import Path

import pandas as pd


ARCHIVE = Path(r"D:\LABOTARY\svmot_data_v1.0.zip")


def turning_tracks(gt: pd.DataFrame) -> int:
    count = 0
    for _, group in gt.groupby(1):
        p = {int(row[0]): (float(row[2]) + float(row[4]) / 2,
                           float(row[3]) + float(row[5]) / 2)
             for row in group.itertuples(index=False, name=None)}
        found = False
        for frame, center in p.items():
            if frame - 5 not in p or frame + 5 not in p:
                continue
            left, right = p[frame - 5], p[frame + 5]
            ax, ay = center[0] - left[0], center[1] - left[1]
            bx, by = right[0] - center[0], right[1] - center[1]
            la, lb = math.hypot(ax, ay), math.hypot(bx, by)
            if la < 5 or lb < 5:
                continue
            cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
            if math.degrees(math.acos(cosine)) >= 30:
                found = True
                break
        count += int(found)
    return count


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        names = sorted({p.split("/")[1] for p in archive.namelist()
                        if p.startswith("data/seq_")})
        classes = archive.read("classes.txt").decode("utf-8").splitlines()
        output = {"archive": str(ARCHIVE), "archive_bytes": ARCHIVE.stat().st_size,
                  "entries": len(infos), "classes_1_based": classes, "sequences": {}}
        for name in names:
            prefix = f"data/{name}/"
            config = configparser.ConfigParser()
            config.read_string(archive.read(prefix + "seqinfo.ini").decode("utf-8"))
            gt = pd.read_csv(io.BytesIO(archive.read(prefix + "gt/gt.txt")), header=None)
            det = pd.read_csv(io.BytesIO(archive.read(prefix + "det/det.txt")), header=None)
            occ = json.loads(archive.read(prefix + "occlusion.json"))
            occluded = stopped = pairwise = 0
            for frame in occ:
                for obj in frame.get("objects", []):
                    stopped += int(bool(obj.get("stopped")))
                    pairwise += len(obj.get("occlusions", []))
                    occluded += int(bool(obj.get("occlusions")))
            img_infos = [info for info in infos if info.filename.startswith(prefix + "img/")
                         and info.filename.lower().endswith(".jpg")]
            gt_boxes = set(zip(gt[0], gt[2], gt[3], gt[4], gt[5]))
            det_boxes = set(zip(det[0], det[2], det[3], det[4], det[5]))
            visibility = pd.to_numeric(gt[8], errors="coerce")
            output["sequences"][name] = {
                "frames": int(config["Sequence"]["seqlength"]),
                "fps": int(config["Sequence"]["framerate"]),
                "size": [int(config["Sequence"]["imwidth"]), int(config["Sequence"]["imheight"])],
                "images": len(img_infos),
                "image_uncompressed_bytes": sum(info.file_size for info in img_infos),
                "gt_boxes": len(gt), "ids": int(gt[1].nunique()),
                "gt_class_counts": {str(int(k)): int(v) for k, v in gt[7].value_counts().items()},
                "visibility_lt_05": int((visibility < 0.5).sum()),
                "visibility_lt_08": int((visibility < 0.8).sum()),
                "occluded_object_frames": occluded,
                "pairwise_occlusions": pairwise,
                "stopped_object_frames": stopped,
                "turning_tracks_rule_5f_30deg": turning_tracks(gt),
                "det_boxes": len(det),
                "det_exact_box_overlap_gt": len(gt_boxes & det_boxes),
                "det_conf_median": float(det[6].median()),
            }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
