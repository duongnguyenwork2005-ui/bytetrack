"""Derive motorcycle-only GT/tracker files from the fixed UTUAV pilot runs."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


TRACKERS = ("utuav_diag_cv", "utuav_diag_ekf", "utuav_diag_ukf",
            "utuav_diag_pf100-s0", "utuav_diag_pf100-s1", "utuav_diag_pf100-s2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--processed-root", type=Path, required=True)
    ap.add_argument("--source-split", default="UTUAV-pilot")
    ap.add_argument("--target-split", default="UTUAV-pilot-moto")
    ap.add_argument("--trackers", nargs="+", default=list(TRACKERS))
    args = ap.parse_args()
    specs = json.loads(args.selection.read_text(encoding="utf-8"))["sequences"]
    source_split = args.processed_root / args.source_split
    target_split = args.processed_root / args.target_split
    summary = {}
    for spec in specs:
        name = spec["name"]
        first, last = spec["first_source_frame"], spec["last_source_frame"]
        rows = []
        for frame, number in enumerate(range(first, last + 1), start=1):
            prefix = "v1" if spec["scene"] == "B" else "v2"
            xml = args.extracted / name / "annotations" / f"{prefix}_{number:06d}.xml"
            root = ET.parse(xml).getroot()
            for obj in root.findall("object"):
                cls = obj.findtext("name", default="")
                if cls not in {"motorbike", "motorbike_stat"}:
                    continue
                source_id = int(obj.findtext("./attributes/attribute[name='track_id']/value"))
                box = obj.find("bndbox")
                x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
                x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
                class_id = 1 if cls == "motorbike" else 8
                rows.append(f"{frame},{source_id+1},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},1,{class_id},1")
        target = target_split / name
        (target / "gt").mkdir(parents=True, exist_ok=True)
        (target / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (target / "seqinfo.ini").write_bytes((source_split / name / "seqinfo.ini").read_bytes())
        summary[name] = {"motorbike_gt_rows": sum(row.split(",")[7] == "1" for row in rows),
                         "static_motorbike_distractors": sum(row.split(",")[7] == "8" for row in rows)}
        for tracker in args.trackers:
            source_dir = args.processed_root / "trackers" / source_split.name / tracker
            mot = pd.read_csv(source_dir / "data" / f"{name}.txt", header=None)
            cls = pd.read_csv(source_dir / "data_cls" / f"{name}.csv")
            if cls.duplicated(["frame", "id"]).any() or mot.duplicated([0, 1]).any():
                raise ValueError(f"duplicate tracker frame/id: {tracker}/{name}")
            picked = mot.merge(cls[cls["cls"] == 3], left_on=[0, 1], right_on=["frame", "id"],
                               how="inner", validate="one_to_one")[list(range(mot.shape[1]))]
            out = args.processed_root / "trackers" / target_split.name / tracker / "data"
            out.mkdir(parents=True, exist_ok=True)
            picked.to_csv(out / f"{name}.txt", header=False, index=False)
    seqmap = args.processed_root / "seqmaps" / f"{target_split.name}.txt"
    seqmap.write_text("name\n" + "\n".join(spec["name"] for spec in specs) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
