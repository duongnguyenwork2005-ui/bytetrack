"""Rank contiguous UTUAV validation windows using GT motion, before tracker runs."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_archive", type=Path)
    ap.add_argument("--ranges-json", type=Path, required=True)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--step", type=int, default=30)
    args = ap.parse_args()
    manifest = json.loads(args.ranges_json.read_text(encoding="utf-8"))
    ranges = {}
    for group, info in manifest["frame_ranges"].items():
        if group.endswith("/validation"):
            scene = "B" if "/B_" in group else "C"
            ranges[scene] = (info["min"], info["max"])
    if set(ranges) != {"B", "C"}:
        raise ValueError(f"missing B/C validation ranges: {ranges}")

    # scene -> track_id -> frame -> (center x, center y, class)
    tracks = {scene: collections.defaultdict(dict) for scene in ranges}
    xml_count = collections.Counter()
    class_count = collections.Counter()
    with tarfile.open(args.labels_archive, "r|gz") as archive:
        for member in archive:
            match = re.fullmatch(r"([BC])_6class_PascalVOC/Annotations/v[12]_(\d+)\.xml", member.name)
            if not match:
                continue
            scene, number = match.group(1), int(match.group(2))
            low, high = ranges[scene]
            if not low <= number <= high:
                continue
            xml_count[scene] += 1
            root = ET.parse(archive.extractfile(member)).getroot()
            for obj in root.findall("object"):
                cls = obj.findtext("name", default="")
                class_count[f"{scene}:{cls}"] += 1
                if cls not in {"motorbike", "car", "bus"}:
                    continue
                id_text = obj.findtext("./attributes/attribute[name='track_id']/value")
                box = obj.find("bndbox")
                if id_text is None or box is None:
                    continue
                cx = (float(box.findtext("xmin")) + float(box.findtext("xmax"))) / 2
                cy = (float(box.findtext("ymin")) + float(box.findtext("ymax"))) / 2
                tracks[scene][int(id_text)][number] = (cx, cy, cls)

    windows = []
    for scene, (low, high) in ranges.items():
        for start in range(low, high - args.length + 2, args.step):
            end = start + args.length - 1
            turn_ids = set()
            motorcycle_turn_ids = set()
            gap_ids = set()
            motorcycle_gap_ids = set()
            motorcycle_observations = 0
            for track_id, positions in tracks[scene].items():
                frames = sorted(f for f in positions if start <= f <= end)
                if not frames:
                    continue
                cls = positions[frames[0]][2]
                if cls == "motorbike":
                    motorcycle_observations += len(frames)
                if any(2 <= b - a <= 10 for a, b in zip(frames, frames[1:])):
                    gap_ids.add(track_id)
                    if cls == "motorbike":
                        motorcycle_gap_ids.add(track_id)
                for f in frames:
                    if f - 5 not in positions or f + 5 not in positions:
                        continue
                    p0, p1, p2 = positions[f - 5], positions[f], positions[f + 5]
                    ax, ay = p1[0] - p0[0], p1[1] - p0[1]
                    bx, by = p2[0] - p1[0], p2[1] - p1[1]
                    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
                    if la < 5 or lb < 5:
                        continue
                    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
                    if math.degrees(math.acos(cosine)) >= 30:
                        turn_ids.add(track_id)
                        if cls == "motorbike":
                            motorcycle_turn_ids.add(track_id)
            windows.append({"scene": scene, "start": start, "end": end,
                            "turn_tracks": len(turn_ids), "motorcycle_turn_tracks": len(motorcycle_turn_ids),
                            "gap_tracks_proxy": len(gap_ids), "motorcycle_gap_tracks_proxy": len(motorcycle_gap_ids),
                            "motorcycle_observations": motorcycle_observations})
    score = lambda w: (w["motorcycle_turn_tracks"] * 3 + w["turn_tracks"]
                       + w["motorcycle_gap_tracks_proxy"] * 2)
    print(json.dumps({"ranges": ranges, "xml_count": xml_count, "class_count": class_count,
                      "length": args.length, "step": args.step,
                      "top_turn_windows": sorted(windows, key=score, reverse=True)[:12],
                      "straight_controls": sorted(
                          (w for w in windows if w["motorcycle_observations"] >= 100),
                          key=lambda w: (w["motorcycle_turn_tracks"], w["turn_tracks"],
                                         -w["motorcycle_observations"]))[:8]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
