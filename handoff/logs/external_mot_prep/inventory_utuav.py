"""Read-only inventory of one UTUAV tar.gz archive; prints a compact JSON summary."""

from __future__ import annotations

import argparse
import collections
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--xml", action="store_true", help="parse PascalVOC annotations")
    args = parser.parse_args()
    groups = collections.Counter()
    suffixes = collections.Counter()
    classes = collections.Counter()
    occluded = collections.Counter()
    bad_xml = []
    example_paths = []
    frame_numbers = collections.defaultdict(list)
    files = total_bytes = objects = with_id = 0
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            files += 1
            total_bytes += member.size
            path = Path(member.name)
            groups["/".join(path.parts[:4])] += 1
            suffixes[path.suffix.lower()] += 1
            if path.suffix.lower() == ".jpg":
                match = re.search(r"(\d+)$", path.stem)
                if match:
                    frame_numbers["/".join(path.parts[:3])].append(int(match.group(1)))
            if len(example_paths) < 8:
                example_paths.append(member.name)
            if args.xml and path.suffix.lower() == ".xml":
                try:
                    root = ET.parse(archive.extractfile(member)).getroot()
                    for obj in root.findall("object"):
                        objects += 1
                        classes[obj.findtext("name", default="")] += 1
                        occluded[obj.findtext("occluded", default="")] += 1
                        if obj.find("./attributes/attribute[name='track_id']/value") is not None:
                            with_id += 1
                except (ET.ParseError, OSError, ValueError) as exc:
                    if len(bad_xml) < 10:
                        bad_xml.append([member.name, str(exc)])
    frame_ranges = {}
    for group, values in frame_numbers.items():
        ordered = sorted(values)
        runs = []
        start = last = ordered[0]
        for number in ordered[1:]:
            if number != last + 1:
                runs.append([start, last, last - start + 1])
                start = number
            last = number
        runs.append([start, last, last - start + 1])
        frame_ranges[group] = {"count": len(ordered), "min": ordered[0],
                               "max": ordered[-1], "longest_runs": sorted(runs, key=lambda r: -r[2])[:5]}
    print(json.dumps({
        "archive": str(args.archive), "files": files,
        "uncompressed_file_bytes": total_bytes,
        "groups": groups, "suffixes": suffixes,
        "classes": classes, "occluded_values": occluded,
        "objects": objects, "objects_with_track_id": with_id,
        "bad_xml_examples": bad_xml, "example_paths": example_paths,
        "frame_ranges": frame_ranges,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
