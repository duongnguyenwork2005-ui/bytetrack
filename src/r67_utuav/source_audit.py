"""Read-only audit of UTUAV validation XML geometry before conversion."""
import json
import math
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare as prep


def main():
    selected = prep.wanted()
    counts = {name: {"frames": 0, "objects": 0, "out_of_bounds": 0,
                     "negative_or_zero_size": 0, "nonfinite": 0, "duplicate_ids": 0,
                     "unknown_class": 0, "occluded_nonzero": 0, "truncated_true": 0,
                     "max_excursion_px": 0, "max_excursion_example": None,
                     "examples": []} for name in prep.RANGES}
    seen = set()
    with tarfile.open(prep.LABEL_ARCHIVE, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            import re
            match = re.fullmatch(r"([BC])_6class_PascalVOC/Annotations/v[12]_(\d+)\.xml", member.name)
            if match is None:
                continue
            key = match.group(1), int(match.group(2))
            if key not in selected:
                continue
            name, _, _ = selected[key]
            row = counts[name]
            root = ET.fromstring(archive.extractfile(member).read())
            row["frames"] += 1
            ids = set()
            for obj in root.findall("object"):
                row["objects"] += 1
                cls = obj.findtext("name")
                tid = int(obj.findtext("./attributes/attribute[name='track_id']/value"))
                if tid in ids: row["duplicate_ids"] += 1
                ids.add(tid)
                if cls not in prep.CLASSES: row["unknown_class"] += 1
                if obj.findtext("occluded") not in (None, "0"): row["occluded_nonzero"] += 1
                if obj.findtext("truncated") not in (None, "0"): row["truncated_true"] += 1
                b = obj.find("bndbox")
                box = {x:float(b.findtext(x)) for x in ("xmin", "ymin", "xmax", "ymax")}
                x1,y1,x2,y2 = box.values()
                if not all(map(math.isfinite, box.values())): row["nonfinite"] += 1
                if x2 <= x1 or y2 <= y1: row["negative_or_zero_size"] += 1
                if x1 < 0 or y1 < 0 or x2 > 3840 or y2 > 2160:
                    row["out_of_bounds"] += 1
                    excursion = max(0, -x1, -y1, x2-3840, y2-2160)
                    if excursion > row["max_excursion_px"]:
                        row["max_excursion_px"] = excursion
                        row["max_excursion_example"] = {"frame": key[1], "id": tid, "box": box}
                    if len(row["examples"]) < 20:
                        row["examples"].append({"frame": key[1], "id": tid,
                            "class": cls, "truncated": obj.findtext("truncated"), "box": box})
            seen.add(key)
    prep.require(seen == set(selected), "audit did not cover every selected XML")
    prep.write(prep.LOG / "source_xml_audit.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
