"""Audit all Urban1 labels for independent intersection-turn coverage."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = OUT.parent / "review106_vehicle_datasets"
sys.path.insert(0, str(OUT.parent / "review99_roundabout_turns"))
from audit_turns import labels_for_camera


def main():
    manifest = json.loads((SOURCE / "gram_manifest.json").read_text(encoding="utf-8"))
    archive = SOURCE / manifest["annotation_kit"]["file"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["annotation_kit"]["sha256"]
    rows, classes, trajectory, occluded = [], Counter(), Counter(), Counter()
    with ZipFile(archive) as source:
        names = [name for name in source.namelist()
                 if name.startswith("GRAM-RTMv4/Annotations/Urban1/xml/") and name.endswith(".xml")]
        names.sort(key=lambda name: int(Path(name).stem))
        indices = [int(Path(name).stem) for name in names]
        assert indices == list(range(23435))
        for index, name in zip(indices, names):
            document = ET.fromstring(source.read(name))
            for obj in document.findall("object"):
                identity = int(obj.findtext("ID"))
                x1, y1, x2, y2 = [float(obj.findtext("bndbox/" + k))
                                  for k in ("xmin", "ymin", "xmax", "ymax")]
                assert np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1
                rows.append([index + 1, identity, x1, y1, x2 - x1, y2 - y1, 1, 1, -1])
                classes[obj.findtext("class")] += 1
                trajectory[obj.findtext("trajectory")] += 1
                occluded[obj.findtext("occluded", "missing")] += 1
    array = np.asarray(rows)
    unique, inverse, counts = np.unique(array[:, :2], axis=0, return_inverse=True, return_counts=True)
    duplicates = int((counts[inverse] > 1).sum())
    array = array[counts[inverse] == 1]
    gt = OUT / "gram_Urban1_full_gt.csv"
    np.savetxt(gt, array, delimiter=",", fmt="%.10g")
    ids, present, groups, episodes = labels_for_camera(gt, "Urban1", 23435, 25)
    result = {
        "source": "https://gram.web.uah.es/data/datasets/rtm/index.html",
        "scope": "All Urban1 annotations, GT-only, no tracking scores or images",
        "frames": 23435, "fps": 25, "boxes": len(array), "ids": len(ids),
        "classes": dict(classes), "trajectory_attribute": dict(trajectory),
        "occluded_attribute": dict(occluded), "duplicate_frame_id_rows_removed": duplicates,
        "angle_groups": {name: {"gt_boxes": int((index >= 0).sum()),
                               "episodes": sum(e["level"] == name for e in episodes)}
                         for name, index in groups.items()},
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "gt_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(),
        "download_blocker": "Official Urban1 image SharePoint URL redirects to university Microsoft login",
        "test_only": "Author restricts images to evaluation, without tuning",
    }
    (OUT / "gram_full_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
