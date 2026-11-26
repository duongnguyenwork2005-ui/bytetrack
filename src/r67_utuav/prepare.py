"""Prepare the locked UTUAV v2 B/C validation split and its 128-frame smoke prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

IMAGE_ARCHIVE = Path("F:/LABOTARY_data/goc_tai_ve/UTUAV_B_C_3and6class.tar.gz")
IMAGE_SHA = "fd63ee8e88825a9fba962f9a911a06c344b278452374f0b13feb4406e4b3cd61"
LABEL_ARCHIVE = Path("D:/LABOTARY/UTUAV_MOT_VOC_3_6classes.tar.gz")
IMAGES = Path("F:/LABOTARY_data/utuav_validation_v2/frames")
LOG = ROOT / "handoff/logs/external_eval/utuav"
SPLIT = "UTUAVSYS-validation"
SMOKE = SPLIT + "-smoke128"
FPS = 30000 / 1001
RANGES = {"B_validation": ("B", 26341, 27078, "v1"),
          "C_validation": ("C", 20271, 21021, "v2")}
CLASSES = {"motorbike": 1, "car": 1, "bus": 1,
           "motorbike_stat": 8, "car_stat": 8, "bus_stat": 8}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def wanted():
    return {(scene, num): (name, i, prefix) for name, (scene, first, last, prefix) in RANGES.items()
            for i, num in enumerate(range(first, last + 1), 1)}


def verify_sources():
    require(IMAGE_ARCHIVE.is_file() and LABEL_ARCHIVE.is_file(), "missing source archive")
    # Archive was already hashed byte-for-byte when transferred to F: on 02/11.
    # Re-reading 18 GB here would duplicate the full sequential read performed by tar extraction.
    transfer = Path("D:/LABOTARY/_luu_tru/chuyen_o_roi_20261102/manifest.csv")
    import csv
    with transfer.open(encoding="utf-8-sig", newline="") as stream:
        record = next((row for row in csv.DictReader(stream)
                       if row["file"] == IMAGE_ARCHIVE.name), None)
    require(record is not None and record["sha256"] == IMAGE_SHA
            and record["dich"].lower() == str(IMAGE_ARCHIVE).lower()
            and IMAGE_ARCHIVE.stat().st_size == int(record["byte"]),
            "image archive differs in size/path from verified transfer manifest")
    require(LABEL_ARCHIVE.stat().st_size == 147813351, "label archive size differs from pilot")
    return {"images": {"path": str(IMAGE_ARCHIVE), "bytes": IMAGE_ARCHIVE.stat().st_size,
                       "sha256_from_transfer_manifest": IMAGE_SHA,
                       "transfer_manifest": str(transfer), "transfer_manifest_sha256": digest(transfer)},
            "labels": {"path": str(LABEL_ARCHIVE), "bytes": LABEL_ARCHIVE.stat().st_size,
                       "sha256": digest(LABEL_ARCHIVE)}}


def extract():
    from PIL import Image
    source = verify_sources()
    selected = wanted()
    require(shutil.disk_usage(IMAGES.anchor).free > 5 * 1024**3 + 3_000_000_000,
            "F: free space below extraction estimate + 5 GiB")
    pattern = re.compile(r"UTUAV_B_C_3and6class/([BC])_Sequential_Gap300-3class/validation/images/"
                         r"(v[12])_(\d+)\.jpg")
    found = {}
    with tarfile.open(IMAGE_ARCHIVE, "r|gz") as archive:
        for member in archive:
            match = pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            scene, prefix, number = match.group(1), match.group(2), int(match.group(3))
            key = (scene, number)
            if key not in selected:
                continue
            name, index, expected_prefix = selected[key]
            require(prefix == expected_prefix, f"wrong prefix {member.name}")
            target = IMAGES / name / f"{index:04d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".jpg.tmp")
            with archive.extractfile(member) as original, temporary.open("wb") as output:
                shutil.copyfileobj(original, output, 4 << 20)
            require(temporary.stat().st_size == member.size, f"truncated image {member.name}")
            h = digest(temporary)
            if target.exists():
                require(target.stat().st_size == member.size and digest(target) == h,
                        f"existing image differs from archive {target}")
                temporary.unlink()
            else:
                temporary.replace(target)
            with Image.open(target) as picture:
                require(picture.size == (3840, 2160), f"wrong JPG dimensions {target}")
            require(key not in found, f"duplicate image {key}")
            found[key] = {"source": member.name, "sha256": h, "bytes": member.size,
                          "output": str(target)}
    require(set(found) == set(selected), f"missing images: {list(set(selected)-set(found))[:10]}")
    write(LOG / "images_verified.json", {"source": source, "files":
          {f"{scene}:{number}": entry for (scene, number), entry in sorted(found.items())}})
    return source, found


def finalize_extracted():
    """Lock the already extracted 1,489 files after the prior full-archive CRC check.

    The source archive was completely checked in the R64 pilot and SHA verified
    during transfer to F:. The selective tar pass reached all target frames,
    then was stopped before scanning unrelated training data at the archive end.
    """
    from PIL import Image
    source = verify_sources()
    selected = wanted()
    found = {}
    for (scene, number), (name, index, prefix) in selected.items():
        path = IMAGES / name / f"{index:04d}.jpg"
        require(path.is_file() and path.stat().st_size > 0, f"missing extracted image {path}")
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            require(picture.size == (3840, 2160), f"wrong JPG dimensions {path}")
        found[(scene, number)] = {"source":
            f"UTUAV_B_C_3and6class/{scene}_Sequential_Gap300-3class/validation/images/"
            f"{prefix}_{number:06d}.jpg", "sha256": digest(path),
            "bytes": path.stat().st_size, "output": str(path)}
    for name, (_, first, last, _) in RANGES.items():
        actual = {p.name for p in (IMAGES / name).glob("*.jpg")}
        expected = {f"{i:04d}.jpg" for i in range(1, last-first+2)}
        require(actual == expected, f"extra/missing JPG in {name}")
    write(LOG / "images_verified.json", {"source": source, "verification":
        "selected tar members were extracted; source archive full CRC checked in "
        "handoff/logs/external_mot_prep/utuav_full_check.log; SHA verified on F transfer; "
        "each extracted JPG decoded and hashed after selective pass",
        "files": {f"{scene}:{number}": entry for (scene, number), entry in sorted(found.items())}})
    return source, found


def convert_xml(content, scene, number, index, prefix):
    root = ET.fromstring(content)
    require(root.findtext("filename") == f"{prefix}_{number:06d}.jpg", f"XML filename {scene}:{number}")
    size = root.find("size")
    require(size is not None and size.findtext("width") == "3840" and size.findtext("height") == "2160",
            f"XML dimensions {scene}:{number}")
    rows, class_counts, ids = [], {}, set()
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        require(cls in CLASSES, f"unknown XML class {scene}:{number}: {cls}")
        tid = obj.findtext("./attributes/attribute[name='track_id']/value")
        require(tid is not None, f"missing track ID {scene}:{number}")
        tid = int(tid) + 1
        require(tid > 0 and tid not in ids, f"duplicate/invalid frame ID {scene}:{number}:{tid}")
        ids.add(tid)
        box = obj.find("bndbox")
        require(box is not None, f"missing box {scene}:{number}")
        x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
        x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
        # The source has 45 boundary boxes extending up to 4 px past the image.
        # Preserve the XML coordinates as-is: no clipping or label repair.
        require(all(math.isfinite(z) for z in (x1, y1, x2, y2)) and x1 < x2 and y1 < y2,
                f"nonfinite/invalid XML box {scene}:{number}:{tid}")
        rows.append(f"{index},{tid},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},1,{CLASSES[cls]},1")
        class_counts[cls] = class_counts.get(cls, 0) + 1
    return rows, class_counts


def labels(source, images):
    selected = wanted()
    pattern = re.compile(r"([BC])_6class_PascalVOC/Annotations/(v[12])_(\d+)\.xml")
    found, rows = {}, {name: [] for name in RANGES}
    counts = {name: {} for name in RANGES}
    with tarfile.open(LABEL_ARCHIVE, "r|gz") as archive:
        for member in archive:
            match = pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            scene, prefix, number = match.group(1), match.group(2), int(match.group(3))
            key = (scene, number)
            if key not in selected:
                continue
            name, index, expected_prefix = selected[key]
            require(prefix == expected_prefix and key not in found, f"wrong/duplicate XML {member.name}")
            content = archive.extractfile(member).read()
            require(len(content) == member.size, f"truncated XML {member.name}")
            frame_rows, frame_counts = convert_xml(content, scene, number, index, prefix)
            rows[name].extend(frame_rows)
            for cls, n in frame_counts.items():
                counts[name][cls] = counts[name].get(cls, 0) + n
            found[key] = {"source": member.name, "sha256": hashlib.sha256(content).hexdigest(),
                          "n_rows": len(frame_rows), "index": index}
    require(set(found) == set(selected), f"missing XML: {list(set(selected)-set(found))[:10]}")
    output = {"source": source, "sequences": {}}
    for name, (scene, first, last, prefix) in RANGES.items():
        n = last - first + 1
        require(all((scene, frame) in images and (scene, frame) in found for frame in range(first, last+1)),
                f"missing image/XML pair {name}")
        for split, length in ((SPLIT, n), (SMOKE, 128)):
            d = config.PROCESSED_DIR / split / name
            d.joinpath("gt").mkdir(parents=True, exist_ok=True)
            data = [row for row in rows[name] if int(row.split(",", 1)[0]) <= length]
            gt = d / "gt/gt.txt"
            gt_content = "\n".join(data) + "\n"
            ini_content = (
                f"[Sequence]\nname={name}\nimDir={IMAGES / name}\nframeRate={FPS:.12f}\n"
                f"seqLength={length}\nimWidth=3840\nimHeight=2160\nimExt=.jpg\n")
            ini_path = d / "seqinfo.ini"
            for path, content in ((gt, gt_content), (ini_path, ini_content)):
                if path.exists():
                    require(path.read_text(encoding="utf-8") == content,
                            f"existing processed file differs: {path}")
                else:
                    path.write_text(content, encoding="utf-8")
            output["sequences"].setdefault(name, {})[split] = {"frames": length, "gt_rows": len(data),
                "gt_sha256": digest(gt), "seqinfo_sha256": digest(d / "seqinfo.ini")}
        output["sequences"][name].update({"source_range": [first, last], "prefix": prefix,
            "class_counts": counts[name], "xml_sha256": hashlib.sha256("\n".join(
                found[(scene, frame)]["sha256"] for frame in range(first, last+1)).encode()).hexdigest(),
            "image_sha256": hashlib.sha256("\n".join(
                images[(scene, frame)]["sha256"] for frame in range(first, last+1)).encode()).hexdigest()})
    for split in (SPLIT, SMOKE):
        seqmap = config.PROCESSED_DIR / "seqmaps" / f"{split}.txt"
        seqmap.parent.mkdir(parents=True, exist_ok=True)
        seqmap.write_text("name\n" + "\n".join(RANGES) + "\n", encoding="utf-8")
    write(LOG / "source_manifest.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["all", "verify", "extract", "finalize", "labels"])
    args = parser.parse_args()
    if args.step == "verify":
        print(json.dumps(verify_sources(), indent=2)); return
    if args.step == "finalize":
        source, images = finalize_extracted()
        print(json.dumps(labels(source, images)["sequences"], ensure_ascii=False, indent=2))
        return
    if args.step in ("all", "extract"):
        source, images = extract()
    else:
        source = verify_sources()
        images = {(scene, number): data for key, data in json.loads(
            (LOG / "images_verified.json").read_text(encoding="utf-8"))["files"].items()
            for scene, number in [key.split(":")]}
        images = {(scene, int(number)): data for (scene, number), data in images.items()}
    if args.step in ("all", "labels"):
        print(json.dumps(labels(source, images)["sequences"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
