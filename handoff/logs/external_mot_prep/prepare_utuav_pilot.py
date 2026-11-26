"""Extract only preselected UTUAV pilot frames and export MOT GT; no tracker changes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_MAP = {"motorbike": 1, "car": 1, "bus": 1,
             "motorbike_stat": 8, "car_stat": 8, "bus_stat": 8}


def selected(specs: list[dict]) -> dict[tuple[str, int], dict]:
    result = {}
    for spec in specs:
        for number in range(spec["first_source_frame"], spec["last_source_frame"] + 1):
            key = (spec["scene"], number)
            if key in result:
                raise ValueError(f"overlapping selection: {key}")
            result[key] = spec
    return result


def write_member(archive: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refuse overwrite: {target}")
    with archive.extractfile(member) as source, target.open("xb") as sink:
        shutil.copyfileobj(source, sink, length=1024 * 1024)
    if target.stat().st_size != member.size:
        raise IOError(f"size mismatch: {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-archive", type=Path, required=True)
    ap.add_argument("--labels-archive", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.selection.read_text(encoding="utf-8"))
    specs = plan["sequences"]
    wanted = selected(specs)
    seen_images, seen_xml = set(), set()

    image_pattern = re.compile(
        r"UTUAV_B_C_3and6class/([BC])_Sequential_Gap300-3class/validation/images/v[12]_(\d+)\.jpg")
    with tarfile.open(args.images_archive, "r|gz") as archive:
        for member in archive:
            match = image_pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            key = (match.group(1), int(match.group(2)))
            if key not in wanted:
                continue
            spec = wanted[key]
            write_member(archive, member, args.out / "frames" / spec["name"] / Path(member.name).name)
            seen_images.add(key)
    if seen_images != set(wanted):
        raise ValueError(f"missing images: {sorted(set(wanted) - seen_images)[:10]}")

    label_pattern = re.compile(r"([BC])_6class_PascalVOC/Annotations/v[12]_(\d+)\.xml")
    with tarfile.open(args.labels_archive, "r|gz") as archive:
        for member in archive:
            match = label_pattern.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            key = (match.group(1), int(match.group(2)))
            if key not in wanted:
                continue
            spec = wanted[key]
            write_member(archive, member, args.out / spec["name"] / "annotations" / Path(member.name).name)
            seen_xml.add(key)
    if seen_xml != set(wanted):
        raise ValueError(f"missing XML labels: {sorted(set(wanted) - seen_xml)[:10]}")

    summary = {}
    for spec in specs:
        name, scene = spec["name"], spec["scene"]
        first, last = spec["first_source_frame"], spec["last_source_frame"]
        image_dir = args.out / "frames" / name
        xml_dir = args.out / name / "annotations"
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(xml_dir.glob("*.xml"))
        expected = last - first + 1
        if len(images) != expected or len(labels) != expected:
            raise ValueError(f"count mismatch: {name} images={len(images)} XML={len(labels)}")
        rows, class_counts = [], {label: 0 for label in CLASS_MAP}
        seen_ids = set()
        for i, label in enumerate(labels, start=1):
            frame_number = int(label.stem.rsplit("_", 1)[1])
            if frame_number != first + i - 1 or image_dir.joinpath(label.stem + ".jpg").is_file() is False:
                raise ValueError(f"image/XML sequence mismatch: {label}")
            root = ET.parse(label).getroot()
            if (root.findtext("filename") or "") != label.stem + ".jpg":
                raise ValueError(f"XML filename mismatch: {label}")
            for obj in root.findall("object"):
                cls = obj.findtext("name", default="")
                if cls not in CLASS_MAP:
                    raise ValueError(f"unknown class {cls!r} in {label}")
                id_text = obj.findtext("./attributes/attribute[name='track_id']/value")
                if id_text is None:
                    raise ValueError(f"missing track_id in {label}")
                track_id = int(id_text) + 1  # MOT convention: positive ids; source id may be zero
                if (i, track_id) in seen_ids:
                    raise ValueError(f"duplicate frame/id in {label}: {track_id}")
                seen_ids.add((i, track_id))
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError(f"missing box in {label}")
                x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
                x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid box in {label}")
                rows.append(f"{i},{track_id},{x1:.4f},{y1:.4f},{x2-x1:.4f},{y2-y1:.4f},1,{CLASS_MAP[cls]},1")
                class_counts[cls] += 1
        seq_dir = args.processed / name
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (seq_dir / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir={image_dir.resolve()}\nframeRate=30\nseqLength={expected}\n"
            "imWidth=3840\nimHeight=2160\nimExt=.jpg\n", encoding="utf-8")
        summary[name] = {"scene": scene, "source_frames": [first, last], "n_frames": expected,
                         "gt_objects": len(rows), "class_counts": class_counts,
                         "images_bytes": sum(path.stat().st_size for path in images)}
    seqmap = args.processed.parent / "seqmaps" / f"{args.processed.name}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(spec["name"] for spec in specs) + "\n", encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps({
        "selection": plan, "images_archive": str(args.images_archive),
        "images_archive_size": args.images_archive.stat().st_size,
        "labels_archive": str(args.labels_archive),
        "labels_archive_size": args.labels_archive.stat().st_size,
        "summary": summary, "gt_split": args.processed.name,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
