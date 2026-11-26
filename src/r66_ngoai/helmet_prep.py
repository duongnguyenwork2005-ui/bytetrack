"""Verify OSF HELMET files, extract only locked test36 images, and build MOT GT.

Commands are deliberately separate: verify -> extract -> gt. No score-dependent choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from helmet_manifest import ROOT, SOURCE, OUTPUT, parse_boxes, select, sha256

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/r64_ngoai"))
import config  # noqa: E402
import danh_gia_ngoai as r64  # noqa: E402

LOG = ROOT / "handoff/logs/external_eval/r66/helmet"
IMAGES = Path("F:/LABOTARY_data/helmet_test36")
SPLIT = "HELMET-test36-motorcycle-v1"
SPLIT_SMOKE = SPLIT + "-smoke12"


def manifest() -> dict:
    m = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if m["selected_clips"] != 36 or m["selected_frames"] != 3600 or len(m["videos"]) != 36:
        raise ValueError("locked HELMET manifest has changed")
    if sha256(SOURCE / "data_split.csv") != m["source"]["data_split_csv_sha256"]:
        raise ValueError("data_split.csv changed after selection")
    if sha256(SOURCE / "annotation.zip") != m["source"]["annotation_zip_sha256"]:
        raise ValueError("annotation.zip changed after selection")
    with (SOURCE / "data_split.csv").open(encoding="utf-8-sig", newline="") as file:
        choices = select(list(csv.DictReader(file)))
    if choices != [{k: row[k] for k in ("group", "video_id", "rank_in_group")} for row in m["videos"]]:
        raise ValueError("selected clips or their ranks differ from original test split")
    return m


def listed_files() -> dict[str, dict]:
    files = {}
    for name in ("osf_helmet_root_listing.json", "osf_helmet_image_listing.json"):
        listing = json.loads((LOG.parent / name).read_text(encoding="utf-8"))
        for x in listing["data"]:
            a = x["attributes"]
            if a["kind"] == "file":
                files[a["name"]] = {"size": int(a["size"]), "sha256": a["extra"]["hashes"]["sha256"]}
    if {f"part_{i}.zip" for i in range(1, 8)} - set(files):
        raise ValueError("OSF listing lacks an image part")
    return files


def verify() -> None:
    manifest()
    files = listed_files()
    LOG.mkdir(parents=True, exist_ok=True)
    partial = LOG / "nguon_kiem_sha_crc_tung_file.json"
    check = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    missing = []
    for name, expected in sorted(files.items()):
        path = SOURCE / name
        stat = path.stat() if path.is_file() else None
        size = stat.st_size if stat else -1
        if size != expected["size"]:
            missing.append(name)
            check.pop(name, None)
            print(f"PENDING {name}: {size}/{expected['size']} bytes", flush=True)
            continue
        record = check.get(name, {})
        if (record.get("sha256") == expected["sha256"] and record.get("size") == size
                and record.get("mtime_ns") == stat.st_mtime_ns
                and record.get("crc_ok") == (path.suffix == ".zip")):
            print(f"OK (verified previously) {name}", flush=True)
            continue
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"{name}: SHA256 mismatch")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(f"{name}: CRC failed for {bad}")
                n = len(z.namelist())
        else:
            n = None
        if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != size:
            raise ValueError(f"{name} changed during source verification")
        check[name] = {**expected, "mtime_ns": stat.st_mtime_ns, "crc_ok": path.suffix == ".zip", "entries": n}
        r64.ghi_json_nguyen_tu(partial, check)
        print(f"OK {name}: {size} bytes, SHA256" + (", CRC" if n is not None else ""), flush=True)
    if missing:
        raise ValueError(f"missing/incomplete OSF files: {missing}")
    for name in ("Readme.txt", "data_split.csv"):
        shutil.copy2(SOURCE / name, LOG / name)
    r64.ghi_json_nguyen_tu(LOG / "nguon_kiem_sha_crc.json", check)


def verified_sources() -> dict:
    marker = LOG / "nguon_kiem_sha_crc.json"
    if not marker.exists():
        raise ValueError("run verify first")
    check = json.loads(marker.read_text(encoding="utf-8"))
    files = listed_files()
    if set(check) != set(files):
        raise ValueError("source verification marker does not cover the OSF listing")
    for name, expected in files.items():
        p = SOURCE / name
        stat = p.stat()
        row = check[name]
        if (row.get("sha256") != expected["sha256"] or row.get("size") != stat.st_size
                or stat.st_size != expected["size"] or row.get("mtime_ns") != stat.st_mtime_ns
                or row.get("crc_ok") != (p.suffix == ".zip")):
            raise ValueError(f"{name}: source changed; run verify again")
    return check


def frame_bytes(data: bytes, info: zipfile.ZipInfo) -> str:
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise ValueError(f"{info.filename}: extracted JPEG does not match verified ZIP entry")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.size != (1920, 1080):
            raise ValueError(f"{info.filename}: expected a 1920x1080 JPEG, got {image.format} {image.size}")
    return hashlib.sha256(data).hexdigest()


def extract() -> None:
    m = manifest()
    check = verified_sources()
    selected = {str(row["video_id"]) for row in m["videos"]}
    members: dict[str, list[tuple[Path, str, int]]] = defaultdict(list)
    for part in sorted(SOURCE.glob("part_*.zip")):
        if part.name not in check:
            continue
        with zipfile.ZipFile(part) as z:
            for info in z.infolist():
                bits = info.filename.split("/")
                if len(bits) == 3 and bits[1] in selected and bits[2].endswith(".jpg"):
                    frame = int(Path(bits[2]).stem)
                    members[bits[1]].append((part, info.filename, frame))
    for clip in selected:
        frames = [x[2] for x in members[clip]]
        if sorted(frames) != list(range(1, 101)):
            raise ValueError(f"{clip}: expected exactly 100 unique frames, found {len(frames)}")
    if shutil.disk_usage(IMAGES.parent).free < 8 * 1024 ** 3:
        raise ValueError("F: has less than 8 GiB free")
    extracted = {}
    for clip in sorted(selected):
        dest = IMAGES / clip
        dest.mkdir(parents=True, exist_ok=True)
        parts = defaultdict(list)
        frame_hashes = {}
        for part, name, frame in members[clip]:
            parts[part].append((name, frame))
        for part, entries in parts.items():
            with zipfile.ZipFile(part) as z:
                for name, frame in entries:
                    target = dest / f"{frame}.jpg"
                    info = z.getinfo(name)
                    data = target.read_bytes() if target.exists() else z.read(info)
                    frame_hashes[str(frame)] = frame_bytes(data, info)
                    if not target.exists():
                        with tempfile.NamedTemporaryFile(dir=dest, suffix=".tmp", delete=False) as sink:
                            sink.write(data)
                        os.replace(sink.name, target)
        if sorted(int(p.stem) for p in dest.glob("*.jpg")) != list(range(1, 101)):
            raise ValueError(f"{clip}: incomplete extraction")
        extracted[clip] = {"part": sorted(p.name for p in parts), "frames": 100,
                           "sha256": frame_hashes, "manifest_sha256": sha256(OUTPUT)}
        print(f"extracted {clip}", flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    r64.ghi_json_nguyen_tu(LOG / "giai_nen_test36.json", extracted)


def gt() -> None:
    m = manifest()
    extracted = json.loads((LOG / "giai_nen_test36.json").read_text(encoding="utf-8"))
    if set(extracted) != {v["video_id"] for v in m["videos"]}:
        raise ValueError("test36 images not completely extracted")
    out = config.PROCESSED_DIR / SPLIT
    smoke_out = config.PROCESSED_DIR / SPLIT_SMOKE
    metadata = {"manifest_sha256": sha256(OUTPUT), "annotation_zip_sha256": sha256(SOURCE / "annotation.zip"),
                "videos": {}}
    with zipfile.ZipFile(SOURCE / "annotation.zip") as z:
        for choice in m["videos"]:
            clip = str(choice["video_id"])
            if sorted(int(p.stem) for p in (IMAGES / clip).glob("*.jpg")) != list(range(1, 101)):
                raise ValueError(f"{clip}: image sequence incomplete")
            record = extracted[clip]
            if (record.get("manifest_sha256") != metadata["manifest_sha256"] or any(
                    record.get("sha256", {}).get(str(frame)) != sha256(IMAGES / clip / f"{frame}.jpg")
                    for frame in range(1, 101))):
                raise ValueError(f"{clip}: image content differs from verified extraction")
            name = f"annotation/{clip}.csv"
            raw = z.read(name)
            if hashlib.sha256(raw).hexdigest() != choice["annotation_sha256"]:
                raise ValueError(f"{clip}: annotation changed")
            boxes, audit = parse_boxes(raw, clip)
            if any(choice.get(k) != v for k, v in audit.items()):
                raise ValueError(f"{clip}: annotation audit disagrees with locked manifest")
            ids = {track: i + 1 for i, track in enumerate(sorted({track for track, _ in boxes}))}
            gt_rows = [[frame, ids[track], *box, 1, 1, 1] for (track, frame), box in boxes.items()]
            a = np.asarray(gt_rows, dtype=np.float64)
            a = a[np.lexsort((a[:, 1], a[:, 0]))]
            check = r64.kiem_gt(a, 100, clip)
            folder = out / clip
            (folder / "gt").mkdir(parents=True, exist_ok=True)
            gt_path = folder / "gt/gt.txt"
            np.savetxt(gt_path, a, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f",
                                                          "%d", "%d", "%.3f"])
            (folder / "seqinfo.ini").write_text(
                f"[Sequence]\nname={clip}\nimDir={IMAGES / clip}\nframeRate=10\nseqLength=100\n"
                "imWidth=1920\nimHeight=1080\nimExt=.jpg\n", encoding="utf-8")
            metadata["videos"][clip] = {"group": choice["group"], "gt_rows": len(a), "tracks": len(ids),
                                         "id_map": ids, "gt_sha256": sha256(gt_path), "check": check}
            if choice["rank_in_group"] == 1:
                smoke_clip = smoke_out / clip
                (smoke_clip / "gt").mkdir(parents=True, exist_ok=True)
                shutil.copy2(gt_path, smoke_clip / "gt/gt.txt")
                shutil.copy2(folder / "seqinfo.ini", smoke_clip / "seqinfo.ini")
    seqmap_root = config.PROCESSED_DIR / "seqmaps"
    seqmap_root.mkdir(parents=True, exist_ok=True)
    for split, choices in ((SPLIT, m["videos"]),
                           (SPLIT_SMOKE, [v for v in m["videos"] if v["rank_in_group"] == 1])):
        (seqmap_root / f"{split}.txt").write_text(
            "name\n" + "\n".join(str(x["video_id"]) for x in choices) + "\n", encoding="utf-8")
    (out / "gt_log.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (smoke_out / "gt_log.json").write_text(json.dumps({"from": str(out / "gt_log.json"),
        "manifest_sha256": metadata["manifest_sha256"], "videos": {
            k: v for k, v in metadata["videos"].items() if any(
                c["video_id"] == k and c["rank_in_group"] == 1 for c in m["videos"])}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"GT {SPLIT}: 36 clips, {sum(v['gt_rows'] for v in metadata['videos'].values())} boxes; "
          f"smoke 12 clips", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["verify", "extract", "gt"])
    step = parser.parse_args().step
    {"verify": verify, "extract": extract, "gt": gt}[step]()
