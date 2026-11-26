"""Validate downloaded archives, annotation databases and source video without tracking."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import io
import json
import sqlite3

import cv2
from PIL import Image

BASE = Path("F:/LABOTARY_data/goc_tai_ve/UrbanTracker-R69")
SOURCE = "https://www.jpjodoin.com/urbantracker/dataset"
FILES = {
    "rene_annotations.zip": ("rene/rene_annotations.zip", 299207),
    "rene_frames.zip": ("rene/rene_frames.zip", 1569006177),
    "rene_mask.png": ("rene/rene_mask.png", 8423),
    "rene_video.mov": ("rene/rene_video.mov", 199595856),
    "stmarc_annotations.zip": ("stmarc/stmarc_annotations.zip", 284598),
    "stmarc_frames.zip": ("stmarc/stmarc_frames.zip", 225012984),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


records = {}
for name, (relative_url, expected_size) in FILES.items():
    path = BASE / name
    assert path.stat().st_size == expected_size, name
    record = dict(path=str(path), url=f"{SOURCE}/{relative_url}",
                  bytes=expected_size, local_sha256=sha256(path))
    if name.endswith(".zip"):
        with ZipFile(path) as archive:
            assert archive.testzip() is None, name
            record["zip_crc"] = "pass"
            databases = {}
            for member in archive.namelist():
                if not member.endswith(".sqlite"):
                    continue
                connection = sqlite3.connect(":memory:")
                connection.deserialize(archive.read(member))
                assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"
                count = connection.execute(
                    "select min(frame_number), max(frame_number), count(distinct frame_number), "
                    "count(distinct object_id), count(*) from bounding_boxes"
                ).fetchone()
                databases[member] = dict(zip(
                    ["first_gt_frame", "last_gt_frame", "annotated_frames", "ids", "boxes"], count))
                connection.close()
            if databases:
                record["databases"] = databases
            frames = [n for n in archive.namelist() if n.lower().endswith(".jpg")]
            if frames:
                frame_numbers = sorted(int(Path(n).stem) for n in frames)
                assert frame_numbers == list(range(1, len(frames) + 1))
                record["frames"] = dict(count=len(frames), first=frames[0], last=frames[-1])
                for index in (0, len(frames) // 2, len(frames) - 1):
                    with Image.open(io.BytesIO(archive.read(frames[index]))) as image:
                        assert image.size == (1280, 720)
                        image.load()
    elif name.endswith(".png"):
        with Image.open(path) as image:
            image.verify()
        record["png_decode"] = "pass"
    elif name.endswith(".mov"):
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        metadata_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        record["video"] = dict(metadata_frames=metadata_count, fps=capture.get(cv2.CAP_PROP_FPS),
                               width=capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                               height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape[:2] == (720, 1280)
            decoded += 1
        capture.release()
        assert decoded == metadata_count == 8501
        record["video"]["decoded_frames"] = decoded
    records[name] = record

result = dict(
    checked_date="2026-11-03",
    source_page="https://www.jpjodoin.com/urbantracker/dataset.html",
    sha256_note="Local digests for reproducibility, not publisher checksums.",
    files=records,
    partial_files=[dict(path=str(p), bytes=p.stat().st_size, usable=False)
                   for p in BASE.glob("*.partial")],
    notes=[
        "Download and file validation only. No detector or tracker executed.",
        "Rene source MOV reports 30000/1001 fps. Web description rounds to 30 fps.",
        "St-Marc page reports 30 fps; ZIP has no temporal metadata. Verify against source video before benchmarking.",
        "Ground-truth frame indexing versus image/video indexing requires an alignment audit before conversion.",
        "Database object counts differ from webpage prose; see per-database counts.",
        "Rene GT covers frames 7200 through 8199 inclusive, despite README mentioning 8200.",
    ],
)
Path(__file__).with_name("datasets.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({name: record.get("video", record.get("frames", {"validated": True}))
                  for name, record in records.items()}, indent=2))
