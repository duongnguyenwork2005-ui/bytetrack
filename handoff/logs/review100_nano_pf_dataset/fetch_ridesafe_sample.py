"""Fetch a matched RideSafe clip and labels using bounded HTTP range reads."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import requests

REVISION = "e2a6637"
ORIGIN = f"https://huggingface.co/datasets/DeepBug/RideSafe-400/resolve/{REVISION}/"
VIDEO = "videoset1_videos_part1.zip"
LABELS = "videoset1_xml_annots_with_rider_motor_poly.zip"
LIMIT = 180 * 1024 * 1024
BLOCK = 8 * 1024 * 1024


class RangeFile(io.RawIOBase):
    def __init__(self, session, name):
        self.session = session
        self.url = ORIGIN + name
        self.offset = 0
        self.size = None
        self.transferred = 0
        self.block = (-1, b"")
        self.fetch(0, 0)

    def fetch(self, start, end):
        with self.session.get(self.url, params={"download": "true", "range_start": start},
                              headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=(15, 60)) as response:
            expected_prefix = f"bytes {start}-{end}/"
            header = response.headers.get("Content-Range", "")
            if response.status_code != 206 or not header.startswith(expected_prefix):
                raise IOError(f"Unexpected range response {response.status_code}: {header}")
            size = int(header.split("/")[-1])
            if self.size is not None and self.size != size:
                raise IOError("Source archive size changed")
            self.size = size
            count = end - start + 1
            if self.transferred + count > LIMIT:
                raise IOError("Pilot transfer exceeds limit")
            data = response.raw.read(count + 1)
            if len(data) != count:
                raise IOError("Range body has incorrect length")
            self.transferred += count
            return data

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        self.offset = (0, self.offset, self.size)[whence] + offset
        if self.offset < 0:
            raise ValueError("Negative offset")
        return self.offset

    def read(self, count=-1):
        stop = self.size if count < 0 else min(self.size, self.offset + count)
        data = bytearray()
        while self.offset < stop:
            start = self.offset // BLOCK * BLOCK
            if self.block[0] != start:
                self.block = (start, self.fetch(start, min(self.size, start + BLOCK) - 1))
            chunk = self.block[1][self.offset - start:stop - start]
            data.extend(chunk)
            self.offset += len(chunk)
        return bytes(data)


def entries(archive, suffix):
    matching = [item for item in archive.infolist() if item.filename.lower().endswith(suffix)]
    by_stem = {PurePosixPath(item.filename).stem: item for item in matching}
    if len(by_stem) != len(matching):
        raise ValueError("Duplicate file stems")
    return by_stem


def save(archive, item, directory):
    name = PurePosixPath(item.filename).name
    target = directory / name
    partial = directory / (name + ".partial")
    with archive.open(item) as source, partial.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=BLOCK)
    partial.replace(target)
    with target.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    return {"member": item.filename, "file": name, "bytes": item.file_size, "sha256": sha256}


def main():
    root = Path(sys.argv[1])
    excluded_day = sys.argv[2] if len(sys.argv) > 2 else None
    root.mkdir(parents=True, exist_ok=True)
    previous = Path(__file__).resolve().parents[1] / "review98_pf_hotspots/ridesafe_videoset1_audit.json"
    audited = json.loads(previous.read_text(encoding="utf-8"))["videos"]
    eligible = {PurePosixPath(row["file"]).stem for row in audited
                if row["motorcycle_frames"] >= 290
                and row["box_counts"].get("motorcycle|multiframe_visible_tracks", 0) >= 10}
    with requests.Session() as session:
        video_file = RangeFile(session, VIDEO)
        label_file = RangeFile(session, LABELS)
        with zipfile.ZipFile(video_file) as videos, zipfile.ZipFile(label_file) as labels:
            video_entries = entries(videos, ".mp4")
            label_entries = entries(labels, ".xml")
            candidates = {key for key in eligible & video_entries.keys() & label_entries.keys()
                          if key[:8] != excluded_day}
            clip = min(candidates, key=lambda key: (video_entries[key].compress_size, key))
            print(f"Selected {clip} from {len(candidates)} eligible clips", flush=True)
            result = {"revision": REVISION, "clip": clip, "eligible_in_part1": len(candidates),
                      "excluded_recording_day": excluded_day,
                      "video": save(videos, video_entries[clip], root),
                      "labels": save(labels, label_entries[clip], root),
                      "transferred_bytes": video_file.transferred + label_file.transferred}
    (root / "source.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
