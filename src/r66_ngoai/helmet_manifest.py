"""Lock the R66 HELMET test36 sample before inspecting detection or tracking scores.

Only reads the small split and annotation archives. It does not read or unpack images.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("F:/LABOTARY_data/goc_tai_ve/HELMET")
OUTPUT = ROOT / "handoff/logs/external_eval/r66/helmet/manifest_test36.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[str]] = defaultdict(list)
    ids = [r["video_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate video_id in split")
    for row in rows:
        if row["Set"] == "test":
            video = row["video_id"]
            group, suffix = video.rsplit("_", 1)
            if not suffix.isdigit() or not group or Path(video).name != video:
                raise ValueError(f"invalid clip name: {video}")
            groups[group].append(video)
    if len(groups) != 12 or sum(map(len, groups.values())) != 182:
        raise ValueError("test split differs from the pre-registered 12 groups / 182 clips")
    selected = []
    for group, videos in sorted(groups.items()):
        if len(videos) < 3:
            raise ValueError(f"{group}: fewer than three test clips")
        for rank, video in enumerate(sorted(videos, key=lambda v: (
            hashlib.sha256(("r66-helmet-v1|" + v).encode("utf-8")).hexdigest(), v))[:3], 1):
            selected.append({"group": group, "video_id": video, "rank_in_group": rank})
    return selected


def parse_boxes(raw: bytes, video: str):
    """Validate identity within one clip; collapse only byte-equivalent geometry values."""
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    seen: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    duplicate_equal = 0
    labels = Counter()
    for row in rows:
        track = row["track_id"]
        frame = int(row["frame_id"])
        box = tuple(float(row[k]) for k in ("x", "y", "w", "h"))
        x, y, w, h = box
        if (not track.strip() or not 1 <= frame <= 100 or not all(map(math.isfinite, box))
                or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1920 or y + h > 1080):
            raise ValueError(f"invalid frame/box: {video} {track} {frame} {box}")
        key = (track, frame)
        if key in seen:
            if seen[key] != box:
                raise ValueError(f"conflicting motorcycle boxes: {video} {track} {frame}")
            duplicate_equal += 1
        else:
            seen[key] = box
        labels[row["label"]] += 1
    return seen, {"annotation_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows),
            "unique_track_frame": len(seen), "duplicate_equal_box": duplicate_equal,
            "track_count": len({k[0] for k in seen}), "frame_min": min((f for _, f in seen), default=None),
            "frame_max": max((f for _, f in seen), default=None), "labels": dict(sorted(labels.items()))}


def annotation_audit(z: zipfile.ZipFile, video: str) -> dict:
    return parse_boxes(z.read(f"annotation/{video}.csv"), video)[1]


def main() -> None:
    split = SOURCE / "data_split.csv"
    annotations = SOURCE / "annotation.zip"
    rows = list(csv.DictReader(split.open(encoding="utf-8-sig", newline="")))
    videos = select(rows)
    with zipfile.ZipFile(annotations) as z:
        for row in videos:
            row.update(annotation_audit(z, str(row["video_id"])))
    out = {"protocol": "R66 HELMET-test36-motorcycle-v1", "rule": "12 scene groups; first 3 clips per group ordered by sha256(UTF-8 'r66-helmet-v1|' + video_id), tie by video_id",
           "source": {"data_split_csv_sha256": sha256(split), "annotation_zip_sha256": sha256(annotations)},
           "all_split_counts": dict(Counter(row["Set"] for row in rows)), "groups": 12,
           "selected_clips": 36, "selected_frames": 3600, "videos": videos}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text(encoding="utf-8")) != out:
            raise ValueError(f"manifest differs from locked file: {OUTPUT}")
    else:
        OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"locked {len(videos)} clips / 3600 frames; annotation rows {sum(v['rows'] for v in videos)}; "
          f"equal-box duplicates {sum(v['duplicate_equal_box'] for v in videos)}; {OUTPUT}")


if __name__ == "__main__":
    main()
