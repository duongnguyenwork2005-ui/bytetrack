"""Download two RideSafe clips from distinct recording days outside the R100 pilot."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import requests

LOGS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOGS / "review100_nano_pf_dataset"))
import fetch_ridesafe_sample as archive  # noqa: E402

VIDEO_PARTS = (archive.VIDEO, "videoset1_videos_part2.zip")
EVAL_DAYS = {"20211109", "20211125"}


def available_clips(part, eligible):
    with requests.Session() as session:
        remote = archive.RangeFile(session, part)
        with zipfile.ZipFile(remote) as zf:
            return [(info.compress_size, stem, part) for stem, info in archive.entries(zf, ".mp4").items()
                    if stem in eligible]


def fetch_clip(root, selection):
    _, stem, part = selection
    folder = root / stem
    folder.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        video = archive.RangeFile(session, part)
        labels = archive.RangeFile(session, archive.LABELS)
        with zipfile.ZipFile(video) as videos, zipfile.ZipFile(labels) as xml:
            record = {"revision": archive.REVISION, "clip": stem, "video_archive": part,
                      "video": archive.save(videos, archive.entries(videos, ".mp4")[stem], folder),
                      "labels": archive.save(xml, archive.entries(xml, ".xml")[stem], folder),
                      "transferred_bytes": video.transferred + labels.transferred}
    (folder / "source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)
    return record


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    audited = json.loads((LOGS / "review98_pf_hotspots/ridesafe_videoset1_audit.json").read_text(encoding="utf-8"))
    eligible = {PurePosixPath(row["file"]).stem for row in audited["videos"]
                if row["motorcycle_frames"] >= 240
                and row["box_counts"].get("motorcycle|multiframe_visible_tracks", 0) >= 12
                and PurePosixPath(row["file"]).stem[:8] not in EVAL_DAYS}
    options = sorted(item for part in VIDEO_PARTS for item in available_clips(part, eligible))
    chosen = []
    days = set()
    for item in options:
        day = item[1][:8]
        if day not in days:
            chosen.append(item)
            days.add(day)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise ValueError(f"Only {len(chosen)} eligible recording days outside the pilot")
    print(f"Selected {chosen}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda item: fetch_clip(root, item), chosen))
    (root / "train_sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
