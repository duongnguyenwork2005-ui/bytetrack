"""Fetch fresh RideSafe clips on separate development and test recording dates."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import sys

LOGS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOGS / "review101_detector_dair_nano"))
from fetch_training_clips import VIDEO_PARTS, available_clips, fetch_clip  # noqa: E402

DAYS = {"dev": "20221010", "test": "20221017"}


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    audit = json.loads((LOGS / "review98_pf_hotspots/ridesafe_videoset1_audit.json").read_text(encoding="utf-8"))
    eligible = {PurePosixPath(row["file"]).stem for row in audit["videos"]
                if row["motorcycle_frames"] >= 100
                and row["box_counts"].get("motorcycle|multiframe_visible_tracks", 0) >= 5
                and PurePosixPath(row["file"]).stem[:8] in DAYS.values()}
    options = sorted(item for part in VIDEO_PARTS for item in available_clips(part, eligible))
    selections = {split: [item for item in options if item[1][:8] == day][:2] for split, day in DAYS.items()}
    assert all(len(items) == 2 for items in selections.values())
    print(json.dumps(selections), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {split: [pool.submit(fetch_clip, root / split, item) for item in choices]
                for split, choices in selections.items()}
        records = {split: [job.result() for job in futures] for split, futures in jobs.items()}
    assert len({row["video"]["sha256"] for rows in records.values() for row in rows}) == 4
    (root / "sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
