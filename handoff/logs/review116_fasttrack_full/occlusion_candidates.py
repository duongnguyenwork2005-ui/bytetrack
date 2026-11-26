"""Find R115-style candidate events and render images for manual verification."""

from collections import defaultdict
from pathlib import Path
import json
import sys
import time
from zipfile import ZipFile

import cv2
import numpy as np

from download import DATA, HERE, PRIOR

PANELS = DATA / "occlusion_review_r116"


def candidate_events(rows):
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    frame_breaks = np.flatnonzero(np.diff(rows[:, 0])) + 1
    cars = {}
    overlaps = defaultdict(list)
    for frame_rows in np.split(rows, frame_breaks):
        target = frame_rows[frame_rows[:, 7] == 6]
        blocker = frame_rows[np.isin(frame_rows[:, 7], [2, 3, 4, 5])]
        frame = int(frame_rows[0, 0])
        for row in target:
            cars[int(row[1]), frame] = row
        top_left, other_left = target[:, None, 2:4], blocker[None, :, 2:4]
        bottom_right = top_left + target[:, None, 4:6]
        other_right = other_left + blocker[None, :, 4:6]
        shared = np.maximum(0, np.minimum(bottom_right, other_right) - np.maximum(top_left, other_left))
        area = target[:, None, 4:6].prod(axis=-1)
        fraction = np.divide(shared.prod(axis=-1), area, out=np.zeros(shared.shape[:2]), where=area > 0)
        possible = (fraction >= .45) & (other_right[:, :, 1] > bottom_right[:, :, 1] + 4)
        for i, j in zip(*np.where(possible & (target[:, None, 5] >= 10))):
            overlaps[int(target[i, 1]), int(blocker[j, 1])].append((frame, fraction[i, j]))
    events = []
    for (car, blocker), points in overlaps.items():
        points = np.asarray(points)
        for chunk in np.split(points, np.flatnonzero(np.diff(points[:, 0]) != 1) + 1):
            first, last = int(chunk[0, 0]), int(chunk[-1, 0])
            peak, fraction = chunk[chunk[:, 1].argmax()]
            before, after = first - 10, last + 10
            if (6 <= len(chunk) <= 120 and fraction >= .65 and
                    (car, before) in cars and (car, after) in cars):
                events.append({"car": car, "blocker": blocker, "before": before, "first": first,
                               "peak": int(peak), "last": last, "after": after,
                               "peak_overlap": round(float(fraction), 5)})
    events.sort(key=lambda e: (e["before"], e["car"], e["blocker"]))
    for idx, event in enumerate(events, 1):
        event["number"] = idx
    return events, cars


def image_at(archive, name, frame):
    path = f"{name}/img1/{frame:06}.jpg"
    picture = cv2.imdecode(np.frombuffer(archive.read(path), np.uint8), cv2.IMREAD_COLOR)
    assert picture is not None, path
    return picture


def tile(image, caption, width=341, height=154):
    assert image.size
    scale = min(width / image.shape[1], (height - 18) / image.shape[0])
    sized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    cell[:sized.shape[0], :sized.shape[1]] = sized
    cv2.putText(cell, caption, (4, height - 5), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
    return cell


def draw_event(archive, name, event, cars):
    car = event["car"]
    frames = [event["before"], event["peak"], event["after"]]
    boxes = np.rint([cars[car, t][2:6] for t in frames]).astype(int)
    padding = int(np.clip(np.max(boxes[:, 3]), 30, 110))
    x0, y0 = np.maximum(0, boxes[:, :2].min(axis=0) - padding)
    x1, y1 = (boxes[:, :2] + boxes[:, 2:]).max(axis=0) + padding
    cells = []
    for t, (x, y, w, h) in zip(frames, boxes):
        picture = image_at(archive, name, t)
        cv2.rectangle(picture, (x, y), (x+w, y+h), (0, 255, 255), 1)
        text = f"#{event['number']} car {car} via {event['blocker']} | f{t}"
        cells.append(tile(picture[y0:y1, x0:x1], text))
    return np.hstack(cells)


def class_panel(archive, name, rows):
    cells = []
    for code in sorted(set(rows[:, 7].astype(int))):
        relevant = rows[rows[:, 7] == code]
        area = relevant[:, 4] * relevant[:, 5]
        picks, seen = [], set()
        for row in relevant[np.argsort(area)[::-1]]:
            if row[1] not in seen:
                picks.append(row)
                seen.add(row[1])
            if len(picks) == 2:
                break
        row_cells = []
        for row in picks:
            picture = image_at(archive, name, int(row[0]))
            x, y, w, h = np.rint(row[2:6]).astype(int)
            object_crop = picture[max(0, y-6):y+h+6, max(0, x-6):x+w+6]
            if object_crop.size:
                row_cells.append(tile(object_crop, f"class {code} | gt {int(row[1])}", 252, 87))
        if row_cells:
            row_cells.extend([np.zeros_like(row_cells[0])]*(2-len(row_cells)))
            cells.append(np.hstack(row_cells))
    assert cv2.imwrite(str(PANELS / f"{name}_classes.jpg"), np.vstack(cells))


def run(name):
    PANELS.mkdir(exist_ok=True)
    with ZipFile(DATA / f"{name}.zip") as archive:
        gt = np.loadtxt(archive.open(f"{name}/gt/gt.txt"), delimiter=",")
        if len(np.unique(gt[:, :2], axis=0)) != len(gt):
            raise ValueError(f"{name}: duplicate identity in frame")
        events, cars = candidate_events(gt)
        output = HERE / "candidates"
        output.mkdir(exist_ok=True)
        class_panel(archive, name, gt)
        for i in range(0, len(events), 6):
            board = np.vstack([draw_event(archive, name, event, cars) for event in events[i:i+6]])
            assert cv2.imwrite(str(PANELS / f"{name}_{i//6+1:02}.jpg"), board)
        (output / f"{name}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(name, len(events), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--wait-new"]:
        manifest = json.loads((HERE / "sources.json").read_text())
        remaining = {Path(item["path"]).stem for item in manifest["archives"]} - PRIOR
        deadline = time.monotonic() + 30 * 60
        while remaining:
            for name in sorted(remaining):
                if (DATA / f"{name}.zip").is_file():
                    run(name)
                    remaining.remove(name)
            if remaining:
                if time.monotonic() > deadline:
                    raise TimeoutError(sorted(remaining))
                time.sleep(4)
    else:
        for name in sys.argv[1:]:
            run(Path(name).stem)
