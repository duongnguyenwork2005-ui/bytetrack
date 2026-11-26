"""Find car tracks that cross two fixed occluders in the left-turn camera."""

from zipfile import ZipFile

import cv2
import numpy as np

from occlusion_candidates import DATA, HERE, PANELS, draw_event
from run import score

SEQUENCE = "task_day_left_turn"
REGIONS = {"median_pole": (888, 215, 914, 470), "bridge_support": (1637, 254, 1805, 479)}


def candidates(cars):
    found = []
    for name, rect in REGIONS.items():
        top_left, bottom_right = np.array(rect[:2]), np.array(rect[2:])
        for identity in np.unique(cars[:, 1]):
            rows = cars[cars[:, 1] == identity]
            rows = rows[np.argsort(rows[:, 0])]
            lt, rb = rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]
            center = (lt + rb) / 2
            shared = np.maximum(0, np.minimum(rb, bottom_right) - np.maximum(lt, top_left))
            overlap = shared.prod(axis=1) / rows[:, 4:6].prod(axis=1)
            inside = (overlap >= .20) & (rows[:, 5] >= 10)
            cuts = np.flatnonzero(np.diff(rows[inside, 0]) != 1) + 1
            for indexes in np.split(np.flatnonzero(inside), cuts):
                if not 3 <= len(indexes) <= 180:
                    continue
                first, last = int(rows[indexes[0], 0]), int(rows[indexes[-1], 0])
                before, after = first - 10, last + 10
                a, b = np.searchsorted(rows[:, 0], [before, after])
                if b >= len(rows) or rows[a, 0] != before or rows[b, 0] != after:
                    continue
                low, high = sorted((center[a, 0], center[b, 0]))
                if max(overlap[a], overlap[b]) >= .15 or not (low < top_left[0] and high > bottom_right[0]):
                    continue
                peak = int(rows[indexes[overlap[indexes].argmax()], 0])
                found.append({"car": int(identity), "blocker": name, "before": before, "peak": peak,
                              "after": after, "first": first, "last": last})
    found.sort(key=lambda e: (e["blocker"], e["before"], e["car"]))
    for number, row in enumerate(found, 1):
        row["number"] = number
    return found


def run():
    with ZipFile(DATA / f"{SEQUENCE}.zip") as archive:
        gt = np.loadtxt(archive.open(f"{SEQUENCE}/gt/gt.txt"), delimiter=",")
        cars = gt[gt[:, 7] == 6]
        assert (cars[:, 4:6] > 0).all()
        lookup = {(int(row[1]), int(row[0])): row for row in cars}
        events = candidates(cars)
        for i in range(0, len(events), 6):
            board = np.vstack([draw_event(archive, SEQUENCE, event, lookup) for event in events[i:i+6]])
            assert cv2.imwrite(str(PANELS / f"fixed_{i//6+1:02}.jpg"), board)
    score.write_json(HERE / "fixed_candidates.json", {"sequence": SEQUENCE, "regions": REGIONS, "events": events})
    print(len(events), flush=True)


if __name__ == "__main__":
    run()
