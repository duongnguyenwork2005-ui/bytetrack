"""Keep full-rate YOLO on low-FPS inputs and near signs of a new or lost object."""
import cv2
import numpy as np


class AdaptiveCadence:
    width = 160
    height = 90
    tile = 10

    def __init__(self, source_shape, regions, guard_lost):
        h, w = source_shape
        self.scale = np.array([self.width / w, self.height / h] * 2, dtype=np.float32)
        self.ignore = np.zeros((self.height, self.width), dtype=bool)
        for x, y, width, height in regions:
            self.cover(self.ignore, (x, y, x + width, y + height), 0)
        self.known = self.ignore.copy()
        self.previous = None
        self.observed = np.empty((0, 4), dtype=np.float32)
        self.guard_lost = guard_lost
        self.lost = False
        self.counts = dict(odd=0, outside_motion=0, recently_lost=0, skipped=0, missing_previous=0)

    def cover(self, mask, box, padding):
        x1, y1, x2, y2 = np.asarray(box) * self.scale
        left, top = np.floor((x1, y1)).astype(int) - padding
        right, bottom = np.ceil((x2, y2)).astype(int) + padding
        left, right = np.clip((left, right), 0, self.width)
        top, bottom = np.clip((top, bottom), 0, self.height)
        mask[top:bottom, left:right] = True

    def reset(self):
        self.previous = None
        self.observed = self.observed[:0]
        self.known[:] = self.ignore
        self.lost = False

    def should_skip(self, index, image):
        small = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        previous = self.previous
        self.previous = gray
        if index % 2:
            reason = "odd"
        elif previous is None:
            reason = "missing_previous"
        elif self.guard_lost and self.lost:
            reason = "recently_lost"
        else:
            changed = (cv2.absdiff(previous, gray) > 16) & ~self.known
            tiles = changed.reshape(self.height // self.tile, self.tile, self.width // self.tile, self.tile)
            reason = "outside_motion" if tiles.sum(axis=(1, 3)).max() >= 12 else "skipped"
        self.counts[reason] += 1
        return reason == "skipped"

    def observe(self, detections):
        boxes = detections[detections[:, 4] > .25, :4].copy()
        if self.guard_lost:
            self.lost = self.disappearance(self.observed * self.scale, boxes * self.scale)
        self.observed = boxes
        self.known[:] = self.ignore
        for box in boxes:
            self.cover(self.known, box, 4)

    def disappearance(self, previous, current):
        if len(previous) == 0:
            return False
        margin = 4
        inside = ((previous[:, :2] > margin).all(axis=1)
                  & (previous[:, 2] < self.width - margin) & (previous[:, 3] < self.height - margin))
        old = previous[inside]
        if len(old) == 0:
            return False
        if len(current) == 0:
            return True
        top_left = np.maximum(old[:, None, :2], current[None, :, :2])
        bottom_right = np.minimum(old[:, None, 2:], current[None, :, 2:])
        intersection = (bottom_right - top_left).clip(0).prod(axis=2)
        old_area = (old[:, 2:] - old[:, :2]).prod(axis=1)
        current_area = (current[:, 2:] - current[:, :2]).prod(axis=1)
        union = old_area[:, None] + current_area[None, :] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        return bool((iou.max(axis=1) < .15).any())


def make_scheduler(mode, cadence, fps, source_shape, regions):
    if mode not in ("off", "motion", "guarded") or cadence not in (1, 2):
        raise ValueError("Unsupported cadence configuration")
    if mode != "off" and cadence != 1:
        raise ValueError("Adaptive and fixed stride must not be combined")
    if mode == "off" or fps < 15:
        return None
    return AdaptiveCadence(source_shape, regions, mode == "guarded")
