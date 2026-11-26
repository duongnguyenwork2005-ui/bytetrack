"""VisDrone dropObjects.m ignore-region geometry (MATLAB 1-based indexing)."""
from __future__ import annotations

import numpy as np


def _matlab_rounded_positive(boxes: np.ndarray) -> np.ndarray:
    # MATLAB round(x) rounds positive half-integers upward. Values <= 1 are
    # clamped to 1 by dropObjects.m, so this also covers negative coordinates.
    return np.maximum(1, np.floor(np.asarray(boxes, dtype=float) + 0.5).astype(np.int64))


def ignored_fraction(boxes: np.ndarray, regions: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return the exact area ratio used by VisDrone's dropObjects.m.

    Both arguments are N x 4 xywh arrays. The source builds an inclusive
    region map, then queries its integral image at (x,y) and (x+w,y+h).
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    regions = np.asarray(regions, dtype=float).reshape(-1, 4)
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), dtype=float)
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")

    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, w, h in _matlab_rounded_positive(regions):
        # Region endpoints are inclusive in MATLAB's 1-based coordinates.
        mask[y - 1:min(height, y + h), x - 1:min(width, x + w)] = 1
    integral = mask.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)

    pos = _matlab_rounded_positive(boxes)
    x = np.clip(pos[:, 0], 1, width)
    y = np.clip(pos[:, 1], 1, height)
    w = pos[:, 2]
    h = pos[:, 3]
    x2 = np.minimum(width, x + w)
    y2 = np.maximum(1, np.minimum(height, y + h))
    covered = (integral[y - 1, x - 1] + integral[y2 - 1, x2 - 1]
               - integral[y - 1, x2 - 1] - integral[y2 - 1, x - 1])
    return covered / (w * h)
