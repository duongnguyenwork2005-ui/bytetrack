"""Opt-in PF predictive association cost for R91 exploration.

For a lost track, average IoU over its weighted predicted particles. The
experiment uses an equal blend with baseline box IoU; this is an association
change, so its score cannot be described as a pure motion-filter ablation.
"""
from __future__ import annotations

import numpy as np


def expected_particle_iou(track, detections):
    x = np.asarray(track.particles, dtype=np.float64)
    logw = np.asarray(track.logw, dtype=np.float64)
    logw = logw - np.max(logw)
    weights = np.exp(logw)
    weights /= weights.sum()
    h = np.maximum(x[:, 6], 1e-3)
    w = np.maximum(x[:, 5] * h, 1e-3)
    left = x[:, 0] - w / 2
    top = x[:, 1] - h / 2
    boxes = np.array([d.tlwh for d in detections], dtype=np.float64)
    bx, by, bw, bh = boxes.T
    iw = np.maximum(0, np.minimum((left + w)[:, None], (bx + bw)[None, :]) - np.maximum(left[:, None], bx[None, :]))
    ih = np.maximum(0, np.minimum((top + h)[:, None], (by + bh)[None, :]) - np.maximum(top[:, None], by[None, :]))
    inter = iw * ih
    union = (w * h)[:, None] + (bw * bh)[None, :] - inter
    return np.sum(weights[:, None] * inter / np.maximum(union, 1e-9), axis=0)
