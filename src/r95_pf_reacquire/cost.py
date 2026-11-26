"""Opt-in conservative PF particle rescue for stranded Lost track/detection pairs."""
from __future__ import annotations

import numpy as np

from ul_backend import TrackState


def posterior_similarity(track, detection) -> float:
    """Weighted bounded kernel on centre and box size, without GT information."""
    x = np.asarray(track.particles, dtype=np.float64)
    logw = np.asarray(track.logw, dtype=np.float64)
    box = np.asarray(detection.tlwh, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 7 or len(x) == 0 or logw.shape != (len(x),):
        return 0.0
    if not np.isfinite(x).all() or not np.isfinite(logw).all() or not np.isfinite(box).all():
        return 0.0
    if box.shape != (4,) or box[2] <= 0 or box[3] <= 0:
        return 0.0
    h = np.maximum(x[:, 6], 1e-3)
    w = np.maximum(x[:, 5] * h, 1e-3)
    dcx, dcy = box[0] + box[2] / 2, box[1] + box[3] / 2
    # Half the mean box width/height gives a dimensionless spatial kernel.
    sx = np.maximum((w + box[2]) / 4, 1.0)
    sy = np.maximum((h + box[3]) / 4, 1.0)
    q = ((x[:, 0] - dcx) / sx) ** 2 + ((x[:, 1] - dcy) / sy) ** 2
    q += (np.log(w / box[2]) / 0.5) ** 2 + (np.log(h / box[3]) / 0.5) ** 2
    weights = np.exp(logw - np.max(logw))
    weights /= weights.sum()
    return float(np.dot(weights, np.exp(-0.5 * q)))


def _clear_best(scores: np.ndarray, best_ix: int) -> bool:
    """Best entry must exceed every alternative by a factor of 1.5."""
    best = float(scores[best_ix])
    if best <= 0:
        return False
    second = max((float(v) for i, v in enumerate(scores) if i != best_ix), default=0.0)
    return best >= 1.5 * second


def rescue_costs(tracks, detections, baseline: np.ndarray, frame_id: int,
                 match_thresh: float = 0.8, max_lost: int = 8):
    """Only change mutually unique, initially unmatchable Lost/detection pairs.

    ``baseline`` is the post-fuse-score ByteTrack cost matrix. Returned costs
    are also post-fuse-score, so the caller must not fuse them a second time.
    """
    costs = np.asarray(baseline).copy()
    if not len(tracks) or not len(detections):
        return costs, []
    if costs.shape != (len(tracks), len(detections)):
        raise ValueError("association matrix shape mismatch")
    rows = [i for i, track in enumerate(tracks)
            if track.state == TrackState.Lost
            and 1 <= int(frame_id) - int(track.end_frame) <= max_lost
            and np.all(costs[i, :] > match_thresh)]
    cols = [j for j in range(len(detections)) if np.all(costs[:, j] > match_thresh)]
    if not rows or not cols:
        return costs, []
    sim = np.array([[posterior_similarity(tracks[i], detections[j]) for j in cols] for i in rows])
    rescued = []
    for ri, i in enumerate(rows):
        cj = int(np.argmax(sim[ri]))
        j = cols[cj]
        similarity = float(sim[ri, cj])
        score = float(detections[j].score)
        if not np.isfinite(score) or similarity < 0.55 or similarity * score <= 1 - match_thresh:
            continue
        if int(np.argmax(sim[:, cj])) != ri:
            continue
        if not _clear_best(sim[ri], cj) or not _clear_best(sim[:, cj], ri):
            continue
        new_cost = 1.0 - similarity * score
        costs[i, j] = new_cost
        rescued.append((i, j, similarity, new_cost))
    return costs, rescued
