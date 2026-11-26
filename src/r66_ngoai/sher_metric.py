"""ROI and raw-field combination for R66. Scores follow TrackEval COMBINED_SEQ."""
from __future__ import annotations

import numpy as np


def roi_contains(mask, x, y, w, h):
    """Membership of the continuous box centre, using floor on the half-open image."""
    cx = np.asarray(x, dtype=float) + np.asarray(w, dtype=float) * 0.5
    cy = np.asarray(y, dtype=float) + np.asarray(h, dtype=float) * 0.5
    cx, cy = np.broadcast_arrays(cx, cy)
    valid = (cx >= 0) & (cy >= 0) & (cx < mask.shape[1]) & (cy < mask.shape[0])
    inside = np.zeros(valid.shape, dtype=bool)
    inside[valid] = mask[np.floor(cy[valid]).astype(int), np.floor(cx[valid]).astype(int)] == 1
    return inside


def combine(raw, videos=None):
    """Combine sufficient statistics across clips, with zero association at zero TP."""
    keys = list(raw) if videos is None else list(videos)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Expected a nonempty list of distinct sequences')
    records = [raw[key] for key in keys]

    def array(name):
        values = np.asarray([row[name] for row in records], dtype=float)
        if values.shape != (len(keys), 19) or not np.isfinite(values).all():
            raise ValueError(f'Invalid HOTA field {name}')
        return values

    def count(name):
        return sum(int(row[name]) for row in records)

    weights = array('HOTA_TP')
    tp = weights.sum(axis=0)
    det = tp / np.maximum(1, tp + array('HOTA_FP').sum(axis=0) + array('HOTA_FN').sum(axis=0))
    assoc = (array('AssA') * weights).sum(axis=0) / np.maximum(1, tp)
    idtp, idfn, idfp = (count(k) for k in ('IDTP', 'IDFN', 'IDFP'))
    ctp, fp, fn, idsw = (count(k) for k in ('CLR_TP', 'CLR_FP', 'CLR_FN', 'IDSW'))
    return {'HOTA': float(np.sqrt(det * assoc).mean()), 'DetA': float(det.mean()),
            'AssA': float(assoc.mean()), 'IDF1': 2 * idtp / max(1, 2 * idtp + idfn + idfp),
            'MOTA': (ctp - fp - idsw) / max(1, ctp + fn), 'IDSW': idsw, 'FP': fp, 'FN': fn}
