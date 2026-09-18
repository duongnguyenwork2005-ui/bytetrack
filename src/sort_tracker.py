"""
sort_tracker.py -- SORT (Bewley et al. 2016) voi bo loc hoan doi duoc (Nhom B, Giai doan 2).

Ba bien the, CUNG association + vong doi track (SORT nguyen ban), CHI khac bo loc bbox:
    sort_orig       KF 7-D [u, v, s, r, u', v', s'] nguyen ban (s = dien tich, r = ti le), viet lai
                    bang numpy (filterpy khong cai; ma goc GPL khong sao chep) - DOI CHUNG nguyen ban
    sort_cv         KalmanFilterXYAH cua ultralytics (dung bo loc cua ByteTrack-CV) - de co lap
                    motion model khi so voi sort_ekf_ctrv (cung tham so hoa xyah, cung nhieu)
    sort_ekf_ctrv   EKF-CTRV cua de tai (src/ekf_ctrv.py) + khoi tao 2 khung hinh nhu CTRVSTrack

Vong doi SORT nguyen ban (khac ByteTrack):
    - moi frame: predict moi track -> IoU(track, det) -> Hungarian -> loai cap IoU < iou_threshold
    - det khong khop -> track moi; track khong khop: time_since_update += 1
    - xoa track khi time_since_update > max_age (mac dinh 1: mat 1 frame la xoa - khong bac cau che khuat)
    - xuat track khi time_since_update < 1 va (hit_streak >= min_hits hoac frame <= min_hits)
    - KHONG co vong association 2, KHONG co track_buffer, KHONG dung diem tin cay khi ghep
    - detection dau vao: conf > det_thresh (nguyen ban khong loc; o day mac dinh 0.25 = dieu kien A)
Tham so nguyen ban: iou_threshold 0.3, max_age 1, min_hits 3. Ban trong repo ByteTrack dung
max_age 30 -> de lam ablation "SORT + bo dem" (sort_max_age trong .yaml), khong phai doi chung.

Giao dien giong BYTETracker de run_tracking.py dung chung:
    tracker = SORT(args); out = tracker.update(results, img=None) -> (N, 8) [x1,y1,x2,y2,id,score,cls,idx]
    reset(), reset_id(), multi_predict(tracks) (de ElapsedStepsMixin lap k buoc), frame_id.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from tracker_ctrv import ctrv_kwargs_from_args  # noqa: E402

from ul_backend import KalmanFilterXYAH  # noqa: E402


# ---------------------------------------------------------------------------
# Bo loc bbox (BoxFilter): initiate(xyxy) / predict() / update(xyxy) / xyxy
# ---------------------------------------------------------------------------
def xyxy_to_xyah(b):
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h])


def xyah_to_xyxy(m):
    cx, cy, a, h = (float(v) for v in m[:4])
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class SortKF7:
    """KF nguyen ban cua SORT: state [u, v, s, r, du, dv, ds], quan sat [u, v, s, r].

    Tham so (filterpy mac dinh + doi nhu ma goc): R = I4, R[2:,2:] *= 10; P = I7, P[4:,4:] *= 1000,
    P *= 10; Q = I7, Q[-1,-1] *= 0.01, Q[4:,4:] *= 0.01. F: van toc hang; guard s + ds <= 0 -> ds = 0.
    """

    def __init__(self):
        self.F = np.eye(7); self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7)); self.H[np.arange(4), np.arange(4)] = 1.0
        self.R = np.eye(4); self.R[2:, 2:] *= 10.0
        self.Q = np.eye(7); self.Q[-1, -1] *= 0.01; self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7); self.P = np.eye(7)

    @staticmethod
    def to_z(b):
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2, y1 + h / 2, w * h, w / h])

    def initiate(self, xyxy):
        self.x = np.zeros(7); self.x[:4] = self.to_z(xyxy)
        self.P = np.eye(7); self.P[4:, 4:] *= 1000.0; self.P *= 10.0

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, xyxy):
        z = self.to_z(xyxy)
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, (self.P @ self.H.T).T).T
        self.x = self.x + K @ (z - self.H @ self.x)
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T     # dang Joseph (filterpy)

    @property
    def xyxy(self):
        s, r = max(float(self.x[2]), 1e-6), max(float(self.x[3]), 1e-6)
        w = np.sqrt(s * r); h = s / w
        return np.array([self.x[0] - w / 2, self.x[1] - h / 2, self.x[0] + w / 2, self.x[1] + h / 2])


class XYAHFilter:
    """Bo loc KF-CV cua ultralytics (giong ByteTrack-CV) bọc thanh BoxFilter."""

    def __init__(self):
        self.kf = KalmanFilterXYAH(); self.mean = None; self.cov = None

    def initiate(self, xyxy):
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(xyxy))

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)

    def update(self, xyxy):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(xyxy))

    @property
    def xyxy(self):
        return xyah_to_xyxy(self.mean)


class CTRVFilter:
    """EKF-CTRV cua de tai bọc thanh BoxFilter, kem khoi tao 2 khung hinh nhu CTRVSTrack.

    `frame` (so buoc da troi qua ke tu quan sat truoc) do track cung cap -> n_frames dung ca khi
    SORT co max_age > 1 (track duoc update lai sau vai frame mat).
    """

    def __init__(self, **ctrv_kwargs):
        self.kf = EKFTrackerCTRV(**ctrv_kwargs)
        self.mean = None; self.cov = None
        self._motion_initialized = False
        self._last_obs_xy = None
        self._steps_since_obs = 0

    def initiate(self, xyxy):
        z = xyxy_to_xyah(xyxy)
        self.mean, self.cov = self.kf.initiate(z)
        self._last_obs_xy = (float(z[0]), float(z[1]))
        self._motion_initialized = False
        self._steps_since_obs = 0

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self._steps_since_obs += 1

    def update(self, xyxy):
        z = xyxy_to_xyah(xyxy)
        if not self._motion_initialized:
            self.mean, self.cov, ok = self.kf.try_initiate_from_motion(
                self.mean, self.cov, z, n_frames=max(1, self._steps_since_obs), ref_pos=self._last_obs_xy)
            self._motion_initialized = bool(ok)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, z)
        self._last_obs_xy = (float(z[0]), float(z[1]))
        self._steps_since_obs = 0

    @property
    def xyxy(self):
        cx, cy, a, h = self.mean[0], self.mean[1], self.mean[5], self.mean[6]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


FILTERS = {"sort_orig": SortKF7, "sort_cv": XYAHFilter, "sort_ekf_ctrv": CTRVFilter}


# ---------------------------------------------------------------------------
# Track + tracker
# ---------------------------------------------------------------------------
class SORTTrack:
    _count = 0

    def __init__(self, box_filter, xyxy, score, cls, idx):
        SORTTrack._count += 1
        self.track_id = SORTTrack._count
        self.filter = box_filter
        self.filter.initiate(xyxy)
        # VONG DOI Y HET upstream abewley/sort (third_party/sort_upstream/sort.py, commit bce9f0d,
        # KalmanBoxTracker.__init__): hits/hit_streak khoi tao 0 va CHI tang khi co update ghep.
        # (Ban truoc dat = 1 -> track duoc xuat som hon upstream DUNG MOT FRAME; loi do review doc lap
        #  phat hien, xem results/phase2/REVIEW_moc2_codex.md va src/test_sort_upstream_parity.py.)
        self.hits = 0; self.hit_streak = 0; self.age = 0; self.time_since_update = 0
        self.score = float(score); self.cls = cls; self.idx = idx

    def predict(self):
        self.filter.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, xyxy, score, cls, idx):
        self.filter.update(xyxy)
        self.hits += 1; self.hit_streak += 1; self.time_since_update = 0
        self.score = float(score); self.cls = cls; self.idx = idx

    @property
    def xyxy(self):
        return self.filter.xyxy


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (n, m) giua hai tap box xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def associate(dets_xyxy: np.ndarray, trks_xyxy: np.ndarray, iou_threshold: float):
    """Ghep cap nhu SORT nguyen ban: Hungarian tren -IoU (neu ma tran nguong khong don anh),
    roi loai cap IoU < nguong. Tra ve (matches (k,2) [det, trk], unmatched_dets, unmatched_trks)."""
    if len(trks_xyxy) == 0 or len(dets_xyxy) == 0:
        return np.empty((0, 2), int), np.arange(len(dets_xyxy)), np.arange(len(trks_xyxy))
    iou = iou_batch(dets_xyxy, trks_xyxy)
    a = (iou > iou_threshold).astype(int)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched = np.stack(np.where(a), axis=1)
    else:
        r, c = linear_sum_assignment(-iou)
        matched = np.stack([r, c], axis=1)
    keep = iou[matched[:, 0], matched[:, 1]] >= iou_threshold
    matches = matched[keep]
    um_d = np.setdiff1d(np.arange(len(dets_xyxy)), matches[:, 0])
    um_t = np.setdiff1d(np.arange(len(trks_xyxy)), matches[:, 1])
    return matches, um_d, um_t


class SORT:
    """SORT voi bo loc chon qua args.tracker_type (sort_orig | sort_cv | sort_ekf_ctrv)."""

    def __init__(self, args):
        self.args = args
        self.filter_cls = FILTERS[args.tracker_type]
        self.iou_threshold = float(getattr(args, "sort_iou_threshold", 0.3))
        self.max_age = int(getattr(args, "sort_max_age", 1))
        self.min_hits = int(getattr(args, "sort_min_hits", 3))
        self.det_thresh = float(getattr(args, "det_thresh", getattr(args, "track_high_thresh", 0.25)))
        self.tracks: list[SORTTrack] = []
        self.frame_id = 0
        self.reset_id()

    def _new_filter(self):
        if self.filter_cls is CTRVFilter:
            return CTRVFilter(**ctrv_kwargs_from_args(self.args))
        return self.filter_cls()

    @staticmethod
    def reset_id():
        SORTTrack._count = 0

    def reset(self):
        self.tracks = []; self.frame_id = 0; self.reset_id()

    def multi_predict(self, tracks):
        for t in tracks:
            t.predict()

    def update(self, results, img=None, feats=None, **kwargs) -> np.ndarray:
        self.frame_id += 1
        xyxy = np.asarray(results.xyxy, dtype=np.float64).reshape(-1, 4)
        conf = np.asarray(results.conf, dtype=np.float64).reshape(-1)
        cls = np.asarray(results.cls).reshape(-1)
        keep = conf > self.det_thresh
        idx_all = np.flatnonzero(keep)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        self.multi_predict(self.tracks)
        trks = np.array([t.xyxy for t in self.tracks]).reshape(-1, 4)
        valid = np.isfinite(trks).all(axis=1) if len(trks) else np.zeros(0, bool)
        for t, ok in zip(self.tracks, valid):       # track NaN -> coi nhu khong khop (nhu ma goc xoa hang NaN)
            if not ok:
                t.time_since_update = max(t.time_since_update, self.max_age + 1)
        matches, um_d, um_t = associate(xyxy, np.where(valid[:, None], trks, -1e6) if len(trks) else trks,
                                        self.iou_threshold)
        for d, k in matches:
            self.tracks[k].update(xyxy[d], conf[d], cls[d], int(idx_all[d]))
        for d in um_d:
            self.tracks.append(SORTTrack(self._new_filter(), xyxy[d], conf[d], cls[d], int(idx_all[d])))

        out = []
        alive = []
        for t in self.tracks:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_id <= self.min_hits):
                b = t.xyxy
                out.append([*b.tolist(), t.track_id, t.score, float(t.cls), float(t.idx)])
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive
        return np.asarray(out, dtype=np.float32).reshape(-1, 8)


SORT_TRACKERS = {name: SORT for name in FILTERS}
