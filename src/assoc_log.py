"""
assoc_log.py -- Ghi LOG QUYET DINH GHEP cua BYTETracker (ultralytics 8.4.x) ma KHONG doi hanh vi.

Muc dich (Moc 2, muc 2): truy nguyen nhan mat ID quanh su kien che khuat. Voi moi frame ghi:
    tracks : moi track trong pool SAU buoc du doan (tracked + lost) va cac track unconfirmed:
             track_id, state (1 Tracked / 2 Lost / 4 Removed / 0 unconfirmed-new), is_activated,
             n_lost = frame_id - end_frame (so buoc da mat), tracklet_len, score cu,
             box DU DOAN (tlwh - dung cho IoU o vong ghep), matched_det (idx det trong ket qua goc, -1),
             stage (1 = vong 1 high, 2 = vong 2 low, 3 = unconfirmed, 4 = hook sau vong 1 (lost_recovery),
             0 = khong ghep)
    dets   : moi detection: det_idx (chi so trong mang detection goc), tier (high/low), conf, cls,
             box tlwh, matched_track (-1), stage, new_track_id (neu sinh track moi), reject_reason
             ('' | 'below_new_track_thresh' | 'unmatched_low' ...)
    events : (frame, kind, track_id, n_lost, det_idx): kind in {lost, refind, new, removed,
             removed_unconfirmed}
IoU / chi phi fused (1 - IoU*conf) duoc tinh lai offline tu box (xac dinh) -> khong luu ma tran.

Cach bat: `with_assoc_log(cls)` tao lop con (mixin dat TRUOC lop tracker, tuong tu
time_units.with_elapsed_steps). Runner (run_tracking.track_video) dat `tr.log_frame = frame_idx`
truoc moi update de ghi chi so frame CUA VIDEO (khac tracker.frame_id khi bo frame).
Kiem tra: output tracker co/khong log BIT-IDENTICAL (test_assoc_log.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ul_backend import TrackState  # noqa: E402

TRACK_COLS = ["frame", "tracker_frame_id", "track_id", "state", "is_activated", "n_lost", "tracklet_len",
              "score", "x", "y", "w", "h", "matched_det", "stage"]
DET_COLS = ["frame", "tracker_frame_id", "det_idx", "tier", "conf", "cls", "x", "y", "w", "h",
            "matched_track", "stage", "new_track_id", "reject_reason"]
EVENT_COLS = ["frame", "tracker_frame_id", "kind", "track_id", "n_lost", "det_idx"]


class AssocLog:
    def __init__(self):
        self.tracks: list[tuple] = []
        self.dets: list[tuple] = []
        self.events: list[tuple] = []

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (pd.DataFrame(self.tracks, columns=TRACK_COLS), pd.DataFrame(self.dets, columns=DET_COLS),
                pd.DataFrame(self.events, columns=EVENT_COLS))

    def clear(self):
        self.tracks.clear(); self.dets.clear(); self.events.clear()


def _tlwh(t) -> tuple[float, float, float, float]:
    b = np.asarray(t.tlwh, dtype=float)
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


class AssociationLogMixin:
    """Mixin ghi log; moi hook goi super() y nguyen roi doc lai trang thai -> khong doi ket qua."""

    log_frame: int = -1          # runner dat = chi so frame cua video truoc moi update

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assoc_log = AssocLog()
        self._lg_pool_state: dict[int, int] = {}
        self._lg_track_rows: dict[int, list] = {}      # track_id -> row (list) de dien matched/stage
        self._lg_det_rows: dict[tuple[str, int], list] = {}

    # ---- helpers ------------------------------------------------------------------
    def _lg_frame(self) -> tuple[int, int]:
        return int(self.log_frame), int(self.frame_id)

    def _lg_add_track(self, t, state_code: int):
        fr, fid = self._lg_frame()
        n_lost = int(fid - t.end_frame) if t.end_frame is not None else 0
        x, y, w, h = _tlwh(t)
        row = [fr, fid, int(t.track_id), state_code, bool(t.is_activated), n_lost, int(t.tracklet_len),
               float(t.score), x, y, w, h, -1, 0]
        self._lg_track_rows[int(t.track_id)] = row
        self.assoc_log.tracks.append(row)

    def _lg_add_det(self, d, tier: str):
        fr, fid = self._lg_frame()
        x, y, w, h = _tlwh(d)
        row = [fr, fid, int(d.idx), tier, float(d.score), int(d.cls) if d.cls is not None else -1,
               x, y, w, h, -1, 0, -1, ""]
        self._lg_det_rows[(tier, int(d.idx))] = row
        self.assoc_log.dets.append(row)

    def _lg_mark_matches(self, tracks, dets, tier: str, stage: int, pre_state: dict, refind_kind: bool):
        """Sau mot vong ghep: track co frame_id == self.frame_id va idx trung det -> ghep."""
        fid = int(self.frame_id)
        by_idx = {int(d.idx): d for d in dets}
        for t in tracks:
            if int(t.frame_id) != fid or int(t.track_id) not in self._lg_track_rows:
                continue
            row = self._lg_track_rows[int(t.track_id)]
            if row[13] != 0:            # da ghep o vong truoc
                continue
            d = by_idx.get(int(t.idx))
            if d is None:
                continue
            row[12], row[13] = int(d.idx), stage
            drow = self._lg_det_rows.get((tier, int(d.idx)))
            if drow is not None:
                drow[10], drow[11] = int(t.track_id), stage
            if refind_kind and pre_state.get(int(t.track_id)) == TrackState.Lost:
                self.assoc_log.events.append([row[0], fid, "refind", int(t.track_id), row[5], int(d.idx)])

    # ---- hooks --------------------------------------------------------------------
    def update(self, results, img=None, feats=None, **kwargs):
        self._lg_track_rows = {}
        self._lg_det_rows = {}
        self._lg_pool_state = {}
        return super().update(results, img, feats, **kwargs)

    def multi_predict(self, tracks):
        super().multi_predict(tracks)          # (ElapsedSteps: k lan) -> ghi box DU DOAN sau tat ca buoc
        for t in tracks:
            self._lg_pool_state[int(t.track_id)] = int(t.state)
            self._lg_add_track(t, int(t.state))

    def _first_association(self, strack_pool, detections, activated, refind):
        for d in detections:
            self._lg_add_det(d, "high")
        u_track, u_detection = super()._first_association(strack_pool, detections, activated, refind)
        self._lg_mark_matches(strack_pool, detections, "high", 1, self._lg_pool_state, refind_kind=True)
        return u_track, u_detection

    def _post_first_association(self, strack_pool, detections, u_track, u_detection, activated, refind):
        u_track, u_detection = super()._post_first_association(strack_pool, detections, u_track, u_detection,
                                                               activated, refind)
        # stage 4 = ghep o hook sau vong 1 (vd lost_recovery.LostRecoveryMixin); mac dinh khong co gi
        self._lg_mark_matches(strack_pool, detections, "high", 4, self._lg_pool_state, refind_kind=True)
        return u_track, u_detection

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        for d in detections_second:
            self._lg_add_det(d, "low")
        n_lost0 = len(lost)
        super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)
        self._lg_mark_matches(strack_pool, detections_second, "low", 2, self._lg_pool_state, refind_kind=False)
        fr, fid = self._lg_frame()
        for t in lost[n_lost0:]:
            self.assoc_log.events.append([fr, fid, "lost", int(t.track_id), 0, -1])
        for key, drow in self._lg_det_rows.items():
            if key[0] == "low" and drow[10] < 0:
                drow[13] = "unmatched_low"

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        for t in unconfirmed:
            self._lg_add_track(t, 0)                 # 0 = unconfirmed (chua activated, khong du doan)
        n_rm0 = len(removed)
        dets_left = [detections[i] for i in u_detection]
        u_detection2, dets2 = super()._unconfirmed_association(unconfirmed, u_detection, detections, activated, removed)
        self._lg_mark_matches(unconfirmed, dets_left, "high", 3, {}, refind_kind=False)
        fr, fid = self._lg_frame()
        for t in removed[n_rm0:]:
            self.assoc_log.events.append([fr, fid, "removed_unconfirmed", int(t.track_id), 0, -1])
        return u_detection2, dets2

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        n0 = len(activated)
        super()._init_new_tracks(u_detection, detections, activated, refind)
        fr, fid = self._lg_frame()
        new_by_idx = {int(t.idx): t for t in activated[n0:]}
        for inew in u_detection:
            d = detections[inew]
            drow = self._lg_det_rows.get(("high", int(d.idx)))
            t = new_by_idx.get(int(d.idx))
            if t is not None:
                if drow is not None:
                    drow[12] = int(t.track_id)
                self.assoc_log.events.append([fr, fid, "new", int(t.track_id), 0, int(d.idx)])
            elif drow is not None:
                drow[13] = "below_new_track_thresh"

    def _remove_stale_lost(self, removed):
        # ultralytics (merge_track_pools) giu track da Removed trong lost_stracks them MOT frame, nen
        # _remove_stale_lost co the danh dau lai cung track o frame ke tiep -> chi ghi lan DAU (state != Removed).
        pre = {id(t): int(t.state) for t in self.lost_stracks}
        n0 = len(removed)
        super()._remove_stale_lost(removed)
        fr, fid = self._lg_frame()
        for t in removed[n0:]:
            if pre.get(id(t)) == TrackState.Removed:
                continue
            self.assoc_log.events.append([fr, fid, "removed", int(t.track_id), int(fid - t.end_frame), -1])


def with_assoc_log(tracker_cls):
    """Lop con `Logged<Ten>` = AssociationLogMixin + tracker_cls (chi cho lop ke thua BYTETracker)."""
    if issubclass(tracker_cls, AssociationLogMixin):
        return tracker_cls
    # Lop nen phai lay tu BACKEND DANG DUNG: tren backend `np`, `tracker_cls` ke thua BYTETracker cua ban
    # VENDOR, nen so voi lop cua ultralytics se luon False (review vong 5, P2-3 - sua import thoi chua du).
    from ul_backend import BYTETracker
    if not issubclass(tracker_cls, BYTETracker):
        raise TypeError(f"assoc_log chi ho tro lop ke thua BYTETracker, nhan {tracker_cls.__name__}")
    return type(f"Logged{tracker_cls.__name__}", (AssociationLogMixin, tracker_cls), {})


def save_log(log: AssocLog, out_dir, video: str) -> None:
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tr, de, ev = log.frames()
    for name, df in (("tracks", tr), ("dets", de), ("events", ev)):
        df.to_parquet(out_dir / f"{video}_{name}.parquet", index=False, engine="fastparquet")


def load_log(log_dir, video: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from pathlib import Path
    d = Path(log_dir)
    return tuple(pd.read_parquet(d / f"{video}_{n}.parquet", engine="fastparquet") for n in ("tracks", "dets", "events"))
