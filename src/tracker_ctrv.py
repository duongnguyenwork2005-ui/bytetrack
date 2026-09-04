"""
tracker_ctrv.py -- Ghep bo loc CTRV (EKF / UKF) vao ByteTrack cua ultralytics.

MUC TIEU: thay DUY NHAT motion model, giu nguyen 100% phan detection va data
association, de phep so sanh o Giai doan 5 la cong bang.


===========================================================================
VI SAO CHI CAN OVERRIDE 3 CHO?
===========================================================================
Da doc va doi chieu truc tiep ma nguon ultralytics/trackers/byte_tracker.py:
trong toan bo file chi co DUNG 3 vi tri cham vao bo cuc vector trang thai:

    dong  82 : mean_state[7] = 0        (trong STrack.predict)
    dong  94 : multi_mean[i][7] = 0     (trong STrack.multi_predict)
    dong 167 : ret = self.mean[:4]      (trong property STrack.tlwh)

Chi so 7 la `vh` trong bo cuc baseline [x, y, a, h, vx, vy, va, vh].
Trong bo cuc CTRV [cx, cy, v, theta, omega, a, h, va, vh] thi `vh` la chi so 8.

Ngoai ra `ultralytics/trackers/utils/matching.py` (phan ghep cap track <->
detection) KHONG he doc `mean` hay `covariance` - no chi dung IoU tren bbox
dang tlwh. Vi vay thay motion model KHONG anh huong data association.


===========================================================================
VE VIEC "XOA vh KHI TRACK BI MAT"
===========================================================================
Baseline dat vh = 0 khi track khong o trang thai Tracked (dang bi mat), de
bbox khong tu phinh to / co lai vo han trong luc khong co detection.
Ta lam Y HET (chi doi chi so 7 -> 8).

CO Y KHONG xoa `omega`: khi xe bi che khuat giua khuc cua, viec tiep tuc quay
theo omega da hoc duoc chinh la uu the ma de tai muon do luong. Xoa omega se
lam CTRV thoai hoa thanh CV va triet tieu toan bo y nghia cua thi nghiem.


===========================================================================
CACH DUNG VOI ultralytics
===========================================================================
    import tracker_ctrv
    tracker_ctrv.register()                       # dang ky vao TRACKER_MAP
    model.track(frame, tracker="configs/bytetrack_ekf_ctrv.yaml", persist=True)

File yaml chi khac bytetrack.yaml o dong `tracker_type`, moi tham so ghep cap
(track_high_thresh, match_thresh, track_buffer...) giu nguyen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402

from ultralytics.trackers.basetrack import TrackState  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker, STrack  # noqa: E402


class CTRVSTrack(STrack):
    """STrack dung bo loc CTRV thay cho Kalman Filter tuyen tinh.

    Bo cuc trang thai: [cx, cy, v, theta, omega, a, h, va, vh]  (9 chieu)
    thay vi baseline:  [x, y, a, h, vx, vy, va, vh]             (8 chieu)
    """

    # Bo loc dung chung cho multi_predict. Lop con (UKF o Giai doan 4) chi can
    # ghi de 2 thuoc tinh nay la du.
    shared_kalman = EKFTrackerCTRV()
    filter_class = EKFTrackerCTRV

    # Chi so trong vector trang thai
    IDX_CX, IDX_CY, IDX_A, IDX_H = 0, 1, 5, 6
    IDX_VH = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Co danh dau da uoc luong duoc huong tu 2 quan sat dau tien chua.
        # Xem muc 5 trong docstring cua ekf_ctrv.py.
        self._motion_initialized = False

    # ------------------------------------------------------------------
    # Override 1: predict - doi chi so vh tu 7 sang 8
    # ------------------------------------------------------------------
    def predict(self):
        """Du doan trang thai ke tiep bang bo loc CTRV."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            # Track dang bi mat: khoa vh lai (giong baseline) nhung VAN GIU omega
            mean_state[self.IDX_VH] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    # ------------------------------------------------------------------
    # Override 2: multi_predict - dung shared_kalman cua lop nay
    # ------------------------------------------------------------------
    @classmethod
    def multi_predict(cls, stracks):
        """Du doan hang loat cho nhieu track.

        Khai bao la classmethod (baseline la staticmethod) de lop con UKF tu
        dong dung dung `shared_kalman` cua no ma khong phai viet lai ham nay.
        """
        if not stracks:
            return
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][cls.IDX_VH] = 0
        multi_mean, multi_covariance = cls.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    # ------------------------------------------------------------------
    # Override 3: tlwh - doc dung chi so cua bo cuc CTRV
    # ------------------------------------------------------------------
    @property
    def tlwh(self) -> np.ndarray:
        """Bbox dang (top-left x, top-left y, width, height) tu trang thai hien tai.

        Baseline doc mean[:4] vi bo cuc cua no la [x, y, a, h, ...].
        Bo cuc CTRV dat a, h o chi so 5, 6 nen phai lay rieng.
        """
        if self.mean is None:
            return self._tlwh.copy()
        cx = self.mean[self.IDX_CX]
        cy = self.mean[self.IDX_CY]
        a = self.mean[self.IDX_A]
        h = self.mean[self.IDX_H]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h])

    # ------------------------------------------------------------------
    # Hook khoi tao 2 khung hinh
    # ------------------------------------------------------------------
    def _init_motion_if_needed(self, new_track: STrack, frame_id: int) -> None:
        """Uoc luong v va theta tu quan sat thu hai (chi chay dung 1 lan).

        Khong co buoc nay, bo loc CTRV se bi ket cung khi khoi tao v = 0
        (moi phan tu Jacobian lien quan theta deu ti le voi v) - chi tiet va
        bang chung thuc nghiem xem ekf_ctrv.py muc 5 va unit test 9c.
        """
        if self._motion_initialized or self.mean is None:
            return
        n_frames = max(1, frame_id - self.frame_id)
        z = self.convert_coords(new_track.tlwh)
        self.mean, self.covariance = self.kalman_filter.initiate_from_motion(
            self.mean, self.covariance, z, n_frames=n_frames
        )
        self._motion_initialized = True

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self._motion_initialized = False

    def update(self, new_track: STrack, frame_id: int):
        self._init_motion_if_needed(new_track, frame_id)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        self._init_motion_if_needed(new_track, frame_id)
        super().re_activate(new_track, frame_id, new_id)


class CTRVByteTracker(BYTETracker):
    """ByteTracker dung bo loc CTRV. Toan bo logic ghep cap giu nguyen."""

    track_class = CTRVSTrack

    def get_kalmanfilter(self):
        """Tra ve bo loc CTRV thay vi KalmanFilterXYAH."""
        return self.track_class.filter_class()

    def multi_predict(self, tracks):
        """Goi multi_predict cua lop track tuong ung.

        Bat buoc phai override: ham goc cua BYTETracker goi thang
        `STrack.multi_predict(tracks)` (hard-code ten lop), nen neu khong
        override thi no se dung Kalman Filter cua baseline chu khong phai CTRV.
        """
        self.track_class.multi_predict(tracks)


# ---------------------------------------------------------------------------
# Dang ky vao ultralytics
# ---------------------------------------------------------------------------
#: Ten tracker_type -> lop tracker. Giai doan 4 se them "bytetrack_ukf_ctrv".
CTRV_TRACKERS = {
    "bytetrack_ekf_ctrv": CTRVByteTracker,
}


def register() -> None:
    """Dang ky cac tracker CTRV vao TRACKER_MAP cua ultralytics.

    Phai goi TRUOC khi goi model.track(...), neu khong ultralytics se bao
    'Only [...] are supported for now' vi khong biet tracker_type moi.
    """
    from ultralytics.trackers import track as _track_mod

    _track_mod.TRACKER_MAP.update(CTRV_TRACKERS)
