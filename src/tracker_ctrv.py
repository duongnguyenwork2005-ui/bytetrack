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
    # hoac configs/bytetrack_ukf_ctrv.yaml cho UKF

File yaml chi khac bytetrack.yaml o dong `tracker_type`, moi tham so ghep cap
(track_high_thresh, match_thresh, track_buffer...) giu nguyen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402
from ekf_ctrv_clamped import EKFTrackerCTRVClamped  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402

from lost_association import LostAwareAssociation  # noqa: E402

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
        # Vi tri QUAN SAT gan nhat (cx, cy) - moc de tinh do dich chuyen khi
        # khoi tao chuyen dong. Khac vi tri trong `mean` (da qua predict).
        self._last_obs_xy: tuple[float, float] | None = None

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
    # Override 2: multi_predict - dung bo loc CUA TRACKER (khong phai shared_kalman)
    # ------------------------------------------------------------------
    @classmethod
    def multi_predict(cls, stracks):
        """Du doan hang loat cho nhieu track.

        LOI DA SUA (Giai doan 2). Ban cu dung `cls.shared_kalman` - mot instance
        tao MOT LAN luc import voi tham so mac dinh trong config.py. Nhung
        `BYTETracker.update()` CHI goi `multi_predict` (khong goi `predict` tung
        track), nen moi tham so nhieu dat tren bo loc cua tracker (tu
        `get_kalmanfilter()`, vd doc tu file .yaml) BI BO QUA HOAN TOAN o buoc
        du doan. Da do duoc: dat std_theta = 1.0 tren bo loc cua tracker, sau
        multi_predict P[theta,theta] van tang dung bang 0.010^2 cua mac dinh.
        Voi cau hinh mac dinh cua Giai doan 1 hai bo loc trung nhau nen ket qua
        cu KHONG bi anh huong - xem test_filter_config.py (bit-identical).

        Nay dung `stracks[0].kalman_filter`: moi track cua CUNG mot tracker deu
        duoc `activate()` voi cung mot instance (`BYTETracker.kalman_filter`),
        nen lay cua track dau tien la du. `shared_kalman` chi con la du phong
        cho truong hop track chua co bo loc (khong xay ra trong pool cua ByteTrack).
        """
        if not stracks:
            return
        kf = stracks[0].kalman_filter
        if kf is None:
            kf = cls.shared_kalman
        multi_mean = np.asarray([st.mean for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][cls.IDX_VH] = 0
        multi_mean, multi_covariance = kf.multi_predict(multi_mean, multi_covariance)
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
        """Uoc luong v va theta tu quan sat thu hai. THU LAI cho den khi thanh cong.

        Khong co buoc nay, bo loc CTRV se bi ket cung khi khoi tao v = 0
        (moi phan tu Jacobian lien quan theta deu ti le voi v) - chi tiet va
        bang chung thuc nghiem xem ekf_ctrv.py muc 5 va unit test 9c.

        HAI DIEM PHAI DUNG (xem src/repro_init_covariance.py):
        1. CHI danh dau da xong khi bo loc BAO thanh cong. Ban cu luon dat co
           `_motion_initialized = True`, ke ca khi xe dung yen nen bo loc tra ve
           nguyen trang. Hau qua: xe dung yen luc dau roi moi chay se KHONG BAO
           GIO duoc khoi tao huong. Do duoc: UKF ket han o v = -0,015 px/frame
           sau 6 frame chay that (dung phai ~8), EKF hoc sai thanh v = 1,88 va
           theta = 127,6 do (dung phai 90).
        2. Dung VI TRI QUAN SAT truoc do lam moc, khong dung vi tri trong `mean`.
           `mean` la vi tri DA PREDICT; khi thu lai o cac frame sau, v co the da
           khac 0 nen predict lam dich vi tri, va `dx, dy` se thanh phan du sau
           predict chu khong phai do dich chuyen that.
        """
        if self._motion_initialized or self.mean is None:
            return
        z = self.convert_coords(new_track.tlwh)
        ref = self._last_obs_xy
        # `self.frame_id` la frame cua quan sat gan nhat truoc do, cung la frame
        # ung voi `_last_obs_xy` -> khoang cach frame dung bang hieu hai so nay.
        n_frames = max(1, frame_id - self.frame_id)
        self.mean, self.covariance, ok = self.kalman_filter.try_initiate_from_motion(
            self.mean, self.covariance, z, n_frames=n_frames, ref_pos=ref
        )
        self._motion_initialized = bool(ok)

    def _remember_obs(self, new_track: STrack) -> None:
        """Ghi lai vi tri QUAN SAT vua nhan, de lan khoi tao sau do moc dung."""
        z = self.convert_coords(new_track.tlwh)
        self._last_obs_xy = (float(z[0]), float(z[1]))

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self._motion_initialized = False
        # Quan sat dau tien cua track chinh la moc cho lan khoi tao chuyen dong
        self._last_obs_xy = (float(self.mean[self.IDX_CX]),
                             float(self.mean[self.IDX_CY]))

    def update(self, new_track: STrack, frame_id: int):
        self._init_motion_if_needed(new_track, frame_id)
        super().update(new_track, frame_id)
        self._remember_obs(new_track)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        self._init_motion_if_needed(new_track, frame_id)
        super().re_activate(new_track, frame_id, new_id)
        self._remember_obs(new_track)


class ClampedSTrack(CTRVSTrack):
    """STrack dung EKF+CTRV co chan omega (xem ekf_ctrv_clamped.py).

    Them vao de KIEM CHUNG xem khuyet diem "omega khong bi chan" co phai la
    nguyen nhan khien CTRV thua CV khong. Ban goc CTRVSTrack giu nguyen.
    """

    shared_kalman = EKFTrackerCTRVClamped()
    filter_class = EKFTrackerCTRVClamped


class UKFSTrack(CTRVSTrack):
    """STrack dung UKF thay cho EKF.

    Chi can doi 2 thuoc tinh: bo cuc trang thai, cach doc tlwh, cach xu ly
    khoi tao 2 khung hinh... tat ca deu ke thua tu CTRVSTrack va giu nguyen.
    """

    shared_kalman = UKFTrackerCTRV()
    filter_class = UKFTrackerCTRV


class PFSTrack(CTRVSTrack):
    """STrack dung Particle Filter CTRV (Giai doan 2). Xem pf_ctrv.py.

    Trang thai cua track la DAM MAY particle (`particles` (N,9), `logw` (N,)) + rng rieng.
    `mean` / `covariance` la uoc luong tong hop (dung cho tlwh -> IoU association, va cho
    project / gating_distance). Sau activate(), `self.kalman_filter` la PFTrackView -> cac
    ham STrack.update / re_activate / _init_motion_if_needed cua lop cha chay nguyen ven
    nhung tac dong len particle cua chinh track nay.
    """

    shared_kalman = None            # KHONG dung: PF phai di qua bo loc cua tracker (co seed)
    filter_class = PFCTRV

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particles: np.ndarray | None = None
        self.logw: np.ndarray | None = None
        self.rng: np.random.Generator | None = None

    def activate(self, kalman_filter, frame_id: int):
        # Lop cha: gan kalman_filter (PFCTRV dung chung), cap track_id, mean/cov = initiate(z)
        # -> mean tai frame activate la XAC DINH (= quan sat), giong KF/EKF/UKF.
        super().activate(kalman_filter, frame_id)
        pf = kalman_filter
        self.rng = pf.make_rng(self.track_id)               # SeedSequence([seed, video, track_id])
        self.particles = pf.sample_initial(self.mean, self.covariance, self.rng)
        self.logw = np.full(pf.num_particles, -np.log(pf.num_particles))
        self.kalman_filter = PFTrackView(pf, self)

    def predict(self):
        lost = self.state != TrackState.Tracked
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance, lost=lost)

    @classmethod
    def multi_predict(cls, stracks):
        """Du doan hang loat: gom particle cua MOI track -> mot lan f_batch (vector hoa), nhieu
        qua trinh lay tu rng CUA TUNG TRACK (ket qua khong phu thuoc thu tu track trong pool,
        va trung khit voi goi predict() tung track)."""
        if not stracks:
            return
        pf = stracks[0].kalman_filter.pf
        VH, H = pf.VH, pf.H
        Xs = []
        for st in stracks:
            X = st.particles
            if st.state != TrackState.Tracked and np.any(X[:, VH] != 0.0):
                X = X.copy()
                X[:, VH] = 0.0
            Xs.append(X)
        allX = np.concatenate(Xs, axis=0)
        allF = pf.f_batch(allX)
        std = pf.process_noise_std_batch(np.maximum(allX[:, H], 1e-3))
        off = 0
        for st in stracks:
            n = len(st.particles)
            Xn = allF[off:off + n] + st.rng.standard_normal((n, pf.NDIM)) * std[off:off + n]
            Xn[:, pf.THETA] = normalize_angle(Xn[:, pf.THETA])
            Xn[:, H] = np.maximum(Xn[:, H], 1e-3)
            st.particles = Xn
            st.mean, st.covariance = pf.estimate_state(Xn, st.logw)
            off += n


#: Khoa trong file .yaml cua tracker -> ten tham so cua EKFTrackerCTRV.__init__.
#: Khong co khoa nao trong .yaml -> bo loc dung mac dinh config.py (hanh vi cu).
CTRV_YAML_KEYS = {
    "ctrv_dt": "dt",
    "ctrv_std_theta": "std_theta",
    "ctrv_std_omega": "std_omega",
    "ctrv_init_std_theta": "init_std_theta",
    "ctrv_init_std_omega": "init_std_omega",
    "ctrv_heading_std_after_init": "heading_std_after_init",
    "ctrv_min_speed_for_heading": "min_speed_for_heading",
}


def ctrv_kwargs_from_args(args) -> dict:
    """Doc cac khoa `ctrv_*` (neu co) tu namespace cau hinh tracker.

    Chi lay khoa co mat va khac None, de file .yaml cu (khong co khoa nao) van
    tao bo loc y het truoc day.
    """
    out = {}
    for key, name in CTRV_YAML_KEYS.items():
        val = getattr(args, key, None)
        if val is not None:
            out[name] = float(val)
    return out


class CTRVByteTracker(LostAwareAssociation, BYTETracker):
    """ByteTracker dung bo loc CTRV. Toan bo logic ghep cap giu nguyen.

    Ke thua them `LostAwareAssociation` (Giai doan B): ghi de ham chi phi lien
    ket CHI cho track dang LOST, bat/tat qua file .yaml. Mac dinh TAT -> hanh vi
    trung khit ban goc.
    """

    track_class = CTRVSTrack

    def get_kalmanfilter(self):
        """Tra ve bo loc CTRV thay vi KalmanFilterXYAH.

        Tham so nhieu / khoi tao lay tu file .yaml (khoa `ctrv_*`, xem
        CTRV_YAML_KEYS); khong co khoa -> mac dinh config.py. Instance nay duoc
        gan cho MOI track khi activate() va la bo loc duy nhat chay trong
        multi_predict (xem CTRVSTrack.multi_predict).
        """
        return self.track_class.filter_class(**ctrv_kwargs_from_args(self.args))

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
class UKFByteTracker(CTRVByteTracker):
    """ByteTracker dung UKF + CTRV."""

    track_class = UKFSTrack


#: Khoa .yaml cua Particle Filter -> tham so PFCTRV.__init__
PF_YAML_KEYS = {
    "pf_num_particles": ("num_particles", int),
    "pf_seed": ("seed", int),
    "pf_resample_ess_ratio": ("resample_ess_ratio", float),
    "pf_estimate": ("estimate", str),
    "pf_roughening": ("roughening", float),
    "pf_degenerate_maha2": ("degenerate_maha2", float),
}


class PFByteTracker(CTRVByteTracker):
    """ByteTracker dung Particle Filter + CTRV (Giai doan 2). Association giu nguyen 100%.

    Tham so PF doc tu .yaml (PF_YAML_KEYS), tham so CTRV (ctrv_*) nhu EKF/UKF. Output phu thuoc
    seed: runner goi set_seed(seed, video) SAU reset() (reset() tao lai bo loc).
    """

    track_class = PFSTrack

    def get_kalmanfilter(self):
        kw = {}
        for key, (name, typ) in PF_YAML_KEYS.items():
            val = getattr(self.args, key, None)
            if val is not None:
                kw[name] = typ(val)
        return PFCTRV(**kw, **ctrv_kwargs_from_args(self.args))

    def set_seed(self, seed: int, video: str = "") -> None:
        self.kalman_filter.set_seed(seed, video)

    def pf_stats(self) -> dict:
        kf = self.kalman_filter
        return {"n_update": kf.n_update, "n_resample": kf.n_resample, "n_degenerate": kf.n_degenerate,
                "num_particles": kf.num_particles, "seed": kf.seed}


class CVByteTracker(LostAwareAssociation, BYTETracker):
    """Baseline KF + CV, CHI khac ban goc o ham chi phi lien ket cho track LOST.

    BAT BUOC phai co lop nay cho Giai doan B: neu chi bat DIoU/expanded gate cho
    EKF/UKF ma de baseline chay IoU goc thi ta da doi HAI bien cung luc (motion
    model VA ham chi phi), pha vo nguyen tac co lap bien cua toan de tai.
    Voi ca hai co TAT, lop nay hanh xu trung khit `BYTETracker` goc.
    """

    pass


class ClampedByteTracker(CTRVByteTracker):
    """ByteTracker dung EKF + CTRV co chan omega."""

    track_class = ClampedSTrack


#: Ten tracker_type -> lop tracker tuong ung (dang ky vao ultralytics).
CTRV_TRACKERS = {
    "bytetrack_ekf_ctrv": CTRVByteTracker,
    "bytetrack_ukf_ctrv": UKFByteTracker,
    "bytetrack_cv_lostassoc": CVByteTracker,
    "bytetrack_ekf_ctrv_clamped": ClampedByteTracker,
    "bytetrack_pf_ctrv": PFByteTracker,
}


def register() -> None:
    """Dang ky cac tracker CTRV vao TRACKER_MAP cua ultralytics.

    Phai goi TRUOC khi goi model.track(...), neu khong ultralytics se bao
    'Only [...] are supported for now' vi khong biet tracker_type moi.
    """
    from ultralytics.trackers import track as _track_mod

    _track_mod.TRACKER_MAP.update(CTRV_TRACKERS)
