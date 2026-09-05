"""
ekf_ctrv.py -- Extended Kalman Filter voi mo hinh chuyen dong CTRV
               (Constant Turn Rate and Velocity).

Class `EKFTrackerCTRV` duoc thiet ke de THAY THE TRUC TIEP lop
`KalmanFilterXYAH` cua ultralytics trong ByteTrack: cung ten phuong thuc,
cung y nghia tham so (initiate / predict / multi_predict / project / update /
gating_distance). Nho vay phan DETECTION va DATA ASSOCIATION khong phai sua
mot dong nao - xem src/tracker_ctrv.py.


===========================================================================
1. VECTOR TRANG THAI (9 chieu)
===========================================================================
        x = [ cx, cy, v, theta, omega, a, h, va, vh ]^T
        chi so:  0   1  2    3      4   5  6   7   8

    cx, cy : tam bounding box                       [px]
    v      : toc do dai theo huong theta            [px/frame]
    theta  : huong di chuyen                        [rad]
    omega  : toc do quay (turn rate)                [rad/frame]
    a      : ty le khung  a = w/h                   [-]
    h      : chieu cao bounding box                 [px]
    va, vh : dao ham cua a va h                     [1/frame], [px/frame]

VI SAO CHIA LAM 2 KHOI?
    - Khoi (cx, cy, v, theta, omega): mo hinh CTRV - PHI TUYEN. Day chinh la
      thu duy nhat khac baseline.
    - Khoi (a, h, va, vh): van la Constant Velocity TUYEN TINH, GIU NGUYEN
      100% giong baseline KalmanFilterXYAH cua ultralytics (ke ca cac he so
      nhieu 1/20, 1/160, 1e-2, 1e-5).

    Muc dich: co lap dung MOT bien nghien cuu. Neu doi luon ca dong hoc kich
    thuoc bbox thi khi ket qua thay doi ta se khong biet la do CTRV hay do
    thay doi kia. Baseline dung state 8 chieu [x, y, a, h, vx, vy, va, vh];
    ta thay 4 chieu dau tien lien quan chuyen dong tam (x, y, vx, vy) bang 5
    chieu CTRV (cx, cy, v, theta, omega), giu nguyen (a, h, va, vh).


===========================================================================
2. HAM CHUYEN TRANG THAI f(x)  -- mo hinh CTRV
===========================================================================
Gia thiet cua CTRV: trong 1 buoc dt, xe di voi toc do dai v KHONG DOI va
toc do quay omega KHONG DOI => quy dao la mot CUNG TRON.

Dat:
        s0 = sin(theta),              c0 = cos(theta)
        s1 = sin(theta + omega*dt),   c1 = cos(theta + omega*dt)

TRUONG HOP |omega| > eps  (xe dang quay - quy dao la cung tron ban kinh R = v/omega):

        cx' = cx + (v/omega) * ( s1 - s0 )
        cy' = cy + (v/omega) * ( c0 - c1 )
        v'     = v
        theta' = theta + omega*dt
        omega' = omega

    Dan xuat: van toc la ( v*cos(theta(t)), v*sin(theta(t)) ) voi
    theta(t) = theta + omega*t. Lay tich phan tu 0 den dt:
        cx' - cx = integral( v*cos(theta + omega*t) )dt
                 = (v/omega) * [ sin(theta + omega*t) ]  tu 0 den dt
                 = (v/omega) * ( s1 - s0 )                      (dpcm)
        cy' - cy = integral( v*sin(theta + omega*t) )dt
                 = (v/omega) * [ -cos(theta + omega*t) ] tu 0 den dt
                 = (v/omega) * ( c0 - c1 )                      (dpcm)

TRUONG HOP |omega| <= eps  (xe di gan nhu thang - CHIA CHO 0!):

        cx' = cx + v*c0*dt
        cy' = cy + v*s0*dt

    Day chinh la GIOI HAN cua cong thuc tren khi omega -> 0. Chung minh bang
    khai trien Taylor cua s1 quanh omega = 0:
        s1 = s0 + omega*dt*c0 - (omega*dt)^2/2 * s0 + O(omega^3)
        => (v/omega)(s1 - s0) = v*dt*c0 - (v*omega*dt^2/2)*s0 + O(omega^2)
        khi omega -> 0 thi ve phai -> v*dt*c0                    (dpcm)

    Neu khong xu ly rieng truong hop nay, cong thuc (v/omega) se cho ra
    inf/NaN va lam hong toan bo bo loc.


===========================================================================
3. MA TRAN JACOBIAN F = df/dx  (9x9)
===========================================================================
EKF tuyen tinh hoa f quanh diem uoc luong hien tai roi truyen hiep phuong sai:

        P' = F * P * F^T + Q

Vi f khong tron lan 2 khoi, F co dang KHOI DUONG CHEO:

        F = [ F_ctrv (5x5)      0      ]
            [      0       F_size (4x4)]

--- 3a. Khoi CTRV, truong hop |omega| > eps ---

    d(cx')/d(cx)    = 1
    d(cx')/d(v)     = ( s1 - s0 ) / omega
    d(cx')/d(theta) = (v/omega) * ( c1 - c0 )
    d(cx')/d(omega) = -(v/omega^2)*( s1 - s0 ) + (v/omega)*dt*c1

    d(cy')/d(cy)    = 1
    d(cy')/d(v)     = ( c0 - c1 ) / omega
    d(cy')/d(theta) = (v/omega) * ( s1 - s0 )
    d(cy')/d(omega) = -(v/omega^2)*( c0 - c1 ) + (v/omega)*dt*s1

    d(v')/d(v) = 1
    d(theta')/d(theta) = 1 ,  d(theta')/d(omega) = dt
    d(omega')/d(omega) = 1

    (Luu y dau: d(c1)/d(theta) = -s1 va d(c1)/d(omega) = -s1*dt,
     nen d(c0 - c1)/d(theta) = -s0 + s1 = s1 - s0
     va  d(c0 - c1)/d(omega) = + s1*dt .)

--- 3b. Khoi CTRV, truong hop |omega| <= eps (gioi han) ---

    d(cx')/d(v)     = c0*dt
    d(cx')/d(theta) = -v*s0*dt
    d(cx')/d(omega) = -0.5 * v * dt^2 * s0      <-- KHONG phai 0!
    d(cy')/d(v)     = s0*dt
    d(cy')/d(theta) = v*c0*dt
    d(cy')/d(omega) = +0.5 * v * dt^2 * c0      <-- KHONG phai 0!

    Hai dao ham theo omega lay tu so hang bac 1 cua khai trien Taylor o muc 2:
        cx' = cx + v*dt*c0 - (1/2)*v*omega*dt^2*s0 + O(omega^2)
        cy' = cy + v*dt*s0 + (1/2)*v*omega*dt^2*c0 + O(omega^2)
    Neu dat 2 dao ham nay = 0 (loi thuong gap), bo loc se KHONG BAO GIO hoc
    duoc omega khi xe dang di thang -> khong the phat hien luc xe bat dau re.

--- 3c. Khoi kich thuoc (tuyen tinh, giong baseline) ---

        a' = a + va*dt ,  h' = h + vh*dt ,  va' = va ,  vh' = vh
        F_size = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]


===========================================================================
4. MO HINH DO (measurement model) -- TUYEN TINH
===========================================================================
        z = [cx, cy, a, h]      (DUNG Y HET baseline)

        z = H * x  voi H la ma tran CHON (4x9), chi lay chi so 0, 1, 5, 6:

        H = [1 0 0 0 0 0 0 0 0]
            [0 1 0 0 0 0 0 0 0]
            [0 0 0 0 0 1 0 0 0]
            [0 0 0 0 0 0 1 0 0]

HE QUA QUAN TRONG: vi h(x) tuyen tinh nen Jacobian do H la HANG SO, va buoc
`update` cua EKF TRUNG KHIT voi buoc update cua KF chuan (khong co xap xi nao).
=> Toan bo tinh phi tuyen (va do do toan bo su khac biet so voi baseline)
   nam GON trong buoc `predict`. Day chinh la dieu de tai muon do luong.
=> Ngoai ra do khong gian do y het baseline, ByteTrack van ghep cap
   track <-> detection bang IoU tren cung mot loai bbox => phan data
   association hoan toan khong bi anh huong.


===========================================================================
5. KHOI TAO 2 KHUNG HINH (two-frame initialization) -- BAT BUOC voi CTRV
===========================================================================
VAN DE: khi track vua sinh ra ta chi co 1 quan sat, khong the biet huong.
Neu dat v = 0 va theta = 0 nhu cach baseline dat vx = vy = 0 thi bo loc
CTRV bi KET CUNG, ly do:

    - Cac phan tu Jacobian lien quan theta deu ti le voi v:
          d(cx')/d(theta) = -v*s0*dt ,  d(cy')/d(theta) = v*c0*dt
      => v = 0 lam ca hai bang 0 => theta khong bao gio nhan duoc thong tin
         tu sai so vi tri => theta dung yen mai o gia tri khoi tao.
    - Voi theta = 0 thi d(cx')/d(v) = 1 nhung d(cy')/d(v) = 0
      => v chi "cam nhan" duoc chuyen dong theo truc x.
    => Mot xe di THANG DUNG theo truc y se khong bao gio hoc duoc v lan theta:
       bo loc thoai hoa thanh "dung yen theo y".

    Baseline KF khong gap loi nay vi CV la tuyen tinh: vx va vy doc lap,
    moi chieu tu hoc duoc tu sai so vi tri cua chinh no.

CACH XU LY (chuan trong tai lieu ve CTRV/CTRA): uoc luong v va theta tu
HAI quan sat dau tien - xem `initiate_from_motion()`. Day KHONG phai la
"uu ai" cho EKF, ma la sua mot khiem khuyet cua cach tham so hoa toa do cuc;
sau buoc nay ca hai bo loc deu xuat phat tu cung mot luong thong tin
(2 quan sat dau). Cung cach xu ly nay se duoc dung lai cho UKF o Giai doan 4
de dam bao cong bang.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def normalize_angle(a):
    """Dua goc ve khoang [-pi, pi).

    Can thiet vi theta cong don omega*dt moi frame se tang vo han neu khong
    goi lai, gay mat on dinh so hoc khi tinh sin/cos va khi so sanh goc.
    """
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class EKFTrackerCTRV:
    """Extended Kalman Filter, mo hinh chuyen dong CTRV, thay the KalmanFilterXYAH.

    Interface giong het `ultralytics.trackers.utils.kalman_filter.KalmanFilterXYAH`:
        initiate(measurement)                   -> (mean[9], cov[9,9])
        predict(mean, cov)                      -> (mean[9], cov[9,9])
        multi_predict(mean[N,9], cov[N,9,9])    -> batch
        project(mean, cov, confidence=None)     -> (mean[4], cov[4,4])
        update(mean, cov, measurement, conf)    -> (mean[9], cov[9,9])
        gating_distance(...)                    -> khoang cach Mahalanobis

    Do do co the gan thang vao ByteTrack ma khong sua phan association.
    """

    # Chi so cac bien trong vector trang thai - dat ten de code de doc
    CX, CY, V, THETA, OMEGA, A, H, VA, VH = range(9)
    NDIM = 9

    def __init__(self, dt: float | None = None):
        self.dt = config.CTRV_DT if dt is None else dt
        self.omega_eps = config.CTRV_OMEGA_EPS

        # He so nhieu LAY Y HET baseline ultralytics de dam bao cong bang
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # Nhieu qua trinh cho 2 bien rieng cua CTRV (khong co trong baseline),
        # dat theo so lieu omega thuc do o Giai doan 1 - xem config.py
        self._std_theta = config.CTRV_STD_THETA
        self._std_omega = config.CTRV_STD_OMEGA

        # Ma tran do H (4x9): chon cx, cy, a, h. Hang so vi mo hinh do tuyen tinh.
        self._update_mat = np.zeros((4, self.NDIM))
        self._update_mat[0, self.CX] = 1.0
        self._update_mat[1, self.CY] = 1.0
        self._update_mat[2, self.A] = 1.0
        self._update_mat[3, self.H] = 1.0

    # ------------------------------------------------------------------
    # Ham chuyen trang thai f(x) va Jacobian F
    # ------------------------------------------------------------------
    def f(self, x: np.ndarray) -> np.ndarray:
        """Ham chuyen trang thai CTRV. Xem muc 2 trong docstring dau file."""
        dt = self.dt
        cx, cy, v, theta, omega = x[self.CX], x[self.CY], x[self.V], x[self.THETA], x[self.OMEGA]
        a, h, va, vh = x[self.A], x[self.H], x[self.VA], x[self.VH]

        c0, s0 = np.cos(theta), np.sin(theta)

        if abs(omega) > self.omega_eps:
            # --- Quy dao cung tron ---
            theta1 = theta + omega * dt
            c1, s1 = np.cos(theta1), np.sin(theta1)
            cx_new = cx + (v / omega) * (s1 - s0)
            cy_new = cy + (v / omega) * (c0 - c1)
        else:
            # --- Gioi han omega -> 0: di thang deu (tranh chia cho 0) ---
            theta1 = theta + omega * dt
            cx_new = cx + v * c0 * dt
            cy_new = cy + v * s0 * dt

        out = np.empty_like(x)
        out[self.CX] = cx_new
        out[self.CY] = cy_new
        out[self.V] = v
        out[self.THETA] = normalize_angle(theta1)
        out[self.OMEGA] = omega
        # Khoi kich thuoc: Constant Velocity tuyen tinh (giong baseline)
        out[self.A] = a + va * dt
        out[self.H] = h + vh * dt
        out[self.VA] = va
        out[self.VH] = vh
        return out

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Ma tran Jacobian F = df/dx (9x9). Xem muc 3 trong docstring dau file."""
        dt = self.dt
        v, theta, omega = x[self.V], x[self.THETA], x[self.OMEGA]
        c0, s0 = np.cos(theta), np.sin(theta)

        F = np.eye(self.NDIM)

        if abs(omega) > self.omega_eps:
            theta1 = theta + omega * dt
            c1, s1 = np.cos(theta1), np.sin(theta1)

            F[self.CX, self.V] = (s1 - s0) / omega
            F[self.CX, self.THETA] = (v / omega) * (c1 - c0)
            F[self.CX, self.OMEGA] = -(v / omega**2) * (s1 - s0) + (v / omega) * dt * c1

            F[self.CY, self.V] = (c0 - c1) / omega
            F[self.CY, self.THETA] = (v / omega) * (s1 - s0)
            F[self.CY, self.OMEGA] = -(v / omega**2) * (c0 - c1) + (v / omega) * dt * s1
        else:
            # Gioi han omega -> 0, lay tu khai trien Taylor bac 2 (xem muc 3b).
            F[self.CX, self.V] = c0 * dt
            F[self.CX, self.THETA] = -v * s0 * dt
            F[self.CX, self.OMEGA] = -0.5 * v * dt**2 * s0

            F[self.CY, self.V] = s0 * dt
            F[self.CY, self.THETA] = v * c0 * dt
            F[self.CY, self.OMEGA] = 0.5 * v * dt**2 * c0

        F[self.THETA, self.OMEGA] = dt
        # Khoi kich thuoc tuyen tinh
        F[self.A, self.VA] = dt
        F[self.H, self.VH] = dt
        return F

    # ------------------------------------------------------------------
    # Nhieu qua trinh Q va nhieu do R
    # ------------------------------------------------------------------
    def _process_noise(self, h: float) -> np.ndarray:
        """Ma tran nhieu qua trinh Q (9x9), duong cheo.

        cx, cy, a, h, va, vh dung Y HET cong thuc cua baseline (ti le theo
        chieu cao h). theta va omega dung hang so do duoc o Giai doan 1.
        """
        std = np.empty(self.NDIM)
        std[self.CX] = self._std_weight_position * h
        std[self.CY] = self._std_weight_position * h
        std[self.V] = self._std_weight_velocity * h      # v cung don vi px/frame nhu vx, vy
        std[self.THETA] = self._std_theta
        std[self.OMEGA] = self._std_omega
        std[self.A] = 1e-2                                # giong baseline
        std[self.H] = self._std_weight_position * h
        std[self.VA] = 1e-5                               # giong baseline
        std[self.VH] = self._std_weight_velocity * h
        return np.diag(np.square(std))

    # ------------------------------------------------------------------
    # Interface tuong thich KalmanFilterXYAH
    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray):
        """Tao track moi tu 1 quan sat (cx, cy, a, h).

        v, theta, omega deu chua biet -> dat 0 nhung gan phuong sai LON.
        Buoc `initiate_from_motion()` (goi o lan cap nhat dau tien) moi la
        cho thuc su uoc luong duoc v va theta - xem muc 5 trong docstring.
        """
        measurement = np.asarray(measurement, dtype=np.float64)
        cx, cy, a, h = measurement

        mean = np.zeros(self.NDIM)
        mean[self.CX] = cx
        mean[self.CY] = cy
        mean[self.V] = 0.0
        mean[self.THETA] = 0.0
        mean[self.OMEGA] = 0.0
        mean[self.A] = a
        mean[self.H] = h
        mean[self.VA] = 0.0
        mean[self.VH] = 0.0

        std = np.empty(self.NDIM)
        # cx, cy, h, va, vh: giong het baseline
        std[self.CX] = 2 * self._std_weight_position * h
        std[self.CY] = 2 * self._std_weight_position * h
        std[self.V] = 10 * self._std_weight_velocity * h   # tuong ung vx, vy cua baseline
        std[self.THETA] = config.CTRV_INIT_STD_THETA        # huong hoan toan chua biet
        std[self.OMEGA] = config.CTRV_INIT_STD_OMEGA
        std[self.A] = 1e-2
        std[self.H] = 2 * self._std_weight_position * h
        std[self.VA] = 1e-5
        std[self.VH] = 10 * self._std_weight_velocity * h

        return mean, np.diag(np.square(std))

    def initiate_from_motion(self, mean: np.ndarray, covariance: np.ndarray,
                             measurement: np.ndarray, n_frames: float = 1.0):
        """Uoc luong v va theta tu 2 quan sat dau tien (xem muc 5 docstring).

        Goi DUY NHAT MOT LAN, ngay truoc lan `update()` dau tien cua track.

        Args:
            mean, covariance: trang thai hien tai (sau `initiate`, truoc update)
            measurement: quan sat thu hai (cx, cy, a, h)
            n_frames: so frame da troi qua giua 2 quan sat. Thuong = 1, nhung
                neu track bi mat dau vai frame roi moi khop lai thi phai chia
                dung so frame do, khong se uoc luong toc do qua cao.

        Returns:
            (mean, covariance) da duoc gan v, theta va thu hep phuong sai tuong ung.
            Neu vat the gan nhu dung yen (dich chuyen < nguong) thi GIU NGUYEN,
            vi luc do huong khong the uoc luong dang tin cay.
        """
        mean = mean.copy()
        covariance = covariance.copy()

        dx = float(measurement[0]) - mean[self.CX]
        dy = float(measurement[1]) - mean[self.CY]
        speed = float(np.hypot(dx, dy)) / (self.dt * max(n_frames, 1.0))

        if speed < config.CTRV_MIN_SPEED_FOR_HEADING:
            return mean, covariance   # dung yen -> chua the biet huong

        mean[self.V] = speed
        mean[self.THETA] = normalize_angle(np.arctan2(dy, dx))

        # Da co thong tin -> thu hep do bat dinh cua v va theta
        covariance[self.THETA, self.THETA] = config.CTRV_HEADING_STD_AFTER_INIT ** 2
        covariance[self.V, self.V] = (2 * self._std_weight_velocity * mean[self.H] * 10) ** 2
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Buoc du doan cua EKF:  x' = f(x) ,  P' = F P F^T + Q."""
        F = self.jacobian(mean)
        Q = self._process_noise(mean[self.H])

        new_mean = self.f(mean)
        new_cov = F @ covariance @ F.T + Q
        return new_mean, new_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Du doan cho nhieu track cung luc (mean: [N,9], covariance: [N,9,9]).

        Cai dat bang vong lap goi `predict` cho tung track: mo hinh CTRV phi
        tuyen nen khong the dung chung mot ma tran F cho moi track (moi track
        co theta, omega, v rieng). Ma tran chi 9x9 nen chi phi khong dang ke,
        va cach viet nay de doc / de debug hon ban vector hoa.
        """
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        out_mean = np.empty_like(mean)
        out_cov = np.empty_like(covariance)
        for i in range(mean.shape[0]):
            out_mean[i], out_cov[i] = self.predict(mean[i], covariance[i])
        return out_mean, out_cov

    def _measurement_noise(self, h: float, confidence: float | None = None) -> np.ndarray:
        """Ma tran nhieu do R (4x4) cho quan sat (cx, cy, a, h).

        Lay Y HET baseline KalmanFilterXYAH.project() de khong tao ra khac biet
        nao ngoai motion model. `confidence` la co che NSA-Kalman cua baseline:
        detection cang chac chan thi nhieu do cang nho.
        """
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))
        if confidence is not None:
            R *= max(1.0 - float(confidence), 0.05)
        return R

    def project(self, mean: np.ndarray, covariance: np.ndarray, confidence: float | None = None):
        """Chieu trang thai xuong khong gian do (cx, cy, a, h)."""
        innovation_cov = self._measurement_noise(mean[self.H], confidence)
        H = self._update_mat
        proj_mean = H @ mean
        proj_cov = H @ covariance @ H.T
        return proj_mean, proj_cov + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray, confidence: float | None = None):
        """Buoc cap nhat cua EKF.

        Vi mo hinh do TUYEN TINH (z = H x), buoc nay trung khit voi KF chuan:
            S = H P H^T + R
            K = P H^T S^-1
            x = x + K (z - H x)
            P = P - K S K^T
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        H = self._update_mat

        # K = P H^T S^-1, giai bang he phuong trinh cho on dinh so hoc hon nghich dao
        kalman_gain = np.linalg.solve(projected_cov, (covariance @ H.T).T).T
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        new_mean[self.THETA] = normalize_angle(new_mean[self.THETA])
        # Ep doi xung lai de tranh tich luy sai so lam mat tinh doi xung cua P
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray,
                        only_position: bool = False, metric: str = "maha") -> np.ndarray:
        """Khoang cach Mahalanobis giua trang thai du doan va cac quan sat.

        ByteTrack mac dinh ghep cap bang IoU nen ham nay khong duoc goi, nhung
        van cai dat day du de giu tuong thich hoan toan voi interface goc.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = np.asarray(measurements, dtype=np.float64) - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T)
            return np.sum(z * z, axis=0)
        raise ValueError("Invalid distance metric")
