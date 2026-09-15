"""
time_units.py -- Thoi gian, don vi va mo hinh nhieu cua cac bo loc (Giai doan 2).

Quyet dinh da chot voi nguoi dung: bo loc GIU DON VI BUOC (dt = 1 buoc = 1 frame
duoc xu ly), KHONG them che do "dt theo giay" rieng cho CTRV. Ly do: baseline
KF-CV cua ultralytics khong co dt (ma tran F co dt = 1 co dinh), nen neu chi
CTRV chay theo giay thi phep so sanh Nhom A them mot bien. Module nay cung cap:

    1. Bang AUDIT mo hinh nhieu + don vi tung tham so cua 4 bo loc (muc A).
    2. Quy tac doi don vi state/covariance sang px/s, rad/s CHI DE BAO CAO (muc B).
    3. ElapsedStepsMixin: phan anh so buoc da troi qua khi bo frame / khoang
       thoi gian khong deu vao buoc du doan va vao bo dem track_buffer (muc C).
    4. track_buffer theo giay -> so buoc, nhat quan giua cac FPS (muc D).

===========================================================================
A. AUDIT: dt xuat hien o dau, tung phan tu nhieu co don vi gi
===========================================================================
Ky hieu: h = chieu cao bbox [px]; "buoc" = mot lan goi predict.

--- A1. KF-CV baseline (ultralytics KalmanFilterXYAH, state [x,y,a,h,vx,vy,va,vh]) ---
    F      : x' = x + vx*dt voi dt = 1.0 HARD-CODE (kalman_filter.py dong 45-50).
    Q (Q duoc cong MOT LAN moi buoc, duong cheo, ti le theo h):
        std(x,y,h) = h/20   [px]          std(a) = 1e-2   [-]
        std(vx,vy,vh) = h/160 [px/buoc]   std(va) = 1e-5  [1/buoc]
    R      : std(x,y,h) = h/20 [px], std(a) = 1e-1; nhan (1 - conf) neu co conf.
    P0     : std(x,y,h) = 2h/20, std(vx,vy,vh) = 10h/160, std(a)=1e-2, std(va)=1e-5.
    => Moi phan tu deu la GIA SO THEO BUOC (discrete white noise), khong co
       tham so nao mang don vi giay. Van toc vx, vy [px/buoc].

--- A2. EKF-CTRV (src/ekf_ctrv.py, state [cx,cy,v,theta,omega,a,h,va,vh]) ---
    f(x)   : cx' = cx + (v/omega)(sin(theta+omega*dt) - sin theta) ..., dt = config.CTRV_DT = 1.0.
    Q      : cx, cy, h : h/20 [px]  (giong A1)       v : h/160 [px/buoc] (giong vx cua A1)
             theta     : CTRV_STD_THETA = 0.010 [rad/buoc]   -- gia so random-walk cua HUONG
             omega     : CTRV_STD_OMEGA = 0.005 [rad/buoc^2] -- gia so random-walk cua TURN RATE
             a, va, vh : giong A1.
    R, P0  : giong A1 cho cx,cy,a,h,va,vh; P0(theta) = 1.5 [rad], P0(omega) = 0.10 [rad/buoc];
             sau khoi tao 2 frame: std(theta) = 0.30 [rad], std(v) = 20h/160 [px/buoc].
    Y NGHIA: theta va omega la HAI LOAI dai luong khac nhau:
        * std_theta la nhieu cong them vao huong moi buoc, NGOAI phan omega*dt da
          giai thich (mo hinh hoa truot/nhieu do huong). Neu coi la tich phan cua
          white noise lien tuc thi phuong sai ti le voi thoi gian: std ~ sqrt(dt).
        * std_omega la muc THAY DOI cua turn rate moi buoc (gia toc goc). Gia tri
          0.005 duoc chon o Giai doan 1 tu quan sat "xe di tu 0 len 3 do/frame
          trong ~1 s" (config.py) - tuc mot GIOI HAN VAT LY cua gia toc goc, ket
          hop voi quet do nhay tren benchmark mo phong. Day KHONG PHAI mot white
          noise lien tuc co mat do pho biet truoc, nen KHONG duoc tu dong ap
          cong thuc sqrt(dt) khi doi FPS. Voi FPS moi, gia tri theo buoc phai
          duoc CHON LAI bang cung quy trinh (thong ke |d omega| theo buoc tu GT
          + quet do nhay tren dev) - xem PHASE2_REPORT.md.
        * Phan cx, cy, v, a, h dung ti le theo h giong baseline: KHONG doi theo FPS
          de giu "chi motion model la bien nghien cuu" - nhung phai ghi nhan rang
          o 5 fps mot buoc dai gap 5 lan nen cung gia tri h/20 la mot gia thiet
          nhieu KHAC ve mat vat ly (day la bien cua NHOM C, khong phai doi don vi).

--- A3. UKF-CTRV (src/ukf_ctrv.py) ---
    Ke thua Y HET f, Q, R, P0 cua A2 (Q tinh tai h TRUOC du doan, giong EKF).
    Khac EKF chi o cach lan truyen hiep phuong sai (sigma point). Khong co tham
    so thoi gian rieng. Tham so alpha/beta/kappa khong mang don vi thoi gian.

--- A4. PF-CTRV (src/pf_ctrv.py) ---
    Bootstrap PF: X' = f(X) + w, w ~ N(0, Q(h)) voi f, Q, R LAY Y HET A2 (cung
    gia thiet nhieu, chi khac cach bieu dien phan bo). Khong co tham so thoi gian
    rieng; num_particles, resample_ess_ratio, seed khong mang don vi thoi gian.

KET LUAN AUDIT
    - Khong bo loc nao co tham so "theo giay". Tat ca deu theo buoc.
    - Doi FPS  =  doi do dai mot buoc, KHONG phai doi don vi state.
    - Vi vay: (i) doi don vi chi la phep DOI CACH DOC (muc B), khong doi bo loc;
      (ii) viec chon lai nhieu cho FPS moi la mot buoc TUNING/THI NGHIEM co ghi
      chep, khong phai quy doi cong thuc; (iii) "che do giay" chua ton tai nen
      KHONG co khang dinh tuong duong buoc/giay nao trong bao cao.

===========================================================================
B. DOI DON VI SANG px/s, rad/s (chi de bao cao / phan tich)
===========================================================================
Voi fps la tan suat cac buoc (frame duoc xu ly), Delta t = 1/fps [s]:
        v [px/buoc] -> v * fps [px/s]           omega [rad/buoc] -> omega * fps [rad/s]
        va [1/buoc] -> va * fps [1/s]           vh [px/buoc]     -> vh * fps [px/s]
        (cx, cy, theta, a, h khong doi)
Hiep phuong sai doi bang phep bien doi TUONG HOP  P' = S P S^T,  S = diag(s_i),
s_i = fps cho cac bien van toc / turn rate, 1 cho bien con lai. Phep nay doi
NHAT QUAN ca phan tu ngoai duong cheo (vd Cov(v, omega) nhan fps^2,
Cov(cx, v) nhan fps) va bao toan tinh ban xac dinh duong (dinh luat quan tinh
Sylvester, xem ekf_ctrv.set_marginal_variance). Ham `convert_state_units`.

===========================================================================
C. BO FRAME / KHOANG THOI GIAN KHONG DEU
===========================================================================
BYTETracker.update() goi multi_predict DUNG MOT LAN va tang frame_id them 1 moi
lan goi, bat ke bao nhieu thoi gian da troi qua. Neu runner bo qua mot frame
(anh hong, timestamp thua) thi (a) du doan chi tien 1 buoc trong khi doi tuong
da di 2 buoc, (b) track_buffer dem theo so lan goi chu khong theo thoi gian,
(c) n_frames trong khoi tao 2 khung hinh (tracker_ctrv._init_motion_if_needed,
= frame_id - frame_id_truoc) sai -> van toc bi thoi phong.

ElapsedStepsMixin.update(results, img=None, n_steps=k):
    - frame_id += (k - 1) TRUOC khi goi update goc (goc cong them 1 -> tong +k).
    - multi_predict duoc goi k lan lien tiep: moi lan cong Q mot lan (dung voi
      gia thiet gia so theo buoc o muc A) va lan truyen f k lan (voi CTRV, lan
      truyen k buoc cung tron lien tiep = lan truyen mot cung tron k*dt).
    - Track LOST: vh bi khoa = 0 o moi buoc (giong hanh vi goc tung buoc).
    Voi k = 1 hanh vi TRUNG KHIT tracker goc (test_time_units.py).
    So buoc k = round(Delta t_thuc / Delta t_danh_nghia), >= 1 (`steps_between`).

===========================================================================
D. THOI GIAN GIU TRACK
===========================================================================
track_buffer trong .yaml la SO BUOC. De nhat quan giua cac FPS, runner nhan
`--track-buffer-sec` va doi: buoc = round(giay * fps). 1.2 s -> 30 buoc (25 fps,
= gia tri Giai doan 1) hoac 6 buoc (5 fps, BDD100K). Ghi ca hai vao metadata.
"""
from __future__ import annotations

import math

import numpy as np

#: Chi so cac bien VAN TOC / TURN RATE (nhan fps khi doi sang don vi giay)
RATE_INDICES = {
    "ctrv": [2, 4, 7, 8],   # v, omega, va, vh   trong [cx,cy,v,theta,omega,a,h,va,vh]
    "xyah": [4, 5, 6, 7],   # vx, vy, va, vh     trong [x,y,a,h,vx,vy,va,vh]
}


# ---------------------------------------------------------------------------
# B. Doi don vi
# ---------------------------------------------------------------------------
def unit_scale_vector(layout: str, fps: float, to_seconds: bool = True) -> np.ndarray:
    """Vector he so s_i de doi state/covariance giua don vi buoc va don vi giay.

    to_seconds=True : [.. /buoc] -> [.. /s]  (nhan fps cho bien van toc)
    to_seconds=False: nguoc lai (chia fps).
    """
    if layout not in RATE_INDICES:
        raise ValueError(f"layout phai la {list(RATE_INDICES)}, nhan {layout!r}")
    n = 9 if layout == "ctrv" else 8
    s = np.ones(n)
    f = float(fps) if to_seconds else 1.0 / float(fps)
    s[RATE_INDICES[layout]] = f
    return s


def convert_state_units(mean: np.ndarray, cov: np.ndarray | None, fps: float,
                        layout: str = "ctrv", to_seconds: bool = True):
    """Doi don vi cua (mean, cov) giua "theo buoc" va "theo giay" (muc B).

        mean' = S mean ,   P' = S P S^T ,   S = diag(s)

    Phep tuong hop doi nhat quan moi phan tu, ke ca ngoai duong cheo. Tra ve
    ban sao; cov=None -> chi doi mean.
    """
    s = unit_scale_vector(layout, fps, to_seconds)
    mean_out = np.asarray(mean, dtype=np.float64) * s
    if cov is None:
        return mean_out, None
    cov_out = np.asarray(cov, dtype=np.float64) * np.outer(s, s)
    return mean_out, cov_out


# ---------------------------------------------------------------------------
# C. So buoc da troi qua
# ---------------------------------------------------------------------------
def steps_between(dt_actual: float, dt_nominal: float, tol: float = 0.25) -> tuple[int, bool]:
    """So buoc danh nghia giua hai frame da xu ly.

    Tra ve (k, is_regular). k = max(1, round(dt_actual / dt_nominal)).
    is_regular = False khi ti so lech qua `tol` so voi so nguyen gan nhat (khoang
    thoi gian "le" - runner ghi lai de bao cao, van dung k).
    """
    if dt_nominal <= 0:
        raise ValueError("dt_nominal phai > 0")
    if not math.isfinite(dt_actual) or dt_actual <= 0:
        return 1, False
    ratio = dt_actual / dt_nominal
    k = max(1, int(round(ratio)))
    return k, abs(ratio - round(ratio)) <= tol


class ElapsedStepsMixin:
    """Mixin cho cac lop ke thua BYTETracker: update(..., n_steps=k) (muc C).

    Phai dung TRUOC lop tracker trong MRO:
        class MyTracker(ElapsedStepsMixin, CTRVByteTracker): ...
    Voi n_steps = 1 (mac dinh) hanh vi trung khit lop goc.
    """

    _pending_steps: int = 1

    def update(self, results, img=None, feats=None, n_steps: int = 1, **kwargs):
        k = int(n_steps)
        if k < 1:
            raise ValueError(f"n_steps phai >= 1, nhan {n_steps}")
        self._pending_steps = k
        # update goc se cong them 1 -> frame_id tong cong tang k buoc
        self.frame_id += k - 1
        try:
            return super().update(results, img, feats, **kwargs)
        finally:
            self._pending_steps = 1

    def multi_predict(self, tracks):
        # Goi predict goc k lan: moi lan lan truyen f mot buoc va cong Q mot lan.
        for _ in range(self._pending_steps):
            super().multi_predict(tracks)


def with_elapsed_steps(tracker_cls):
    """Tao lop con `Elapsed<Ten>` = ElapsedStepsMixin + tracker_cls (neu chua co)."""
    if issubclass(tracker_cls, ElapsedStepsMixin):
        return tracker_cls
    return type(f"Elapsed{tracker_cls.__name__}", (ElapsedStepsMixin, tracker_cls), {})


# ---------------------------------------------------------------------------
# D. track_buffer theo giay
# ---------------------------------------------------------------------------
def track_buffer_frames(seconds: float, fps: float) -> int:
    """Doi thoi gian giu track (giay) sang so buoc: round(seconds * fps), toi thieu 1."""
    if seconds <= 0 or fps <= 0:
        raise ValueError("seconds va fps phai > 0")
    return max(1, int(round(seconds * fps)))
