"""
pf_ctrv.py -- Particle Filter (bootstrap / SIR) voi mo hinh chuyen dong CTRV.

`PFCTRV` KE THUA `EKFTrackerCTRV` de dung CHUNG:
    - ham chuyen trang thai f(x)  (CTRV, ke ca nhanh |omega| ~ 0)      -> f_batch() vector hoa
    - ma tran nhieu qua trinh Q(h) (_process_noise)                     -> nhieu cong them moi particle
    - ma tran nhieu do R(h, conf)  (_measurement_noise)                 -> likelihood Gaussian
    - initiate(), try_initiate_from_motion() (uoc luong v, theta tu 2 quan sat)
    - project(), gating_distance() (tren (mean, cov) tong hop tu particle)
=> Cung gia thiet mo hinh + nhieu voi EKF/UKF. Khac biet DUY NHAT: cach bieu dien
   phan bo hau nghiem (dam may particle co trong so thay vi Gaussian).

===========================================================================
1. THUAT TOAN (bootstrap PF, moi track mot dam may N particle)
===========================================================================
    Khoi tao : X_i ~ N(x0, P0)  voi (x0, P0) = initiate(z)  (theta goi ve [-pi, pi))
    Du doan  : X_i <- f(X_i) + w_i,  w_i ~ N(0, Q(h_i))     (Q cheo, lay theo h cua TUNG particle)
               trong so KHONG doi (bootstrap: proposal = prior)
    Cap nhat : log w_i <- log w_i - 1/2 (z - H X_i)^T R^-1 (z - H X_i)   (R cheo -> tinh theo cot)
               chuan hoa bang log-sum-exp (on dinh so hoc; khong bao gio exp(-1e8) truc tiep)
    Tong hop : mean = sum w_i X_i  (theta: TRUNG BINH VONG atan2(sum w sin, sum w cos))
               cov  = sum w_i d_i d_i^T,  d_i = X_i - mean, thanh phan theta goi ve [-pi, pi)
    Resample : ESS = 1 / sum w_i^2 < ratio * N  -> systematic resampling (MOT so ngau nhien u ~ U[0,1/N))
               sau do trong so deu 1/N
    Roughening (tuy chon, mac dinh TAT): cong nhieu nho sau resample de tranh trung lap particle.

===========================================================================
2. CAC TRUONG HOP SUY BIEN (da xu ly, co bo dem)
===========================================================================
    - Thieu measurement (track LOST): chi du doan; dam may loang dan theo Q. vh = 0 cho moi
      particle (giong quy tac cua ByteTrack/CTRVSTrack).
    - Trong so suy bien: z cach xa moi particle (Mahalanobis^2 lon nhat > pf_degenerate_maha2)
      -> KHONG nhan trong so (se thanh 1 particle "thang" roi copy N lan), ma "dat lai vi tri":
      cx, cy, a, h cua moi particle <- z + N(0, P0), giu v/theta/omega, trong so deu.
      Dem trong `n_degenerate`. (Trong ByteTrack association IoU da chan truoc nen hiem.)
    - logsumexp voi moi loglik = -inf (khong the xay ra voi Gaussian; van chan): trong so deu.

===========================================================================
3. TAI LAP (reproducibility)
===========================================================================
    rng cua TUNG track = np.random.default_rng(SeedSequence([pf_seed, video_salt, track_id])).
    track_id do ByteTrack cap (xac dinh khi detection + association xac dinh), nen ket qua
    KHONG phu thuoc thu tu xu ly cac track trong mot frame va cung seed -> bit-identical.
    Cac buoc ngau nhien: sample_initial, propagate (nhieu), resample (1 so u), roughening,
    reseed (khoi tao 2 khung hinh, dat lai vi tri).

===========================================================================
4. TONG HOP -> BBOX CHO ASSOCIATION
===========================================================================
    ByteTrack ghep cap bang IoU tren bbox tlwh doc tu track.mean[cx, cy, a, h]. PF gan
    track.mean = uoc luong co trong so (pf_estimate = 'mean') hoac particle trong so lon nhat
    ('map', tinh TRUOC khi resample). track.covariance = hiep phuong sai thuc nghiem co trong
    so -> project()/gating_distance() (Mahalanobis) van dung dung nghia neu tracker khac can.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + float(np.log(np.sum(np.exp(a - m))))


def systematic_resample(weights: np.ndarray, u: float) -> np.ndarray:
    """Chi so particle duoc chon (N,), systematic resampling voi MOT so ngau nhien u ~ U[0, 1)."""
    n = len(weights)
    positions = (np.arange(n) + u) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0                      # chong sai so lam tron
    return np.searchsorted(cumsum, positions, side="right").clip(0, n - 1)


class PFCTRV(EKFTrackerCTRV):
    """Particle Filter CTRV. Bo loc DUNG CHUNG cho moi track cua mot tracker; trang thai
    particle cua tung track nam trong PFSTrack (tracker_ctrv.py), duoc truyen vao cac ham."""

    def __init__(self, num_particles: int = 300, seed: int = 0, resample_ess_ratio: float = 0.5,
                 estimate: str = "mean", roughening: float = 0.0, degenerate_maha2: float = 1e4,
                 **ctrv_kwargs):
        super().__init__(**ctrv_kwargs)
        if num_particles < 2:
            raise ValueError("num_particles phai >= 2")
        if estimate not in ("mean", "map"):
            raise ValueError("estimate phai la 'mean' hoac 'map'")
        self.num_particles = int(num_particles)
        self.seed = int(seed)
        self.resample_ess_ratio = float(resample_ess_ratio)
        self.estimate = estimate
        self.roughening = float(roughening)
        self.degenerate_maha2 = float(degenerate_maha2)
        self.video_salt = 0
        # bo dem chan doan (cong don tren toan bo track cua tracker)
        self.n_resample = 0
        self.n_update = 0
        self.n_degenerate = 0
        # Moc 2: chan doan theo tung update (tuy chon, khong doi hanh vi): enable_diag() -> diag_rows
        self.diag_rows: list | None = None

    def enable_diag(self) -> None:
        self.diag_rows = []

    def diag_frame(self):
        import pandas as pd
        cols = ["track_id", "frame", "kind", "ess_ratio", "resampled", "n_unique", "spread_xy", "spread_theta",
                "spread_v", "degenerate", "maha2_best", "n_lost"]
        return pd.DataFrame(self.diag_rows or [], columns=cols)

    def _diag_spread(self, X: np.ndarray, logw: np.ndarray) -> tuple[float, float, float]:
        w = np.exp(logw - _logsumexp(logw))
        mx, my = float(np.sum(w * X[:, self.CX])), float(np.sum(w * X[:, self.CY]))
        sxy = float(np.sqrt(np.sum(w * ((X[:, self.CX] - mx) ** 2 + (X[:, self.CY] - my) ** 2))))
        c, s_ = float(np.sum(w * np.cos(X[:, self.THETA]))), float(np.sum(w * np.sin(X[:, self.THETA])))
        r = min(1.0, float(np.hypot(c, s_)))
        sth = float(np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0)))      # do lech chuan vong
        mv = float(np.sum(w * X[:, self.V]))
        sv = float(np.sqrt(np.sum(w * (X[:, self.V] - mv) ** 2)))
        return sxy, sth, sv

    # ------------------------------------------------------------------
    # RNG
    # ------------------------------------------------------------------
    def set_seed(self, seed: int, video: str = "") -> None:
        self.seed = int(seed)
        self.video_salt = int(hashlib.sha1(str(video).encode()).hexdigest()[:8], 16) if video else 0

    def make_rng(self, track_id: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.video_salt, int(track_id)]))

    # ------------------------------------------------------------------
    # Ham chuyen trang thai vector hoa
    # ------------------------------------------------------------------
    def f_batch(self, X: np.ndarray) -> np.ndarray:
        """f(x) cho K particle cung luc: X (K, 9) -> (K, 9). Cung cong thuc voi EKFTrackerCTRV.f."""
        dt = self.dt
        X = np.asarray(X, dtype=np.float64)
        v, th, om = X[:, self.V], X[:, self.THETA], X[:, self.OMEGA]
        c0, s0 = np.cos(th), np.sin(th)
        th1 = th + om * dt
        turning = np.abs(om) > self.omega_eps
        om_safe = np.where(turning, om, 1.0)          # tranh chia cho 0 o nhanh khong dung
        c1, s1 = np.cos(th1), np.sin(th1)
        dx_turn = (v / om_safe) * (s1 - s0)
        dy_turn = (v / om_safe) * (c0 - c1)
        dx_lin = v * c0 * dt
        dy_lin = v * s0 * dt
        out = X.copy()
        out[:, self.CX] = X[:, self.CX] + np.where(turning, dx_turn, dx_lin)
        out[:, self.CY] = X[:, self.CY] + np.where(turning, dy_turn, dy_lin)
        out[:, self.THETA] = normalize_angle(th1)
        out[:, self.A] = X[:, self.A] + X[:, self.VA] * dt
        out[:, self.H] = X[:, self.H] + X[:, self.VH] * dt
        return out

    def process_noise_std_batch(self, h: np.ndarray) -> np.ndarray:
        """Do lech chuan nhieu qua trinh cho tung particle (K, 9) - cung cong thuc _process_noise."""
        K = len(h)
        std = np.empty((K, self.NDIM))
        std[:, self.CX] = self._std_weight_position * h
        std[:, self.CY] = self._std_weight_position * h
        std[:, self.V] = self._std_weight_velocity * h
        std[:, self.THETA] = self._std_theta
        std[:, self.OMEGA] = self._std_omega
        std[:, self.A] = 1e-2
        std[:, self.H] = self._std_weight_position * h
        std[:, self.VA] = 1e-5
        std[:, self.VH] = self._std_weight_velocity * h
        return std

    # ------------------------------------------------------------------
    # Cac buoc PF
    # ------------------------------------------------------------------
    def sample_initial(self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """N particle ~ N(mean, cov). cov cua initiate() la duong cheo -> lay can tung phan tu."""
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        X = mean[None, :] + rng.standard_normal((self.num_particles, self.NDIM)) * std[None, :]
        X[:, self.THETA] = normalize_angle(X[:, self.THETA])
        X[:, self.H] = np.maximum(X[:, self.H], 1e-3)      # chieu cao phai duong
        return X

    def propagate(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mot buoc du doan: f(X) + N(0, Q(h)). h lay cua TUNG particle (truoc khi du doan)."""
        std = self.process_noise_std_batch(np.maximum(X[:, self.H], 1e-3))
        Xn = self.f_batch(X) + rng.standard_normal(X.shape) * std
        Xn[:, self.THETA] = normalize_angle(Xn[:, self.THETA])
        Xn[:, self.H] = np.maximum(Xn[:, self.H], 1e-3)
        return Xn

    def log_likelihood(self, X: np.ndarray, z: np.ndarray, R_diag: np.ndarray) -> np.ndarray:
        """log N(z | H x_i, R) bo hang so; R cheo (4,) -> tinh theo cot. (N,)"""
        Z = X[:, [self.CX, self.CY, self.A, self.H]]
        d = Z - np.asarray(z, dtype=np.float64)[None, :]
        return -0.5 * np.sum(d * d / R_diag[None, :], axis=1)

    def estimate_state(self, X: np.ndarray, logw: np.ndarray):
        """(mean, cov) co trong so. theta: trung binh vong; residual theta goi ve [-pi, pi)."""
        w = np.exp(logw - _logsumexp(logw))
        if self.estimate == "map":
            mean = X[int(np.argmax(w))].copy()
        else:
            mean = w @ X
            th = X[:, self.THETA]
            mean[self.THETA] = np.arctan2(w @ np.sin(th), w @ np.cos(th))
        d = X - mean[None, :]
        d[:, self.THETA] = normalize_angle(d[:, self.THETA])
        cov = (d * w[:, None]).T @ d
        cov = 0.5 * (cov + cov.T)
        return mean, cov

    def ess(self, logw: np.ndarray) -> float:
        w = np.exp(logw - _logsumexp(logw))
        return 1.0 / float(np.sum(w * w))

    def resample_if_needed(self, X: np.ndarray, logw: np.ndarray, rng: np.random.Generator):
        """Systematic resampling khi ESS < ratio * N. Tra ve (X, logw, da_resample)."""
        n = len(logw)
        w = np.exp(logw - _logsumexp(logw))
        if 1.0 / float(np.sum(w * w)) >= self.resample_ess_ratio * n:
            return X, logw, False
        idx = systematic_resample(w, float(rng.random()))
        Xr = X[idx].copy()
        if self.roughening > 0.0:
            std = self.process_noise_std_batch(np.maximum(Xr[:, self.H], 1e-3)) * self.roughening
            Xr += rng.standard_normal(Xr.shape) * std
            Xr[:, self.THETA] = normalize_angle(Xr[:, self.THETA])
        self.n_resample += 1
        return Xr, np.full(n, -np.log(n)), True

    def update_particles(self, X: np.ndarray, logw: np.ndarray, z: np.ndarray, conf: float | None,
                         rng: np.random.Generator, diag_key: tuple | None = None):
        """Cap nhat trong so voi quan sat z = (cx, cy, a, h). Tra ve (X, logw, mean, cov).

        diag_key (Moc 2): (track_id, frame, n_lost) de ghi chan doan khi enable_diag() - khong doi ket qua."""
        self.n_update += 1
        z = np.asarray(z, dtype=np.float64)
        h_ref = float(max(z[3], 1e-3))              # R theo chieu cao quan sat (~ mean[H] cua EKF)
        R_diag = np.diag(self._measurement_noise(h_ref, conf))
        ll = self.log_likelihood(X, z, R_diag)
        best = float(np.max(ll)) if len(ll) else -np.inf
        maha2_best = float(-2.0 * best) if np.isfinite(best) else float("inf")
        if not np.isfinite(best) or (-2.0 * best) > self.degenerate_maha2:
            # Suy bien: khong particle nao gan quan sat -> dat lai vi tri quanh z, giu chuyen dong
            self.n_degenerate += 1
            m0, P0 = self.initiate(z)
            std0 = np.sqrt(np.diag(P0))
            for k in (self.CX, self.CY, self.A, self.H):
                X[:, k] = m0[k] + rng.standard_normal(len(X)) * std0[k]
            X[:, self.H] = np.maximum(X[:, self.H], 1e-3)
            logw = np.full(len(X), -np.log(len(X)))
            mean, cov = self.estimate_state(X, logw)
            if self.diag_rows is not None and diag_key is not None:
                sxy, sth, sv = self._diag_spread(X, logw)
                self.diag_rows.append([diag_key[0], diag_key[1], "update", 0.0, True, len(X), sxy, sth, sv,
                                       True, maha2_best, diag_key[2]])
            return X, logw, mean, cov
        logw = logw + ll
        logw -= _logsumexp(logw)
        ess_ratio = self.ess(logw) / len(logw) if self.diag_rows is not None else None
        mean, cov = self.estimate_state(X, logw)       # uoc luong TRUOC khi resample ('map' co nghia)
        X, logw, resampled = self.resample_if_needed(X, logw, rng)
        if self.diag_rows is not None and diag_key is not None:
            sxy, sth, sv = self._diag_spread(X, logw)
            n_unique = int(len(np.unique(X[:, self.CX])))          # so particle khac nhau sau resample
            self.diag_rows.append([diag_key[0], diag_key[1], "update", float(ess_ratio), bool(resampled), n_unique,
                                   sxy, sth, sv, False, maha2_best, diag_key[2]])
        return X, logw, mean, cov

    def reseed_motion(self, X: np.ndarray, speed: float, heading: float, h: float,
                      rng: np.random.Generator) -> np.ndarray:
        """Khoi tao 2 khung hinh cho particle: v ~ N(speed, sigma_v^2), theta ~ N(heading, sigma_theta^2).

        sigma_v = 2 * std_weight_velocity * h * 10 va sigma_theta = heading_std_after_init: dung
        cac phuong sai ma EKF dat sau try_initiate_from_motion (cung luong thong tin).
        """
        n = len(X)
        sig_v = 2 * self._std_weight_velocity * h * 10
        X = X.copy()
        X[:, self.V] = speed + rng.standard_normal(n) * sig_v
        X[:, self.THETA] = normalize_angle(heading + rng.standard_normal(n) * self._heading_std_after_init)
        return X


class PFTrackView:
    """Bo loc "nhin tu mot track": giu giao dien KalmanFilterXYAH (predict/update/...) de
    STrack.update / re_activate / _init_motion_if_needed cua ultralytics + CTRVSTrack goi
    KHONG can sua, nhung tac dong len dam may particle cua chinh track do.

    STrack.update goi:  self.mean, self.covariance = self.kalman_filter.update(mean, cov, z)
    -> o day update() cap nhat track.particles / track.logw va tra ve uoc luong tong hop.
    """

    def __init__(self, pf: PFCTRV, track):
        self.pf = pf
        self.track = track

    # --- cac ham STrack / CTRVSTrack goi ---
    def predict(self, mean_state, covariance, lost: bool | None = None):
        """Mot buoc du doan. Khi track LOST, quy tac cua ByteTrack (vh = 0) ap len MOI particle.
        `lost` do PFSTrack.predict truyen tuong minh (KHONG suy tu mean_state[vh] == 0: ngay
        sau activate mean[vh] = 0 mot cach xac dinh trong khi track van Tracked)."""
        t = self.track
        if lost is None:
            from ultralytics.trackers.basetrack import TrackState
            lost = t.state != TrackState.Tracked
        X = t.particles
        if lost and np.any(X[:, self.pf.VH] != 0.0):
            X = X.copy()
            X[:, self.pf.VH] = 0.0
        t.particles = self.pf.propagate(X, t.rng)
        return self.pf.estimate_state(t.particles, t.logw)

    def update(self, mean, covariance, measurement, confidence=None):
        t = self.track
        key = None
        if self.pf.diag_rows is not None:
            fid = int(getattr(t, "frame_id", -1))
            key = (int(t.track_id), fid, int(fid - t.end_frame) if t.end_frame is not None else 0)
        t.particles, t.logw, m, P = self.pf.update_particles(t.particles, t.logw,
                                                             measurement, confidence, t.rng, diag_key=key)
        return m, P

    def try_initiate_from_motion(self, mean, covariance, measurement, n_frames=1.0, ref_pos=None):
        """Dung EKF de quyet dinh (dung yen hay khong) va lay speed/heading, roi reseed particle."""
        m_ekf, P_ekf, ok = self.pf.try_initiate_from_motion(mean, covariance, measurement,
                                                            n_frames=n_frames, ref_pos=ref_pos)
        if not ok:
            return mean, covariance, False
        t = self.track
        t.particles = self.pf.reseed_motion(t.particles, float(m_ekf[self.pf.V]), float(m_ekf[self.pf.THETA]),
                                            float(max(mean[self.pf.H], 1e-3)), t.rng)
        t.logw = np.full(len(t.particles), -np.log(len(t.particles)))
        m, P = self.pf.estimate_state(t.particles, t.logw)
        if self.pf.diag_rows is not None:
            sxy, sth, sv = self.pf._diag_spread(t.particles, t.logw)
            self.pf.diag_rows.append([int(t.track_id), int(getattr(t, "frame_id", -1)), "init_motion", 1.0, False,
                                      len(t.particles), sxy, sth, sv, False, 0.0, 0])
        return m, P, True

    def initiate_from_motion(self, mean, covariance, measurement, n_frames=1.0, ref_pos=None):
        m, P, _ = self.try_initiate_from_motion(mean, covariance, measurement, n_frames, ref_pos)
        return m, P

    # --- uy quyen phan con lai cho bo loc dung chung (initiate, project, gating_distance, f, ...) ---
    def __getattr__(self, name):
        # Chi goi khi thuoc tinh KHONG co tren instance. Chan de quy khi copy/pickle
        # (deepcopy tao instance rong roi hoi 'pf' -> __getattr__('pf') -> self.pf -> ...).
        if name in ("pf", "track") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.pf, name)
