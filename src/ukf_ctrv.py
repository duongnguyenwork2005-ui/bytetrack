"""
ukf_ctrv.py -- Unscented Kalman Filter voi mo hinh chuyen dong CTRV.

`UKFTrackerCTRV` KE THUA `EKFTrackerCTRV` de dung chung Y HET:
    - ham chuyen trang thai f(x)        (mo hinh CTRV, ke ca xu ly omega ~ 0)
    - ma tran nhieu qua trinh Q
    - ma tran nhieu do R  (ham project)
    - cach khoi tao initiate() va initiate_from_motion()

=> Khac biet DUY NHAT giua EKF va UKF trong de tai nay la CACH TRUYEN HIEP
   PHUONG SAI qua ham phi tuyen. Nho vay phep so sanh EKF vs UKF la tuyet doi
   cong bang (khong lan bat ky bien nao khac).


===========================================================================
1. Y TUONG: TAI SAO KHONG DUNG JACOBIAN?
===========================================================================
EKF tuyen tinh hoa f quanh MOT diem (diem uoc luong hien tai) roi dung
Jacobian de truyen hiep phuong sai:
        P' = F P F^T + Q
Cach nay bo qua moi thanh phan bac >= 2 cua f. Khi f cong manh (xe re gap,
omega lon) hoac khi do bat dinh P lon (bi che khuat lau), xap xi bac 1 sai nhieu.

UKF khong tuyen tinh hoa gi ca. Thay vao do no:
    1. Chon mot bo diem dai dien (SIGMA POINT) mo ta dung ky vong va hiep
       phuong sai cua phan bo hien tai.
    2. Cho TUNG diem di qua ham f PHI TUYEN THAT (khong xap xi).
    3. Tinh lai ky vong va hiep phuong sai tu dam may diem da bien doi.

Ket qua chinh xac toi bac 3 cua khai trien Taylor voi phan bo Gaussian
(so voi bac 1 cua EKF), va KHONG can tinh dao ham -> khong the sai Jacobian.


===========================================================================
2. SINH SIGMA POINT (Unscented Transform co ti le)
===========================================================================
Voi n = 9 chieu trang thai, sinh 2n + 1 = 19 diem:

        lambda = alpha^2 * (n + kappa) - n
        L = cholesky( (n + lambda) * P )        (ma tran tam giac duoi)

        X[0]     = x
        X[i]     = x + L[:, i-1]        voi i = 1..n
        X[n+i]   = x - L[:, i-1]        voi i = 1..n

Trong so (khac nhau cho ky vong va hiep phuong sai):

        Wm[0] = lambda / (n + lambda)
        Wc[0] = lambda / (n + lambda) + (1 - alpha^2 + beta)
        Wm[i] = Wc[i] = 1 / ( 2 * (n + lambda) )        voi i >= 1

CHON THAM SO - xem config.py muc UKF_ALPHA. Tom tat: dung alpha = 1, kappa = 0
(lambda = 0). Gia tri "mac dinh sach giao khoa" alpha = 1e-3 LAM HONG bo loc khi
n = 9 vi n + lambda -> 0 (chia cho 0, trong so no len 10^6).


===========================================================================
3. DIEM CAN CAN THAN NHAT: THETA LA GOC, KHONG PHAI SO THUC
===========================================================================
Khi tinh ky vong co trong so cua cac sigma point, thanh phan theta KHONG
duoc lay trung binh cong thong thuong.

VI DU HONG: hai goc +179 do va -179 do thuc chat chi cach nhau 2 do, nhung
trung binh cong cua chung la 0 do - lech han 180 do so voi thuc te!

CACH DUNG (trung binh vong / circular mean):

        theta_mean = atan2( sum(Wm_i * sin(theta_i)) , sum(Wm_i * cos(theta_i)) )

Tuong tu, khi tinh hieu (Y_i - x_mean) de lap hiep phuong sai, thanh phan
theta phai duoc goi ve [-pi, pi) truoc khi binh phuong, neu khong mot cap goc
sat nhau vat qua bien +-pi se tao ra sai so gia ~ 2*pi.

Day la loi kinh dien cua UKF co bien goc, tuong duong ve muc do nguy hiem voi
loi chia cho 0 tai omega ~ 0 cua EKF. Ca hai deu co unit test rieng.


===========================================================================
4. VE BUOC UPDATE
===========================================================================
Mo hinh do cua ta TUYEN TINH: z = H x voi H la ma tran chon [cx, cy, a, h].
Voi ham tuyen tinh, Unscented Transform tra ve KET QUA DUNG BANG phep bien
doi tuyen tinh thong thuong => buoc update cua UKF trung khit voi KF/EKF.

Script van cai dat DAY DU buoc update bang sigma point (dung dinh nghia UKF)
chu khong di duong tat, va co unit test kiem chung rang no khop voi buoc
update cua EKF toi sai so may (~1e-10). Nho vay:
    - Trinh bay trong luan van dung la mot UKF hoan chinh.
    - Dong thoi chung minh duoc cai dat khong co loi.

=> Moi khac biet ve KET QUA giua EKF va UKF deu den tu buoc PREDICT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402


class UKFTrackerCTRV(EKFTrackerCTRV):
    """Unscented Kalman Filter, mo hinh CTRV.

    Cung interface voi EKFTrackerCTRV (initiate / predict / multi_predict /
    project / update / gating_distance) nen hoan doi duoc trong pipeline ma
    khong sua gi o ByteTrack.
    """

    def __init__(self, dt: float | None = None,
                 alpha: float | None = None, beta: float | None = None,
                 kappa: float | None = None):
        super().__init__(dt=dt)
        self.alpha = config.UKF_ALPHA if alpha is None else alpha
        self.beta = config.UKF_BETA if beta is None else beta
        self.kappa = config.UKF_KAPPA if kappa is None else kappa

        n = self.NDIM
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        denom = n + self.lambda_

        # --- Kiem tra tham so co dung duoc khong ---
        # Che do hong KHONG phai la denom = 0 chinh xac, ma la denom RAT NHO:
        # luc do trong so Wm[0] = lambda/denom no len hang tram nghin, bo loc
        # phan ky ngay. Vi du alpha=1e-3, n=9: denom = 9e-6 (khong phai 0 nen
        # phep kiem tra "abs(denom) < 1e-9" KHONG bat duoc) nhung Wm[0] = -999999.
        # Vi vay kiem tra thang tren DO LON CUA TRONG SO.
        if denom <= 1e-9:
            raise ValueError(
                f"Tham so sigma point hong: n + lambda = {denom:.3e} <= 0 (chia cho 0 "
                f"hoac can bac hai cua so am). alpha={self.alpha}, kappa={self.kappa}, "
                f"n={n}. Xem config.py muc UKF_ALPHA."
            )
        wm0 = self.lambda_ / denom
        if abs(wm0) > 1e3:
            raise ValueError(
                f"Tham so sigma point hong: trong so Wm[0] = {wm0:.1f} qua lon "
                f"(n + lambda = {denom:.3e} qua nho) - bo loc se phan ky do chia cho 0. "
                f"alpha={self.alpha}, kappa={self.kappa}, n={n}. "
                f"Voi n = {n} hay dung alpha = 1.0, kappa = 0.0. Xem config.py muc UKF_ALPHA."
            )
        self._n_plus_lambda = denom

        # Trong so cho ky vong (Wm) va hiep phuong sai (Wc)
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * denom))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / denom
        self.Wc[0] = self.lambda_ / denom + (1.0 - self.alpha**2 + self.beta)

    # ------------------------------------------------------------------
    # Sinh sigma point
    # ------------------------------------------------------------------
    def sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """Sinh 2n+1 sigma point tu (mean, covariance). Tra ve mang (2n+1, n).

        Dung phan tich Cholesky cua (n + lambda) * P. Neu P mat tinh xac dinh
        duong do sai so tich luy, them mot luong jitter nho tang dan tren duong
        cheo cho toi khi phan tich duoc - bo loc chay tiep thay vi sap.
        """
        n = self.NDIM
        P = 0.5 * (covariance + covariance.T)   # ep doi xung truoc khi phan tich
        scaled = self._n_plus_lambda * P

        jitter = 0.0
        for _ in range(6):
            try:
                L = np.linalg.cholesky(scaled + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-9 * float(np.trace(scaled)) / n + 1e-12)
        else:
            # Truong hop cuc doan: dung tri rieng, ep cac tri rieng am ve 0
            w, V = np.linalg.eigh(scaled)
            L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))

        sigmas = np.empty((2 * n + 1, n))
        sigmas[0] = mean
        for i in range(n):
            sigmas[1 + i] = mean + L[:, i]
            sigmas[1 + n + i] = mean - L[:, i]
        return sigmas

    # ------------------------------------------------------------------
    # Trung binh / hieu co xet bien goc
    # ------------------------------------------------------------------
    def _state_mean(self, sigmas: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Ky vong co trong so cua cac sigma point, dung TRUNG BINH VONG cho theta.

        Xem muc 3 trong docstring dau file: lay trung binh cong cho goc se sai
        hoan toan khi cac goc vat qua bien +-pi.
        """
        mean = weights @ sigmas          # trung binh thong thuong cho moi thanh phan
        th = sigmas[:, self.THETA]
        mean[self.THETA] = np.arctan2(weights @ np.sin(th), weights @ np.cos(th))
        return mean

    def _state_residual(self, sigmas: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """Hieu (sigma - mean), goi thanh phan theta ve [-pi, pi)."""
        d = sigmas - mean
        d[:, self.THETA] = normalize_angle(d[:, self.THETA])
        return d

    # ------------------------------------------------------------------
    # Buoc du doan
    # ------------------------------------------------------------------
    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Buoc du doan cua UKF (Unscented Transform).

        1. Sinh sigma point tu (x, P)
        2. Cho TUNG diem di qua ham CTRV phi tuyen f() - khong xap xi gi
        3. Gop lai thanh ky vong + hiep phuong sai moi, cong nhieu qua trinh Q
        """
        sigmas = self.sigma_points(mean, covariance)
        # Buoc 2: f() ke thua tu EKFTrackerCTRV -> mo hinh chuyen dong y het EKF,
        # ke ca cach xu ly diem ky di omega ~ 0 cho TUNG sigma point rieng biet.
        propagated = np.array([self.f(s) for s in sigmas])

        new_mean = self._state_mean(propagated, self.Wm)
        d = self._state_residual(propagated, new_mean)
        # QUAN TRONG: Q phai tinh tai chieu cao TRUOC khi du doan (mean[H]), giong
        # het EKF va baseline ultralytics. Neu tinh tai new_mean[H] (chieu cao SAU
        # du doan) thi Q se khac EKF mot chut va phep so sanh khong con cong bang.
        new_cov = (d * self.Wc[:, None]).T @ d + self._process_noise(mean[self.H])

        new_mean[self.THETA] = normalize_angle(new_mean[self.THETA])
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov

    # ------------------------------------------------------------------
    # Buoc cap nhat
    # ------------------------------------------------------------------
    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray, confidence: float | None = None):
        """Buoc cap nhat cua UKF bang sigma point.

            Z_i   = h(X_i)                        (chieu sang khong gian do)
            z_hat = sum( Wm_i * Z_i )
            S     = sum( Wc_i * (Z_i - z_hat)(Z_i - z_hat)^T ) + R
            Pxz   = sum( Wc_i * (X_i - x)(Z_i - z_hat)^T )
            K     = Pxz * S^-1
            x     = x + K (z - z_hat)
            P     = P - K S K^T

        Vi h tuyen tinh, ket qua trung khit voi buoc update cua EKF/KF - da co
        unit test kiem chung (sai so ~1e-10).
        """
        n = self.NDIM
        sigmas = self.sigma_points(mean, covariance)

        # Chieu sang khong gian do: z = H x  (ma tran chon cx, cy, a, h)
        H = self._update_mat
        Z = sigmas @ H.T                       # (2n+1, 4)
        z_hat = self.Wm @ Z

        dz = Z - z_hat
        # Nhieu do R: dung chung ham voi EKF -> chi khac nhau o buoc predict
        R = self._measurement_noise(mean[self.H], confidence)
        S = (dz * self.Wc[:, None]).T @ dz + R

        dx = self._state_residual(sigmas, mean)
        Pxz = (dx * self.Wc[:, None]).T @ dz   # (n, 4)

        kalman_gain = np.linalg.solve(S, Pxz.T).T
        innovation = np.asarray(measurement, dtype=np.float64) - z_hat

        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ S @ kalman_gain.T

        new_mean[self.THETA] = normalize_angle(new_mean[self.THETA])
        new_cov = 0.5 * (new_cov + new_cov.T)
        return new_mean, new_cov
