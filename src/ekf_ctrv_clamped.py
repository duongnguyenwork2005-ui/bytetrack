"""
ekf_ctrv_clamped.py -- EKF + CTRV co CHAN toc do quay omega.

VI SAO CAN BAN NAY
------------------
`diagnose_ekf_circle.py` do duoc tren 50 doan che cua 48 track dai: ban EKF+CTRV
goc khong dat gioi han nao cho omega, dan den

    - trung vi |omega| = 0,366 do/frame  (ground truth: 0,441)  -> binh thuong OK
    - nhung phan vi 90 = 19,9 do/frame, lon nhat 111,4 do/frame
    - 28% doan ngoai suy voi omega vuot nguong vat ly 2,9 do/frame (72 do/giay)
    - 26% doan co ban kinh quy dao R = v/|omega| < 100 px, nho hon ca chiec xe
    - 20% doan box du doan di HET IT NHAT MOT VONG TRON

CTRV ngoai suy tren duong tron ban kinh R = v/|omega|, quet mot cung |omega|*T
sau T frame mat quan sat. Nen quay vong tron TU NO la hanh vi dung cua mo hinh;
khuyet diem la khong co gi ngan omega nhan gia tri phi vat ly khi no duoc uoc
luong tu nhieu annotation hoac tu hiep phuong sai chua hoi tu.

NGUONG CHON
-----------
    OMEGA_CLAMP = 0,05 rad/frame = 2,9 do/frame = 72 do/giay o 25 fps

Xe hoi re nga tu quet khoang 90 do trong 2-4 giay, tuc 18-45 do/giay. Nguong
72 do/giay do do rong rai gap 1,6-4 lan chuyen dong that, chi cat phan duoi
bat kha thi. Day la ky thuat tieu chuan cho CTRV, khong phai meo lam dep so lieu.

CHAN O CA HAI CHO
-----------------
    sau `update`  : khong de omega phi ly di vao trang thai va tich luy
    truoc `predict`: khong de ngoai suy tren omega phi ly du no den tu dau
                     (vi du duoc `initiate_from_motion` gan vao)

LUU Y VE TINH NHAT QUAN CUA BO LOC
----------------------------------
Cat gia tri trung binh ma KHONG dong thoi sua hiep phuong sai P la mot phep
CAN THIEP ngoai khuon kho Kalman: sau khi cat, P khong con la hiep phuong sai
hau nghiem dung cua trang thai da cat. Ta chap nhan dieu nay vi (a) day la cach
lam pho bien trong cai dat CTRV thuc te, va (b) tac dong duoc do bang thuc
nghiem chu khong suy luan - xem `omega_clamp_experiment.py`.

BAN NAY LA BIEN THE THEM VAO, KHONG THAY THE ban goc: `ekf_ctrv.py` giu nguyen
de moi ket qua da bao cao van tai lap duoc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402

#: Nguong toc do quay toi da [rad/frame]. Xem docstring dau file.
OMEGA_CLAMP = 0.05


class EKFTrackerCTRVClamped(EKFTrackerCTRV):
    """EKF + CTRV, chan |omega| <= OMEGA_CLAMP. Moi thu khac giu nguyen ban goc."""

    def _clip_omega(self, mean: np.ndarray) -> np.ndarray:
        mean[self.OMEGA] = float(np.clip(mean[self.OMEGA], -OMEGA_CLAMP, OMEGA_CLAMP))
        return mean

    def predict(self, mean, covariance):
        return super().predict(self._clip_omega(mean.copy()), covariance)

    def update(self, mean, covariance, measurement, confidence=None):
        m, c = super().update(mean, covariance, measurement, confidence)
        return self._clip_omega(m), c

    def multi_predict(self, mean, covariance):
        m = np.asarray(mean, dtype=np.float64).copy()
        m[:, self.OMEGA] = np.clip(m[:, self.OMEGA], -OMEGA_CLAMP, OMEGA_CLAMP)
        return super().multi_predict(m, covariance)

    def initiate_from_motion(self, mean, covariance, measurement, n_frames: float = 1.0):
        m, c = super().initiate_from_motion(mean, covariance, measurement, n_frames)
        return self._clip_omega(m), c
