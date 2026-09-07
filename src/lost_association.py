"""
lost_association.py -- Giai doan B: ham chi phi lien ket thay the CHO RIENG
track dang o trang thai LOST.

===========================================================================
VAN DE (da do o phaseB_iou_diagnosis.py)
===========================================================================
Tai thoi diem ByteTrack thu lien ket lai mot track da mat:
    % IoU = 0   : KF 48.7% | EKF 54.3% | UKF 51.7%
    % IoU < 0.2 : KF 69.6% | EKF 73.6% | UKF 74.4%   (nguong ghep cua ByteTrack)

IoU co "vach dung": box du doan lech 70px va lech 400px deu cho IoU = 0 y het
nhau. Ham chi phi mat hoan toan kha nang phan biet du doan tot voi du doan te
o dung vung ma phan lon truong hop roi vao.

===========================================================================
HAI CACH THAY THE - VA HAI CAI BAY DA KIEM CHUNG BANG SO
===========================================================================

--- BAY 1: khong duoc thay THANG DIoU vao cost = 1 - DIoU ---
ByteTrack dung match_thresh = 0.8 tren cost = 1 - IoU, tuc can IoU > 0.2.
DIoU <= 0 khi hai box KHONG giao nhau, nen cost = 1 - DIoU >= 1 > 0.8:
    lech  70px: DIoU = -0.253 -> cost = 1.253  -> VAN BI TU CHOI
    lech 400px: DIoU = -0.747 -> cost = 1.747  -> VAN BI TU CHOI
=> Thay thang DIoU KHONG cuu duoc mot truong hop nao. Phai CHUAN HOA ve [0,1]:

    pseudo_iou = (DIoU + 1) / 2      thuoc [0, 1]
    cost       = 1 - pseudo_iou = (1 - DIoU) / 2

Sau khi chuan hoa (da kiem chung):
    lech  70px -> cost 0.626  NHAN     lech 250px -> cost 0.817  tu choi
    lech 150px -> cost 0.741  NHAN     lech 400px -> cost 0.874  tu choi
=> Gate gian ra toi ~150px nhung VAN co gradient va van tu choi du doan qua te.

--- BAY 2: gian no box roi tinh IoU thi TU LAM LOANG chinh no ---
IoU chia cho HOP dien tich. Gian box du doan len k lan thi dien tich tang k^2,
nen IoU toi da co the dat duoc chi con (dien tich detection)/(dien tich da gian):
    k = 2 -> IoU toi da 0.250      k = 3 -> IoU toi da 0.111  < 0.2 !
    k = 5 -> IoU toi da 0.040
=> Gian 3x tro len thi KHONG BAO GIO vuot nguong 0.2 duoc, du du doan trung tam.
Phai doi mau so: dung IoA (Intersection over Area cua DETECTION) thay vi IoU:

    IoA = dien_tich_giao / dien_tich_detection

IoA khong bi loang khi gian box track, va van bang 1.0 khi detection nam tron
trong vung gate. Da kiem chung: k=2 nhan lech ~70px, k=3 nhan ~100px,
k=5 nhan ~150px.

===========================================================================
HE SO GIAN NO TANG THEO SO FRAME DA MAT
===========================================================================
    k = min(1 + rate * so_frame_da_mat, k_max)

Y nghia: cang mat lau thi do bat dinh vi tri cang lon, nen cang phai chap nhan
vung tim kiem rong hon. Day la cach xap xi don gian cua viec noi gate theo
trace(P) - dung so frame vi no de giai thich va khong phu thuoc cach dat Q.

===========================================================================
NGUYEN TAC THUC NGHIEM
===========================================================================
- Chi ap dung cho track LOST. Track dang Tracked giu NGUYEN IoU goc de khong
  pha baseline.
- Ap dung NHU NHAU cho ca 3 motion model (co ca ban cho KF + CV baseline),
  giu dung nguyen tac chi motion model la bien nghien cuu.
- Mac dinh TAT ca hai co. Bat qua file cau hinh .yaml cua tracker.
"""
from __future__ import annotations

import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching


def _tlwh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.asarray(b, dtype=np.float64).copy()
    out[..., 2:] += out[..., :2]
    return out


def normalized_diou_cost(tracks_tlwh: np.ndarray, dets_tlwh: np.ndarray) -> np.ndarray:
    """Ma tran chi phi (n_track, n_det) dung DIoU DA CHUAN HOA ve [0, 1].

        DIoU = IoU - d^2 / c^2
        d = khoang cach 2 tam ; c = duong cheo box bao nho nhat chua ca hai
        cost = (1 - DIoU) / 2

    Xem docstring dau file muc "BAY 1" ve ly do bat buoc phai chuan hoa.
    """
    n, m = len(tracks_tlwh), len(dets_tlwh)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    a = _tlwh_to_xyxy(tracks_tlwh)[:, None, :]     # (n,1,4)
    b = _tlwh_to_xyxy(dets_tlwh)[None, :, :]       # (1,m,4)

    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

    acx, acy = (a[..., 0] + a[..., 2]) / 2, (a[..., 1] + a[..., 3]) / 2
    bcx, bcy = (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2
    d2 = (acx - bcx) ** 2 + (acy - bcy) ** 2

    ex1 = np.minimum(a[..., 0], b[..., 0]); ey1 = np.minimum(a[..., 1], b[..., 1])
    ex2 = np.maximum(a[..., 2], b[..., 2]); ey2 = np.maximum(a[..., 3], b[..., 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2

    diou = iou - np.where(c2 > 0, d2 / np.maximum(c2, 1e-9), 0.0)
    return ((1.0 - diou) / 2.0).astype(np.float32)


def expanded_ioa_cost(tracks_tlwh: np.ndarray, dets_tlwh: np.ndarray,
                      k: np.ndarray) -> np.ndarray:
    """Ma tran chi phi dung gate DA GIAN NO + IoA (chia cho dien tich DETECTION).

        cost = 1 - dien_tich_giao / dien_tich_detection

    `k`: he so gian no cho TUNG track (mang do dai n_track).
    Xem docstring dau file muc "BAY 2" ve ly do phai dung IoA chu khong phai IoU.
    """
    n, m = len(tracks_tlwh), len(dets_tlwh)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    t = np.asarray(tracks_tlwh, dtype=np.float64).copy()
    cx, cy = t[:, 0] + t[:, 2] / 2, t[:, 1] + t[:, 3] / 2
    w, h = t[:, 2] * k, t[:, 3] * k
    big = np.c_[cx - w / 2, cy - h / 2, w, h]

    a = _tlwh_to_xyxy(big)[:, None, :]
    b = _tlwh_to_xyxy(dets_tlwh)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_det = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    ioa = np.where(area_det > 0, inter / np.maximum(area_det, 1e-9), 0.0)
    return (1.0 - np.clip(ioa, 0.0, 1.0)).astype(np.float32)


class LostAwareAssociation:
    """Mixin ghi de `get_dists` cua BYTETracker cho RIENG cac track dang LOST.

    Doc 3 tham so tu file cau hinh tracker (.yaml), mac dinh deu TAT:
        lost_cost        : "iou" (goc) | "diou" (DIoU chuan hoa)
        lost_expand_rate : 0.0 = tat. Vd 0.10 -> k = 1 + 0.10 * so_frame_mat
        lost_expand_max  : tran cua k (mac dinh 3.0)

    Neu ca hai deu tat -> hanh vi TRUNG KHIT baseline goc.
    """

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)

        cost_mode = str(getattr(self.args, "lost_cost", "iou")).lower()
        rate = float(getattr(self.args, "lost_expand_rate", 0.0))
        kmax = float(getattr(self.args, "lost_expand_max", 3.0))

        if (cost_mode != "iou" or rate > 0.0) and len(tracks) and len(detections):
            lost = [i for i, t in enumerate(tracks) if t.state != TrackState.Tracked]
            if lost:
                t_tlwh = np.array([tracks[i].tlwh for i in lost], dtype=np.float64)
                d_tlwh = np.array([d.tlwh for d in detections], dtype=np.float64)

                n_lost = np.array(
                    [max(0, self.frame_id - tracks[i].end_frame) for i in lost],
                    dtype=np.float64)
                k = np.minimum(1.0 + rate * n_lost, kmax) if rate > 0.0 else None

                alt = None
                if cost_mode == "diou":
                    alt = normalized_diou_cost(t_tlwh, d_tlwh)
                    if k is not None:
                        # Gop kieu "de dai": chap nhan neu THOA MAN it nhat mot tieu chi.
                        # DA DO VA THAY CO HAI (EKF -8 loi rong, IDSW +1532) - giu lai
                        # de tai lap ket qua, KHONG dung nua.
                        alt = np.minimum(alt, expanded_ioa_cost(t_tlwh, d_tlwh, k))

                elif cost_mode == "gatediou":
                    # --- BAN SUA (sau khi do thay 2 cach tren deu lam IDSW no) ---
                    # Tach vai tro cua hai thanh phan, thay vi de ca hai vua loc vua
                    # xep hang:
                    #   gate gian no -> chi LOC UNG VIEN (trong hay ngoai vung tim kiem)
                    #   DIoU chuan hoa -> XEP HANG cac ung vien con lai
                    #
                    # VI SAO: IoA bao hoa o 1.0 nen MOI detection nam gon trong gate deu
                    # cho cost = 0.0, ke ca xe khac. Da kiem chung bang so: xe dung va xe
                    # khac nho trong gate deu cost 0.000 -> Hungarian noi bua -> IDSW
                    # tang 5-6 lan. DIoU khong bao hoa: no giam dan theo khoang cach tam
                    # nen van phan biet duoc ung vien gan voi ung vien xa.
                    alt = normalized_diou_cost(t_tlwh, d_tlwh)
                    if k is not None:
                        outside = expanded_ioa_cost(t_tlwh, d_tlwh, k) >= 1.0
                        # Ngoai gate -> dat cost = 1.0 (> match_thresh 0.8) de bi tu choi.
                        alt = np.where(outside, 1.0, alt)

                elif rate > 0.0:
                    alt = expanded_ioa_cost(t_tlwh, d_tlwh, k)

                if alt is not None:
                    dists[np.array(lost), :] = alt

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists
