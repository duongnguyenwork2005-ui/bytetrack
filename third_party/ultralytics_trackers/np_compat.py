"""
np_compat.py -- Lop tuong thich NumPy cho ban vendor cua ultralytics.trackers.

Bon module tracker cua ultralytics (`basetrack.py`, `byte_tracker.py`, `utils/matching.py`,
`utils/kalman_filter.py`, `utils/stracks.py`) KHONG import torch dong nao. Torch chi vao qua
`ultralytics/__init__.py` va qua SAU ky hieu tien ich duoi day. File nay cung cap dung sau ky hieu do
bang NumPy/SciPy, nho vay ban vendor chay duoc trong moi truong KHONG cai torch/Ultralytics
(muc tieu: Jetson Nano).

File nay la ma DUY NHAT duoc viet them; nam file tracker giu NGUYEN VAN ban upstream, chi doi 5 dong
`import` (xem PROVENANCE.json muc `import_rewrites`).

`bbox_ioa` va `xywh2ltwh` duoc CHEP NGUYEN phan than ham tu ultralytics (ca hai von la NumPy thuan) de
bao dam hanh vi giong het, khong phai cai dat lai.
"""
from __future__ import annotations

import logging

import numpy as np

__all__ = ["LOGGER", "batch_probiou", "bbox_ioa", "check_requirements", "linear_sum_assignment",
           "xywh2ltwh"]

#: `ultralytics.utils.LOGGER` chi duoc dung de goi `.warning(...)` trong byte_tracker.py.
LOGGER = logging.getLogger("bytetrack_np")


def check_requirements(*args, **kwargs) -> bool:
    """No-op: ban vendor khai bao dependency truc tiep (numpy, scipy, lap) thay vi cai dat luc chay."""
    return True


def xywh2ltwh(x):
    """[x, y, w, h] -> [x1, y1, w, h] (x1,y1 = goc tren-trai).

    Chep tu `ultralytics.utils.ops.xywh2ltwh`, bo nhanh torch (ban vendor luon nhan np.ndarray).
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Giao tren dien tich box2 (hoac IoU chuan neu `iou=True`).

    Chep NGUYEN VAN tu `ultralytics.utils.metrics.bbox_ioa` (ban goc da la NumPy thuan).
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def batch_probiou(obb1, obb2, eps: float = 1e-7):
    """OBB (box xoay, 5 cot xywhr) -- NGOAI PHAM VI backend NumPy.

    `matching.iou_distance` chi goi ham nay khi box co 5 cot. Pipeline cua de tai dung box THANG TRUC
    (4 cot) o moi duong chay, nen nhanh nay khong bao gio duoc goi. Ban upstream cai dat bang torch;
    thay vi cai dat lai mot phep tinh chua tung duoc kiem chung o day, ta BAO LOI RO RANG.
    """
    raise NotImplementedError(
        "batch_probiou (OBB, box xoay) khong duoc ho tro trong backend NumPy: ban upstream cai dat bang "
        "torch va pipeline cua de tai chi dung box thang truc (4 cot). Neu can OBB, dung backend "
        "ultralytics tren PC."
    )


def linear_sum_assignment(cost_matrix):
    """Ghep cap chi phi nho nhat -- dung `scipy.optimize.linear_sum_assignment`.

    `ultralytics.utils.ops.linear_sum_assignment` uu tien chinh SciPy khi co (va chi dung ban NumPy du
    phong khi thieu SciPy), nen goi thang SciPy cho ket qua giong ban upstream. Duong chay mac dinh cua
    `matching.linear_assignment` la `lap.lapjv` (use_lap=True); ham nay chi duoc dung khi use_lap=False.
    """
    from scipy.optimize import linear_sum_assignment as _lsa

    return _lsa(np.asarray(cost_matrix))
