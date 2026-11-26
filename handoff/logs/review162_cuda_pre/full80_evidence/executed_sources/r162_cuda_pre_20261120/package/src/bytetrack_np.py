"""
bytetrack_np.py -- Duong chay ByteTrack **khong phu thuoc torch/Ultralytics**.

MUC DICH: chay vong lap ByteTrack + bo loc (KF-CV, EKF-CTRV) tu detection cache tren mot may KHONG cai
torch va Ultralytics - dieu kien bat buoc de do CV so voi EKF tren Jetson Nano (JetPack 4.6 khong chay
duoc ultralytics 8.4).

CACH DUNG
    BYTETRACK_BACKEND=np python -c "import bytetrack_np as b; ..."

    from bytetrack_np import Detections, make_tracker_np
    tr  = make_tracker_np("bytetrack_ekf_ctrv", cfg)
    det = Detections.from_cache_array(arr, orig_shape)     # arr: (N,6) xyxy+conf+cls cua det_cache
    out = tr.update(det, None, n_steps=1)                  # (M,7): xyxy, track_id, conf, cls, idx

HOP DONG DETECTION (thay cho `ultralytics.engine.results.Boxes`)
    Detections.xywh  float32 (N,4)  tam + kich thuoc (cx, cy, w, h), toa do ANH GOC (pixel)
    Detections.xyxy  float32 (N,4)  goc tren-trai / goc duoi-phai, toa do ANH GOC (pixel)
    Detections.conf  float32 (N,)   do tin cay [0,1]
    Detections.cls   float32 (N,)   chi so lop COCO (giu float32 giong `Boxes.cls` de so sanh duoc)
    len(Detections)  = N
`BYTETracker.update` chi doc dung `.xywh`, `.conf`, `.cls` va `len()` (xem `stracks.parse_bboxes`), nen
day la TOAN BO be mat can thiet - khong co tensor, khong co device.

GIOI HAN CO Y
    - Chi CV va EKF-CTRV. UKF/PF/SORT chua mo tren backend nay (khong can cho muc tieu CV/EKF tren Nano).
    - Box THANG TRUC (4 cot). OBB (box xoay) khong ho tro - xem `np_compat.batch_probiou`.
    - Khong co detector: backend nay chay TU CACHE. Tao cache van can GPU + Ultralytics tren PC.

DEPENDENCIES THUC TE: numpy, scipy, lap (xem third_party/ultralytics_trackers/PROVENANCE.json).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ul_backend  # noqa: E402

__all__ = ["Detections", "make_tracker_np", "tracker_registry_np"]


class Detections:
    """Detection dang NumPy thuan, thay cho `Boxes` cua Ultralytics.

    Giu nguyen quy uoc cua `Boxes`: toa do theo ANH GOC (khong chuan hoa), dtype float32, thu tu hang
    GIU NGUYEN nhu mang detection dau vao (ByteTracker dung chi so hang lam `track.idx`).
    """

    __slots__ = ("xyxy", "conf", "cls", "orig_shape")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray,
                 orig_shape: tuple[int, int] | None = None):
        self.xyxy = np.ascontiguousarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.ascontiguousarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.ascontiguousarray(cls, dtype=np.float32).reshape(-1)
        if not (len(self.xyxy) == len(self.conf) == len(self.cls)):
            raise ValueError(f"so hang khong khop: xyxy={len(self.xyxy)} conf={len(self.conf)} "
                             f"cls={len(self.cls)}")
        self.orig_shape = tuple(orig_shape) if orig_shape is not None else None

    @classmethod
    def from_cache_array(cls, arr: np.ndarray, orig_shape: tuple[int, int] | None = None) -> Detections:
        """Tu mang (N,6) cua `det_cache` [x1, y1, x2, y2, conf, cls] - dung dinh dang ma
        `det_cache.to_boxes` dua cho `Boxes`, nen hai backend nhan CUNG mot so lieu dau vao."""
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 6)
        return cls(a[:, :4], a[:, 4], a[:, 5], orig_shape)

    @property
    def xywh(self) -> np.ndarray:
        """(cx, cy, w, h) float32 - dung cong thuc cua `Boxes.xywh` (`ops.xyxy2xywh`)."""
        x = self.xyxy
        out = np.empty_like(x)
        out[:, 0] = (x[:, 0] + x[:, 2]) / 2
        out[:, 1] = (x[:, 1] + x[:, 3]) / 2
        out[:, 2] = x[:, 2] - x[:, 0]
        out[:, 3] = x[:, 3] - x[:, 1]
        return out

    def __getitem__(self, idx) -> Detections:
        """Cat theo mask/chi so, tra ve `Detections` moi -- giong `Boxes.__getitem__`.

        `BYTETracker._split_detections` cat detection thanh nhom diem cao / diem thap bang mask
        boolean, nen day la thao tac BAT BUOC cua hop dong (xem byte_tracker.py:318).
        """
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx], self.orig_shape)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __repr__(self) -> str:
        return f"Detections(n={len(self)}, orig_shape={self.orig_shape})"


def tracker_registry_np() -> dict:
    """tracker_type -> lop tracker, chay duoc KHONG can torch/Ultralytics.

    Chi gom CV va EKF-CTRV (pham vi da chot). `tracker_ctrv` duoc import SAU khi backend duoc dat, nen
    cac lop o day ke thua lop nen cua ban VENDOR.
    """
    if ul_backend.backend_name() != "np":
        raise RuntimeError(
            "tracker_registry_np() can BYTETRACK_BACKEND=np. Dat bien moi truong (hoac goi "
            "ul_backend.use_backend('np')) TRUOC khi import module tracker.")
    import tracker_ctrv                                     # noqa: PLC0415

    return {"bytetrack": tracker_ctrv.CVByteTracker,
            "bytetrack_cv": tracker_ctrv.CVByteTracker,
            "bytetrack_ekf_ctrv": tracker_ctrv.CTRVByteTracker}


def make_tracker_np(tracker_type: str, cfg):
    """Tao tracker theo `tracker_type` tren backend NumPy. `cfg` giong het backend hien tai."""
    reg = tracker_registry_np()
    if tracker_type not in reg:
        raise ValueError(f"tracker_type {tracker_type!r} chua duoc mo tren backend NumPy; "
                         f"hien co {sorted(reg)}")
    import time_units                                       # noqa: PLC0415

    tr = time_units.with_elapsed_steps(reg[tracker_type])(args=cfg)
    tr.reset()
    return tr
