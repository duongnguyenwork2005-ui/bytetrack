"""
base.py -- Giao dien chung cho moi dataset MOT dung trong Giai doan 2.

Mot dataset phai cho biet:
    - fps danh nghia (de doi track_buffer theo giay va bao cao van toc px/s)
    - kich thuoc anh (H, W) - dung cho `Boxes(orig_shape)` cua ultralytics
    - danh sach video cua mot phan (part: train / val / test ...)
    - danh sach frame cua mot video: chi so frame (1-based, LIEN TUC theo thu tu
      xu ly), duong dan anh, thoi diem (giay) - de tinh Delta t thuc
    - vung bo qua (ignored regions) cua video, neu dataset co
    - bang class cua dataset va anh xa tu class detector (COCO id) sang class dataset
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRef:
    """Mot frame trong chuoi xu ly cua mot video."""
    index: int          # chi so frame 1-based, lien tuc (1, 2, 3, ...)
    path: Path | None   # duong dan anh (None neu chua co anh - vd fixture khong anh)
    time_s: float       # thoi diem [s] ke tu frame dau (danh nghia hoac tu timestamp)
    name: str = ""      # ten frame goc trong dataset (vd BDD: "<video>-0000001.jpg")


class DatasetSpec:
    """Lop co so. Lop con ghi de cac thuoc tinh / phuong thuc duoi day."""

    name: str = "base"
    fps: float = 25.0
    img_w: int = 0
    img_h: int = 0
    #: ten class cua dataset -> id class dung khi xuat MOTChallenge (1 = lop danh gia)
    categories: tuple[str, ...] = ()
    #: COCO class id (detector yolov8n pretrained) -> ten class cua dataset
    coco_to_dataset: dict[int, str] = {}

    @property
    def dt_nominal(self) -> float:
        return 1.0 / self.fps

    @property
    def orig_shape(self) -> tuple[int, int]:
        return (self.img_h, self.img_w)

    def list_videos(self, part: str) -> list[str]:
        raise NotImplementedError

    def frames(self, video: str, part: str) -> list[FrameRef]:
        raise NotImplementedError

    def ignored_regions(self, video: str, part: str) -> list[tuple[float, float, float, float]]:
        """(left, top, w, h) cac vung to xam truoc khi dua vao detector. Mac dinh: khong co."""
        return []

    def gt_dataframe(self, part: str):
        """DataFrame annotation (schema chung: video, frame, track_id, left, top, width, height,
        category, occlusion_ratio, ...). Lop con cai."""
        raise NotImplementedError
