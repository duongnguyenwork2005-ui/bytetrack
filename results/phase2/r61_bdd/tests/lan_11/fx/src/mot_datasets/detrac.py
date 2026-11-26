"""
detrac.py -- UA-DETRAC: bao lai config.py + cac file da co cua Giai doan 1.

KHONG doi hanh vi gi: danh sach video, thu tu frame (sorted img*.jpg, chi so
1-based nhu baseline_track.py), vung bo qua (gt/ignored_regions.txt do
to_motchallenge.py tao) deu lay tu cac ham/thu muc cu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from .base import DatasetSpec, FrameRef  # noqa: E402

#: part -> ten split MOTChallenge da tao o Giai doan 1 (data/processed/<split>)
PART_TO_SPLIT = {"train": "DETRAC-all", "test": "DETRAC-test", "sample": "DETRAC-sample"}


class DetracDataset(DatasetSpec):
    name = "detrac"
    fps = float(config.FPS)          # 25
    img_w = config.IMG_W             # 960
    img_h = config.IMG_H             # 540
    categories = ("car", "bus", "van", "others")
    # Giai doan 1 gop moi loai xe -> 1 lop "vehicle"; detector COCO: 2 car, 3 motorcycle, 5 bus, 7 truck
    coco_to_dataset = {2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle"}
    # vehicle_type cua GT -> COCO id (chi dung cho detection tong hop tu GT, det_cache --source gt)
    category_to_coco = {"car": 2, "van": 2, "bus": 5, "others": 7, "vehicle": 2}

    def __init__(self, img_root: Path | None = None):
        self.img_root = Path(img_root) if img_root else config.find_images_root()

    @staticmethod
    def split_name(part: str) -> str:
        return PART_TO_SPLIT.get(part, part)

    def list_videos(self, part: str) -> list[str]:
        if part == "sample":
            return list(config.SAMPLE_VIDEOS)
        return config.list_videos(part)

    def frames(self, video: str, part: str = "train") -> list[FrameRef]:
        if self.img_root is None:
            raise FileNotFoundError("Khong tim thay thu muc anh UA-DETRAC (config.find_images_root())")
        paths = sorted((self.img_root / video).glob("img*.jpg"))
        if not paths:
            raise FileNotFoundError(f"Khong co anh trong {self.img_root / video}")
        dt = self.dt_nominal
        return [FrameRef(index=i, path=p, time_s=(i - 1) * dt, name=p.name)
                for i, p in enumerate(paths, start=1)]

    def ignored_regions(self, video: str, part: str = "train"):
        p = config.PROCESSED_DIR / self.split_name(part) / video / "gt" / "ignored_regions.txt"
        if not p.exists() or p.stat().st_size == 0:
            return []
        arr = np.loadtxt(p, delimiter=",", ndmin=2)
        return [tuple(float(v) for v in row) for row in arr]

    def gt_dataframe(self, part: str = "train"):
        import pandas as pd
        return pd.read_parquet(config.ann_parquet(part))
